// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reading 3MF, the only mesh format here that states its own units.
//!
//! A 3MF is an OPC package: a ZIP holding XML. The container is [`zip`]'s job;
//! this module is the part that knows what the XML means, which is more than it
//! first appears. STL and OBJ hand over a triangle soup already in its final
//! place. A 3MF hands over a *scene*: objects that may be built out of other
//! objects, each instance carrying its own affine transform, in units the file
//! chooses. Flattening that is where a reader either produces the solid the
//! author saw or a plausible looking wrong one.
//!
//! # The three ways to read a wrong solid
//!
//! **Units.** `unit` is honoured and scaled to millimetres. A file in metres
//! read as millimetres arrives a thousand times too small, which looks like a
//! successful import of a speck rather than like an error. This is the only
//! legitimate source of scale in the whole importer.
//!
//! **Transforms.** 3MF writes an affine as twelve numbers in row-major 4x3
//! order, and treats a point as a *row* vector multiplied on the left. Get the
//! convention backwards and a translation still lands correctly while every
//! rotation comes out transposed -- which is to say inverted -- so a test with a
//! translation alone proves nothing. See [`instance_transform`].
//!
//! **Extensions.** 3MF is extensible, and two of its extensions put the real
//! geometry somewhere this reader does not look: beam lattice replaces the mesh
//! with a strut graph, and slice replaces it with stacked contours. A file that
//! marks either as required is refused by name, because importing the fallback
//! mesh from one would silently hand back a different solid than the file
//! describes.
//!
//! # What is deliberately not read
//!
//! Materials, colours, textures and metadata, because a signed distance field
//! has nowhere to put any of them -- the same reason [`super::obj`] discards
//! them. Object `type` is ignored too: a support or an "other" object still
//! contributes surface, and deciding what to do with that is the sculptor's
//! call rather than the reader's.

mod zip;

use glam::{Mat4, Vec3};
use roxmltree::{Document, Node};
use rustc_hash::FxHashMap;

use super::{ImportError, SoupWelder, finite};
use crate::export::ExportMesh;

/// The relationship type that points at the root model part.
///
/// Following this rather than reaching straight for `3D/3dmodel.model` is the
/// difference between reading a package and guessing at one: the name is a
/// convention that most writers follow and the specification does not require.
const MODEL_RELATIONSHIP: &str = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel";

/// Extensions whose geometry lives somewhere this reader does not look.
const REFUSED_EXTENSIONS: [(&str, &str); 2] = [
    (
        "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02",
        "the beam lattice extension, whose geometry is a strut graph rather than the mesh",
    ),
    (
        "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07",
        "the slice extension, whose geometry is stacked contours rather than the mesh",
    ),
];

/// How deep a chain of components may nest before it is called a mistake.
///
/// Cycles are caught exactly, by the identifiers on the stack, so this only
/// bounds the legitimate-but-absurd case -- a chain deep enough to overflow the
/// stack while never repeating an object. Counted across model parts, not
/// within one, because a chain that hops between parts is still a chain.
const MAX_COMPONENT_DEPTH: usize = 64;

/// Model parts a package may hold before it is called a mistake.
///
/// The production extension splits a scene across parts, and every one of them
/// is inflated and parsed. A package naming thousands would be asking this
/// reader to do a great deal of work from a very small file.
const MAX_MODEL_PARTS: usize = 256;

pub fn read(bytes: &[u8]) -> Result<ExportMesh, ImportError> {
    let archive = zip::Archive::open(bytes)?;

    let relationships = archive.read("_rels/.rels")?.ok_or_else(|| {
        ImportError::Malformed(
            "it is a ZIP but has no `_rels/.rels`, so it is not an OPC package".to_string(),
        )
    })?;
    let root = model_part(&relationships)?;

    // Every model part the package reaches, root first. Inflated up front
    // rather than as the walk discovers them, for two reasons: a part reached
    // from a hundred components would otherwise be inflated and CRC checked a
    // hundred times, and the borrow checker will not let an `objects` map hold
    // nodes from a document whose backing text is still being pushed into a
    // growing collection.
    let parts = gather_parts(&archive, root)?;
    read_parts(&parts)
}

/// Inflate the root model part and everything it reaches through `path`.
///
/// The production extension lets a build item or a component name geometry in
/// another part. This follows those, transitively, refusing a package that
/// names one it does not contain rather than importing a model with a piece
/// missing.
fn gather_parts(
    archive: &zip::Archive,
    root: String,
) -> Result<Vec<(String, String)>, ImportError> {
    let mut parts: Vec<(String, String)> = Vec::new();
    let mut seen: FxHashMap<String, ()> = FxHashMap::default();
    let mut pending = vec![root];

    while let Some(name) = pending.pop() {
        if seen.insert(part_key(&name), ()).is_some() {
            continue;
        }
        if parts.len() >= MAX_MODEL_PARTS {
            return Err(ImportError::Malformed(format!(
                "it is split across more than {MAX_MODEL_PARTS} model parts"
            )));
        }

        let bytes = archive.read(&name)?.ok_or_else(|| {
            ImportError::Malformed(format!(
                "it points at model part `{name}`, which is not in the package"
            ))
        })?;
        let text = String::from_utf8(bytes).map_err(|_| {
            ImportError::Malformed(format!("its model part `{name}` is not valid UTF-8"))
        })?;

        // Scanned rather than walked: this pass only needs the names, and
        // reaching them through the component structure would mean resolving
        // ids before every part is available.
        //
        // The substring test in front of it is not micro-optimisation. Almost
        // every 3MF in existence is a single part with no `path` anywhere, and
        // parsing one twice -- once to find nothing, once to read it -- put
        // 235 ms on a 41 MB package in the import bench. Scanning the bytes for
        // the attribute name costs a memchr pass. A false positive only means
        // doing the parse that used to happen unconditionally.
        if text.contains("path") {
            let document = parse(&text, "model")?;
            for node in document.descendants() {
                if !matches!(local(node), "component" | "item") {
                    continue;
                }
                if let Some(path) = node.attributes().find(|a| a.name() == "path") {
                    pending.push(path.value().to_string());
                }
            }
        }
        parts.push((name, text));
    }
    Ok(parts)
}

/// How a part name is compared, matching what [`zip::Archive::read`] does.
fn part_key(name: &str) -> String {
    name.trim_start_matches('/').to_ascii_lowercase()
}

/// One part's resources, ready to be walked.
struct Part<'a> {
    model: Node<'a, 'a>,
    objects: FxHashMap<&'a str, Node<'a, 'a>>,
}

fn read_parts(parts: &[(String, String)]) -> Result<ExportMesh, ImportError> {
    let documents: Vec<Document> =
        parts.iter().map(|(_, text)| parse(text, "model")).collect::<Result<Vec<_>, _>>()?;

    let mut resolved: Vec<Part> = Vec::with_capacity(documents.len());
    let mut unit = 1.0f32;
    for (index, document) in documents.iter().enumerate() {
        let model = document.root_element();
        if local(model) != "model" {
            return Err(ImportError::Malformed(format!(
                "its model part starts with `<{}>` rather than `<model>`",
                local(model)
            )));
        }
        // Every part, not only the root: a part is free to declare that its own
        // geometry needs an extension this reader cannot follow.
        refuse_unreadable_extensions(model)?;

        let scale = unit_scale(model)?;
        if index == 0 {
            unit = scale;
        } else if scale != unit {
            // Refused rather than honoured per part. The specification puts
            // `unit` on the root model and says nothing useful about a sub-part
            // disagreeing, so either answer is a guess -- and a guess here is
            // the wrong-solid failure this module exists to avoid. No writer in
            // practice emits a mismatch.
            return Err(ImportError::Unsupported(format!(
                "its model parts disagree about units, `{}` against the root's `{}`",
                model.attribute("unit").unwrap_or("millimeter"),
                documents[0].root_element().attribute("unit").unwrap_or("millimeter"),
            )));
        }

        let mut objects: FxHashMap<&str, Node> = FxHashMap::default();
        match child(model, "resources") {
            Some(resources) => {
                for object in children(resources, "object") {
                    let id = object.attribute("id").ok_or_else(|| {
                        ImportError::Malformed("an object in it has no id".to_string())
                    })?;
                    objects.insert(id, object);
                }
            }
            // Only the root has to carry one. A part that holds nothing is
            // pointless rather than malformed, and the build walk will say so
            // if something references into it.
            None if index == 0 => {
                return Err(ImportError::Malformed(
                    "its model part has no resources section".to_string(),
                ));
            }
            None => {}
        }
        resolved.push(Part { model, objects });
    }

    // Units are folded into the root transform rather than applied afterwards,
    // so that there is exactly one place where a coordinate becomes millimetres
    // and no chance of a path that skips it.
    let root = Mat4::from_scale(Vec3::splat(unit));

    let mut welder = SoupWelder::new();
    let mut stack: Vec<(usize, &str)> = Vec::new();
    let mut items = 0usize;
    // Only the ROOT part's build is the scene. A sub-part written by a slicer
    // usually has no build at all, and where it has one it describes that
    // part's own plate rather than this import.
    if let Some(build) = child(resolved[0].model, "build") {
        for item in children(build, "item") {
            let part = referenced_part(item, 0, parts)?;
            let id = item.attribute("objectid").ok_or_else(|| {
                ImportError::Malformed("a build item in it names no object".to_string())
            })?;
            items += 1;
            emit(
                parts,
                &resolved,
                part,
                id,
                root * instance_transform(item)?,
                &mut stack,
                &mut welder,
            )?;
        }
    }

    if items == 0 {
        return Err(ImportError::Malformed(
            "its build section is empty, so the file describes nothing to print".to_string(),
        ));
    }
    welder.finish()
}

/// Which part a build item or component names, defaulting to the one it is in.
///
/// The attribute is matched on its local name because its prefix is whatever
/// the file declared -- `p:path` in every writer seen, but the specification
/// only fixes the namespace.
fn referenced_part(
    node: Node,
    current: usize,
    parts: &[(String, String)],
) -> Result<usize, ImportError> {
    let Some(path) = node.attributes().find(|attribute| attribute.name() == "path") else {
        // No path: it resolves against the part it appears in, NOT against the
        // root. Ids are unique only within a part, so treating them as global
        // would silently pick up whichever object happened to share the number.
        return Ok(current);
    };
    let wanted = part_key(path.value());
    parts.iter().position(|(name, _)| part_key(name) == wanted).ok_or_else(|| {
        ImportError::Malformed(format!(
            "it points at model part `{}`, which is not in the package",
            path.value()
        ))
    })
}

/// Find the model part by following `_rels/.rels`.
fn model_part(relationships: &[u8]) -> Result<String, ImportError> {
    let text = std::str::from_utf8(relationships).map_err(|_| {
        ImportError::Malformed("its relationships part is not valid UTF-8".to_string())
    })?;
    let document = parse(text, "relationships")?;

    for node in document.root_element().children().filter(|node| local(*node) == "Relationship") {
        if node.attribute("Type") != Some(MODEL_RELATIONSHIP) {
            continue;
        }
        let target = node.attribute("Target").ok_or_else(|| {
            ImportError::Malformed("its model relationship has no target".to_string())
        })?;
        // A target is either absolute within the package or relative to the
        // package root, and both mean the same entry once the leading slash is
        // gone. Nothing else is resolved, because nothing here touches a
        // filesystem -- see the note in `zip`.
        return Ok(target.trim_start_matches('/').to_string());
    }
    Err(ImportError::Malformed(
        "it has no 3D model relationship, so there is nothing in it to import".to_string(),
    ))
}

/// Refuse a file whose geometry is in an extension this reader does not read.
///
/// `requiredextensions` holds namespace *prefixes*, not the namespace URIs
/// themselves, so each one has to be resolved against the declarations on the
/// model element before it means anything. Matching the prefixes literally
/// would be matching on names a file is free to choose -- `b`, `bl`,
/// `beamlattice` are all the same extension.
fn refuse_unreadable_extensions(model: Node) -> Result<(), ImportError> {
    let Some(required) = model.attribute("requiredextensions") else {
        return Ok(());
    };
    for prefix in required.split_ascii_whitespace() {
        let Some(namespace) = model.lookup_namespace_uri(Some(prefix)) else {
            continue;
        };
        for (refused, what) in REFUSED_EXTENSIONS {
            if namespace == refused {
                return Err(ImportError::Unsupported(format!(
                    "it requires {what} -- importing it would produce a different solid than the \
                     file describes"
                )));
            }
        }
    }
    Ok(())
}

/// How many millimetres one of the file's units is.
///
/// Millimetre when unstated, which is what the specification defaults to.
fn unit_scale(model: Node) -> Result<f32, ImportError> {
    let unit = model.attribute("unit").unwrap_or("millimeter");
    match unit {
        "micron" => Ok(0.001),
        "millimeter" => Ok(1.0),
        "centimeter" => Ok(10.0),
        "inch" => Ok(25.4),
        "foot" => Ok(304.8),
        "meter" => Ok(1000.0),
        other => Err(ImportError::Malformed(format!(
            "it is measured in `{other}`, which is not one of the units 3MF defines"
        ))),
    }
}

/// Read the `transform` attribute of a build item or a component.
///
/// The twelve numbers are the rows of a 4x3, and 3MF multiplies a point into
/// them from the *left* as a row vector: `x' = x*m00 + y*m10 + z*m20 + m30`.
/// glam multiplies a column vector from the right, so the same affine is the
/// transpose of the naive reading -- which happens to mean the twelve numbers
/// drop straight into a column-major array in file order, with `0 0 0 1`
/// inserted as the fourth column of each row. That coincidence is exactly why
/// this is worth a comment: the wrong reading also compiles, also round trips a
/// translation, and only shows up as a rotation applied backwards.
fn instance_transform(node: Node) -> Result<Mat4, ImportError> {
    let Some(text) = node.attribute("transform") else {
        return Ok(Mat4::IDENTITY);
    };

    let mut values = [0.0f32; 12];
    let mut words = text.split_ascii_whitespace();
    for slot in &mut values {
        let word = words.next().ok_or_else(|| {
            ImportError::Malformed(
                "a transform in it has fewer than the twelve numbers 3MF transforms have"
                    .to_string(),
            )
        })?;
        *slot = word.parse::<f32>().ok().filter(|value| value.is_finite()).ok_or_else(|| {
            ImportError::Malformed(format!("`{word}` in a transform is not a usable number"))
        })?;
    }
    if words.next().is_some() {
        return Err(ImportError::Malformed(
            "a transform in it has more than the twelve numbers 3MF transforms have".to_string(),
        ));
    }

    let [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11] = values;
    Ok(Mat4::from_cols_array(&[
        m0, m1, m2, 0.0, //
        m3, m4, m5, 0.0, //
        m6, m7, m8, 0.0, //
        m9, m10, m11, 1.0,
    ]))
}

/// Add one object, and everything it is built out of, to the soup.
fn emit<'a>(
    parts: &[(String, String)],
    resolved: &[Part<'a>],
    part: usize,
    id: &'a str,
    world: Mat4,
    stack: &mut Vec<(usize, &'a str)>,
    welder: &mut SoupWelder,
) -> Result<(), ImportError> {
    // An object that contains itself, directly or through a chain, would
    // recurse until the stack ran out. That is a crash rather than an error
    // message, and a hand written file can reach it by accident.
    //
    // The key is the PART and the id together. Ids are unique only within a
    // part, so on the id alone a legitimate chain from part A's object 1 into
    // part B's object 1 reads as an object containing itself -- and a genuine
    // A:1 -> B:1 -> A:1 cycle would slip through.
    if stack.contains(&(part, id)) {
        return Err(ImportError::Malformed(format!(
            "object {id} in it contains itself, so it cannot be flattened"
        )));
    }
    if stack.len() >= MAX_COMPONENT_DEPTH {
        return Err(ImportError::Malformed(format!(
            "its components nest more than {MAX_COMPONENT_DEPTH} deep"
        )));
    }
    let object = *resolved[part].objects.get(id).ok_or_else(|| {
        ImportError::Malformed(format!("it references object {id}, which it does not contain"))
    })?;

    stack.push((part, id));
    if let Some(mesh) = child(object, "mesh") {
        emit_mesh(mesh, world, welder)?;
    }
    if let Some(components) = child(object, "components") {
        for component in children(components, "component") {
            let into = referenced_part(component, part, parts)?;
            let child_id = component.attribute("objectid").ok_or_else(|| {
                ImportError::Malformed("a component in it names no object".to_string())
            })?;
            // The child's own transform applies first and the parent's on top,
            // which in glam's column convention is the parent on the left.
            emit(
                parts,
                resolved,
                into,
                child_id,
                world * instance_transform(component)?,
                stack,
                welder,
            )?;
        }
    }
    stack.pop();
    Ok(())
}

/// Add one instance of one mesh, already placed in the world.
///
/// The vertices are re-read for every instance rather than cached. An object
/// used a hundred times pays for a hundred parses of its vertex list, which is
/// a real cost and a deliberate one: the alternative is a cache keyed by object
/// holding untransformed positions, and that is more machinery than a format
/// where deep instancing is rare deserves.
fn emit_mesh(mesh: Node, world: Mat4, welder: &mut SoupWelder) -> Result<(), ImportError> {
    let mut positions: Vec<Vec3> = Vec::new();
    if let Some(vertices) = child(mesh, "vertices") {
        for (number, vertex) in children(vertices, "vertex").enumerate() {
            let read = Vec3::new(
                coordinate(vertex, "x", number)?,
                coordinate(vertex, "y", number)?,
                coordinate(vertex, "z", number)?,
            );
            positions.push(finite(world.transform_point3(read), &format!("vertex {number}"))?);
        }
    }

    // A transform that mirrors turns every triangle inside out: the corners
    // still describe the same three points, but in the order that points the
    // face the other way. The voxeliser takes its inside from the winding, so
    // an unflipped mirrored instance arrives as a hole rather than as a solid.
    let mirrored = world.determinant() < 0.0;

    let Some(triangles) = child(mesh, "triangles") else {
        return Ok(());
    };
    for triangle in children(triangles, "triangle") {
        let a = corner(triangle, "v1", positions.len())?;
        let b = corner(triangle, "v2", positions.len())?;
        let c = corner(triangle, "v3", positions.len())?;
        let corners = if mirrored {
            [positions[a], positions[c], positions[b]]
        } else {
            [positions[a], positions[b], positions[c]]
        };
        welder.triangle(corners);
    }
    Ok(())
}

fn coordinate(vertex: Node, name: &str, number: usize) -> Result<f32, ImportError> {
    vertex.attribute(name).and_then(|text| text.parse::<f32>().ok()).ok_or_else(|| {
        ImportError::Malformed(format!("vertex {number} in it has no usable {name}"))
    })
}

/// Resolve one corner of a triangle, which is a zero based index into the
/// object's own vertex list.
fn corner(triangle: Node, name: &str, vertices: usize) -> Result<usize, ImportError> {
    let index = triangle
        .attribute(name)
        .and_then(|text| text.parse::<usize>().ok())
        .ok_or_else(|| ImportError::Malformed(format!("a triangle in it has no usable {name}")))?;
    if index >= vertices {
        return Err(ImportError::Malformed(format!(
            "a triangle in it uses vertex {index} of an object with {vertices} vertices"
        )));
    }
    Ok(index)
}

fn parse<'a>(text: &'a str, what: &str) -> Result<Document<'a>, ImportError> {
    Document::parse(text).map_err(|error| {
        ImportError::Malformed(format!("its {what} part is not valid XML: {error}"))
    })
}

/// The local name of an element, ignoring whatever namespace it is in.
///
/// Namespaces are deliberately not matched on. The core namespace has been
/// fixed since 2015 and every element read here has a name unique within it, so
/// matching the local name costs nothing and buys tolerance of files that put
/// the core namespace on a prefix rather than defaulting it -- which is legal,
/// and which a namespace-strict reader would reject for no benefit.
fn local<'i>(node: Node<'_, 'i>) -> &'i str {
    node.tag_name().name()
}

fn child<'a, 'i>(node: Node<'a, 'i>, name: &str) -> Option<Node<'a, 'i>> {
    children(node, name).next()
}

fn children<'a, 'i, 'n>(
    node: Node<'a, 'i>,
    name: &'n str,
) -> impl Iterator<Item = Node<'a, 'i>> + 'n
where
    'a: 'n,
{
    node.children().filter(move |node| node.is_element() && local(*node) == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::threemf::crc32;
    use crate::testing::Noise;

    /// How an entry is put into a test package.
    #[derive(Clone, Copy, PartialEq)]
    enum How {
        Stored,
        Deflated,
        /// Deflated, written the way a streaming writer has to: general purpose
        /// flag bit 3 set, the local header's sizes and CRC left at zero, and a
        /// data descriptor after the data. The central directory still holds
        /// the truth, which is the whole point of the case.
        DeflatedStreaming,
    }

    /// Build a ZIP the way a real writer would, rather than the way ours does.
    fn package(parts: &[(&str, &[u8], How)]) -> Vec<u8> {
        let mut body: Vec<u8> = Vec::new();
        let mut central: Vec<(String, u32, u32, u32, u32, u16, u16)> = Vec::new();

        for (name, data, how) in parts {
            let offset = body.len() as u32;
            let crc = crc32(data);
            let uncompressed = data.len() as u32;
            let (method, packed) = match how {
                How::Stored => (0u16, data.to_vec()),
                How::Deflated | How::DeflatedStreaming => (
                    8u16,
                    yazi::compress(data, yazi::Format::Raw, yazi::CompressionLevel::Default)
                        .unwrap(),
                ),
            };
            let compressed = packed.len() as u32;
            let streaming = *how == How::DeflatedStreaming;
            let flags: u16 = if streaming { 1 << 3 } else { 0 };

            body.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            body.extend_from_slice(&20u16.to_le_bytes());
            body.extend_from_slice(&flags.to_le_bytes());
            body.extend_from_slice(&method.to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&if streaming { 0 } else { crc }.to_le_bytes());
            body.extend_from_slice(&if streaming { 0 } else { compressed }.to_le_bytes());
            body.extend_from_slice(&if streaming { 0 } else { uncompressed }.to_le_bytes());
            body.extend_from_slice(&(name.len() as u16).to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(name.as_bytes());
            body.extend_from_slice(&packed);
            if streaming {
                body.extend_from_slice(&0x0807_4b50u32.to_le_bytes());
                body.extend_from_slice(&crc.to_le_bytes());
                body.extend_from_slice(&compressed.to_le_bytes());
                body.extend_from_slice(&uncompressed.to_le_bytes());
            }

            central.push((
                (*name).to_string(),
                crc,
                compressed,
                uncompressed,
                offset,
                method,
                flags,
            ));
        }

        let directory_offset = body.len() as u32;
        for (name, crc, compressed, uncompressed, offset, method, flags) in &central {
            body.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            body.extend_from_slice(&20u16.to_le_bytes());
            body.extend_from_slice(&20u16.to_le_bytes());
            body.extend_from_slice(&flags.to_le_bytes());
            body.extend_from_slice(&method.to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&crc.to_le_bytes());
            body.extend_from_slice(&compressed.to_le_bytes());
            body.extend_from_slice(&uncompressed.to_le_bytes());
            body.extend_from_slice(&(name.len() as u16).to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&0u16.to_le_bytes());
            body.extend_from_slice(&0u32.to_le_bytes());
            body.extend_from_slice(&offset.to_le_bytes());
            body.extend_from_slice(name.as_bytes());
        }
        let directory_size = body.len() as u32 - directory_offset;

        body.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body.extend_from_slice(&(central.len() as u16).to_le_bytes());
        body.extend_from_slice(&(central.len() as u16).to_le_bytes());
        body.extend_from_slice(&directory_size.to_le_bytes());
        body.extend_from_slice(&directory_offset.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body
    }

    fn rels(target: &str) -> String {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <Relationships \
             xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n\
             <Relationship Target=\"{target}\" Id=\"rel0\" Type=\"{MODEL_RELATIONSHIP}\"/>\n\
             </Relationships>\n"
        )
    }

    /// A unit tetrahedron, which is the smallest thing with a volume and so the
    /// smallest thing whose winding can be observed.
    const TETRAHEDRON: &str = "\
   <vertices>
    <vertex x=\"0\" y=\"0\" z=\"0\"/>
    <vertex x=\"1\" y=\"0\" z=\"0\"/>
    <vertex x=\"0\" y=\"1\" z=\"0\"/>
    <vertex x=\"0\" y=\"0\" z=\"1\"/>
   </vertices>
   <triangles>
    <triangle v1=\"0\" v2=\"2\" v3=\"1\"/>
    <triangle v1=\"0\" v2=\"1\" v3=\"3\"/>
    <triangle v1=\"0\" v2=\"3\" v3=\"2\"/>
    <triangle v1=\"1\" v2=\"2\" v3=\"3\"/>
   </triangles>";

    /// A model part with the given `<model>` attributes, resources and build.
    fn model(attributes: &str, resources: &str, build: &str) -> String {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <model xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\" \
             {attributes}>\n\
             <resources>\n{resources}</resources>\n\
             <build>\n{build}</build>\n\
             </model>\n"
        )
    }

    /// One object holding the tetrahedron, built once with `transform`.
    fn one_tetrahedron(attributes: &str, transform: &str) -> String {
        model(
            attributes,
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            &format!("  <item objectid=\"1\" {transform}/>\n"),
        )
    }

    /// Wrap a model part into a package, deflated, at the conventional name.
    fn packaged(model: &str) -> Vec<u8> {
        let relationships = rels("/3D/3dmodel.model");
        package(&[
            ("_rels/.rels", relationships.as_bytes(), How::Deflated),
            ("3D/3dmodel.model", model.as_bytes(), How::Deflated),
        ])
    }

    fn corners_of(mesh: &ExportMesh, triangle: usize) -> [Vec3; 3] {
        mesh.triangles[triangle].map(|index| mesh.positions[index as usize])
    }

    /// The winding a correctly read tetrahedron has, used to prove that a
    /// mirrored instance is not silently accepted with the opposite one.
    #[test]
    fn the_corner_order_of_a_triangle_is_kept_as_written() {
        let mesh = read(&packaged(&one_tetrahedron("", ""))).unwrap();
        let [a, b, c] = corners_of(&mesh, 0);
        assert_eq!([a, b, c], [Vec3::ZERO, Vec3::Y, Vec3::X], "the corners were reordered");
    }

    #[test]
    fn a_deflated_package_is_inflated_rather_than_only_a_stored_one() {
        // Every 3MF in the wild is deflated. Our own writer stores, so testing
        // against its output alone would leave the whole inflate path unrun.
        let bytes = packaged(&one_tetrahedron("unit=\"millimeter\"", ""));
        let mesh = read(&bytes).unwrap();
        assert_eq!(mesh.triangles.len(), 4);
        assert_eq!(mesh.positions.len(), 4);
    }

    #[test]
    fn a_stored_package_is_read_too() {
        let relationships = rels("/3D/3dmodel.model");
        let model = one_tetrahedron("", "");
        let bytes = package(&[
            ("_rels/.rels", relationships.as_bytes(), How::Stored),
            ("3D/3dmodel.model", model.as_bytes(), How::Stored),
        ]);
        assert_eq!(read(&bytes).unwrap().triangles.len(), 4);
    }

    /// Flag bit 3 zeroes the local header's sizes, so a reader that takes them
    /// from there sees an empty entry and produces an empty model.
    #[test]
    fn an_entry_with_a_data_descriptor_is_read_from_the_central_directory() {
        let relationships = rels("/3D/3dmodel.model");
        let model = one_tetrahedron("", "");
        let bytes = package(&[
            ("_rels/.rels", relationships.as_bytes(), How::DeflatedStreaming),
            ("3D/3dmodel.model", model.as_bytes(), How::DeflatedStreaming),
        ]);
        assert_eq!(
            read(&bytes).unwrap().triangles.len(),
            4,
            "the sizes were taken from the local header, which a streaming writer zeroes"
        );
    }

    #[test]
    fn the_model_part_is_found_through_the_relationships_rather_than_by_name() {
        let relationships = rels("/somewhere/else.model");
        let model = one_tetrahedron("", "");
        let bytes = package(&[
            ("_rels/.rels", relationships.as_bytes(), How::Deflated),
            ("somewhere/else.model", model.as_bytes(), How::Deflated),
            // A decoy at the conventional name, holding a different mesh, so
            // that a reader which hardcodes the path fails loudly here.
            ("3D/3dmodel.model", b"<model/>", How::Deflated),
        ]);
        assert_eq!(read(&bytes).unwrap().triangles.len(), 4);
    }

    #[test]
    fn a_model_in_metres_is_scaled_to_millimetres() {
        let bytes = packaged(&one_tetrahedron("unit=\"meter\"", ""));
        let mesh = read(&bytes).unwrap();
        let far = mesh.positions.iter().fold(0.0f32, |far, position| far.max(position.length()));
        assert_eq!(far, 1000.0, "a metre did not become a thousand millimetres");
    }

    #[test]
    fn an_inch_becomes_twenty_five_point_four_millimetres() {
        let bytes = packaged(&one_tetrahedron("unit=\"inch\"", ""));
        let mesh = read(&bytes).unwrap();
        let far = mesh.positions.iter().fold(0.0f32, |far, position| far.max(position.length()));
        assert!((far - 25.4).abs() < 1e-3, "an inch came out as {far} mm");
    }

    #[test]
    fn a_unit_this_reader_does_not_know_is_refused_rather_than_assumed_to_be_millimetres() {
        let bytes = packaged(&one_tetrahedron("unit=\"furlong\"", ""));
        assert!(matches!(read(&bytes), Err(ImportError::Malformed(_))));
    }

    /// The test the transform convention lives or dies by. A translation alone
    /// passes under either reading of the twelve numbers; a rotation does not.
    #[test]
    fn a_build_item_transform_rotates_as_well_as_translates() {
        // A quarter turn about Z, then a translation. In 3MF's row-major 4x3
        // the rotation rows are the images of the basis vectors, so X goes to
        // +Y: the first three numbers are `0 1 0`.
        let transform = "transform=\"0 1 0 -1 0 0 0 0 1 10 20 30\"";
        let bytes = packaged(&one_tetrahedron("", transform));
        let mesh = read(&bytes).unwrap();

        let expected = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0))
            * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        for corner in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
            let wanted = expected.transform_point3(corner);
            assert!(
                mesh.positions.iter().any(|position| position.distance(wanted) < 1e-4),
                "no vertex landed at {wanted}, so the transform was read transposed"
            );
        }
    }

    #[test]
    fn a_transform_that_is_not_twelve_numbers_is_refused() {
        // A sixteen number transform is what a writer produces when it exports
        // a full 4x4 by mistake. Silently taking the first twelve of those
        // would place every instance somewhere plausible and wrong.
        for transform in [
            "transform=\"1 0 0 0 1 0 0 0 1\"",
            "transform=\"1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1\"",
            "transform=\"1 0 0 0 1 0 0 0 1 0 0 nope\"",
        ] {
            let bytes = packaged(&one_tetrahedron("", transform));
            assert!(
                matches!(read(&bytes), Err(ImportError::Malformed(_))),
                "`{transform}` was accepted"
            );
        }
    }

    #[test]
    fn nested_component_transforms_compose_outermost_last() {
        // Object 2 is object 1 moved along X, and the build moves object 2
        // along Y. A reader that composes the other way puts the result at
        // (0, 5) and (3, 0) rather than at (3, 5).
        let resources = format!(
            "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n\
             \x20 <object id=\"2\" type=\"model\">\n   <components>\n    \
             <component objectid=\"1\" transform=\"1 0 0 0 1 0 0 0 1 3 0 0\"/>\n   \
             </components>\n  </object>\n"
        );
        let text = model(
            "",
            &resources,
            "  <item objectid=\"2\" transform=\"1 0 0 0 1 0 0 0 1 0 5 0\"/>\n",
        );
        let mesh = read(&packaged(&text)).unwrap();
        assert!(
            mesh.positions
                .iter()
                .any(|position| position.distance(Vec3::new(3.0, 5.0, 0.0)) < 1e-4),
            "the component and item transforms did not compose: {:?}",
            mesh.positions
        );
    }

    #[test]
    fn several_build_items_are_merged_into_one_mesh() {
        let text = model(
            "",
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "  <item objectid=\"1\"/>\n  \
             <item objectid=\"1\" transform=\"1 0 0 0 1 0 0 0 1 100 0 0\"/>\n",
        );
        let mesh = read(&packaged(&text)).unwrap();
        assert_eq!(mesh.triangles.len(), 8, "the second build item was dropped");
        assert_eq!(mesh.positions.len(), 8, "the two instances were welded together");
    }

    /// A mirror keeps the same three points and reverses which side is out, so
    /// the winding has to be reversed with it. The direct measure of getting
    /// that wrong is the sign of the volume the surface encloses: a solid
    /// turned inside out encloses a negative one, and the voxeliser would read
    /// it as a hole in space rather than as a body.
    #[test]
    fn a_mirroring_transform_flips_the_winding_so_the_solid_is_not_inside_out() {
        fn enclosed_volume(mesh: &ExportMesh) -> f32 {
            mesh.triangles
                .iter()
                .map(|corners| {
                    let [a, b, c] = corners.map(|index| mesh.positions[index as usize]);
                    a.dot(b.cross(c)) / 6.0
                })
                .sum()
        }

        let plain = read(&packaged(&one_tetrahedron("", ""))).unwrap();
        assert!(
            (enclosed_volume(&plain) - 1.0 / 6.0).abs() < 1e-5,
            "the fixture itself is not wound outwards, so this test would prove nothing"
        );

        let transform = "transform=\"-1 0 0 0 1 0 0 0 1 0 0 0\"";
        let mirrored = read(&packaged(&one_tetrahedron("", transform))).unwrap();
        let volume = enclosed_volume(&mirrored);
        assert!(
            (volume - 1.0 / 6.0).abs() < 1e-5,
            "a mirrored instance encloses {volume}, so its faces were left pointing inwards"
        );
    }

    #[test]
    fn the_beam_lattice_extension_is_refused_by_name() {
        let attributes = "xmlns:b=\"http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02\" \
                          requiredextensions=\"b\"";
        let error = read(&packaged(&one_tetrahedron(attributes, ""))).unwrap_err();
        assert!(
            matches!(&error, ImportError::Unsupported(why) if why.contains("beam lattice")),
            "expected a refusal naming the extension, got: {error}"
        );
    }

    #[test]
    fn the_slice_extension_is_refused_by_name() {
        let attributes = "xmlns:whatever=\"http://schemas.microsoft.com/3dmanufacturing/slice/2015/07\" \
                          requiredextensions=\"whatever\"";
        let error = read(&packaged(&one_tetrahedron(attributes, ""))).unwrap_err();
        assert!(
            matches!(&error, ImportError::Unsupported(why) if why.contains("slice")),
            "the prefix was matched literally rather than resolved: {error}"
        );
    }

    #[test]
    fn a_required_extension_this_reader_does_not_know_about_is_not_refused() {
        // Materials and production are required by plenty of ordinary files and
        // do not move the geometry, so refusing everything unknown would refuse
        // files this reader handles correctly.
        let attributes = "xmlns:m=\"http://schemas.microsoft.com/3dmanufacturing/material/2015/02\" \
                          requiredextensions=\"m\"";
        assert!(read(&packaged(&one_tetrahedron(attributes, ""))).is_ok());
    }

    /// Package a root part plus one named sub-part, the way the production
    /// extension splits a scene and the way every Bambu Studio and OrcaSlicer
    /// project file on disk is laid out.
    fn packaged_with_part(root: &str, name: &str, part: &str) -> Vec<u8> {
        let relationships = rels("/3D/3dmodel.model");
        package(&[
            ("_rels/.rels", relationships.as_bytes(), How::Deflated),
            ("3D/3dmodel.model", root.as_bytes(), How::Deflated),
            (name, part.as_bytes(), How::Deflated),
        ])
    }

    /// The production namespace declaration, which every such file carries.
    const PRODUCTION: &str =
        "xmlns:p=\"http://schemas.microsoft.com/3dmanufacturing/production/2015/06\"";

    #[test]
    fn geometry_in_another_model_part_is_followed() {
        // The layout of `Howling-Wolf-solid.3mf` and of every project file
        // Bambu Studio or OrcaSlicer writes: a root part of a few hundred bytes
        // whose object is a single component pointing into the part that
        // actually holds the mesh.
        let root = model(
            PRODUCTION,
            "  <object id=\"2\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/Objects/wolf.model\" objectid=\"1\"/>\n   \
             </components>\n  </object>\n",
            "  <item objectid=\"2\"/>\n",
        );
        let part = model(
            PRODUCTION,
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "",
        );

        let mesh = read(&packaged_with_part(&root, "3D/Objects/wolf.model", &part))
            .expect("a package split across parts should read");
        assert_eq!(mesh.triangles.len(), 4, "the tetrahedron did not come through");
        assert_eq!(mesh.positions.len(), 4);
    }

    #[test]
    fn a_sub_part_needs_no_build_of_its_own() {
        // Only the root's build is the scene. A sub-part written by a slicer
        // usually has no build at all, and the single-part reader required a
        // non-empty one -- so this is the check that the requirement was moved
        // rather than merely relaxed everywhere.
        let root = model(
            PRODUCTION,
            "  <object id=\"9\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\"/>\n   \
             </components>\n  </object>\n",
            "  <item objectid=\"9\"/>\n",
        );
        // No <build> element at all.
        let part = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <model xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\" \
             {PRODUCTION}>\n<resources>\n  \
             <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n\
             </resources>\n</model>\n"
        );
        assert!(read(&packaged_with_part(&root, "3D/sub.model", &part)).is_ok());
    }

    #[test]
    fn a_transform_composes_across_a_part_boundary() {
        // The multi-part twin of `nested_component_transforms_compose_outermost_last`,
        // and the reason it matters here: the wolf's build item carries a 2x
        // scale that has to reach geometry living in a different file.
        let root = model(
            PRODUCTION,
            "  <object id=\"2\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\" \
             transform=\"1 0 0 0 1 0 0 0 1 10 0 0\"/>\n   </components>\n  </object>\n",
            "  <item objectid=\"2\" transform=\"2 0 0 0 2 0 0 0 2 0 0 0\"/>\n",
        );
        let part = model(
            PRODUCTION,
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "",
        );

        let mesh = read(&packaged_with_part(&root, "3D/sub.model", &part)).expect("it should read");
        // The component shifts by 10 and the item then doubles everything, so
        // the origin vertex lands at 20 rather than at 10.
        let origin = mesh
            .positions
            .iter()
            .find(|p| p.y.abs() < 1.0e-5 && p.z.abs() < 1.0e-5)
            .expect("the tetrahedron's origin corner");
        assert!(
            (origin.x - 20.0).abs() < 1.0e-4,
            "the outer transform did not reach across the part boundary: {origin}"
        );
    }

    #[test]
    fn object_ids_are_local_to_their_part() {
        // Ids are unique only within a part, and both parts here call their
        // object 1. Flattening every part into one map would make the root's
        // own object 1 shadow the sub-part's, and the import would silently
        // come out as the wrong geometry rather than as an error.
        let root = model(
            PRODUCTION,
            // The root has its own object 1 -- a mesh, not the one wanted.
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n  \
                 <object id=\"2\" type=\"model\">\n   <components>\n    \
                 <component p:path=\"/3D/sub.model\" objectid=\"1\"/>\n   </components>\n  </object>\n"
            ),
            "  <item objectid=\"2\"/>\n",
        );
        // The sub-part's object 1 is a single triangle, so it is telling apart
        // from the root's four.
        let part = model(
            PRODUCTION,
            "  <object id=\"1\" type=\"model\">\n   <mesh>\n   <vertices>\n    \
             <vertex x=\"0\" y=\"0\" z=\"0\"/>\n    <vertex x=\"5\" y=\"0\" z=\"0\"/>\n    \
             <vertex x=\"0\" y=\"5\" z=\"0\"/>\n   </vertices>\n   <triangles>\n    \
             <triangle v1=\"0\" v2=\"1\" v3=\"2\"/>\n   </triangles>\n   </mesh>\n  </object>\n",
            "",
        );

        let mesh = read(&packaged_with_part(&root, "3D/sub.model", &part)).expect("it should read");
        assert_eq!(
            mesh.triangles.len(),
            1,
            "the root's object 1 was used instead of the sub-part's"
        );
    }

    #[test]
    fn a_cycle_across_two_parts_is_refused() {
        // The cycle guard keys on (part, id). On the id alone this would either
        // recurse until the stack ran out, or -- with the naive fix -- report
        // the legitimate A:1 -> B:1 hop as an object containing itself.
        let root = model(
            PRODUCTION,
            "  <object id=\"1\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\"/>\n   </components>\n  </object>\n",
            "  <item objectid=\"1\"/>\n",
        );
        let part = model(
            PRODUCTION,
            "  <object id=\"1\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/3dmodel.model\" objectid=\"1\"/>\n   </components>\n  </object>\n",
            "",
        );
        assert!(
            matches!(
                read(&packaged_with_part(&root, "3D/sub.model", &part)),
                Err(ImportError::Malformed(_))
            ),
            "a cycle that hops between parts has to be an error, not a stack overflow"
        );
    }

    #[test]
    fn the_same_id_in_two_parts_is_not_a_cycle() {
        // The control for the test above: keying the stack on the id alone
        // would refuse this, and it is exactly what the wolf does.
        let root = model(
            PRODUCTION,
            "  <object id=\"1\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\"/>\n   </components>\n  </object>\n",
            "  <item objectid=\"1\"/>\n",
        );
        let part = model(
            PRODUCTION,
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "",
        );
        assert!(
            read(&packaged_with_part(&root, "3D/sub.model", &part)).is_ok(),
            "part A's object 1 pointing at part B's object 1 is not a cycle"
        );
    }

    #[test]
    fn a_path_naming_a_part_that_is_not_there_is_refused_rather_than_dropped() {
        // The failure this replaces the old blanket refusal with. Silently
        // emitting nothing would hand back a model missing a piece, with a
        // success message.
        let text = model(
            PRODUCTION,
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "  <item objectid=\"1\" p:path=\"/3D/other.model\"/>\n",
        );
        let error = read(&packaged(&text)).expect_err("a missing part must not read as a success");
        assert!(
            matches!(error, ImportError::Malformed(ref message) if message.contains("other.model")),
            "the error should name the part that is missing: {error}"
        );
    }

    #[test]
    fn model_parts_that_disagree_about_units_are_refused() {
        // Either answer -- honour the sub-part's unit, or inherit the root's --
        // is a guess, and a guess about units is the thousand-times-too-small
        // import this module's header warns about.
        let root = model(
            &format!("{PRODUCTION} unit=\"millimeter\""),
            "  <object id=\"2\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\"/>\n   </components>\n  </object>\n",
            "  <item objectid=\"2\"/>\n",
        );
        let part = model(
            &format!("{PRODUCTION} unit=\"meter\""),
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "",
        );
        assert!(matches!(
            read(&packaged_with_part(&root, "3D/sub.model", &part)),
            Err(ImportError::Unsupported(_))
        ));
    }

    #[test]
    fn a_sub_part_requiring_an_unreadable_extension_is_refused_too() {
        // The extension check used to run on the root only. A sub-part is free
        // to declare that its geometry is a strut graph, and that is exactly
        // the part the geometry is coming from.
        let root = model(
            PRODUCTION,
            "  <object id=\"2\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\"/>\n   </components>\n  </object>\n",
            "  <item objectid=\"2\"/>\n",
        );
        let part = model(
            &format!(
                "{PRODUCTION} xmlns:b=\"{}\" requiredextensions=\"b\"",
                REFUSED_EXTENSIONS[0].0
            ),
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "",
        );
        assert!(matches!(
            read(&packaged_with_part(&root, "3D/sub.model", &part)),
            Err(ImportError::Unsupported(_))
        ));
    }

    #[test]
    fn a_part_reached_twice_is_inflated_once() {
        // Two components naming the same part. Reading it per reference would
        // make a small package cost arbitrarily much work, which is the same
        // class of thing the zip bomb ceiling exists to stop -- and the
        // geometry still has to arrive twice.
        let root = model(
            PRODUCTION,
            "  <object id=\"2\" type=\"model\">\n   <components>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\"/>\n    \
             <component p:path=\"/3D/sub.model\" objectid=\"1\" \
             transform=\"1 0 0 0 1 0 0 0 1 50 0 0\"/>\n   </components>\n  </object>\n",
            "  <item objectid=\"2\"/>\n",
        );
        let part = model(
            PRODUCTION,
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "",
        );
        let mesh = read(&packaged_with_part(&root, "3D/sub.model", &part)).expect("it should read");
        assert_eq!(mesh.triangles.len(), 8, "both instances should be present");
    }

    #[test]
    fn an_object_that_contains_itself_is_refused_rather_than_recursed_into() {
        let resources = "  <object id=\"1\" type=\"model\">\n   <components>\n    \
                         <component objectid=\"2\"/>\n   </components>\n  </object>\n  \
                         <object id=\"2\" type=\"model\">\n   <components>\n    \
                         <component objectid=\"1\"/>\n   </components>\n  </object>\n";
        let text = model("", resources, "  <item objectid=\"1\"/>\n");
        assert!(
            matches!(read(&packaged(&text)), Err(ImportError::Malformed(_))),
            "a component cycle has to be an error, not a stack overflow"
        );
    }

    #[test]
    fn a_triangle_index_past_the_end_is_refused_rather_than_wrapping() {
        let resources = "  <object id=\"1\" type=\"model\">\n   <mesh>\n    <vertices>\n     \
                         <vertex x=\"0\" y=\"0\" z=\"0\"/>\n     \
                         <vertex x=\"1\" y=\"0\" z=\"0\"/>\n     \
                         <vertex x=\"0\" y=\"1\" z=\"0\"/>\n    </vertices>\n    <triangles>\n     \
                         <triangle v1=\"0\" v2=\"1\" v3=\"9\"/>\n    </triangles>\n   \
                         </mesh>\n  </object>\n";
        let text = model("", resources, "  <item objectid=\"1\"/>\n");
        assert!(matches!(read(&packaged(&text)), Err(ImportError::Malformed(_))));
    }

    #[test]
    fn a_build_item_naming_an_object_that_is_not_there_is_refused() {
        let text = model(
            "",
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "  <item objectid=\"7\"/>\n",
        );
        assert!(matches!(read(&packaged(&text)), Err(ImportError::Malformed(_))));
    }

    #[test]
    fn an_empty_build_says_so_rather_than_reporting_an_empty_mesh() {
        let text = model(
            "",
            &format!(
                "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n"
            ),
            "",
        );
        let error = read(&packaged(&text)).unwrap_err();
        assert!(matches!(&error, ImportError::Malformed(why) if why.contains("build")), "{error}");
    }

    #[test]
    fn a_package_with_no_model_relationship_is_refused() {
        let bytes = package(&[("_rels/.rels", b"<Relationships/>", How::Deflated)]);
        assert!(matches!(read(&bytes), Err(ImportError::Malformed(_))));
    }

    #[test]
    fn something_that_is_not_a_zip_at_all_is_refused_rather_than_panicking() {
        assert!(read(b"not a zip").is_err());
        assert!(read(&[]).is_err());
        assert!(read(&[0u8; 5000]).is_err());
    }

    #[test]
    fn a_truncated_package_does_not_panic() {
        let whole = packaged(&one_tetrahedron("", ""));
        for keep in [1, 8, 40, whole.len() / 3, whole.len() / 2, whole.len() - 4] {
            let cut = &whole[..keep.min(whole.len())];
            // Some cuts land on a valid prefix and some do not; what matters is
            // that none of them reach a slice out of bounds.
            let _ = read(cut);
        }
    }

    /// A package whose central directory declares more than the bomb ceiling
    /// has to be refused before anything is allocated for it.
    #[test]
    fn a_package_that_declares_a_huge_unpacked_size_is_refused_before_it_inflates() {
        let mut bytes = packaged(&one_tetrahedron("", ""));
        // Rewrite the uncompressed size of the last central directory entry.
        let central = bytes
            .windows(4)
            .rposition(|window| window == 0x0201_4b50u32.to_le_bytes())
            .expect("the fixture has a central directory");
        let huge = (3u64 * 1024 * 1024 * 1024) as u32;
        bytes[central + 24..central + 28].copy_from_slice(&huge.to_le_bytes());

        let error = read(&bytes).unwrap_err();
        assert!(
            matches!(&error, ImportError::Malformed(why) if why.contains("unpacks")),
            "expected the bomb ceiling to fire, got: {error}"
        );
    }

    /// The ceiling on the total is only half of it. An entry that declares a
    /// modest size and then inflates past it has to be stopped at the
    /// declaration, or the ceiling is a number nothing enforces.
    #[test]
    fn an_entry_that_inflates_past_its_declared_size_is_stopped_at_the_declaration() {
        let mut bytes = packaged(&one_tetrahedron("", ""));
        let central = bytes
            .windows(4)
            .rposition(|window| window == 0x0201_4b50u32.to_le_bytes())
            .expect("the fixture has a central directory");
        bytes[central + 24..central + 28].copy_from_slice(&16u32.to_le_bytes());

        let error = read(&bytes).unwrap_err();
        assert!(
            matches!(&error, ImportError::Malformed(why) if why.contains("zip bomb")),
            "expected the inflate to stop at the declared size, got: {error}"
        );
    }

    #[test]
    fn a_part_whose_checksum_does_not_match_is_refused() {
        let relationships = rels("/3D/3dmodel.model");
        let model = one_tetrahedron("", "");
        let mut bytes = package(&[
            ("_rels/.rels", relationships.as_bytes(), How::Stored),
            ("3D/3dmodel.model", model.as_bytes(), How::Stored),
        ]);
        // Corrupt a byte of the stored model, well past both headers.
        let at = bytes.len() / 2;
        bytes[at] ^= 0xff;
        assert!(read(&bytes).is_err(), "a corrupt part was accepted");
    }

    /// The one that proves this reader inverts the writer this project ships,
    /// rather than merely agreeing with the test author about what 3MF is.
    /// Until it existed, `export::threemf` was only checked by counting
    /// signatures in its own output.
    #[test]
    fn our_own_exported_3mf_reads_back_with_the_same_geometry() {
        let mut volume = crate::volume::Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 8.0);
        volume.mark_everything_dirty();
        let (exported, report) = volume.export_mesh();
        assert!(report.is_printable(), "the fixture is not a good mesh: {}", report.summary());

        let mut bytes = Vec::new();
        crate::export::threemf::write(&exported, &mut bytes).unwrap();
        let read_back = read(&bytes).unwrap();

        assert_eq!(
            read_back.triangles.len(),
            exported.triangles.len(),
            "the triangle count changed across a write and a read"
        );
        assert_eq!(
            read_back.positions.len(),
            exported.positions.len(),
            "3MF already shares its vertices, so welding should recover exactly them"
        );
    }

    #[test]
    fn a_corrupted_model_part_is_answered_rather_than_survived() {
        // `tests/hostile_meshes.rs` corrupts whole files, which for 3MF means
        // corrupting a deflate stream under a CRC: the container catches most
        // of it and the XML underneath is barely reached. This corrupts the
        // XML and then packages it correctly, so the container is never the
        // thing that objects and the parser is always the thing under test.
        //
        // The property is the same one: every input gets an answer. An index
        // past the end of `vertices` is the failure this is really watching
        // for -- it is one flipped digit away in any real file, and it panics
        // the voxeliser rather than the reader.
        let valid = one_tetrahedron("unit=\"millimeter\"", "");
        let mut accepted = 0usize;
        let mut tried = 0usize;

        for strategy in 0..2 {
            for run in 0..2_000u64 {
                let seed = (strategy as u64) << 32 | run | 1;
                let mut noise = Noise::seeded(seed);
                let mut bytes = valid.clone().into_bytes();

                if strategy == 0 {
                    // Anywhere, any printable byte. Most of these break well
                    // formedness and are refused by the XML parser, which is
                    // the container's job done properly rather than a gap.
                    for _ in 0..1 + noise.below(6) {
                        let at = noise.below(bytes.len());
                        bytes[at] = 0x20 + (noise.below(95) as u8);
                    }
                } else {
                    // Digits only, which leaves the markup well formed and
                    // lands every mutation on an index, a coordinate or an
                    // object id. This is the strategy that reaches the
                    // reader's own reasoning, and the one that would find an
                    // unchecked `v1`.
                    let digits: Vec<usize> = bytes
                        .iter()
                        .enumerate()
                        .filter(|(_, byte)| byte.is_ascii_digit())
                        .map(|(at, _)| at)
                        .collect();
                    if digits.is_empty() {
                        continue;
                    }
                    for _ in 0..1 + noise.below(4) {
                        let at = digits[noise.below(digits.len())];
                        bytes[at] = b'0' + (noise.below(10) as u8);
                    }
                }

                let Ok(text) = String::from_utf8(bytes) else {
                    continue;
                };

                tried += 1;
                if let Ok(mesh) = read(&packaged(&text)) {
                    accepted += 1;
                    for (index, triangle) in mesh.triangles.iter().enumerate() {
                        for corner in triangle {
                            assert!(
                                (*corner as usize) < mesh.positions.len(),
                                "strategy {strategy} run {run}: triangle {index} refers to \
                                 vertex {corner} of {}",
                                mesh.positions.len()
                            );
                        }
                    }
                    for position in &mesh.positions {
                        assert!(
                            position.is_finite(),
                            "strategy {strategy} run {run}: {position:?}"
                        );
                    }
                }
            }
        }

        // The control: without it a reader that rejected every mutant would
        // pass this without the parser ever having run. Measured at 814 of
        // 4000 when this was written -- 152 of 2000 for the scattergun
        // strategy, which mostly breaks well formedness, against 662 of 2000
        // for the digits, which by construction does not. The floor is set
        // under the scattergun rate on its own, so it holds even if the XML
        // parser is one day swapped for a stricter one.
        assert!(
            accepted > tried / 10,
            "only {accepted} of {tried} corrupted parts got as far as being parsed, \
             so this is measuring the container and not the reader"
        );
    }

    #[test]
    fn nesting_far_past_the_component_guard_is_refused_rather_than_overflowing() {
        // `MAX_COMPONENT_DEPTH` exists so a package cannot drive the component
        // walk as deep as it likes. This drives it two orders of magnitude
        // past the guard: the guard has to be what stops it, because the
        // alternative is a stack overflow, which aborts the process outright
        // and cannot be caught or reported.
        let deep = 8_000;
        let mut resources = String::from(
            "  <object id=\"1\" type=\"model\">\n   <mesh>\n{TETRAHEDRON}\n   </mesh>\n  </object>\n",
        );
        resources = resources.replace("{TETRAHEDRON}", TETRAHEDRON);
        for id in 2..=deep {
            resources.push_str(&format!(
                "  <object id=\"{id}\" type=\"model\">\n   <components>\n    \
                 <component objectid=\"{}\"/>\n   </components>\n  </object>\n",
                id - 1
            ));
        }
        let part = model("", &resources, &format!("  <item objectid=\"{deep}\"/>\n"));

        let Err(error) = read(&packaged(&part)) else {
            panic!("nesting far past the guard should be refused, not read");
        };
        assert!(
            format!("{error}").contains("deep"),
            "refused for the wrong reason, which means the guard is not what stopped it: {error}"
        );
    }
}
