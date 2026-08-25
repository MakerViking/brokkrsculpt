// SPDX-License-Identifier: AGPL-3.0-only

//! 3MF, the format that actually states its units.
//!
//! A 3MF is a ZIP container holding a little XML. Unlike STL it shares vertices,
//! and unlike both STL and OBJ it says outright that the model is in
//! millimetres, so there is nothing for a slicer to guess.
//!
//! # Why the ZIP is written by hand
//!
//! Only a handful of small entries are needed and 3MF permits them stored
//! rather than deflated, so the whole container is a few hundred lines of well
//! specified header layout and one CRC. That is a poor trade against pulling in
//! a zip crate and a compression crate, and the format is fixed forever, so
//! there is nothing here to keep up with.
//!
//! What this writer produces is a deliberately degenerate subset of the format:
//! stored entries only, one namespace, always millimetres, and no transforms --
//! a body's position IS its brick occupancy, so there is nothing for a
//! transform to say. [`crate::import::threemf`] reads it back in the tests,
//! which is what stops this module being checked only against its own
//! assumptions -- but that reader is not tested against this writer alone,
//! precisely because agreeing with it would prove nothing about a real file.
//!
//! # Colour is not in the specification's terms
//!
//! Two of the five entries -- `Metadata/model_settings.config` and
//! `Metadata/project_settings.config` -- are not 3MF at all. They are the
//! slicer's own sidecars, and together with the `paint_color` attribute on each
//! `<triangle>` they are how per-filament colour actually travels. The
//! specification's materials extension is not read by the target and is not
//! written here; see [`PAINT_CODE`] for the evidence and the encoding.
//!
//! **Verified against OrcaSlicer 2.4.0-alpha on 2026-08-20**, not merely
//! believed: a four-band sphere written by `a_banded_sphere_is_written_for_a_slicer_to_judge`
//! opens with its bands on filaments 1 to 4, the unpainted cap on the base
//! filament, and the four slot colours taken from our settings part.
//!
//! Coordinates are rotated to Z-up on the way out. See [`crate::orientation`].
//! 3MF carries no vertex normals, so positions are all there is to rotate.

use std::io::{self, Write};

use super::ExportMesh;
use crate::orientation::to_print_space;

/// The core specification namespace, which every consumer keys off.
const CORE_NAMESPACE: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";

/// The filament slot a triangle is printed with, as the slicers in the
/// PrusaSlicer lineage encode it.
///
/// # Why not the specification's own materials extension
///
/// Because it is not what the target reads. Measured against two real
/// multi-colour projects -- `Happy_Piglet_Multi-Color.3mf` (46.9 MB of model
/// XML) and `1plate+3color+3MF.3mf` -- there are **zero** occurrences of
/// `basematerials`, `colorgroup`, `texture2d` or `pid=` between them. Not rare:
/// absent. What they carry is a `paint_color` attribute on nearly every
/// `<triangle>`. Writing the specification's route instead produces a file that
/// validates perfectly and prints in one colour.
///
/// # The encoding
///
/// From PrusaSlicer's `TriangleSelector::serialize`, which BambuStudio and
/// OrcaSlicer inherit: nibbles are consumed in reverse string order and bits
/// LSB-first. Two bits of split count (always `00` here -- see below), then two
/// bits of state, and a state of 3 or more escapes to `11` followed by four
/// bits of *state - 3*.
///
/// **This writer only ever emits leaf codes.** The long strings in real files
/// (`"84886844AA84886844AA848828823"`) are a subdivision tree for painting
/// *finer* than a source triangle. Surface nets already produces triangles
/// smaller than any paint feature, so there is nothing to subdivide -- a
/// structural advantage of a voxel sculptor over a mesh painter, and the reason
/// the recursive encoder is not here. A *reader* would still need it.
///
/// Verified by decoding both files above; the codes for slots 1-4 appear there
/// as `"4"`, `"8"`, `"0C"` and `"1C"`.
const PAINT_CODE: [&str; 16] =
    ["4", "8", "0C", "1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "AC", "BC", "CC", "DC"];

/// The code for a 1-based filament slot, or `None` when the slot is unassigned
/// or out of range.
fn paint_code(slot: u8) -> Option<&'static str> {
    if slot == 0 { None } else { PAINT_CODE.get(slot as usize - 1).copied() }
}

/// The filament slots a package declares, so a slicer opening it shows the
/// colours the sculpt was painted with rather than whatever is loaded.
///
/// **Slot number is the contract; the colours here are a hint.** A 3MF carries
/// which filament prints a triangle, not what colour it is, so if the slicer
/// ignores this the assignment still lands -- the user just sees their own
/// slot colours instead of ours.
#[derive(Debug, Clone)]
pub struct Filaments {
    /// `#RRGGBB` per slot, in slot order.
    pub colours: Vec<String>,
    /// Material per slot, in slot order. Padded with `PLA` when short.
    pub materials: Vec<String>,
    /// The slot an unpainted triangle prints with, 1-based.
    pub base: u8,
}

impl Default for Filaments {
    /// Four slots, the U1's count, in colours that are obvious when they land
    /// on the wrong one.
    fn default() -> Self {
        Self {
            colours: ["#FFFFFF", "#FF0000", "#00B050", "#2850E0"].map(str::to_string).to_vec(),
            materials: vec!["PLA".to_string(); 4],
            base: 1,
        }
    }
}

/// Build the model XML for a document's bodies.
///
/// Coordinates are written with enough digits to round trip an `f32` exactly,
/// because a 3MF is often the archival copy of a model.
///
/// **Each body is its own `<object>` with its own `<item>` in the build**, and
/// the ids are 1-based positions in the list. That is the shape a slicer needs
/// to keep them as separate parts on the plate: one object with several
/// disjoint shells loads as a single part that cannot be moved, assigned a
/// filament, or deleted independently. Indices stay per object -- 3MF numbers
/// vertices within an object, unlike OBJ -- so there is no running offset here
/// and adding one would be the bug.
fn model_xml(bodies: &[(&str, &ExportMesh)]) -> String {
    let capacity: usize =
        bodies.iter().map(|(_, mesh)| mesh.positions.len() * 48 + mesh.triangles.len() * 40).sum();
    let mut xml = String::with_capacity(capacity);
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str(&format!(
        "<model unit=\"millimeter\" xml:lang=\"en-US\" xmlns=\"{CORE_NAMESPACE}\">\n"
    ));
    xml.push_str(" <metadata name=\"Application\">BrokkrSculpt</metadata>\n");
    xml.push_str(" <resources>\n");

    for (index, (_, mesh)) in bodies.iter().enumerate() {
        let id = index + 1;
        xml.push_str(&format!("  <object id=\"{id}\" type=\"model\">\n   <mesh>\n"));

        xml.push_str("    <vertices>\n");
        for position in &mesh.positions {
            let position = to_print_space(*position);
            xml.push_str(&format!(
                "     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
                position.x, position.y, position.z
            ));
        }
        xml.push_str("    </vertices>\n");

        xml.push_str("    <triangles>\n");
        for [a, b, c] in &mesh.triangles {
            // The triangle's filament is the slot of its FIRST corner, which is
            // the same rule the renderer's provoking vertex uses. Picking a
            // different corner here would let the preview and the file disagree
            // at every colour boundary, and nothing would catch it but the eye.
            let slot = mesh.slots.get(*a as usize).copied().unwrap_or(0);
            match paint_code(slot) {
                Some(code) => xml.push_str(&format!(
                    "     <triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\" paint_color=\"{code}\"/>\n"
                )),
                // Unassigned. The object's own `extruder` metadata decides,
                // which is what an unpainted triangle means in a real file.
                None => {
                    xml.push_str(&format!("     <triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\"/>\n"))
                }
            }
        }
        xml.push_str("    </triangles>\n");

        xml.push_str("   </mesh>\n  </object>\n");
    }
    xml.push_str(" </resources>\n");

    xml.push_str(" <build>\n");
    for index in 0..bodies.len() {
        xml.push_str(&format!("  <item objectid=\"{}\"/>\n", index + 1));
    }
    xml.push_str(" </build>\n");
    xml.push_str("</model>\n");
    xml
}

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

const RELATIONSHIPS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

/// Which filament each object prints with when a triangle says nothing, and
/// what each one is called.
///
/// Not part of the 3MF specification: it is the slicer's own sidecar, and the
/// reference for its shape is a real project file rather than a document.
///
/// One `<object>` per body, ids matching [`model_xml`]'s, because the slicer's
/// object list is what shows the names -- a document of eleven bodies all
/// called BrokkrSculpt would be eleven rows nobody could tell apart.
///
/// **`[Content_Types].xml` deliberately does not declare a `config`
/// extension.** The real packages ship these parts undeclared and are read
/// anyway, so declaring one would be inventing a rule the target does not
/// follow.
fn model_settings_xml(bodies: &[(&str, &ExportMesh)], filaments: &Filaments) -> String {
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<config>\n");
    for (index, (name, _)) in bodies.iter().enumerate() {
        xml.push_str(&format!(
            "  <object id=\"{}\">\n    <metadata key=\"name\" value=\"{}\"/>\n    \
             <metadata key=\"extruder\" value=\"{}\"/>\n  </object>\n",
            index + 1,
            escape(name),
            filaments.base.max(1)
        ));
    }
    xml.push_str("</config>\n");
    xml
}

/// The five characters XML cannot carry raw, in an attribute value.
///
/// A body's name is user text: it reaches this writer from a rename box and,
/// through a project file, from whoever made that file. `<` alone would produce
/// a package that no slicer opens, and `"` would end the attribute and let the
/// rest of the name be read as markup -- which is an injection into a document
/// somebody else's parser trusts. The name field is capped at 32 bytes and the
/// reader repairs bad UTF-8, so nothing else about it needs defending here.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(character),
        }
    }
    out
}

/// The slot colours, so the slicer shows what the sculpt was painted with.
///
/// A real one of these is a 437-key versioned preset blob, and coupling to all
/// of it would break on every slicer release. This writes the three keys that
/// carry the answer and nothing else, on the theory that a settings file is
/// merged over the active preset rather than replacing it.
///
/// **Whether that theory holds is exactly what the first export spike
/// measures.** If the slicer refuses a partial file, drop this part entirely
/// and keep [`model_settings_xml`]; the slot assignment still lands and only
/// the colours come from the user's own setup instead of ours.
fn project_settings_json(filaments: &Filaments) -> String {
    let list = |values: &[String]| {
        let quoted: Vec<String> = values.iter().map(|v| format!("\"{v}\"")).collect();
        quoted.join(", ")
    };
    let mut materials = filaments.materials.clone();
    materials.resize(filaments.colours.len(), "PLA".to_string());
    format!(
        "{{\n  \"from\": \"project\",\n  \"filament_colour\": [{}],\n  \"filament_type\": [{}]\n}}\n",
        list(&filaments.colours),
        list(&materials)
    )
}

/// Write `mesh` as a 3MF package, with default filament slots.
pub fn write(mesh: &ExportMesh, out: &mut impl Write) -> io::Result<()> {
    write_with(mesh, &Filaments::default(), out)
}

/// Write `mesh` as a 3MF package, declaring `filaments` as the slot setup.
pub fn write_with(
    mesh: &ExportMesh,
    filaments: &Filaments,
    out: &mut impl Write,
) -> io::Result<()> {
    write_with_all(&[("BrokkrSculpt", mesh)], filaments, out)
}

/// Write a document's bodies as one 3MF package, with default filament slots.
pub fn write_all(bodies: &[(&str, &ExportMesh)], out: &mut impl Write) -> io::Result<()> {
    write_with_all(bodies, &Filaments::default(), out)
}

/// Write a document's bodies as one 3MF package, declaring `filaments` as the
/// slot setup.
///
/// The extension point of this module: every other entry point here is a
/// wrapper over it, so a caller with both several bodies and a filament setup
/// to declare has one call rather than a combination that does not exist.
pub fn write_with_all(
    bodies: &[(&str, &ExportMesh)],
    filaments: &Filaments,
    out: &mut impl Write,
) -> io::Result<()> {
    let model = model_xml(bodies);
    let model_settings = model_settings_xml(bodies, filaments);
    let project_settings = project_settings_json(filaments);
    let mut zip = ZipWriter::new();
    // Order matters to some readers: the content types part has to come first.
    zip.add("[Content_Types].xml", CONTENT_TYPES.as_bytes());
    zip.add("_rels/.rels", RELATIONSHIPS.as_bytes());
    zip.add("3D/3dmodel.model", model.as_bytes());
    zip.add("Metadata/model_settings.config", model_settings.as_bytes());
    zip.add("Metadata/project_settings.config", project_settings.as_bytes());
    out.write_all(&zip.finish())
}

/// A minimal ZIP writer that stores entries without compressing them.
///
/// Everything here is fixed by the ZIP specification: signatures, field order
/// and little endian sizes. The version fields say 2.0, which is what "stored,
/// no encryption, no directories" requires, and the timestamps are left at zero
/// so the same mesh always produces the same bytes.
struct ZipWriter {
    body: Vec<u8>,
    entries: Vec<CentralEntry>,
}

struct CentralEntry {
    name: String,
    crc: u32,
    size: u32,
    offset: u32,
}

impl ZipWriter {
    fn new() -> Self {
        Self { body: Vec::new(), entries: Vec::new() }
    }

    fn add(&mut self, name: &str, data: &[u8]) {
        let offset = self.body.len() as u32;
        let crc = crc32(data);
        let size = data.len() as u32;

        // Local file header.
        self.body.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        self.body.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.body.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.body.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        self.body.extend_from_slice(&0u16.to_le_bytes()); // modification time
        self.body.extend_from_slice(&0u16.to_le_bytes()); // modification date
        self.body.extend_from_slice(&crc.to_le_bytes());
        self.body.extend_from_slice(&size.to_le_bytes()); // compressed size
        self.body.extend_from_slice(&size.to_le_bytes()); // uncompressed size
        self.body.extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.body.extend_from_slice(&0u16.to_le_bytes()); // extra field length
        self.body.extend_from_slice(name.as_bytes());
        self.body.extend_from_slice(data);

        self.entries.push(CentralEntry { name: name.to_string(), crc, size, offset });
    }

    fn finish(mut self) -> Vec<u8> {
        let directory_offset = self.body.len() as u32;

        for entry in &self.entries {
            self.body.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            self.body.extend_from_slice(&20u16.to_le_bytes()); // version made by
            self.body.extend_from_slice(&20u16.to_le_bytes()); // version needed
            self.body.extend_from_slice(&0u16.to_le_bytes()); // flags
            self.body.extend_from_slice(&0u16.to_le_bytes()); // method: stored
            self.body.extend_from_slice(&0u16.to_le_bytes()); // modification time
            self.body.extend_from_slice(&0u16.to_le_bytes()); // modification date
            self.body.extend_from_slice(&entry.crc.to_le_bytes());
            self.body.extend_from_slice(&entry.size.to_le_bytes());
            self.body.extend_from_slice(&entry.size.to_le_bytes());
            self.body.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
            self.body.extend_from_slice(&0u16.to_le_bytes()); // extra field length
            self.body.extend_from_slice(&0u16.to_le_bytes()); // comment length
            self.body.extend_from_slice(&0u16.to_le_bytes()); // starting disk
            self.body.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
            self.body.extend_from_slice(&0u32.to_le_bytes()); // external attributes
            self.body.extend_from_slice(&entry.offset.to_le_bytes());
            self.body.extend_from_slice(entry.name.as_bytes());
        }

        let directory_size = self.body.len() as u32 - directory_offset;
        let count = self.entries.len() as u16;

        // End of central directory.
        self.body.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        self.body.extend_from_slice(&0u16.to_le_bytes()); // this disk
        self.body.extend_from_slice(&0u16.to_le_bytes()); // disk with the directory
        self.body.extend_from_slice(&count.to_le_bytes());
        self.body.extend_from_slice(&count.to_le_bytes());
        self.body.extend_from_slice(&directory_size.to_le_bytes());
        self.body.extend_from_slice(&directory_offset.to_le_bytes());
        self.body.extend_from_slice(&0u16.to_le_bytes()); // comment length

        self.body
    }
}

/// CRC32 as ZIP uses it: the reflected IEEE polynomial, computed a byte at a
/// time. Small enough that a lookup table would be the only reason to do more.
///
/// Shared with [`crate::import::threemf`], which needs the same check value to
/// verify an entry it has just unpacked. Note that `yazi` ships an Adler-32,
/// which is what zlib wraps a stream in and is *not* what a ZIP entry stores.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let carry = crc & 1;
            crc >>= 1;
            if carry != 0 {
                crc ^= 0xedb8_8320;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Volume;
    use glam::Vec3;

    fn exported() -> ExportMesh {
        let mut volume = Volume::new(3.0);
        volume.seed_sphere(Vec3::ZERO, 12.0);
        let (mesh, report) = volume.export_mesh();
        assert!(report.is_printable());
        mesh
    }

    /// The one-body list, under the name the single-mesh entry points use.
    ///
    /// Every test below this line predates the N-body writer and is about what
    /// one object's XML says, so they all go through here rather than each
    /// spelling out a slice literal.
    fn solo(mesh: &ExportMesh) -> Vec<(&str, &ExportMesh)> {
        vec![("BrokkrSculpt", mesh)]
    }

    #[test]
    fn crc32_matches_the_known_check_value() {
        // The value every CRC32 implementation is tested against.
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn the_package_starts_with_a_zip_signature() {
        let mut bytes = Vec::new();
        write(&exported(), &mut bytes).unwrap();
        assert_eq!(&bytes[..4], &0x0403_4b50u32.to_le_bytes());
    }

    #[test]
    fn the_model_xml_declares_millimetres_and_references_its_object() {
        // The whole reason to prefer 3MF: the units are not a guess.
        let xml = model_xml(&solo(&exported()));
        assert!(xml.contains("unit=\"millimeter\""), "{xml:.400}");
        assert!(xml.contains(CORE_NAMESPACE));
        assert!(xml.contains("<object id=\"1\""));
        assert!(xml.contains("<item objectid=\"1\"/>"), "the build has to reference the object");
    }

    #[test]
    fn every_vertex_and_triangle_reaches_the_xml() {
        let mesh = exported();
        let xml = model_xml(&solo(&mesh));
        assert_eq!(xml.matches("<vertex ").count(), mesh.positions.len());
        assert_eq!(xml.matches("<triangle ").count(), mesh.triangles.len());
    }

    #[test]
    fn the_vertices_in_the_xml_are_z_up() {
        // The file is Z-up while the sculpt is Y-up. 3MF is the archival copy,
        // so getting this wrong here is the version that outlives the others.
        let mesh = exported();
        let xml = model_xml(&solo(&mesh));

        let written: Vec<Vec3> = xml
            .lines()
            .filter(|line| line.contains("<vertex "))
            .map(|line| {
                let value = |name: &str| {
                    let at =
                        line.find(name).expect("a vertex has all three coordinates") + name.len();
                    let rest = &line[at..];
                    rest[..rest.find('"').expect("the attribute is quoted")]
                        .parse::<f32>()
                        .expect("coordinates are floats")
                };
                Vec3::new(value("x=\""), value("y=\""), value("z=\""))
            })
            .collect();

        let expected: Vec<Vec3> = mesh.positions.iter().copied().map(to_print_space).collect();
        assert_eq!(written, expected);

        // The sphere sits at the origin, so a mesh that had been left Y-up
        // would be indistinguishable by extent alone. Compare axis by axis.
        let height = |vectors: &[Vec3], axis: fn(&Vec3) -> f32| {
            vectors.iter().map(axis).fold(f32::NEG_INFINITY, f32::max)
        };
        assert_eq!(height(&written, |v| v.z), height(&mesh.positions, |v| v.y));
    }

    #[test]
    fn triangle_indices_are_zero_based_and_within_range() {
        // 3MF counts from zero, unlike OBJ. Getting that backwards would shift
        // every face by one.
        let mesh = exported();
        let xml = model_xml(&solo(&mesh));
        let first = xml.lines().find(|line| line.contains("<triangle ")).expect("has a triangle");
        let [a, b, c] = mesh.triangles[0];
        assert!(first.contains(&format!("v1=\"{a}\"")), "{first}");
        assert!(first.contains(&format!("v2=\"{b}\"")));
        assert!(first.contains(&format!("v3=\"{c}\"")));

        for [a, b, c] in &mesh.triangles {
            for index in [a, b, c] {
                assert!((*index as usize) < mesh.positions.len());
            }
        }
    }

    #[test]
    fn the_same_mesh_always_produces_the_same_bytes() {
        // Timestamps left at zero on purpose, so a file can be compared or
        // deduplicated.
        let mesh = exported();
        let mut first = Vec::new();
        let mut second = Vec::new();
        write(&mesh, &mut first).unwrap();
        write(&mesh, &mut second).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn the_central_directory_lists_every_part() {
        let mut bytes = Vec::new();
        write(&exported(), &mut bytes).unwrap();

        let signatures =
            bytes.windows(4).filter(|window| *window == 0x0201_4b50u32.to_le_bytes()).count();
        assert_eq!(signatures, 5, "expected five central directory entries");

        for name in [
            "[Content_Types].xml",
            "_rels/.rels",
            "3D/3dmodel.model",
            "Metadata/model_settings.config",
            "Metadata/project_settings.config",
        ] {
            assert!(
                bytes.windows(name.len()).any(|window| window == name.as_bytes()),
                "{name} is missing from the package"
            );
        }
    }

    // --- filament slots --------------------------------------------------

    /// A mesh whose vertices are assigned slots by a closure of their index.
    fn with_slots(slot_of: impl Fn(usize) -> u8) -> ExportMesh {
        let mut mesh = exported();
        mesh.slots = (0..mesh.positions.len()).map(&slot_of).collect();
        mesh
    }

    #[test]
    fn the_paint_codes_are_the_ones_real_files_carry() {
        // Decoded from `Happy_Piglet_Multi-Color.3mf` and
        // `1plate+3color+3MF.3mf`, which between them use slots 1 through 5.
        // Getting these wrong produces a file that opens, slices, and prints in
        // the wrong colours -- there is no error anywhere in that chain.
        assert_eq!(paint_code(1), Some("4"));
        assert_eq!(paint_code(2), Some("8"));
        assert_eq!(paint_code(3), Some("0C"));
        assert_eq!(paint_code(4), Some("1C"));
        assert_eq!(paint_code(5), Some("2C"));
        assert_eq!(paint_code(16), Some("DC"));

        // The escape rule the table encodes, re-derived rather than copied:
        // states of 3 and above are "11" plus four bits of state - 3, which in
        // the reversed-nibble form is the hex digit of (slot - 3) then "C".
        for slot in 3..=16u8 {
            assert_eq!(
                paint_code(slot),
                Some(format!("{:X}C", slot - 3).as_str()),
                "slot {slot} does not follow the escape rule"
            );
        }
    }

    #[test]
    fn slot_zero_and_out_of_range_carry_no_code() {
        assert_eq!(paint_code(0), None, "0 means unassigned, not filament zero");
        assert_eq!(paint_code(17), None);
        assert_eq!(paint_code(255), None);
    }

    #[test]
    fn a_triangle_takes_the_slot_of_its_first_corner() {
        // The same rule the renderer's provoking vertex uses. If the writer
        // picked a different corner the file and the preview would disagree at
        // every colour boundary, silently.
        let mut mesh = exported();
        mesh.slots = vec![0; mesh.positions.len()];
        let [a, b, c] = mesh.triangles[0];
        mesh.slots[a as usize] = 2;
        mesh.slots[b as usize] = 4;
        mesh.slots[c as usize] = 4;

        let xml = model_xml(&solo(&mesh));
        let line = xml
            .lines()
            .find(|line| line.contains(&format!("v1=\"{a}\" v2=\"{b}\" v3=\"{c}\"")))
            .expect("the triangle should be in the file");
        assert!(line.contains("paint_color=\"8\""), "took a corner other than the first: {line}");
    }

    #[test]
    fn an_unassigned_mesh_writes_exactly_what_it_used_to() {
        // The property that keeps every existing export byte-identical: a mesh
        // nobody painted must not gain a single attribute.
        let xml = model_xml(&solo(&exported()));
        assert!(!xml.contains("paint_color"), "an unpainted mesh grew colour attributes");

        // And an all-zero slots vector is the same thing as an empty one.
        let zeroed = with_slots(|_| 0);
        assert_eq!(model_xml(&solo(&zeroed)), xml, "zeros are not the same as unassigned");
    }

    #[test]
    fn every_assigned_triangle_is_painted_including_the_base_slot() {
        // Deliberately NOT omitting the base extruder's triangles. The real
        // piglet file carries `paint_color` on 469,982 of its 469,983
        // triangles, including the 430,995 painted with its own base extruder,
        // so omission is an optimisation nothing in the target relies on. Until
        // a round trip proves the slicer reads "absent" as "base", write it.
        let mesh = with_slots(|index| (index % 4 + 1) as u8);
        let xml = model_xml(&solo(&mesh));
        let painted = xml.matches("paint_color=").count();
        assert_eq!(painted, mesh.triangles.len(), "some assigned triangles came out unpainted");
    }

    #[test]
    fn the_settings_parts_say_which_filament_and_what_colour() {
        let filaments = Filaments {
            colours: vec!["#112233".into(), "#445566".into()],
            materials: vec!["PETG".into()],
            base: 2,
        };
        let model_settings = model_settings_xml(&solo(&exported()), &filaments);
        assert!(model_settings.contains("key=\"extruder\" value=\"2\""), "{model_settings}");

        let project = project_settings_json(&filaments);
        assert!(project.contains("\"#112233\", \"#445566\""), "{project}");
        // Short material lists are padded rather than mismatched, because a
        // slicer reading two colours and one type has to guess.
        assert!(project.contains("\"PETG\", \"PLA\""), "{project}");
        assert!(project.contains("\"from\": \"project\""));
    }

    #[test]
    fn a_painted_mesh_still_reads_back_as_the_same_geometry() {
        // The reader ignores `paint_color`, and must keep ignoring it rather
        // than choking: this is the guard that adding colour did not break the
        // round trip the writer is otherwise only checked by.
        let mesh = with_slots(|index| (index % 4 + 1) as u8);
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();
        let read_back = crate::import::threemf::read(&bytes).expect("it should read back");
        assert_eq!(read_back.positions.len(), mesh.positions.len());
        assert_eq!(read_back.triangles.len(), mesh.triangles.len());
    }

    #[test]
    fn a_painted_mesh_is_written_deterministically_too() {
        let mesh = with_slots(|index| (index % 4 + 1) as u8);
        let mut first = Vec::new();
        let mut second = Vec::new();
        write(&mesh, &mut first).unwrap();
        write(&mesh, &mut second).unwrap();
        assert_eq!(first, second);
    }

    // --- several bodies ----------------------------------------------------

    /// **The bytes a known mesh produces are what the committed golden holds.**
    ///
    /// This is the test that actually pins the package. Its neighbour below
    /// compares `write` against `write_all`, which since the refactor is the
    /// *same code* on both sides: `write` is a two-hop wrapper over
    /// [`write_with_all`]. A committed file cannot move with the code.
    ///
    /// A golden is safe here only because the zip writer is the one in this
    /// file: entries are stored rather than deflated and every timestamp is
    /// zero, so there is no compressor version and no clock to make the bytes
    /// drift. The golden covers the whole package --- content types, the
    /// relationships part, the model XML and both slicer sidecars --- which is
    /// the point, since a slicer refuses the package over any one of them.
    ///
    /// The golden's cube carries mixed filament slots, so the `paint_color`
    /// route and the bare `<triangle>` are both in the pinned bytes. See
    /// [`crate::export::golden`].
    #[test]
    fn a_known_mesh_writes_the_bytes_committed_in_the_golden() {
        let mesh = crate::export::golden::cube();
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();
        crate::export::golden::assert_bytes("export-cube.3mf", &bytes);
    }

    /// **A single body writes byte for byte what the single-mesh writer always
    /// has**, name included: [`write`] passes `BrokkrSculpt` through, so the
    /// one-body package is unchanged by the N-body path. That makes the name the
    /// only parameter reaching the one-body path --- and proves nothing more,
    /// because `write` now *is* `write_with_all`. The bytes themselves are
    /// pinned by [`a_known_mesh_writes_the_bytes_committed_in_the_golden`]; keep
    /// both.
    ///
    /// The application now names the body instead, so a real one-body export
    /// carries `value="Body 1"` in the slicer's sidecar where it used to say
    /// `BrokkrSculpt`. The model part -- the geometry, the units, the ids -- is
    /// identical, and the alternative was eleven objects in a slicer's list all
    /// called the same thing.
    #[test]
    fn one_body_through_the_document_writer_is_byte_identical() {
        let mesh = exported();
        let mut alone = Vec::new();
        let mut through = Vec::new();
        write(&mesh, &mut alone).unwrap();
        write_all(&[("BrokkrSculpt", &mesh)], &mut through).unwrap();
        assert_eq!(alone, through, "the N-body path changed the one-body package");
    }

    /// Each body is its own `<object>` with its own `<item>`, and the indices
    /// stay per object.
    ///
    /// The index rule is the opposite of OBJ's and that is exactly why it is
    /// pinned: 3MF numbers vertices within an object, so a running offset
    /// copied over from the OBJ writer would send every triangle of the second
    /// body past the end of its own vertex list.
    #[test]
    fn each_body_is_its_own_object_and_its_own_build_item() {
        let mesh = exported();
        let xml = model_xml(&[("Left", &mesh), ("Right", &mesh)]);

        assert_eq!(xml.matches("<object id=").count(), 2);
        assert!(xml.contains("<object id=\"1\" type=\"model\">"), "{xml:.400}");
        assert!(xml.contains("<object id=\"2\" type=\"model\">"));
        assert!(xml.contains("<item objectid=\"1\"/>"));
        assert!(xml.contains("<item objectid=\"2\"/>"));
        assert_eq!(xml.matches("<vertex ").count(), mesh.positions.len() * 2);
        assert_eq!(xml.matches("<triangle ").count(), mesh.triangles.len() * 2);

        // Every index is inside ONE body's vertex list, which is what "per
        // object" means and what an offset would break.
        let highest = mesh.positions.len() - 1;
        for line in xml.lines().filter(|line| line.contains("<triangle ")) {
            for name in ["v1=\"", "v2=\"", "v3=\""] {
                let at = line.find(name).expect("a triangle has three corners") + name.len();
                let rest = &line[at..];
                let index: usize = rest[..rest.find('"').unwrap()].parse().unwrap();
                assert!(index <= highest, "index {index} is past the end of its own object");
            }
        }

        // And the sidecar names them, so the slicer's object list is readable.
        let settings =
            model_settings_xml(&[("Left", &mesh), ("Right", &mesh)], &Filaments::default());
        assert!(settings.contains("<object id=\"1\">"), "{settings}");
        assert!(settings.contains("value=\"Left\""), "{settings}");
        assert!(settings.contains("<object id=\"2\">"));
        assert!(settings.contains("value=\"Right\""));
    }

    /// A body's name is user text and it lands inside an XML attribute.
    ///
    /// A rename box will take `<` and `"` without complaint, and the reader
    /// repairs a name out of a file rather than refusing it, so this writer is
    /// the last thing between somebody else's name and somebody else's parser.
    /// Unescaped, a quote ends the attribute and the rest of the name is read
    /// as markup.
    #[test]
    fn a_name_with_xml_in_it_is_escaped_rather_than_written_through() {
        let mesh = exported();
        let hostile = "a<b>c&d\"e'f";
        let settings = model_settings_xml(&[(hostile, &mesh)], &Filaments::default());
        assert!(
            settings.contains("value=\"a&lt;b&gt;c&amp;d&quot;e&apos;f\""),
            "a name went into the XML unescaped: {settings}"
        );
        assert!(!settings.contains("value=\"a<"), "{settings}");
    }
}
