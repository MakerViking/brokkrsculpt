// SPDX-License-Identifier: AGPL-3.0-or-later

//! 3MF, the format that actually states its units.
//!
//! A 3MF is a ZIP container holding a little XML. Unlike STL it shares vertices,
//! and unlike both STL and OBJ it says outright that the model is in
//! millimetres, so there is nothing for a slicer to guess.
//!
//! # Why the ZIP is written by hand
//!
//! Only three small entries are needed and 3MF permits them stored rather than
//! deflated, so the whole container is a few hundred lines of well specified
//! header layout and one CRC. That is a poor trade against pulling in a zip
//! crate and a compression crate, and the format is fixed forever, so there is
//! nothing here to keep up with.
//!
//! What this writer produces is a deliberately degenerate subset of the format:
//! stored entries only, one namespace, always millimetres, one object and no
//! transforms. [`crate::import::threemf`] reads it back in the tests, which is
//! what stops this module being checked only against its own assumptions -- but
//! that reader is not tested against this writer alone, precisely because
//! agreeing with it would prove nothing about a real file.

use std::io::{self, Write};

use super::ExportMesh;

/// The core specification namespace, which every consumer keys off.
const CORE_NAMESPACE: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";

/// Build the model XML for a mesh.
///
/// Coordinates are written with enough digits to round trip an `f32` exactly,
/// because a 3MF is often the archival copy of a model.
fn model_xml(mesh: &ExportMesh) -> String {
    let mut xml = String::with_capacity(mesh.positions.len() * 48 + mesh.triangles.len() * 40);
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str(&format!(
        "<model unit=\"millimeter\" xml:lang=\"en-US\" xmlns=\"{CORE_NAMESPACE}\">\n"
    ));
    xml.push_str(" <metadata name=\"Application\">BrokkrSculpt</metadata>\n");
    xml.push_str(" <resources>\n  <object id=\"1\" type=\"model\">\n   <mesh>\n");

    xml.push_str("    <vertices>\n");
    for position in &mesh.positions {
        xml.push_str(&format!(
            "     <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>\n",
            position.x, position.y, position.z
        ));
    }
    xml.push_str("    </vertices>\n");

    xml.push_str("    <triangles>\n");
    for [a, b, c] in &mesh.triangles {
        xml.push_str(&format!("     <triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\"/>\n"));
    }
    xml.push_str("    </triangles>\n");

    xml.push_str("   </mesh>\n  </object>\n </resources>\n");
    xml.push_str(" <build>\n  <item objectid=\"1\"/>\n </build>\n");
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

/// Write `mesh` as a 3MF package.
pub fn write(mesh: &ExportMesh, out: &mut impl Write) -> io::Result<()> {
    let model = model_xml(mesh);
    let mut zip = ZipWriter::new();
    // Order matters to some readers: the content types part has to come first.
    zip.add("[Content_Types].xml", CONTENT_TYPES.as_bytes());
    zip.add("_rels/.rels", RELATIONSHIPS.as_bytes());
    zip.add("3D/3dmodel.model", model.as_bytes());
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
        let xml = model_xml(&exported());
        assert!(xml.contains("unit=\"millimeter\""), "{xml:.400}");
        assert!(xml.contains(CORE_NAMESPACE));
        assert!(xml.contains("<object id=\"1\""));
        assert!(xml.contains("<item objectid=\"1\"/>"), "the build has to reference the object");
    }

    #[test]
    fn every_vertex_and_triangle_reaches_the_xml() {
        let mesh = exported();
        let xml = model_xml(&mesh);
        assert_eq!(xml.matches("<vertex ").count(), mesh.positions.len());
        assert_eq!(xml.matches("<triangle ").count(), mesh.triangles.len());
    }

    #[test]
    fn triangle_indices_are_zero_based_and_within_range() {
        // 3MF counts from zero, unlike OBJ. Getting that backwards would shift
        // every face by one.
        let mesh = exported();
        let xml = model_xml(&mesh);
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
    fn the_central_directory_lists_all_three_parts() {
        let mut bytes = Vec::new();
        write(&exported(), &mut bytes).unwrap();

        let signatures =
            bytes.windows(4).filter(|window| *window == 0x0201_4b50u32.to_le_bytes()).count();
        assert_eq!(signatures, 3, "expected three central directory entries");

        for name in ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"] {
            assert!(
                bytes.windows(name.len()).any(|window| window == name.as_bytes()),
                "{name} is missing from the package"
            );
        }
    }
}
