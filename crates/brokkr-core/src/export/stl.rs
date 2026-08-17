// SPDX-License-Identifier: AGPL-3.0-or-later

//! Binary STL.
//!
//! The oldest and least capable of the three, and still what every slicer opens
//! first. It carries no units, no colour and no vertex sharing: each triangle
//! repeats its three corners in full. A welded mesh therefore comes back out of
//! an STL unwelded, which is exactly why the format has such a reputation for
//! producing models that will not print. Nothing can be done about that from
//! this side beyond writing corners that agree bit for bit, which welding
//! already guarantees.
//!
//! Binary rather than ASCII: a tenth of the size and no float formatting to
//! argue about.

use std::io::{self, Write};

use super::ExportMesh;

/// Bytes an STL takes, so a caller can size a buffer or check free space.
pub fn size_of(mesh: &ExportMesh) -> usize {
    80 + 4 + mesh.triangles.len() * 50
}

/// Write `mesh` as binary STL.
///
/// The 80 byte header deliberately does not begin with "solid": some readers
/// sniff that word to decide a file is ASCII, and would then fail on the binary
/// body.
pub fn write(mesh: &ExportMesh, out: &mut impl Write) -> io::Result<()> {
    let mut header = [0u8; 80];
    let banner = b"Exported by BrokkrSculpt. Units are millimetres.";
    header[..banner.len()].copy_from_slice(banner);
    out.write_all(&header)?;

    let count = u32::try_from(mesh.triangles.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "STL cannot hold more than about four billion triangles",
        )
    })?;
    out.write_all(&count.to_le_bytes())?;

    for triangle in &mesh.triangles {
        let [a, b, c] = triangle.map(|index| mesh.positions[index as usize]);
        // STL stores a face normal. Some readers trust it over the winding, so
        // it has to agree with the winding rather than being left at zero.
        let normal = (b - a).cross(c - a).try_normalize().unwrap_or(glam::Vec3::Z);

        for vector in [normal, a, b, c] {
            for component in vector.to_array() {
                out.write_all(&component.to_le_bytes())?;
            }
        }
        // Attribute byte count, unused.
        out.write_all(&0u16.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Volume;
    use glam::Vec3;

    fn sphere() -> Volume {
        let mut volume = Volume::new(2.0);
        volume.seed_sphere(Vec3::ZERO, 16.0);
        volume
    }

    /// Read a binary STL back into triangles, so the tests check the bytes
    /// rather than the code that wrote them.
    fn parse(bytes: &[u8]) -> Vec<[Vec3; 3]> {
        assert!(bytes.len() >= 84, "too short to be an STL");
        let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 84 + count * 50, "length disagrees with the triangle count");

        let float = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        (0..count)
            .map(|index| {
                let base = 84 + index * 50;
                // Skip the normal, read the three corners.
                let corner = |n: usize| {
                    let at = base + 12 + n * 12;
                    Vec3::new(float(at), float(at + 4), float(at + 8))
                };
                [corner(0), corner(1), corner(2)]
            })
            .collect()
    }

    #[test]
    fn the_written_bytes_parse_back_to_the_same_triangles() {
        let volume = sphere();
        let (mesh, report) = volume.export_mesh();
        assert!(report.is_printable());

        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).expect("writing to a vector cannot fail");
        assert_eq!(bytes.len(), size_of(&mesh), "size_of disagrees with what was written");

        let triangles = parse(&bytes);
        assert_eq!(triangles.len(), mesh.triangles.len());
        for (written, original) in triangles.iter().zip(&mesh.triangles) {
            let expected = original.map(|index| mesh.positions[index as usize]);
            assert_eq!(*written, expected);
        }
    }

    #[test]
    fn the_header_does_not_start_with_solid() {
        // Readers that sniff for that word would treat the file as ASCII and
        // then choke on the binary body.
        let (mesh, _) = sphere().export_mesh();
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();
        assert!(!bytes.starts_with(b"solid"));
    }

    #[test]
    fn face_normals_agree_with_the_winding() {
        let (mesh, _) = sphere().export_mesh();
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();

        let float = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        for index in 0..mesh.triangles.len().min(500) {
            let base = 84 + index * 50;
            let stored = Vec3::new(float(base), float(base + 4), float(base + 8));
            let [a, b, c] = mesh.triangles[index].map(|i| mesh.positions[i as usize]);
            let Some(expected) = (b - a).cross(c - a).try_normalize() else {
                continue;
            };
            assert!(
                stored.dot(expected) > 0.99,
                "triangle {index} stored {stored:?} against {expected:?}"
            );
        }
    }

    #[test]
    fn an_empty_mesh_writes_a_valid_empty_file() {
        let mesh = ExportMesh::default();
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();
        assert_eq!(bytes.len(), 84);
        assert_eq!(parse(&bytes).len(), 0);
    }
}
