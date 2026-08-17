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
//!
//! Coordinates are rotated to Z-up on the way out. See [`crate::orientation`].

use std::io::{self, Write};

use super::ExportMesh;
use crate::orientation::to_print_space;

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
        // Rotated to Z-up before anything is measured off it, so the face
        // normal below is derived from the corners as they are written rather
        // than from the sculpt's own axes.
        let [a, b, c] = triangle.map(|index| to_print_space(mesh.positions[index as usize]));
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

    /// A model that is clearly taller than it is wide, so that "which way is
    /// up" is a question the bounding box can answer.
    fn tall_model() -> Volume {
        use crate::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp};

        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let brush =
            Brush { kind: BrushKind::Draw, radius: 10.0, strength: 1.0, ..Brush::default() };

        for step in 1..=4 {
            let at = Vec3::new(0.0, 16.0 + step as f32 * 6.0, 0.0);
            brush.apply(&mut volume, &Stamp::new(at, Vec3::Y, BrushDirection::Add), &mut scratch);
        }
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
        // The file is Z-up while the sculpt is Y-up, so what comes back out is
        // the rotated corner, not the one held in the mesh.
        let volume = sphere();
        let (mesh, report) = volume.export_mesh();
        assert!(report.is_printable());

        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).expect("writing to a vector cannot fail");
        assert_eq!(bytes.len(), size_of(&mesh), "size_of disagrees with what was written");

        let triangles = parse(&bytes);
        assert_eq!(triangles.len(), mesh.triangles.len());
        for (written, original) in triangles.iter().zip(&mesh.triangles) {
            let expected = original.map(|index| to_print_space(mesh.positions[index as usize]));
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
        // Both sides of the comparison are in file space: the file is Z-up
        // while the sculpt is Y-up, and a normal left in sculpt space next to a
        // rotated corner is exactly the bug this checks for.
        let (mesh, _) = sphere().export_mesh();
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();

        let float = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        for index in 0..mesh.triangles.len().min(500) {
            let base = 84 + index * 50;
            let stored = Vec3::new(float(base), float(base + 4), float(base + 8));
            let [a, b, c] =
                mesh.triangles[index].map(|i| to_print_space(mesh.positions[i as usize]));
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

    #[test]
    fn the_tallest_axis_moves_from_y_in_the_sculpt_to_z_in_the_file() {
        // End to end: the sculpt is Y-up and the file is Z-up, so a model that
        // stands tall on screen has to stand tall in a slicer too. A bare
        // sphere cannot show this -- its extents are equal on every axis -- so
        // the fixture is drawn out along Y first.
        let volume = tall_model();
        let (mesh, report) = volume.export_mesh();
        assert!(report.is_printable(), "{}", report.summary());

        // Measured off the mesh rather than `world_bounds`, which is brick
        // quantised: these are the very points the file is written from, so the
        // two bounding boxes are comparable exactly.
        let mut sculpt_min = Vec3::splat(f32::INFINITY);
        let mut sculpt_max = Vec3::splat(f32::NEG_INFINITY);
        for position in &mesh.positions {
            sculpt_min = sculpt_min.min(*position);
            sculpt_max = sculpt_max.max(*position);
        }
        let sculpt_extent = sculpt_max - sculpt_min;
        assert!(
            sculpt_extent.y > sculpt_extent.x && sculpt_extent.y > sculpt_extent.z,
            "the fixture is meant to be tallest in Y: {sculpt_extent:?}"
        );

        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();

        let mut file_min = Vec3::splat(f32::INFINITY);
        let mut file_max = Vec3::splat(f32::NEG_INFINITY);
        for corners in parse(&bytes) {
            for corner in corners {
                file_min = file_min.min(corner);
                file_max = file_max.max(corner);
            }
        }
        let file_extent = file_max - file_min;

        assert!(
            file_extent.z > file_extent.x && file_extent.z > file_extent.y,
            "the file should be tallest in Z: {file_extent:?}"
        );
        // Not merely tallest: the same measurement, moved onto the other axis.
        // A rigid rotation cannot change how big the model is.
        assert_eq!(file_extent.z, sculpt_extent.y, "the height changed on the way out");
        assert_eq!(file_extent.x, sculpt_extent.x, "the axis rotated about moved");
        assert_eq!(file_extent.y, sculpt_extent.z);
    }

    #[test]
    fn the_written_winding_still_faces_outward() {
        // The end to end form of the determinant test in `crate::orientation`:
        // a mapping that swapped Y and Z would put the model up the right way
        // and turn it inside out, which nothing else here would notice.
        let (mesh, _) = sphere().export_mesh();
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();

        let float = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let mut outward = 0;
        let mut counted = 0;
        for (index, corners) in parse(&bytes).into_iter().enumerate() {
            let base = 84 + index * 50;
            let stored = Vec3::new(float(base), float(base + 4), float(base + 8));
            // The sphere is centred on the origin, so outward is simply away
            // from it, and the rotation leaves the origin where it is.
            let centre = (corners[0] + corners[1] + corners[2]) / 3.0;
            let Some(direction) = centre.try_normalize() else {
                continue;
            };
            counted += 1;
            if stored.dot(direction) > 0.0 {
                outward += 1;
            }
        }

        assert!(counted > 500, "not enough triangles to conclude anything");
        let fraction = outward as f32 / counted as f32;
        assert!(fraction > 0.99, "only {:.1}% of faces wound outward", fraction * 100.0);
    }
}
