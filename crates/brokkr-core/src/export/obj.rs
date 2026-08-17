// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wavefront OBJ.
//!
//! Text, shares vertices, and carries normals, so unlike STL a welded mesh
//! survives a round trip through it. Carries no units: every tool that reads it
//! for printing assumes millimetres, which is what the sculpt is in.
//!
//! Indices are one based, which is the single most common way to get OBJ wrong.

use std::io::{self, Write};

use super::ExportMesh;

/// Write `mesh` as an OBJ.
///
/// Normals are emitted and referenced per face corner. They are averaged across
/// the triangles meeting at each vertex, so the surface reads as smooth rather
/// than faceted in anything that shades it.
pub fn write(mesh: &ExportMesh, out: &mut impl Write) -> io::Result<()> {
    let mut out = io::BufWriter::new(out);

    writeln!(out, "# Exported by BrokkrSculpt")?;
    writeln!(out, "# Units are millimetres")?;
    writeln!(out, "# {} vertices, {} triangles", mesh.positions.len(), mesh.triangles.len())?;
    writeln!(out, "o BrokkrSculpt")?;

    for position in &mesh.positions {
        writeln!(out, "v {} {} {}", position.x, position.y, position.z)?;
    }
    for normal in &mesh.normals {
        writeln!(out, "vn {} {} {}", normal.x, normal.y, normal.z)?;
    }

    let has_normals = mesh.normals.len() == mesh.positions.len();
    for triangle in &mesh.triangles {
        // OBJ counts from one, not zero.
        let [a, b, c] = triangle.map(|index| index + 1);
        if has_normals {
            writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}")?;
        } else {
            writeln!(out, "f {a} {b} {c}")?;
        }
    }

    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Volume;
    use glam::Vec3;

    fn exported() -> ExportMesh {
        let mut volume = Volume::new(2.0);
        volume.seed_sphere(Vec3::ZERO, 16.0);
        let (mesh, report) = volume.export_mesh();
        assert!(report.is_printable());
        mesh
    }

    fn text(mesh: &ExportMesh) -> String {
        let mut bytes = Vec::new();
        write(mesh, &mut bytes).expect("writing to a vector cannot fail");
        String::from_utf8(bytes).expect("OBJ is text")
    }

    #[test]
    fn the_counts_in_the_file_match_the_mesh() {
        let mesh = exported();
        let written = text(&mesh);
        assert_eq!(
            written.lines().filter(|line| line.starts_with("v ")).count(),
            mesh.positions.len()
        );
        assert_eq!(
            written.lines().filter(|line| line.starts_with("vn ")).count(),
            mesh.normals.len()
        );
        assert_eq!(
            written.lines().filter(|line| line.starts_with("f ")).count(),
            mesh.triangles.len()
        );
    }

    #[test]
    fn face_indices_are_one_based_and_within_range() {
        // Off by one here is the classic OBJ bug: the model loads, and every
        // face refers to the wrong vertex.
        let mesh = exported();
        let written = text(&mesh);

        let mut checked = 0;
        for line in written.lines().filter(|line| line.starts_with("f ")) {
            for field in line.split_whitespace().skip(1) {
                let index: usize = field
                    .split('/')
                    .next()
                    .expect("a face field has at least one part")
                    .parse()
                    .expect("face indices are integers");
                assert!(index >= 1, "OBJ indices start at one, got {index}");
                assert!(index <= mesh.positions.len(), "index {index} is past the end");
                checked += 1;
            }
        }
        assert_eq!(checked, mesh.triangles.len() * 3);
    }

    #[test]
    fn the_first_face_refers_to_the_first_triangle() {
        let mesh = exported();
        let written = text(&mesh);
        let first = written.lines().find(|line| line.starts_with("f ")).expect("has a face");
        let [a, b, c] = mesh.triangles[0];
        assert_eq!(
            first,
            format!("f {}//{} {}//{} {}//{}", a + 1, a + 1, b + 1, b + 1, c + 1, c + 1)
        );
    }

    #[test]
    fn positions_round_trip_through_the_text() {
        let mesh = exported();
        let written = text(&mesh);
        let parsed: Vec<Vec3> = written
            .lines()
            .filter(|line| line.starts_with("v "))
            .map(|line| {
                let mut parts = line
                    .split_whitespace()
                    .skip(1)
                    .map(|value| value.parse::<f32>().expect("coordinates are floats"));
                Vec3::new(parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap())
            })
            .collect();
        assert_eq!(parsed, mesh.positions);
    }

    #[test]
    fn an_empty_mesh_still_writes_a_readable_file() {
        let written = text(&ExportMesh::default());
        assert!(written.contains("BrokkrSculpt"));
        assert_eq!(written.lines().filter(|line| line.starts_with("f ")).count(), 0);
    }
}
