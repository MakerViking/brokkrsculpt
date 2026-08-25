// SPDX-License-Identifier: AGPL-3.0-only

//! Wavefront OBJ.
//!
//! Text, shares vertices, and carries normals, so unlike STL a welded mesh
//! survives a round trip through it. Carries no units: every tool that reads it
//! for printing assumes millimetres, which is what the sculpt is in.
//!
//! Indices are one based, which is the single most common way to get OBJ wrong.
//!
//! Coordinates and normals are rotated to Z-up on the way out. See
//! [`crate::orientation`].

use std::io::{self, Write};

use super::ExportMesh;
use crate::orientation::to_print_space;

/// Write `mesh` as an OBJ, as the one object `BrokkrSculpt`.
///
/// Normals are emitted and referenced per face corner. They are averaged across
/// the triangles meeting at each vertex, so the surface reads as smooth rather
/// than faceted in anything that shades it.
pub fn write(mesh: &ExportMesh, out: &mut impl Write) -> io::Result<()> {
    write_all(&[("BrokkrSculpt", mesh)], out)
}

/// Write several bodies as one OBJ, each under its own `o` block.
///
/// **The vertex indices are file-global and one-based, so each body's faces
/// carry a running offset**: OBJ numbers `v` lines from the top of the file and
/// an `o` line starts a group rather than a new numbering. Getting that wrong
/// would put every body after the first one's faces on the previous body's
/// vertices, which loads without complaint and prints as rubble.
///
/// Each body keeps its own welded vertices; they are never merged. The weld key
/// is a lattice cell and every body shares the lattice, so a shared weld map
/// would fuse two unrelated bodies into one vertex wherever their cells
/// coincide -- which is exactly where two bodies touch, and would tie them
/// together with triangles neither one has.
pub fn write_all(bodies: &[(&str, &ExportMesh)], out: &mut impl Write) -> io::Result<()> {
    let mut out = io::BufWriter::new(out);

    let vertices: usize = bodies.iter().map(|(_, mesh)| mesh.positions.len()).sum();
    let triangles: usize = bodies.iter().map(|(_, mesh)| mesh.triangles.len()).sum();
    writeln!(out, "# Exported by BrokkrSculpt")?;
    writeln!(out, "# Units are millimetres")?;
    writeln!(out, "# {vertices} vertices, {triangles} triangles")?;

    let mut offset: u32 = 0;
    for (name, mesh) in bodies {
        writeln!(out, "o {name}")?;

        for position in &mesh.positions {
            let position = to_print_space(*position);
            writeln!(out, "v {} {} {}", position.x, position.y, position.z)?;
        }
        for normal in &mesh.normals {
            // Rotated alongside the positions. A normal left in sculpt space
            // beside a rotated vertex shades as if the light were coming from
            // the wrong place, which reads as a material problem rather than an
            // axis one.
            let normal = to_print_space(*normal);
            writeln!(out, "vn {} {} {}", normal.x, normal.y, normal.z)?;
        }

        let has_normals = mesh.normals.len() == mesh.positions.len();
        for triangle in &mesh.triangles {
            // OBJ counts from one, not zero, and from the top of the FILE.
            let [a, b, c] = triangle.map(|index| index + offset + 1);
            if has_normals {
                writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}")?;
            } else {
                writeln!(out, "f {a} {b} {c}")?;
            }
        }
        offset += mesh.positions.len() as u32;
    }

    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Volume;
    use crate::orientation::from_print_space;
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

    /// Every vector written under one keyword, read back out of the text.
    fn vectors(written: &str, keyword: &str) -> Vec<Vec3> {
        written
            .lines()
            .filter(|line| line.starts_with(keyword))
            .map(|line| {
                let mut parts = line
                    .split_whitespace()
                    .skip(1)
                    .map(|value| value.parse::<f32>().expect("coordinates are floats"));
                Vec3::new(parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap())
            })
            .collect()
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
        // The file is Z-up while the sculpt is Y-up, so the text holds the
        // rotated vertex. Rotating what was read back exercises both halves of
        // the mapping at once.
        let mesh = exported();
        let parsed = vectors(&text(&mesh), "v ");

        let expected: Vec<Vec3> = mesh.positions.iter().copied().map(to_print_space).collect();
        assert_eq!(parsed, expected);
        assert_eq!(
            parsed.into_iter().map(from_print_space).collect::<Vec<_>>(),
            mesh.positions,
            "rotating back has to land on the sculpt again"
        );
    }

    #[test]
    fn normals_are_rotated_with_the_positions_they_belong_to() {
        // A rotated vertex carrying a sculpt space normal still renders, and
        // still looks nearly right, so nothing but a test catches it.
        let mesh = exported();
        let written = text(&mesh);

        let expected: Vec<Vec3> = mesh.normals.iter().copied().map(to_print_space).collect();
        assert_eq!(vectors(&written, "vn "), expected);

        // And still outward, measured in the file's own space: the sphere is
        // centred on the origin, which the rotation leaves where it is.
        let positions = vectors(&written, "v ");
        let outward = positions
            .iter()
            .zip(&expected)
            .filter(|(position, normal)| position.normalize_or_zero().dot(**normal) > 0.0)
            .count();
        let fraction = outward as f32 / expected.len() as f32;
        assert!(fraction > 0.99, "only {:.1}% of normals faced outward", fraction * 100.0);
    }

    #[test]
    fn an_empty_mesh_still_writes_a_readable_file() {
        let written = text(&ExportMesh::default());
        assert!(written.contains("BrokkrSculpt"));
        assert_eq!(written.lines().filter(|line| line.starts_with("f ")).count(), 0);
    }

    /// **The bytes a known mesh produces are what the committed golden holds.**
    ///
    /// This is the test that actually pins the file. Its neighbour below
    /// compares `write` against `write_all`, which since the refactor is the
    /// *same code* on both sides: `write` is a one-line wrapper. Mutating the
    /// `# Exported by BrokkrSculpt` header line left the entire workspace suite
    /// green, which is how that gap was found rather than argued. A committed
    /// file cannot move with the code.
    ///
    /// The golden is written with the name `BrokkrSculpt`, so it holds the
    /// `o BrokkrSculpt` line the previous build emitted. See
    /// [`crate::export::golden`].
    #[test]
    fn a_known_mesh_writes_the_bytes_committed_in_the_golden() {
        let mesh = crate::export::golden::cube();
        let mut bytes = Vec::new();
        write(&mesh, &mut bytes).unwrap();
        crate::export::golden::assert_bytes("export-cube.obj", &bytes);
    }

    /// **A single body writes byte for byte what the single-mesh writer always
    /// has**, name included: [`write`] passes `BrokkrSculpt` through to
    /// [`write_all`], so the one-body file is unchanged by the N-body path.
    /// That makes the name the only parameter reaching the one-body path --- and
    /// proves nothing more, because `write` now *is* `write_all`. The bytes
    /// themselves are pinned by
    /// [`a_known_mesh_writes_the_bytes_committed_in_the_golden`]; keep both.
    ///
    /// The application now names the body instead, which is the one byte that
    /// moves in a real one-body export -- `o Body 1` where it used to say
    /// `o BrokkrSculpt`. Everything measured off the file is identical, and the
    /// alternative was an eleven-body OBJ with eleven objects all called the
    /// same thing.
    #[test]
    fn one_body_through_the_document_writer_is_byte_identical() {
        let mesh = exported();
        let mut alone = Vec::new();
        let mut through = Vec::new();
        write(&mesh, &mut alone).unwrap();
        write_all(&[("BrokkrSculpt", &mesh)], &mut through).unwrap();
        assert_eq!(alone, through, "the N-body path changed the one-body file");
    }

    /// **Face indices are file-global, so every body after the first carries a
    /// running offset.** This is the one thing an OBJ writer can get wrong that
    /// still loads: a second body's faces would land on the first body's
    /// vertices, and the model opens as a tangle rather than an error.
    #[test]
    fn each_body_gets_its_own_object_block_with_offset_faces() {
        let mesh = exported();
        let mut bytes = Vec::new();
        write_all(&[("Left", &mesh), ("Right", &mesh)], &mut bytes).unwrap();
        let written = String::from_utf8(bytes).expect("OBJ is text");

        let objects: Vec<&str> = written.lines().filter(|line| line.starts_with("o ")).collect();
        assert_eq!(objects, vec!["o Left", "o Right"], "one o block per body, named");

        let vertices = written.lines().filter(|line| line.starts_with("v ")).count();
        assert_eq!(vertices, mesh.positions.len() * 2, "the bodies were not both written");
        assert!(
            written.contains(&format!("# {vertices} vertices")),
            "the header must count the whole file"
        );

        let faces: Vec<&str> = written.lines().filter(|line| line.starts_with("f ")).collect();
        assert_eq!(faces.len(), mesh.triangles.len() * 2);

        let index_of = |line: &str| -> usize {
            line.split_whitespace().nth(1).unwrap().split('/').next().unwrap().parse().unwrap()
        };
        // Every index in range for the file as a whole...
        for line in &faces {
            for field in line.split_whitespace().skip(1) {
                let index: usize = field.split('/').next().unwrap().parse().unwrap();
                assert!(index >= 1 && index <= vertices, "index {index} is out of range");
            }
        }
        // ...and the second body's first face is exactly one body further on.
        let first = index_of(faces[0]);
        let second = index_of(faces[mesh.triangles.len()]);
        assert_eq!(
            second,
            first + mesh.positions.len(),
            "the second body's faces were not offset past the first body's vertices"
        );
    }
}
