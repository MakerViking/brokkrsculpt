// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reading Wavefront OBJ.
//!
//! Only the geometry is read: `v` for positions and `f` for faces. Normals
//! (`vn`), texture coordinates (`vt`), materials, groups and smoothing are
//! parsed past and discarded, because a signed distance field has nowhere to
//! put any of them -- the surface is rebuilt from the triangles alone.
//!
//! Faces with more than three corners are triangulated as a fan. That is exact
//! for a convex face and the standard reading of the format; a concave n-gon
//! would need ear clipping, and OBJ files with concave faces are rare enough
//! that carrying a polygon triangulator for them is not a trade worth making.
//! The failure is visible rather than silent -- a fan across a concave face
//! puts triangles outside it, which shows up as surface where none belongs.

use glam::Vec3;

use super::{ImportError, SoupWelder, finite};
use crate::export::ExportMesh;

pub fn read(bytes: &[u8]) -> Result<ExportMesh, ImportError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| ImportError::Malformed("an OBJ file must be text, and this is not".into()))?;

    // The file's own vertex list, in file order. OBJ indices refer into this,
    // so it has to be kept whole even though the welder builds its own.
    let mut listed: Vec<Vec3> = Vec::new();
    let mut welder = SoupWelder::new();
    let mut corners: Vec<Vec3> = Vec::new();

    for (number, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("");
        let mut words = line.split_ascii_whitespace();
        match words.next() {
            Some("v") => {
                let mut read_one = || -> Result<f32, ImportError> {
                    words.next().and_then(|word| word.parse::<f32>().ok()).ok_or_else(|| {
                        ImportError::Malformed(format!("line {}: bad vertex", number + 1))
                    })
                };
                let position = Vec3::new(read_one()?, read_one()?, read_one()?);
                listed.push(finite(position, &format!("line {}", number + 1))?);
            }
            Some("f") => {
                corners.clear();
                for word in words {
                    // `v`, `v/vt`, `v//vn` and `v/vt/vn` all put the position
                    // index first, so everything past the first slash is for
                    // data this importer has no use for.
                    let field = word.split('/').next().unwrap_or("");
                    let Ok(index) = field.parse::<i64>() else {
                        return Err(ImportError::Malformed(format!(
                            "line {}: `{word}` is not a face index",
                            number + 1
                        )));
                    };
                    corners.push(resolve(index, &listed, number + 1)?);
                }
                if corners.len() < 3 {
                    return Err(ImportError::Malformed(format!(
                        "line {}: a face needs at least three corners, this one has {}",
                        number + 1,
                        corners.len()
                    )));
                }
                // Fan from the first corner.
                for corner in 1..corners.len() - 1 {
                    welder.triangle([corners[0], corners[corner], corners[corner + 1]]);
                }
            }
            _ => {}
        }
    }

    welder.finish()
}

/// Turn an OBJ face index into a position.
///
/// OBJ indices are **one based**, and a negative index counts backwards from
/// the most recent vertex. Both are easy to get wrong in the direction that
/// still parses: an off-by-one silently shifts every face onto its neighbour's
/// corners, which produces a mesh that looks like scrambled noise rather than
/// an error.
fn resolve(index: i64, listed: &[Vec3], line: usize) -> Result<Vec3, ImportError> {
    let count = listed.len() as i64;
    let resolved = if index > 0 {
        index - 1
    } else if index < 0 {
        count + index
    } else {
        return Err(ImportError::Malformed(format!("line {line}: index 0 is not valid in OBJ")));
    };
    if resolved < 0 || resolved >= count {
        return Err(ImportError::Malformed(format!(
            "line {line}: index {index} is outside the {count} vertices declared so far"
        )));
    }
    Ok(listed[resolved as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: &str = "\
# a unit square as two triangles
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3
f 1 3 4
";

    #[test]
    fn positions_and_faces_are_read() {
        let mesh = read(SQUARE.as_bytes()).unwrap();
        assert_eq!(mesh.triangles.len(), 2);
        assert_eq!(mesh.positions.len(), 4, "the shared corners were not welded");
    }

    #[test]
    fn indices_are_one_based() {
        let mesh = read(SQUARE.as_bytes()).unwrap();
        let [a, _, _] = mesh.triangles[0].map(|index| mesh.positions[index as usize]);
        assert_eq!(a, Vec3::ZERO, "face index 1 should mean the first vertex, not the second");
    }

    #[test]
    fn negative_indices_count_back_from_the_end() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n";
        let mesh = read(text.as_bytes()).unwrap();
        let corners = mesh.triangles[0].map(|index| mesh.positions[index as usize]);
        assert_eq!(corners, [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)]);
    }

    #[test]
    fn the_extra_fields_of_a_face_reference_are_ignored() {
        let text = "\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 3//1
";
        let mesh = read(text.as_bytes()).unwrap();
        assert_eq!(mesh.triangles.len(), 1, "a v/vt/vn face reference was not understood");
    }

    #[test]
    fn a_quad_is_fanned_into_two_triangles() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        let mesh = read(text.as_bytes()).unwrap();
        assert_eq!(mesh.triangles.len(), 2);
    }

    #[test]
    fn an_index_past_the_end_is_refused_rather_than_wrapping() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n";
        assert!(
            matches!(read(text.as_bytes()), Err(ImportError::Malformed(_))),
            "an out of range index was accepted, which would silently scramble the mesh"
        );
    }

    #[test]
    fn index_zero_is_refused() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n";
        assert!(matches!(read(text.as_bytes()), Err(ImportError::Malformed(_))));
    }

    #[test]
    fn comments_and_unknown_keywords_are_skipped() {
        let text = "\
mtllib thing.mtl
o Cube
g group
s off
usemtl red
v 0 0 0 # trailing comment
v 1 0 0
v 0 1 0
f 1 2 3
";
        assert_eq!(read(text.as_bytes()).unwrap().triangles.len(), 1);
    }

    #[test]
    fn a_file_with_no_faces_is_empty_rather_than_a_blank_model() {
        assert!(matches!(read(b"v 0 0 0\nv 1 0 0\n"), Err(ImportError::Empty)));
    }

    /// The counterpart of the STL round trip: this reader must invert the
    /// writer this project already ships.
    #[test]
    fn our_own_exported_obj_reads_back_with_the_same_geometry() {
        let mut volume = crate::volume::Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 8.0);
        volume.mark_everything_dirty();
        let (exported, report) = volume.export_mesh();
        assert!(report.is_printable(), "the fixture is not a good mesh: {}", report.summary());

        let mut bytes = Vec::new();
        crate::export::obj::write(&exported, &mut bytes).unwrap();
        let read_back = read(&bytes).unwrap();

        assert_eq!(read_back.triangles.len(), exported.triangles.len());
        assert_eq!(read_back.positions.len(), exported.positions.len());
    }
}
