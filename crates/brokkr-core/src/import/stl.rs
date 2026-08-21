// SPDX-License-Identifier: AGPL-3.0-only

//! Reading STL, both flavours.
//!
//! STL is a triangle soup with no shared vertices, no units and no orientation
//! convention. The printing world reads it as millimetres and Z up; this reader
//! reports the numbers exactly as written and leaves both conventions to the
//! caller, so that "what the file says" and "what we do about it" stay separable
//! and separately testable.

use glam::Vec3;

use super::{ImportError, SoupWelder, finite};
use crate::export::ExportMesh;

/// The fixed size of one binary STL triangle record: a normal and three
/// corners, twelve little endian `f32`s, plus the attribute byte count.
const RECORD_BYTES: usize = 50;
/// Eighty bytes of free-form header, then a `u32` triangle count.
const HEADER_BYTES: usize = 84;

/// Read either flavour, deciding which by shape rather than by the leading
/// word.
///
/// The usual test -- "does it start with `solid`?" -- is wrong often enough to
/// matter: plenty of exporters write that word into the eighty byte header of a
/// *binary* file, and a reader that trusts it then fails on the binary body.
/// The arithmetic is decisive instead: a binary STL is exactly
/// `84 + 50 * count` bytes long, and no ASCII file lands on that by accident
/// for a plausible count. Our own writer sidesteps the question entirely by
/// starting its header with something else, and this reader does not rely on
/// that.
pub fn read(bytes: &[u8]) -> Result<ExportMesh, ImportError> {
    if looks_binary(bytes) { read_binary(bytes) } else { read_ascii(bytes) }
}

fn declared_count(bytes: &[u8]) -> Option<usize> {
    let field: [u8; 4] = bytes.get(80..84)?.try_into().ok()?;
    Some(u32::from_le_bytes(field) as usize)
}

fn looks_binary(bytes: &[u8]) -> bool {
    if bytes.len() < HEADER_BYTES {
        return false;
    }
    let Some(count) = declared_count(bytes) else {
        return false;
    };
    // Checked in `usize` after the length guard above, so the multiply cannot
    // wrap on a 64 bit target for any count a `u32` can hold.
    bytes.len() == HEADER_BYTES + count * RECORD_BYTES
}

fn read_binary(bytes: &[u8]) -> Result<ExportMesh, ImportError> {
    let count = declared_count(bytes)
        .ok_or_else(|| ImportError::Malformed("the STL header is truncated".to_string()))?;

    let mut welder = SoupWelder::new();
    for index in 0..count {
        let start = HEADER_BYTES + index * RECORD_BYTES;
        let record = &bytes[start..start + RECORD_BYTES];
        // The stored face normal is deliberately ignored. Readers disagree
        // about whether it or the winding wins, exporters routinely write
        // zeroes, and the voxeliser takes its orientation from the winding --
        // so trusting the normal here would introduce a second, conflicting
        // source of truth for which way a face points.
        let mut corners = [Vec3::ZERO; 3];
        for (corner, slot) in corners.iter_mut().enumerate() {
            let base = 12 + corner * 12;
            *slot = Vec3::new(
                float_at(record, base),
                float_at(record, base + 4),
                float_at(record, base + 8),
            );
            *slot = finite(*slot, &format!("triangle {index}"))?;
        }
        welder.triangle(corners);
    }
    welder.finish()
}

/// Read a little endian `f32` from a slice known to be long enough.
fn float_at(record: &[u8], at: usize) -> f32 {
    let field: [u8; 4] = record[at..at + 4].try_into().expect("the record is a fixed 50 bytes");
    f32::from_le_bytes(field)
}

/// Read the text flavour.
///
/// Deliberately lenient about structure and strict about numbers: `facet`,
/// `outer loop` and `endfacet` are skipped rather than validated, because every
/// real failure seen in the wild is a bad coordinate rather than a missing
/// keyword, and a reader that rejects an otherwise fine file over a missing
/// `endsolid` helps nobody.
fn read_ascii(bytes: &[u8]) -> Result<ExportMesh, ImportError> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        ImportError::Malformed(
            "it is neither a binary STL of the declared size nor valid text".to_string(),
        )
    })?;

    let mut welder = SoupWelder::new();
    let mut corners: Vec<Vec3> = Vec::with_capacity(3);

    for (number, line) in text.lines().enumerate() {
        let mut words = line.split_ascii_whitespace();
        if words.next() != Some("vertex") {
            continue;
        }
        let mut read_one = || -> Result<f32, ImportError> {
            words
                .next()
                .and_then(|word| word.parse::<f32>().ok())
                .ok_or_else(|| ImportError::Malformed(format!("line {}: bad vertex", number + 1)))
        };
        let position = Vec3::new(read_one()?, read_one()?, read_one()?);
        corners.push(finite(position, &format!("line {}", number + 1))?);

        if corners.len() == 3 {
            welder.triangle([corners[0], corners[1], corners[2]]);
            corners.clear();
        }
    }

    if !corners.is_empty() {
        return Err(ImportError::Malformed(format!(
            "it ends part way through a triangle, with {} of 3 corners",
            corners.len()
        )));
    }
    welder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a binary STL the way an exporter would, header word and all.
    fn binary(triangles: &[[Vec3; 3]], header: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; 80];
        bytes[..header.len()].copy_from_slice(header);
        bytes.extend_from_slice(&(triangles.len() as u32).to_le_bytes());
        for corners in triangles {
            for value in [0.0f32, 0.0, 0.0] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for corner in corners {
                for value in corner.to_array() {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            bytes.extend_from_slice(&0u16.to_le_bytes());
        }
        bytes
    }

    fn one_triangle() -> [Vec3; 3] {
        [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)]
    }

    #[test]
    fn a_binary_file_round_trips_its_corners() {
        let bytes = binary(&[one_triangle()], b"whatever");
        let mesh = read(&bytes).unwrap();
        assert_eq!(mesh.triangles.len(), 1);
        assert_eq!(mesh.positions.len(), 3);
        let [a, b, c] = mesh.triangles[0].map(|index| mesh.positions[index as usize]);
        assert_eq!([a, b, c], one_triangle());
    }

    /// The trap this reader is built around: a binary file whose header starts
    /// with the word that is supposed to mean ASCII.
    #[test]
    fn a_binary_file_whose_header_says_solid_is_still_read_as_binary() {
        let bytes = binary(&[one_triangle()], b"solid exported by something careless");
        let mesh = read(&bytes).expect("a binary STL claiming to be ASCII should still parse");
        assert_eq!(mesh.triangles.len(), 1, "it was parsed as text and produced nothing");
    }

    #[test]
    fn an_ascii_file_is_read() {
        let text = "\
solid thing
 facet normal 0 0 1
  outer loop
   vertex 0 0 0
   vertex 1 0 0
   vertex 0 1 0
  endloop
 endfacet
endsolid thing
";
        let mesh = read(text.as_bytes()).unwrap();
        assert_eq!(mesh.triangles.len(), 1);
        let [a, b, c] = mesh.triangles[0].map(|index| mesh.positions[index as usize]);
        assert_eq!([a, b, c], one_triangle());
    }

    /// Every corner of an STL soup is written out separately, so a closed shape
    /// arrives with each position repeated. Welding is what makes the triangle
    /// count meaningful and the vertex count small.
    #[test]
    fn repeated_corners_are_welded_into_one_vertex() {
        let shared = Vec3::new(0.0, 0.0, 0.0);
        let mesh = read(&binary(
            &[
                [shared, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
                [shared, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)],
            ],
            b"x",
        ))
        .unwrap();
        assert_eq!(mesh.triangles.len(), 2);
        assert_eq!(mesh.positions.len(), 4, "the shared corners were not welded");
    }

    #[test]
    fn a_degenerate_triangle_is_dropped_rather_than_carried() {
        let point = Vec3::new(2.0, 2.0, 2.0);
        let bytes = binary(&[one_triangle(), [point, point, point]], b"x");
        let mesh = read(&bytes).unwrap();
        assert_eq!(mesh.triangles.len(), 1, "a zero area triangle survived");
    }

    #[test]
    fn a_file_with_no_triangles_is_an_error_rather_than_an_empty_model() {
        let bytes = binary(&[], b"x");
        assert!(matches!(read(&bytes), Err(ImportError::Empty)));
    }

    /// A count field that does not match the file length must not be believed.
    /// Trusting it is how a truncated download becomes a panic on a slice.
    #[test]
    fn a_truncated_binary_file_does_not_panic() {
        let mut bytes = binary(&[one_triangle(), one_triangle()], b"x");
        bytes.truncate(bytes.len() - 20);
        // It no longer satisfies the binary size rule, so it is tried as text
        // and rejected there. What matters is that it neither panics nor
        // returns a mesh.
        assert!(read(&bytes).is_err(), "a truncated file parsed as though it were whole");
    }

    #[test]
    fn a_not_a_number_coordinate_is_refused() {
        let bytes = binary(
            &[[Vec3::new(f32::NAN, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)]],
            b"x",
        );
        assert!(
            matches!(read(&bytes), Err(ImportError::Malformed(_))),
            "a NaN corner was accepted, and would become a field full of NaN"
        );
    }

    #[test]
    fn an_ascii_file_that_stops_mid_triangle_is_refused() {
        let text = "solid x\n vertex 0 0 0\n vertex 1 0 0\n";
        assert!(matches!(read(text.as_bytes()), Err(ImportError::Malformed(_))));
    }

    /// The one that proves this reader inverts the writer, rather than merely
    /// agreeing with the test author about what STL looks like.
    #[test]
    fn our_own_exported_stl_reads_back_with_the_same_geometry() {
        let mut volume = crate::volume::Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 8.0);
        volume.mark_everything_dirty();
        let (exported, report) = volume.export_mesh();
        assert!(report.is_printable(), "the fixture is not a good mesh: {}", report.summary());

        let mut bytes = Vec::new();
        crate::export::stl::write(&exported, &mut bytes).unwrap();
        let read_back = read(&bytes).unwrap();

        assert_eq!(
            read_back.triangles.len(),
            exported.triangles.len(),
            "the triangle count changed across a write and a read"
        );
        assert_eq!(
            read_back.positions.len(),
            exported.positions.len(),
            "welding on exact bits should recover exactly the vertices that were written"
        );
    }
}
