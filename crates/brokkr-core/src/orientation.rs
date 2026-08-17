// SPDX-License-Identifier: AGPL-3.0-or-later

//! The one place the sculpt's axes become a file's axes.
//!
//! BrokkrSculpt sculpts in a right handed, Y-up world: the camera builds its up
//! vector out of `Vec3::Y`, and every brush, pattern and gradient is written
//! against that. STL, OBJ and 3MF are read by slicers that assume Z-up, because
//! Z is the axis a printer builds along. Written straight through, a model
//! sculpted standing up therefore arrives in a slicer lying on its back.
//!
//! So the writers rotate on the way out and the readers rotate on the way in.
//! The two are exact inverses, which is what keeps a round trip through a file
//! an identity rather than a slow drift of the model onto its side.
//!
//! # Why a rotation, and not the axis swap that looks equivalent
//!
//! Sending `(x, y, z)` to `(x, z, y)` also puts sculpt up at +Z, and is the
//! obvious thing to reach for. It is a reflection: its determinant is -1. That
//! silently reverses the winding of every triangle and inverts every normal, so
//! the exported surface is inside out. Nothing catches it. The mesh still loads,
//! still passes a watertight check -- closure is about which edges are shared,
//! not which way round they run -- and still renders, because most viewers light
//! back faces anyway. The first sign of trouble is a print with its walls on the
//! wrong side.
//!
//! [`to_print_space`] is therefore a real rotation, +90 degrees about X, and the
//! tests pin its determinant rather than only checking that up came out at +Z.

use glam::Vec3;

/// Sculpt space (Y-up) to file space (Z-up): +90 degrees about X.
///
/// `(x, y, z) -> (x, -z, y)`, so sculpt up `+Y` becomes print up `+Z`. Being a
/// rotation, it preserves handedness, so triangle winding and normals stay
/// correct without any compensating flip.
///
/// Apply it to normals as well as positions. A rotated position carrying an
/// unrotated normal reads as a shading artefact rather than as the orientation
/// bug it is.
pub fn to_print_space(v: Vec3) -> Vec3 {
    Vec3::new(v.x, -v.z, v.y)
}

/// File space (Z-up) back to sculpt space (Y-up): the inverse of
/// [`to_print_space`], -90 degrees about X.
///
/// `(x, y, z) -> (x, z, -y)`. This is what an importer applies, so a mesh
/// exported and read back lands exactly where it started.
pub fn from_print_space(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.z, -v.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat3;

    /// A spread that would hide a sign error if any component were zero.
    const SAMPLES: [Vec3; 6] = [
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-4.5, 0.25, 7.75),
        Vec3::new(0.0, -12.0, 0.0),
        Vec3::new(31.0, -31.0, 31.0),
        Vec3::new(0.001, 0.002, -0.003),
        Vec3::ZERO,
    ];

    #[test]
    fn sculpt_up_becomes_print_up() {
        assert_eq!(to_print_space(Vec3::Y), Vec3::Z);
        assert_eq!(from_print_space(Vec3::Z), Vec3::Y);
    }

    #[test]
    fn the_axis_the_model_is_rotated_about_does_not_move() {
        assert_eq!(to_print_space(Vec3::X), Vec3::X);
        assert_eq!(from_print_space(Vec3::X), Vec3::X);
    }

    #[test]
    fn a_round_trip_through_a_file_is_an_identity() {
        for sample in SAMPLES {
            assert_eq!(from_print_space(to_print_space(sample)), sample, "out and back: {sample}");
            assert_eq!(to_print_space(from_print_space(sample)), sample, "in and back: {sample}");
        }
    }

    #[test]
    fn a_rotated_export_is_not_a_mirror_of_the_sculpt() {
        // The whole reason this module exists rather than an inline axis swap.
        // A swap of Y and Z also lands up at +Z, but its determinant is -1, so
        // it reverses every triangle's winding and inverts every normal while
        // still passing every other check the export makes.
        let rotation = Mat3::from_cols(
            to_print_space(Vec3::X),
            to_print_space(Vec3::Y),
            to_print_space(Vec3::Z),
        );
        assert_eq!(rotation.determinant(), 1.0, "the mapping must be a rotation, not a reflection");

        let inverse = Mat3::from_cols(
            from_print_space(Vec3::X),
            from_print_space(Vec3::Y),
            from_print_space(Vec3::Z),
        );
        assert_eq!(inverse.determinant(), 1.0);
    }

    #[test]
    fn handedness_survives_the_rotation() {
        // Stated the way winding actually depends on it: the cross product of
        // the first two basis vectors still gives the third.
        let x = to_print_space(Vec3::X);
        let y = to_print_space(Vec3::Y);
        let z = to_print_space(Vec3::Z);
        assert_eq!(x.cross(y), z);
        assert_eq!(y.cross(z), x);
        assert_eq!(z.cross(x), y);
    }

    #[test]
    fn rotating_a_cross_product_is_the_cross_product_of_the_rotated_vectors() {
        // This is the property the STL writer leans on: it computes the face
        // normal from already rotated corners, and that has to agree with
        // rotating the normal computed from the original ones.
        let a = Vec3::new(1.0, -2.0, 0.5);
        let b = Vec3::new(-0.25, 3.0, 4.0);
        assert_eq!(to_print_space(a.cross(b)), to_print_space(a).cross(to_print_space(b)));
    }

    #[test]
    fn lengths_and_angles_are_untouched() {
        // A rotation is rigid, so an exported model is the same size and shape
        // as the sculpt, not a sheared or scaled version of it.
        for sample in SAMPLES {
            assert_eq!(to_print_space(sample).length(), sample.length());
        }
        let a = Vec3::new(2.0, 3.0, -1.0);
        let b = Vec3::new(-5.0, 0.5, 2.0);
        assert_eq!(to_print_space(a).dot(to_print_space(b)), a.dot(b));
    }
}
