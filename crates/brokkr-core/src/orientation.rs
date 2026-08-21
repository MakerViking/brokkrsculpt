// SPDX-License-Identifier: AGPL-3.0-only

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

use glam::{IVec3, Vec3};

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

/// One of the six axis aligned directions: which way a cube face points.
///
/// Naming a face and naming what it should become is how the user says which
/// way is up, so these are the two ends of an [`AxisRotation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Facing {
    /// `+X`.
    Right,
    /// `-X`.
    Left,
    /// `+Y`, which is up in sculpt space.
    Up,
    /// `-Y`.
    Down,
    /// `+Z`.
    Front,
    /// `-Z`.
    Back,
}

impl Facing {
    pub const ALL: [Facing; 6] =
        [Facing::Right, Facing::Left, Facing::Up, Facing::Down, Facing::Front, Facing::Back];

    /// The unit vector this direction points along.
    pub const fn normal(self) -> IVec3 {
        match self {
            Facing::Right => IVec3::new(1, 0, 0),
            Facing::Left => IVec3::new(-1, 0, 0),
            Facing::Up => IVec3::new(0, 1, 0),
            Facing::Down => IVec3::new(0, -1, 0),
            Facing::Front => IVec3::new(0, 0, 1),
            Facing::Back => IVec3::new(0, 0, -1),
        }
    }

    pub const fn opposite(self) -> Facing {
        match self {
            Facing::Right => Facing::Left,
            Facing::Left => Facing::Right,
            Facing::Up => Facing::Down,
            Facing::Down => Facing::Up,
            Facing::Front => Facing::Back,
            Facing::Back => Facing::Front,
        }
    }

    /// Which of the six directions a vector points along, or `None` when it is
    /// not close enough to any of them to be one.
    ///
    /// The navigation cube hands over face normals it built from its own table,
    /// so this is an exact match in practice; the tolerance is only there so a
    /// value that has been through a matrix still resolves.
    pub fn nearest(v: Vec3) -> Option<Facing> {
        Facing::ALL.into_iter().find(|facing| v.dot(facing.normal().as_vec3()) > 0.9 * v.length())
    }
}

/// A rotation by a multiple of 90 degrees, as a signed permutation of the axes.
///
/// This is the only kind of re-orientation the sculpt offers, and the
/// restriction is what makes it free of error: it maps voxels exactly onto
/// voxels, so a model can be turned without resampling it. An arbitrary angle
/// would have to go through [`crate::Volume::resampled`]'s trilinear path and
/// would blur the surface a little every time.
///
/// Stored as the images of the three basis vectors, which is a matrix in its
/// columns. Being integer, its determinant is exact, and
/// [`AxisRotation::taking`] can never hand back a reflection by accident --
/// the failure mode the module documentation above describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisRotation {
    /// `columns[i]` is where basis vector `i` ends up.
    columns: [IVec3; 3],
}

impl AxisRotation {
    pub const IDENTITY: AxisRotation = AxisRotation { columns: [IVec3::X, IVec3::Y, IVec3::Z] };

    /// The rotation that turns `from` into `to`, taking the shortest way.
    ///
    /// A quarter turn when the two are perpendicular. When they are opposite
    /// there is no shortest way -- every half turn about an axis perpendicular
    /// to `from` sends it to `to`, and they differ in what they do to
    /// everything else -- so one is chosen by a fixed rule rather than left to
    /// whichever the arithmetic happened to produce.
    pub fn taking(from: Facing, to: Facing) -> AxisRotation {
        if from == to {
            return AxisRotation::IDENTITY;
        }
        if from == to.opposite() {
            let half = AxisRotation::quarter_turn(first_axis_perpendicular_to(from));
            return half.then(half);
        }
        AxisRotation::quarter_turn(from.normal().cross(to.normal()))
    }

    /// A quarter turn about a unit basis vector, positive by the right hand
    /// rule.
    ///
    /// `R(v) = (v . k) k + k x v`, which for a basis vector `k` gives exact
    /// integers.
    fn quarter_turn(axis: IVec3) -> AxisRotation {
        let turn = |v: IVec3| axis * axis.dot(v) + axis.cross(v);
        AxisRotation { columns: [turn(IVec3::X), turn(IVec3::Y), turn(IVec3::Z)] }
    }

    /// This rotation followed by `next`.
    pub fn then(self, next: AxisRotation) -> AxisRotation {
        AxisRotation { columns: self.columns.map(|column| next.apply_ivec(column)) }
    }

    /// The rotation that undoes this one.
    ///
    /// A rotation matrix's inverse is its transpose, and for a signed
    /// permutation that is exact.
    pub fn inverse(self) -> AxisRotation {
        let c = self.columns;
        AxisRotation {
            columns: [
                IVec3::new(c[0].x, c[1].x, c[2].x),
                IVec3::new(c[0].y, c[1].y, c[2].y),
                IVec3::new(c[0].z, c[1].z, c[2].z),
            ],
        }
    }

    pub fn is_identity(self) -> bool {
        self == AxisRotation::IDENTITY
    }

    pub fn apply_ivec(self, v: IVec3) -> IVec3 {
        self.columns[0] * v.x + self.columns[1] * v.y + self.columns[2] * v.z
    }

    pub fn apply(self, v: Vec3) -> Vec3 {
        self.columns[0].as_vec3() * v.x
            + self.columns[1].as_vec3() * v.y
            + self.columns[2].as_vec3() * v.z
    }

    /// Where `from` ends up.
    pub fn applied_to(self, from: Facing) -> Option<Facing> {
        Facing::nearest(self.apply_ivec(from.normal()).as_vec3())
    }

    /// The determinant, which must be +1. A signed permutation with determinant
    /// -1 is a reflection: it would turn every triangle inside out.
    pub fn determinant(self) -> i32 {
        let c = self.columns;
        c[0].dot(c[1].cross(c[2]))
    }

    /// For each source axis, which destination axis it feeds and whether the
    /// index runs backwards along it.
    ///
    /// This is the form [`crate::Volume::rotated`] wants: it lets a whole brick
    /// be permuted with three lookups instead of a matrix multiply per voxel.
    pub(crate) fn axis_map(self) -> [(usize, bool); 3] {
        let mut map = [(0usize, false); 3];
        for (source, column) in self.columns.iter().enumerate() {
            for destination in 0..3 {
                match column[destination] {
                    0 => {}
                    m => map[source] = (destination, m < 0),
                }
            }
        }
        map
    }

    /// Where an integer voxel index lands.
    ///
    /// **A voxel index labels the cell spanning `i..i+1`, not the point `i`.**
    /// That is why a negated axis sends `i` to `-i - 1` rather than to `-i`,
    /// and it is not a detail: `-i` leaves a block of 32 indices straddling two
    /// bricks, so every brick of the rotated model would be one voxel out of
    /// step with the lattice. The result still meshes cleanly and still passes
    /// `is_printable`, which is exactly what makes it worth spelling out here.
    ///
    /// The price is that this is a rotation about the point half a voxel below
    /// the origin on each axis rather than about the origin itself, so a
    /// rotated model is displaced by at most one voxel per negated axis. At the
    /// default 0.25 mm voxel that is a quarter of a millimetre.
    pub fn apply_voxel(self, voxel: IVec3) -> IVec3 {
        let mut out = IVec3::ZERO;
        for (source, (destination, flipped)) in self.axis_map().into_iter().enumerate() {
            out[destination] = if flipped { -voxel[source] - 1 } else { voxel[source] };
        }
        out
    }
}

/// The lowest numbered axis that `facing` does not lie along.
///
/// Only used to settle the half turn case, where any perpendicular axis is
/// equally correct and the point is that the answer never changes.
fn first_axis_perpendicular_to(facing: Facing) -> IVec3 {
    let along = facing.normal().abs();
    [IVec3::X, IVec3::Y, IVec3::Z]
        .into_iter()
        .find(|axis| axis.dot(along) == 0)
        .expect("a unit axis is perpendicular to two others")
}

/// Which way a mesh's own up points, guessed from it standing on a plane.
///
/// Nothing in an STL, OBJ or 3MF states which axis is up, and the two
/// conventions in the world disagree: the printing tools that
/// [`from_print_space`] is written for are Z-up, while a mesh from a modelling
/// or generation tool is usually Y-up. Read as the wrong one, a model arrives
/// lying on its back and would be exported that way onto the plate.
///
/// The one tell that does not depend on the format is that such a model is
/// almost always built standing on the ground: its bounding box sits exactly on
/// zero along one axis and extends away from it. That is what this looks for.
///
/// `None` when nothing suggests an answer, or when more than one axis does --
/// a model touching zero on two axes says nothing about which is up. **This is
/// a guess and must be offered rather than applied**; a wrong one costs the
/// user a click, a silent wrong one costs them a print.
pub fn resting_up(positions: &[Vec3]) -> Option<Facing> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for position in positions {
        minimum = minimum.min(*position);
        maximum = maximum.max(*position);
    }
    let extent = maximum - minimum;
    let longest = extent.max_element();
    if !longest.is_finite() || longest <= 0.0 {
        return None;
    }
    // Relative, because the same model arrives in metres, millimetres or the
    // normalised units a generator emits, and an absolute epsilon would mean
    // something different in each.
    let tolerance = longest * 1.0e-5;

    let mut found = None;
    for axis in 0..3 {
        // Named rather than indexed out of `Facing::ALL`: an ordering the
        // arithmetic depends on is one a reorder can break in silence.
        let (positive, negative) = match axis {
            0 => (Facing::Right, Facing::Left),
            1 => (Facing::Up, Facing::Down),
            _ => (Facing::Front, Facing::Back),
        };
        let candidate = if minimum[axis].abs() <= tolerance && maximum[axis] > tolerance {
            Some(positive)
        } else if maximum[axis].abs() <= tolerance && minimum[axis] < -tolerance {
            Some(negative)
        } else {
            None
        };
        if candidate.is_some() {
            if found.is_some() {
                // Resting on two planes at once. A corner says nothing.
                return None;
            }
            found = candidate;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::{BRICK_DIM, BrickCoord};
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

    // --- AxisRotation ----------------------------------------------------

    /// Voxels spread over both signs of every axis, including the boundary
    /// cases where a negated index changes brick.
    const VOXELS: [IVec3; 8] = [
        IVec3::new(0, 0, 0),
        IVec3::new(31, 31, 31),
        IVec3::new(32, 0, -1),
        IVec3::new(-1, -1, -1),
        IVec3::new(-32, -33, 64),
        IVec3::new(5, -7, 11),
        IVec3::new(-100, 250, -3),
        IVec3::new(1000, -1000, 0),
    ];

    #[test]
    fn taking_actually_takes_the_face_where_it_was_asked_to() {
        for from in Facing::ALL {
            for to in Facing::ALL {
                let rotation = AxisRotation::taking(from, to);
                assert_eq!(
                    rotation.applied_to(from),
                    Some(to),
                    "{from:?} -> {to:?} did not land on its target"
                );
            }
        }
    }

    #[test]
    fn no_re_orientation_is_ever_a_reflection() {
        // The same failure the module documentation describes for the export
        // rotation, and it is more tempting here: a signed axis permutation
        // taking one face to another is easy to write with determinant -1, and
        // it would turn the whole model inside out while still meshing,
        // exporting and rendering perfectly.
        for from in Facing::ALL {
            for to in Facing::ALL {
                let rotation = AxisRotation::taking(from, to);
                assert_eq!(rotation.determinant(), 1, "{from:?} -> {to:?} is a reflection");
            }
        }
    }

    #[test]
    fn turning_a_face_to_another_and_back_is_an_identity() {
        for from in Facing::ALL {
            for to in Facing::ALL {
                let there = AxisRotation::taking(from, to);
                let back = AxisRotation::taking(to, from);
                assert!(
                    there.then(back).is_identity(),
                    "{from:?} -> {to:?} -> {from:?} did not come home"
                );
                assert_eq!(
                    there.inverse(),
                    back,
                    "the inverse of {from:?} -> {to:?} is not the way back"
                );
            }
        }
    }

    #[test]
    fn four_quarter_turns_return_every_voxel_exactly() {
        // The whole lossless claim, stated at the level it has to hold: a
        // rotation is a permutation of the lattice, so turning four times is
        // the identity on the index itself and not merely close to it. This is
        // also the cheapest test that catches the off-by-one described on
        // `apply_voxel` -- the wrong rule drifts by one voxel per turn.
        let quarter = AxisRotation::taking(Facing::Up, Facing::Front);
        for voxel in VOXELS {
            let mut turned = voxel;
            for _ in 0..4 {
                turned = quarter.apply_voxel(turned);
            }
            assert_eq!(turned, voxel, "four turns moved {voxel}");
        }
    }

    #[test]
    fn a_rotated_brick_is_still_a_brick() {
        // The property `Volume::rotated` is built on. Every voxel of a brick
        // has to land in the SAME destination brick, and that brick has to be
        // the one the rotation names at brick granularity -- otherwise a brick
        // straddles two and the rebuilt field is one voxel out of step with
        // the lattice on every negated axis.
        for from in Facing::ALL {
            for to in Facing::ALL {
                let rotation = AxisRotation::taking(from, to);
                for voxel in VOXELS {
                    let brick = BrickCoord::containing(voxel);
                    assert_eq!(
                        BrickCoord::containing(rotation.apply_voxel(voxel)),
                        BrickCoord(rotation.apply_voxel(brick.0)),
                        "{from:?} -> {to:?} split the brick holding {voxel}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_voxel_of_one_brick_lands_in_one_brick() {
        // The test above samples; this one is exhaustive over a single brick,
        // including a negative coordinate, because the failure is at the two
        // ends of the index range and a sample can miss it.
        let rotation = AxisRotation::taking(Facing::Up, Facing::Right);
        for origin in [BrickCoord::new(0, 0, 0), BrickCoord::new(-2, 3, -1)] {
            let expected = BrickCoord(rotation.apply_voxel(origin.0));
            let mut seen = std::collections::HashSet::new();
            for z in 0..BRICK_DIM as i32 {
                for y in 0..BRICK_DIM as i32 {
                    for x in 0..BRICK_DIM as i32 {
                        let landed = rotation.apply_voxel(origin.origin() + IVec3::new(x, y, z));
                        assert_eq!(BrickCoord::containing(landed), expected);
                        seen.insert(landed);
                    }
                }
            }
            // And it is a permutation, not a collapse: every voxel got its own
            // destination.
            assert_eq!(seen.len(), BRICK_DIM * BRICK_DIM * BRICK_DIM);
        }
    }

    #[test]
    fn the_half_turn_choice_is_fixed_rather_than_incidental() {
        // Turning a face to its opposite has no shortest way -- two different
        // half turns both do it, and they disagree about where everything else
        // goes. Which one this is does not matter; that it cannot quietly
        // change under a refactor does, because the model would start landing
        // in a different pose from the same click.
        let rotation = AxisRotation::taking(Facing::Up, Facing::Down);
        assert_eq!(rotation.apply(Vec3::Y), -Vec3::Y);
        assert_eq!(rotation.apply(Vec3::X), Vec3::X, "the half turn is about X");
        assert_eq!(rotation.apply(Vec3::Z), -Vec3::Z);
    }

    #[test]
    fn turning_a_face_onto_itself_does_nothing_at_all() {
        for facing in Facing::ALL {
            assert!(AxisRotation::taking(facing, facing).is_identity());
        }
        for voxel in VOXELS {
            assert_eq!(AxisRotation::IDENTITY.apply_voxel(voxel), voxel);
        }
    }

    #[test]
    fn nearest_resolves_the_six_faces_and_rejects_a_diagonal() {
        for facing in Facing::ALL {
            assert_eq!(Facing::nearest(facing.normal().as_vec3()), Some(facing));
        }
        // A cube corner points at three faces at once and is not one of them.
        assert_eq!(Facing::nearest(Vec3::new(1.0, 1.0, 1.0)), None);
    }

    // --- resting_up ------------------------------------------------------

    /// A box standing on the plane `axis == 0`, extending `sign` from it.
    fn standing_on(axis: usize, sign: f32) -> Vec<Vec3> {
        let mut tall = Vec3::new(1.0, 1.0, 1.0);
        tall[axis] = 4.0 * sign;
        let mut low = Vec3::new(-1.0, -1.0, -1.0);
        low[axis] = 0.0;
        vec![low, tall]
    }

    #[test]
    fn a_model_standing_on_a_plane_says_which_way_is_up() {
        assert_eq!(resting_up(&standing_on(0, 1.0)), Some(Facing::Right));
        assert_eq!(resting_up(&standing_on(0, -1.0)), Some(Facing::Left));
        assert_eq!(resting_up(&standing_on(1, 1.0)), Some(Facing::Up));
        assert_eq!(resting_up(&standing_on(1, -1.0)), Some(Facing::Down));
        assert_eq!(resting_up(&standing_on(2, 1.0)), Some(Facing::Front));
        assert_eq!(resting_up(&standing_on(2, -1.0)), Some(Facing::Back));
    }

    #[test]
    fn the_nightwing_reads_as_y_up_in_the_file_and_as_back_once_imported() {
        // The bounds actually measured off
        // `Meshy_AI_Prismatic_Nightwing_...obj`, which is the file that started
        // this: Y floored at exactly zero is the ground plane, X is the
        // wingspan and Z the depth.
        let file = [Vec3::new(-0.1596, 0.0, -0.1429), Vec3::new(0.1595, 0.2, 0.1449)];
        assert_eq!(resting_up(&file), Some(Facing::Up), "the file is Y-up");

        // And what the importer does to it. Read as Z-up, the model's own up
        // ends up pointing backwards in sculpt space, which is the 90 degrees
        // onto its back that the user sees.
        let imported: Vec<Vec3> = file.iter().map(|v| from_print_space(*v)).collect();
        assert_eq!(resting_up(&imported), Some(Facing::Back));

        // So this is the turn that puts it upright, and it undoes exactly the
        // rotation the import applied.
        let fix = AxisRotation::taking(Facing::Back, Facing::Up);
        for sample in file {
            let there_and_back = fix.apply(from_print_space(sample));
            assert!(
                (there_and_back - sample).length() < 1.0e-6,
                "{sample} did not come back to itself: {there_and_back}"
            );
        }
    }

    #[test]
    fn a_print_ready_model_sitting_on_the_bed_needs_no_turn() {
        // The case that must NOT produce a prompt, and the reason the guess is
        // stated as "which way is up" rather than "is this file Y-up": an STL
        // exported for a slicer sits on z = 0, and read as Z-up it arrives
        // already standing. The heuristic has to be quiet here or it would ask
        // on nearly every print file in existence.
        let file = [Vec3::new(-10.0, -10.0, 0.0), Vec3::new(10.0, 10.0, 40.0)];
        let imported: Vec<Vec3> = file.iter().map(|v| from_print_space(*v)).collect();
        assert_eq!(resting_up(&imported), Some(Facing::Up), "it is already upright");
        assert!(AxisRotation::taking(Facing::Up, Facing::Up).is_identity());
    }

    #[test]
    fn a_centred_model_and_a_cornered_one_both_say_nothing() {
        // Centred on the origin: touches no plane, so there is no tell.
        let centred = [Vec3::splat(-5.0), Vec3::splat(5.0)];
        assert_eq!(resting_up(&centred), None);

        // In the positive octant: touches three planes at once, and a corner
        // says nothing about which of them is the floor.
        let cornered = [Vec3::ZERO, Vec3::new(3.0, 4.0, 5.0)];
        assert_eq!(resting_up(&cornered), None);
    }

    #[test]
    fn a_degenerate_mesh_does_not_produce_a_guess() {
        assert_eq!(resting_up(&[]), None);
        assert_eq!(resting_up(&[Vec3::ZERO]), None, "a single point has no extent");
        assert_eq!(resting_up(&[Vec3::ZERO, Vec3::new(f32::NAN, 1.0, 1.0)]), None);
    }

    #[test]
    fn the_tolerance_is_relative_to_the_model_not_absolute() {
        // The same shape arrives in metres, millimetres and a generator's
        // normalised units, and has to read the same way in all three.
        for scale in [0.001, 1.0, 1000.0] {
            let scaled: Vec<Vec3> = standing_on(1, 1.0).iter().map(|v| *v * scale).collect();
            assert_eq!(resting_up(&scaled), Some(Facing::Up), "failed at scale {scale}");
        }
    }
}
