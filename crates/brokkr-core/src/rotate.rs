// SPDX-License-Identifier: AGPL-3.0-only

//! Turning the whole model onto a different face.
//!
//! An imported mesh does not always arrive standing up. The printing formats
//! are read as Z-up (see [`crate::orientation`]), but an OBJ from a modelling
//! or generation tool is usually Y-up, and there is nothing in the bytes that
//! says which. So the user has to be able to say "that face is down", and this
//! is what carries it out.
//!
//! # Why this is not a resample
//!
//! [`Volume::resampled`] rebuilds the field by trilinear sampling, and every
//! pass through it costs a little of the surface. A rotation does not have to
//! work that way, because the only re-orientations offered are multiples of a
//! quarter turn, and those map the lattice exactly onto itself:
//!
//! - Distance values are stored in voxels, and a rotation preserves distance,
//!   so not one value is recomputed or rescaled.
//! - [`crate::BRICK_DIM`] is the same on all three axes, so a brick turns into
//!   a brick rather than across a boundary. Tiles stay tiles at no cost at all,
//!   and a dense brick is one permuted copy of its 32768 values.
//!
//! The result is bit identical to the original after four quarter turns, which
//! is the property `four_quarter_turns_of_a_sculpt_are_bit_identical` pins.
//! Nothing else in the engine can claim that.
//!
//! # What it costs, and what is deliberately not kept
//!
//! Work is proportional to the resident dense bricks -- one 128 KB read and one
//! 128 KB write each, across every core. There is no structural shortcut to
//! find because there is no sampling to avoid.
//!
//! Undo history is **not** carried across, for the same reason
//! [`Volume::resampled`]'s caller drops it: every entry names a brick of a
//! volume that no longer exists. A full field snapshot would also be pointless
//! here, since it would exceed the history budget on any real model. The
//! recovery path is that a rotation is exactly reversible -- turn it back and
//! the field returns unchanged, which is not true of anything else that
//! rebuilds the volume.

use rayon::prelude::*;

use crate::brick::{BRICK_DIM, Brick, BrickCoord, brick_index};
use crate::orientation::AxisRotation;
use crate::volume::Volume;

impl Volume {
    /// Rebuild this volume turned by `rotation`.
    ///
    /// Exact: the result holds the same distance values as this one, moved to
    /// different voxels. See the module documentation for why that is possible
    /// and what it does not preserve.
    pub fn rotated(&self, rotation: AxisRotation) -> Volume {
        let mut rotated = Volume::new(self.voxel_size());

        // Which destination axis each source axis feeds, and whether the index
        // runs backwards along it. Resolved once for the whole volume rather
        // than as a matrix multiply per voxel.
        let map = rotation.axis_map();
        let last = BRICK_DIM - 1;

        let coords: Vec<BrickCoord> = self.brick_coords().collect();
        let built: Vec<(BrickCoord, Brick)> = coords
            .par_iter()
            .map(|coord| {
                let source = self.brick(*coord).expect("the coordinate came from the brick map");
                // The same rule at brick granularity as at voxel granularity.
                // `a_rotated_brick_is_still_a_brick` is what says those two
                // agree, and the whole approach rests on it.
                let destination = BrickCoord(rotation.apply_voxel(coord.0));

                let turned = match source {
                    // A tile holds one value everywhere, so turning it is
                    // nothing at all -- and it stays a tile, which is what
                    // keeps a solid interior from being allocated.
                    Brick::Uniform(value) => Brick::Uniform(*value),
                    Brick::Dense(data) => {
                        let mut turned = Brick::dense_filled(0.0);
                        let out = turned.make_dense();
                        for z in 0..BRICK_DIM {
                            for y in 0..BRICK_DIM {
                                for x in 0..BRICK_DIM {
                                    let from = [x, y, z];
                                    let mut to = [0usize; 3];
                                    for (axis, (into, flipped)) in map.into_iter().enumerate() {
                                        to[into] =
                                            if flipped { last - from[axis] } else { from[axis] };
                                    }
                                    out[brick_index(to[0], to[1], to[2])] =
                                        data[brick_index(x, y, z)];
                                }
                            }
                        }
                        turned
                    }
                };
                (destination, turned)
            })
            .collect();

        // No `is_collapsible` pass: a permutation of a brick's values holds
        // exactly the values it held before, so a brick that was worth storing
        // densely still is. Scanning 32768 values to learn that would cost as
        // much as the copy.
        for (coord, brick) in built {
            rotated.insert_brick(coord, brick);
        }
        rotated.mark_everything_dirty();
        rotated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::INSIDE;
    use crate::orientation::Facing;
    use glam::Vec3;

    fn sphere(voxel_size: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        volume
    }

    /// Every stored voxel, in a deterministic order, for comparing two volumes
    /// value by value rather than through a sampled surface.
    fn field(volume: &Volume) -> Vec<(BrickCoord, Vec<f32>)> {
        let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
        coords.sort_unstable();
        coords
            .into_iter()
            .map(|coord| {
                let brick = volume.brick(coord).expect("came from the map");
                let mut values = Vec::with_capacity(BRICK_DIM * BRICK_DIM * BRICK_DIM);
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            values.push(brick.get(x, y, z));
                        }
                    }
                }
                (coord, values)
            })
            .collect()
    }

    #[test]
    fn four_quarter_turns_of_a_sculpt_are_bit_identical() {
        // The claim the whole module rests on, and the reason a rotation is not
        // a resample. Not "close to": the same bits, because no value was ever
        // recomputed.
        let source = sphere(0.5);
        let quarter = AxisRotation::taking(Facing::Up, Facing::Front);

        let mut turned = source.rotated(quarter);
        for _ in 0..3 {
            turned = turned.rotated(quarter);
        }

        assert_eq!(field(&turned), field(&source), "four quarter turns did not come home");
    }

    #[test]
    fn a_turn_and_its_inverse_are_bit_identical_too() {
        // Stated separately from the four-turn case because this is the one the
        // user reaches for: the recovery path for a wrong click is to turn it
        // back, and it has to return the sculpt rather than something like it.
        let source = sphere(0.5);
        for from in Facing::ALL {
            for to in Facing::ALL {
                let there = AxisRotation::taking(from, to);
                let back = there.inverse();
                let round_trip = source.rotated(there).rotated(back);
                assert_eq!(
                    field(&round_trip),
                    field(&source),
                    "{from:?} -> {to:?} and back did not return the field"
                );
            }
        }
    }

    #[test]
    fn a_turn_moves_the_model_the_way_it_says() {
        // A sphere is symmetric, so it would pass every test above even if the
        // rotation did nothing. Put material somewhere off axis and check it
        // arrives where the rotation promised.
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::new(0.0, 30.0, 0.0), 8.0);

        // Up becomes front, so material that was above the origin should end up
        // in front of it.
        let turned = volume.rotated(AxisRotation::taking(Facing::Up, Facing::Front));

        assert!(volume.sample_world(Vec3::new(0.0, 30.0, 0.0)) < 0.0, "the fixture is wrong");
        assert!(
            turned.sample_world(Vec3::new(0.0, 0.0, 30.0)) < 0.0,
            "the material did not arrive in front of the origin"
        );
        assert!(
            turned.sample_world(Vec3::new(0.0, 30.0, 0.0)) >= 0.0,
            "the material is still above the origin, so nothing turned"
        );
    }

    #[test]
    fn the_interior_stays_tiles_rather_than_becoming_dense() {
        // The same guarantee `resampled` has to work for, except here it is
        // free: a tile turns into a tile without being looked at.
        let source = sphere(0.25);
        assert!(
            source.stats().uniform_bricks > 0,
            "the test needs a source with an interior to preserve"
        );

        let turned = source.rotated(AxisRotation::taking(Facing::Up, Facing::Right));
        assert_eq!(
            turned.stats().uniform_bricks,
            source.stats().uniform_bricks,
            "a solid interior should have cost nothing to turn"
        );
        assert_eq!(turned.stats().dense_bricks, source.stats().dense_bricks);
    }

    #[test]
    fn a_turned_model_is_still_printable_and_the_same_size() {
        // Rotating is a rigid motion, so the surface it meshes to has to be the
        // same surface. A reflection would also keep the triangle count and
        // would export inside out, which is why `orientation` pins the
        // determinant separately.
        let source = sphere(0.5);
        let (before, before_report) = source.export_mesh();
        let turned = source.rotated(AxisRotation::taking(Facing::Up, Facing::Down));
        let (after, after_report) = turned.export_mesh();

        assert_eq!(after.positions.len(), before.positions.len());
        assert_eq!(after.triangles.len(), before.triangles.len());
        assert!(before_report.is_printable(), "the fixture is not printable");
        assert!(
            after_report.is_printable(),
            "turning the model broke it: {}",
            after_report.summary()
        );
    }

    #[test]
    fn an_empty_volume_turns_into_an_empty_volume() {
        let empty = Volume::new(0.5);
        let turned = empty.rotated(AxisRotation::taking(Facing::Up, Facing::Back));
        assert_eq!(turned.brick_count(), 0);
        assert_eq!(turned.voxel_size(), 0.5);
    }

    #[test]
    fn the_identity_turn_gives_the_field_back() {
        let source = sphere(0.5);
        let same = source.rotated(AxisRotation::IDENTITY);
        assert_eq!(field(&same), field(&source));
    }

    #[test]
    fn every_brick_of_a_turned_model_is_dirty() {
        // `voxelise` and `project::read` both mark everything, so a caller can
        // never install a rebuilt volume and get an invisible model. This one
        // has to do the same.
        //
        // The dirty set is deliberately larger than the stored bricks -- it
        // takes in each one's 26 neighbours, because a brick with no voxels of
        // its own still owns the quads on its low faces -- so the claim to make
        // is containment, not equality.
        let source = sphere(0.5);
        let mut turned = source.rotated(AxisRotation::taking(Facing::Up, Facing::Left));
        assert!(turned.brick_count() > 0, "the fixture is empty");

        let stored: Vec<BrickCoord> = turned.brick_coords().collect();
        let mut dirty = Vec::new();
        turned.take_dirty(&mut dirty);
        let dirty: std::collections::HashSet<BrickCoord> = dirty.into_iter().collect();
        for coord in stored {
            assert!(dirty.contains(&coord), "{coord:?} carries geometry and was not marked dirty");
        }
    }

    #[test]
    fn a_solid_tile_keeps_its_value() {
        // The tile path skips the voxel loop entirely, so it is the one place a
        // value could be dropped without any test of the surface noticing.
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 40.0);
        let turned = volume.rotated(AxisRotation::taking(Facing::Front, Facing::Up));
        let tiles: Vec<f32> = turned
            .brick_coords()
            .filter_map(|coord| match turned.brick(coord) {
                Some(Brick::Uniform(value)) => Some(*value),
                _ => None,
            })
            .collect();
        assert!(!tiles.is_empty(), "the fixture has no interior tiles");
        assert!(tiles.iter().all(|value| *value == INSIDE), "an interior tile lost its value");
    }
}
