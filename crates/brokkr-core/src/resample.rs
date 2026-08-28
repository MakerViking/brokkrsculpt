// SPDX-License-Identifier: AGPL-3.0-only

//! Rebuilding the volume at a different voxel size.
//!
//! Resolution is uniform and fixed across the whole volume, which is what keeps
//! the sculpt loop simple and its cost predictable. The price is that getting
//! finer detail is not something a brush can do: the entire field has to be
//! resampled onto a finer lattice. That is a deliberate, explicit operation, the
//! same model Nomad's voxel remesher uses.
//!
//! # Two things that are easy to get wrong
//!
//! Distance values are stored in voxels, not world units, so they have to be
//! rescaled by the ratio of the two voxel sizes. Copying them across unchanged
//! would move the surface.
//!
//! Sampling every voxel of every brick in the new bounding box would take
//! minutes on a large model. Instead each new brick first asks what the old
//! volume holds over its extent: regions that are entirely empty or entirely
//! solid are answered from the brick structure alone, without sampling anything,
//! and only the shell is resampled.

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use crate::brick::{BRICK_DIM, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, brick_index};
use crate::region::FieldRegion;
use crate::volume::Volume;

/// Samples above which a new brick reads through the volume rather than
/// gathering the old field first.
///
/// Four million is 16 MB of `f32` per worker, which is a sensible ceiling for
/// something every core holds at once.
pub(crate) const MAX_GATHERED_SAMPLES: i64 = 4 << 20;

/// What the old volume holds over some region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Coverage {
    /// Nothing but empty space, so the new brick can stay absent.
    Empty,
    /// Nothing but the inside of a solid, so the new brick is a tile.
    Solid,
    /// Contains a surface, or a mixture, so it has to be resampled.
    Surface,
}

impl Volume {
    /// What this volume holds over an inclusive world space box, answered from
    /// the brick structure without sampling.
    pub(crate) fn coverage(&self, minimum: Vec3, maximum: Vec3) -> Coverage {
        let (v_min, v_max) = self.voxel_bounds(minimum, maximum);
        let b_min = BrickCoord::containing(v_min).0;
        let b_max = BrickCoord::containing(v_max).0;

        let mut any_empty = false;
        let mut any_solid = false;
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    match self.brick(BrickCoord::new(bx, by, bz)) {
                        None => any_empty = true,
                        Some(Brick::Uniform(value)) if *value <= INSIDE => any_solid = true,
                        // A tile of anything other than fully inside, or a brick
                        // with real values, means there could be a surface here.
                        Some(_) => return Coverage::Surface,
                    }
                    if any_empty && any_solid {
                        return Coverage::Surface;
                    }
                }
            }
        }

        match (any_empty, any_solid) {
            (_, true) => Coverage::Solid,
            _ => Coverage::Empty,
        }
    }

    /// Rebuild this volume at a different voxel size.
    ///
    /// Cost is proportional to the surface, not to the bounding box: empty and
    /// solid regions are recognised from the brick structure and cost nothing.
    /// The work that remains runs across every core.
    pub fn resampled(&self, voxel_size: f32) -> Volume {
        assert!(voxel_size > 0.0, "voxel size must be positive");

        let mut resampled = Volume::new(voxel_size);
        // Before the empty-field return, not after it: a mask can protect empty
        // space -- that is what stops Draw growing material there -- so a body
        // with no bricks at all can still have one to carry.
        *resampled.mask_mut() = self.mask().resampled(self.voxel_size(), voxel_size);
        let Some((world_min, world_max)) = self.world_bounds() else {
            return resampled;
        };

        // Values are in voxels, so shrinking the voxel size stretches them.
        let rescale = self.voxel_size() / voxel_size;
        // A margin so the new surface's narrow band has room either side.
        let margin = Vec3::splat(NARROW_BAND * voxel_size * 2.0);
        let new_min =
            BrickCoord::containing(((world_min - margin) / voxel_size).floor().as_ivec3()).0;
        let new_max =
            BrickCoord::containing(((world_max + margin) / voxel_size).ceil().as_ivec3()).0;

        let mut coords = Vec::new();
        for bz in new_min.z..=new_max.z {
            for by in new_min.y..=new_max.y {
                for bx in new_min.x..=new_max.x {
                    coords.push(BrickCoord::new(bx, by, bz));
                }
            }
        }

        let dim = BRICK_DIM as i32;
        let built: Vec<(BrickCoord, Brick)> = coords
            .par_iter()
            .map_init(FieldRegion::new, |region, coord| {
                let origin = coord.origin();
                let brick_min = origin.as_vec3() * voxel_size;
                let brick_max = (origin + IVec3::splat(dim - 1)).as_vec3() * voxel_size;

                // Answer the cheap cases from structure alone.
                match self.coverage(brick_min, brick_max) {
                    Coverage::Empty => return None,
                    Coverage::Solid => return Some((*coord, Brick::Uniform(INSIDE))),
                    Coverage::Surface => {}
                }

                // Gather the old field over this brick's extent once, then read
                // it from a flat array rather than through the brick map. A
                // trilinear sample is eight reads, and doing those as hash
                // lookups is what would make this take minutes.
                //
                // The region covering one new brick grows with the ratio of the
                // two voxel sizes, though. Refining keeps it small, which is the
                // case that matters, but coarsening by a large factor would ask
                // for a box of billions of samples. Past a limit, read through
                // the volume instead: slower per sample, but a coarsening step
                // has far fewer bricks to fill.
                let (old_lo, old_hi) = self.voxel_bounds(brick_min, brick_max);
                let gathered_samples =
                    (old_hi - old_lo + IVec3::splat(3)).as_i64vec3().element_product();
                let gathered = gathered_samples <= MAX_GATHERED_SAMPLES;
                if gathered {
                    self.snapshot(old_lo, old_hi, region);
                }

                let mut brick = Brick::dense_filled(OUTSIDE);
                let data = brick.make_dense();
                let scale = self.voxel_size();
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let voxel = origin + IVec3::new(x as i32, y as i32, z as i32);
                            let world = voxel.as_vec3() * voxel_size;
                            let value = if gathered {
                                region.sample(world / scale)
                            } else {
                                self.sample_world(world)
                            } * rescale;
                            data[brick_index(x, y, z)] = value.clamp(INSIDE, OUTSIDE);
                        }
                    }
                }

                // A brick that resampled to nothing but empty or solid does not
                // need its allocation.
                match brick.is_collapsible() {
                    Some(value) if value >= OUTSIDE => None,
                    Some(value) => Some((*coord, Brick::Uniform(value))),
                    None => Some((*coord, brick)),
                }
            })
            .flatten()
            .collect();

        for (coord, brick) in built {
            resampled.insert_brick(coord, brick);
        }
        resampled.mark_everything_dirty();
        resampled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp};

    const RADIUS: f32 = 20.0;

    fn sphere(voxel_size: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::ZERO, RADIUS);
        volume
    }

    /// Where the surface actually sits along an axis, found by walking outward.
    fn measured_radius(volume: &Volume) -> f32 {
        let step = volume.voxel_size() * 0.1;
        let mut last = 0.0;
        for index in 0..20_000 {
            let t = index as f32 * step;
            if volume.sample_world(Vec3::new(t, 0.0, 0.0)) >= 0.0 {
                return last;
            }
            last = t;
        }
        last
    }

    /// The detail button rebuilds the field on a new lattice, and the mask has
    /// to arrive at the same WORLD point rather than the same voxel index.
    ///
    /// Dropping it there would be silent: the resample already clears the undo
    /// history, so there is nothing to press to get the protection back.
    #[test]
    fn resampling_carries_the_mask_to_the_same_world_point() {
        use crate::{PROTECTED, UNMASKED};
        use glam::IVec3;

        let mut coarse = sphere(1.0);
        let world = Vec3::new(RADIUS, 0.0, 0.0);
        let cell = (world / coarse.voxel_size()).round().as_ivec3();
        coarse.mask_mut().write(cell, PROTECTED);

        let fine = coarse.resampled(0.5);

        let fine_cell = (world / 0.5).round().as_ivec3();
        assert_eq!(fine.voxel_size(), 0.5, "the fixture did not resample");
        assert_eq!(
            fine.mask().at(fine_cell),
            PROTECTED,
            "the mask is not on the world point it was painted on"
        );
        assert_eq!(
            fine.mask().at(fine_cell + IVec3::new(0, 0, 40)),
            UNMASKED,
            "the mask spread across the model"
        );
    }

    /// A mask over empty space is real -- it is what stops Draw growing
    /// material there -- so a body with no bricks at all can still have one to
    /// carry, and the early return for an empty field must not drop it.
    #[test]
    fn resampling_an_empty_body_still_carries_its_mask() {
        use crate::PROTECTED;

        let mut empty = Volume::new(1.0);
        empty.mask_mut().write(IVec3::new(4, 4, 4), PROTECTED);
        assert_eq!(empty.brick_count(), 0, "the fixture must hold no geometry");

        let fine = empty.resampled(0.5);
        assert_eq!(fine.mask().at(IVec3::new(8, 8, 8)), PROTECTED);
    }

    /// Mask All is a polarity bit over an EMPTY brick map, so for that state the
    /// bit is the entire mask and a resample that copies no bricks carries
    /// nothing else -- and the resample clears the undo history, so losing it
    /// leaves nothing to press to get the protection back.
    ///
    /// The reading assertion is the load-bearing one: `inverted()` pins the
    /// current representation, but `at()` on a cell nothing ever wrote is what
    /// still holds if the bool is ever refactored away.
    #[test]
    fn resampling_carries_mask_all_even_though_it_moves_no_brick() {
        use crate::PROTECTED;

        let mut coarse = sphere(1.0);
        coarse.mask_mut().set_inverted(true);
        assert_eq!(coarse.mask().map_bytes(), 0, "Mask All must store no bricks at all");

        let fine = coarse.resampled(0.5);

        assert_eq!(fine.voxel_size(), 0.5, "the fixture did not resample");
        assert!(fine.mask().inverted(), "the resample dropped the mask's polarity");
        assert_eq!(
            fine.mask().at(IVec3::new(-317, 44, 900)),
            PROTECTED,
            "a body that was fully masked came back sculptable"
        );
    }

    #[test]
    fn resampling_finer_keeps_the_surface_where_it_was() {
        // The mistake this guards against: values are stored in voxels, so
        // copying them across without rescaling moves the surface.
        let coarse = sphere(1.0);
        let before = measured_radius(&coarse);

        let fine = coarse.resampled(0.25);
        let after = measured_radius(&fine);

        assert!(
            (after - before).abs() < 1.0,
            "the surface moved from {before} to {after} when resampled"
        );
        assert!((after - RADIUS).abs() < 1.0, "and it should still be near {RADIUS}, got {after}");
        assert_eq!(fine.voxel_size(), 0.25);
    }

    #[test]
    fn resampling_finer_gives_more_detail_and_stays_watertight() {
        let coarse = sphere(1.0);
        let (_, coarse_report) = coarse.export_mesh();

        let fine = coarse.resampled(0.4);
        let (_, fine_report) = fine.export_mesh();

        assert!(
            fine_report.triangles > coarse_report.triangles * 3,
            "expected far more triangles: {} against {}",
            fine_report.triangles,
            coarse_report.triangles
        );
        assert!(
            fine_report.is_printable(),
            "a resampled model must still print: {}",
            fine_report.summary()
        );
    }

    #[test]
    fn resampling_coarser_works_too() {
        let fine = sphere(0.5);
        let coarse = fine.resampled(2.0);
        let (_, report) = coarse.export_mesh();

        assert!(report.is_printable(), "{}", report.summary());
        assert!((measured_radius(&coarse) - RADIUS).abs() < 2.5);
    }

    #[test]
    fn a_sculpted_feature_survives_the_resample() {
        // Detail that a brush put there has to come through, or the operation
        // would quietly throw away the user's work.
        let mut volume = sphere(0.5);
        let mut scratch = BrushScratch::new();
        let brush = Brush { kind: BrushKind::Draw, radius: 4.0, strength: 0.9, ..Brush::default() };
        let at = Vec3::new(0.0, RADIUS, 0.0);
        let normal = volume.gradient_world(at);
        for _ in 0..6 {
            brush.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        }

        let bump_before = volume.sample_world(at);
        let resampled = volume.resampled(0.25);
        let bump_after = resampled.sample_world(at);

        // Both are inside the added material, and in the same place.
        assert!(bump_before < 0.0, "the brush should have added material: {bump_before}");
        assert!(bump_after < 0.0, "the added material did not survive the resample: {bump_after}");
    }

    #[test]
    fn the_interior_stays_tiles_rather_than_becoming_dense() {
        // Resampling a solid ball must not allocate its whole inside. Without
        // the structural shortcut this is what would blow the memory budget.
        //
        // The source has to be several bricks across for there to be an interior
        // at all: at a 1 mm voxel a 20 mm ball is barely more than one brick
        // wide, and every brick of it is shell.
        let source = sphere(0.25);
        assert!(
            source.stats().uniform_bricks > 0,
            "the test needs a source with an interior to preserve"
        );

        let fine = source.resampled(0.125);
        let stats = fine.stats();
        assert!(stats.uniform_bricks > 0, "the interior should have stayed tiles");

        // The direct statement: the brick at the centre of the ball holds no
        // detail, so it must not be carrying 128 KB of voxels.
        let centre = BrickCoord::containing(IVec3::ZERO);
        match fine.brick(centre) {
            Some(Brick::Uniform(value)) => {
                assert!(*value <= INSIDE, "the centre of a solid ball should read as inside")
            }
            Some(Brick::Dense(_)) => {
                panic!("the brick at the centre of the ball was filled in with voxels")
            }
            None => panic!("the centre of a solid ball should not be empty space"),
        }

        assert!(
            stats.dense_bricks < fine.brick_count(),
            "every brick came out dense, so nothing was recognised as solid"
        );
    }

    #[test]
    fn resampling_an_empty_volume_gives_an_empty_volume() {
        let empty = Volume::new(1.0);
        let resampled = empty.resampled(0.5);
        assert_eq!(resampled.brick_count(), 0);
        assert_eq!(resampled.voxel_size(), 0.5);
    }

    #[test]
    fn resampling_to_the_same_size_is_close_to_a_copy() {
        let original = sphere(1.0);
        let again = original.resampled(1.0);
        assert!((measured_radius(&again) - measured_radius(&original)).abs() < 0.2);
    }

    #[test]
    fn everything_is_marked_dirty_so_the_renderer_rebuilds() {
        // A resample replaces every brick. Anything left unmarked would keep
        // showing the old model's triangles.
        let mut resampled = sphere(1.0).resampled(0.5);
        assert!(resampled.dirty_count() >= resampled.brick_count());
        let mut dirty = Vec::new();
        resampled.take_dirty(&mut dirty);
        assert!(!dirty.is_empty());
    }
}
