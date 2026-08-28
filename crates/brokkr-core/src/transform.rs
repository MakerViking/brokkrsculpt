// SPDX-License-Identifier: AGPL-3.0-only

//! Moving, turning and scaling one body through a [`Similarity`].
//!
//! Two routes, and which one a placement takes is decided once by
//! [`Similarity::route`] rather than separately here, in the status line and in
//! the undo entry.
//!
//! # [`Volume::shifted`] is a gather, not a resample
//!
//! [`Volume::duplicated`]'s header says a sub-brick offset "would have to
//! rebuild every brick from two neighbours, which is the resample this whole
//! lattice exists to avoid, and it would cost the surface a little detail on
//! the way through". The first half is true and the second is not, at
//! whole-voxel granularity: every destination voxel takes exactly one source
//! voxel's `f32`, with no interpolation and no arithmetic performed on the
//! value at all. There is no detail to lose because nothing is recomputed. What
//! it does cost is the rebuild -- eight source bricks may feed one destination
//! brick -- which is why it delegates to `duplicated` whenever the offset turns
//! out to be brick aligned.
//!
//! Distance values are stored in voxels and a translation preserves distance,
//! so, exactly as in [`crate::rotate`], not one value is rescaled.
//!
//! # [`Volume::warped`] is [`Volume::resampled`] with a map in the middle
//!
//! Same skeleton: walk destination bricks, ask the source what it holds over
//! the region each one came from, answer the empty and solid cases from the
//! brick structure and sample only the shell. Three differences, each of which
//! is a place this could go quietly wrong:
//!
//! * The region a destination brick came from is [`Similarity::inverse_bounds`]
//!   of its world box rather than the box itself. A box derived from the
//!   SOURCE bounds instead would drop the corners of a turned model, which
//!   looks like a mesher bug rather than a transform one.
//! * The value rescale is `by.scale` and not a voxel-size ratio, because
//!   `voxel_size` does not change. That is the invariant the approved plan's
//!   refusal of per-body scale exists to protect --
//!   `Document::lattice_agrees()` still holds by construction, because nothing
//!   here touches the lattice.
//! * The destination footprint has to cover where the model is GOING. A
//!   rotation of forty-five degrees pushes the corners of a bounding box out by
//!   a factor of root two, and a scale pushes them out by the scale.
//!
//! # What repeating it costs, and who is responsible for bounding that
//!
//! One `warped` pass is one trilinear resample, which is lossy in the way every
//! pass through [`Volume::resampled`] is lossy. Nothing here can make that
//! cheaper. What CAN be bounded is how many passes a body suffers, and that is
//! the caller's job and not this module's: the gizmo holds the field as it was
//! when it armed and re-bakes every release FROM that, so thirty adjustments
//! cost one pass rather than thirty. See `Brokkr::rebake_gizmo`.

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use crate::brick::{BRICK_DIM, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, brick_index};
use crate::similarity::Similarity;
use crate::volume::Volume;

thread_local! {
    /// How many trilinear passes [`Volume::warped`] has run on this thread.
    ///
    /// Thread local rather than global for the reason
    /// [`crate::volume::copies_made_on_this_thread`] gives: the test suite runs
    /// in parallel threads and a shared counter would race.
    ///
    /// **It exists because "one gesture is one pass" is the claim the whole
    /// anti-degradation design rests on, and nothing else can see it.** Erosion
    /// after thirty passes is real but hard to assert on without a tolerance
    /// that would also pass for two; the number of passes is the thing that was
    /// actually promised, so it is the thing that is counted.
    static WARPS_MADE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The value of that counter. Take it before and after and compare.
pub fn warps_made_on_this_thread() -> usize {
    WARPS_MADE.with(std::cell::Cell::get)
}

/// The destination box for a source box: the eight corners forward.
///
/// Replaces `by.inverse().inverse_bounds(..)`, which no longer type-checks and
/// could not be made to: with a per-axis scale the inverse of this map is not
/// itself expressible as one. Same eight-corner argument, same conservative
/// direction.
pub(crate) fn forward_bounds(by: Similarity, low: Vec3, high: Vec3) -> (Vec3, Vec3) {
    let mut result_low = Vec3::splat(f32::INFINITY);
    let mut result_high = Vec3::splat(f32::NEG_INFINITY);
    for corner in 0..8 {
        let point = Vec3::new(
            if corner & 1 == 0 { low.x } else { high.x },
            if corner & 2 == 0 { low.y } else { high.y },
            if corner & 4 == 0 { low.z } else { high.z },
        );
        let moved = by.transform_point(point);
        result_low = result_low.min(moved);
        result_high = result_high.max(moved);
    }
    (result_low, result_high)
}

impl Volume {
    /// This field moved by a whole number of VOXELS.
    ///
    /// Value-exact: see the module header for why a gather at whole-voxel
    /// granularity is not a resample. Delegates to [`Volume::duplicated`] when
    /// the offset is a whole number of bricks, which moves `Box` pointers
    /// instead of voxels.
    pub fn shifted(&self, offset_voxels: IVec3) -> Volume {
        let dim = BRICK_DIM as i32;
        if offset_voxels.rem_euclid(IVec3::splat(dim)) == IVec3::ZERO {
            return self.duplicated(offset_voxels / dim);
        }

        let mut shifted = Volume::new(self.voxel_size());

        // Every destination brick some source brick reaches. A source brick
        // moved by a sub-brick offset straddles up to eight of them, and the
        // destination is built from that set rather than from a bounding box so
        // that a sparse shell stays sparse.
        let mut wanted: rustc_hash::FxHashSet<BrickCoord> = rustc_hash::FxHashSet::default();
        for coord in self.brick_coords() {
            let low = BrickCoord::containing(coord.origin() + offset_voxels).0;
            let high = BrickCoord::containing(coord.max_voxel() + offset_voxels).0;
            for bz in low.z..=high.z {
                for by in low.y..=high.y {
                    for bx in low.x..=high.x {
                        wanted.insert(BrickCoord::new(bx, by, bz));
                    }
                }
            }
        }

        let coords: Vec<BrickCoord> = wanted.into_iter().collect();
        let built: Vec<(BrickCoord, Brick)> = coords
            .par_iter()
            .filter_map(|coord| {
                let source_low = coord.origin() - offset_voxels;
                let source_high = coord.max_voxel() - offset_voxels;

                // The cheap case, and on a solid interior it is most of them:
                // if every source brick this one reads from holds one value,
                // so does this one, and its 32768 voxels never exist.
                if let Some(value) = self.uniform_over(source_low, source_high) {
                    return (value < OUTSIDE).then_some((*coord, Brick::Uniform(value)));
                }

                let mut brick = Brick::dense_filled(OUTSIDE);
                let data = brick.make_dense();
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let voxel = source_low + IVec3::new(x as i32, y as i32, z as i32);
                            // `sample_voxel`, not `sample_world`: one stored
                            // f32 copied across, which is what makes this
                            // exact.
                            data[brick_index(x, y, z)] = self.sample_voxel(voxel);
                        }
                    }
                }
                match brick.is_collapsible() {
                    Some(value) if value >= OUTSIDE => None,
                    Some(value) => Some((*coord, Brick::Uniform(value))),
                    None => Some((*coord, brick)),
                }
            })
            .collect();

        for (coord, brick) in built {
            shifted.insert_brick(coord, brick);
        }
        // The mask moves with the body it protects, to the same world voxel,
        // and by its OWN bricks: protection over empty space is real, so a mask
        // brick with no field brick under it would be left behind by a walk of
        // `brick_coords`.
        *shifted.mask_mut() = self.mask().shifted(offset_voxels);
        shifted.mark_everything_dirty();
        shifted
    }

    /// This field rebuilt through a similarity, onto the SAME lattice.
    ///
    /// `voxel_size` is untouched, which is what keeps every body in a document
    /// on one lattice -- the invariant the approved plan's refusal of per-body
    /// scale exists to protect. Scaling a body here changes what it OCCUPIES,
    /// not what a voxel measures.
    ///
    /// Lossy, in the same way and to the same degree as
    /// [`Volume::resampled`]: one trilinear pass. Callers that can take the
    /// exact route must, and [`Similarity::route`] is what tells them whether
    /// they can.
    pub fn warped(&self, by: Similarity) -> Volume {
        self.warped_inner(by, true)
    }

    /// The pass, with the source gather switchable so a test can prove that
    /// reading through a gathered region and reading through the brick map
    /// give the same field, rather than inferring it from a bench run.
    fn warped_inner(&self, by: Similarity, gather: bool) -> Volume {
        WARPS_MADE.with(|made| made.set(made.get() + 1));
        let voxel_size = self.voxel_size();
        let mut warped = Volume::new(voxel_size);
        // Before the empty-field return, exactly as in `resampled` and for the
        // same reason: a mask can protect empty space, so a body with no bricks
        // at all can still have one to carry.
        *warped.mask_mut() = self.mask().warped(by, voxel_size);
        let Some((world_min, world_max)) = self.world_bounds() else {
            return warped;
        };
        if !by.scale.is_finite() || by.scale.min_element() <= 0.0 {
            return warped;
        }

        // Where the content is GOING. The forward image of the source box, by
        // the same eight-corner argument `inverse_bounds` makes backwards.
        let (moved_min, moved_max) = forward_bounds(by, world_min, world_max);
        // Room either side for the new surface's narrow band, and the band is
        // measured on the destination lattice.
        let margin = Vec3::splat(NARROW_BAND * voxel_size * 2.0);
        let new_min =
            BrickCoord::containing(((moved_min - margin) / voxel_size).floor().as_ivec3()).0;
        let new_max =
            BrickCoord::containing(((moved_max + margin) / voxel_size).ceil().as_ivec3()).0;

        let mut coords = Vec::new();
        for bz in new_min.z..=new_max.z {
            for by_ in new_min.y..=new_max.y {
                for bx in new_min.x..=new_max.x {
                    coords.push(BrickCoord::new(bx, by_, bz));
                }
            }
        }

        let dim = BRICK_DIM as i32;
        let built: Vec<(BrickCoord, Brick)> = coords
            .par_iter()
            .map_init(crate::region::FieldRegion::new, |region, coord| {
                let origin = coord.origin();
                let brick_min = origin.as_vec3() * voxel_size;
                let brick_max = (origin + IVec3::splat(dim - 1)).as_vec3() * voxel_size;

                // Answer the cheap cases from the source's brick structure, over
                // the region this brick actually came from.
                let (source_min, source_max) = by.inverse_bounds(brick_min, brick_max);
                match self.coverage(source_min, source_max) {
                    crate::resample::Coverage::Empty => return None,
                    crate::resample::Coverage::Solid => {
                        return Some((*coord, Brick::Uniform(INSIDE)));
                    }
                    crate::resample::Coverage::Surface => {}
                }

                // Gather the source over this brick's own inverse image once,
                // then read it from a flat array. A trilinear sample is eight
                // reads and `sample_world` does each as a hash lookup, so a
                // dense brick was paying 262,144 of them -- measured at 68.7 ns
                // a sample against 20.1 ns through a gathered region. This is
                // the same treatment `Volume::resampled` has always had; the
                // two loops are siblings and only this one was missing it.
                //
                // The `+1` padding `snapshot` applies is provably enough:
                // `inverse_transform_point` is affine, so the image of the
                // destination brick's box lies inside the box `inverse_bounds`
                // returns, and only a trilinear sample's own neighbours can
                // reach one voxel past it.
                //
                // Past `MAX_GATHERED_SAMPLES` fall back to reading through the
                // volume, for the reason `resampled` gives: a large scale-up
                // asks for a source box of billions of samples, and it has
                // proportionally few bricks to fill.
                let (source_lo, source_hi) = self.voxel_bounds(source_min, source_max);
                let gathered_samples =
                    (source_hi - source_lo + IVec3::splat(3)).as_i64vec3().element_product();
                let gathered = gather && gathered_samples <= crate::resample::MAX_GATHERED_SAMPLES;
                if gathered {
                    self.snapshot(source_lo, source_hi, region);
                }

                let mut brick = Brick::dense_filled(OUTSIDE);
                let data = brick.make_dense();
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let voxel = origin + IVec3::new(x as i32, y as i32, z as i32);
                            let world = voxel.as_vec3() * voxel_size;
                            // `d'(p) = s * d(T_inverse(p))`. For a UNIFORM
                            // scale that is exactly the field of the
                            // transformed solid. For a per-axis one it is
                            // `s_min` -- the zero set stays exact and every
                            // other distance is underestimated, which is the
                            // sound direction for a sphere trace: it steps
                            // short, never through. `Volume::redistance` is
                            // what puts the magnitudes back afterwards.
                            //
                            // **The saturation plateau is NOT scaled with the
                            // measured distances**, and this is a legality
                            // repair rather than an accuracy one. A stored
                            // `+/-NARROW_BAND` does not mean "three voxels
                            // away", it means "further than the band reaches
                            // and we stopped counting". Multiplying it by `s`
                            // turns the whole far field into what reads as a
                            // measured distance: at `s = 0.5` every value lands
                            // inside `+/-1.5` and not one voxel is saturated
                            // any more -- measured, 971,210 saturated voxels
                            // became zero. `in_band` then accepts the plateau,
                            // `measurable` stops skipping it, and a flat region
                            // has no gradient, so `gradient_drift` reports 1.0
                            // for a field whose real drift is 0.
                            //
                            // Writing the band edge back is a LIE about
                            // distance -- the true clearance is only `3s` -- and
                            // it is only sound because `redistance` runs
                            // straight afterwards and re-solves the band
                            // outward from the interface. It must NOT be
                            // adopted as a policy in place of redistancing: at
                            // `MIN_SCALE` it would claim three voxels of
                            // clearance where 0.15 exists, and `raycast`
                            // sphere-marches on exactly these values.
                            let from = by.inverse_transform_point(world);
                            let source = if gathered {
                                region.sample(from / voxel_size)
                            } else {
                                self.sample_world(from)
                            };
                            let value = if source.abs() >= OUTSIDE {
                                source
                            } else {
                                source * by.min_scale()
                            };
                            data[brick_index(x, y, z)] = value.clamp(INSIDE, OUTSIDE);
                        }
                    }
                }

                match brick.is_collapsible() {
                    Some(value) if value >= OUTSIDE => None,
                    Some(value) => Some((*coord, Brick::Uniform(value))),
                    None => Some((*coord, brick)),
                }
            })
            .flatten()
            .collect();

        for (coord, brick) in built {
            warped.insert_brick(coord, brick);
        }
        warped.mark_everything_dirty();
        warped
    }

    /// The one value every voxel of an inclusive source voxel box holds, or
    /// `None` when the region carries detail.
    ///
    /// Absent bricks count as [`OUTSIDE`], which is what they read as. The
    /// field's sibling of `MaskField::uniform_over`, and what lets a solid
    /// interior survive a shift without being allocated.
    fn uniform_over(&self, low: IVec3, high: IVec3) -> Option<f32> {
        let b_low = BrickCoord::containing(low).0;
        let b_high = BrickCoord::containing(high).0;
        let mut found: Option<f32> = None;
        for bz in b_low.z..=b_high.z {
            for by in b_low.y..=b_high.y {
                for bx in b_low.x..=b_high.x {
                    let value = self.brick_fill(BrickCoord::new(bx, by, bz))?;
                    match found {
                        None => found = Some(value),
                        Some(held) if held == value => {}
                        Some(_) => return None,
                    }
                }
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orientation::{AxisRotation, Facing};
    use crate::redistance::GRADIENT_TOLERANCE;
    use crate::similarity::Bake;
    use glam::Quat;

    const RADIUS: f32 = 20.0;

    fn sphere(voxel_size: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::ZERO, RADIUS);
        volume
    }

    /// Every stored voxel, in a deterministic order, so two volumes can be
    /// compared value by value rather than through a sampled surface.
    ///
    /// The same helper `rotate.rs` uses, and deliberately the same shape: a
    /// claim of bit identity that is only checked at the surface is not a claim
    /// of bit identity.
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

    /// Where the surface sits along +X, found by walking outward.
    fn measured_radius(volume: &Volume, from: Vec3) -> f32 {
        let step = volume.voxel_size() * 0.1;
        let mut last = 0.0;
        for index in 0..20_000 {
            let t = index as f32 * step;
            if volume.sample_world(from + Vec3::new(t, 0.0, 0.0)) >= 0.0 {
                return last;
            }
            last = t;
        }
        last
    }

    /// The claim the exact route rests on. Not "close to": the same bits,
    /// because no value was ever recomputed.
    #[test]
    fn a_whole_voxel_shift_and_its_inverse_are_bit_identical() {
        let source = sphere(0.5);
        for offset in [
            IVec3::new(1, 0, 0),
            IVec3::new(-7, 13, 5),
            IVec3::new(31, -31, 1),
            // Brick aligned, which takes the `duplicated` route instead.
            IVec3::new(32, 64, -32),
        ] {
            let there = source.shifted(offset);
            let back = there.shifted(-offset);
            assert_eq!(
                field(&back),
                field(&source),
                "shifting by {offset:?} and back moved values"
            );
        }
    }

    #[test]
    fn a_brick_aligned_shift_agrees_with_duplicated() {
        let source = sphere(0.5);
        let offset_bricks = IVec3::new(2, -1, 3);
        let shifted = source.shifted(offset_bricks * BRICK_DIM as i32);
        let duplicated = source.duplicated(offset_bricks);
        assert_eq!(field(&shifted), field(&duplicated));
    }

    #[test]
    fn a_sub_brick_shift_moves_the_model_exactly_that_far() {
        let volume = sphere(0.5);
        // Seven voxels along X at a 0.5 mm voxel is 3.5 mm.
        let shifted = volume.shifted(IVec3::new(7, 0, 0));
        let before = measured_radius(&volume, Vec3::ZERO);
        let after = measured_radius(&shifted, Vec3::new(3.5, 0.0, 0.0));
        assert!((after - before).abs() < 1.0e-4, "{before} against {after}");
        // And the old place is empty now.
        assert!(shifted.sample_world(Vec3::new(-RADIUS + 1.0, 0.0, 0.0)) >= 0.0);
    }

    /// A shift must not allocate a solid interior. Without the `uniform_over`
    /// shortcut every destination brick would be gathered densely, and only
    /// `is_collapsible` would notice afterwards -- correct, and 32768 reads per
    /// brick to learn it.
    #[test]
    fn the_interior_stays_tiles_rather_than_becoming_dense() {
        let source = sphere(0.25);
        assert!(source.stats().uniform_bricks > 0, "the fixture has no interior to preserve");

        let shifted = source.shifted(IVec3::new(5, -3, 9));
        let stats = shifted.stats();
        assert!(stats.uniform_bricks > 0, "the interior should have stayed tiles");
        match shifted.brick(BrickCoord::containing(IVec3::ZERO)) {
            Some(Brick::Uniform(value)) => assert!(*value <= INSIDE),
            Some(Brick::Dense(_)) => panic!("the centre of a solid ball was filled in with voxels"),
            None => panic!("the centre of a solid ball should not be empty space"),
        }
    }

    #[test]
    fn a_shift_carries_the_mask_to_the_same_world_voxel() {
        use crate::{PROTECTED, UNMASKED};

        let mut volume = sphere(0.5);
        let on_the_body = IVec3::new(3, 40, 7);
        // Out in empty space, where no field brick exists to carry a mask that
        // merely rode along with one.
        let in_empty_space = IVec3::new(400, 400, 400);
        volume.mask_mut().write(on_the_body, PROTECTED);
        volume.mask_mut().write(in_empty_space, 200);

        let offset = IVec3::new(5, -3, 9);
        let shifted = volume.shifted(offset);

        assert_eq!(shifted.mask().at(on_the_body + offset), PROTECTED);
        assert_eq!(shifted.mask().at(in_empty_space + offset), 200);
        assert_eq!(shifted.mask().at(on_the_body), UNMASKED, "the mask did not move with the body");
    }

    /// Mask All is a polarity bit over an EMPTY brick map, so for that state the
    /// bit is the entire mask and a move that copies no brick carries nothing
    /// else.
    #[test]
    fn a_shift_carries_mask_all_even_though_it_moves_no_brick() {
        use crate::PROTECTED;

        let mut volume = sphere(0.5);
        volume.mask_mut().set_inverted(true);
        assert_eq!(volume.mask().map_bytes(), 0, "Mask All must store no bricks at all");

        let shifted = volume.shifted(IVec3::new(7, 1, -2));
        assert!(shifted.mask().inverted(), "the move dropped the mask's polarity");
        assert_eq!(shifted.mask().at(IVec3::new(-317, 44, 900)), PROTECTED);
    }

    #[test]
    fn every_brick_of_a_moved_model_is_dirty() {
        let mut shifted = sphere(0.5).shifted(IVec3::new(3, 3, 3));
        let stored: Vec<BrickCoord> = shifted.brick_coords().collect();
        assert!(!stored.is_empty(), "the fixture is empty");
        let mut dirty = Vec::new();
        shifted.take_dirty(&mut dirty);
        let dirty: std::collections::HashSet<BrickCoord> = dirty.into_iter().collect();
        for coord in stored {
            assert!(dirty.contains(&coord), "{coord:?} carries geometry and was not marked dirty");
        }
    }

    #[test]
    fn shifting_an_empty_volume_gives_an_empty_volume() {
        let empty = Volume::new(0.5);
        let shifted = empty.shifted(IVec3::new(3, -4, 5));
        assert_eq!(shifted.brick_count(), 0);
        assert_eq!(shifted.voxel_size(), 0.5);
    }

    /// The lossy route has to agree with the exact one wherever both apply, or
    /// the routing decision changes the answer rather than only its cost.
    #[test]
    fn a_quarter_turn_through_warped_agrees_with_rotated() {
        let mut volume = Volume::new(0.5);
        // Off axis, so a rotation that did nothing would be caught.
        volume.seed_sphere(Vec3::new(0.0, 30.0, 0.0), 8.0);

        let turns = AxisRotation::taking(Facing::Up, Facing::Front);
        let exact = volume.rotated(turns);
        let warped = volume.warped(Similarity::about(
            Vec3::ZERO,
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            Vec3::ONE,
            Vec3::ZERO,
        ));

        for point in [
            Vec3::new(0.0, 0.0, 30.0),
            Vec3::new(0.0, 0.0, 22.0),
            Vec3::new(0.0, 0.0, 38.0),
            Vec3::new(4.0, 0.0, 30.0),
        ] {
            let a = exact.sample_world(point);
            let b = warped.sample_world(point);
            assert!(
                (a - b).abs() < 0.3,
                "at {point:?} the exact turn says {a} and the warp says {b}"
            );
        }
    }

    /// **The test that catches a destination box derived from the SOURCE
    /// bounds.** A forty-five degree turn pushes a box's corners out by root
    /// two; a footprint that did not grow would clip them off and the model
    /// would come back with its corners shaved, which reads as a mesher bug.
    #[test]
    fn a_forty_five_degree_turn_of_a_box_keeps_its_corners() {
        let mut volume = Volume::new(0.5);
        // A bar, so the corners are far from the turn's centre and would be the
        // first thing a tight footprint lost.
        volume.seed_sphere(Vec3::new(-25.0, 0.0, 0.0), 6.0);
        volume.seed_sphere(Vec3::new(25.0, 0.0, 0.0), 6.0);

        let turn = Similarity::about(
            Vec3::ZERO,
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            Vec3::ONE,
            Vec3::ZERO,
        );
        let turned = volume.warped(turn);

        for end in [Vec3::new(-25.0, 0.0, 0.0), Vec3::new(25.0, 0.0, 0.0)] {
            let moved = turn.transform_point(end);
            assert!(
                turned.sample_world(moved) < 0.0,
                "the ball that was at {end:?} did not arrive at {moved:?}"
            );
        }
    }

    #[test]
    fn a_uniform_scale_makes_the_model_that_much_bigger() {
        let volume = sphere(0.5);
        for scale in [0.5_f32, 2.0] {
            let scaled = volume.warped(Similarity::about(
                Vec3::ZERO,
                Quat::IDENTITY,
                Vec3::splat(scale),
                Vec3::ZERO,
            ));
            let measured = measured_radius(&scaled, Vec3::ZERO);
            assert!(
                (measured - RADIUS * scale).abs() < 1.0,
                "scaling by {scale} gave a radius of {measured}, expected {}",
                RADIUS * scale
            );
            // The lattice is untouched, which is the invariant that lets a
            // scaled body sit beside its unscaled siblings.
            assert_eq!(scaled.voxel_size(), volume.voxel_size());
        }
    }

    /// **Reading through a gathered region must give the same field as reading
    /// through the brick map.**
    ///
    /// `Volume::warped` used to call `sample_world` per destination voxel, and
    /// a trilinear sample is eight reads, so a dense brick paid 262,144 hash
    /// lookups. [`Volume::resampled`] has always gathered the source region
    /// once and read a flat array instead; the two loops are siblings and only
    /// this one was missing it.
    ///
    /// A speed change that quietly moved a voxel would be the worst kind, so
    /// this drives both paths over a squash, a shrink and a grow and compares
    /// the storage bit for bit -- including which bricks exist at all, since a
    /// difference of one ULP either side of `is_collapsible`'s exact test would
    /// change a brick's representation rather than its values.
    #[test]
    fn a_gathered_warp_is_bit_identical_to_a_sampled_one() {
        let volume = sphere(0.5);
        for scale in [Vec3::new(1.0, 0.6, 1.0), Vec3::splat(0.5), Vec3::splat(1.7)] {
            let by = Similarity::about(Vec3::ZERO, Quat::IDENTITY, scale, Vec3::new(3.0, 0.0, 1.0));
            let gathered = volume.warped_inner(by, true);
            let sampled = volume.warped_inner(by, false);
            assert!(
                gathered.stats().dense_bricks > 0,
                "no dense bricks at scale {scale:?}, so nothing was actually sampled"
            );
            crate::testing::assert_same_field(
                &gathered,
                &sampled,
                &format!("gathering the source changed the field at scale {scale:?}"),
            );
        }
    }

    /// **A shrunk body must measure the size it was shrunk to.**
    ///
    /// `surface_bounds` rejects the far field with `value.abs() >= NARROW_BAND`,
    /// which is only a valid test while the saturation constant IS
    /// `NARROW_BAND`. A shrink used to scale the plateau along with the measured
    /// distances, so nothing was saturated any more, the predicate stopped
    /// rejecting anything, and the whole volume read as surface -- 47.75 mm
    /// against an exact 36.000 on a 0.25 mm lattice, 32.6% out.
    ///
    /// That is not a cosmetic number. `set_working_size` divides by this span,
    /// under a docstring calling the operation free and lossless, so a body
    /// scaled down and then set to a size would have exported at the wrong
    /// physical size -- a wrong part out of the slicer.
    #[test]
    fn the_measured_size_of_a_shrunk_body_is_the_size_it_was_shrunk_to() {
        for voxel_size in [0.25_f32, 0.125] {
            for scale in [0.6_f32, 0.3] {
                let volume = sphere(voxel_size);
                let shrunk = volume.warped(Similarity::about(
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(scale),
                    Vec3::ZERO,
                ));
                let (lo, hi) = shrunk.surface_bounds().expect("the shrunk body has a surface");
                let measured = (hi - lo).max_element();
                let expected = RADIUS * 2.0 * scale;
                assert!(
                    (measured - expected).abs() <= voxel_size * 4.0,
                    "a {RADIUS} mm-radius ball scaled by {scale} at a {voxel_size} mm voxel \
                     measured {measured} mm across, expected {expected} mm"
                );
            }
        }
    }

    /// **A shrink must leave a band that still reaches the band edge.**
    ///
    /// `d' = s * d(T_inv(p))` is exact for a similarity, so a uniformly scaled
    /// body is a true distance field and its drift must measure as zero. It did
    /// not, and the reason was the saturation plateau rather than the geometry:
    /// the multiply scaled the clamp along with the measured distances, so at
    /// `s = 0.5` the whole volume landed inside `+/-1.5` and **not one voxel was
    /// saturated** -- measured, 971,210 became zero. `measurable` skips a voxel
    /// only when its stencil touches saturation, so the flat far field was then
    /// measured, and a flat region has no gradient, so `gradient_drift` came
    /// back with 1.0.
    ///
    /// It was permanent, too. `rebake_gizmo` used to skip `redistance` when the
    /// scale was uniform, on the correct-sounding reasoning that a similarity
    /// leaves a true distance field, so nothing ever put the band back and the
    /// body reported a drift of 1.0 for the rest of its life. Since the brush
    /// residual is meant to be gated on measured drift, wiring it up would have
    /// fired it on every body that had ever been scaled and never converged.
    ///
    /// Asserted on the pair, not on `warped` alone, because that is the pair
    /// production runs: `warped` deliberately leaves a cliff at the band edge
    /// and `redistance` is what resolves it. Testing `warped` by itself would
    /// pin the intermediate state and forbid the fix.
    #[test]
    fn a_uniform_shrink_leaves_a_full_depth_band() {
        let volume = sphere(0.5);
        let before = volume.gradient_drift();
        assert!(before <= GRADIENT_TOLERANCE, "the fixture already drifts by {before}");

        let mut scaled = volume.warped(Similarity::about(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::splat(0.5),
            Vec3::ZERO,
        ));
        scaled.redistance();

        let deepest = scaled
            .brick_coords()
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|coord| match scaled.brick(coord) {
                Some(Brick::Dense(data)) => data.iter().copied().fold(None::<f32>, |worst, v| {
                    Some(worst.map_or(v.abs(), |w: f32| w.max(v.abs())))
                }),
                _ => None,
            })
            .fold(0.0f32, f32::max);
        assert!(
            deepest >= NARROW_BAND,
            "the deepest value in the band is {deepest}, so the shrink carried the saturation \
             plateau down with it and the band no longer reaches its own edge"
        );

        let drift = scaled.gradient_drift();
        assert!(
            drift <= GRADIENT_TOLERANCE,
            "a uniform scale is an exact distance field, so the drift should be near zero \
             and it measured {drift}"
        );
    }

    /// A distance field that is not a distance field breaks the sphere trace,
    /// the mesher and the curvature the brushes read. `d' = s * d(T_inv(p))` is
    /// exact for a similarity, so the gradient must come back unit length --
    /// which is the property `Bake::Resample` is allowed to be lossy about the
    /// DETAIL of and not about the METRIC of.
    #[test]
    fn a_scaled_field_is_still_a_distance_field() {
        let volume = sphere(0.5);
        let scaled = volume.warped(Similarity::about(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::splat(1.7),
            Vec3::ZERO,
        ));
        let expected = RADIUS * 1.7;

        // Along the surface, a little outside it, where the field is not
        // clamped and the values are real.
        for direction in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 1.0, 1.0).normalize()] {
            for offset in [-1.0_f32, 0.0, 1.0] {
                let at = direction * (expected + offset);
                let h = scaled.voxel_size();
                let gradient = Vec3::new(
                    scaled.sample_world(at + Vec3::X * h) - scaled.sample_world(at - Vec3::X * h),
                    scaled.sample_world(at + Vec3::Y * h) - scaled.sample_world(at - Vec3::Y * h),
                    scaled.sample_world(at + Vec3::Z * h) - scaled.sample_world(at - Vec3::Z * h),
                ) / (2.0 * h);
                // Values are in voxels, so a unit gradient measures 1 per voxel.
                let length = gradient.length() * h;
                assert!(
                    (0.85..=1.15).contains(&length),
                    "|grad| was {length} at {at:?}, so the field is no longer metric"
                );
            }
        }
    }

    #[test]
    fn warping_by_the_identity_leaves_the_surface_where_it_was() {
        let volume = sphere(0.5);
        let again = volume.warped(Similarity::IDENTITY);
        assert!(
            (measured_radius(&again, Vec3::ZERO) - measured_radius(&volume, Vec3::ZERO)).abs()
                < 0.2
        );
        // And the routing says it should never have been called at all.
        assert_eq!(Similarity::IDENTITY.route(volume.voxel_size()), Bake::Identity);
    }

    #[test]
    fn a_warped_model_is_still_printable() {
        let volume = sphere(0.5);
        let placement = Similarity::about(
            Vec3::ZERO,
            Quat::from_euler(glam::EulerRot::YXZ, 0.4, 0.3, 0.0),
            Vec3::splat(1.3),
            Vec3::new(3.0, -2.0, 1.0),
        );
        let (_, report) = volume.warped(placement).export_mesh();
        assert!(report.is_printable(), "{}", report.summary());
    }

    #[test]
    fn a_warp_carries_the_mask_to_where_the_body_went() {
        use crate::PROTECTED;

        let mut volume = sphere(0.5);
        let cell = IVec3::new(0, 40, 0);
        volume.mask_mut().write(cell, PROTECTED);

        let placement = Similarity::moving(Vec3::new(6.0, 0.0, 0.0));
        let warped = volume.warped(placement);

        let moved = placement.transform_point(cell.as_vec3() * 0.5) / 0.5;
        assert_eq!(
            warped.mask().at(moved.round().as_ivec3()),
            PROTECTED,
            "the protection did not travel with the body"
        );
    }

    #[test]
    fn warping_an_empty_volume_gives_an_empty_volume() {
        let empty = Volume::new(0.5);
        let warped = empty.warped(Similarity::moving(Vec3::splat(3.0)));
        assert_eq!(warped.brick_count(), 0);
        assert_eq!(warped.voxel_size(), 0.5);
    }

    #[test]
    fn a_nonsense_scale_gives_an_empty_volume_rather_than_a_field_of_infinities() {
        let volume = sphere(0.5);
        for scale in [0.0_f32, -1.0, f32::NAN] {
            let warped = volume.warped(Similarity {
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(scale),
                translation: Vec3::ZERO,
            });
            assert_eq!(warped.brick_count(), 0, "scale {scale}");
        }
    }
}
