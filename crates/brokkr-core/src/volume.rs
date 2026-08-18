// SPDX-License-Identifier: AGPL-3.0-or-later

//! The sparse brick volume: storage, sampling, editing and dirty tracking.

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::apron::ApronBuffer;
use crate::brick::{
    BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, apron_index,
    brick_index,
};
use crate::mesh::{BrickMesh, MeshScratch, mesh_apron};
use crate::region::FieldRegion;
use crate::undo::StrokeEdit;

/// Counters for the debug overlay and the memory budget.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VolumeStats {
    /// Bricks holding a full voxel array.
    pub dense_bricks: usize,
    /// Bricks collapsed to a single tile value, which cost no voxel storage.
    pub uniform_bricks: usize,
    /// Bytes of voxel data plus the map that indexes it.
    pub resident_bytes: usize,
}

/// What is known about one brick before an edit decides whether to touch it.
///
/// The useful part is [`BrickPreview::uniform`]. Most of a large brush's box is
/// deep interior or far exterior, saturated at a single value across whole
/// bricks, and a brush that resamples or averages cannot change a region that
/// already reads as one value everywhere. Answering that before the brick is
/// made dense is what keeps a 20 mm brush from allocating and rewriting thirty
/// megabytes it will not change.
#[derive(Debug, Clone, Copy)]
pub struct BrickPreview {
    pub coord: BrickCoord,
    /// Inclusive world voxel range of this brick that the edit's box covers.
    pub lo: IVec3,
    pub hi: IVec3,
    /// `Some(v)` when every voxel of this brick holds `v`, which covers both a
    /// uniform tile and an absent brick -- absent reads as [`OUTSIDE`]. `None`
    /// when the brick carries detail.
    ///
    /// This says nothing about its neighbours. A brush that answers
    /// [`BrickVerdict::OnlyNearDifferentNeighbours`] is telling the volume it
    /// leaves `v` alone, and the volume works out how much of the brick is far
    /// enough from anything else for that to hold.
    pub uniform: Option<f32>,
}

/// What an edit wants done with one brick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickVerdict {
    /// Nothing in it can change, so do not read it, make it dense, record it
    /// for undo or write it.
    Skip,
    /// Resolve every voxel of it that the edit's box covers.
    Whole,
    /// It holds one value that this edit leaves alone, so only the part of it
    /// within reach of a neighbour holding something different can change.
    ///
    /// Only legal when [`BrickPreview::uniform`] is `Some`. This is where the
    /// bulk of a large brush's saving comes from, and it is much stronger than
    /// asking for the whole neighbourhood to be uniform: a brick is 32 voxels
    /// across and a stamp reads two, so a tile sitting against the surface
    /// still has 96 percent of itself out of reach.
    OnlyNearDifferentNeighbours,
    /// It holds one value everywhere, and the edit has worked out for itself
    /// which part of it can change. Only that inclusive voxel range is
    /// resolved, and the rest is left exactly as it was.
    ///
    /// [`BrickVerdict::OnlyNearDifferentNeighbours`] is this with the volume
    /// doing the working out, which it can only do for an edit whose reads come
    /// from the volume. A warping gesture reads a copy locked before the
    /// gesture started and displaces by tens of voxels along the drag, so
    /// neither the grid nor the shape of the reach is the volume's to reason
    /// about, and it answers for itself. See [`crate::brush::MoveStroke`].
    OnlyWithin(IVec3, IVec3),
}

/// One brick an edit has decided to visit, and where inside it.
#[derive(Debug, Clone, Copy)]
struct Visit {
    coord: BrickCoord,
    /// Inclusive voxel range of this brick that the edit's box covers.
    lo: IVec3,
    hi: IVec3,
}

/// One brick lifted out of the map for the duration of an edit.
///
/// Taking bricks out is what lets the writing phase hold several mutable bricks
/// at once, and it costs a pointer move each rather than a scan of the volume.
struct Taken {
    coord: BrickCoord,
    brick: Brick,
    /// Inclusive voxel range of this brick that the edit's box covers.
    lo: IVec3,
    hi: IVec3,
    existed: bool,
    was_uniform: bool,
    recorded_now: bool,
    changed: bool,
}

/// Working space for [`Volume::edit_voxels_where`], kept between calls.
///
/// A stroke lays down thousands of stamps and the budget forbids allocating in
/// that path, so all three buffers live on the volume and are reused.
#[derive(Default)]
struct EditScratch {
    /// One entry per brick of the edit's box plus a ring around it: `Some(v)`
    /// when every voxel of that brick holds `v`. Laid out with X fastest, from
    /// the grown box's minimum brick.
    fills: Vec<Option<f32>>,
    /// Bricks the classifier kept.
    visits: Vec<Visit>,
    /// Bricks lifted out of the map by the parallel path.
    taken: Vec<Taken>,
}

/// Apply an edit to one brick's voxels, returning whether anything changed.
///
/// A generic function rather than a closure, so that the brush's per voxel
/// maths, which for the resampling brushes is an eight tap trilinear read,
/// inlines into the loop.
#[inline]
fn write_voxels<F>(
    data: &mut [f32; BRICK_VOXELS],
    origin: IVec3,
    lo: IVec3,
    hi: IVec3,
    voxel_size: f32,
    edit: &F,
) -> bool
where
    F: Fn(IVec3, Vec3, f32) -> f32 + Sync,
{
    let mut changed = false;
    for wz in lo.z..=hi.z {
        for wy in lo.y..=hi.y {
            for wx in lo.x..=hi.x {
                let voxel = IVec3::new(wx, wy, wz);
                let index = brick_index(
                    (wx - origin.x) as usize,
                    (wy - origin.y) as usize,
                    (wz - origin.z) as usize,
                );
                let position = voxel.as_vec3() * voxel_size;
                let old = data[index];
                let new = edit(voxel, position, old).clamp(INSIDE, OUTSIDE);
                if new != old {
                    data[index] = new;
                    changed = true;
                }
            }
        }
    }
    changed
}

/// The part of a uniform brick that an edit could still change, given that the
/// brick holds `value` everywhere and can read `reach` voxels past its own
/// faces.
///
/// Returns the inclusive voxel range within `reach` of a neighbouring brick
/// that holds something other than `value`, or `None` when all 26 of them hold
/// `value` too and the brick is therefore untouchable.
///
/// A bounding box of the union rather than the union itself, which is exact
/// whenever the differing neighbours are all on one side -- the usual case,
/// because what makes a neighbour differ is the surface passing through it.
fn reachable_from_elsewhere<F>(
    brick: IVec3,
    value: f32,
    reach: i32,
    at: &F,
    fills: &[Option<f32>],
) -> Option<(IVec3, IVec3)>
where
    F: Fn(IVec3) -> usize,
{
    let dim = BRICK_DIM as i32;
    let origin = brick * dim;
    let last = origin + IVec3::splat(dim - 1);

    let mut lo = IVec3::splat(i32::MAX);
    let mut hi = IVec3::splat(i32::MIN);

    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let offset = IVec3::new(dx, dy, dz);
                if offset == IVec3::ZERO {
                    continue;
                }
                if fills[at(brick + offset)] == Some(value) {
                    continue;
                }
                // The slab of this brick lying within `reach` of that
                // neighbour, per axis: the near face, the far face, or all of
                // it when the neighbour is not offset along that axis at all.
                let mut near_lo = origin;
                let mut near_hi = last;
                for axis in 0..3 {
                    match offset[axis] {
                        -1 => near_hi[axis] = origin[axis] + reach - 1,
                        1 => near_lo[axis] = last[axis] - reach + 1,
                        _ => {}
                    }
                }
                if near_lo.cmpgt(near_hi).any() {
                    continue;
                }
                lo = lo.min(near_lo);
                hi = hi.max(near_hi);
            }
        }
    }

    (lo.cmple(hi).all()).then_some((lo, hi))
}

/// A sparse grid of bricks at a fixed world space voxel size.
///
/// Only bricks that carry detail are stored. Absent bricks read as [`OUTSIDE`],
/// and solid interiors are stored as [`Brick::Uniform`] tiles, so empty space
/// and solid space both cost nothing.
pub struct Volume {
    voxel_size: f32,
    bricks: FxHashMap<BrickCoord, Brick>,
    dirty: FxHashSet<BrickCoord>,
    /// Prior contents of every brick touched since the stroke began, or `None`
    /// when no stroke is in progress. The inner `None` means the brick did not
    /// exist, which undo has to restore just as faithfully as any content.
    recorder: Option<FxHashMap<BrickCoord, Option<Brick>>>,
    scratch: EditScratch,
}

impl Volume {
    /// An empty volume with the given world space voxel size.
    ///
    /// Resolution is uniform and independent of object size. Making detail
    /// finer means resampling the whole volume, which is a deliberate explicit
    /// operation, not something a brush does.
    pub fn new(voxel_size: f32) -> Self {
        assert!(voxel_size > 0.0, "voxel size must be positive");
        Self {
            voxel_size,
            bricks: FxHashMap::default(),
            dirty: FxHashSet::default(),
            recorder: None,
            scratch: EditScratch::default(),
        }
    }

    #[inline]
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    // ---------------------------------------------------------------- sampling

    /// Distance value at a world voxel coordinate.
    #[inline]
    pub fn sample_voxel(&self, voxel: IVec3) -> f32 {
        let coord = BrickCoord::containing(voxel);
        match self.bricks.get(&coord) {
            None => OUTSIDE,
            Some(brick) => {
                let local = voxel - coord.origin();
                brick.get(local.x as usize, local.y as usize, local.z as usize)
            }
        }
    }

    /// Trilinearly interpolated distance at a world space point.
    pub fn sample_world(&self, point: Vec3) -> f32 {
        let grid = point / self.voxel_size;
        let base = grid.floor();
        let frac = grid - base;
        let b = base.as_ivec3();

        let c = |dx: i32, dy: i32, dz: i32| self.sample_voxel(b + IVec3::new(dx, dy, dz));

        let x00 = lerp(c(0, 0, 0), c(1, 0, 0), frac.x);
        let x10 = lerp(c(0, 1, 0), c(1, 1, 0), frac.x);
        let x01 = lerp(c(0, 0, 1), c(1, 0, 1), frac.x);
        let x11 = lerp(c(0, 1, 1), c(1, 1, 1), frac.x);

        let y0 = lerp(x00, x10, frac.y);
        let y1 = lerp(x01, x11, frac.y);
        lerp(y0, y1, frac.z)
    }

    /// Surface normal at a world space point, from central differences.
    ///
    /// Returns `Vec3::Y` where the field is flat, so callers never receive a
    /// zero vector.
    pub fn gradient_world(&self, point: Vec3) -> Vec3 {
        let h = self.voxel_size;
        let gradient = Vec3::new(
            self.sample_world(point + Vec3::X * h) - self.sample_world(point - Vec3::X * h),
            self.sample_world(point + Vec3::Y * h) - self.sample_world(point - Vec3::Y * h),
            self.sample_world(point + Vec3::Z * h) - self.sample_world(point - Vec3::Z * h),
        );
        gradient.try_normalize().unwrap_or(Vec3::Y)
    }

    // ------------------------------------------------------------------- apron

    /// Fill `apron` with this brick's voxels plus the one voxel halo taken from
    /// its 26 neighbours.
    ///
    /// The buffer covers world voxels `origin - 1` through `origin + BRICK_DIM`
    /// inclusive on each axis. Every sample is written exactly once: the 27
    /// source bricks tile the buffer exactly, and absent ones contribute
    /// [`OUTSIDE`], so there is no prefill pass.
    ///
    /// The brick itself need not exist. An absent brick next to an edited one
    /// still has to be meshed, because its apron sees the edit and the tiling
    /// scheme assigns those boundary quads to it.
    pub fn gather_apron(&self, coord: BrickCoord, apron: &mut ApronBuffer) {
        let dim = BRICK_DIM as i32;
        let apron_lo = coord.origin() - IVec3::ONE;
        let apron_hi = coord.origin() + IVec3::splat(dim);
        let samples = apron.samples_mut();

        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let neighbour = BrickCoord(coord.0 + IVec3::new(dx, dy, dz));
                    let n_lo = neighbour.origin();
                    let n_hi = neighbour.max_voxel();

                    // World voxel range this neighbour contributes.
                    let lo = apron_lo.max(n_lo);
                    let hi = apron_hi.min(n_hi);
                    if lo.cmpgt(hi).any() {
                        continue;
                    }

                    let run = (hi.x - lo.x + 1) as usize;
                    let brick = self.bricks.get(&neighbour);

                    for wz in lo.z..=hi.z {
                        for wy in lo.y..=hi.y {
                            let a_start = apron_index(
                                (lo.x - apron_lo.x) as usize,
                                (wy - apron_lo.y) as usize,
                                (wz - apron_lo.z) as usize,
                            );
                            let dst = &mut samples[a_start..a_start + run];

                            match brick {
                                None => dst.fill(OUTSIDE),
                                Some(Brick::Uniform(value)) => dst.fill(*value),
                                Some(Brick::Dense(data)) => {
                                    let b_start = brick_index(
                                        (lo.x - n_lo.x) as usize,
                                        (wy - n_lo.y) as usize,
                                        (wz - n_lo.z) as usize,
                                    );
                                    dst.copy_from_slice(&data[b_start..b_start + run]);
                                }
                            }
                        }
                    }
                }
            }
        }

        apron.set_coord(coord);
    }

    // ----------------------------------------------------------------- meshing

    /// Mesh one brick into `out`, in world space.
    ///
    /// This is the only public path from voxels to triangles, and it gathers
    /// the apron itself. That is what makes the apron rule structural rather
    /// than a convention someone has to remember.
    pub fn mesh_brick(&self, coord: BrickCoord, scratch: &mut MeshScratch, out: &mut BrickMesh) {
        self.gather_apron(coord, &mut scratch.apron);
        mesh_apron(&scratch.apron, coord, self.voxel_size, &mut scratch.surface_nets, out);
    }

    /// Mesh many bricks at once, across every core.
    ///
    /// `out` must already hold one mesh per coordinate; they are reused rather
    /// than allocated, so a stroke settles into steady state without touching
    /// the allocator. Each worker keeps its own scratch, which is why this does
    /// not take one.
    ///
    /// Meshing is read only on the volume and every brick is independent, so
    /// this is close to linear in the core count. Worth it: at the M2 target
    /// size a full mesh takes over two seconds on one core, and a stroke's
    /// remesh reaches 70 percent of its budget.
    pub fn mesh_bricks(&self, coords: &[BrickCoord], out: &mut [BrickMesh]) {
        assert_eq!(coords.len(), out.len(), "one output mesh per brick");

        // Below this the thread hand off costs more than the meshing saves.
        const PARALLEL_THRESHOLD: usize = 4;

        if coords.len() < PARALLEL_THRESHOLD {
            let mut scratch = MeshScratch::new();
            for (coord, mesh) in coords.iter().zip(out.iter_mut()) {
                self.mesh_brick(*coord, &mut scratch, mesh);
            }
            return;
        }

        out.par_iter_mut().zip(coords.par_iter()).for_each_init(
            MeshScratch::new,
            |scratch, (mesh, coord)| {
                self.mesh_brick(*coord, scratch, mesh);
            },
        );
    }

    // ---------------------------------------------------------- dirty tracking

    /// Mark every brick overlapping an inclusive world voxel range as needing a
    /// remesh, expanded by one voxel because a brick's apron reads one voxel
    /// into each neighbour.
    pub fn mark_dirty_voxel_range(&mut self, min_voxel: IVec3, max_voxel: IVec3) {
        let lo = BrickCoord::containing(min_voxel - IVec3::ONE).0;
        let hi = BrickCoord::containing(max_voxel + IVec3::ONE).0;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    self.dirty.insert(BrickCoord::new(x, y, z));
                }
            }
        }
    }

    #[inline]
    pub fn mark_dirty(&mut self, coord: BrickCoord) {
        self.dirty.insert(coord);
    }

    /// Mark every brick that can carry geometry: those with data, plus their
    /// neighbours, because a brick with no voxels of its own still owns the
    /// quads on its low faces.
    ///
    /// This is a load time operation, used after seeding or opening a model.
    /// It is proportional to the whole volume and so must never run per frame.
    pub fn mark_everything_dirty(&mut self) {
        let stored: Vec<BrickCoord> = self.bricks.keys().copied().collect();
        for coord in stored {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        self.dirty.insert(BrickCoord(coord.0 + IVec3::new(dx, dy, dz)));
                    }
                }
            }
        }
    }

    #[inline]
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Move the dirty set into `out`, keeping both allocations.
    pub fn take_dirty(&mut self, out: &mut Vec<BrickCoord>) {
        out.clear();
        out.extend(self.dirty.drain());
    }

    // ------------------------------------------------------------------ editing

    /// Inclusive voxel bounds covering a world space box.
    pub fn voxel_bounds(&self, min_world: Vec3, max_world: Vec3) -> (IVec3, IVec3) {
        (
            (min_world / self.voxel_size).floor().as_ivec3(),
            (max_world / self.voxel_size).ceil().as_ivec3(),
        )
    }

    /// World space position of a voxel's centre.
    #[inline]
    pub fn voxel_position(&self, voxel: IVec3) -> Vec3 {
        voxel.as_vec3() * self.voxel_size
    }

    /// Copy the field over an inclusive voxel box into `region`, grown by one
    /// voxel on every side so gradients and neighbour averages are available
    /// throughout the box.
    ///
    /// Brushes read from this copy rather than from the volume so that every
    /// voxel sees the same starting field, whatever order the writes happen in.
    ///
    /// Copied brick by brick in contiguous runs along X rather than by sampling
    /// each voxel. A stamp of radius 12 voxels covers about 20 000 samples, and
    /// looking every one of them up through the brick map put the fast drag
    /// case within one percent of the edit budget on its own.
    ///
    /// A plane of the box at a time, across every core once there is enough of
    /// it to be worth the hand off. The planes are disjoint and the volume is
    /// only read, so there is nothing to synchronise. It is worth doing: a
    /// 20 mm brush at a quarter millimetre voxel copies sixteen megabytes, and
    /// one core moves that at about the speed one core can write memory, which
    /// was a third of the whole stamp.
    ///
    /// Every element of the box is written exactly once -- the bricks tile it
    /// -- which is what lets [`FieldRegion::resize`] hand back a dirty buffer
    /// rather than zeroing it first.
    pub fn snapshot(&self, lo: IVec3, hi: IVec3, region: &mut FieldRegion) {
        /// Samples in the box below which the copy stays on one core.
        const PARALLEL_SNAPSHOT_THRESHOLD: usize = 8 * BRICK_VOXELS;

        let lo = lo - IVec3::ONE;
        let hi = hi + IVec3::ONE;
        let size = hi - lo + IVec3::ONE;
        let values = region.resize(lo, hi);

        let b_min = BrickCoord::containing(lo).0;
        let b_max = BrickCoord::containing(hi).0;
        let plane = (size.x * size.y) as usize;

        let fill_plane = |wz: i32, slab: &mut [f32]| {
            let bz = BrickCoord::containing(IVec3::new(lo.x, lo.y, wz)).0.z;
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let coord = BrickCoord::new(bx, by, bz);
                    let brick_lo = coord.origin();
                    let brick_hi = coord.max_voxel();

                    // The part of this brick's column that falls inside the box.
                    let from_x = lo.x.max(brick_lo.x);
                    let to_x = hi.x.min(brick_hi.x);
                    let from_y = lo.y.max(brick_lo.y);
                    let to_y = hi.y.min(brick_hi.y);
                    if from_x > to_x || from_y > to_y {
                        continue;
                    }

                    let run = (to_x - from_x + 1) as usize;
                    let brick = self.bricks.get(&coord);

                    for wy in from_y..=to_y {
                        let start = ((from_x - lo.x) + (wy - lo.y) * size.x) as usize;
                        let destination = &mut slab[start..start + run];

                        match brick {
                            None => destination.fill(OUTSIDE),
                            Some(Brick::Uniform(value)) => destination.fill(*value),
                            Some(Brick::Dense(data)) => {
                                let source = brick_index(
                                    (from_x - brick_lo.x) as usize,
                                    (wy - brick_lo.y) as usize,
                                    (wz - brick_lo.z) as usize,
                                );
                                destination.copy_from_slice(&data[source..source + run]);
                            }
                        }
                    }
                }
            }
        };

        if values.len() >= PARALLEL_SNAPSHOT_THRESHOLD {
            values.par_chunks_mut(plane).enumerate().for_each(|(index, slab)| {
                fill_plane(lo.z + index as i32, slab);
            });
        } else {
            for (index, slab) in values.chunks_mut(plane).enumerate() {
                fill_plane(lo.z + index as i32, slab);
            }
        }
    }

    /// Apply `edit` to every voxel in an inclusive voxel box, allocating bricks
    /// as needed and marking the affected region dirty.
    ///
    /// `edit` receives the voxel coordinate, its world position and its current
    /// value, and returns the new value, which is clamped to the narrow band.
    ///
    /// Cost is proportional to the box, never to the size of the model. That
    /// property is the whole point of the brick grid and must not regress.
    pub fn edit_voxels(
        &mut self,
        v_min: IVec3,
        v_max: IVec3,
        edit: impl Fn(IVec3, Vec3, f32) -> f32 + Sync,
    ) {
        self.edit_voxels_where(v_min, v_max, 0, |_| BrickVerdict::Whole, edit);
    }

    /// Apply `edit` to an inclusive voxel box, leaving out whatever `decide`
    /// says cannot change.
    ///
    /// A skipped brick is not read, not made dense, not recorded for undo and
    /// not written. That is what makes it worth having: a brush box grows with
    /// the cube of the radius, but the part of it near the surface does not,
    /// and everything else is deep interior or far exterior that the edit
    /// cannot change. Promoting those to dense costs 128 KB and a memset each
    /// before the edit discovers it had nothing to do.
    ///
    /// `reach` is how many voxels past the one it is writing the edit can read,
    /// and it is what [`BrickVerdict::OnlyNearDifferentNeighbours`] is resolved
    /// against. Declaring too little is a silently wrong result, so an edit
    /// that starts resampling from further away has to widen this at the same
    /// time. Zero means it reads only the voxel it writes.
    ///
    /// `decide` is called once per brick of the box, in an unspecified order,
    /// and must be a pure function of the preview: it decides whether work
    /// happens, so an answer that varies for the same brick makes the result
    /// depend on iteration order.
    pub fn edit_voxels_where(
        &mut self,
        v_min: IVec3,
        v_max: IVec3,
        reach: i32,
        decide: impl Fn(&BrickPreview) -> BrickVerdict,
        edit: impl Fn(IVec3, Vec3, f32) -> f32 + Sync,
    ) {
        /// Voxels actually being written below which the edit stays on one
        /// core.
        ///
        /// Measured, and it matters in both directions. Sending a small stamp to
        /// a thread pool cost more than it saved and put the fast drag case over
        /// its budget. Two bricks' worth of voxels sits comfortably between the
        /// two regimes.
        const PARALLEL_VOXEL_THRESHOLD: i64 = 2 * BRICK_VOXELS as i64;

        // Lifted off the volume for the duration so the planning phase can hold
        // it while borrowing the brick map, and put back at the end so the next
        // stamp reuses the same allocations.
        let mut scratch = std::mem::take(&mut self.scratch);
        self.plan_visits(v_min, v_max, reach, &decide, &mut scratch);

        let voxels: i64 = scratch
            .visits
            .iter()
            .map(|visit| (visit.hi - visit.lo + IVec3::ONE).as_i64vec3().element_product())
            .sum();

        if voxels >= PARALLEL_VOXEL_THRESHOLD {
            self.edit_planned_across_cores(&scratch.visits, &mut scratch.taken, &edit);
        } else {
            self.edit_planned_on_one_core(&scratch.visits, &edit);
        }

        self.scratch = scratch;
        self.mark_dirty_voxel_range(v_min, v_max);
    }

    /// Work out which bricks of a box the edit will visit, and how much of
    /// each.
    ///
    /// Two passes. The first records, for every brick of the box and the ring
    /// of bricks around it, whether it holds a single value everywhere -- one
    /// map lookup each, and an absent brick counts as [`OUTSIDE`]. The second
    /// offers each brick of the box to `decide`, and for the ones it answers
    /// [`BrickVerdict::OnlyNearDifferentNeighbours`] about, narrows the range
    /// to whatever sits within `reach` voxels of a neighbour that holds
    /// something else.
    fn plan_visits<D>(
        &self,
        v_min: IVec3,
        v_max: IVec3,
        reach: i32,
        decide: &D,
        scratch: &mut EditScratch,
    ) where
        D: Fn(&BrickPreview) -> BrickVerdict,
    {
        let dim = BRICK_DIM as i32;
        let b_min = BrickCoord::containing(v_min).0;
        let b_max = BrickCoord::containing(v_max).0;
        // One brick of margin, because a voxel at a brick's face reads into the
        // brick next door. `reach` never exceeds a brick, which is checked
        // below rather than assumed.
        let grid_min = b_min - IVec3::ONE;
        let grid_size = b_max - b_min + IVec3::splat(3);

        scratch.fills.clear();
        scratch.fills.reserve(grid_size.as_i64vec3().element_product().max(0) as usize);
        for z in 0..grid_size.z {
            for y in 0..grid_size.y {
                for x in 0..grid_size.x {
                    scratch.fills.push(self.brick_fill(BrickCoord(grid_min + IVec3::new(x, y, z))));
                }
            }
        }

        let at = |brick: IVec3| {
            let local = brick - grid_min;
            (local.x + local.y * grid_size.x + local.z * grid_size.x * grid_size.y) as usize
        };

        scratch.visits.clear();
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let brick = IVec3::new(bx, by, bz);
                    let coord = BrickCoord(brick);
                    let origin = coord.origin();
                    let mut lo = v_min.max(origin);
                    let mut hi = v_max.min(coord.max_voxel());
                    if lo.cmpgt(hi).any() {
                        continue;
                    }

                    let uniform = scratch.fills[at(brick)];
                    match decide(&BrickPreview { coord, lo, hi, uniform }) {
                        BrickVerdict::Skip => continue,
                        BrickVerdict::Whole => {}
                        BrickVerdict::OnlyNearDifferentNeighbours => {
                            let Some(value) = uniform else {
                                debug_assert!(
                                    false,
                                    "a brick with detail in it has no constant to leave alone"
                                );
                                scratch.visits.push(Visit { coord, lo, hi });
                                continue;
                            };
                            // Reach is clamped to a brick so that only the 26
                            // immediate neighbours can be within it. A brush
                            // that reads further than that would have to look
                            // at more of them, and clamping keeps the answer
                            // conservative rather than wrong.
                            let reach = reach.clamp(0, dim);
                            let Some((near_lo, near_hi)) =
                                reachable_from_elsewhere(brick, value, reach, &at, &scratch.fills)
                            else {
                                // Nothing else is close enough to reach in, so
                                // the whole brick stays the value it is.
                                continue;
                            };
                            lo = lo.max(near_lo);
                            hi = hi.min(near_hi);
                            if lo.cmpgt(hi).any() {
                                continue;
                            }
                        }
                        BrickVerdict::OnlyWithin(near_lo, near_hi) => {
                            debug_assert!(
                                uniform.is_some(),
                                "a brick with detail in it has no constant to leave alone"
                            );
                            lo = lo.max(near_lo);
                            hi = hi.min(near_hi);
                            if lo.cmpgt(hi).any() {
                                continue;
                            }
                        }
                    }

                    scratch.visits.push(Visit { coord, lo, hi });
                }
            }
        }
    }

    /// One brick at a time, resolving and writing each before moving on.
    ///
    /// Kept separate from the parallel version rather than sharing its three
    /// phase shape, because that shape has to lift each brick out of the map and
    /// put it back, and doing that thousands of times a stroke leaves enough
    /// deletion markers in the table to cost 20 percent. This path only ever
    /// looks a brick up.
    fn edit_planned_on_one_core<F>(&mut self, visits: &[Visit], edit: &F)
    where
        F: Fn(IVec3, Vec3, f32) -> f32 + Sync,
    {
        let voxel_size = self.voxel_size;
        for visit in visits {
            let coord = visit.coord;
            let origin = coord.origin();

            // The prior contents have to be captured before the brick is
            // promoted to dense, because that is what destroys them.
            let recorded_now = self.record_for_undo(coord);
            let existed = self.bricks.contains_key(&coord);
            let brick = self.bricks.entry(coord).or_insert(Brick::Uniform(OUTSIDE));
            let was_uniform = matches!(brick, Brick::Uniform(_));
            let data = brick.make_dense();

            let changed = write_voxels(data, origin, visit.lo, visit.hi, voxel_size, edit);
            if !changed {
                self.undo_promotion(coord, existed, was_uniform, recorded_now);
            }
        }
    }

    /// Lift the planned bricks out of the map, write them across every core,
    /// then put them back.
    ///
    /// Removing them is what allows several to be held mutably at once, and it
    /// costs a pointer move each rather than a scan of the whole volume.
    ///
    /// The promotion to dense happens inside the parallel phase rather than
    /// while the bricks are being lifted out. It is a 128 KB allocation and
    /// memset per brick, and a large brush plans enough of them that doing it
    /// on one thread was a third of the cost of the whole edit.
    fn edit_planned_across_cores<F>(&mut self, visits: &[Visit], taken: &mut Vec<Taken>, edit: &F)
    where
        F: Fn(IVec3, Vec3, f32) -> f32 + Sync,
    {
        let voxel_size = self.voxel_size;

        taken.clear();
        taken.reserve(visits.len());
        for visit in visits {
            let coord = visit.coord;
            let recorded_now = self.record_for_undo(coord);
            let removed = self.bricks.remove(&coord);
            let existed = removed.is_some();
            let brick = removed.unwrap_or(Brick::Uniform(OUTSIDE));
            let was_uniform = matches!(brick, Brick::Uniform(_));

            taken.push(Taken {
                coord,
                brick,
                lo: visit.lo,
                hi: visit.hi,
                existed,
                was_uniform,
                recorded_now,
                changed: false,
            });
        }

        // Every brick is now a disjoint piece of memory that this thread owns,
        // and `edit` only reads, so there is nothing to synchronise.
        taken.par_iter_mut().for_each(|entry| {
            let origin = entry.coord.origin();
            let data = entry.brick.make_dense();
            entry.changed = write_voxels(data, origin, entry.lo, entry.hi, voxel_size, edit);
        });

        for entry in taken.drain(..) {
            self.bricks.insert(entry.coord, entry.brick);
            if !entry.changed {
                self.undo_promotion(
                    entry.coord,
                    entry.existed,
                    entry.was_uniform,
                    entry.recorded_now,
                );
            }
        }
    }

    /// How many bricks the last edit decided to visit.
    ///
    /// For tests that need to see the skipping actually happening rather than
    /// just agreeing with the unskipped path, which it would also do if it
    /// skipped nothing.
    pub fn last_visited_bricks(&self) -> usize {
        self.scratch.visits.len()
    }

    /// Roll back a brick that an edit's box clipped but never actually reached.
    ///
    /// A brush box always catches a few of those, and leaving them behind would
    /// mean a 128 KB allocation and an undo entry for a brick nothing touched.
    fn undo_promotion(
        &mut self,
        coord: BrickCoord,
        existed: bool,
        was_uniform: bool,
        recorded_now: bool,
    ) {
        if let Some(recorder) = self.recorder.as_mut().filter(|_| recorded_now) {
            recorder.remove(&coord);
        }
        if !was_uniform {
            return;
        }
        let value = match self.bricks.get(&coord) {
            Some(Brick::Dense(data)) => data[0],
            Some(Brick::Uniform(value)) => *value,
            None => OUTSIDE,
        };
        if existed {
            self.bricks.insert(coord, Brick::Uniform(value));
        } else {
            self.bricks.remove(&coord);
        }
    }

    /// Apply `edit` to every voxel whose world position falls inside the
    /// inclusive world space box.
    pub fn edit_box(
        &mut self,
        min_world: Vec3,
        max_world: Vec3,
        edit: impl Fn(Vec3, f32) -> f32 + Sync,
    ) {
        let (v_min, v_max) = self.voxel_bounds(min_world, max_world);
        self.edit_voxels(v_min, v_max, |_, position, value| edit(position, value));
    }

    // -------------------------------------------------------------------- undo

    /// Begin recording the prior contents of every brick a stroke touches.
    ///
    /// One stroke is one undo entry, so this is called when the button goes
    /// down and [`Volume::end_stroke`] when it comes back up. Recording is per
    /// brick and happens on first touch only, so a stroke that passes over the
    /// same brick a hundred times still costs one snapshot of it.
    pub fn begin_stroke(&mut self) {
        self.recorder = Some(FxHashMap::default());
    }

    /// Finish recording and return the undo entry, or `None` if the stroke
    /// changed nothing.
    pub fn end_stroke(&mut self) -> Option<StrokeEdit> {
        let recorder = self.recorder.take()?;
        StrokeEdit::from_recording(recorder)
    }

    /// True while a stroke is being recorded.
    #[inline]
    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Capture a brick's prior contents if a stroke is being recorded and this
    /// is the first time it has been touched. Returns whether it recorded.
    pub(crate) fn record_for_undo(&mut self, coord: BrickCoord) -> bool {
        let Some(recorder) = self.recorder.as_ref() else {
            return false;
        };
        if recorder.contains_key(&coord) {
            return false;
        }
        let prior = self.bricks.get(&coord).cloned();
        self.recorder.as_mut().expect("checked just above").insert(coord, prior);
        true
    }

    /// Swap a set of bricks back into the volume and return what was there, so
    /// the caller can undo the undo.
    ///
    /// Every restored brick and its neighbours are marked dirty, because a
    /// brick's apron reads one voxel into each of them.
    pub fn apply_edit(&mut self, edit: StrokeEdit) -> StrokeEdit {
        let mut inverse = Vec::with_capacity(edit.len());
        for (coord, brick) in edit.into_bricks() {
            let previous = match brick {
                Some(brick) => self.bricks.insert(coord, brick),
                None => self.bricks.remove(&coord),
            };
            inverse.push((coord, previous));
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        self.dirty.insert(BrickCoord(coord.0 + IVec3::new(dx, dy, dz)));
                    }
                }
            }
        }
        StrokeEdit::from_bricks(inverse)
    }

    /// Seed a sphere, replacing anything already in the volume within its
    /// bounds.
    ///
    /// Bricks fully outside the band are left absent and bricks fully inside
    /// become uniform tiles, so a large sphere allocates only its shell.
    pub fn seed_sphere(&mut self, centre: Vec3, radius: f32) {
        let voxel_size = self.voxel_size;
        let band = NARROW_BAND * voxel_size;
        let v_min = ((centre - radius - band * 2.0) / voxel_size).floor().as_ivec3();
        let v_max = ((centre + radius + band * 2.0) / voxel_size).ceil().as_ivec3();
        let b_min = BrickCoord::containing(v_min).0;
        let b_max = BrickCoord::containing(v_max).0;

        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let coord = BrickCoord::new(bx, by, bz);
                    let origin = coord.origin();
                    let box_min = origin.as_vec3() * voxel_size;
                    let box_max = coord.max_voxel().as_vec3() * voxel_size;

                    // Distance from the sphere centre to the nearest and
                    // farthest points of the brick decides whether the brick
                    // needs voxels at all.
                    let nearest = centre.clamp(box_min, box_max).distance(centre);
                    let farthest = (centre - box_min).abs().max((centre - box_max).abs()).length();

                    if nearest - radius >= band {
                        self.bricks.remove(&coord);
                        self.dirty.insert(coord);
                        continue;
                    }
                    if farthest - radius <= -band {
                        self.bricks.insert(coord, Brick::Uniform(INSIDE));
                        self.dirty.insert(coord);
                        continue;
                    }

                    let mut brick = Brick::dense_filled(OUTSIDE);
                    let data = brick.make_dense();
                    for z in 0..BRICK_DIM {
                        for y in 0..BRICK_DIM {
                            for x in 0..BRICK_DIM {
                                let position = (origin + IVec3::new(x as i32, y as i32, z as i32))
                                    .as_vec3()
                                    * voxel_size;
                                let distance = position.distance(centre) - radius;
                                data[brick_index(x, y, z)] =
                                    (distance / voxel_size).clamp(INSIDE, OUTSIDE);
                            }
                        }
                    }
                    self.bricks.insert(coord, brick);
                    self.dirty.insert(coord);
                }
            }
        }
    }

    // ------------------------------------------------------------------- stats

    pub fn stats(&self) -> VolumeStats {
        let mut stats = VolumeStats::default();
        let mut voxel_bytes = 0;
        for brick in self.bricks.values() {
            match brick {
                Brick::Uniform(_) => stats.uniform_bricks += 1,
                Brick::Dense(_) => stats.dense_bricks += 1,
            }
            voxel_bytes += brick.heap_bytes();
        }
        // Hash map overhead: one key plus one enum slot per entry, and the map
        // keeps roughly 8/7 of that in capacity.
        let entry = size_of::<BrickCoord>() + size_of::<Brick>();
        stats.resident_bytes = voxel_bytes + self.bricks.capacity() * entry;
        stats
    }

    #[inline]
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Iterate the coordinates of every stored brick. Used by tests and by the
    /// initial full mesh after seeding.
    /// A world space bounding radius for whatever the volume holds, measured
    /// from the origin.
    ///
    /// Derived from the brick extents rather than from the surface, so it costs
    /// a walk of the map's keys instead of a mesh: it is used to size interface
    /// affordances -- how far a mirror plane should reach, what a brush radius
    /// means as a fraction of the model -- and those need "about how big" rather
    /// than a tight bound.
    ///
    /// `None` when the volume is empty, which is the caller's cue to fall back
    /// rather than divide by zero.
    pub fn bounding_radius(&self) -> Option<f32> {
        let mut furthest: f32 = 0.0;
        let mut any = false;
        for coord in self.bricks.keys() {
            any = true;
            // The corner of the brick that is furthest from the origin.
            let low = coord.origin().as_vec3() * self.voxel_size;
            let high = coord.max_voxel().as_vec3() * self.voxel_size;
            furthest = furthest.max(low.abs().max(high.abs()).length());
        }
        any.then_some(furthest)
    }

    pub fn brick_coords(&self) -> impl Iterator<Item = BrickCoord> + '_ {
        self.bricks.keys().copied()
    }

    /// The brick at a coordinate, if it is stored.
    #[inline]
    pub(crate) fn brick(&self, coord: BrickCoord) -> Option<&Brick> {
        self.bricks.get(&coord)
    }

    /// The single value every voxel of a brick holds, or `None` when it carries
    /// detail.
    ///
    /// An absent brick counts as [`OUTSIDE`], which is what it reads as. One
    /// map lookup, and it is the whole basis of the skipping: an edit that
    /// cannot change a constant only has to be told which bricks are one.
    #[inline]
    pub(crate) fn brick_fill(&self, coord: BrickCoord) -> Option<f32> {
        match self.bricks.get(&coord) {
            None => Some(OUTSIDE),
            Some(Brick::Uniform(value)) => Some(*value),
            Some(Brick::Dense(_)) => None,
        }
    }

    /// Put a brick in directly. Used when building a volume from another one.
    pub(crate) fn insert_brick(&mut self, coord: BrickCoord, brick: Brick) {
        self.bricks.insert(coord, brick);
    }

    /// Take a brick out, handing over its storage.
    ///
    /// For an operation that rewrites a brick in place: cloning it out and
    /// inserting the copy back costs a second 128 KB copy on top of the one
    /// `record_for_undo` already makes, and the original is about to be
    /// discarded anyway.
    pub(crate) fn take_brick(&mut self, coord: BrickCoord) -> Option<Brick> {
        self.bricks.remove(&coord)
    }

    /// Drop a brick entirely, so it reads as empty space again.
    ///
    /// Used by the plane cut: a brick wholly past the cut is `OUTSIDE`
    /// everywhere, and an absent brick already reads that way, so removing it
    /// is both the correct answer and the free one.
    pub(crate) fn remove_brick(&mut self, coord: BrickCoord) {
        self.bricks.remove(&coord);
    }

    /// World space bounds of every stored brick, or `None` when empty.
    ///
    /// Brick extents rather than the surface itself, so this is a little loose,
    /// which is what a caller wanting to cover the content needs.
    pub fn world_bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut minimum = IVec3::splat(i32::MAX);
        let mut maximum = IVec3::splat(i32::MIN);
        for coord in self.bricks.keys() {
            minimum = minimum.min(coord.origin());
            maximum = maximum.max(coord.max_voxel());
        }
        if minimum.x > maximum.x {
            return None;
        }
        Some((minimum.as_vec3() * self.voxel_size, maximum.as_vec3() * self.voxel_size))
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_bricks_read_as_outside() {
        let volume = Volume::new(0.1);
        assert_eq!(volume.sample_voxel(IVec3::new(1000, -2000, 3)), OUTSIDE);
    }

    #[test]
    fn seeding_a_sphere_allocates_only_the_shell() {
        let mut volume = Volume::new(1.0);
        // Radius 60 voxels: the interior spans several whole bricks, which
        // must become uniform tiles rather than dense allocations.
        volume.seed_sphere(Vec3::ZERO, 60.0);
        let stats = volume.stats();
        assert!(stats.uniform_bricks > 0, "interior should collapse to tiles");
        assert!(stats.dense_bricks > 0, "shell should be dense");
        // A dense-everything implementation would allocate the full 5 cubed
        // brick bounding box or more.
        assert!(
            stats.dense_bricks < 100,
            "shell should not need {} dense bricks",
            stats.dense_bricks
        );
    }

    #[test]
    fn interior_of_a_seeded_sphere_reads_as_inside() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 60.0);
        assert_eq!(volume.sample_voxel(IVec3::ZERO), INSIDE);
        assert!(volume.sample_world(Vec3::new(0.0, 0.0, 0.0)) < 0.0);
        assert!(volume.sample_world(Vec3::new(200.0, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn gather_apron_writes_every_sample() {
        // Fill the neighbourhood with distinguishable values, then check the
        // apron reproduces the world field exactly at every one of its 34 cubed
        // samples. A gather that missed a region would leave a stale value.
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(16.0), 20.0);

        let mut apron = ApronBuffer::new();
        let coord = BrickCoord::new(0, 0, 0);
        volume.gather_apron(coord, &mut apron);

        let origin = coord.origin();
        for z in 0..crate::brick::APRON_DIM {
            for y in 0..crate::brick::APRON_DIM {
                for x in 0..crate::brick::APRON_DIM {
                    let world = origin - IVec3::ONE + IVec3::new(x as i32, y as i32, z as i32);
                    assert_eq!(
                        apron.get(x, y, z),
                        volume.sample_voxel(world),
                        "apron mismatch at local ({x}, {y}, {z})"
                    );
                }
            }
        }
        assert_eq!(apron.coord(), Some(coord));
    }

    /// Run one edit down the serial path and the same edit down the parallel
    /// one, with no bricks skipped, and hand back both volumes.
    fn both_paths(
        build: impl Fn() -> Volume,
        lo: IVec3,
        hi: IVec3,
        edit: impl Fn(IVec3, Vec3, f32) -> f32 + Sync,
    ) -> (Volume, Volume) {
        let mut one_core = build();
        let mut scratch = EditScratch::default();
        one_core.plan_visits(lo, hi, 0, &|_: &BrickPreview| BrickVerdict::Whole, &mut scratch);
        let visits = std::mem::take(&mut scratch.visits);
        one_core.edit_planned_on_one_core(&visits, &edit);
        one_core.mark_dirty_voxel_range(lo, hi);

        let mut many_cores = build();
        many_cores.plan_visits(lo, hi, 0, &|_: &BrickPreview| BrickVerdict::Whole, &mut scratch);
        let visits = std::mem::take(&mut scratch.visits);
        let mut taken = Vec::new();
        many_cores.edit_planned_across_cores(&visits, &mut taken, &edit);
        many_cores.mark_dirty_voxel_range(lo, hi);

        (one_core, many_cores)
    }

    #[test]
    fn the_two_edit_paths_agree_exactly() {
        // There are two implementations of the same edit, chosen by how much
        // work the plan holds, because the one that parallelises has to lift
        // bricks out of the map and doing that for small stamps costs more than
        // it saves. Two implementations of one thing need pinning together.
        let build = || {
            let mut volume = Volume::new(1.0);
            volume.seed_sphere(Vec3::new(6.0, -3.0, 2.0), 30.0);
            volume.take_dirty(&mut Vec::new());
            volume
        };

        // A box spanning several bricks, including a brick it clips but whose
        // voxels the edit leaves alone, so the rollback path is exercised too.
        let lo = IVec3::new(-40, -20, -10);
        let hi = IVec3::new(20, 30, 40);
        let edit = |voxel: IVec3, _position: Vec3, value: f32| {
            if voxel.x > 10 { value } else { value - 0.25 }
        };

        let (one_core, many_cores) = both_paths(build, lo, hi, edit);

        assert_eq!(
            one_core.stats(),
            many_cores.stats(),
            "the two paths left different amounts of storage allocated"
        );
        for z in lo.z - 2..=hi.z + 2 {
            for y in lo.y - 2..=hi.y + 2 {
                for x in lo.x - 2..=hi.x + 2 {
                    let voxel = IVec3::new(x, y, z);
                    assert_eq!(
                        one_core.sample_voxel(voxel),
                        many_cores.sample_voxel(voxel),
                        "the two edit paths disagree at {voxel:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn both_edit_paths_record_the_same_undo_entry() {
        let build = || {
            let mut volume = Volume::new(1.0);
            volume.seed_sphere(Vec3::ZERO, 30.0);
            volume.take_dirty(&mut Vec::new());
            volume.begin_stroke();
            volume
        };
        let lo = IVec3::new(-40, -20, -10);
        let hi = IVec3::new(20, 30, 40);
        let edit = |voxel: IVec3, _: Vec3, value: f32| {
            if voxel.x > 10 { value } else { value - 0.25 }
        };

        let (mut one_core, mut many_cores) = both_paths(build, lo, hi, edit);

        let from_one = one_core.end_stroke().expect("the edit changed something");
        let from_many = many_cores.end_stroke().expect("the edit changed something");
        assert_eq!(from_one.len(), from_many.len(), "different numbers of bricks snapshotted");
        assert_eq!(from_one.bytes(), from_many.bytes());
    }

    #[test]
    fn a_narrowed_brick_covers_exactly_what_the_neighbour_that_differs_can_reach() {
        // One differing neighbour at a time, all 26 of them. The realistic
        // case below cannot pin this down: around a real surface several
        // neighbours differ at once, and the bounding box of their slabs hides
        // a slab that is a voxel short on one face, or a corner neighbour that
        // never gets looked at.
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let offset = IVec3::new(dx, dy, dz);
                    if offset == IVec3::ZERO {
                        continue;
                    }
                    for reach in [1, 2, 7] {
                        let middle = BrickCoord::new(1, 1, 1);
                        let mut volume = Volume::new(1.0);
                        // The brick and all 26 around it hold one value, so the
                        // only thing that can reach into it is the one made
                        // dense below.
                        for z in 0..3 {
                            for y in 0..3 {
                                for x in 0..3 {
                                    volume.insert_brick(
                                        BrickCoord::new(x, y, z),
                                        Brick::Uniform(INSIDE),
                                    );
                                }
                            }
                        }
                        volume.insert_brick(
                            BrickCoord(middle.0 + offset),
                            Brick::dense_filled(OUTSIDE),
                        );

                        let mut scratch = EditScratch::default();
                        volume.plan_visits(
                            middle.origin(),
                            middle.max_voxel(),
                            reach,
                            &|preview: &BrickPreview| {
                                assert_eq!(preview.uniform, Some(INSIDE), "wrong brick offered");
                                BrickVerdict::OnlyNearDifferentNeighbours
                            },
                            &mut scratch,
                        );

                        // Per axis: the near face, the far face, or all of it
                        // where the neighbour is not offset along that axis.
                        let mut want_lo = middle.origin();
                        let mut want_hi = middle.max_voxel();
                        for axis in 0..3 {
                            match offset[axis] {
                                -1 => want_hi[axis] = middle.origin()[axis] + reach - 1,
                                1 => want_lo[axis] = middle.max_voxel()[axis] - reach + 1,
                                _ => {}
                            }
                        }

                        assert_eq!(scratch.visits.len(), 1, "offset {offset:?} reach {reach}");
                        let visit = scratch.visits[0];
                        assert_eq!(
                            (visit.lo, visit.hi),
                            (want_lo, want_hi),
                            "a neighbour at {offset:?} reaching {reach} voxels should narrow the \
                             brick to {want_lo:?}..{want_hi:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn narrowing_a_uniform_brick_leaves_out_only_what_nothing_can_reach() {
        // The invariant the whole optimisation rests on. When a brush says it
        // leaves a constant alone, the planner narrows the brick to whatever
        // sits within `reach` of a neighbour holding something else -- and
        // everything it leaves out has to be genuinely unreachable, or a stamp
        // quietly stops working part way into a brick.
        //
        // Checked against the field itself rather than against the planner's
        // own reasoning: find where the brick's neighbourhood stops holding the
        // value, grow that by the reach, and the visit has to cover it.
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(64.0), 46.0);

        for reach in [1, 2, 5] {
            let mut scratch = EditScratch::default();
            let lo = IVec3::splat(-8);
            let hi = IVec3::splat(136);
            let uniforms = std::cell::RefCell::new(Vec::new());
            volume.plan_visits(
                lo,
                hi,
                reach,
                &|preview: &BrickPreview| match preview.uniform {
                    Some(value) => {
                        uniforms.borrow_mut().push((preview.coord, value));
                        BrickVerdict::OnlyNearDifferentNeighbours
                    }
                    None => BrickVerdict::Skip,
                },
                &mut scratch,
            );

            let uniforms = uniforms.into_inner();
            assert!(!uniforms.is_empty(), "the test needs uniform bricks to be worth running");
            let mut narrowed = 0;
            let mut dropped = 0;

            for (coord, value) in uniforms {
                let visit =
                    scratch.visits.iter().find(|visit| visit.coord == coord).map(|v| (v.lo, v.hi));

                // Where the neighbourhood stops holding `value`, as a box.
                let from = coord.origin() - IVec3::splat(reach);
                let to = coord.max_voxel() + IVec3::splat(reach);
                let mut different_lo = IVec3::splat(i32::MAX);
                let mut different_hi = IVec3::splat(i32::MIN);
                for z in from.z..=to.z {
                    for y in from.y..=to.y {
                        for x in from.x..=to.x {
                            let voxel = IVec3::new(x, y, z);
                            if volume.sample_voxel(voxel) != value {
                                different_lo = different_lo.min(voxel);
                                different_hi = different_hi.max(voxel);
                            }
                        }
                    }
                }

                // Anything within `reach` of a differing voxel could change, so
                // the visit has to cover all of it. Covering more than that is
                // allowed and does happen: the planner asks whether a whole
                // neighbouring brick differs, not whether the voxels close
                // enough to matter do, so it can keep a slab that nothing
                // actually reaches into.
                let must_lo = (different_lo - IVec3::splat(reach)).max(coord.origin());
                let must_hi = (different_hi + IVec3::splat(reach)).min(coord.max_voxel());
                let must_change = must_lo.cmple(must_hi).all();

                let Some((visit_lo, visit_hi)) = visit else {
                    assert!(
                        !must_change,
                        "brick {coord:?} was dropped whole at reach {reach}, but \
                         {must_lo:?}..{must_hi:?} of it is within reach of something else"
                    );
                    dropped += 1;
                    continue;
                };
                if must_change {
                    assert!(
                        visit_lo.cmple(must_lo).all() && visit_hi.cmpge(must_hi).all(),
                        "brick {coord:?} at reach {reach} was narrowed to \
                         {visit_lo:?}..{visit_hi:?}, which leaves out {must_lo:?}..{must_hi:?}"
                    );
                }
                if (visit_lo, visit_hi) != (coord.origin(), coord.max_voxel()) {
                    narrowed += 1;
                }
            }

            assert!(dropped > 0, "no uniform brick was dropped whole at reach {reach}");
            assert!(narrowed > 0, "no uniform brick was narrowed to a slab at reach {reach}");
        }
    }

    #[test]
    fn meshing_in_parallel_gives_the_same_result_as_one_at_a_time() {
        // Meshing across cores is only safe because bricks are independent and
        // the volume is read only during it. If that ever stopped being true
        // the difference would be intermittent and awful to chase, so pin it.
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::new(10.0, -6.0, 4.0), 40.0);

        let mut coords: Vec<BrickCoord> = Vec::new();
        volume.take_dirty(&mut coords);
        coords.sort();
        assert!(coords.len() > 8, "the test needs enough bricks to be worth parallelising");

        let mut scratch = MeshScratch::new();
        let mut serial = BrickMesh::default();

        let mut parallel = vec![BrickMesh::default(); coords.len()];
        volume.mesh_bricks(&coords, &mut parallel);

        for (coord, from_many) in coords.iter().zip(parallel.iter()) {
            volume.mesh_brick(*coord, &mut scratch, &mut serial);
            assert_eq!(
                serial.vertices, from_many.vertices,
                "brick {coord:?} meshed differently in parallel"
            );
            assert_eq!(serial.indices, from_many.indices, "brick {coord:?} indices differ");
        }
    }

    #[test]
    fn meshing_reuses_the_buffers_it_is_given() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        let mut coords: Vec<BrickCoord> = Vec::new();
        volume.take_dirty(&mut coords);

        let mut meshes = vec![BrickMesh::default(); coords.len()];
        volume.mesh_bricks(&coords, &mut meshes);
        let capacities: Vec<usize> = meshes.iter().map(|mesh| mesh.vertices.capacity()).collect();

        volume.mesh_bricks(&coords, &mut meshes);
        for (mesh, capacity) in meshes.iter().zip(capacities) {
            assert_eq!(mesh.vertices.capacity(), capacity, "a buffer was reallocated");
        }
    }

    #[test]
    fn a_snapshot_reproduces_the_field_exactly() {
        // The snapshot copies brick by brick for speed rather than sampling
        // each voxel, so it can silently disagree with the volume at a brick
        // boundary. Every brush reads through it, so that would be wrong
        // everywhere at once.
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::new(20.0, -12.0, 33.0), 26.0);

        let mut region = FieldRegion::new();
        // Reused, and `FieldRegion::resize` deliberately does not clear: a
        // snapshot claims to write every element of the box, and the way that
        // claim breaks is a stale value left over from the last stamp. So take
        // one snapshot elsewhere first, and let this one land on top of it.
        volume.snapshot(IVec3::splat(-200), IVec3::splat(-140), &mut region);
        // Deliberately straddling several bricks, including negative ones.
        let lo = IVec3::new(-40, -20, 25);
        let hi = IVec3::new(6, 14, 70);
        volume.snapshot(lo, hi, &mut region);

        for z in lo.z - 1..=hi.z + 1 {
            for y in lo.y - 1..=hi.y + 1 {
                for x in lo.x - 1..=hi.x + 1 {
                    let voxel = IVec3::new(x, y, z);
                    assert_eq!(
                        region.get(voxel),
                        volume.sample_voxel(voxel),
                        "snapshot disagrees at {voxel:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn editing_marks_neighbours_dirty_only_when_the_edit_reaches_them() {
        let mut volume = Volume::new(1.0);

        // Well inside brick (0,0,0): only that brick becomes dirty.
        volume.edit_box(Vec3::splat(10.0), Vec3::splat(12.0), |_, d| d - 1.0);
        let mut dirty: Vec<_> = Vec::new();
        volume.take_dirty(&mut dirty);
        assert_eq!(dirty, vec![BrickCoord::new(0, 0, 0)]);

        // Touching the far face of brick (0,0,0) must also dirty the neighbour
        // whose apron reads those voxels.
        volume.edit_box(Vec3::new(10.0, 10.0, 31.0), Vec3::new(12.0, 12.0, 31.0), |_, d| d - 1.0);
        volume.take_dirty(&mut dirty);
        dirty.sort();
        assert!(dirty.contains(&BrickCoord::new(0, 0, 0)));
        assert!(dirty.contains(&BrickCoord::new(0, 0, 1)), "apron neighbour must remesh");
    }

    #[test]
    fn edits_that_change_nothing_do_not_leave_allocations_behind() {
        let mut volume = Volume::new(1.0);
        // An edit that returns the value unchanged over empty space.
        volume.edit_box(Vec3::splat(1.0), Vec3::splat(40.0), |_, d| d);
        assert_eq!(volume.stats().dense_bricks, 0);
        assert_eq!(volume.brick_count(), 0);
    }
}
