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
    /// Working space for [`Volume::edit_voxels`], kept between calls. A stroke
    /// lays down thousands of stamps, and the budget forbids allocating in that
    /// path.
    edit_scratch: Vec<Taken>,
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
            edit_scratch: Vec::new(),
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
    pub fn snapshot(&self, lo: IVec3, hi: IVec3, region: &mut FieldRegion) {
        let lo = lo - IVec3::ONE;
        let hi = hi + IVec3::ONE;
        let size = hi - lo + IVec3::ONE;
        let values = region.resize(lo, hi);

        let b_min = BrickCoord::containing(lo).0;
        let b_max = BrickCoord::containing(hi).0;

        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let coord = BrickCoord::new(bx, by, bz);
                    let brick_lo = coord.origin();

                    // The part of this brick that falls inside the box.
                    let clip_lo = lo.max(brick_lo);
                    let clip_hi = hi.min(coord.max_voxel());
                    if clip_lo.cmpgt(clip_hi).any() {
                        continue;
                    }

                    let run = (clip_hi.x - clip_lo.x + 1) as usize;
                    let brick = self.bricks.get(&coord);

                    for wz in clip_lo.z..=clip_hi.z {
                        for wy in clip_lo.y..=clip_hi.y {
                            let local = IVec3::new(clip_lo.x, wy, wz) - lo;
                            let start =
                                (local.x + local.y * size.x + local.z * size.x * size.y) as usize;
                            let destination = &mut values[start..start + run];

                            match brick {
                                None => destination.fill(OUTSIDE),
                                Some(Brick::Uniform(value)) => destination.fill(*value),
                                Some(Brick::Dense(data)) => {
                                    let source = brick_index(
                                        (clip_lo.x - brick_lo.x) as usize,
                                        (wy - brick_lo.y) as usize,
                                        (wz - brick_lo.z) as usize,
                                    );
                                    destination.copy_from_slice(&data[source..source + run]);
                                }
                            }
                        }
                    }
                }
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
    ///
    /// There are two implementations because a brush covers a fixed world radius
    /// and so touches cubically more voxels as the voxel size shrinks. A default
    /// brush at a quarter millimetre voxel is a few thousand voxels and belongs
    /// on one core. The same brush at the sizes M2 targets is over a million and
    /// takes six times less wall clock across cores.
    pub fn edit_voxels(
        &mut self,
        v_min: IVec3,
        v_max: IVec3,
        edit: impl Fn(IVec3, Vec3, f32) -> f32 + Sync,
    ) {
        /// Voxels in the box below which the edit stays on one core.
        ///
        /// Measured, and it matters in both directions. Sending a small stamp to
        /// a thread pool cost more than it saved and put the fast drag case over
        /// its budget. Two bricks' worth of voxels sits comfortably between the
        /// two regimes.
        const PARALLEL_VOXEL_THRESHOLD: i64 = 2 * BRICK_VOXELS as i64;

        let voxels_in_box =
            (v_max - v_min + IVec3::ONE).max(IVec3::ZERO).as_i64vec3().element_product();

        if voxels_in_box >= PARALLEL_VOXEL_THRESHOLD {
            self.edit_voxels_across_cores(v_min, v_max, &edit);
        } else {
            self.edit_voxels_on_one_core(v_min, v_max, &edit);
        }

        self.mark_dirty_voxel_range(v_min, v_max);
    }

    /// One brick at a time, resolving and writing each before moving on.
    ///
    /// Kept separate from the parallel version rather than sharing its three
    /// phase shape, because that shape has to lift each brick out of the map and
    /// put it back, and doing that thousands of times a stroke leaves enough
    /// deletion markers in the table to cost 20 percent. This path only ever
    /// looks a brick up.
    fn edit_voxels_on_one_core<F>(&mut self, v_min: IVec3, v_max: IVec3, edit: &F)
    where
        F: Fn(IVec3, Vec3, f32) -> f32 + Sync,
    {
        let voxel_size = self.voxel_size;
        let b_min = BrickCoord::containing(v_min).0;
        let b_max = BrickCoord::containing(v_max).0;

        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let coord = BrickCoord::new(bx, by, bz);
                    let origin = coord.origin();
                    let lo = v_min.max(origin);
                    let hi = v_max.min(coord.max_voxel());
                    if lo.cmpgt(hi).any() {
                        continue;
                    }

                    // The prior contents have to be captured before the brick is
                    // promoted to dense, because that is what destroys them.
                    let recorded_now = self.record_for_undo(coord);
                    let existed = self.bricks.contains_key(&coord);
                    let brick = self.bricks.entry(coord).or_insert(Brick::Uniform(OUTSIDE));
                    let was_uniform = matches!(brick, Brick::Uniform(_));
                    let data = brick.make_dense();

                    let changed = write_voxels(data, origin, lo, hi, voxel_size, edit);
                    if !changed {
                        self.undo_promotion(coord, existed, was_uniform, recorded_now);
                    }
                }
            }
        }
    }

    /// Lift the affected bricks out of the map, write them across every core,
    /// then put them back.
    ///
    /// Removing them is what allows several to be held mutably at once, and it
    /// costs a pointer move each rather than a scan of the whole volume.
    fn edit_voxels_across_cores<F>(&mut self, v_min: IVec3, v_max: IVec3, edit: &F)
    where
        F: Fn(IVec3, Vec3, f32) -> f32 + Sync,
    {
        let voxel_size = self.voxel_size;
        let b_min = BrickCoord::containing(v_min).0;
        let b_max = BrickCoord::containing(v_max).0;

        let mut taken = std::mem::take(&mut self.edit_scratch);
        taken.clear();
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let coord = BrickCoord::new(bx, by, bz);
                    let lo = v_min.max(coord.origin());
                    let hi = v_max.min(coord.max_voxel());
                    if lo.cmpgt(hi).any() {
                        continue;
                    }

                    let recorded_now = self.record_for_undo(coord);
                    let existed = self.bricks.contains_key(&coord);
                    let mut brick = self.bricks.remove(&coord).unwrap_or(Brick::Uniform(OUTSIDE));
                    let was_uniform = matches!(brick, Brick::Uniform(_));
                    brick.make_dense();

                    taken.push(Taken {
                        coord,
                        brick,
                        lo,
                        hi,
                        existed,
                        was_uniform,
                        recorded_now,
                        changed: false,
                    });
                }
            }
        }

        // Every brick is now a disjoint piece of memory that this thread owns,
        // and `edit` only reads, so there is nothing to synchronise.
        taken.par_iter_mut().for_each(|entry| {
            let data = match &mut entry.brick {
                Brick::Dense(data) => data,
                Brick::Uniform(_) => unreachable!("made dense above"),
            };
            entry.changed =
                write_voxels(data, entry.coord.origin(), entry.lo, entry.hi, voxel_size, edit);
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

        self.edit_scratch = taken;
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

    /// Put a brick in directly. Used when building a volume from another one.
    pub(crate) fn insert_brick(&mut self, coord: BrickCoord, brick: Brick) {
        self.bricks.insert(coord, brick);
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

    #[test]
    fn the_two_edit_paths_agree_exactly() {
        // There are two implementations of the same edit, chosen by how much
        // work the box holds, because the one that parallelises has to lift
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

        let mut one_core = build();
        one_core.edit_voxels_on_one_core(lo, hi, &edit);
        let mut many_cores = build();
        many_cores.edit_voxels_across_cores(lo, hi, &edit);

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

        let mut one_core = build();
        one_core.edit_voxels_on_one_core(lo, hi, &edit);
        let mut many_cores = build();
        many_cores.edit_voxels_across_cores(lo, hi, &edit);

        let from_one = one_core.end_stroke().expect("the edit changed something");
        let from_many = many_cores.end_stroke().expect("the edit changed something");
        assert_eq!(from_one.len(), from_many.len(), "different numbers of bricks snapshotted");
        assert_eq!(from_one.bytes(), from_many.bytes());
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
