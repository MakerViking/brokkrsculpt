// SPDX-License-Identifier: AGPL-3.0-only

//! The sparse brick volume: storage, sampling, editing and dirty tracking.

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::apron::ApronBuffer;
use crate::brick::{
    BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, apron_index,
    brick_index,
};
use crate::mask::{MaskBrick, MaskEdit, MaskField, MaskSlab, PROTECTED, UNMASKED};
use crate::mesh::{BrickMesh, MeshScratch, mesh_apron};
use crate::region::FieldRegion;
use crate::undo::StrokeEdit;

/// Below this many bricks in ONE call, the thread hand off costs more than the
/// meshing saves.
///
/// "In one call" is the load-bearing half of that sentence and is why this is a
/// module constant rather than a local of [`Volume::mesh_bricks`]:
/// [`crate::body::Document::mesh_dirty`] batches dirty bricks from every body
/// so that the comparison is made once, against the real total, instead of once
/// per body against a fraction of it.
pub(crate) const PARALLEL_MESH_THRESHOLD: usize = 4;

/// Counters for the debug overlay and the memory budget.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VolumeStats {
    /// Bricks holding a full voxel array.
    pub dense_bricks: usize,
    /// Bricks collapsed to a single tile value, which cost no voxel storage.
    pub uniform_bricks: usize,
    /// Mask bricks, of which [`VolumeStats::mask_dense_bricks`] carry an array.
    ///
    /// Counted apart from the field's bricks rather than folded into them,
    /// because a mask brick can exist where no field brick does -- a protection
    /// value over empty space is exactly what stops Draw growing material there
    /// -- so the two censuses do not cover the same coordinates.
    pub mask_bricks: usize,
    /// Mask bricks holding a full byte array, at 32,768 bytes each.
    pub mask_dense_bricks: usize,
    /// Bytes of mask data, excluding the map that indexes it.
    pub mask_bytes: usize,
    /// Bytes of voxel data, mask data, and the maps that index them.
    pub resident_bytes: usize,
}

thread_local! {
    /// How many times [`Volume::duplicated`] has run on THIS thread.
    ///
    /// Read it through [`copies_made_on_this_thread`].
    ///
    /// This exists so that a caller which promises to refuse a copy *before*
    /// allocating it can be held to that promise. The promise is worth a
    /// counter: the 765 MB dragon is 1.53 GiB of memory traffic, and a guard
    /// that has been hoisted below the allocation still refuses, still reports
    /// the right numbers, and still leaves the document untouched -- so every
    /// assertion a test can make about the DOCUMENT passes either way. Nothing
    /// but a count of the copies distinguishes the two.
    ///
    /// **Not `#[cfg(test)]`**, because the caller that has to be pinned lives
    /// in another crate and links this one built without `cfg(test)`. The cost
    /// is one non-atomic increment per duplicate, against a copy that is
    /// already measured in gibibytes.
    ///
    /// **Thread-local rather than a global**, because the test harness runs
    /// tests in parallel threads: a shared counter would make the assertion
    /// race against every other test that duplicates anything.
    static COPIES_MADE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The value of that counter, for a test that needs to see a copy NOT happen.
///
/// Take it before and after and compare the difference; the absolute value is
/// whatever the rest of the thread has done.
pub fn copies_made_on_this_thread() -> usize {
    COPIES_MADE.with(std::cell::Cell::get)
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
    /// `Some(p)` when every voxel of this brick resolves to the same
    /// protection `p`, which covers an absent mask brick as well as a collapsed
    /// tile. `None` when the mask carries detail across it.
    ///
    /// **RESOLVED protection, with polarity already applied**, so
    /// `Some(`[`crate::PROTECTED`]`)` means "no brush may touch any of this" whichever
    /// way the polarity is pointing. Reading it as the stored byte instead
    /// would fire the skip on the fully FREE bricks the moment Invert is on,
    /// which is the loudest possible way to get masking wrong.
    ///
    /// Always `Some(`[`UNMASKED`]`)` for an edit that passed `use_mask` as
    /// false, so a caller that does not mask sees a mask that protects nothing
    /// rather than a mask it has to reason about.
    pub mask: Option<u8>,
}

/// What an edit wants done with one brick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickVerdict {
    /// Nothing in it can change, so do not read it, make it dense, record it
    /// for undo or write it.
    ///
    /// A fully protected brick is the strongest case there is for this, and the
    /// only one that needs no reasoning about neighbours at all: a mask kills
    /// the WRITE rather than the read, so there is nothing that could reach in
    /// and no rim to leave behind. The consequence is worth stating plainly --
    /// a stroke over a masked region gets cheaper, not dearer.
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

/// What the last edit's planning phase decided, brick by brick and voxel by
/// voxel.
///
/// Bricks alone cannot answer whether the skipping is working. A brick the
/// classifier narrowed to a two voxel slab still counts as one visited brick,
/// so a brick ratio reads a 94 percent saving as no saving at all. The voxel
/// counts are the ones to look at; the verdict tallies say which rule earned
/// them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlanStats {
    /// Bricks the edit's box covers, whatever was decided about them.
    pub bricks_in_box: usize,
    /// Bricks the edit ruled out entirely, by either verdict.
    pub bricks_skipped: usize,
    /// Bricks resolved over the whole of what the box covers.
    pub bricks_whole: usize,
    /// Bricks narrowed to the part within reach of a different neighbour,
    /// by [`BrickVerdict::OnlyNearDifferentNeighbours`] or
    /// [`BrickVerdict::OnlyWithin`].
    pub bricks_narrowed: usize,
    /// Voxels the edit's box covers.
    pub voxels_in_box: i64,
    /// Voxels the edit will actually visit.
    pub voxels_visited: i64,
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
    /// What the last planning pass decided. Counters only, never read by the
    /// edit itself.
    plan: PlanStats,
}

/// One brick's protection, resolved once per brick and applied per voxel.
///
/// The whole point of resolving it here rather than inside the edit is that
/// `edit` is `Fn(..) + Sync` and the parallel path runs it across cores, so
/// there is nowhere to cache a map lookup and no interior mutability to hang
/// one off. A probe per voxel is millions of `FxHashMap` lookups on a large
/// stamp against a couple of hundred if the slab is resolved per brick.
///
/// `pub(crate)` for [`crate::clip`], which is not a brush and still has to
/// resolve exactly the same thing. **A second copy of the polarity rule is how
/// an inverted mask gets sculpted straight through**, so the cut borrows this
/// one rather than reading the mask itself: the two constants below, the
/// absent-means-stored-zero reading and the hoist on
/// [`Freedom::uniform`] are decided once, here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Freedom<'a> {
    slab: MaskSlab<'a>,
    /// The two constants that turn a stored byte into the factor in `0..=1`
    /// that the edit multiplies by, in one fused multiply-add.
    ///
    /// `(1.0, -1/255)` under normal polarity and `(0.0, 1/255)` under
    /// inversion, so Invert stays a bool and polarity costs the same either
    /// way. Both forms land exactly on 0.0 and exactly on 1.0 at the ends of
    /// the byte range -- checked over all 256 bytes and both polarities by
    /// `every_mask_factor_stays_inside_zero_to_one`, because the brush weight
    /// staying inside `0..=1` is what makes smooth, flatten and clay blend
    /// toward their target instead of extrapolating away from it.
    offset: f32,
    scale: f32,
}

impl Freedom<'_> {
    /// Nothing is protected: every voxel feels the whole of the edit.
    ///
    /// What an edit that passed `use_mask` as false gets, and it is a real
    /// resolution rather than a sentinel -- [`Freedom::uniform`] answers
    /// `Some(1.0)`, so the loop hoists exactly as it does for an unmasked body.
    const OPEN: Freedom<'static> = Freedom { slab: MaskSlab::Free, offset: 1.0, scale: 0.0 };

    /// This brick's protection, or [`Freedom::OPEN`] when the edit is not
    /// masked at all.
    pub(crate) fn resolve(mask: Option<&MaskField>, coord: BrickCoord) -> Freedom<'_> {
        let Some(mask) = mask else {
            return Freedom::OPEN;
        };
        const STEP: f32 = 1.0 / 255.0;
        // Absent means STORED BYTE 0, never "free". Under inversion -- which is
        // exactly what Mask All is -- byte 0 resolves to fully protected, so an
        // arm that short-circuited an absent brick to 1.0 would sculpt at full
        // strength straight through the single most-used masking state.
        let (offset, scale) = if mask.inverted() { (0.0, STEP) } else { (1.0, -STEP) };
        Freedom { slab: mask.slab(coord), offset, scale }
    }

    /// The factor the whole brick shares, or `None` when it carries detail.
    /// This is the test the voxel loop hoists on.
    #[inline]
    pub(crate) fn uniform(&self) -> Option<f32> {
        self.slab.fill().map(|byte| self.offset + self.scale * byte as f32)
    }

    /// The factor for one voxel of a brick that carries detail.
    #[inline]
    pub(crate) fn at(&self, index: usize) -> f32 {
        self.offset + self.scale * self.slab.byte_at(index) as f32
    }
}

/// Apply an edit to one brick's voxels, returning whether anything changed.
///
/// A generic function rather than a closure, so that the brush's per voxel
/// maths, which for the resampling brushes is an eight tap trilinear read,
/// inlines into the loop.
///
/// `free` is the fourth argument handed to the edit: how much of itself the
/// edit is allowed to apply to this voxel, in `0..=1`. It is resolved from the
/// brick's mask through the `brick_index` the field write already computed, so
/// a masked voxel costs one extra `u8` load and no address arithmetic.
#[inline]
fn write_voxels<F>(
    data: &mut [f32; BRICK_VOXELS],
    origin: IVec3,
    lo: IVec3,
    hi: IVec3,
    voxel_size: f32,
    freedom: Freedom<'_>,
    edit: &F,
) -> bool
where
    F: Fn(IVec3, Vec3, f32, f32) -> f32 + Sync,
{
    // Hoisted, so an unmasked body -- and a body whose mask collapsed this
    // brick to a tile -- pays nothing per voxel at all.
    let uniform = freedom.uniform();
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
                let free = match uniform {
                    Some(free) => free,
                    None => freedom.at(index),
                };
                let position = voxel.as_vec3() * voxel_size;
                let old = data[index];
                let new = edit(voxel, position, old, free).clamp(INSIDE, OUTSIDE);
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
    /// How much of a brush stroke each voxel is allowed to feel.
    ///
    /// A second sparse map BESIDE the bricks and not inside them, and inside
    /// the volume rather than beside it. [`crate::mask`]'s module documentation
    /// gives the three reasons for the first and the two for the second; each
    /// of them is a silent failure rather than a preference.
    mask: MaskField,
    dirty: FxHashSet<BrickCoord>,
    /// Prior contents of every brick touched since the stroke began, or `None`
    /// when no stroke is in progress. The inner `None` means the brick did not
    /// exist, which undo has to restore just as faithfully as any content.
    recorder: Option<FxHashMap<BrickCoord, Option<Brick>>>,
    /// The same, for the mask, and a second map rather than a widened first
    /// one.
    ///
    /// A sculpt stroke fills the first and leaves this empty, because it never
    /// writes the mask; a mask stroke does the reverse. Merging them would put
    /// a `None` in the field list for every mask brick a paint stroke created
    /// and vice versa, and `apply_edit` would then remove field bricks a mask
    /// stroke never touched.
    mask_recorder: Option<FxHashMap<BrickCoord, Option<MaskBrick>>>,
    /// The mask's polarity when the current stroke opened, once something in
    /// the stroke has changed it. See [`crate::undo::StrokeEdit`].
    mask_polarity: Option<bool>,
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
            mask: MaskField::default(),
            dirty: FxHashSet::default(),
            recorder: None,
            mask_recorder: None,
            mask_polarity: None,
            scratch: EditScratch::default(),
        }
    }

    #[inline]
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// This body's mask.
    ///
    /// **`pub` rather than `pub(crate)`, which increment 18 wrote and this
    /// increment had to widen.** The plan requires a masked row in
    /// `benches/budget.rs`, or the budget gate only ever measures unmasked
    /// strokes and stays green through a regression to a map lookup per voxel;
    /// a bench links this crate as an ordinary dependency, so it cannot reach a
    /// `pub(crate)` item and there is no other route to a mask on a `Volume`.
    /// Increment 21 needs one from `brokkr-app` for the same reason.
    ///
    /// The half of the original claim that was load-bearing survives: a mask is
    /// only reachable THROUGH the volume that owns it, so no expression can
    /// hand one body's mesher, serialiser or brush another body's mask. What is
    /// given up is "nothing outside this crate can hold a [`MaskField`]", which
    /// never protected those three -- all of them live inside this crate.
    #[inline]
    pub fn mask(&self) -> &MaskField {
        &self.mask
    }

    #[inline]
    pub fn mask_mut(&mut self) -> &mut MaskField {
        &mut self.mask
    }

    /// Swap a whole mask onto this body and hand back the one that was there.
    ///
    /// **The engine under Clear, Mask All and every absolute filter**, and a
    /// swap rather than a rewrite for the reason
    /// [`crate::undo::Change::NodeRemoved`] moves a node rather than cloning
    /// it: the mask that comes off is the ONLY copy of itself, so a Clear on a
    /// gigabyte of protection allocates nothing at all and peak memory does not
    /// rise -- it merely does not fall until the history entry holding it is
    /// evicted.
    ///
    /// Every brick either side of the swap is marked dirty, with its
    /// neighbours, because the mask is baked into a vertex attribute at mesh
    /// time: a mask brick that goes away is only off the screen once the bricks
    /// it covered have been remeshed, which is the same class of bug as undoing
    /// a carve and seeing the old triangles stay.
    ///
    /// **Polarity travels with the field and is not a separate argument.** That
    /// is what makes Mask All one change and not two -- `cleared(true)` is
    /// clear-then-invert as a single value -- and it is what makes undoing one
    /// put both halves back together, which is the only state either half is
    /// meaningful in.
    pub fn replace_mask(&mut self, mask: MaskField) -> MaskField {
        // Marked while each field is a local, because `mark_brick_and_
        // neighbours_dirty` needs `&mut self` and `self.mask.brick_coords()`
        // would be holding a shared borrow of it.
        self.mark_mask_bricks_dirty(&mask);
        let previous = std::mem::replace(&mut self.mask, mask);
        self.mark_mask_bricks_dirty(&previous);
        // **Past the one that just left, and not merely one on from its own.**
        // An absolute filter builds every step of a drag from the SAME
        // snapshot, so each of them arrives carrying `snapshot.revision + 1` --
        // one number for a blur at 0.3 and a blur at 0.9. The standing card is
        // keyed on it. See [`MaskField::advance_revision_past`].
        self.mask.advance_revision_past(previous.revision());
        previous
    }

    /// Take this body's mask off, leaving an empty one of the same polarity.
    ///
    /// The grab half of an absolute filter gesture: the snapshot the filter
    /// re-applies from is MOVED out to the caller, so a drag holds the snapshot
    /// and the field being built and never a third copy. Dropping the returned
    /// value is how the caller discards the previous change of the same
    /// gesture before building the next one.
    pub fn take_mask(&mut self) -> MaskField {
        let blank = self.mask.cleared(self.mask.inverted());
        self.replace_mask(blank)
    }

    /// Flip the mask's polarity, and hand back the undo entry for it.
    ///
    /// O(1) in time, memory and undo, whatever the mask holds: the entry
    /// carries one bool. Without it, ctrl+I on a lightly masked 45,567-brick
    /// model allocates 1.04 GiB from one keystroke.
    ///
    /// **It marks NOTHING dirty, and that is the payoff of resolving polarity
    /// in the shader rather than baking it into the mesh.** The vertex
    /// attribute carries the STORED byte -- see
    /// [`crate::MaskField::bytes_at_cells`] -- and `Uniforms::mask_inverted`
    /// resolves it per draw, so an Invert is one word written to the GPU
    /// instead of a whole-body remesh: 71 ms on the reference dragon and
    /// roughly 475 ms at the brick count the mesh pool is sized for.
    pub fn flip_mask_polarity(&mut self) -> StrokeEdit {
        let was = self.mask.inverted();
        self.mask.set_inverted(!was);
        StrokeEdit::from_parts(Vec::new(), Vec::new(), Some(was))
    }

    /// Mark every brick a mask holds an entry for, and their neighbours.
    ///
    /// `mask` is a parameter and never `self.mask`, which is what lets this
    /// iterate one field while dirtying the volume that is about to hold it.
    fn mark_mask_bricks_dirty(&mut self, mask: &MaskField) {
        for coord in mask.brick_coords() {
            self.mark_brick_and_neighbours_dirty(coord);
        }
    }

    /// Run `work` with this volume's mask lifted off it.
    ///
    /// The mask lives INSIDE the volume -- see [`crate::mask`] for why -- and
    /// everything that applies it needs `&mut self` at the same moment, which
    /// the borrow checker forbids while `&self.mask` is alive.
    /// [`Volume::edit_voxels_where`] makes exactly this move inline; this exists
    /// because [`crate::clip`] is a different module and so cannot reach the
    /// field at all.
    ///
    /// A scoped helper rather than a `take`/`put` pair on purpose. Between those
    /// two an early return leaves the body's mask behind, and a mask that
    /// vanishes on one code path is a loss nothing downstream would report.
    ///
    /// `None` is handed over when the mask protects nothing anywhere, so a
    /// caller on an unmasked body can take the path it took before masks
    /// existed rather than a path that merely computes the same answer.
    pub(crate) fn with_mask_lifted<R>(
        &mut self,
        work: impl FnOnce(&mut Self, Option<&MaskField>) -> R,
    ) -> R {
        let mask = std::mem::take(&mut self.mask);
        let out = work(self, (!mask.is_free()).then_some(&mask));
        self.mask = mask;
        out
    }

    /// Scale the whole model in world space, without touching a single voxel.
    ///
    /// This is free, and the reason is worth stating because it is not obvious:
    /// **distances are stored in voxels, not in millimetres.** The field is
    /// therefore scale-free, and how big the model is in the world is decided
    /// entirely by `voxel_size`. Making the model 2.7 times smaller is the same
    /// operation as making the voxel 2.7 times smaller, and neither resamples
    /// anything -- the bricks are bit-identical afterwards.
    ///
    /// **It buys no detail.** The model still has exactly as many voxels across
    /// it as it did before, so the finest feature it can hold is unchanged. What
    /// it changes is what one voxel MEASURES, which is the only thing that
    /// decides whether the detail already there is enough for a given printer.
    /// A caller offering this to a user should say so; see the detail panel.
    pub fn rescale(&mut self, factor: f32) {
        assert!(factor.is_finite() && factor > 0.0, "a scale factor must be finite and positive");
        self.voxel_size *= factor;
    }

    /// A deep copy of this field, with every brick moved by `offset_bricks`.
    ///
    /// **Named, and deliberately not `impl Clone`.** A `Volume` is the heaviest
    /// thing this application holds, and `.clone()` is one syllable that would
    /// hide all of it: the 765 MB dragon is 6,120 dense bricks, so a copy of it
    /// is 6,120 allocations and 1.53 GiB of memory traffic. Nothing else in the
    /// engine copies a field at all -- a delete MOVES the volume into the undo
    /// entry, and resample, rotate and clip each build a new one while dropping
    /// the old -- so duplicate is the first operation whose whole cost IS the
    /// copy, and the call site is where that has to be visible. Deriving
    /// `Clone` would also make it available to every `#[derive]` above every
    /// type that ever comes to hold a `Volume`, which is how a gigabyte gets
    /// copied by a struct update nobody read.
    ///
    /// # Why this is not `rotated(identity)`
    ///
    /// [`Volume::rotated`] walks 32,768 scalars per dense brick, because a
    /// quarter turn genuinely moves every voxel to a different index inside its
    /// brick. A translation by WHOLE BRICKS moves no voxel within its brick, so
    /// each one is copied by `Brick::clone` -- a single 128 KB memcpy for a
    /// dense brick, and nothing at all for a tile. The offset is in bricks
    /// rather than in voxels for exactly that reason: a sub-brick offset would
    /// have to rebuild every brick from two neighbours, which is the resample
    /// this whole lattice exists to avoid, and it would cost the surface a
    /// little detail on the way through.
    ///
    /// Serial where [`Volume::rotated`] is parallel, and that is a choice
    /// rather than an omission: this is a memcpy of the brick map and so is
    /// bound by memory bandwidth, not by arithmetic, and there is no per-voxel
    /// work for the other cores to take. The 1.53 GiB is the cost, and no
    /// scheduling makes it smaller.
    ///
    /// The copy comes back with **every** brick marked dirty, because nothing
    /// else will mark them -- the same rule and the same reason as
    /// [`crate::primitive::build`]. A body whose bricks were never meshed sits
    /// in the document, exports correctly, and is invisible on screen for the
    /// rest of the session. This project has shipped that twice.
    pub fn duplicated(&self, offset_bricks: IVec3) -> Volume {
        COPIES_MADE.with(|made| made.set(made.get() + 1));
        let mut copy = Volume::new(self.voxel_size);
        copy.bricks.reserve(self.bricks.len());
        for (coord, brick) in &self.bricks {
            copy.bricks.insert(BrickCoord(coord.0 + offset_bricks), brick.clone());
        }
        // The mask moves with the body it protects. A copy that arrived
        // unmasked would have no undo and nothing on screen to say so: the
        // duplicate looks right, and the protection the user painted is gone.
        copy.mask = self.mask.translated(offset_bricks);
        // A stroke in progress is NOT carried over. `recorder` holds the prior
        // contents of the bricks this volume's own gesture touched, and those
        // name an undo entry that belongs to the body being copied; a copy that
        // arrived mid-recording would end that stroke by restoring the
        // original's bricks into the wrong body. `Volume::new` leaves it
        // `None`, which is "no stroke here", which is the truth about a field
        // that has never been edited.
        copy.mark_everything_dirty();
        copy
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
    ///
    /// **It also fills [`BrickMesh::mask`], and that is the whole reason the
    /// mask lives inside the volume** rather than beside it: this function
    /// takes only `&self`, so there is no expression that could hand one body's
    /// mesher another body's mask. The apron is not widened to carry it -- the
    /// lookup is by world cell, which is an exact identity for a vertex and is
    /// shared by both bricks either side of a seam, so there is no halo to
    /// gather and no step at a brick boundary. See [`crate::mask`].
    pub fn mesh_brick(&self, coord: BrickCoord, scratch: &mut MeshScratch, out: &mut BrickMesh) {
        self.gather_apron(coord, &mut scratch.apron);
        mesh_apron(&scratch.apron, coord, self.voxel_size, &mut scratch.surface_nets, out);
        self.mask.bytes_at_cells(coord, &out.cells, &mut out.mask);
        debug_assert_eq!(
            out.mask.len(),
            out.vertices.len(),
            "one mask byte per vertex, or the pool writes a short attribute slice"
        );
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

        if coords.len() < PARALLEL_MESH_THRESHOLD {
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
        self.drain_dirty(|coord| out.push(coord));
    }

    /// Move the dirty set out one coordinate at a time, keeping the set's
    /// allocation.
    ///
    /// A callback rather than a returned iterator so that a caller collecting
    /// several volumes' dirty sets into ONE list -- which is what
    /// [`crate::body::Document::take_dirty`] does, so that the remesh can tag
    /// each coordinate with the body it came from -- needs no scratch vector
    /// per volume. A per-body scratch would be an allocation on the stroke
    /// path, which is the one path in this engine that must never touch the
    /// allocator.
    pub fn drain_dirty(&mut self, mut visit: impl FnMut(BrickCoord)) {
        for coord in self.dirty.drain() {
            visit(coord);
        }
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
    ///
    /// **Not masked.** The voxeliser and the tests are the callers, and neither
    /// is a brush. The plane cut is NOT one of them despite the shape of what it
    /// does: it classifies bricks itself, because its `Removes` verdict deletes
    /// a whole brick rather than writing voxels and no multiply in this path
    /// could reach that. It answers for the mask on its own -- see
    /// [`crate::clip`].
    pub fn edit_voxels(
        &mut self,
        v_min: IVec3,
        v_max: IVec3,
        edit: impl Fn(IVec3, Vec3, f32) -> f32 + Sync,
    ) {
        self.edit_voxels_where(
            v_min,
            v_max,
            0,
            false,
            |_| BrickVerdict::Whole,
            |voxel, position, value, _free| edit(voxel, position, value),
        );
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
    ///
    /// # `use_mask` is a `bool` and not an `Option<&MaskField>`
    ///
    /// Forced by the borrow checker rather than preferred. The mask lives
    /// inside the volume, and both callers hold a `&mut Volume`, so
    /// `volume.edit_voxels_where(.., &volume.mask, ..)` would need a shared
    /// borrow of `self.mask` outliving call activation while `&mut self` is
    /// live; two phase borrows do not rescue it. The fix is the one `scratch`
    /// already uses below: lift the mask off `self` for the duration and put it
    /// back. That also means no expression can hand one body's edit another
    /// body's mask, which is half the reason the mask lives inside `Volume` at
    /// all.
    ///
    /// An edit that passes `false` sees a `free` of exactly 1.0 at every voxel
    /// and a [`BrickPreview::mask`] of `Some(`[`UNMASKED`]`)` at every brick.
    pub fn edit_voxels_where(
        &mut self,
        v_min: IVec3,
        v_max: IVec3,
        reach: i32,
        use_mask: bool,
        decide: impl Fn(&BrickPreview) -> BrickVerdict,
        edit: impl Fn(IVec3, Vec3, f32, f32) -> f32 + Sync,
    ) {
        /// Voxels actually being written below which the edit stays on one
        /// core.
        ///
        /// Measured, and it matters in both directions. Sending a small stamp to
        /// a thread pool cost more than it saved and put the fast drag case over
        /// its budget. Two bricks' worth of voxels sits comfortably between the
        /// two regimes.
        const PARALLEL_VOXEL_THRESHOLD: i64 = 2 * BRICK_VOXELS as i64;

        // Both lifted off the volume for the duration so the planning phase can
        // hold them while borrowing the brick map, and put back at the end so
        // the next stamp reuses the same allocations. See the note above on why
        // the mask has to travel this way rather than as a parameter.
        let mut scratch = std::mem::take(&mut self.scratch);
        let mask = std::mem::take(&mut self.mask);
        let masking = use_mask.then_some(&mask);
        self.plan_visits(v_min, v_max, reach, masking, &decide, &mut scratch);

        let voxels: i64 = scratch
            .visits
            .iter()
            .map(|visit| (visit.hi - visit.lo + IVec3::ONE).as_i64vec3().element_product())
            .sum();

        if voxels >= PARALLEL_VOXEL_THRESHOLD {
            self.edit_planned_across_cores(&scratch.visits, &mut scratch.taken, masking, &edit);
        } else {
            self.edit_planned_on_one_core(&scratch.visits, masking, &edit);
        }

        self.mask = mask;
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
    ///
    /// The mask is answered per brick from its own map -- one lookup, no ring,
    /// because a mask kills the write and so needs no neighbour reasoning.
    fn plan_visits<D>(
        &self,
        v_min: IVec3,
        v_max: IVec3,
        reach: i32,
        mask: Option<&MaskField>,
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
        scratch.plan = PlanStats::default();
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
                    scratch.plan.bricks_in_box += 1;
                    scratch.plan.voxels_in_box +=
                        (hi - lo + IVec3::ONE).as_i64vec3().element_product();

                    let uniform = scratch.fills[at(brick)];
                    // Resolved protection, not the stored byte: see
                    // `BrickPreview::mask`. An unmasked edit sees a mask that
                    // protects nothing.
                    let protection = match mask {
                        None => Some(UNMASKED),
                        Some(mask) => mask.protection_fill(coord),
                    };
                    match decide(&BrickPreview { coord, lo, hi, uniform, mask: protection }) {
                        BrickVerdict::Skip => {
                            scratch.plan.bricks_skipped += 1;
                            continue;
                        }
                        BrickVerdict::Whole => scratch.plan.bricks_whole += 1,
                        BrickVerdict::OnlyNearDifferentNeighbours => {
                            let Some(value) = uniform else {
                                debug_assert!(
                                    false,
                                    "a brick with detail in it has no constant to leave alone"
                                );
                                scratch.plan.bricks_whole += 1;
                                scratch.plan.voxels_visited +=
                                    (hi - lo + IVec3::ONE).as_i64vec3().element_product();
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
                                scratch.plan.bricks_skipped += 1;
                                continue;
                            };
                            lo = lo.max(near_lo);
                            hi = hi.min(near_hi);
                            if lo.cmpgt(hi).any() {
                                scratch.plan.bricks_skipped += 1;
                                continue;
                            }
                            scratch.plan.bricks_narrowed += 1;
                        }
                        BrickVerdict::OnlyWithin(near_lo, near_hi) => {
                            debug_assert!(
                                uniform.is_some(),
                                "a brick with detail in it has no constant to leave alone"
                            );
                            lo = lo.max(near_lo);
                            hi = hi.min(near_hi);
                            if lo.cmpgt(hi).any() {
                                scratch.plan.bricks_skipped += 1;
                                continue;
                            }
                            scratch.plan.bricks_narrowed += 1;
                        }
                    }

                    scratch.plan.voxels_visited +=
                        (hi - lo + IVec3::ONE).as_i64vec3().element_product();
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
    fn edit_planned_on_one_core<F>(&mut self, visits: &[Visit], mask: Option<&MaskField>, edit: &F)
    where
        F: Fn(IVec3, Vec3, f32, f32) -> f32 + Sync,
    {
        let voxel_size = self.voxel_size;
        for visit in visits {
            let coord = visit.coord;
            let origin = coord.origin();
            let freedom = Freedom::resolve(mask, coord);

            // The prior contents have to be captured before the brick is
            // promoted to dense, because that is what destroys them.
            let recorded_now = self.record_for_undo(coord);
            let existed = self.bricks.contains_key(&coord);
            let brick = self.bricks.entry(coord).or_insert(Brick::Uniform(OUTSIDE));
            let was_uniform = matches!(brick, Brick::Uniform(_));
            let data = brick.make_dense();

            let changed = write_voxels(data, origin, visit.lo, visit.hi, voxel_size, freedom, edit);
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
    ///
    /// # What makes this safe to run across cores
    ///
    /// Each entry owns its brick outright for the duration, so the parallel
    /// phase shares nothing mutable; `edit` is `Sync` and only reads, and what
    /// the resampling brushes read is a *copy* taken before the edit rather
    /// than the volume being written. The one precondition that is not
    /// self-evident is that `visits` holds each brick at most once -- a repeat
    /// would find the second `remove` empty, edit a fresh empty brick, and then
    /// overwrite the first entry's work on the way back in, losing an edit
    /// silently and only sometimes. `plan_visits` walks a brick range and emits
    /// one visit per brick, so it holds; the assertion below is what keeps it
    /// holding if that ever stops being how visits are produced.
    fn edit_planned_across_cores<F>(
        &mut self,
        visits: &[Visit],
        taken: &mut Vec<Taken>,
        mask: Option<&MaskField>,
        edit: &F,
    ) where
        F: Fn(IVec3, Vec3, f32, f32) -> f32 + Sync,
    {
        debug_assert!(
            visits.windows(2).all(|pair| {
                let (a, b) = (pair[0].coord.0, pair[1].coord.0);
                (a.z, a.y, a.x) < (b.z, b.y, b.x)
            }),
            "visits must be strictly ascending, and so each brick at most once"
        );

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
        // and `edit` only reads, so there is nothing to synchronise. The mask
        // is shared and read-only for the whole stroke, so resolving each
        // brick's slab in here rather than while lifting the bricks out costs a
        // map lookup per brick on a worker instead of on this thread.
        taken.par_iter_mut().for_each(|entry| {
            let origin = entry.coord.origin();
            let freedom = Freedom::resolve(mask, entry.coord);
            let data = entry.brick.make_dense();
            entry.changed =
                write_voxels(data, origin, entry.lo, entry.hi, voxel_size, freedom, edit);
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

    /// What the last edit's planning phase decided.
    ///
    /// The voxel counts are the honest measure of the skipping; see
    /// [`PlanStats`] for why the brick counts alone mislead.
    pub fn last_plan(&self) -> PlanStats {
        self.scratch.plan
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
    ///
    /// **Opening a recorder over a live one is a bug and no longer silent.**
    /// It used to overwrite the map, which threw away every brick recorded so
    /// far: the stroke went on carving, the entry that was finally pushed
    /// restored only the tail of it, and nothing anywhere said so. With more
    /// than one body it is also how a recorder ends up open on the body the
    /// user is NOT sculpting -- and `record_for_undo` does nothing at all when
    /// no recorder is open, so that carve is pushed nowhere, never sets
    /// `unsaved`, and does not even raise the discard prompt on the way out.
    /// A caller that cannot know whether a stroke is already live asks
    /// [`Volume::is_recording`] first.
    pub fn begin_stroke(&mut self) {
        debug_assert!(
            self.recorder.is_none(),
            "a stroke is already being recorded, and opening another would discard it"
        );
        self.recorder = Some(FxHashMap::default());
        self.mask_recorder = Some(FxHashMap::default());
        self.mask_polarity = None;
    }

    /// Finish recording and return the undo entry, or `None` if the stroke
    /// changed nothing.
    ///
    /// **The only place the mask collapses.** Deliberately here rather than per
    /// edit: [`crate::Brick::is_collapsible`]'s saturated-only rule is an
    /// anti-thrash heuristic for a brick the next stamp is about to rewrite, and
    /// waiting for the end of the gesture removes the thrash it guards against.
    ///
    /// The collapse runs BEFORE the recording is taken and that is safe: it
    /// changes how a mask brick is stored and not one value any voxel reads
    /// back, while the recorder holds bricks copied out before the stroke wrote
    /// them.
    pub fn end_stroke(&mut self) -> Option<StrokeEdit> {
        self.mask.collapse();
        let recorder = self.recorder.take()?;
        let masks = self.mask_recorder.take().unwrap_or_default();
        StrokeEdit::from_recording(recorder, masks, self.mask_polarity.take())
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
    /// brick's apron reads one voxel into each of them. **The mask half marks
    /// them too**, and that is not symmetry for its own sake: the mask is baked
    /// into a vertex attribute at mesh time, so a mask brick put back by an undo
    /// is only on screen once the bricks it covers have been remeshed. Undoing a
    /// mask stroke and seeing the old tint stay is the same class of bug as
    /// undoing a carve and seeing the old triangles stay.
    pub fn apply_edit(&mut self, edit: StrokeEdit) -> StrokeEdit {
        let (bricks, masks, polarity) = edit.into_parts();
        let mut inverse = Vec::with_capacity(bricks.len());
        for (coord, brick) in bricks {
            let previous = match brick {
                Some(brick) => self.bricks.insert(coord, brick),
                None => self.bricks.remove(&coord),
            };
            inverse.push((coord, previous));
            self.mark_brick_and_neighbours_dirty(coord);
        }

        let mut mask_inverse = Vec::with_capacity(masks.len());
        for (coord, brick) in masks {
            let previous = self.mask.brick(coord).cloned();
            self.mask.restore_brick(coord, brick);
            mask_inverse.push((coord, previous));
            self.mark_brick_and_neighbours_dirty(coord);
        }

        let polarity = polarity.map(|was| {
            let now = self.mask.inverted();
            self.mask.set_inverted(was);
            // **Nothing is marked dirty, deliberately.** Increment 21 wrote a
            // `mark_everything_dirty` here on the reading that polarity is
            // resolved at read, so every brick reads differently. It is
            // resolved at read by the SHADER: the vertex attribute carries the
            // stored byte and `Uniforms::mask_inverted` applies the flip per
            // draw, so a remesh would rebuild every brick to the bytes it
            // already holds. Undoing an Invert has to cost what the Invert cost
            // -- one word to the GPU -- or ctrl+Z after a Mask All is a 475 ms
            // whole-body remesh at the brick count the pool is sized for.
            now
        });

        StrokeEdit::from_parts(inverse, mask_inverse, polarity)
    }

    /// Mark a brick and the twenty-six around it as needing a remesh.
    ///
    /// A brick's apron reads one voxel into each neighbour, so a change inside
    /// one moves triangles that belong to the bricks around it.
    fn mark_brick_and_neighbours_dirty(&mut self, coord: BrickCoord) {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    self.dirty.insert(BrickCoord(coord.0 + IVec3::new(dx, dy, dz)));
                }
            }
        }
    }

    /// Apply `edit` to the protection of every voxel in an inclusive voxel box.
    ///
    /// The mask twin of [`Volume::edit_voxels_where`], and deliberately a far
    /// simpler machine than it. There is no planning phase, no parallel path and
    /// no skipping: a mask brush writes one byte per voxel with no neighbour
    /// reads and no promotion of the field, so the work is memory-bound and
    /// small, and the three mechanisms that earn their keep on the field side
    /// would each need their own proof here for no measured gain. The one thing
    /// it does keep is the rollback of an unpromoted brick -- see
    /// [`crate::mask::MaskField::edit_brick`], where a mask brush painting over
    /// empty space makes that the common case rather than the rim case.
    ///
    /// `edit` receives each voxel, its world position, and the protection that
    /// voxel currently resolves to, and returns the protection it should resolve
    /// to. **Polarity is applied on both sides**, so a caller painting
    /// protection paints protection whichever way Invert is pointing.
    pub fn edit_mask(&mut self, lo: IVec3, hi: IVec3, edit: impl Fn(IVec3, Vec3, u8) -> u8) {
        let voxel_size = self.voxel_size;
        let b_min = BrickCoord::containing(lo).0;
        let b_max = BrickCoord::containing(hi).0;

        // Lifted off `self` for the duration, the same move
        // `edit_voxels_where` makes and for the same reason: the recorder and
        // the dirty set are fields of the same struct as the mask, and the
        // borrow checker will not hold one mutably while the other is written.
        let mut mask = std::mem::take(&mut self.mask);
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let coord = BrickCoord::new(bx, by, bz);
                    let origin = coord.origin();
                    let brick_hi = coord.max_voxel();
                    let from = lo.max(origin);
                    let to = hi.min(brick_hi);
                    if from.cmpgt(to).any() {
                        continue;
                    }
                    let MaskEdit::Changed(prior) =
                        mask.edit_brick(coord, from, to, voxel_size, &edit)
                    else {
                        continue;
                    };
                    // On first touch only, so a stroke that passes over the
                    // same brick fifty times still costs one snapshot of it.
                    if let Some(recorder) = self.mask_recorder.as_mut() {
                        recorder.entry(coord).or_insert(prior);
                    }
                }
            }
        }
        self.mask = mask;
        self.mark_dirty_voxel_range(lo, hi);
    }

    /// Copy the resolved protection over a box into a region, as `0..=255`
    /// floats.
    ///
    /// A [`FieldRegion`] and not a `u8` twin of it, and that is reuse rather
    /// than laziness: the blur a mask brush applies is exactly
    /// [`FieldRegion::neighbour_average`], which already exists, is already
    /// tested against a ramp and a constant, and is already the kernel Smooth
    /// runs on the field. Four bytes per voxel instead of one buys that, on a
    /// box the smoothing brush already snapshots at the same size.
    ///
    /// The region is grown by one voxel on every side by [`Volume::snapshot`]'s
    /// own rule, so the average is available at every voxel of the core box
    /// without a bounds check at the rim.
    pub fn snapshot_mask(&self, lo: IVec3, hi: IVec3, region: &mut FieldRegion) {
        let lo = lo - IVec3::ONE;
        let hi = hi + IVec3::ONE;
        let size = hi - lo + IVec3::ONE;
        let mask = &self.mask;
        let values = region.resize(lo, hi);

        let b_min = BrickCoord::containing(lo).0;
        let b_max = BrickCoord::containing(hi).0;
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let coord = BrickCoord::new(bx, by, bz);
                    // **Resolved once per brick and not once per voxel.** The
                    // slab exists precisely so a hot loop does no map lookups,
                    // and a per-voxel `at` here would be 3.5 million probes on
                    // the box an r80 brush snapshots.
                    let slab = mask.slab(coord);
                    let uniform = slab.fill().map(|byte| mask.resolve(byte) as f32);
                    let origin = coord.origin();
                    let from = lo.max(origin);
                    let to = hi.min(coord.max_voxel());
                    if from.cmpgt(to).any() {
                        continue;
                    }
                    for z in from.z..=to.z {
                        for y in from.y..=to.y {
                            for x in from.x..=to.x {
                                let index =
                                    (x - lo.x) + (y - lo.y) * size.x + (z - lo.z) * size.x * size.y;
                                let local = IVec3::new(x, y, z) - origin;
                                values[index as usize] = match uniform {
                                    Some(value) => value,
                                    None => {
                                        let byte = slab.byte_at(brick_index(
                                            local.x as usize,
                                            local.y as usize,
                                            local.z as usize,
                                        ));
                                        mask.resolve(byte) as f32
                                    }
                                };
                            }
                        }
                    }
                }
            }
        }
    }

    /// Mean resolved protection over this body, as a fraction in `0..=1`.
    ///
    /// **What the standing overlay card's percentage is.** Denominated over the
    /// union of the bricks that hold geometry and the bricks that hold
    /// protection, which is the honest denominator for both directions of the
    /// mistake: counting only mask bricks would report a dab on one brick as
    /// 100%, and counting only field bricks would leave protection painted in
    /// empty space -- exactly what stops Draw growing material there -- out of
    /// the number entirely.
    ///
    /// Costs a 32 KB scan per DENSE mask brick and nothing at all per tile or
    /// per unmasked brick, so it is affordable at the end of a gesture and is
    /// not affordable per frame. That is what
    /// [`crate::mask::MaskField::revision`] is for: the caller caches this and
    /// asks the revision whether the cache is still good.
    pub fn mask_fill(&self) -> f32 {
        let mut bricks = 0u64;
        let mut sum = 0u64;
        for coord in self.bricks.keys().copied() {
            bricks += 1;
            sum += self.mask.protection_sum(coord);
        }
        for coord in self.mask.brick_coords() {
            if self.bricks.contains_key(&coord) {
                continue;
            }
            bricks += 1;
            sum += self.mask.protection_sum(coord);
        }
        if bricks == 0 {
            return 0.0;
        }
        sum as f32 / (bricks as f32 * BRICK_VOXELS as f32 * PROTECTED as f32)
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
        // And the mask, on the same basis. It is part of `resident_bytes`
        // because that is the number the 6 GiB ceiling is checked against, and
        // a generated mask writes a value at every surface voxel -- leaving it
        // out under-reports by up to 25% at the moment the document is largest.
        self.mask.add_to_stats(&mut stats);
        stats
    }

    #[inline]
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// How big the CONTENT is, wherever in the world it sits: half the diagonal
    /// of [`Volume::world_bounds`].
    ///
    /// Derived from the brick extents rather than from the surface, so it costs
    /// a walk of the map's keys instead of a mesh: it is used to size interface
    /// affordances -- how far a mirror plane should reach, what a brush radius
    /// means as a fraction of the model, how far back to put the camera -- and
    /// those need "about how big" rather than a tight bound.
    ///
    /// **This replaced `bounding_radius`, which measured from the world
    /// origin** as `furthest.max(low.abs().max(high.abs()).length())`. That was
    /// indistinguishable from this one while there was a single body seeded at
    /// the origin, and wrong the moment a body sits anywhere else: a 5 mm cube
    /// 100 mm out reported over 100 mm, so the camera framed empty space and a
    /// Dynamic brush came out twenty times too big. It was deleted rather than
    /// kept beside this, because the two differ only in a case that did not
    /// exist yet and the wrong one was the one with the obvious name.
    ///
    /// **`surface_bounds` was refused for this**, though it would be tighter.
    /// It scans every dense brick and its own documentation says it is a
    /// user-action operation that must not run per frame -- and this figure is
    /// refreshed on every remesh, which is to say on every pointer event of
    /// every stroke.
    ///
    /// `None` when the volume is empty, which is the caller's cue to fall back
    /// rather than divide by zero.
    pub fn content_radius(&self) -> Option<f32> {
        let (low, high) = self.world_bounds()?;
        Some((high - low).length() * 0.5)
    }

    /// Iterate the coordinates of every stored brick. Used by tests and by the
    /// initial full mesh after seeding.
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

    /// World space bounds of the SURFACE, or `None` when there is none.
    ///
    /// [`Volume::world_bounds`] answers from brick extents, which round out to
    /// whole 32 voxel bricks and so over-report by up to 8 mm a side at a
    /// 0.25 mm voxel. That is fine for framing a camera and useless for telling
    /// someone how big their print will be, which is what this is for.
    ///
    /// Costs a scan of every dense brick, so it is a user-action operation and
    /// must not be called per frame.
    pub fn surface_bounds(&self) -> Option<(Vec3, Vec3)> {
        let coords: Vec<BrickCoord> = self.brick_coords().collect();
        let found = coords
            .par_iter()
            .filter_map(|coord| {
                // A uniform tile is entirely interior or entirely empty, so it
                // holds no surface and contributes no bound of its own.
                let Some(Brick::Dense(data)) = self.brick(*coord) else {
                    return None;
                };
                let origin = coord.origin();
                let mut lo = IVec3::MAX;
                let mut hi = IVec3::MIN;
                for (index, value) in data.iter().enumerate() {
                    if value.abs() >= NARROW_BAND {
                        continue;
                    }
                    let local = IVec3::new(
                        (index % BRICK_DIM) as i32,
                        ((index / BRICK_DIM) % BRICK_DIM) as i32,
                        (index / (BRICK_DIM * BRICK_DIM)) as i32,
                    );
                    lo = lo.min(origin + local);
                    hi = hi.max(origin + local);
                }
                (lo != IVec3::MAX).then_some((lo, hi))
            })
            .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1)));
        found.map(|(lo, hi)| (self.voxel_position(lo), self.voxel_position(hi)))
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
    use crate::mask::PROTECTED;

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
    ///
    /// `mask` is threaded through because the two paths resolve it in different
    /// places -- the serial one on this thread while it holds the brick, the
    /// parallel one inside the rayon closure -- so a mask that arrived at the
    /// wrong brick would show up here and nowhere else.
    fn both_paths(
        build: impl Fn() -> Volume,
        lo: IVec3,
        hi: IVec3,
        mask: Option<&MaskField>,
        edit: impl Fn(IVec3, Vec3, f32, f32) -> f32 + Sync,
    ) -> (Volume, Volume) {
        let mut one_core = build();
        let mut scratch = EditScratch::default();
        one_core.plan_visits(
            lo,
            hi,
            0,
            mask,
            &|_: &BrickPreview| BrickVerdict::Whole,
            &mut scratch,
        );
        let visits = std::mem::take(&mut scratch.visits);
        one_core.edit_planned_on_one_core(&visits, mask, &edit);
        one_core.mark_dirty_voxel_range(lo, hi);

        let mut many_cores = build();
        many_cores.plan_visits(
            lo,
            hi,
            0,
            mask,
            &|_: &BrickPreview| BrickVerdict::Whole,
            &mut scratch,
        );
        let visits = std::mem::take(&mut scratch.visits);
        let mut taken = Vec::new();
        many_cores.edit_planned_across_cores(&visits, &mut taken, mask, &edit);
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

        let (one_core, many_cores) =
            both_paths(build, lo, hi, None, move |voxel, position, value, _free| {
                edit(voxel, position, value)
            });

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

        let (mut one_core, mut many_cores) =
            both_paths(build, lo, hi, None, move |voxel, position, value, _free| {
                edit(voxel, position, value)
            });

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
                            None,
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
                None,
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
    fn the_plan_counters_account_for_every_brick_of_the_box() {
        // The counters are what the skipping is judged by, so they have to be
        // arithmetic rather than an impression. Every brick of the box lands
        // in exactly one of the three tallies, and the voxels actually visited
        // are a subset of the ones the box covers.
        //
        // The narrowed count is the one worth pinning: a brick narrowed to a
        // slab still counts as one visited brick, so a brick ratio reads a
        // saving of ninety odd percent inside that brick as no saving at all.
        // That is what sent a previous session looking in the wrong place.
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(64.0), 46.0);

        let mut scratch = EditScratch::default();
        volume.plan_visits(
            IVec3::splat(-8),
            IVec3::splat(136),
            2,
            None,
            &|preview: &BrickPreview| match preview.uniform {
                Some(_) => BrickVerdict::OnlyNearDifferentNeighbours,
                None => BrickVerdict::Whole,
            },
            &mut scratch,
        );

        let plan = scratch.plan;
        assert_eq!(
            plan.bricks_skipped + plan.bricks_whole + plan.bricks_narrowed,
            plan.bricks_in_box,
            "the three verdicts do not add up to the box: {plan:?}"
        );
        assert!(plan.bricks_narrowed > 0 && plan.bricks_skipped > 0, "nothing to count: {plan:?}");
        assert!(plan.voxels_visited < plan.voxels_in_box, "nothing was left out: {plan:?}");

        let visited: i64 = scratch
            .visits
            .iter()
            .map(|visit| (visit.hi - visit.lo + IVec3::ONE).as_i64vec3().element_product())
            .sum();
        assert_eq!(visited, plan.voxels_visited, "the voxel count disagrees with the visits");
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

    /// One mask byte per vertex, taken from that vertex's own lattice cell.
    ///
    /// The attribute is what the viewport tints with, so a mesher that sampled
    /// the wrong cell -- or sampled the brick's origin, or the position rather
    /// than the cell -- would put the tint somewhere other than the protection
    /// it is reporting, and every gesture downstream would still behave
    /// correctly. Nothing else in the workspace can see that.
    #[test]
    fn every_vertex_carries_the_stored_mask_byte_of_its_own_lattice_cell() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        // Half the ball, so the meshed bricks straddle the mask's own edge.
        volume.edit_mask(IVec3::splat(-30), IVec3::new(0, 30, 30), |_, _, _| PROTECTED);

        let mut coords: Vec<BrickCoord> = Vec::new();
        volume.take_dirty(&mut coords);
        let mut scratch = MeshScratch::new();
        let mut mesh = BrickMesh::default();

        let mut protected = 0;
        let mut free = 0;
        for coord in &coords {
            volume.mesh_brick(*coord, &mut scratch, &mut mesh);
            assert_eq!(mesh.mask.len(), mesh.vertices.len(), "brick {coord:?} is a byte short");
            for (cell, byte) in mesh.cells.iter().zip(&mesh.mask) {
                assert_eq!(
                    *byte,
                    volume.mask().byte_at(*cell),
                    "brick {coord:?} sampled the wrong cell for {cell:?}"
                );
                match *byte {
                    PROTECTED => protected += 1,
                    UNMASKED => free += 1,
                    _ => {}
                }
            }
        }
        assert!(protected > 0 && free > 0, "the fixture masked all or none of the surface");
    }

    /// **The tint is continuous across a brick seam**, and this is that claim at
    /// the byte where the pixels can only agree or disagree with it.
    ///
    /// Two bricks that both emit a vertex at a shared seam derive its cell from
    /// the same world coordinate, so they look up the same voxel. Sampling by
    /// POSITION instead would split the pair -- the two bricks compute the same
    /// seam vertex at different intermediate magnitudes and the results differ
    /// in the last bits -- and the visible result is a hard line of mismatched
    /// tint every 32 voxels.
    #[test]
    fn two_bricks_agree_about_the_mask_at_every_seam_cell_they_share() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 40.0);
        // A gradient rather than a block, so a wrong cell by one voxel is a
        // wrong byte rather than the same byte from the same flat region.
        volume.edit_mask(IVec3::splat(-50), IVec3::splat(50), |cell, _, _| {
            (cell.x.rem_euclid(255)) as u8
        });

        let mut scratch = MeshScratch::new();
        let mut here = BrickMesh::default();
        let mut next = BrickMesh::default();

        let mut shared = 0;
        for x in -1..=0 {
            let coord = BrickCoord::new(x, 0, 0);
            volume.mesh_brick(coord, &mut scratch, &mut here);
            volume.mesh_brick(BrickCoord::new(x + 1, 0, 0), &mut scratch, &mut next);

            let held: FxHashMap<IVec3, u8> =
                here.cells.iter().copied().zip(here.mask.iter().copied()).collect();
            for (cell, byte) in next.cells.iter().zip(&next.mask) {
                let Some(other) = held.get(cell) else {
                    continue;
                };
                assert_eq!(other, byte, "the two bricks disagree about cell {cell:?}");
                shared += 1;
            }
        }
        assert!(shared > 0, "the fixture found no seam vertex at all, so it proved nothing");
    }

    /// The STORED byte and never the resolved one, which is what makes Invert a
    /// uniform write instead of a remesh of the whole body.
    #[test]
    fn flipping_the_polarity_does_not_change_one_byte_of_the_mesh() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        volume.edit_mask(IVec3::splat(-30), IVec3::new(0, 30, 30), |_, _, _| PROTECTED);

        let coord = BrickCoord::new(-1, 0, 0);
        let mut scratch = MeshScratch::new();
        let mut before = BrickMesh::default();
        volume.mesh_brick(coord, &mut scratch, &mut before);
        assert!(before.mask.contains(&PROTECTED), "the fixture masked nothing");

        volume.mask_mut().set_inverted(true);
        let mut after = BrickMesh::default();
        volume.mesh_brick(coord, &mut scratch, &mut after);
        assert_eq!(before.mask, after.mask, "the polarity was baked into the attribute");
    }

    /// An unmasked body still carries the attribute, as a run of zeros.
    ///
    /// Not an optimisation left on the floor: the pool writes this into a
    /// vertex buffer that the previous tenant of the slice wrote too, so a body
    /// that supplied nothing would be tinted with whatever was there before.
    #[test]
    fn an_unmasked_body_meshes_one_zero_per_vertex() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        let mut coords: Vec<BrickCoord> = Vec::new();
        volume.take_dirty(&mut coords);

        let mut scratch = MeshScratch::new();
        let mut mesh = BrickMesh::default();
        let mut vertices = 0;
        for coord in &coords {
            volume.mesh_brick(*coord, &mut scratch, &mut mesh);
            assert_eq!(mesh.mask.len(), mesh.vertices.len());
            assert!(mesh.mask.iter().all(|byte| *byte == UNMASKED));
            vertices += mesh.vertices.len();
        }
        assert!(vertices > 0, "the fixture meshed nothing");
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

    /// The failure this replaced `bounding_radius` to avoid: the same shape,
    /// moved away from the origin, used to report its DISTANCE instead of its
    /// size, and everything sized off it -- the camera, the mirror plane, the
    /// Dynamic brush -- came out proportionally wrong.
    ///
    /// The tolerance is one brick diagonal because the bound rounds out to
    /// whole 32-voxel bricks, so where the sphere happens to fall against the
    /// lattice can add a brick to a side. It is nowhere near the 100 mm the old
    /// measure would have reported, which is what the second assertion pins.
    #[test]
    fn the_content_radius_measures_the_content_and_not_its_distance_from_the_origin() {
        let mut here = Volume::new(0.5);
        here.seed_sphere(Vec3::ZERO, 5.0);
        let mut far = Volume::new(0.5);
        far.seed_sphere(Vec3::new(100.0, 0.0, 0.0), 5.0);

        let centred = here.content_radius().expect("a seeded sphere has content");
        let displaced = far.content_radius().expect("a seeded sphere has content");
        let brick_diagonal = BRICK_DIM as f32 * 0.5 * 3.0f32.sqrt();
        assert!(
            (centred - displaced).abs() <= brick_diagonal,
            "the same 5 mm ball measured {centred} at the origin and {displaced} at 100 mm"
        );
        assert!(displaced < 50.0, "a 5 mm ball 100 mm out reported a radius of {displaced}");

        let (low, _) = far.world_bounds().expect("a seeded sphere has bounds");
        assert!(low.x > 50.0, "the fixture is not actually off origin");
    }

    #[test]
    fn an_empty_volume_has_no_content_radius() {
        assert_eq!(Volume::new(0.25).content_radius(), None);
    }

    /// **The whole promise of duplicate**: the copy is the same field, and
    /// writing to one does not touch the other.
    ///
    /// Bit-identical rather than approximately equal, and per BRICK rather than
    /// by sampling a handful of points. Sampling would pass on a copy that had
    /// been rebuilt by trilinear resampling, which is exactly the implementation
    /// this one is written to avoid -- and a resample loses a little of the
    /// surface every time it runs, so a duplicate that quietly went through one
    /// would degrade a model a user duplicated a few times with nothing on
    /// screen saying so. Uniform tiles are compared as tiles for the same
    /// reason: a copy that promoted an interior tile to a dense brick would
    /// hold the same values and cost 128 KB apiece.
    #[test]
    fn a_copy_holds_the_same_field_brick_for_brick() {
        // The same fixture as `seeding_a_sphere_allocates_only_the_shell`,
        // because it is the one sized to produce both kinds of brick: the
        // interior spans whole bricks, so there are tiles as well as a shell.
        let mut source = Volume::new(1.0);
        source.seed_sphere(Vec3::ZERO, 60.0);
        let copy = source.duplicated(IVec3::ZERO);

        assert_eq!(copy.voxel_size(), source.voxel_size());
        assert_eq!(copy.brick_count(), source.brick_count(), "the copy holds a different map");
        assert_eq!(copy.stats().dense_bricks, source.stats().dense_bricks);
        assert_eq!(copy.stats().uniform_bricks, source.stats().uniform_bricks);
        assert!(source.stats().uniform_bricks > 0, "the fixture must exercise tiles as well");

        for coord in source.brick_coords() {
            match (source.brick(coord), copy.brick(coord)) {
                (Some(Brick::Uniform(here)), Some(Brick::Uniform(there))) => {
                    assert_eq!(here, there, "the tile at {coord:?} differs");
                }
                (Some(Brick::Dense(here)), Some(Brick::Dense(there))) => {
                    assert_eq!(here, there, "the brick at {coord:?} differs");
                }
                (here, there) => {
                    panic!("the brick at {coord:?} changed kind: {here:?} became {there:?}");
                }
            }
        }
    }

    /// Two bodies, not one body in two rows. The copy owns its own storage, so
    /// a stroke on either leaves the other exactly as it was.
    #[test]
    fn writing_to_a_copy_leaves_the_original_alone() {
        let mut source = Volume::new(0.5);
        source.seed_sphere(Vec3::ZERO, 20.0);
        let before = source.sample_voxel(IVec3::ZERO);

        let mut copy = source.duplicated(IVec3::ZERO);
        copy.edit_box(Vec3::splat(-30.0), Vec3::splat(30.0), |_, _| OUTSIDE);

        assert_eq!(copy.sample_voxel(IVec3::ZERO), OUTSIDE, "the copy was not written to");
        assert_eq!(
            source.sample_voxel(IVec3::ZERO),
            before,
            "writing to the copy reached the original, so the two share storage"
        );
    }

    /// A whole-brick offset moves bricks and changes not one voxel, which is
    /// the property that lets the copy be a memcpy instead of a resample.
    #[test]
    fn an_offset_copy_moves_whole_bricks_and_rewrites_no_voxel() {
        let mut source = Volume::new(0.5);
        source.seed_sphere(Vec3::ZERO, 20.0);
        let offset = IVec3::new(3, -2, 7);
        let copy = source.duplicated(offset);

        assert_eq!(copy.brick_count(), source.brick_count());
        for coord in source.brick_coords() {
            let moved = BrickCoord(coord.0 + offset);
            let here = source.brick(coord).expect("the coordinate came from the source");
            let there = copy.brick(moved).expect("every brick moved by the offset");
            match (here, there) {
                (Brick::Uniform(a), Brick::Uniform(b)) => assert_eq!(a, b),
                (Brick::Dense(a), Brick::Dense(b)) => assert_eq!(a, b),
                _ => panic!("the brick at {coord:?} changed kind on the way to {moved:?}"),
            }
        }
    }

    /// **A copy that never reaches the GPU is the failure this project has
    /// shipped twice**: the row is in the document, the export is right, and
    /// the viewport is empty for the rest of the session.
    ///
    /// The dirty set covers each stored brick's neighbours as well, because a
    /// brick with no voxels of its own still owns the quads on its low faces —
    /// so the count is at least the brick count and not equal to it.
    #[test]
    fn a_copy_arrives_with_every_brick_dirty() {
        let mut source = Volume::new(0.5);
        source.seed_sphere(Vec3::ZERO, 20.0);
        // Drained, so the source is in the state a body is in after its first
        // remesh -- which is the state every body a user can duplicate is in.
        source.take_dirty(&mut Vec::new());
        assert_eq!(source.dirty_count(), 0, "the fixture must start clean");

        let copy = source.duplicated(IVec3::ZERO);
        assert!(
            copy.dirty_count() >= copy.brick_count(),
            "a copy of {} bricks arrived with {} dirty",
            copy.brick_count(),
            copy.dirty_count()
        );
    }

    /// A mask brick costs exactly 32,768 bytes and `resident_bytes` counts
    /// every one of them.
    ///
    /// **Asserted in BYTES rather than as a 1.25x ratio**, and the ratio fails
    /// on two counts: `resident_bytes` carries a map-overhead term that adding
    /// a mask does not scale, and a fully masked volume's mask bricks collapse
    /// to tiles at end of stroke and cost nothing at all, so the ratio there is
    /// 1.0. This is the assertion that stops the 6 GiB guard under-predicting
    /// by a quarter, and it is written before anything can create a mask.
    #[test]
    fn a_dense_mask_costs_exactly_32768_bytes_a_brick_and_resident_bytes_says_so() {
        let coords = [BrickCoord::new(0, 0, 0), BrickCoord::new(1, 0, 0), BrickCoord::new(0, 1, 0)];
        let mut volume = Volume::new(0.5);
        for coord in coords {
            volume.insert_brick(coord, Brick::dense_filled(0.5));
        }
        let before = volume.stats();
        assert_eq!(before.mask_bricks, 0, "an untouched volume reported a mask");
        assert_eq!(before.mask_bytes, 0);

        for coord in coords {
            let origin = coord.origin();
            for z in 0..BRICK_DIM as i32 {
                for y in 0..BRICK_DIM as i32 {
                    for x in 0..BRICK_DIM as i32 {
                        // One feathered voxel per brick, so the brick cannot
                        // collapse to a tile and the dense case is the one
                        // being measured.
                        let value = if (x, y, z) == (0, 0, 0) { 128 } else { crate::PROTECTED };
                        volume.mask_mut().write(origin + IVec3::new(x, y, z), value);
                    }
                }
            }
        }
        volume.mask_mut().collapse();

        let after = volume.stats();
        assert_eq!(after.mask_bricks, coords.len());
        assert_eq!(after.mask_dense_bricks, coords.len(), "a feathered brick collapsed anyway");
        assert_eq!(
            after.mask_bytes,
            32_768 * coords.len(),
            "a dense mask brick is exactly a quarter of a field brick"
        );
        assert_eq!(after.dense_bricks, before.dense_bricks, "the field census moved");
        assert_eq!(
            after.resident_bytes - before.resident_bytes,
            after.mask_bytes + volume.mask().map_bytes(),
            "resident_bytes has to grow by the mask's bytes plus its own map, to the byte"
        );
    }

    /// The mask travels with the body it protects, at the same WORLD point.
    ///
    /// A duplicate that arrived unmasked has no undo and nothing on screen to
    /// say so: the copy looks right and the protection the user painted is
    /// gone.
    #[test]
    fn a_copy_carries_the_mask_to_the_same_world_point() {
        let mut source = Volume::new(0.5);
        source.seed_sphere(Vec3::ZERO, 20.0);
        let cell = IVec3::new(3, 4, 5);
        source.mask_mut().write(cell, crate::PROTECTED);

        let offset = IVec3::new(2, 0, 0);
        let copy = source.duplicated(offset);
        let moved = cell + offset * BRICK_DIM as i32;

        assert_eq!(copy.mask().at(moved), crate::PROTECTED, "the copy lost the mask");
        assert_eq!(copy.mask().at(cell), crate::UNMASKED, "the mask did not move with the body");
        assert_eq!(source.mask().at(cell), crate::PROTECTED, "the original's mask moved");
    }

    /// The mask factor handed to every edit stays inside `0..=1`, over all 256
    /// stored bytes and both polarities.
    ///
    /// The hard constraint the blending brushes rest on. Smooth, flatten and
    /// clay all compute `value + (target - value) * weight` and are stable only
    /// while `weight` is a legal lerp factor: above 1 it extrapolates past the
    /// target, below 0 it extrapolates away from it, and neither shows up as a
    /// panic or a NaN. It shows up as a surface that grows a crust over a
    /// stroke, which is a bug report and not a test failure.
    ///
    /// All 256 bytes and both polarities, because the resolution is one fused
    /// multiply-add over constants that differ per polarity and the ends of the
    /// byte range are exactly where a rounding error would land.
    #[test]
    fn every_mask_factor_stays_inside_zero_to_one() {
        let coord = BrickCoord::new(0, 0, 0);
        let origin = coord.origin();
        // One voxel per protection value, laid along X and wrapping onto the
        // next row, so a single dense brick carries the whole range.
        let cell_of = |value: u8| {
            origin + IVec3::new(value as i32 % BRICK_DIM as i32, value as i32 / BRICK_DIM as i32, 0)
        };

        for inverted in [false, true] {
            let mut mask = MaskField::default();
            mask.set_inverted(inverted);
            for value in 0..=u8::MAX {
                mask.write(cell_of(value), value);
            }

            let freedom = Freedom::resolve(Some(&mask), coord);
            assert!(
                freedom.uniform().is_none(),
                "inverted {inverted}: the fixture has to carry detail, or nothing per voxel is \
                 being measured"
            );

            for value in 0..=u8::MAX {
                let local = cell_of(value) - origin;
                let index = brick_index(local.x as usize, local.y as usize, local.z as usize);
                let free = freedom.at(index);
                assert!(
                    (0.0..=1.0).contains(&free),
                    "inverted {inverted}: protection {value} resolved to a factor of {free}, \
                     which is not a legal lerp factor"
                );
                // The value the user painted really is the protection felt.
                let wanted = (u8::MAX - value) as f32 / u8::MAX as f32;
                assert!(
                    (free - wanted).abs() < 1.0e-6,
                    "inverted {inverted}: protection {value} should leave {wanted} of the brush \
                     and left {free}"
                );
            }

            // The ends exactly, because "close to zero" is a brush that still
            // moves fully protected material and "close to one" is a stroke
            // that never quite reaches its target.
            let free_at = |value: u8| {
                let local = cell_of(value) - origin;
                freedom.at(brick_index(local.x as usize, local.y as usize, local.z as usize))
            };
            assert_eq!(free_at(PROTECTED), 0.0, "inverted {inverted}: fully protected is not zero");
            assert_eq!(free_at(UNMASKED), 1.0, "inverted {inverted}: unmasked is not one");
        }

        // The two hoisted arms, which is what an unmasked body and a collapsed
        // brick actually take.
        assert_eq!(Freedom::OPEN.uniform(), Some(1.0), "an unmasked edit is not fully free");
        assert_eq!(Freedom::OPEN.at(17), 1.0, "an unmasked edit is not fully free per voxel");
        assert_eq!(
            Freedom::resolve(None, coord).uniform(),
            Some(1.0),
            "an edit that passed use_mask false is not fully free"
        );

        // Absent means STORED BYTE 0, never "free". Under inversion -- which is
        // what Mask All is -- an absent brick is fully protected, and an arm
        // that short-circuited it to 1.0 would sculpt at full strength straight
        // through the most-used masking state there is.
        let mut empty = MaskField::default();
        assert_eq!(Freedom::resolve(Some(&empty), coord).uniform(), Some(1.0));
        empty.set_inverted(true);
        assert_eq!(
            Freedom::resolve(Some(&empty), coord).uniform(),
            Some(0.0),
            "an absent brick under inversion was read as free"
        );
    }

    /// Both edit paths resolve the same mask at the same voxel.
    ///
    /// They resolve it in different places -- the serial one on this thread
    /// while it holds the brick, the parallel one inside the rayon closure --
    /// so a slab fetched for the wrong brick shows up here and in no
    /// field-versus-field comparison, because both paths would fetch the same
    /// wrong slab.
    ///
    /// The edit writes the factor it was handed straight into the field, so the
    /// assertion is against the mask itself rather than against the other path.
    #[test]
    fn both_edit_paths_resolve_the_mask_at_the_voxel_it_belongs_to() {
        let lo = IVec3::new(-40, -20, -10);
        let hi = IVec3::new(20, 30, 40);

        for inverted in [false, true] {
            let mut mask = MaskField::default();
            mask.set_inverted(inverted);
            // A gradient that varies along every axis and crosses brick
            // boundaries, so a slab taken from the neighbouring brick or an
            // index computed from the wrong origin lands on a different value.
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let cell = IVec3::new(x, y, z);
                        let value = (x * 7 + y * 13 + z * 29).rem_euclid(256) as u8;
                        mask.write(cell, value);
                    }
                }
            }

            let build = || {
                let mut volume = Volume::new(1.0);
                volume.seed_sphere(Vec3::new(6.0, -3.0, 2.0), 30.0);
                volume.take_dirty(&mut Vec::new());
                volume
            };
            // The factor itself, mapped into the narrow band so the clamp in
            // `write_voxels` cannot swallow it.
            let (one_core, many_cores) =
                both_paths(build, lo, hi, Some(&mask), |_, _, _, free| INSIDE + free);

            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        let cell = IVec3::new(x, y, z);
                        let wanted = INSIDE + (u8::MAX - mask.at(cell)) as f32 / u8::MAX as f32;
                        assert!(
                            (one_core.sample_voxel(cell) - wanted).abs() < 1.0e-6,
                            "inverted {inverted}: the serial path read the wrong mask at \
                             {cell:?}: {} against {wanted}",
                            one_core.sample_voxel(cell)
                        );
                        assert!(
                            (many_cores.sample_voxel(cell) - wanted).abs() < 1.0e-6,
                            "inverted {inverted}: the parallel path read the wrong mask at \
                             {cell:?}: {} against {wanted}",
                            many_cores.sample_voxel(cell)
                        );
                    }
                }
            }
        }
    }

    /// The planner sees a fully protected brick as unskippable-by-mask only
    /// when the protection RESOLVES to 255, not when the stored byte does.
    ///
    /// Reading the stored byte instead would fire the skip on the fully FREE
    /// bricks the moment Invert is on, which is masking inside out.
    #[test]
    fn the_previewed_mask_is_resolved_protection_and_not_the_stored_byte() {
        let coord = BrickCoord::new(0, 0, 0);
        let origin = coord.origin();

        let mut mask = MaskField::default();
        assert_eq!(mask.protection_fill(coord), Some(UNMASKED), "an absent brick is not free");
        mask.set_inverted(true);
        assert_eq!(
            mask.protection_fill(coord),
            Some(PROTECTED),
            "an absent brick under inversion is not fully protected"
        );

        // Fully protected under each polarity, painted and then collapsed the
        // way a stroke leaves it.
        for inverted in [false, true] {
            let mut mask = MaskField::default();
            mask.set_inverted(inverted);
            for z in 0..BRICK_DIM as i32 {
                for y in 0..BRICK_DIM as i32 {
                    for x in 0..BRICK_DIM as i32 {
                        mask.write(origin + IVec3::new(x, y, z), PROTECTED);
                    }
                }
            }
            mask.collapse();
            assert_eq!(
                mask.protection_fill(coord),
                Some(PROTECTED),
                "inverted {inverted}: a fully protected brick did not read as protected"
            );
            // And a brick that carries detail is not uniform, whatever it holds.
            mask.write(origin, 128);
            assert_eq!(
                mask.protection_fill(coord),
                None,
                "inverted {inverted}: a brick with a feathered voxel read as uniform"
            );
        }
    }

    /// Copying a volume mid-stroke must not carry the recording with it. The
    /// prior contents in there name bricks of the body being copied, and
    /// restoring them into the copy would undo the wrong body.
    #[test]
    fn a_copy_taken_mid_stroke_is_not_recording() {
        let mut source = Volume::new(0.5);
        source.seed_sphere(Vec3::ZERO, 20.0);
        source.begin_stroke();
        source.edit_box(Vec3::splat(-5.0), Vec3::splat(5.0), |_, _| INSIDE);
        assert!(source.is_recording(), "the fixture must actually be mid-stroke");

        let mut copy = source.duplicated(IVec3::ZERO);
        assert!(!copy.is_recording(), "the copy inherited the original's recording");
        assert!(copy.end_stroke().is_none(), "the copy handed back an edit it never made");
    }
}
