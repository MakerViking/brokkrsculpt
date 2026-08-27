// SPDX-License-Identifier: AGPL-3.0-only

//! The sculpt mask: eight bits of protection per voxel, held per body.
//!
//! A mask says how much of a brush stroke a voxel is allowed to feel. Zero is
//! free to sculpt, 255 is fully protected, and everything between is a real
//! value rather than a rounding of one -- see "soft, not a bitmask" below.
//!
//! # A second sparse map, inside [`crate::Volume`], never inside [`Brick`]
//!
//! [`MaskField`] mirrors the brick map exactly: absent means unmasked, a
//! [`MaskBrick::Uniform`] tile costs no heap at all, and a dense brick is
//! 32,768 bytes against the field brick's 131,072 -- exactly +25%.
//!
//! **Not inside `Brick`**, and each of these three is fatal on its own.
//! `Volume::record_for_undo` clones the whole brick, so a mask carried inside
//! one would ride every ordinary sculpt stroke's undo entry whether or not the
//! mask changed. `Volume::undo_promotion` removes a brick entirely when no
//! DISTANCE changed, so a stamp that touched a brick without moving the surface
//! would silently delete its mask. And a mask over empty space is exactly what
//! stops Draw growing material there, so the mask has to exist where no field
//! brick does -- inside `Brick` that would mean a 128 KB field allocation per
//! masked empty brick, appearing in the brick census, the memory guard and the
//! save file as geometry that is not there.
//!
//! **Not outside `Volume` either**, and that is forced by two call sites rather
//! than preferred. `Volume::mesh_brick` is the only path from voxels to
//! triangles and takes only `&self`, and threading a mask through it would make
//! an expression that hands one body's mesher another body's mask possible.
//! `Volume::stats` walks its own brick map and nothing else, so a mask held
//! beside the volume is invisible to `resident_bytes` -- which is the number
//! the 6 GiB ceiling is checked against and the number a deleted body reports
//! to the undo allowance. The body owns the volume and the volume owns the
//! mask, which is a reading of "a mask belongs to a body" rather than a literal
//! one, and it is named as such.
//!
//! **[`Brick::is_collapsible`] needs no change at all, and that is the point.**
//! It never sees the mask, so a field brick that is uniform in distance but
//! varied in protection still collapses and still releases its 128 KB, with the
//! mask entry untouched and still correct. If anyone ever moves the mask inside
//! `Brick`, that function has to answer for two arrays and a masked saturated
//! brick can never release its allocation again.
//!
//! # Soft, not a bitmask
//!
//! Eight bits buy two things a single bit cannot. Blur, Mask by Cavity and Mask
//! by Thickness all write a value at every surface voxel, and a hard 0/1 result
//! makes their output unusable: Blender's box and lasso masks write hard values
//! and broke "Set Pivot to Mask Border" for years, because border detection
//! needs values BETWEEN the extremes to find a transition, and the documented
//! workaround was "blur the mask first". The second is the Move brush: a step in
//! the mask is a fold in the geometry, because the warp stops being invertible
//! once the combined gradient reaches one. **Every path that writes the mask
//! writes a feathered edge, never a step.**
//!
//! # Three mechanisms keep it sparse, in order of what they are worth
//!
//! The polarity bool, which makes Mask All, Clear All and Invert All O(1) in
//! time, memory and undo -- without it, one keystroke on a lightly-masked
//! 45,567-brick model allocates 1.04 GiB. Absent-means-unmasked, which is the
//! whole of the hand-painted case and costs a user who never masks nothing at
//! all. And [`MaskBrick::Uniform`], collapsed at end of stroke only.
//!
//! **Absent means STORED BYTE 0, never "free".** Under normal polarity those
//! coincide; under inversion -- which is exactly what Mask All is -- an absent
//! brick is fully protected. Reading absent as free would sculpt at full
//! strength straight through the single most-used masking state.
//!
//! [`Brick`]: crate::Brick
//! [`Brick::is_collapsible`]: crate::Brick::is_collapsible

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::brick::{BRICK_DIM, BRICK_VOXELS, BrickCoord, brick_index};
use crate::orientation::AxisRotation;
use crate::project::MAX_VOLUME_BYTES;
use crate::similarity::Similarity;
use crate::volume::VolumeStats;

/// A voxel a brush may change freely.
pub const UNMASKED: u8 = 0;

/// A voxel no brush may change at all.
pub const PROTECTED: u8 = u8::MAX;

/// How much under an exact fit [`MaskField::would_fit`] suggests.
///
/// Three percent, matching the resample guard and `GrowthGuard` and for the
/// same reason: the estimate is an estimate, and a prediction that lands at
/// exactly 100% of a ceiling helps nobody.
const MARGIN: f32 = 1.03;

/// The width of the padded protection block one filtered brick is built from.
///
/// A 3x3x3 kernel reads one voxel outside the brick on each side, so the block
/// is the brick plus a one-voxel apron -- the same reason
/// [`crate::Volume::mesh_brick`] has one, arrived at independently.
const PADDED_DIM: usize = BRICK_DIM + 2;

/// Voxels in that padded block: 39,304 bytes, built once per dense brick.
const PADDED_VOXELS: usize = PADDED_DIM * PADDED_DIM * PADDED_DIM;

/// One of the four whole-mask filters.
///
/// **`max`, `min` and a 3x3x3 box over the SOFT mask, and deliberately not
/// [`crate::cavity`]'s operators**, which dilate and erode a bitmask: a
/// protection value is eight bits and every one of these has to keep the
/// intermediate levels, because a step in the mask is a fold in the geometry
/// under Move and because a hard 0/1 result is what broke Blender's mask border
/// detection for years. See [`crate::mask`]'s "soft, not a bitmask".
///
/// Each is applied ABSOLUTELY, at an amount in `0..=1`, against the mask as it
/// stood when the gesture began -- never accumulated from the last frame. That
/// is ZBrush's top masking complaint answered: its BlurMask is "press
/// repeatedly for progressively more blur" with a strength that varies with the
/// subtool, and Maxon's own later fix is documented as "absolute rather than
/// accumulative".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskFilter {
    /// The mean of the 27 voxels around each one. Softens an edge.
    Blur,
    /// The value pushed away from that mean. **Not blur's inverse**, and Maxon
    /// concedes the same of its own pair: the round trip "will not normally
    /// give you exactly the original mask". A blur discards what it averaged
    /// away and no later operation can put it back.
    Sharpen,
    /// The greatest protection in the 27. Spreads a mask outward by a voxel.
    Grow,
    /// The least. Pulls it back in by a voxel.
    ///
    /// **Grow then Shrink does not round-trip, and that is morphology and not
    /// a bug**: the pair is a closing, so a gap narrower than two voxels is
    /// bridged by the Grow and the Shrink cannot reopen it. Two protected
    /// blobs two voxels apart come back as one.
    Shrink,
}

impl MaskFilter {
    pub const ALL: [MaskFilter; 4] =
        [MaskFilter::Blur, MaskFilter::Sharpen, MaskFilter::Grow, MaskFilter::Shrink];

    pub fn label(self) -> &'static str {
        match self {
            MaskFilter::Blur => "Blur",
            MaskFilter::Sharpen => "Sharpen",
            MaskFilter::Grow => "Grow",
            MaskFilter::Shrink => "Shrink",
        }
    }

    /// The verb as it reads inside a refusal, which is a lower-case sentence.
    pub fn verb(self) -> &'static str {
        match self {
            MaskFilter::Blur => "blur",
            MaskFilter::Sharpen => "sharpen",
            MaskFilter::Grow => "grow",
            MaskFilter::Shrink => "shrink",
        }
    }

    /// The verb in the past tense, for the status line after it has run.
    ///
    /// Spelled out rather than suffixed, because two of the four are irregular
    /// and "growred the mask" is the kind of thing that ships.
    pub fn done(self) -> &'static str {
        match self {
            MaskFilter::Blur => "blurred",
            MaskFilter::Sharpen => "sharpened",
            MaskFilter::Grow => "grew",
            MaskFilter::Shrink => "shrank",
        }
    }
}

/// The protection values of one brick.
///
/// Mirrors [`crate::Brick`] one for one, including the allocate-on-the-heap
/// discipline: 32 KB is small enough to build on the stack and large enough
/// that doing it in a rayon worker is not worth the argument.
#[derive(Debug, Clone)]
pub enum MaskBrick {
    /// Every voxel holds this byte. Costs no allocation.
    Uniform(u8),
    /// Per voxel stored bytes, indexed by [`brick_index`].
    Dense(Box<[u8; BRICK_VOXELS]>),
}

impl MaskBrick {
    /// A dense brick with every voxel set to `value`.
    pub(crate) fn dense_filled(value: u8) -> Self {
        let data: Box<[u8]> = vec![value; BRICK_VOXELS].into_boxed_slice();
        let data: Box<[u8; BRICK_VOXELS]> = data.try_into().expect("length is BRICK_VOXELS");
        MaskBrick::Dense(data)
    }

    /// Promote a uniform brick so individual voxels can be written.
    pub(crate) fn make_dense(&mut self) -> &mut [u8; BRICK_VOXELS] {
        if let MaskBrick::Uniform(value) = *self {
            *self = MaskBrick::dense_filled(value);
        }
        match self {
            MaskBrick::Dense(data) => data,
            MaskBrick::Uniform(_) => unreachable!("just promoted to dense"),
        }
    }

    /// Heap bytes this brick holds, excluding the enum itself.
    #[inline]
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            MaskBrick::Uniform(_) => 0,
            MaskBrick::Dense(_) => BRICK_VOXELS,
        }
    }

    /// The single byte every voxel holds, or `None` when it carries detail.
    ///
    /// **Deliberately looser than [`crate::Brick::is_collapsible`], which
    /// collapses only saturated bricks.** That rule is an anti-thrash heuristic
    /// for a surface-adjacent brick about to be rewritten by the next stamp of
    /// the same stroke; moving the mask's collapse to end of stroke removes the
    /// thrash it guards against, so a mid-range uniform mask brick is worth
    /// collapsing here where a mid-band field brick is not.
    ///
    /// `pub(crate)` for [`crate::generate`], which builds whole bricks outside
    /// this module and has to collapse them itself: a generated mask that
    /// stored a dense block of one repeated byte would cost 32 KB a brick for a
    /// tile, and the half-space recipe produces those by the thousand.
    pub(crate) fn is_collapsible(&self) -> Option<u8> {
        match self {
            MaskBrick::Uniform(value) => Some(*value),
            MaskBrick::Dense(data) => {
                let first = data[0];
                data.iter().all(|byte| *byte == first).then_some(first)
            }
        }
    }
}

/// One body's mask.
///
/// See the module documentation for where this lives and why. Reachable only
/// THROUGH the volume that owns it, via [`crate::Volume::mask`] and
/// [`crate::Volume::mask_mut`]: no expression can hand the mesher, the
/// serialiser or the brush the wrong body's. Those two accessors are `pub`
/// rather than `pub(crate)` -- see their own documentation for what that gave
/// up and why it was not the load-bearing half.
#[derive(Debug, Default)]
pub struct MaskField {
    bricks: FxHashMap<BrickCoord, MaskBrick>,
    /// Resolved at READ, never materialised. This one bool is what makes
    /// Invert and Mask All O(1) in time, memory and undo.
    inverted: bool,
    /// How many times this mask has been changed, counting up and never down.
    ///
    /// **The whole of what makes the standing overlay card affordable.** That
    /// card shows a percentage of protection, `view()` runs at display rate, and
    /// summing protection over a 45,567-brick model is not something a frame may
    /// do -- so the caller caches the number and asks this whether the cache is
    /// still good. A `u64` counter compares in one instruction where the answer
    /// it stands in for is a gigabyte of reads.
    ///
    /// Monotonic **per lineage**: every transform carries `self.revision + 1`
    /// into the mask it builds, so a resample or a rotation cannot hand a body
    /// a fresh mask wearing a stamp its cache has already seen. A mask that
    /// starts life at zero belongs to a body that did not exist before, and so
    /// is keyed under an id no cache holds.
    revision: u64,
}

/// One brick's mask, resolved for the duration of a stamp so that the voxel
/// loop does no map lookups.
///
/// The three arms are what let the hot loop hoist: [`MaskSlab::Free`] and
/// [`MaskSlab::Uniform`] are loop-invariant, only [`MaskSlab::Dense`] is per
/// voxel.
#[derive(Debug, Clone, Copy)]
pub enum MaskSlab<'a> {
    /// The body has no mask at all AND polarity is normal. Only then -- this is
    /// the arm a caller may short-circuit on, and reaching it under inversion
    /// would sculpt straight through a Mask All.
    Free,
    /// No entry for this brick, or a collapsed one. Carries the STORED byte,
    /// which is 0 for an absent brick and not the resolved protection.
    Uniform(u8),
    Dense(&'a [u8; BRICK_VOXELS]),
}

impl MaskSlab<'_> {
    /// The stored byte, before polarity.
    #[inline]
    pub(crate) fn byte_at(&self, index: usize) -> u8 {
        match self {
            MaskSlab::Free => UNMASKED,
            MaskSlab::Uniform(byte) => *byte,
            MaskSlab::Dense(data) => data[index],
        }
    }

    /// The one stored byte the whole brick holds, or `None` when it carries
    /// detail. This is the test a hot loop hoists on.
    #[inline]
    pub(crate) fn fill(&self) -> Option<u8> {
        match self {
            MaskSlab::Free => Some(UNMASKED),
            MaskSlab::Uniform(byte) => Some(*byte),
            MaskSlab::Dense(_) => None,
        }
    }
}

/// What one call to [`MaskField::edit_brick`] did.
///
/// Two states rather than a `bool` plus an out-parameter, because the two
/// answers carry different things: a brick that changed owes the stroke
/// recorder its prior contents, and one that did not owes it nothing at all and
/// must not be recorded -- an entry for an unchanged brick is 32 KB of undo
/// budget spent on putting back what is already there.
pub(crate) enum MaskEdit {
    /// Nothing inside the box moved. The entry is exactly as it was found.
    Unchanged,
    /// It changed, and this is what the brick held before -- `None` for a brick
    /// that had no entry, which undo puts back by removing the entry again.
    Changed(Option<MaskBrick>),
}

impl MaskField {
    /// Polarity applied, in both directions.
    ///
    /// One function and not two, because the map from stored byte to protection
    /// is its own inverse: `255 - (255 - b) == b`. That is why [`MaskField::at`]
    /// and [`MaskField::write`] round-trip under either polarity, and why Invert
    /// stays a bool rather than a pass over the map.
    ///
    /// `pub(crate)` so that a caller who has already resolved a whole brick's
    /// [`MaskSlab`] can turn its stored bytes into protection without going back
    /// through the map for each one.
    #[inline]
    pub(crate) fn resolve(&self, byte: u8) -> u8 {
        if self.inverted { PROTECTED - byte } else { byte }
    }

    /// Whether this mask resolves to "nothing is protected" everywhere.
    ///
    /// The condition for [`MaskSlab::Free`], and both halves are load-bearing:
    /// an empty map under inversion is fully protected, not free.
    ///
    /// `pub` because it is also the condition the standing overlay card is
    /// shown on, and that question is asked from `brokkr-app`.
    #[inline]
    pub fn is_free(&self) -> bool {
        self.bricks.is_empty() && !self.inverted
    }

    /// How many times this mask has changed. See the field's own documentation.
    #[inline]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Force this mask's revision past `floor`.
    ///
    /// **What [`crate::Volume::replace_mask`] uses to make "same body, same
    /// revision" mean "same mask" for a whole gesture, and it is not
    /// belt-and-braces.** An absolute filter re-applies from ONE snapshot, so
    /// every field it builds carries `snapshot.revision + 1` -- the same number
    /// for a blur at 0.3 and a blur at 0.9. The standing card is keyed on that
    /// number, so without this the card would show the percentage of the first
    /// step of a drag for the whole of it.
    ///
    /// A bump is always safe in the other direction too: it can only cost a
    /// cache miss, where a collision costs a wrong number on screen.
    #[inline]
    pub(crate) fn advance_revision_past(&mut self, floor: u64) {
        self.revision = self.revision.max(floor) + 1;
    }

    /// Record that something in here changed.
    ///
    /// Every write goes through this rather than touching `revision` inline, so
    /// a new mutator cannot be added without the cache being told about it.
    #[inline]
    fn touch(&mut self) {
        self.revision += 1;
    }

    /// Whether protection is read inverted.
    #[inline]
    pub fn inverted(&self) -> bool {
        self.inverted
    }

    /// Flip or set the polarity.
    ///
    /// O(1) whatever the mask holds, which is the whole reason the bool exists;
    /// see the module documentation.
    #[inline]
    pub fn set_inverted(&mut self, inverted: bool) {
        if self.inverted != inverted {
            self.touch();
        }
        self.inverted = inverted;
    }

    /// One brick's mask, resolved once so the voxel loop can hoist.
    pub(crate) fn slab(&self, coord: BrickCoord) -> MaskSlab<'_> {
        if self.is_free() {
            return MaskSlab::Free;
        }
        match self.bricks.get(&coord) {
            None => MaskSlab::Uniform(UNMASKED),
            Some(MaskBrick::Uniform(byte)) => MaskSlab::Uniform(*byte),
            Some(MaskBrick::Dense(data)) => MaskSlab::Dense(data),
        }
    }

    /// The STORED byte, polarity NOT applied.
    ///
    /// Private to the serialiser and the mesh-attribute writer -- the only two
    /// callers that legitimately want it, because the file and the vertex
    /// buffer both carry the polarity separately. Everything else reads
    /// [`MaskField::at`].
    pub(crate) fn byte_at(&self, cell: IVec3) -> u8 {
        let coord = BrickCoord::containing(cell);
        let local = cell - coord.origin();
        self.slab(coord).byte_at(brick_index(local.x as usize, local.y as usize, local.z as usize))
    }

    /// Protection in 0..=255 with polarity applied. Everything else reads this.
    #[inline]
    pub fn at(&self, cell: IVec3) -> u8 {
        self.resolve(self.byte_at(cell))
    }

    /// Append the STORED byte at each of `cells`, which mostly belong to
    /// `coord`.
    ///
    /// The mesh-attribute writer's call, and a method here rather than a loop
    /// over [`MaskField::byte_at`] at the call site because of what it costs:
    /// one brick meshes to around 1100 vertices, and a map lookup each is a
    /// per-vertex hash on the remesh path that a stroke has 8 ms for. Every
    /// vertex of a brick comes from that brick or from the one-voxel border
    /// around it -- [`crate::BrickMesh::cells`] is derived from the apron -- so
    /// the lookup hoists for the great majority of them and only the border
    /// falls back.
    ///
    /// **Stored bytes, polarity NOT applied**, for the reason
    /// [`MaskField::byte_at`] gives: the vertex buffer carries the polarity
    /// separately, as a uniform the shader resolves.
    pub(crate) fn bytes_at_cells(&self, coord: BrickCoord, cells: &[IVec3], out: &mut Vec<u8>) {
        if self.bricks.is_empty() {
            // Whatever the polarity: an inverted mask with an empty map is
            // stored zeros that the shader inverts, not stored 255s.
            out.resize(out.len() + cells.len(), UNMASKED);
            return;
        }

        let slab = self.slab(coord);
        let origin = coord.origin();
        let dim = BRICK_DIM as i32;
        out.reserve(cells.len());
        for cell in cells {
            let local = *cell - origin;
            let byte = if local.min_element() >= 0 && local.max_element() < dim {
                slab.byte_at(brick_index(local.x as usize, local.y as usize, local.z as usize))
            } else {
                self.byte_at(*cell)
            };
            out.push(byte);
        }
    }

    /// Write a protection value, polarity applied.
    ///
    /// `value` is what [`MaskField::at`] will hand back, so a caller painting
    /// protection paints protection whichever way the polarity is pointing.
    ///
    /// **A caller must write a FEATHERED value and never a step.** That is not
    /// checkable here -- one voxel cannot see its neighbours' gradient -- and it
    /// is a rule rather than a preference for three independent reasons, given
    /// in the module documentation.
    pub fn write(&mut self, cell: IVec3, value: u8) {
        let stored = self.resolve(value);
        let coord = BrickCoord::containing(cell);
        let local = cell - coord.origin();
        let index = brick_index(local.x as usize, local.y as usize, local.z as usize);
        self.touch();
        match self.bricks.get_mut(&coord) {
            // Writing what the tile already holds must not promote it: a stroke
            // that grazes a fully protected brick would otherwise turn 0 heap
            // bytes into 32 KB for no change at all.
            Some(MaskBrick::Uniform(held)) if *held == stored => {}
            Some(brick) => brick.make_dense()[index] = stored,
            // An absent brick already stores 0 everywhere, so writing 0 into one
            // is the difference between a mask that costs nothing and one that
            // allocates a brick per voxel the brush passed over.
            None if stored == UNMASKED => {}
            None => {
                let mut brick = MaskBrick::dense_filled(UNMASKED);
                brick.make_dense()[index] = stored;
                self.bricks.insert(coord, brick);
            }
        }
    }

    /// One brick's stored entry, for the stroke recorder.
    ///
    /// The prior contents an undo entry has to put back, taken BEFORE
    /// [`MaskField::edit_brick`] promotes anything. `None` is a brick with no
    /// entry, which stores 0 everywhere -- and restoring that means removing
    /// the entry again, not writing zeroes into one.
    #[inline]
    pub(crate) fn brick(&self, coord: BrickCoord) -> Option<&MaskBrick> {
        self.bricks.get(&coord)
    }

    /// Put one brick's stored entry back exactly as it was.
    ///
    /// The undo side of [`MaskField::edit_brick`], and the writer the project
    /// reader uses too -- those two are the only callers that hand over a whole
    /// brick, and they want the same thing: bytes that were stored somewhere
    /// else, put back untouched. Polarity is NOT applied: an entry recorded
    /// from this map is stored bytes, and the polarity it was read under is
    /// carried separately -- by the undo entry in one case and by the file's
    /// `mask_flags` in the other.
    pub(crate) fn restore_brick(&mut self, coord: BrickCoord, brick: Option<MaskBrick>) {
        self.touch();
        match brick {
            Some(brick) => self.bricks.insert(coord, brick),
            None => self.bricks.remove(&coord),
        };
    }

    /// Rewrite the protection of one brick over the part of it inside a box.
    ///
    /// `edit` is handed each voxel, its world position and the protection it
    /// currently resolves to, and returns the protection it should resolve to.
    /// Polarity is applied on both sides, so a painter paints protection
    /// whichever way it is pointing -- the same guarantee [`MaskField::write`]
    /// gives, at a brick's worth of voxels for one map lookup instead of one
    /// lookup per voxel.
    ///
    /// **A brick the box clipped but never changed is left exactly as it was**,
    /// promotion included. Without that rollback a stroke's bounding cube turns
    /// the rim of empty bricks it grazes into 32 KB allocations and undo entries
    /// for bricks nothing touched -- which is the same trap
    /// [`crate::Volume::undo_promotion`] exists for on the field side, and it
    /// bites harder here because a mask brush is normally painting over empty
    /// space.
    pub(crate) fn edit_brick(
        &mut self,
        coord: BrickCoord,
        lo: IVec3,
        hi: IVec3,
        voxel_size: f32,
        edit: &impl Fn(IVec3, Vec3, u8) -> u8,
    ) -> MaskEdit {
        let origin = coord.origin();
        let existed = self.bricks.contains_key(&coord);
        let prior = self.bricks.get(&coord).cloned();
        let brick = self.bricks.entry(coord).or_insert(MaskBrick::Uniform(UNMASKED));
        let was_uniform = matches!(brick, MaskBrick::Uniform(_));
        let uniform_byte = match brick {
            MaskBrick::Uniform(byte) => *byte,
            MaskBrick::Dense(_) => UNMASKED,
        };
        let inverted = self.inverted;
        let resolve = |byte: u8| if inverted { PROTECTED - byte } else { byte };

        let data = brick.make_dense();
        let mut changed = false;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let voxel = IVec3::new(x, y, z);
                    let local = voxel - origin;
                    let index = brick_index(local.x as usize, local.y as usize, local.z as usize);
                    let position = voxel.as_vec3() * voxel_size;
                    let stored = resolve(edit(voxel, position, resolve(data[index])));
                    if stored != data[index] {
                        data[index] = stored;
                        changed = true;
                    }
                }
            }
        }

        if changed {
            self.touch();
            return MaskEdit::Changed(prior);
        }
        // Nothing moved, so the promotion above is undone rather than left to
        // `collapse` at the end of the stroke: between here and there the brick
        // is 32 KB of a value it already had, and a large brush grazes hundreds
        // of them per stamp.
        if was_uniform {
            if existed {
                self.bricks.insert(coord, MaskBrick::Uniform(uniform_byte));
            } else {
                self.bricks.remove(&coord);
            }
        }
        MaskEdit::Unchanged
    }

    /// Sum of the resolved protection over one brick, in units of one voxel.
    ///
    /// `BRICK_VOXELS * 255` is a fully protected brick. Answered without a voxel
    /// loop for a tile or an absent brick, which after [`MaskField::collapse`]
    /// is most of them, and that is what makes the overlay card's percentage
    /// affordable at all.
    pub(crate) fn protection_sum(&self, coord: BrickCoord) -> u64 {
        match self.protection_fill(coord) {
            Some(byte) => byte as u64 * BRICK_VOXELS as u64,
            None => match self.bricks.get(&coord) {
                Some(MaskBrick::Dense(data)) => {
                    data.iter().map(|byte| self.resolve(*byte) as u64).sum()
                }
                // `protection_fill` answered `None`, which only a dense brick
                // does, so this is unreachable -- and answering zero rather
                // than panicking keeps a percentage out of the crash path.
                _ => 0,
            },
        }
    }

    /// Every brick this mask holds an entry for.
    ///
    /// `pub` for the same reason [`crate::Volume::mask`] is, and with the same
    /// half of the claim surviving: it is reachable only THROUGH the volume
    /// that owns it, so no expression can enumerate one body's mask against
    /// another's. What it buys outside this crate is a test that can compare
    /// two masks byte for byte instead of sampling the cells it happened to
    /// think of -- which is exactly the difference between catching an undo
    /// that restores most of a mask and not catching it.
    pub fn brick_coords(&self) -> impl Iterator<Item = BrickCoord> + '_ {
        self.bricks.keys().copied()
    }

    /// Collapse uniform bricks and drop the ones that hold nothing.
    ///
    /// Called from [`crate::Volume::end_stroke`] only, which is what makes the
    /// looser collapse rule in [`MaskBrick::is_collapsible`] safe.
    ///
    /// **Only stored 0 is pruned.** The colour argument -- that a saturated
    /// brick's paint has no surface to sit on -- does not transfer: a protection
    /// value over solid interior is still read by Blur across the brick boundary
    /// and still decides whether Draw may grow material into that space.
    ///
    /// Costs a scan of every dense mask brick, which is nothing while the mask
    /// is empty and is the reason the stroke recorder will narrow this to the
    /// bricks one stroke actually touched.
    ///
    /// **Deliberately does NOT [`MaskField::touch`].** Every voxel reads back
    /// exactly what it read back before -- a tile of `b` and a dense brick of
    /// `b` are the same mask -- so a cache keyed on the revision is still
    /// correct afterwards, and bumping it here would throw away the percentage
    /// at the end of every single stroke including the ones that changed
    /// nothing.
    pub fn collapse(&mut self) {
        self.bricks.retain(|_, brick| match brick.is_collapsible() {
            Some(UNMASKED) => false,
            Some(byte) => {
                *brick = MaskBrick::Uniform(byte);
                true
            }
            None => true,
        });
    }

    /// Why `extra` bytes of mask would not fit under the memory ceiling, and
    /// the coarser voxel size that would.
    ///
    /// `None` is the answer that means "go ahead". The message is a fragment in
    /// the same shape `too_fine_for_the_pool` and [`crate::GrowthGuard`] return,
    /// so the caller supplies the verb and this supplies the numbers.
    ///
    /// # The two parameters the plan did not have
    ///
    /// `resident_bytes` is the whole DOCUMENT's, not this body's: the ceiling is
    /// document-wide, and a mask that fits beside its own body can still be the
    /// thing that pushes a five-body document over. `voxel_size` is the
    /// document's too. **A `MaskField` can derive neither** -- it holds bricks
    /// and a bool -- and the suggested size is a square law over the current one
    /// (a surface has fixed area, so coarsening the voxel by `k` divides the
    /// whole shell, mask included, by `k * k`), so it cannot be produced from
    /// `&self` alone. Recorded as a deviation from the four-signature API rather
    /// than worked around.
    pub fn would_fit(
        &self,
        extra: usize,
        resident_bytes: usize,
        voxel_size: f32,
    ) -> Option<(String, f32)> {
        let wanted = resident_bytes as f64 + extra as f64;
        if wanted <= MAX_VOLUME_BYTES {
            return None;
        }
        let gigabyte = 1024.0 * 1024.0 * 1024.0;
        let coarser = voxel_size * (wanted / MAX_VOLUME_BYTES).sqrt() as f32 * MARGIN;
        let why = format!(
            "it needs about {:.1} GB of memory against a {:.0} GB ceiling -- {coarser:.3} mm is \
             the finest voxel that fits",
            wanted / gigabyte,
            MAX_VOLUME_BYTES / gigabyte,
        );
        Some((why, coarser))
    }

    // -------------------------------------------------- the whole-mask verbs

    /// Whether every voxel of this body resolves to [`PROTECTED`].
    ///
    /// True of an empty map read inverted and of nothing else, which is exactly
    /// what Mask All produces -- so the verb can decline to push a second
    /// history entry for a state the document is already in.
    #[inline]
    pub fn protects_everything(&self) -> bool {
        self.bricks.is_empty() && self.inverted
    }

    /// An empty mask at `inverted`, carrying this one's lineage.
    ///
    /// **Clear and Mask All are both this**, which is why neither is a walk:
    /// Clear is `cleared(false)`, Mask All is `cleared(true)`, and the map that
    /// was there is MOVED into the history entry rather than rewritten. An
    /// [`FxHashMap::default`] allocates nothing, so both verbs cost one bool
    /// and one move however large the mask was.
    ///
    /// The revision is carried forward for the reason the field gives: a cache
    /// keyed on `(body, revision)` must never see a different mask wearing a
    /// stamp it has already answered for.
    pub fn cleared(&self, inverted: bool) -> MaskField {
        MaskField { bricks: FxHashMap::default(), inverted, revision: self.revision + 1 }
    }

    /// A mask holding exactly these bricks, at NORMAL polarity, carrying this
    /// one's lineage.
    ///
    /// What [`crate::generate`] hands back. Normal polarity is the whole of the
    /// deviation from [`MaskField::cleared`] and it is deliberate: a generator
    /// computes protection directly, so inheriting an inversion would read
    /// every byte it just wrote back upside down -- a cavity mask applied on
    /// top of Mask All would protect precisely the flat ground it was asked to
    /// leave free. Invert afterwards if that was the intent; it is still one
    /// bool and no bricks.
    ///
    /// Entries that hold [`UNMASKED`] everywhere are dropped rather than
    /// stored, so a recipe that finds nothing produces a mask that
    /// [`MaskField::is_free`] accepts and no overlay card appears for.
    pub(crate) fn generated(&self, bricks: Vec<(BrickCoord, MaskBrick)>) -> MaskField {
        let mut out = MaskField {
            bricks: FxHashMap::default(),
            inverted: false,
            revision: self.revision + 1,
        };
        out.bricks.reserve(bricks.len());
        for (coord, brick) in bricks {
            if matches!(brick.is_collapsible(), Some(UNMASKED)) {
                continue;
            }
            out.bricks.insert(coord, brick);
        }
        out
    }

    /// Bytes this mask holds, on the same basis [`MaskField::add_to_stats`]
    /// charges them to `resident_bytes`.
    ///
    /// **What a global filter has to predict before it runs.** A filter needs
    /// the old mask while it writes the new one, and history then holds the old
    /// one, so the peak is `field + 2 x mask` -- and `resident_bytes` already
    /// counts one of those two. This is the other. See the caller's guard and
    /// [`MaskField::would_fit`].
    pub fn bytes(&self) -> usize {
        self.bricks.values().map(MaskBrick::heap_bytes).sum::<usize>() + self.map_bytes()
    }

    /// This mask with a 3x3x3 filter blended in at `amount`.
    ///
    /// `amount` is a fraction of the way from the value a voxel holds to the
    /// value the kernel produces, clamped to `0..=1`, so the whole family stays
    /// inside the range the brush weight rule requires and every result is a
    /// blend toward a legal target rather than an extrapolation away from one.
    /// At `0.0` the answer is this mask again; at `1.0` it is one clean pass of
    /// the kernel. **The caller re-applies it from an unchanged snapshot on
    /// every change of the amount**, which is what makes the gesture absolute.
    ///
    /// # Cost, and the one thing that makes it affordable
    ///
    /// A destination brick whose whole 3x3x3 source neighbourhood holds one
    /// stored byte is answered from the brick structure with no voxel loop at
    /// all, because every one of these four filters is the identity on a
    /// constant. After [`MaskField::collapse`] that is most of a painted mask
    /// and all of a Mask All, so what is actually walked is the SHELL where
    /// protection changes -- which is a surface, not a volume. The bricks that
    /// do get walked pay 27 reads a voxel over a padded copy.
    ///
    /// The destination set is every brick this mask holds an entry for AND the
    /// twenty-six around each, because all four filters can move protection one
    /// voxel across a brick boundary -- Grow and Blur outward under normal
    /// polarity, Shrink and Sharpen outward under inversion, where an absent
    /// brick resolves to [`PROTECTED`] rather than to nothing.
    pub fn filtered(&self, filter: MaskFilter, amount: f32) -> MaskField {
        let amount = amount.clamp(0.0, 1.0);
        let mut out = self.cleared(self.inverted);
        // An empty map is a constant field under either polarity -- stored 0
        // everywhere, resolving to 0 or to 255 -- and all four filters are the
        // identity on a constant. Nothing to walk and nothing to allocate.
        if self.bricks.is_empty() {
            return out;
        }

        let mut wanted: FxHashSet<BrickCoord> = FxHashSet::default();
        wanted.reserve(self.bricks.len() * 2);
        for coord in self.bricks.keys() {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        wanted.insert(BrickCoord(coord.0 + IVec3::new(dx, dy, dz)));
                    }
                }
            }
        }

        let coords: Vec<BrickCoord> = wanted.into_iter().collect();
        let built: Vec<(BrickCoord, MaskBrick)> = coords
            .par_iter()
            // One padded block per WORKER and not one per brick. It is 39,304
            // bytes and the shell of a large mask is thousands of bricks, so
            // allocating it inside the closure would be thousands of 39 KB
            // allocations per step of a slider drag.
            .map_init(
                || vec![UNMASKED; PADDED_VOXELS],
                |padded, coord| {
                    let (brick, _took_the_shortcut) =
                        self.filter_brick(padded, *coord, filter, amount);
                    brick.map(|brick| (*coord, brick))
                },
            )
            .flatten()
            .collect();

        out.bricks.reserve(built.len());
        for (coord, brick) in built {
            out.bricks.insert(coord, brick);
        }
        out
    }

    /// One destination brick of [`MaskField::filtered`], or `None` when it comes
    /// out unmasked and so needs no entry at all -- and, beside it, whether the
    /// uniform shortcut answered it instead of the voxel loop.
    ///
    /// **The bool is the only way anything can observe the shortcut, and that
    /// is why it is returned rather than inferred.** The shortcut produces
    /// bit-for-bit the same brick the slow path would (a dense block of one
    /// repeated byte collapses straight back to [`MaskBrick::Uniform`]), so a
    /// test that looks at the value cannot tell which path ran -- and the whole
    /// affordability argument for a global filter is that the slow path runs
    /// over the shell and not the volume. `filtered` drops it; the test named
    /// for the shortcut counts it. A test-only predicate that re-asked
    /// `uniform_over` would not do: deleting the shortcut from here would leave
    /// it green, which is the failure this exists to make impossible.
    ///
    /// `padded` is scratch, reused across the bricks one worker handles. Every
    /// one of its voxels is written by the gather below before anything reads
    /// it -- the 3x3x3 block of source bricks covers the padded box exactly --
    /// so it is deliberately not cleared between bricks.
    fn filter_brick(
        &self,
        padded: &mut [u8],
        coord: BrickCoord,
        filter: MaskFilter,
        amount: f32,
    ) -> (Option<MaskBrick>, bool) {
        let origin = coord.origin();
        let low = origin - IVec3::ONE;
        let high = coord.max_voxel() + IVec3::ONE;

        // The cheap case, and after a collapse it is most of them. Polarity is
        // uniform over the whole field, so one stored byte everywhere is one
        // protection value everywhere, and all four filters return it unchanged.
        if let Some(byte) = self.uniform_over(low, high) {
            return ((byte != UNMASKED).then_some(MaskBrick::Uniform(byte)), true);
        }

        // Resolved protection over the brick and its one-voxel apron, gathered
        // once so the kernel below does no map lookups at all.
        let b_low = BrickCoord::containing(low).0;
        let b_high = BrickCoord::containing(high).0;
        for bz in b_low.z..=b_high.z {
            for by in b_low.y..=b_high.y {
                for bx in b_low.x..=b_high.x {
                    let source = BrickCoord::new(bx, by, bz);
                    let slab = self.slab(source);
                    let uniform = slab.fill().map(|byte| self.resolve(byte));
                    let source_origin = source.origin();
                    let from = low.max(source_origin);
                    let to = high.min(source.max_voxel());
                    if from.cmpgt(to).any() {
                        continue;
                    }
                    for z in from.z..=to.z {
                        for y in from.y..=to.y {
                            for x in from.x..=to.x {
                                let at = IVec3::new(x, y, z);
                                let local = at - source_origin;
                                let protection = match uniform {
                                    Some(value) => value,
                                    None => self.resolve(slab.byte_at(brick_index(
                                        local.x as usize,
                                        local.y as usize,
                                        local.z as usize,
                                    ))),
                                };
                                let into = at - low;
                                padded[padded_index(
                                    into.x as usize,
                                    into.y as usize,
                                    into.z as usize,
                                )] = protection;
                            }
                        }
                    }
                }
            }
        }

        let mut brick = MaskBrick::dense_filled(UNMASKED);
        let data = brick.make_dense();
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let held = padded[padded_index(x + 1, y + 1, z + 1)];
                    let target = kernel(padded, filter, x + 1, y + 1, z + 1);
                    let blended = held as f32 + (target - held as f32) * amount;
                    let protection = blended.round().clamp(0.0, PROTECTED as f32) as u8;
                    data[brick_index(x, y, z)] = self.resolve(protection);
                }
            }
        }
        let answer = match brick.is_collapsible() {
            Some(UNMASKED) => None,
            Some(byte) => Some(MaskBrick::Uniform(byte)),
            None => Some(brick),
        };
        (answer, false)
    }

    // -------------------------------------------------------------- accounting

    /// Bytes the brick map itself costs, on the same basis
    /// [`crate::Volume::stats`] charges its own map.
    #[inline]
    pub(crate) fn map_bytes(&self) -> usize {
        self.bricks.capacity() * (size_of::<BrickCoord>() + size_of::<MaskBrick>())
    }

    /// Add this mask's census and its bytes to a volume's.
    ///
    /// **`resident_bytes` has to include the mask**, because that number is
    /// what the 6 GiB ceiling is checked against by a square law and what a
    /// removed body charges to the undo allowance. A generated mask writes a
    /// value at every surface voxel, so leaving it out makes both under-report
    /// by up to 25% at exactly the moment the document is largest.
    pub(crate) fn add_to_stats(&self, stats: &mut VolumeStats) {
        let mut bytes = 0;
        for brick in self.bricks.values() {
            stats.mask_bricks += 1;
            if matches!(brick, MaskBrick::Dense(_)) {
                stats.mask_dense_bricks += 1;
            }
            bytes += brick.heap_bytes();
        }
        stats.mask_bytes += bytes;
        stats.resident_bytes += bytes + self.map_bytes();
    }

    // -------------------------------------------------------------- transforms

    /// A deep copy, with every brick moved by `offset_bricks`.
    ///
    /// The mask half of [`crate::Volume::duplicated`], and whole bricks for the
    /// same reason: a translation by whole bricks moves no voxel within its
    /// brick, so each one is a `clone` and a tile stays a tile.
    pub(crate) fn translated(&self, offset_bricks: IVec3) -> MaskField {
        let mut copy = MaskField {
            bricks: FxHashMap::default(),
            inverted: self.inverted,
            // Carried forward rather than reset, so a resample or a rotation
            // cannot hand a body a fresh mask wearing a stamp its cache has
            // already seen. See the field.
            revision: self.revision + 1,
        };
        copy.bricks.reserve(self.bricks.len());
        for (coord, brick) in &self.bricks {
            copy.bricks.insert(BrickCoord(coord.0 + offset_bricks), brick.clone());
        }
        copy
    }

    /// This mask turned by `rotation`.
    ///
    /// Exact, and for the same reason the field's rotation is: a quarter turn
    /// maps the lattice onto itself, so a brick turns into a brick and a tile
    /// costs nothing at all.
    pub(crate) fn rotated(&self, rotation: AxisRotation) -> MaskField {
        let map = rotation.axis_map();
        let last = BRICK_DIM - 1;
        let mut turned = MaskField {
            bricks: FxHashMap::default(),
            inverted: self.inverted,
            // Carried forward rather than reset, so a resample or a rotation
            // cannot hand a body a fresh mask wearing a stamp its cache has
            // already seen. See the field.
            revision: self.revision + 1,
        };
        turned.bricks.reserve(self.bricks.len());
        for (coord, brick) in &self.bricks {
            let destination = BrickCoord(rotation.apply_voxel(coord.0));
            let moved = match brick {
                MaskBrick::Uniform(byte) => MaskBrick::Uniform(*byte),
                MaskBrick::Dense(data) => {
                    let mut moved = MaskBrick::dense_filled(UNMASKED);
                    let out = moved.make_dense();
                    for z in 0..BRICK_DIM {
                        for y in 0..BRICK_DIM {
                            for x in 0..BRICK_DIM {
                                let from = [x, y, z];
                                let mut to = [0usize; 3];
                                for (axis, (into, flipped)) in map.into_iter().enumerate() {
                                    to[into] = if flipped { last - from[axis] } else { from[axis] };
                                }
                                out[brick_index(to[0], to[1], to[2])] = data[brick_index(x, y, z)];
                            }
                        }
                    }
                    moved
                }
            };
            turned.bricks.insert(destination, moved);
        }
        turned
    }

    /// This mask on a different lattice, at the same world positions.
    ///
    /// Nearest neighbour, where the field is trilinear, and that is deliberate:
    /// interpolating protection would soften a mask edge every time the detail
    /// button is pressed, and the value being carried is an authored intent
    /// rather than a measurement. Coarsening still loses feathering, because
    /// there are fewer voxels to hold it in.
    ///
    /// Cost is proportional to what the MASK covers rather than to the model: a
    /// destination brick whose whole source footprint holds one byte is answered
    /// from the brick structure without a voxel loop, which after
    /// [`MaskField::collapse`] is the ordinary case.
    pub(crate) fn resampled(&self, from_voxel: f32, to_voxel: f32) -> MaskField {
        let mut out = MaskField {
            bricks: FxHashMap::default(),
            inverted: self.inverted,
            // Carried forward rather than reset, so a resample or a rotation
            // cannot hand a body a fresh mask wearing a stamp its cache has
            // already seen. See the field.
            revision: self.revision + 1,
        };
        if self.bricks.is_empty() {
            return out;
        }

        // Destination bricks any source mask brick reaches. Built from the mask
        // alone: a mask over empty space has no field brick to be found through.
        let ratio = from_voxel / to_voxel;
        let dim = BRICK_DIM as i32;
        let mut wanted: FxHashSet<BrickCoord> = FxHashSet::default();
        for coord in self.bricks.keys() {
            let low = (coord.origin().as_vec3() * ratio).floor().as_ivec3() - IVec3::ONE;
            let high = (coord.max_voxel().as_vec3() * ratio).ceil().as_ivec3() + IVec3::ONE;
            let b_low = BrickCoord::containing(low).0;
            let b_high = BrickCoord::containing(high).0;
            for bz in b_low.z..=b_high.z {
                for by in b_low.y..=b_high.y {
                    for bx in b_low.x..=b_high.x {
                        wanted.insert(BrickCoord::new(bx, by, bz));
                    }
                }
            }
        }

        let inverse = to_voxel / from_voxel;
        let coords: Vec<BrickCoord> = wanted.into_iter().collect();
        let built: Vec<(BrickCoord, MaskBrick)> = coords
            .par_iter()
            .filter_map(|coord| {
                let origin = coord.origin();
                let source_low = source_cell(origin, inverse);
                let source_high = source_cell(origin + IVec3::splat(dim - 1), inverse);

                // The cheap case, and after a collapse it is most of them.
                if let Some(byte) = self.uniform_over(source_low, source_high) {
                    return (byte != UNMASKED).then_some((*coord, MaskBrick::Uniform(byte)));
                }

                let mut brick = MaskBrick::dense_filled(UNMASKED);
                let data = brick.make_dense();
                // One map lookup per run of destination voxels that share a
                // source brick, rather than one per voxel: at any refinement
                // ratio that is at least a 32-fold saving along X.
                let mut cached: Option<(BrickCoord, MaskSlab<'_>)> = None;
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let cell = source_cell(
                                origin + IVec3::new(x as i32, y as i32, z as i32),
                                inverse,
                            );
                            let source = BrickCoord::containing(cell);
                            let slab = match cached {
                                Some((held, slab)) if held == source => slab,
                                _ => {
                                    let slab = self.slab(source);
                                    cached = Some((source, slab));
                                    slab
                                }
                            };
                            let local = cell - source.origin();
                            data[brick_index(x, y, z)] = slab.byte_at(brick_index(
                                local.x as usize,
                                local.y as usize,
                                local.z as usize,
                            ));
                        }
                    }
                }
                match brick.is_collapsible() {
                    Some(UNMASKED) => None,
                    Some(byte) => Some((*coord, MaskBrick::Uniform(byte))),
                    None => Some((*coord, brick)),
                }
            })
            .collect();

        out.bricks.reserve(built.len());
        for (coord, brick) in built {
            out.bricks.insert(coord, brick);
        }
        out
    }

    /// This mask moved by whole VOXELS, sub-brick offsets included.
    ///
    /// The mask half of [`crate::Volume::shifted`], and value-exact for the
    /// same reason: every destination cell takes exactly one source cell's
    /// byte. [`MaskField::translated`] moves whole bricks and cannot express a
    /// sub-brick offset at all; this one gathers, and delegates to it when the
    /// offset happens to be brick aligned so that the common case still moves
    /// `Box` pointers rather than bytes.
    ///
    /// Nearest neighbour is not a choice here, it is the absence of one: at
    /// whole-voxel granularity there is nothing between two cells to
    /// interpolate.
    pub(crate) fn shifted(&self, offset_voxels: IVec3) -> MaskField {
        let dim = BRICK_DIM as i32;
        if offset_voxels.rem_euclid(IVec3::splat(dim)) == IVec3::ZERO {
            return self.translated(offset_voxels / dim);
        }

        let mut out = MaskField {
            bricks: FxHashMap::default(),
            inverted: self.inverted,
            // Carried forward rather than reset, for the reason
            // [`MaskField::translated`] gives.
            revision: self.revision + 1,
        };
        if self.bricks.is_empty() {
            return out;
        }

        // Every destination brick any source brick reaches. A source brick
        // shifted by a sub-brick offset straddles up to eight of them.
        let mut wanted: FxHashSet<BrickCoord> = FxHashSet::default();
        for coord in self.bricks.keys() {
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
        let built: Vec<(BrickCoord, MaskBrick)> = coords
            .par_iter()
            .filter_map(|coord| {
                let source_low = coord.origin() - offset_voxels;
                let source_high = coord.max_voxel() - offset_voxels;
                if let Some(byte) = self.uniform_over(source_low, source_high) {
                    return (byte != UNMASKED).then_some((*coord, MaskBrick::Uniform(byte)));
                }

                let mut brick = MaskBrick::dense_filled(UNMASKED);
                let data = brick.make_dense();
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let cell = source_low + IVec3::new(x as i32, y as i32, z as i32);
                            data[brick_index(x, y, z)] = self.byte_at(cell);
                        }
                    }
                }
                match brick.is_collapsible() {
                    Some(UNMASKED) => None,
                    Some(byte) => Some((*coord, MaskBrick::Uniform(byte))),
                    None => Some((*coord, brick)),
                }
            })
            .collect();

        out.bricks.reserve(built.len());
        for (coord, brick) in built {
            out.bricks.insert(coord, brick);
        }
        out
    }

    /// This mask rebuilt through a similarity, onto the SAME lattice.
    ///
    /// Nearest neighbour, for the reason [`MaskField::resampled`] gives:
    /// protection is an authored intent rather than a measurement, and
    /// interpolating it would soften a painted edge on every pass.
    ///
    /// The destination footprint is found by pushing each source brick's own
    /// box FORWARD through the map, rather than by walking the field's bounds.
    /// Protection over empty space is real -- it is what stops Draw growing
    /// material there -- so a mask brick with no field brick under it has to be
    /// carried, and it would be invisible to a walk of the field.
    pub(crate) fn warped(&self, by: Similarity, voxel_size: f32) -> MaskField {
        let mut out = MaskField {
            bricks: FxHashMap::default(),
            inverted: self.inverted,
            revision: self.revision + 1,
        };
        if self.bricks.is_empty() {
            return out;
        }

        let mut wanted: FxHashSet<BrickCoord> = FxHashSet::default();
        for coord in self.bricks.keys() {
            let source_low = coord.origin().as_vec3() * voxel_size;
            let source_high = coord.max_voxel().as_vec3() * voxel_size;
            // The forward image of a box, by the same eight-corner argument
            // [`Similarity::inverse_bounds`] makes for the backward one --
            // which is what `inverse().inverse_bounds(..)` says.
            let (low, high) = by.inverse().inverse_bounds(source_low, source_high);
            let low = BrickCoord::containing((low / voxel_size).floor().as_ivec3() - IVec3::ONE).0;
            let high = BrickCoord::containing((high / voxel_size).ceil().as_ivec3() + IVec3::ONE).0;
            for bz in low.z..=high.z {
                for byy in low.y..=high.y {
                    for bx in low.x..=high.x {
                        wanted.insert(BrickCoord::new(bx, byy, bz));
                    }
                }
            }
        }

        let inverse = by.inverse();
        let coords: Vec<BrickCoord> = wanted.into_iter().collect();
        let built: Vec<(BrickCoord, MaskBrick)> = coords
            .par_iter()
            .filter_map(|coord| {
                let origin = coord.origin();
                let mut brick = MaskBrick::dense_filled(UNMASKED);
                let data = brick.make_dense();
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let world = (origin + IVec3::new(x as i32, y as i32, z as i32))
                                .as_vec3()
                                * voxel_size;
                            let source = inverse.transform_point(world) / voxel_size;
                            data[brick_index(x, y, z)] = self.byte_at(source.round().as_ivec3());
                        }
                    }
                }
                match brick.is_collapsible() {
                    Some(UNMASKED) => None,
                    Some(byte) => Some((*coord, MaskBrick::Uniform(byte))),
                    None => Some((*coord, brick)),
                }
            })
            .collect();

        out.bricks.reserve(built.len());
        for (coord, brick) in built {
            out.bricks.insert(coord, brick);
        }
        out
    }

    /// The one stored byte covering an inclusive source cell box, or `None` when
    /// it carries detail.
    ///
    /// Absent bricks count as stored 0, which is what they read as.
    fn uniform_over(&self, low: IVec3, high: IVec3) -> Option<u8> {
        let b_low = BrickCoord::containing(low).0;
        let b_high = BrickCoord::containing(high).0;
        let mut found: Option<u8> = None;
        for bz in b_low.z..=b_high.z {
            for by in b_low.y..=b_high.y {
                for bx in b_low.x..=b_high.x {
                    let byte = match self.bricks.get(&BrickCoord::new(bx, by, bz)) {
                        None => UNMASKED,
                        Some(MaskBrick::Uniform(byte)) => *byte,
                        Some(MaskBrick::Dense(_)) => return None,
                    };
                    match found {
                        None => found = Some(byte),
                        Some(held) if held == byte => {}
                        Some(_) => return None,
                    }
                }
            }
        }
        found
    }

    /// Take the greater protection of the two, everywhere the other one has
    /// something to say.
    ///
    /// **`max` and not the field's `min`**, and the two rules do not contradict
    /// each other: the union of two solids is the lower distance, and the union
    /// of two protections is the higher one, because a merge must not be a way
    /// to unprotect what either body protected. Resolved protection on both
    /// sides and stored in this field's polarity, so merging an inverted mask
    /// into a normal one means what it says.
    ///
    /// `also` names bricks the other body's mask has no entry for but which
    /// still carry protection -- which under inversion is every brick it holds
    /// geometry in. Absent is stored 0, so under normal polarity there is
    /// nothing there to merge and the caller passes nothing.
    pub(crate) fn union_max_from(
        &mut self,
        other: &MaskField,
        also: impl Iterator<Item = BrickCoord>,
    ) {
        let coords: FxHashSet<BrickCoord> = if other.inverted {
            other.bricks.keys().copied().chain(also).collect()
        } else {
            other.bricks.keys().copied().collect()
        };
        for coord in coords {
            let incoming = other.slab(coord);
            // Both sides uniform answers without allocating anything, which is
            // what keeps merging two fully masked bodies free.
            if let (Some(source), Some(target)) = (incoming.fill(), self.fill(coord)) {
                let stored = self.resolve(other.resolve(source).max(self.resolve(target)));
                if stored == target {
                    continue;
                }
                if stored == UNMASKED {
                    self.bricks.remove(&coord);
                } else {
                    self.bricks.insert(coord, MaskBrick::Uniform(stored));
                }
                continue;
            }

            // An absent target brick stores 0 everywhere, which is what a
            // uniform tile of 0 is, so there is no separate arm for it.
            let mut brick = self.bricks.remove(&coord).unwrap_or(MaskBrick::Uniform(UNMASKED));
            let data = brick.make_dense();
            for (index, held) in data.iter_mut().enumerate() {
                let winner = other.resolve(incoming.byte_at(index)).max(self.resolve(*held));
                *held = self.resolve(winner);
            }
            self.bricks.insert(coord, brick);
        }
    }

    /// The one stored byte a brick holds, or `None` when it carries detail.
    ///
    /// An absent brick is stored 0, which is what it reads as.
    #[inline]
    fn fill(&self, coord: BrickCoord) -> Option<u8> {
        match self.bricks.get(&coord) {
            None => Some(UNMASKED),
            Some(MaskBrick::Uniform(byte)) => Some(*byte),
            Some(MaskBrick::Dense(_)) => None,
        }
    }

    /// The one PROTECTION value a brick resolves to everywhere, or `None` when
    /// it carries detail.
    ///
    /// Polarity applied, which is the whole difference between this and
    /// [`MaskField::fill`] and the reason both exist. The planner skips a brick
    /// at a resolved [`PROTECTED`], and under inversion that is a stored 0 or
    /// an absent brick -- so a planner reading the stored byte would skip the
    /// fully FREE bricks the moment Invert is on.
    #[inline]
    pub(crate) fn protection_fill(&self, coord: BrickCoord) -> Option<u8> {
        self.fill(coord).map(|byte| self.resolve(byte))
    }
}

/// The source cell a destination voxel reads from, nearest neighbour.
#[inline]
fn source_cell(destination: IVec3, inverse_ratio: f32) -> IVec3 {
    (destination.as_vec3() * inverse_ratio).round().as_ivec3()
}

/// Index into the padded protection block [`MaskField::filter_brick`] builds.
///
/// Its own function rather than [`brick_index`] with a different stride,
/// because the two differ only in that stride and mixing them up would read
/// the right number of bytes from the wrong places -- a filter that looks
/// nearly right and smears diagonally.
#[inline]
fn padded_index(x: usize, y: usize, z: usize) -> usize {
    x + y * PADDED_DIM + z * PADDED_DIM * PADDED_DIM
}

/// What one filter makes of the 27 protection values around a voxel.
///
/// Coordinates are into the padded block, so the caller passes the CENTRE and
/// every neighbour is in range by construction -- there is no bounds test in
/// here and there must not be one, because this runs 27 times per voxel over
/// 32,768 voxels a brick.
#[inline]
fn kernel(padded: &[u8], filter: MaskFilter, x: usize, y: usize, z: usize) -> f32 {
    let mut sum = 0u32;
    let mut lowest = PROTECTED;
    let mut highest = UNMASKED;
    for dz in 0..3 {
        for dy in 0..3 {
            for dx in 0..3 {
                let value = padded[padded_index(x + dx - 1, y + dy - 1, z + dz - 1)];
                sum += value as u32;
                lowest = lowest.min(value);
                highest = highest.max(value);
            }
        }
    }
    let mean = sum as f32 / 27.0;
    match filter {
        MaskFilter::Blur => mean,
        // The value pushed as far the other side of its neighbourhood mean as
        // it currently sits this side of it: the unsharp mask, clamped by the
        // caller. On a settled region it is the identity, which is what makes
        // repeated gestures converge rather than ring.
        MaskFilter::Sharpen => {
            let held = padded[padded_index(x, y, z)] as f32;
            (held + (held - mean)).clamp(UNMASKED as f32, PROTECTED as f32)
        }
        MaskFilter::Grow => highest as f32,
        MaskFilter::Shrink => lowest as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(x: i32, y: i32, z: i32) -> IVec3 {
        IVec3::new(x, y, z)
    }

    #[test]
    fn an_untouched_mask_is_free_and_costs_nothing() {
        let mask = MaskField::default();
        assert!(mask.is_free(), "an untouched mask has to hoist out of the voxel loop");
        assert_eq!(mask.at(cell(0, 0, 0)), UNMASKED);
        assert!(matches!(mask.slab(BrickCoord::new(0, 0, 0)), MaskSlab::Free));
    }

    #[test]
    fn writing_a_value_reads_it_back_at_the_same_cell_and_nowhere_else() {
        let mut mask = MaskField::default();
        mask.write(cell(5, 6, 7), 200);
        assert_eq!(mask.at(cell(5, 6, 7)), 200);
        assert_eq!(mask.at(cell(5, 6, 8)), UNMASKED);
        assert_eq!(mask.at(cell(-5, 6, 7)), UNMASKED, "a negative cell landed in the wrong brick");
    }

    /// The rule the whole polarity design rests on: an absent brick stores 0,
    /// which is unmasked normally and FULLY PROTECTED once inverted. Reading it
    /// as "free" would sculpt straight through a Mask All.
    #[test]
    fn an_absent_brick_is_fully_protected_under_inversion_and_free_without_it() {
        let mut mask = MaskField::default();
        assert_eq!(mask.at(cell(1000, 1000, 1000)), UNMASKED);
        mask.set_inverted(true);
        assert_eq!(mask.at(cell(1000, 1000, 1000)), PROTECTED);
        assert!(!mask.is_free(), "an inverted empty mask protects everything and cannot hoist");
        assert_eq!(mask.byte_at(cell(1000, 1000, 1000)), UNMASKED, "the STORED byte is still 0");
    }

    #[test]
    fn a_value_written_under_inversion_reads_back_as_the_value_written() {
        // `at` and `write` are both in protection space, so painting protection
        // paints protection whichever way the polarity points.
        let mut mask = MaskField::default();
        mask.set_inverted(true);
        mask.write(cell(2, 2, 2), 40);
        assert_eq!(mask.at(cell(2, 2, 2)), 40);
        assert_eq!(mask.byte_at(cell(2, 2, 2)), PROTECTED - 40, "polarity was not applied");
    }

    #[test]
    fn writing_nothing_into_empty_space_allocates_nothing() {
        let mut mask = MaskField::default();
        mask.write(cell(9, 9, 9), UNMASKED);
        let mut stats = VolumeStats::default();
        mask.add_to_stats(&mut stats);
        assert_eq!(stats.mask_bricks, 0, "an unmasked write allocated a brick");
    }

    #[test]
    fn a_uniform_mask_brick_costs_no_heap_and_a_dense_one_costs_exactly_32768() {
        assert_eq!(MaskBrick::Uniform(PROTECTED).heap_bytes(), 0);
        assert_eq!(MaskBrick::dense_filled(1).heap_bytes(), 32_768);
    }

    #[test]
    fn a_fully_masked_brick_collapses_to_a_tile_and_costs_no_heap() {
        let mut mask = MaskField::default();
        for z in 0..BRICK_DIM as i32 {
            for y in 0..BRICK_DIM as i32 {
                for x in 0..BRICK_DIM as i32 {
                    mask.write(cell(x, y, z), PROTECTED);
                }
            }
        }
        let mut before = VolumeStats::default();
        mask.add_to_stats(&mut before);
        assert_eq!(before.mask_dense_bricks, 1);
        assert_eq!(before.mask_bytes, 32_768);

        mask.collapse();
        let mut after = VolumeStats::default();
        mask.add_to_stats(&mut after);
        assert_eq!(after.mask_bricks, 1, "the brick still has to protect its voxels");
        assert_eq!(after.mask_dense_bricks, 0);
        assert_eq!(after.mask_bytes, 0, "a collapsed tile has to cost no heap at all");
        assert_eq!(mask.at(cell(4, 4, 4)), PROTECTED, "collapsing lost the value");
    }

    #[test]
    fn a_brick_that_ends_a_stroke_unmasked_is_dropped_entirely() {
        let mut mask = MaskField::default();
        mask.write(cell(1, 1, 1), 60);
        mask.write(cell(1, 1, 1), UNMASKED);
        mask.collapse();
        let mut stats = VolumeStats::default();
        mask.add_to_stats(&mut stats);
        assert_eq!(stats.mask_bricks, 0, "a brick protecting nothing was kept");
        assert!(mask.is_free(), "and the mask should be free again");
    }

    #[test]
    fn a_mid_range_mask_brick_collapses_where_a_mid_band_field_brick_would_not() {
        // Deliberately looser than `Brick::is_collapsible`; see its doc comment.
        let mut mask = MaskField::default();
        for z in 0..BRICK_DIM as i32 {
            for y in 0..BRICK_DIM as i32 {
                for x in 0..BRICK_DIM as i32 {
                    mask.write(cell(x, y, z), 128);
                }
            }
        }
        mask.collapse();
        let mut stats = VolumeStats::default();
        mask.add_to_stats(&mut stats);
        assert_eq!(stats.mask_dense_bricks, 0, "a uniform mid-range brick kept its 32 KB");
        assert_eq!(mask.at(cell(0, 0, 0)), 128);
    }

    #[test]
    fn writing_the_value_a_tile_already_holds_does_not_promote_it() {
        let mut mask = MaskField::default();
        for z in 0..BRICK_DIM as i32 {
            for y in 0..BRICK_DIM as i32 {
                for x in 0..BRICK_DIM as i32 {
                    mask.write(cell(x, y, z), PROTECTED);
                }
            }
        }
        mask.collapse();
        mask.write(cell(3, 3, 3), PROTECTED);
        let mut stats = VolumeStats::default();
        mask.add_to_stats(&mut stats);
        assert_eq!(stats.mask_bytes, 0, "grazing a protected tile re-allocated it");
    }

    #[test]
    fn a_mask_that_fits_the_ceiling_says_nothing() {
        let mask = MaskField::default();
        assert_eq!(mask.would_fit(1024, 1024 * 1024, 0.25), None);
    }

    #[test]
    fn a_mask_over_the_ceiling_names_a_coarser_voxel_that_would_fit() {
        let mask = MaskField::default();
        // Four times the ceiling, so the square law wants twice the voxel size.
        let resident = (MAX_VOLUME_BYTES * 4.0) as usize;
        let (why, coarser) =
            mask.would_fit(0, resident, 0.25).expect("four times the ceiling has to refuse");
        assert!(why.contains("24.0 GB"), "the message should name the cost: {why}");
        assert!(why.contains("0.515 mm"), "the message should name a coarser voxel: {why}");
        assert!((coarser - 0.515).abs() < 0.001, "the suggested size was {coarser}");
    }

    #[test]
    fn a_translated_mask_keeps_its_values_and_its_polarity() {
        let mut mask = MaskField::default();
        mask.set_inverted(true);
        mask.write(cell(1, 2, 3), 90);
        let moved = mask.translated(IVec3::new(1, 0, 0));
        assert_eq!(moved.at(cell(1 + BRICK_DIM as i32, 2, 3)), 90);
        assert!(moved.inverted(), "a copy lost its polarity");
        assert_eq!(
            moved.at(cell(500, 500, 500)),
            PROTECTED,
            "the copy of an inverted mask has to protect the space it says nothing about"
        );
    }

    #[test]
    fn merging_takes_the_greater_protection_of_the_two() {
        let mut into = MaskField::default();
        into.write(cell(1, 1, 1), 200);
        into.write(cell(2, 2, 2), 10);
        let mut from = MaskField::default();
        from.write(cell(1, 1, 1), 30);
        from.write(cell(2, 2, 2), 250);

        into.union_max_from(&from, std::iter::empty());
        assert_eq!(into.at(cell(1, 1, 1)), 200, "the merge unprotected a protected voxel");
        assert_eq!(into.at(cell(2, 2, 2)), 250);
    }

    #[test]
    fn merging_an_inverted_mask_into_a_normal_one_merges_what_it_resolves_to() {
        let mut into = MaskField::default();
        into.write(cell(1, 1, 1), 10);
        let mut from = MaskField::default();
        from.set_inverted(true);
        // Stored 0 everywhere, which under inversion is fully protected.
        from.write(cell(1, 1, 1), PROTECTED);

        into.union_max_from(&from, std::iter::once(BrickCoord::new(0, 0, 0)));
        assert_eq!(into.at(cell(1, 1, 1)), PROTECTED);
        assert!(!into.inverted(), "the target's polarity is not the source's to change");
    }

    /// The mesh-attribute writer's hoisted lookup must answer exactly what the
    /// per-cell one does, including for the border cells it does not hoist.
    ///
    /// A vertex of a brick can come from the one-voxel border around it, and
    /// those fall off the fast path. Getting the bounds test wrong by one is a
    /// wrong byte on exactly the vertices that sit on a brick seam, which is
    /// the one place a wrong byte is visible as a hard line.
    #[test]
    fn the_hoisted_lookup_agrees_with_the_per_cell_one_inside_the_brick_and_over_its_border() {
        let mut mask = MaskField::default();
        let dim = BRICK_DIM as i32;
        // Values in three neighbouring bricks, so the border really differs.
        for (at, value) in [(cell(0, 0, 0), 40), (cell(dim, 5, 5), 120), (cell(-1, 5, 5), 200)] {
            mask.write(at, value);
        }

        let coord = BrickCoord::new(0, 0, 0);
        let cells: Vec<IVec3> = vec![
            cell(0, 0, 0),
            cell(5, 5, 5),
            cell(dim - 1, dim - 1, dim - 1),
            // Outside the brick on each side, which is the border a vertex can
            // legitimately come from.
            cell(-1, 5, 5),
            cell(dim, 5, 5),
            cell(5, -1, 5),
            cell(5, 5, dim),
        ];

        let mut out = Vec::new();
        mask.bytes_at_cells(coord, &cells, &mut out);
        let expected: Vec<u8> = cells.iter().map(|at| mask.byte_at(*at)).collect();
        assert_eq!(out, expected);
        assert!(
            out.contains(&200) && out.contains(&120),
            "the fixture's border values are missing"
        );
    }

    /// An empty map is a run of zeros whatever the polarity, because the
    /// polarity is the shader's to apply and not the mesher's.
    #[test]
    fn an_inverted_empty_mask_still_meshes_as_stored_zeros() {
        let mut mask = MaskField::default();
        mask.set_inverted(true);
        let cells = vec![cell(0, 0, 0), cell(9, 9, 9)];
        let mut out = Vec::new();
        mask.bytes_at_cells(BrickCoord::new(0, 0, 0), &cells, &mut out);
        assert_eq!(out, vec![UNMASKED, UNMASKED]);
        assert_eq!(mask.at(cell(0, 0, 0)), PROTECTED, "the fixture is not really inverted");
    }

    // ------------------------------------------------- the whole-mask verbs

    /// A block of protection, so a filter has an edge to work on.
    fn blob(low: IVec3, high: IVec3, value: u8) -> MaskField {
        let mut mask = MaskField::default();
        for z in low.z..=high.z {
            for y in low.y..=high.y {
                for x in low.x..=high.x {
                    mask.write(cell(x, y, z), value);
                }
            }
        }
        mask
    }

    /// Every stored byte, in a stable order, so two masks can be compared bit
    /// for bit rather than by the sample the test happened to think of.
    fn fingerprint(mask: &MaskField) -> Vec<(BrickCoord, Vec<u8>)> {
        let mut coords: Vec<BrickCoord> = mask.brick_coords().collect();
        coords.sort_by_key(|coord| (coord.0.x, coord.0.y, coord.0.z));
        coords
            .into_iter()
            .map(|coord| {
                let slab = mask.slab(coord);
                let bytes = (0..BRICK_VOXELS).map(|index| slab.byte_at(index)).collect();
                (coord, bytes)
            })
            .collect()
    }

    /// **Invert is a bool and nothing else.** Twice round has to be the mask
    /// that went in, byte for byte, or ctrl+I is a slow corruption rather than
    /// an O(1) view change.
    #[test]
    fn inverting_twice_is_bit_identical() {
        let mut mask = blob(cell(2, 2, 2), cell(9, 9, 9), 180);
        mask.collapse();
        let before = fingerprint(&mask);
        let revision = mask.revision();

        mask.set_inverted(true);
        mask.set_inverted(false);

        assert_eq!(fingerprint(&mask), before, "two flips did not come back to the same bytes");
        assert!(!mask.inverted());
        assert!(mask.revision() > revision, "a flip has to move the card's cache key");
    }

    /// Clear and Mask All are the same move with a different bool, and neither
    /// allocates: the map that comes off is handed over whole.
    #[test]
    fn clearing_and_masking_all_hand_over_the_old_map_and_allocate_nothing() {
        let mask = blob(cell(0, 0, 0), cell(4, 4, 4), 200);
        let bytes_before = mask.bytes();
        assert!(bytes_before > 0, "the fixture holds no mask at all");

        let cleared = mask.cleared(false);
        assert!(cleared.is_free(), "Clear left something protected");
        assert_eq!(cleared.bytes(), 0, "an empty map has to cost nothing");

        let all = mask.cleared(true);
        assert!(all.protects_everything());
        assert_eq!(all.at(cell(1000, 1000, 1000)), PROTECTED, "Mask All missed empty space");
        assert_eq!(all.bytes(), 0, "Mask All allocated a map");
        // The old map is untouched by either, which is what lets the caller
        // move it into the history entry.
        assert_eq!(mask.bytes(), bytes_before);
    }

    /// Mask All over an empty map is one bool and no map at all.
    #[test]
    fn masking_all_over_an_empty_map_allocates_nothing() {
        let mask = MaskField::default();
        let all = mask.cleared(true);
        assert_eq!(all.bytes(), 0);
        assert!(all.brick_coords().next().is_none(), "Mask All minted a brick");
        assert_eq!(all.at(cell(3, 4, 5)), PROTECTED);
    }

    /// A filter at zero is the mask that went in, value for value.
    ///
    /// It still produces a new field -- the caller needs one to put on the body
    /// -- but every voxel of it reads back what it read before, which is what
    /// "absolute" means at the bottom of the slider.
    #[test]
    fn a_filter_at_zero_changes_no_value() {
        let mask = blob(cell(1, 1, 1), cell(6, 6, 6), 240);
        for filter in MaskFilter::ALL {
            let out = mask.filtered(filter, 0.0);
            for at in [cell(1, 1, 1), cell(3, 3, 3), cell(6, 6, 6), cell(7, 7, 7), cell(0, 0, 0)] {
                assert_eq!(out.at(at), mask.at(at), "{filter:?} moved a value at zero");
            }
        }
    }

    /// The whole point of an absolute filter: the same amount from the same
    /// snapshot lands in the same place however many times it is asked for.
    #[test]
    fn blurring_at_one_twice_from_the_same_snapshot_equals_blurring_once() {
        let mut snapshot = blob(cell(2, 2, 2), cell(8, 8, 8), 255);
        snapshot.collapse();

        let once = snapshot.filtered(MaskFilter::Blur, 1.0);
        let again = snapshot.filtered(MaskFilter::Blur, 1.0);
        assert_eq!(fingerprint(&once), fingerprint(&again));

        // And it is genuinely a blur, not a copy: the rim of the block has to
        // come down off 255 and the space outside it has to come up off 0.
        assert!(once.at(cell(2, 2, 2)) < 255, "the corner of the block did not soften");
        assert!(once.at(cell(1, 5, 5)) > 0, "nothing spread outside the block");
    }

    /// Blur softens an edge and Sharpen steepens it, over the same fixture, so
    /// neither can pass by doing nothing.
    ///
    /// Sharpen is measured on a HALF blur rather than a full one, and that is
    /// the unsharp mask being correct rather than the fixture being tuned: a
    /// full box blur of a step is a straight ramp, a straight ramp has no
    /// second derivative, and the identity is the right answer on one.
    #[test]
    fn blur_softens_a_step_and_sharpen_steepens_it() {
        // A half space: protected at x <= 5, free above it.
        let mut mask = MaskField::default();
        for z in 0..12 {
            for y in 0..12 {
                for x in 0..=5 {
                    mask.write(cell(x, y, z), PROTECTED);
                }
            }
        }
        let blurred = mask.filtered(MaskFilter::Blur, 0.5);
        assert!(blurred.at(cell(5, 5, 5)) < PROTECTED, "the last protected column stayed hard");
        assert!(blurred.at(cell(6, 5, 5)) > UNMASKED, "the first free column stayed empty");

        let sharpened = blurred.filtered(MaskFilter::Sharpen, 1.0);
        assert!(
            sharpened.at(cell(5, 5, 5)) > blurred.at(cell(5, 5, 5)),
            "sharpen did not push the protected side back up"
        );
        assert!(
            sharpened.at(cell(6, 5, 5)) < blurred.at(cell(6, 5, 5)),
            "sharpen did not push the free side back down"
        );
    }

    /// Grow spreads protection by a voxel and Shrink pulls it back.
    #[test]
    fn grow_spreads_protection_by_one_voxel_and_shrink_pulls_it_back() {
        let mask = blob(cell(4, 4, 4), cell(7, 7, 7), PROTECTED);

        let grown = mask.filtered(MaskFilter::Grow, 1.0);
        assert_eq!(grown.at(cell(3, 4, 4)), PROTECTED, "grow did not reach the next voxel");
        assert_eq!(grown.at(cell(2, 4, 4)), UNMASKED, "grow reached two voxels, not one");

        let shrunk = mask.filtered(MaskFilter::Shrink, 1.0);
        assert_eq!(shrunk.at(cell(4, 4, 4)), UNMASKED, "shrink did not pull the rim in");
        assert_eq!(shrunk.at(cell(5, 5, 5)), PROTECTED, "shrink ate the middle as well");
    }

    /// **Grow then Shrink is a closing and does not round-trip**, which the
    /// filter's own doc comment says out loud so nobody reports it as a bug.
    #[test]
    fn growing_then_shrinking_over_a_hard_edge_does_not_round_trip() {
        // Two blocks with a two-voxel gap between them. Grow bridges the gap,
        // and Shrink cannot reopen what is no longer an edge.
        let mut mask = blob(cell(2, 2, 2), cell(4, 8, 8), PROTECTED);
        for z in 2..=8 {
            for y in 2..=8 {
                for x in 7..=9 {
                    mask.write(cell(x, y, z), PROTECTED);
                }
            }
        }
        assert_eq!(mask.at(cell(5, 5, 5)), UNMASKED, "the fixture has no gap");
        assert_eq!(mask.at(cell(6, 5, 5)), UNMASKED, "the fixture has no gap");

        let closed = mask.filtered(MaskFilter::Grow, 1.0).filtered(MaskFilter::Shrink, 1.0);
        assert_eq!(
            closed.at(cell(5, 5, 5)),
            PROTECTED,
            "the round trip reopened the gap, so the doc comment is now wrong"
        );
    }

    /// A filter has to mean the same thing whichever way the polarity points,
    /// because `at` and `write` are both in protection space and a user
    /// blurring an inverted mask is blurring what they can see.
    #[test]
    fn a_filter_under_inversion_works_on_protection_and_not_on_stored_bytes() {
        // Stored: a block of 255 inside an inverted field, which resolves to a
        // block of FREE inside a fully protected body.
        let mut mask = MaskField::default();
        mask.set_inverted(true);
        for z in 4..=7 {
            for y in 4..=7 {
                for x in 4..=7 {
                    mask.write(cell(x, y, z), UNMASKED);
                }
            }
        }
        assert_eq!(mask.at(cell(5, 5, 5)), UNMASKED);
        assert_eq!(mask.at(cell(0, 0, 0)), PROTECTED);

        // Grow takes the GREATER protection, so the protected surround eats
        // into the free block -- exactly what it does the other way up.
        let grown = mask.filtered(MaskFilter::Grow, 1.0);
        assert!(grown.inverted(), "the filter dropped the polarity");
        assert_eq!(grown.at(cell(4, 5, 5)), PROTECTED, "grow read stored bytes, not protection");
        assert_eq!(grown.at(cell(5, 5, 5)), UNMASKED, "grow ate the whole block");
    }

    /// Half an amount is half the way there, which is what makes the slider a
    /// slider and not a button pressed repeatedly.
    #[test]
    fn half_an_amount_lands_half_way_to_the_filtered_value() {
        let mask = blob(cell(4, 4, 4), cell(7, 7, 7), PROTECTED);
        let full = mask.filtered(MaskFilter::Grow, 1.0).at(cell(3, 4, 4));
        let half = mask.filtered(MaskFilter::Grow, 0.5).at(cell(3, 4, 4));
        assert_eq!(full, PROTECTED);
        assert!(half > UNMASKED && half < full, "half a grow landed at {half}");
        assert!((half as i32 - 128).abs() <= 1, "half of 255 is 128, not {half}");
    }

    /// The uniform fast path answers the interior without a voxel loop, gives
    /// the same value the loop would, and leaves only the SHELL to be walked.
    ///
    /// **This is the whole of what makes a global filter affordable, and until
    /// increment 24's review it was not actually checked.** The shortcut
    /// returns bit-for-bit what the slow path returns -- a dense block of one
    /// repeated byte collapses straight back to [`MaskBrick::Uniform`] -- so an
    /// assertion about the VALUE passes with the shortcut deleted. Confirmed:
    /// with `if false &&` in front of the `uniform_over` test the whole
    /// workspace stayed green. So this counts the shortcut instead, through the
    /// bool [`MaskField::filter_brick`] returns, which only the shortcut itself
    /// can set.
    ///
    /// The numbers are the property stated in numbers. A solid `n x n x n`
    /// block of uniform bricks has `(n - 2) ^ 3` bricks whose whole 3x3x3
    /// neighbourhood is inside it, and every other destination brick is walked:
    /// at `n = 5` that is 27 of 343, which looks like nothing, and at the
    /// 46,656 bricks (`n = 36`) the pool is sized for it is 39,304 of 54,872 --
    /// the shell is a surface and the interior is a volume, so the ratio only
    /// improves with size. Tiles are inserted directly, which is the state
    /// [`MaskField::collapse`] leaves a painted mask in and is the state the
    /// shortcut exists for.
    #[test]
    fn the_uniform_shortcut_answers_the_interior_and_leaves_only_the_shell_to_be_walked() {
        const SPAN: i32 = 5;

        let mut mask = MaskField::default();
        for z in 0..SPAN {
            for y in 0..SPAN {
                for x in 0..SPAN {
                    mask.bricks.insert(BrickCoord::new(x, y, z), MaskBrick::Uniform(PROTECTED));
                }
            }
        }

        // The destination set `filtered` builds for this fixture: every brick
        // the mask holds, plus the twenty-six around each.
        let mut padded = vec![UNMASKED; PADDED_VOXELS];
        let mut shortcut: Vec<BrickCoord> = Vec::new();
        let mut walked = 0usize;
        for z in -1..=SPAN {
            for y in -1..=SPAN {
                for x in -1..=SPAN {
                    let coord = BrickCoord::new(x, y, z);
                    let (_, took) = mask.filter_brick(&mut padded, coord, MaskFilter::Blur, 1.0);
                    if took {
                        shortcut.push(coord);
                    } else {
                        walked += 1;
                    }
                }
            }
        }

        let interior = ((SPAN - 2) * (SPAN - 2) * (SPAN - 2)) as usize;
        assert_eq!(
            shortcut.len(),
            interior,
            "{} bricks took the uniform shortcut where {interior} should have, and {walked} were \
             walked: the filter is paying 27 reads a voxel over the interior",
            shortcut.len()
        );
        // And they are the interior specifically, not any 27 bricks: a shortcut
        // taken on a brick the protection actually changes across would be a
        // wrong answer rather than a slow one.
        for coord in &shortcut {
            assert!(
                coord.0.min_element() >= 1 && coord.0.max_element() <= SPAN - 2,
                "{coord:?} is on the shell and was answered without looking at its voxels"
            );
        }

        // The value the shortcut gives is the value the loop would give, for
        // every filter -- otherwise the interior of a Mask All quietly diverges
        // from its shell.
        for filter in MaskFilter::ALL {
            let out = mask.filtered(filter, 1.0);
            let middle = BrickCoord::new(SPAN / 2, SPAN / 2, SPAN / 2);
            assert!(
                matches!(out.slab(middle), MaskSlab::Uniform(PROTECTED)),
                "{filter:?} did not leave the interior alone"
            );
            assert_eq!(
                out.at(middle.origin() + IVec3::splat(16)),
                PROTECTED,
                "{filter:?} lost the middle"
            );
        }
    }

    /// **The out-of-memory guard, and the arithmetic behind the plan's figure.**
    ///
    /// A filter holds the old mask while it writes the new one, and history
    /// then holds the old one, so the peak is `field + 2 x mask`. A fully
    /// painted mask is 25% of the field it sits on -- 32,768 bytes a brick
    /// against 131,072 -- so that peak is 1.5 times the field, and 1.5 times
    /// 4 GiB is exactly the 6 GiB ceiling. On the measured 0.0565 mm dragon
    /// that is 6.22 GiB from one button press.
    ///
    /// The boundary itself goes to "fits": [`MaskField::would_fit`] passes when
    /// the prediction lands ON the ceiling, which is increment 18's rule and
    /// not this one's to change. So the refusal fires just past a 4 GiB field
    /// rather than at it.
    #[test]
    fn the_filter_refusal_fires_just_past_a_four_gibibyte_field() {
        let mask = MaskField::default();
        let gibibyte = 1024.0 * 1024.0 * 1024.0;
        let field = (4.0 * gibibyte) as usize;
        // What a fully painted mask on that field costs, and what `resident`
        // already counts one copy of.
        let painted = field / 4;

        assert_eq!(
            mask.would_fit(painted, field + painted, 0.0565),
            None,
            "1.5 x 4 GiB is exactly the ceiling, so it has to be allowed through"
        );
        let (why, coarser) = mask
            .would_fit(painted + 1, field + painted, 0.0565)
            .expect("one byte past the ceiling has to refuse");
        assert!(why.contains("6.0 GB"), "the refusal should name the cost: {why}");
        assert!(coarser > 0.0565, "the suggestion has to be COARSER: {coarser}");
    }

    /// What the guard is handed, and it has to be the mask's real size rather
    /// than the brick count.
    #[test]
    fn a_masks_own_byte_count_matches_what_it_charges_the_volume() {
        let mut mask = blob(cell(0, 0, 0), cell(20, 20, 20), 90);
        mask.collapse();
        let mut stats = VolumeStats::default();
        mask.add_to_stats(&mut stats);
        assert_eq!(mask.bytes(), stats.mask_bytes + mask.map_bytes());
        assert!(mask.bytes() > 0, "the fixture holds no mask at all");
    }
}
