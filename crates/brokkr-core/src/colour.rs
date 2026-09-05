// SPDX-License-Identifier: AGPL-3.0-only

//! The painted filament slot: one byte per voxel, held per body.
//!
//! A voxel's byte is a **filament slot number, never a colour**. Slot 0 means
//! unpainted and prints with whatever the package declares as its base; 1
//! upwards name the tool heads. The whole reason this is an index rather than
//! an RGB triple is measured: `u32` RGBA per voxel is +100% against a dense
//! field brick where a slot byte is +25%, and a slicer wants to be told which
//! filament prints a triangle rather than what colour to mix.
//!
//! # A second sparse map, inside [`crate::Volume`], never inside [`Brick`]
//!
//! [`ColourField`] mirrors the brick map and [`crate::mask::MaskField`] exactly:
//! absent means unpainted, a [`ColourBrick::Uniform`] tile costs no heap at all,
//! and a dense brick is 32,768 bytes against the field brick's 131,072.
//!
//! **The three reasons `mask` gives for staying out of [`Brick`] all hold here
//! word for word**, and a fourth is specific to paint. `Volume::record_for_undo`
//! clones the whole brick, so colour carried inside one would make a PAINT
//! stroke snapshot the 128 KB distance array it did not touch as well as the
//! 32 KB it did -- 192 KiB a brick, which is **1,365** brick-snapshots against
//! the 256 MB history budget where a parallel field gives 8,192. Painting would
//! cost more undo depth than sculpting, which is the wrong way round for the
//! cheaper operation.
//!
//! The earlier plan for this feature decided the opposite, and its argument was
//! real: making `Brick::is_collapsible` answer for two arrays would let the
//! compiler walk an implementer to every site that rebuilds a brick and might
//! silently drop colour. That sweep was done by hand instead -- `rotated`,
//! `shifted`, `resampled`, `warped`, the merge, the split, the duplicate and
//! the Move brush each carry the paint, each with a test that fails without
//! it -- and the compiler's help would not have bought back the undo
//! arithmetic.
//!
//! # Paint lives in the band, and only the band is ever read
//!
//! A slot is written only at a voxel the field puts in the narrow band -- see
//! [`ColourField::paintable`] -- and read only at mesh vertices, which sit in
//! the cells the surface passes through. A slot can be LEFT behind outside the
//! band, by a carve that removes the material under it or a Move that drags
//! the material away, and that is tolerated rather than scrubbed: nothing
//! reads it, and scrubbing at stroke end would have to record every brick it
//! touched or an undo of the carve would come back unpainted. The two places
//! that copy paint from one body to another do consult the field, because
//! there a stale slot WOULD be read: a merge admits an incoming slot only where
//! the incoming field is in band, and a split hands each half only the slots
//! under its own band.
//!
//! # No polarity, and that is not an omission
//!
//! A mask has an `inverted` bool because "protect everything except this" is a
//! thing people ask for and because the map from stored byte to protection is
//! its own inverse. A slot index has no inverse: there is no meaningful "every
//! filament except 3", so there is nothing to resolve at read and the byte
//! stored is the byte meant.
//!
//! # What is deliberately not here
//!
//! **No interior colour.** Only voxels the surface can reach are worth painting
//! -- the slicer derives the inside from the shell -- and refusing to paint a
//! saturated interior is what keeps a solid body's interior bricks at zero heap
//! instead of promoting each to 32 KB the first time a stamp passes over it.
//! The predicate for that is strict and lives in [`ColourField::paintable`].

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::brick::{BRICK_DIM, BRICK_VOXELS, BrickCoord, brick_index};
use crate::orientation::AxisRotation;
use crate::similarity::Similarity;
use crate::volume::VolumeStats;
use glam::{IVec3, Vec3};

/// The slot an unpainted voxel carries.
///
/// **Reserved, and this is load-bearing.** An imported palette is built by
/// median cut, which fills 0..255 and would otherwise hand index 0 to the
/// model's most common colour -- so every cut face, every newly inflated
/// surface and every voxel the import never reached would print in the
/// dominant colour, render perfectly, and pass every test. Palettes fill from
/// 1.
pub const UNPAINTED: u8 = 0;

/// The painted slots of one brick.
///
/// Mirrors [`crate::Brick`] and [`crate::mask::MaskBrick`] one for one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColourBrick {
    /// Every voxel holds this slot. Costs no allocation.
    Uniform(u8),
    /// Per voxel stored slots, indexed by [`brick_index`].
    Dense(Box<[u8; BRICK_VOXELS]>),
}

impl ColourBrick {
    /// A dense brick with every voxel set to `slot`.
    pub(crate) fn dense_filled(slot: u8) -> Self {
        let data: Box<[u8]> = vec![slot; BRICK_VOXELS].into_boxed_slice();
        let data: Box<[u8; BRICK_VOXELS]> = data.try_into().expect("length is BRICK_VOXELS");
        ColourBrick::Dense(data)
    }

    /// Promote a uniform brick so individual voxels can be written.
    pub(crate) fn make_dense(&mut self) -> &mut [u8; BRICK_VOXELS] {
        if let ColourBrick::Uniform(slot) = *self {
            *self = ColourBrick::dense_filled(slot);
        }
        match self {
            ColourBrick::Dense(data) => data,
            ColourBrick::Uniform(_) => unreachable!("just promoted to dense"),
        }
    }

    /// Heap bytes this brick holds, excluding the enum itself.
    #[inline]
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            ColourBrick::Uniform(_) => 0,
            ColourBrick::Dense(_) => BRICK_VOXELS,
        }
    }

    /// The single slot every voxel holds, or `None` when it carries detail.
    ///
    /// Looser than [`crate::Brick::is_collapsible`] for the same reason the
    /// mask's is: that rule is an anti-thrash heuristic for a brick the next
    /// stamp will rewrite, and collapsing at end of stroke removes the thrash.
    pub(crate) fn is_collapsible(&self) -> Option<u8> {
        match self {
            ColourBrick::Uniform(slot) => Some(*slot),
            ColourBrick::Dense(data) => {
                let first = data[0];
                data.iter().all(|slot| *slot == first).then_some(first)
            }
        }
    }
}

/// One brick's colour, resolved for the duration of a stamp so the voxel loop
/// does no map lookups.
///
/// The three arms are what let the hot loop hoist: [`ColourSlab::Unpainted`]
/// and [`ColourSlab::Uniform`] are loop-invariant, only [`ColourSlab::Dense`]
/// is per voxel. The mask's own note applies: one brick meshes to around 1100
/// vertices, and a map lookup each is a per-vertex hash on a path a stroke has
/// 8 ms for.
#[derive(Debug, Clone, Copy)]
pub enum ColourSlab<'a> {
    /// The body has no colour at all.
    Unpainted,
    /// No entry for this brick, or a collapsed one.
    Uniform(u8),
    Dense(&'a [u8; BRICK_VOXELS]),
}

impl ColourSlab<'_> {
    /// The stored slot at a voxel.
    #[inline]
    pub(crate) fn slot_at(&self, index: usize) -> u8 {
        match self {
            ColourSlab::Unpainted => UNPAINTED,
            ColourSlab::Uniform(slot) => *slot,
            ColourSlab::Dense(data) => data[index],
        }
    }
}

/// What [`ColourField::edit_brick`] did to one brick.
///
/// [`crate::mask::MaskEdit`]'s twin, for the same caller shape: the recorder
/// wants the prior contents on first change and nothing at all otherwise.
#[derive(Debug)]
pub(crate) enum ColourEdit {
    /// At least one voxel changed. Carries what the brick held before, which is
    /// `None` for a brick that had no entry.
    Changed(Option<ColourBrick>),
    /// Nothing changed, and any promotion made on the way in has been undone.
    Unchanged,
}

/// One body's painted slots.
///
/// See the module documentation for where this lives and why. Reachable only
/// through the volume that owns it, so no expression can hand the mesher, the
/// serialiser or the brush the wrong body's.
#[derive(Debug, Default, Clone)]
pub struct ColourField {
    bricks: FxHashMap<BrickCoord, ColourBrick>,
    /// How many times this has changed, counting up and never down.
    ///
    /// The same job the mask's revision does: a viewer that caches anything
    /// derived from the colour compares one `u64` rather than re-reading a
    /// gigabyte. Carried forward by every transform, so a resample cannot hand
    /// a body a fresh field wearing a stamp a cache has already seen.
    revision: u64,
}

impl ColourField {
    /// Whether this body has been painted at all.
    ///
    /// The short circuit every read path takes first: an unpainted document
    /// pays nothing anywhere, which is what makes the whole feature free for
    /// someone who never presses the button.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    /// How many times this has changed.
    #[inline]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    /// Whether a voxel at this distance may be painted.
    ///
    /// **Strict on both sides, and that is the whole point.** Every distance in
    /// the field is clamped to `INSIDE..=OUTSIDE`, so `|distance| <=
    /// NARROW_BAND` is vacuously true -- it selects 100% of voxels, including
    /// the saturated interior that is 86.8% of a real dense import. A paint
    /// stamp using it would promote a zero-heap uniform interior brick to a
    /// full 32 KB allocation, visible in nothing but `resident_bytes`.
    ///
    /// This is [`crate::redistance::in_band`]'s test, and the same shape has
    /// already shipped a 32.6% measurement error elsewhere in this crate under
    /// a docstring calling the operation lossless.
    #[inline]
    pub fn paintable(distance: f32) -> bool {
        distance > crate::brick::INSIDE && distance < crate::brick::OUTSIDE
    }

    /// The slot stored at a voxel.
    pub fn at(&self, cell: IVec3) -> u8 {
        let coord = BrickCoord::containing(cell);
        let Some(brick) = self.bricks.get(&coord) else { return UNPAINTED };
        let local = cell - coord.origin();
        match brick {
            ColourBrick::Uniform(slot) => *slot,
            ColourBrick::Dense(data) => {
                data[brick_index(local.x as usize, local.y as usize, local.z as usize)]
            }
        }
    }

    /// This brick's slots, resolved once for a stamp.
    pub(crate) fn slab(&self, coord: BrickCoord) -> ColourSlab<'_> {
        if self.bricks.is_empty() {
            return ColourSlab::Unpainted;
        }
        match self.bricks.get(&coord) {
            None => ColourSlab::Uniform(UNPAINTED),
            Some(ColourBrick::Uniform(slot)) => ColourSlab::Uniform(*slot),
            Some(ColourBrick::Dense(data)) => ColourSlab::Dense(data),
        }
    }

    /// Paint one voxel.
    ///
    /// Writing [`UNPAINTED`] into a brick that has no entry is the difference
    /// between a field that costs nothing and one that allocates 32 KB per
    /// brick a brush merely passed over.
    pub fn write(&mut self, cell: IVec3, slot: u8) {
        let coord = BrickCoord::containing(cell);
        let local = cell - coord.origin();
        let index = brick_index(local.x as usize, local.y as usize, local.z as usize);
        self.touch();
        match self.bricks.get_mut(&coord) {
            // Writing what the tile already holds must not promote it.
            Some(ColourBrick::Uniform(held)) if *held == slot => {}
            Some(brick) => brick.make_dense()[index] = slot,
            None if slot == UNPAINTED => {}
            None => {
                let mut brick = ColourBrick::Uniform(UNPAINTED);
                brick.make_dense()[index] = slot;
                self.bricks.insert(coord, brick);
            }
        }
    }

    /// Apply `edit` to every voxel of one brick inside an inclusive box, and
    /// say what changed.
    ///
    /// `paintable` answers, by brick index, whether the FIELD allows a voxel
    /// to be painted at all -- see [`ColourField::paintable`] -- and a voxel it
    /// refuses is never offered to `edit`. The predicate is the volume's to
    /// supply because the distances live there; this type never sees them.
    ///
    /// **A brick the box clipped but never changed is left exactly as it was**,
    /// promotion included, for the reason [`crate::mask::MaskField::edit_brick`]
    /// gives and one more of its own: a paint stamp's bounding cube grazes the
    /// saturated bricks around the shell it is painting, and without the
    /// rollback every one of those would be a 32 KB allocation for a brick no
    /// voxel of which was paintable.
    pub(crate) fn edit_brick(
        &mut self,
        coord: BrickCoord,
        lo: IVec3,
        hi: IVec3,
        voxel_size: f32,
        paintable: &impl Fn(usize) -> bool,
        edit: &impl Fn(IVec3, Vec3, u8) -> u8,
    ) -> ColourEdit {
        let origin = coord.origin();
        let existed = self.bricks.contains_key(&coord);
        let prior = self.bricks.get(&coord).cloned();
        let brick = self.bricks.entry(coord).or_insert(ColourBrick::Uniform(UNPAINTED));
        let was_uniform = matches!(brick, ColourBrick::Uniform(_));
        let uniform_slot = match brick {
            ColourBrick::Uniform(slot) => *slot,
            ColourBrick::Dense(_) => UNPAINTED,
        };

        let data = brick.make_dense();
        let mut changed = false;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let voxel = IVec3::new(x, y, z);
                    let local = voxel - origin;
                    let index = brick_index(local.x as usize, local.y as usize, local.z as usize);
                    if !paintable(index) {
                        continue;
                    }
                    let position = voxel.as_vec3() * voxel_size;
                    let slot = edit(voxel, position, data[index]);
                    if slot != data[index] {
                        data[index] = slot;
                        changed = true;
                    }
                }
            }
        }

        if changed {
            self.touch();
            return ColourEdit::Changed(prior);
        }
        if was_uniform {
            if existed {
                self.bricks.insert(coord, ColourBrick::Uniform(uniform_slot));
            } else {
                self.bricks.remove(&coord);
            }
        }
        ColourEdit::Unchanged
    }

    /// One brick's entry, for undo to copy before it puts another back.
    pub(crate) fn brick(&self, coord: BrickCoord) -> Option<&ColourBrick> {
        self.bricks.get(&coord)
    }

    /// Put a brick back exactly as a recording found it, `None` meaning it had
    /// no entry.
    pub(crate) fn restore_brick(&mut self, coord: BrickCoord, brick: Option<ColourBrick>) {
        self.touch();
        match brick {
            Some(brick) => {
                self.bricks.insert(coord, brick);
            }
            None => {
                self.bricks.remove(&coord);
            }
        }
    }

    /// Append the stored slot at each of `cells`, which mostly belong to
    /// `coord`.
    ///
    /// The mesh-attribute writer's call, and a method here rather than a loop
    /// at the call site for the reason the mask's twin gives: the lookup hoists
    /// for the great majority of a brick's vertices and only the one-voxel
    /// border falls back.
    pub(crate) fn slots_at_cells(&self, coord: BrickCoord, cells: &[IVec3], out: &mut Vec<u8>) {
        if self.bricks.is_empty() {
            out.resize(out.len() + cells.len(), UNPAINTED);
            return;
        }

        let slab = self.slab(coord);
        let origin = coord.origin();
        let dim = BRICK_DIM as i32;
        out.reserve(cells.len());
        for cell in cells {
            let local = *cell - origin;
            let slot = if local.min_element() >= 0 && local.max_element() < dim {
                slab.slot_at(brick_index(local.x as usize, local.y as usize, local.z as usize))
            } else {
                self.at(*cell)
            };
            out.push(slot);
        }
    }

    /// Drop every brick that holds one slot everywhere, and every brick that
    /// holds nothing.
    ///
    /// Called at end of stroke rather than per stamp, which is what lets the
    /// collapse rule be looser than the field's.
    pub(crate) fn collapse(&mut self) {
        self.bricks.retain(|_, brick| match brick.is_collapsible() {
            Some(UNPAINTED) => false,
            Some(slot) => {
                *brick = ColourBrick::Uniform(slot);
                true
            }
            None => true,
        });
    }

    /// Add this field's census and its bytes to a volume's.
    ///
    /// **`resident_bytes` has to include the colour.** That number is what the
    /// 6 GiB ceiling is checked against, what a removed body charges to the
    /// undo allowance, and what every bug report quotes as the model's size.
    /// The mask's equivalent exists solely because that hole was found once;
    /// missing it here would under-report by up to 25% at exactly the moment
    /// the document is largest.
    pub(crate) fn add_to_stats(&self, stats: &mut VolumeStats) {
        let mut bytes = 0;
        for brick in self.bricks.values() {
            stats.colour_bricks += 1;
            if matches!(brick, ColourBrick::Dense(_)) {
                stats.colour_dense_bricks += 1;
            }
            bytes += brick.heap_bytes();
        }
        stats.colour_bytes += bytes;
        stats.resident_bytes += bytes + self.map_bytes();
    }

    /// What the map itself costs, separate from the bricks it holds.
    fn map_bytes(&self) -> usize {
        self.bricks.capacity()
            * (std::mem::size_of::<BrickCoord>() + std::mem::size_of::<ColourBrick>())
    }

    /// A deep copy with every brick moved by whole bricks.
    ///
    /// Whole bricks move no voxel within its brick, so each one is a `clone`
    /// and a tile stays a tile.
    pub(crate) fn translated(&self, offset_bricks: IVec3) -> ColourField {
        let mut copy = ColourField { bricks: FxHashMap::default(), revision: self.revision + 1 };
        copy.bricks.reserve(self.bricks.len());
        for (coord, brick) in &self.bricks {
            copy.bricks.insert(BrickCoord(coord.0 + offset_bricks), brick.clone());
        }
        copy
    }

    /// A fresh field carrying this one's revision forward, for the transforms.
    fn successor(&self) -> ColourField {
        ColourField { bricks: FxHashMap::default(), revision: self.revision + 1 }
    }

    /// This paint turned by `rotation`.
    ///
    /// Exact, as the mask's and the field's are: a quarter turn maps the
    /// lattice onto itself, so a brick turns into a brick and a tile costs
    /// nothing.
    pub(crate) fn rotated(&self, rotation: AxisRotation) -> ColourField {
        let map = rotation.axis_map();
        let last = BRICK_DIM - 1;
        let mut turned = self.successor();
        turned.bricks.reserve(self.bricks.len());
        for (coord, brick) in &self.bricks {
            let destination = BrickCoord(rotation.apply_voxel(coord.0));
            let moved = match brick {
                ColourBrick::Uniform(slot) => ColourBrick::Uniform(*slot),
                ColourBrick::Dense(data) => {
                    let mut moved = ColourBrick::dense_filled(UNPAINTED);
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

    /// This paint moved by whole VOXELS, sub-brick offsets included.
    ///
    /// Value-exact: every destination cell takes exactly one source cell's
    /// slot. Delegates to [`ColourField::translated`] when the offset is brick
    /// aligned, so the common case moves `Box` pointers rather than bytes.
    pub(crate) fn shifted(&self, offset_voxels: IVec3) -> ColourField {
        let dim = BRICK_DIM as i32;
        if offset_voxels.rem_euclid(IVec3::splat(dim)) == IVec3::ZERO {
            return self.translated(offset_voxels / dim);
        }
        let mut wanted: FxHashSet<BrickCoord> = FxHashSet::default();
        for coord in self.bricks.keys() {
            let low = BrickCoord::containing(coord.origin() + offset_voxels).0;
            let high = BrickCoord::containing(coord.max_voxel() + offset_voxels).0;
            wanted.extend(bricks_between(low, high));
        }
        self.gathered(
            wanted,
            |coord| Some((coord.origin() - offset_voxels, coord.max_voxel() - offset_voxels)),
            |cell| cell - offset_voxels,
        )
    }

    /// This paint on a different lattice, at the same world positions.
    ///
    /// Nearest neighbour, and here that is not a choice at all: there is no
    /// slot between two slots to interpolate toward. Coarsening loses paint
    /// narrower than the new voxel, because there is nowhere to hold it.
    pub(crate) fn resampled(&self, from_voxel: f32, to_voxel: f32) -> ColourField {
        let ratio = from_voxel / to_voxel;
        let inverse = to_voxel / from_voxel;
        // Grown by HALF THE RATIO and not by one voxel: the destination cells
        // that round back to a source brick's last cell reach
        // `(max + 0.5) * ratio`, so past a 2x refinement a pad of one drops
        // the paint on every brick's far face. See `gather_pad`.
        let pad = IVec3::splat(gather_pad(ratio));
        let mut wanted: FxHashSet<BrickCoord> = FxHashSet::default();
        for coord in self.bricks.keys() {
            let low = (coord.origin().as_vec3() * ratio).floor().as_ivec3() - pad;
            let high = (coord.max_voxel().as_vec3() * ratio).ceil().as_ivec3() + pad;
            wanted.extend(bricks_between(
                BrickCoord::containing(low).0,
                BrickCoord::containing(high).0,
            ));
        }
        let source = move |cell: IVec3| (cell.as_vec3() * inverse).round().as_ivec3();
        self.gathered(
            wanted,
            |coord| Some((source(coord.origin()), source(coord.max_voxel()))),
            source,
        )
    }

    /// This paint rebuilt through a similarity, onto the SAME lattice.
    ///
    /// Nearest neighbour, for the reason [`ColourField::resampled`] gives. The
    /// destination footprint is each source brick's box pushed FORWARD through
    /// the map, as the mask does, so paint is found by its own bricks and not
    /// by a walk of the field's.
    pub(crate) fn warped(&self, by: Similarity, voxel_size: f32) -> ColourField {
        // The same half-ratio pad `resampled` needs, at the largest scale the
        // similarity applies along any axis.
        let pad = IVec3::splat(gather_pad(by.scale.max_element()));
        let mut wanted: FxHashSet<BrickCoord> = FxHashSet::default();
        for coord in self.bricks.keys() {
            let source_low = coord.origin().as_vec3() * voxel_size;
            let source_high = coord.max_voxel().as_vec3() * voxel_size;
            let (low, high) = crate::transform::forward_bounds(by, source_low, source_high);
            let low = BrickCoord::containing((low / voxel_size).floor().as_ivec3() - pad).0;
            let high = BrickCoord::containing((high / voxel_size).ceil().as_ivec3() + pad).0;
            wanted.extend(bricks_between(low, high));
        }
        self.gathered(
            wanted,
            // A warped box is not a box, so there is no uniform fast path.
            |_| None,
            |cell| {
                (by.inverse_transform_point(cell.as_vec3() * voxel_size) / voxel_size)
                    .round()
                    .as_ivec3()
            },
        )
    }

    /// Build every `wanted` destination brick by reading one source cell per
    /// voxel, nearest neighbour.
    ///
    /// The one gather loop the three lattice transforms share. `source_box`
    /// names the inclusive source box a destination brick reads from when
    /// that footprint IS a box, which lets a brick whose whole source holds
    /// one slot be answered from the map without a voxel loop -- after a
    /// collapse, most of them. A brick that gathers to one slot everywhere
    /// collapses on the way out, and one that gathers to nothing is dropped.
    fn gathered(
        &self,
        wanted: FxHashSet<BrickCoord>,
        source_box: impl Fn(&BrickCoord) -> Option<(IVec3, IVec3)> + Sync,
        source_of: impl Fn(IVec3) -> IVec3 + Sync,
    ) -> ColourField {
        let mut out = self.successor();
        if self.bricks.is_empty() {
            return out;
        }
        let coords: Vec<BrickCoord> = wanted.into_iter().collect();
        let built: Vec<(BrickCoord, ColourBrick)> = coords
            .par_iter()
            .filter_map(|coord| {
                if let Some((low, high)) = source_box(coord)
                    && let Some(slot) = self.uniform_over(low, high)
                {
                    return (slot != UNPAINTED).then_some((*coord, ColourBrick::Uniform(slot)));
                }
                let origin = coord.origin();
                let mut brick = ColourBrick::dense_filled(UNPAINTED);
                let data = brick.make_dense();
                // One map lookup per run of destination voxels that share a
                // source brick rather than one per voxel, as the mask's own
                // gather does: along X that is a 32-fold saving at any ratio.
                let mut cached: Option<(BrickCoord, ColourSlab<'_>)> = None;
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let cell = source_of(origin + IVec3::new(x as i32, y as i32, z as i32));
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
                            data[brick_index(x, y, z)] = slab.slot_at(brick_index(
                                local.x as usize,
                                local.y as usize,
                                local.z as usize,
                            ));
                        }
                    }
                }
                match brick.is_collapsible() {
                    Some(UNPAINTED) => None,
                    Some(slot) => Some((*coord, ColourBrick::Uniform(slot))),
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

    /// The one slot covering an inclusive source cell box, or `None` when it
    /// carries detail. Absent bricks count as unpainted, which is what they
    /// read as.
    fn uniform_over(&self, low: IVec3, high: IVec3) -> Option<u8> {
        let mut found: Option<u8> = None;
        for coord in bricks_between(BrickCoord::containing(low).0, BrickCoord::containing(high).0) {
            let slot = match self.bricks.get(&coord) {
                None => UNPAINTED,
                Some(ColourBrick::Uniform(slot)) => *slot,
                Some(ColourBrick::Dense(_)) => return None,
            };
            match found {
                None => found = Some(slot),
                Some(held) if held == slot => {}
                Some(_) => return None,
            }
        }
        found
    }

    /// Take the other body's slot at every voxel where it has one AND
    /// `admits` says the incoming material is really there.
    ///
    /// The colour half of a merge. Where both are painted the incoming paint
    /// wins, because the incoming body's material is what the merge is adding
    /// and that material carries its own slot; where the incoming voxel is
    /// unpainted, ours stands. Neither rule can unpaint anything.
    ///
    /// `admits` is the caller's view of the incoming FIELD -- in band or not --
    /// because a slot can be left behind outside the band (see the module
    /// doc) and a stale slot admitted here would overwrite fresh paint on the
    /// target under material the incoming body does not even have.
    ///
    /// `record` is handed every brick's prior contents before it changes, so
    /// the caller's stroke recorder can put the target's own paint back on
    /// undo. Without that a merge is the one operation that destroys paint
    /// nothing can recover.
    pub(crate) fn union_from(
        &mut self,
        other: &ColourField,
        admits: impl Fn(BrickCoord, usize) -> bool,
        mut record: impl FnMut(BrickCoord, Option<ColourBrick>),
    ) -> Vec<BrickCoord> {
        let mut changed = Vec::new();
        for (coord, incoming) in &other.bricks {
            if matches!(incoming, ColourBrick::Uniform(UNPAINTED)) {
                continue;
            }
            let theirs = other.slab(*coord);
            let mut brick =
                self.bricks.get(coord).cloned().unwrap_or(ColourBrick::Uniform(UNPAINTED));
            let data = brick.make_dense();
            let mut moved = false;
            for (index, held) in data.iter_mut().enumerate() {
                let slot = theirs.slot_at(index);
                if slot != UNPAINTED && *held != slot && admits(*coord, index) {
                    *held = slot;
                    moved = true;
                }
            }
            if !moved {
                continue;
            }
            record(*coord, self.bricks.get(coord).cloned());
            match brick.is_collapsible() {
                Some(slot) => self.bricks.insert(*coord, ColourBrick::Uniform(slot)),
                None => self.bricks.insert(*coord, brick),
            };
            self.touch();
            changed.push(*coord);
        }
        changed
    }

    /// Every brick with an entry, for the serialiser and the tests.
    pub(crate) fn bricks(&self) -> &FxHashMap<BrickCoord, ColourBrick> {
        &self.bricks
    }

    /// Insert a brick wholesale, for the reader and the transforms.
    pub(crate) fn insert(&mut self, coord: BrickCoord, brick: ColourBrick) {
        self.touch();
        self.bricks.insert(coord, brick);
    }
}

/// How many destination voxels past a source brick's scaled box a
/// nearest-neighbour gather can reach, for a magnification of `ratio`.
///
/// The destination cells that round to the source cell `max` are
/// `[(max - 0.5) * ratio, (max + 0.5) * ratio)`, which is half a ratio past
/// where `ceil(max * ratio)` stops. One voxel covers that up to 2x; past it
/// the paint on every brick's far face was dropped, and so was the mask's.
pub(crate) fn gather_pad(ratio: f32) -> i32 {
    (0.5 * ratio).ceil().max(1.0) as i32
}

/// Every brick coordinate in an inclusive box of brick coordinates.
fn bricks_between(low: IVec3, high: IVec3) -> impl Iterator<Item = BrickCoord> {
    (low.z..=high.z).flat_map(move |z| {
        (low.y..=high.y).flat_map(move |y| (low.x..=high.x).map(move |x| BrickCoord::new(x, y, z)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::{INSIDE, NARROW_BAND, OUTSIDE};

    #[test]
    fn an_unpainted_field_costs_nothing_and_reads_as_slot_zero() {
        let field = ColourField::default();
        assert!(field.is_empty());
        assert_eq!(field.at(IVec3::new(3, 4, 5)), UNPAINTED);
        let mut stats = VolumeStats::default();
        field.add_to_stats(&mut stats);
        assert_eq!(stats.colour_bytes, 0);
        assert_eq!(stats.resident_bytes, 0, "an empty map charged the volume for itself");
    }

    #[test]
    fn painting_a_voxel_and_reading_it_back_agree() {
        let mut field = ColourField::default();
        let cell = IVec3::new(1, 2, 3);
        field.write(cell, 3);
        assert_eq!(field.at(cell), 3);
        // Its neighbour is untouched.
        assert_eq!(field.at(cell + IVec3::X), UNPAINTED);
    }

    #[test]
    fn painting_slot_zero_where_nothing_is_painted_allocates_nothing() {
        let mut field = ColourField::default();
        field.write(IVec3::new(9, 9, 9), UNPAINTED);
        assert!(field.is_empty(), "an unpainted write allocated a brick");
    }

    #[test]
    fn writing_what_a_tile_already_holds_does_not_promote_it() {
        let mut field = ColourField::default();
        let coord = BrickCoord::containing(IVec3::ZERO);
        field.insert(coord, ColourBrick::Uniform(2));

        field.write(IVec3::new(1, 1, 1), 2);
        let mut stats = VolumeStats::default();
        field.add_to_stats(&mut stats);
        assert_eq!(stats.colour_dense_bricks, 0, "a no-op write cost 32 KB");
        assert_eq!(stats.colour_bytes, 0);

        // A different slot does promote it, and only then.
        field.write(IVec3::new(1, 1, 1), 5);
        let mut stats = VolumeStats::default();
        field.add_to_stats(&mut stats);
        assert_eq!(stats.colour_dense_bricks, 1);
        assert_eq!(stats.colour_bytes, BRICK_VOXELS);
    }

    /// The correction that must not be re-lost: the obvious predicate selects
    /// every voxel in the volume, because every distance is clamped into the
    /// band before it is ever stored.
    #[test]
    fn the_paint_predicate_excludes_the_saturated_interior() {
        assert!(ColourField::paintable(0.0), "the surface itself must be paintable");
        assert!(ColourField::paintable(NARROW_BAND - 0.5));
        assert!(ColourField::paintable(-NARROW_BAND + 0.5));

        // The two saturated values, which are what a solid interior and open
        // space actually store, and which are 86.8% of a real dense import.
        assert!(!ColourField::paintable(INSIDE), "a solid interior voxel was paintable");
        assert!(!ColourField::paintable(OUTSIDE), "an empty exterior voxel was paintable");

        // And the vacuous form really is vacuous, which is why it is not used.
        assert!(INSIDE.abs() <= NARROW_BAND && OUTSIDE.abs() <= NARROW_BAND);
    }

    #[test]
    fn a_brick_painted_all_one_slot_collapses_to_a_tile() {
        let mut field = ColourField::default();
        field.insert(BrickCoord::containing(IVec3::ZERO), ColourBrick::dense_filled(4));
        field.collapse();
        let mut stats = VolumeStats::default();
        field.add_to_stats(&mut stats);
        assert_eq!(stats.colour_bricks, 1, "the entry was dropped rather than collapsed");
        assert_eq!(stats.colour_dense_bricks, 0, "a uniform brick kept its allocation");
        assert_eq!(field.at(IVec3::new(2, 2, 2)), 4, "collapsing changed what it holds");
    }

    #[test]
    fn a_brick_painted_back_to_nothing_is_dropped_entirely() {
        let mut field = ColourField::default();
        field.insert(BrickCoord::containing(IVec3::ZERO), ColourBrick::dense_filled(UNPAINTED));
        field.collapse();
        assert!(field.is_empty(), "an all-unpainted brick kept an entry");
    }

    #[test]
    fn the_revision_counts_up_so_a_cache_can_compare_one_number() {
        let mut field = ColourField::default();
        let before = field.revision();
        field.write(IVec3::ZERO, 1);
        assert!(field.revision() > before);
        // A transform carries it forward rather than resetting it.
        assert!(field.translated(IVec3::X).revision() > field.revision());
    }

    #[test]
    fn slots_at_cells_reads_the_brick_and_its_border() {
        let mut field = ColourField::default();
        let coord = BrickCoord::containing(IVec3::ZERO);
        field.write(IVec3::new(0, 0, 0), 7);
        // A cell one voxel outside this brick, which is what the apron reaches.
        let outside = coord.origin() - IVec3::X;
        field.write(outside, 9);

        let cells = vec![IVec3::new(0, 0, 0), outside, IVec3::new(5, 5, 5)];
        let mut out = Vec::new();
        field.slots_at_cells(coord, &cells, &mut out);
        assert_eq!(out, vec![7, 9, UNPAINTED], "the border fell back to the wrong brick");
    }

    /// The rollback the module documentation promises: a brick the box reaches
    /// but the predicate refuses everywhere is not left as a 32 KB allocation.
    #[test]
    fn a_brick_nothing_could_paint_is_not_promoted() {
        let mut field = ColourField::default();
        let coord = BrickCoord::containing(IVec3::ZERO);
        let lo = coord.origin();
        let hi = coord.max_voxel();

        let outcome = field.edit_brick(coord, lo, hi, 1.0, &|_| false, &|_, _, _| 3);
        assert!(matches!(outcome, ColourEdit::Unchanged));
        assert!(field.is_empty(), "a brick no voxel of which was paintable got an entry");

        // The same when the edit writes what is already there.
        field.insert(coord, ColourBrick::Uniform(3));
        let outcome = field.edit_brick(coord, lo, hi, 1.0, &|_| true, &|_, _, _| 3);
        assert!(matches!(outcome, ColourEdit::Unchanged));
        assert!(
            matches!(field.brick(coord), Some(ColourBrick::Uniform(3))),
            "a no-op edit left the tile promoted"
        );
    }

    #[test]
    fn edit_brick_offers_only_the_paintable_voxels_and_reports_the_prior() {
        let mut field = ColourField::default();
        let coord = BrickCoord::containing(IVec3::ZERO);
        let lo = coord.origin();
        let hi = coord.max_voxel();
        // Only index 5 -- voxel (5, 0, 0) -- may be painted.
        let outcome = field.edit_brick(coord, lo, hi, 1.0, &|index| index == 5, &|_, _, _| 2);
        let ColourEdit::Changed(prior) = outcome else { panic!("the paintable voxel changed") };
        assert!(prior.is_none(), "the brick did not exist before, and the prior must say so");
        assert_eq!(field.at(IVec3::new(5, 0, 0)), 2);
        assert_eq!(field.at(IVec3::new(6, 0, 0)), UNPAINTED, "a refused voxel was painted");

        // A second edit reports the brick as it stood after the first.
        let outcome = field.edit_brick(coord, lo, hi, 1.0, &|_| true, &|_, _, _| 7);
        let ColourEdit::Changed(Some(prior)) = outcome else { panic!("the prior went missing") };
        assert_eq!(prior.is_collapsible(), None, "the prior should be the dense brick");
    }

    #[test]
    fn restoring_a_brick_puts_back_exactly_what_was_recorded() {
        let mut field = ColourField::default();
        let coord = BrickCoord::containing(IVec3::ZERO);
        field.write(IVec3::new(1, 1, 1), 4);
        let before = field.revision();
        field.restore_brick(coord, None);
        assert!(field.is_empty(), "restoring `None` should remove the entry");
        assert!(
            field.revision() > before,
            "a restore must move the revision, or a cache keeps the old slots"
        );
        field.restore_brick(coord, Some(ColourBrick::Uniform(9)));
        assert_eq!(field.at(IVec3::new(30, 30, 30)), 9);
    }

    /// The far face of a source brick at a refinement past 2x: the cells
    /// that round back to source cell 63 run to 63.5 * ratio, and a pad of one
    /// voxel stopped short of them.
    #[test]
    fn a_fine_resample_keeps_the_paint_on_a_bricks_far_face() {
        let mut field = ColourField::default();
        field.write(IVec3::new(63, 0, 0), 3);
        let fine = field.resampled(1.0, 0.104);
        for x in 601..=610 {
            let cell = IVec3::new(x, 0, 0);
            let source = (cell.as_vec3() * 0.104).round().as_ivec3();
            let expected = if source.x == 63 { 3 } else { UNPAINTED };
            assert_eq!(fine.at(cell), expected, "destination {x} reads from source {}", source.x);
        }
        assert_eq!(gather_pad(1.0 / 0.104), 5);
        assert_eq!(gather_pad(1.0), 1, "no refinement still pads by the one voxel it always did");
    }

    #[test]
    fn a_union_takes_incoming_paint_only_where_the_field_admits_it_and_records() {
        let mut ours = ColourField::default();
        ours.write(IVec3::new(1, 0, 0), 1);
        let mut theirs = ColourField::default();
        theirs.write(IVec3::new(1, 0, 0), 3);
        theirs.write(IVec3::new(2, 0, 0), 3);

        let mut recorded = Vec::new();
        let changed = ours.union_from(
            &theirs,
            |_, index| index != brick_index(1, 0, 0),
            |coord, prior| recorded.push((coord, prior)),
        );
        assert_eq!(
            ours.at(IVec3::new(1, 0, 0)),
            1,
            "a slot the field does not admit overwrote ours"
        );
        assert_eq!(ours.at(IVec3::new(2, 0, 0)), 3, "an admitted slot did not arrive");
        assert_eq!(changed.len(), 1);
        assert_eq!(recorded.len(), 1, "the prior was not handed to the recorder");
        assert!(
            matches!(&recorded[0].1, Some(ColourBrick::Dense(_))),
            "the prior recorded is not the brick as it stood"
        );
    }

    #[test]
    fn an_empty_field_fills_the_attribute_run_with_zeros() {
        let field = ColourField::default();
        let cells = vec![IVec3::ZERO; 4];
        let mut out = vec![255];
        field.slots_at_cells(BrickCoord::containing(IVec3::ZERO), &cells, &mut out);
        assert_eq!(out, vec![255, 0, 0, 0, 0], "the fast path wrote the wrong length or value");
    }
}
