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
//! silently drop colour. That sweep is owed either way -- see `reads_the_colour`
//! in the transform modules -- and it does not buy back the undo arithmetic.
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

use rustc_hash::FxHashMap;

use crate::brick::{BRICK_DIM, BRICK_VOXELS, BrickCoord, brick_index};
use crate::volume::VolumeStats;
use glam::IVec3;

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

    /// The one slot the whole brick holds, or `None` when it carries detail.
    #[inline]
    pub(crate) fn fill(&self) -> Option<u8> {
        match self {
            ColourSlab::Unpainted => Some(UNPAINTED),
            ColourSlab::Uniform(slot) => Some(*slot),
            ColourSlab::Dense(_) => None,
        }
    }
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

    #[test]
    fn an_empty_field_fills_the_attribute_run_with_zeros() {
        let field = ColourField::default();
        let cells = vec![IVec3::ZERO; 4];
        let mut out = vec![255];
        field.slots_at_cells(BrickCoord::containing(IVec3::ZERO), &cells, &mut out);
        assert_eq!(out, vec![255, 0, 0, 0, 0], "the fast path wrote the wrong length or value");
    }
}
