// SPDX-License-Identifier: AGPL-3.0-only

//! Brick geometry, storage and the constants that define the volume lattice.

use glam::IVec3;

/// Voxels along one edge of a brick.
///
/// This is a tuning knob, not a magic number. Larger bricks mean fewer
/// allocations but more wasted work on small edits. Every call site must refer
/// to this constant so the whole engine moves together when it changes.
pub const BRICK_DIM: usize = 32;

/// Voxels in a brick.
pub const BRICK_VOXELS: usize = BRICK_DIM * BRICK_DIM * BRICK_DIM;

/// Edge length of a brick's sample array once the one voxel halo is included.
///
/// See [`crate::ApronBuffer`] for why the halo exists.
pub const APRON_DIM: usize = BRICK_DIM + 2;

/// Samples in an apron buffer.
pub const APRON_VOXELS: usize = APRON_DIM * APRON_DIM * APRON_DIM;

/// Narrow band half width, in voxels.
///
/// Distance values are clamped to plus or minus this. Beyond the band the
/// field carries no detail, which is what lets empty and solid regions be
/// stored as a single number instead of an array.
pub const NARROW_BAND: f32 = 3.0;

/// Distance value for a voxel known to be outside the surface, far from it.
pub const OUTSIDE: f32 = NARROW_BAND;

/// Distance value for a voxel known to be inside the surface, far from it.
pub const INSIDE: f32 = -NARROW_BAND;

/// Integer coordinate of a brick in the lattice.
///
/// Brick `c` covers world voxels `c * BRICK_DIM` through
/// `c * BRICK_DIM + BRICK_DIM - 1` inclusive, on every axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrickCoord(pub IVec3);

// glam's IVec3 has no ordering, but a deterministic order is useful for stable
// iteration and for readable test failures, so derive one from the components.
impl Ord for BrickCoord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0.z, self.0.y, self.0.x).cmp(&(other.0.z, other.0.y, other.0.x))
    }
}

impl PartialOrd for BrickCoord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl BrickCoord {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// The brick containing a given world voxel.
    ///
    /// Uses floor division so negative coordinates land in the correct brick.
    /// Plain integer division truncates toward zero, which would make bricks
    /// -1 and 0 share a coordinate and silently corrupt the lattice.
    #[inline]
    pub fn containing(voxel: IVec3) -> Self {
        Self(voxel.div_euclid(IVec3::splat(BRICK_DIM as i32)))
    }

    /// World voxel coordinate of this brick's minimum corner.
    #[inline]
    pub fn origin(self) -> IVec3 {
        self.0 * BRICK_DIM as i32
    }

    /// Inclusive world voxel coordinate of this brick's maximum corner.
    #[inline]
    pub fn max_voxel(self) -> IVec3 {
        self.origin() + IVec3::splat(BRICK_DIM as i32 - 1)
    }
}

/// Index of a voxel within a brick's dense array.
///
/// X is the fastest varying axis, so a run of X values is contiguous and can
/// be copied with a single memcpy. The apron gather relies on this.
#[inline]
pub const fn brick_index(x: usize, y: usize, z: usize) -> usize {
    x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM
}

/// Index of a sample within an apron buffer. X is the fastest varying axis.
#[inline]
pub const fn apron_index(x: usize, y: usize, z: usize) -> usize {
    x + y * APRON_DIM + z * APRON_DIM * APRON_DIM
}

/// The contents of one brick.
///
/// `Uniform` is the VDB tile concept and it is not an optimisation, it is
/// required for correctness. A brick deep inside a solid holds no detail, but
/// its value still has to read as negative when a neighbouring brick gathers
/// its apron. If absent bricks simply defaulted to "outside", every brick on
/// the inner boundary of a solid would grow a spurious surface.
#[derive(Debug, Clone)]
pub enum Brick {
    /// Every voxel holds this value. Costs no allocation.
    Uniform(f32),
    /// Per voxel distance values, indexed by [`brick_index`].
    Dense(Box<[f32; BRICK_VOXELS]>),
}

impl Brick {
    /// A dense brick with every voxel set to `value`.
    ///
    /// Allocates on the heap directly rather than building the array on the
    /// stack, which at 128 KB would overflow a default thread stack.
    pub fn dense_filled(value: f32) -> Self {
        let data: Box<[f32]> = vec![value; BRICK_VOXELS].into_boxed_slice();
        // The length is BRICK_VOXELS by construction, so this cannot fail.
        let data: Box<[f32; BRICK_VOXELS]> = data.try_into().expect("length is BRICK_VOXELS");
        Brick::Dense(data)
    }

    /// Promote a uniform brick to dense so individual voxels can be written.
    /// Dense bricks are returned unchanged.
    pub fn make_dense(&mut self) -> &mut [f32; BRICK_VOXELS] {
        if let Brick::Uniform(value) = *self {
            *self = Brick::dense_filled(value);
        }
        match self {
            Brick::Dense(data) => data,
            Brick::Uniform(_) => unreachable!("just promoted to dense"),
        }
    }

    /// Value at a voxel local to this brick. Coordinates must be in range.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        match self {
            Brick::Uniform(value) => *value,
            Brick::Dense(data) => data[brick_index(x, y, z)],
        }
    }

    /// Heap bytes this brick holds, excluding the enum itself.
    #[inline]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Brick::Uniform(_) => 0,
            Brick::Dense(_) => BRICK_VOXELS * size_of::<f32>(),
        }
    }

    /// True if every voxel holds the same value, so the brick can be collapsed
    /// back to a tile and its allocation released.
    pub fn is_collapsible(&self) -> Option<f32> {
        match self {
            Brick::Uniform(value) => Some(*value),
            Brick::Dense(data) => {
                let first = data[0];
                // Only fully saturated bricks are worth collapsing. A brick of
                // uniform mid band values would be surface adjacent and about
                // to be written again.
                if (first == OUTSIDE || first == INSIDE) && data.iter().all(|v| *v == first) {
                    Some(first)
                } else {
                    None
                }
            }
        }
    }
}

/// A brick as undo history keeps it: run length encoded, never read directly.
///
/// **Measured before it was written.** On a 3,100,109 triangle import at
/// 0.168 mm, the 4,265 dense bricks hold 533.1 MB and **86.8% of their voxels
/// are saturated** at `INSIDE` or `OUTSIDE` -- a narrow band field is a thin
/// shell of varying distance wrapped in two enormous constants. Encoded as
/// runs the same bricks come to 121.5 MB, **4.39x smaller**, which is what buys
/// the undo depth: the stroke budget holds four times the strokes for the same
/// bytes, and a body that could not be moved at all now fits with room over.
///
/// **History only, never the live volume.** Sculpting needs random access to
/// every voxel and a run table cannot give it without a search; the live
/// [`Brick`] is untouched and no edit path pays anything for this. The cost is
/// one pass over a brick when it enters history and one when it leaves, which
/// is the same order as the memcpy it replaces.
///
/// Runs are `(f32, u16)` packed as six bytes with no padding -- a `struct` of
/// those two would be eight, aligned, and throw away a quarter of the saving.
/// A run is capped at `u16::MAX`, which is twice `BRICK_VOXELS`, so the cap can
/// never split a run that a uniform brick would not have collapsed already.
#[derive(Debug, Clone, PartialEq)]
pub enum StoredBrick {
    /// Every voxel holds this value; the tile case, free as it is live.
    Uniform(f32),
    /// Six bytes per run: little endian `f32` value, little endian `u16` count.
    Runs(Box<[u8]>),
    /// Verbatim, for a brick the run table would make BIGGER.
    ///
    /// **A run is six bytes and a voxel is four**, so a brick where no two
    /// neighbours share a value encodes to 1.5x its own size. Real geometry
    /// never does that -- 86.8% of voxels are saturated -- but "never on the
    /// data I measured" is not a bound, and an encoder that can lose is one
    /// that will lose on the one model that matters. Whichever is smaller
    /// wins, so this is at worst exactly what it replaced.
    Dense(Box<[f32; BRICK_VOXELS]>),
}

/// Bytes one run occupies: an `f32` and a `u16`, packed.
const RUN_BYTES: usize = 6;

impl StoredBrick {
    /// Encode a live brick for history.
    pub fn encode(brick: &Brick) -> Self {
        let values = match brick {
            Brick::Uniform(value) => return StoredBrick::Uniform(*value),
            Brick::Dense(values) => values,
        };
        let mut runs: Vec<u8> = Vec::with_capacity(1024);
        let mut current = values[0];
        let mut count: u32 = 1;
        // Compared by BITS, not by value: two NaNs that differ in payload are
        // different voxels to a bit for bit restore, and `-0.0 == 0.0` would
        // silently merge two values a later read can tell apart.
        for value in values.iter().skip(1) {
            if value.to_bits() == current.to_bits() && count < u16::MAX as u32 {
                count += 1;
            } else {
                runs.extend_from_slice(&current.to_le_bytes());
                runs.extend_from_slice(&(count as u16).to_le_bytes());
                current = *value;
                count = 1;
            }
        }
        runs.extend_from_slice(&current.to_le_bytes());
        runs.extend_from_slice(&(count as u16).to_le_bytes());
        if runs.len() >= BRICK_VOXELS * size_of::<f32>() {
            return StoredBrick::Dense(Box::new(**values));
        }
        runs.shrink_to_fit();
        StoredBrick::Runs(runs.into_boxed_slice())
    }

    /// Rebuild the live brick this was encoded from, bit for bit.
    pub fn decode(&self) -> Brick {
        let runs = match self {
            StoredBrick::Uniform(value) => return Brick::Uniform(*value),
            StoredBrick::Dense(values) => return Brick::Dense(Box::new(**values)),
            StoredBrick::Runs(runs) => runs,
        };
        let mut values: Vec<f32> = Vec::with_capacity(BRICK_VOXELS);
        let (runs, _) = runs.as_chunks::<RUN_BYTES>();
        for run in runs {
            let value = f32::from_le_bytes([run[0], run[1], run[2], run[3]]);
            let count = u16::from_le_bytes([run[4], run[5]]) as usize;
            values.resize(values.len() + count, value);
        }
        debug_assert_eq!(values.len(), BRICK_VOXELS, "a run table decoded to the wrong length");
        // A table that decoded short would index out of bounds later, which is
        // a corrupted undo rather than a panic anyone can act on. Pad rather
        // than trust: `OUTSIDE` is what an absent voxel already reads as.
        values.resize(BRICK_VOXELS, OUTSIDE);
        let data: Box<[f32]> = values.into_boxed_slice();
        let data: Box<[f32; BRICK_VOXELS]> = data.try_into().expect("length is BRICK_VOXELS");
        Brick::Dense(data)
    }

    /// What this costs the budget that holds it.
    pub fn heap_bytes(&self) -> usize {
        match self {
            StoredBrick::Uniform(_) => 0,
            StoredBrick::Runs(runs) => runs.len(),
            StoredBrick::Dense(_) => BRICK_VOXELS * size_of::<f32>(),
        }
    }
}

impl Brick {
    /// What this brick WOULD cost in history, without encoding it.
    ///
    /// For the planners that have to quote a gesture's undo weight before
    /// running it -- `merge` prices a whole body this way. Counting runs is the
    /// same pass `StoredBrick::encode` makes and allocates nothing, and it has
    /// to be the encoder's own arithmetic rather than a ratio: a 4.39x average
    /// applied to a brick that happens to be all shell would under-quote it,
    /// and the budget it is quoting against is the one that decides whether the
    /// gesture is offered at all.
    pub fn encoded_bytes(&self) -> usize {
        let values = match self {
            Brick::Uniform(_) => return 0,
            Brick::Dense(values) => values,
        };
        let mut runs = 1usize;
        let mut count: u32 = 1;
        let mut current = values[0];
        for value in values.iter().skip(1) {
            if value.to_bits() == current.to_bits() && count < u16::MAX as u32 {
                count += 1;
            } else {
                runs += 1;
                current = *value;
                count = 1;
            }
        }
        (runs * RUN_BYTES).min(BRICK_VOXELS * size_of::<f32>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brick_containing_handles_negative_voxels() {
        assert_eq!(BrickCoord::containing(IVec3::new(0, 0, 0)), BrickCoord::new(0, 0, 0));
        assert_eq!(BrickCoord::containing(IVec3::new(31, 0, 0)), BrickCoord::new(0, 0, 0));
        assert_eq!(BrickCoord::containing(IVec3::new(32, 0, 0)), BrickCoord::new(1, 0, 0));
        // The case truncating division gets wrong.
        assert_eq!(BrickCoord::containing(IVec3::new(-1, 0, 0)), BrickCoord::new(-1, 0, 0));
        assert_eq!(BrickCoord::containing(IVec3::new(-32, 0, 0)), BrickCoord::new(-1, 0, 0));
        assert_eq!(BrickCoord::containing(IVec3::new(-33, 0, 0)), BrickCoord::new(-2, 0, 0));
    }

    #[test]
    fn brick_origin_round_trips_through_containing() {
        for c in [BrickCoord::new(0, 0, 0), BrickCoord::new(3, -2, 7), BrickCoord::new(-4, -4, -4)]
        {
            assert_eq!(BrickCoord::containing(c.origin()), c);
            assert_eq!(BrickCoord::containing(c.max_voxel()), c);
        }
    }

    #[test]
    fn uniform_bricks_cost_no_heap() {
        assert_eq!(Brick::Uniform(OUTSIDE).heap_bytes(), 0);
        assert_eq!(Brick::dense_filled(0.0).heap_bytes(), 128 * 1024);
    }

    #[test]
    fn make_dense_preserves_the_tile_value() {
        let mut brick = Brick::Uniform(INSIDE);
        let data = brick.make_dense();
        assert!(data.iter().all(|v| *v == INSIDE));
    }
}
