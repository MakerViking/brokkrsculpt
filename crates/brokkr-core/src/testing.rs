// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers shared between the unit tests of several modules.
//!
//! Only compiled under `cfg(test)`, so nothing here reaches a release binary.
//!
//! The integration tests under `tests/` cannot see this module -- they link the
//! crate as an ordinary dependency, compiled without `cfg(test)` -- so
//! `tests/hostile_meshes.rs` carries its own copy of [`Noise`]. That is a
//! duplication the compilation model forces rather than one worth removing.

use crate::brick::{Brick, BrickCoord};
use crate::volume::Volume;

/// A seeded xorshift.
///
/// Hand rolled rather than pulled in: `rand` is not a dependency of this
/// workspace, and adding one to shuffle bytes in a test would be a poor trade.
/// The point of seeding it is that a failure is reproducible from the seed
/// printed alongside it, on any machine and in CI, which is what makes a
/// randomised test usable as a gate.
pub struct Noise(pub u64);

impl Noise {
    /// A generator from `seed`, spread through a multiply so that neighbouring
    /// seeds do not produce neighbouring streams.
    pub fn seeded(seed: u64) -> Self {
        Self((seed | 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `0..limit`, or zero when the range is empty.
    pub fn below(&mut self, limit: usize) -> usize {
        if limit == 0 { 0 } else { (self.next() % limit as u64) as usize }
    }

    pub fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

/// Assert two volumes hold the same field, brick for brick.
///
/// Compared through the storage rather than by sampling every voxel,
/// which makes it a memcmp per brick instead of a hash lookup per voxel.
/// The representation has to match too: a brick the unskipped path made
/// dense and then found it had not changed is rolled back to the tile it
/// was, so a difference there would mean one path is leaving 128 KB behind.
///
/// Shared rather than copied because the plane cut needs exactly the
/// comparison the brush's skipping tests need -- "bit-identical" is the same
/// claim whichever operation is making it, and two copies of it would drift.
pub fn assert_same_field(a: &Volume, b: &Volume, what: &str) {
    let mut left: Vec<BrickCoord> = a.brick_coords().collect();
    let mut right: Vec<BrickCoord> = b.brick_coords().collect();
    left.sort();
    right.sort();
    assert_eq!(left, right, "{what}: different bricks are stored");

    for coord in left {
        match (a.brick(coord), b.brick(coord)) {
            (Some(Brick::Uniform(x)), Some(Brick::Uniform(y))) => {
                assert_eq!(x, y, "{what}: tile {coord:?} differs");
            }
            (Some(Brick::Dense(x)), Some(Brick::Dense(y))) => {
                assert!(x[..] == y[..], "{what}: brick {coord:?} differs");
            }
            (x, y) => panic!(
                "{what}: brick {coord:?} is stored differently: {:?} against {:?}",
                x.map(|brick| matches!(brick, Brick::Dense(_))),
                y.map(|brick| matches!(brick, Brick::Dense(_))),
            ),
        }
    }
}
