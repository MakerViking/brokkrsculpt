// SPDX-License-Identifier: AGPL-3.0-only

//! Helpers shared between the unit tests of several modules.
//!
//! Only compiled under `cfg(test)`, so nothing here reaches a release binary.
//!
//! The integration tests under `tests/` cannot see this module -- they link the
//! crate as an ordinary dependency, compiled without `cfg(test)` -- so
//! `tests/hostile_meshes.rs` carries its own copy of [`Noise`]. That is a
//! duplication the compilation model forces rather than one worth removing.

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
