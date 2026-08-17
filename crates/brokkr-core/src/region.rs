// SPDX-License-Identifier: AGPL-3.0-or-later

//! A read only snapshot of the field over a small box of voxels.
//!
//! Several brushes are not per voxel functions. Smooth needs a neighbour
//! average, pinch needs the difference between a value and that average, and
//! draw needs the local gradient. Reading those straight from the volume while
//! writing to it would mean each voxel sees a mixture of old and new values,
//! and the result would depend on iteration order.
//!
//! So a stamp runs in two phases: copy the affected box out, then write back
//! using only the copy. The box is grown by one voxel on every side so a
//! gradient or neighbour average is available at every voxel of the core box
//! without a bounds check at the rim.

use glam::{IVec3, Vec3};

/// Reusable storage for one stamp's worth of field values.
#[derive(Debug, Default, Clone)]
pub struct FieldRegion {
    /// Inclusive minimum voxel of the stored box, already grown by one.
    lo: IVec3,
    /// Inclusive maximum voxel of the stored box.
    hi: IVec3,
    size: IVec3,
    values: Vec<f32>,
}

impl FieldRegion {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resize to cover an inclusive voxel box, keeping the allocation when it
    /// is already big enough. Called once per stamp, so it must not allocate in
    /// the steady state of a stroke.
    pub(crate) fn resize(&mut self, lo: IVec3, hi: IVec3) -> &mut [f32] {
        self.lo = lo;
        self.hi = hi;
        self.size = hi - lo + IVec3::ONE;
        let count = (self.size.x * self.size.y * self.size.z) as usize;
        self.values.clear();
        self.values.resize(count, 0.0);
        &mut self.values
    }

    /// Value at a world voxel coordinate.
    ///
    /// Coordinates outside the stored box clamp to its edge rather than
    /// panicking. Callers stay inside the core box, so a clamp only ever
    /// happens at the outer rim of the padding.
    #[inline]
    pub fn get(&self, voxel: IVec3) -> f32 {
        let local = (voxel - self.lo).clamp(IVec3::ZERO, self.size - IVec3::ONE);
        let index = local.x + local.y * self.size.x + local.z * self.size.x * self.size.y;
        self.values[index as usize]
    }

    /// Mean of the six axis neighbours, which is the smoothing kernel.
    #[inline]
    pub fn neighbour_average(&self, voxel: IVec3) -> f32 {
        let sum = self.get(voxel + IVec3::X)
            + self.get(voxel - IVec3::X)
            + self.get(voxel + IVec3::Y)
            + self.get(voxel - IVec3::Y)
            + self.get(voxel + IVec3::Z)
            + self.get(voxel - IVec3::Z);
        sum / 6.0
    }

    /// Trilinearly interpolated value at a fractional voxel coordinate.
    ///
    /// Used by brushes that warp the field rather than adding to it. The
    /// coordinate must stay within one voxel of the core box, which is what the
    /// padding is for.
    #[inline]
    pub fn sample(&self, voxel: Vec3) -> f32 {
        let base = voxel.floor();
        let frac = voxel - base;
        let b = base.as_ivec3();

        let corner = |dx: i32, dy: i32, dz: i32| self.get(b + IVec3::new(dx, dy, dz));
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;

        let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), frac.x);
        let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), frac.x);
        let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), frac.x);
        let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), frac.x);

        lerp(lerp(x00, x10, frac.y), lerp(x01, x11, frac.y), frac.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a region holding a linear ramp along X, whose gradient is known.
    fn ramp() -> FieldRegion {
        let mut region = FieldRegion::new();
        let lo = IVec3::splat(-2);
        let hi = IVec3::splat(2);
        let size = hi - lo + IVec3::ONE;
        let values = region.resize(lo, hi);
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let index = x + y * size.x + z * size.x * size.y;
                    values[index as usize] = (lo.x + x) as f32;
                }
            }
        }
        region
    }

    #[test]
    fn values_read_back_at_their_world_coordinate() {
        let region = ramp();
        assert_eq!(region.get(IVec3::new(-2, 0, 0)), -2.0);
        assert_eq!(region.get(IVec3::new(0, 1, -1)), 0.0);
        assert_eq!(region.get(IVec3::new(2, 2, 2)), 2.0);
    }

    #[test]
    fn reads_outside_the_box_clamp_to_its_edge() {
        let region = ramp();
        assert_eq!(region.get(IVec3::new(99, 0, 0)), 2.0);
        assert_eq!(region.get(IVec3::new(-99, 0, 0)), -2.0);
    }

    #[test]
    fn a_flat_field_averages_to_itself() {
        let mut region = FieldRegion::new();
        region.resize(IVec3::splat(-1), IVec3::splat(1)).fill(7.0);
        assert_eq!(region.neighbour_average(IVec3::ZERO), 7.0);
        assert_eq!(region.sample(Vec3::new(0.25, -0.5, 0.75)), 7.0);
    }

    #[test]
    fn sampling_between_voxels_interpolates() {
        let region = ramp();
        assert!((region.sample(Vec3::new(0.5, 0.0, 0.0)) - 0.5).abs() < 1.0e-6);
        assert!((region.sample(Vec3::new(-1.25, 1.0, 1.0)) + 1.25).abs() < 1.0e-6);
        // Exactly on a voxel it agrees with the integer lookup.
        assert_eq!(region.sample(Vec3::new(1.0, 0.0, 0.0)), region.get(IVec3::new(1, 0, 0)));
    }

    #[test]
    fn the_neighbour_average_of_a_ramp_is_its_centre_value() {
        // A linear field is its own average, which is what makes smoothing a
        // no op on flat and evenly sloped surfaces.
        let region = ramp();
        assert!((region.neighbour_average(IVec3::ZERO) - 0.0).abs() < 1.0e-6);
        assert!((region.neighbour_average(IVec3::new(1, 0, 0)) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn resizing_keeps_the_allocation() {
        let mut region = FieldRegion::new();
        region.resize(IVec3::splat(-5), IVec3::splat(5));
        let capacity = region.values.capacity();
        region.resize(IVec3::splat(-3), IVec3::splat(3));
        assert_eq!(region.values.capacity(), capacity, "a smaller box should reuse the buffer");
    }
}
