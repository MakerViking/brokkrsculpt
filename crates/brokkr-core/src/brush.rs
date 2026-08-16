// SPDX-License-Identifier: AGPL-3.0-or-later

//! Brushes: falloff weighted modification of the distance field.

use glam::Vec3;

use crate::volume::Volume;

/// Which way a brush moves the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushDirection {
    /// Add clay: move the surface out along its normal.
    Add,
    /// Remove clay: move the surface in.
    Subtract,
}

impl BrushDirection {
    #[inline]
    fn sign(self) -> f32 {
        match self {
            // Distance is negative inside, so making a value more negative
            // grows the solid.
            BrushDirection::Add => -1.0,
            BrushDirection::Subtract => 1.0,
        }
    }
}

/// The draw brush: displaces the surface within a sphere of influence.
#[derive(Debug, Clone, Copy)]
pub struct DrawBrush {
    /// Radius of influence in world units.
    pub radius: f32,
    /// Fraction of the radius the centre of the brush displaces per
    /// application. Keep well below 1 or a single application will punch
    /// through the narrow band.
    pub strength: f32,
}

impl Default for DrawBrush {
    fn default() -> Self {
        Self { radius: 1.0, strength: 0.15 }
    }
}

impl DrawBrush {
    /// Smooth radial falloff, 1 at the centre and 0 at the rim, with zero
    /// derivative at both ends so repeated strokes do not leave a visible
    /// ring at the brush edge.
    #[inline]
    pub fn falloff(normalised_distance: f32) -> f32 {
        let t = (1.0 - normalised_distance).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Apply one stamp of the brush centred on a world space point.
    ///
    /// Work is proportional to the brush volume, never to the size of the
    /// model.
    ///
    /// The edit subtracts a falloff weighted amount from the field rather than
    /// re-deriving a true distance function. That is the standard approach and
    /// it is cheap, but it does not preserve the eikonal property: after many
    /// overlapping stamps the gradient magnitude drifts from 1 and the surface
    /// moves slightly less per stamp than the nominal displacement. Clamping to
    /// the narrow band bounds the drift. A renormalisation pass belongs with
    /// the GPU rewrite, not here.
    pub fn apply(&self, volume: &mut Volume, centre: Vec3, direction: BrushDirection) {
        if self.radius <= 0.0 || self.strength <= 0.0 {
            return;
        }

        let voxel_size = volume.voxel_size();
        let radius = self.radius;
        let inv_radius = 1.0 / radius;
        // Stored values are in voxels, so the displacement has to be too.
        let peak = self.strength * (radius / voxel_size) * direction.sign();
        let extent = Vec3::splat(radius);

        volume.edit_box(centre - extent, centre + extent, |position, value| {
            let distance = position.distance(centre) * inv_radius;
            if distance >= 1.0 {
                return value;
            }
            value + peak * Self::falloff(distance)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::{INSIDE, OUTSIDE};

    #[test]
    fn falloff_is_one_at_the_centre_and_zero_at_the_rim() {
        assert_eq!(DrawBrush::falloff(0.0), 1.0);
        assert_eq!(DrawBrush::falloff(1.0), 0.0);
        assert_eq!(DrawBrush::falloff(1.5), 0.0);
        assert!((DrawBrush::falloff(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn falloff_is_monotonic() {
        let mut previous = f32::INFINITY;
        for step in 0..=100 {
            let value = DrawBrush::falloff(step as f32 / 100.0);
            assert!(value <= previous + 1e-6, "falloff rose at {step}");
            previous = value;
        }
    }

    #[test]
    fn adding_clay_pushes_the_field_negative() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(64.0), 20.0);

        let surface = Vec3::new(64.0 + 20.0, 64.0, 64.0);
        let before = volume.sample_world(surface);
        let brush = DrawBrush { radius: 8.0, strength: 0.2 };
        brush.apply(&mut volume, surface, BrushDirection::Add);
        let after = volume.sample_world(surface);

        assert!(after < before, "add should lower the distance: {before} then {after}");
        assert!((INSIDE..=OUTSIDE).contains(&after), "value left the narrow band");
    }

    #[test]
    fn subtracting_clay_pushes_the_field_positive() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(64.0), 20.0);

        let surface = Vec3::new(64.0 + 20.0, 64.0, 64.0);
        let before = volume.sample_world(surface);
        let brush = DrawBrush { radius: 8.0, strength: 0.2 };
        brush.apply(&mut volume, surface, BrushDirection::Subtract);
        assert!(volume.sample_world(surface) > before);
    }

    #[test]
    fn a_stroke_far_from_the_model_does_not_touch_existing_bricks() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(64.0), 20.0);
        let before = volume.brick_count();

        let brush = DrawBrush { radius: 4.0, strength: 0.2 };
        brush.apply(&mut volume, Vec3::splat(2000.0), BrushDirection::Add);

        // It allocates where it painted, but the model's bricks are untouched
        // and the work did not scale with them.
        assert!(volume.brick_count() > before);
        assert_eq!(volume.sample_voxel(glam::IVec3::splat(64)), INSIDE);
    }
}
