// SPDX-License-Identifier: AGPL-3.0-or-later

//! Brushes: falloff weighted modification of the distance field.
//!
//! Every brush reads from a snapshot of the affected box taken before any
//! writing starts, so each voxel sees the same starting field and the result
//! does not depend on which voxel happened to be visited first. See
//! [`crate::region`].
//!
//! # Working in an implicit field
//!
//! A mesh sculptor moves vertices. There are no vertices here, so each brush is
//! whatever operation on the distance field produces the same visible effect.
//! They fall into two groups.
//!
//! Three of them blend the value toward a target, which is inherently stable
//! because the target is itself a legal distance:
//!
//! - Smooth blends toward the average of the neighbouring values.
//! - Flatten blends toward the tangent plane under the cursor.
//! - Clay blends toward a plane held slightly outside the surface, kept to the
//!   direction that adds material, which is how clay is actually built up.
//!
//! Three of them move material, and those have to be handled with more care.
//! Inflate offsets the whole level set, which moves every point of the surface
//! along its own normal, and is the natural operation on a distance field.
//! Draw and pinch instead resample the field from a shifted position: draw
//! reads from behind along the stroke normal, which slides the patch outward,
//! and pinch reads from slightly nearer the brush axis, which squeezes a ridge
//! into a crease.
//!
//! Draw and pinch were both first written the obvious way, as a value the
//! brush adds or amplifies, and both had to be rewritten. Anything that
//! multiplies a displacement by the local gradient, or that amplifies the
//! difference from a local average, has gain above one somewhere and turns its
//! own rounding error into visible crust over the course of a stroke. Warping
//! where the field is read from cannot introduce detail that was not there.
//!
//! None of these preserve the eikonal property: after many overlapping stamps
//! the gradient magnitude drifts from 1 and the surface moves slightly less per
//! stamp than the nominal displacement. Clamping to the narrow band bounds the
//! drift, and [`MAX_STAMP_VOXELS`] keeps any single stamp small enough that the
//! field stays well formed. A renormalisation pass belongs with the GPU
//! rewrite, not here.

use glam::Vec3;

use crate::region::FieldRegion;
use crate::volume::Volume;

/// How far outside the surface the clay plane sits, as a fraction of radius.
///
/// Zero would make clay identical to flatten. Too large and it stops following
/// the form and just deposits a slab.
const CLAY_OFFSET: f32 = 0.35;

/// Most a single stamp may move the surface, in voxels.
///
/// This one is load bearing. The field only carries a real distance inside the
/// narrow band, so a stamp that displaces further than about a voxel saturates
/// everything it touches, and the next stamp has no usable field left to read.
/// The result is a crust of aliased spikes rather than a smooth push, and every
/// value in it is still legally inside the band, so nothing but looking at it
/// catches the problem.
///
/// Strokes get their reach from laying down many small stamps, not from one
/// big one, which is also what makes stroke interpolation worth having.
const MAX_STAMP_VOXELS: f32 = 1.0;

/// Displacement per stamp as a fraction of the brush radius, before the cap.
///
/// Only small brushes ever fall below the cap, which is the point: a brush one
/// voxel across should not shove the surface a whole voxel per stamp.
const STAMP_FRACTION_OF_RADIUS: f32 = 0.25;

/// How far pinch drags the field toward the brush axis per stamp, in voxels.
///
/// Kept under one voxel so the warped read stays inside the snapshot's padding.
const PINCH_PULL_VOXELS: f32 = 0.85;

/// Which way a brush moves the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushDirection {
    /// Add clay: move the surface out along its normal.
    Add,
    /// Remove clay: move the surface in.
    Subtract,
}

impl BrushDirection {
    /// Sign to apply to a distance value to grow or shrink the solid.
    ///
    /// Distance is negative inside, so making a value more negative grows it.
    #[inline]
    pub fn field_sign(self) -> f32 {
        match self {
            BrushDirection::Add => -1.0,
            BrushDirection::Subtract => 1.0,
        }
    }

    /// Sign along the surface normal: outward when adding.
    #[inline]
    pub fn outward_sign(self) -> f32 {
        match self {
            BrushDirection::Add => 1.0,
            BrushDirection::Subtract => -1.0,
        }
    }

    #[inline]
    pub fn inverted(self) -> Self {
        match self {
            BrushDirection::Add => BrushDirection::Subtract,
            BrushDirection::Subtract => BrushDirection::Add,
        }
    }
}

/// Shape of the brush's radial weighting.
///
/// Every curve is 1 at the centre, 0 at the rim and monotonic in between, so
/// swapping between them changes the shape of a stroke without changing how far
/// it reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FalloffCurve {
    /// Smoothstep. Zero slope at both ends, so repeated strokes leave no ring
    /// at the brush edge. The default for good reason.
    #[default]
    Smooth,
    /// Straight line. Predictable, with a faint edge where the slope stops.
    Linear,
    /// Concentrated at the centre, for detail work and small features.
    Sharp,
    /// Broad and flat with a quick roll off at the rim, for filling areas.
    Wide,
}

impl FalloffCurve {
    /// Weight at a distance from the centre, given as a fraction of the radius.
    #[inline]
    pub fn weight(self, normalised_distance: f32) -> f32 {
        let t = (1.0 - normalised_distance).clamp(0.0, 1.0);
        match self {
            FalloffCurve::Smooth => t * t * (3.0 - 2.0 * t),
            FalloffCurve::Linear => t,
            FalloffCurve::Sharp => t * t * t,
            FalloffCurve::Wide => {
                let inverse = 1.0 - t;
                1.0 - inverse * inverse * inverse
            }
        }
    }

    pub const ALL: [FalloffCurve; 4] =
        [FalloffCurve::Smooth, FalloffCurve::Linear, FalloffCurve::Sharp, FalloffCurve::Wide];

    pub fn label(self) -> &'static str {
        match self {
            FalloffCurve::Smooth => "Smooth",
            FalloffCurve::Linear => "Linear",
            FalloffCurve::Sharp => "Sharp",
            FalloffCurve::Wide => "Wide",
        }
    }
}

impl std::fmt::Display for FalloffCurve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Which operation a brush performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrushKind {
    /// Push the surface out along the stroke direction.
    #[default]
    Draw,
    /// Build material up toward a plane held just outside the surface.
    Clay,
    /// Blend toward the average of the surrounding field.
    Smooth,
    /// Offset the whole level set, moving every point along its own normal.
    Inflate,
    /// Squeeze the surface toward the brush axis, sharpening ridges into
    /// creases.
    Pinch,
    /// Blend toward the tangent plane under the cursor.
    Flatten,
}

impl BrushKind {
    pub const ALL: [BrushKind; 6] = [
        BrushKind::Draw,
        BrushKind::Clay,
        BrushKind::Smooth,
        BrushKind::Inflate,
        BrushKind::Pinch,
        BrushKind::Flatten,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BrushKind::Draw => "Draw",
            BrushKind::Clay => "Clay",
            BrushKind::Smooth => "Smooth",
            BrushKind::Inflate => "Inflate",
            BrushKind::Pinch => "Pinch",
            BrushKind::Flatten => "Flatten",
        }
    }

    /// Whether inverting the stroke means anything for this brush.
    ///
    /// Smooth and flatten are their own opposite: there is no such thing as
    /// unsmoothing toward a plane. Holding the invert key with them selected
    /// does nothing, which is worth saying in the interface rather than leaving
    /// the user to wonder.
    pub fn is_directional(self) -> bool {
        !matches!(self, BrushKind::Smooth | BrushKind::Flatten)
    }
}

impl std::fmt::Display for BrushKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Mirroring applied to every stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Symmetry {
    #[default]
    Off,
    /// Mirror across the world plane x = 0.
    X,
}

impl Symmetry {
    /// The mirrored twin of a stamp, or `None` when symmetry is off.
    ///
    /// A stamp landing on the mirror plane is applied twice at nearly the same
    /// place. That is deliberate: the two falloffs overlap smoothly, whereas
    /// suppressing the twin near the plane would put a visible step in the
    /// stroke strength exactly where the user is trying to work.
    pub fn mirror(self, stamp: &Stamp) -> Option<Stamp> {
        match self {
            Symmetry::Off => None,
            Symmetry::X => Some(Stamp {
                centre: Vec3::new(-stamp.centre.x, stamp.centre.y, stamp.centre.z),
                normal: Vec3::new(-stamp.normal.x, stamp.normal.y, stamp.normal.z),
                ..*stamp
            }),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Symmetry::Off => "Off",
            Symmetry::X => "X",
        }
    }
}

/// One application of a brush at a point.
#[derive(Debug, Clone, Copy)]
pub struct Stamp {
    /// Where the brush is centred, in world space.
    pub centre: Vec3,
    /// Outward surface normal at that point, used by the directional brushes
    /// and by the flatten and clay planes.
    pub normal: Vec3,
    /// Stylus pressure from 0 to 1, scaling strength. Defaults to 1, which is
    /// what a mouse always reports.
    pub pressure: f32,
    pub direction: BrushDirection,
}

impl Stamp {
    pub fn new(centre: Vec3, normal: Vec3, direction: BrushDirection) -> Self {
        Self { centre, normal, pressure: 1.0, direction }
    }

    pub fn with_pressure(mut self, pressure: f32) -> Self {
        self.pressure = pressure.clamp(0.0, 1.0);
        self
    }
}

/// Reusable working memory for stamping.
///
/// Holding one across a stroke is what keeps sculpting out of the allocator.
#[derive(Debug, Default)]
pub struct BrushScratch {
    region: FieldRegion,
}

impl BrushScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A configured brush.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brush {
    pub kind: BrushKind,
    /// Radius of influence in world units.
    pub radius: f32,
    /// How hard one stamp bites. Keep well below 1: a stroke lays down many
    /// overlapping stamps, so this is per stamp and not per stroke.
    pub strength: f32,
    pub falloff: FalloffCurve,
}

impl Default for Brush {
    fn default() -> Self {
        Self { kind: BrushKind::Draw, radius: 3.0, strength: 0.15, falloff: FalloffCurve::Smooth }
    }
}

impl Brush {
    /// Spacing between stamps along a stroke, in world units.
    ///
    /// A quarter of the radius keeps consecutive stamps heavily overlapped, so
    /// a fast drag leaves a continuous cut instead of a dotted trail. Never
    /// smaller than a voxel, or a slow drag would stamp the same voxels over
    /// and over for no visible gain.
    pub fn spacing(&self, voxel_size: f32) -> f32 {
        (self.radius * 0.25).max(voxel_size)
    }

    /// Apply one stamp, plus its mirror when symmetry is on.
    pub fn apply_symmetric(
        &self,
        volume: &mut Volume,
        stamp: &Stamp,
        symmetry: Symmetry,
        scratch: &mut BrushScratch,
    ) {
        self.apply(volume, stamp, scratch);
        if let Some(mirrored) = symmetry.mirror(stamp) {
            self.apply(volume, &mirrored, scratch);
        }
    }

    /// Apply one stamp.
    ///
    /// Work is proportional to the brush volume, never to the size of the
    /// model.
    pub fn apply(&self, volume: &mut Volume, stamp: &Stamp, scratch: &mut BrushScratch) {
        if self.radius <= 0.0 || self.strength <= 0.0 || stamp.pressure <= 0.0 {
            return;
        }

        let voxel_size = volume.voxel_size();
        let extent = Vec3::splat(self.radius);
        let (lo, hi) = volume.voxel_bounds(stamp.centre - extent, stamp.centre + extent);

        volume.snapshot(lo, hi, &mut scratch.region);
        let region = &scratch.region;

        let inverse_radius = 1.0 / self.radius;
        let gain = self.strength * stamp.pressure;
        // Displacement at full weight, in voxels, capped so one stamp can never
        // saturate the narrow band. See MAX_STAMP_VOXELS.
        let displacement =
            (self.radius / voxel_size * STAMP_FRACTION_OF_RADIUS).min(MAX_STAMP_VOXELS);
        let field_sign = stamp.direction.field_sign();

        // Reference plane for clay and flatten, in world space. Clay holds it
        // just outside the surface so material builds up to it.
        let plane_point = match self.kind {
            BrushKind::Clay => {
                stamp.centre
                    + stamp.normal * (self.radius * CLAY_OFFSET * stamp.direction.outward_sign())
            }
            _ => stamp.centre,
        };

        let kind = self.kind;
        let falloff = self.falloff;
        let centre = stamp.centre;
        let stroke_normal = stamp.normal;
        let direction = stamp.direction;

        volume.edit_voxels(lo, hi, |voxel, position, value| {
            let distance = position.distance(centre) * inverse_radius;
            if distance >= 1.0 {
                return value;
            }
            let weight = falloff.weight(distance) * gain;
            if weight <= 0.0 {
                return value;
            }

            match kind {
                BrushKind::Inflate => value + field_sign * weight * displacement,

                BrushKind::Draw => {
                    // Translate the field along the stroke normal, which slides
                    // this patch of surface out from under the cursor.
                    //
                    // The tempting version, weighting an offset by how much the
                    // local gradient faces the stroke, does not survive a
                    // stroke: wherever the field is flat or saturated that
                    // gradient is noise, and multiplying a displacement by it
                    // turns the noise into geometry. A resample cannot
                    // introduce detail that was not already there.
                    let shift = stroke_normal
                        * (weight * displacement * voxel_size * direction.outward_sign());
                    region.sample((position - shift) / voxel_size)
                }

                BrushKind::Smooth => {
                    let average = region.neighbour_average(voxel);
                    value + (average - value) * weight
                }

                BrushKind::Pinch => {
                    // Read the field from slightly closer to the brush axis,
                    // which drags the surface sideways and squeezes whatever
                    // ridge runs through the brush into a crease.
                    //
                    // The obvious alternative, an unsharp mask on the values,
                    // looks the same for one stamp and then compounds: it is a
                    // high pass filter with gain above 1, so across a stroke it
                    // amplifies its own rounding error into a crust. Warping
                    // the domain only ever resamples what is already there.
                    let to_centre = centre - position;
                    // Along the surface only. Pulling along the normal as well
                    // would make pinch quietly double as inflate.
                    let lateral = to_centre - stroke_normal * to_centre.dot(stroke_normal);
                    let Some(direction_to_axis) = lateral.try_normalize() else {
                        return value;
                    };
                    let spread = match direction {
                        BrushDirection::Add => 1.0,
                        BrushDirection::Subtract => -1.0,
                    };
                    let pull =
                        direction_to_axis * (weight * PINCH_PULL_VOXELS * voxel_size * spread);
                    region.sample((position + pull) / voxel_size)
                }

                BrushKind::Flatten => {
                    let plane = (position - plane_point).dot(stroke_normal) / voxel_size;
                    value + (plane - value) * weight
                }

                BrushKind::Clay => {
                    let plane = (position - plane_point).dot(stroke_normal) / voxel_size;
                    let blended = value + (plane - value) * weight;
                    // Keep only the half of the operation that moves material
                    // the way the user asked for. Without this, clay carves
                    // away any bump standing above its plane.
                    match direction {
                        BrushDirection::Add => blended.min(value),
                        BrushDirection::Subtract => blended.max(value),
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::{INSIDE, OUTSIDE};

    fn sphere() -> Volume {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 24.0);
        volume
    }

    /// A point on the surface of the seeded sphere, and its outward normal.
    fn surface(volume: &Volume) -> (Vec3, Vec3) {
        let point = Vec3::new(24.0, 0.0, 0.0);
        (point, volume.gradient_world(point))
    }

    fn brush(kind: BrushKind) -> Brush {
        Brush { kind, radius: 8.0, strength: 0.4, falloff: FalloffCurve::Smooth }
    }

    #[test]
    fn every_falloff_curve_runs_from_one_to_zero_and_never_rises() {
        for curve in FalloffCurve::ALL {
            assert!((curve.weight(0.0) - 1.0).abs() < 1.0e-6, "{curve} is not 1 at the centre");
            assert_eq!(curve.weight(1.0), 0.0, "{curve} is not 0 at the rim");
            assert_eq!(curve.weight(2.5), 0.0, "{curve} leaks past the rim");

            let mut previous = f32::INFINITY;
            for step in 0..=100 {
                let value = curve.weight(step as f32 / 100.0);
                assert!(value <= previous + 1.0e-6, "{curve} rose at step {step}");
                assert!((0.0..=1.0).contains(&value), "{curve} left the unit range");
                previous = value;
            }
        }
    }

    #[test]
    fn sharp_is_tighter_than_smooth_and_wide_is_broader() {
        let half = 0.5;
        assert!(FalloffCurve::Sharp.weight(half) < FalloffCurve::Smooth.weight(half));
        assert!(FalloffCurve::Wide.weight(half) > FalloffCurve::Smooth.weight(half));
    }

    #[test]
    fn adding_and_removing_move_the_field_in_opposite_directions() {
        for kind in [BrushKind::Draw, BrushKind::Clay, BrushKind::Inflate] {
            let mut volume = sphere();
            let (point, normal) = surface(&volume);
            let before = volume.sample_world(point);
            let mut scratch = BrushScratch::new();

            brush(kind).apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Add),
                &mut scratch,
            );
            let added = volume.sample_world(point);

            let mut volume = sphere();
            brush(kind).apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Subtract),
                &mut scratch,
            );
            let removed = volume.sample_world(point);

            assert!(added < before, "{kind} did not add material: {before} then {added}");
            assert!(removed > before, "{kind} did not remove material: {before} then {removed}");
        }
    }

    #[test]
    fn every_brush_keeps_values_inside_the_narrow_band() {
        for kind in BrushKind::ALL {
            let mut volume = sphere();
            let (point, normal) = surface(&volume);
            let mut scratch = BrushScratch::new();
            let brush = Brush { kind, radius: 8.0, strength: 0.9, falloff: FalloffCurve::Smooth };

            for _ in 0..40 {
                brush.apply(
                    &mut volume,
                    &Stamp::new(point, normal, BrushDirection::Add),
                    &mut scratch,
                );
            }

            for step in -12..=12 {
                let probe = point + normal * step as f32;
                let value = volume.sample_world(probe);
                assert!(
                    (INSIDE..=OUTSIDE).contains(&value),
                    "{kind} left the band at {probe:?}: {value}"
                );
                assert!(value.is_finite(), "{kind} produced a non finite value");
            }
        }
    }

    #[test]
    fn smooth_reduces_roughness() {
        // Build a bumpy patch, then check smoothing flattens the variation.
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let poke = Brush {
            kind: BrushKind::Draw,
            radius: 3.0,
            strength: 0.8,
            falloff: FalloffCurve::Sharp,
        };
        for offset in [-6.0_f32, -2.0, 2.0, 6.0] {
            let at = point + Vec3::new(0.0, offset, 0.0);
            poke.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        }

        let roughness = |volume: &Volume| {
            let mut total = 0.0;
            for step in -8..=8 {
                let a = volume.sample_world(point + Vec3::new(0.0, step as f32, 0.0));
                let b = volume.sample_world(point + Vec3::new(0.0, step as f32 + 1.0, 0.0));
                total += (b - a).abs();
            }
            total
        };

        let before = roughness(&volume);
        let smooth = Brush {
            kind: BrushKind::Smooth,
            radius: 10.0,
            strength: 0.9,
            falloff: FalloffCurve::Wide,
        };
        for _ in 0..12 {
            smooth.apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Add),
                &mut scratch,
            );
        }
        let after = roughness(&volume);

        assert!(after < before, "smoothing did not reduce roughness: {before} then {after}");
    }

    #[test]
    fn pinch_is_the_opposite_of_smooth() {
        // Smooth pulls the surface toward its local average and pinch squeezes
        // it toward the brush axis. They should not move a probe beside the
        // stroke the same way, which is what a sign error would look like.
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let probe = point + Vec3::new(0.0, 2.0, 0.0);
        let base = volume.sample_world(probe);

        let mut smoothed = sphere();
        brush(BrushKind::Smooth).apply(
            &mut smoothed,
            &Stamp::new(point, normal, BrushDirection::Add),
            &mut scratch,
        );
        let after_smooth = smoothed.sample_world(probe);

        brush(BrushKind::Pinch).apply(
            &mut volume,
            &Stamp::new(point, normal, BrushDirection::Add),
            &mut scratch,
        );
        let after_pinch = volume.sample_world(probe);

        // Smooth pulls toward the local average, pinch pushes away from it.
        assert!(
            (after_smooth - base).signum() != (after_pinch - base).signum()
                || (after_smooth - base).abs() < 1.0e-7,
            "smooth moved to {after_smooth} and pinch to {after_pinch} from {base}, same direction"
        );
    }

    #[test]
    fn flatten_pulls_a_bump_down_toward_the_plane() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let bump = Brush {
            kind: BrushKind::Draw,
            radius: 4.0,
            strength: 0.9,
            falloff: FalloffCurve::Sharp,
        };
        for _ in 0..4 {
            bump.apply(&mut volume, &Stamp::new(point, normal, BrushDirection::Add), &mut scratch);
        }

        // Height of the surface along the normal, found by walking outward.
        let height = |volume: &Volume| {
            let mut last = 0.0;
            for step in 0..80 {
                let t = step as f32 * 0.25;
                if volume.sample_world(point + normal * t) >= 0.0 {
                    return last;
                }
                last = t;
            }
            last
        };

        let raised = height(&volume);
        let flatten = Brush {
            kind: BrushKind::Flatten,
            radius: 8.0,
            strength: 0.8,
            falloff: FalloffCurve::Wide,
        };
        for _ in 0..10 {
            flatten.apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Add),
                &mut scratch,
            );
        }
        assert!(height(&volume) < raised, "flatten did not bring the bump down");
    }

    #[test]
    fn clay_only_adds_when_adding() {
        // Clay's plane sits outside the surface, so without the one sided clamp
        // it would shave off anything already standing proud of that plane.
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let spike = Brush {
            kind: BrushKind::Draw,
            radius: 2.5,
            strength: 1.0,
            falloff: FalloffCurve::Sharp,
        };
        for _ in 0..6 {
            spike.apply(&mut volume, &Stamp::new(point, normal, BrushDirection::Add), &mut scratch);
        }

        let probes: Vec<Vec3> =
            (-6..=6).map(|step| point + Vec3::new(0.0, step as f32, 0.0) + normal * 0.5).collect();
        let before: Vec<f32> = probes.iter().map(|p| volume.sample_world(*p)).collect();

        brush(BrushKind::Clay).apply(
            &mut volume,
            &Stamp::new(point, normal, BrushDirection::Add),
            &mut scratch,
        );

        for (probe, was) in probes.iter().zip(before) {
            let now = volume.sample_world(*probe);
            assert!(now <= was + 1.0e-6, "clay removed material at {probe:?}: {was} then {now}");
        }
    }

    #[test]
    fn draw_follows_the_stroke_direction_and_inflate_does_not() {
        // This is the whole difference between the two. Draw translates the
        // patch along the direction the stroke is pushing, so a stroke coming
        // in at an angle to the surface bites by the cosine of that angle.
        // Inflate offsets the level set, so every point moves along its own
        // normal by the same amount whichever way the stroke points.
        let mut scratch = BrushScratch::new();
        let point = Vec3::new(24.0, 0.0, 0.0);
        // The surface normal at `point` is along X, so this comes in at 45
        // degrees to it.
        let tilted = Vec3::new(1.0, 1.0, 0.0).normalize();

        let mut bite = |kind: BrushKind, stroke_normal: Vec3| {
            let mut volume = sphere();
            let before = volume.sample_world(point);
            let brush = Brush { kind, radius: 8.0, strength: 1.0, falloff: FalloffCurve::Smooth };
            brush.apply(
                &mut volume,
                &Stamp::new(point, stroke_normal, BrushDirection::Add),
                &mut scratch,
            );
            before - volume.sample_world(point)
        };

        let draw_square_on = bite(BrushKind::Draw, Vec3::X);
        let draw_at_an_angle = bite(BrushKind::Draw, tilted);
        assert!(draw_square_on > 0.0, "draw did nothing at all");
        assert!(
            draw_at_an_angle < draw_square_on * 0.9,
            "draw ignored the stroke direction: {draw_at_an_angle} against {draw_square_on}"
        );
        // Should land near cos 45 degrees, which is what makes it a projection
        // rather than an arbitrary reduction.
        let ratio = draw_at_an_angle / draw_square_on;
        assert!(
            (ratio - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.15,
            "the bite fell off by {ratio}, which is not the cosine of the angle"
        );

        let inflate_square_on = bite(BrushKind::Inflate, Vec3::X);
        let inflate_at_an_angle = bite(BrushKind::Inflate, tilted);
        assert!(
            (inflate_at_an_angle - inflate_square_on).abs() < 1.0e-5,
            "inflate should not care about the stroke direction: \
             {inflate_at_an_angle} against {inflate_square_on}"
        );
    }

    #[test]
    fn no_brush_carves_a_pit_where_it_should_raise_a_bump() {
        // The failure mode that only shows up on screen: a stamp big enough to
        // saturate the narrow band destroys the gradient it just wrote, and the
        // next stamp reads noise from it. The surface comes out as a crust of
        // aliased spikes while every value stays legally inside the band.
        //
        // A stroke of overlapping stamps must leave a surface that is still
        // smooth, measured here as the height varying gently across the bump.
        for kind in [BrushKind::Draw, BrushKind::Inflate, BrushKind::Clay, BrushKind::Pinch] {
            let mut volume = sphere();
            let mut scratch = BrushScratch::new();
            let (point, normal) = surface(&volume);
            let brush = Brush { kind, radius: 8.0, strength: 0.9, falloff: FalloffCurve::Smooth };

            for step in -4..=4 {
                let at = point + Vec3::new(0.0, step as f32 * 2.0, 0.0);
                for _ in 0..4 {
                    brush.apply(
                        &mut volume,
                        &Stamp::new(at, normal, BrushDirection::Add),
                        &mut scratch,
                    );
                }
            }

            // Walk across the worked area and measure the surface height at
            // each step. Neighbouring samples should differ by a fraction of a
            // voxel, not leap about.
            let heights: Vec<f32> = (-8..=8)
                .map(|step| {
                    let probe = point + Vec3::new(0.0, step as f32, 0.0);
                    let mut last = 0.0;
                    for walk in 0..200 {
                        let t = walk as f32 * 0.1;
                        if volume.sample_world(probe + normal * t) >= 0.0 {
                            return last;
                        }
                        last = t;
                    }
                    last
                })
                .collect();

            let worst_jump =
                heights.windows(2).map(|pair| (pair[1] - pair[0]).abs()).fold(0.0_f32, f32::max);
            assert!(
                worst_jump < 2.0,
                "{kind} left a surface that jumps {worst_jump} units between neighbouring \
                 samples, which is the crust a saturating stamp produces: {heights:?}"
            );
        }
    }

    #[test]
    fn pressure_scales_the_bite() {
        let (point, normal) = surface(&sphere());
        let mut scratch = BrushScratch::new();
        let brush = brush(BrushKind::Draw);

        let mut bite = |pressure: f32| {
            let mut volume = sphere();
            let before = volume.sample_world(point);
            let stamp = Stamp::new(point, normal, BrushDirection::Add).with_pressure(pressure);
            brush.apply(&mut volume, &stamp, &mut scratch);
            before - volume.sample_world(point)
        };

        let light = bite(0.25);
        let full = bite(1.0);
        assert!(light > 0.0, "a light touch should still cut");
        assert!(light < full, "pressure did not scale the stroke: {light} against {full}");
        assert_eq!(bite(0.0), 0.0, "zero pressure should do nothing");
    }

    #[test]
    fn x_symmetry_mirrors_the_stroke_across_the_origin() {
        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let at = Vec3::new(12.0, 20.0, 0.0);
        let normal = volume.gradient_world(at);
        let mirrored_probe = Vec3::new(-at.x, at.y, at.z);
        let before = volume.sample_world(mirrored_probe);

        brush(BrushKind::Draw).apply_symmetric(
            &mut volume,
            &Stamp::new(at, normal, BrushDirection::Add),
            Symmetry::X,
            &mut scratch,
        );

        assert!(
            volume.sample_world(mirrored_probe) < before,
            "the mirrored half of the stroke never landed"
        );
    }

    #[test]
    fn symmetry_off_leaves_the_other_side_untouched() {
        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let at = Vec3::new(12.0, 20.0, 0.0);
        let normal = volume.gradient_world(at);
        let mirrored_probe = Vec3::new(-at.x, at.y, at.z);
        let before = volume.sample_world(mirrored_probe);

        brush(BrushKind::Draw).apply_symmetric(
            &mut volume,
            &Stamp::new(at, normal, BrushDirection::Add),
            Symmetry::Off,
            &mut scratch,
        );
        assert_eq!(volume.sample_world(mirrored_probe), before);
    }

    #[test]
    fn a_stamp_far_from_the_model_does_not_disturb_it() {
        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let probe = Vec3::new(24.0, 0.0, 0.0);
        let before = volume.sample_world(probe);

        brush(BrushKind::Draw).apply(
            &mut volume,
            &Stamp::new(Vec3::splat(2000.0), Vec3::Y, BrushDirection::Add),
            &mut scratch,
        );
        assert_eq!(volume.sample_world(probe), before);
    }

    #[test]
    fn smooth_and_flatten_declare_themselves_non_directional() {
        assert!(!BrushKind::Smooth.is_directional());
        assert!(!BrushKind::Flatten.is_directional());
        assert!(BrushKind::Draw.is_directional());
    }

    #[test]
    fn stamp_spacing_never_falls_below_a_voxel() {
        let tiny = Brush { radius: 0.01, ..Brush::default() };
        assert_eq!(tiny.spacing(0.25), 0.25);
        let large = Brush { radius: 8.0, ..Brush::default() };
        assert_eq!(large.spacing(0.25), 2.0);
    }
}
