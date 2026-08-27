// SPDX-License-Identifier: AGPL-3.0-only

//! World space overlay geometry: the brush cursor and the mirror planes.
//!
//! Both are built here, in the application, and handed to `brokkr-gpu` as plain
//! coloured vertices. `brokkr-core` never learns that a screen exists, and the
//! GPU crate never learns what a brush is.
//!
//! # Why the brush needs a cursor at all
//!
//! Before this there was none: nothing told you where the brush was or how big
//! it was until after you had pressed. Every sculpting tool draws a ring for the
//! same reason a pen has a visible nib.
//!
//! The ring is **pushed onto the surface** rather than drawn as a flat disc at
//! the hit point: the field is a signed distance, so `position - value *
//! gradient` walks toward the surface, and that is the difference between a ring
//! that wraps a form and a decal floating through it.
//!
//! It takes a few steps rather than one, and the reason is specific to this
//! engine. Distances are **clamped to the narrow band**, ±3 voxels, so however
//! far a point actually is from the surface one step can only ever move it three
//! voxels. A wide brush on a tight form starts further out than that: at a 10 mm
//! radius on a 20 mm ball the rim begins 2.36 mm off the surface, which is more
//! than the band carries at a 0.5 mm voxel. One step left the ring visibly
//! floating; see [`SURFACE_STEPS`].

use brokkr_core::{Brush, BrushDirection, FalloffCurve, MaskOp, MirrorAxis, Symmetry, Volume};
use brokkr_gpu::OverlayBatch;
use glam::Vec3;

use crate::theme::{self, linear};

/// Segments in a ring. Sixty four is smooth at any size the interface offers
/// and is still only 128 line vertices.
const RING_SEGMENTS: usize = 64;

/// Gradient steps used to settle a ring vertex onto the surface.
///
/// One is not enough: the narrow band clamp caps a single step at three voxels,
/// and a wide brush's rim starts further out than that. Four covers twelve
/// voxels, which is past anything the radius slider can ask for, and the loop
/// stops early once it is within a tenth of a voxel.
const SURFACE_STEPS: usize = 4;

/// How far outside the model a mirror plane reaches, as a fraction of the
/// model's bounding radius. Enough to read as a plane rather than as a patch.
const PLANE_OVERSHOOT: f32 = 1.25;

/// Opacity of a mirror plane. Low on purpose: it is a reference, and it sits in
/// front of the thing being worked on.
const PLANE_ALPHA: f32 = 0.10;

/// What the pointer is currently doing, which the ring's colour reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorMood {
    /// Adding material.
    Add,
    /// Removing it — `ctrl`, or the eraser end of the stylus.
    Subtract,
    /// Mid gesture, resizing the brush rather than sculpting with it.
    Sizing,
    /// Over a body that is not the active one, where a press SELECTS rather
    /// than sculpts.
    ///
    /// **This overrides the add/subtract pair rather than sitting beside it.**
    /// Holding ctrl over another body would otherwise draw the red ring that
    /// means "this takes away" over a press that takes nothing away at all.
    ///
    /// Drawn as an unfilled ring — the outer ring alone, in a neutral colour —
    /// and deliberately not as a fourth hue: the three hues are spent, and the
    /// difference this is reporting is a verb rather than an intensity.
    Selecting,
    /// Painting protection in. The mask tool's plain left drag.
    ///
    /// **A hue outside the matcap's own gamut**, which is the same argument the
    /// mask's viewport tint is made on: the clay is deliberately warm, its
    /// coolest pixel is a fill-lit cavity at `b - r = +17`, and a ring that
    /// merely desaturates would read as "a bit darker" rather than as a
    /// different tool. A cool blue at full saturation cannot be mistaken for
    /// anything the model can be.
    Masking,
    /// Taking it away — `ctrl`, `alt`, or the eraser end of the stylus.
    ///
    /// Not [`CursorMood::Subtract`]'s red. Red means "this removes material",
    /// and unmasking removes no material at all — it is the *reverse* of the
    /// blue ring beside it, so it is drawn as the pale end of the same hue
    /// rather than as a different one. Add and Subtract can afford two hues
    /// because they really are two things; these two are one thing and its
    /// inverse.
    Unmasking,
}

/// The distance, as a fraction of the radius, at which a falloff curve has
/// fallen to half weight.
///
/// Drawn as an inner ring, which is ZBrush's Focal Shift readout: it is what
/// makes the four curves legible as shapes rather than as four names. Found by
/// bisection rather than by a table per curve, so adding a curve to
/// [`FalloffCurve`] cannot leave a stale number here.
fn half_weight_distance(falloff: FalloffCurve) -> f32 {
    let (mut low, mut high) = (0.0f32, 1.0f32);
    // Twenty steps resolves to about one part in a million, far finer than a
    // pixel at any radius the interface offers.
    for _ in 0..20 {
        let middle = (low + high) * 0.5;
        if falloff.weight(middle) > 0.5 {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low + high) * 0.5
}

/// Two unit vectors spanning the plane across `normal`.
fn tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let normal = normal.normalize_or(Vec3::Y);
    // Cross with whichever world axis the normal is least aligned with, so the
    // product never collapses.
    let a = normal.abs();
    let axis = if a.x <= a.y && a.x <= a.z {
        Vec3::X
    } else if a.y <= a.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let u = normal.cross(axis).normalize_or(Vec3::X);
    (u, normal.cross(u).normalize_or(Vec3::Z))
}

/// Walk a point onto the surface along the gradient.
///
/// Enough steps to cross the narrow band several times over. Each step can move
/// at most [`brokkr_core::NARROW_BAND`] voxels because that is where the field
/// saturates, so a single step is not enough for a wide brush on a tight form.
fn onto_surface(volume: &Volume, at: Vec3) -> Vec3 {
    let voxel_size = volume.voxel_size();
    let mut at = at;
    for _ in 0..SURFACE_STEPS {
        let value = volume.sample_world(at);
        if !value.is_finite() {
            return at;
        }
        // Distances are held in voxels, so convert before stepping in world
        // space. Within a tenth of a voxel is closer than a pixel at any sane
        // zoom, so stop rather than keep sampling.
        if value.abs() < 0.1 {
            break;
        }
        at -= volume.gradient_world(at) * (value * voxel_size);
    }
    at
}

/// A ring of the given radius around a surface point, following the surface.
fn push_ring(
    batch: &mut OverlayBatch,
    volume: &Volume,
    centre: Vec3,
    normal: Vec3,
    radius: f32,
    colour: [f32; 4],
) {
    if radius <= 0.0 {
        return;
    }
    let (u, v) = tangent_frame(normal);
    let point = |step: usize| {
        let angle = step as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        let (sin, cos) = angle.sin_cos();
        onto_surface(volume, centre + (u * cos + v * sin) * radius)
    };

    let mut previous = point(0);
    for step in 1..=RING_SEGMENTS {
        let next = point(step % RING_SEGMENTS);
        batch.push_line(previous, next, colour);
        previous = next;
    }
}

/// Rebuild the whole world overlay for one frame.
///
/// Clears and refills `batch` rather than returning a new one, because this runs
/// on input events and the per frame path must not allocate.
///
/// `volume` is the body the pick returned rather than the active one, which is
/// what keeps the ring on the surface it is actually over. Building it against
/// a volume that does not contain the point is worse than stale: far from any
/// surface `sample_world` returns the clamped outside value and `gradient_world`
/// falls through `try_normalize` to `Vec3::Y`, so `onto_surface` would draw a
/// confident flat ring three millimetres below nothing.
///
/// `mirror_centre` is where the enabled mirror planes are drawn, and it is the
/// same number the engine mirrors about — see `MIRROR_CENTRE`. Drawing them at
/// the world origin regardless would be a promise the sculpt does not keep the
/// day the centre moves.
///
/// `hover` is the surface point under the pointer, or `None` when the pointer is
/// off the model — in which case there is no ring, which is itself the useful
/// signal that a press would do nothing.
#[allow(clippy::too_many_arguments)]
pub fn build(
    batch: &mut OverlayBatch,
    volume: &Volume,
    brush: &Brush,
    symmetry: Symmetry,
    mirror_centre: Vec3,
    hover: Option<Vec3>,
    mood: CursorMood,
    model_radius: f32,
) {
    batch.clear();

    // --- mirror planes ----------------------------------------------------
    // Only the enabled ones, which is the whole point: turning on X should show
    // where X cuts. Drawn first so the ring reads over them.
    //
    // All three share the accent rather than getting a colour each: the tokens
    // hold no second and third hue, the strip already says which axes are on,
    // and a plane's orientation says which one it is.
    let reach = (model_radius * PLANE_OVERSHOOT).max(1.0);
    let plane_colour = linear(theme::ACCENT, PLANE_ALPHA);
    for axis in MirrorAxis::ALL {
        if !symmetry.axis(axis) {
            continue;
        }
        // The two world axes the plane spans.
        let (u, v) = match axis {
            MirrorAxis::X => (Vec3::Y, Vec3::Z),
            MirrorAxis::Y => (Vec3::X, Vec3::Z),
            MirrorAxis::Z => (Vec3::X, Vec3::Y),
        };
        batch.push_quad(
            mirror_centre + (-u - v) * reach,
            mirror_centre + (u - v) * reach,
            mirror_centre + (u + v) * reach,
            mirror_centre + (-u + v) * reach,
            plane_colour,
        );
    }

    // --- the brush cursor -------------------------------------------------
    let Some(centre) = hover else {
        return;
    };
    let normal = volume.gradient_world(centre);
    let colour = match mood {
        CursorMood::Add => linear(theme::ACCENT, 0.95),
        // Red for removing material, which is the convention and reads instantly
        // as "this takes away".
        CursorMood::Subtract => linear(theme::ERROR, 0.95),
        CursorMood::Sizing => linear(theme::ACCENT_HOT, 1.0),
        CursorMood::Selecting => linear(theme::TEXT, 0.95),
        CursorMood::Masking => linear(theme::MASK, 0.95),
        CursorMood::Unmasking => linear(theme::MASK_PALE, 0.95),
    };

    push_ring(batch, volume, centre, normal, brush.radius, colour);
    // The inner ring is where the brush has fallen to half strength. Skipped
    // when it would sit on top of the outer one -- and skipped entirely while a
    // press would only select, because there is no stroke for a falloff to
    // describe and the empty middle is what makes that ring read as unfilled.
    if mood == CursorMood::Selecting {
        return;
    }
    let inner = brush.radius * half_weight_distance(brush.falloff);
    if inner > brush.radius * 0.08 && inner < brush.radius * 0.92 {
        push_ring(batch, volume, centre, normal, inner, [colour[0], colour[1], colour[2], 0.45]);
    }
}

/// Which mood a cursor should have.
///
/// `selecting` is answered before the stroke direction because it overrides it:
/// see [`CursorMood::Selecting`]. Sizing outranks even that, because a sizing
/// gesture is not a press on anything.
///
/// `masking` is `Some` only while the mask tool is live, and it outranks the
/// add/subtract pair for the same reason `selecting` does: in mask mode a press
/// changes no geometry at all, so a ring saying "this adds" or "this removes"
/// would be describing a stroke that is not going to happen. It sits BELOW
/// sizing, because a sizing drag is not a press on anything whichever tool is
/// chosen. Blur takes the masking colour rather than a third: it is neither
/// adding protection nor taking it away, and the ring's job here is to say
/// which tool is live, which the strip's live `blur` label already qualifies.
pub fn mood(
    direction: BrushDirection,
    sizing: bool,
    selecting: bool,
    masking: Option<MaskOp>,
) -> CursorMood {
    if sizing {
        return CursorMood::Sizing;
    }
    if let Some(op) = masking {
        return match op {
            MaskOp::Lower => CursorMood::Unmasking,
            MaskOp::Raise | MaskOp::Blur => CursorMood::Masking,
        };
    }
    if selecting {
        return CursorMood::Selecting;
    }
    match direction {
        BrushDirection::Add => CursorMood::Add,
        BrushDirection::Subtract => CursorMood::Subtract,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brokkr_core::BrushKind;

    fn sphere() -> Volume {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        volume
    }

    fn brush(radius: f32, falloff: FalloffCurve) -> Brush {
        Brush { kind: BrushKind::Draw, radius, falloff, ..Brush::default() }
    }

    /// Every curve's half weight point, checked against the curve itself rather
    /// than against numbers written down here.
    #[test]
    fn the_inner_ring_sits_where_the_brush_is_at_half_strength() {
        for curve in FalloffCurve::ALL {
            let d = half_weight_distance(curve);
            assert!((0.0..=1.0).contains(&d), "{curve} gave {d}");
            assert!(
                (curve.weight(d) - 0.5).abs() < 1.0e-3,
                "{curve} is {} at {d}, not half",
                curve.weight(d)
            );
        }
    }

    /// Sharp concentrates toward the centre and wide spreads out, so their half
    /// weight rings must sit on opposite sides of the linear one. This is the
    /// property that makes the inner ring worth drawing.
    #[test]
    fn a_sharper_curve_has_a_tighter_inner_ring() {
        let sharp = half_weight_distance(FalloffCurve::Sharp);
        let linear_curve = half_weight_distance(FalloffCurve::Linear);
        let wide = half_weight_distance(FalloffCurve::Wide);
        assert!(sharp < linear_curve, "sharp {sharp} should be tighter than linear");
        assert!(wide > linear_curve, "wide {wide} should be broader than linear");
    }

    #[test]
    fn no_hover_means_no_ring_at_all() {
        let volume = sphere();
        let mut batch = OverlayBatch::default();
        build(
            &mut batch,
            &volume,
            &brush(3.0, FalloffCurve::Smooth),
            Symmetry::OFF,
            Vec3::ZERO,
            None,
            CursorMood::Add,
            20.0,
        );
        assert!(batch.is_empty(), "a pointer off the model should draw nothing");
    }

    #[test]
    fn the_ring_lands_on_the_surface_at_the_brush_radius() {
        let volume = sphere();
        let mut batch = OverlayBatch::default();
        let centre = Vec3::new(0.0, 0.0, 20.0);
        let radius = 4.0;
        build(
            &mut batch,
            &volume,
            &brush(radius, FalloffCurve::Smooth),
            Symmetry::OFF,
            Vec3::ZERO,
            Some(centre),
            CursorMood::Add,
            20.0,
        );

        assert!(!batch.lines.is_empty(), "the ring did not draw");
        let mut on_surface = 0;
        let mut at_radius = 0;
        for vertex in &batch.lines {
            let p = Vec3::from_array(vertex.position);
            // Pushed onto the surface, so the field should read near zero.
            if volume.sample_world(p).abs() < 1.0 {
                on_surface += 1;
            }
            // The outer ring is at the radius; the inner one is closer, so only
            // check that nothing exceeds it.
            let across = (p - centre).length();
            assert!(across <= radius * 1.2, "a ring vertex was {across} from the centre");
            if (across - radius).abs() < radius * 0.15 {
                at_radius += 1;
            }
        }
        assert!(on_surface > batch.lines.len() / 2, "most of the ring should lie on the surface");
        assert!(at_radius > 0, "no vertex sat at the brush radius");
    }

    /// The ring has to wrap a form rather than float through it, which is the
    /// whole reason for the Newton step.
    #[test]
    fn the_ring_follows_curvature_rather_than_staying_flat() {
        let volume = sphere();
        let mut batch = OverlayBatch::default();
        let centre = Vec3::new(0.0, 0.0, 20.0);
        // A radius large enough that a flat disc would visibly leave the sphere.
        build(
            &mut batch,
            &volume,
            &brush(10.0, FalloffCurve::Smooth),
            Symmetry::OFF,
            Vec3::ZERO,
            Some(centre),
            CursorMood::Add,
            20.0,
        );

        // A flat ring would keep every vertex at z = 20. Following the sphere
        // pulls the rim back toward the centre of the model.
        let deepest = batch.lines.iter().map(|v| v.position[2]).fold(f32::MAX, f32::min);
        assert!(deepest < 19.0, "the ring stayed flat: nearest z was {deepest}");
    }

    #[test]
    fn only_the_enabled_mirror_planes_are_drawn() {
        let volume = sphere();
        let mut batch = OverlayBatch::default();
        let quad_vertices = 6;

        for (symmetry, expected) in [
            (Symmetry::OFF, 0),
            (Symmetry::X, 1),
            (Symmetry::OFF.with_axis(MirrorAxis::Y, true), 1),
            (Symmetry::X.with_axis(MirrorAxis::Z, true), 2),
        ] {
            build(
                &mut batch,
                &volume,
                &brush(3.0, FalloffCurve::Smooth),
                symmetry,
                Vec3::ZERO,
                None,
                CursorMood::Add,
                20.0,
            );
            assert_eq!(
                batch.surfaces.len(),
                expected * quad_vertices,
                "{} should draw {expected} plane(s)",
                symmetry.label()
            );
        }
    }

    #[test]
    fn a_mirror_plane_lies_on_its_own_axis_and_reaches_past_the_model() {
        let volume = sphere();
        let mut batch = OverlayBatch::default();
        let model_radius = 20.0;
        build(
            &mut batch,
            &volume,
            &brush(3.0, FalloffCurve::Smooth),
            Symmetry::X,
            Vec3::ZERO,
            None,
            CursorMood::Add,
            model_radius,
        );

        let mut reach: f32 = 0.0;
        for vertex in &batch.surfaces {
            let p = Vec3::from_array(vertex.position);
            assert_eq!(p.x, 0.0, "the X mirror plane left the x = 0 plane");
            reach = reach.max(p.length());
        }
        assert!(reach > model_radius, "the plane stopped inside the model: {reach}");
    }

    #[test]
    fn carving_and_sizing_change_the_cursor_colour() {
        let volume = sphere();
        let centre = Vec3::new(0.0, 0.0, 20.0);
        let colour_for = |mood| {
            let mut batch = OverlayBatch::default();
            build(
                &mut batch,
                &volume,
                &brush(3.0, FalloffCurve::Smooth),
                Symmetry::OFF,
                Vec3::ZERO,
                Some(centre),
                mood,
                20.0,
            );
            batch.lines[0].colour
        };

        let add = colour_for(CursorMood::Add);
        assert_ne!(add, colour_for(CursorMood::Subtract), "carving looks like adding");
        assert_ne!(add, colour_for(CursorMood::Sizing), "sizing looks like adding");
        // The two mask rings against everything else and against each other.
        // Masking sharing a colour with adding is the failure this is really
        // about: it would say "this stroke changes the model" over one that
        // changes no geometry at all.
        let masking = colour_for(CursorMood::Masking);
        let unmasking = colour_for(CursorMood::Unmasking);
        assert_ne!(masking, add, "masking looks like adding");
        assert_ne!(masking, colour_for(CursorMood::Subtract), "masking looks like carving");
        assert_ne!(masking, unmasking, "unmasking looks like masking");
        assert_ne!(unmasking, colour_for(CursorMood::Subtract), "unmasking looks like carving");
    }

    #[test]
    fn the_mood_follows_the_stroke_direction_unless_sizing() {
        assert_eq!(mood(BrushDirection::Add, false, false, None), CursorMood::Add);
        assert_eq!(mood(BrushDirection::Subtract, false, false, None), CursorMood::Subtract);
        // Sizing wins: mid gesture the pointer is not sculpting at all, so the
        // direction it would have used is not what the ring should report.
        assert_eq!(mood(BrushDirection::Add, true, false, None), CursorMood::Sizing);
        assert_eq!(mood(BrushDirection::Subtract, true, false, None), CursorMood::Sizing);
        // And over a body that is not the active one the press selects, which
        // outranks the direction and is outranked by sizing.
        assert_eq!(mood(BrushDirection::Subtract, false, true, None), CursorMood::Selecting);
        assert_eq!(mood(BrushDirection::Add, true, true, None), CursorMood::Sizing);
    }

    /// Mask mode outranks the stroke direction, and is outranked by sizing.
    ///
    /// The first half is the one worth pinning: in mask mode a press changes no
    /// geometry, so an Add or Subtract ring would be describing a stroke that is
    /// not going to happen -- and `stroke_direction` is still computed and still
    /// passed, because the caller has no reason to suppress it.
    #[test]
    fn the_mask_tool_owns_the_ring_whichever_way_the_brush_is_pointing() {
        for direction in [BrushDirection::Add, BrushDirection::Subtract] {
            assert_eq!(mood(direction, false, false, Some(MaskOp::Raise)), CursorMood::Masking);
            assert_eq!(mood(direction, false, false, Some(MaskOp::Lower)), CursorMood::Unmasking);
            // Blur is neither adding protection nor taking it away, and the ring
            // says which tool is live rather than which of the three.
            assert_eq!(mood(direction, false, false, Some(MaskOp::Blur)), CursorMood::Masking);
            // A sizing drag is not a press on anything, whichever tool is live.
            assert_eq!(mood(direction, true, false, Some(MaskOp::Raise)), CursorMood::Sizing);
        }
    }
}
