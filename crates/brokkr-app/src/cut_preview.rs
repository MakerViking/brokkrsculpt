// SPDX-License-Identifier: AGPL-3.0-only

//! The live cut preview: what the drag so far is going to remove.
//!
//! Built here and handed to `brokkr-gpu` as plain coloured vertices, exactly as
//! the brush cursor is, and for the same reasons -- `brokkr-core` never learns
//! that a screen exists and the GPU crate never learns what a cut is.
//!
//! # Why this exists at all
//!
//! Before it, the cut had **no preview of any kind**. `DragKind::Cutting` fell
//! into a do-nothing arm on every motion event, nothing anywhere drew a line,
//! and the comment above that arm said the opposite. The user dragged a
//! destructive, irreversible-except-by-undo gesture across their model seeing
//! nothing but the brush ring -- which implies a press would sculpt.
//!
//! # What is drawn is what goes
//!
//! **The decimated hull, and never the raw stroke.** The hull is what the
//! planes are built from, so drawing the stroke would show one shape and remove
//! another, and the difference is exactly where the surprises are: a lasso with
//! a dent in it is cut as though the dent were not there, and a hand-drawn loop
//! is simplified to sixteen sides. Both are visible before the button comes up
//! precisely because this draws the hull.
//!
//! For a straight drag the hull is two points and there is no region, so what
//! is drawn instead is the line and a shaded band on the side that goes -- the
//! side convention being the one thing about the plane cut that cannot be
//! worked out by looking at it.
//!
//! # Why it is drawn at the focus depth
//!
//! The cutter is a pyramid from the eye, so it has no single depth. It is drawn
//! on the plane through the camera's target, facing the camera, because that is
//! where the model the user is looking at actually is. The consequence is worth
//! stating plainly rather than hiding: the region is screen-exact and the
//! **taper is not shown**. A small loop takes slightly more than its outline at
//! the back of the model than at the front. Every tool with a perspective
//! camera has this, and the alternative -- drawing the frustum's true silhouette
//! -- is a solid that occludes the thing being cut.

use brokkr_gpu::OverlayBatch;
use glam::{Vec2, Vec3};

use crate::cut::CutShape;
use crate::theme::{self, linear};

/// Opacity of the shaded region that is going to be removed.
///
/// Low, and lower than the mirror planes': this sits directly over the material
/// it is about to take, and a fill dense enough to read as a colour would hide
/// the shape of the thing being cut at the moment that shape matters most.
const DOOMED_ALPHA: f32 = 0.16;

/// Opacity of the same region when the cut will go all the way through.
///
/// Denser, because it is taking more. **The point is that the difference is
/// visible WHILE shift is held**, not inferred from the key state at the moment
/// of release: the research names a silent mode flip at release as the top
/// pitfall of this whole family of tools, and this design ships exactly one
/// modifier precisely so that the one it ships can be previewed.
const THROUGH_ALPHA: f32 = 0.34;

/// Opacity of the outline. Nearly solid -- the outline is the measurement, and
/// it has to be legible over both the model and the background.
const OUTLINE_ALPHA: f32 = 0.95;

/// How far the shaded band beside a straight cut line reaches, as a multiple of
/// the model's radius.
///
/// The plane is infinite and the band cannot be, so it reaches comfortably past
/// anything on screen and stops. Matching `cursor`'s `PLANE_OVERSHOOT` in
/// spirit: enough that it reads as "everything that side" rather than as a
/// rectangle of its own.
const BAND_OVERSHOOT: f32 = 3.0;

/// Add the cut preview to an overlay batch.
///
/// `at` maps a point in widget pixels onto the world, and is the caller's
/// `ray_through` closed over the camera -- passed in rather than taken as a
/// camera so that this module stays testable without one.
///
/// `through` says the cut will not be depth-capped -- shift, or a straight drag,
/// which is infinite by definition. It only changes how densely the region is
/// shaded, and that is the whole job: the difference between "this takes the
/// lump" and "this takes everything behind it too" has to be on screen before
/// the button comes up.
///
/// `doomed` is the world-space normal of the cut plane, for the straight drag
/// only: the side it points at is the side that goes. **It is passed in rather
/// than worked out here, and that is the point.** The side convention is the
/// one thing about the plane cut nobody can derive by looking -- it falls out of
/// the ray order, the handedness of the camera basis and the y-flip in the
/// pixel-to-NDC step -- so the preview takes the answer from the same plane the
/// cut is about to use instead of deriving it a second time. A second derivation
/// is a second chance to get it backwards, and getting it backwards here means
/// the preview confidently shades the half the user is about to keep.
///
/// Appends rather than clearing, because the cut preview shares the sculpt
/// overlay batch with the brush ring and the mirror planes. There is no fourth
/// pipeline: `overlay.rs` records that the next overlay should reuse the third
/// rather than add to the list, and this one does.
pub fn build(
    batch: &mut OverlayBatch,
    hull: &[Vec2],
    shape: CutShape,
    model_radius: f32,
    doomed: Option<Vec3>,
    through: bool,
    at: impl Fn(Vec2) -> Vec3,
) {
    let outline = linear(theme::ERROR, OUTLINE_ALPHA);
    let fill = linear(theme::ERROR, if through { THROUGH_ALPHA } else { DOOMED_ALPHA });

    match shape {
        // Two points and no region: draw the line, and shade the side that
        // goes. The side is the one piece of information a user cannot recover
        // by looking, so it is the piece the preview owes them.
        CutShape::Line => {
            let [from, to] = hull else {
                return;
            };
            let (start, end) = (at(*from), at(*to));
            batch.push_line(start, end, outline);

            let (Some(along), Some(normal)) = ((end - start).try_normalize(), doomed) else {
                return;
            };
            // The plane's normal, with the component along the line taken out,
            // which leaves the direction across the line that the plane removes.
            // Gram-Schmidt rather than a cross product with the view: a cross
            // product needs a second vector to be right about, and this needs
            // none.
            let Some(across) = (normal - along * normal.dot(along)).try_normalize() else {
                return;
            };
            let reach = (model_radius * BAND_OVERSHOOT).max(1.0);
            let far = along * reach;
            batch.push_quad(
                start - far,
                end + far,
                end + far + across * reach,
                start - far + across * reach,
                fill,
            );
        }
        CutShape::Curve | CutShape::Lasso => {
            if hull.len() < 3 {
                return;
            }
            let points: Vec<Vec3> = hull.iter().map(|point| at(*point)).collect();
            for index in 0..points.len() {
                batch.push_line(points[index], points[(index + 1) % points.len()], outline);
            }
            // A fan from the first vertex, which is watertight for a convex
            // polygon and is the reason `push_triangle` exists rather than
            // faking a fan out of quads.
            for index in 1..points.len() - 1 {
                batch.push_triangle(points[0], points[index], points[index + 1], fill);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat projection standing in for a camera: pixels straight onto the
    /// z = 0 plane. Enough for every question this module answers, all of which
    /// are about how many primitives come out and in what arrangement.
    fn flat(point: Vec2) -> Vec3 {
        Vec3::new(point.x, point.y, 0.0)
    }

    fn square() -> Vec<Vec2> {
        vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0), Vec2::new(0.0, 10.0)]
    }

    /// A closed shape draws a closed outline: exactly one segment per hull
    /// edge, including the one back to the start.
    #[test]
    fn a_lasso_draws_a_closed_loop_of_one_segment_per_edge() {
        let mut batch = OverlayBatch::default();
        build(&mut batch, &square(), CutShape::Lasso, 30.0, None, false, flat);

        // Two vertices per segment.
        assert_eq!(batch.lines.len(), 4 * 2, "the outline is not one segment per edge");
        // A fan over n points is n - 2 triangles, three vertices each.
        assert_eq!(batch.surfaces.len(), 2 * 3, "the fill is not a fan over the hull");
    }

    /// The outline must actually close. An open outline reads as a curve and
    /// would say the region is not enclosed when it is.
    #[test]
    fn the_outline_returns_to_where_it_started() {
        let mut batch = OverlayBatch::default();
        build(&mut batch, &square(), CutShape::Lasso, 30.0, None, false, flat);

        let first = batch.lines.first().expect("an outline").position;
        let last = batch.lines.last().expect("an outline").position;
        assert_eq!(first, last, "the outline did not close: {first:?} to {last:?}");
    }

    /// The fill sits over the region and never outside it, which is what makes
    /// "what is shaded is what goes" true rather than approximately true.
    #[test]
    fn the_fill_stays_inside_the_outline() {
        let mut batch = OverlayBatch::default();
        build(&mut batch, &square(), CutShape::Lasso, 30.0, None, false, flat);

        for vertex in &batch.surfaces {
            let [x, y, _] = vertex.position;
            assert!(
                (0.0..=10.0).contains(&x) && (0.0..=10.0).contains(&y),
                "a fill vertex left the hull: {:?}",
                vertex.position
            );
        }
    }

    /// A straight drag has no region, so it draws its line and a band on one
    /// side -- and the band must never reach across to the side being KEPT,
    /// which is the half the user is relying on surviving.
    ///
    /// Asserted as "never on the wrong side" rather than "always on the right
    /// side": the quad is anchored ON the line, so two of its corners sit
    /// exactly at zero, and a strict test would read the anchor as a straddle.
    #[test]
    fn a_line_never_shades_the_side_it_keeps() {
        let mut batch = OverlayBatch::default();
        let hull = vec![Vec2::new(0.0, 5.0), Vec2::new(10.0, 5.0)];
        // The plane's normal points at +y, so +y is the side that goes.
        build(&mut batch, &hull, CutShape::Line, 30.0, Some(Vec3::Y), false, flat);

        assert_eq!(batch.lines.len(), 2, "a line is one segment");
        assert!(!batch.surfaces.is_empty(), "a line drew no doomed side");
        for vertex in &batch.surfaces {
            assert!(
                vertex.position[1] >= 5.0,
                "the band reached onto the kept side: {:?}",
                vertex.position
            );
        }
        assert!(
            batch.surfaces.iter().any(|v| v.position[1] > 5.0),
            "the band has no width, so it names no side at all"
        );
    }

    /// And the other normal shades the other side. Without this the test above
    /// would pass on a band that ignored `doomed` and always went one way.
    #[test]
    fn flipping_the_plane_flips_the_shaded_side() {
        let hull = vec![Vec2::new(0.0, 5.0), Vec2::new(10.0, 5.0)];
        let side = |normal| {
            let mut batch = OverlayBatch::default();
            build(&mut batch, &hull, CutShape::Line, 30.0, Some(normal), false, flat);
            batch.surfaces.iter().map(|v| v.position[1]).fold(5.0_f32, f32::max)
        };
        assert!(side(Vec3::Y) > 5.0, "+y did not shade upward");
        assert!(
            side(-Vec3::Y) <= 5.0,
            "-y shaded the same side as +y, so the preview ignores which side goes"
        );
    }

    /// Nothing is drawn for a shape that has no region, rather than a
    /// degenerate primitive that the renderer would silently drop.
    #[test]
    fn a_degenerate_hull_draws_nothing() {
        for (hull, shape) in [
            (vec![], CutShape::Lasso),
            (vec![Vec2::ZERO, Vec2::new(1.0, 1.0)], CutShape::Lasso),
            (vec![], CutShape::Line),
            (vec![Vec2::ZERO], CutShape::Line),
        ] {
            let mut batch = OverlayBatch::default();
            build(&mut batch, &hull, shape, 30.0, Some(Vec3::Y), false, flat);
            assert!(batch.is_empty(), "a degenerate {shape:?} drew something");
        }
    }

    /// The preview appends. It shares the sculpt batch with the brush ring and
    /// the mirror planes, and clearing here would delete them.
    #[test]
    fn building_the_preview_keeps_what_was_already_in_the_batch() {
        let mut batch = OverlayBatch::default();
        batch.push_line(Vec3::ZERO, Vec3::X, [1.0, 1.0, 1.0, 1.0]);
        let before = batch.lines.len();

        build(&mut batch, &square(), CutShape::Lasso, 30.0, None, false, flat);

        assert!(batch.lines.len() > before, "the preview drew nothing");
        assert_eq!(
            batch.lines[0].position,
            Vec3::ZERO.to_array(),
            "the preview cleared the batch it was given"
        );
    }

    /// The one modifier this tool ships is previewed rather than inferred, so
    /// the two states have to actually look different.
    #[test]
    fn cutting_through_is_shaded_differently_from_cutting_to_depth() {
        let alpha = |through| {
            let mut batch = OverlayBatch::default();
            build(&mut batch, &square(), CutShape::Lasso, 30.0, None, through, flat);
            batch.surfaces[0].colour[3]
        };
        assert_ne!(
            alpha(false),
            alpha(true),
            "shift changes what the cut takes and the preview says nothing about it"
        );
        assert!(alpha(true) > alpha(false), "taking MORE should not read as fainter");
    }
}
