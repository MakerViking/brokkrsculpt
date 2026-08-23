// SPDX-License-Identifier: AGPL-3.0-only

//! The icon set: minimal stroke icons on a 24 by 24 grid.
//!
//! The same house style as SindriCAD's `src/ui/icons.ts`, and deliberately so —
//! the two applications are siblings and should look it. Each icon also exists
//! as a hand-written file under `assets/icons/`, carrying the same numbers, the
//! way `logo.rs` and `assets/brand/brokkrsculpt-mark.svg` already do.
//!
//! # House style, hold to it when adding entries
//!
//! A 24 by 24 grid with a roughly 20 by 20 live area, [`STROKE_WIDTH`] set once
//! for the whole set, round caps and joins, everything stroked except the solid
//! dots, and **no colour anywhere in the data**. Colour arrives from the call
//! site, which is the whole of the next section.
//!
//! # There is no `currentColor`, and that is the one real difference
//!
//! SindriCAD's icons carry no colour at all: `stroke="currentColor"` plus the
//! CSS cascade means an icon is whatever colour its control computes to, and
//! hover and selected states need no icon-specific rules.
//!
//! iced has no cascade. A canvas draws with an explicit colour and cannot see
//! the button it sits inside — so the colour has to be handed in, and it has to
//! agree with the button's own style. That agreement is not decorative:
//! [`crate::theme::tool_button`] draws its text `TEXT_DIM`, while
//! [`crate::theme::tool_button_active`] fills with `ACCENT` and drops its text
//! to a near-black `ON_ACCENT`. An icon that kept one colour across both states
//! is invisible in one of them.
//!
//! **So never pick an icon colour beside a button style. Call
//! [`crate::theme::tool_toggle`], which returns both from one `selected`.** A
//! copied constant drifts silently; nothing fails, the strip just stops
//! agreeing with itself.
//!
//! # Why this costs six crates and the mark did not
//!
//! `logo.rs` draws the mark out of `fill_quad` rectangles and its header
//! explains why: `canvas` costs six crates, `image` seventy one, and a logo
//! cannot cost six. That reasoning still holds for a logo. It does not hold for
//! a set of two dozen icons with circles and diagonals in them, which
//! axis-aligned rectangles cannot draw at 24 pixels. `svg` was measured at the
//! same time and costs **thirty one** — resvg, usvg, rustybuzz and four raster
//! decoders for bitmaps an icon never carries. Six is the price of the set.

use iced::widget::canvas;
use iced::{Color, Element, Point, Renderer, Size, Theme};

/// The grid every icon is drawn on, matching the `viewBox` of its SVG twin.
///
/// Geometry below is in these coordinates literally, never in pixels: the frame
/// is scaled once at draw time, so a number here and the same number in
/// `assets/icons/*.svg` mean the same thing and can be read side by side.
pub const VIEWBOX: f32 = 24.0;

/// Stroke width for the whole set, in grid units.
///
/// Set once here rather than per icon, exactly as SindriCAD sets it once on the
/// wrapping `<svg>`. Scaling the frame scales this with it, which is what an
/// SVG `stroke-width` does inside a `viewBox`.
const STROKE_WIDTH: f32 = 1.6;

/// One segment of an icon path, in grid coordinates.
///
/// A small vocabulary on purpose. It is `&'static` data rather than a closure
/// so the tests can walk every icon and check its bounds — which is how
/// SindriCAD found paths that were empty or outside the viewBox.
#[derive(Clone, Copy, PartialEq)]
enum Seg {
    /// Start a new sub-path.
    Move(f32, f32),
    /// Straight line from the current point.
    Line(f32, f32),
    /// An axis-aligned rectangle with a corner radius: `(x, y, w, h, r)`.
    Rect(f32, f32, f32, f32, f32),
}

/// How a sub-path is inked.
#[derive(Clone, Copy, PartialEq)]
enum Ink {
    /// The default, and nearly everything: a stroked outline.
    Stroke,
}

/// One sub-path of an icon, and how it is drawn.
struct Sub {
    segs: &'static [Seg],
    ink: Ink,
}

/// A whole icon: one or more sub-paths.
struct Glyph {
    subs: &'static [Sub],
}

// --- the set -----------------------------------------------------------------

/// Close: an X. Two crossing strokes rather than a glyph, so it cannot be
/// resolved through font fallback into something else.
const CLOSE: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(6.5, 6.5), Seg::Line(17.5, 17.5)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(17.5, 6.5), Seg::Line(6.5, 17.5)], ink: Ink::Stroke },
    ],
};

/// Minimise: a single rule along the bottom of the live area.
///
/// Low rather than centred, because the gesture it stands for is "put it down
/// there" — and because a centred rule is the same drawing as a "remove" or a
/// collapsed section, which is a twin this set does not need.
const MINIMISE: Glyph = Glyph {
    subs: &[Sub { segs: &[Seg::Move(6.5, 16.5), Seg::Line(17.5, 16.5)], ink: Ink::Stroke }],
};

/// Maximise: the window's outline.
const MAXIMISE: Glyph =
    Glyph { subs: &[Sub { segs: &[Seg::Rect(6.5, 6.5, 11.0, 11.0, 1.0)], ink: Ink::Stroke }] };

/// Every icon in the set.
///
/// An enum rather than a string lookup for the reason SindriCAD moved to
/// `keyof typeof PATHS`: a misspelling becomes a compile error instead of a
/// silently empty icon that nobody notices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IconName {
    Close,
    Minimise,
    Maximise,
}

impl IconName {
    /// Every variant, so the tests can sweep the whole set and a new icon
    /// cannot be added without being checked.
    ///
    /// Only the tests walk it today, hence the `allow` — the same argument
    /// `theme.rs` makes at module scope for its unused tokens. This is the
    /// set's index, and anything that needs the whole set rather than one icon
    /// starts here.
    #[allow(dead_code)]
    pub const ALL: [IconName; 3] = [IconName::Close, IconName::Minimise, IconName::Maximise];

    fn glyph(self) -> &'static Glyph {
        match self {
            IconName::Close => &CLOSE,
            IconName::Minimise => &MINIMISE,
            IconName::Maximise => &MAXIMISE,
        }
    }
}

// --- drawing -----------------------------------------------------------------

/// The canvas program for one icon.
///
/// **It deliberately does not implement `update`.** The default returns `None`,
/// and `Canvas` only captures an event when a program returns `Some` with
/// `Captured` — so an icon inside a button never swallows the press that should
/// have fired it. That failure is not hypothetical here: the viewport captured
/// `ButtonReleased` window-wide from M0 until 2026-08-22 and made every button
/// in the properties panel unclickable for two releases. Adding an `update` to
/// this type would be the same bug wearing a different hat.
struct IconProgram {
    glyph: &'static Glyph,
    colour: Color,
}

impl<Message> canvas::Program<Message> for IconProgram {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());

        // One scale for the whole icon, so the geometry above stays in grid
        // coordinates and the stroke width scales with it the way an SVG's
        // does inside its viewBox.
        frame.scale(bounds.width.min(bounds.height) / VIEWBOX);

        for sub in self.glyph.subs {
            let path = build(sub.segs);
            match sub.ink {
                Ink::Stroke => frame.stroke(
                    &path,
                    canvas::Stroke {
                        style: canvas::Style::Solid(self.colour),
                        width: STROKE_WIDTH,
                        line_cap: canvas::LineCap::Round,
                        line_join: canvas::LineJoin::Round,
                        ..canvas::Stroke::default()
                    },
                ),
            }
        }

        vec![frame.into_geometry()]
    }
}

/// Turn a sub-path's segments into a path, in grid coordinates.
fn build(segs: &[Seg]) -> canvas::Path {
    canvas::Path::new(|builder| {
        for seg in segs {
            match *seg {
                Seg::Move(x, y) => builder.move_to(Point::new(x, y)),
                Seg::Line(x, y) => builder.line_to(Point::new(x, y)),
                Seg::Rect(x, y, w, h, r) => builder.rounded_rectangle(
                    Point::new(x, y),
                    Size::new(w, h),
                    iced::border::Radius::new(r),
                ),
            }
        }
    })
}

/// An icon, `size` pixels square, in `colour`.
///
/// The colour is not optional and has no default on purpose — see the module
/// header. Take it from [`crate::theme::tool_toggle`] wherever the control has
/// a selected state.
pub fn icon<'a, Message: 'a>(name: IconName, size: f32, colour: Color) -> Element<'a, Message> {
    canvas(IconProgram { glyph: name.glyph(), colour }).width(size).height(size).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk every point an icon names, so a glyph can be measured without
    /// caring which segment produced a coordinate.
    fn points(glyph: &Glyph) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for sub in glyph.subs {
            for seg in sub.segs {
                match *seg {
                    Seg::Move(x, y) | Seg::Line(x, y) => out.push((x, y)),
                    // Both corners, so a rectangle cannot hang out of the grid
                    // at the far end while its origin sits comfortably inside.
                    Seg::Rect(x, y, w, h, _) => {
                        out.push((x, y));
                        out.push((x + w, y + h));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn every_icon_stays_inside_the_grid() {
        // The stroke straddles the path, so half of it lies outside the
        // coordinates named here. Allowing for that is what makes this a real
        // check rather than one that passes on a clipped icon.
        let margin = STROKE_WIDTH / 2.0;
        for name in IconName::ALL {
            for (x, y) in points(name.glyph()) {
                assert!(
                    x - margin >= 0.0 && x + margin <= VIEWBOX,
                    "{name:?} reaches x={x}, which strokes outside the {VIEWBOX} grid"
                );
                assert!(
                    y - margin >= 0.0 && y + margin <= VIEWBOX,
                    "{name:?} reaches y={y}, which strokes outside the {VIEWBOX} grid"
                );
            }
        }
    }

    #[test]
    fn no_icon_is_empty() {
        // "Long enough to draw something" is per segment kind, not a segment
        // count: a lone `Rect` is a whole drawing, while a lone `Move` is a
        // pen put down and never moved. Counting segments instead called the
        // maximise icon empty, which is how this wording was arrived at.
        fn draws_something(sub: &Sub) -> bool {
            sub.segs.iter().any(|seg| matches!(seg, Seg::Line(..) | Seg::Rect(..)))
        }

        for name in IconName::ALL {
            let glyph = name.glyph();
            assert!(!glyph.subs.is_empty(), "{name:?} has no sub-paths");
            assert!(
                glyph.subs.iter().all(draws_something),
                "{name:?} has a sub-path that puts the pen down and never draws"
            );
        }
    }

    /// The automated half of the twins check.
    ///
    /// Rendering all 108 of SindriCAD's icons together turned up four pairs
    /// that were the same drawing, two of them in the same ribbon. This cannot
    /// catch a pair that merely *looks* alike -- that is what the contact sheet
    /// is for -- but it does catch the copy-paste that produces an exact one.
    #[test]
    fn no_two_icons_are_the_same_drawing() {
        for (i, a) in IconName::ALL.iter().enumerate() {
            for b in &IconName::ALL[i + 1..] {
                let (ga, gb) = (a.glyph(), b.glyph());
                let same = ga.subs.len() == gb.subs.len()
                    && ga.subs.iter().zip(gb.subs).all(|(x, y)| x.ink == y.ink && x.segs == y.segs);
                assert!(!same, "{a:?} and {b:?} are the same drawing");
            }
        }
    }

    #[test]
    fn all_lists_each_icon_once() {
        // What actually guards this list is `glyph()`: its match is exhaustive,
        // so a new variant cannot compile without being given a drawing, and
        // `ALL`'s declared length cannot change without being retyped. What
        // neither catches is the copy-paste that adds a variant to `ALL` twice
        // and leaves another one out, which is exactly what would make every
        // sweep above quietly skip an icon.
        for (i, name) in IconName::ALL.iter().enumerate() {
            assert!(
                !IconName::ALL[i + 1..].contains(name),
                "{name:?} appears twice in IconName::ALL, so something else is missing from it"
            );
        }
    }
}
