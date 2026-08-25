// SPDX-License-Identifier: AGPL-3.0-only

//! The icon set: minimal stroke icons on a 24 by 24 grid.
//!
//! The same house style as SindriCAD's `src/ui/icons.ts`, and deliberately so —
//! the two applications are siblings and should look it.
//!
//! Every icon also exists as a file under `assets/icons/`, **generated from
//! this data** by the `write_the_svg_twins` test rather than maintained beside
//! it. `logo.rs` and `assets/brand/brokkrsculpt-mark.svg` are hand-kept twins
//! and rely on nobody editing one without the other; at twenty-one icons that
//! bet is not worth taking, and generating them means the contact sheet cannot
//! flatter a drawing the application does not actually have.
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

use brokkr_core::{BrushKind, PatternKind};
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
///
/// Named fields on everything past two numbers: `Rect(6.5, 6.5, 11.0, 11.0,
/// 1.0)` reads as five anonymous floats at the call site, and a cubic has six.
#[derive(Clone, Copy, PartialEq)]
enum Seg {
    /// Start a new sub-path.
    Move(f32, f32),
    /// Straight line from the current point.
    Line(f32, f32),
    /// Quadratic Bézier from the current point, through one control point.
    Quad { cx: f32, cy: f32, x: f32, y: f32 },
    /// Cubic Bézier from the current point, through two control points.
    Cubic { c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32 },
    /// A whole circle. Stroked it is an outline, filled it is a dot.
    Circle { cx: f32, cy: f32, r: f32 },
    /// An axis-aligned rectangle with a corner radius.
    Rect { x: f32, y: f32, w: f32, h: f32, r: f32 },
    /// Join the last point back to where this sub-path started.
    Close,
}

/// How a sub-path is inked.
#[derive(Clone, Copy, PartialEq)]
enum Ink {
    /// The default, and nearly everything: a stroked outline.
    Stroke,
    /// Stroked, but broken — a *reference* rather than a thing.
    ///
    /// Worth its own variant because in several icons the dash carries the
    /// whole meaning. SindriCAD's draft angle is the clearest case: without the
    /// dashed vertical saying "this is where it started", a lone trapezoid
    /// reads as a generic taper. Here it is what makes `MOVE` a displacement
    /// rather than a bulge, and what makes `CUT_PLANE` a cut rather than a
    /// solid with a line on it.
    Dashed,
    /// Solid. The house style reserves this for dots and for the one or two
    /// shapes whose whole meaning is that they are solid — a play triangle
    /// outlined reads as a direction, not as a button.
    Fill,
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

/// Minimise: a single rule across the live area.
///
/// It sat four units lower than this at first, to keep it away from a centred
/// rule that the collapse marker was going to be. The collapse marker became a
/// caret instead, so that reason evaporated — and seen at 14 px beside the
/// square and the X, the low rule simply read as misaligned. Centred now,
/// sharing their axis.
const MINIMISE: Glyph = Glyph {
    subs: &[Sub { segs: &[Seg::Move(6.5, 12.0), Seg::Line(17.5, 12.0)], ink: Ink::Stroke }],
};

/// Maximise: the window's outline.
const MAXIMISE: Glyph = Glyph {
    subs: &[Sub {
        segs: &[Seg::Rect { x: 6.5, y: 6.5, w: 11.0, h: 11.0, r: 1.0 }],
        ink: Ink::Stroke,
    }],
};

// --- the viewport's own chrome -----------------------------------------------

/// Info: the circled "i" that opens the stats readout.
///
/// The dot is fatter than a period would be for the reason `PATTERN_NOISE`
/// records: at [`crate::theme::ICON_CHROME`] a grid unit is 0.58 of a pixel, so
/// a radius under about one unit draws thinner than the stroke beside it and
/// reads as a smudge rather than as a dot. At 1.2 it is half again the stroke's
/// width, which is what a tittle looks like.
const INFO: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Circle { cx: 12.0, cy: 12.0, r: 8.5 }], ink: Ink::Stroke },
        Sub { segs: &[Seg::Circle { cx: 12.0, cy: 7.8, r: 1.2 }], ink: Ink::Fill },
        Sub { segs: &[Seg::Move(12.0, 11.3), Seg::Line(12.0, 16.6)], ink: Ink::Stroke },
    ],
};

// --- the timeline's transport ------------------------------------------------

/// Play: a solid triangle.
///
/// Filled rather than outlined because an outlined triangle reads as "this way"
/// — a direction — where a solid one reads as a button. SindriCAD's transport
/// quartet is solid for the same reason.
const PLAY: Glyph = Glyph {
    subs: &[Sub {
        segs: &[Seg::Move(8.5, 5.5), Seg::Line(18.0, 12.0), Seg::Line(8.5, 18.5), Seg::Close],
        ink: Ink::Fill,
    }],
};

/// Stop: a solid square.
const STOP: Glyph = Glyph {
    subs: &[Sub {
        segs: &[Seg::Rect { x: 7.0, y: 7.0, w: 10.0, h: 10.0, r: 1.0 }],
        ink: Ink::Fill,
    }],
};

// --- section headings --------------------------------------------------------

/// An open section: a caret pointing down at the contents it is showing.
///
/// A caret rather than the minus sign this replaced. A centred horizontal rule
/// would have been the same drawing as [`MINIMISE`], and while the two are far
/// apart on screen, "the same drawing means the same thing" is a promise worth
/// keeping across a set. It is also what SindriCAD's tree uses.
const CARET_DOWN: Glyph = Glyph {
    subs: &[Sub {
        segs: &[Seg::Move(6.5, 9.5), Seg::Line(12.0, 15.0), Seg::Line(17.5, 9.5)],
        ink: Ink::Stroke,
    }],
};

/// A closed section: the same caret, turned to point at its heading.
const CARET_RIGHT: Glyph = Glyph {
    subs: &[Sub {
        segs: &[Seg::Move(9.5, 6.5), Seg::Line(15.0, 12.0), Seg::Line(9.5, 17.5)],
        ink: Ink::Stroke,
    }],
};

// --- the pattern modifier ----------------------------------------------------

/// No pattern: the universal "none".
///
/// A ring with a bar through it rather than a bare dash, because this sits in a
/// row of six and has to read as *off* rather than as one more texture.
const PATTERN_NONE: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Circle { cx: 12.0, cy: 12.0, r: 6.5 }], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(7.4, 16.6), Seg::Line(16.6, 7.4)], ink: Ink::Stroke },
    ],
};

/// Noise: scattered dots, deliberately at no spacing you could call a grid.
///
/// The grid is what `WEAVE` is, so an evenly spaced field of dots would have
/// been a twin of it. Irregularity *is* the subject here.
///
/// Five fat dots rather than seven small ones. At `ICON_INLINE` a radius of
/// 1.3 grid units is **0.6 of a pixel**, and seven of those is not a texture,
/// it is a smudge — which is exactly how it looked in the pattern list before
/// anyone measured it.
const PATTERN_NOISE: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Circle { cx: 7.0, cy: 8.0, r: 1.9 }], ink: Ink::Fill },
        Sub { segs: &[Seg::Circle { cx: 13.2, cy: 5.8, r: 1.9 }], ink: Ink::Fill },
        Sub { segs: &[Seg::Circle { cx: 17.6, cy: 10.8, r: 1.9 }], ink: Ink::Fill },
        Sub { segs: &[Seg::Circle { cx: 8.8, cy: 15.4, r: 1.9 }], ink: Ink::Fill },
        Sub { segs: &[Seg::Circle { cx: 16.0, cy: 17.2, r: 1.9 }], ink: Ink::Fill },
    ],
};

/// Scales: overlapping arcs, offset row to row the way real scales lie.
///
/// Two big scallops and one below rather than three small ones. Three arcs on
/// a 24 grid are about four pixels each in the pattern list, and four-pixel
/// arcs abutting each other do not read as scales, they read as a squiggle.
const PATTERN_SCALES: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[Seg::Move(2.5, 12.0), Seg::Quad { cx: 7.25, cy: 3.5, x: 12.0, y: 12.0 }],
            ink: Ink::Stroke,
        },
        Sub {
            segs: &[Seg::Move(12.0, 12.0), Seg::Quad { cx: 16.75, cy: 3.5, x: 21.5, y: 12.0 }],
            ink: Ink::Stroke,
        },
        Sub {
            segs: &[Seg::Move(7.25, 21.0), Seg::Quad { cx: 12.0, cy: 12.5, x: 16.75, y: 21.0 }],
            ink: Ink::Stroke,
        },
    ],
};

/// Hair: strokes that all comb the same way.
///
/// Hair is the one pattern evaluated along the stroke rather than in world
/// space — it combs along the drag — so every line leaning together is the
/// honest drawing of it.
const PATTERN_HAIR: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[
                Seg::Move(4.5, 19.5),
                Seg::Cubic { c1x: 5.5, c1y: 13.0, c2x: 7.0, c2y: 8.0, x: 9.5, y: 4.5 },
            ],
            ink: Ink::Stroke,
        },
        Sub {
            segs: &[
                Seg::Move(9.5, 19.5),
                Seg::Cubic { c1x: 10.5, c1y: 13.0, c2x: 12.0, c2y: 8.0, x: 14.5, y: 4.5 },
            ],
            ink: Ink::Stroke,
        },
        Sub {
            segs: &[
                Seg::Move(14.5, 19.5),
                Seg::Cubic { c1x: 15.5, c1y: 13.0, c2x: 17.0, c2y: 8.0, x: 19.5, y: 4.5 },
            ],
            ink: Ink::Stroke,
        },
    ],
};

/// Weave: threads crossing, with the verticals broken where they pass under.
///
/// The breaks are the whole icon. An unbroken grid is a grid — SindriCAD's
/// `texture` — and says nothing about interlacing.
const PATTERN_WEAVE: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(4.0, 9.0), Seg::Line(20.0, 9.0)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(4.0, 15.0), Seg::Line(20.0, 15.0)], ink: Ink::Stroke },
        // Broken at y=9, so it passes under the upper thread and over the lower.
        Sub { segs: &[Seg::Move(9.0, 4.0), Seg::Line(9.0, 7.6)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(9.0, 10.4), Seg::Line(9.0, 20.0)], ink: Ink::Stroke },
        // And the mirror of it, so the over-under alternates like real weave.
        Sub { segs: &[Seg::Move(15.0, 4.0), Seg::Line(15.0, 13.6)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(15.0, 16.4), Seg::Line(15.0, 20.0)], ink: Ink::Stroke },
    ],
};

/// Cracks: one split that forks, because a crack that never forks is a line.
///
/// **One** branch, not two. Two branches leave three strokes meeting near the
/// middle, and at small sizes three strokes radiating from a point is a
/// sparkle — which is what this drew in the pattern list, and a sparkle is
/// SindriCAD's `cleanUp`, the opposite of carving damage in.
const PATTERN_CRACKS: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[
                Seg::Move(13.2, 2.6),
                Seg::Line(10.0, 10.5),
                Seg::Line(13.2, 15.5),
                Seg::Line(10.6, 21.4),
            ],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Move(10.0, 10.5), Seg::Line(3.6, 13.4)], ink: Ink::Stroke },
    ],
};

// --- the brushes -------------------------------------------------------------
//
// Seven tools that all push a surface around, which is exactly the setup that
// produces twins: SindriCAD rendered its 108 icons together and found four
// pairs that were the same drawing, and a sculpting strip is worse, because
// "draw", "clay", "inflate" and "pinch" are all a lump appearing on a curve.
//
// So none of these is a variation on a lump. Each one draws the SUBJECT of its
// operation instead -- what the tool is *about* rather than what its result
// vaguely looks like. Smooth is a rough line above a calm one; pinch is two
// arrows squeezing a crease; move is where the surface was against where it is.
// That is also SindriCAD's own answer to its twins, arrived at the same way.

/// Draw: one dome deposited on a flat surface.
///
/// The plainest of the seven on purpose — it is the default brush, and the
/// baseline the other six read as departures from.
const BRUSH_DRAW: Glyph = Glyph {
    subs: &[Sub {
        segs: &[
            Seg::Move(3.5, 17.0),
            Seg::Line(8.5, 17.0),
            Seg::Quad { cx: 12.0, cy: 8.0, x: 15.5, y: 17.0 },
            Seg::Line(20.5, 17.0),
        ],
        ink: Ink::Stroke,
    }],
};

/// Clay: material added in slabs.
///
/// Stacked rather than domed, which is the whole difference from `BRUSH_DRAW`:
/// clay builds a surface up in layers toward a target, and two blocks sitting
/// on the ground say that where a second smooth hump would only have said
/// "draw again".
const BRUSH_CLAY: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(3.5, 18.5), Seg::Line(20.5, 18.5)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Rect { x: 6.0, y: 13.5, w: 12.0, h: 3.2, r: 0.8 }], ink: Ink::Stroke },
        Sub { segs: &[Seg::Rect { x: 8.5, y: 8.8, w: 7.0, h: 3.2, r: 0.8 }], ink: Ink::Stroke },
    ],
};

/// Smooth: a rough line, and the calm one it becomes.
///
/// Two lines rather than one transitioning, because a single line that starts
/// jagged and ends smooth needs a transition zone, and at 18 px the transition
/// is where all the pixels go and none of the meaning is.
const BRUSH_SMOOTH: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[
                Seg::Move(3.5, 9.0),
                Seg::Line(6.0, 5.5),
                Seg::Line(8.5, 9.0),
                Seg::Line(11.0, 5.5),
                Seg::Line(13.5, 9.0),
                Seg::Line(16.0, 5.5),
                Seg::Line(18.5, 9.0),
                Seg::Line(20.5, 7.0),
            ],
            ink: Ink::Stroke,
        },
        Sub {
            segs: &[
                Seg::Move(3.5, 17.0),
                Seg::Cubic { c1x: 8.0, c1y: 13.5, c2x: 16.0, c2y: 20.5, x: 20.5, y: 16.0 },
            ],
            ink: Ink::Stroke,
        },
    ],
};

/// Inflate: a closed form swelling along its own normals.
///
/// A whole body rather than a patch of surface, because that is the actual
/// difference from draw — inflate pushes everywhere at once, so the icon has
/// to have an "everywhere" in it for the radiating ticks to mean anything.
const BRUSH_INFLATE: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Circle { cx: 12.0, cy: 12.0, r: 5.5 }], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(15.9, 8.1), Seg::Line(18.6, 5.4)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(8.1, 8.1), Seg::Line(5.4, 5.4)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(15.9, 15.9), Seg::Line(18.6, 18.6)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(8.1, 15.9), Seg::Line(5.4, 18.6)], ink: Ink::Stroke },
    ],
};

/// Pinch: two arrows squeezing material onto a crease.
///
/// Deliberately NOT "a sharp peak instead of a round dome". That was the first
/// drawing, and beside `BRUSH_DRAW` at 18 px it was a twin — the eye reads
/// "bump on a line" for both and the corner radius is a couple of pixels of
/// difference. Arrows converging say what pinch does; a pointier lump does not.
const BRUSH_PINCH: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(12.0, 4.0), Seg::Line(12.0, 20.0)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(4.5, 12.0), Seg::Line(9.3, 12.0)], ink: Ink::Stroke },
        Sub {
            segs: &[Seg::Move(7.4, 10.1), Seg::Line(9.3, 12.0), Seg::Line(7.4, 13.9)],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Move(19.5, 12.0), Seg::Line(14.7, 12.0)], ink: Ink::Stroke },
        Sub {
            segs: &[Seg::Move(16.6, 10.1), Seg::Line(14.7, 12.0), Seg::Line(16.6, 13.9)],
            ink: Ink::Stroke,
        },
    ],
};

/// Flatten: a bump with its top taken off, level with a plane.
///
/// Two passes to get here, both driven by the contact sheet. It began as humps
/// with a plane line through them, which was two curves plus a straight — and
/// `BRUSH_SMOOTH` is two curves, so at 14 px both were "a squiggle" and
/// `BRUSH_DRAW` was a third. Flattening the top fixed that pair and left a new
/// one: round-topped bump against square-topped bump is a couple of pixels of
/// difference. The dashed level is what separates them, and it is the honest
/// drawing anyway — flatten blends toward a plane, and this is the plane.
const BRUSH_FLATTEN: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[
                Seg::Move(3.5, 18.0),
                Seg::Line(7.5, 18.0),
                Seg::Line(9.5, 11.5),
                Seg::Line(14.5, 11.5),
                Seg::Line(16.5, 18.0),
                Seg::Line(20.5, 18.0),
            ],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Move(2.8, 11.5), Seg::Line(21.2, 11.5)], ink: Ink::Dashed },
    ],
};

/// Move: where the surface was, against where it is now.
///
/// The dashed original is the icon. Without it this is a curve, and a curve is
/// every other brush's result too — displacement is only visible as a
/// comparison, which is exactly why Move locks the field at stroke start and
/// warps from that snapshot rather than integrating increments.
const BRUSH_MOVE: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(6.0, 4.5), Seg::Line(6.0, 19.5)], ink: Ink::Dashed },
        Sub {
            segs: &[
                Seg::Move(6.0, 4.5),
                Seg::Cubic { c1x: 19.5, c1y: 8.5, c2x: 19.5, c2y: 15.5, x: 6.0, y: 19.5 },
            ],
            ink: Ink::Stroke,
        },
    ],
};

/// The plane cut: a solid, and the plane going through it.
const CUT_PLANE: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[
                Seg::Move(5.0, 9.0),
                Seg::Line(12.0, 5.0),
                Seg::Line(19.0, 9.0),
                Seg::Line(19.0, 16.0),
                Seg::Line(12.0, 20.0),
                Seg::Line(5.0, 16.0),
                Seg::Close,
            ],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Move(3.5, 14.0), Seg::Line(20.5, 10.5)], ink: Ink::Dashed },
    ],
};

// --- the body panel ----------------------------------------------------------

/// Shown: an eye.
///
/// Two arcs meeting at the corners rather than a circle in a lens, because at
/// [`crate::theme::ICON_INLINE`] -- 11 px, where a grid unit is 0.46 of a pixel
/// -- an outlined pupil inside an outlined lens is two rings 1.5 px apart and
/// reads as a blob. The pupil is filled for the reason [`INFO`]'s dot is.
const EYE: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[
                Seg::Move(3.5, 12.0),
                Seg::Quad { cx: 12.0, cy: 4.5, x: 20.5, y: 12.0 },
                Seg::Quad { cx: 12.0, cy: 19.5, x: 3.5, y: 12.0 },
                Seg::Close,
            ],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Circle { cx: 12.0, cy: 12.0, r: 2.4 }], ink: Ink::Fill },
    ],
};

/// Hidden: the same eye, struck through, and **with the pupil gone**.
///
/// The pupil is dropped rather than kept under the slash, and that is the whole
/// design of this pair. `no_two_icons_are_the_same_drawing` is exact structural
/// equality, so an eye and an eye-plus-hairline pass it trivially while being
/// one dot apart at 11 px. Losing the dark centre changes the icon's whole
/// weight, which is a difference that survives being small — and the row's name
/// is muted alongside it, so the eye is never the only signal.
const EYE_OFF: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[
                Seg::Move(3.5, 12.0),
                Seg::Quad { cx: 12.0, cy: 4.5, x: 20.5, y: 12.0 },
                Seg::Quad { cx: 12.0, cy: 19.5, x: 3.5, y: 12.0 },
                Seg::Close,
            ],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Move(5.0, 19.5), Seg::Line(19.0, 4.5)], ink: Ink::Stroke },
    ],
};

/// Add: a plus.
///
/// Upright, where [`CLOSE`] is the same two strokes turned forty-five degrees.
/// The pair is the one place in this set where orientation carries the entire
/// meaning, which is why both are centred on the same axes and drawn at the same
/// length.
const PLUS: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(12.0, 6.0), Seg::Line(12.0, 18.0)], ink: Ink::Stroke },
        Sub { segs: &[Seg::Move(6.0, 12.0), Seg::Line(18.0, 12.0)], ink: Ink::Stroke },
    ],
};

/// Delete: a bin, with its lid and its handle.
///
/// Three sub-paths rather than a tapered box on its own: at 11 px the taper is
/// under a pixel of difference across the whole glyph, and it is the lid line
/// that makes this a bin rather than a bucket.
const TRASH: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(5.0, 7.0), Seg::Line(19.0, 7.0)], ink: Ink::Stroke },
        Sub {
            segs: &[
                Seg::Move(7.0, 7.0),
                Seg::Line(8.2, 19.5),
                Seg::Line(15.8, 19.5),
                Seg::Line(17.0, 7.0),
            ],
            ink: Ink::Stroke,
        },
        Sub {
            segs: &[
                Seg::Move(10.0, 7.0),
                Seg::Line(10.0, 4.5),
                Seg::Line(14.0, 4.5),
                Seg::Line(14.0, 7.0),
            ],
            ink: Ink::Stroke,
        },
    ],
};

/// Duplicate: one sheet in front of another.
///
/// **The sheet behind is an L and not a second rectangle**, and that is the
/// whole difference between a copy icon and a smudge at 11 px. Drawn whole, its
/// right and bottom edges run *underneath* the front sheet, so at shipping size
/// the two overlap into a grid of five or six lines and read as a table. The L
/// keeps every stroke of the back sheet clear of the front one -- the upright
/// sits left of `x = 8.5` and the bar sits above `y = 8.5` -- so nothing
/// crosses anything and the offset is what the eye picks up.
const COPY: Glyph = Glyph {
    subs: &[
        Sub {
            segs: &[Seg::Move(5.0, 15.0), Seg::Line(5.0, 5.0), Seg::Line(15.0, 5.0)],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Rect { x: 8.5, y: 8.5, w: 11.0, h: 11.0, r: 1.5 }], ink: Ink::Stroke },
    ],
};

/// Solo: a ring around a single dot — "this one, and nothing else".
///
/// **A bullseye and not a bare ring.** A lone stroked circle at
/// [`crate::theme::ICON_INLINE`] is a 6 px `o` that reads as a letter, and it
/// would sit one column away from [`EYE`] in the same row where the two have to
/// be told apart at a glance. The centre dot is what makes it a target: the
/// outer ring is the document and the dot is the one row inside it that is
/// left, which is exactly what the mode does.
///
/// The dot is [`EYE`]'s pupil at the same radius on purpose — 2.4 is the size
/// this set has already settled on for a filled centre, and copying it is what
/// keeps the two icons looking like they came from one hand. The ring is what
/// carries the difference, and the two are never the same drawing because
/// [`EYE`]'s outer sub-path is an almond of two quadratics.
const SOLO: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Circle { cx: 12.0, cy: 12.0, r: 7.0 }], ink: Ink::Stroke },
        Sub { segs: &[Seg::Circle { cx: 12.0, cy: 12.0, r: 2.4 }], ink: Ink::Fill },
    ],
};

/// Merge down: an arrow dropping onto the row beneath it.
///
/// **A bar and not a second sheet**, which is what keeps this clear of [`COPY`]
/// two columns along in the same verb row. Copy's meaning is two objects, so
/// anything here built out of two rectangles would read as a copy with a
/// decoration. The bar is the row below and the arrow is what lands in it: one
/// object, one direction, and nothing that resembles a page.
///
/// It is also not [`CARET_DOWN`] with a line under it. The caret is the section
/// heading's disclosure and appears on folder rows in this very panel, so the
/// arrow carries a shaft the caret does not have — at
/// [`crate::theme::ICON_INLINE`] the shaft is the whole of the difference, and
/// the two are never seen at the same size in the same column.
const MERGE_DOWN: Glyph = Glyph {
    subs: &[
        Sub { segs: &[Seg::Move(12.0, 4.0), Seg::Line(12.0, 12.8)], ink: Ink::Stroke },
        Sub {
            segs: &[Seg::Move(8.0, 8.8), Seg::Line(12.0, 12.8), Seg::Line(16.0, 8.8)],
            ink: Ink::Stroke,
        },
        Sub { segs: &[Seg::Move(5.0, 17.5), Seg::Line(19.0, 17.5)], ink: Ink::Stroke },
    ],
};

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
    Info,
    Play,
    Stop,
    CaretDown,
    CaretRight,
    PatternNone,
    PatternNoise,
    PatternScales,
    PatternHair,
    PatternWeave,
    PatternCracks,
    BrushDraw,
    BrushClay,
    BrushSmooth,
    BrushInflate,
    BrushPinch,
    BrushFlatten,
    BrushMove,
    CutPlane,
    Eye,
    EyeOff,
    Plus,
    Trash,
    Copy,
    Solo,
    MergeDown,
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
    pub const ALL: [IconName; 29] = [
        IconName::Close,
        IconName::Minimise,
        IconName::Maximise,
        IconName::Info,
        IconName::Play,
        IconName::Stop,
        IconName::CaretDown,
        IconName::CaretRight,
        IconName::PatternNone,
        IconName::PatternNoise,
        IconName::PatternScales,
        IconName::PatternHair,
        IconName::PatternWeave,
        IconName::PatternCracks,
        IconName::BrushDraw,
        IconName::BrushClay,
        IconName::BrushSmooth,
        IconName::BrushInflate,
        IconName::BrushPinch,
        IconName::BrushFlatten,
        IconName::BrushMove,
        IconName::CutPlane,
        IconName::Eye,
        IconName::EyeOff,
        IconName::Plus,
        IconName::Trash,
        IconName::Copy,
        IconName::Solo,
        IconName::MergeDown,
    ];

    /// The icon standing for a sculpting brush.
    ///
    /// Exhaustive for the same reason [`Self::for_pattern`] is: an eighth brush
    /// cannot be added without being drawn.
    pub fn for_brush(kind: BrushKind) -> Self {
        match kind {
            BrushKind::Draw => IconName::BrushDraw,
            BrushKind::Clay => IconName::BrushClay,
            BrushKind::Smooth => IconName::BrushSmooth,
            BrushKind::Inflate => IconName::BrushInflate,
            BrushKind::Pinch => IconName::BrushPinch,
            BrushKind::Flatten => IconName::BrushFlatten,
            BrushKind::Move => IconName::BrushMove,
        }
    }

    /// The icon standing for a surface pattern.
    ///
    /// A match rather than a lookup table, so a new `PatternKind` cannot be
    /// added without being given a drawing — the same guarantee SindriCAD gets
    /// from typing `FEATURE_META` against `IconName`, where a feature type with
    /// no icon is a compile error rather than a blank square at run time.
    pub fn for_pattern(kind: PatternKind) -> Self {
        match kind {
            PatternKind::None => IconName::PatternNone,
            PatternKind::Noise => IconName::PatternNoise,
            PatternKind::Scales => IconName::PatternScales,
            PatternKind::Hair => IconName::PatternHair,
            PatternKind::Weave => IconName::PatternWeave,
            PatternKind::Cracks => IconName::PatternCracks,
        }
    }

    fn glyph(self) -> &'static Glyph {
        match self {
            IconName::Close => &CLOSE,
            IconName::Minimise => &MINIMISE,
            IconName::Maximise => &MAXIMISE,
            IconName::Info => &INFO,
            IconName::Play => &PLAY,
            IconName::Stop => &STOP,
            IconName::CaretDown => &CARET_DOWN,
            IconName::CaretRight => &CARET_RIGHT,
            IconName::PatternNone => &PATTERN_NONE,
            IconName::PatternNoise => &PATTERN_NOISE,
            IconName::PatternScales => &PATTERN_SCALES,
            IconName::PatternHair => &PATTERN_HAIR,
            IconName::PatternWeave => &PATTERN_WEAVE,
            IconName::PatternCracks => &PATTERN_CRACKS,
            IconName::BrushDraw => &BRUSH_DRAW,
            IconName::BrushClay => &BRUSH_CLAY,
            IconName::BrushSmooth => &BRUSH_SMOOTH,
            IconName::BrushInflate => &BRUSH_INFLATE,
            IconName::BrushPinch => &BRUSH_PINCH,
            IconName::BrushFlatten => &BRUSH_FLATTEN,
            IconName::BrushMove => &BRUSH_MOVE,
            IconName::CutPlane => &CUT_PLANE,
            IconName::Eye => &EYE,
            IconName::EyeOff => &EYE_OFF,
            IconName::Plus => &PLUS,
            IconName::Trash => &TRASH,
            IconName::Copy => &COPY,
            IconName::Solo => &SOLO,
            IconName::MergeDown => &MERGE_DOWN,
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
                Ink::Stroke => {
                    frame.stroke(&path, stroke_of(self.colour, canvas::LineDash::default()));
                }
                // In grid units, so the dash scales with everything else. It
                // started at 1.6 on / 2.2 off, which is 1.2 px on at the tool
                // strip's 18 px -- under the stroke width, so it greyed out
                // into a faint solid line and `BRUSH_MOVE` read as a plain "D"
                // with no reference to be displaced from. Sized so the ON
                // segment stays wider than the stroke at the smallest size the
                // set is drawn at.
                Ink::Dashed => {
                    let dash = canvas::LineDash { segments: &DASH, offset: 0 };
                    frame.stroke(&path, stroke_of(self.colour, dash));
                }
                Ink::Fill => frame.fill(&path, self.colour),
            }
        }

        vec![frame.into_geometry()]
    }
}

/// The house style's stroke, in whichever dash pattern is wanted.
///
/// A function rather than a closure over `self.colour`: the returned `Stroke`
/// borrows the dash segments, and a closure cannot express that its output
/// lives exactly as long as its input, so the borrow checker refuses it.
fn stroke_of(colour: Color, dash: canvas::LineDash<'_>) -> canvas::Stroke<'_> {
    canvas::Stroke {
        style: canvas::Style::Solid(colour),
        width: STROKE_WIDTH,
        line_cap: canvas::LineCap::Round,
        line_join: canvas::LineJoin::Round,
        line_dash: dash,
    }
}

/// Turn a sub-path's segments into a path, in grid coordinates.
fn build(segs: &[Seg]) -> canvas::Path {
    canvas::Path::new(|builder| {
        for seg in segs {
            match *seg {
                Seg::Move(x, y) => builder.move_to(Point::new(x, y)),
                Seg::Line(x, y) => builder.line_to(Point::new(x, y)),
                Seg::Quad { cx, cy, x, y } => {
                    builder.quadratic_curve_to(Point::new(cx, cy), Point::new(x, y));
                }
                Seg::Cubic { c1x, c1y, c2x, c2y, x, y } => {
                    builder.bezier_curve_to(
                        Point::new(c1x, c1y),
                        Point::new(c2x, c2y),
                        Point::new(x, y),
                    );
                }
                Seg::Circle { cx, cy, r } => builder.circle(Point::new(cx, cy), r),
                Seg::Rect { x, y, w, h, r } => builder.rounded_rectangle(
                    Point::new(x, y),
                    Size::new(w, h),
                    iced::border::Radius::new(r),
                ),
                Seg::Close => builder.close(),
            }
        }
    })
}

/// The dash pattern, in grid units.
///
/// A constant because the SVG twins have to emit the identical
/// `stroke-dasharray`, and two places writing "2.6 2.2" independently is the
/// drift this module goes out of its way to avoid.
const DASH: [f32; 2] = [2.6, 2.2];

/// An icon, `size` pixels square, in `colour`.
///
/// The colour is not optional and has no default on purpose — see the module
/// header. Take it from [`crate::theme::tool_toggle`] wherever the control has
/// a selected state.
pub fn icon<'a, Message: 'a>(name: IconName, size: f32, colour: Color) -> Element<'a, Message> {
    canvas(IconProgram { glyph: name.glyph(), colour }).width(size).height(size).into()
}

/// The kebab-case name an icon's SVG twin is filed under.
#[cfg(test)]
fn file_stem(name: IconName) -> String {
    let raw = format!("{name:?}");
    let mut out = String::new();
    for (i, ch) in raw.char_indices() {
        if ch.is_uppercase() && i > 0 {
            out.push('-');
        }
        out.extend(ch.to_lowercase());
    }
    out
}

/// One icon as a standalone SVG document.
///
/// **Generated from the same `&'static` data the application draws**, rather
/// than hand-written beside it. `logo.rs` and `brokkrsculpt-mark.svg` are twins
/// maintained by hand and rely on nobody editing one without the other; with
/// twenty-one icons that bet is not worth taking, and generating them means the
/// contact sheet cannot flatter a drawing the application does not have.
///
/// `currentColor` throughout, so the files drop into a web page or a README and
/// take their colour from it — which is how SindriCAD's set works, and the one
/// part of its design iced could not give us at run time.
///
/// Test-only: the application draws from [`Glyph`] directly and never reads
/// these files, so this exists purely to write them and to be checked.
#[cfg(test)]
fn to_svg(name: IconName) -> String {
    fn n(v: f32) -> String {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }

    let mut body = String::new();
    for sub in name.glyph().subs {
        let extra = match sub.ink {
            Ink::Stroke => String::new(),
            Ink::Dashed => format!(r#" stroke-dasharray="{} {}""#, n(DASH[0]), n(DASH[1])),
            Ink::Fill => r#" fill="currentColor" stroke="none""#.to_string(),
        };

        // A sub-path is either one whole shape or a run of path commands. Both
        // exist because a circle written as four arcs is unreadable in a file
        // somebody may want to open in an editor.
        match sub.segs {
            [Seg::Circle { cx, cy, r }] => body.push_str(&format!(
                "\n  <circle cx=\"{}\" cy=\"{}\" r=\"{}\"{extra}/>",
                n(*cx),
                n(*cy),
                n(*r)
            )),
            [Seg::Rect { x, y, w, h, r }] => body.push_str(&format!(
                "\n  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"{}\"{extra}/>",
                n(*x),
                n(*y),
                n(*w),
                n(*h),
                n(*r)
            )),
            segs => {
                let mut d = String::new();
                for seg in segs {
                    if !d.is_empty() {
                        d.push(' ');
                    }
                    match *seg {
                        Seg::Move(x, y) => d.push_str(&format!("M{} {}", n(x), n(y))),
                        Seg::Line(x, y) => d.push_str(&format!("L{} {}", n(x), n(y))),
                        Seg::Quad { cx, cy, x, y } => {
                            d.push_str(&format!("Q{} {} {} {}", n(cx), n(cy), n(x), n(y)));
                        }
                        Seg::Cubic { c1x, c1y, c2x, c2y, x, y } => d.push_str(&format!(
                            "C{} {} {} {} {} {}",
                            n(c1x),
                            n(c1y),
                            n(c2x),
                            n(c2y),
                            n(x),
                            n(y)
                        )),
                        Seg::Circle { cx, cy, r } => {
                            // Only reachable if a circle is ever mixed into a
                            // run; approximated so the file still draws.
                            d.push_str(&format!(
                                "M{} {} a{} {} 0 1 0 {} 0 a{} {} 0 1 0 -{} 0",
                                n(cx - r),
                                n(cy),
                                n(r),
                                n(r),
                                n(r * 2.0),
                                n(r),
                                n(r),
                                n(r * 2.0)
                            ));
                        }
                        Seg::Rect { x, y, w, h, .. } => d.push_str(&format!(
                            "M{} {} L{} {} L{} {} L{} {} Z",
                            n(x),
                            n(y),
                            n(x + w),
                            n(y),
                            n(x + w),
                            n(y + h),
                            n(x),
                            n(y + h)
                        )),
                        Seg::Close => d.push('Z'),
                    }
                }
                body.push_str(&format!("\n  <path d=\"{d}\"{extra}/>"));
            }
        }
    }

    format!(
        concat!(
            // No double hyphen anywhere in here. XML forbids `--` inside a
            // comment, and the first version of this line named the cargo
            // command that generates the file -- which contains `-- --ignored`
            // -- so all twenty-one files were unparseable XML. They rendered
            // in nothing, and the only reason it surfaced is that something
            // tried to rasterise one.
            "<!-- Generated from crates/brokkr-app/src/icon.rs by the\n",
            "     write_the_svg_twins test. Edit the drawing there, not here. -->\n",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" fill=\"none\"\n",
            "     stroke=\"currentColor\" stroke-width=\"{}\" stroke-linecap=\"round\"\n",
            "     stroke-linejoin=\"round\">{}\n</svg>\n"
        ),
        n(VIEWBOX),
        n(VIEWBOX),
        n(STROKE_WIDTH),
        body
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk every point an icon names, so a glyph can be measured without
    /// caring which segment produced a coordinate.
    ///
    /// A curve's control points are included even though the curve itself
    /// stays inside its hull. That makes the bounds check **conservative**: a
    /// design could in principle be refused for a control point that never
    /// draws anywhere near the edge. That is the safe direction to be wrong in,
    /// and no icon in the set has needed the slack.
    fn points(glyph: &Glyph) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for sub in glyph.subs {
            for seg in sub.segs {
                match *seg {
                    Seg::Move(x, y) | Seg::Line(x, y) => out.push((x, y)),
                    Seg::Quad { cx, cy, x, y } => {
                        out.push((cx, cy));
                        out.push((x, y));
                    }
                    Seg::Cubic { c1x, c1y, c2x, c2y, x, y } => {
                        out.push((c1x, c1y));
                        out.push((c2x, c2y));
                        out.push((x, y));
                    }
                    Seg::Circle { cx, cy, r } => {
                        out.push((cx - r, cy - r));
                        out.push((cx + r, cy + r));
                    }
                    // Both corners, so a rectangle cannot hang out of the grid
                    // at the far end while its origin sits comfortably inside.
                    Seg::Rect { x, y, w, h, .. } => {
                        out.push((x, y));
                        out.push((x + w, y + h));
                    }
                    Seg::Close => {}
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
            sub.segs.iter().any(|seg| {
                matches!(
                    seg,
                    Seg::Line(..)
                        | Seg::Quad { .. }
                        | Seg::Cubic { .. }
                        | Seg::Circle { .. }
                        | Seg::Rect { .. }
                )
            })
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

    #[test]
    fn every_svg_twin_carries_the_house_style_and_no_colour() {
        for name in IconName::ALL {
            let svg = to_svg(name);
            assert!(svg.contains(r#"viewBox="0 0 24 24""#), "{name:?} lost its viewBox");
            assert!(svg.contains(r#"stroke-width="1.6""#), "{name:?} lost the house stroke");
            assert!(svg.contains(r#"stroke="currentColor""#), "{name:?} lost currentColor");
            // The one rule the whole set rests on. A colour baked into a twin
            // would look right on a dark contact sheet and then refuse to
            // follow a selected button, which is the failure this module goes
            // out of its way to design against.
            assert!(
                !svg.contains('#') && !svg.to_lowercase().contains("rgb("),
                "{name:?} has a colour baked into it"
            );
            assert!(
                svg.contains("<path") || svg.contains("<circle") || svg.contains("<rect"),
                "{name:?} emitted no shapes"
            );

            // XML forbids `--` inside a comment, and the header comment here
            // once named the cargo command that writes these files -- which
            // contains one. Every file was unparseable and rendered as nothing
            // at all, which no amount of looking at the application would have
            // shown, because the application does not read them.
            let comment = svg.split("-->").next().unwrap_or_default();
            let opened = comment.strip_prefix("<!--").unwrap_or(comment);
            assert!(
                !opened.contains("--"),
                "{name:?} has a double hyphen in its comment, so the file is not valid XML"
            );
        }
    }

    /// Write the SVG twins and a contact sheet, then go and look at them.
    ///
    /// `#[ignore]`d because it writes into the repository — the same reason the
    /// bug reporter's live test is ignored for filing a real report. Run it
    /// with `cargo test -p brokkr-app -- --ignored write_the_svg_twins
    /// --nocapture` after changing any drawing.
    ///
    /// The sheet renders every icon at the three sizes the application actually
    /// uses, because that is the only way this catches anything. SindriCAD
    /// found four twins and two illegible drawings only once the whole set was
    /// rendered together **at shipping size**, and judging enlarged had hidden
    /// all six. This project has now repeated that twice: a pinch that was a
    /// twin of draw at 18 px, and a noise whose dots came out 0.6 px across.
    #[test]
    #[ignore]
    fn write_the_svg_twins() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = root.join("assets/icons");
        std::fs::create_dir_all(&dir).expect("creating assets/icons");

        let mut cells = String::new();
        for name in IconName::ALL {
            let stem = file_stem(name);
            let svg = to_svg(name);
            std::fs::write(dir.join(format!("{stem}.svg")), &svg).expect("writing an icon");
            cells.push_str(&format!(
                "<figure><div class=\"row\">\
                 <span class=\"s24\">{svg}</span>\
                 <span class=\"s18\">{svg}</span>\
                 <span class=\"s14\">{svg}</span></div>\
                 <figcaption>{stem}</figcaption></figure>\n"
            ));
        }

        let sheet = format!(
            "<!doctype html><meta charset=utf-8>\n\
             <title>BrokkrSculpt icons</title>\n\
             <style>\n\
             body{{background:#0e0f12;color:#9aa3af;\
             font:13px/1.4 Inter,system-ui,sans-serif;margin:0;padding:28px}}\n\
             h1{{color:#e6e8ec;font-size:15px;font-weight:600;margin:0 0 4px}}\n\
             p{{color:#6b7280;font-size:11px;margin:0 0 24px}}\n\
             .grid{{display:grid;\
             grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:18px}}\n\
             figure{{margin:0;background:#16181d;border:1px solid #262a31;\
             border-radius:6px;padding:14px 10px}}\n\
             .row{{display:flex;align-items:center;justify-content:center;\
             gap:12px;min-height:34px}}\n\
             .s24 svg{{width:24px;height:24px;display:block}}\n\
             .s18 svg{{width:18px;height:18px;display:block}}\n\
             .s14 svg{{width:14px;height:14px;display:block}}\n\
             figcaption{{color:#6b7280;font-size:10px;text-align:center;\
             margin-top:10px}}\n\
             </style>\n\
             <h1>BrokkrSculpt icons</h1>\n\
             <p>Every icon at 24, 18 and 14 pixels &mdash; the three sizes the \
             application draws. Judge them here, at these sizes, never enlarged.</p>\n\
             <div class=\"grid\">\n{cells}</div>\n"
        );
        let sheet_path = root.join("docs/icons.html");
        std::fs::write(&sheet_path, sheet).expect("writing the contact sheet");

        println!("wrote {} icons to {}", IconName::ALL.len(), dir.display());
        println!("contact sheet: {}", sheet_path.display());
    }
}
