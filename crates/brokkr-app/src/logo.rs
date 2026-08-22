// SPDX-License-Identifier: AGPL-3.0-only

//! The logo mark: one voxel, with two chipped off it.
//!
//! The same geometry as `assets/brand/brokkrsculpt-mark.svg`, in the SVG's own
//! coordinates so the two can be read side by side, drawn with the molten
//! gradient SindriCAD's mark shares.
//!
//! # Why this is drawn one rectangle at a time
//!
//! iced can draw an SVG or an image, and both were measured before this was
//! written: the `canvas` feature costs **6 crates** (the whole `lyon`
//! tessellator) and the `image` feature costs **71** (the `image` crate with
//! every codec it ships, `rav1e` and `exr` included). This workspace refuses
//! dependencies that cost one or two, so a logo cannot cost six.
//!
//! What the `advanced` feature already provides is `fill_quad`, which draws an
//! axis-aligned rectangle. So the mark is rasterised into horizontal runs of
//! constant colour and each run is one quad. At the eighteen pixels the header
//! uses this is a few dozen quads among the hundreds the panel already draws.
//! If the mark ever needs to be larger than an icon, this is the wrong
//! technique and `canvas` becomes worth its six crates.
//!
//! # There is no window icon
//!
//! There was one, briefly, generated from this same geometry. It was removed
//! because it could not work: `winit`'s Wayland backend implements
//! `set_window_icon` as an empty function, since Wayland has no protocol for a
//! client to hand its compositor an icon — the compositor matches the
//! surface's `app_id` against an installed `.desktop` file instead. SindriCAD
//! does not carry one either. If a window icon is ever wanted, it comes from
//! `packaging/brokkrsculpt.desktop` and the icon theme, not from here.

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{self, Widget};
use iced::{Color, Element, Length, Rectangle, Size};

/// The square of design space the widget maps onto, matching the mark SVG's
/// coordinates so the two can be read side by side.
///
/// Not the SVG's whole 256 viewBox: the artwork occupies x 40..211 and
/// y 16..220, so the widget is mapped onto a square just covering that. A
/// widget mapped onto the full viewBox would draw the mark two-thirds size
/// with a margin of nothing.
const DESIGN_MIN: (f32, f32) = (23.5, 16.0);
const DESIGN_SPAN: f32 = 204.0;

/// A filled convex polygon in design coordinates.
struct Face {
    points: &'static [(f32, f32)],
}

// --- the block ---------------------------------------------------------------

const TOP: Face = Face { points: &[(96.0, 96.0), (152.0, 128.0), (96.0, 160.0), (40.0, 128.0)] };
const LEFT: Face = Face { points: &[(40.0, 128.0), (96.0, 160.0), (96.0, 220.0), (40.0, 188.0)] };
const RIGHT: Face =
    Face { points: &[(96.0, 160.0), (152.0, 128.0), (152.0, 188.0), (96.0, 220.0)] };

/// The edges that make the block read as a solid: its outline, and the three
/// that meet at the near corner.
const EDGES: [((f32, f32), (f32, f32)); 9] = [
    ((96.0, 96.0), (152.0, 128.0)),
    ((152.0, 128.0), (152.0, 188.0)),
    ((152.0, 188.0), (96.0, 220.0)),
    ((96.0, 220.0), (40.0, 188.0)),
    ((40.0, 188.0), (40.0, 128.0)),
    ((40.0, 128.0), (96.0, 96.0)),
    ((96.0, 160.0), (40.0, 128.0)),
    ((96.0, 160.0), (152.0, 128.0)),
    ((96.0, 160.0), (96.0, 220.0)),
];

// --- the chisel --------------------------------------------------------------
//
// Drawn rotated in the SVG; the rotation is baked into these absolute points so
// the two representations share one set of numbers and cannot drift.

const HANDLE: Face = Face { points: &[(70.9, 38.5), (99.2, 16.4), (116.5, 38.5), (88.1, 60.6)] };
const BLADE: Face = Face { points: &[(89.6, 54.4), (110.1, 38.4), (155.6, 96.7), (135.1, 112.7)] };
const SHINE: Face = Face { points: &[(89.6, 54.4), (97.4, 48.2), (143.0, 106.5), (135.1, 112.7)] };
const TIP: Face = Face { points: &[(135.1, 112.7), (155.6, 96.7), (160.1, 123.6)] };

/// The two voxels the chisel has taken off, as axis-aligned squares. The SVG
/// rotates them; at icon size that rotation is under half a pixel.
const CHIPS: [(f32, f32, f32); 2] = [(176.0, 150.0, 15.0), (200.0, 182.0, 11.0)];

/// An sRGB hex triple as fractions, so the constants can be read against the
/// SVG they come from.
const fn hex(r: u8, g: u8, b: u8) -> [f32; 3] {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

/// Where a point of the widget's unit square lands in design space.
fn to_design(u: f32, v: f32) -> (f32, f32) {
    (DESIGN_MIN.0 + u * DESIGN_SPAN, DESIGN_MIN.1 + v * DESIGN_SPAN)
}

/// Whether a point is inside a convex polygon wound consistently.
fn inside(face: &Face, x: f32, y: f32) -> bool {
    let points = face.points;
    let mut sign = 0.0f32;
    for index in 0..points.len() {
        let (ax, ay) = points[index];
        let (bx, by) = points[(index + 1) % points.len()];
        let cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        if cross != 0.0 {
            if sign != 0.0 && cross.signum() != sign {
                return false;
            }
            sign = cross.signum();
        }
    }
    true
}

/// Distance from a point to a line segment, for stroking the edges.
fn distance_to(x: f32, y: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let length = dx * dx + dy * dy;
    let t = if length > 0.0 {
        (((x - a.0) * dx + (y - a.1) * dy) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (px, py) = (a.0 + dx * t, a.1 + dy * t);
    ((x - px).powi(2) + (y - py).powi(2)).sqrt()
}

/// The molten gradient down the block: cool at the top, hot through the middle,
/// cooling again to a deep red at the bottom.
///
/// **These are the SVG's sRGB hex values and the arithmetic stays in sRGB**,
/// because that is the space SVG composites its opacity overlays in. Treating
/// them as linear and encoding on the way out washes the whole mark to pastel,
/// which is what the first version of this did.
fn molten(t: f32) -> [f32; 3] {
    const COOL: [f32; 3] = hex(0x33, 0x51, 0x7A);
    const HOT: [f32; 3] = hex(0xFF, 0x7C, 0x36);
    const DEEP: [f32; 3] = hex(0xDE, 0x3A, 0x16);
    let (from, to, k) =
        if t < 0.30 { (COOL, HOT, t / 0.30) } else { (HOT, DEEP, (t - 0.30) / 0.70) };
    [
        from[0] + (to[0] - from[0]) * k,
        from[1] + (to[1] - from[1]) * k,
        from[2] + (to[2] - from[2]) * k,
    ]
}

/// The colour at a point of the mark, or `None` where the mark is not.
///
/// First match wins, so the order here is the SVG's painting order reversed:
/// what the SVG draws last is tested first.
fn shade(x: f32, y: f32) -> Option<[f32; 3]> {
    for (cx, cy, size) in CHIPS {
        if x >= cx && x <= cx + size && y >= cy && y <= cy + size {
            let edge = 3.0;
            let border =
                x < cx + edge || x > cx + size - edge || y < cy + edge || y > cy + size - edge;
            return Some(if border { hex(0xFF, 0x5A, 0x2C) } else { [1.0, 1.0, 1.0] });
        }
    }

    // The chisel sits over the block.
    if inside(&TIP, x, y) {
        return Some(hex(0x84, 0x96, 0xAA));
    }
    if inside(&SHINE, x, y) {
        return Some(hex(0xE8, 0xEE, 0xF6));
    }
    if inside(&BLADE, x, y) {
        return Some(hex(0xB9, 0xC4, 0xD2));
    }
    if inside(&HANDLE, x, y) {
        return Some(hex(0x6B, 0x4F, 0x35));
    }

    let t = ((y - 96.0) / (220.0 - 96.0)).clamp(0.0, 1.0);
    let base = molten(t);

    // Face colour first, then the edges over it -- but only where the block
    // already is, so the stroke never widens the silhouette.
    if inside(&TOP, x, y) || inside(&RIGHT, x, y) || inside(&LEFT, x, y) {
        let nearest = EDGES.iter().map(|(a, b)| distance_to(x, y, *a, *b)).fold(f32::MAX, f32::min);
        let face = face_colour(&base, x, y);
        if nearest < 2.0 {
            const EDGE: [f32; 3] = hex(0x16, 0x28, 0x3D);
            return Some([
                face[0] + (EDGE[0] - face[0]) * 0.55,
                face[1] + (EDGE[1] - face[1]) * 0.55,
                face[2] + (EDGE[2] - face[2]) * 0.55,
            ]);
        }
        return Some(face);
    }
    None
}

/// Which of the three faces a point is on, shaded as the SVG shades it.
fn face_colour(base: &[f32; 3], x: f32, y: f32) -> [f32; 3] {
    if inside(&TOP, x, y) {
        // Lit, the same 0.14 white the SVG lays over the top face.
        return [
            base[0] + (1.0 - base[0]) * 0.14,
            base[1] + (1.0 - base[1]) * 0.14,
            base[2] + (1.0 - base[2]) * 0.14,
        ];
    }
    if inside(&RIGHT, x, y) {
        // In shadow, the SVG's 0.22 black.
        return [base[0] * 0.78, base[1] * 0.78, base[2] * 0.78];
    }
    *base
}

/// Vertical resolution the mark is sampled at.
///
/// Independent of the widget's pixel size on purpose: it is the number of
/// horizontal bands the cube is cut into, and 44 is where the diagonals stop
/// looking like stairs at header size. Higher costs quads for nothing.
const BANDS: usize = 44;

/// The mark, sized to a square of `side` logical pixels.
pub struct Mark {
    side: f32,
}

pub fn mark(side: f32) -> Mark {
    Mark { side }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer> for Mark
where
    Renderer: renderer::Renderer,
{
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fixed(self.side), Length::Fixed(self.side))
    }

    fn layout(
        &mut self,
        _tree: &mut widget::Tree,
        _renderer: &Renderer,
        _limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(Size::new(self.side, self.side))
    }

    fn draw(
        &self,
        _tree: &widget::Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: iced::advanced::mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let band = self.side / BANDS as f32;

        for row in 0..BANDS {
            // Sampled down the middle of the band, so a run takes the colour
            // of the material it covers rather than of its edge.
            let v = (row as f32 + 0.5) / BANDS as f32;

            // One quad per run of equal colour. Equal to a byte, because the
            // gradient changes continuously and comparing floats would emit a
            // quad per sample.
            let mut run: Option<(usize, [u8; 3])> = None;
            for column in 0..=BANDS {
                let colour = (column < BANDS).then(|| {
                    let u = (column as f32 + 0.5) / BANDS as f32;
                    let (x, y) = to_design(u, v);
                    shade(x, y).map(quantise)
                });
                let colour = colour.flatten();

                match (run, colour) {
                    (Some((_, held)), Some(found)) if held == found => {}
                    (Some((start, held)), _) => {
                        fill_run(renderer, bounds, band, row, start, column, held);
                        run = colour.map(|c| (column, c));
                    }
                    (None, Some(found)) => run = Some((column, found)),
                    (None, None) => {}
                }
            }
        }
    }
}

/// One run of equal colour, as a rectangle.
fn fill_run<Renderer: renderer::Renderer>(
    renderer: &mut Renderer,
    bounds: Rectangle,
    band: f32,
    row: usize,
    from: usize,
    to: usize,
    colour: [u8; 3],
) {
    let x = bounds.x + from as f32 * band;
    let y = bounds.y + row as f32 * band;
    // Overlapped by a hair, because adjacent quads that merely touch leave a
    // seam of background showing through at fractional scale factors.
    let width = (to - from) as f32 * band + 0.5;
    renderer.fill_quad(
        Quad { bounds: Rectangle { x, y, width, height: band + 0.5 }, ..Quad::default() },
        Color::from_rgb8(colour[0], colour[1], colour[2]),
    );
}

fn quantise(colour: [f32; 3]) -> [u8; 3] {
    [
        (colour[0].clamp(0.0, 1.0) * 255.0) as u8,
        (colour[1].clamp(0.0, 1.0) * 255.0) as u8,
        (colour[2].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

impl<'a, Message, Theme, Renderer> From<Mark> for Element<'a, Message, Theme, Renderer>
where
    Renderer: renderer::Renderer + 'a,
    Message: 'a,
    Theme: 'a,
{
    fn from(mark: Mark) -> Self {
        Element::new(mark)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mark that is entirely empty, or entirely filled, is not a mark. Both
    /// are what a broken polygon test produces, and neither would be noticed
    /// in an eighteen pixel corner of a header.
    #[test]
    fn the_cube_covers_some_of_the_square_but_not_all_of_it() {
        let mut covered = 0;
        let steps = 64;
        for row in 0..steps {
            for column in 0..steps {
                let u = (column as f32 + 0.5) / steps as f32;
                let v = (row as f32 + 0.5) / steps as f32;
                let (x, y) = to_design(u, v);
                if shade(x, y).is_some() {
                    covered += 1;
                }
            }
        }
        let total = steps * steps;
        assert!(
            covered > total / 8 && covered < total * 3 / 4,
            "the cube covers {covered} of {total} samples, which is not a cube"
        );
    }

    /// The gradient must actually run: a flat fill means `molten` collapsed.
    #[test]
    fn the_top_of_the_cube_is_cooler_than_the_bottom() {
        let top = molten(0.05);
        let bottom = molten(0.95);
        assert!(top[2] > bottom[2], "the top should carry the cool blue");
        assert!(bottom[0] > bottom[2], "the bottom should be hot");
    }

    /// The three faces must differ, or the cube reads as a flat hexagon. This
    /// is what the first version got wrong by leaving the edges out.
    #[test]
    fn the_three_faces_are_shaded_apart() {
        // A point well inside each face, away from the edge stroke and from
        // the chisel, which is painted over the block.
        let top = shade(70.0, 126.0).expect("top face");
        let left = shade(62.0, 170.0).expect("left face");
        let right = shade(130.0, 170.0).expect("right face");
        assert!(top[0] != left[0] && left[0] != right[0], "faces should not share a colour");
        assert!(right[0] < left[0], "the right face is the one in shadow");
    }

    /// The chisel must be ON TOP of the block, or the mark is just a cube
    /// again. This samples a point the blade covers and requires steel rather
    /// than molten orange.
    #[test]
    fn the_chisel_is_painted_over_the_block() {
        let blade = shade(122.6, 75.5).expect("the blade covers this point");
        assert!(blade[2] > blade[0], "steel is blue-ish; this came out warm, so the block won");
        let handle = shade(93.7, 38.5).expect("the handle covers this point");
        assert!(handle[0] > handle[2], "the handle is wood, warmer than it is blue");
    }
}
