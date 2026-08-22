// SPDX-License-Identifier: AGPL-3.0-only

//! The logo mark, as a widget.
//!
//! # Why this is drawn one rectangle at a time
//!
//! iced can draw an SVG or an image, and both were measured before this was
//! written: the `canvas` feature costs **6 crates** (the whole `lyon`
//! tessellator) and the `image` feature costs **71** (the `image` crate with
//! every codec it ships, rav1e and exr included). This workspace refuses
//! dependencies that cost one or two, so a logo cannot cost six.
//!
//! What `advanced` already provides is [`Renderer::fill_quad`], which draws an
//! axis-aligned rectangle. So the mark is rasterised into horizontal runs of
//! constant colour and each run is one quad — the same
//! [`crate::icon::shade`] the window icon uses, sampled at widget scale, so
//! the header mark and the window icon and `assets/brand/brokkrsculpt-mark.svg`
//! are all the same geometry expressed three ways.
//!
//! At the 18 pixel size the header uses this comes to a few dozen quads, drawn
//! once per frame among the hundreds the panel already draws. If it ever needs
//! to be larger than an icon, this is the wrong technique and `canvas` becomes
//! worth its six crates.

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::{self, Widget};
use iced::{Color, Element, Length, Rectangle, Size};

use crate::icon;

/// The mark, sized to a square of `side` logical pixels.
pub struct Mark {
    side: f32,
}

/// Vertical resolution the mark is sampled at.
///
/// Independent of the widget's pixel size on purpose: it is the number of
/// horizontal bands the cube is cut into, and 44 is where the diagonals stop
/// looking like stairs at header size. Higher costs quads for nothing.
const BANDS: usize = 44;

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
            // Sample down the middle of the band, so a run's colour is the
            // colour of the material it covers rather than of its edge.
            let fy = (row as f32 + 0.5) / BANDS as f32 * icon::DESIGN;

            // Walk the row and emit one quad per run of equal colour. Equal
            // to a byte, because the gradient changes continuously and
            // comparing floats would emit a quad per sample.
            let mut run: Option<(usize, [u8; 3])> = None;
            for column in 0..=BANDS {
                let colour = if column == BANDS {
                    None
                } else {
                    let fx = (column as f32 + 0.5) / BANDS as f32 * icon::DESIGN;
                    let (x, y) = icon::to_design(fx, fy);
                    icon::shade(x, y).map(quantise)
                };

                match (&run, colour) {
                    (Some((start, held)), Some(found)) if *held == found => {}
                    (Some((start, held)), _) => {
                        let start = *start;
                        let held = *held;
                        flush(renderer, bounds, band, row, start, column, held);
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
fn flush<Renderer: renderer::Renderer>(
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
