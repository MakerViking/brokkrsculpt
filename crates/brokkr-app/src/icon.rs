// SPDX-License-Identifier: AGPL-3.0-only

//! The window icon, drawn rather than shipped.
//!
//! The same mark as `assets/brand/brokkrsculpt-mark.svg`: one voxel with two
//! chipped off it, in the molten gradient SindriCAD's mark shares.
//!
//! # This does nothing on Wayland, and that is not a bug here
//!
//! `winit`'s Wayland backend implements `set_window_icon` as an empty function
//! (`platform_impl/linux/wayland/window/mod.rs`, `pub(crate) fn
//! set_window_icon(&self, _window_icon: Option<PlatformIcon>) {}`). **Wayland
//! has no protocol for a client to hand its compositor an icon** -- the
//! compositor matches the surface's `app_id` against an installed `.desktop`
//! file and takes the icon from the icon theme. So on the session this is
//! developed in, the window shows a generic fallback however correct the
//! pixels below are, and `packaging/brokkrsculpt.desktop` is what actually
//! fixes it. The `app_id` is already `brokkrsculpt`, which is the name that
//! file must have.
//!
//! It is still worth having: X11 and Windows both use it, and it costs one
//! rasterise at startup.
//!
//! Generated in code for the reason [`brokkr_gpu`]'s matcap is — it keeps the
//! repository free of a binary asset, and it means the icon cannot drift out of
//! step with the palette, because it is drawn from `theme`'s own colours. The
//! geometry below is the SVG's, in the SVG's coordinates, so the two can be
//! compared side by side.

use crate::theme;

/// Edge length of the generated icon.
///
/// 64 is what a Wayland compositor and a taskbar actually display; larger sizes
/// are downscaled by the shell anyway, and this is rasterised at startup.
const SIZE: u32 = 64;

/// The mark's design space, matching the SVG's `viewBox`.
pub(crate) const DESIGN: f32 = 256.0;

/// A filled convex polygon in design coordinates.
struct Face {
    points: &'static [(f32, f32)],
}

/// The cube's three visible faces, in the SVG's coordinates and translated the
/// same way (`translate(18,34) scale(1.02)` folded in below at sample time).
const TOP: Face = Face { points: &[(110.0, 44.0), (166.0, 76.0), (110.0, 108.0), (54.0, 76.0)] };
const LEFT: Face = Face { points: &[(54.0, 76.0), (110.0, 108.0), (110.0, 172.0), (54.0, 140.0)] };
const RIGHT: Face =
    Face { points: &[(110.0, 108.0), (166.0, 76.0), (166.0, 140.0), (110.0, 172.0)] };

/// The two chipped voxels, as axis-aligned squares. The SVG rotates them; at 64
/// pixels a rotation of sixteen degrees is under half a pixel of difference at
/// the corners, so they are left square and the SVG stays the reference.
const CHIPS: [(f32, f32, f32); 2] = [(171.0, 44.0, 16.0), (193.0, 20.0, 12.0)];

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

/// The edges that make the cube read as a solid: the outline, and the three
/// that meet at the near corner. Same segments the SVG strokes.
const EDGES: [((f32, f32), (f32, f32)); 9] = [
    ((110.0, 44.0), (166.0, 76.0)),
    ((166.0, 76.0), (166.0, 140.0)),
    ((166.0, 140.0), (110.0, 172.0)),
    ((110.0, 172.0), (54.0, 140.0)),
    ((54.0, 140.0), (54.0, 76.0)),
    ((54.0, 76.0), (110.0, 44.0)),
    ((110.0, 108.0), (54.0, 76.0)),
    ((110.0, 108.0), (166.0, 76.0)),
    ((110.0, 108.0), (110.0, 172.0)),
];

/// Undo the mark group's transform, so [`shade`] can be sampled in the design
/// space the SVG's own coordinates are written in.
pub(crate) fn to_design(fx: f32, fy: f32) -> (f32, f32) {
    ((fx - 18.0) / 1.02, (fy - 34.0) / 1.02)
}

/// An sRGB hex triple as fractions, so the constants below can be read against
/// the SVG they come from.
const fn hex(r: u8, g: u8, b: u8) -> [f32; 3] {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

/// Distance from a point to a line segment, for stroking the edges above.
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

/// The molten gradient at a height, matching the SVG's stops: cool at the top,
/// hot through the middle, cooling again to a deep red at the bottom.
///
/// **These are the SVG's sRGB hex values and the arithmetic stays in sRGB**,
/// because that is the space SVG composites its opacity overlays in. Treating
/// them as linear and encoding on the way out washes the whole mark to pastel,
/// which is what the first version of this did.
fn molten(t: f32) -> [f32; 3] {
    // Written as the SVG's hex, divided, so the two cannot drift apart:
    // #33517A, #FF7C36, #DE3A16.
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

/// Rasterise the mark to RGBA, ready for [`iced::window::icon::from_rgba`].
///
/// Supersampled four by four, because the cube is all diagonals and a hard
/// edged 64 pixel hexagon looks broken next to every other icon on the bar.
pub fn rgba() -> Vec<u8> {
    const SS: u32 = 4;
    let mut out = vec![0u8; (SIZE * SIZE * 4) as usize];

    for py in 0..SIZE {
        for px in 0..SIZE {
            let mut accum = [0.0f32; 4];
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = (px * SS + sx) as f32 / (SIZE * SS) as f32 * DESIGN;
                    let fy = (py * SS + sy) as f32 / (SIZE * SS) as f32 * DESIGN;
                    let (x, y) = to_design(fx, fy);

                    if let Some(sample) = shade(x, y) {
                        accum[0] += sample[0];
                        accum[1] += sample[1];
                        accum[2] += sample[2];
                        accum[3] += 1.0;
                    }
                }
            }
            let taken = (SS * SS) as f32;
            let alpha = accum[3] / taken;
            let index = ((py * SIZE + px) * 4) as usize;
            if alpha > 0.0 {
                // Premultiplied average of the covered samples only, so an edge
                // pixel takes the colour of what covers it rather than fading
                // toward black.
                out[index] = byte(accum[0] / accum[3]);
                out[index + 1] = byte(accum[1] / accum[3]);
                out[index + 2] = byte(accum[2] / accum[3]);
            }
            out[index + 3] = (alpha * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// The colour at a point of the mark, or `None` where the mark is not.
pub(crate) fn shade(x: f32, y: f32) -> Option<[f32; 3]> {
    // Chips first: they sit over everything, and they are the accent squares.
    for (cx, cy, size) in CHIPS {
        if x >= cx && x <= cx + size && y >= cy && y <= cy + size {
            let edge = 3.0;
            let border =
                x < cx + edge || x > cx + size - edge || y < cy + edge || y > cy + size - edge;
            return Some(if border { colour(theme::ACCENT) } else { [1.0, 1.0, 1.0] });
        }
    }

    // The cube. The gradient runs over the mark's own height, as the SVG's
    // objectBoundingBox gradient does.
    let t = ((y - 44.0) / (172.0 - 44.0)).clamp(0.0, 1.0);
    let base = molten(t);

    // Face colour first, then the edges are drawn over it -- but only where
    // the cube already is, so the stroke never widens the silhouette.
    let on_cube = inside(&TOP, x, y) || inside(&RIGHT, x, y) || inside(&LEFT, x, y);
    if on_cube {
        let nearest = EDGES.iter().map(|(a, b)| distance_to(x, y, *a, *b)).fold(f32::MAX, f32::min);
        if nearest < 2.0 {
            // #16283D at the SVG's 0.55, over whatever face this is.
            let face = face_colour(&base, x, y);
            const EDGE: [f32; 3] = hex(0x16, 0x28, 0x3D);
            return Some([
                face[0] + (EDGE[0] - face[0]) * 0.55,
                face[1] + (EDGE[1] - face[1]) * 0.55,
                face[2] + (EDGE[2] - face[2]) * 0.55,
            ]);
        }
        return Some(face_colour(&base, x, y));
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

fn colour(c: iced::Color) -> [f32; 3] {
    [c.r, c.g, c.b]
}

/// One sRGB channel as a byte. No transfer function: see `molten`.
fn byte(srgb: f32) -> u8 {
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The icon, or `None` if it could not be built -- which would be a bug here,
/// not a condition, so the caller simply goes without rather than failing to
/// start over a picture.
pub fn icon() -> Option<iced::window::Icon> {
    iced::window::icon::from_rgba(rgba(), SIZE, SIZE).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The icon has to be the right shape for `from_rgba`, which otherwise
    /// refuses it at startup and leaves the window with a default icon and no
    /// explanation.
    #[test]
    fn the_icon_is_the_size_it_claims() {
        let pixels = rgba();
        assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
        assert!(icon().is_some(), "from_rgba refused the buffer");
    }

    /// A mark that is entirely transparent, or entirely opaque, is not a mark.
    /// Both are what a broken polygon test produces, and neither would be
    /// noticed in a 64 pixel corner of a taskbar.
    #[test]
    fn the_mark_covers_some_of_the_icon_but_not_all_of_it() {
        let pixels = rgba();
        let opaque = pixels.chunks_exact(4).filter(|p| p[3] > 128).count();
        let total = (SIZE * SIZE) as usize;
        assert!(
            opaque > total / 8 && opaque < total * 3 / 4,
            "the cube covers {opaque} of {total} pixels, which is not a cube"
        );
    }

    #[test]
    #[ignore = "writes a preview to /tmp"]
    fn dump_preview() {
        let px = rgba();
        let mut pam =
            b"P7\nWIDTH 64\nHEIGHT 64\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n".to_vec();
        pam.extend_from_slice(&px);
        std::fs::write("/tmp/icon.pam", pam).unwrap();
    }

    /// The gradient must actually run: a flat fill means `molten` collapsed.
    #[test]
    fn the_top_of_the_cube_is_cooler_than_the_bottom() {
        let top = molten(0.05);
        let bottom = molten(0.95);
        assert!(top[2] > bottom[2], "the top should carry the cool blue");
        assert!(bottom[0] > bottom[2], "the bottom should be hot");
    }
}
