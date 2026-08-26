// SPDX-License-Identifier: AGPL-3.0-only

//! A procedurally generated clay matcap.
//!
//! Matcap shading looks up a colour by the view space normal, so all the
//! lighting is baked into one small image. That is the whole shading model for
//! M0: no lights, no shadows, no PBR.
//!
//! Generating the image in code rather than shipping one keeps the repository
//! free of a binary asset whose licence would have to be tracked, and it makes
//! the look a set of numbers that can be tuned in one place.

use glam::Vec3;

/// Edge length of the generated matcap. 256 is plenty: the image is sampled
/// smoothly across a hemisphere of normals, so detail beyond this is invisible.
pub const MATCAP_SIZE: u32 = 256;

/// Base clay colour, a warm neutral grey that sits calmly against the dark UI.
///
/// Kept well below white on purpose. A bright surface flattens into glossy
/// plastic and hides exactly the shallow form a sculptor needs to judge, so the
/// lit side lands near 0.4 in linear light rather than near 1.
const BASE: Vec3 = Vec3::new(0.40, 0.382, 0.362);
/// Key light direction in view space, up and to the left, slightly toward the
/// viewer.
const KEY: Vec3 = Vec3::new(-0.35, 0.55, 0.76);
/// Fill light from below right, cool, so cavities do not read as flat black.
const FILL: Vec3 = Vec3::new(0.58, -0.42, 0.45);
const FILL_COLOUR: Vec3 = Vec3::new(0.09, 0.11, 0.15);
/// Rim colour, warm to echo the amber accent without shouting.
const RIM_COLOUR: Vec3 = Vec3::new(0.42, 0.28, 0.18);
/// Ambient floor, so nothing is pure black.
const AMBIENT: f32 = 0.16;

/// Generate the matcap as sRGB encoded RGBA bytes, row 0 being the top of the
/// sphere.
///
/// The bytes are sRGB encoded because the texture is uploaded as
/// `Rgba8UnormSrgb`, so the sampler hands the shader linear values back.
pub fn clay() -> Vec<u8> {
    let size = MATCAP_SIZE as usize;
    let mut pixels = vec![0u8; size * size * 4];

    let key = KEY.normalize();
    let fill = FILL.normalize();
    let view = Vec3::Z;
    let halfway = (key + view).normalize();

    for y in 0..size {
        for x in 0..size {
            // Pixel centre mapped to [-1, 1]. Row 0 is the top of the sphere,
            // which is why the y term is negated: the shader samples with the
            // texture v axis flipped to match.
            let nx = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let ny = -((y as f32 + 0.5) / size as f32 * 2.0 - 1.0);

            let radius_squared = nx * nx + ny * ny;
            let normal = if radius_squared <= 1.0 {
                Vec3::new(nx, ny, (1.0 - radius_squared).sqrt())
            } else {
                // Outside the disc there is no normal. Extend the silhouette
                // outward so bilinear filtering near the rim has something
                // sensible to blend with instead of black.
                Vec3::new(nx, ny, 0.0).normalize()
            };

            // Wrapped diffuse: clay is soft and the terminator should not be a
            // hard line. Wrapping too far washes the form out, so this is a
            // gentle wrap rather than a full half Lambert.
            let lambert = normal.dot(key);
            let diffuse = ((lambert + 0.20) / 1.20).clamp(0.0, 1.0);
            let fill_amount = normal.dot(fill).max(0.0).powf(1.5);
            let rim = (1.0 - normal.z.max(0.0)).powf(3.5);
            let specular = normal.dot(halfway).max(0.0).powf(48.0);

            let colour = BASE * (AMBIENT + 0.90 * diffuse)
                + FILL_COLOUR * fill_amount
                + RIM_COLOUR * rim * 0.45
                + Vec3::splat(specular * 0.16);

            let index = (y * size + x) * 4;
            pixels[index] = encode_srgb(colour.x);
            pixels[index + 1] = encode_srgb(colour.y);
            pixels[index + 2] = encode_srgb(colour.z);
            pixels[index + 3] = 255;
        }
    }

    pixels
}

/// Linear value to an sRGB encoded byte.
fn encode_srgb(linear: f32) -> u8 {
    let linear = linear.clamp(0.0, 1.0);
    let encoded =
        if linear <= 0.003_130_8 { linear * 12.92 } else { 1.055 * linear.powf(1.0 / 2.4) - 0.055 };
    (encoded * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_image_is_the_expected_size_and_fully_opaque() {
        let pixels = clay();
        assert_eq!(pixels.len(), (MATCAP_SIZE * MATCAP_SIZE * 4) as usize);
        assert!(pixels.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn the_lit_side_is_brighter_than_the_shadowed_side() {
        let pixels = clay();
        let size = MATCAP_SIZE as usize;
        let luma = |x: usize, y: usize| {
            let i = (y * size + x) * 4;
            pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32
        };
        // The key light comes from up and to the left, so the upper left of the
        // disc must be brighter than the lower right.
        let lit = luma(size / 4, size / 4);
        let shadowed = luma(size * 3 / 4, size * 3 / 4);
        assert!(lit > shadowed, "upper left {lit} should outshine lower right {shadowed}");
    }

    #[test]
    fn srgb_encoding_hits_the_known_end_points() {
        assert_eq!(encode_srgb(0.0), 0);
        assert_eq!(encode_srgb(1.0), 255);
        // Mid grey in linear light is a good deal brighter once encoded.
        assert!(encode_srgb(0.5) > 180);
    }
}
