// SPDX-License-Identifier: AGPL-3.0-only

//! What two filaments look like printed in alternating layers.
//!
//! A port of `filament_mixer::lerp` from Snapmaker OrcaSlicer's
//! `filament_mixer_model.h` (MIT, Justin Hayes): a degree-4 polynomial in the
//! two colours and the ratio, fitted to approximate Mixbox's pigment mixing.
//! Blue and yellow come out green, which is what a layer stack of them looks
//! like from arm's length, where an average in RGB or Oklab gives grey.
//!
//! **Ported rather than approximated because the number has to agree with
//! the slicer**, not merely look plausible. Snapmaker Orca draws a mixed
//! filament's swatch with exactly this function, so a swatch here computed any
//! other way would show the user one colour while painting and another in the
//! slicer they print from. The tests pin it against output of the C++ compiled
//! from the same header.

use super::pigment_table::{COEF, INTERCEPT, POWERS};

/// The colour of `a` and `b` mixed, `t` of the way from `a` to `b`.
///
/// `t` is an `f32` because the C++ takes a `float`, and widening a different
/// value to `f64` first would move the last bit of some results.
pub fn lerp(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    if t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }

    let inputs = [a[0], a[1], a[2], b[0], b[1], b[2]]
        .map(f64::from)
        .into_iter()
        .chain([f64::from(t)])
        .collect::<Vec<_>>();

    let mut sum = INTERCEPT;
    for (powers, weights) in POWERS.iter().zip(COEF.iter()) {
        // Same evaluation order as the C++: a product of repeated
        // multiplications, input by input. `powi` would round differently.
        let mut feature = 1.0;
        for (&input, &exponent) in inputs.iter().zip(powers.iter()) {
            let mut power = 1.0;
            for _ in 0..exponent {
                power *= input;
            }
            feature *= power;
        }
        for (channel, weight) in sum.iter_mut().zip(weights) {
            *channel += feature * weight;
        }
    }

    // Truncate, then clamp: `static_cast<int>` rounds toward zero, which is
    // what the fitting script's `astype(int)` did too.
    sum.map(|channel| (channel as i64).clamp(0, 255) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inputs and outputs of the C++ `filament_mixer::lerp`, compiled with
    /// `g++ -O0` from Snapmaker/OrcaSlicer `229efa64b`'s header and run on
    /// these inputs. Covers the U1's default slot colours, the extremes, and
    /// the three ratios Full Spectrum uses.
    ///
    /// Printed with `%.9g`, which round-trips an `f32` but is longer than the
    /// shortest form clippy would like; the digits are the C++'s, so they stay.
    #[allow(clippy::excessive_precision)]
    const FROM_THE_CPP: &[Case] = include!("pigment_golden.in");

    /// Colour A, colour B, the ratio, and what the C++ made of them.
    type Case = ([u8; 3], [u8; 3], f32, [u8; 3]);

    #[test]
    fn blue_and_yellow_make_green_as_the_header_documents() {
        // The example in the header's own usage comment.
        assert_eq!(lerp([0, 33, 133], [252, 211, 0], 0.5), [47, 141, 56]);
    }

    #[test]
    fn every_result_matches_the_cpp_exactly() {
        for &(a, b, t, expected) in FROM_THE_CPP {
            assert_eq!(lerp(a, b, t), expected, "lerp({a:?}, {b:?}, {t})");
        }
    }

    #[test]
    fn the_ends_are_the_inputs_untouched() {
        let a = [12, 200, 7];
        let b = [250, 3, 99];
        assert_eq!(lerp(a, b, 0.0), a);
        assert_eq!(lerp(a, b, 1.0), b);
    }
}
