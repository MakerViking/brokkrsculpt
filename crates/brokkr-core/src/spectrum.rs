// SPDX-License-Identifier: AGPL-3.0-only

//! Full Spectrum: more colours than the printer has tool heads.
//!
//! Two filaments printed in alternating layers read, from arm's length, as a
//! third colour. RatDoux's Full Spectrum made that a slicer feature, and
//! Snapmaker OrcaSlicer ships it as *mixed filaments*: a U1 with four heads
//! prints every pair of them at 1:1, 2:1 and 1:2 -- eighteen mixes, twenty-two
//! colours in all.
//!
//! # A mix is a slot number like any other
//!
//! Slots `1..=physical` are tool heads; a mix is numbered after them, so on a
//! U1 slots 5 to 22 are the eighteen mixes in [`full_spectrum`]'s order. That
//! is exactly how Snapmaker Orca numbers them -- physical count plus one
//! onward, in the order the rows appear in `mixed_filament_definitions` -- so
//! the byte a voxel stores, the `paint_color` a triangle carries and the
//! filament the slicer prints are the same number with no table between them.
//!
//! **The order is therefore a contract, not a presentation choice.** A sculpt
//! painted in slot 9 means "1 and 3, two layers of 1 to one of 3" for as long
//! as this list is generated the same way; reordering it would repaint every
//! saved document. The test `the_order_is_pinned` exists to make that loud.
//! Like a head's slot, a mix's means what the machine in front of you makes
//! of it: on a three-head printer slot 9 is a different mix, exactly as it is
//! in Snapmaker Orca's own numbering.
//!
//! # What only Snapmaker Orca understands
//!
//! `mixed_filament_definitions` is Snapmaker's key (first in its v2.3.3).
//! Stock OrcaSlicer 2.5 implements colour mixing with a different, per-slot
//! scheme and drops these rows; stock 2.4 reads only one `paint_color` chunk,
//! so a slot above 17 is misread there. A plain four-colour export is
//! unaffected by any of this: the definitions are written only when a mix is
//! actually painted.

pub mod pigment;
mod pigment_table;

/// The layer ratios each pair is offered at, A layers to B layers.
///
/// Snapmaker's own auto-generation offers 1:1 only, which is ten colours from
/// four heads. 2:1 and 1:2 are what make the spread worth painting with, and
/// a percentage of 33 or 67 is what its own Ratio editor writes for them.
pub const RATIOS: [(u8, u8); 3] = [(1, 1), (2, 1), (1, 2)];

/// One mixed filament: two tool heads alternated by layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mix {
    /// The first tool head, 1-based.
    pub a: u8,
    /// The second tool head, 1-based.
    pub b: u8,
    /// Consecutive layers of `a` per cycle.
    pub layers_a: u8,
    /// Consecutive layers of `b` per cycle.
    pub layers_b: u8,
}

impl Mix {
    /// The share of `b`, as the whole percentage the slicer stores.
    ///
    /// Snapmaker Orca turns this back into layers by rounding the major share
    /// over the minor (`effective_pair_preview_ratios`): 33 becomes 2:1, 67
    /// becomes 1:2, 50 becomes 1:1. Round to nearest here so that inversion
    /// lands on the ratio this mix was made from.
    pub fn b_percent(&self) -> u8 {
        let total = u32::from(self.layers_a) + u32::from(self.layers_b);
        ((u32::from(self.layers_b) * 100 + total / 2) / total) as u8
    }

    /// What this mix looks like, given the colours loaded in the tool heads.
    ///
    /// The slicer's own swatch: with its default settings it counts the layers
    /// of one cycle and mixes `b` in at `layers_b / (layers_a + layers_b)`,
    /// computed in `f32` exactly as here. `None` when either head is not in
    /// `physical`.
    pub fn colour(&self, physical: &[[u8; 3]]) -> Option<[u8; 3]> {
        let a = *physical.get(usize::from(self.a).checked_sub(1)?)?;
        let b = *physical.get(usize::from(self.b).checked_sub(1)?)?;
        let t = f32::from(self.layers_b) / (f32::from(self.layers_a) + f32::from(self.layers_b));
        Some(pigment::lerp(a, b, t))
    }
}

/// Every mix of `physical` tool heads, in slot order.
///
/// Pairs in ascending order, `(1,2), (1,3) .. (3,4)`, each at every one of
/// [`RATIOS`]. Fewer than two heads mix nothing.
pub fn full_spectrum(physical: usize) -> Vec<Mix> {
    let heads = u8::try_from(physical).unwrap_or(u8::MAX);
    let mut mixes = Vec::new();
    for a in 1..=heads {
        for b in a.saturating_add(1)..=heads {
            for (layers_a, layers_b) in RATIOS {
                mixes.push(Mix { a, b, layers_a, layers_b });
            }
        }
    }
    mixes
}

/// The value of Snapmaker Orca's `mixed_filament_definitions` for `mixes`.
///
/// One row per mix, `;`-separated, in the field order of
/// `MixedFilamentManager::serialize_custom_entries`:
///
/// ```text
/// a,b,enabled,custom,mix_b_percent,pointillism,g<ids>,w<weights>,m<mode>,
/// z<sublayers>,xa<offset>,xb<offset>,d<deleted>,o<origin_auto>,u<id>,cm<ui>
/// ```
///
/// **Every row is written `custom` and not `origin_auto`.** On load the
/// slicer drops any auto row it would not have generated itself, and
/// auto-generation is off by default and always off on its command line -- so
/// rows marked auto would vanish, renumbering every mix after them. Mode 2 is
/// `Simple`, plain layer alternation; `cm0` is its Ratio editor, which is the
/// one that shows a percentage. Stable ids are 1-based row positions.
pub fn definitions(mixes: &[Mix]) -> String {
    let rows: Vec<String> = mixes
        .iter()
        .enumerate()
        .map(|(index, mix)| {
            format!(
                "{},{},1,1,{},0,g,w,m2,z0,xa0,xb0,d0,o0,u{},cm0",
                mix.a,
                mix.b,
                mix.b_percent(),
                index + 1
            )
        })
        .collect();
    rows.join(";")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_heads_make_eighteen_mixes_and_twenty_two_colours() {
        assert_eq!(full_spectrum(4).len(), 18);
        assert_eq!(4 + full_spectrum(4).len(), 22);
        assert!(full_spectrum(1).is_empty());
        assert!(full_spectrum(0).is_empty());
        assert_eq!(full_spectrum(2).len(), 3);
    }

    #[test]
    fn the_order_is_pinned() {
        // Slot numbers are stored in documents. If this fails, every sculpt
        // painted with a mix now prints in a different one.
        let slots: Vec<(usize, Mix)> =
            full_spectrum(4).into_iter().enumerate().map(|(i, m)| (i + 5, m)).collect();
        assert_eq!(slots[0], (5, Mix { a: 1, b: 2, layers_a: 1, layers_b: 1 }));
        assert_eq!(slots[1], (6, Mix { a: 1, b: 2, layers_a: 2, layers_b: 1 }));
        assert_eq!(slots[2], (7, Mix { a: 1, b: 2, layers_a: 1, layers_b: 2 }));
        assert_eq!(slots[3], (8, Mix { a: 1, b: 3, layers_a: 1, layers_b: 1 }));
        assert_eq!(slots[17], (22, Mix { a: 3, b: 4, layers_a: 1, layers_b: 2 }));
    }

    #[test]
    fn the_percentage_inverts_to_the_ratio_the_way_the_slicer_does_it() {
        // A port of `effective_pair_preview_ratios`, kept as small as it can
        // be: the major share over the minor, rounded, then reduced.
        fn slicer_ratio(percent_b: u8) -> (u32, u32) {
            let b = u32::from(percent_b);
            let a = 100 - b;
            let (major, minor) = if b >= a { (b, a) } else { (a, b) };
            let layers = ((f64::from(major) / f64::from(minor.max(1))).round() as u32).max(1);
            if b >= a { (1, layers) } else { (layers, 1) }
        }
        for (layers_a, layers_b) in RATIOS {
            let mix = Mix { a: 1, b: 2, layers_a, layers_b };
            assert_eq!(slicer_ratio(mix.b_percent()), (u32::from(layers_a), u32::from(layers_b)));
        }
        let percents: Vec<u8> = RATIOS
            .iter()
            .map(|&(a, b)| Mix { a: 1, b: 2, layers_a: a, layers_b: b }.b_percent())
            .collect();
        assert_eq!(percents, [50, 33, 67]);
    }

    #[test]
    fn rows_are_in_the_slicers_own_field_order() {
        // Shaped like the row in Snapmaker's `test_mixed_filament.cpp`:
        // "1,4,1,1,63,0,g,w,m2,z0,xa0,xb0,d0,o0,u1,cm0".
        let mixes = [
            Mix { a: 1, b: 4, layers_a: 1, layers_b: 1 },
            Mix { a: 3, b: 4, layers_a: 2, layers_b: 1 },
        ];
        assert_eq!(
            definitions(&mixes),
            "1,4,1,1,50,0,g,w,m2,z0,xa0,xb0,d0,o0,u1,cm0;3,4,1,1,33,0,g,w,m2,z0,xa0,xb0,d0,o0,u2,cm0"
        );
        assert_eq!(definitions(&[]), "");
    }

    #[test]
    fn the_swatch_is_the_pigment_mix_at_the_layer_share() {
        let heads = [[0, 33, 133], [252, 211, 0]];
        let even = Mix { a: 1, b: 2, layers_a: 1, layers_b: 1 };
        assert_eq!(even.colour(&heads), Some([47, 141, 56]));
        let two_to_one = Mix { a: 1, b: 2, layers_a: 2, layers_b: 1 };
        assert_eq!(two_to_one.colour(&heads), Some(pigment::lerp(heads[0], heads[1], 1.0 / 3.0)));
        assert_eq!(Mix { a: 1, b: 3, layers_a: 1, layers_b: 1 }.colour(&heads), None);
        assert_eq!(Mix { a: 0, b: 2, layers_a: 1, layers_b: 1 }.colour(&heads), None);
    }
}
