// SPDX-License-Identifier: AGPL-3.0-only

//! The filament slots, and what is loaded in them.
//!
//! A sculpt stores a **slot number**, never a colour: slot 3 means "print this
//! with whatever is in tool head 3". Which filament that actually is belongs to
//! the machine, not to the model, so this lives in a config file beside
//! `printer.conf` rather than inside a `.brokkr` document. Opening someone
//! else's sculpt therefore shows it in *your* filament, which is the correct
//! answer -- the assignment travelled, the pigment did not.
//!
//! SindriCAD reached the same conclusion from the other end and wrote it down
//! in `nearestPaletteSlot`: "The palette is the U1's filament list -- four
//! physical slots -- not a display palette, so a slot means 'print this in
//! filament N'."
//!
//! # The file
//!
//! A flat `key = value` file, the shape `printer.conf`, `spacemouse.conf` and
//! `welcome.conf` already use, read with [`crate::paths::entries`]:
//!
//! ```text
//! slot1 = White, #E8E8E8, PLA
//! slot2 = Black, #202020, PETG
//! ```
//!
//! Name, colour, material -- and every field after the name is optional, so
//! `slot2 = Black` is a legal line. **A line this module cannot make sense of
//! leaves that slot at its default rather than failing the file**, the same
//! forgiveness `printer.rs` gives an unparseable port. The failure worth
//! avoiding is a hand-edited typo in slot 4 silently discarding slots 1 to 3.

use brokkr_core::export::threemf::{Filaments, MAX_SLOTS};

/// The file the palette lives in.
const FILE: &str = "filaments.conf";

/// What is loaded in one tool head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Slot {
    /// What to call it in the panel. Free text -- a filament's name is whatever
    /// the user calls it, or whatever the printer reports.
    pub name: String,
    /// `#RRGGBB`, upper case.
    pub colour: String,
    /// `PLA`, `PETG`, `ABS`, or whatever the printer says.
    pub material: String,
}

/// The filament slots this machine has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Palette {
    pub slots: Vec<Slot>,
}

impl Default for Palette {
    /// **Derived from [`Filaments::default`], not restated.**
    ///
    /// That table is baked into the byte-pinned `export-cube.3mf` golden. A
    /// second copy of it here is the shape that has already drifted once in
    /// this repository and gated a whole panel off; the names are generated
    /// rather than tabulated for the same reason.
    fn default() -> Self {
        let filaments = Filaments::default();
        let slots = filaments
            .colours
            .iter()
            .enumerate()
            .map(|(index, colour)| Slot {
                name: format!("Slot {}", index + 1),
                colour: normalise_colour(colour).unwrap_or_else(|| colour.to_ascii_uppercase()),
                material: filaments
                    .materials
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| "PLA".to_string()),
            })
            .collect();
        Self { slots }
    }
}

/// A field as it can safely be written to the config file and read back.
///
/// **The config format is one line per slot, comma separated**, so a value
/// containing a comma, an `=`, a `#` or a newline does not survive the round
/// trip -- and these values are not this application's text. They come off the
/// printer: a vendor of `"Polymaker, Inc."` makes a name whose comma silently
/// becomes a field boundary, so on the next launch the colour is read as the
/// material and the user's synced colour is gone. A newline is worse: it ends
/// the line and whatever follows is parsed as another `slotN = ...` entry.
///
/// Separators become spaces rather than being dropped, so a name stays legible
/// rather than turning into a run-on word.
fn clean_field(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|character| match character {
            ',' | '=' | '#' => ' ',
            control if control.is_control() => ' ',
            other => other,
        })
        .collect();
    // Collapse the runs those substitutions can leave behind.
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `#RRGGBB` in upper case, or `None` if it is not a colour.
///
/// **Upper case is not cosmetic.** The rest of this pipeline is upper case by
/// literal -- `PAINT_CODE`, `Filaments::default` -- and lower-case hex has been
/// reported as silently ignored by slicers in this lineage. Normalising on the
/// way in means a hand-edited `#e8e8e8` and a printer that answers in lower
/// case both land as the same bytes in the exported package.
///
/// A three-digit `#RGB` is refused rather than expanded: it is not a form
/// anything in this pipeline produces, and quietly doubling nibbles would make
/// a typo look like a deliberate colour.
fn normalise_colour(text: &str) -> Option<String> {
    let body = text.strip_prefix('#')?;
    if body.len() != 6 || !body.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("#{}", body.to_ascii_uppercase()))
}

impl Palette {
    /// The slots as the sculpt shader wants them: linear RGB by slot number.
    ///
    /// Index 0 is unpainted and never read. Every slot this machine does not
    /// have -- past the end of the list, or past what the shader can hold --
    /// is left at [`brokkr_gpu::UNKNOWN_FILAMENT`], so a sculpt painted for a
    /// machine with more heads shows where rather than showing clay.
    pub fn shader_palette(&self) -> [[f32; 4]; brokkr_gpu::PALETTE_SLOTS] {
        let mut palette = [brokkr_gpu::UNKNOWN_FILAMENT; brokkr_gpu::PALETTE_SLOTS];
        for (index, slot) in self.slots.iter().enumerate().take(brokkr_gpu::PALETTE_SLOTS - 1) {
            palette[index + 1] = slot.swatch().into_linear();
        }
        palette
    }

    /// The palette on this machine, or the defaults when there is no file.
    pub fn load() -> Self {
        Self::load_from(crate::paths::config_file(FILE).as_deref())
    }

    /// Split out so the tests can point it at a temp file, the way
    /// `printer::configured` and `welcome::read_from` are.
    pub fn load_from(path: Option<&std::path::Path>) -> Self {
        let Some(text) = path.and_then(|path| std::fs::read_to_string(path).ok()) else {
            return Self::default();
        };
        Self::parse(&text)
    }

    /// Read a palette out of the config text.
    ///
    /// Starts from the defaults and overwrites what the file names, so a file
    /// that mentions only `slot2` keeps the other three rather than producing a
    /// one-slot palette. A `slotN` beyond [`MAX_SLOTS`] is dropped: the writer
    /// cannot encode it, and silently keeping it would produce a palette whose
    /// tail can never reach a triangle.
    pub fn parse(text: &str) -> Self {
        let mut palette = Self::default();
        for (key, value) in crate::paths::entries(text) {
            let Some(index) = key
                .strip_prefix("slot")
                .and_then(|digits| digits.parse::<usize>().ok())
                .filter(|number| (1..=MAX_SLOTS).contains(number))
                .map(|number| number - 1)
            else {
                continue;
            };
            // A file may name a slot past the four defaults; grow to reach it,
            // filling the gap with defaults rather than with empty rows.
            while palette.slots.len() <= index {
                let position = palette.slots.len() + 1;
                palette.slots.push(Slot {
                    name: format!("Slot {position}"),
                    colour: "#FFFFFF".to_string(),
                    material: "PLA".to_string(),
                });
            }
            palette.slots[index].apply_line(value);
        }
        palette
    }

    /// The config text for this palette, ready to write back.
    pub fn to_config(&self) -> String {
        let mut text = String::from(
            "# BrokkrSculpt filament slots: name, colour, material.\n\
             # Written by \"Sync filaments from printer\", and safe to edit by hand.\n",
        );
        for (index, slot) in self.slots.iter().enumerate() {
            text.push_str(&format!(
                "slot{} = {}, {}, {}\n",
                index + 1,
                slot.name,
                slot.colour,
                slot.material
            ));
        }
        text
    }

    /// Write the palette to this machine's config file.
    ///
    /// **Inert under `cfg(test)`, deliberately**, the way the update journal's
    /// `record` is: an application test that sends `PaletteSynced` reaches
    /// this through the real `update`, and on 2026-09-05 one did and replaced
    /// the U1's synced `filaments.conf` on the developer's machine with the
    /// test's fixture. A test wanting the file goes through
    /// [`Palette::to_config`] and a path of its own.
    pub fn save(&self) -> Result<(), String> {
        if cfg!(test) {
            return Ok(());
        }
        let Some(path) = crate::paths::config_file(FILE) else {
            return Err("no config directory to write to".to_string());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|why| format!("could not create {}: {why}", parent.display()))?;
        }
        std::fs::write(&path, self.to_config())
            .map_err(|why| format!("could not write {}: {why}", path.display()))
    }

    /// What the 3MF writer needs.
    ///
    /// `base` is slot 1: it is the slot an *unpainted* triangle prints with,
    /// and until there is a paint tool that is every triangle.
    pub fn as_filaments(&self) -> Filaments {
        Filaments {
            colours: self.slots.iter().map(|slot| slot.colour.clone()).collect(),
            materials: self.slots.iter().map(|slot| slot.material.clone()).collect(),
            base: 1,
        }
    }

    /// Take the filament the printer reports into the palette.
    ///
    /// **An empty tool head leaves its slot alone.** The machine reports a slot
    /// with nothing loaded as present-but-blank, and overwriting a filament the
    /// user named and coloured with a blank row would lose work in exchange for
    /// no information. Returns which 1-based slots were left untouched, so the
    /// caller can say so rather than reporting a silent partial sync.
    ///
    /// **A head that says nothing counts as empty even when it claims to be
    /// present.** `filament_exist` is optional and an absent entry is read as
    /// loaded, so a machine that pads its columns with blanks and omits that
    /// array would otherwise replace a named slot with `Slot N` and an empty
    /// material -- and an empty material silently becomes PLA on the next
    /// launch, then goes out in every subsequent export.
    pub fn sync_from_printer(&mut self, reported: &[crate::printer::Filament]) -> Vec<usize> {
        let mut skipped = Vec::new();
        for (index, filament) in reported.iter().take(MAX_SLOTS).enumerate() {
            let says_nothing = filament.vendor.trim().is_empty()
                && filament.material.trim().is_empty()
                && filament.colour.is_empty();
            if !filament.present || says_nothing {
                skipped.push(index + 1);
                continue;
            }
            while self.slots.len() <= index {
                let position = self.slots.len() + 1;
                self.slots.push(Slot {
                    name: format!("Slot {position}"),
                    colour: "#FFFFFF".to_string(),
                    material: "PLA".to_string(),
                });
            }
            let slot = &mut self.slots[index];
            // Each field only when the machine actually named it, which is the
            // same rule `apply_line` applies to a hand-edited file.
            let name = clean_field(&filament.label(index + 1));
            if !name.is_empty() {
                slot.name = name;
            }
            let material = clean_field(&filament.material);
            if !material.is_empty() {
                slot.material = material;
            }
            if let Some(colour) = normalise_colour(&filament.colour) {
                slot.colour = colour;
            }
        }
        skipped
    }
}

impl Slot {
    /// Overwrite this slot from one `name, colour, material` line.
    ///
    /// Each field is taken only if it is there and makes sense, so a line with
    /// a mistyped colour still sets the name. An empty name is ignored rather
    /// than stored: a nameless row in the panel is indistinguishable from a
    /// rendering bug.
    fn apply_line(&mut self, value: &str) {
        let mut fields = value.split(',').map(str::trim);
        if let Some(name) = fields.next().map(clean_field).filter(|name| !name.is_empty()) {
            self.name = name;
        }
        if let Some(colour) = fields.next().and_then(normalise_colour) {
            self.colour = colour;
        }
        if let Some(material) =
            fields.next().map(clean_field).filter(|material| !material.is_empty())
        {
            self.material = material;
        }
    }

    /// The swatch colour as iced wants it.
    ///
    /// Falls back to mid grey rather than to black or to a panic: this runs in
    /// `view()`, and the colour has already been validated on the way in, so a
    /// failure here means a bug elsewhere and should be visible without taking
    /// the window down.
    pub fn swatch(&self) -> iced::Color {
        parse_rgb(&self.colour).unwrap_or(iced::Color::from_rgb(0.5, 0.5, 0.5))
    }
}

/// `#RRGGBB` to a colour, for the panel.
fn parse_rgb(text: &str) -> Option<iced::Color> {
    let body = text.strip_prefix('#')?;
    if body.len() != 6 {
        return None;
    }
    let channel = |at: usize| u8::from_str_radix(&body[at..at + 2], 16).ok();
    Some(iced::Color::from_rgb8(channel(0)?, channel(2)?, channel(4)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_palette_is_linear_by_slot_number_and_unknown_past_the_end() {
        let mut palette = Palette::default();
        palette.slots[0].colour = "#FF0000".to_string();
        let shader = palette.shader_palette();
        assert_eq!(shader[0], brokkr_gpu::UNKNOWN_FILAMENT, "slot 0 is unpainted, never a colour");
        assert_eq!(shader[1], [1.0, 0.0, 0.0, 1.0], "slot 1 should be the first filament");
        // Linear, not sRGB: mid grey encodes as #808080 and lands near 0.216.
        palette.slots[1].colour = "#808080".to_string();
        let grey = palette.shader_palette()[2][0];
        assert!(
            (grey - 0.216).abs() < 0.01,
            "slot colours must reach the shader linear, got {grey}"
        );
        // Past the machine's slots is the unknown-filament colour, not clay.
        assert_eq!(shader[palette.slots.len() + 1], brokkr_gpu::UNKNOWN_FILAMENT);
        // And a palette longer than the shader can hold does not panic.
        while palette.slots.len() < brokkr_gpu::PALETTE_SLOTS + 4 {
            palette.slots.push(palette.slots[0].clone());
        }
        let _ = palette.shader_palette();
    }

    #[test]
    fn the_default_palette_is_the_writers_own_slots() {
        let palette = Palette::default();
        let filaments = Filaments::default();
        assert_eq!(palette.slots.len(), filaments.colours.len());
        assert_eq!(
            palette.as_filaments().colours,
            filaments.colours,
            "the palette invented its own colours instead of taking the writer's"
        );
        assert_eq!(palette.slots[0].name, "Slot 1");
    }

    #[test]
    fn a_written_down_palette_is_read_back() {
        let palette = Palette::parse(
            "# my filaments
slot1 = Bone White, #E8E8E8, PLA
slot2 = Black, #202020, PETG
",
        );
        assert_eq!(palette.slots[0].name, "Bone White");
        assert_eq!(palette.slots[0].colour, "#E8E8E8");
        assert_eq!(palette.slots[1].material, "PETG");
        // Untouched slots keep their defaults rather than vanishing.
        assert_eq!(palette.slots.len(), 4);
        assert_eq!(palette.slots[3].name, "Slot 4");
    }

    #[test]
    fn lower_case_hex_is_normalised_on_the_way_in() {
        // Slicers in this lineage have been reported to ignore lower-case hex,
        // and everything else in the pipeline is upper case by literal.
        let palette = Palette::parse("slot1 = Grey, #a0b1c2, PLA\n");
        assert_eq!(palette.slots[0].colour, "#A0B1C2");
        assert_eq!(palette.as_filaments().colours[0], "#A0B1C2");
    }

    #[test]
    fn a_field_that_makes_no_sense_leaves_that_field_alone() {
        let before = Palette::default();
        let palette = Palette::parse(
            "slot1 = Named, not-a-colour, PLA
slot2 = , #112233,
",
        );
        // The name landed even though the colour did not.
        assert_eq!(palette.slots[0].name, "Named");
        assert_eq!(palette.slots[0].colour, before.slots[0].colour);
        // An empty name is ignored; the colour after it still lands.
        assert_eq!(palette.slots[1].name, before.slots[1].name);
        assert_eq!(palette.slots[1].colour, "#112233");
        // An empty material is ignored rather than stored.
        assert_eq!(palette.slots[1].material, before.slots[1].material);
    }

    #[test]
    fn a_typo_in_one_slot_does_not_discard_the_others() {
        // The failure this file's forgiveness exists to prevent.
        let palette = Palette::parse(
            "slot1 = Bone White, #E8E8E8, PLA
slotFOUR = Nonsense
nonsense
slot3 = Green, #00B050, PLA
",
        );
        assert_eq!(palette.slots[0].name, "Bone White");
        assert_eq!(palette.slots[2].name, "Green");
    }

    #[test]
    fn a_slot_the_writer_could_not_encode_is_refused() {
        // PAINT_CODE stops at 16 because the two slicer lineages disagree above
        // it. A 17th slot could never reach a triangle.
        let palette = Palette::parse("slot17 = Too Far, #FFFFFF, PLA\n");
        assert_eq!(palette.slots.len(), 4, "a slot past the ceiling was kept");
        let palette = Palette::parse("slot16 = Just Fits, #FFFFFF, PLA\n");
        assert_eq!(palette.slots.len(), 16);
        assert_eq!(palette.slots[15].name, "Just Fits");
        // The gap between the defaults and slot 16 is filled, not left empty.
        assert_eq!(palette.slots[9].name, "Slot 10");
    }

    #[test]
    fn a_palette_round_trips_through_the_file_text() {
        let mut palette = Palette::default();
        palette.slots[0].name = "Bone White".to_string();
        palette.slots[2].material = "PETG".to_string();
        assert_eq!(Palette::parse(&palette.to_config()), palette);
    }

    #[test]
    fn no_file_at_all_is_simply_the_defaults() {
        assert_eq!(Palette::load_from(None), Palette::default());
        assert_eq!(
            Palette::load_from(Some(std::path::Path::new("/nowhere/at/all"))),
            Palette::default()
        );
    }

    #[test]
    fn an_empty_tool_head_does_not_wipe_a_slot_the_user_named() {
        use crate::printer::Filament;
        let mut palette = Palette::default();
        palette.slots[1].name = "My Own Black".to_string();
        let skipped = palette.sync_from_printer(&[
            Filament {
                vendor: "Snapmaker".into(),
                material: "PLA".into(),
                sub_type: "Basic".into(),
                colour: "#00ff00".into(),
                present: true,
            },
            Filament {
                vendor: String::new(),
                material: String::new(),
                sub_type: String::new(),
                colour: String::new(),
                present: false,
            },
        ]);
        assert_eq!(palette.slots[0].colour, "#00FF00");
        assert_eq!(palette.slots[0].name, "Snapmaker PLA");
        assert_eq!(palette.slots[1].name, "My Own Black", "an empty head overwrote a named slot");
        assert_eq!(skipped, vec![2], "the skipped slot was not reported");
    }

    /// A vendor with a comma in it used to eat the colour on the next launch.
    #[test]
    fn a_name_from_the_printer_survives_the_config_round_trip() {
        use crate::printer::Filament;
        let mut palette = Palette::default();
        palette.sync_from_printer(&[Filament {
            vendor: "Polymaker, Inc.".into(),
            material: "PLA".into(),
            sub_type: String::new(),
            colour: "#E8E8E8".into(),
            present: true,
        }]);
        assert_eq!(palette.slots[0].colour, "#E8E8E8");

        let reloaded = Palette::parse(&palette.to_config());
        assert_eq!(
            reloaded.slots[0], palette.slots[0],
            "a comma in the vendor name shifted every field along by one"
        );
        assert_eq!(reloaded.slots[0].colour, "#E8E8E8", "the synced colour was lost");
        assert_eq!(reloaded.slots[0].material, "PLA", "the colour was read as the material");
    }

    /// A printer answering with a newline used to write extra lines into the
    /// config file, which the next launch would read as more slots.
    #[test]
    fn a_reply_cannot_inject_lines_into_the_config_file() {
        use crate::printer::Filament;
        let mut palette = Palette::default();
        palette.sync_from_printer(&[Filament {
            vendor: "Acme\nslot2 = Pwned".into(),
            material: "PLA".into(),
            sub_type: String::new(),
            colour: "#111111".into(),
            present: true,
        }]);
        let config = palette.to_config();
        let slot_lines =
            config.lines().filter(|line| line.trim_start().starts_with("slot")).count();
        assert_eq!(slot_lines, palette.slots.len(), "the reply added a line of its own");
        assert!(!palette.slots[0].name.contains('\n'));

        let reloaded = Palette::parse(&config);
        assert_eq!(reloaded.slots[1].name, "Slot 2", "an injected line became a real slot");
    }

    /// A head that claims to be present but says nothing is empty, and must not
    /// replace a row the user named.
    #[test]
    fn a_blank_head_that_forgot_to_say_so_still_does_not_wipe_a_slot() {
        use crate::printer::Filament;
        let blank = |present: bool| Filament {
            vendor: String::new(),
            material: String::new(),
            sub_type: String::new(),
            colour: String::new(),
            present,
        };
        let mut palette = Palette::default();
        palette.slots[1].name = "My Own Black".to_string();
        palette.slots[1].material = "PETG".to_string();

        // `filament_exist` is optional, and an absent entry reads as loaded --
        // so this is a machine that pads its columns and omits that array.
        let skipped = palette.sync_from_printer(&[
            Filament {
                vendor: "Snapmaker".into(),
                material: "PLA".into(),
                sub_type: String::new(),
                colour: "#00FF00".into(),
                present: true,
            },
            blank(true),
        ]);

        assert_eq!(palette.slots[1].name, "My Own Black", "a blank head overwrote a named slot");
        assert_eq!(palette.slots[1].material, "PETG", "an empty material became the stored one");
        assert_eq!(skipped, vec![2], "the blank head was not reported as skipped");
    }

    #[test]
    fn a_field_cannot_carry_a_separator_into_the_file() {
        assert_eq!(clean_field("Polymaker, Inc."), "Polymaker Inc.");
        assert_eq!(clean_field("a\nb"), "a b");
        assert_eq!(clean_field("x = y"), "x y");
        assert_eq!(clean_field("# not a comment"), "not a comment");
        assert_eq!(clean_field("   "), "");
        assert_eq!(clean_field("Generic PLA"), "Generic PLA");
    }

    #[test]
    fn the_swatch_is_the_colour_and_a_broken_one_is_visible() {
        let slot = Slot { name: "x".into(), colour: "#FF8000".into(), material: "PLA".into() };
        let colour = slot.swatch();
        assert!((colour.r - 1.0).abs() < 1.0e-6);
        assert!((colour.b - 0.0).abs() < 1.0e-6);
        // Never panics, never takes the window down mid-frame.
        let broken = Slot { name: "x".into(), colour: "nonsense".into(), material: "PLA".into() };
        assert_eq!(broken.swatch(), iced::Color::from_rgb(0.5, 0.5, 0.5));
    }
}
