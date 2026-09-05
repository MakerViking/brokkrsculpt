// SPDX-License-Identifier: AGPL-3.0-only

//! Design tokens, taken from SindriCAD so the two applications read as a
//! family.
//!
//! SindriCAD is a Tauri application with a web front end and its tokens live in
//! `src/styles.css` under `:root`. The stacks diverge on purpose, because a
//! sculpt loop cannot route through a webview at these frame rates, but the
//! look should not. Every widget is built from the values here. Do not
//! hand tune a colour at a call site: change the token.

// The full SindriCAD token set is mirrored here on purpose, so a new widget
// reaches for an existing token instead of inventing a colour. Tokens with no
// consumer yet are not dead code, they are the palette.
#![allow(dead_code)]

use iced::{Color, Font, Padding};

const fn rgb(red: u8, green: u8, blue: u8) -> Color {
    Color { r: red as f32 / 255.0, g: green as f32 / 255.0, b: blue as f32 / 255.0, a: 1.0 }
}

const fn rgba(red: u8, green: u8, blue: u8, alpha: f32) -> Color {
    Color { r: red as f32 / 255.0, g: green as f32 / 255.0, b: blue as f32 / 255.0, a: alpha }
}

// Surfaces, deepest first.
/// `--bg`, the application background.
pub const BG: Color = rgb(0x0e, 0x0f, 0x12);
/// `--bg-deep`, an inset well.
pub const BG_DEEP: Color = rgb(0x0b, 0x0d, 0x10);
/// `--bg-vignette`, the centre of the viewport vignette.
pub const BG_VIGNETTE: Color = rgb(0x15, 0x17, 0x1c);
/// `--panel-hi`, the gradient stop above a panel.
pub const PANEL_HI: Color = rgb(0x18, 0x1a, 0x20);
/// `--panel`, docked panels.
pub const PANEL: Color = rgb(0x16, 0x18, 0x1d);
/// `--panel-2`, nested surfaces, inputs and popups.
pub const PANEL_2: Color = rgb(0x1c, 0x1f, 0x26);
/// `--raised`, a hovered interactive surface.
pub const RAISED: Color = rgb(0x22, 0x26, 0x2e);
/// `--raised-2`, pressed or strongly hovered.
pub const RAISED_2: Color = rgb(0x2a, 0x2f, 0x38);

// Lines.
/// `--line`, a hairline divider.
pub const LINE: Color = rgb(0x26, 0x2a, 0x31);
/// `--line-strong`, the border of a focusable control.
pub const LINE_STRONG: Color = rgb(0x32, 0x38, 0x43);

// Text.
/// `--text`, primary.
pub const TEXT: Color = rgb(0xe6, 0xe8, 0xec);
/// `--text-dim`, secondary and labels.
pub const TEXT_DIM: Color = rgb(0x9a, 0xa3, 0xaf);
/// `--text-mute`, tertiary and hints.
pub const TEXT_MUTE: Color = rgb(0x6b, 0x72, 0x80);

// Accent and status.
/// `--accent`, the amber the whole family is built around.
pub const ACCENT: Color = rgb(0xff, 0x7a, 0x3c);
/// `--accent-hot`, the hovered accent.
pub const ACCENT_HOT: Color = rgb(0xff, 0x9a, 0x5c);
/// `--accent-tint`, an accent wash behind a selected control.
pub const ACCENT_TINT: Color = rgba(0xff, 0x7a, 0x3c, 0.14);
/// `--on-accent`, text on a filled amber surface.
pub const ON_ACCENT: Color = rgb(0x16, 0x0a, 0x04);
/// `--ok`.
pub const OK: Color = rgb(0x4a, 0xd0, 0x7d);
/// `--warn`.
pub const WARN: Color = rgb(0xe0, 0xa0, 0x20);
/// `--error`.
pub const ERROR: Color = rgb(0xff, 0x5c, 0x5c);
/// `--error-tint`, a red wash behind a destructive control being hovered.
///
/// A wash rather than a fill, because the icon on top of it is drawn at a fixed
/// colour and cannot lighten to stay legible the way text does — see
/// `icon.rs`'s header.
pub const ERROR_TINT: Color = rgba(0xff, 0x5c, 0x5c, 0.22);

/// The mask's cool blue: the brush ring while protection is being painted, and
/// the hue the viewport tint shifts toward.
///
/// **Not in SindriCAD's token set, and not derivable from one.** Every colour
/// above is warm or neutral, deliberately — the clay matcap is warm and
/// `--accent` already means "a press adds". This has to be a hue the matcap
/// cannot produce, or a masked surface reads as a slightly different shade of
/// clay. Generated from the matcap's own values: its whole luminance swing is
/// 102 levels and its coolest pixel is a fill-lit cavity at `b - r = +17`,
/// where this sits at `b - r = +100`. That is the measurement the choice rests
/// on, and it is why a "gentle cool desaturation" was refused — at full
/// strength it moves a lit pixel by 36 levels against the matcap's own 102, so
/// a fully masked lit pixel would be brighter than an unmasked shadowed one.
pub const MASK: Color = rgb(0x58, 0x87, 0xf0);

/// The same hue, pale, for the ring that says a drag is REMOVING protection.
///
/// Deliberately not [`ERROR`]. Red means "this removes material" everywhere
/// else in the application and unmasking removes none at all; masking and
/// unmasking are one thing and its inverse, so they are one hue and two
/// lightnesses rather than two hues.
pub const MASK_PALE: Color = rgb(0xa8, 0xc4, 0xf8);

/// The transform gizmo's three axes: X, Y, Z, in that order.
///
/// # Why these are authored rather than taken from what is already here
///
/// The palette above holds no triad. [`ERROR`] is the obvious red for X and is
/// exactly the wrong choice: it already means "this stroke removes material" on
/// the brush ring, and a red arrow beside a red ring on the same surface would
/// make one of the two lie. [`OK`] and [`MASK`] are similarly spoken for.
///
/// So these are the industry triad -- red X, green Y, blue Z, which every
/// modelling tool a user of this one has touched already uses -- shifted away
/// from the tokens they would otherwise collide with: the X is a rose rather
/// than [`ERROR`]'s alarm red, and the Z is a cyan rather than [`MASK`]'s
/// indigo. Against the warm clay matcap all three read as instrument colours
/// rather than as material.
pub const AXIS_X: Color = rgb(0xf0, 0x5a, 0x76);
pub const AXIS_Y: Color = rgb(0x6a, 0xd8, 0x62);
pub const AXIS_Z: Color = rgb(0x4a, 0xa8, 0xf0);

/// The same three when the pointer is over them.
///
/// **Less saturated, not brighter.** Brightening a saturated colour to say
/// "hovered" runs it past the top of the channel it is already at -- Godot's
/// gizmo went through this (PR #50597) and ended up with a hover state that
/// glowed white on the axis you could least afford to lose track of. Washing
/// toward white keeps the hue readable and still reads as lit.
pub const AXIS_X_HOVER: Color = rgb(0xff, 0xa8, 0xb6);
pub const AXIS_Y_HOVER: Color = rgb(0xb4, 0xf0, 0xae);
pub const AXIS_Z_HOVER: Color = rgb(0xa6, 0xd6, 0xff);

/// The three axis colours and their hover variants, indexed by axis.
pub const AXIS: [Color; 3] = [AXIS_X, AXIS_Y, AXIS_Z];
pub const AXIS_HOVER: [Color; 3] = [AXIS_X_HOVER, AXIS_Y_HOVER, AXIS_Z_HOVER];

/// Convert a theme colour to the linear space the overlay shader expects.
///
/// The tokens in this module are sRGB, because they came from SindriCAD's CSS.
/// Handing those straight to the shader would draw every overlay too bright,
/// and inconsistently with the model beside it.
///
/// **Here rather than at the call site**, because it was written out verbatim
/// in `cursor.rs` and again in `navcube.rs`, and the gizmo would have been the
/// third copy. Three copies of a colour-space conversion is three chances for
/// one overlay to end up in a different space from the one beside it.
pub fn linear(colour: Color, alpha: f32) -> [f32; 4] {
    let to_linear = |c: f32| c.powf(2.2);
    [to_linear(colour.r), to_linear(colour.g), to_linear(colour.b), alpha]
}

// Radii, from `--r-sm` through `--r-lg`.
pub const RADIUS_SM: f32 = 6.0;
pub const RADIUS_MD: f32 = 8.0;
pub const RADIUS_LG: f32 = 10.0;

// Icon sizes. These live here rather than at the call sites for the reason
// SindriCAD needed a whole CSS sizing table: an icon does not inherit a font
// size, so every control that used to hold a character needs its box named
// somewhere. Naming them together is also what keeps the set looking like a
// set.
/// Window controls and the timeline's transport.
pub const ICON_CHROME: f32 = 14.0;
/// The tool strip, above the tool's name.
///
/// Eighteen rather than the twenty-four the grid is drawn on, because the strip
/// has a vertical budget: eight brushes, three mirror toggles, the cut and the
/// mask have to fit a 768-high window without scrolling. A properties panel
/// taller than its window has already happened here once and put every
/// SpaceMouse binding out of reach. Adding the mask as the eleventh button was
/// paid for by two subtractions rather than by shrinking this — see
/// `panel.rs`'s `tool_strip`.
pub const ICON_TOOL: f32 = 18.0;
/// Inline with a line of text: a section's collapse marker.
pub const ICON_INLINE: f32 = 11.0;

// Spacing scale, `--s-1` through `--s-6`.
pub const S1: f32 = 4.0;
pub const S2: f32 = 6.0;
pub const S3: f32 = 8.0;
pub const S4: f32 = 12.0;
pub const S5: f32 = 16.0;
pub const S6: f32 = 20.0;

// Typography. SindriCAD sets Inter at 13px with a 10px uppercase caption. Inter
// is not bundled, so the request falls back to whatever the system provides,
// which is the same thing the web build does.
pub const FONT: Font = Font::with_name("Inter");
pub const TEXT_SIZE: f32 = 13.0;
pub const TEXT_SIZE_SMALL: f32 = 11.0;
pub const CAPTION_SIZE: f32 = 10.0;
/// The one heading big enough to be read as a title, on the welcome screen.
/// Nothing inside the application proper uses it: panels are dense by design
/// and a heading this size in one would cost a row of controls.
pub const TITLE_SIZE: f32 = 20.0;
/// Monospace for the debug overlay, where columns of numbers must not jitter.
pub const MONO: Font = Font::MONOSPACE;

/// Standard padding inside a docked panel.
pub const PANEL_PADDING: Padding = Padding { top: S4, right: S5, bottom: S4, left: S5 };

/// The iced palette, so the built in widget styles land in the right family
/// before any per widget styling is applied.
pub fn palette() -> iced::theme::Palette {
    iced::theme::Palette {
        background: BG,
        text: TEXT,
        primary: ACCENT,
        success: OK,
        warning: WARN,
        danger: ERROR,
    }
}

pub fn theme() -> iced::Theme {
    iced::Theme::custom("BrokkrSculpt".to_string(), palette())
}

/// A docked panel: flat fill, hairline border, medium radius.
pub fn panel(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(PANEL.into()),
        text_color: Some(TEXT),
        border: iced::Border { color: LINE, width: 1.0, radius: RADIUS_MD.into() },
        ..Default::default()
    }
}

/// A filament swatch: a block of one slot's colour.
///
/// **The first style here that takes a colour**, and so the first that returns
/// a closure rather than a `fn`. Every other style in this module is a plain
/// function pointer because the palette is fixed at compile time; a filament's
/// colour is not, so the value has to be captured. `iced` takes any
/// `Fn(&Theme) -> Style` for a container, so this costs nothing but the move.
///
/// It carries a border in the panel's own line colour rather than none, because
/// a white filament against the panel would otherwise have no edge at all and
/// read as a hole rather than as a swatch.
pub fn swatch(colour: Color) -> impl Fn(&iced::Theme) -> iced::widget::container::Style {
    move |_theme: &iced::Theme| iced::widget::container::Style {
        background: Some(colour.into()),
        border: iced::Border { color: LINE, width: 1.0, radius: RADIUS_SM.into() },
        ..Default::default()
    }
}

/// The viewport well, deeper than the surrounding chrome.
pub fn viewport_well(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(BG_VIGNETTE.into()),
        border: iced::Border { color: LINE, width: 1.0, radius: RADIUS_MD.into() },
        ..Default::default()
    }
}

/// The debug overlay card, floating over the viewport.
pub fn overlay_card(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Color { a: 0.82, ..BG_DEEP }.into()),
        text_color: Some(TEXT_DIM),
        border: iced::Border { color: LINE_STRONG, width: 1.0, radius: RADIUS_SM.into() },
        ..Default::default()
    }
}

/// A tool in the strip that is not the current one.
///
/// Built from the existing tokens rather than new colours, per this file's own
/// rule: hovering and pressing walk up the raised scale, and the border is the
/// same hairline every other focusable control uses.
pub fn tool_button(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::widget::button::Status;
    let background = match status {
        Status::Hovered => RAISED,
        Status::Pressed => RAISED_2,
        Status::Active | Status::Disabled => PANEL_2,
    };
    iced::widget::button::Style {
        background: Some(background.into()),
        text_color: TEXT_DIM,
        border: iced::Border { color: LINE_STRONG, width: 1.0, radius: RADIUS_SM.into() },
        ..Default::default()
    }
}

/// The tool that is currently selected: filled with the accent.
///
/// A fill rather than a border, because at this size a one pixel outline is
/// not readable at a glance, and knowing which brush is live at a glance is
/// the entire point of having a strip instead of a dropdown.
pub fn tool_button_active(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::widget::button::Status;
    let background = match status {
        Status::Hovered | Status::Pressed => ACCENT_HOT,
        Status::Active | Status::Disabled => ACCENT,
    };
    iced::widget::button::Style {
        background: Some(background.into()),
        text_color: ON_ACCENT,
        border: iced::Border { color: ACCENT, width: 1.0, radius: RADIUS_SM.into() },
        ..Default::default()
    }
}

/// The destructive answer in a prompt: throwing work away.
///
/// Outlined rather than filled, so the accent-filled safe answer stays the one
/// the eye lands on. A filled red button next to a filled orange one reads as
/// two equally weighted choices, which is the wrong shape for "save" versus
/// "lose it".
pub fn danger_button(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::widget::button::Status;
    let background = match status {
        Status::Hovered => RAISED,
        Status::Pressed => RAISED_2,
        Status::Active | Status::Disabled => PANEL_2,
    };
    iced::widget::button::Style {
        background: Some(background.into()),
        text_color: ERROR,
        border: iced::Border { color: ERROR, width: 1.0, radius: RADIUS_SM.into() },
        ..Default::default()
    }
}

/// The dimming layer behind a modal prompt.
///
/// Dark but translucent: the question is about the sculpt, so the sculpt has to
/// stay visible behind it.
///
/// **Decoration only.** This used to say its real job was swallowing presses,
/// and that was simply false — a styled `container` forwards every event it is
/// given. The widget that swallows them is `panel::modal_layer`'s `opaque`
/// wrapper; this only makes the dead area look dead.
pub fn scrim(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(rgba(0x00, 0x00, 0x00, 0.55).into()),
        ..Default::default()
    }
}

/// A menu that drops from the bar, or any surface you must be able to read.
///
/// Opaque, unlike [`overlay_card`]. That card is 0.82 alpha because it floats
/// over the model and should not hide it; a menu floats over the *debug
/// overlay*, and at two translucent layers the text of both became unreadable.
pub fn menu_card(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(PANEL_2.into()),
        text_color: Some(TEXT),
        border: iced::Border { color: LINE_STRONG, width: 1.0, radius: RADIUS_MD.into() },
        ..Default::default()
    }
}

/// A collapsible section heading. Flat and borderless, so it reads as a
/// heading rather than as a button, but brightens on hover so it is
/// discoverable as one.
pub fn section_heading(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::widget::button::Status;
    iced::widget::button::Style {
        background: None,
        text_color: match status {
            Status::Hovered | Status::Pressed => TEXT,
            Status::Active | Status::Disabled => TEXT_MUTE,
        },
        ..Default::default()
    }
}

/// The shape of every button style in this module.
///
/// Worth a name because a style passed as a *value* rather than called needs
/// one: each `fn` here has its own distinct type, so a closure that takes one
/// infers that exact function and then refuses the next, which reads as a
/// baffling "expected fn item, found a different fn item".
pub type ButtonStyle =
    fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style;

/// The style *and* the icon colour for a control that can be selected.
///
/// **Always take both from here; never pick one beside the other.** Nothing in
/// the type system ties an icon's colour to the button it sits in, and the two
/// disagree badly if they drift: [`tool_button_active`] fills with `ACCENT` and
/// drops its foreground to a near-black `ON_ACCENT`, where [`tool_button`]
/// leaves it `TEXT_DIM` on a dark panel. Text follows that automatically
/// because the button sets `text_color`. An icon cannot — it is a canvas drawn
/// at a colour fixed when the widget is built — so a call site that chose the
/// style here and the colour by hand would eventually put a near-black icon on
/// a near-black panel, and nothing would fail. See `icon.rs`'s header.
pub fn tool_toggle(selected: bool) -> (ButtonStyle, Color) {
    if selected { (tool_button_active, ON_ACCENT) } else { (tool_button, TEXT_DIM) }
}

/// A window control: minimise, maximise, close.
///
/// **Its hover feedback is the background, not the text colour, and that is
/// forced rather than chosen.** [`section_heading`] — which these three used
/// while they were Unicode characters — signals hover *only* by lifting its
/// text from `TEXT_MUTE` to `TEXT`. An icon is a canvas drawn at a colour fixed
/// when the widget is built, so it cannot follow that, and keeping
/// `section_heading` would have left the window controls with no hover response
/// at all. Moving the affordance to the background is also what every title bar
/// does. See `icon.rs`'s header for why the colour cannot cascade.
pub fn window_control(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::widget::button::Status;
    iced::widget::button::Style {
        background: match status {
            Status::Hovered => Some(RAISED.into()),
            Status::Pressed => Some(RAISED_2.into()),
            Status::Active | Status::Disabled => None,
        },
        text_color: TEXT_DIM,
        border: iced::Border { radius: RADIUS_SM.into(), ..Default::default() },
        ..Default::default()
    }
}

/// Close, which is [`window_control`] that goes red.
///
/// A tint rather than a filled red: the icon on top is a fixed colour and would
/// be unreadable on a saturated fill, where text would simply have been given
/// `ON_ACCENT`. The tint still reads as "this one is different".
pub fn window_control_close(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    use iced::widget::button::Status;
    iced::widget::button::Style {
        background: match status {
            Status::Hovered => Some(ERROR_TINT.into()),
            Status::Pressed => Some(ERROR.into()),
            Status::Active | Status::Disabled => None,
        },
        text_color: TEXT_DIM,
        border: iced::Border { radius: RADIUS_SM.into(), ..Default::default() },
        ..Default::default()
    }
}

/// The strip itself: a narrow docked column against the viewport.
pub fn tool_strip(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(PANEL.into()),
        text_color: Some(TEXT),
        border: iced::Border { color: LINE, width: 1.0, radius: RADIUS_MD.into() },
        ..Default::default()
    }
}

/// The timeline's empty track.
///
/// Darker than a panel so the strip reads as a groove cut into the layout
/// rather than another surface floating on it.
pub fn timeline_track(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(BG_DEEP.into()),
        border: iced::Border { radius: RADIUS_SM.into(), width: 1.0, color: LINE },
        ..Default::default()
    }
}

/// One stored key.
pub fn timeline_key(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(TEXT_DIM.into()),
        border: iced::Border { radius: 2.0.into(), ..Default::default() },
        ..Default::default()
    }
}

/// A key under the pointer, or being dragged.
pub fn timeline_key_lit(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(ACCENT.into()),
        border: iced::Border { radius: 2.0.into(), ..Default::default() },
        ..Default::default()
    }
}

/// The playhead.
pub fn timeline_playhead(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style { background: Some(ACCENT_HOT.into()), ..Default::default() }
}

// --- the body panel ----------------------------------------------------------

/// A row of the body list that is not the active one.
///
/// No background at all, so the panel behind it shows through and a list of
/// twelve reads as a list rather than as twelve cards.
pub fn body_row(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        border: iced::Border { radius: RADIUS_SM.into(), ..Default::default() },
        ..Default::default()
    }
}

/// The row edits land on.
///
/// [`ACCENT_TINT`] rather than a solid fill: at 0.14 alpha the name stays
/// legible in `TEXT`, where a solid accent would need the near-black `ON_ACCENT`
/// and make the selected row the only one in the list with dark text.
pub fn body_row_active(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(ACCENT_TINT.into()),
        border: iced::Border { radius: RADIUS_SM.into(), ..Default::default() },
        ..Default::default()
    }
}

/// A row solo is not showing.
///
/// **The dim is the second signal, and it is not decoration.** While solo is on,
/// an out-of-scope row's own eye is still on, so its glyph is still an open eye
/// — ten rows claiming "visible" over a viewport drawing one. `ICON_INLINE` is
/// 11 px and the muted ink alone is a subtle difference at that size, so the
/// row itself recedes: [`BG_DEEP`] is the inset-well colour, darker than the
/// panel behind it, which reads as "this is switched off" rather than as "this
/// is selected".
pub fn body_row_out_of_scope(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(BG_DEEP.into()),
        border: iced::Border { radius: RADIUS_SM.into(), ..Default::default() },
        ..Default::default()
    }
}

/// The two-pixel bar down the inside edge of the active row.
///
/// The tint alone is 0.14 alpha over a dark panel, which survives a screenshot
/// and not a bright room. The bar is what makes the selection readable at a
/// glance, and it is the same device SindriCAD's tree uses.
pub fn body_row_marker(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(ACCENT.into()),
        border: iced::Border { radius: 1.0.into(), ..Default::default() },
        ..Default::default()
    }
}

/// The two-pixel accent line that says where a dragged row would land.
///
/// The same [`ACCENT`] as the selection marker, and the same 1 px radius, so
/// the two read as one vocabulary: orange means "this row" and an orange line
/// means "this gap". It is drawn INDENTED to the depth the block would take,
/// which is how a line between two rows says which of them it would become a
/// sibling of.
pub fn drop_line(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(ACCENT.into()),
        border: iced::Border { radius: 1.0.into(), ..Default::default() },
        ..Default::default()
    }
}

/// A closed folder the drop would go INSIDE.
///
/// The tint alone would be the active row's style exactly, and the active row
/// is very often the row being dragged -- so this adds the accent outline. The
/// filled row rather than a line is the third indicator every shipped tree drag
/// gets wrong: without it, "into this folder" and "after this folder" draw the
/// same thing and the user finds out which on release.
pub fn body_row_drop_into(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(ACCENT_TINT.into()),
        border: iced::Border { color: ACCENT, width: 1.0, radius: RADIUS_SM.into() },
        ..Default::default()
    }
}

/// The well a body's picture sits in.
///
/// A flat 28 px inset with a hairline border. The picture is a `shader` widget
/// inside it with a pixel of padding, so the border still frames the cell -- and
/// a body with no cell to draw falls back to the bare well, which is what this
/// was before there were pictures.
///
/// **`BG_DEEP` here and `THUMBNAIL_BACKGROUND` in `brokkr-gpu` are the same
/// colour and have to stay that way.** An unrendered cell is filled with that
/// constant, so a mismatch shows as a picture sitting in a slightly different
/// hole from the rows either side of it. `brokkr-gpu` must not depend on the
/// application, which is why there are two of them.
pub fn body_thumbnail(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(BG_DEEP.into()),
        border: iced::Border { color: LINE, width: 1.0, radius: RADIUS_SM.into() },
        ..Default::default()
    }
}
