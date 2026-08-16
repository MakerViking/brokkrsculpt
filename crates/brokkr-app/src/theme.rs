// SPDX-License-Identifier: AGPL-3.0-or-later

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

// Radii, from `--r-sm` through `--r-lg`.
pub const RADIUS_SM: f32 = 6.0;
pub const RADIUS_MD: f32 = 8.0;
pub const RADIUS_LG: f32 = 10.0;

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
