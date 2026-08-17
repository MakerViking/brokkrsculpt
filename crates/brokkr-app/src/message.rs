// SPDX-License-Identifier: AGPL-3.0-or-later

//! Messages exchanged between the widget tree and the application.

use brokkr_core::{BrushKind, FalloffCurve};
use iced::Vector;

/// Which pointer button an event refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerButton {
    Left,
    Right,
    Middle,
}

/// Input from the viewport, in widget local pixels.
#[derive(Debug, Clone, Copy)]
pub enum PointerEvent {
    Moved {
        position: Vector,
        size: Vector,
    },
    Pressed {
        button: PointerButton,
        position: Vector,
        size: Vector,
    },
    Released {
        button: PointerButton,
    },
    /// Positive is a scroll toward the model.
    Scrolled {
        amount: f32,
    },
    Modifiers {
        shift: bool,
        control: bool,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    Pointer(PointerEvent),
    /// One presented frame, used to drive the frame rate readout and to keep
    /// the viewport redrawing while a stroke is in progress.
    Frame,
    BrushKindChanged(BrushKind),
    BrushRadiusChanged(f32),
    BrushStrengthChanged(f32),
    FalloffChanged(FalloffCurve),
    SymmetryToggled(bool),
    PressureToggled(bool),
    PressureCurveChanged(f32),
    ResetPressurePeak,
    Undo,
    Redo,
    /// Throw the model away and start from a fresh sphere.
    ResetSphere,
}
