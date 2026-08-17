// SPDX-License-Identifier: AGPL-3.0-or-later

//! Messages exchanged between the widget tree and the application.

use brokkr_core::{BrushKind, FalloffCurve, MirrorAxis, PatternKind};
use iced::Vector;

use crate::spacemouse::{Action, Axis, ButtonAction, Mode};

/// A file format the sculpt can be written to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Binary STL. Universal, carries no units, repeats every vertex.
    Stl,
    /// Wavefront OBJ. Text, shares vertices, carries normals.
    Obj,
    /// 3MF. The only one of the three that states its units.
    ThreeMf,
}

impl ExportFormat {
    pub const ALL: [ExportFormat; 3] =
        [ExportFormat::Stl, ExportFormat::Obj, ExportFormat::ThreeMf];

    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Stl => "STL",
            ExportFormat::Obj => "OBJ",
            ExportFormat::ThreeMf => "3MF",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Stl => "stl",
            ExportFormat::Obj => "obj",
            ExportFormat::ThreeMf => "3mf",
        }
    }
}

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

/// One change to the SpaceMouse settings.
///
/// Gathered into one message rather than a variant each, because there are
/// twelve of them and they all do the same thing: patch the config and, for
/// the ones that are not mid drag, write it out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpaceMouseSetting {
    Mode(Mode),
    Deadzone(f32),
    PanSens(f32),
    ZoomSens(f32),
    OrbitSens(f32),
    /// Which raw axis drives an action.
    Binding(Action, Axis),
    Invert(Action, bool),
    Button(usize, ButtonAction),
    /// Back to the values ported from SindriCAD.
    Reset,
    /// Persist the current settings. Sent when a slider is released rather
    /// than on every step of a drag, which would rewrite the file sixty times
    /// a second.
    Save,
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
    /// Which surface pattern multiplies into the brush. See
    /// `brokkr_core::pattern`: one control that modifies every brush, rather
    /// than a brush per pattern.
    PatternChanged(PatternKind),
    PatternScaleChanged(f32),
    PatternDepthChanged(f32),
    /// Flip one mirror plane. A toggle rather than an explicit on/off,
    /// because both the strip button and the keyboard act on what is currently
    /// set rather than knowing it.
    SymmetryAxisToggled(MirrorAxis),
    /// Multiply the brush radius, for the keyboard nudge. Multiplicative
    /// because the radius spans fifty to one and a fixed step would crawl at
    /// the top and jump at the bottom.
    BrushRadiusScaled(f32),
    PressureToggled(bool),
    PressureCurveChanged(f32),
    TiltToggled(bool),
    ResetPressurePeak,
    Undo,
    Redo,
    /// Throw the model away and start from a fresh sphere.
    ResetSphere,
    /// Write the sculpt out.
    Export(ExportFormat),
    /// Rebuild the volume at a different voxel size, which is the explicit
    /// operation that increases or reduces detail.
    Resample(f32),
    SpaceMouse(SpaceMouseSetting),
}
