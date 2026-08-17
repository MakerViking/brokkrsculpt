// SPDX-License-Identifier: AGPL-3.0-or-later

//! Messages exchanged between the widget tree and the application.

use brokkr_core::{BrushKind, FalloffCurve, MirrorAxis, PatternKind};
use iced::Vector;

use crate::app::SizingTarget;
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
        alt: bool,
    },
}

/// A menu on the bar along the top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopMenu {
    File,
    Help,
}

impl TopMenu {
    pub const ALL: [TopMenu; 2] = [TopMenu::File, TopMenu::Help];

    pub fn label(self) -> &'static str {
        match self {
            TopMenu::File => "File",
            TopMenu::Help => "Help",
        }
    }
}

/// What the user answered to the unsaved-work prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmChoice {
    /// Write the file first, then go on with whatever was asked for.
    Save,
    /// Throw the changes away and go on.
    Discard,
    /// Stay where we are and do nothing.
    Cancel,
}

/// A finished import, on its way from a background task to the update loop.
///
/// Hand written rather than a `Box` or a plain field, and the reason is not
/// stylistic: `Message` derives `Debug` and `Clone`, `Volume` derives neither,
/// and `Box<T>` is `Clone` only when `T` is. `Arc<Mutex<T>>` is `Clone` for any
/// `T` but is `Debug` only when `T` is, so the `Debug` impl below is what makes
/// the pair work. Do not "simplify" this into a `Box` -- it will not compile.
///
/// The slot is taken from rather than read, because the volume moves out into
/// the application exactly once.
#[derive(Clone)]
pub struct ImportPayload(
    pub std::sync::Arc<std::sync::Mutex<Option<Result<Imported, brokkr_core::import::ImportError>>>>,
);

impl ImportPayload {
    pub fn new(result: Result<Imported, brokkr_core::import::ImportError>) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(Some(result))))
    }

    /// Take the result out. A second call yields `None`, which is correct: a
    /// message can in principle be delivered twice and the volume must move
    /// once.
    pub fn take(&self) -> Option<Result<Imported, brokkr_core::import::ImportError>> {
        self.0.lock().ok().and_then(|mut slot| slot.take())
    }
}

impl std::fmt::Debug for ImportPayload {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str("ImportPayload(..)")
    }
}

/// What a successful import produced.
pub struct Imported {
    pub volume: brokkr_core::Volume,
    pub report: brokkr_core::voxelise::VoxeliseReport,
    pub source: std::path::PathBuf,
    pub elapsed_ms: f64,
}

/// A collapsible block of the properties panel.
///
/// The panel has more in it than fits a 1080 high window, and a scrollable
/// alone left every settings section below the fold — including the SpaceMouse
/// bindings, which made a deliberately rebindable puck practically
/// unrebindable. Collapsing keeps every heading reachable without a scroll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelSection {
    Pattern,
    Pen,
    SpaceMouse,
    Detail,
    Export,
}

impl PanelSection {
    pub const ALL: [PanelSection; 5] = [
        PanelSection::Pattern,
        PanelSection::Pen,
        PanelSection::SpaceMouse,
        PanelSection::Detail,
        PanelSection::Export,
    ];

    pub fn title(self) -> &'static str {
        match self {
            PanelSection::Pattern => "PATTERN",
            PanelSection::Pen => "PEN",
            PanelSection::SpaceMouse => "SPACEMOUSE",
            PanelSection::Detail => "DETAIL",
            PanelSection::Export => "EXPORT",
        }
    }

    /// Whether it starts open. The two long ones start closed so every
    /// heading is visible without scrolling.
    pub fn open_by_default(self) -> bool {
        !matches!(self, PanelSection::Pen | PanelSection::SpaceMouse)
    }
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
    /// Flip every axis at once.
    InvertAll,
    /// Back to the built in defaults.
    Reset,
    /// Persist the current settings. Sent when a slider is released rather
    /// than on every step of a drag, which would rewrite the file sixty times
    /// a second.
    Save,
}

/// Not `Copy`: one variant carries the raw text of a numeric field being typed
/// into, which has to be owned so a half typed value survives.
#[derive(Debug, Clone)]
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
    /// A hold-and-drag adjustment of one brush number has begun. ZBrush's `s`
    /// and `u`: the fast path for radius in every sculpting tool is a drag
    /// against a live ring, not a slider in a panel.
    SizingStarted(SizingTarget),
    SizingEnded,
    /// Whether the radius tracks the model's size (ZBrush calls it Dynamic)
    /// rather than staying a fixed number of millimetres.
    DynamicRadiusToggled(bool),
    /// Close the right-click menu.
    MenuClosed,
    /// Open a top bar menu, or close it if it is the one already open.
    TopMenuToggled(TopMenu),
    /// Put the diagnostics on the clipboard, for pasting into a bug report.
    DiagnosticsCopied,
    /// Open the issue tracker in a browser, with what we know prefilled.
    IssueOpened,
    /// A numeric field in the menu was typed into. Carries the raw text, since
    /// a half typed value has to survive until it parses.
    MenuFieldEdited(SizingTarget, String),
    /// Commit whatever is in the field being edited and stop editing it.
    MenuFieldSubmitted,
    PressureToggled(bool),
    PressureCurveChanged(f32),
    TiltToggled(bool),
    ResetPressurePeak,
    Undo,
    Redo,
    /// Throw the model away and start from a fresh sphere.
    ResetSphere,
    /// Write the sculpt out. Opens a dialog first.
    Export(ExportFormat),
    /// Start a fresh sculpt.
    NewSculpt,
    /// Open a saved sculpt: ask for a file, then load whatever came back.
    OpenRequested,
    OpenChosen(Option<std::path::PathBuf>),
    /// Open a named file straight from the recent list, with no dialog.
    OpenRecent(std::path::PathBuf),
    /// Load the crash net left behind by a session that did not save.
    RecoverAutosave,
    /// Import a mesh: ask for a file, then read and voxelise whatever came back.
    ImportRequested,
    ImportChosen(Option<std::path::PathBuf>),
    /// The finished import, or why it failed.
    ///
    /// The payload is carried in a shared slot because `Message` derives `Debug`
    /// and `Clone` while `Volume` derives neither, so it cannot be put in a
    /// variant directly. See [`ImportPayload`].
    ImportLoaded(ImportPayload),
    /// Save. Over the current file if there is one, otherwise as a new one.
    SaveRequested,
    SaveAsRequested,
    SaveChosen(Option<std::path::PathBuf>),
    /// Export, in two halves for the same reason: ask, then write.
    ExportRequested(ExportFormat),
    ExportChosen(ExportFormat, Option<std::path::PathBuf>),
    /// Rebuild the volume at a different voxel size, which is the explicit
    /// operation that increases or reduces detail.
    Resample(f32),
    /// The window manager asked to close. Carries the window, because
    /// `exit_on_close_request(false)` means nothing closes it but us.
    CloseRequested(iced::window::Id),
    /// An answer to the unsaved-work prompt.
    ConfirmAnswered(ConfirmChoice),
    /// A path chosen by the Save button *inside* the prompt. Separate from
    /// `SaveChosen` because this one continues to the pending action after a
    /// successful write, and must not after a failed one.
    SavedThenContinue(Option<std::path::PathBuf>),
    SpaceMouse(SpaceMouseSetting),
    /// Open or close one block of the properties panel.
    SectionToggled(PanelSection),
}
