// SPDX-License-Identifier: AGPL-3.0-only

//! Messages exchanged between the widget tree and the application.

use brokkr_core::{BrushKind, FalloffCurve, MaskFilter, MirrorAxis, PatternKind};
use iced::Vector;

use crate::app::{SizingTarget, Tool};
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
    ///
    /// Carries where the pointer was, because the zoom is anchored on what is
    /// under it rather than on the camera's target. Legal to add without
    /// weakening the capture rule: the scroll arm already gates on
    /// `cursor.position_in(bounds)`, so there is no case where a scroll is
    /// routed here without a position to go with it.
    Scrolled {
        amount: f32,
        position: Vector,
        size: Vector,
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
    /// Which way the mesh's own up pointed, if it could be guessed, already
    /// expressed in sculpt space. See [`brokkr_core::resting_up`].
    ///
    /// Carried from the import task because it has to be measured on the mesh,
    /// and by the time the volume exists the mesh is gone.
    pub resting_up: Option<brokkr_core::Facing>,
}

/// A collapsible block of the properties panel.
///
/// The panel has more in it than fits a 1080 high window, and a scrollable
/// alone left every settings section below the fold — including the SpaceMouse
/// bindings, which made a deliberately rebindable puck practically
/// unrebindable. Collapsing keeps every heading reachable without a scroll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelSection {
    Bodies,
    Pattern,
    Pen,
    SpaceMouse,
    Detail,
    Export,
}

impl PanelSection {
    pub const ALL: [PanelSection; 6] = [
        PanelSection::Bodies,
        PanelSection::Pattern,
        PanelSection::Pen,
        PanelSection::SpaceMouse,
        PanelSection::Detail,
        PanelSection::Export,
    ];

    pub fn title(self) -> &'static str {
        match self {
            PanelSection::Bodies => "BODIES",
            PanelSection::Pattern => "PATTERN",
            PanelSection::Pen => "PEN",
            PanelSection::SpaceMouse => "SPACEMOUSE",
            PanelSection::Detail => "DETAIL",
            PanelSection::Export => "EXPORT",
        }
    }

    /// Whether it starts open. The long ones start closed so every heading is
    /// visible without scrolling.
    ///
    /// **DETAIL joined that list when BODIES arrived, and it is the sixth
    /// section that pays for it.** The budget is a 1080 high window with every
    /// heading reachable, and the handoff already reports DETAIL's advice line
    /// clipped below the fold at that height -- so it was the block already
    /// costing more than it showed.
    pub fn open_by_default(self) -> bool {
        !matches!(self, PanelSection::Pen | PanelSection::SpaceMouse | PanelSection::Detail)
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
    /// The pointer moved over the timeline strip, at this offset in pixels
    /// from its left edge.
    ///
    /// Carried separately from `Pointer` because it is a different widget with
    /// its own coordinate space, and because `mouse_area`'s press carries no
    /// position at all -- this is what a press then acts on.
    TimelineHover(f32),
    TimelinePressed,
    TimelineReleased,
    TimelineLeft,
    /// Remove the key under the pointer.
    TimelineRemoveKey,
    TimelinePlayToggled,
    /// The strip was laid out this many pixels wide.
    TimelineResized(f32),

    /// Export the sculpt to a staging file and open it in OrcaSlicer.
    OpenInSlicer,
    /// Ask the configured printer what it is doing.
    PrinterChecked,
    /// What it said.
    PrinterAnswered(Result<String, String>),

    /// Open the bug report dialog.
    BugReportOpened,
    /// The description editor did something.
    BugReportEdited(iced::widget::text_editor::Action),
    /// Include the session trail and diagnostics, or do not.
    BugReportDetailToggled(bool),
    /// Send it to TinkerAtlas.
    BugReportSubmitted,
    /// The send finished, with what to show for it.
    BugReportFinished(Result<String, String>),
    /// Put the exact payload on the clipboard instead of sending it.
    BugReportCopied,
    BugReportDismissed,
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
    /// A key was pressed and nothing in the widget tree wanted it.
    ///
    /// **Carries the key, not the message the key means.** The subscription
    /// that raises it cannot decide that itself: `iced::event::listen_with`
    /// takes a bare `fn` pointer (`iced_futures-0.14.0/src/event.rs:26`), so
    /// its callback captures nothing and can never see whether a modal card is
    /// up over the document. Decoding therefore happens in `Brokkr::on_key`,
    /// which can — and that is the one place the modal guard has to live.
    ///
    /// A press that spells nothing is still sent, and dropped there. The
    /// alternative, filtering in the subscription, would mean two places that
    /// know which keys exist.
    KeyPressed {
        key: iced::keyboard::Key,
        modifiers: iced::keyboard::Modifiers,
    },
    /// A left press that no widget in the tree wanted. **Nothing acts on it;
    /// it exists so that something arrives.**
    ///
    /// This is the application's only blur signal, and it is needed because
    /// `text_input` blurs itself without saying so: any left press that is not
    /// over its own bounds sets `state.is_focused = None`
    /// (`text_input.rs:723-735`) and does not capture, and no ancestor stops
    /// it -- `Row::update` forwards every event to every child regardless of
    /// capture (`row.rs:261-271`). The bodies list is a FIXED six or eight
    /// rows tall, so with a handful of bodies there is empty scrollable below
    /// the last row that belongs to no widget at all. A press there used to
    /// produce no message whatsoever, which left `Brokkr::renaming` set and
    /// the rename field DRAWN BUT DEAD: the user's next keystroke reached
    /// `key_event` as Ignored and fired a tool shortcut -- `s` starting a
    /// brush-radius drag, `x`/`y`/`z` flipping the mirror planes -- which is
    /// verbatim the class of bug `viewport::route_pointer`'s header records as
    /// fixed.
    ///
    /// Left button only, and that is not an oversight: `text_input` blurs on a
    /// LEFT press, so a right press leaves the field focused and must leave
    /// the rename alone. Presses that a widget did want never get here at all
    /// -- `key_event` drops captured events, and the field, a row's
    /// `mouse_area` and the viewport shader all capture their own.
    PressedNothing,
    /// Close the right-click menu.
    MenuClosed,
    /// Turn the model so the face the cube menu was opened on becomes this
    /// direction.
    OrientFace(brokkr_core::Facing),
    /// Answer the "this looks like it came in lying down" prompt. `false`
    /// leaves the model exactly as imported.
    OrientPromptAnswered(bool),
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
    /// Show the welcome screen, from the Help menu.
    WelcomeOpened,
    /// Dismiss it. Escape does this too.
    WelcomeClosed,
    /// The "show this on startup" tick, which is written through immediately:
    /// a preference that only lands when the dialog is dismissed the right way
    /// is one that silently forgets.
    WelcomeOnStartupSet(bool),
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
    /// Typing in the working-size field. Held as text so a half-typed number
    /// is not rounded or rejected mid-keystroke.
    WorkingSizeTyped(String),
    /// Commit the working-size field: scale the model so its longest dimension
    /// is that many millimetres. Free, and buys no detail -- see
    /// `Volume::rescale`.
    WorkingSizeCommitted,
    /// The title bar was pressed: start moving the window. The bar IS the
    /// title bar -- the window is undecorated -- so this is what replaces the
    /// one the compositor would otherwise have drawn.
    TitleBarDragged,
    /// Double-click on the title bar, which everywhere means maximise.
    TitleBarDoubleClicked,
    WindowMinimise,
    WindowClose,
    /// A press on one of the window's resize edges. Undecorated windows get no
    /// resize border from the compositor, so the application draws its own.
    ResizeStarted(iced::window::Direction),
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
    /// Show or hide the stats readout over the viewport. Collapsed it is one
    /// icon in the corner; open it is seven lines of numbers.
    StatsToggled,
    /// Choose what a left drag does. Pressing the live tool goes back to
    /// [`crate::app::Tool::Sculpt`], so this covers arming AND disarming and
    /// the cut costs no second variant.
    ToolChanged(Tool),

    // --- the body panel ------------------------------------------------------
    /// Make this body the one edits land on.
    BodySelected(brokkr_core::NodeId),
    /// Flip one row's own eye.
    ///
    /// Its OWN eye, never the resolved answer: an ancestor folder's eye and solo
    /// are masks applied on top, and a click here must not silently rewrite a
    /// bit the user cannot see.
    BodyVisibilityToggled(brokkr_core::NodeId),
    /// Flip the eye of whichever body is active, for `ctrl+comma`.
    ///
    /// A variant of its own rather than `BodyVisibilityToggled(active)`, because
    /// `viewport::shortcut` is a **pure decode** of a key and cannot see the
    /// document -- that separation is what lets every shortcut be tested without
    /// a window, and it is worth one message to keep.
    ActiveBodyVisibilityToggled,
    /// Turn every eye in the document on, for `ctrl+alt+comma`. The way out of
    /// having hidden something and lost track of what.
    EveryBodyShown,
    /// Open or close the small menu the `+` button drops.
    PrimitiveMenuToggled,
    /// Add a primitive as a NEW body, placed clear of everything already there.
    PrimitiveAdded(brokkr_core::PrimitiveKind),
    /// Remove the active body. May raise a prompt first; see
    /// `Brokkr::pending_delete`.
    BodyDeleted,
    /// Go ahead with a delete the prompt asked about.
    BodyDeleteConfirmed,
    BodyDeleteCancelled,
    /// Start renaming a row: a DOUBLE click on it, which is Photoshop's
    /// gesture and not the plan's single click.
    ///
    /// Single click had to go: the name cell is `Length::Fill` and therefore
    /// the largest target in the row, so a single click there is how a body
    /// gets selected. Giving the name its own `mouse_area` does not rescue it
    /// either -- `MouseArea::update` captures a left press whenever
    /// `on_double_click` is set, double or not (`mouse_area.rs:394-398`), so an
    /// inner area would eat the press the row's own selection depends on.
    BodyRenameBegan(brokkr_core::NodeId),
    /// A keystroke in the rename field, already clamped to what the file
    /// format's fixed name field can hold.
    BodyRenameEdited(String),
    /// Enter in the rename field. See `Brokkr::commit_rename` for where the
    /// commit actually happens, which is NOT this arm.
    BodyRenameSubmitted,
    /// Copy the active body, in place, as a new row directly below it.
    ///
    /// No payload: the active row is the subject, exactly as `BodyDeleted`'s is.
    /// A `NodeId` here would be a second answer to "which body" beside
    /// `Document::active`, and the two would disagree the first time a verb
    /// button was pressed while a stale id sat in a queued message.
    BodyDuplicated,
    /// Merge the active body down into the body directly below it.
    ///
    /// No payload, for the reason `BodyDuplicated` gives. May raise a prompt
    /// first; see `Brokkr::pending_merge`.
    BodyMergedDown,
    /// Go ahead with a merge the prompt asked about.
    BodyMergeConfirmed,
    BodyMergeCancelled,
    /// Split the active body into its loose parts.
    ///
    /// No payload, for the reason `BodyDuplicated` gives. Always raises the
    /// preview card first, because the number of parts is the whole of what
    /// there is to decide about; see `Brokkr::pending_split`.
    BodySplit,
    /// Go ahead with the split the preview card described.
    BodySplitConfirmed,
    BodySplitCancelled,
    /// `ctrl+G`: wrap the active row in a new folder, in place, no dialog.
    BodyGrouped,
    /// `ctrl+shift+G`: dissolve the folder the active row sits in.
    ///
    /// The PARENT and not the row, because the active row is always a body and
    /// a body has nothing to dissolve. That is also what makes the pair
    /// symmetric: ctrl+G then ctrl+shift+G leaves the document as it was.
    BodyUngrouped,
    /// The pointer moved over row `over` while a row is being dragged,
    /// `fraction` of the way down that row's height.
    ///
    /// **A fraction and not a pixel offset**, because which of a row's bands
    /// the pointer is in is the whole of the gesture and a band is a share of
    /// the row rather than a number of pixels -- the list is 32 px a row with
    /// pictures and 22 without. `mouse_area::on_move` hands over a point in the
    /// row's own space and the panel divides by the height it laid the row out
    /// at, which is why that height is `Length::Fixed` and not left to the
    /// content.
    ///
    /// Raised only while a drag is armed: the panel attaches no `on_move` at
    /// all otherwise, so an idle pointer over the list costs nothing.
    BodyRowDragged {
        over: usize,
        fraction: f32,
    },
    /// Fold a folder's children away in the panel, or show them again.
    ///
    /// Names the folder rather than acting on the active row: the chevron is
    /// drawn on the folder it belongs to, and a folder is never the active row.
    FolderCollapseToggled(brokkr_core::NodeId),
    /// Delete a folder and everything in it.
    ///
    /// **Deliberately not `BodyDeleted` with a folder in it.** The verb row's
    /// Delete names `Document::active`, which always holds a field, and this
    /// names a folder -- so no state of the panel, collapsed least of all, can
    /// make one of them do the other's job. In ZBrush it can, and a user
    /// reported losing an unrecoverable hour to it.
    FolderDeleted(brokkr_core::NodeId),
    /// Show only this row's subtree, and leave every eye alone.
    ///
    /// A view MODE and never a document change: it is `Option<NodeId>` on
    /// `Brokkr`, it is passed to `resolve_visibility` as a parameter, and it is
    /// written nowhere. Entering it on a row that is hidden is the one exception
    /// — that turns the row's own eye and its ancestors' back on, because
    /// "show me only this" with nothing on screen is not a mode anyone asked
    /// for — and *that* part is an ordinary undoable edit.
    SoloEntered(brokkr_core::NodeId),
    /// Leave solo. Escape, `ctrl+alt+comma`, and the header indicator's own
    /// exit all send this.
    ///
    /// It restores nothing, because it changed nothing: every hand-set eye is
    /// exactly where the user left it. That is the whole reason solo is a mode
    /// rather than a saved-visibility vector — every shipped version of the
    /// vector design loses the hand-set set on the way out.
    SoloExited,
    /// Show or hide the thumbnail column. Session state, so it must NOT dirty
    /// the document: nothing about it is written to the file.
    ThumbnailsToggled,
    /// Show or hide the mask's tint on the model.
    ///
    /// **It governs the TINT and nothing else.** The standing mask card is
    /// unconditional, so switching this off cannot reach the state "a mask is
    /// active and nothing on screen says so". It is view state, so it must not
    /// dirty the document, and it changes no protection and marks no brick
    /// dirty: the polarity and the strength are uniforms the shader reads.
    ShowMaskToggled,
    /// How strongly protection is tinted, 0..1.
    ///
    /// A view strength and never a protection strength — see
    /// [`brokkr_gpu::Uniforms::mask_tint`].
    MaskTintChanged(f32),
    /// The held peek key went down, or came back up.
    ///
    /// Kept alongside the toggle rather than replaced by it: it is the faster
    /// gesture for a momentary look, and the way back for a user who switched
    /// the tint off and forgot. Two messages rather than one toggle because the
    /// key repeats while it is held, and a toggle would strobe.
    MaskPeekStarted,
    MaskPeekEnded,
    /// Put the camera back on the active body, keeping the angles.
    ///
    /// The recovery half of a free camera. A camera that can go anywhere can
    /// end up anywhere, and the answer is one keystroke back rather than a
    /// fence around where it may go.
    CameraFramedOnActive,
    /// Throw the active body's mask away. The map is MOVED into the history
    /// entry, so this allocates nothing however large the mask was.
    MaskCleared,
    /// Flip which side of the mask is protected. One bool, no bricks, no bytes.
    MaskInverted,
    /// Protect the whole body: clear, then invert, as one change.
    MaskAllApplied,
    /// Which of the four absolute filters the amount slider drives.
    MaskFilterChosen(MaskFilter),
    /// The amount slider moved, to a fraction in `0..=1`.
    ///
    /// **Absolute, and re-applied from the mask as it stood when the drag
    /// began** -- never accumulated. ZBrush's BlurMask is "press repeatedly for
    /// progressively more blur" and that is its top masking complaint; Maxon's
    /// own later fix is documented as "absolute rather than accumulative".
    MaskAmountChanged(f32),
    /// The amount slider was let go, which commits the whole drag as ONE entry.
    MaskAmountReleased,
    /// Build a whole mask out of the active body's geometry.
    ///
    /// The parameters are NOT carried here: they are the two sliders beside the
    /// buttons, so a press means "with what is on screen" and the message stays
    /// the same shape however many recipes there turn out to be.
    MaskGenerated(MaskGenerator),
    /// The feature-size slider moved, in MILLIMETRES.
    ///
    /// Millimetres and never voxels, which is where this beats ZBrush outright
    /// rather than matching it: its cavity masking is resolution-relative, so
    /// the same model at a different subdivision masks differently. A body here
    /// has one fixed voxel size, so "narrower than 1.5 mm" is stable and means
    /// something to somebody choosing a nozzle.
    MaskFeatureChanged(f32),
    /// The thickness slider moved, in VOXELS.
    ///
    /// Voxels and not millimetres, and that is the opposite choice from the one
    /// above for a real reason: the ceiling is
    /// [`brokkr_core::MAX_THICKNESS_VOXELS`], which is a property of the narrow
    /// band and not of the model, so a millimetre slider would have a maximum
    /// that moved every time the body was resampled. The millimetres are shown
    /// beside the number.
    MaskThicknessChanged(f32),
    /// Split the active body in two along its mask.
    ///
    /// No preview card, unlike [`Message::BodySplit`]: a masked split always
    /// makes exactly two bodies, so there is nothing a card could tell the user
    /// that they do not already see in the tint.
    BodySplitMasked,
}

/// Which mask a generator button asked for.
///
/// Three and not four: the half-space is a DRAG and not a button, because it
/// needs a plane and the cut tool already has the gesture that makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskGenerator {
    /// Concave detail narrower than the feature slider.
    Cavity,
    /// Everything flat at that scale: the same pass read the other way.
    Smoothness,
    /// Solid material thinner than the thickness slider.
    Thickness,
}

impl MaskGenerator {
    pub const ALL: [MaskGenerator; 3] =
        [MaskGenerator::Cavity, MaskGenerator::Smoothness, MaskGenerator::Thickness];

    pub fn label(self) -> &'static str {
        match self {
            MaskGenerator::Cavity => "Cavity",
            MaskGenerator::Smoothness => "Smooth",
            MaskGenerator::Thickness => "Thin",
        }
    }
}
