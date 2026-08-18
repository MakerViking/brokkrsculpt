// SPDX-License-Identifier: AGPL-3.0-or-later

//! Application state, input handling and the widget tree.

mod panel;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use brokkr_core::{
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, History, HistoryStats,
    MirrorAxis, Stamp, Stroke, Symmetry, Volume, VolumeStats, lean_normal, raycast,
};
use glam::{Vec2, Vec3};
use iced::{Subscription, Task};

use crate::camera::OrbitCamera;
use crate::cursor;
use crate::message::{
    ConfirmChoice, ExportFormat, Message, PanelSection, PointerButton, PointerEvent,
    SpaceMouseSetting, TopMenu,
};
use crate::navcube;
use crate::spacemouse::{
    Action as PuckAction, AxisBinding, ButtonAction, Config as SpaceMouseConfig, SpaceMouse,
};
use crate::tablet::Tablet;
use crate::viewport::SharedFrame;

/// World units are millimetres, because the output of this program is meant to
/// be printed.
///
/// A 60 mm ball at a quarter millimetre voxel is 240 voxels across, which is
/// the 256 cubed effective volume the milestones are measured against.
const MODEL_RADIUS_MM: f32 = 30.0;
const VOXEL_SIZE_MM: f32 = 0.25;

/// Range the voxel size may be resampled to, in millimetres.
///
/// The lower bound is where the mesh pool has been measured to hold the result:
/// a 60 mm ball at 0.055 mm comes to 11.2 million triangles and 6.2 million
/// vertices against a pool of 8 million. Going finer would overflow it and put
/// an incomplete model on screen, so the interface will not offer it.
/// Range the brush radius may be nudged to with the keyboard, in millimetres.
/// The same range the slider offers, so the two cannot disagree.
pub(crate) const MIN_RADIUS_MM: f32 = 0.25;
pub(crate) const MAX_RADIUS_MM: f32 = 12.0;

/// Range the brush strength may take, matching the slider.
pub(crate) const MIN_STRENGTH: f32 = 0.02;
pub(crate) const MAX_STRENGTH: f32 = 0.80;

/// How fast a hold-and-drag moves the brush numbers.
///
/// Radius is in log space so the same drag is the same proportion at any size;
/// about 250 px covers the whole range, which is a comfortable sweep.
const RADIUS_PER_PIXEL: f32 = 0.006;
const STRENGTH_PER_PIXEL: f32 = 0.002;

const FINEST_VOXEL_MM: f32 = 0.06;
const COARSEST_VOXEL_MM: f32 = 2.0;

/// Largest angle a fully tilted pen steers the stroke by.
///
/// Tablets report tilt against their own range, so this is applied to the
/// normalised value rather than trusting a device to report degrees. Sixty
/// degrees is about as far as a pen can be leaned while still drawing.
const MAX_TILT: f32 = std::f32::consts::PI / 3.0;

/// Frame intervals kept for the rate readout. At 60 fps this averages over
/// about a second, which is long enough to be steady and short enough to react.
const FRAME_HISTORY: usize = 60;

/// How close a flown-to pitch may come to straight up or down.
///
/// The camera clamps this itself, but a flight interpolates the field directly,
/// so it has to respect the same limit or a top view would collapse the view
/// matrix on arrival.
const PITCH_SAFE: f32 = std::f32::consts::FRAC_PI_2 - 0.02;

/// How far the pointer may travel during a right press and still count as a
/// click rather than an orbit.
///
/// Right drag orbits, so this is the whole difficulty of putting a menu on right
/// click: too tight and a click that wobbles a pixel opens nothing, too loose
/// and a small deliberate orbit opens a menu over the model. Four pixels and a
/// quarter second are a starting point to tune by feel, which is why both are
/// named.
const CLICK_SLOP_PX: f32 = 4.0;
const CLICK_MS: u128 = 250;

/// What a held pointer button is currently doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragKind {
    Orbit,
    Pan,
    Sculpt(BrushDirection),
    /// The pointer is resizing the brush, not using it.
    Sizing,
    /// The pointer is dragging the line of a plane cut. Nothing happens to the
    /// model until the button comes back up, so the line can be adjusted.
    Cutting,
}

/// A drag in progress, tagged with the button that started it so that
/// releasing a different button does not cancel it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Drag {
    button: PointerButton,
    kind: DragKind,
    /// Where the button went down, and when.
    origin: Vec2,
    pressed_at: Instant,
    /// Whether the pointer has travelled far enough to count as a drag rather
    /// than a click. Once true it stays true: a drag that returns to where it
    /// started is still a drag.
    moved: bool,
}

/// Timings for the debug overlay.
#[derive(Debug, Default)]
struct Perf {
    last_frame: Option<Instant>,
    frame_ms: VecDeque<f32>,
    edit_ms: f32,
    remesh_ms: f32,
    dirty_bricks: usize,
    stamps: usize,
    /// Pressure the last stroke step ran at, so the overlay can show a pen
    /// working without the user having to guess.
    pressure: f32,
    /// Cost of the one off full mesh at load. Kept apart from the per stroke
    /// numbers so a 70 ms load does not sit in the slot that is supposed to
    /// show an 8 ms budget.
    load_ms: f32,
}

impl Perf {
    /// Record a presented frame and return how long it took, in milliseconds.
    ///
    /// The SpaceMouse scales its motion by this, so it reads the same clock
    /// the overlay does rather than keeping a second one that could disagree.
    fn record_frame(&mut self) -> f32 {
        let now = Instant::now();
        let Some(previous) = self.last_frame.replace(now) else {
            return 0.0;
        };
        let elapsed_ms = now.duration_since(previous).as_secs_f32() * 1000.0;
        if self.frame_ms.len() == FRAME_HISTORY {
            self.frame_ms.pop_front();
        }
        self.frame_ms.push_back(elapsed_ms);
        elapsed_ms
    }

    fn average_frame_ms(&self) -> f32 {
        if self.frame_ms.is_empty() {
            return 0.0;
        }
        self.frame_ms.iter().sum::<f32>() / self.frame_ms.len() as f32
    }

    fn worst_frame_ms(&self) -> f32 {
        self.frame_ms.iter().copied().fold(0.0, f32::max)
    }
}

pub struct Brokkr {
    volume: Volume,
    camera: OrbitCamera,
    brush: Brush,
    symmetry: Symmetry,
    tablet: Tablet,
    spacemouse: SpaceMouse,
    /// Whether stylus pressure scales the brush. Off means every stamp runs at
    /// full strength, which is also what happens when there is no pen.
    pressure_enabled: bool,
    /// Exponent applied to raw pressure. Below 1 makes light touches bite
    /// harder, above 1 gives finer control at the light end.
    pressure_curve: f32,
    /// Whether leaning the pen steers the stroke.
    tilt_enabled: bool,
    stroke: Stroke,
    history: History,
    shared: Arc<SharedFrame>,
    /// Buffers the dirty bricks are meshed into, refilled from the renderer's
    /// recycled ones so a stroke never allocates.
    mesh_buffers: Vec<BrickMesh>,
    brush_scratch: BrushScratch,
    /// Stamp centres produced by the current pointer event. Reused so a stroke
    /// does not allocate.
    stamp_centres: Vec<Vec3>,
    dirty: Vec<BrickCoord>,
    drag: Option<Drag>,
    /// Last pointer position in widget pixels, for drag deltas.
    cursor: Option<Vec2>,
    viewport_size: Vec2,
    shift: bool,
    control: bool,
    /// Alt, which inverts the brush the same way control does.
    alt: bool,
    perf: Perf,
    volume_stats: VolumeStats,
    history_stats: HistoryStats,
    /// Current voxel size, which the resample operation changes.
    voxel_size: f32,
    /// What the last export or resample did, for the interface to show.
    status: String,
    /// Which blocks of the properties panel are open, in `PanelSection::ALL`
    /// order.
    expanded: [bool; PanelSection::ALL.len()],
    /// The brush ring and mirror planes, handed to the renderer through
    /// `SharedFrame`. Held here and swapped rather than rebuilt into a fresh
    /// allocation each time.
    overlay: brokkr_gpu::OverlayBatch,
    /// Where the pointer last met the surface, which is where the ring goes.
    hover: Option<Vec3>,
    /// A hold-and-drag brush resize in progress.
    sizing: Option<Sizing>,
    /// Whether the radius tracks the model's size rather than staying a fixed
    /// number of millimetres.
    dynamic_radius: bool,
    /// The model's bounding radius, refreshed on remesh. `Volume::bounding_radius`
    /// walks the brick map, so it is not something to call per frame.
    model_radius: f32,
    /// The navigation cube's geometry, swapped to the renderer the same way.
    cube: brokkr_gpu::OverlayBatch,
    /// The cube part under the pointer, lit so a click's effect is visible
    /// before the click.
    cube_hover: Option<navcube::Part>,
    /// A camera move in progress after clicking the cube.
    flight: Option<Flight>,
    /// Where the right-click menu is open, in widget pixels.
    menu: Option<Vec2>,
    /// Which top bar menu is open, if any.
    pub(crate) top_menu: Option<TopMenu>,
    /// The file this sculpt was opened from or last saved to. `None` until it
    /// has one, which is what makes the first Save behave as Save As.
    project_path: Option<std::path::PathBuf>,
    /// Whether the field has changed since it was last written to a file.
    ///
    /// Named `unsaved` rather than `dirty` on purpose: `Brokkr::dirty` is the
    /// remesh scratch list and `Volume` has its own `mark_dirty`/`take_dirty`
    /// vocabulary, so a second meaning for the word would cost a reader every
    /// time.
    ///
    /// Deliberately tracks the *field* only, not the camera, brush or mirror
    /// state that a `.brokkr` file also carries. `rescale_radius` mutates
    /// `brush.radius` from inside `remesh_dirty`, which runs on every stroke,
    /// undo, redo, resample, open and reset, so a flag covering brush state
    /// would be set permanently and mean nothing. A save still records the
    /// newer camera; losing an unsaved camera nudge is not worth a prompt.
    unsaved: bool,
    /// An action waiting on the user's answer to "you have unsaved work".
    ///
    /// Its own field rather than a variant of `menu`, because the two dismissal
    /// routes that close a menu -- a press in the viewport and
    /// `Message::MenuClosed` -- must NOT close this. A prompt that a stray click
    /// dismisses is worse than no prompt: the user learns to ignore it.
    confirm: Option<PendingAction>,
    /// Whether the next left drag cuts the model instead of sculpting it.
    ///
    /// A mode rather than a modifier because a cut is destructive and
    /// irreversible-looking: arming it deliberately is worth one click, and the
    /// tool strip shows the armed state so it can never be a surprise.
    cut_armed: bool,
    /// Where a cut line started, while it is being dragged.
    cut_from: Option<Vec2>,
    /// Sculpts opened or saved recently, for the File menu.
    recent: crate::recent::Recent,
    /// Where the crash net is written.
    ///
    /// Carried rather than recomputed from the environment so a test can point
    /// it at a temporary directory. Without that, running the suite would
    /// delete a real autosave belonging to a real session.
    autosave_file: Option<std::path::PathBuf>,
    /// When the crash net was last written. See [`Brokkr::maybe_autosave`].
    last_autosave: Instant,
    /// A numeric field in the menu being typed into.
    ///
    /// Held as text rather than parsed on every keystroke, because a half typed
    /// value like `2.` does not parse and snapping the field back to the old
    /// number mid-edit makes it unusable.
    menu_edit: Option<(SizingTarget, String)>,
}

/// An animated camera move to an orientation.
///
/// Instant would be disorienting on a sculpt: with nothing but a shaded form to
/// go on there is no way to tell a jump from a different model. A quarter second
/// is enough to follow and short enough not to be in the way.
#[derive(Debug, Clone, Copy)]
struct Flight {
    from: (f32, f32, f32),
    to: (f32, f32, f32),
    elapsed_ms: f32,
}

/// How long a click on the navigation cube takes to arrive.
const FLIGHT_MS: f32 = 260.0;

/// Something the user asked for that would discard unsaved work, held until
/// they say what to do about it.
///
/// Each of these throws the current field away: New reseeds the sphere, Open
/// swaps a file in, and Quit ends the process. They are the complete set of
/// ways work can be lost, which is why the gate lives in one place.
///
/// `Clone` rather than `Copy` because one variant carries a path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PendingAction {
    NewSculpt,
    /// Open, by way of the file dialog.
    Open,
    /// Open a specific file, chosen from the recent list, so no dialog.
    OpenRecent(std::path::PathBuf),
    /// Load the crash net. Its own variant rather than an `OpenRecent` of the
    /// autosave path, because the document that comes back must NOT be owned by
    /// that path -- see [`Brokkr::recover_autosave`].
    RecoverAutosave,
    /// Import a mesh. Discards the document exactly as Open does, so it goes
    /// through the same gate.
    Import,
    /// Carries the window, because with `exit_on_close_request(false)` the
    /// close has to be issued by us against that specific window.
    Quit(iced::window::Id),
}

impl PendingAction {
    /// What the prompt says is about to happen.
    pub(crate) fn describe(&self) -> &'static str {
        match self {
            PendingAction::NewSculpt => "Starting a new sculpt",
            PendingAction::Open | PendingAction::OpenRecent(_) => "Opening another file",
            PendingAction::RecoverAutosave => "Recovering the autosave",
            PendingAction::Import => "Importing a mesh",
            PendingAction::Quit(_) => "Quitting",
        }
    }
}

/// Where bug reports go.
pub const ISSUE_URL: &str = "https://github.com/MakerViking/brokkrsculpt/issues/new";

/// The commit this was built from, when the build system supplied one.
///
/// Shown in About, and not only as a nicety: shipping binaries carries an AGPL
/// obligation to offer the corresponding source, and a commit is what ties an
/// artifact to it. `unknown` in a local build, which is honest.
pub fn build_commit() -> &'static str {
    option_env!("BROKKR_COMMIT").unwrap_or("unknown")
}

/// The extension a sculpt is saved with.
pub const PROJECT_EXTENSION: &str = "brokkr";

/// Ask for a sculpt to open.
///
/// Async, and driven through a `Task`, because the portal dialog is a round trip
/// to another process: the blocking form would stall the event loop, which is
/// the same thread that draws.
///
/// The dialog opens unparented. iced 0.14 gives no way to hand `rfd` a window
/// handle -- the one route, `iced::window::run`, hands back a borrowed handle on
/// the event loop thread and its return type must be `Send`, so it cannot reach
/// an async task. In practice the portal still centres the dialog on the right
/// screen; verify that rather than assume it.
async fn pick_project_to_open() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Open sculpt")
        .add_filter("BrokkrSculpt", &[PROJECT_EXTENSION])
        .pick_file()
        .await
        .map(|handle| handle.path().to_path_buf())
}

async fn pick_project_to_save() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Save sculpt")
        .add_filter("BrokkrSculpt", &[PROJECT_EXTENSION])
        .set_file_name(format!("sculpt.{PROJECT_EXTENSION}"))
        .save_file()
        .await
        .map(|handle| {
            let path = handle.path().to_path_buf();
            // A portal can hand back a name with no extension. Adding it keeps
            // the file recognisable to the open dialog's own filter.
            if path.extension().is_some() { path } else { path.with_extension(PROJECT_EXTENSION) }
        })
}

async fn pick_export_target(format: ExportFormat) -> Option<std::path::PathBuf> {
    let extension = format.extension();
    rfd::AsyncFileDialog::new()
        .set_title(format!("Export {}", format.label()))
        .add_filter(format.label(), &[extension])
        .set_file_name(format!("sculpt.{extension}"))
        .save_file()
        .await
        .map(|handle| {
            let path = handle.path().to_path_buf();
            if path.extension().is_some() { path } else { path.with_extension(extension) }
        })
}

/// Ask for a mesh to import.
///
/// `pick_file`, so unlike the two save dialogs there is no extension fixup: the
/// file already exists and its name is not ours to correct.
async fn pick_mesh_to_import() -> Option<std::path::PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Import mesh")
        .add_filter("Mesh", &brokkr_core::MESH_EXTENSIONS)
        .pick_file()
        .await
        .map(|handle| handle.path().to_path_buf())
}

/// A hold-and-drag gesture adjusting one brush number.
///
/// Holds where the pointer started and what the value was, so the gesture is
/// absolute: a drag out and back returns to exactly where it began, where
/// accumulating per-event deltas would drift.
#[derive(Debug, Clone, Copy)]
struct Sizing {
    what: SizingTarget,
    from_pixel: Vec2,
    original: f32,
}

/// Which number a sizing drag is moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizingTarget {
    Radius,
    Strength,
}

impl Brokkr {
    pub fn new() -> Self {
        Self::with_devices(Tablet::start(), SpaceMouse::start())
    }

    /// Build with a given pressure source and an inert puck, which is what
    /// almost every test wants.
    ///
    /// Points the recent list and the crash net at a per-test temporary
    /// directory. This is not tidiness: the defaults are the real
    /// `~/.config/brokkrsculpt/recent` and the real autosave, and `save_project`
    /// both records into one and deletes the other -- so a plain `cargo test`
    /// would rewrite someone's recent list and destroy a crash net belonging to
    /// a live session.
    #[cfg(test)]
    fn with_tablet(tablet: Tablet) -> Self {
        let mut app = Self::with_devices(tablet, SpaceMouse::inert());
        let scratch = std::env::temp_dir().join(format!(
            "brokkr-test-state-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        app.recent = crate::recent::Recent::load_from(Some(scratch.join("recent")));
        app.autosave_file = Some(scratch.join("autosave.brokkr"));
        // `with_devices` has already looked for a crash net, and at that point
        // it was still looking at the real one -- so on a machine that has an
        // autosave the status line would arrive pre-filled and every test
        // asserting on it would fail for a reason outside its own scope.
        app.status = String::new();
        app
    }

    /// Build with given input devices, so tests do not go looking through
    /// `/dev/input` and spawn reader threads.
    fn with_devices(tablet: Tablet, spacemouse: SpaceMouse) -> Self {
        let shared = SharedFrame::new();
        let mut volume = Volume::new(VOXEL_SIZE_MM);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM);
        // Everything the sphere touches plus a one brick margin, because bricks
        // with no voxels of their own still own the quads on their low faces.
        volume.mark_everything_dirty();

        let mut app = Self {
            volume,
            camera: OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM),
            brush: Brush::default(),
            symmetry: Symmetry::OFF,
            tablet,
            spacemouse,
            pressure_enabled: true,
            pressure_curve: 1.0,
            tilt_enabled: true,
            stroke: Stroke::new(),
            history: History::default(),
            shared,
            mesh_buffers: Vec::new(),
            brush_scratch: BrushScratch::new(),
            stamp_centres: Vec::new(),
            dirty: Vec::new(),
            drag: None,
            cursor: None,
            viewport_size: Vec2::new(1280.0, 720.0),
            shift: false,
            control: false,
            alt: false,
            perf: Perf::default(),
            volume_stats: VolumeStats::default(),
            history_stats: HistoryStats::default(),
            voxel_size: VOXEL_SIZE_MM,
            status: String::new(),
            expanded: PanelSection::ALL.map(PanelSection::open_by_default),
            overlay: brokkr_gpu::OverlayBatch::default(),
            hover: None,
            sizing: None,
            dynamic_radius: false,
            model_radius: MODEL_RADIUS_MM,
            cube: brokkr_gpu::OverlayBatch::default(),
            cube_hover: None,
            flight: None,
            menu: None,
            top_menu: None,
            project_path: None,
            unsaved: false,
            confirm: None,
            cut_armed: false,
            cut_from: None,
            recent: crate::recent::Recent::load(),
            autosave_file: Self::default_autosave_path(),
            last_autosave: Instant::now(),
            menu_edit: None,
        };
        app.remesh_dirty();
        // Otherwise the overlay reports a zero byte budget until the first
        // stroke happens to refresh it.
        app.history_stats = app.history.stats();
        app.perf.load_ms = app.perf.remesh_ms;
        app.perf.remesh_ms = 0.0;
        app.perf.dirty_bricks = 0;
        app.publish_camera();
        app.refresh_overlay();
        // A crash net left by a previous session is worth nothing if nobody
        // knows it is there, and the File menu is not somewhere you look
        // unprompted.
        if app.has_autosave() {
            app.status = "an autosave from a previous session is in File > Recover".to_string();
        }
        app
    }

    /// What the sculpt is called, for the window title and the header.
    ///
    /// One function so the two cannot drift: the header used to derive this
    /// itself.
    pub(crate) fn document_name(&self) -> String {
        match &self.project_path {
            Some(path) => path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            None => "untitled".to_string(),
        }
    }

    /// The window title, which is where the unsaved marker is most visible.
    ///
    /// iced re-evaluates this every frame through `program::with_title`, so the
    /// star appears the moment a stroke lands with nothing else to wire.
    pub fn title(&self) -> String {
        let star = if self.unsaved { "*" } else { "" };
        format!("{}{star} — BrokkrSculpt", self.document_name())
    }

    pub fn subscription(&self) -> Subscription<Message> {
        Subscription::batch([
            // Drives the frame rate readout and keeps the viewport presenting
            // while a stroke is in flight.
            iced::window::frames().map(|_| Message::Frame),
            // The window manager's close button. `main` sets
            // `exit_on_close_request(false)`, so this is the ONLY thing that
            // ends the application -- if this subscription is ever dropped the
            // window becomes unclosable.
            iced::window::close_requests().map(Message::CloseRequested),
        ])
    }

    /// Run an action that discards the document, now that it has been allowed.
    fn run_pending(&mut self, action: PendingAction) -> Task<Message> {
        match action {
            PendingAction::NewSculpt => {
                self.reset_sculpt();
                Task::none()
            }
            PendingAction::Open => Task::perform(pick_project_to_open(), Message::OpenChosen),
            PendingAction::OpenRecent(path) => {
                self.open_project(&path);
                Task::none()
            }
            PendingAction::RecoverAutosave => {
                self.recover_autosave();
                Task::none()
            }
            PendingAction::Import => Task::perform(pick_mesh_to_import(), Message::ImportChosen),
            // Destroying the last window is what ends a non-daemon iced
            // application (`iced_winit` sends `Control::Exit` once the window
            // manager is empty), so this really does quit.
            PendingAction::Quit(id) => iced::window::close(id),
        }
    }

    /// Do `action`, or ask first if there is unsaved work to lose.
    fn guard(&mut self, action: PendingAction) -> Task<Message> {
        if self.unsaved {
            // The prompt is about to draw over the viewport, and a menu left
            // open would sit on top of it.
            self.top_menu = None;
            self.menu = None;
            self.confirm = Some(action);
            return Task::none();
        }
        self.run_pending(action)
    }

    /// Act on the answer to the unsaved-work prompt.
    fn answer_confirm(&mut self, choice: ConfirmChoice) -> Task<Message> {
        match choice {
            ConfirmChoice::Cancel => {
                self.confirm = None;
                Task::none()
            }
            ConfirmChoice::Discard => match self.confirm.take() {
                Some(action) => self.run_pending(action),
                None => Task::none(),
            },
            ConfirmChoice::Save => match self.project_path.clone() {
                // It has a file already, so this is a plain write and the
                // answer is known immediately.
                Some(path) => {
                    self.save_project(&path);
                    if self.unsaved {
                        // Failed. `status` says why; the prompt stays up so the
                        // user can still choose to discard or cancel rather
                        // than losing the work to a write that did not happen.
                        return Task::none();
                    }
                    match self.confirm.take() {
                        Some(action) => self.run_pending(action),
                        None => Task::none(),
                    }
                }
                // No file yet, so ask for one. The prompt stays up until the
                // dialog comes back, which is what `SavedThenContinue` settles.
                None => Task::perform(pick_project_to_save(), Message::SavedThenContinue),
            },
        }
    }

    fn publish_camera(&mut self) {
        self.shared.set_camera(self.camera);
        // The cube shows the camera's orientation, so it is stale the moment the
        // camera moves. This is the one place that knows that happened.
        self.refresh_cube();
    }

    /// Rebuild the navigation cube and hand it to the renderer.
    fn refresh_cube(&mut self) {
        let mut batch = std::mem::take(&mut self.cube);
        navcube::build(&mut batch, &self.camera, self.cube_hover);
        self.shared.swap_cube(&mut batch);
        self.cube = batch;
    }

    /// Start an animated move to a cube part's orientation.
    fn fly_to(&mut self, part: navcube::Part) {
        let (yaw, pitch) = navcube::orientation(part.direction, self.camera.yaw);
        // Clamped the way a drag would be, so a top or bottom face lands just
        // short of the pole rather than collapsing the view matrix.
        let pitch = pitch.clamp(-PITCH_SAFE, PITCH_SAFE);
        // The shortest way round, so a click never spins the model several times
        // to reach a heading a few degrees away.
        let yaw = self.camera.yaw + OrbitCamera::shortest_angle_delta(self.camera.yaw, yaw);
        self.flight = Some(Flight {
            from: (self.camera.yaw, self.camera.pitch, self.camera.roll),
            to: (yaw, pitch, 0.0),
            elapsed_ms: 0.0,
        });
    }

    /// Advance a camera flight. Returns whether the camera moved.
    fn advance_flight(&mut self, elapsed_ms: f32) -> bool {
        let Some(mut flight) = self.flight else {
            return false;
        };
        flight.elapsed_ms += elapsed_ms.clamp(0.0, 50.0);
        let t = (flight.elapsed_ms / FLIGHT_MS).clamp(0.0, 1.0);
        // Smoothstep, so it eases out of the old view and into the new one
        // rather than starting and stopping abruptly.
        let eased = t * t * (3.0 - 2.0 * t);
        let lerp = |from: f32, to: f32| from + (to - from) * eased;

        self.camera.yaw = lerp(flight.from.0, flight.to.0);
        self.camera.pitch = lerp(flight.from.1, flight.to.1);
        self.camera.roll = lerp(flight.from.2, flight.to.2);

        self.flight = (t < 1.0).then_some(flight);
        true
    }

    /// Mesh every brick the volume has marked dirty and hand the results to the
    /// renderer. Never touches a brick that was not marked.
    fn remesh_dirty(&mut self) {
        self.volume.take_dirty(&mut self.dirty);
        self.perf.dirty_bricks = self.dirty.len();
        if self.dirty.is_empty() {
            return;
        }

        let started = Instant::now();
        let count = self.dirty.len();
        while self.mesh_buffers.len() < count {
            self.mesh_buffers.push(self.shared.take_mesh());
        }
        // Across every core. At the sizes M2 targets this is the difference
        // between a remesh at 70 percent of its budget and one at under 10.
        self.volume.mesh_bricks(&self.dirty, &mut self.mesh_buffers[..count]);
        for (coord, mesh) in self.dirty.iter().zip(self.mesh_buffers.drain(..count)) {
            self.shared.publish(*coord, mesh);
        }
        self.perf.remesh_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.volume_stats = self.volume.stats();

        let previous = self.model_radius;
        self.model_radius = self.volume.bounding_radius().unwrap_or(MODEL_RADIUS_MM);
        self.rescale_radius(previous);
    }

    /// Rebuild the brush ring and mirror planes and hand them to the renderer.
    ///
    /// Called from the places that change what it looks like -- pointer motion,
    /// a brush or mirror setting, a resample -- and deliberately NOT from the
    /// frame handler: the geometry is world space, so a camera moving under it
    /// needs no rebuild, and a raycast per frame would be waste.
    fn refresh_overlay(&mut self) {
        // Moved out and back rather than borrowed in place: `build` also needs
        // the volume and the brush, which are fields of the same `self`. The
        // two buffers then rotate -- this frame's goes to the renderer and last
        // frame's comes back with its capacity -- so nothing allocates once
        // warm.
        let mut batch = std::mem::take(&mut self.overlay);
        cursor::build(
            &mut batch,
            &self.volume,
            &self.effective_brush(),
            self.symmetry,
            self.hover,
            cursor::mood(self.stroke_direction(), self.sizing.is_some()),
            self.model_radius,
        );
        self.shared.swap_overlay(&mut batch);
        self.overlay = batch;
    }

    /// Where the pointer meets the surface, remembered for the cursor ring.
    fn update_hover(&mut self, pixel: Vec2) {
        self.hover = self.surface_under(pixel);
    }

    /// Keep the brush a constant fraction of the model when asked to.
    ///
    /// ZBrush calls this Dynamic. Without it a brush tuned on a 60 mm ball is
    /// the wrong size the moment the model is resampled or grows.
    fn rescale_radius(&mut self, previous_model_radius: f32) {
        if !self.dynamic_radius || previous_model_radius <= 0.0 || self.model_radius <= 0.0 {
            return;
        }
        let factor = self.model_radius / previous_model_radius;
        if (factor - 1.0).abs() > 1.0e-4 {
            self.brush.radius = (self.brush.radius * factor).clamp(MIN_RADIUS_MM, MAX_RADIUS_MM);
        }
    }

    /// The world space ray through a point in widget pixels.
    fn ray_through(&self, pixel: Vec2) -> (Vec3, Vec3) {
        let aspect = self.viewport_size.x / self.viewport_size.y.max(1.0);
        let ndc = OrbitCamera::ndc_from_pixels(pixel, self.viewport_size);
        self.camera.ray(ndc, aspect)
    }

    /// Where the cursor meets the surface, if it does.
    fn surface_under(&self, pixel: Vec2) -> Option<Vec3> {
        let (origin, ray) = self.ray_through(pixel);
        raycast(&self.volume, origin, ray, self.camera.far).map(|hit| hit.position)
    }

    /// Apply the brush along the stroke path up to the point under the cursor.
    ///
    /// The stroke walks from its previous stamp to the new one at a fixed
    /// spacing, so a fast drag lays a continuous cut instead of a dotted trail.
    /// The stamps are applied one after another rather than batched, because
    /// each one has to see the field the previous one left behind.
    fn sculpt_to(&mut self, pixel: Vec2, direction: BrushDirection, start: bool) {
        let Some(point) = self.surface_under(pixel) else {
            // The cursor ran off the model. The stroke stays live so coming
            // back onto it continues rather than restarting, but nothing is
            // stamped in mid air.
            return;
        };

        let started = Instant::now();
        self.stamp_centres.clear();
        if start {
            self.stroke.begin(point, &mut self.stamp_centres);
        } else {
            let spacing = self.effective_brush().spacing(self.volume.voxel_size());
            self.stroke.advance(point, spacing, &mut self.stamp_centres);
        }

        // Sampled once for the whole event rather than per stamp: the pen has
        // not moved between the stamps that one pointer event interpolates, so
        // re-reading it would only add jitter.
        let pressure = self.tablet.stamp_pressure(self.pressure_enabled, self.pressure_curve);
        let lean = self.pen_lean();
        let brush = self.effective_brush();
        // Which way the drag is going, for the patterns that comb. Zero until
        // the stroke has moved far enough to have a direction, which the
        // pattern copes with by picking any direction across the surface.
        let tangent = self.stroke.direction().unwrap_or(Vec3::ZERO);

        for index in 0..self.stamp_centres.len() {
            let centre = self.stamp_centres[index];
            // Take the normal from the field at each stamp rather than reusing
            // the one from the raycast, so a stroke curving around a form stays
            // oriented to the surface it is actually on.
            // Leaning the pen rotates the direction the brush pushes in, which
            // steers every brush at once because they all read this normal.
            let normal = lean_normal(self.volume.gradient_world(centre), lean);
            let stamp =
                Stamp::new(centre, normal, direction).with_pressure(pressure).with_tangent(tangent);
            brush.apply_symmetric(&mut self.volume, &stamp, self.symmetry, &mut self.brush_scratch);
        }
        self.perf.stamps = self.stamp_centres.len();
        self.perf.pressure = pressure;
        self.perf.edit_ms = started.elapsed().as_secs_f32() * 1000.0;

        self.remesh_dirty();
    }

    /// The brush a stamp actually runs with.
    ///
    /// Holding shift smooths, which is the convention every sculpting tool
    /// uses and the single biggest ergonomic win available here. Shift already
    /// modified right drag into a pan, and this is left drag, so there is no
    /// clash.
    ///
    /// Deliberately does NOT swap `self.brush.kind` and swap it back: nothing
    /// then has to notice a key being released while the window is unfocused,
    /// and the tool strip does not flicker between two highlights during a
    /// stroke. The selection is never touched, so there is nothing to restore.
    fn effective_brush(&self) -> Brush {
        if self.shift { Brush { kind: BrushKind::Smooth, ..self.brush } } else { self.brush }
    }

    /// Direction for a new stroke, honouring the invert modifier, the eraser
    /// end of the stylus, and the fact that some brushes have no opposite.
    ///
    /// The two inverts combine rather than override: holding the modifier while
    /// using the eraser gives back the additive brush, which is the same
    /// behaviour every drawing application has.
    fn stroke_direction(&self) -> BrushDirection {
        // Either modifier inverts. Alt is what ZBrush and Nomad both use, and
        // control is what this had first; keeping both costs nothing and means
        // neither habit is wrong. They do NOT compound -- holding both is still
        // one inversion, because a user pressing both means "invert", not
        // "invert twice".
        let modifier = self.control || self.alt;
        let inverted = modifier != self.eraser_in_use();
        if inverted && self.brush.kind.is_directional() {
            BrushDirection::Subtract
        } else {
            BrushDirection::Add
        }
    }

    /// Whether the eraser end of the stylus is the one in range.
    fn eraser_in_use(&self) -> bool {
        let pen = self.tablet.state();
        pen.in_proximity && pen.eraser
    }

    /// The world space lean of the pen, as a vector whose length is the tilt
    /// angle in radians.
    ///
    /// Tilt arrives in the tablet's own frame, which lines up with the screen,
    /// so it has to be carried into world space through the camera basis before
    /// it can steer anything.
    fn pen_lean(&self) -> Vec3 {
        if !self.tilt_enabled {
            return Vec3::ZERO;
        }
        let pen = self.tablet.state();
        if !pen.in_proximity {
            return Vec3::ZERO;
        }

        let magnitude = pen.tilt.length().min(1.0);
        if magnitude < 1.0e-4 {
            return Vec3::ZERO;
        }

        // Screen y grows downward and the camera's up axis grows upward, and a
        // positive tilt on that axis means the pen is leaning toward the user,
        // which is toward the bottom of the screen. Hence the subtraction.
        let direction =
            (self.camera.right() * pen.tilt.x - self.camera.up() * pen.tilt.y).normalize_or_zero();
        direction * (magnitude * MAX_TILT)
    }

    /// Turn the dragged line into a plane and cut with it.
    ///
    /// The plane is the one containing the eye and both ends of the line, so it
    /// is exactly the surface the line sweeps out going away from the viewer --
    /// which is what makes a screen-space drag mean something in three
    /// dimensions, and why the cut passes through the whole model rather than
    /// stopping at the first surface.
    ///
    /// **Which side goes is set by the drag direction**: the material to the
    /// LEFT of the arrow, as drawn on screen, is removed -- so a left to right
    /// drag takes the top half. Drag the other way to keep the other half.
    ///
    /// That sentence is copied from what
    /// `a_left_to_right_drag_removes_a_consistent_side` actually observes, not
    /// from reasoning about the cross product. The sign depends on the ray
    /// order, the handedness of the camera basis and whether the pixel-to-NDC
    /// step flips Y, and getting it backwards means the tool removes the half
    /// the user meant to keep.
    fn finish_cut(&mut self) {
        let Some(from) = self.cut_from.take() else {
            return;
        };
        let Some(to) = self.cursor else {
            return;
        };
        // A click is not a cut. Without this, arming the tool and clicking once
        // would take an arbitrary half of the model away.
        if from.distance(to) < CLICK_SLOP_PX {
            self.status = "cut cancelled: drag a line across the model".to_string();
            return;
        }

        let (eye, first) = self.ray_through(from);
        let (_, second) = self.ray_through(to);
        // Both rays leave the same eye, so their cross product is normal to the
        // plane through the two of them. The order is what decides the sign,
        // and therefore which side is cut.
        let Some(plane) = brokkr_core::ClipPlane::new(eye, second.cross(first)) else {
            self.status = "cut cancelled: that line has no direction".to_string();
            return;
        };

        self.volume.begin_stroke();
        let changed = self.volume.clip(plane);
        match self.volume.end_stroke() {
            Some(edit) if changed > 0 => {
                self.history.push(edit);
                self.history_stats = self.history.stats();
                self.unsaved = true;
                self.status = format!("cut {changed} bricks");
                self.remesh_dirty();
                self.refresh_overlay();
            }
            _ => {
                // The plane missed the model. Nothing changed, so nothing is
                // recorded -- an undo entry for a no-op would be worse than
                // none.
                self.status = "the cut missed the model".to_string();
            }
        }
        // One cut per arming. A destructive tool that stays live is how a
        // stray click removes half the model.
        self.cut_armed = false;
    }

    fn finish_stroke(&mut self) {
        self.stroke.end();
        if let Some(edit) = self.volume.end_stroke() {
            self.history.push(edit);
            self.history_stats = self.history.stats();
            // Inside the guard on purpose: a press and release that never
            // touched the model produces no edit, and must not raise a
            // "discard your work?" prompt on the way out.
            self.unsaved = true;
        }
    }

    /// Where exported files go.
    ///
    /// A fixed directory rather than a file dialog: Iced has no file picker and
    /// pulling one in is a dependency and a portal setup for something a first
    /// version can do without. The path is shown in the interface so it is never
    /// a guess.
    /// Everything worth pasting into a bug report.
    ///
    /// Deliberately a string on the clipboard rather than an upload: there is no
    /// server, no endpoint, no telemetry and no consent question. The user can
    /// read exactly what they are sending before they send it.
    ///
    /// The session type is here because it is the single most likely cause of a
    /// bad report: running on XWayland puts the window outside the X screen's
    /// bounds and the compositor stops requesting frames, which reads as a 1 fps
    /// bug and is not one.
    pub(crate) fn diagnostics(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out, "BrokkrSculpt {} ({})", env!("CARGO_PKG_VERSION"), build_commit());
        let _ = writeln!(
            out,
            "session: {} on {}",
            std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".into()),
            std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_else(|_| "unknown".into()),
        );
        let _ = writeln!(out, "wgpu: {}", self.shared.adapter_summary());
        let _ = writeln!(
            out,
            "model: voxel {:.3} mm, {} dense + {} uniform bricks, {:.1} MB",
            self.voxel_size,
            self.volume_stats.dense_bricks,
            self.volume_stats.uniform_bricks,
            self.volume_stats.resident_bytes as f64 / (1024.0 * 1024.0),
        );
        let pool = self.shared.stats();
        let _ = writeln!(
            out,
            "view: {} triangles, {} drawn / {} culled, {:.1} fps",
            pool.triangles,
            pool.drawn,
            pool.culled,
            if self.perf.average_frame_ms() > 0.0 {
                1000.0 / self.perf.average_frame_ms()
            } else {
                0.0
            },
        );
        let _ = writeln!(out, "tablet: {}", self.tablet.diagnosis().explain());
        let _ = writeln!(out, "spacemouse: {}", self.spacemouse.diagnosis().explain());
        if !self.status.is_empty() {
            let _ = writeln!(out, "last message: {}", self.status);
        }
        out
    }

    /// Throw the sculpt away and start again from a fresh sphere.
    ///
    /// Extracted from the old `ResetSphere` arm so the menu and the panel button
    /// share one path rather than drifting.
    fn reset_sculpt(&mut self) {
        let mut volume = Volume::new(self.voxel_size);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM);
        volume.mark_everything_dirty();
        // The old bricks have to be cleared from the pool too, or their
        // triangles stay on screen. Marking them dirty meshes them to nothing,
        // which releases their slices.
        for coord in self.volume.brick_coords() {
            volume.mark_dirty(coord);
        }
        self.volume = volume;
        // History refers to bricks of the volume that just went away, so keeping
        // it would let undo splice pieces of the discarded model into this one.
        self.history.clear();
        self.history_stats = self.history.stats();
        self.camera = OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM);
        self.project_path = None;
        self.unsaved = false;
        self.status = String::new();
        self.publish_camera();
        self.remesh_dirty();
        self.refresh_overlay();
    }

    /// How long between crash-net writes.
    ///
    /// Two minutes is a compromise against the write itself: a large sculpt is
    /// tens of megabytes and the write happens on the thread that draws, so
    /// often enough to be worth having and rare enough that the hitch is not
    /// something you sculpt against.
    const AUTOSAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(120);

    /// Where the crash net lives.
    ///
    /// `$XDG_STATE_HOME`, not the config directory: this is recoverable state
    /// rather than settings, and it must never sit next to the user's own
    /// files where it could be mistaken for one.
    pub(crate) fn default_autosave_path() -> Option<std::path::PathBuf> {
        let base = std::env::var_os("XDG_STATE_HOME")
            .map(std::path::PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|home| std::path::PathBuf::from(home).join(".local").join("state"))
            })?;
        Some(base.join("brokkrsculpt").join("autosave.brokkr"))
    }

    /// Write the crash net if it is due.
    ///
    /// Driven from the frame tick rather than a timer, because iced's `time`
    /// module is empty under this feature set -- the defaults here include
    /// `thread-pool`, not `tokio` or `smol`, so `iced::time::every` does not
    /// exist to be called.
    ///
    /// Known limitation: `window::frames()` stops arriving when the compositor
    /// stops asking for frames, so autosave stalls on a minimised or fully
    /// occluded window. That is the acceptable direction -- the case worth
    /// protecting is a crash during active sculpting, and during active
    /// sculpting there are frames. Revisit if iced gains a timer here.
    fn maybe_autosave(&mut self) {
        if !self.unsaved
            // Never mid-stroke: the write would land in the middle of a drag
            // and read as a stutter in the brush.
            || self.stroke.is_active()
            || self.drag.is_some()
            || self.last_autosave.elapsed() < Self::AUTOSAVE_INTERVAL
        {
            return;
        }
        self.write_autosave();
        self.last_autosave = Instant::now();
    }

    /// Write the crash net, reporting only to the log.
    ///
    /// Deliberately does NOT clear `unsaved` and does NOT touch
    /// `project_path`: this is not a save, it is something to recover from, and
    /// treating it as a save would suppress the very prompt that protects the
    /// real file. It also never writes over the user's own file.
    ///
    /// Failures go to the log and nowhere else. `status` is the line the user
    /// is reading for the action they asked for, and an autosave is not one.
    fn write_autosave(&mut self) {
        let Some(path) = self.autosave_file.clone() else {
            return;
        };
        if let Some(parent) = path.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            log::warn!("could not create {}: {error}", parent.display());
            return;
        }

        // Through a temporary and a rename, so a crash during the write cannot
        // destroy the previous autosave -- which would be the one moment it was
        // most needed.
        let temporary = path.with_extension("brokkr.tmp");
        let state = self.project_state();
        // `File::create` fails with an `io::Error` and `project::write` with a
        // `ProjectError`, so neither `?` nor `and_then` will chain them.
        // Reporting is identical either way -- the log line and nothing else.
        let write = match std::fs::File::create(&temporary) {
            Ok(file) => {
                let mut writer = std::io::BufWriter::new(file);
                brokkr_core::project::write(&mut writer, &self.volume, &state)
                    .map_err(|error| error.to_string())
            }
            Err(error) => Err(error.to_string()),
        };
        match write {
            Ok(()) => match std::fs::rename(&temporary, &path) {
                Ok(()) => log::info!("autosaved to {}", path.display()),
                Err(error) => log::warn!("could not replace {}: {error}", path.display()),
            },
            Err(error) => {
                log::warn!("could not autosave to {}: {error}", temporary.display());
                std::fs::remove_file(&temporary).ok();
            }
        }
    }

    /// Throw the crash net away, once its work is safely somewhere else.
    fn clear_autosave(&self) {
        if let Some(path) = self.autosave_file.as_ref() {
            std::fs::remove_file(path).ok();
        }
    }

    /// The session state a `.brokkr` file carries alongside the field.
    ///
    /// Shared by the explicit save and the autosave so the two cannot record
    /// different things.
    fn project_state(&self) -> brokkr_core::ProjectState {
        brokkr_core::ProjectState {
            camera_target: self.camera.target,
            camera_distance: self.camera.distance,
            camera_yaw: self.camera.yaw,
            camera_pitch: self.camera.pitch,
            camera_roll: self.camera.roll,
            brush_radius: self.brush.radius,
            brush_strength: self.brush.strength,
            mirror: MirrorAxis::ALL.map(|axis| self.symmetry.axis(axis)),
        }
    }

    /// Whether there is a crash net to offer.
    pub(crate) fn has_autosave(&self) -> bool {
        self.autosave_file.as_ref().is_some_and(|path| path.is_file())
    }

    /// Load the crash net, and deliberately do not adopt its path.
    ///
    /// `open_project` would leave `project_path` pointing at the autosave file,
    /// and the next plain Save would then write the user's work back into the
    /// crash net -- a file under `$XDG_STATE_HOME` that they never chose and
    /// that the next successful save deletes. So the recovered document is
    /// deliberately homeless and unsaved: Save behaves as Save As, which is
    /// what makes them name it somewhere real.
    fn recover_autosave(&mut self) {
        let Some(path) = self.autosave_file.clone() else {
            return;
        };
        self.open_project(&path);
        if self.status.contains("could not") {
            return;
        }
        self.project_path = None;
        self.unsaved = true;
        // `open_project` records whatever it opened. The autosave is a crash
        // net, not a document, and has no business in the recent list.
        self.recent.forget(&path);
        self.status = "recovered the autosave — save it somewhere".to_string();
    }

    fn open_project(&mut self, path: &std::path::Path) {
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) => {
                self.status = format!("could not open {}: {error}", path.display());
                // A file that has been moved or deleted since it was last used
                // should stop being offered.
                self.recent.forget(path);
                return;
            }
        };
        let mut reader = std::io::BufReader::new(file);
        let (volume, state) = match brokkr_core::project::read(&mut reader) {
            Ok(loaded) => loaded,
            Err(error) => {
                // The message says what was expected as well as what was found,
                // because "a file from a different build" and "a partial
                // download" look identical without it.
                self.status = format!("could not read {}: {error}", path.display());
                return;
            }
        };

        // Clear the outgoing model from the mesh pool before swapping, exactly
        // as a reset does.
        let stale: Vec<BrickCoord> = self.volume.brick_coords().collect();
        self.volume = volume;
        for coord in stale {
            self.volume.mark_dirty(coord);
        }

        self.voxel_size = self.volume.voxel_size();
        self.camera = OrbitCamera {
            target: state.camera_target,
            distance: state.camera_distance,
            yaw: state.camera_yaw,
            pitch: state.camera_pitch,
            roll: state.camera_roll,
            ..OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM)
        };
        self.brush.radius = state.brush_radius.clamp(MIN_RADIUS_MM, MAX_RADIUS_MM);
        self.brush.strength = state.brush_strength.clamp(MIN_STRENGTH, MAX_STRENGTH);
        self.symmetry = MirrorAxis::ALL
            .into_iter()
            .zip(state.mirror)
            .fold(Symmetry::OFF, |set, (axis, on)| set.with_axis(axis, on));

        // History belongs to the model that just went away.
        self.history.clear();
        self.history_stats = self.history.stats();
        self.project_path = Some(path.to_path_buf());
        self.unsaved = false;
        self.recent.record(path);
        self.status = format!("opened {}", path.display());
        self.publish_camera();
        self.remesh_dirty();
        self.refresh_overlay();
    }

    /// Swap an imported model in, modelled on `open_project`.
    ///
    /// Four things here are individually silent when wrong.
    ///
    /// `project_path` is cleared, where `open_project` sets it. An import must
    /// not adopt the mesh's path, or the next plain Save would write a `.brokkr`
    /// container straight over the user's `.stl` with no dialog and no warning.
    /// This is the most damaging mistake available in this function.
    ///
    /// The stale brick loop clears the outgoing model from the renderer's mesh
    /// pool. Skipping it compiles and leaves the previous model's triangles
    /// drawn on top of the new one.
    ///
    /// `mark_everything_dirty` is done by `voxelise` itself, exactly as
    /// `project::read` does it, which is why neither this nor `open_project`
    /// appears to need it.
    ///
    /// The camera is framed on `MODEL_RADIUS_MM` like every other call site, not
    /// on `bounding_radius`: that is measured from the origin over brick
    /// extents, so for a centred model it over-reports by up to about 1.7 times
    /// plus padding and the camera sits visibly too far back.
    fn adopt_import(&mut self, imported: crate::message::Imported) {
        let stale: Vec<BrickCoord> = self.volume.brick_coords().collect();
        self.volume = imported.volume;
        for coord in stale {
            self.volume.mark_dirty(coord);
        }

        self.voxel_size = self.volume.voxel_size();
        self.camera = OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM);
        self.history.clear();
        self.history_stats = self.history.stats();
        self.project_path = None;
        // An import is unsaved work by definition: there is no `.brokkr` file
        // holding it, and quitting now would lose the whole thing.
        self.unsaved = true;
        self.status = format!(
            "imported {}: {}, {:.0} ms",
            imported
                .source
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| imported.source.display().to_string()),
            imported.report.summary(),
            imported.elapsed_ms,
        );
        log::info!("{}", self.status);
        self.publish_camera();
        self.remesh_dirty();
        self.refresh_overlay();
    }

    fn save_project(&mut self, path: &std::path::Path) {
        let state = self.project_state();

        let file = match std::fs::File::create(path) {
            Ok(file) => file,
            Err(error) => {
                self.status = format!("could not write {}: {error}", path.display());
                return;
            }
        };
        let mut writer = std::io::BufWriter::new(file);
        match brokkr_core::project::write(&mut writer, &self.volume, &state) {
            Ok(()) => {
                self.project_path = Some(path.to_path_buf());
                // Only here. Both failure arms leave the flag set, or a failed
                // write would let the next quit go through silently.
                self.unsaved = false;
                self.recent.record(path);
                // The work is safely in a file the user chose, so the crash net
                // has nothing left to protect and a stale one would only offer
                // to restore something older than what is on disk.
                self.clear_autosave();
                self.status = format!("saved {}", path.display());
            }
            Err(error) => self.status = format!("could not write {}: {error}", path.display()),
        }
    }

    fn export_directory() -> std::path::PathBuf {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        home.unwrap_or_else(std::env::temp_dir).join("brokkrsculpt")
    }

    /// Weld the sculpt into one mesh and write it out.
    ///
    /// Refuses to write anything that would not print. A slicer given a mesh
    /// with holes either rejects it or fills it wrong, and finding that out
    /// after a failed print is worse than being told here.
    /// Write the sculpt out to a chosen path.
    ///
    /// The path now comes from a dialog rather than a fixed directory, so there
    /// is nothing to create and nothing to guess.
    fn export(&mut self, format: ExportFormat, path: &std::path::Path) {
        let started = Instant::now();
        let (mesh, report) = self.volume.export_mesh();

        if !report.is_printable() {
            self.status = format!("not exported, {}", report.summary());
            log::error!("refusing to export: {}", report.summary());
            return;
        }

        let write = || -> std::io::Result<()> {
            let file = std::fs::File::create(path)?;
            let mut file = std::io::BufWriter::new(file);
            match format {
                ExportFormat::Stl => brokkr_core::export::stl::write(&mesh, &mut file)?,
                ExportFormat::Obj => brokkr_core::export::obj::write(&mesh, &mut file)?,
                ExportFormat::ThreeMf => brokkr_core::export::threemf::write(&mesh, &mut file)?,
            }
            std::io::Write::flush(&mut file)
        };

        match write() {
            Ok(()) => {
                let bytes = std::fs::metadata(path).map(|data| data.len()).unwrap_or(0);
                self.status = format!(
                    "{} to {} ({:.1} MB, {:.0} ms)",
                    report.summary(),
                    path.display(),
                    bytes as f64 / (1024.0 * 1024.0),
                    started.elapsed().as_secs_f64() * 1000.0
                );
                log::info!("{}", self.status);
            }
            Err(error) => {
                self.status = format!("could not write {}: {error}", path.display());
                log::error!("{}", self.status);
            }
        }
    }

    /// The voxel size a request actually lands on.
    fn clamped_voxel_size(requested: f32) -> f32 {
        requested.clamp(FINEST_VOXEL_MM, COARSEST_VOXEL_MM)
    }

    /// Rebuild the volume at a different voxel size.
    fn resample(&mut self, voxel_size: f32) {
        let voxel_size = Self::clamped_voxel_size(voxel_size);
        if (voxel_size - self.voxel_size).abs() < 1.0e-6 {
            return;
        }

        let started = Instant::now();
        let mut resampled = self.volume.resampled(voxel_size);
        // The old bricks have to be cleared out of the renderer's pool, and
        // after a resample their coordinates mean something different, so they
        // are marked in the new volume to remesh to nothing.
        for coord in self.volume.brick_coords() {
            resampled.mark_dirty(coord);
        }

        self.volume = resampled;
        self.voxel_size = voxel_size;
        // Past the early return above, so resampling to the size already in use
        // stays the no-op its test asserts it is.
        self.unsaved = true;
        // History refers to bricks of a volume that no longer exists, and at a
        // different resolution, so keeping it would splice nonsense back in.
        self.history.clear();
        self.history_stats = self.history.stats();
        self.remesh_dirty();

        self.status = format!(
            "resampled to {voxel_size:.3} mm, {} dense bricks, {:.0} MB, {:.0} ms",
            self.volume_stats.dense_bricks,
            self.volume_stats.resident_bytes as f64 / (1024.0 * 1024.0),
            started.elapsed().as_secs_f64() * 1000.0
        );
        log::info!("{}", self.status);
    }

    /// Apply one frame of puck motion, and whatever its buttons asked for.
    fn drive_from_spacemouse(&mut self, elapsed_ms: f32) {
        for action in self.spacemouse.take_presses() {
            match action {
                ButtonAction::None => {}
                ButtonAction::Undo => self.undo(),
                ButtonAction::Redo => self.redo(),
                // Recentre and refit without losing which way the model is
                // being looked at, which is what makes this useful mid sculpt.
                ButtonAction::FrameModel => {
                    let framed = OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM);
                    self.camera = OrbitCamera {
                        yaw: self.camera.yaw,
                        pitch: self.camera.pitch,
                        roll: self.camera.roll,
                        ..framed
                    };
                    self.publish_camera();
                }
                // Everything back to the start, roll included. This is the way
                // out of a view that has been twisted somewhere confusing.
                ButtonAction::ResetView => {
                    self.camera = OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM);
                    self.publish_camera();
                }
                ButtonAction::ToggleSymmetry => {
                    self.symmetry = self.symmetry.toggled(MirrorAxis::X);
                }
            }
        }

        let motion = self.spacemouse.motion();
        let moved = self.spacemouse.config.apply(
            &motion,
            elapsed_ms,
            &mut self.camera,
            self.viewport_size.y,
        );
        if moved {
            self.publish_camera();
        }
    }

    fn configure_spacemouse(&mut self, setting: SpaceMouseSetting) {
        let config = &mut self.spacemouse.config;
        // Whether this change should reach the disk now. A slider mid drag
        // should not: it sends a message per step, and `Save` follows on
        // release.
        let mut persist = true;
        match setting {
            SpaceMouseSetting::Mode(mode) => config.mode = mode,
            SpaceMouseSetting::Deadzone(value) => {
                config.deadzone = value;
                persist = false;
            }
            SpaceMouseSetting::PanSens(value) => {
                config.pan_sens = value;
                persist = false;
            }
            SpaceMouseSetting::ZoomSens(value) => {
                config.zoom_sens = value;
                persist = false;
            }
            SpaceMouseSetting::OrbitSens(value) => {
                config.orbit_sens = value;
                persist = false;
            }
            SpaceMouseSetting::Binding(action, source) => {
                let invert = config.binding(action).invert;
                config.set_binding(action, AxisBinding { source, invert });
            }
            SpaceMouseSetting::Invert(action, invert) => {
                let source = config.binding(action).source;
                config.set_binding(action, AxisBinding { source, invert });
            }
            SpaceMouseSetting::Button(index, action) => {
                if let Some(slot) = config.buttons.get_mut(index) {
                    *slot = action;
                }
            }
            SpaceMouseSetting::InvertAll => config.invert_all(),
            SpaceMouseSetting::Reset => *config = SpaceMouseConfig::default(),
            SpaceMouseSetting::Save => {}
        }
        if persist {
            self.spacemouse.config.save();
        }
    }

    /// Take a keystroke in one of the menu's numeric fields.
    ///
    /// The text is kept whatever it says, so `2.` and an empty field survive
    /// being typed through. The value only moves when the text parses to
    /// something inside the same bounds the sliders use -- a field that silently
    /// accepted 500 mm would be worse than one that ignores it.
    fn edit_menu_field(&mut self, which: SizingTarget, text: String) {
        if let Ok(value) = text.trim().parse::<f32>()
            && value.is_finite()
        {
            match which {
                SizingTarget::Radius => {
                    self.brush.radius = value.clamp(MIN_RADIUS_MM, MAX_RADIUS_MM);
                }
                SizingTarget::Strength => {
                    self.brush.strength = value.clamp(MIN_STRENGTH, MAX_STRENGTH);
                }
            }
        }
        self.menu_edit = Some((which, text));
    }

    /// The text a menu field should show: what is being typed, or the current
    /// value formatted.
    pub(crate) fn menu_field_text(&self, which: SizingTarget) -> String {
        if let Some((editing, text)) = &self.menu_edit
            && *editing == which
        {
            return text.clone();
        }
        match which {
            SizingTarget::Radius => format!("{:.2}", self.brush.radius),
            SizingTarget::Strength => format!("{:.2}", self.brush.strength),
        }
    }

    /// Move the number a sizing gesture is holding, from how far the pointer
    /// has travelled since it began.
    ///
    /// Horizontal only, and absolute against where the drag started, so out and
    /// back returns to exactly the original value.
    fn apply_sizing(&mut self, to: Vec2) {
        let Some(sizing) = self.sizing else {
            return;
        };
        let travel = to.x - sizing.from_pixel.x;
        match sizing.what {
            // Multiplicative, matching the [ and ] keys: the radius spans fifty
            // to one, so a fixed amount per pixel would crawl at one end and
            // jump at the other.
            SizingTarget::Radius => {
                let factor = (travel * RADIUS_PER_PIXEL).exp();
                self.brush.radius = (sizing.original * factor).clamp(MIN_RADIUS_MM, MAX_RADIUS_MM);
            }
            // Additive: strength is already a small linear range.
            SizingTarget::Strength => {
                self.brush.strength = (sizing.original + travel * STRENGTH_PER_PIXEL)
                    .clamp(MIN_STRENGTH, MAX_STRENGTH);
            }
        }
    }

    /// Undo and redo both count as changes.
    ///
    /// Undoing back to exactly the state that was last saved leaves the flag
    /// set, so the prompt on the way out is a false positive. Telling that case
    /// apart needs a save point recorded in the history, and the history is
    /// cleared by open, resample and reset, so the marker would have to be
    /// invalidated in three more places. Over-prompting costs a click;
    /// under-prompting costs the sculpt.
    fn undo(&mut self) {
        if self.history.undo(&mut self.volume) {
            self.history_stats = self.history.stats();
            self.unsaved = true;
            self.remesh_dirty();
        }
    }

    fn redo(&mut self) {
        if self.history.redo(&mut self.volume) {
            self.history_stats = self.history.stats();
            self.unsaved = true;
            self.remesh_dirty();
        }
    }

    fn on_pointer(&mut self, event: PointerEvent) {
        // While the unsaved-work prompt is up, the pointer belongs to it. The
        // capture layer in `view` already swallows presses, but the viewport is
        // a shader widget that sees events wherever the cursor is, so this is
        // the guarantee rather than the belt: without it a press behind the
        // card would sculpt into a document the user is about to discard.
        if self.confirm.is_some() {
            return;
        }
        match event {
            PointerEvent::Modifiers { shift, control, alt } => {
                self.shift = shift;
                self.control = control;
                self.alt = alt;
                // Both change what a press would do, so the ring has to say so
                // before the press rather than after.
                self.refresh_overlay();
            }
            PointerEvent::Pressed { button, position, size } => {
                self.viewport_size = Vec2::new(size.x, size.y);
                let position = Vec2::new(position.x, position.y);
                self.cursor = Some(position);

                // An open menu swallows the next press: closing it is what the
                // click was for, and sculpting as well would be a surprise.
                if self.menu.take().is_some() || self.top_menu.take().is_some() {
                    return;
                }

                // A press on the navigation cube belongs to the cube. Checked
                // before anything else, or clicking it would also carve a divot
                // out of the model behind it.
                if button == PointerButton::Left
                    && let Some(part) = navcube::pick(&self.camera, self.viewport_size, position)
                {
                    self.fly_to(part);
                    return;
                }

                let kind = match button {
                    // A cut outranks sculpting: the mode was armed deliberately
                    // and the next left drag is the line, not a stroke.
                    PointerButton::Left if self.cut_armed => {
                        self.cut_from = Some(position);
                        DragKind::Cutting
                    }
                    // Left sculpts -- unless a hold-and-drag resize is in
                    // progress, in which case the pointer belongs to that
                    // gesture and a press must not lay down a stroke.
                    PointerButton::Left if self.sizing.is_some() => DragKind::Sizing,
                    PointerButton::Left => DragKind::Sculpt(self.stroke_direction()),
                    // Right and middle move the camera. Shift slides instead of
                    // turning.
                    PointerButton::Right | PointerButton::Middle => {
                        if self.shift {
                            DragKind::Pan
                        } else {
                            DragKind::Orbit
                        }
                    }
                };
                self.drag = Some(Drag {
                    button,
                    kind,
                    origin: position,
                    pressed_at: Instant::now(),
                    moved: false,
                });

                if let DragKind::Sculpt(direction) = kind {
                    // One stroke is one undo entry, so recording opens here and
                    // closes when the button comes back up.
                    self.volume.begin_stroke();
                    self.sculpt_to(position, direction, true);
                }
                self.update_hover(position);
                self.refresh_overlay();
            }
            PointerEvent::Released { button } => {
                if let Some(drag) = self.drag.filter(|drag| drag.button == button) {
                    if matches!(drag.kind, DragKind::Cutting) {
                        self.finish_cut();
                    }
                    if matches!(drag.kind, DragKind::Sculpt(_)) {
                        self.finish_stroke();
                    }
                    // A right press that neither moved nor lingered was a click,
                    // and a click opens the tool's own controls where the hand
                    // already is. Anything else was an orbit or a pan.
                    if button == PointerButton::Right
                        && !drag.moved
                        && drag.pressed_at.elapsed().as_millis() < CLICK_MS
                    {
                        self.menu = Some(drag.origin);
                    }
                    self.drag = None;
                }
                self.refresh_overlay();
            }
            PointerEvent::Moved { position, size } => {
                self.viewport_size = Vec2::new(size.x, size.y);
                let position = Vec2::new(position.x, position.y);
                let delta = self.cursor.map(|previous| position - previous).unwrap_or(Vec2::ZERO);
                self.cursor = Some(position);

                // Past the slop this is a drag, not a click, and it stays one
                // even if it comes back: otherwise an orbit that ends where it
                // began would open the menu.
                if let Some(drag) = &mut self.drag
                    && position.distance(drag.origin) > CLICK_SLOP_PX
                {
                    drag.moved = true;
                }

                // A held sizing key owns the pointer whether or not a button
                // is down, so this comes before the drag cases.
                if self.sizing.is_some() {
                    self.apply_sizing(position);
                    self.refresh_overlay();
                    return;
                }

                match self.drag.map(|drag| drag.kind) {
                    Some(DragKind::Sculpt(direction)) => self.sculpt_to(position, direction, false),
                    Some(DragKind::Orbit) => {
                        self.camera.orbit(delta);
                        self.publish_camera();
                    }
                    Some(DragKind::Pan) => {
                        self.camera.pan(delta, self.viewport_size.y);
                        self.publish_camera();
                    }
                    // The cut line is only drawn while it is being dragged;
                    // the model is not touched until the button comes up, so
                    // the line can be adjusted freely.
                    Some(DragKind::Cutting) | Some(DragKind::Sizing) | None => {}
                }

                // Over the cube: light the part under the pointer, and draw no
                // brush ring, because a press there will not sculpt.
                let over_cube = navcube::pick(&self.camera, self.viewport_size, position);
                if over_cube != self.cube_hover {
                    self.cube_hover = over_cube;
                    self.refresh_cube();
                }
                if over_cube.is_some() {
                    self.hover = None;
                    self.refresh_overlay();
                    return;
                }

                // The ring follows the pointer, and the surface under it moves
                // as a stroke cuts into it.
                self.update_hover(position);
                self.refresh_overlay();
            }
            PointerEvent::Scrolled { amount } => {
                self.camera.zoom(amount);
                self.publish_camera();
            }
        }
    }

    /// Handle one message.
    ///
    /// Returns a [`Task`] because anything that touches the filesystem -- a
    /// dialog, a save, an export -- must not run here. The event loop is the
    /// same thread that draws, so a blocking call freezes the window, which is
    /// exactly the export freeze this replaces.
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Pointer(event) => self.on_pointer(event),
            Message::Frame => {
                let elapsed_ms = self.perf.record_frame();
                if self.advance_flight(elapsed_ms) {
                    self.publish_camera();
                }
                self.drive_from_spacemouse(elapsed_ms);
                self.maybe_autosave();
            }
            Message::BrushKindChanged(kind) => self.brush.kind = kind,
            Message::FalloffChanged(curve) => self.brush.falloff = curve,
            Message::BrushRadiusChanged(radius) => self.brush.radius = radius,
            Message::BrushStrengthChanged(strength) => self.brush.strength = strength,
            Message::SymmetryAxisToggled(axis) => self.symmetry = self.symmetry.toggled(axis),
            Message::PatternChanged(kind) => self.brush.pattern.kind = kind,
            Message::PatternScaleChanged(scale) => self.brush.pattern.scale_mm = scale,
            Message::PatternDepthChanged(depth) => self.brush.pattern.depth = depth,
            Message::SizingStarted(what) => {
                // Nothing to anchor the gesture to until the pointer has been
                // seen; and a gesture already running must not restart, or
                // holding the key through a drag would reset it every event.
                if self.sizing.is_none()
                    && let Some(from_pixel) = self.cursor
                {
                    self.sizing = Some(Sizing {
                        what,
                        from_pixel,
                        original: match what {
                            SizingTarget::Radius => self.brush.radius,
                            SizingTarget::Strength => self.brush.strength,
                        },
                    });
                }
            }
            Message::SizingEnded => self.sizing = None,
            Message::TopMenuToggled(which) => {
                // Clicking the open one closes it, which is what a menu bar
                // does everywhere.
                self.top_menu = (self.top_menu != Some(which)).then_some(which);
            }
            Message::DiagnosticsCopied => {
                let report = self.diagnostics();
                self.status = "diagnostics copied".to_string();
                self.top_menu = None;
                return iced::clipboard::write(report);
            }
            Message::IssueOpened => {
                self.top_menu = None;
                self.status = format!("report bugs at {ISSUE_URL}");
                log::info!("{}", self.diagnostics());
            }
            Message::MenuClosed => {
                // Escape is the only sender. Against an open prompt it means
                // Cancel, which is the harmless answer -- an explicit arm
                // rather than letting the clears below reach `confirm`, since
                // dismissing the prompt by accident is how work gets lost.
                if self.confirm.is_some() {
                    return self.answer_confirm(ConfirmChoice::Cancel);
                }
                // Escape is also the way out of an armed cut, which is the only
                // mode in the application that changes what a click does.
                self.cut_armed = false;
                self.cut_from = None;
                self.menu = None;
                self.menu_edit = None;
                self.top_menu = None;
            }
            Message::MenuFieldEdited(which, text) => self.edit_menu_field(which, text),
            Message::MenuFieldSubmitted => self.menu_edit = None,
            Message::CutToggled => {
                self.cut_armed = !self.cut_armed;
                self.cut_from = None;
                self.status = if self.cut_armed {
                    "cut armed: drag a line across the model, the left of the arrow goes"
                        .to_string()
                } else {
                    String::new()
                };
            }
            Message::DynamicRadiusToggled(on) => self.dynamic_radius = on,
            Message::BrushRadiusScaled(factor) => {
                self.brush.radius =
                    (self.brush.radius * factor).clamp(MIN_RADIUS_MM, MAX_RADIUS_MM);
            }
            Message::PressureToggled(on) => self.pressure_enabled = on,
            Message::PressureCurveChanged(curve) => self.pressure_curve = curve,
            Message::TiltToggled(on) => self.tilt_enabled = on,
            Message::ResetPressurePeak => self.tablet.reset_peak(),
            Message::Undo => self.undo(),
            Message::Redo => self.redo(),
            Message::Export(format) => {
                return Task::perform(pick_export_target(format), move |path| {
                    Message::ExportChosen(format, path)
                });
            }
            Message::Resample(voxel_size) => self.resample(voxel_size),
            Message::SpaceMouse(setting) => self.configure_spacemouse(setting),
            Message::SectionToggled(section) => {
                let open = &mut self.expanded[section as usize];
                *open = !*open;
            }
            // Both the panel button and File > New come here, so the two
            // cannot drift apart. Both discard the document, so both ask.
            Message::ResetSphere | Message::NewSculpt => {
                return self.guard(PendingAction::NewSculpt);
            }

            // --- the unsaved-work prompt -------------------------------------
            Message::CloseRequested(id) => return self.guard(PendingAction::Quit(id)),
            Message::ConfirmAnswered(choice) => return self.answer_confirm(choice),
            Message::SavedThenContinue(path) => {
                let Some(path) = path else {
                    // The save dialog was dismissed. That is not an answer to
                    // the prompt, so the prompt stays up.
                    return Task::none();
                };
                self.save_project(&path);
                if self.unsaved {
                    // The write failed and `status` says why. Leave the prompt
                    // up rather than quitting on a file that was never written.
                    return Task::none();
                }
                if let Some(action) = self.confirm.take() {
                    return self.run_pending(action);
                }
            }

            // --- files -------------------------------------------------------
            // Each is two halves: ask for a path off the UI thread, then act on
            // whatever came back. A `None` means the dialog was dismissed, which
            // is not an error and must leave everything alone.
            Message::OpenRequested => return self.guard(PendingAction::Open),
            Message::OpenRecent(path) => return self.guard(PendingAction::OpenRecent(path)),
            Message::RecoverAutosave => return self.guard(PendingAction::RecoverAutosave),

            // --- importing a mesh --------------------------------------------
            Message::ImportRequested => return self.guard(PendingAction::Import),
            Message::ImportChosen(path) => {
                let Some(path) = path else {
                    return Task::none();
                };
                self.status = format!("importing {}…", path.display());
                let voxel_size = self.voxel_size;
                // Off the update loop, unlike open and save. Measured by
                // `cargo bench -p brokkr-core --bench import`: a 542k triangle
                // sphere reads in 62 ms and voxelises in 148 to 210 ms
                // depending on voxel size -- so not the seconds this was first
                // written for, but still twelve dropped frames if it ran on the
                // thread that draws, and a scan or a fine voxel size scales it
                // straight past that.
                return Task::perform(
                    async move {
                        let started = Instant::now();
                        let outcome = brokkr_core::import::read_path(&path).and_then(|mesh| {
                            let options = brokkr_core::voxelise::VoxeliseOptions::at(voxel_size);
                            brokkr_core::voxelise::voxelise(&mesh, &options).map(
                                |(volume, report)| crate::message::Imported {
                                    volume,
                                    report,
                                    source: path.clone(),
                                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                                },
                            )
                        });
                        crate::message::ImportPayload::new(outcome)
                    },
                    Message::ImportLoaded,
                );
            }
            Message::ImportLoaded(payload) => {
                match payload.take() {
                    Some(Ok(imported)) => self.adopt_import(imported),
                    // The "could not" prefix is load bearing: the header colours
                    // the status line by that substring, so a message without it
                    // renders in muted grey as though the import had worked.
                    Some(Err(error)) => {
                        self.status = format!("could not import: {error}");
                        log::error!("{}", self.status);
                    }
                    // Already taken, which can only happen if the message were
                    // delivered twice. Nothing to do, and nothing is lost.
                    None => {}
                }
            }
            Message::OpenChosen(path) => {
                if let Some(path) = path {
                    self.open_project(&path);
                }
            }
            Message::SaveRequested => match self.project_path.clone() {
                // Save over the file it came from; with no file yet this is a
                // Save As, which is what a first Save should do.
                Some(path) => self.save_project(&path),
                None => return Task::perform(pick_project_to_save(), Message::SaveChosen),
            },
            Message::SaveAsRequested => {
                return Task::perform(pick_project_to_save(), Message::SaveChosen);
            }
            Message::SaveChosen(path) => {
                if let Some(path) = path {
                    self.save_project(&path);
                }
            }
            Message::ExportRequested(format) => {
                return Task::perform(pick_export_target(format), move |path| {
                    Message::ExportChosen(format, path)
                });
            }
            Message::ExportChosen(format, path) => {
                if let Some(path) = path {
                    self.export(format, &path);
                }
            }
        }
        Task::none()
    }
}

impl Default for Brokkr {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `update` returns a `Task` now. Tests do not run the iced runtime, so
    /// there is nothing to hand it to and dropping it is correct — but it must
    /// be dropped deliberately rather than by `#[allow]`, or a test that should
    /// have driven a dialog would pass silently.
    fn update(app: &mut Brokkr, message: Message) {
        let task = app.update(message);
        drop(task);
    }

    use crate::tablet::PenState;
    use brokkr_core::BrushKind;
    use iced::Vector;
    use std::time::Duration;

    const SIZE: Vector = Vector { x: 800.0, y: 600.0 };

    fn centre_of_viewport() -> Vector {
        Vector::new(SIZE.x / 2.0, SIZE.y / 2.0)
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// Run a camera flight to completion in controlled time.
    ///
    /// `Message::Frame` scales by the real clock, and consecutive calls in a
    /// test are microseconds apart, so driving it that way would never arrive.
    fn finish_flight(app: &mut Brokkr) {
        for _ in 0..200 {
            if app.flight.is_none() {
                return;
            }
            app.advance_flight(16.0);
            app.publish_camera();
        }
        panic!("the flight never finished");
    }

    fn press(app: &mut Brokkr, at: Vector) {
        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: at,
            size: SIZE,
        });
    }

    fn release(app: &mut Brokkr) {
        app.on_pointer(PointerEvent::Released { button: PointerButton::Left });
    }

    /// The whole input to geometry path, with no window and no GPU: a press at
    /// the centre of the viewport must raycast onto the sphere, stamp the
    /// brush, and leave meshed bricks waiting for upload.
    #[test]
    fn a_press_at_the_centre_of_the_viewport_changes_the_model() {
        let mut app = app();
        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let before = app.volume.sample_world(front);

        press(&mut app, centre_of_viewport());

        assert!(app.perf.edit_ms > 0.0, "no edit was timed, so the raycast missed the sphere");
        assert!(app.perf.dirty_bricks > 0, "the stroke dirtied nothing");
        assert!(
            app.volume.sample_world(front) < before,
            "adding clay should have pushed the field negative at the surface"
        );
    }

    #[test]
    fn the_history_budget_is_reported_before_anything_is_drawn() {
        let app = app();
        assert!(
            app.history_stats.budget_bytes > 0,
            "the overlay would show a zero byte history budget until the first stroke"
        );
    }

    #[test]
    fn a_press_that_misses_the_model_changes_nothing() {
        let mut app = app();
        let bricks_before = app.volume.brick_count();

        // The far corner of the viewport looks past the sphere into empty space.
        press(&mut app, Vector::new(2.0, 2.0));

        assert_eq!(app.volume.brick_count(), bricks_before, "a miss must not allocate");
        assert_eq!(app.perf.dirty_bricks, 0, "a miss must not schedule a remesh");
    }

    #[test]
    fn orbiting_moves_the_camera_without_touching_the_model() {
        let mut app = app();
        let yaw = app.camera.yaw;
        let bricks = app.volume.brick_count();

        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Right,
            position: centre_of_viewport(),
            size: SIZE,
        });
        app.on_pointer(PointerEvent::Moved { position: Vector::new(460.0, 300.0), size: SIZE });

        assert_ne!(app.camera.yaw, yaw, "a right drag should have orbited");
        assert_eq!(app.volume.brick_count(), bricks, "orbiting must not sculpt");
    }

    #[test]
    fn releasing_a_different_button_does_not_cancel_a_stroke() {
        let mut app = app();
        press(&mut app, centre_of_viewport());
        app.on_pointer(PointerEvent::Released { button: PointerButton::Right });
        assert!(app.drag.is_some(), "the left button drag should still be live");

        release(&mut app);
        assert!(app.drag.is_none());
    }

    #[test]
    fn a_finished_stroke_becomes_exactly_one_undo_entry() {
        let mut app = app();
        press(&mut app, centre_of_viewport());
        for offset in 1..8 {
            app.on_pointer(PointerEvent::Moved {
                position: centre_of_viewport() + Vector::new(offset as f32 * 6.0, 0.0),
                size: SIZE,
            });
        }
        assert_eq!(app.history_stats.undo_entries, 0, "history should wait for the button up");

        release(&mut app);
        assert_eq!(
            app.history_stats.undo_entries, 1,
            "a whole drag is one entry, not one per pointer event"
        );
    }

    #[test]
    fn undo_returns_the_model_to_where_it_started() {
        let mut app = app();
        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let before = app.volume.sample_world(front);

        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert_ne!(app.volume.sample_world(front), before);

        update(&mut app, Message::Undo);
        assert_eq!(app.volume.sample_world(front), before, "undo did not restore the field");
        assert!(app.perf.dirty_bricks > 0, "undo must schedule a remesh or the screen goes stale");

        update(&mut app, Message::Redo);
        assert_ne!(app.volume.sample_world(front), before, "redo did not reapply the stroke");
    }

    #[test]
    fn a_drag_stamps_more_than_once_along_its_path() {
        // Without interpolation a fast drag leaves a dotted trail, so check the
        // stroke actually produced several stamps from one pointer event.
        let mut app = app();
        app.brush.radius = 1.0;
        press(&mut app, centre_of_viewport());

        app.on_pointer(PointerEvent::Moved {
            position: centre_of_viewport() + Vector::new(80.0, 0.0),
            size: SIZE,
        });
        assert!(
            app.perf.stamps > 1,
            "one long pointer move should interpolate, got {} stamps",
            app.perf.stamps
        );
    }

    #[test]
    fn symmetry_sculpts_both_sides_at_once() {
        let mut app = app();
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::X));
        // Nothing has told the application how big the viewport is yet, and
        // the ray depends on it.
        app.viewport_size = Vec2::new(SIZE.x, SIZE.y);

        // Aim off to one side so the mirrored half lands somewhere distinct.
        let off_centre = Vector::new(SIZE.x * 0.38, SIZE.y * 0.5);
        let hit = app
            .surface_under(Vec2::new(off_centre.x, off_centre.y))
            .expect("the test needs a point that is on the model");
        let mirrored = Vec3::new(-hit.x, hit.y, hit.z);
        let before = app.volume.sample_world(mirrored);

        press(&mut app, off_centre);
        release(&mut app);

        assert!(
            app.volume.sample_world(mirrored) < before,
            "the mirrored half of the stroke never landed"
        );
    }

    /// The ZBrush convention, and the biggest ergonomic win in this round.
    #[test]
    fn holding_shift_sculpts_with_smooth_without_changing_the_selection() {
        let mut app = app();
        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        assert_eq!(app.effective_brush().kind, BrushKind::Draw);

        update(
            &mut app,
            Message::Pointer(PointerEvent::Modifiers { shift: true, control: false, alt: false }),
        );
        assert_eq!(app.effective_brush().kind, BrushKind::Smooth, "shift should smooth");
        assert_eq!(
            app.brush.kind,
            BrushKind::Draw,
            "the selection itself must not change, or the tool strip would flicker \
             and a key released out of focus would strand the wrong brush"
        );

        update(
            &mut app,
            Message::Pointer(PointerEvent::Modifiers { shift: false, control: false, alt: false }),
        );
        assert_eq!(app.effective_brush().kind, BrushKind::Draw, "releasing shift should restore");
    }

    #[test]
    fn holding_shift_actually_smooths_the_model_rather_than_drawing_on_it() {
        let mut app = app();
        // A move first, so the application knows the viewport size and
        // `surface_under` agrees with where a press will actually land. The
        // default size is not SIZE, and disagreeing here probes a point the
        // brush never touches.
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        let probe = app
            .surface_under(Vec2::new(SIZE.x / 2.0, SIZE.y / 2.0))
            .expect("the centre of the viewport should hit the sphere");
        let flat = app.volume.sample_world(probe);

        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        for _ in 0..4 {
            press(&mut app, centre_of_viewport());
            release(&mut app);
        }
        let raised = app.volume.sample_world(probe);
        assert!(raised < flat, "drawing should have pushed the surface out past the probe");

        // Now the same gesture with shift held. Nothing about the selection
        // changes, but the strokes must smooth rather than pile on more.
        update(
            &mut app,
            Message::Pointer(PointerEvent::Modifiers { shift: true, control: false, alt: false }),
        );
        for _ in 0..12 {
            press(&mut app, centre_of_viewport());
            release(&mut app);
        }
        let smoothed = app.volume.sample_world(probe);

        assert_ne!(smoothed, raised, "holding shift changed nothing at all");
        assert!(
            smoothed > raised,
            "shift should have flattened the bump back, not built on it: \
             {flat} flat, {raised} raised, {smoothed} after smoothing"
        );
        assert_eq!(app.brush.kind, BrushKind::Draw, "the selection must be untouched");
    }

    #[test]
    fn a_fresh_application_starts_with_every_mirror_plane_off() {
        let app = app();
        assert!(app.symmetry.is_off(), "symmetry was {:?}", app.symmetry);
        for axis in MirrorAxis::ALL {
            assert!(!app.symmetry.axis(axis), "{} was on at startup", axis.label());
        }
    }

    #[test]
    fn each_mirror_axis_reaches_the_opposite_side() {
        for (axis, probe) in [
            (MirrorAxis::X, Vec3::new(-1.0, 1.0, 1.0)),
            (MirrorAxis::Y, Vec3::new(1.0, -1.0, 1.0)),
            (MirrorAxis::Z, Vec3::new(1.0, 1.0, -1.0)),
        ] {
            let mut app = app();
            update(&mut app, Message::SymmetryAxisToggled(axis));
            assert!(app.symmetry.axis(axis), "{} did not turn on", axis.label());

            // On the surface, and off all three planes, so only the axis
            // under test can carry the stroke to the probe point.
            let at = Vec3::new(14.0, 14.0, 18.0).normalize() * MODEL_RADIUS_MM;
            let normal = app.volume.gradient_world(at);
            let before = app.volume.sample_world(at * probe);

            app.volume.begin_stroke();
            app.brush.apply_symmetric(
                &mut app.volume,
                &Stamp::new(at, normal, BrushDirection::Add),
                app.symmetry,
                &mut app.brush_scratch,
            );

            assert!(
                app.volume.sample_world(at * probe) < before,
                "{} symmetry never reached the other side",
                axis.label()
            );

            // ...and toggling it again turns it back off.
            update(&mut app, Message::SymmetryAxisToggled(axis));
            assert!(!app.symmetry.axis(axis));
        }
    }

    /// The fast path for radius in every sculpting tool is a drag, not a
    /// slider. Absolute against where the drag began, so out and back returns
    /// to exactly the original value rather than drifting.
    #[test]
    fn holding_the_sizing_key_and_dragging_scales_the_radius_reversibly() {
        let mut app = app();
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        let original = app.brush.radius;

        update(&mut app, Message::SizingStarted(SizingTarget::Radius));
        assert!(app.sizing.is_some(), "the gesture did not start");

        let at = |dx: f32| Vector::new(SIZE.x / 2.0 + dx, SIZE.y / 2.0);
        app.on_pointer(PointerEvent::Moved { position: at(120.0), size: SIZE });
        let grown = app.brush.radius;
        assert!(grown > original, "dragging right should grow the brush");

        app.on_pointer(PointerEvent::Moved { position: at(-120.0), size: SIZE });
        assert!(app.brush.radius < original, "dragging left should shrink it");

        // Back where it started: absolute, not accumulated.
        app.on_pointer(PointerEvent::Moved { position: at(0.0), size: SIZE });
        assert!(
            (app.brush.radius - original).abs() < 1.0e-4,
            "the gesture drifted: {original} became {}",
            app.brush.radius
        );

        update(&mut app, Message::SizingEnded);
        assert!(app.sizing.is_none());
        // ...and the pointer goes back to sculpting.
        app.on_pointer(PointerEvent::Moved { position: at(200.0), size: SIZE });
        assert!((app.brush.radius - original).abs() < 1.0e-4, "sizing outlived its key");
    }

    /// The failure this would otherwise cause is the worst kind: a drag that
    /// silently carves the model while the user thinks they are resizing.
    #[test]
    fn a_sizing_drag_does_not_sculpt() {
        let mut app = app();
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        let probe = app
            .surface_under(Vec2::new(SIZE.x / 2.0, SIZE.y / 2.0))
            .expect("the centre should hit the sphere");
        let before = app.volume.sample_world(probe);

        update(&mut app, Message::SizingStarted(SizingTarget::Radius));
        press(&mut app, centre_of_viewport());
        for step in 1..=8 {
            app.on_pointer(PointerEvent::Moved {
                position: Vector::new(SIZE.x / 2.0 + step as f32 * 10.0, SIZE.y / 2.0),
                size: SIZE,
            });
        }
        release(&mut app);

        assert_eq!(app.volume.sample_world(probe), before, "a sizing drag cut into the model");
        assert!(!app.history.can_undo(), "a sizing drag recorded an undo entry");
    }

    #[test]
    fn the_sizing_gesture_stays_inside_the_slider_bounds() {
        for (target, low, high) in [
            (SizingTarget::Radius, MIN_RADIUS_MM, MAX_RADIUS_MM),
            (SizingTarget::Strength, MIN_STRENGTH, MAX_STRENGTH),
        ] {
            for direction in [-1.0f32, 1.0] {
                let mut app = app();
                app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
                update(&mut app, Message::SizingStarted(target));
                app.on_pointer(PointerEvent::Moved {
                    position: Vector::new(SIZE.x / 2.0 + direction * 5000.0, SIZE.y / 2.0),
                    size: SIZE,
                });
                let value = match target {
                    SizingTarget::Radius => app.brush.radius,
                    SizingTarget::Strength => app.brush.strength,
                };
                assert!((low..=high).contains(&value), "{target:?} left its range at {value}");
            }
        }
    }

    /// ZBrush calls this Dynamic. Without it a brush tuned on one model is the
    /// wrong size the moment the model changes scale.
    /// The whole point of the format: sculpt, save, reopen, and get the same
    /// thing back. Driven through the application's own open and save rather
    /// than through the format directly, because the interesting failures are
    /// in the wiring -- a stale mesh pool, history that outlives its model, a
    /// camera that does not follow.
    #[test]
    fn a_sculpt_survives_a_save_and_a_reopen() {
        let directory =
            std::env::temp_dir().join(format!("brokkr-roundtrip-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("sculpt.brokkr");

        let mut app = app();
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        let probe = app
            .surface_under(Vec2::new(SIZE.x / 2.0, SIZE.y / 2.0))
            .expect("the centre should hit the sphere");

        // Make something worth keeping, and move the camera and settings so the
        // session state has something to prove.
        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        for _ in 0..4 {
            press(&mut app, centre_of_viewport());
            release(&mut app);
        }
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::Y));
        update(&mut app, Message::BrushRadiusChanged(5.5));
        app.camera.yaw = 1.25;
        let sculpted = app.volume.sample_world(probe);
        let expected_bricks: usize = app.volume.brick_coords().count();

        app.save_project(&path);
        assert!(app.status.starts_with("saved"), "save reported: {}", app.status);
        assert_eq!(app.project_path.as_deref(), Some(path.as_path()));

        // A different session entirely, as reopening after quitting would be.
        let mut reopened = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        reopened.open_project(&path);
        assert!(reopened.status.starts_with("opened"), "open reported: {}", reopened.status);

        assert_eq!(reopened.volume.sample_world(probe), sculpted, "the field came back different");
        assert_eq!(reopened.volume.brick_coords().count(), expected_bricks);
        assert!((reopened.camera.yaw - 1.25).abs() < 1.0e-5, "the camera did not follow");
        assert!((reopened.brush.radius - 5.5).abs() < 1.0e-5, "the brush did not follow");
        assert!(reopened.symmetry.axis(MirrorAxis::Y), "the mirror planes did not follow");

        // And it was drawn and is printable, which is what "loaded" means.
        // Note `open_project` meshes immediately, so the dirty set is already
        // drained by now -- what proves it happened is how much it meshed.
        assert!(
            reopened.perf.dirty_bricks > 0,
            "the reopened model was never meshed, so it would load into an empty screen"
        );
        let (_, report) = reopened.volume.export_mesh();
        assert!(report.is_printable(), "the reopened model is not printable: {}", report.summary());

        // History belongs to the model that went away.
        assert!(!reopened.history.can_undo(), "undo outlived the model it belonged to");

        std::fs::remove_dir_all(&directory).ok();
    }

    // --- the unsaved marker --------------------------------------------------
    //
    // Quitting with unsaved work used to lose it silently. These pin the flag
    // that the confirm prompt reads, and in particular the three places it must
    // NOT be set, since a flag that is always on is the same as no flag at all.

    #[test]
    fn a_stroke_marks_the_document_unsaved_and_a_save_clears_it() {
        let directory = std::env::temp_dir().join(format!("brokkr-unsaved-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("sculpt.brokkr");

        let mut app = app();
        assert!(!app.unsaved, "a freshly opened sphere is not unsaved work");

        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert!(app.unsaved, "a stroke did not mark the document unsaved");

        app.save_project(&path);
        assert!(app.status.starts_with("saved"), "save reported: {}", app.status);
        assert!(!app.unsaved, "a successful save did not clear the marker");

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A press and release that never met the model produces no undo entry, so
    /// it must produce no prompt either. Without the guard inside
    /// `finish_stroke` every stray click in empty space would arm the marker.
    #[test]
    fn a_press_that_misses_the_model_does_not_mark_it_unsaved() {
        let mut app = app();
        press(&mut app, Vector::new(4.0, 4.0));
        release(&mut app);
        assert!(
            !app.unsaved,
            "a click into empty space armed the unsaved marker, so every stray \
             click would raise a discard prompt"
        );
        assert!(
            !app.history.can_undo(),
            "the corner of the viewport hit the model after all, so this test \
             proves nothing -- move the press"
        );

        // The control: the same gesture at the centre does arm it, so the
        // assertion above is about the miss and not about a press path that
        // silently does nothing.
        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert!(app.unsaved, "the control press missed too, so the test is vacuous");
    }

    /// A write that failed did not save anything, so the marker has to survive
    /// it. Clearing it in the wrong place lets the next quit go through with no
    /// prompt and no file.
    #[test]
    fn a_failed_save_leaves_it_unsaved() {
        let mut app = app();
        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert!(app.unsaved);

        let nowhere = std::env::temp_dir()
            .join(format!("brokkr-absent-{}", std::process::id()))
            .join("no")
            .join("such")
            .join("directory")
            .join("sculpt.brokkr");
        app.save_project(&nowhere);

        assert!(app.status.contains("could not"), "a failed save reported: {}", app.status);
        assert!(app.unsaved, "a failed save cleared the unsaved marker");
        assert!(app.project_path.is_none(), "a failed save claimed the file");
    }

    /// The puck's buttons call `undo`/`redo` directly rather than sending
    /// `Message::Undo`, so the marker has to be set in the methods. Calling them
    /// the way `drive_from_spacemouse` does is what proves that.
    #[test]
    fn undo_and_redo_mark_it_unsaved_through_the_direct_call_path() {
        let mut app = app();
        press(&mut app, centre_of_viewport());
        release(&mut app);
        app.unsaved = false;

        app.undo();
        assert!(app.unsaved, "undo did not mark the document unsaved");

        app.unsaved = false;
        app.redo();
        assert!(app.unsaved, "redo did not mark the document unsaved");
    }

    /// Undo with nothing to undo changes nothing, so it must not arm the marker
    /// either -- the flag lives inside the `history.undo` guard for this.
    #[test]
    fn undo_with_an_empty_history_does_not_mark_it_unsaved() {
        let mut app = app();
        app.undo();
        assert!(!app.unsaved, "an undo that did nothing armed the unsaved marker");
    }

    /// `resample` returns early when the size is already in use. The marker sits
    /// past that return, so the no-op stays a no-op.
    #[test]
    fn resampling_to_the_current_size_leaves_the_flag_alone() {
        let mut app = app();
        app.resample(app.voxel_size);
        assert!(!app.unsaved, "a resample that did nothing armed the unsaved marker");

        app.resample(app.voxel_size * 2.0);
        assert!(app.unsaved, "a real resample did not mark the document unsaved");
    }

    #[test]
    fn the_title_carries_the_file_name_and_the_unsaved_star() {
        let mut app = app();
        assert_eq!(app.title(), "untitled — BrokkrSculpt");

        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert_eq!(app.title(), "untitled* — BrokkrSculpt");

        app.project_path = Some(std::path::PathBuf::from("/tmp/dragon.brokkr"));
        assert_eq!(app.title(), "dragon.brokkr* — BrokkrSculpt");

        app.unsaved = false;
        assert_eq!(app.title(), "dragon.brokkr — BrokkrSculpt");
    }

    // --- the confirm-before-discard prompt -----------------------------------

    /// An app with a stroke in it and nowhere to save it, which is the state
    /// every one of these is about.
    fn app_with_unsaved_work() -> Brokkr {
        let mut app = app();
        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert!(app.unsaved, "the fixture did not actually dirty the document");
        app
    }

    #[test]
    fn new_with_unsaved_work_raises_the_prompt_and_does_not_reset() {
        let mut app = app_with_unsaved_work();
        let sculpted = app.volume.brick_coords().count();

        update(&mut app, Message::NewSculpt);

        assert_eq!(app.confirm, Some(PendingAction::NewSculpt), "no prompt was raised");
        assert!(app.unsaved, "the document was reset behind the prompt");
        assert_eq!(
            app.volume.brick_coords().count(),
            sculpted,
            "New threw the sculpt away before the user had answered"
        );
    }

    /// The panel's Reset sphere button shares an arm with File > New, so it
    /// discards the document identically and must ask identically.
    #[test]
    fn reset_sphere_asks_as_well_since_it_shares_the_arm() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::ResetSphere);
        assert_eq!(app.confirm, Some(PendingAction::NewSculpt));
    }

    #[test]
    fn quitting_with_unsaved_work_raises_the_prompt() {
        let mut app = app_with_unsaved_work();
        let id = iced::window::Id::unique();
        update(&mut app, Message::CloseRequested(id));
        assert_eq!(
            app.confirm,
            Some(PendingAction::Quit(id)),
            "the close request was not intercepted, so quitting still loses work"
        );
    }

    /// With nothing to lose, every gated action must behave exactly as it did
    /// before the gate existed.
    #[test]
    fn a_clean_document_is_not_prompted_at_all() {
        let mut app = app();
        update(&mut app, Message::NewSculpt);
        assert!(app.confirm.is_none(), "a clean document was prompted");

        update(&mut app, Message::CloseRequested(iced::window::Id::unique()));
        assert!(app.confirm.is_none(), "a clean document was prompted on quit");

        update(&mut app, Message::OpenRequested);
        assert!(app.confirm.is_none(), "a clean document was prompted on open");
    }

    /// The prompt draws over the viewport but the shader widget still sees
    /// events wherever the cursor is. A press behind the card must neither
    /// dismiss the prompt nor sculpt into a document about to be discarded.
    #[test]
    fn a_viewport_press_does_not_dismiss_the_prompt_or_sculpt() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::NewSculpt);
        let before = app.volume.sample_world(Vec3::ZERO);
        let entries = app.history_stats.undo_entries;

        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.confirm, Some(PendingAction::NewSculpt), "a stray click dismissed it");
        assert_eq!(app.volume.sample_world(Vec3::ZERO), before, "it sculpted behind the prompt");
        assert_eq!(
            app.history_stats.undo_entries, entries,
            "it recorded a stroke behind the prompt"
        );
    }

    /// Escape answers Cancel. It must not fall through to the menu clears,
    /// which would drop the prompt without answering it.
    #[test]
    fn escape_cancels_the_prompt_rather_than_dropping_it() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::NewSculpt);

        update(&mut app, Message::MenuClosed);

        assert!(app.confirm.is_none(), "escape left the prompt up");
        assert!(app.unsaved, "escape discarded the work it was asking about");
        assert!(app.history.can_undo(), "escape reset the sculpt");
    }

    #[test]
    fn discard_runs_the_pending_action() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::NewSculpt);
        update(&mut app, Message::ConfirmAnswered(ConfirmChoice::Discard));

        assert!(app.confirm.is_none());
        assert!(!app.unsaved, "the fresh sphere is not unsaved work");
        assert!(!app.history.can_undo(), "the reset did not happen");
    }

    #[test]
    fn cancel_leaves_everything_exactly_as_it_was() {
        let mut app = app_with_unsaved_work();
        let bricks = app.volume.brick_coords().count();
        update(&mut app, Message::NewSculpt);
        update(&mut app, Message::ConfirmAnswered(ConfirmChoice::Cancel));

        assert!(app.confirm.is_none());
        assert!(app.unsaved, "cancel cleared the unsaved marker");
        assert_eq!(app.volume.brick_coords().count(), bricks, "cancel reset the sculpt anyway");
    }

    /// Save-then-continue, for a document that already has a file.
    #[test]
    fn saving_from_the_prompt_writes_the_file_and_then_acts() {
        let directory =
            std::env::temp_dir().join(format!("brokkr-confirm-save-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("sculpt.brokkr");

        let mut app = app_with_unsaved_work();
        app.save_project(&path);
        assert!(!app.unsaved);

        // Dirty it again so the prompt has something to be about.
        press(&mut app, Vector::new(SIZE.x / 2.0 + 12.0, SIZE.y / 2.0));
        release(&mut app);
        assert!(app.unsaved);

        update(&mut app, Message::NewSculpt);
        update(&mut app, Message::ConfirmAnswered(ConfirmChoice::Save));

        assert!(app.confirm.is_none(), "the prompt stayed up after a good save");
        assert!(!app.history.can_undo(), "the pending New did not run after the save");
        // And the file on disk is the sculpt as it was, not the fresh sphere.
        let mut reopened = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        reopened.open_project(&path);
        assert!(reopened.status.starts_with("opened"), "open reported: {}", reopened.status);

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A save that fails must not be treated as a save. Continuing here would
    /// quit on a file that was never written, which is the exact loss the
    /// prompt exists to prevent.
    #[test]
    fn a_failed_save_from_the_prompt_does_not_continue() {
        let mut app = app_with_unsaved_work();
        app.project_path = Some(
            std::env::temp_dir()
                .join(format!("brokkr-absent-{}", std::process::id()))
                .join("nope")
                .join("sculpt.brokkr"),
        );

        update(&mut app, Message::NewSculpt);
        update(&mut app, Message::ConfirmAnswered(ConfirmChoice::Save));

        assert!(app.status.contains("could not"), "reported: {}", app.status);
        assert_eq!(
            app.confirm,
            Some(PendingAction::NewSculpt),
            "the prompt was dismissed by a save that did not happen"
        );
        assert!(app.unsaved, "a failed save cleared the marker");
        assert!(app.history.can_undo(), "the sculpt was discarded on a failed save");
    }

    /// Dismissing the file dialog is not an answer to the prompt.
    #[test]
    fn dismissing_the_save_dialog_leaves_the_prompt_up() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::NewSculpt);
        update(&mut app, Message::SavedThenContinue(None));

        assert_eq!(app.confirm, Some(PendingAction::NewSculpt));
        assert!(app.history.can_undo(), "the sculpt was discarded");
    }

    // --- the recent list and the crash net -----------------------------------

    /// A scratch directory for one test's recent list and autosave, so nothing
    /// here can reach the real ones.
    fn scratch(name: &str) -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!("brokkr-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn saving_and_opening_both_record_into_the_recent_list() {
        let directory = scratch("recent-record");
        let path = directory.join("sculpt.brokkr");

        let mut app = app_with_unsaved_work();
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));
        app.save_project(&path);
        assert_eq!(app.recent.paths().len(), 1, "a save did not record the file");

        let mut reopened = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        reopened.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));
        reopened.open_project(&path);
        assert!(reopened.status.starts_with("opened"), "open reported: {}", reopened.status);
        assert_eq!(reopened.recent.paths()[0], std::path::absolute(&path).unwrap());

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A file that has been moved or deleted should stop being offered, rather
    /// than sitting in the menu failing every time it is clicked.
    #[test]
    fn opening_a_missing_recent_file_drops_it_from_the_list() {
        let directory = scratch("recent-missing");
        let gone = directory.join("gone.brokkr");

        let mut app = app();
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));
        app.recent.record(&gone);
        assert_eq!(app.recent.paths().len(), 1);

        app.open_project(&gone);
        assert!(app.status.contains("could not"), "reported: {}", app.status);
        assert!(app.recent.is_empty(), "a file that cannot be opened stayed in the list");

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn the_autosave_round_trips_but_is_not_a_save() {
        let directory = scratch("autosave");
        let mut app = app_with_unsaved_work();
        app.autosave_file = Some(directory.join("autosave.brokkr"));
        let probe = Vec3::ZERO;
        let sculpted = app.volume.sample_world(probe);

        app.write_autosave();

        assert!(app.has_autosave(), "nothing was written");
        assert!(app.unsaved, "the autosave cleared the unsaved marker, so quitting would not ask");
        assert!(app.project_path.is_none(), "the autosave claimed to be the document's file");

        let mut recovered = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        recovered.autosave_file = app.autosave_file.clone();
        recovered.recover_autosave();

        assert_eq!(recovered.volume.sample_world(probe), sculpted, "the field came back different");
        assert!(
            recovered.project_path.is_none(),
            "a recovered autosave adopted its own path, so the next Save would write back into \
             the crash net instead of asking for a real file"
        );
        assert!(recovered.unsaved, "a recovered autosave must still want saving somewhere real");
        assert!(recovered.recent.is_empty(), "the crash net was listed as a recent document");

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn an_explicit_save_throws_the_crash_net_away() {
        let directory = scratch("autosave-cleared");
        let mut app = app_with_unsaved_work();
        app.autosave_file = Some(directory.join("autosave.brokkr"));
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));
        app.write_autosave();
        assert!(app.has_autosave());

        app.save_project(&directory.join("sculpt.brokkr"));
        assert!(app.status.starts_with("saved"), "save reported: {}", app.status);
        assert!(
            !app.has_autosave(),
            "a stale crash net survived a real save, so File > Recover would offer to restore \
             something older than the file on disk"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    // --- the plane cut -------------------------------------------------------

    /// Which side of the dragged line is removed, pinned by observation rather
    /// than by reasoning about cross products on paper.
    ///
    /// This is the one thing about the cut that cannot be got right by thinking
    /// about it: the sign depends on the ray order, the handedness of the
    /// camera basis, and whether the pixel-to-NDC step flips Y. Get it backwards
    /// and the tool takes the half the user meant to keep -- which is
    /// destructive, and only obvious after the fact.
    #[test]
    fn a_left_to_right_drag_removes_a_consistent_side() {
        let mut app = app();
        // A known camera: looking straight down -Z at the origin.
        app.camera.yaw = 0.0;
        app.camera.pitch = 0.0;
        app.publish_camera();

        let middle_y = SIZE.y / 2.0;
        update(&mut app, Message::CutToggled);
        assert!(app.cut_armed, "the cut did not arm");

        // Drag left to right across the middle of the viewport.
        press(&mut app, Vector::new(SIZE.x * 0.1, middle_y));
        app.on_pointer(PointerEvent::Moved {
            position: Vector::new(SIZE.x * 0.9, middle_y),
            size: SIZE,
        });
        release(&mut app);

        assert!(!app.cut_armed, "the cut stayed armed after being used");
        assert!(app.unsaved, "a cut did not mark the document unsaved");

        let above = app.volume.sample_world(Vec3::new(0.0, 12.0, 0.0));
        let below = app.volume.sample_world(Vec3::new(0.0, -12.0, 0.0));
        assert_ne!(
            above < 0.0,
            below < 0.0,
            "the cut removed both halves or neither: above {above}, below {below}"
        );
        // Record which one it actually is, so a change of sign fails here
        // rather than in someone's sculpt. If this assertion is what fails
        // after a camera change, check the ray order in `finish_cut` before
        // editing the expectation.
        assert!(
            below < 0.0 && above > 0.0,
            "a left to right drag should keep the LOWER half on screen: \
             above {above}, below {below}"
        );
    }

    /// A cut is destructive, so a click must never be one.
    #[test]
    fn a_click_with_the_cut_armed_does_nothing() {
        let mut app = app();
        update(&mut app, Message::CutToggled);
        let before = app.volume.brick_coords().count();

        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.volume.brick_coords().count(), before, "a click cut the model");
        assert!(!app.unsaved, "a click that cut nothing marked the document unsaved");
        assert!(app.status.contains("cancelled"), "reported: {}", app.status);
    }

    #[test]
    fn a_cut_is_undoable_through_the_application() {
        let mut app = app();
        let probe = Vec3::new(0.0, 12.0, 0.0);
        let before = app.volume.sample_world(probe);

        update(&mut app, Message::CutToggled);
        press(&mut app, Vector::new(SIZE.x * 0.1, SIZE.y / 2.0));
        app.on_pointer(PointerEvent::Moved {
            position: Vector::new(SIZE.x * 0.9, SIZE.y / 2.0),
            size: SIZE,
        });
        release(&mut app);
        assert_ne!(app.volume.sample_world(probe), before, "the cut did nothing to undo");

        app.undo();
        assert_eq!(app.volume.sample_world(probe), before, "undo did not restore the cut");
    }

    /// Escape is the way out of every other mode, and a destructive one must
    /// not be the exception.
    #[test]
    fn escape_disarms_the_cut() {
        let mut app = app();
        update(&mut app, Message::CutToggled);
        assert!(app.cut_armed);
        update(&mut app, Message::MenuClosed);
        assert!(!app.cut_armed, "escape left a destructive mode armed");
    }

    // --- importing a mesh ----------------------------------------------------

    /// Drives the import through the application rather than the voxeliser,
    /// because the interesting failures are all in the wiring: a stale mesh
    /// pool, history outliving its model, and above all a document that thinks
    /// it belongs to the STL it came from.
    #[test]
    fn importing_a_mesh_replaces_the_model_without_claiming_its_file() {
        let directory = scratch("import");
        let path = directory.join("cube.stl");

        // Write a real STL through this project's own writer, so the test
        // exercises the reader rather than an assumption about the format.
        let mut source = brokkr_core::Volume::new(0.5);
        source.seed_sphere(Vec3::ZERO, 14.0);
        source.mark_everything_dirty();
        let (mesh, report) = source.export_mesh();
        assert!(report.is_printable(), "the fixture is not printable");
        let mut bytes = Vec::new();
        brokkr_core::export::stl::write(&mesh, &mut bytes).unwrap();
        std::fs::write(&path, &bytes).unwrap();

        let mut app = app();
        app.project_path = Some(directory.join("previous.brokkr"));
        press(&mut app, centre_of_viewport());
        release(&mut app);

        let imported = brokkr_core::import::read_path(&path).expect("the STL should read");
        let options = brokkr_core::voxelise::VoxeliseOptions::at(app.voxel_size);
        let (volume, voxel_report) =
            brokkr_core::voxelise::voxelise(&imported, &options).expect("it should voxelise");
        app.adopt_import(crate::message::Imported {
            volume,
            report: voxel_report,
            source: path.clone(),
            elapsed_ms: 0.0,
        });

        assert!(
            app.project_path.is_none(),
            "the import adopted the mesh's path, so the next plain Save would write a .brokkr \
             container straight over the user's .stl"
        );
        assert!(app.unsaved, "an imported model is unsaved work and quitting would lose it");
        assert!(!app.history.can_undo(), "history outlived the model it belonged to");
        assert!(app.perf.dirty_bricks > 0, "nothing was meshed, so the import is invisible");
        assert!(app.status.starts_with("imported"), "reported: {}", app.status);

        let (_, after) = app.volume.export_mesh();
        assert!(after.is_printable(), "the imported model is not printable: {}", after.summary());

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A file that cannot be read must leave the sculpt on screen alone, and
    /// must say so in a way the header renders as an error.
    #[test]
    fn a_mesh_that_cannot_be_read_reports_and_changes_nothing() {
        let directory = scratch("import-bad");
        let path = directory.join("broken.stl");
        std::fs::write(&path, b"this is not an STL of any flavour").unwrap();

        let mut app = app();
        press(&mut app, centre_of_viewport());
        release(&mut app);
        let before = app.volume.sample_world(Vec3::ZERO);

        let payload = crate::message::ImportPayload::new(
            brokkr_core::import::read_path(&path).map(|_| unreachable!("it should not parse")),
        );
        update(&mut app, Message::ImportLoaded(payload));

        assert!(
            app.status.contains("could not"),
            "the failure does not contain the substring the header colours as an error: {}",
            app.status
        );
        assert_eq!(
            app.volume.sample_world(Vec3::ZERO),
            before,
            "a failed import changed the model"
        );
        assert!(app.history.can_undo(), "a failed import cleared history");

        std::fs::remove_dir_all(&directory).ok();
    }

    /// Import discards the document, so it goes through the same gate as Open.
    #[test]
    fn importing_with_unsaved_work_raises_the_prompt() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::ImportRequested);
        assert_eq!(app.confirm, Some(PendingAction::Import));
    }

    #[test]
    fn the_autosave_tick_waits_for_the_interval_and_for_the_stroke_to_end() {
        let directory = scratch("autosave-tick");
        let mut app = app();
        app.autosave_file = Some(directory.join("autosave.brokkr"));

        // Clean document: nothing to protect, so nothing is written however
        // long it has been.
        app.last_autosave = Instant::now() - Duration::from_secs(600);
        app.maybe_autosave();
        assert!(!app.has_autosave(), "a clean document was autosaved");

        // Dirty, but the interval has not elapsed.
        press(&mut app, centre_of_viewport());
        release(&mut app);
        app.last_autosave = Instant::now();
        app.maybe_autosave();
        assert!(!app.has_autosave(), "it autosaved before the interval was up");

        // Dirty and overdue, but mid-drag: a write here reads as a stutter in
        // the brush.
        app.last_autosave = Instant::now() - Duration::from_secs(600);
        press(&mut app, centre_of_viewport());
        app.maybe_autosave();
        assert!(!app.has_autosave(), "it autosaved in the middle of a stroke");

        // Released, dirty and overdue.
        release(&mut app);
        app.maybe_autosave();
        assert!(app.has_autosave(), "an overdue autosave never happened");

        std::fs::remove_dir_all(&directory).ok();
    }
    #[test]
    fn a_bad_file_reports_and_changes_nothing() {
        let directory = std::env::temp_dir().join(format!("brokkr-bad-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("not-a-sculpt.brokkr");
        std::fs::write(&path, b"this is not a sculpt").unwrap();

        let mut app = app();
        let before = app.volume.brick_coords().count();
        app.open_project(&path);

        assert!(app.status.contains("not a BrokkrSculpt file"), "reported: {}", app.status);
        assert_eq!(app.volume.brick_coords().count(), before, "a failed open lost the model");
        assert!(app.project_path.is_none(), "a failed open claimed the file");

        app.open_project(&directory.join("does-not-exist.brokkr"));
        assert!(app.status.contains("could not open"), "reported: {}", app.status);
        assert_eq!(app.volume.brick_coords().count(), before);

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn a_dynamic_radius_keeps_its_proportion_to_the_model() {
        let proportion = |state: &Brokkr| state.brush.radius / state.model_radius;

        // Off by default: the radius means a physical size, so a resample must
        // leave it alone.
        let mut fixed = app();
        let before = fixed.brush.radius;
        update(&mut fixed, Message::Resample(VOXEL_SIZE_MM / 2.0));
        assert_eq!(fixed.brush.radius, before, "a fixed radius must survive a resample");

        // On, and the model doubles: the brush should follow it.
        let mut dynamic = app();
        update(&mut dynamic, Message::DynamicRadiusToggled(true));
        let before = proportion(&dynamic);
        dynamic.volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM * 2.0);
        dynamic.volume.mark_everything_dirty();
        dynamic.remesh_dirty();

        assert!(
            dynamic.brush.radius > fixed.brush.radius,
            "a dynamic radius should have grown with the model"
        );
        assert!(
            (proportion(&dynamic) - before).abs() < 0.15 * before,
            "the brush lost its proportion: {before} became {}",
            proportion(&dynamic)
        );
    }

    /// Clicking the cube must move the camera and must NOT sculpt. Getting that
    /// wrong takes a divot out of the model every time someone reaches for a
    /// standard view.
    #[test]
    fn clicking_the_navigation_cube_flies_the_camera_without_sculpting() {
        let mut app = app();
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        let probe = app
            .surface_under(Vec2::new(SIZE.x / 2.0, SIZE.y / 2.0))
            .expect("the centre should hit the sphere");
        let before_field = app.volume.sample_world(probe);
        let before_yaw = app.camera.yaw;

        let (origin, size) = crate::navcube::corner_rect(Vec2::new(SIZE.x, SIZE.y));
        let middle = origin + size * 0.5;
        press(&mut app, Vector::new(middle.x, middle.y));
        release(&mut app);

        assert!(app.flight.is_some(), "clicking the cube did not start a move");
        assert_eq!(
            app.volume.sample_world(probe),
            before_field,
            "clicking the cube carved the model"
        );
        assert!(!app.history.can_undo(), "clicking the cube recorded an undo entry");
        assert!(app.drag.is_none(), "clicking the cube started a drag");

        finish_flight(&mut app);
        assert!(app.flight.is_none(), "the move never finished");
        assert_ne!(app.camera.yaw, before_yaw, "the camera did not actually move");
        assert!(app.camera.view().to_cols_array().iter().all(|v| v.is_finite()));
    }

    /// A click on Top asks for a pitch of ninety degrees, which is exactly where
    /// the view matrix collapses. The flight interpolates the field directly, so
    /// it has to respect the same limit the drag path does.
    #[test]
    fn flying_to_a_pole_stops_short_of_collapsing_the_view() {
        for direction in [Vec3::Y, Vec3::NEG_Y] {
            let mut app = app();
            app.fly_to(crate::navcube::Part { direction, extremes: 1 });
            finish_flight(&mut app);
            assert!(app.flight.is_none());
            assert!(
                app.camera.pitch.abs() < std::f32::consts::FRAC_PI_2,
                "{direction:?} reached the pole: pitch {}",
                app.camera.pitch
            );
            assert!(app.camera.view().to_cols_array().iter().all(|v| v.is_finite()));
            assert!((app.camera.right().length() - 1.0).abs() < 1.0e-4);
        }
    }

    /// The camera must not spin several times round to reach a heading a few
    /// degrees away, which is what interpolating an unwrapped yaw would do.
    #[test]
    fn a_flight_takes_the_short_way_round_and_unwinds_any_roll() {
        let mut app = app();
        // Many turns of yaw, and a roll from the puck, as a session would
        // accumulate.
        app.camera.yaw = std::f32::consts::TAU * 3.0 + 0.2;
        app.camera.roll = 0.7;

        app.fly_to(crate::navcube::Part { direction: Vec3::Z, extremes: 1 });
        let flight = app.flight.expect("no flight started");
        let travel = (flight.to.0 - flight.from.0).abs();
        assert!(travel <= std::f32::consts::PI + 1.0e-4, "the flight would spin {travel} radians");

        finish_flight(&mut app);
        assert!(
            app.camera.roll.abs() < 1.0e-4,
            "a standard view should be level, roll was {}",
            app.camera.roll
        );
    }

    /// The flight is driven off the frame tick, so that link needs its own
    /// test: the tests above supply their own time to get a result at all.
    #[test]
    fn the_frame_tick_is_what_advances_a_flight() {
        let mut app = app();
        app.fly_to(crate::navcube::Part { direction: Vec3::X, extremes: 1 });
        // The first frame has no previous one to measure against, so it
        // contributes nothing; the second is the one that moves.
        update(&mut app, Message::Frame);
        update(&mut app, Message::Frame);
        let elapsed = app.flight.expect("the flight ended early").elapsed_ms;
        assert!(elapsed > 0.0, "the frame tick did not advance the flight");
    }

    fn right_press(app: &mut Brokkr, at: Vector) {
        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Right,
            position: at,
            size: SIZE,
        });
    }

    fn right_release(app: &mut Brokkr) {
        app.on_pointer(PointerEvent::Released { button: PointerButton::Right });
    }

    /// Right drag already orbits, so the menu has to be told apart from it by
    /// movement and time. All three outcomes matter, and getting any of them
    /// wrong makes either the menu or the orbit unusable.
    #[test]
    fn a_right_click_opens_the_menu_but_a_right_drag_orbits() {
        let centre = centre_of_viewport();
        let moved_to = |dx: f32| Vector::new(SIZE.x / 2.0 + dx, SIZE.y / 2.0);

        // A click that does not move: the menu.
        let mut click = app();
        right_press(&mut click, centre);
        right_release(&mut click);
        assert!(click.menu.is_some(), "a right click did not open the menu");

        // A drag: orbits, and opens nothing.
        let mut orbit = app();
        let before = orbit.camera.yaw;
        right_press(&mut orbit, centre);
        orbit.on_pointer(PointerEvent::Moved { position: moved_to(60.0), size: SIZE });
        right_release(&mut orbit);
        assert!(orbit.menu.is_none(), "an orbit opened the menu");
        assert_ne!(orbit.camera.yaw, before, "the orbit did not happen");

        // A drag that returns to where it started is still a drag.
        let mut returned = app();
        right_press(&mut returned, centre);
        for dx in [40.0, 0.0] {
            returned.on_pointer(PointerEvent::Moved { position: moved_to(dx), size: SIZE });
        }
        right_release(&mut returned);
        assert!(returned.menu.is_none(), "an orbit that came back opened the menu");

        // Inside the slop is still a click: a hand never holds perfectly still.
        let mut wobble = app();
        right_press(&mut wobble, centre);
        wobble.on_pointer(PointerEvent::Moved {
            position: moved_to(CLICK_SLOP_PX * 0.5),
            size: SIZE,
        });
        right_release(&mut wobble);
        assert!(wobble.menu.is_some(), "a click that wobbled a pixel opened nothing");
    }

    #[test]
    fn the_menu_closes_on_escape_and_on_a_click_elsewhere() {
        let mut app = app();
        right_press(&mut app, centre_of_viewport());
        right_release(&mut app);
        assert!(app.menu.is_some());

        update(&mut app, Message::MenuClosed);
        assert!(app.menu.is_none(), "escape did not close it");

        // Open again, then click elsewhere: the click closes it and must not
        // also sculpt, or dismissing a menu would carve the model.
        right_press(&mut app, centre_of_viewport());
        right_release(&mut app);
        let probe = app
            .surface_under(Vec2::new(SIZE.x / 2.0, SIZE.y / 2.0))
            .expect("the centre should hit the sphere");
        let before = app.volume.sample_world(probe);

        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert!(app.menu.is_none(), "a click elsewhere did not close the menu");
        assert_eq!(app.volume.sample_world(probe), before, "dismissing the menu sculpted");
    }

    /// A numeric field is the reason the menu beats the panel, and the reason it
    /// is fiddly: half typed text has to survive, but a value outside the
    /// slider's range must not be accepted quietly.
    #[test]
    fn a_typed_field_survives_being_half_written_and_still_clamps() {
        let mut app = app();

        // Part way through typing "2.5": neither "" nor "2." parses, and
        // snapping the field back mid-edit would make it unusable.
        for text in ["", "2", "2.", "2.5"] {
            update(&mut app, Message::MenuFieldEdited(SizingTarget::Radius, text.to_string()));
            assert_eq!(app.menu_field_text(SizingTarget::Radius), text, "the text was lost");
        }
        assert!((app.brush.radius - 2.5).abs() < 1.0e-5, "the value never arrived");

        // Junk leaves the value alone rather than zeroing it.
        update(&mut app, Message::MenuFieldEdited(SizingTarget::Radius, "banana".into()));
        assert!((app.brush.radius - 2.5).abs() < 1.0e-5, "junk moved the value");

        // Out of range is clamped, not accepted.
        update(&mut app, Message::MenuFieldEdited(SizingTarget::Radius, "500".into()));
        assert_eq!(app.brush.radius, MAX_RADIUS_MM);
        update(&mut app, Message::MenuFieldEdited(SizingTarget::Strength, "-5".into()));
        assert_eq!(app.brush.strength, MIN_STRENGTH);

        // Once submitted the field goes back to showing the real value.
        update(&mut app, Message::MenuFieldSubmitted);
        assert_eq!(app.menu_field_text(SizingTarget::Radius), format!("{:.2}", app.brush.radius));
    }

    #[test]
    fn hovering_the_cube_hides_the_brush_ring() {
        let mut app = app();
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        assert!(app.hover.is_some(), "the model should be under the pointer");

        let (origin, size) = crate::navcube::corner_rect(Vec2::new(SIZE.x, SIZE.y));
        let middle = origin + size * 0.5;
        app.on_pointer(PointerEvent::Moved {
            position: Vector::new(middle.x, middle.y),
            size: SIZE,
        });

        assert!(app.cube_hover.is_some(), "the cube should be under the pointer");
        // No ring: a press here will not sculpt, and drawing one would promise
        // that it would.
        assert!(app.hover.is_none());
        assert!(app.shared.overlay_snapshot().lines.is_empty());
    }

    #[test]
    fn the_cursor_ring_appears_where_the_pointer_meets_the_model() {
        let mut app = app();
        assert!(app.hover.is_none(), "nothing has been pointed at yet");

        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        assert!(app.hover.is_some(), "the centre of the viewport should hit the sphere");
        // The geometry lives in the shared frame, not in `self.overlay`: the
        // application swaps its buffer over to the renderer and keeps the empty
        // one back, which is what stops either side allocating.
        let drawn = app.shared.overlay_snapshot();
        assert!(!drawn.lines.is_empty(), "no ring was handed to the renderer");

        // A corner of the viewport misses the sphere, so there is nothing to
        // draw and that absence is the signal a press would do nothing.
        app.on_pointer(PointerEvent::Moved { position: Vector::new(2.0, 2.0), size: SIZE });
        assert!(app.hover.is_none(), "a corner should miss the model");
        assert!(
            app.shared.overlay_snapshot().lines.is_empty(),
            "a ring was drawn for a pointer that is off the model"
        );
    }

    #[test]
    fn the_radius_keys_stay_inside_the_range_the_slider_offers() {
        let mut app = app();
        for _ in 0..200 {
            update(&mut app, Message::BrushRadiusScaled(1.5));
        }
        assert_eq!(app.brush.radius, MAX_RADIUS_MM);
        for _ in 0..200 {
            update(&mut app, Message::BrushRadiusScaled(1.0 / 1.5));
        }
        assert_eq!(app.brush.radius, MIN_RADIUS_MM);
    }

    /// ZBrush and Nomad both invert on alt; this had control first. Both work,
    /// and holding both is still one inversion rather than two.
    #[test]
    fn either_modifier_inverts_and_holding_both_is_not_a_double_negative() {
        let mut app = app();
        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        assert_eq!(app.stroke_direction(), BrushDirection::Add);

        let modifiers = |app: &mut Brokkr, control, alt| {
            app.on_pointer(PointerEvent::Modifiers { shift: false, control, alt });
        };

        modifiers(&mut app, true, false);
        assert_eq!(app.stroke_direction(), BrushDirection::Subtract, "control should invert");

        modifiers(&mut app, false, true);
        assert_eq!(app.stroke_direction(), BrushDirection::Subtract, "alt should invert");

        modifiers(&mut app, true, true);
        assert_eq!(
            app.stroke_direction(),
            BrushDirection::Subtract,
            "both held is one inversion, not two"
        );

        modifiers(&mut app, false, false);
        assert_eq!(app.stroke_direction(), BrushDirection::Add, "releasing should restore");
    }

    /// The eraser end still combines with a modifier the way it always did:
    /// inverting an inverted stroke gives back the additive brush.
    #[test]
    fn alt_combines_with_the_eraser_the_way_control_does() {
        for (control, alt) in [(true, false), (false, true)] {
            let mut app = app();
            update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
            app.tablet.simulate(pen(glam::Vec2::ZERO, true));
            assert_eq!(app.stroke_direction(), BrushDirection::Subtract, "the eraser alone");

            app.on_pointer(PointerEvent::Modifiers { shift: false, control, alt });
            assert_eq!(
                app.stroke_direction(),
                BrushDirection::Add,
                "a modifier over the eraser should give back the additive brush"
            );
        }
    }

    #[test]
    fn control_does_not_invert_a_brush_that_has_no_opposite() {
        let mut app = app();
        app.control = true;

        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        assert_eq!(app.stroke_direction(), BrushDirection::Subtract);

        update(&mut app, Message::BrushKindChanged(BrushKind::Smooth));
        assert_eq!(
            app.stroke_direction(),
            BrushDirection::Add,
            "smooth has no opposite, so inverting it should do nothing"
        );
    }

    fn pen(tilt: glam::Vec2, eraser: bool) -> PenState {
        PenState { in_proximity: true, pressure: 1.0, eraser, tilt }
    }

    #[test]
    fn the_eraser_end_inverts_the_brush() {
        let mut app = app();
        assert_eq!(app.stroke_direction(), BrushDirection::Add);

        app.tablet.simulate(pen(glam::Vec2::ZERO, true));
        assert_eq!(app.stroke_direction(), BrushDirection::Subtract);

        // The modifier and the eraser combine rather than override, so holding
        // one while using the other gives back the additive brush.
        app.control = true;
        assert_eq!(app.stroke_direction(), BrushDirection::Add);
    }

    #[test]
    fn the_eraser_does_nothing_to_a_brush_with_no_opposite() {
        let mut app = app();
        update(&mut app, Message::BrushKindChanged(BrushKind::Smooth));
        app.tablet.simulate(pen(glam::Vec2::ZERO, true));
        assert_eq!(app.stroke_direction(), BrushDirection::Add);
    }

    #[test]
    fn a_pen_that_is_away_cannot_erase() {
        let app = app();
        app.tablet.simulate(PenState { eraser: true, ..PenState::NONE });
        assert_eq!(app.stroke_direction(), BrushDirection::Add);
    }

    #[test]
    fn leaning_the_pen_produces_a_world_space_lean_along_the_camera_axes() {
        let mut app = app();
        assert_eq!(app.pen_lean(), Vec3::ZERO, "no pen means no lean");

        app.tablet.simulate(pen(glam::Vec2::new(1.0, 0.0), false));
        let lean = app.pen_lean();
        assert!(
            (lean.length() - MAX_TILT).abs() < 1.0e-4,
            "a fully tilted pen should lean by the maximum angle, got {}",
            lean.length()
        );
        assert!(
            lean.normalize().dot(app.camera.right()) > 0.999,
            "tilt on the x axis should lean along the camera's right axis"
        );

        // Tilt on the y axis leans down the screen, which is away from up.
        app.tablet.simulate(pen(glam::Vec2::new(0.0, 1.0), false));
        assert!(app.pen_lean().normalize().dot(app.camera.up()) < -0.999);

        update(&mut app, Message::TiltToggled(false));
        assert_eq!(app.pen_lean(), Vec3::ZERO, "turning tilt off must disable it entirely");
    }

    #[test]
    fn leaning_the_pen_moves_where_the_clay_lands() {
        // The end to end statement of what tilt is for: the same stroke on the
        // same spot puts material somewhere else when the pen is leaned.
        //
        // Measured as the difference between the two sides of the stroke rather
        // than against an upright stroke. Leaning also reduces how far the
        // brush pushes outward, by the cosine of the angle, and that term is
        // larger than the sideways one at this scale. Comparing left against
        // right cancels it, because it applies to both equally.
        let sculpt = |tilt: glam::Vec2| {
            let mut app = app();
            app.viewport_size = Vec2::new(SIZE.x, SIZE.y);
            app.brush.strength = 0.8;
            app.brush.radius = 6.0;
            app.tablet.simulate(pen(tilt, false));

            let hit = app
                .surface_under(Vec2::new(SIZE.x / 2.0, SIZE.y / 2.0))
                .expect("the centre of the view is on the model");
            let sideways = app.camera.right() * 4.0;

            // One stamp moves the surface by a fraction of a voxel, so the
            // stroke has to be laid down repeatedly to be measurable.
            for _ in 0..8 {
                press(&mut app, centre_of_viewport());
                release(&mut app);
            }
            (app.volume.sample_world(hit + sideways), app.volume.sample_world(hit - sideways))
        };

        let (right_upright, left_upright) = sculpt(glam::Vec2::ZERO);
        assert!(
            (right_upright - left_upright).abs() < 0.02,
            "an upright pen should build up evenly on both sides: \
             {right_upright} against {left_upright}"
        );

        let (right_leaned, left_leaned) = sculpt(glam::Vec2::new(1.0, 0.0));
        assert!(
            right_leaned < left_leaned - 0.02,
            "leaning right should pile material to the right: \
             right {right_leaned}, left {left_leaned}"
        );

        let (right_other_way, left_other_way) = sculpt(glam::Vec2::new(-1.0, 0.0));
        assert!(
            left_other_way < right_other_way - 0.02,
            "leaning left should pile material to the left: \
             right {right_other_way}, left {left_other_way}"
        );
    }

    #[test]
    fn resetting_discards_history_that_refers_to_the_old_model() {
        // Undoing into a volume the entry was not recorded against would splice
        // pieces of the discarded model back in.
        let mut app = app();
        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert!(app.history.can_undo());

        // Reset now goes through the unsaved-work prompt, because it discards
        // the document. Answering Discard is what the button used to do on its
        // own; the history assertions below are unchanged.
        update(&mut app, Message::ResetSphere);
        update(&mut app, Message::ConfirmAnswered(ConfirmChoice::Discard));

        assert!(!app.history.can_undo(), "reset must clear history");
        assert!(!app.history.can_redo());
    }

    #[test]
    fn every_brush_can_be_driven_from_the_interface_without_panicking() {
        // Cheap breadth: each brush goes through the whole application path
        // once, which is where a bad plane or a zero normal would surface.
        for kind in BrushKind::ALL {
            let mut app = app();
            update(&mut app, Message::BrushKindChanged(kind));
            update(&mut app, Message::BrushStrengthChanged(0.6));
            press(&mut app, centre_of_viewport());
            app.on_pointer(PointerEvent::Moved {
                position: centre_of_viewport() + Vector::new(30.0, 12.0),
                size: SIZE,
            });
            release(&mut app);
            assert_eq!(app.history_stats.undo_entries, 1, "{kind} recorded no undo entry");
        }
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;

    /// `update` returns a `Task` now. Tests do not run the iced runtime, so
    /// there is nothing to hand it to and dropping it is correct — but it must
    /// be dropped deliberately rather than by `#[allow]`, or a test that should
    /// have driven a dialog would pass silently.
    fn update(app: &mut Brokkr, message: Message) {
        let task = app.update(message);
        drop(task);
    }

    use brokkr_core::BrickCoord;

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// The starting ball has to straddle every mirror plane equally, or turning
    /// on symmetry works against a model that was never centred.
    ///
    /// It always has -- but "I measured it once" is not a guarantee, and the
    /// defaults it depends on (`MODEL_RADIUS_MM`, `VOXEL_SIZE_MM`, and the fact
    /// that voxel index 0 sits exactly on the origin) are three separate things
    /// a future change could move. This is what makes it permanent.
    #[test]
    fn the_diagnostics_carry_what_a_bug_report_needs() {
        let app = app();
        let report = app.diagnostics();
        for expected in ["BrokkrSculpt", "session:", "model:", "view:", "tablet:", "spacemouse:"] {
            assert!(report.contains(expected), "diagnostics are missing {expected}:\n{report}");
        }
        // The commit is what ties a binary to its source, which is an AGPL
        // obligation rather than a nicety.
        assert!(report.contains(build_commit()));
    }

    #[test]
    fn the_default_model_is_centred_on_every_mirror_plane() {
        let app = app();
        let (mesh, _) = app.volume.export_mesh();
        assert!(!mesh.positions.is_empty());

        let mut low = Vec3::splat(f32::MAX);
        let mut high = Vec3::splat(f32::MIN);
        for position in &mesh.positions {
            low = low.min(*position);
            high = high.max(*position);
        }

        // Well under a voxel: the lattice is symmetric about the origin, so the
        // only error available is f32 rounding.
        let tolerance = VOXEL_SIZE_MM * 0.01;
        let midpoint = (low + high) * 0.5;
        for (axis, offset) in [("x", midpoint.x), ("y", midpoint.y), ("z", midpoint.z)] {
            assert!(offset.abs() < tolerance, "the model is off centre on {axis} by {offset} mm");
        }

        // And it reaches the same distance either way along each axis, which is
        // what "equal sides of the centreline" actually means.
        for (axis, lo, hi) in [("x", low.x, high.x), ("y", low.y, high.y), ("z", low.z, high.z)] {
            assert!(
                (hi + lo).abs() < tolerance,
                "the model reaches {hi} one way on {axis} and {lo} the other"
            );
        }

        // The field itself, in mirrored pairs, not just the mesh it produced.
        for step in 0..24 {
            let probe = Vec3::new(
                MODEL_RADIUS_MM - 1.0 + step as f32 * 0.05,
                step as f32 * 0.37,
                step as f32 * -0.21,
            );
            for flip in
                [Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, -1.0)]
            {
                let here = app.volume.sample_world(probe);
                let mirrored = app.volume.sample_world(probe * flip);
                assert!(
                    (here - mirrored).abs() < 1.0e-4,
                    "the field disagrees across {flip:?}: {here} against {mirrored}"
                );
            }
        }
    }

    #[test]
    fn the_default_model_exports_watertight() {
        // The most basic promise of a printing tool: what it opens with can be
        // printed.
        let app = app();
        let (_, report) = app.volume.export_mesh();
        assert!(
            report.is_printable(),
            "the model the application starts with does not print: {}",
            report.summary()
        );
    }

    #[test]
    fn a_sculpted_model_exports_watertight() {
        let mut app = app();
        app.viewport_size = Vec2::new(800.0, 600.0);
        app.brush.strength = 0.7;
        for offset in 0..6 {
            app.on_pointer(PointerEvent::Pressed {
                button: PointerButton::Left,
                position: iced::Vector::new(400.0 + offset as f32 * 8.0, 300.0),
                size: iced::Vector::new(800.0, 600.0),
            });
            app.on_pointer(PointerEvent::Released { button: PointerButton::Left });
        }

        let (_, report) = app.volume.export_mesh();
        assert!(report.is_printable(), "{}", report.summary());
    }

    #[test]
    fn resampling_finer_increases_detail_and_keeps_it_printable() {
        let mut app = app();
        let (_, before) = app.volume.export_mesh();
        let coarse_voxel = app.voxel_size;

        update(&mut app, Message::Resample(coarse_voxel / 2.0));

        assert!((app.voxel_size - coarse_voxel / 2.0).abs() < 1.0e-6);
        let (_, after) = app.volume.export_mesh();
        assert!(
            after.triangles > before.triangles * 2,
            "finer voxels should give far more triangles: {} against {}",
            after.triangles,
            before.triangles
        );
        assert!(after.is_printable(), "{}", after.summary());
        assert!(app.status.contains("resampled"), "the interface should say what happened");
    }

    #[test]
    fn resampling_will_not_go_past_the_limits() {
        // Memory grows as the inverse cube of the voxel size, so the bounds stop
        // a click walking the model past what the mesh pool has been measured to
        // hold. Checked on the clamp rather than by resampling, because actually
        // building a model at the limit is a gigabyte of work.
        assert_eq!(Brokkr::clamped_voxel_size(0.0000001), FINEST_VOXEL_MM);
        assert_eq!(Brokkr::clamped_voxel_size(1000.0), COARSEST_VOXEL_MM);
        assert_eq!(Brokkr::clamped_voxel_size(0.25), 0.25);

        // And the interface offers a step that lands inside the limits: two
        // halvings from the default are allowed, three are not.
        const _: () = assert!(FINEST_VOXEL_MM > VOXEL_SIZE_MM / 8.0);
        const _: () = assert!(FINEST_VOXEL_MM < VOXEL_SIZE_MM / 2.0);
    }

    #[test]
    fn resampling_clears_history_and_schedules_a_full_remesh() {
        let mut app = app();
        app.viewport_size = Vec2::new(800.0, 600.0);
        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: iced::Vector::new(400.0, 300.0),
            size: iced::Vector::new(800.0, 600.0),
        });
        app.on_pointer(PointerEvent::Released { button: PointerButton::Left });
        assert!(app.history.can_undo());

        let old_coords: Vec<BrickCoord> = app.volume.brick_coords().collect();
        let finer = app.voxel_size / 2.0;
        update(&mut app, Message::Resample(finer));

        assert!(
            !app.history.can_undo(),
            "history refers to bricks at the old resolution and must be dropped"
        );
        assert!(app.perf.dirty_bricks > old_coords.len(), "the whole model has to be remeshed");
    }

    #[test]
    fn resampling_to_the_current_size_does_nothing() {
        let mut app = app();
        let before = app.volume.brick_count();
        let same = app.voxel_size;
        update(&mut app, Message::Resample(same));
        assert_eq!(app.volume.brick_count(), before);
        assert!(app.status.is_empty(), "nothing happened, so nothing should be reported");
    }
}
