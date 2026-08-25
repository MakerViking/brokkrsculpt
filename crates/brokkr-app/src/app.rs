// SPDX-License-Identifier: AGPL-3.0-only

//! Application state, input handling and the widget tree.

mod panel;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use brokkr_core::{
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, Change, Document, Entry,
    History, HistoryStats, MAX_VOLUME_BYTES, MirrorAxis, MoveStroke, NodeId, NodeMeta, Stamp,
    Stroke, Symmetry, UndoOutcome, Volume, VolumeStats, lean_normal,
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

/// The point every enabled mirror plane passes through: the lattice origin.
///
/// **One constant, so there is one answer.** `Symmetry::mirrors` and
/// `Symmetry::flips` take the centre as a parameter rather than assuming it,
/// and `cursor::build` draws the planes where this says they are, so the
/// mirroring the engine performs and the mirroring the viewport promises come
/// from the same number.
///
/// The value is the origin because **the axis and the centre must have the same
/// scope**. The three axis switches are global -- one setting for the whole
/// document -- and a global switch whose plane came from the selected body
/// would make "X on" a different physical plane depending on which row is
/// highlighted, with the only evidence on screen a translucent patch that
/// shrinks with that body's radius. A single world-fixed mirror on a shared
/// lattice is defensible and it is what the interface has always drawn.
///
/// **Be honest about what that does not fix.** With the centre pinned here, a
/// dent carved into a body sitting at x = +80 still gets its twin at x = -80:
/// free-floating geometry that exports as an extra, unprintable shell. The
/// parameter alone buys nothing today. What stops that happening is
/// [`Brokkr::mirror_refusal`], which will not let an axis be enabled while the
/// active body sits wholly to one side of it. The real fix is a per-body
/// centre, and it is deferred with the per-body axis it has to move with.
const MIRROR_CENTRE: Vec3 = Vec3::ZERO;

/// Range the brush radius may be nudged to with the keyboard, in millimetres.
/// The same range the slider offers, so the two cannot disagree.
///
/// It was 12 mm, which was too small to grab a form with Move. The new ceiling
/// is measured rather than chosen, and `cargo bench -p brokkr-core` is what
/// measured it: a brush covers a fixed world radius, so its cost grows with the
/// cube of it, and what a drag has to fit inside is the 16 ms frame -- edit plus
/// remesh, per pointer event.
///
/// 30 mm was tried first, on an estimate that undercounted the remesh, and the
/// bench refused it flatly. 25 mm passed every single stamp row and then failed
/// the **fast drag**, which is the case that decides this: fewer pointer
/// samples means more interpolated stamps per event, and it came to 15.7 ms p95
/// with a 17.8 ms worst against the 16 ms frame. A brush that is fluid until
/// you hurry is not fluid.
///
/// So 20 mm, where all of it passes, and the bench's own sweep tops out at the
/// same place. **Move the two together or neither** -- a slider that goes past
/// what the gate covers is a promise nothing checks.
///
/// The cap is the same for every brush, deliberately. Move is among the
/// cheapest and could afford more, but a per-brush ceiling means the radius
/// jumps when the tool changes, and a number that moves on its own is worse
/// than a number that is merely smaller than you wanted.
pub(crate) const MIN_RADIUS_MM: f32 = 0.25;
pub(crate) const MAX_RADIUS_MM: f32 = 20.0;

/// The most voxels of radius a brush may reach, whatever that is in
/// millimetres.
///
/// **A brush costs what it costs in VOXELS, and the ceiling beside this one is
/// in millimetres, so the two disconnect the moment the voxel size moves.**
/// Measured 2026-08-21, one Draw stamp, with the application running so
/// pessimistic: 80 voxels of radius took 5.01 ms, 96 took 5.22 ms, and 192 took
/// **21.04 ms**. The cost tracks the voxel radius and nothing else -- doubling
/// it costs four times as much, because the work is the surface inside the
/// brush.
///
/// At the 0.25 mm default, 100 voxels is 25 mm, which is above
/// [`MAX_RADIUS_MM`] and so changes nothing at all. At the 0.03125 mm a resin
/// print wants, it is 3.1 mm -- and without it the 20 mm slider would reach 640
/// voxels, which is roughly a quarter of a second per stamp and not a usable
/// tool.
///
/// The budget bench sweeps radius in millimetres at one voxel size, so it has
/// never covered this. Sweeping voxel radius instead is the honest fix and is
/// not done.
pub(crate) const MAX_RADIUS_VOXELS: f32 = 100.0;

/// Range the brush strength may take, matching the slider.
pub(crate) const MIN_STRENGTH: f32 = 0.02;
pub(crate) const MAX_STRENGTH: f32 = 0.80;

/// How fast a hold-and-drag moves the brush numbers.
///
/// Radius is in log space so the same drag is the same proportion at any size;
/// about 250 px covers the whole range, which is a comfortable sweep.
const RADIUS_PER_PIXEL: f32 = 0.006;
const STRENGTH_PER_PIXEL: f32 = 0.002;

/// Range the voxel size may be resampled to, in millimetres.
///
/// **The lower bound is a resin printing target, not a pool limit.** It used to
/// be 0.06 and justified as the most the mesh pool could hold -- which stopped
/// being the reason the moment `resample` grew a preflight of its own. What
/// fits is now decided per model, against the real reservation on the GPU, and
/// refused with a message naming the size that would work. So this bound exists
/// only to stop the interface offering something absurd.
///
/// 0.03 rather than 0.035, which is roughly what a consumer resin printer
/// resolves, because **the halving ladder from the 0.25 default lands on
/// 0.03125 and never on 0.035**. A floor of 0.035 would leave the finest step
/// permanently one press out of reach. The print size field reaches
/// intermediate sizes continuously; the buttons only halve.
const FINEST_VOXEL_MM: f32 = 0.03;
const COARSEST_VOXEL_MM: f32 = 2.0;

/// Roughly what a consumer resin printer resolves in XY, in millimetres.
///
/// A reference point for the detail advice, not a limit on anything. Mono LCD
/// panels in the 8K class land near this; the exact figure varies by machine
/// and it is used only to say "you are at or below this", which does not need
/// to be exact.
const RESIN_XY_MM: f32 = 0.035;

/// A typical filament nozzle's extrusion width, in millimetres. The other end
/// of the same comparison: detail finer than this cannot be printed by FDM at
/// all, so a voxel well under it is being spent on nothing unless the model is
/// bound for resin.
const FDM_LINE_MM: f32 = 0.4;

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
    /// The press landed on a body that was not the active one, so it chose that
    /// body and nothing else.
    ///
    /// It is a drag kind rather than nothing at all because the release has to
    /// know what the press meant: a gesture that never opened a recorder must
    /// not be finished as if it had. Dragging after it does nothing, which is
    /// the correct answer -- the body under the cursor changed, and a stroke
    /// the user had not chosen a target for is exactly the surprise this
    /// ordering exists to remove.
    Selecting,
    /// The press had nothing it was allowed to do: the pointer was over
    /// nothing that is drawn, and the body edits would land on is hidden.
    ///
    /// A drag kind rather than no drag at all for the same reason
    /// [`DragKind::Selecting`] is one -- the release has to know that this
    /// press opened no recorder -- and a kind of its own rather than
    /// `Selecting` because no body was chosen. **Without it the press falls
    /// through to a stroke**, and a stroke whose surface comes from
    /// `pick_body`, which does not consult the eye, carves the invisible body
    /// and marks the document unsaved with nothing on screen changing.
    Refused,
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
    /// Every body in the sculpt, on one lattice.
    ///
    /// This was a bare `Volume`. The application still holds exactly one body
    /// and nothing in the interface can create a second, but the voxel size,
    /// the dirty set and the meshing all belong to the document from here --
    /// which is what makes "the volume" stop being a thing a call site can mean
    /// by accident.
    doc: Document,
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
    /// Strength per brush, so switching tools restores what that tool was last
    /// set to rather than carrying a number that means something different.
    ///
    /// Strength is not the same quantity for every brush -- for Move it is the
    /// fraction of the drag the surface follows -- so one shared slider value
    /// made Move look broken at Draw's default. Indexed by `BrushKind::ALL`.
    strengths: [f32; BrushKind::ALL.len()],
    /// Where a Move gesture grabbed the surface, fixed for the whole drag.
    ///
    /// The drag target is the pointer projected into the view plane through
    /// this point, so the surface follows the cursor across the screen rather
    /// than crawling around the form. See `view_plane_point`.
    move_grab: Option<Vec3>,
    /// The field Move locked when the current gesture began, if the current
    /// gesture is a Move. Kept here rather than rebuilt per event, because
    /// holding it across the whole gesture is the entire point of the brush.
    move_stroke: MoveStroke,
    /// Stamp centres produced by the current pointer event. Reused so a stroke
    /// does not allocate.
    stamp_centres: Vec<Vec3>,
    /// Bricks waiting to be remeshed, each tagged with the body it belongs to.
    ///
    /// One list across the whole document rather than one per body, because
    /// [`Document::mesh_dirty`] has to make the serial-or-parallel decision
    /// once over the real total -- see its documentation.
    dirty: Vec<(NodeId, BrickCoord)>,
    /// What is drawn, indexed by node position, and the bodies among it that
    /// are not. Both are kept rather than rebuilt because
    /// [`Brokkr::publish_visibility`] runs on the frame tick.
    shown: Vec<bool>,
    hidden_bodies: Vec<NodeId>,
    /// The one row solo is showing, or `None` for the whole document.
    ///
    /// **A field of the APPLICATION, never of the document.** Solo is the third
    /// input to [`brokkr_core::resolve_visibility`] and it reaches it as a
    /// parameter, which is what makes it structurally impossible to persist:
    /// `project::write` takes a `&Document` and a `&ProjectState`, so there is
    /// no expression that hands it solo. Do not move this onto `Document` or
    /// `View` to tidy a call site — the guarantee is the type signature, not
    /// the discipline, and `View` is restored by both a reopen and a timeline
    /// key.
    ///
    /// It is also why leaving solo restores the hand-set eyes bit for bit:
    /// nothing was ever written, so there is nothing to restore. Every shipped
    /// version of the save-a-vector design loses that set — Photoshop's
    /// alt-click eye remembers it only "if you haven't changed anything else",
    /// and Plasticity's own manual says Unisolate makes everything visible
    /// instead.
    solo: Option<NodeId>,
    drag: Option<Drag>,
    /// Last pointer position in widget pixels, for drag deltas.
    cursor: Option<Vec2>,
    viewport_size: Vec2,
    shift: bool,
    control: bool,
    /// Alt, which inverts the brush the same way control does.
    alt: bool,
    perf: Perf,
    /// What the whole document costs, summed over every body, refreshed on
    /// remesh.
    ///
    /// **There is deliberately no `voxel_size` field beside this.** There used
    /// to be, and it was an application-side copy of a `Volume` property that
    /// four separate sites re-derived; with a document lattice there is exactly
    /// one correct value and `doc.voxel_size()` is it.
    doc_stats: VolumeStats,
    history_stats: HistoryStats,
    /// What the last export or resample did, for the interface to show.
    status: String,
    /// Which blocks of the properties panel are open, in `PanelSection::ALL`
    /// order.
    expanded: [bool; PanelSection::ALL.len()],
    /// Whether the stats readout over the viewport is showing.
    ///
    /// Closed to begin with: it is seven lines of monospace across the corner
    /// of the model, and the answers it holds are wanted occasionally rather
    /// than continuously. The icon that opens it stays put either way.
    stats_open: bool,
    /// The brush ring and mirror planes, handed to the renderer through
    /// `SharedFrame`. Held here and swapped rather than rebuilt into a fresh
    /// allocation each time.
    overlay: brokkr_gpu::OverlayBatch,
    /// Where the pointer last met the surface, which is where the ring goes.
    hover: Option<Vec3>,
    /// Which body that surface belonged to, or `None` when the pointer is off
    /// the model.
    ///
    /// The ring is built against this body rather than the active one. Without
    /// it, hovering a second body draws a ring sampled from a field that does
    /// not contain the point -- a confident flat ring below nothing -- and the
    /// mood cannot say that a press there would select rather than carve.
    hover_body: Option<NodeId>,
    /// A hold-and-drag brush resize in progress.
    sizing: Option<Sizing>,
    /// Whether the radius tracks the model's size rather than staying a fixed
    /// number of millimetres.
    dynamic_radius: bool,
    /// How big the active body is, refreshed on remesh.
    /// [`Volume::content_radius`] walks the brick map, so it is not something
    /// to call per frame.
    model_radius: f32,
    /// Which body `model_radius` was last measured on.
    ///
    /// **This is what makes "Dynamic never resizes the brush because the
    /// SELECTION changed" unrepresentable rather than merely avoided.** Without
    /// it, a selection handler that does the obvious thing -- set the active
    /// body, then remesh -- would compare a 5 mm rivet's radius against the
    /// 200 mm bust's and scale the brush by forty. See
    /// [`Brokkr::rescale_radius`].
    model_radius_body: NodeId,
    /// What is typed in the working-size field, before it is committed. Held as
    /// text so a half-typed number is not rounded or rejected mid-keystroke.
    pub(crate) working_size_field: String,
    /// The cached one-line readout of what the current resolution means.
    ///
    /// CACHED because computing it walks every dense brick, and the widget tree
    /// is rebuilt every frame. Refreshed only when something that changes the
    /// answer happens -- import, open, resample, resize, re-orient. Sculpting
    /// moves the model's extent slightly and deliberately does NOT refresh it:
    /// a per-stroke scan of the whole volume is exactly the shape of work this
    /// engine exists to avoid, and a millimetre of drift in a readout that
    /// exists to answer "is 0.25 mm fine enough" changes nothing.
    pub(crate) detail_advice: String,
    /// Stored views on the strip above the viewport, and the state of
    /// scrubbing through them.
    timeline: crate::timeline::Timeline,
    /// What this session has done, for a bug report.
    breadcrumbs: crate::breadcrumbs::Breadcrumbs,
    /// The status line as it was when it was last recorded as a breadcrumb.
    ///
    /// The trail is taken by watching `status` change rather than by calling
    /// `crumb` at each of the two dozen places that set it. A trail that
    /// depends on remembering to add a line is a trail that goes quietly
    /// incomplete, and the failure is invisible until the one report that
    /// needed it.
    crumbed_status: String,
    /// The bug report dialog, open or not.
    bug_report: Option<BugReport>,
    /// Whether the once-per-session facts have been recorded yet.
    ///
    /// Not done in the constructor: the renderer has not looked at the adapter
    /// by then, and the devices are still being scanned. The first frame is the
    /// earliest point at which there is anything true to say.
    facts_recorded: bool,
    /// The navigation cube's geometry, swapped to the renderer the same way.
    cube: brokkr_gpu::OverlayBatch,
    /// The cube part under the pointer, lit so a click's effect is visible
    /// before the click.
    cube_hover: Option<navcube::Part>,
    /// A camera move in progress after clicking the cube.
    flight: Option<Flight>,
    /// Where the right-click menu is open, in widget pixels.
    menu: Option<Vec2>,
    /// The navigation cube's own menu: where it is drawn and the face it acts
    /// on. See [`CubeMenu`].
    pub(crate) cube_menu: Option<CubeMenu>,
    /// An imported model whose own up is not up, and which way it points.
    ///
    /// Offered rather than applied: nothing in a mesh file states which axis is
    /// up, so this is a guess, and a guess that silently turned the model would
    /// be worse than one the user waves away. `None` once answered either way.
    pub(crate) orient_prompt: Option<brokkr_core::Facing>,
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
    /// When the pointer last did anything, so the autosave can wait for a pause.
    last_activity: Instant,
    /// A numeric field in the menu being typed into.
    ///
    /// Held as text rather than parsed on every keystroke, because a half typed
    /// value like `2.` does not parse and snapping the field back to the old
    /// number mid-edit makes it unusable.
    menu_edit: Option<(SizingTarget, String)>,
    /// Whether the body list draws its thumbnail column.
    ///
    /// **Session state, and deliberately not in the file.** Photoshop's Panel
    /// Options offers None alongside three sizes and the standard advice for a
    /// heavy document is None, so an off switch is faithful to the reference
    /// rather than a retreat from it -- and by the rule that nothing outside the
    /// file may dirty the document, toggling it must not set `unsaved`.
    thumbnails: bool,
    /// Whether the `+` button's little menu of primitives is open.
    ///
    /// A plain flag rather than a `stack!` layer: the menu is drawn inside the
    /// panel column, under the verb row, so it needs no scrim and cannot fall
    /// into the trap that iced 0.14 stack layers do not block what is beneath
    /// them.
    adding: bool,
    /// A body delete waiting on an answer, because it is large enough that undo
    /// may not be able to hold it.
    pending_delete: Option<PendingDelete>,
    /// How large a delete has to be before it asks first.
    ///
    /// [`brokkr_core::DEFAULT_RECLAIM_BUDGET`] by default, and one number rather
    /// than two that happen to coincide: a delete that would be evicted from
    /// history before it could be undone is exactly the delete that has to warn.
    ///
    /// Carried rather than a constant read in place so that a test can point it
    /// at a small number -- the alternative is a fixture that really does hold
    /// 512 MB of bricks, which is half a gigabyte of allocation inside a test
    /// process running alongside every other test in the crate.
    delete_prompt_bytes: usize,
    /// A merge waiting on an answer, because the one entry it would push is
    /// large enough that undo may not be able to hold it.
    pending_merge: Option<PendingMerge>,
    /// How large the entry a merge would push has to be before it asks first.
    ///
    /// The same [`brokkr_core::DEFAULT_RECLAIM_BUDGET`] as the delete prompt,
    /// carried rather than read in place for the same reason: a test that
    /// pointed this at the real 512 MB would have to build half a gigabyte of
    /// bricks.
    ///
    /// **One number against the SUM of the two halves, which is an
    /// approximation, and here is where it is loose.** The bricks half is
    /// charged to the 256 MB stroke budget and the consumed body to the 512 MB
    /// reclaim allowance, so the exactly-right test is two comparisons. The sum
    /// against 512 MB is within a factor of two of both and errs on the side of
    /// asking: a fully overlapping merge records its priors twice over -- once
    /// as bricks and once as the body -- so anything that would breach the
    /// 256 MB stroke budget on its own is already over 512 MB summed. What it
    /// permits is a large DISJOINT merge, whose bricks half is 32 bytes each;
    /// that one is exactly as recoverable as deleting the same body, which does
    /// not prompt below the same number either.
    merge_prompt_bytes: usize,
    /// The row being renamed and what has been typed into its field so far.
    ///
    /// The typed text is held HERE rather than written into the node on every
    /// keystroke, so that one rename is one undo entry rather than one per
    /// letter, and so that Escape has something to revert to. It is already
    /// clamped to [`brokkr_core::MAX_NAME_BYTES`] -- see
    /// [`Brokkr::commit_rename`] for why the clamp is on the way in and not on
    /// the way out.
    ///
    /// **Never persisted, and there is nowhere it could be**: nothing in
    /// `ProjectState` names it, and a rename in flight when the application
    /// quits is simply a rename that did not happen.
    renaming: Option<(NodeId, String)>,
}

/// A delete the user has been asked about, held until they answer.
///
/// It carries the size it measured rather than re-measuring on the way out:
/// `Volume::stats` walks a body's whole brick map, and the number the prompt
/// showed has to be the number the decision was made about.
pub(crate) struct PendingDelete {
    pub(crate) id: NodeId,
    pub(crate) name: String,
    pub(crate) bytes: usize,
    /// How many bodies go with it, which for a folder is not one.
    ///
    /// **Folders make the prompt the common path rather than the exception**,
    /// and a card that says "Delete Group 1?" over a size without saying how
    /// many parts are inside it is asking the user to remember what they put
    /// there. It is counted here beside the bytes, and for the same reason:
    /// the number the prompt shows has to be the number the decision was made
    /// about.
    pub(crate) bodies: usize,
}

/// A merge the user has been asked about, held until they answer.
///
/// It carries everything the card shows, measured once, for the reason
/// [`PendingDelete`] carries its size: the walk that produced the number is a
/// walk of a brick map, and the number the prompt showed has to be the number
/// the decision was made about.
///
/// The `source` id and not an index: a queued message could in principle arrive
/// after the list has moved under it, and an index would then name a different
/// row. `apply_merge` re-resolves the target from the id, so a source that has
/// gone simply does nothing.
pub(crate) struct PendingMerge {
    pub(crate) source: brokkr_core::NodeId,
    pub(crate) source_name: String,
    pub(crate) target_name: String,
    /// The whole entry, which is the number the user is being asked about.
    pub(crate) bytes: usize,
    /// The bricks half, charged to the stroke budget.
    pub(crate) stroke_bytes: usize,
    /// The consumed body, charged to the reclaim allowance.
    pub(crate) reclaim_bytes: usize,
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

/// The navigation cube's right-click menu while it is open.
///
/// Fusion's ViewCube, and the answer to "down isn't down": a left click on the
/// cube moves the camera, a right click asks what that face of the *model*
/// should become. The two need different state because the second one has to
/// remember which face was picked until the answer arrives.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CubeMenu {
    /// Where to draw it, in widget pixels.
    pub(crate) at: Vec2,
    /// The face the menu was opened on, which is the one that will be moved.
    pub(crate) facing: brokkr_core::Facing,
}

/// The bug report dialog while it is open.
///
/// Held on the application rather than passed around because the description
/// editor owns its own cursor and selection state, which has to outlive a
/// redraw.
pub(crate) struct BugReport {
    description: iced::widget::text_editor::Content,
    /// Whether to attach the diagnostics and the session trail.
    ///
    /// On by default, and the reason the dialog shows the whole payload: a
    /// report without them is rarely answerable, and a user who can read
    /// exactly what is attached can decide for themselves rather than being
    /// asked to trust a sentence about it.
    with_detail: bool,
    /// True while the request is in flight, so the button cannot be pressed
    /// twice and file two rows for one bug.
    sending: bool,
}

impl BugReport {
    fn new() -> Self {
        Self {
            description: iced::widget::text_editor::Content::new(),
            with_detail: true,
            sending: false,
        }
    }
}

/// Something the user asked for that would discard unsaved work, held until
/// they say what to do about it.
///
/// Each of these throws the current document away: New reseeds the sphere, Open
/// swaps a file in, and Quit ends the process.
///
/// **They are not, as this comment used to claim, "the complete set of ways
/// work can be lost", and that sentence was becoming a lie a reader would
/// trust.** Two things are true instead. The gate lives in one place, so the
/// list of *guarded* actions is complete by construction. And what counts as
/// something to lose is no longer just `unsaved`: importing a mesh replaces
/// every body in the document, so with a saved multi-body project open it used
/// to discard all of them silently, having asked nothing, because nothing was
/// unsaved. See [`Brokkr::would_lose_work`].
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
    /// Import a mesh, REPLACING the document. It discards every body exactly as
    /// Open does, so it goes through the same gate -- and, unlike Open, it does
    /// not put a file the user chose in their place, which is why it is guarded
    /// on the body count as well as on `unsaved`.
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
            PendingAction::Import => "Importing a mesh in place of every body",
            PendingAction::Quit(_) => "Quitting",
        }
    }
}

/// Whether a world space box has material on both sides of a mirror plane
/// through [`MIRROR_CENTRE`].
///
/// Touching counts as straddling: a body whose surface reaches the plane
/// exactly has its twin land against it rather than out in space, which is a
/// join and not a free-floating shell.
fn straddles((low, high): (Vec3, Vec3), axis: MirrorAxis) -> bool {
    let component = match axis {
        MirrorAxis::X => 0,
        MirrorAxis::Y => 1,
        MirrorAxis::Z => 2,
    };
    low[component] <= MIRROR_CENTRE[component] && MIRROR_CENTRE[component] <= high[component]
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

/// Turn a window event the widget tree did not want into a message.
///
/// This is the callback `Brokkr::subscription` hands to
/// `iced::event::listen_with`, lifted out of it and named so that it can be
/// tested. It costs nothing to lift: `listen_with` takes a bare `fn` pointer
/// (`iced_futures-0.14.0/src/event.rs:26`), so the inline version was already
/// a non-capturing closure and this is the same value with a name on it.
///
/// It was NOT tested while it was inline, and that was the gap: every keyboard
/// guard test synthesises `Message::KeyPressed` directly, which is downstream
/// of here. Deleting the `KeyPressed` arm would have killed the entire
/// keyboard -- no undo, no brush digits, no mirror keys -- with all of those
/// tests still green, reporting a guard working perfectly on a message the
/// application could no longer produce.
///
/// The two things it decides, and neither belongs anywhere else:
///
/// * Captured events are dropped. That is what makes the shortcuts
///   focus-aware -- a focused text input consumes its own keystrokes -- and
///   inverting it is how `1`-`7`, `s`, `u`, `x`, `y` and `z` were stolen from
///   every text field for a year.
/// * A press forwards the raw key; it does not decide what the key MEANS.
///   There is no `self` here and no way to get one, so whether a modal is up
///   is invisible from inside. `Brokkr::on_key` decides, and holds the guard.
///
/// It is also where a left press that nobody wanted becomes
/// [`Message::PressedNothing`], for the same reason and by the same test: this
/// is the application's "nobody claimed that" tap, and an unclaimed left press
/// is exactly what blurs a focused `text_input`. See that message.
fn key_event(
    event: iced::Event,
    status: iced::event::Status,
    _window: iced::window::Id,
) -> Option<Message> {
    if status == iced::event::Status::Captured {
        return None;
    }
    match event {
        iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { key, modifiers, .. }) => {
            Some(Message::KeyPressed { key, modifiers })
        }
        // An ignored LEFT press is a blur, and nothing else in the application
        // can see one. Left only, because that is the button `text_input`
        // blurs on.
        iced::Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
            Some(Message::PressedNothing)
        }
        // Releasing a sizing key ends the gesture. Without this the pointer
        // would keep resizing the brush instead of going back to sculpting.
        //
        // Not routed through `on_key` and deliberately not guarded by the
        // modal check: a modal that opened mid-gesture must still let the
        // gesture end, and ending one that never started is already a no-op.
        iced::Event::Keyboard(iced::keyboard::Event::KeyReleased { key, .. }) => match key {
            iced::keyboard::Key::Character(character)
                if matches!(character.to_ascii_lowercase().as_str(), "s" | "u") =>
            {
                Some(Message::SizingEnded)
            }
            _ => None,
        },
        _ => None,
    }
}

/// The rename field's widget id, so that beginning a rename can focus it.
///
/// **The only widget in the application with an id at all.** One constant and
/// not one per row: there is at most one rename in flight, so the field exists
/// in exactly one row at a time and `operation::focus` finds it wherever that
/// row happens to be. An id per row would also break the moment folders let a
/// row move.
pub(crate) const RENAME_FIELD: &str = "brokkr-body-rename";

/// Whether a message leaves a rename in flight alone, or ends it.
///
/// **The list is inverted on purpose, and that is the whole reason this is a
/// function rather than a call at each site that ends a rename.** Committing
/// keeps what the user typed and closing without committing throws it away, so
/// the safe default for a message nobody thought about is to commit. Written
/// the other way round -- a list of messages that DO commit -- a message added
/// next year would default to leaving a field open over a document that has
/// moved on underneath it, and nothing would fail.
///
/// The four that keep it open:
///
/// * the field's own keystrokes, which are the rename;
/// * every key press, because Escape is captured by the focused field itself
///   (`text_input.rs:1235-1244`) and only reaches the application once the
///   field has already been blurred -- at which point it must still REVERT,
///   and a commit here would have eaten it first. Keeping every key press
///   rather than picking Escape out of them keeps the "no key is decoded
///   outside `viewport::shortcut`" rule intact;
/// * `MenuClosed`, which is what Escape becomes and which reverts;
/// * the frame tick and a pointer merely moving. A pointer PRESS is the user
///   leaving, and does commit.
///
/// Note which press that is: `Message::Pointer(Pressed)` is raised only for a
/// press INSIDE the viewport (`viewport::route_pointer` bounds-checks it), so
/// it is not the whole of "the user clicked away". A press that landed on
/// nothing -- the empty scrollable under the last body row, the panel
/// background -- arrives as [`Message::PressedNothing`], which is on no list
/// here and therefore commits. That message exists for exactly this, because
/// the field has already blurred itself by then.
fn keeps_the_rename_open(message: &Message) -> bool {
    match message {
        Message::BodyRenameEdited(_) => true,
        Message::KeyPressed { .. } | Message::MenuClosed => true,
        Message::Frame => true,
        Message::Pointer(event) => !matches!(event, PointerEvent::Pressed { .. }),
        _ => false,
    }
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

        let doc = Document::from_volume(volume);
        let mut app = Self {
            model_radius_body: doc.active(),
            doc,
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
            strengths: BrushKind::ALL.map(BrushKind::default_strength),
            move_grab: None,
            move_stroke: MoveStroke::new(),
            stamp_centres: Vec::new(),
            dirty: Vec::new(),
            shown: Vec::new(),
            hidden_bodies: Vec::new(),
            solo: None,
            drag: None,
            cursor: None,
            viewport_size: Vec2::new(1280.0, 720.0),
            shift: false,
            control: false,
            alt: false,
            perf: Perf::default(),
            doc_stats: VolumeStats::default(),
            history_stats: HistoryStats::default(),
            status: String::new(),
            expanded: PanelSection::ALL.map(PanelSection::open_by_default),
            stats_open: false,
            overlay: brokkr_gpu::OverlayBatch::default(),
            hover: None,
            hover_body: None,
            sizing: None,
            dynamic_radius: false,
            model_radius: MODEL_RADIUS_MM,
            working_size_field: String::new(),
            detail_advice: String::new(),
            timeline: crate::timeline::Timeline::new(),
            breadcrumbs: crate::breadcrumbs::Breadcrumbs::new(),
            crumbed_status: String::new(),
            bug_report: None,
            facts_recorded: false,
            cube: brokkr_gpu::OverlayBatch::default(),
            cube_hover: None,
            flight: None,
            menu: None,
            cube_menu: None,
            orient_prompt: None,
            top_menu: None,
            project_path: None,
            unsaved: false,
            confirm: None,
            cut_armed: false,
            recent: crate::recent::Recent::load(),
            autosave_file: Self::default_autosave_path(),
            last_autosave: Instant::now(),
            last_activity: Instant::now(),
            menu_edit: None,
            thumbnails: true,
            adding: false,
            pending_delete: None,
            delete_prompt_bytes: brokkr_core::DEFAULT_RECLAIM_BUDGET,
            pending_merge: None,
            merge_prompt_bytes: brokkr_core::DEFAULT_RECLAIM_BUDGET,
            renaming: None,
        };
        app.remesh_dirty();
        // Otherwise the overlay reports a zero byte budget until the first
        // stroke happens to refresh it.
        app.history_stats = app.history.stats();
        app.perf.load_ms = app.perf.remesh_ms;
        app.perf.remesh_ms = 0.0;
        app.perf.dirty_bricks = 0;
        app.publish_camera();
        app.refresh_detail_advice();
        app.refresh_overlay();
        // Before the first message, so the renderer is never acting on an empty
        // hidden set it was never given.
        app.publish_visibility();
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
            // Keyboard shortcuts, on events the widget tree IGNORED. They
            // lived in the shader widget for a year, captured window-wide,
            // "because the shader already receives every event" -- and that
            // capture stole 1-7, s, u, x, y and z from every text field in
            // the application, because the shader traverses before the panel.
            // Filtering on Ignored is what makes them focus-aware: a focused
            // text input consumes its keystrokes, and a shortcut fires only
            // when nothing else wanted the key.
            //
            // What that callback must NOT do is decide what a key means.
            // `listen_with` takes a bare `fn` pointer, not a closure, so there
            // is no `self` there and no way to get one: whether a modal is up,
            // whether a gesture is in flight, whether the document can even
            // take the change are all invisible from inside it. It forwards
            // the key and `on_key` decides. Reaching for a static or a global
            // to dodge that is how the two halves drift apart.
            //
            // It is `key_event`, a named function rather than an inline
            // closure, purely so the tests can call it -- see the tests on it
            // for what silently breaks when it is wrong.
            iced::event::listen_with(key_event),
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

    /// Do `action`, or ask first if there is work to lose.
    fn guard(&mut self, action: PendingAction) -> Task<Message> {
        if self.would_lose_work(&action) {
            // The prompt is about to draw over the viewport, and a menu left
            // open would sit on top of it.
            self.top_menu = None;
            self.menu = None;
            self.confirm = Some(action);
            return Task::none();
        }
        self.run_pending(action)
    }

    /// Whether carrying `action` out would cost the user something they cannot
    /// get back with one gesture.
    ///
    /// `unsaved` for everything, and for an import **the body count as well,
    /// whether or not anything is unsaved**. Import replaces every body with
    /// the mesh it reads. With a saved five-body project open, `unsaved` is
    /// false, so it used to discard all five having asked nothing: no dialog,
    /// no prompt, nothing in the status line. The file on disk survives, which
    /// is the only reason this is a prompt and not a refusal — reassembling
    /// five bodies by hand is not a recovery a user should have to make because
    /// a menu item did more than its name said.
    ///
    /// Open is deliberately NOT guarded the same way: it puts a document the
    /// user picked in place of this one, which is what its name promises.
    fn would_lose_work(&self, action: &PendingAction) -> bool {
        self.unsaved || (matches!(action, PendingAction::Import) && self.doc.body_count() > 1)
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
    /// Remesh after the WHOLE model has been replaced.
    ///
    /// Empties the mesh pool first, because the pool's allocator never splits
    /// or merges blocks: a rebuild changes every brick's mesh size at once, so
    /// the free lists fill with granule classes nothing asks for again while
    /// the bump pointer climbs. Two or three trips up and down the detail
    /// buttons exhausted an 11M vertex pool with roughly 7.4M live -- observed
    /// on the dragon, 2755 bricks missing from the screen, on 2026-08-22.
    ///
    /// The reset is only sound paired with marking everything dirty, so the
    /// two live in one function and neither is callable by accident.
    ///
    /// # This is also why no swap site marks the OUTGOING model's bricks
    ///
    /// `reset_sculpt`, `open_project`, `adopt_import` and `orient` each used to
    /// collect the departing volume's brick coordinates and mark them dirty in
    /// the incoming one, so that they would mesh to nothing and release their
    /// pool slices. All four of those loops were dead, and had been since
    /// before the pool learned about bodies: all four call this function, and
    /// `MeshPool::reset` clears its slot map wholesale regardless of what any
    /// key says. The loops cost a walk of the outgoing model plus a remesh of
    /// however many coordinates it had, to release slices that had already
    /// been released.
    ///
    /// The reason to say so here rather than to leave four comments behind is
    /// that the trick reads as necessary, and it is the thing a fifth swap site
    /// would copy. **The rule is: call this, and the outgoing model is gone.**
    /// Anything that removes ONE body while the others stay is a different
    /// problem and has a different answer -- `SharedFrame::forget_body`.
    fn rebuild_everything(&mut self) {
        self.shared.request_pool_reset();
        self.doc.mark_everything_dirty();
        self.remesh_dirty();
        // Solo cannot survive a whole-document swap, and the reason it is
        // cleared here rather than left to `forget_a_vanished_solo` is that the
        // incoming document numbers its rows from 1 again: a stale id does not
        // dangle, it names a real and perfectly innocent body in the new
        // document and hides everything else. Three of the four callers replace
        // the whole document, and the fourth -- `orient` -- turns the model
        // without changing which rows exist, so clearing there costs the user a
        // mode they can press one button to get back.
        self.solo = None;
        // Same shape of problem for the eye: an id the outgoing document had
        // hidden could name a perfectly visible body in the incoming one.
        // `update` would put that right on the next message, and the next
        // message is at worst one frame away, but "at worst one frame" is how
        // long a body would be missing from the screen with nothing in the
        // panel saying why.
        self.publish_visibility();
    }

    fn remesh_dirty(&mut self) {
        self.doc.take_dirty(&mut self.dirty);
        self.perf.dirty_bricks = self.dirty.len();
        if self.dirty.is_empty() {
            return;
        }

        let started = Instant::now();
        let count = self.dirty.len();
        while self.mesh_buffers.len() < count {
            self.mesh_buffers.push(self.shared.take_mesh());
        }
        // Across every core, and across every BODY in one call: the threshold
        // that chooses between the serial and the parallel path counts the
        // coordinates in one call, so meshing body by body would drop a
        // document's worth of scattered dirty bricks onto a single thread. At
        // the sizes M2 targets the parallel path is the difference between a
        // remesh at 70 percent of its budget and one at under 10.
        self.doc.mesh_dirty(&self.dirty, &mut self.mesh_buffers[..count]);
        for ((body, coord), mesh) in self.dirty.iter().zip(self.mesh_buffers.drain(..count)) {
            self.shared.publish(*body, *coord, mesh);
        }
        self.perf.remesh_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.doc_stats = self.doc.totals();

        // Dynamic resizes the brush when the MODEL changed under it. Comparing
        // against a radius measured on a DIFFERENT body would resize it because
        // the selection changed, which is a different thing entirely and must
        // never happen -- see `rescale_radius`. So the comparison is made only
        // when the previous measurement was of the same body.
        let same_body = self.model_radius_body == self.doc.active();
        let previous = self.model_radius;
        self.refresh_model_radius();
        if same_body {
            self.rescale_radius(previous);
        }
    }

    /// Re-measure the active body, for the camera, the mirror plane and the
    /// Dynamic brush.
    ///
    /// **Separate from [`Brokkr::rescale_radius`] on purpose.** This one is
    /// safe to call after a selection change; that one is not.
    fn refresh_model_radius(&mut self) {
        self.model_radius = self.doc.active_volume().content_radius().unwrap_or(MODEL_RADIUS_MM);
        self.model_radius_body = self.doc.active();
    }

    /// Choose the body that edits land on, with everything that has to follow.
    ///
    /// The one place a selection changes, so that the three things which follow
    /// one cannot be forgotten: the measured radius the camera and the mirror
    /// planes are sized by, the mirror axes the newly chosen body may not be
    /// allowed to use, and a status line saying what was chosen — there is no
    /// panel to show it yet.
    ///
    /// **It re-measures and must never rescale the brush**: selecting changes
    /// no geometry, and a Dynamic brush that resized because the selection
    /// changed would carve at a radius the user never chose. See
    /// [`Brokkr::rescale_radius`], which spells out that failure in full.
    ///
    /// **It deliberately does not set `unsaved`.** Which row is active is
    /// written to the file, so this does leave the document a shade out of step
    /// with what is on disk; raising a "discard your work?" prompt because
    /// somebody clicked a different body would be far worse, and no geometry is
    /// at stake either way.
    ///
    /// **A press on a FOLDER row selects the first body inside it**, because
    /// the active row always holds a field -- that invariant is what keeps
    /// `Option<NodeId>` out of every signature downstream, and it is not worth
    /// trading for a second notion of "which row is selected" that the brush,
    /// the mirror and the solo scope would all then have to disambiguate
    /// against. A folder's own row keeps its own affordances: the chevron, the
    /// eye and the trash all name the folder they sit on.
    fn select_body(&mut self, row: NodeId) {
        let Some(body) = self.first_body_in(row) else {
            return;
        };
        let name = self.doc.node(body).map_or("another body", |node| node.name.as_str());
        let note = format!("selected {name}");
        self.select_body_saying(body, note);
    }

    /// The row itself when it holds a field, otherwise the first body under it.
    ///
    /// `None` only for a row that is not in the document; a folder always has
    /// at least one child, and a document always has at least one body.
    fn first_body_in(&self, row: NodeId) -> Option<NodeId> {
        let range = self.doc.subtree_of(row)?;
        self.doc.nodes()[range].iter().find(|node| node.is_body()).map(|node| node.id)
    }

    /// [`Brokkr::select_body`] with the caller's own line instead of "selected
    /// X".
    ///
    /// The note has to be set HERE rather than by the caller afterwards,
    /// because the mirror refusal below must have the last word: which body is
    /// selected is visible in the panel, and a mirror plane going off is not
    /// visible anywhere else. A caller that set its own status after selecting
    /// would silently eat that refusal.
    fn select_body_saying(&mut self, body: NodeId, note: String) {
        if body == self.doc.active() {
            return;
        }
        self.doc.set_active(body);
        self.refresh_model_radius();
        self.status = note;
        self.refuse_mirrors_the_body_does_not_straddle("turned off");
    }

    /// Add one primitive as a NEW body, and select it.
    ///
    /// **The whole feature, from the user's chair: press `+`, pick Cube, and a
    /// cube appears in the list and on screen.** Everything in the eight
    /// increments before this one exists so that these few lines are the only
    /// thing that had to be written to get there.
    ///
    /// One [`Change::NodeAdded`], so ctrl+Z removes it and ctrl+shift+Z brings
    /// it back. **The camera does not move**: framing is something the user set,
    /// and a tool that re-frames on every add makes a twelve-body layout
    /// impossible to build.
    fn add_primitive(&mut self, kind: brokkr_core::PrimitiveKind) {
        // Refused BEFORE anything allocates, and named rather than silently
        // ignored. The reader's own clamps do not cover this: nothing built by
        // the interface goes through the reader.
        if self.doc.body_count() >= brokkr_core::MAX_BODIES {
            self.status = format!(
                "could not add a {}: this document holds {} bodies, which is the limit",
                kind.label().to_lowercase(),
                self.doc.body_count()
            );
            return;
        }
        if self.doc.node_count() >= brokkr_core::MAX_NODES {
            self.status = format!(
                "could not add a {}: this document holds {} rows, which is the limit",
                kind.label().to_lowercase(),
                self.doc.node_count()
            );
            return;
        }

        let (centre, half) = brokkr_core::primitive::placement(&self.doc, MODEL_RADIUS_MM);

        // The size ceilings, checked BEFORE `build` allocates a single brick.
        // `MAX_BODIES` is nowhere near the binding limit: a cube is sized off
        // the biggest body, so at a fine voxel the very first one can be tens
        // of gigabytes, and sixty-four of anything is not what runs the machine
        // out of room. The pool figure is the WATERMARK, because adding a body
        // empties nothing and the bump pointer is what overflows -- see
        // `GrowthGuard`, whose doc comment is the argument in full.
        let pool = self.shared.stats();
        // A capacity of zero is a pool that has never reported: before the
        // first frame, and in every headless test. Judging against it would
        // refuse every add there has ever been, so the vertex ceiling is simply
        // not applied and the memory one still is. The resample guard makes the
        // same call one line into itself -- `if pool.vertices_reserved == 0 {
        // return None }` -- for the same reason.
        let headroom = if pool.vertex_capacity == 0 {
            u64::MAX
        } else {
            pool.vertex_capacity.saturating_sub(pool.vertices_watermark)
        };
        let (bytes, vertices) = brokkr_core::primitive::cost(kind, self.doc.voxel_size(), half);
        if let Some((why, workable)) = self.doc.growth_guard(headroom).no_room_for(bytes, vertices)
        {
            self.status = format!(
                "could not add a {} {:.1} mm across: {why} ({:.1} mm)",
                kind.label().to_lowercase(),
                half * 2.0,
                half * 2.0 * workable,
            );
            return;
        }

        let volume = brokkr_core::primitive::build(kind, self.doc.voxel_size(), centre, half);
        let id = self.doc.add_body(kind.label(), volume);
        let at = self.doc.node_count() - 1;

        let before = self.history.stats();
        self.history.push(Entry::new(vec![Change::NodeAdded { at, id }]));
        self.record_history(before);
        self.unsaved = true;

        let note = format!("added a {} {:.1} mm across", kind.label().to_lowercase(), half * 2.0);
        self.select_body_saying(id, note);
        self.remesh_dirty();
        self.refresh_overlay();
    }

    /// Copy the active body in place, as a new row directly below it.
    ///
    /// **In place, with no offset, and that is the reference behaviour rather
    /// than a shortcut.** Photoshop and ZBrush both duplicate where the
    /// original stands. There is no move gizmo in this application, so an
    /// auto-offset copy would land in a position the user then cannot adjust;
    /// the row appearing in the list is the feedback, which is why the "a
    /// primitive at the origin is invisible on the first press" argument that
    /// governs [`Brokkr::add_primitive`] does not carry over here.
    ///
    /// **ZBrush is inverted on all three counts, deliberately.** Duplicating
    /// "object" there renames the ORIGINAL to "object1", hands the copy the
    /// original's name, and leaves the original selected. Users worked that out
    /// from their own undo history, and it breaks GoZ round-trips because there
    /// the name is the identity key. Here the original keeps its name, the copy
    /// gets " copy", and the copy becomes active -- names are free to collide
    /// because [`NodeId`] is the identity and nothing downstream keys off one.
    ///
    /// # What it costs, and what is refused rather than evicted
    ///
    /// The copy is the expensive operation in this application: 765 MB of
    /// dragon is 6,120 dense bricks, so it is 6,120 allocations and 1.53 GiB of
    /// memory traffic, and [`Volume::duplicated`]'s own header is why that is
    /// spelled at the call site instead of hidden behind a `.clone()`.
    ///
    /// So it is refused before it allocates, against both ceilings, with the
    /// numbers in the message -- never allowed through to be recovered from
    /// afterwards. The history is not the constraint: one `Change::NodeAdded`
    /// is a few bytes against a 256 MB budget, and it is the RAM and the mesh
    /// pool that the second copy of a large body runs out of.
    fn duplicate_active_body(&mut self) {
        let id = self.doc.active();
        // Refused BEFORE anything allocates, and named rather than silently
        // ignored, exactly as the add path refuses. Nothing built by the
        // interface goes through the reader, so the reader's clamps do not
        // cover this.
        if self.doc.body_count() >= brokkr_core::MAX_BODIES {
            self.status = format!(
                "could not duplicate: this document holds {} bodies, which is the limit",
                self.doc.body_count()
            );
            return;
        }
        if self.doc.node_count() >= brokkr_core::MAX_NODES {
            self.status = format!(
                "could not duplicate: this document holds {} rows, which is the limit",
                self.doc.node_count()
            );
            return;
        }
        let Some(at) = self.doc.index_of(id) else {
            return;
        };
        let Some(node) = self.doc.node(id) else {
            return;
        };
        let name = node.name.clone();
        // Read before the copy is built, because the copy has to land BESIDE
        // its source rather than at the top level: a body inside a folder that
        // is duplicated at depth 0 ends the folder's run at the copy, and every
        // sibling below it falls out of the folder.
        let depth = node.depth();
        let Some(source) = node.volume() else {
            // A folder, once folders exist. Duplicating a subtree is its own
            // operation and belongs with the increment that can build one.
            return;
        };

        // Measured and not estimated, which no other add path in this
        // application can say: the copy is this body's brick map again, brick
        // for brick, so the voxel data it will hold is the voxel data this one
        // holds, to the byte. Only the brick map's own capacity can differ, and
        // that is the small term.
        let bytes = source.stats().resident_bytes as f64;
        if let Some(why) = self.no_room_to_duplicate(bytes) {
            self.status = format!("could not duplicate {name}: {why}");
            return;
        }

        // Whole bricks, and zero of them. See `Volume::duplicated` for why the
        // offset is measured in bricks at all: anything finer would be a
        // resample of the field rather than a copy of it.
        let copy = source.duplicated(glam::IVec3::ZERO);
        let copy_name = brokkr_core::name_that_fits(&format!("{name} copy")).to_string();
        // Directly below the row it came from and at the same depth, which is
        // where the user is looking and inside whatever folder the source sits
        // in. `insert_body` and not `add_body`: at sixty-four rows the bottom
        // of the list is off screen.
        let new_id = self.doc.insert_body(at + 1, depth, copy_name.clone(), copy);

        let before = self.history.stats();
        self.history.push(Entry::new(vec![Change::NodeAdded { at: at + 1, id: new_id }]));
        self.record_history(before);
        self.unsaved = true;

        self.select_body_saying(new_id, format!("duplicated {name} as {copy_name}"));
        self.remesh_dirty();
        self.refresh_overlay();
    }

    /// Why a second copy of a body costing `bytes` will not fit, or `None` to
    /// go ahead.
    ///
    /// # The vertex figure is apportioned from the pool, not estimated
    ///
    /// [`brokkr_core::primitive::cost`] predicts vertices from a closed-form
    /// surface area, because a cube that does not exist yet has no other
    /// answer. A duplicate does: the body is meshed and its triangles are on
    /// the GPU right now, and the copy's meshes are bit-identical to them. So
    /// the honest basis is what the pool has actually RESERVED, taken as this
    /// body's share of the document -- which is the same call
    /// [`Brokkr::too_fine_for_the_pool`] makes one screen up, and for the same
    /// reason its header gives: starting from a measured reservation bakes in
    /// the allocator's own padding rather than guessing at it.
    ///
    /// The share is by resident bytes and the two figures are of the same
    /// vintage -- `doc_stats` and the pool snapshot are both written by a
    /// remesh. Bytes are the right apportionment because both costs are a shell
    /// over a surface: a body holding a third of the document's voxel data has
    /// about a third of its surface and therefore about a third of its
    /// vertices.
    ///
    /// # A pool that has never reported does not veto
    ///
    /// `vertex_capacity` is zero before the first frame and in every headless
    /// test. Judging against it would refuse every duplicate there has ever
    /// been, so the vertex ceiling is simply not applied there and the memory
    /// one still is -- the same short-circuit `add_primitive` and the resample
    /// guard each make, one line into themselves.
    fn no_room_to_duplicate(&self, bytes: f64) -> Option<String> {
        let pool = self.shared.stats();
        let (headroom, vertices) = if pool.vertex_capacity == 0 {
            (u64::MAX, 0.0)
        } else {
            // The WATERMARK, never `reserved`: a duplicate empties nothing, so
            // what runs the pool out of room is the bump pointer rather than
            // the live count. `GrowthGuard`'s own header is the argument in
            // full, and this project has shipped the other reading twice.
            let headroom = pool.vertex_capacity.saturating_sub(pool.vertices_watermark);
            let share = bytes / self.doc_stats.resident_bytes.max(1) as f64;
            (headroom, pool.vertices_reserved as f64 * share)
        };
        self.doc.growth_guard(headroom).no_room_for_a_copy(bytes, vertices)
    }

    /// Merge the active body down into the body directly below it, asking first
    /// when the one entry it pushes may be too big for undo to hold.
    ///
    /// **The union of two signed distance fields is their `min`, and on a shared
    /// lattice that is all it is** -- brick `c` covers the same world box in
    /// both bodies, so there is no resampling and nothing is interpolated. See
    /// `brokkr_core::merge` for the engine half; everything here is the
    /// question of whether to run it and what to say afterwards.
    ///
    /// # Refused by name, never greyed and never silent
    ///
    /// The two ways there is nothing to merge into -- the bottom of a list, and
    /// a folder on the next line -- are said out loud, exactly as the other
    /// verbs in that row refuse. A merge into a folder is ZBrush's MergeVisible
    /// and its universal "the button did nothing" reaction; naming the folder is
    /// the whole difference.
    fn merge_active_body_down(&mut self) {
        let source = self.doc.active();
        let name = self.doc.node(source).map_or_else(String::new, |node| node.name.clone());
        let target = match self.doc.merge_target(source) {
            brokkr_core::MergeTarget::Body(target) => target,
            brokkr_core::MergeTarget::Bottom => {
                self.status = format!("could not merge {name}: there is no body below it");
                return;
            }
            brokkr_core::MergeTarget::Folder(folder) => {
                let folder = self
                    .doc
                    .node(folder)
                    .map_or_else(|| "a folder".to_string(), |node| node.name.clone());
                self.status = format!(
                    "could not merge {name}: {folder} is a folder, and a merge joins two bodies"
                );
                return;
            }
        };

        // Microseconds and no allocation: two map lookups per brick of the
        // SOURCE, which is what makes it affordable to ask before the merge
        // rather than to discover the size afterwards.
        let Some(plan) = self.doc.merge_plan(source) else {
            return;
        };
        if plan.bytes() >= self.merge_prompt_bytes {
            let target_name =
                self.doc.node(target).map_or_else(String::new, |node| node.name.clone());
            self.pending_merge = Some(PendingMerge {
                source,
                source_name: name,
                target_name,
                bytes: plan.bytes(),
                stroke_bytes: plan.stroke_bytes,
                reclaim_bytes: plan.reclaim_bytes,
            });
            return;
        }
        self.apply_merge(source);
    }

    /// Run the merge and put the renderer back in step with it.
    ///
    /// **The three lines after the merge are the same unit `Brokkr::remove_body`
    /// documents, and for the same reason**: a merge consumes a body, that
    /// body's bricks are in nobody's dirty set any more, so
    /// `SharedFrame::forget_body` is what releases its slots -- queued rather
    /// than called -- and the whole-document remesh is the other half.
    /// `MeshPool::forget_body` clears the pool-full banner when it frees space,
    /// and a brick the pool refused while it was full was dropped with its
    /// coordinate long gone from the dirty set; freeing the space without
    /// re-offering everything takes the warning down and leaves the geometry
    /// missing.
    ///
    /// # This is deliberately NOT `rebuild_everything`, and the plan asked for
    /// one
    ///
    /// The plan for this increment says to call it when `PoolStats.overflowed`
    /// is non-zero afterwards. That was written before increment 6 landed
    /// `forget_body`, which now does strictly more of what the rebuild was
    /// wanted for -- it resets the allocators of any buffer pair it empties and
    /// clears `overflowed` itself -- and it does it without the one thing
    /// `rebuild_everything` also does: **clear solo.** Solo is cleared there
    /// because a whole-document swap renumbers the rows, and a merge is not a
    /// swap. Dropping the user's view mode on a merge would be a regression
    /// against increment 13 for no gain, so the rebuild is not called and this
    /// paragraph is why.
    fn apply_merge(&mut self, source: brokkr_core::NodeId) {
        let source_name = self.doc.node(source).map_or_else(String::new, |node| node.name.clone());
        let Some(outcome) = self.doc.merge_down(source) else {
            return;
        };
        let target_name =
            self.doc.node(outcome.target).map_or_else(String::new, |node| node.name.clone());
        self.shared.forget_body(source);
        self.doc.mark_everything_dirty();

        let before = self.history.stats();
        self.history.push(outcome.entry);
        self.unsaved = true;
        self.status = if outcome.bricks == 0 {
            format!("merged {source_name} into {target_name} — it was already inside it")
        } else {
            format!("merged {source_name} into {target_name} — {} bricks changed", outcome.bricks)
        };
        // After the merge's own line, so an eviction wins the status.
        self.record_history(before);
        self.refresh_model_radius();
        self.remesh_dirty();
        self.refresh_overlay();
    }

    /// Begin renaming a row: hold its current name as the field's text, and
    /// return the task that focuses the field and selects what is in it.
    ///
    /// Select-all rather than a caret at the end, because a rename is far more
    /// often a replacement than an edit -- "Cube 2" is a placeholder, and
    /// having to clear it first would make every rename two gestures.
    fn begin_rename(&mut self, id: NodeId) -> Task<Message> {
        let Some(node) = self.doc.node(id) else {
            return Task::none();
        };
        self.renaming = Some((id, node.name.clone()));
        // Chained rather than batched: `select_all` acts on the field's cursor
        // and there is no reason to find out what a runtime that ran them the
        // other way round would do to it.
        iced::widget::operation::focus(RENAME_FIELD)
            .chain(iced::widget::operation::select_all(RENAME_FIELD))
    }

    /// Put the typed name on the row, as one undoable change.
    ///
    /// **The single commit point**, called from [`Brokkr::update`]'s guard and
    /// nowhere else, so that Enter, a click on another row, a press in the
    /// viewport and a menu command all end a rename the same way. See
    /// [`keeps_the_rename_open`].
    ///
    /// Three things it refuses, each because the alternative is a silent
    /// change the user never sees:
    ///
    /// * a name that is only whitespace. The reader repairs an empty name field
    ///   to `Body {n}` (`project.rs`, `read_name`), so committing one would
    ///   rename the body to something else entirely on the next open;
    /// * a name equal to the one already there, which would cost a real undo
    ///   press for a change that is not one;
    /// * a row that has gone. Nothing can reach this today -- the guard runs
    ///   before `dispatch`, so the message that deletes a row commits the
    ///   rename against the document that still holds it -- and it is a
    ///   `let else` rather than an `expect` because "a rename outlived its
    ///   row" is not worth a crash.
    ///
    /// The clamp to [`brokkr_core::MAX_NAME_BYTES`] is applied on the way IN,
    /// in the `BodyRenameEdited` arm, so that the field cannot show a
    /// thirty-third byte that the file would then drop. It is applied again
    /// here anyway: this is the last point before the document, and one line
    /// is cheaper than proving that no other route into `renaming` will ever
    /// exist.
    fn commit_rename(&mut self) {
        let Some((id, typed)) = self.renaming.take() else {
            return;
        };
        let Some(before) = self.doc.meta(id) else {
            return;
        };
        // Trimmed on both sides of the cut: leading and trailing space first,
        // then the cut, then the tail again in case the cut landed just after
        // a space.
        let name = brokkr_core::name_that_fits(typed.trim()).trim_end();
        if name.is_empty() {
            self.status = format!("a body needs a name — kept {}", before.name);
            return;
        }
        if name == before.name {
            return;
        }

        let after = NodeMeta { name: name.to_string(), ..before.clone() };
        let change = self.outline_change(|doc| doc.set_meta(&after));
        let stats = self.history.stats();
        self.history.push(Entry::new(vec![change]));
        self.record_history(stats);
        // The name is written to the file, so changing it is a change to the
        // document.
        self.unsaved = true;
        self.status = format!("renamed {} to {name}", before.name);
    }

    /// Snapshot the outline, make one edit to it, and hand back the change that
    /// undoes it.
    ///
    /// [`Change::Outline`] is a permutation plus field edits over a fixed id
    /// set, so its two halves have to be taken either side of the same
    /// mutation. Getting that wrong is silent -- the undo reapplies the state
    /// it was meant to replace -- so the pairing lives here rather than at each
    /// call site.
    ///
    /// **[`Brokkr::toggle_visibility`] deliberately does not use it**, and that
    /// is the one exception: it has to look at the RESOLVED mask between the
    /// two snapshots and may put the eye back and refuse, so its `before` has
    /// to be taken before a mutation it might undo by hand.
    ///
    /// It costs two whole-outline clones -- about 10 KB at 128 rows -- and it
    /// runs at a user action, never per frame.
    fn outline_change(&mut self, edit: impl FnOnce(&mut brokkr_core::Document)) -> Change {
        let before = self.doc.outline();
        edit(&mut self.doc);
        Change::Outline { before, after: self.doc.outline() }
    }

    /// Throw a rename away and leave the row exactly as it was.
    ///
    /// Escape, by way of `MenuClosed`. Nothing to undo, because nothing was
    /// ever written to the document: the typed text only ever lived in
    /// `renaming`.
    fn cancel_rename(&mut self) {
        self.renaming = None;
    }

    /// Flip one row's OWN eye, as one undoable change.
    ///
    /// **Hiding the body that is active moves the selection**, evaluated against
    /// [`Document::saved_visibility`] and never against what solo is showing: a
    /// view mode must not be able to veto a structural rule. When there is no
    /// other visible body to move to, the hide is refused outright rather than
    /// leaving the application with an active body nobody can see and every
    /// press reporting that it is hidden.
    ///
    /// # An eye click outside the solo scope is refused, and that is not fussy
    ///
    /// While solo is on, every out-of-scope row still shows an open eye, because
    /// its own bit really is on -- solo is a mask over that bit and never a
    /// write to it. So ten rows claim "visible" over a viewport drawing one.
    /// Letting the click through means the user turns an eye off, sees nothing
    /// change, clicks again, sees nothing change -- and each of those clicks
    /// sets `unsaved` and arms an autosave of a multi-gigabyte document, with
    /// the whole effect arriving later, when they leave the mode. The refusal
    /// costs a status line and names the mode that caused it.
    fn toggle_visibility(&mut self, id: NodeId) {
        let Some(before_meta) = self.doc.meta(id) else {
            return;
        };
        if !self.in_solo_scope(id) {
            self.status =
                format!("solo is on — leave it before changing {}'s eye", before_meta.name);
            return;
        }
        let outline_before = self.doc.outline();
        let after = NodeMeta { visible: !before_meta.visible, ..before_meta.clone() };
        self.doc.set_meta(&after);

        // Resolved rather than read off the bit, because an ancestor folder can
        // hide a row whose own eye is on.
        let mut saved = Vec::new();
        self.doc.saved_visibility(&mut saved);
        let active_is_hidden = self
            .doc
            .index_of(self.doc.active())
            .and_then(|index| saved.get(index))
            .is_some_and(|shown| !shown);

        let mut note =
            format!("{} {}", if after.visible { "showing" } else { "hiding" }, after.name);
        let mut moved_to = None;
        if active_is_hidden {
            let replacement = self
                .doc
                .nodes()
                .iter()
                .zip(&saved)
                .find(|(node, shown)| **shown && node.is_body())
                .map(|(node, _)| (node.id, node.name.clone()));
            match replacement {
                Some((next, name)) => {
                    note = format!("hiding {} — {name} is now selected", after.name);
                    moved_to = Some(next);
                }
                None => {
                    self.doc.set_meta(&before_meta);
                    self.status =
                        "cannot hide the last visible body — there would be nothing to sculpt"
                            .to_string();
                    return;
                }
            }
        }

        let before = self.history.stats();
        self.history.push(Entry::new(vec![Change::Outline {
            before: outline_before,
            after: self.doc.outline(),
        }]));
        self.record_history(before);
        // The eye is written to the file, so flipping it is a change to the
        // document.
        self.unsaved = true;
        match moved_to {
            Some(next) => self.select_body_saying(next, note),
            None => self.status = note,
        }
    }

    /// Turn every eye in the document on, as ONE undo entry, and leave solo.
    ///
    /// The way out of having hidden things and lost track of what. A no-op when
    /// nothing is hidden, and it says so rather than pushing an empty entry that
    /// would cost the user a real undo.
    ///
    /// One [`Change::Outline`] whatever the count, because the outline snapshot
    /// is whole-document either way: N eyes for the price of one.
    ///
    /// **Solo goes first and unconditionally**, before the "already showing"
    /// return: a document with every eye on and solo still on shows exactly one
    /// subtree, which is the opposite of what the user just asked for. This is
    /// solo's second exit and the lesser one -- it is also a document change,
    /// and Escape exists precisely so that leaving the mode need not be.
    fn show_everything(&mut self) {
        let left_solo = self.solo.take().is_some();
        let hidden: Vec<NodeMeta> = self
            .doc
            .nodes()
            .iter()
            .filter(|node| !node.visible)
            .map(|node| NodeMeta { visible: true, ..node.meta() })
            .collect();
        if hidden.is_empty() {
            self.status = if left_solo {
                "left solo — everything is showing".to_string()
            } else {
                "everything is already showing".to_string()
            };
            return;
        }

        let shown = hidden.len();
        let change = self.outline_change(|doc| {
            for meta in &hidden {
                doc.set_meta(meta);
            }
        });
        let before = self.history.stats();
        self.history.push(Entry::new(vec![change]));
        self.record_history(before);
        self.unsaved = true;
        let rows = if shown == 1 { "row" } else { "rows" };
        self.status = if left_solo {
            format!("left solo, showing {shown} hidden {rows}")
        } else {
            format!("showing {shown} hidden {rows}")
        };
    }

    // --- solo, which is a MODE ------------------------------------------------

    /// Whether a row is inside the solo scope, and so whether a click on it
    /// means what it looks like.
    ///
    /// `true` whenever solo is off, which is what lets every caller be a plain
    /// guard rather than an `Option` dance. A subtree is a contiguous preorder
    /// run, so this is a range test and never a search -- the same property
    /// [`brokkr_core::resolve_visibility`] leans on to make the scope test one
    /// integer comparison.
    ///
    /// A solo naming a row that is not in the document answers `true` for
    /// everything, which is the safe direction: the answer only ever refuses,
    /// and [`Brokkr::forget_a_vanished_solo`] clears that state on the same
    /// pass anyway.
    fn in_solo_scope(&self, id: NodeId) -> bool {
        let Some(solo) = self.solo else {
            return true;
        };
        match (self.doc.subtree_of(solo), self.doc.index_of(id)) {
            (Some(range), Some(at)) => range.contains(&at),
            _ => true,
        }
    }

    /// Show only this row's subtree.
    ///
    /// **Nothing about the mode is written to the document**, which is the whole
    /// design: leaving it restores the hand-set eyes bit for bit because none of
    /// them was ever touched. See [`Brokkr::solo`].
    ///
    /// # Soloing a hidden row turns its eye on, and that part IS an edit
    ///
    /// [`brokkr_core::resolve_visibility`] narrows and never widens -- soloing a
    /// row whose eye is off would otherwise leave the screen empty with the
    /// indicator claiming to be showing something. The resolver deliberately
    /// does not rewrite a bit the user set, and says so; it is this handler's
    /// job, and here it is an ordinary undoable change that sets `unsaved`,
    /// exactly as clicking the eye would be.
    ///
    /// The ancestors go with it. An ancestor's eye is an AND-mask, so turning on
    /// a row inside a hidden folder and stopping there shows nothing at all --
    /// the same empty screen by a longer route. At most [`brokkr_core::MAX_DEPTH`]
    /// rows are touched and they land in ONE [`Change::Outline`], because the
    /// snapshot is whole-document either way.
    fn enter_solo(&mut self, id: NodeId) {
        let Some(node) = self.doc.node(id) else {
            return;
        };
        let name = node.name.clone();

        // The row and every folder above it. Bounded by `MAX_DEPTH`, and walked
        // upward through `parent_of`, which preorder makes a backward scan
        // rather than a search.
        let mut chain = vec![id];
        while let Some(parent) = self.doc.parent_of(*chain.last().expect("seeded with the row")) {
            chain.push(parent);
        }
        let closed: Vec<NodeMeta> = chain
            .iter()
            .filter_map(|row| self.doc.meta(*row))
            .filter(|meta| !meta.visible)
            .map(|meta| NodeMeta { visible: true, ..meta })
            .collect();

        self.solo = Some(id);
        if closed.is_empty() {
            self.status = format!("solo: {name} — escape leaves it");
            return;
        }

        let change = self.outline_change(|doc| {
            for meta in &closed {
                doc.set_meta(meta);
            }
        });
        let before = self.history.stats();
        self.history.push(Entry::new(vec![change]));
        self.record_history(before);
        // Turning an eye on is written to the file, so this half of the gesture
        // is a change to the document even though the mode itself is not.
        self.unsaved = true;
        self.status = format!("solo: {name} — its eye was off and is now on");
    }

    /// Leave solo.
    ///
    /// **It restores nothing, because it changed nothing.** That single line is
    /// what the mode buys over the saved-visibility vector every other tool
    /// ships: Photoshop's alt-click eye remembers the previous set only "if you
    /// haven't changed anything else", Plasticity's manual says Unisolate
    /// "does not step back to the previous hierarchical isolation layer --
    /// everything becomes visible instead", and Blender's Alt+H is documented as
    /// ruining the scene configuration. Blender's Local View is the mode version
    /// and it exits correctly, for exactly this reason.
    ///
    /// Silent when solo was already off, so that Escape falling through to the
    /// armed cut does not first announce leaving a mode nobody was in.
    fn exit_solo(&mut self) {
        if self.solo.take().is_some() {
            self.status = "left solo".to_string();
        }
    }

    /// Delete the active body, asking first when undo may not be able to hold
    /// it.
    ///
    /// **The active row is always a body**, so this can never take a folder by
    /// accident -- which is how "deleting a body inside a collapsed folder
    /// deletes the body, never the folder" is guaranteed rather than
    /// remembered. ZBrush gets that wrong and a user reported losing an
    /// unrecoverable hour to it. Deleting a folder is [`Brokkr::delete_folder`],
    /// reached from that folder's own row and from nowhere else.
    fn delete_active_body(&mut self) {
        self.delete_row(self.doc.active());
    }

    /// Delete a folder and everything in it, asking first when undo may not be
    /// able to hold the lot.
    ///
    /// Its own entry point rather than a mode of the Delete verb, and that is
    /// the structural half of the rule above: the verb names the active body
    /// and this names a folder, so no state of the panel -- collapsed least of
    /// all -- can make one of them do the other's job.
    fn delete_folder(&mut self, id: NodeId) {
        if self.doc.node(id).is_none_or(brokkr_core::Node::is_body) {
            return;
        }
        self.delete_row(id);
    }

    /// One row and its whole subtree, prompted on the SUM of what it takes.
    ///
    /// **The prompt is on the SIZE and nothing else**, and the threshold is the
    /// same 512 MB as the reclaim allowance because a delete that would be
    /// evicted before it could be undone is exactly the one that has to warn.
    /// It is measured over the whole subtree, so a folder delete asks about
    /// what it is really taking -- and folders make the prompt the common case
    /// rather than the exception.
    fn delete_row(&mut self, id: NodeId) {
        if self.doc.subtree_body_count(id) >= self.doc.body_count() {
            self.status = "cannot delete the last body — a sculpt always holds one".to_string();
            return;
        }
        let Some(node) = self.doc.node(id) else {
            return;
        };
        let name = node.name.clone();
        let bytes = self.subtree_bytes(id);
        if bytes >= self.delete_prompt_bytes {
            let bodies = self.doc.subtree_body_count(id);
            self.pending_delete = Some(PendingDelete { id, name, bytes, bodies });
            return;
        }
        self.remove_body(id);
    }

    /// What deleting this row would take with it, in bytes.
    ///
    /// A sum over the whole subtree, which for a body is the body and for a
    /// folder is every field beneath it. Walks each brick map, so it belongs to
    /// a user action and never to a frame.
    fn subtree_bytes(&self, id: NodeId) -> usize {
        let Some(range) = self.doc.subtree_of(id) else {
            return 0;
        };
        self.doc.nodes()[range]
            .iter()
            .filter_map(brokkr_core::Node::volume)
            .map(|volume| volume.stats().resident_bytes)
            .sum()
    }

    /// Take one row and its subtree out of the document, recording every node
    /// so undo can put the lot back.
    ///
    /// The volumes MOVE into the entry -- `Volume` has no `Clone` at all -- so a
    /// delete allocates nothing and peak memory does not rise; it merely does
    /// not fall until the entry is evicted.
    ///
    /// **N removals in ONE entry**, so one ctrl+Z restores a folder of three
    /// bodies whole. [`brokkr_core::Document::delete_subtree`] records them in
    /// the order that makes undo put the FOLDER back before the bodies, and a
    /// folder the delete leaves empty is dissolved into the same entry.
    ///
    /// **The three lines after the removal are one unit and increment 6 wrote
    /// down why.** A removed body's bricks are in nobody's dirty set, so
    /// `SharedFrame::forget_body` is what releases its slots -- queued rather
    /// than called, because the pool lives inside the pipeline Iced owns and
    /// releasing slots while that body's meshes still sit in `pending` would
    /// re-upload them into fresh slices on the same frame. And the whole-document
    /// remesh is the other half: `MeshPool::forget_body` clears the pool-full
    /// banner when it frees space, and a brick the pool refused while it was
    /// full was dropped on the floor with its coordinate long gone from the
    /// dirty set. Freeing the space without re-offering everything takes the
    /// warning down and leaves the geometry missing.
    fn remove_body(&mut self, id: NodeId) {
        let name = self.doc.node(id).map_or_else(String::new, |node| node.name.clone());
        let Some(changes) = self.doc.delete_subtree(id) else {
            return;
        };
        let mut bodies = 0usize;
        for change in &changes {
            if let Change::NodeRemoved { node, .. } = change
                && node.is_body()
            {
                bodies += 1;
                self.shared.forget_body(node.id);
            }
        }
        self.doc.mark_everything_dirty();

        let before = self.history.stats();
        self.history.push(Entry::new(changes));
        self.unsaved = true;
        self.status = if bodies > 1 {
            format!("deleted {name} and the {bodies} bodies in it")
        } else {
            format!("deleted {name}")
        };
        // After the delete's own line, so an eviction that took an older
        // deleted body with it wins the status.
        self.record_history(before);
        self.refresh_model_radius();
        self.remesh_dirty();
        self.refresh_overlay();
    }

    // --- folders -------------------------------------------------------------

    /// Wrap the active row in a new folder, in place, with no dialog.
    ///
    /// `ctrl+G`. Photoshop's own chord, and its own behaviour: the group
    /// appears where the row was and the row is inside it, still selected.
    /// Pressed again it nests -- which is how a tree gets deeper than one level
    /// without a drag, and it is the only way to reach depth seven before
    /// increment 17 lands.
    ///
    /// The name is the first `Group n` nothing else is using, counted over the
    /// folders that exist rather than over the ids handed out, so deleting
    /// Group 2 and grouping again gives Group 2 back rather than Group 9.
    fn group_active_body(&mut self) {
        let id = self.doc.active();
        if self.doc.node_count() >= brokkr_core::MAX_NODES {
            self.status = format!(
                "could not group: this document holds {} rows, which is the limit",
                self.doc.node_count()
            );
            return;
        }
        let name = self.next_group_name();
        let Some((folder, changes)) = self.doc.group(id, name.clone()) else {
            self.status = format!(
                "could not group: {} folders deep is as far as the panel goes",
                brokkr_core::MAX_DEPTH
            );
            return;
        };
        let _ = folder;

        let before = self.history.stats();
        self.history.push(Entry::new(changes));
        self.record_history(before);
        // The tree is written to the file, so grouping is a change to the
        // document.
        self.unsaved = true;
        self.status = format!("grouped into {name}");
    }

    /// Dissolve the folder the active row sits in; its children rise out of it.
    ///
    /// `ctrl+shift+G`. It acts on the PARENT rather than on the active row,
    /// because the active row is always a body and a body has nothing to
    /// dissolve -- and because that is what makes the pair symmetric: ctrl+G
    /// then ctrl+shift+G leaves the document exactly as it was.
    fn ungroup_active_body(&mut self) {
        let Some(parent) = self.doc.parent_of(self.doc.active()) else {
            self.status = "nothing to ungroup — this row is not in a folder".to_string();
            return;
        };
        let name = self.doc.node(parent).map_or_else(String::new, |node| node.name.clone());
        let Some(changes) = self.doc.ungroup(parent) else {
            return;
        };

        let before = self.history.stats();
        self.history.push(Entry::new(changes));
        self.record_history(before);
        self.unsaved = true;
        self.status = format!("dissolved {name}");
    }

    /// Re-parent the active row into a folder, or out to the top level.
    ///
    /// The panel's `Move to ▸` list. **Increment 17 deletes this the day a drag
    /// lands**: two routes to one operation is two sets of drop rules to keep
    /// consistent in a 214 px panel, and neither ever gets removed once
    /// shipped.
    fn move_to_folder(&mut self, into: Option<NodeId>) {
        let id = self.doc.active();
        let where_to = match into {
            Some(folder) => self
                .doc
                .node(folder)
                .map_or_else(|| "a folder".to_string(), |node| node.name.clone()),
            None => "the top level".to_string(),
        };
        // The list leaves the row's current container out, so this only fires
        // on a message that was queued before the list was rebuilt -- and
        // "already there" is a very different thing to be told from "could
        // not".
        if self.doc.parent_of(id) == into {
            self.status = format!("this body is already in {where_to}");
            return;
        }
        let Some(changes) = self.doc.move_to_folder(id, into) else {
            self.status = format!("could not move this body to {where_to}");
            return;
        };

        let before = self.history.stats();
        self.history.push(Entry::new(changes));
        self.record_history(before);
        self.unsaved = true;
        self.status = format!("moved to {where_to}");
    }

    /// Fold a folder's children away, or show them again.
    ///
    /// **Collapse changes only what is DRAWN.** It never changes what a command
    /// does, which is the ZBrush failure this whole design is written against:
    /// there, deleting a subtool inside a closed folder deletes the folder.
    ///
    /// Recorded like any other field of the outline, because `collapsed` is
    /// written to the file -- a change the next save keeps and no ctrl+Z can
    /// reach would be the only such change in the application.
    fn toggle_collapse(&mut self, folder: NodeId) {
        let Some(node) = self.doc.node(folder) else {
            return;
        };
        let collapsed = !node.collapsed;
        let Some(changes) = self.doc.set_collapsed(folder, collapsed) else {
            return;
        };

        let before = self.history.stats();
        self.history.push(Entry::new(changes));
        self.record_history(before);
        self.unsaved = true;
    }

    /// The first `Group n` no folder in the document is called.
    ///
    /// Counted over the names in use rather than off a running number, so the
    /// list does not creep up to `Group 40` in a document that has never held
    /// more than three folders. At most [`brokkr_core::MAX_NODES`] tries, and
    /// it runs once per ctrl+G.
    fn next_group_name(&self) -> String {
        (1..=brokkr_core::MAX_NODES + 1)
            .map(|n| format!("Group {n}"))
            .find(|name| self.doc.folders().all(|folder| &folder.name != name))
            .expect("more candidate names than there are rows to use them")
    }

    /// Turn off every enabled mirror plane the active body sits wholly to one
    /// side of, and say which.
    ///
    /// **This is the mitigation for the centre being the lattice origin.** With
    /// [`MIRROR_CENTRE`] pinned there, a dent carved into a body at x = +80 has
    /// its twin written at x = -80: free-floating geometry, in empty space, that
    /// exports as an extra shell no slicer can print and nothing on screen
    /// explains. Refusing the mirror is what stops the first primitive a user
    /// adds from growing one.
    ///
    /// Turning the axis off rather than keeping it on and quietly ignoring it
    /// is the honest form: there is then one answer to "is X mirroring on", the
    /// strip highlight and the drawn plane agree with the sculpt, and nothing
    /// has to carry an effective-versus-actual pair.
    ///
    /// # What it costs, and when it is allowed to run
    ///
    /// [`Volume::surface_bounds`] scans every dense brick and its own
    /// documentation forbids calling it per frame, so this asks only at a user
    /// action and only while a mirror is actually enabled — the `is_off` check
    /// is what keeps the common case free. The two call sites are enabling an
    /// axis and choosing a different body, which are the two ways the pairing
    /// of "this axis" with "this body" can change.
    ///
    /// **The residual, said out loud:** sculpting a body across the plane while
    /// the axis is off does not turn it back on by itself. The user toggles the
    /// axis and it is re-measured. Re-measuring per stroke would put a
    /// full-model scan on every button release, which is the thing the cache in
    /// `brokkr-core` refuses to hold for exactly this reason.
    fn refuse_mirrors_the_body_does_not_straddle(&mut self, verb: &str) {
        if self.symmetry.is_off() {
            return;
        }
        let Some(bounds) = self.doc.active_volume().surface_bounds() else {
            // No surface, so there is nothing to be on one side of. A mirror of
            // nothing is nothing.
            return;
        };
        for axis in MirrorAxis::ALL {
            if !self.symmetry.axis(axis) || straddles(bounds, axis) {
                continue;
            }
            self.symmetry = self.symmetry.with_axis(axis, false);
            self.status = self.mirror_refusal(axis, verb);
        }
    }

    /// Whether the active body has material on both sides of one mirror plane.
    ///
    /// Separate from the sweep above because enabling an axis asks about one
    /// axis and a selection asks about all three, and the sweep is what makes
    /// one scan of the field answer for all of them.
    fn mirror_straddles(&self, axis: MirrorAxis) -> bool {
        self.doc.active_volume().surface_bounds().is_none_or(|bounds| straddles(bounds, axis))
    }

    /// What the user is told when a mirror plane misses the body.
    fn mirror_refusal(&self, axis: MirrorAxis, verb: &str) -> String {
        let name = self.doc.node(self.doc.active()).map_or("the body", |node| node.name.as_str());
        format!(
            "{} mirroring {verb}: {name} sits entirely to one side of the {} plane, so every \
             mirrored stroke would land in empty space",
            axis.label(),
            axis.label(),
        )
    }

    /// Turn one mirror plane on or off, refusing to turn one on that the
    /// active body does not straddle.
    ///
    /// **The one gate every way of enabling a mirror goes through.** There is
    /// more than one way: the symmetry strip sends a message, and a SpaceMouse
    /// button mapped to `ToggleSymmetry` toggles X directly. While the refusal
    /// lived inside the message arm the SpaceMouse route turned X on
    /// unrefused, and every stroke after it wrote its twin into empty space --
    /// with the strip highlight and the drawn plane both agreeing that nothing
    /// was wrong, and the status line saying nothing at all. One function
    /// rather than two spellings of one.
    ///
    /// Turning one OFF is never refused: the refusal exists to stop material
    /// appearing where the user cannot see it, and nothing appears when a plane
    /// goes away.
    fn toggle_mirror(&mut self, axis: MirrorAxis) {
        if self.symmetry.axis(axis) {
            self.symmetry = self.symmetry.with_axis(axis, false);
        } else if self.mirror_straddles(axis) {
            self.symmetry = self.symmetry.with_axis(axis, true);
        } else {
            self.status = self.mirror_refusal(axis, "refused");
        }
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
        // The body the pick returned, and not the active one. The ring is
        // pushed onto the surface by sampling the field it is over, so building
        // it from a body that does not contain the hover point draws a
        // confident ring in mid air; see `cursor::build`.
        let hovered = self.hover_body.unwrap_or_else(|| self.doc.active());
        let volume = self.doc.volume(hovered).unwrap_or_else(|| self.doc.active_volume());
        let selecting = self.hover_body.is_some_and(|body| body != self.doc.active());
        cursor::build(
            &mut batch,
            volume,
            &self.effective_brush(),
            self.symmetry,
            MIRROR_CENTRE,
            self.hover,
            cursor::mood(self.stroke_direction(), self.sizing.is_some(), selecting),
            self.model_radius,
        );
        self.shared.swap_overlay(&mut batch);
        self.overlay = batch;
    }

    /// Where the pointer meets the surface, and on which body, remembered for
    /// the cursor ring and read by the press.
    fn update_hover(&mut self, pixel: Vec2) {
        match self.pick(pixel) {
            Some((body, hit)) => {
                self.hover = Some(hit.position);
                self.hover_body = Some(body);
            }
            None => {
                self.hover = None;
                self.hover_body = None;
            }
        }
    }

    /// The same, except that a live stroke owns the ring.
    ///
    /// **While a stroke is running the ring belongs to the body being carved
    /// and to no other.** [`Brokkr::update_hover`] picks across every drawn
    /// body, so dragging a stroke over a second body moved the ring onto that
    /// body's surface -- and [`Brokkr::refresh_overlay`] builds the ring out of
    /// the hovered body's field and colours it as `CursorMood::Selecting` the
    /// moment the hovered body is not the active one. The cursor was telling
    /// the user "a press here would select", on the wrong surface, during a
    /// press that was carving something else. The carve itself was never
    /// affected, because the stroke asks [`Brokkr::surface_under`]; only the
    /// overlay lied, which is worse than it sounds for a tool whose whole
    /// feedback loop is that ring.
    ///
    /// Asking `surface_under` here costs one march of one body, where the pick
    /// it replaces marched every drawn one, so the stroke path got cheaper
    /// rather than dearer.
    fn refresh_hover(&mut self, pixel: Vec2) {
        if !matches!(self.drag.map(|drag| drag.kind), Some(DragKind::Sculpt(_))) {
            self.update_hover(pixel);
            return;
        }
        self.hover = self.surface_under(pixel);
        // Paired with the position rather than set unconditionally: a stroke
        // dragged off the model has no ring, and a `hover_body` naming a body
        // no ring is being drawn against would be a fact with no owner.
        self.hover_body = self.hover.map(|_| self.doc.active());
    }

    /// The nearest DRAWN surface under a point in widget pixels, and the body
    /// it belongs to.
    ///
    /// The visibility mask is the one `publish_visibility` already keeps, so
    /// the pick, the renderer and the panel cannot disagree about what is on
    /// screen — and a body the user cannot see cannot be picked, hovered or
    /// carved.
    fn pick(&self, pixel: Vec2) -> Option<(NodeId, brokkr_core::Hit)> {
        let (origin, ray) = self.ray_through(pixel);
        self.doc.pick(origin, ray, self.camera.far, &self.shown)
    }

    /// Whether the body edits land on is one of the ones being drawn.
    ///
    /// **The pick alone does not answer this**, and that is the whole reason
    /// this exists. [`Document::pick`] refuses a hidden body, but a stroke
    /// takes its surface from [`Document::pick_body`], which asks one named
    /// body and deliberately does not consult the eye -- a live stroke keeps
    /// carving the body it started on. So a press while the ACTIVE body is
    /// hidden picks nothing, falls through to a sculpt, and then happily
    /// marches the invisible body and carves it.
    ///
    /// The mask is `publish_visibility`'s, which is recomputed after every
    /// message, so it is never stale by the time a pointer event reads it. A
    /// body that is not in the document at all reads as not drawn, which is
    /// the safe direction: the answer only ever refuses.
    fn active_is_drawn(&self) -> bool {
        self.doc
            .index_of(self.doc.active())
            .is_some_and(|index| self.shown.get(index).copied().unwrap_or(false))
    }

    /// Keep the brush a constant fraction of the model when asked to.
    ///
    /// ZBrush calls this Dynamic. Without it a brush tuned on a 60 mm ball is
    /// the wrong size the moment the model is resampled or grows.
    ///
    /// **This must never run because the SELECTION changed.** It exists to
    /// survive a *resample*, where the geometry moved under a fixed brush;
    /// choosing a different body changes no geometry at all. Once
    /// `model_radius` means the active body's, clicking a 5 mm rivet beside a
    /// 200 mm bust with Dynamic on would take a 5 mm brush to `5 * 0.025 =
    /// 0.125`, clamped up to [`MIN_RADIUS_MM`], and clicking back would take it
    /// to `0.25 * 40 = 10 mm` -- the brush doubled and the user never touched
    /// the slider. Worse, the press that resized it is the same press that
    /// selects, so the next press carves at a radius they never chose.
    /// Selection changes go through [`Brokkr::refresh_model_radius`] instead,
    /// which re-measures for the camera and the mirror plane and leaves the
    /// brush alone.
    fn rescale_radius(&mut self, previous_model_radius: f32) {
        if !self.dynamic_radius || previous_model_radius <= 0.0 || self.model_radius <= 0.0 {
            return;
        }
        let factor = self.model_radius / previous_model_radius;
        if (factor - 1.0).abs() > 1.0e-4 {
            self.brush.radius =
                (self.brush.radius * factor).clamp(MIN_RADIUS_MM, self.max_radius());
        }
    }

    /// The largest brush radius that is usable at the current voxel size.
    ///
    /// The lower of the millimetre ceiling and the voxel one. See
    /// [`MAX_RADIUS_VOXELS`] for why the second exists: at a resin voxel the
    /// millimetre ceiling alone reaches a brush that takes a quarter of a
    /// second per stamp.
    pub(crate) fn max_radius(&self) -> f32 {
        MAX_RADIUS_MM.min(MAX_RADIUS_VOXELS * self.doc.voxel_size()).max(MIN_RADIUS_MM)
    }

    /// The world space ray through a point in widget pixels.
    fn ray_through(&self, pixel: Vec2) -> (Vec3, Vec3) {
        let aspect = self.viewport_size.x / self.viewport_size.y.max(1.0);
        let ndc = OrbitCamera::ndc_from_pixels(pixel, self.viewport_size);
        self.camera.ray(ndc, aspect)
    }

    /// Where the cursor meets the ACTIVE body's surface, if it does.
    ///
    /// A live stroke keeps carving the body it started on however many others
    /// the cursor passes over, so this asks one body rather than picking. It
    /// goes through [`Document::pick_body`] rather than calling `raycast`
    /// directly, so that it shares the box gate and the box advance with the
    /// picker: two marches that disagree about where the surface is would put
    /// the ring in one place and the stroke in another.
    fn surface_under(&self, pixel: Vec2) -> Option<Vec3> {
        let (origin, ray) = self.ray_through(pixel);
        self.doc.pick_body(self.doc.active(), origin, ray, self.camera.far).map(|hit| hit.position)
    }

    /// Open the undo recorder on the body this stroke is about to write to, if
    /// a press has not already opened it.
    ///
    /// **This is called where the edit happens and not where the button goes
    /// down, and that ordering is the whole point.** Recording used to open in
    /// the `Pressed` arm before anything had been raycast, which with more than
    /// one body means opening it on the wrong one -- and `record_for_undo` does
    /// nothing at all when the volume it is called on has no recorder, so the
    /// carve would land, no history entry would be pushed, `unsaved` would stay
    /// false, and quitting would not even raise the discard prompt. One press,
    /// unrecoverable work.
    ///
    /// Lazy rather than moved into the press for one behaviour worth keeping: a
    /// press that starts off the model and drags onto it sculpts, and there is
    /// no body to open a recorder on until it arrives.
    fn arm_recorder(&mut self) {
        let volume = self.doc.active_volume_mut();
        if !volume.is_recording() {
            volume.begin_stroke();
        }
    }

    /// Apply the brush along the stroke path up to the point under the cursor.
    ///
    /// The stroke walks from its previous stamp to the new one at a fixed
    /// spacing, so a fast drag lays a continuous cut instead of a dotted trail.
    /// The stamps are applied one after another rather than batched, because
    /// each one has to see the field the previous one left behind.
    ///
    /// Move is the exception, and takes none of that path. It locked the field
    /// when the button went down and re-warps that same copy by the whole drag
    /// so far, so it wants the raw pointer position and one pass, not a trail
    /// of stamps that each build on the last. See `MoveStroke`.
    /// Index of a brush in the per-kind strength table.
    fn strength_slot(kind: BrushKind) -> usize {
        BrushKind::ALL.iter().position(|candidate| *candidate == kind).unwrap_or(0)
    }

    /// Where the pointer is, in the plane through `through` that faces the
    /// camera.
    ///
    /// This is what makes a grab feel like a grab. Raycasting the pointer onto
    /// the surface instead gives a target that crawls ALONG the form: drag
    /// sideways across a ball and the hit point slides around its curve, so the
    /// vector from the grab point stays short and keeps turning, and the result
    /// is a smear rather than a pull. It is also a feedback loop, because the
    /// surface being raycast is the one currently being deformed.
    ///
    /// Projecting into the view plane instead means the surface follows the
    /// cursor the way the hand expects, in the plane of the screen, and it
    /// keeps working when the cursor is dragged off the model entirely -- which
    /// is exactly when a grab is most useful.
    fn view_plane_point(&self, pixel: Vec2, through: Vec3) -> Vec3 {
        let (origin, ray) = self.ray_through(pixel);
        let Some(facing) = (self.camera.target - self.camera.eye()).try_normalize() else {
            return through;
        };
        let denominator = ray.dot(facing);
        // The ray is parallel to the plane, which cannot happen for a pointer
        // inside the viewport but is cheap to refuse.
        if denominator.abs() < 1.0e-6 {
            return through;
        }
        origin + ray * ((through - origin).dot(facing) / denominator)
    }

    /// One event of a Move gesture.
    ///
    /// Kept apart from `sculpt_to` because Move shares almost none of it: no
    /// stroke interpolation, no per-stamp normals, no trail of stamps building
    /// on each other. It locks the field once and re-warps that copy by the
    /// whole drag so far.
    fn move_to(&mut self, pixel: Vec2, start: bool) {
        let brush = self.effective_brush();
        let pressure = self.tablet.stamp_pressure(self.pressure_enabled, self.pressure_curve);
        let started = Instant::now();

        // The anchor is the one thing that needs the surface, and it is taken
        // once. After that the gesture is about the pointer.
        if start || !self.move_stroke.is_active() {
            let Some(point) = self.surface_under(pixel) else {
                return;
            };
            // Past the last early return, so the recorder opens on the body
            // that is about to be warped and not a moment sooner.
            self.arm_recorder();
            self.move_stroke.begin(
                self.doc.active_volume(),
                &brush,
                point,
                self.symmetry,
                MIRROR_CENTRE,
            );
            self.move_grab = Some(point);
        }
        let Some(grab) = self.move_grab else {
            return;
        };

        let target = self.view_plane_point(pixel, grab);
        self.move_stroke.drag_to(self.doc.active_volume_mut(), target, pressure);

        // Re-anchor once this lock has warped as far as it safely can, and let
        // the drag carry on from where the material now is.
        //
        // Without this a gesture stops dead at `Brush::max_drag`, which is a
        // quarter of the fold threshold and only a millimetre and a half for
        // the default 3 mm brush -- the surface simply refuses to follow the
        // cursor any further, which is exactly the "it does something, but not
        // what I expected" complaint. The cap itself cannot just be raised: past
        // the fold threshold a domain warp turns the field back through itself.
        // Chaining fold-safe warps gives unlimited reach without ever crossing
        // it. ZBrush can drag as far as it likes because it moves mesh vertices,
        // which stretch; a field has to be re-locked instead.
        //
        // Safe inside one stroke because `record_for_undo` captures a brick the
        // first time it is touched and not again, so a chain of locks is still
        // one undo entry.
        if self.move_stroke.is_at_the_limit() {
            let carried = grab + self.move_stroke.applied();
            self.move_stroke.begin(
                self.doc.active_volume(),
                &brush,
                carried,
                self.symmetry,
                MIRROR_CENTRE,
            );
            self.move_grab = Some(carried);
        }

        self.perf.stamps = 1;
        self.perf.pressure = pressure;
        self.perf.edit_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.remesh_dirty();
        self.hover = Some(grab);
        self.refresh_overlay();
    }

    fn sculpt_to(&mut self, pixel: Vec2, direction: BrushDirection, start: bool) {
        // Move is handled before anything else, because it wants the pointer
        // rather than the surface under the pointer, and it must keep working
        // once the cursor has been dragged off the model.
        if self.effective_brush().kind == BrushKind::Move {
            self.move_to(pixel, start);
            return;
        }

        let Some(point) = self.surface_under(pixel) else {
            // The cursor ran off the model. The stroke stays live so coming
            // back onto it continues rather than restarting, but nothing is
            // stamped in mid air.
            return;
        };

        // Past the early return, so the recorder opens on the body that is
        // about to be carved and only once there is something to carve. See
        // `arm_recorder`.
        self.arm_recorder();

        let started = Instant::now();
        self.stamp_centres.clear();
        if start {
            self.stroke.begin(point, &mut self.stamp_centres);
        } else {
            let spacing = self.effective_brush().spacing(self.doc.voxel_size());
            self.stroke.advance(point, spacing, &mut self.stamp_centres);
        }

        // Sampled once for the whole event rather than per stamp: the pen has
        // not moved between the stamps that one pointer event interpolates, so
        // re-reading it would only add jitter.
        let pressure = self.tablet.stamp_pressure(self.pressure_enabled, self.pressure_curve);
        let brush = self.effective_brush();

        {
            // A locked copy of the field is only good while nothing else writes
            // over it, and this is about to.
            self.move_stroke.end();
            self.move_grab = None;

            let lean = self.pen_lean();
            // Which way the drag is going, for the patterns that comb. Zero
            // until the stroke has moved far enough to have a direction, which
            // the pattern copes with by picking any direction across the
            // surface.
            let tangent = self.stroke.direction().unwrap_or(Vec3::ZERO);

            for index in 0..self.stamp_centres.len() {
                let centre = self.stamp_centres[index];
                // Take the normal from the field at each stamp rather than
                // reusing the one from the raycast, so a stroke curving around
                // a form stays oriented to the surface it is actually on.
                // Leaning the pen rotates the direction the brush pushes in,
                // which steers every brush at once because they all read this
                // normal.
                let normal = lean_normal(self.doc.active_volume().gradient_world(centre), lean);
                let stamp = Stamp::new(centre, normal, direction)
                    .with_pressure(pressure)
                    .with_tangent(tangent);
                brush.apply_symmetric(
                    self.doc.active_volume_mut(),
                    &stamp,
                    self.symmetry,
                    MIRROR_CENTRE,
                    &mut self.brush_scratch,
                );
            }
            self.perf.stamps = self.stamp_centres.len();
        }
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
    /// **It cuts every body the user can SEE**, which is [`Document::clip`]'s
    /// whole job: solo narrows the set, a hidden body the line passes over is
    /// left bit-identical, and the whole gesture is one undo entry.
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
    fn finish_cut(&mut self, from: Vec2) {
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

        // **The cut crosses every VISIBLE body**, which is what
        // `Document::clip` is for: solo narrows the set, a hidden body the line
        // passes over comes back bit-identical, and the whole gesture is ONE
        // undo entry of N `Change::Bricks`. Direct manipulation acts on what is
        // drawn.
        let visible = self.drawn_nodes();
        let outcome = self.doc.clip(plane, &visible);
        if outcome.bricks > 0 {
            let before = self.history.stats();
            if let Some(entry) = outcome.entry {
                self.history.push(entry);
            }
            self.unsaved = true;
            self.status = if outcome.bodies_cut > 1 {
                format!("cut {} bricks across {} bodies", outcome.bricks, outcome.bodies_cut)
            } else {
                format!("cut {} bricks", outcome.bricks)
            };
            // After the cut's own message on purpose: an eviction that took
            // a deleted body with it outranks a count of bricks.
            self.record_history(before);
            self.remesh_dirty();
            self.refresh_overlay();
        } else {
            // Nothing changed, so nothing is recorded -- an undo entry for a
            // no-op would be worse than none. The two ways to get here read
            // very differently to a user, so they are told apart: a line that
            // went nowhere near anything, and a line that really did pass over
            // a body and found only empty space inside its box.
            self.status = if outcome.bodies_crossed > 0 {
                format!(
                    "the cut crossed {} bodies and found nothing to remove",
                    outcome.bodies_crossed
                )
            } else {
                "the cut missed the model".to_string()
            };
        }
        // One cut per arming. A destructive tool that stays live is how a
        // stray click removes half the model.
        self.cut_armed = false;
    }

    fn finish_stroke(&mut self) {
        self.stroke.end();
        // Releases the locked field. One gesture is one lock, so a new drag
        // starts from the surface as it now stands rather than re-warping the
        // last one's copy.
        self.move_stroke.end();
        self.move_grab = None;
        let body = self.doc.active();
        if let Some(edit) = self.doc.active_volume_mut().end_stroke() {
            let before = self.history.stats();
            self.history.push(Entry::stroke(body, edit));
            self.record_history(before);
            // Inside the guard on purpose: a press and release that never
            // touched the model produces no edit, and must not raise a
            // "discard your work?" prompt on the way out.
            self.unsaved = true;
        }
    }

    /// Take the history's counters, and say out loud when an eviction has taken
    /// a deleted body with it.
    ///
    /// Entries are only ever dropped by a push, so this belongs at the push
    /// sites and nowhere else. The reason a body gets a status line when a
    /// dropped stroke gets only a counter in the stats readout: a dropped
    /// stroke costs the user a redo they did not ask for, and a dropped body
    /// costs them the body. Two folder deletes that each pass the reclaim
    /// allowance on their own and then evict each other are the case no
    /// per-operation prompt can catch, which is why the eviction has to be the
    /// one that speaks.
    fn record_history(&mut self, before: HistoryStats) {
        self.history_stats = self.history.stats();
        if self.history_stats.dropped_bodies > before.dropped_bodies {
            self.status =
                "undo history is full — a deleted body can no longer be brought back".to_string();
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
    /// Note the status line in the trail if it has changed since last frame.
    ///
    /// Watching one string beat a `set_status` helper through two dozen call
    /// sites: this cannot be forgotten by the next person to add one, and the
    /// per-frame cost is a string comparison that allocates only when something
    /// actually happened.
    ///
    /// The consequence to know about is that a status set and replaced within
    /// a single frame is not recorded. Nothing here does that -- a status is
    /// something a user reads -- and the alternative was a trail with holes in
    /// it nobody would notice.
    fn record_status_change(&mut self) {
        if self.status == self.crumbed_status {
            return;
        }
        self.crumbed_status.clone_from(&self.status);
        self.breadcrumbs.crumb(&self.status);
    }

    /// Record the things that are true of this session rather than the things
    /// that happened in it.
    ///
    /// Called once the renderer and the input devices have been looked at, so
    /// there is something to say. These are what make a report from a machine
    /// nobody here owns answerable: which graphics stack, which session type,
    /// whether the devices the application steers with were found at all.
    fn record_session_facts(&mut self) {
        self.breadcrumbs.sticky_fact(&format!("[wgpu] {}", self.shared.adapter_summary()));
        self.breadcrumbs.sticky_fact(&format!(
            "[session] {} on {}",
            std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".into()),
            std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_else(|_| "unknown".into()),
        ));
        self.breadcrumbs.sticky_fact(&format!("[tablet] {}", self.tablet.diagnosis().explain()));
        self.breadcrumbs
            .sticky_fact(&format!("[spacemouse] {}", self.spacemouse.diagnosis().explain()));
    }

    /// Build the report the dialog would send, or `None` when there is nothing
    /// to send.
    ///
    /// One function for the preview, the clipboard and the upload, so what is
    /// shown is what is sent. A description of the payload that is assembled
    /// separately from the payload is a description that goes stale.
    pub(crate) fn assemble_report(&self) -> Option<crate::report::Report> {
        let draft = self.bug_report.as_ref()?;
        let description = draft.description.text();
        if description.trim().is_empty() {
            return None;
        }
        let (diagnostics, trail) = if draft.with_detail {
            (self.diagnostics(), self.breadcrumbs.all())
        } else {
            (String::new(), Vec::new())
        };
        Some(crate::report::Report::new(
            &description,
            &diagnostics,
            &trail,
            None,
            &format!("{} ({})", env!("CARGO_PKG_VERSION"), build_commit()),
            &self.shared.adapter_summary(),
        ))
    }

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
            self.doc.voxel_size(),
            self.doc_stats.dense_bricks,
            self.doc_stats.uniform_bricks,
            self.doc_stats.resident_bytes as f64 / (1024.0 * 1024.0),
        );
        let pool = self.shared.stats();
        let _ = writeln!(
            out,
            "view: {} triangles, {} drawn / {} culled / {} hidden, {:.1} fps",
            pool.triangles,
            pool.drawn,
            pool.culled,
            pool.hidden,
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
        let trail = self.breadcrumbs.all();
        if !trail.is_empty() {
            let _ = writeln!(out, "\ntrail:");
            for crumb in crate::report::trim_breadcrumbs(&trail) {
                let _ = writeln!(out, "  {crumb}");
            }
        }
        out
    }

    /// Throw the sculpt away and start again from a fresh sphere.
    ///
    /// Extracted from the old `ResetSphere` arm so the menu and the panel button
    /// share one path rather than drifting.
    fn reset_sculpt(&mut self) {
        let mut volume = Volume::new(self.doc.voxel_size());
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM);
        volume.mark_everything_dirty();
        self.doc = Document::from_volume(volume);
        // History refers to bricks of the volume that just went away, so keeping
        // it would let undo splice pieces of the discarded model into this one.
        self.history.clear();
        self.history_stats = self.history.stats();
        self.camera = OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM);
        self.project_path = None;
        self.unsaved = false;
        self.status = String::new();
        self.publish_camera();
        self.rebuild_everything();
        self.refresh_detail_advice();
        self.refresh_overlay();
    }

    /// How long between crash-net writes.
    ///
    /// Two minutes is a compromise against the write itself: a large sculpt is
    /// tens of megabytes and the write happens on the thread that draws, so
    /// often enough to be worth having and rare enough that the hitch is not
    /// something you sculpt against.
    const AUTOSAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(120);

    /// How long the pointer must have been still before the crash net is
    /// written.
    ///
    /// The write happens on the thread that draws, and it is not small:
    /// measured at **1.55 GB/s**, so a 159 MB model takes 100 ms and the 671 MB
    /// autosave seen from a real scan session would take about 430 ms -- some
    /// twenty six dropped frames, every two minutes, in the middle of
    /// sculpting. That is exactly the kind of thing that gets reported as "it
    /// stutters sometimes" and is miserable to track down.
    ///
    /// Waiting for a pause is the cheap fix, and it works because sculpting is
    /// full of pauses: the hitch lands while the user is looking at the model
    /// rather than dragging across it, where it is invisible. It does not make
    /// the write faster, and if that ever matters the next step is encoding to
    /// a buffer on this thread and doing the file write from a `Task`, the way
    /// import already does.
    const AUTOSAVE_IDLE_GAP: std::time::Duration = std::time::Duration::from_secs(3);

    /// Where the crash net lives.
    ///
    /// `$XDG_STATE_HOME`, not the config directory: this is recoverable state
    /// rather than settings, and it must never sit next to the user's own
    /// files where it could be mistaken for one.
    pub(crate) fn default_autosave_path() -> Option<std::path::PathBuf> {
        crate::paths::state_file("autosave.brokkr")
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
            // And not while the pointer is still moving. See the constant.
            || self.last_activity.elapsed() < Self::AUTOSAVE_IDLE_GAP
        {
            return;
        }
        let started = Instant::now();
        self.write_autosave();
        let taken = started.elapsed().as_secs_f64() * 1000.0;
        if taken > 50.0 {
            // Visible in the log rather than silent, so the next person to
            // wonder where a hitch came from has the number in front of them.
            log::warn!("the autosave took {taken:.0} ms on the draw thread");
        }
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
                brokkr_core::project::write(&mut writer, &self.doc, &state)
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
        brokkr_core::ProjectState { view: self.current_view(), keys: self.timeline.keys.clone() }
    }

    /// Put the camera where a view says, and nothing else.
    ///
    /// The counterpart to `apply_view`, and the difference between them is the
    /// whole policy of the timeline: **going to a key restores everything it
    /// holds, and playing through the keys moves only the camera.** A
    /// fly-through that reached over and changed the brush, or switched a
    /// mirror plane on halfway, would be alarming rather than useful.
    fn fly_camera_to(&mut self, view: &brokkr_core::View) {
        self.camera.target = view.camera_target;
        self.camera.distance = view.camera_distance;
        self.camera.yaw = view.camera_yaw;
        self.camera.pitch = view.camera_pitch;
        self.camera.roll = view.camera_roll;
        self.publish_camera();
        self.refresh_overlay();
    }

    /// Where the camera is and what the brush is set to, right now.
    ///
    /// The same function serves saving a file and storing a timeline key, which
    /// is the point of `View` being one type: a key cannot come to restore less
    /// than a reopen does.
    fn current_view(&self) -> brokkr_core::View {
        brokkr_core::View {
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

    /// Put the camera, brush and mirror planes back the way a view records
    /// them, clamping every number to the range the interface offers.
    ///
    /// Clamped rather than trusted for the same reason `open_project` clamps:
    /// a view can come from a file, and the radius ceiling has already moved
    /// once. A saved 12 mm brush is fine; a saved 40 mm one from some future
    /// build must not put the slider somewhere it cannot be dragged back from.
    ///
    /// The mirror planes are put back and then **swept**, for the same reason
    /// choosing a body sweeps them: a file can name a mirror the body it was
    /// saved with does not straddle -- it was saved from a build without the
    /// refusal, or with a different body active -- and a restored mirror
    /// carves twins into empty space exactly as an enabled one does. The sweep
    /// costs one measurement of the field, at a file open or a key press,
    /// which is a user action; see
    /// [`Brokkr::refuse_mirrors_the_body_does_not_straddle`].
    fn apply_view(&mut self, view: &brokkr_core::View) {
        self.camera = OrbitCamera {
            target: view.camera_target,
            distance: view.camera_distance,
            yaw: view.camera_yaw,
            pitch: view.camera_pitch,
            roll: view.camera_roll,
            ..OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM)
        };
        self.brush.radius = view.brush_radius.clamp(MIN_RADIUS_MM, self.max_radius());
        self.brush.strength = view.brush_strength.clamp(MIN_STRENGTH, MAX_STRENGTH);
        self.symmetry = MirrorAxis::ALL
            .into_iter()
            .zip(view.mirror)
            .fold(Symmetry::OFF, |set, (axis, on)| set.with_axis(axis, on));
        self.refuse_mirrors_the_body_does_not_straddle("turned off");
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
        let (doc, state) = match brokkr_core::project::read(&mut reader) {
            Ok(loaded) => loaded,
            Err(error) => {
                // The message says what was expected as well as what was found,
                // because "a file from a different build" and "a partial
                // download" look identical without it.
                self.status = format!("could not read {}: {error}", path.display());
                return;
            }
        };

        self.doc = doc;
        // Before `apply_view`, so that a mirror plane the loaded body does not
        // straddle wins the status line: which file was opened is visible in
        // the title and the recent list, and a mirror plane going off is not
        // visible anywhere else. Same ordering, and same reason, as
        // `select_body`.
        self.status = format!("opened {}", path.display());
        self.apply_view(&state.view);
        self.timeline.adopt(state.keys);

        // History belongs to the model that just went away.
        self.history.clear();
        self.history_stats = self.history.stats();
        self.project_path = Some(path.to_path_buf());
        self.unsaved = false;
        self.recent.record(path);
        self.publish_camera();
        self.rebuild_everything();
        self.refresh_detail_advice();
        self.refresh_overlay();
    }

    /// Swap an imported model in, modelled on `open_project`.
    ///
    /// Three things here are individually silent when wrong.
    ///
    /// `project_path` is cleared, where `open_project` sets it. An import must
    /// not adopt the mesh's path, or the next plain Save would write a `.brokkr`
    /// container straight over the user's `.stl` with no dialog and no warning.
    /// This is the most damaging mistake available in this function.
    ///
    /// `mark_everything_dirty` is done by `voxelise` itself, exactly as
    /// `project::read` does it, which is why neither this nor `open_project`
    /// appears to need it.
    ///
    /// The camera is framed on `MODEL_RADIUS_MM` like every other call site,
    /// not on the model's own measured radius: the bound rounds out to whole
    /// bricks, so for a centred model it over-reports by up to about 1.7 times
    /// plus padding and the camera sits visibly too far back.
    fn adopt_import(&mut self, imported: crate::message::Imported) {
        self.doc = Document::from_volume(imported.volume);
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
        // Nothing in an STL, OBJ or 3MF says which axis is up, and the two
        // conventions in the world disagree -- so a mesh that arrived lying
        // down is a normal outcome rather than a corrupt file. Ask; do not
        // turn it silently, because the guess can be wrong and a model turned
        // without being asked about is one the user cannot account for.
        //
        // Filtered against `Up`, so the common case of a print-ready file that
        // was already standing raises nothing.
        self.orient_prompt = imported.resting_up.filter(|up| *up != brokkr_core::Facing::Up);
        self.publish_camera();
        self.rebuild_everything();
        self.refresh_detail_advice();
        self.refresh_overlay();
    }

    /// Save the document, through a temporary file and a rename.
    ///
    /// **Two things here used to be wrong, and both got worse with every body
    /// added.**
    ///
    /// It opened the user's real file with `File::create`, which truncates it
    /// to zero before a byte of the new document is written. The crash net
    /// already had a temp-and-rename and the document did not, so the *net* was
    /// better protected than the thing it protects -- and at the measured
    /// 1.55 GB/s a three gigabyte document is about two seconds of the user's
    /// file existing as a truncated stub. A failed save now leaves the previous
    /// file exactly as it was.
    ///
    /// And it cleared `unsaved` and deleted the crash net **on the writer's
    /// word**. So the file is read back before either happens: its header and
    /// node table only, a few hundred bytes and no geometry, compared against
    /// the document that was meant to go into it. On a mismatch the temporary
    /// is thrown away, the previous file is untouched, `unsaved` stays set, the
    /// autosave survives, and the status says "could not" -- which is what
    /// `panel.rs` colours red.
    fn save_project(&mut self, path: &std::path::Path) {
        let state = self.project_state();

        // Beside the target rather than in a temp directory, so the rename is
        // within one filesystem and therefore atomic.
        let temporary = path.with_extension("brokkr.tmp");
        let file = match std::fs::File::create(&temporary) {
            Ok(file) => file,
            Err(error) => {
                self.status = format!("could not write {}: {error}", path.display());
                return;
            }
        };
        let mut writer = std::io::BufWriter::new(file);
        if let Err(error) = brokkr_core::project::write(&mut writer, &self.doc, &state) {
            self.status = format!("could not write {}: {error}", path.display());
            std::fs::remove_file(&temporary).ok();
            return;
        }
        // `project::write` ends with its own flush and reports a failure there
        // as an error, so nothing is left in the buffer by the time this
        // returns. The drop is to CLOSE the file before it is reopened for the
        // check below.
        drop(writer);

        if let Err(problem) = self.verify_written(&temporary) {
            self.status = format!("could not write {}: {problem}", path.display());
            std::fs::remove_file(&temporary).ok();
            return;
        }

        if let Err(error) = std::fs::rename(&temporary, path) {
            self.status = format!("could not replace {}: {error}", path.display());
            std::fs::remove_file(&temporary).ok();
            return;
        }

        self.project_path = Some(path.to_path_buf());
        // Only here. Every failure arm above leaves the flag set, or a failed
        // write would let the next quit go through silently.
        self.unsaved = false;
        self.recent.record(path);
        // The work is safely in a file the user chose, so the crash net has
        // nothing left to protect and a stale one would only offer to restore
        // something older than what is on disk.
        self.clear_autosave();
        self.status = format!("saved {}", path.display());
    }

    /// Read back what was just written, far enough to know it is this document.
    ///
    /// The header and the node table, which is a few hundred bytes whatever the
    /// sculpt weighs -- the geometry is deliberately not re-read, because doing
    /// that on the draw thread would double the cost of every save.
    ///
    /// What it catches is the class the old code could not: a write that
    /// reported success and produced a file this build will not open, or opens
    /// as a different document. That is not hypothetical -- the format grew a
    /// node table this release, and the first thing to exercise it in the wild
    /// is the autosave, unwatched, every two minutes.
    ///
    /// The check itself is pinned by
    /// `the_save_verification_notices_a_file_that_is_not_this_document`, and
    /// the *gate* in `save_project` that calls it -- a separate thing, and the
    /// one that can be refactored away without a test noticing -- by
    /// `a_save_that_cannot_read_its_own_output_back_keeps_the_previous_file`.
    /// If you change the shape of the failure here, that second test is the one
    /// to look at: it drives this through a temporary that reads back empty.
    fn verify_written(&self, path: &std::path::Path) -> Result<(), String> {
        let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
        let mut reader = std::io::BufReader::new(file);
        let outline =
            brokkr_core::project::read_outline(&mut reader).map_err(|error| error.to_string())?;
        if outline.nodes != self.doc.node_count() || outline.bodies != self.doc.body_count() {
            return Err(format!(
                "the file came back with {} rows and {} bodies, and the document has {} and {}",
                outline.nodes,
                outline.bodies,
                self.doc.node_count(),
                self.doc.body_count()
            ));
        }
        if outline.voxel_size != self.doc.voxel_size() {
            return Err(format!(
                "the file came back at a {} mm voxel, and the document is at {} mm",
                outline.voxel_size,
                self.doc.voxel_size()
            ));
        }
        Ok(())
    }

    fn export_directory() -> std::path::PathBuf {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        home.unwrap_or_else(std::env::temp_dir).join("brokkrsculpt")
    }

    /// Weld every visible body and write them all out to a chosen path.
    ///
    /// Refuses to write anything that would not print. A slicer given a mesh
    /// with holes either rejects it or fills it wrong, and finding that out
    /// after a failed print is worse than being told here.
    ///
    /// **The refusal happens before `File::create`, and that ordering is the
    /// point.** `File::create` truncates, so a verdict taken after it would
    /// leave the user with a zero-byte file where their last good export was.
    /// Everything is welded and validated into memory, the verdict is taken,
    /// and only then is a file opened.
    ///
    /// **The status line names how many bodies were omitted, unconditionally
    /// -- including when the answer is none.** The eye is one bit in a
    /// forty-byte node record with no integrity check of any kind, and it is
    /// the bit that decides whether a part reaches the printer: a single
    /// flipped bit is a perfectly legal value that loads without error. The
    /// brick stream is defended by a checked distance decode and the node table
    /// is defended by nothing, so this line is the whole of that defence and it
    /// costs one format string. Making it conditional -- "only say it when
    /// something was hidden" -- would mean silence is ambiguous between
    /// "nothing was hidden" and "the count was never computed", which is
    /// exactly the reading it exists to prevent.
    ///
    /// `saved_visibility` and never `display_visibility`: a view mode must not
    /// decide what reaches a print.
    fn export(&mut self, format: ExportFormat, path: &std::path::Path) {
        let started = Instant::now();
        let mut visible = Vec::new();
        self.doc.saved_visibility(&mut visible);
        let bodies = self.doc.export_bodies(&visible);
        let omitted = self.doc.body_count().saturating_sub(bodies.len());

        // Per body, never over the union: a document whose second body is
        // empty sums to a printable report while half the print is missing.
        if let Err(why) = brokkr_core::export::document_verdict(&bodies) {
            self.status = format!("not exported, {why}");
            log::error!("refusing to export: {why}");
            return;
        }

        let summary =
            brokkr_core::MeshReport::summed(bodies.iter().map(|(_, _, report)| *report)).summary();
        let parts: Vec<(&str, &brokkr_core::ExportMesh)> =
            bodies.iter().map(|(meta, mesh, _)| (meta.name.as_str(), mesh)).collect();

        let write = || -> std::io::Result<()> {
            let file = std::fs::File::create(path)?;
            let mut file = std::io::BufWriter::new(file);
            match format {
                ExportFormat::Stl => brokkr_core::export::stl::write_all(&parts, &mut file)?,
                ExportFormat::Obj => brokkr_core::export::obj::write_all(&parts, &mut file)?,
                ExportFormat::ThreeMf => {
                    brokkr_core::export::threemf::write_all(&parts, &mut file)?
                }
            }
            std::io::Write::flush(&mut file)
        };

        match write() {
            Ok(()) => {
                let bytes = std::fs::metadata(path).map(|data| data.len()).unwrap_or(0);
                // STL has nowhere to put an object boundary, so several bodies
                // arrive in a slicer as one part. Said here rather than
                // discovered after slicing.
                let fused = matches!(format, ExportFormat::Stl) && parts.len() > 1;
                self.status = format!(
                    "exported {} of {} bodies; {omitted} hidden{} -- {summary} to {} \
                     ({:.1} MB, {:.0} ms)",
                    parts.len(),
                    self.doc.body_count(),
                    if fused { " (STL fuses them into one part)" } else { "" },
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

    /// Export the sculpt to a staging file and open it in OrcaSlicer.
    ///
    /// A staging path under the state directory rather than a dialog: this is a
    /// handoff and not a save, the file is the slicer's input rather than the
    /// user's document, and asking where to put something they will not look
    /// for again is a question with no useful answer.
    ///
    /// 3MF rather than STL, because it carries units. An STL does not, so a
    /// slicer has to guess millimetres, and every slicer guessing right is a
    /// convention rather than a guarantee.
    ///
    /// Runs on the update thread, like `export` already does. The export itself
    /// is the slow half and it is the same work the Export buttons do.
    fn hand_to_slicer(&mut self) {
        let Some(slicer) = crate::slicer::find() else {
            self.status = "could not find OrcaSlicer — install it and try again".to_string();
            return;
        };
        let Some(staged) = crate::paths::state_file("handoff.3mf") else {
            self.status = "could not work out where to put the file".to_string();
            return;
        };
        if let Some(parent) = staged.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            self.status = format!("could not make {}: {error}", parent.display());
            return;
        }

        self.export(ExportFormat::ThreeMf, &staged);
        // `export` refuses a model that would not print, and says so in the
        // status. Handing the slicer a file that was never written would
        // replace that message with a less useful one.
        if !staged.is_file() || self.status.contains("not exported") {
            return;
        }

        match crate::slicer::open(&slicer, &staged) {
            Ok(()) => {
                self.status = format!("opened in {}", slicer.display());
            }
            Err(why) => self.status = format!("could not open the slicer: {why}"),
        }
    }

    /// The voxel size a request actually lands on.
    fn clamped_voxel_size(requested: f32) -> f32 {
        requested.clamp(FINEST_VOXEL_MM, COARSEST_VOXEL_MM)
    }

    /// Rebuild the volume at a different voxel size.
    fn resample(&mut self, voxel_size: f32) {
        let mut voxel_size = Self::clamped_voxel_size(voxel_size);
        if (voxel_size - self.doc.voxel_size()).abs() < 1.0e-6 {
            return;
        }

        // A step the pool cannot hold LANDS ON THE FINEST THAT FITS instead
        // of refusing. The refusal named that size and then made the user go
        // there by hand -- which they could not: the button only halves, so
        // the message pointed at a rung the interface had no way to reach.
        // Observed live at 0.062 mm with 0.043 mm reachable and no path to it.
        let mut capped = None;
        if let Some((why, finest)) = self.too_fine_for_the_pool(voxel_size) {
            let fallback = Self::clamped_voxel_size(finest);
            if fallback < self.doc.voxel_size() * 0.98 {
                capped = Some(voxel_size);
                voxel_size = fallback;
            } else {
                // Already at the fit limit: there is nowhere finer to land.
                self.status = why;
                log::warn!("{}", self.status);
                return;
            }
        }

        let started = Instant::now();
        // Every body at once, because they share the lattice: the old bricks
        // have to be cleared out of the renderer's pool, and after a resample
        // their coordinates mean something different, so they are marked in the
        // new volume to remesh to nothing.
        self.doc.resample(voxel_size);
        // Past the early return above, so resampling to the size already in use
        // stays the no-op its test asserts it is.
        self.unsaved = true;
        // History refers to bricks of a volume that no longer exists, and at a
        // different resolution, so keeping it would splice nonsense back in.
        self.history.clear();
        self.history_stats = self.history.stats();
        self.rebuild_everything();
        self.refresh_detail_advice();

        self.status = match capped {
            Some(wanted) => format!(
                "resampled to {voxel_size:.3} mm -- the finest the mesh pool holds at this size \
                 ({wanted:.3} mm did not fit), {} dense bricks, {:.0} MB, {:.0} ms",
                self.doc_stats.dense_bricks,
                self.doc_stats.resident_bytes as f64 / (1024.0 * 1024.0),
                started.elapsed().as_secs_f64() * 1000.0
            ),
            None => format!(
                "resampled to {voxel_size:.3} mm, {} dense bricks, {:.0} MB, {:.0} ms",
                self.doc_stats.dense_bricks,
                self.doc_stats.resident_bytes as f64 / (1024.0 * 1024.0),
                started.elapsed().as_secs_f64() * 1000.0
            ),
        };
        log::info!("{}", self.status);
    }

    /// Why a resample to `wanted` would not fit the mesh pool, if it would not.
    ///
    /// This is the guard the import path has had all along and this one never
    /// did, and its absence was not theoretical: on the Nightwing dragon the
    /// "finer" button resampled correctly in 296 ms and then reserved 8,525,824
    /// vertices against a pool of 8,000,000. The volume was right, the model on
    /// screen lost parts of itself, and the only thing that said so was a line
    /// on stderr. From the user's chair the button did nothing.
    ///
    /// The prediction is a square law over what is ACTUALLY on the GPU right
    /// now, not an estimate from surface area. A surface has a fixed area, so
    /// halving the voxel size quadruples the vertices on it -- and starting
    /// from a measured reservation means the allocator's own padding is already
    /// baked into the figure rather than guessed at.
    fn too_fine_for_the_pool(&self, wanted: f32) -> Option<(String, f32)> {
        if wanted >= self.doc.voxel_size() {
            return None;
        }

        // System memory first, because the GPU pool is no longer the tighter
        // of the two. Measured on the dragon: 0.0565 mm fits the pool at 48%
        // and costs 4.15 GB of RAM, so a pool-only guard would happily walk a
        // machine into swap or the OOM killer. Same square law -- a surface
        // has fixed area, so halving the voxel quadruples the shell.
        let growth = (self.doc.voxel_size() / wanted).powi(2) as f64;
        let bytes = self.doc_stats.resident_bytes as f64 * growth;
        if bytes > MAX_VOLUME_BYTES {
            let headroom = MAX_VOLUME_BYTES / self.doc_stats.resident_bytes.max(1) as f64;
            let finest = self.doc.voxel_size() / headroom.sqrt() as f32 * 1.03;
            return Some((
                format!(
                    "could not resample to {wanted:.3} mm: it needs about {:.1} GB of memory \
                     against a {:.0} GB ceiling -- {finest:.3} mm is the finest that fits",
                    bytes / (1024.0 * 1024.0 * 1024.0),
                    MAX_VOLUME_BYTES / (1024.0 * 1024.0 * 1024.0),
                ),
                finest,
            ));
        }

        let pool = self.shared.stats();
        if pool.vertices_reserved == 0 {
            return None;
        }
        // Predicted from what is LIVE, because a resample resets the pool and
        // so starts from a clean bump pointer -- the watermark before the
        // reset says nothing about the space after it. Scaling `live` is
        // therefore the honest basis, and it is only honest BECAUSE of the
        // reset; before that existed this prediction was blind to
        // fragmentation and let the pool overflow.
        let vertices = pool.vertices_reserved as f64 * growth;
        let indices = pool.indices_reserved as f64 * growth;
        if vertices <= pool.vertex_capacity as f64 && indices <= pool.index_capacity as f64 {
            return None;
        }

        // The finest size that would fit, from the same square law, with three
        // percent on top because a prediction that lands at exactly 100% of a
        // ceiling helps nobody. Returned so `resample` can GO there rather
        // than tell the user to.
        let headroom = (pool.vertex_capacity as f64 / pool.vertices_reserved as f64)
            .min(pool.index_capacity as f64 / pool.indices_reserved as f64);
        let finest = self.doc.voxel_size() / headroom.sqrt() as f32 * 1.03;
        let why = format!(
            "could not resample to {wanted:.3} mm: it needs about {:.1}M vertices against a pool \
             of {:.1}M -- {finest:.3} mm is the finest that fits at this size",
            vertices / 1.0e6,
            pool.vertex_capacity as f64 / 1.0e6,
        );
        Some((why, finest))
    }

    /// Scale the model so its longest dimension is `longest_mm`.
    ///
    /// Free and lossless: distances are stored in voxels, so this is a change
    /// to `voxel_size` and nothing else, and the bricks are bit-identical
    /// afterwards. It buys **no** detail -- the model has the same number of
    /// voxels across it as before. What it changes is what one voxel measures,
    /// which is what decides whether the detail already there is enough for a
    /// given printer. [`Brokkr::detail_advice`] is what tells the user that.
    fn set_working_size(&mut self, longest_mm: f32) {
        if !(longest_mm.is_finite() && longest_mm > 0.0) {
            self.status = "could not resize: that is not a length".into();
            return;
        }
        let Some((lo, hi)) = self.doc.active_volume().surface_bounds() else {
            self.status = "could not resize: there is no model".into();
            return;
        };
        let current = (hi - lo).max_element();
        if current <= 0.0 {
            self.status = "could not resize: the model has no size".into();
            return;
        }

        let factor = longest_mm / current;
        // The voxel size travels with the model, so the range that bounds it
        // bounds this too -- otherwise a resize could land somewhere the finer
        // and coarser buttons can never get back from.
        let wanted_voxel = self.doc.voxel_size() * factor;
        let clamped = Self::clamped_voxel_size(wanted_voxel);
        if (clamped - wanted_voxel).abs() > 1.0e-9 {
            self.status = format!(
                "could not resize to {longest_mm:.1} mm: that would put the voxel at {:.4} mm, \
                 outside the {FINEST_VOXEL_MM:.2}-{COARSEST_VOXEL_MM:.2} mm range",
                wanted_voxel,
            );
            return;
        }

        let previous_radius = self.model_radius;
        // Every body, because they share the lattice: scaling one alone would
        // hand it a lattice its siblings do not have.
        self.doc.rescale(factor);
        // The brush is in millimetres, so without this a resize leaves it the
        // wrong size relative to the model it is about to be used on.
        self.brush.radius = (self.brush.radius * factor).clamp(MIN_RADIUS_MM, self.max_radius());
        self.unsaved = true;
        // Every brick's world position moved, so everything has to be redrawn
        // even though not one voxel changed.
        self.doc.mark_everything_dirty();
        self.rebuild_everything();
        let _ = previous_radius;
        self.camera = OrbitCamera::framing(Vec3::ZERO, self.model_radius.max(1.0e-3));
        self.publish_camera();
        self.refresh_overlay();
        self.refresh_detail_advice();
        self.status = format!("working size {longest_mm:.1} mm, {}", self.detail_advice);
        log::info!("{}", self.status);
    }

    /// Recompute the cached readout. See the field for why it is cached.
    fn refresh_detail_advice(&mut self) {
        self.detail_advice = self.measure_detail_advice();
    }

    /// What the current resolution actually means for a print, in one line.
    ///
    /// Detail is decided by how many voxels lie across the model, which does
    /// not change when it is scaled. What a printer cares about is what one
    /// voxel MEASURES. Saying both, side by side, is the whole point: it turns
    /// "is 0.25 mm fine enough?" from a guess into arithmetic.
    ///
    /// The resin figure is the comparison worth drawing because it is the
    /// demanding one -- a consumer resin printer resolves around 0.03 mm in XY,
    /// where a filament nozzle lays down 0.4 mm lines and cannot use anything
    /// like this much.
    fn measure_detail_advice(&self) -> String {
        let Some((lo, hi)) = self.doc.active_volume().surface_bounds() else {
            return "no model".into();
        };
        let voxel_size = self.doc.voxel_size();
        let longest = (hi - lo).max_element();
        let across = (longest / voxel_size).round() as i64;
        let verdict = if voxel_size <= RESIN_XY_MM {
            "at or below what a resin printer resolves"
        } else if voxel_size <= FDM_LINE_MM {
            "finer than a filament nozzle, coarser than resin"
        } else {
            "coarse: a filament nozzle would not see the difference"
        };
        format!(
            "{longest:.1} mm across, voxel {voxel_size:.3} mm, {across} voxels wide -- {verdict}"
        )
    }

    /// Turn the whole model, so that a face the user picked becomes the
    /// direction they asked for.
    ///
    /// Exact, unlike [`Brokkr::resample`]: a quarter turn maps voxels onto
    /// voxels, so nothing is resampled and turning back returns the same bits.
    /// That is also why there is no undo entry -- see below.
    ///
    /// **Every body turns, and turning only the active one would be wrong.**
    /// Bodies share one lattice and have no transform, so where a body sits IS
    /// its brick occupancy and the arrangement of the bodies is the only
    /// positional state the document has. Turning one of them would scatter
    /// that arrangement with nothing to put it back, and it would make this
    /// function's own status line -- turn it back the same way to undo -- a
    /// lie, because the bodies that did not turn would come back somewhere new.
    fn orient(&mut self, rotation: brokkr_core::AxisRotation) {
        if rotation.is_identity() {
            return;
        }

        let started = Instant::now();
        self.doc.rotate(rotation);
        self.unsaved = true;
        // Every entry names a brick of a volume that no longer exists, exactly
        // as after a resample. Snapshotting the whole field instead would
        // exceed the history budget on any real model, and it is not needed:
        // turning back is the undo, and it is exact.
        self.history.clear();
        self.history_stats = self.history.stats();
        self.rebuild_everything();
        self.refresh_overlay();

        self.status = format!(
            "turned the model, {:.0} ms -- turn it back the same way to undo",
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
                    // Through the same gate the strip uses, and not a bare
                    // `toggled`: see `toggle_mirror`.
                    self.toggle_mirror(MirrorAxis::X);
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
                    self.brush.radius = value.clamp(MIN_RADIUS_MM, self.max_radius());
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
                self.brush.radius =
                    (sizing.original * factor).clamp(MIN_RADIUS_MM, self.max_radius());
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
        let shown = self.saved_nodes();
        match self.history.undo(&mut self.doc, &shown) {
            UndoOutcome::Applied(_) => self.after_history_step(),
            UndoOutcome::Refused(node) => self.refuse_history_step("undo", node),
            UndoOutcome::Nothing => {}
        }
    }

    fn redo(&mut self) {
        let shown = self.saved_nodes();
        match self.history.redo(&mut self.doc, &shown) {
            UndoOutcome::Applied(_) => self.after_history_step(),
            UndoOutcome::Refused(node) => self.refuse_history_step("redo", node),
            UndoOutcome::Nothing => {}
        }
    }

    /// Which rows the FILE would keep, indexed by node position.
    ///
    /// Undo and redo are the two callers, and they use the saved answer rather
    /// than the drawn one on purpose. **Solo must never veto a structural
    /// operation.** The refusal these feed -- "undo would change X, which is
    /// hidden" -- is about an eye the user set and can see in the panel; a
    /// transient view mode turning ctrl+Z off, with a message calling a body
    /// "hidden" whose eye is plainly open, is a different thing wearing the
    /// same words.
    ///
    /// This is [`Document::saved_visibility`], and it is the allocating wrapper
    /// the keystroke paths want: both are gestures rather than frames. The
    /// resolver fills a buffer the application keeps, which is what increment
    /// 6's GPU mask needs and what this must not become.
    fn saved_nodes(&self) -> Vec<bool> {
        let mut shown = Vec::new();
        self.doc.saved_visibility(&mut shown);
        shown
    }

    /// Which rows are DRAWN, indexed by node position, solo included.
    ///
    /// The plane cut is the caller, and direct manipulation acts on what is
    /// drawn: a cut is a line the user draws across what they can see, so solo
    /// narrows it. Nothing on screen distinguishes an eye-hidden body from a
    /// solo-hidden one -- in both cases it is simply not there -- so a line
    /// drawn across one body cutting five would have no explanation anywhere.
    ///
    /// Allocating, like [`Brokkr::saved_nodes`] and for the same reason: a
    /// gesture, never a frame. The frame-rate answer is
    /// [`Brokkr::publish_visibility`]'s kept buffer.
    fn drawn_nodes(&self) -> Vec<bool> {
        let mut shown = Vec::new();
        self.doc.display_visibility(self.solo, &mut shown);
        shown
    }

    /// Work out what is drawn and tell the renderer, wholesale.
    ///
    /// **This is the second caller of the one visibility rule**, and the reason
    /// the rule is one function: the eye in the panel, the pick gate, the plane
    /// cut and this all have to agree, and they agree by asking rather than by
    /// each remembering. The answer is recomputed in full every time and never
    /// patched -- an incremental version would be a second owner of it, and two
    /// owners is how the eye and the viewport come to disagree after an undo,
    /// leaving a body invisible on screen that still raycasts and still carves.
    ///
    /// Called from [`Brokkr::update`] after every message rather than from the
    /// handful of places that change an eye, because "the handful of places" is
    /// a list that goes out of date silently. It runs on the frame tick too,
    /// so it allocates nothing: both buffers are kept and refilled.
    ///
    /// **Bodies only.** Folder rows have no geometry in the pool, and passing
    /// one to the renderer would be a name it can never match.
    fn publish_visibility(&mut self) {
        self.forget_a_vanished_solo();
        self.doc.display_visibility(self.solo, &mut self.shown);
        self.leave_a_solo_that_shows_nothing();
        self.hidden_bodies.clear();
        for (node, shown) in self.doc.nodes().iter().zip(&self.shown) {
            if !shown && node.is_body() {
                self.hidden_bodies.push(node.id);
            }
        }
        self.shared.set_hidden(&self.hidden_bodies);
    }

    /// Drop solo the moment it stops showing anything, and say so.
    ///
    /// **"Solo always shows something" is an invariant here rather than a
    /// courtesy in [`Brokkr::enter_solo`]**, because entry is not the only way
    /// into the empty state and the other ways are one keystroke each:
    ///
    /// * ctrl+Z. Turning the soloed row's eye on is an ordinary undoable change
    ///   and undo is deliberately not vetoed by solo (see
    ///   [`Brokkr::saved_nodes`]), so the undo straight after soloing a hidden
    ///   row turns that eye back off underneath the mode.
    /// * The soloed row's own eye. A subtree contains its own root, so
    ///   [`Brokkr::in_solo_scope`] passes that click -- correctly, it is the one
    ///   eye in the panel whose state the user can actually see the effect of.
    ///
    /// Both leave every row resolving to false: a black viewport, a badge
    /// naming a row nobody can see, and a status line about something else
    /// entirely. Escape recovers it and nothing on screen points at Escape.
    /// Catching it here costs one fold over a buffer that was just computed
    /// anyway, and it catches the third route nobody has thought of yet -- the
    /// same argument [`Brokkr::forget_a_vanished_solo`] makes one function
    /// down.
    ///
    /// **It replaces whatever status the gesture wrote.** The gesture's own
    /// note is the smaller news: the viewport just changed wholesale, and a
    /// message about which body got selected does not explain that. It refills
    /// `shown` rather than leaving the caller a mask it is about to contradict.
    ///
    /// Bodies only, matching [`Brokkr::publish_visibility`]: a folder resolving
    /// to true draws nothing, so counting folders would call an empty screen
    /// full.
    fn leave_a_solo_that_shows_nothing(&mut self) {
        let Some(id) = self.solo else {
            return;
        };
        let showing =
            self.doc.nodes().iter().zip(&self.shown).any(|(node, shown)| *shown && node.is_body());
        if showing {
            return;
        }
        let name =
            self.doc.node(id).map_or_else(|| "that row".to_string(), |node| node.name.clone());
        self.solo = None;
        self.doc.display_visibility(None, &mut self.shown);
        self.status = format!("left solo — {name} is hidden, so it was showing nothing");
    }

    /// Drop solo the moment the row it names stops existing.
    ///
    /// A dangling `Some(id)` is worse than either state it sits between:
    /// [`brokkr_core::resolve_visibility`] finds no node to scope to and shows
    /// **nothing at all**, while the mode is still on and its only exit -- the
    /// indicator naming the row -- has no name to draw.
    ///
    /// **Here rather than in the delete path, and that is deliberate.** A row
    /// stops existing on a delete, on a folder dissolving, on the undo of a
    /// group, on the redo of a delete, and on a merge and a split when those
    /// land. "The places that remove a row" is a list that goes out of date
    /// silently, which is the argument [`Brokkr::update`] already makes about
    /// the visibility pass itself -- so this rides on the same pass, one
    /// `index_of` over at most [`brokkr_core::MAX_NODES`] rows, allocating
    /// nothing.
    ///
    /// A whole-document swap is a different problem and has a different answer:
    /// the incoming document numbers its rows from 1 again, so a stale id names
    /// a real and perfectly innocent body rather than nothing. See
    /// [`Brokkr::rebuild_everything`], which clears solo outright.
    fn forget_a_vanished_solo(&mut self) {
        if let Some(id) = self.solo
            && self.doc.index_of(id).is_none()
        {
            self.solo = None;
        }
    }

    /// What undo and redo both do once something has actually moved.
    ///
    /// `refresh_overlay` is in here because the brush ring and the mirror
    /// planes are built from the field: sixteen other sites already refresh
    /// them, and undo not doing so has been a staleness bug the whole time
    /// there has been one body to see it on.
    fn after_history_step(&mut self) {
        self.history_stats = self.history.stats();
        self.unsaved = true;
        self.remesh_dirty();
        self.refresh_overlay();
    }

    /// Say which body is in the way, and change nothing else.
    ///
    /// **Deliberately does not reveal or select it.** The eye bit is persisted,
    /// undoable document state, so writing it from inside undo would destroy a
    /// deliberate hide, set `unsaved` for a change the user never made, and
    /// still not reveal the body if an ancestor folder were the one hiding it.
    /// The user reveals it and presses again; the entry is untouched and costs
    /// nothing while it waits.
    fn refuse_history_step(&mut self, verb: &str, node: NodeId) {
        let name = self.doc.node(node).map_or("a hidden body", |node| node.name.as_str());
        self.status = format!("{verb} would change {name}, which is hidden");
        self.refresh_overlay();
    }

    /// Whether a modal card is up, and therefore owns the input.
    ///
    /// One list, read by both halves of the guard: `on_key` for the keyboard
    /// and `on_pointer` for the pointer. Before this existed the pointer had
    /// its own inline pair of checks and the keyboard had none at all, so
    /// `ctrl+Z` under the unsaved-work prompt changed the very volume the
    /// prompt was asking about, `x` flipped a mirror plane behind the card,
    /// and `1`-`6` swapped the brush. The bug-report dialog was in neither
    /// list, so a press beside its card sculpted.
    ///
    /// # Adding to this list is not automatically right
    ///
    /// It answers "is the document unreachable right now", and every card here
    /// is a question the user must answer before anything else happens. **A
    /// modeless overlay that takes the pointer — the split preview, whose
    /// whole gesture is dragging a plane across the model — belongs in the
    /// keyboard guard and NOT in the pointer one.** Split this function in two
    /// at that point rather than widening it and quietly killing the drag.
    fn modal_open(&self) -> bool {
        self.confirm.is_some()
            || self.orient_prompt.is_some()
            || self.bug_report.is_some()
            || self.pending_delete.is_some()
            || self.pending_merge.is_some()
    }

    /// What a key press means, now that nothing in the widget tree wanted it.
    ///
    /// The decode itself is `viewport::shortcut`, which is a pure function of
    /// the key and its modifiers so that it can be tested without a window —
    /// this is the half that needs the application, and it exists so that
    /// there is exactly one place where a key turns into a change.
    fn on_key(
        &mut self,
        key: iced::keyboard::Key,
        modifiers: iced::keyboard::Modifiers,
    ) -> Task<Message> {
        match key {
            // Escape is the one key a modal must still see, so it sits above
            // the guard: it is how the unsaved-work prompt is cancelled and
            // the orientation prompt declined. It does NOT dismiss the bug
            // report, and that is on purpose -- the card holds a description
            // the user typed, and throwing it away on a stray Escape is the
            // same class of loss the confirm prompt exists to prevent.
            iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape) => {
                self.update(Message::MenuClosed)
            }
            iced::keyboard::Key::Character(character) => {
                // THE keyboard modal guard. Everything a shortcut can reach --
                // undo, the mirror planes, the brush numbers -- edits the
                // document or the tool behind a card that is asking about it.
                if self.modal_open() {
                    return Task::none();
                }
                let Some(message) = crate::viewport::shortcut(
                    character.as_str(),
                    modifiers.command(),
                    modifiers.shift(),
                    modifiers.alt(),
                ) else {
                    return Task::none();
                };
                // One level of recursion and never more: `shortcut` returns
                // ordinary messages and cannot return another `KeyPressed`.
                self.update(message)
            }
            iced::keyboard::Key::Named(_) | iced::keyboard::Key::Unidentified => Task::none(),
        }
    }

    fn on_pointer(&mut self, event: PointerEvent) {
        // While a modal prompt is up, the pointer belongs to it. The scrim in
        // `view` now genuinely swallows presses -- it is wrapped in
        // `iced::widget::opaque`, which captures them, where the bare styled
        // container it used to be forwarded every one -- and this is the other
        // half of that guarantee, because the viewport is a shader widget that
        // sees events wherever the cursor is. Both halves or neither: without
        // this a press behind the card would sculpt into a document the user
        // is about to discard, into a model they are about to turn, or into
        // the very state a bug report is describing.
        if self.modal_open() {
            return;
        }
        self.last_activity = Instant::now();
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
                if self.menu.take().is_some()
                    || self.top_menu.take().is_some()
                    || self.cube_menu.take().is_some()
                {
                    return;
                }

                // A press on the navigation cube belongs to the cube. Checked
                // before anything else, or clicking it would also carve a divot
                // out of the model behind it.
                //
                // Both buttons are taken, not just the left one. A right press
                // that fell through would start an orbit and then, on release,
                // open the brush menu on top of the cube -- and the cube's own
                // menu is what a right click there is for.
                if let Some(part) = navcube::pick(&self.camera, self.viewport_size, position) {
                    match button {
                        PointerButton::Left => self.fly_to(part),
                        PointerButton::Right => {
                            // Faces only. An edge or a corner points along no
                            // axis, and a quarter turn is the only kind of
                            // re-orientation offered, so there is nothing the
                            // menu could truthfully offer for one.
                            if part.extremes == 1
                                && let Some(facing) = brokkr_core::Facing::nearest(part.direction)
                            {
                                self.cube_menu = Some(CubeMenu { at: position, facing });
                            }
                        }
                        PointerButton::Middle => {}
                    }
                    return;
                }

                let kind = match button {
                    // A cut outranks sculpting: the mode was armed deliberately
                    // and the next left drag is the line, not a stroke.
                    PointerButton::Left if self.cut_armed => DragKind::Cutting,
                    // Left sculpts -- unless a hold-and-drag resize is in
                    // progress, in which case the pointer belongs to that
                    // gesture and a press must not lay down a stroke.
                    PointerButton::Left if self.sizing.is_some() => DragKind::Sizing,
                    // **The body is resolved from the raycast BEFORE anything
                    // opens a recorder**, which is the whole ordering this
                    // increment exists for; see `arm_recorder`. A press over a
                    // body that is not the active one selects it and carves
                    // nothing, and the press after that sculpts -- the same
                    // rule Photoshop's layer panel has, applied to the model
                    // itself so that direct manipulation acts on what is drawn.
                    // A press over a HIDDEN body picks nothing, because
                    // `Document::pick` never marches one.
                    PointerButton::Left => match self.pick(position) {
                        Some((body, _)) if body != self.doc.active() => {
                            self.select_body(body);
                            DragKind::Selecting
                        }
                        Some(_) => DragKind::Sculpt(self.stroke_direction()),
                        // Nothing drawn under the pointer. That is normally a
                        // press beside the model, which is allowed to become a
                        // stroke the moment it is dragged onto one -- but not
                        // when the body it would carve is itself hidden. See
                        // `active_is_drawn`: the stroke's own raycast does not
                        // consult the eye, so without this the press carves a
                        // body nobody can see. Saying which body is the honest
                        // half; a press that silently does nothing is its own
                        // puzzle.
                        //
                        // **And it must say which of the two hid it.** Under
                        // solo, the active body can be undrawn with its own eye
                        // plainly open -- clicking a row outside the scope
                        // selects it, and `select_body` does not veto that,
                        // because a view mode never vetoes a structural
                        // operation. "Show it before sculpting it" would then be
                        // advice for a state the user is not in, pointing at an
                        // eye that is already on.
                        None if !self.active_is_drawn() => {
                            let active = self.doc.active();
                            let name = self
                                .doc
                                .node(active)
                                .map_or("the active body", |node| node.name.as_str());
                            self.status = if self.in_solo_scope(active) {
                                format!("{name} is hidden — show it before sculpting it")
                            } else {
                                format!("{name} is outside the solo scope — escape leaves solo")
                            };
                            DragKind::Refused
                        }
                        None => DragKind::Sculpt(self.stroke_direction()),
                    },
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
                    self.sculpt_to(position, direction, true);
                }
                self.refresh_hover(position);
                self.refresh_overlay();
            }
            PointerEvent::Released { button } => {
                if let Some(drag) = self.drag.filter(|drag| drag.button == button) {
                    if matches!(drag.kind, DragKind::Cutting) {
                        // The drag already recorded where the button went down.
                        self.finish_cut(drag.origin);
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
                    // the line can be adjusted freely. A press that chose a
                    // body is over: dragging must not turn it into a stroke on
                    // a body the user has only just arrived at.
                    Some(DragKind::Cutting)
                    | Some(DragKind::Sizing)
                    | Some(DragKind::Selecting)
                    | Some(DragKind::Refused)
                    | None => {}
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
                    self.hover_body = None;
                    self.refresh_overlay();
                    return;
                }

                // The ring follows the pointer, and the surface under it moves
                // as a stroke cuts into it.
                self.refresh_hover(position);
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
    ///
    /// **The visibility pass is here rather than inside the arms that change
    /// an eye, and that placement is the guarantee.** `dispatch` returns early
    /// from several arms, and the set of messages that can change what is drawn
    /// is not a list anyone can keep correct -- open, reset, import, undo, redo,
    /// a delete and a solo all belong to it, and increments 9 to 13 add more.
    /// Recomputing it once, after whatever the message did, cannot go out of
    /// date. It costs a walk of at most `MAX_NODES` rows and no allocation.
    ///
    /// **The rename commit is here for the same reason**, and it is the whole
    /// of what the plan calls "blur commits": iced 0.14's `text_input` has no
    /// blur event to hang it on -- it has `on_input`, `on_submit` and
    /// `on_paste` and nothing else (`text_input.rs:172-224`) -- so the only
    /// honest definition of blur is "the user did something that was not
    /// typing in the field", and that is a question about the message, asked
    /// once, before the message is acted on. See [`keeps_the_rename_open`].
    pub fn update(&mut self, message: Message) -> Task<Message> {
        if self.renaming.is_some() && !keeps_the_rename_open(&message) {
            self.commit_rename();
        }
        let task = self.dispatch(message);
        self.publish_visibility();
        task
    }

    /// What one message actually does. See [`Brokkr::update`], which is the
    /// only caller and which is where the visibility pass lives.
    fn dispatch(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Pointer(event) => self.on_pointer(event),
            Message::Frame => {
                let elapsed_ms = self.perf.record_frame();
                if !self.facts_recorded {
                    self.facts_recorded = true;
                    self.record_session_facts();
                }
                self.record_status_change();
                if self.advance_flight(elapsed_ms) {
                    self.publish_camera();
                }
                // Driven off the frame tick for the same reason the autosave
                // is: `iced::time::every` does not exist under this feature
                // set. It stalls on a hidden window, which for playback is
                // exactly right -- nothing is watching it.
                if let Some(pose) = self.timeline.advance(elapsed_ms) {
                    self.fly_camera_to(&pose);
                }
                self.drive_from_spacemouse(elapsed_ms);
                self.maybe_autosave();
            }
            Message::BrushKindChanged(kind) => {
                // Remember what the outgoing brush was set to, and restore what
                // this one was last on.
                self.strengths[Self::strength_slot(self.brush.kind)] = self.brush.strength;
                self.brush.kind = kind;
                self.brush.strength = self.strengths[Self::strength_slot(kind)];
            }
            Message::FalloffChanged(curve) => self.brush.falloff = curve,
            // Clamped rather than trusted. The slider's own range is already
            // voxel-aware, but a message can arrive from a typed field or the
            // right-click menu too, and at a resin lattice an unclamped 20 mm
            // is a quarter of a second per stamp.
            Message::BrushRadiusChanged(radius) => {
                self.brush.radius = radius.clamp(MIN_RADIUS_MM, self.max_radius())
            }
            Message::BrushStrengthChanged(strength) => {
                self.brush.strength = strength;
                self.strengths[Self::strength_slot(self.brush.kind)] = strength;
            }
            Message::SymmetryAxisToggled(axis) => self.toggle_mirror(axis),
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
            Message::OpenInSlicer => {
                self.top_menu = None;
                self.hand_to_slicer();
            }
            Message::PrinterChecked => {
                self.top_menu = None;
                let Some(printer) =
                    crate::printer::configured(crate::paths::config_file("printer").as_deref())
                else {
                    // Says where to write it rather than only that it is
                    // missing: a setting with no interface needs its file named
                    // or it may as well not exist.
                    self.status = match crate::paths::config_file("printer") {
                        Some(path) => format!(
                            "no printer set — put `host = 192.0.2.46` in {}",
                            path.display()
                        ),
                        None => "could not work out where the printer config lives".to_string(),
                    };
                    return Task::none();
                };
                self.status = format!("asking {}…", printer.host);
                return Task::perform(
                    async move {
                        crate::printer::status(&printer.host, printer.port)
                            .map(|status| status.summary())
                    },
                    Message::PrinterAnswered,
                );
            }
            Message::PrinterAnswered(answer) => {
                self.status = match answer {
                    Ok(summary) => summary,
                    Err(why) => format!("could not reach the printer: {why}"),
                };
            }
            Message::BugReportOpened => {
                self.top_menu = None;
                self.bug_report = Some(BugReport::new());
            }
            Message::BugReportEdited(action) => {
                if let Some(draft) = &mut self.bug_report {
                    draft.description.perform(action);
                }
            }
            Message::BugReportDetailToggled(on) => {
                if let Some(draft) = &mut self.bug_report {
                    draft.with_detail = on;
                }
            }
            Message::BugReportDismissed => self.bug_report = None,
            Message::BugReportCopied => {
                let Some(report) = self.assemble_report() else {
                    self.status =
                        "could not build the report: describe the problem first".to_string();
                    return Task::none();
                };
                self.status = "report copied — paste it into an issue".to_string();
                return iced::clipboard::write(report.to_json());
            }
            Message::BugReportSubmitted => {
                let Some(report) = self.assemble_report() else {
                    self.status = "could not send: describe the problem first".to_string();
                    return Task::none();
                };
                if let Some(draft) = &mut self.bug_report {
                    draft.sending = true;
                }
                self.status = "sending the report…".to_string();
                return Task::perform(
                    async move { crate::report::send(report, crate::report::TINKERATLAS) },
                    Message::BugReportFinished,
                );
            }
            Message::BugReportFinished(outcome) => {
                self.bug_report = None;
                self.status = match outcome {
                    Ok(note) => note,
                    // The substring "could not" is what colours the status line
                    // as an error -- see `panel.rs`. A failure worded without it
                    // renders in muted grey as though it had worked.
                    Err(why) => format!("could not send the report: {why}"),
                };
            }
            Message::KeyPressed { key, modifiers } => return self.on_key(key, modifiers),
            // **Deliberately empty, and it must stay empty.** The work this
            // message does happens before `dispatch` is called at all: it is
            // not on `keeps_the_rename_open`'s list, so `Brokkr::update`
            // commits the rename in flight and then arrives here with nothing
            // left to do. Giving it a body would make every click on the panel
            // background do something, which is the opposite of what it is.
            Message::PressedNothing => {}
            Message::MenuClosed => {
                // Escape is the only sender, by way of `on_key`. Against an
                // open prompt it means Cancel, which is the harmless answer --
                // an explicit arm rather than letting the clears below reach
                // `confirm`, since dismissing the prompt by accident is how
                // work gets lost.
                if self.confirm.is_some() {
                    return self.answer_confirm(ConfirmChoice::Cancel);
                }
                // Same reasoning one card down: Escape against the delete
                // prompt means "don't", which is the harmless answer.
                if self.pending_delete.take().is_some() {
                    return Task::none();
                }
                // And the merge prompt, which asks the same kind of question
                // about the same kind of size. Escape means "don't", which
                // again changes nothing.
                if self.pending_merge.take().is_some() {
                    return Task::none();
                }
                // And one card further down: Escape against a rename means
                // "keep the old name", which is the harmless answer, and it
                // must not also disarm the cut on its way past.
                //
                // **This is the SECOND Escape, not the first, and the plan says
                // otherwise.** A focused `text_input` handles Escape itself --
                // it clears its own focus and captures the event
                // (`text_input.rs:1235-1244`) -- so the first press never
                // reaches `key_event` at all. The field is left open and
                // unfocused, still showing what was typed, and the next Escape
                // arrives here. Nothing in the widget tree can be put in front
                // of that: iced hands an event to the content before the
                // container, so the innermost widget always wins.
                if self.renaming.is_some() {
                    self.cancel_rename();
                    return Task::none();
                }
                // And one further down again: solo, BEFORE the armed cut.
                //
                // The order is the point. Both are modes, and with both on, one
                // Escape has to pick -- so it picks the one whose exit costs
                // nothing. Leaving solo puts nineteen bodies back on screen and
                // changes not a byte; disarming the cut throws away a mode the
                // user deliberately armed and would have to arm again. The
                // second Escape then reaches the cut, which is the order every
                // other card here already follows: the harmless answer first.
                //
                // **This is also solo's only exit that is not a document
                // change.** `ctrl+alt+comma` leaves the mode too, but it turns
                // every eye on as it goes -- silently revealing the six bodies
                // the user hid an hour ago -- and exiting a mode must not be an
                // edit. That is why Escape is here rather than left to it.
                if self.solo.is_some() {
                    self.exit_solo();
                    return Task::none();
                }
                // Escape is also the way out of an armed cut, which is the only
                // mode in the application that changes what a click does.
                self.cut_armed = false;
                self.adding = false;
                self.menu = None;
                self.menu_edit = None;
                self.top_menu = None;
                self.cube_menu = None;
                // Unlike `confirm` above, this one IS cleared: declining to
                // turn the model is the safe answer and leaves it exactly as
                // imported, where declining to answer "you have unsaved work"
                // loses the work.
                self.orient_prompt = None;
            }
            Message::MenuFieldEdited(which, text) => self.edit_menu_field(which, text),
            Message::MenuFieldSubmitted => self.menu_edit = None,
            Message::OrientFace(to) => {
                // Taken rather than read: the menu asked its question and has
                // been answered, and leaving it open over a model that just
                // moved would invite a second turn from a face that is no
                // longer there.
                if let Some(menu) = self.cube_menu.take() {
                    self.orient(brokkr_core::AxisRotation::taking(menu.facing, to));
                }
            }
            Message::OrientPromptAnswered(accept) => {
                if let Some(up) = self.orient_prompt.take()
                    && accept
                {
                    self.orient(brokkr_core::AxisRotation::taking(up, brokkr_core::Facing::Up));
                }
            }
            Message::CutToggled => {
                self.cut_armed = !self.cut_armed;
                self.status = if self.cut_armed {
                    "cut armed: drag a line across the model, the left of the arrow goes"
                        .to_string()
                } else {
                    String::new()
                };
            }
            Message::DynamicRadiusToggled(on) => self.dynamic_radius = on,

            // --- the body panel ----------------------------------------------
            Message::BodySelected(id) => self.select_body(id),
            Message::BodyVisibilityToggled(id) => self.toggle_visibility(id),
            Message::ActiveBodyVisibilityToggled => {
                let active = self.doc.active();
                self.toggle_visibility(active);
            }
            Message::EveryBodyShown => self.show_everything(),
            Message::PrimitiveMenuToggled => self.adding = !self.adding,
            Message::PrimitiveAdded(kind) => {
                // Closed whatever happens, including on a refusal: a menu that
                // stays open after a press that was refused reads as a press
                // that never landed.
                self.adding = false;
                self.add_primitive(kind);
            }
            Message::BodyDeleted => self.delete_active_body(),
            Message::BodyDuplicated => self.duplicate_active_body(),
            Message::BodyGrouped => self.group_active_body(),
            Message::BodyUngrouped => self.ungroup_active_body(),
            Message::BodyMovedToFolder(into) => self.move_to_folder(into),
            Message::FolderCollapseToggled(id) => self.toggle_collapse(id),
            Message::FolderDeleted(id) => self.delete_folder(id),
            Message::BodyDeleteConfirmed => {
                if let Some(pending) = self.pending_delete.take() {
                    self.remove_body(pending.id);
                }
            }
            Message::BodyDeleteCancelled => {
                // Nothing else: cancelling changes not one byte of the
                // document, which is the whole promise of the prompt.
                self.pending_delete = None;
            }
            Message::BodyMergedDown => self.merge_active_body_down(),
            Message::BodyMergeConfirmed => {
                if let Some(pending) = self.pending_merge.take() {
                    self.apply_merge(pending.source);
                }
            }
            Message::BodyMergeCancelled => {
                // As above: cancelling a merge changes nothing at all, which is
                // the only reason it is safe to ask.
                self.pending_merge = None;
            }
            // Session state, so deliberately NOT `unsaved`: nothing about it is
            // written to the file.
            Message::ThumbnailsToggled => self.thumbnails = !self.thumbnails,
            // A view mode, and likewise not `unsaved` -- with the one exception
            // `enter_solo` documents, where soloing a row whose eye is off turns
            // that eye on and THAT is an ordinary edit.
            Message::SoloEntered(id) => self.enter_solo(id),
            Message::SoloExited => self.exit_solo(),
            Message::BodyRenameBegan(id) => return self.begin_rename(id),
            Message::BodyRenameEdited(text) => {
                if let Some((_, held)) = &mut self.renaming {
                    // Clamped as it is typed, and not at commit. The field's
                    // job is to show what the file will hold: a thirty-third
                    // byte that looks accepted and is dropped on the next save
                    // is exactly the silent loss `name_that_fits` exists to
                    // stop, moved one layer up where it is more convincing.
                    held.clear();
                    held.push_str(brokkr_core::name_that_fits(&text));
                }
            }
            // Nothing, and that is not an oversight: `update`'s guard has
            // already committed by the time this arrives, because
            // `keeps_the_rename_open` does not list it. Committing here as well
            // would be a second commit point, which is the one thing the guard
            // exists to prevent.
            Message::BodyRenameSubmitted => {}

            Message::TimelineResized(width) => self.timeline.resized(width),
            Message::TimelineHover(x) => {
                self.timeline.hover(x);
                // A drag re-times the key it is holding, which moves it under
                // the playhead, so the view follows the pointer.
                if self.timeline.dragged_key().is_some() {
                    self.unsaved = true;
                }
            }
            Message::TimelinePressed => {
                let view = self.current_view();
                match self.timeline.press(view) {
                    Some(crate::timeline::Pressed::WentTo(index)) => {
                        // Before the view is applied, so that a mirror plane
                        // the body does not straddle wins the line: which key
                        // was gone to is visible in the timeline, and a plane
                        // going off is not visible anywhere else.
                        self.status = format!("key {} of {}", index + 1, self.timeline.keys.len());
                        // Everything a key holds, not only the camera: going
                        // to a key is going back to a working setup.
                        if let Some(key) = self.timeline.keys.get(index).copied() {
                            self.apply_view(&key.view);
                            self.publish_camera();
                            self.refresh_overlay();
                        }
                    }
                    Some(crate::timeline::Pressed::Added(index)) => {
                        self.unsaved = true;
                        self.status =
                            format!("stored key {} of {}", index + 1, self.timeline.keys.len());
                    }
                    None => {}
                }
            }
            Message::TimelineReleased => self.timeline.release(),
            Message::TimelineLeft => self.timeline.leave(),
            Message::TimelineRemoveKey => {
                if self.timeline.remove_under_pointer().is_some() {
                    self.unsaved = true;
                    self.status = match self.timeline.keys.len() {
                        0 => "removed the last key".to_string(),
                        left => format!("removed a key, {left} left"),
                    };
                }
            }
            Message::TimelinePlayToggled => {
                self.timeline.toggle_play();
                if self.timeline.playing {
                    // The playhead may be sitting anywhere; put the camera
                    // where it is before the first frame advances it, or
                    // playback starts with a jump.
                    if let Some(pose) = self.timeline.pose_at(self.timeline.playhead) {
                        self.fly_camera_to(&pose);
                    }
                }
            }
            Message::BrushRadiusScaled(factor) => {
                self.brush.radius =
                    (self.brush.radius * factor).clamp(MIN_RADIUS_MM, self.max_radius());
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
            Message::WorkingSizeTyped(text) => self.working_size_field = text,
            Message::WorkingSizeCommitted => {
                match self.working_size_field.trim().parse::<f32>() {
                    Ok(mm) => self.set_working_size(mm),
                    Err(_) => {
                        self.status = "could not resize: type a size in millimetres".into();
                    }
                }
                self.working_size_field.clear();
            }
            Message::SpaceMouse(setting) => self.configure_spacemouse(setting),
            Message::SectionToggled(section) => {
                let open = &mut self.expanded[section as usize];
                *open = !*open;
            }
            Message::StatsToggled => self.stats_open = !self.stats_open,
            // Both the panel button and File > New come here, so the two
            // cannot drift apart. Both discard the document, so both ask.
            Message::ResetSphere | Message::NewSculpt => {
                return self.guard(PendingAction::NewSculpt);
            }

            // --- the unsaved-work prompt -------------------------------------
            // The window is undecorated, so these four are the title bar the
            // compositor is no longer drawing. Each needs the window's id and
            // none of them has it, so each asks for it first.
            Message::TitleBarDragged => {
                return iced::window::latest().and_then(iced::window::drag);
            }
            Message::TitleBarDoubleClicked => {
                return iced::window::latest().and_then(iced::window::toggle_maximize);
            }
            Message::WindowMinimise => {
                return iced::window::latest().and_then(|id| iced::window::minimize(id, true));
            }
            // Routed through CloseRequested rather than closing directly, so
            // the unsaved-work prompt is the same one the compositor's close
            // button used to reach.
            Message::ResizeStarted(direction) => {
                return iced::window::latest()
                    .and_then(move |id| iced::window::drag_resize(id, direction));
            }
            Message::WindowClose => {
                return iced::window::latest()
                    .and_then(|id| Task::done(Message::CloseRequested(id)));
            }
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
                let voxel_size = self.doc.voxel_size();
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
                            // Measured on the mesh, because by the time there
                            // is a volume the mesh is gone and its bounding box
                            // has been centred on the origin -- which destroys
                            // the one tell there is.
                            let resting_up = brokkr_core::resting_up(&mesh.positions);
                            // `already_reserved` stays at zero, which is what
                            // `at` gives and what a REPLACING import needs:
                            // `adopt_import` swaps the whole document and
                            // `rebuild_everything` empties the pool, so the
                            // model on screen while this runs is not competing
                            // with the one being built. An import that JOINS a
                            // document -- "Import as a new body", deferred out
                            // of this arc -- has to pass the pool's watermark
                            // here instead, or the two will overflow it
                            // between them with nothing reporting it.
                            let options = brokkr_core::voxelise::VoxeliseOptions::at(voxel_size);
                            brokkr_core::voxelise::voxelise(&mesh, &options).map(
                                |(volume, report)| crate::message::Imported {
                                    volume,
                                    report,
                                    source: path.clone(),
                                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                                    resting_up,
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
    use brokkr_core::{BrushKind, Change};
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
        let before = app.doc.active_volume().sample_world(front);

        press(&mut app, centre_of_viewport());

        assert!(app.perf.edit_ms > 0.0, "no edit was timed, so the raycast missed the sphere");
        assert!(app.perf.dirty_bricks > 0, "the stroke dirtied nothing");
        assert!(
            app.doc.active_volume().sample_world(front) < before,
            "adding clay should have pushed the field negative at the surface"
        );
    }

    /// The stats readout starts collapsed, and the icon is what brings it back.
    ///
    /// Pinned because "it is not covering the model" is the whole point of the
    /// control: a default flipped by an unrelated edit to the constructor would
    /// put seven lines of monospace back across the corner of the sculpt with
    /// nothing failing anywhere.
    #[test]
    fn the_stats_readout_starts_collapsed_and_the_icon_toggles_it() {
        let mut app = app();
        assert!(!app.stats_open, "the readout should not be over the model on launch");

        update(&mut app, Message::StatsToggled);
        assert!(app.stats_open, "pressing the info icon should show the numbers");

        update(&mut app, Message::StatsToggled);
        assert!(!app.stats_open, "pressing it again should put them away");
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
        let bricks_before = app.doc.active_volume().brick_count();

        // The far corner of the viewport looks past the sphere into empty space.
        press(&mut app, Vector::new(2.0, 2.0));

        assert_eq!(
            app.doc.active_volume().brick_count(),
            bricks_before,
            "a miss must not allocate"
        );
        assert_eq!(app.perf.dirty_bricks, 0, "a miss must not schedule a remesh");
    }

    #[test]
    fn orbiting_moves_the_camera_without_touching_the_model() {
        let mut app = app();
        let yaw = app.camera.yaw;
        let bricks = app.doc.active_volume().brick_count();

        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Right,
            position: centre_of_viewport(),
            size: SIZE,
        });
        app.on_pointer(PointerEvent::Moved { position: Vector::new(460.0, 300.0), size: SIZE });

        assert_ne!(app.camera.yaw, yaw, "a right drag should have orbited");
        assert_eq!(app.doc.active_volume().brick_count(), bricks, "orbiting must not sculpt");
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

    /// The eviction that has to speak for itself.
    ///
    /// A per-operation prompt cannot catch this: two folder deletes that each
    /// pass the reclaim allowance on their own go on to evict each other, and
    /// the only moment anything knows the body has become unrecoverable is the
    /// eviction. The delete gesture itself does not exist yet, so the entry is
    /// built here by hand -- what is under test is the path from
    /// `HistoryStats::dropped_bodies` to something the user can read.
    #[test]
    fn an_eviction_that_takes_a_deleted_body_with_it_reaches_the_status_line() {
        let mut app = app();
        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(60.0, 0.0, 0.0), 10.0);
        let doomed = app.doc.add_body("Body 2", second);
        app.remesh_dirty();

        // An allowance of one byte, so the next push evicts the delete while
        // the stroke budget is nowhere near its ceiling.
        app.history = History::with_budgets(brokkr_core::DEFAULT_HISTORY_BUDGET, 1);
        let at = app.doc.index_of(doomed).expect("the second body");
        let node = app.doc.remove(at);
        app.history.push(Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]));
        assert!(app.status.is_empty(), "nothing has been evicted yet");

        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.history_stats.dropped_bodies, 1, "the delete was not evicted");
        assert!(
            app.status.contains("deleted body"),
            "the eviction left the user nothing to read: {:?}",
            app.status
        );
    }

    /// A body that comes back from the undo stack has to reach the GPU, and
    /// the count the remesh reports is the only thing in the application that
    /// says it did.
    ///
    /// `perf.dirty_bricks` and not `perf.remesh_ms`: the first is written
    /// before `remesh_dirty`'s early return and the second only past it, so a
    /// restored body that scheduled nothing would leave `remesh_ms` reading
    /// whatever the last real remesh cost.
    ///
    /// The remesh before the delete is load-bearing -- it is what drains the
    /// dirty set the second body was seeded with, so that the only thing that
    /// can mark its bricks afterwards is `Document::insert`.
    #[test]
    fn undoing_a_body_delete_schedules_every_one_of_its_bricks_for_remesh() {
        let mut app = app();
        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(60.0, 0.0, 0.0), 10.0);
        let doomed = app.doc.add_body("Body 2", second);
        app.remesh_dirty();

        let bricks = app.doc.volume(doomed).expect("the second body").brick_count();
        assert!(bricks > 0, "the fixture body must have bricks or this asserts nothing");

        // The delete itself, which has no interface yet: the node moves out of
        // the document and into the entry.
        let at = app.doc.index_of(doomed).expect("the second body");
        let node = app.doc.remove(at);
        app.history.push(Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]));

        update(&mut app, Message::Undo);

        assert!(app.doc.volume(doomed).is_some(), "the body did not come back");
        assert!(
            app.perf.dirty_bricks >= bricks,
            "the restored body scheduled {} bricks, not its {bricks}, so it never reaches the screen",
            app.perf.dirty_bricks
        );
    }

    #[test]
    fn undo_returns_the_model_to_where_it_started() {
        let mut app = app();
        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let before = app.doc.active_volume().sample_world(front);

        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert_ne!(app.doc.active_volume().sample_world(front), before);

        update(&mut app, Message::Undo);
        assert_eq!(
            app.doc.active_volume().sample_world(front),
            before,
            "undo did not restore the field"
        );
        assert!(app.perf.dirty_bricks > 0, "undo must schedule a remesh or the screen goes stale");

        update(&mut app, Message::Redo);
        assert_ne!(
            app.doc.active_volume().sample_world(front),
            before,
            "redo did not reapply the stroke"
        );
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
        let before = app.doc.active_volume().sample_world(mirrored);

        press(&mut app, off_centre);
        release(&mut app);

        assert!(
            app.doc.active_volume().sample_world(mirrored) < before,
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
        let flat = app.doc.active_volume().sample_world(probe);

        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        for _ in 0..4 {
            press(&mut app, centre_of_viewport());
            release(&mut app);
        }
        let raised = app.doc.active_volume().sample_world(probe);
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
        let smoothed = app.doc.active_volume().sample_world(probe);

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
            let normal = app.doc.active_volume().gradient_world(at);
            let before = app.doc.active_volume().sample_world(at * probe);

            app.doc.active_volume_mut().begin_stroke();
            app.brush.apply_symmetric(
                app.doc.active_volume_mut(),
                &Stamp::new(at, normal, BrushDirection::Add),
                app.symmetry,
                MIRROR_CENTRE,
                &mut app.brush_scratch,
            );

            assert!(
                app.doc.active_volume().sample_world(at * probe) < before,
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
        let before = app.doc.active_volume().sample_world(probe);

        update(&mut app, Message::SizingStarted(SizingTarget::Radius));
        press(&mut app, centre_of_viewport());
        for step in 1..=8 {
            app.on_pointer(PointerEvent::Moved {
                position: Vector::new(SIZE.x / 2.0 + step as f32 * 10.0, SIZE.y / 2.0),
                size: SIZE,
            });
        }
        release(&mut app);

        assert_eq!(
            app.doc.active_volume().sample_world(probe),
            before,
            "a sizing drag cut into the model"
        );
        assert!(!app.history.can_undo(), "a sizing drag recorded an undo entry");
    }

    #[test]
    fn the_sizing_gesture_stays_inside_the_slider_bounds() {
        // The radius ceiling now depends on the voxel size, so it is asked of a
        // real app rather than read off a constant.
        let radius_ceiling = app().max_radius();
        for (target, low, high) in [
            (SizingTarget::Radius, MIN_RADIUS_MM, radius_ceiling),
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
        let sculpted = app.doc.active_volume().sample_world(probe);
        let expected_bricks: usize = app.doc.active_volume().brick_coords().count();

        app.save_project(&path);
        assert!(app.status.starts_with("saved"), "save reported: {}", app.status);
        assert_eq!(app.project_path.as_deref(), Some(path.as_path()));

        // A different session entirely, as reopening after quitting would be.
        let mut reopened = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        reopened.open_project(&path);
        assert!(reopened.status.starts_with("opened"), "open reported: {}", reopened.status);

        assert_eq!(
            reopened.doc.active_volume().sample_world(probe),
            sculpted,
            "the field came back different"
        );
        assert_eq!(reopened.doc.active_volume().brick_coords().count(), expected_bricks);
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
        let (_, report) = reopened.doc.active_volume().export_mesh();
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
        assert!(!app.history.can_undo(), "the fixture already had something to undo");
        app.undo();
        assert!(!app.unsaved, "an undo that did nothing armed the unsaved marker");

        // The control. Without it this test asserts the default value and would
        // pass just as happily if `undo` did nothing at all, in either case.
        press(&mut app, centre_of_viewport());
        release(&mut app);
        app.unsaved = false;
        app.undo();
        assert!(
            app.unsaved,
            "an undo with something to undo failed to mark it, so the \
                              assertion above was vacuous"
        );
    }

    /// `resample` returns early when the size is already in use. The marker sits
    /// past that return, so the no-op stays a no-op.
    #[test]
    fn resampling_to_the_current_size_leaves_the_flag_alone() {
        let mut app = app();
        app.resample(app.doc.voxel_size());
        assert!(!app.unsaved, "a resample that did nothing armed the unsaved marker");

        app.resample(app.doc.voxel_size() * 2.0);
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
        let sculpted = app.doc.active_volume().brick_coords().count();

        update(&mut app, Message::NewSculpt);

        assert_eq!(app.confirm, Some(PendingAction::NewSculpt), "no prompt was raised");
        assert!(app.unsaved, "the document was reset behind the prompt");
        assert_eq!(
            app.doc.active_volume().brick_coords().count(),
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
        assert!(!app.unsaved, "the fixture starts dirty, so this proves nothing");

        update(&mut app, Message::NewSculpt);
        assert!(app.confirm.is_none(), "a clean document was prompted");

        update(&mut app, Message::CloseRequested(iced::window::Id::unique()));
        assert!(app.confirm.is_none(), "a clean document was prompted on quit");

        update(&mut app, Message::OpenRequested);
        assert!(app.confirm.is_none(), "a clean document was prompted on open");

        // The control: the same three actions DO prompt once there is something
        // to lose. Without it every assertion above is satisfied by the default.
        for action in [Message::NewSculpt, Message::CloseRequested(iced::window::Id::unique())] {
            let mut dirty = app_with_unsaved_work();
            update(&mut dirty, action);
            assert!(
                dirty.confirm.is_some(),
                "the gate never fires, so the checks above are vacuous"
            );
        }
    }

    /// The prompt draws over the viewport but the shader widget still sees
    /// events wherever the cursor is. A press behind the card must neither
    /// dismiss the prompt nor sculpt into a document about to be discarded.
    #[test]
    fn a_viewport_press_does_not_dismiss_the_prompt_or_sculpt() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::NewSculpt);
        let before = app.doc.active_volume().sample_world(Vec3::ZERO);
        let entries = app.history_stats.undo_entries;

        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.confirm, Some(PendingAction::NewSculpt), "a stray click dismissed it");
        assert_eq!(
            app.doc.active_volume().sample_world(Vec3::ZERO),
            before,
            "it sculpted behind the prompt"
        );
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

    // --- the subscription's key decode ---------------------------------------
    //
    // Everything in the section below this one synthesises `Message::KeyPressed`
    // directly, which is downstream of `key_event` and cannot see it. These
    // four cases are the only thing standing between a wrong `key_event` and a
    // green suite: with the press arm deleted the whole keyboard is dead and
    // every guard test still passes, guarding a message nothing can produce.

    /// A window event carrying `character`, pressed or released.
    ///
    /// The fields beyond `key` and `modifiers` are what a real winit press
    /// fills in and `key_event` ignores; they are here because the literal
    /// does not compile without them.
    fn key_window_event(character: &str, modifiers: iced::keyboard::Modifiers) -> iced::Event {
        iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
            key: iced::keyboard::Key::Character(character.into()),
            modified_key: iced::keyboard::Key::Character(character.into()),
            physical_key: iced::keyboard::key::Physical::Unidentified(
                iced::keyboard::key::NativeCode::Unidentified,
            ),
            location: iced::keyboard::Location::Standard,
            modifiers,
            text: None,
            repeat: false,
        })
    }

    fn key_release_event(character: &str) -> iced::Event {
        iced::Event::Keyboard(iced::keyboard::Event::KeyReleased {
            key: iced::keyboard::Key::Character(character.into()),
            modified_key: iced::keyboard::Key::Character(character.into()),
            physical_key: iced::keyboard::key::Physical::Unidentified(
                iced::keyboard::key::NativeCode::Unidentified,
            ),
            location: iced::keyboard::Location::Standard,
            modifiers: iced::keyboard::Modifiers::empty(),
        })
    }

    /// `listen_with` hands every event to `key_event` regardless of who wanted
    /// it, so dropping the captured ones is the whole of the focus awareness.
    /// Inverting this check is verbatim the bug that shipped for a year:
    /// `1`-`7`, `s`, `u`, `x`, `y` and `z` stolen from every text field,
    /// because a focused input reports its keystrokes captured and the
    /// shortcut fired anyway.
    #[test]
    fn a_key_a_widget_already_consumed_never_reaches_the_application() {
        let typed = key_window_event("z", ctrl());

        let message = key_event(typed, iced::event::Status::Captured, iced::window::Id::unique());

        assert!(
            message.is_none(),
            "a captured key press was forwarded, so shortcuts fire while typing"
        );
    }

    /// The other half: an event nobody claimed becomes a `KeyPressed` carrying
    /// exactly the key and modifiers that arrived. `on_key` decodes from those
    /// two fields alone, so anything lost here is a shortcut that cannot fire
    /// -- dropping the modifiers would turn every `ctrl+z` into a bare `z`.
    #[test]
    fn an_ignored_key_press_is_forwarded_with_its_key_and_modifiers_intact() {
        let mut modifiers = ctrl();
        modifiers.insert(iced::keyboard::Modifiers::SHIFT);
        let pressed = key_window_event("z", modifiers);

        let message = key_event(pressed, iced::event::Status::Ignored, iced::window::Id::unique());

        let Some(Message::KeyPressed { key, modifiers: forwarded }) = message else {
            panic!("an ignored key press produced {message:?} rather than a KeyPressed");
        };
        assert_eq!(key, iced::keyboard::Key::Character("z".into()));
        assert!(forwarded.command(), "the command modifier was dropped");
        assert!(forwarded.shift(), "the shift modifier was dropped");
    }

    /// A left press the widget tree ignored becomes the blur signal, and a
    /// press anyone claimed does not.
    ///
    /// **This arm is the only thing in the application that can see a blur.**
    /// `text_input` unfocuses itself on any left press outside its own bounds
    /// and publishes nothing (`text_input.rs:723-735`), and the bodies list is
    /// a fixed six or eight rows tall, so a press in the empty scrollable
    /// under the last row belongs to no widget. Without this arm that press
    /// produced no message at all, `renaming` stayed set, and the field went
    /// on being drawn while the next keystroke fired a tool shortcut.
    ///
    /// Both controls matter. Captured must stay dropped, or clicking INSIDE
    /// the field would close it. The right button must stay silent, because
    /// `text_input` does not blur on it -- committing there would close a
    /// field that is still focused and still taking keys.
    #[test]
    fn a_left_press_nobody_wanted_is_the_only_blur_the_application_can_see() {
        let press = |button| iced::Event::Mouse(iced::mouse::Event::ButtonPressed(button));

        let ignored = key_event(
            press(iced::mouse::Button::Left),
            iced::event::Status::Ignored,
            iced::window::Id::unique(),
        );
        assert!(
            matches!(ignored, Some(Message::PressedNothing)),
            "a press on nothing produced {ignored:?}, so a rename in flight never hears about it"
        );

        let captured = key_event(
            press(iced::mouse::Button::Left),
            iced::event::Status::Captured,
            iced::window::Id::unique(),
        );
        assert!(
            captured.is_none(),
            "a press a widget wanted produced {captured:?}; clicking inside the field closes it"
        );

        let right = key_event(
            press(iced::mouse::Button::Right),
            iced::event::Status::Ignored,
            iced::window::Id::unique(),
        );
        assert!(
            right.is_none(),
            "a right press produced {right:?}, but the field stays focused through one"
        );
    }

    /// Releases are the one thing `key_event` decodes itself, because ending a
    /// gesture must not go through the modal guard. Only the two sizing keys
    /// mean anything on release; every other release is somebody else's.
    #[test]
    fn releasing_a_sizing_key_ends_the_gesture_and_no_other_release_means_anything() {
        for sizing in ["s", "u"] {
            let message = key_event(
                key_release_event(sizing),
                iced::event::Status::Ignored,
                iced::window::Id::unique(),
            );
            assert!(
                matches!(message, Some(Message::SizingEnded)),
                "releasing {sizing} produced {message:?} rather than ending the sizing drag"
            );
        }

        let message = key_event(
            key_release_event("x"),
            iced::event::Status::Ignored,
            iced::window::Id::unique(),
        );
        assert!(
            message.is_none(),
            "releasing x produced {message:?}; only presses carry meaning for the mirror keys"
        );
    }

    // --- the modal keyboard and pointer guard --------------------------------

    /// Press a character key the way the subscription does, on an event the
    /// widget tree ignored.
    ///
    /// Note what this does NOT do: call `viewport::shortcut` and feed the
    /// result in. That would test the decode and skip the guard, which is the
    /// only thing these tests are about.
    fn key(app: &mut Brokkr, character: &str, modifiers: iced::keyboard::Modifiers) {
        update(
            app,
            Message::KeyPressed {
                key: iced::keyboard::Key::Character(character.into()),
                modifiers,
            },
        );
    }

    fn bare() -> iced::keyboard::Modifiers {
        iced::keyboard::Modifiers::empty()
    }

    /// `command()` is control on this platform, and the shortcut table reads
    /// `command()` rather than the raw bit, so the tests must set what it
    /// reads or they prove nothing.
    fn ctrl() -> iced::keyboard::Modifiers {
        iced::keyboard::Modifiers::CTRL
    }

    /// The undo that used to reach through the card.
    ///
    /// `Message::Undo` had no guard of any kind, so ctrl+Z with the
    /// unsaved-work prompt up rolled back the very stroke the prompt was
    /// asking whether to keep -- and answering Save then wrote out a document
    /// the user had not agreed to.
    #[test]
    fn control_z_under_a_modal_leaves_the_document_alone() {
        let mut app = app_with_unsaved_work();
        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        update(&mut app, Message::NewSculpt);
        assert!(app.confirm.is_some(), "the fixture never raised a prompt");

        let before = app.doc.active_volume().sample_world(front);
        key(&mut app, "z", ctrl());

        assert_eq!(
            app.doc.active_volume().sample_world(front),
            before,
            "ctrl+Z undid a stroke behind the prompt"
        );
        assert!(app.history.can_undo(), "the stroke left the undo stack");
        assert!(app.confirm.is_some(), "the prompt went away on its own");
    }

    /// Every other shortcut, on every modal. The brush and the mirror planes
    /// are not the document, but changing them behind a card the user is
    /// reading is the same surprise, and `1`-`6` did exactly that.
    #[test]
    fn no_shortcut_fires_while_any_modal_card_is_up() {
        fn check(what: &str, raise: impl FnOnce(&mut Brokkr)) {
            let mut app = app_with_unsaved_work();
            let brush = app.brush.kind;
            raise(&mut app);
            assert!(app.modal_open(), "{what} did not count as a modal");

            key(&mut app, "x", bare());
            key(&mut app, "2", bare());

            assert!(!app.symmetry.axis(MirrorAxis::X), "x flipped a mirror plane behind {what}");
            assert_eq!(app.brush.kind, brush, "a digit swapped the brush behind {what}");
        }

        check("the unsaved-work prompt", |app| update(app, Message::NewSculpt));
        check("the bug report", |app| update(app, Message::BugReportOpened));
        check("the orientation prompt", |app| {
            app.adopt_import(imported_with(Some(brokkr_core::Facing::Back)));
        });
    }

    /// The control for both tests above: with nothing modal up, the same three
    /// keystrokes all land. Without this they would pass against a build where
    /// the keyboard was simply dead.
    #[test]
    fn the_same_keys_all_land_with_no_modal_up() {
        let mut app = app_with_unsaved_work();
        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let sculpted = app.doc.active_volume().sample_world(front);
        assert!(!app.modal_open(), "the fixture is not a fair control");

        key(&mut app, "x", bare());
        assert!(app.symmetry.axis(MirrorAxis::X), "x did not reach the mirror planes");

        key(&mut app, "2", bare());
        assert_eq!(app.brush.kind, BrushKind::ALL[1], "the digit did not reach the brush");

        key(&mut app, "z", ctrl());
        assert_ne!(
            app.doc.active_volume().sample_world(front),
            sculpted,
            "ctrl+Z did not undo the stroke"
        );
    }

    /// Escape is the exception, and has to stay one: it is how the card is
    /// answered, so it is the one key that must pass the guard.
    #[test]
    fn escape_still_reaches_a_modal_through_the_key_path() {
        let mut app = app_with_unsaved_work();
        update(&mut app, Message::NewSculpt);

        update(
            &mut app,
            Message::KeyPressed {
                key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape),
                modifiers: bare(),
            },
        );

        assert!(app.confirm.is_none(), "escape was swallowed by the modal guard");
        assert!(app.unsaved, "escape discarded the work it was asking about");
    }

    /// The bug-report card was in neither guard: the pointer early return
    /// named `confirm` and `orient_prompt` only, so a press beside the card
    /// carved the model the report was about while the user described it.
    #[test]
    fn a_press_beside_the_bug_report_card_does_not_sculpt() {
        let mut app = app();
        update(&mut app, Message::BugReportOpened);
        assert!(app.bug_report.is_some(), "the dialog never opened");

        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let before = app.doc.active_volume().sample_world(front);
        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(
            app.doc.active_volume().sample_world(front),
            before,
            "a press reached the model behind the card"
        );
        assert!(app.drag.is_none(), "it started a stroke behind the card");
        assert!(!app.unsaved, "it dirtied a document nobody had touched");
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
        let bricks = app.doc.active_volume().brick_coords().count();
        update(&mut app, Message::NewSculpt);
        update(&mut app, Message::ConfirmAnswered(ConfirmChoice::Cancel));

        assert!(app.confirm.is_none());
        assert!(app.unsaved, "cancel cleared the unsaved marker");
        assert_eq!(
            app.doc.active_volume().brick_coords().count(),
            bricks,
            "cancel reset the sculpt anyway"
        );
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
        let sculpted = app.doc.active_volume().sample_world(probe);

        app.write_autosave();

        assert!(app.has_autosave(), "nothing was written");
        assert!(app.unsaved, "the autosave cleared the unsaved marker, so quitting would not ask");
        assert!(app.project_path.is_none(), "the autosave claimed to be the document's file");

        let mut recovered = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        recovered.autosave_file = app.autosave_file.clone();
        recovered.recover_autosave();

        assert_eq!(
            recovered.doc.active_volume().sample_world(probe),
            sculpted,
            "the field came back different"
        );
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

    /// **A failed save must leave the file it was replacing exactly as it
    /// was.** Until this increment the save opened the user's real file with
    /// `File::create`, which truncates it to zero before a byte of the new
    /// document is written -- so a failure part way through left the work in
    /// neither place.
    ///
    /// The failure is forced by putting a DIRECTORY where the temporary file
    /// wants to be, which is the cheapest way to make `File::create` fail
    /// without depending on permissions, a full disk or a mocked filesystem.
    #[test]
    fn a_save_that_fails_leaves_the_previous_file_and_the_crash_net_alone() {
        let directory = scratch("save-atomic");
        let path = directory.join("sculpt.brokkr");
        std::fs::write(&path, b"the previous save, which must survive").unwrap();

        let mut app = app_with_unsaved_work();
        app.autosave_file = Some(directory.join("autosave.brokkr"));
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));
        app.write_autosave();
        assert!(app.has_autosave(), "the fixture needs a crash net to protect");

        // Where `save_project` will try to put its temporary.
        std::fs::create_dir_all(path.with_extension("brokkr.tmp")).unwrap();

        app.save_project(&path);

        assert!(app.status.contains("could not"), "a failed save reported: {}", app.status);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"the previous save, which must survive",
            "a failed save destroyed the file it was replacing"
        );
        assert!(app.unsaved, "a failed save cleared the unsaved marker");
        assert!(app.project_path.is_none(), "a failed save adopted the path anyway");
        assert!(app.has_autosave(), "a failed save deleted the crash net");

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A save that works leaves the file and nothing else -- no temporary
    /// beside it for the user to find and wonder about.
    #[test]
    fn a_save_that_works_leaves_no_temporary_behind() {
        let directory = scratch("save-tidy");
        let path = directory.join("sculpt.brokkr");
        let mut app = app_with_unsaved_work();
        app.autosave_file = Some(directory.join("autosave.brokkr"));
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));

        app.save_project(&path);

        assert!(app.status.starts_with("saved"), "save reported: {}", app.status);
        assert!(path.is_file(), "the file was not written");
        assert!(
            !path.with_extension("brokkr.tmp").exists(),
            "the temporary the save writes through was left on disk"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// The save reads its own output back before it deletes the crash net.
    ///
    /// `clear_autosave` used to run on the writer's word alone: a write that
    /// reported success and produced a file this build will not open took the
    /// user's work, cleared the unsaved marker and deleted the one copy that
    /// could have recovered it. The check is the header and the node table
    /// only, which is a few hundred bytes whatever the sculpt weighs.
    ///
    /// Driven here by changing the document *after* the file is written, which
    /// is the same disagreement a partial write would produce and the only one
    /// a test can stage without a mocked filesystem.
    #[test]
    fn the_save_verification_notices_a_file_that_is_not_this_document() {
        let directory = scratch("save-verify");
        let path = directory.join("sculpt.brokkr");
        let mut app = app_with_unsaved_work();
        app.autosave_file = Some(directory.join("autosave.brokkr"));
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));

        app.save_project(&path);
        assert!(app.status.starts_with("saved"), "save reported: {}", app.status);
        assert!(app.verify_written(&path).is_ok(), "a file just written failed its own check");

        // One more body than the file on disk knows about.
        app.doc.add_body("Body 2", brokkr_core::Volume::new(app.doc.voxel_size()));
        let problem = app.verify_written(&path).expect_err("a stale file passed the check");
        assert!(problem.contains("2"), "the mismatch did not say what it found: {problem}");

        std::fs::remove_dir_all(&directory).ok();
    }

    /// **`save_project` really returns on that verification** -- the test above
    /// proves the check works, not that anything calls it.
    ///
    /// That distinction was measured, not assumed: deleting the whole
    /// `if let Err(problem) = self.verify_written(&temporary)` gate from
    /// `save_project` left all 279 tests in this crate green. A repair that is
    /// real, correct, unit-tested and never called is a shape this project has
    /// shipped before, and the gate is the one standing between an unreadable
    /// file and `clear_autosave`.
    ///
    /// Driving it needs a temporary that swallows every write and hands back
    /// something else on the read, and `/dev/null` is exactly that: the save
    /// opens the symlink, `project::write` reports success at every step, and
    /// the read-back finds an empty file. Nothing is written through the link
    /// that survives, and the cleanup unlinks the LINK -- `remove_file` does
    /// not follow one -- so `/dev/null` itself is never touched.
    ///
    /// Rejected first: making the document disagree with the file, the trick
    /// the test above uses. It cannot work here, because `save_project` writes
    /// and verifies against the same `self.doc` within one call, so the two can
    /// never disagree without an injection point that does not exist.
    #[test]
    fn a_save_that_cannot_read_its_own_output_back_keeps_the_previous_file() {
        let directory = scratch("save-unreadable");
        let path = directory.join("sculpt.brokkr");
        std::fs::write(&path, b"the previous save, which must survive").unwrap();

        let mut app = app_with_unsaved_work();
        app.autosave_file = Some(directory.join("autosave.brokkr"));
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));
        app.write_autosave();
        assert!(app.has_autosave(), "the fixture needs a crash net to protect");

        // Where `save_project` will put its temporary, pointed somewhere that
        // accepts the write and gives nothing back.
        let temporary = path.with_extension("brokkr.tmp");
        std::os::unix::fs::symlink("/dev/null", &temporary).unwrap();

        app.save_project(&path);

        assert!(
            app.status.contains("could not write"),
            "a file that read back empty reported: {}",
            app.status
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"the previous save, which must survive",
            "an unverifiable save replaced the file it could not prove it had written"
        );
        assert!(app.unsaved, "an unverifiable save cleared the unsaved marker");
        assert!(app.project_path.is_none(), "an unverifiable save adopted the path anyway");
        assert!(
            app.has_autosave(),
            "an unverifiable save deleted the crash net, which is the whole failure the \
             verification exists to stop"
        );
        assert!(
            std::fs::symlink_metadata(&temporary).is_err(),
            "the temporary was left on disk after the verification refused it"
        );
        assert!(
            std::path::Path::new("/dev/null").exists(),
            "the cleanup followed the symlink instead of unlinking it"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// The last failure arm, and the only one reached with a good file in hand:
    /// the write worked, the read-back agreed, and the rename still failed.
    ///
    /// It is here for two reasons. It is the one arm that runs *after* the
    /// verification, so reaching it at all proves execution got past that gate
    /// rather than around it. And it is the arm where the temporary holds the
    /// user's only new copy, so leaving it behind, or clearing `unsaved`
    /// because the bytes did reach a disk somewhere, would both be wrong.
    ///
    /// The failure is forced by putting a DIRECTORY at the TARGET, which makes
    /// `rename` fail with `EISDIR` while leaving every step before it -- create,
    /// write, verify -- entirely successful.
    #[test]
    fn a_save_that_cannot_replace_the_target_keeps_the_crash_net() {
        let directory = scratch("save-rename");
        let path = directory.join("sculpt.brokkr");
        std::fs::create_dir_all(&path).unwrap();

        let mut app = app_with_unsaved_work();
        app.autosave_file = Some(directory.join("autosave.brokkr"));
        app.recent = crate::recent::Recent::load_from(Some(directory.join("recent")));
        app.write_autosave();
        assert!(app.has_autosave(), "the fixture needs a crash net to protect");

        app.save_project(&path);

        assert!(
            app.status.contains("could not replace"),
            "a failed rename reported: {}",
            app.status
        );
        assert!(app.unsaved, "a failed rename cleared the unsaved marker");
        assert!(app.project_path.is_none(), "a failed rename adopted the path anyway");
        assert!(app.has_autosave(), "a failed rename deleted the crash net");
        assert!(
            !path.with_extension("brokkr.tmp").exists(),
            "a failed rename left the temporary behind, where the user would find a file \
             holding work the application says is unsaved"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    ///
    /// One shared number made Move look broken: for Move, strength is the
    /// fraction of the drag the surface follows, so Draw's 0.15 meant the form
    /// crawled at a seventh of the pointer.
    #[test]
    fn each_brush_remembers_its_own_strength() {
        let mut app = app();
        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        assert!((app.brush.strength - 0.15).abs() < 1.0e-6, "draw: {}", app.brush.strength);

        update(&mut app, Message::BrushKindChanged(BrushKind::Move));
        assert!(
            app.brush.strength > 0.9,
            "a grab tool at {} would follow a seventh of the drag",
            app.brush.strength
        );

        // A deliberate change sticks, and survives a round trip through another
        // brush.
        update(&mut app, Message::BrushStrengthChanged(0.5));
        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        assert!((app.brush.strength - 0.15).abs() < 1.0e-6, "draw was overwritten");
        update(&mut app, Message::BrushKindChanged(BrushKind::Move));
        assert!((app.brush.strength - 0.5).abs() < 1.0e-6, "move forgot its setting");
    }

    /// A Move drag must follow the pointer across the SCREEN, not crawl around
    /// the form.
    ///
    /// Raycasting the pointer onto the surface gives a target that slides along
    /// the curve, so the vector from the grab point stays short and keeps
    /// turning. Dragging past the silhouette is the case that separates the two:
    /// there is no surface under the cursor at all there, and a raycast version
    /// simply stops.
    #[test]
    fn a_move_drag_keeps_working_past_the_edge_of_the_model() {
        let mut app = app();
        app.camera.yaw = 0.0;
        app.camera.pitch = 0.0;
        app.publish_camera();
        update(&mut app, Message::BrushKindChanged(BrushKind::Move));
        update(&mut app, Message::BrushRadiusChanged(10.0));

        let probe = Vec3::new(0.0, 0.0, 30.0);
        let before = app.doc.active_volume().sample_world(probe);

        press(&mut app, centre_of_viewport());
        // Well past the silhouette of the ball.
        for step in 1..=40 {
            app.on_pointer(PointerEvent::Moved {
                position: centre_of_viewport() + Vector::new(step as f32 * 12.0, 0.0),
                size: SIZE,
            });
        }
        release(&mut app);

        assert_ne!(
            app.doc.active_volume().sample_world(probe),
            before,
            "the drag stopped at the silhouette, so it is still raycasting the surface"
        );
        assert!(app.history_stats.undo_entries > 0, "a long drag recorded no undo entry");
    }

    /// Move through the APPLICATION, not through `Brush::apply`.
    ///
    /// Every existing Move test builds a `Brush` and a `Stamp` by hand, which
    /// tests the arithmetic and says nothing about whether a drag in the
    /// viewport ever reaches it.
    #[test]
    fn dragging_with_move_selected_actually_changes_the_model() {
        let mut app = app();
        update(&mut app, Message::BrushKindChanged(BrushKind::Move));
        update(&mut app, Message::BrushStrengthChanged(1.0));
        update(&mut app, Message::BrushRadiusChanged(10.0));

        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        let probe = app
            .surface_under(Vec2::new(SIZE.x / 2.0, SIZE.y / 2.0))
            .expect("the centre hits the ball");
        let before = app.doc.active_volume().sample_world(probe);

        press(&mut app, centre_of_viewport());
        for step in 1..=12 {
            app.on_pointer(PointerEvent::Moved {
                position: centre_of_viewport() + Vector::new(step as f32 * 4.0, 0.0),
                size: SIZE,
            });
        }
        release(&mut app);

        assert!(
            app.history_stats.undo_entries > 0,
            "a Move drag recorded no undo entry, so it did nothing at all"
        );
        assert_ne!(
            app.doc.active_volume().sample_world(probe),
            before,
            "a Move drag across the model left the field untouched"
        );
    }

    /// Reach and cost together, since Move trades one against the other.
    ///
    /// Printed rather than asserted: these are the numbers the reach cap was
    /// chosen against, and the point is that a future change can re-run them.
    /// `the_surface_follows_the_pointer_through_the_application` is the one
    /// that fails if the reach collapses again.
    #[test]
    fn measure_move_reach_and_cost() {
        for (radius, strength) in
            [(3.0f32, 0.15f32), (3.0, 1.0), (10.0, 0.15), (10.0, 1.0), (20.0, 0.15), (20.0, 1.0)]
        {
            let mut app = app();
            aim_at_the_front(&mut app);
            update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
            update(&mut app, Message::BrushRadiusChanged(radius.min(6.0)));
            update(&mut app, Message::BrushStrengthChanged(0.8));
            for _ in 0..6 {
                press(&mut app, centre_of_viewport());
                release(&mut app);
            }

            update(&mut app, Message::BrushKindChanged(BrushKind::Move));
            update(&mut app, Message::BrushRadiusChanged(radius));
            update(&mut app, Message::BrushStrengthChanged(strength));

            let before = bump_x(&app);
            app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
            let from = app.surface_under(pixel(centre_of_viewport())).expect("the centre hits");

            press(&mut app, centre_of_viewport());
            let mut worst = 0.0f32;
            let mut total_ms = 0.0f32;
            let mut to = from;
            for step in 1..=30 {
                let at = centre_of_viewport() + Vector::new(step as f32 * 5.0, 0.0);
                app.on_pointer(PointerEvent::Moved { position: at, size: SIZE });
                worst = worst.max(app.perf.edit_ms);
                total_ms += app.perf.edit_ms;
                if let Some(point) = app.surface_under(pixel(at)) {
                    to = point;
                }
            }
            release(&mut app);

            let moved = bump_x(&app) - before;
            eprintln!(
                "radius {radius:>5.1} strength {strength:>4.2}: pointer travelled \
                 {:>5.1} mm, surface followed {moved:>+6.2} mm (cap {:>5.2} mm), \
                 worst event {worst:>6.2} ms, whole drag {total_ms:>7.2} ms",
                from.distance(to),
                Brush { kind: BrushKind::Move, radius, ..app.brush }.max_drag(),
            );
        }
    }

    /// Where the bump raised at the front of the ball sits along X, weighted by
    /// how much material is standing proud of the sphere at each slice.
    ///
    /// A drag has to carry this along with it, and it is measured rather than
    /// assumed to have stayed put -- which is the whole difference between
    /// "something changed" and "the surface followed the pointer".
    fn bump_x(app: &Brokkr) -> f32 {
        let mut weighted = 0.0;
        let mut total = 0.0;
        for step in -120..=120 {
            let x = step as f32 * 0.25;
            // Where the surface crosses along +Z at this slice.
            let mut z = 60.0f32;
            while z > 0.0 {
                if app.doc.active_volume().sample_world(Vec3::new(x, 0.0, z)) < 0.0 {
                    break;
                }
                z -= 0.05;
            }
            let raised = (z - (MODEL_RADIUS_MM * MODEL_RADIUS_MM - x * x).max(0.0).sqrt()).max(0.0);
            weighted += x * raised;
            total += raised;
        }
        assert!(total > 0.0, "there is no bump to measure");
        weighted / total
    }

    fn pixel(at: Vector) -> Vec2 {
        Vec2::new(at.x, at.y)
    }

    /// Look straight down -Z, so a horizontal drag on screen is a drag along
    /// world X and the measurements above are in a frame anyone can check.
    fn aim_at_the_front(app: &mut Brokkr) {
        app.camera.yaw = 0.0;
        app.camera.pitch = 0.0;
        app.publish_camera();
    }

    /// The reach the old incremental Move could not deliver.
    ///
    /// It moved the surface **0.02 mm** for a full viewport drag at the default
    /// radius, which is why it was reported as doing nothing at all. This
    /// asserts a real magnitude rather than "something changed", because
    /// "something changed" is exactly what the broken version also did.
    #[test]
    fn the_surface_follows_the_pointer_through_the_application() {
        let mut app = app();
        aim_at_the_front(&mut app);
        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        update(&mut app, Message::BrushRadiusChanged(6.0));
        update(&mut app, Message::BrushStrengthChanged(0.8));
        for _ in 0..6 {
            press(&mut app, centre_of_viewport());
            release(&mut app);
        }

        update(&mut app, Message::BrushKindChanged(BrushKind::Move));
        update(&mut app, Message::BrushRadiusChanged(10.0));
        update(&mut app, Message::BrushStrengthChanged(1.0));

        let before = bump_x(&app);
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        let from = app.surface_under(pixel(centre_of_viewport())).expect("the centre hits");

        press(&mut app, centre_of_viewport());
        let mut to = from;
        for step in 1..=10 {
            let at = centre_of_viewport() + Vector::new(step as f32 * 5.0, 0.0);
            app.on_pointer(PointerEvent::Moved { position: at, size: SIZE });
            if let Some(point) = app.surface_under(pixel(at)) {
                to = point;
            }
        }
        release(&mut app);

        let pointer = from.distance(to);
        let moved = bump_x(&app) - before;
        assert!(pointer > 3.0, "the test did not drag far enough to mean anything: {pointer} mm");
        assert!(
            moved > pointer * 0.4,
            "a {pointer:.1} mm drag moved the surface {moved:.2} mm, which is the order of \
             magnitude the incremental version failed at"
        );
    }

    /// The property locking the field buys, seen from the application.
    #[test]
    fn a_drag_out_and_back_through_the_application_returns_the_form() {
        let mut app = app();
        aim_at_the_front(&mut app);
        update(&mut app, Message::BrushKindChanged(BrushKind::Draw));
        update(&mut app, Message::BrushRadiusChanged(6.0));
        update(&mut app, Message::BrushStrengthChanged(0.8));
        for _ in 0..6 {
            press(&mut app, centre_of_viewport());
            release(&mut app);
        }

        let probes: Vec<Vec3> = (-40..=40)
            .map(|step| Vec3::new(step as f32 * 0.5, 0.0, MODEL_RADIUS_MM - 2.0))
            .collect();
        let before: Vec<f32> =
            probes.iter().map(|p| app.doc.active_volume().sample_world(*p)).collect();

        update(&mut app, Message::BrushKindChanged(BrushKind::Move));
        update(&mut app, Message::BrushRadiusChanged(10.0));
        update(&mut app, Message::BrushStrengthChanged(1.0));

        press(&mut app, centre_of_viewport());
        let mut path: Vec<f32> = (1..=8).map(|step| step as f32 * 4.0).collect();
        path.extend((0..=8).rev().map(|step| step as f32 * 4.0));
        for offset in path {
            app.on_pointer(PointerEvent::Moved {
                position: centre_of_viewport() + Vector::new(offset, 0.0),
                size: SIZE,
            });
        }
        release(&mut app);

        let worst = probes
            .iter()
            .zip(&before)
            .map(|(probe, was)| (app.doc.active_volume().sample_world(*probe) - was).abs())
            .fold(0.0f32, f32::max);
        // Looser than the core test's thousandth, and for two reasons that are
        // the application's rather than the algorithm's. The world point comes
        // from a raycast against the surface as it currently stands, so coming
        // back to the same pixel is not quite coming back to the same point;
        // and a drag that has shrunk to within a quarter voxel of where it
        // already was is skipped rather than redone. Both are bounded by a
        // fraction of a voxel, which is what this asserts -- the values are in
        // voxels, so a twentieth here is a hundredth of a voxel.
        assert!(worst < 0.05, "a drag out and back left the surface {worst} different");
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

        let above = app.doc.active_volume().sample_world(Vec3::new(0.0, 12.0, 0.0));
        let below = app.doc.active_volume().sample_world(Vec3::new(0.0, -12.0, 0.0));
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
        let before = app.doc.active_volume().brick_coords().count();

        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.doc.active_volume().brick_coords().count(), before, "a click cut the model");
        assert!(!app.unsaved, "a click that cut nothing marked the document unsaved");
        assert!(app.status.contains("cancelled"), "reported: {}", app.status);
    }

    #[test]
    fn a_cut_is_undoable_through_the_application() {
        let mut app = app();
        let probe = Vec3::new(0.0, 12.0, 0.0);
        let before = app.doc.active_volume().sample_world(probe);

        update(&mut app, Message::CutToggled);
        press(&mut app, Vector::new(SIZE.x * 0.1, SIZE.y / 2.0));
        app.on_pointer(PointerEvent::Moved {
            position: Vector::new(SIZE.x * 0.9, SIZE.y / 2.0),
            size: SIZE,
        });
        release(&mut app);
        assert_ne!(
            app.doc.active_volume().sample_world(probe),
            before,
            "the cut did nothing to undo"
        );

        app.undo();
        assert_eq!(
            app.doc.active_volume().sample_world(probe),
            before,
            "undo did not restore the cut"
        );
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
        let options = brokkr_core::voxelise::VoxeliseOptions::at(app.doc.voxel_size());
        let (volume, voxel_report) =
            brokkr_core::voxelise::voxelise(&imported, &options).expect("it should voxelise");
        app.adopt_import(crate::message::Imported {
            volume,
            report: voxel_report,
            source: path.clone(),
            elapsed_ms: 0.0,
            resting_up: brokkr_core::resting_up(&imported.positions),
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

        let (_, after) = app.doc.active_volume().export_mesh();
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
        let before = app.doc.active_volume().sample_world(Vec3::ZERO);

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
            app.doc.active_volume().sample_world(Vec3::ZERO),
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

    /// **A saved multi-body document is still work, and import throws all of it
    /// away.**
    ///
    /// The gate consulted `unsaved` alone, so with a saved five-body project
    /// open, File > Import replaced every body with the mesh and asked nothing:
    /// no dialog, no prompt, no status line. The file survives on disk, which
    /// is why this is a prompt rather than a refusal.
    #[test]
    fn importing_over_a_saved_document_of_several_bodies_asks_first() {
        let mut app = app();
        assert!(!app.unsaved, "the fixture must be clean or this measures the old gate");

        // One body and saved: nothing to lose that Open would not also cost, so
        // no prompt. This half is what says the guard did not simply become
        // "always ask".
        update(&mut app, Message::ImportRequested);
        assert_eq!(app.confirm, None, "a single-body import should not be interrupted");

        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(60.0, 0.0, 0.0), 10.0);
        app.doc.add_body("Body 2", second);
        app.publish_visibility();
        assert!(!app.unsaved, "adding the fixture body must not be what raises the prompt");

        update(&mut app, Message::ImportRequested);
        assert_eq!(
            app.confirm,
            Some(PendingAction::Import),
            "importing discarded a second body without asking"
        );
    }

    /// Two bodies, one in front of the other, both drawn.
    ///
    /// The second ball is placed on the line from the origin to the eye, so a
    /// press at the centre of the viewport meets it before the starting sphere.
    /// `publish_visibility` is called by hand because the fixture builds the
    /// document directly rather than through a message, and the pick reads the
    /// mask the application publishes.
    fn two_bodies_in_a_line(app: &mut Brokkr) -> (NodeId, NodeId) {
        let front = app.camera.eye().normalize() * 55.0;
        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(front, 8.0);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        app.publish_visibility();
        app.remesh_dirty();
        (app.doc.active(), other)
    }

    /// **The press ordering, which is the one that loses work if it is wrong.**
    ///
    /// Recording used to open in the `Pressed` arm before anything had been
    /// raycast. With two bodies that opens it on the wrong one, and
    /// `record_for_undo` does nothing at all when the volume it is called on
    /// has no recorder — so the carve lands, no entry is pushed, `unsaved` stays
    /// false, and quitting does not even raise the discard prompt.
    ///
    /// The rule that falls out is the one Photoshop's layer list has: a press
    /// on something that is not selected selects it, and the press after that
    /// works on it.
    #[test]
    fn a_press_over_another_body_selects_it_and_carves_nothing_until_the_next_press() {
        let mut app = app();
        let (first, other) = two_bodies_in_a_line(&mut app);

        // The surfaces the camera meets first: the starting ball at its own
        // radius, and the second ball at its centre plus its radius, because
        // the ray comes inwards from the eye.
        let front_of_first = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let front_of_other = app.camera.eye().normalize() * 63.0;
        let first_before = app.doc.volume(first).expect("a live body").sample_world(front_of_first);
        let other_before = app.doc.volume(other).expect("a live body").sample_world(front_of_other);

        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.doc.active(), other, "the press did not choose the body it landed on");
        assert!(!app.history.can_undo(), "a press that only selected recorded a stroke");
        assert!(!app.unsaved, "a press that only selected dirtied the document");
        assert_eq!(
            app.doc.volume(first).expect("a live body").sample_world(front_of_first),
            first_before,
            "the press carved the body it was leaving"
        );
        assert_eq!(
            app.doc.volume(other).expect("a live body").sample_world(front_of_other),
            other_before,
            "the press that selected also carved"
        );

        // And now it sculpts, on the body it chose.
        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.doc.active(), other, "the second press moved the selection again");
        assert!(app.history.can_undo(), "the second press recorded no undo entry");
        assert!(app.unsaved, "the second press left the document looking saved");
        assert_ne!(
            app.doc.volume(other).expect("a live body").sample_world(front_of_other),
            other_before,
            "the second press did not carve the selected body"
        );
        assert_eq!(
            app.doc.volume(first).expect("a live body").sample_world(front_of_first),
            first_before,
            "the stroke landed on the body that was NOT selected"
        );
    }

    /// Hiding is a draw-time skip, so the depth buffer where a hidden body sits
    /// is empty. A press there must go through it: carving something invisible
    /// would set `unsaved`, push an entry, pay a remesh and an upload, and
    /// change not one pixel.
    #[test]
    fn a_press_over_a_hidden_body_carves_nothing_and_reaches_what_is_behind_it() {
        let mut app = app();
        let (first, other) = two_bodies_in_a_line(&mut app);

        let mut meta = app.doc.meta(other).expect("the second body");
        meta.visible = false;
        app.doc.set_meta(&meta);
        app.publish_visibility();

        let front_of_first = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let front_of_other = app.camera.eye().normalize() * 63.0;
        let other_before = app.doc.volume(other).expect("a live body").sample_world(front_of_other);
        let first_before = app.doc.volume(first).expect("a live body").sample_world(front_of_first);

        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(app.doc.active(), first, "a hidden body was selected by a press over it");
        assert_eq!(
            app.doc.volume(other).expect("a live body").sample_world(front_of_other),
            other_before,
            "the press carved a body that is not on screen"
        );
        assert_ne!(
            app.doc.volume(first).expect("a live body").sample_world(front_of_first),
            first_before,
            "the press stopped at a body nothing is drawing"
        );
    }

    /// The same rule, for the case the pick cannot answer on its own.
    ///
    /// [`Document::pick`] masks on visibility, so it refuses a hidden body --
    /// but a stroke takes its surface from `pick_body`, which asks one named
    /// body and never consults the eye. So a press while the ACTIVE body is
    /// hidden picked nothing, fell through to a sculpt, and carved the
    /// invisible body: an undo entry pushed, `unsaved` set, a remesh and an
    /// upload paid for, and not one pixel changed. Reachable today only by
    /// opening a file whose active row is hidden; a one-click bug the moment
    /// the panel ships an eye.
    #[test]
    fn a_press_while_the_active_body_is_hidden_carves_nothing_and_says_which_body() {
        let mut app = app();
        let active = app.doc.active();
        let mut meta = app.doc.meta(active).expect("the starting body");
        meta.visible = false;
        app.doc.set_meta(&meta);
        app.publish_visibility();

        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let before = app.doc.active_volume().sample_world(front);

        press(&mut app, centre_of_viewport());
        app.on_pointer(PointerEvent::Moved {
            position: centre_of_viewport() + Vector::new(20.0, 0.0),
            size: SIZE,
        });
        release(&mut app);

        assert_eq!(
            app.doc.active_volume().sample_world(front),
            before,
            "the press carved the body that is not on screen"
        );
        assert!(!app.history.can_undo(), "a press that carved nothing recorded a stroke");
        assert!(!app.unsaved, "a press that carved nothing dirtied the document");
        assert!(
            app.status.contains("hidden") && app.status.contains(Document::FIRST_BODY_NAME),
            "nothing said why the press did nothing: {:?}",
            app.status
        );

        // Shown again, the very same press works: the refusal is about the
        // eye and nothing else.
        meta.visible = true;
        app.doc.set_meta(&meta);
        app.publish_visibility();
        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert_ne!(
            app.doc.active_volume().sample_world(front),
            before,
            "the press stopped working on a body that is drawn"
        );
    }

    /// **A live stroke owns the ring.**
    ///
    /// The `Moved` arm carved the active body and then picked across every
    /// drawn body anyway, so a stroke dragged over a second body moved
    /// `hover_body` onto that second body -- and the ring is built from the
    /// hovered body's field and coloured `CursorMood::Selecting` whenever the
    /// hovered body is not the active one. The cursor said "a press here would
    /// select", on the wrong surface, during a press that was carving
    /// something else.
    #[test]
    fn the_ring_stays_on_the_body_a_live_stroke_is_carving() {
        let mut app = app();
        let (first, other) = two_bodies_in_a_line(&mut app);

        // The control. With no button down the centre of the viewport hovers
        // the SECOND body, which is the whole reason a stroke dragged there
        // could move the ring; without this the test would also pass on a
        // build where the hover never reaches that body at all.
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        assert_eq!(app.hover_body, Some(other), "the fixture never hovers the second body");

        // Off to the side, where only the first body is, so the press starts a
        // stroke instead of choosing the other body.
        let beside = Vector::new(610.0, 300.0);
        press(&mut app, beside);
        assert_eq!(app.doc.active(), first, "the press beside the second ball chose a body");
        assert!(
            matches!(app.drag.map(|drag| drag.kind), Some(DragKind::Sculpt(_))),
            "the press beside the second ball did not start a stroke: {:?}",
            app.drag.map(|drag| drag.kind)
        );

        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        assert_eq!(
            app.hover_body,
            Some(first),
            "mid-stroke the ring hopped onto a body the stroke is not carving"
        );
        assert!(app.hover.is_some(), "the ring vanished from the body being carved");

        // And it comes back the moment the stroke ends.
        release(&mut app);
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        assert_eq!(app.hover_body, Some(other), "the hover never returned to the pick");
    }

    /// **The mitigation for the mirror centre being the lattice origin.**
    ///
    /// With the centre pinned there, mirroring a body that sits wholly to one
    /// side of a plane writes its twin out in empty space: a free-floating
    /// shell that exports and that no slicer can print. The axis is refused
    /// rather than enabled-and-ignored, so there is one answer to "is X on".
    #[test]
    fn a_mirror_the_active_body_does_not_straddle_is_refused_with_a_reason() {
        let mut app = app();

        // The starting ball is centred on the origin, so every plane crosses it
        // and every axis is allowed. This half is what stops the refusal from
        // being "always refuse".
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::X));
        assert!(app.symmetry.axis(MirrorAxis::X), "a body on the origin was refused its mirror");
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::X));
        assert!(!app.symmetry.axis(MirrorAxis::X));

        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(80.0, 0.0, 0.0), 10.0);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        app.publish_visibility();
        app.remesh_dirty();
        app.select_body(other);

        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::X));
        assert!(!app.symmetry.axis(MirrorAxis::X), "a mirror was enabled that misses the body");
        assert!(
            app.status.contains('X') && app.status.contains("Body 2"),
            "the refusal said nothing useful: {:?}",
            app.status
        );

        // The other two planes DO cross it, and must not be refused along with
        // the one that does not.
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::Y));
        assert!(app.symmetry.axis(MirrorAxis::Y), "a plane the body straddles was refused");
    }

    /// **A refusal wired into one event site is not a refusal.**
    ///
    /// The symmetry strip is not the only way to turn X mirroring on: a
    /// SpaceMouse button bound to `ToggleSymmetry` did it with a bare
    /// `Symmetry::toggled`, so a puck user on a body at x = +80 got every
    /// stroke's twin written into empty space, with the strip highlight and
    /// the drawn plane both agreeing that nothing was wrong. This drives the
    /// application's real button path rather than calling the gate, because
    /// the gate was never what was broken.
    #[test]
    fn the_spacemouse_symmetry_button_is_refused_the_mirror_the_strip_is_refused() {
        let mut app = app();
        let first = app.doc.active();
        app.spacemouse.config.buttons[0] = ButtonAction::ToggleSymmetry;

        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(80.0, 0.0, 0.0), 10.0);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        app.publish_visibility();
        app.remesh_dirty();
        app.select_body(other);

        app.spacemouse.simulate_press(0);
        app.drive_from_spacemouse(16.0);

        assert!(
            !app.symmetry.axis(MirrorAxis::X),
            "the puck turned on a mirror plane the body sits entirely to one side of"
        );
        assert!(
            app.status.contains('X') && app.status.contains("Body 2"),
            "the puck's refusal said nothing: {:?}",
            app.status
        );

        // And it still works where the mirror is legitimate, so the fix is a
        // gate and not a disconnected wire.
        app.select_body(first);
        app.spacemouse.simulate_press(0);
        app.drive_from_spacemouse(16.0);
        assert!(
            app.symmetry.axis(MirrorAxis::X),
            "the puck stopped turning on a mirror the body does straddle"
        );
    }

    /// A view out of a file is a third way for a mirror to come on, and a
    /// restored one carves twins into empty space exactly as an enabled one
    /// does -- the file may have been written by a build without the refusal,
    /// or with a different body active.
    #[test]
    fn a_mirror_restored_from_a_view_the_body_does_not_straddle_is_turned_off() {
        let mut app = app();
        let mut off_centre = Volume::new(app.doc.voxel_size());
        off_centre.seed_sphere(Vec3::new(80.0, 0.0, 0.0), 10.0);
        off_centre.mark_everything_dirty();
        app.doc = Document::from_volume(off_centre);
        app.publish_visibility();
        app.remesh_dirty();

        let mut view = app.current_view();
        view.mirror = [true, true, true];
        app.apply_view(&view);

        assert!(!app.symmetry.axis(MirrorAxis::X), "a file turned on a mirror the body misses");
        assert!(
            app.symmetry.axis(MirrorAxis::Y) && app.symmetry.axis(MirrorAxis::Z),
            "the two planes the body does straddle were dropped as well"
        );
    }

    /// The axis and the centre have the same scope, so choosing a body the
    /// enabled plane misses has to resolve the pairing one way or the other.
    /// It turns the plane off and says so, rather than leaving the strip
    /// claiming a mirror the sculpt is not applying.
    #[test]
    fn selecting_a_body_clear_of_an_enabled_mirror_plane_turns_that_plane_off() {
        let mut app = app();
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::X));
        assert!(app.symmetry.axis(MirrorAxis::X));

        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(80.0, 0.0, 0.0), 10.0);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        app.publish_visibility();
        app.remesh_dirty();

        app.select_body(other);
        assert!(
            !app.symmetry.axis(MirrorAxis::X),
            "the mirror stayed on over a body it does not cross"
        );
        assert!(app.status.contains('X'), "nothing said the mirror went off: {:?}", app.status);
    }

    /// **A quarter turn turns the whole document.**
    ///
    /// Bodies share one lattice and have no transform, so their arrangement is
    /// the only positional state there is. `orient` turned `active_volume()`
    /// alone, which with two bodies scatters that arrangement — and makes its
    /// own status line, "turn it back the same way to undo", untrue for every
    /// body that stayed put. The bit-exactness of four turns is measured in
    /// `brokkr-core`; this is the wiring.
    #[test]
    fn turning_the_model_turns_every_body_and_not_just_the_active_one() {
        let mut app = app();
        // Off the axis the turn is about: a ball sitting ON that axis comes
        // back to the same box whether the turn reached it or not, and this
        // test would then pass with `orient` still turning one body.
        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(0.0, 50.0, 0.0), 8.0);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        app.publish_visibility();
        app.remesh_dirty();

        let before =
            app.doc.volume(other).expect("a live body").world_bounds().expect("a seeded ball");
        app.orient(brokkr_core::AxisRotation::taking(
            brokkr_core::Facing::Front,
            brokkr_core::Facing::Up,
        ));
        let after =
            app.doc.volume(other).expect("a live body").world_bounds().expect("still a ball");

        assert_ne!(before, after, "the inactive body did not turn with the model");
        assert!(
            ((after.1 - after.0).length() - (before.1 - before.0).length()).abs() < 1.0e-3,
            "the inactive body changed size rather than orientation: {before:?} to {after:?}"
        );
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

        // Released, dirty and overdue -- but the pointer only just stopped, so
        // the write still waits for the pause.
        release(&mut app);
        app.maybe_autosave();
        assert!(
            !app.has_autosave(),
            "it autosaved the instant the pointer stopped, which is where a 400 ms write \
             lands as a visible hitch"
        );

        // Once the pointer has been still long enough, it writes.
        app.last_activity = Instant::now() - Duration::from_secs(60);
        app.maybe_autosave();
        assert!(app.has_autosave(), "an overdue autosave never happened after a pause");

        std::fs::remove_dir_all(&directory).ok();
    }
    #[test]
    fn a_bad_file_reports_and_changes_nothing() {
        let directory = std::env::temp_dir().join(format!("brokkr-bad-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("not-a-sculpt.brokkr");
        std::fs::write(&path, b"this is not a sculpt").unwrap();

        let mut app = app();
        let before = app.doc.active_volume().brick_coords().count();
        app.open_project(&path);

        assert!(app.status.contains("not a BrokkrSculpt file"), "reported: {}", app.status);
        assert_eq!(
            app.doc.active_volume().brick_coords().count(),
            before,
            "a failed open lost the model"
        );
        assert!(app.project_path.is_none(), "a failed open claimed the file");

        app.open_project(&directory.join("does-not-exist.brokkr"));
        assert!(app.status.contains("could not open"), "reported: {}", app.status);
        assert_eq!(app.doc.active_volume().brick_coords().count(), before);

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
        dynamic.doc.active_volume_mut().seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM * 2.0);
        dynamic.doc.active_volume_mut().mark_everything_dirty();
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

    /// The same ball, moved bodily away from the origin, must report the same
    /// size.
    ///
    /// This is what `content_radius` replaced `bounding_radius` for. The old
    /// measure was taken from the WORLD ORIGIN, so a model 128 mm out reported
    /// roughly 2.6 times its own radius and the Dynamic brush went with it --
    /// and nothing in the interface would have said why the brush had grown.
    ///
    /// 128 mm rather than a round 100 because at the 0.25 mm voxel that is
    /// exactly sixteen bricks, so the sphere's brick footprint is an exact
    /// translation of the one at the origin and the two radii are identical to
    /// the last bit. That is the shared-lattice property this whole design
    /// rests on, exercised from the application's side.
    #[test]
    fn a_body_far_from_the_origin_does_not_resize_the_dynamic_brush() {
        let mut app = app();
        update(&mut app, Message::DynamicRadiusToggled(true));
        let radius = app.brush.radius;
        let at_origin = app.model_radius;

        let mut moved = brokkr_core::Volume::new(app.doc.voxel_size());
        moved.seed_sphere(Vec3::new(128.0, 0.0, 0.0), MODEL_RADIUS_MM);
        moved.mark_everything_dirty();
        app.doc.replace_active_volume(moved);
        app.remesh_dirty();

        assert_eq!(
            app.model_radius, at_origin,
            "the same ball measured {} at the origin and {} at 128 mm",
            at_origin, app.model_radius
        );
        assert_eq!(app.brush.radius, radius, "moving the model resized the brush");
    }

    /// Selecting a different body changes no geometry, so it must not resize
    /// the Dynamic brush -- not on the selection, and not on the next remesh
    /// either, which is where a naive implementation would catch up on it a
    /// frame later.
    ///
    /// The fixture is the one from the decision: a small body beside a large
    /// one. Under a plain `previous == self.model_radius` comparison this took
    /// a 3 mm brush to the 0.25 mm floor and then to 10 mm on the way back --
    /// and the press that resizes it is the same press that selects, so the
    /// next press would carve at a radius nobody chose.
    #[test]
    fn changing_the_active_body_does_not_resize_the_dynamic_brush() {
        let mut app = app();
        update(&mut app, Message::DynamicRadiusToggled(true));
        let radius = app.brush.radius;

        let mut rivet = brokkr_core::Volume::new(app.doc.voxel_size());
        rivet.seed_sphere(Vec3::new(128.0, 0.0, 0.0), 2.5);
        rivet.mark_everything_dirty();
        let second = app.doc.add_body("Body 2", rivet);
        app.remesh_dirty();
        assert_eq!(app.brush.radius, radius, "ADDING a body resized the brush");

        app.doc.set_active(second);
        // Something then dirties a brick -- a stroke, an undo, a resample,
        // anything -- and the remesh that follows measures the rivet where the
        // last measurement was of the ball. That comparison is the whole
        // hazard, so the test has to reach it rather than land on
        // `remesh_dirty`'s empty-set early return.
        app.doc.active_volume_mut().mark_everything_dirty();
        app.remesh_dirty();
        assert_eq!(app.brush.radius, radius, "selecting a smaller body resized the brush");
        assert!(
            app.model_radius < MODEL_RADIUS_MM,
            "the framing radius did not follow the selection: {}",
            app.model_radius
        );

        let first = app.doc.nodes()[0].id;
        app.doc.set_active(first);
        app.doc.active_volume_mut().mark_everything_dirty();
        app.remesh_dirty();
        assert_eq!(app.brush.radius, radius, "selecting back resized the brush");
    }

    /// **The plane cut crosses every VISIBLE body.**
    ///
    /// A cut is a line the user draws across what they can see, so it acts on
    /// what is drawn. `finish_cut` was single-volume before there was a
    /// document, and the rename that gave it one made it silently mean "the
    /// active body" -- which is exactly the kind of change that ships with
    /// nothing failing. This was written as the executable statement of what it
    /// owed and left `#[ignore]`d, alongside a `debug_assert!` at the top of
    /// `finish_cut` saying the same thing from the other side. Both were
    /// discharged by `Document::clip`, which brackets each visible body and
    /// records ONE `Entry` of N `Change::Bricks`.
    #[test]
    fn a_cut_crosses_every_visible_body() {
        let mut app = app();
        app.camera.yaw = 0.0;
        app.camera.pitch = 0.0;
        app.publish_camera();

        // A second ball overlapping the first, so one screen-space line crosses
        // both of them.
        let mut second = brokkr_core::Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(10.0, 0.0, 0.0), MODEL_RADIUS_MM * 0.5);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        app.remesh_dirty();

        let above = Vec3::new(10.0, 12.0, 0.0);
        let before = app.doc.volume(other).expect("a live body").sample_world(above);

        update(&mut app, Message::CutToggled);
        press(&mut app, Vector::new(SIZE.x * 0.1, SIZE.y / 2.0));
        app.on_pointer(PointerEvent::Moved {
            position: Vector::new(SIZE.x * 0.9, SIZE.y / 2.0),
            size: SIZE,
        });
        release(&mut app);

        assert_ne!(
            app.doc.volume(other).expect("a live body").sample_world(above),
            before,
            "the cut passed straight through the inactive body and left it whole"
        );
    }

    /// **Solo narrows the cut**, and that is decided rather than incidental.
    ///
    /// The instinct is that a view mode must never narrow an operation. It does
    /// not survive the case: eight bodies with three eye-hidden and four
    /// solo-hidden means a line drawn across one body cuts five, and **nothing
    /// on screen distinguishes an eye-hidden body from a solo-hidden one** --
    /// in both cases it is simply not there. The rule that survives is narrower
    /// and can be stated in one sentence: direct manipulation acts on what is
    /// drawn; the file and the export act on `saved_visibility`.
    #[test]
    fn solo_narrows_the_cut_to_the_body_it_is_showing() {
        let mut app = app();
        app.camera.yaw = 0.0;
        app.camera.pitch = 0.0;
        app.publish_camera();

        let mut second = brokkr_core::Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(10.0, 0.0, 0.0), MODEL_RADIUS_MM * 0.5);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        app.remesh_dirty();

        let first = app.doc.nodes()[0].id;
        update(&mut app, Message::SoloEntered(first));

        let above = Vec3::new(10.0, 12.0, 0.0);
        let spared = app.doc.volume(other).expect("a live body").sample_world(above);
        let over_the_soloed = Vec3::new(0.0, 12.0, 0.0);
        let cut_away = app.doc.volume(first).expect("a live body").sample_world(over_the_soloed);

        update(&mut app, Message::CutToggled);
        press(&mut app, Vector::new(SIZE.x * 0.1, SIZE.y / 2.0));
        app.on_pointer(PointerEvent::Moved {
            position: Vector::new(SIZE.x * 0.9, SIZE.y / 2.0),
            size: SIZE,
        });
        release(&mut app);

        assert_eq!(
            app.doc.volume(other).expect("a live body").sample_world(above),
            spared,
            "the cut went through a body solo was hiding"
        );
        // ...and the body solo IS showing really was cut, so this is not a test
        // of a cut that missed everything and passed for the wrong reason.
        assert_ne!(
            app.doc.volume(first).expect("a live body").sample_world(over_the_soloed),
            cut_away,
            "the cut missed the soloed body too: {}",
            app.status
        );
        assert!(
            app.status.contains("bricks") && !app.status.contains("bodies"),
            "the cut did not report exactly one body: {}",
            app.status
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
        let before_field = app.doc.active_volume().sample_world(probe);
        let before_yaw = app.camera.yaw;

        let (origin, size) = crate::navcube::corner_rect(Vec2::new(SIZE.x, SIZE.y));
        let middle = origin + size * 0.5;
        press(&mut app, Vector::new(middle.x, middle.y));
        release(&mut app);

        assert!(app.flight.is_some(), "clicking the cube did not start a move");
        assert_eq!(
            app.doc.active_volume().sample_world(probe),
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
        let before = app.doc.active_volume().sample_world(probe);

        press(&mut app, centre_of_viewport());
        release(&mut app);
        assert!(app.menu.is_none(), "a click elsewhere did not close the menu");
        assert_eq!(
            app.doc.active_volume().sample_world(probe),
            before,
            "dismissing the menu sculpted"
        );
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

    /// Undo has to walk the brush ring back onto the surface it restored.
    ///
    /// The ring is not a screen space circle drawn over the model: every one
    /// of its points is pushed onto the field by `cursor::onto_surface`, so
    /// the geometry the renderer is holding is only correct for the field it
    /// was built from. Undo changes the field without the pointer moving,
    /// which is the one way the two can come apart -- and before
    /// [`Brokkr::after_history_step`] refreshed the overlay, ctrl+Z left the
    /// ring standing on the surface the stroke had made, floating off the
    /// model until some later pointer motion happened to rebuild it.
    ///
    /// **Measured rather than argued.** With the default sphere and a
    /// full strength Draw at the centre of the viewport, the stale ring sits
    /// 0.49 voxels off the restored field and a refreshed one sits 0.0016
    /// voxels off it, against the 0.1 voxel tolerance `onto_surface` stops
    /// walking at -- five times the tolerance apart in one direction and
    /// sixty times inside it in the other. The first assertion is there
    /// because a fixture whose stroke did not actually move the surface under
    /// the ring would pass the second one while testing nothing.
    ///
    /// Deleting the `refresh_overlay` call from `after_history_step` fails
    /// this test and, as of writing, nothing else in the workspace.
    #[test]
    fn undoing_a_stroke_walks_the_brush_ring_back_onto_the_surface_it_restores() {
        // What `cursor::onto_surface` treats as "on the surface", in voxels.
        const ON_SURFACE: f32 = 0.1;

        let ring = |app: &Brokkr| -> Vec<Vec3> {
            app.shared
                .overlay_snapshot()
                .lines
                .iter()
                .map(|vertex| Vec3::from_array(vertex.position))
                .collect()
        };
        let worst_off_surface = |app: &Brokkr, points: &[Vec3]| -> f32 {
            points
                .iter()
                .map(|point| app.doc.active_volume().sample_world(*point).abs())
                .fold(0.0f32, f32::max)
        };

        let mut app = app();
        app.on_pointer(PointerEvent::Moved { position: centre_of_viewport(), size: SIZE });
        assert!(app.hover.is_some(), "the centre of the viewport should hit the sphere");

        // Deep enough to move the surface the ring is standing on. The pointer
        // does not move again after this, so nothing but undo can rebuild the
        // overlay.
        app.brush.strength = 1.0;
        press(&mut app, centre_of_viewport());
        release(&mut app);

        let stale = ring(&app);
        assert!(!stale.is_empty(), "no ring was handed to the renderer");

        update(&mut app, Message::Undo);

        let stale_error = worst_off_surface(&app, &stale);
        assert!(
            stale_error > ON_SURFACE,
            "the fixture asserts nothing: the stroke left the ring only {stale_error} voxels \
             off the surface undo restored, which a stale ring would pass"
        );

        let error = worst_off_surface(&app, &ring(&app));
        assert!(
            error <= ON_SURFACE,
            "undo left the brush ring {error} voxels off the surface it restored, so the \
             overlay was never rebuilt"
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
            (
                app.doc.active_volume().sample_world(hit + sideways),
                app.doc.active_volume().sample_world(hit - sideways),
            )
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

    // --- re-orienting the model ------------------------------------------

    /// The middle of the navigation cube, which always picks a face rather than
    /// an edge or a corner.
    fn cube_centre() -> Vector {
        let (origin, size) = crate::navcube::corner_rect(Vec2::new(SIZE.x, SIZE.y));
        let middle = origin + size * 0.5;
        Vector::new(middle.x, middle.y)
    }

    /// A model with material in one identifiable place, so a turn is visible.
    ///
    /// The seeded sphere is centred and symmetric, so every rotation leaves it
    /// looking exactly the same -- a test built on it would pass whether or not
    /// anything turned.
    fn lopsided(app: &mut Brokkr) {
        let mut volume = brokkr_core::Volume::new(app.doc.voxel_size());
        volume.seed_sphere(Vec3::new(0.0, 30.0, 0.0), 8.0);
        volume.mark_everything_dirty();
        app.doc.replace_active_volume(volume);
        app.remesh_dirty();
    }

    /// The collision this feature had to resolve. A right press on the cube
    /// used to fall through to an orbit and then, on release, open the BRUSH
    /// menu on top of the cube -- so the one gesture that should have asked
    /// about orientation asked about brush radius instead.
    #[test]
    fn a_right_click_on_the_cube_opens_the_cubes_menu_and_not_the_brushs() {
        let mut app = app();
        right_press(&mut app, cube_centre());
        right_release(&mut app);

        assert!(app.cube_menu.is_some(), "the cube's own menu did not open");
        assert!(app.menu.is_none(), "the brush menu opened on top of the cube");
        assert!(app.drag.is_none(), "the press started an orbit");
        assert!(app.flight.is_none(), "a right click flew the camera as though it were a left one");
    }

    #[test]
    fn a_left_click_on_the_cube_still_flies_the_camera() {
        // The half of the cube's behaviour that already existed, pinned
        // alongside the new half so a change to the routing cannot take it out
        // silently.
        let mut app = app();
        press(&mut app, cube_centre());
        release(&mut app);

        assert!(app.flight.is_some(), "the left click stopped moving the camera");
        assert!(app.cube_menu.is_none(), "a left click opened the orientation menu");
    }

    #[test]
    fn the_next_press_closes_the_cube_menu_without_sculpting() {
        let mut app = app();
        let probe = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        right_press(&mut app, cube_centre());
        right_release(&mut app);
        assert!(app.cube_menu.is_some());

        let before = app.doc.active_volume().sample_world(probe);
        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert!(app.cube_menu.is_none(), "the menu stayed open");
        assert_eq!(
            app.doc.active_volume().sample_world(probe),
            before,
            "dismissing the menu also sculpted"
        );
    }

    #[test]
    fn escape_closes_the_cube_menu() {
        let mut app = app();
        right_press(&mut app, cube_centre());
        right_release(&mut app);
        update(&mut app, Message::MenuClosed);
        assert!(app.cube_menu.is_none());
    }

    #[test]
    fn choosing_a_direction_turns_the_model_and_leaves_the_camera_alone() {
        let mut app = app();
        lopsided(&mut app);
        let (yaw, pitch, roll) = (app.camera.yaw, app.camera.pitch, app.camera.roll);
        let above = Vec3::new(0.0, 30.0, 0.0);
        let in_front = Vec3::new(0.0, 0.0, 30.0);
        assert!(app.doc.active_volume().sample_world(above) < 0.0, "the fixture is wrong");

        app.cube_menu =
            Some(CubeMenu { at: Vec2::new(700.0, 20.0), facing: brokkr_core::Facing::Up });
        update(&mut app, Message::OrientFace(brokkr_core::Facing::Front));

        // The model moved...
        assert!(
            app.doc.active_volume().sample_world(in_front) < 0.0,
            "the material did not arrive in front of the origin"
        );
        assert!(
            app.doc.active_volume().sample_world(above) >= 0.0,
            "the material is still above the origin"
        );
        // ...and the camera did not. This is the whole difference between
        // turning the model and re-aiming the view, and it is what makes an
        // export land upright.
        assert_eq!((app.camera.yaw, app.camera.pitch, app.camera.roll), (yaw, pitch, roll));

        assert!(app.cube_menu.is_none(), "the menu stayed open over a model that had moved");
        assert!(app.unsaved, "a turned model is unsaved work");
        assert!(!app.history.can_undo(), "history outlived the volume it named bricks of");
        assert!(app.perf.dirty_bricks > 0, "nothing was remeshed, so the turn is invisible");
    }

    #[test]
    fn turning_a_face_onto_itself_changes_nothing() {
        // The menu greys this one out, but the message is public and the guard
        // is in `orient`, not in the widget.
        let mut app = app();
        lopsided(&mut app);
        app.unsaved = false;
        let above = Vec3::new(0.0, 30.0, 0.0);
        let before = app.doc.active_volume().sample_world(above);

        app.cube_menu = Some(CubeMenu { at: Vec2::ZERO, facing: brokkr_core::Facing::Up });
        update(&mut app, Message::OrientFace(brokkr_core::Facing::Up));

        assert_eq!(app.doc.active_volume().sample_world(above), before);
        assert!(!app.unsaved, "a turn that did nothing still marked the document dirty");
    }

    #[test]
    fn turning_back_returns_the_model() {
        // The recovery path, and the reason there is no undo entry: a quarter
        // turn is exact, so the other way round is a real undo rather than an
        // approximation of one.
        let mut app = app();
        lopsided(&mut app);
        let above = Vec3::new(0.0, 30.0, 0.0);
        let before = app.doc.active_volume().sample_world(above);

        app.cube_menu = Some(CubeMenu { at: Vec2::ZERO, facing: brokkr_core::Facing::Up });
        update(&mut app, Message::OrientFace(brokkr_core::Facing::Front));
        app.cube_menu = Some(CubeMenu { at: Vec2::ZERO, facing: brokkr_core::Facing::Front });
        update(&mut app, Message::OrientFace(brokkr_core::Facing::Up));

        assert_eq!(
            app.doc.active_volume().sample_world(above),
            before,
            "turning back did not return the model"
        );
    }

    /// The same import, with its sphere where the camera is looking.
    ///
    /// `adopt_import` frames the ORIGIN at `MODEL_RADIUS_MM`, and
    /// `imported_with`'s sphere is a 6 mm ball twenty millimetres above it: a
    /// ray through the centre of the viewport misses it completely. A test
    /// that presses there to prove the press was STOPPED therefore proves
    /// nothing, which is exactly what
    /// `a_press_behind_the_orientation_prompt_does_not_sculpt` was doing --
    /// verified by removing the guard and watching it pass.
    fn imported_under_the_cursor(
        resting_up: Option<brokkr_core::Facing>,
    ) -> crate::message::Imported {
        let mut volume = brokkr_core::Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM);
        volume.mark_everything_dirty();
        crate::message::Imported { volume, ..imported_with(resting_up) }
    }

    /// Build what a finished import delivers, with a chosen guess about which
    /// way the mesh's own up pointed.
    fn imported_with(resting_up: Option<brokkr_core::Facing>) -> crate::message::Imported {
        let mut volume = brokkr_core::Volume::new(0.5);
        volume.seed_sphere(Vec3::new(0.0, 20.0, 0.0), 6.0);
        volume.mark_everything_dirty();
        crate::message::Imported {
            volume,
            report: brokkr_core::voxelise::VoxeliseReport::default(),
            source: std::path::PathBuf::from("nightwing.obj"),
            elapsed_ms: 0.0,
            resting_up,
        }
    }

    #[test]
    fn an_import_that_came_in_lying_down_raises_the_prompt() {
        let mut app = app();
        app.adopt_import(imported_with(Some(brokkr_core::Facing::Back)));
        assert_eq!(
            app.orient_prompt,
            Some(brokkr_core::Facing::Back),
            "the model is on its back and nothing offered to stand it up"
        );
    }

    #[test]
    fn an_import_that_is_already_upright_is_not_asked_about() {
        // The case that decides whether this feature is usable or a nuisance:
        // an STL exported for a slicer sits on the bed and arrives standing, so
        // the guess resolves to `Up` and must raise nothing at all. Getting
        // this wrong would put a dialog in front of nearly every print file.
        let mut app = app();
        app.adopt_import(imported_with(Some(brokkr_core::Facing::Up)));
        assert!(app.orient_prompt.is_none(), "an upright model was asked about");
    }

    #[test]
    fn an_import_with_no_tell_is_not_guessed_at() {
        let mut app = app();
        app.adopt_import(imported_with(None));
        assert!(app.orient_prompt.is_none());
    }

    #[test]
    fn accepting_the_prompt_stands_the_model_up() {
        let mut app = app();
        app.adopt_import(imported_with(Some(brokkr_core::Facing::Back)));
        // The fixture's material is above the origin; a model whose own up
        // points backwards has to be turned for that to mean anything, so
        // check where it lands rather than that it merely moved.
        let turned =
            brokkr_core::AxisRotation::taking(brokkr_core::Facing::Back, brokkr_core::Facing::Up)
                .apply(Vec3::new(0.0, 20.0, 0.0));

        update(&mut app, Message::OrientPromptAnswered(true));

        assert!(app.orient_prompt.is_none(), "the prompt stayed up after being answered");
        assert!(
            app.doc.active_volume().sample_world(turned) < 0.0,
            "the model did not turn the way promised"
        );
    }

    #[test]
    fn declining_the_prompt_leaves_the_model_exactly_as_imported() {
        let mut app = app();
        app.adopt_import(imported_with(Some(brokkr_core::Facing::Back)));
        let above = Vec3::new(0.0, 20.0, 0.0);
        let before = app.doc.active_volume().sample_world(above);

        update(&mut app, Message::OrientPromptAnswered(false));

        assert!(app.orient_prompt.is_none());
        assert_eq!(
            app.doc.active_volume().sample_world(above),
            before,
            "declining still turned the model"
        );
    }

    #[test]
    fn a_press_behind_the_orientation_prompt_does_not_sculpt() {
        // iced 0.14's `stack!` layers do not block what is underneath them, so
        // the scrim is the widget that swallows presses and this early return
        // is the guarantee behind it. Without both, a click behind the card
        // carves the model the user is being asked about.
        let mut app = app();
        app.adopt_import(imported_under_the_cursor(Some(brokkr_core::Facing::Back)));
        assert!(app.orient_prompt.is_some());

        let probe = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let before = app.doc.active_volume().sample_world(probe);
        let entries = app.history_stats.undo_entries;
        press(&mut app, centre_of_viewport());
        release(&mut app);

        assert_eq!(
            app.doc.active_volume().sample_world(probe),
            before,
            "a press reached the model behind it"
        );
        // The probe above is a fixed point on the ORIGINAL sphere, and an
        // imported model need not have surface there -- with the guard removed
        // by hand this assertion still passed, so it is not the one doing the
        // work. A stroke that lands anywhere at all records an undo entry.
        assert_eq!(
            app.history_stats.undo_entries, entries,
            "it recorded a stroke behind the prompt"
        );
        assert!(app.drag.is_none());
    }

    #[test]
    fn escape_declines_the_orientation_prompt() {
        let mut app = app();
        app.adopt_import(imported_with(Some(brokkr_core::Facing::Back)));
        let above = Vec3::new(0.0, 20.0, 0.0);
        let before = app.doc.active_volume().sample_world(above);

        update(&mut app, Message::MenuClosed);

        assert!(app.orient_prompt.is_none(), "escape did not dismiss it");
        assert_eq!(app.doc.active_volume().sample_world(above), before, "escape turned the model");
    }
}

#[cfg(test)]
mod working_size_tests {
    use super::*;

    fn update(app: &mut Brokkr, message: Message) {
        drop(app.update(message));
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// Scaling is a change to `voxel_size` and nothing else, so the field has
    /// to come through it untouched. If this ever fails, the operation stopped
    /// being free and started resampling.
    #[test]
    fn resizing_scales_the_model_without_touching_a_voxel() {
        let mut app = app();
        let before: Vec<_> = app
            .doc
            .active_volume()
            .brick_coords()
            .map(|c| (c, app.doc.active_volume().sample_voxel(c.origin())))
            .collect();
        let was =
            app.doc.active_volume().surface_bounds().expect("the starting ball has a surface");
        let longest_before = (was.1 - was.0).max_element();
        let voxel_before = app.doc.voxel_size();

        app.working_size_field = "30".into();
        update(&mut app, Message::WorkingSizeCommitted);

        let now = app.doc.active_volume().surface_bounds().expect("still a surface");
        let longest_after = (now.1 - now.0).max_element();
        assert!(
            (longest_after - 30.0).abs() < 0.5,
            "asked for 30 mm and got {longest_after:.2} mm (from {longest_before:.2})"
        );

        let factor = 30.0 / longest_before;
        assert!(
            (app.doc.voxel_size() - voxel_before * factor).abs() < 1.0e-6,
            "the voxel size should have scaled with the model"
        );
        for (coord, value) in before {
            assert_eq!(
                app.doc.active_volume().sample_voxel(coord.origin()),
                value,
                "a voxel changed, so this resampled instead of rescaling"
            );
        }
    }

    /// The detail did NOT improve, and the interface must not imply it did.
    #[test]
    fn resizing_buys_no_extra_detail() {
        let mut app = app();
        let bounds = app.doc.active_volume().surface_bounds().expect("a surface");
        let across_before =
            ((bounds.1 - bounds.0).max_element() / app.doc.voxel_size()).round() as i64;

        app.working_size_field = "12".into();
        update(&mut app, Message::WorkingSizeCommitted);

        let bounds = app.doc.active_volume().surface_bounds().expect("a surface");
        let across_after =
            ((bounds.1 - bounds.0).max_element() / app.doc.voxel_size()).round() as i64;
        assert_eq!(
            across_before, across_after,
            "the model has the same number of voxels across it, whatever size it is"
        );
    }

    /// A size that would push the voxel outside the range the finer and coarser
    /// buttons work in has to be refused, or the model lands somewhere those
    /// buttons can never bring it back from.
    #[test]
    fn a_size_that_would_put_the_voxel_out_of_range_is_refused() {
        let mut app = app();
        let voxel_before = app.doc.voxel_size();

        app.working_size_field = "0.001".into();
        update(&mut app, Message::WorkingSizeCommitted);

        assert_eq!(app.doc.voxel_size(), voxel_before, "the model was resized anyway");
        assert!(app.status.contains("could not resize"), "no refusal shown: {}", app.status);
    }

    #[test]
    fn nonsense_in_the_field_is_refused_rather_than_parsed_as_zero() {
        let mut app = app();
        let voxel_before = app.doc.voxel_size();
        app.working_size_field = "big".into();
        update(&mut app, Message::WorkingSizeCommitted);
        assert_eq!(app.doc.voxel_size(), voxel_before);
        assert!(app.status.contains("could not resize"), "{}", app.status);
    }

    /// The radius ceiling at the default voxel size must be **exactly what it
    /// always was**. The voxel rule is there to stop a fine lattice reaching an
    /// unusable brush, not to change the tool anyone is using today.
    #[test]
    fn the_voxel_rule_does_not_touch_the_brush_at_the_default_size() {
        let app = app();
        assert_eq!(app.doc.voxel_size(), VOXEL_SIZE_MM);
        assert_eq!(
            app.max_radius(),
            MAX_RADIUS_MM,
            "100 voxels is 25 mm at the default, above the millimetre ceiling, so nothing moves"
        );
    }

    /// And at a resin lattice it must bite, because the millimetre ceiling
    /// there is 640 voxels of radius -- measured at roughly a quarter of a
    /// second per stamp.
    ///
    /// The lattice is reached with [`Document::rescale`], which multiplies
    /// `voxel_size` and touches no voxel, rather than with `resample`, which
    /// rebuilds the default sphere's narrow band eight times finer.
    /// `max_radius` reads nothing but `doc.voxel_size()`, and `0.25 * 0.125`
    /// is `0.03125` to the bit, so the resample bought this assertion nothing
    /// and was refused rather than merely not chosen: measured on the built
    /// test binary it cost 4.6 s and 2.4 GB of peak RSS for one test, against
    /// 0.08 s and 75 MB through `rescale`. Cargo runs test binaries and their
    /// threads in parallel, so a 2.4 GB spike here is a spike beside the GPU
    /// offscreen suite on a CI runner. If a later change gives `max_radius` a
    /// reason to read actual voxels, this has to become a real resample --
    /// and then say so here.
    #[test]
    fn the_voxel_rule_caps_the_brush_at_a_resin_lattice() {
        let mut app = app();
        app.doc.rescale(0.125);
        assert!(
            (app.max_radius() - 3.125).abs() < 1.0e-6,
            "expected 100 voxels of radius, got {} mm",
            app.max_radius()
        );

        // And a request past it is actually clamped, not merely reported.
        drop(app.update(Message::BrushRadiusChanged(MAX_RADIUS_MM)));
        assert!(
            app.brush.radius <= app.max_radius() + 1.0e-6,
            "the slider handed out {} mm past a {} mm ceiling",
            app.brush.radius,
            app.max_radius()
        );
    }

    /// A finer step the pool cannot hold lands on the finest that fits,
    /// rather than refusing with directions to a rung the buttons cannot
    /// reach. The refusal is kept only for when there is nowhere finer left.
    #[test]
    fn a_step_that_does_not_fit_lands_on_the_finest_that_does() {
        let mut app = app();
        // A pool measured at 6M of 11M vertices: one halving would predict
        // 24M, far over, but there is real room above the current size.
        app.shared.set_stats_for_tests(brokkr_gpu::PoolStats {
            vertices_reserved: 6_000_000,
            indices_reserved: 33_000_000,
            vertex_capacity: 11_000_000,
            index_capacity: 66_000_000,
            ..Default::default()
        });

        update(&mut app, Message::Resample(VOXEL_SIZE_MM / 2.0));

        let expected = VOXEL_SIZE_MM / (11.0f32 / 6.0).sqrt() * 1.03;
        assert!(
            (app.doc.voxel_size() - expected).abs() < 1.0e-3,
            "expected to land near {expected:.3} mm, landed at {:.3}",
            app.doc.voxel_size()
        );
        assert!(
            app.status.contains("finest the mesh pool holds"),
            "the status must say the step was capped: {}",
            app.status
        );
    }

    /// Memory is checked before the pool, because since the pool grew itself
    /// buffers it is no longer the tighter of the two. A step the pool would
    /// happily hold can still be one that walks the machine into swap.
    #[test]
    fn a_step_that_would_exhaust_memory_is_capped_even_when_the_pool_has_room() {
        let mut app = app();
        // Plenty of pool...
        app.shared.set_stats_for_tests(brokkr_gpu::PoolStats {
            vertices_reserved: 1_000_000,
            indices_reserved: 6_000_000,
            vertex_capacity: brokkr_gpu::TOTAL_VERTEX_CAPACITY,
            index_capacity: brokkr_gpu::TOTAL_INDEX_CAPACITY,
            ..Default::default()
        });
        // ...but the volume is already 4 GB, so one halving would want 16.
        app.doc_stats.resident_bytes = 4 * 1024 * 1024 * 1024;
        let before = app.doc.voxel_size();

        update(&mut app, Message::Resample(before / 2.0));

        assert!(app.doc.voxel_size() < before, "it should still go finer, just not that fine");
        assert!(app.doc.voxel_size() > before / 2.0, "and not all the way to the requested size");
        assert!(
            app.status.contains("finest the mesh pool holds"),
            "the cap should be reported: {}",
            app.status
        );
    }

    /// And when the current size already sits at the fit limit, the refusal
    /// stays a refusal -- there is nowhere finer to land.
    #[test]
    fn at_the_fit_limit_the_refusal_remains() {
        let mut app = app();
        app.shared.set_stats_for_tests(brokkr_gpu::PoolStats {
            vertices_reserved: 10_800_000,
            indices_reserved: 64_000_000,
            vertex_capacity: 11_000_000,
            index_capacity: 66_000_000,
            ..Default::default()
        });
        let before = app.doc.voxel_size();

        update(&mut app, Message::Resample(before / 2.0));

        assert_eq!(app.doc.voxel_size(), before, "there was no room, so nothing should move");
        assert!(app.status.contains("could not resample"), "{}", app.status);
    }

    /// The readout has to name both numbers, because either alone is the
    /// scaling changed no detail, and voxels-across without millimetres cannot
    /// answer "is this enough for my printer".
    #[test]
    fn the_advice_names_both_the_resolution_and_what_it_measures() {
        let app = app();
        let advice = app.measure_detail_advice();
        assert!(advice.contains("voxels wide"), "no voxel count: {advice}");
        assert!(advice.contains("mm across"), "no physical size: {advice}");
        assert!(advice.contains("voxel "), "no voxel size: {advice}");
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
        let (mesh, _) = app.doc.active_volume().export_mesh();
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
                let here = app.doc.active_volume().sample_world(probe);
                let mirrored = app.doc.active_volume().sample_world(probe * flip);
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
        let (_, report) = app.doc.active_volume().export_mesh();
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

        let (_, report) = app.doc.active_volume().export_mesh();
        assert!(report.is_printable(), "{}", report.summary());
    }

    #[test]
    fn resampling_finer_increases_detail_and_keeps_it_printable() {
        let mut app = app();
        let (_, before) = app.doc.active_volume().export_mesh();
        let coarse_voxel = app.doc.voxel_size();

        update(&mut app, Message::Resample(coarse_voxel / 2.0));

        assert!((app.doc.voxel_size() - coarse_voxel / 2.0).abs() < 1.0e-6);
        let (_, after) = app.doc.active_volume().export_mesh();
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

        // And the interface offers a step that lands inside the limits. Three
        // halvings from the default are allowed and four are not: 0.25 halved
        // three times is 0.03125, which is the finest rung of the ladder and
        // the one resin work needs. It was two halvings until 2026-08-21, when
        // the floor came down from 0.06 to 0.03 for exactly that rung.
        const _: () = assert!(FINEST_VOXEL_MM > VOXEL_SIZE_MM / 16.0);
        const _: () = assert!(FINEST_VOXEL_MM < VOXEL_SIZE_MM / 4.0);
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

        let old_coords: Vec<BrickCoord> = app.doc.active_volume().brick_coords().collect();
        let finer = app.doc.voxel_size() / 2.0;
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
        let before = app.doc.active_volume().brick_count();
        let same = app.doc.voxel_size();
        update(&mut app, Message::Resample(same));
        assert_eq!(app.doc.active_volume().brick_count(), before);
        assert!(app.status.is_empty(), "nothing happened, so nothing should be reported");
    }

    // --- the N-body export -------------------------------------------------

    fn scratch(name: &str) -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!("brokkr-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// An application holding `count` small spheres, laid out along X so they
    /// do not touch.
    ///
    /// A coarse lattice and a small radius on purpose: this is about how many
    /// bodies reach a file, and twelve default spheres at 0.25 mm would spend
    /// the whole test welding.
    fn app_with_bodies(count: usize) -> Brokkr {
        let mut app = app();
        let mut first = Volume::new(1.0);
        first.seed_sphere(Vec3::ZERO, 8.0);
        let mut doc = Document::from_volume(first);
        for index in 1..count {
            let mut volume = Volume::new(1.0);
            volume.seed_sphere(Vec3::new(index as f32 * 48.0, 0.0, 0.0), 8.0);
            doc.add_body(format!("Body {}", index + 1), volume);
        }
        app.doc = doc;
        app.rebuild_everything();
        app
    }

    /// Turn one row's eye off through the document's own snapshot path, which
    /// is what the panel will do.
    fn hide(app: &mut Brokkr, at: usize) {
        let id = app.doc.nodes()[at].id;
        let mut meta = app.doc.meta(id).expect("the row is in the document");
        meta.visible = false;
        app.doc.set_meta(&meta);
    }

    /// **Twelve bodies with one hidden export eleven, and the status says so in
    /// those words.**
    ///
    /// Decisions 4 and 10 both rest on this string. The eye is one bit in a
    /// forty-byte node record with no checksum over it, and it is the bit that
    /// decides whether a part reaches the printer -- a flipped bit is a legal
    /// value that loads without any error at all. The brick stream has a
    /// checked distance decode defending it; the node table has nothing. Naming
    /// the count is the whole of that defence, which is why it is asserted
    /// rather than assumed.
    #[test]
    fn a_hidden_body_is_omitted_and_the_status_names_how_many() {
        let directory = scratch("export-hidden");
        let path = directory.join("twelve.obj");
        let mut app = app_with_bodies(12);
        hide(&mut app, 4);

        app.export(ExportFormat::Obj, &path);

        assert!(
            app.status.contains("exported 11 of 12 bodies; 1 hidden"),
            "the omitted count is not in the status: {}",
            app.status
        );
        let written = std::fs::read_to_string(&path).expect("the file should be there");
        let objects: Vec<&str> = written.lines().filter(|line| line.starts_with("o ")).collect();
        assert_eq!(objects.len(), 11, "eleven bodies should have reached the file: {objects:?}");
        assert!(
            !objects.contains(&"o Body 5"),
            "the hidden body reached the file anyway: {objects:?}"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// **And a document with nothing hidden still names the count.**
    ///
    /// Split out from the test above because it is the case a conditional
    /// message would get wrong, and a conditional message is the obvious
    /// "improvement" somebody makes later: if the line only appears when
    /// something was hidden, silence means either "nothing was hidden" or "the
    /// count was never worked out", and those are the two readings the count
    /// exists to keep apart.
    #[test]
    fn a_document_with_nothing_hidden_still_names_the_count() {
        let directory = scratch("export-none-hidden");
        let path = directory.join("one.stl");
        let mut app = app();

        app.export(ExportFormat::Stl, &path);

        assert!(
            app.status.contains("exported 1 of 1 bodies; 0 hidden"),
            "the count is missing from a plain single-body export: {}",
            app.status
        );
        std::fs::remove_dir_all(&directory).ok();
    }

    /// **An export while soloed writes every body the EYE is showing, including
    /// the ones solo is hiding, and names only the eye-hidden count.**
    ///
    /// This is the line where a view mode would be at its most dangerous. Solo
    /// is a way of looking at a document; export is what reaches a printer. If
    /// the two ever share a visibility answer, a user who soloed one part to
    /// inspect it and then pressed Export gets a file with eleven parts missing
    /// and a status line that agrees with them -- and the failure is discovered
    /// after the print. `export` reads `saved_visibility`, and `saved_visibility`
    /// takes no solo parameter at all, so this is a test of a call that has no
    /// other form.
    #[test]
    fn an_export_while_soloed_still_writes_every_body_the_eye_shows() {
        let directory = scratch("export-soloed");
        let path = directory.join("twelve.obj");
        let mut app = app_with_bodies(12);
        hide(&mut app, 4);

        // Solo something else entirely, so eleven of the twelve rows are not
        // drawn and only ONE of them is hidden by an eye.
        let soloed = app.doc.nodes()[7].id;
        drop(app.update(Message::SoloEntered(soloed)));
        assert_eq!(app.solo, Some(soloed), "the fixture never entered solo");
        let mut drawn = Vec::new();
        app.doc.display_visibility(app.solo, &mut drawn);
        assert_eq!(
            drawn.iter().filter(|shown| **shown).count(),
            1,
            "the fixture is not actually hiding anything on screen"
        );

        app.export(ExportFormat::Obj, &path);

        assert!(
            app.status.contains("exported 11 of 12 bodies; 1 hidden"),
            "solo reached the export, or the count now includes it: {}",
            app.status
        );
        let written = std::fs::read_to_string(&path).expect("the file should be there");
        let objects: Vec<&str> = written.lines().filter(|line| line.starts_with("o ")).collect();
        assert_eq!(objects.len(), 11, "solo dropped ten parts out of the print: {objects:?}");
        assert!(
            !objects.contains(&"o Body 5"),
            "the eye-hidden body reached the file anyway: {objects:?}"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A single body's STL is byte for byte what the single-mesh writer
    /// produces, end to end through the application.
    ///
    /// **What this proves on its own is narrower than its name suggests**, and
    /// worth stating so nobody reads it as the byte pin it is not. Both sides
    /// end in `stl::write_all`, so a change to the writer moves them together
    /// and this test cannot see it. What it does see is everything the
    /// application wraps around the writer -- `export_bodies`, the per-body
    /// weld, `document_verdict`, the file creation -- adding nothing to a
    /// one-body file. The bytes themselves are pinned in `brokkr-core` by
    /// `export::stl::tests::a_known_mesh_writes_the_bytes_committed_in_the_golden`
    /// against a committed fixture. The two together are what pins the file a
    /// user receives; neither one does it alone.
    ///
    /// The shipped, verified property: an STL out of this build has been opened
    /// in a slicer and printed. STL carries no object names, so it is the one
    /// format where the N-body path can be identical rather than merely
    /// equivalent -- OBJ and 3MF now carry the body's name where they used to
    /// carry the literal `BrokkrSculpt`, which is the only byte that moves.
    #[test]
    fn a_single_body_export_is_byte_identical_to_the_single_mesh_writer() {
        let directory = scratch("export-identical");
        let path = directory.join("one.stl");
        let mut app = app();

        app.export(ExportFormat::Stl, &path);
        let written = std::fs::read(&path).expect("the file should be there");

        let (mesh, report) = app.doc.active_volume().export_mesh();
        assert!(report.is_printable());
        let mut expected = Vec::new();
        brokkr_core::export::stl::write(&mesh, &mut expected).unwrap();
        assert_eq!(written, expected, "the N-body path changed a one-body STL");

        std::fs::remove_dir_all(&directory).ok();
    }

    /// Several bodies in one STL are named as fused, because the format has
    /// nowhere to put an object boundary and a slicer will load them as one
    /// part. Better said here than discovered after slicing.
    #[test]
    fn an_stl_of_several_bodies_says_they_arrive_as_one_part() {
        let directory = scratch("export-stl-fused");
        let path = directory.join("three.stl");
        let mut app = app_with_bodies(3);

        app.export(ExportFormat::Stl, &path);
        assert!(app.status.contains("exported 3 of 3 bodies"), "{}", app.status);
        assert!(
            app.status.contains("STL fuses them into one part"),
            "an STL of three bodies did not say what it did with them: {}",
            app.status
        );
        std::fs::remove_dir_all(&directory).ok();
    }

    /// **A body that would not print refuses the whole export, and refuses it
    /// before the file is opened.**
    ///
    /// `File::create` truncates, so a verdict taken after it destroys the file
    /// the user was about to print -- and the natural per-body refactor is
    /// exactly the one that moves the check down past the open. The previous
    /// export was left in place here, unread and unchanged.
    #[test]
    fn an_unprintable_body_refuses_before_the_file_is_opened() {
        let directory = scratch("export-refusal");
        let path = directory.join("keepme.obj");
        std::fs::write(&path, b"the previous export, which must survive").unwrap();

        let mut app = app_with_bodies(2);
        // An empty body prints nothing, and the sum over the document would
        // still read as watertight -- see `document_verdict`.
        let empty = app.doc.nodes()[1].id;
        *app.doc.volume_mut(empty).expect("a body") = Volume::new(1.0);

        app.export(ExportFormat::Obj, &path);

        assert!(app.status.starts_with("not exported"), "reported: {}", app.status);
        assert!(app.status.contains("Body 2"), "the refusal must name the body: {}", app.status);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"the previous export, which must survive",
            "the refusal opened and truncated the file before deciding"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// **The detail guard measures the whole document, not the active body.**
    ///
    /// It never did: `too_fine_for_the_pool` reads `doc_stats`, which is a sum
    /// over every body, and the numbers below are what makes the difference
    /// visible. Two bodies of 0.8 GiB each is 1.6 GiB, and one halving of the
    /// voxel quadruples a shell -- 6.4 GiB, past the ceiling. Either body ALONE
    /// would come to 3.2 GiB and be admitted. A guard that asked the active
    /// body would step straight past the ceiling with the second body's bricks
    /// unaccounted for, and the first thing to report it would be the machine
    /// swapping.
    #[test]
    fn a_two_body_document_over_the_ceiling_refuses_the_finer_step() {
        let mut app = app_with_bodies(2);
        assert_eq!(app.doc.body_count(), 2);

        // `doc_stats` really is the sum, measured rather than assumed.
        let per_body: usize =
            app.doc.bodies().map(|(_, volume)| volume.stats().resident_bytes).sum();
        assert_eq!(app.doc_stats.resident_bytes, per_body);
        assert!(
            app.doc_stats.resident_bytes > app.doc.active_volume().stats().resident_bytes,
            "the fixture's two bodies must not cost the same as one"
        );

        let each = 0.8 * 1024.0 * 1024.0 * 1024.0;
        assert!(each * 4.0 < MAX_VOLUME_BYTES, "either body alone has to be admitted");
        assert!(each * 2.0 * 4.0 > MAX_VOLUME_BYTES, "the pair has to be refused");
        app.doc_stats.resident_bytes = (each * 2.0) as usize;

        let before = app.doc.voxel_size();
        update(&mut app, Message::Resample(before / 2.0));

        assert!(app.doc.voxel_size() > before / 2.0, "the halving should not have been allowed");
        assert!(
            app.status.contains("finest the mesh pool holds"),
            "the cap should be reported: {}",
            app.status
        );
    }
}

#[cfg(test)]
mod timeline_tests {
    use super::*;

    /// See the note in `tests::update`.
    fn update(app: &mut Brokkr, message: Message) {
        drop(app.update(message));
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// Put the pointer somewhere along the strip and press.
    ///
    /// Through the real messages rather than by calling `Timeline` directly.
    /// The Move brush shipped twice passing tests that built the objects by
    /// hand and never went through the routing, and both times it did nothing
    /// in the hand -- so what is exercised here is the path a click takes.
    fn click_at(app: &mut Brokkr, fraction: f32) {
        let x = fraction * app.timeline.width();
        update(app, Message::TimelineHover(x));
        update(app, Message::TimelinePressed);
        update(app, Message::TimelineReleased);
    }

    #[test]
    fn clicking_empty_track_stores_the_view_that_was_showing() {
        let mut app = app();
        app.camera.yaw = 1.25;
        app.camera.distance = 88.0;
        update(&mut app, Message::BrushRadiusChanged(7.5));
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::Y));

        click_at(&mut app, 0.4);

        assert_eq!(app.timeline.keys.len(), 1, "a click on empty track stored nothing");
        let key = app.timeline.keys[0];
        assert!((key.at - 0.4).abs() < 0.01, "the key landed at {}", key.at);
        assert_eq!(key.view.camera_yaw, 1.25);
        assert_eq!(key.view.camera_distance, 88.0);
        assert_eq!(key.view.brush_radius, 7.5, "a key has to hold the brush, not only the camera");
        assert_eq!(key.view.mirror, [false, true, false], "the mirror planes were not stored");
        assert!(app.unsaved, "storing a key changed the document and did not say so");
    }

    #[test]
    fn clicking_a_key_puts_everything_it_holds_back() {
        // The property that makes a key a stored working setup rather than a
        // camera angle. Every field has to come back, so this moves all of
        // them away between storing and returning.
        let mut app = app();
        app.camera.yaw = 0.2;
        app.camera.distance = 40.0;
        update(&mut app, Message::BrushRadiusChanged(3.0));
        update(&mut app, Message::BrushStrengthChanged(0.2));
        click_at(&mut app, 0.3);

        app.camera.yaw = 2.9;
        app.camera.distance = 150.0;
        update(&mut app, Message::BrushRadiusChanged(9.0));
        update(&mut app, Message::BrushStrengthChanged(0.6));
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::X));

        click_at(&mut app, 0.3);

        assert_eq!(app.timeline.keys.len(), 1, "clicking the key added a second one");
        assert_eq!(app.camera.yaw, 0.2);
        assert_eq!(app.camera.distance, 40.0);
        assert_eq!(app.brush.radius, 3.0);
        assert_eq!(app.brush.strength, 0.2);
        assert!(!app.symmetry.axis(MirrorAxis::X), "the mirror plane was not restored");
    }

    #[test]
    fn dragging_a_key_past_its_neighbour_keeps_hold_of_it() {
        // The list stays sorted, so a dragged key's index moves under the
        // drag. Losing track of it here drops the drag at exactly the moment
        // a user is watching for it.
        let mut app = app();
        click_at(&mut app, 0.2);
        click_at(&mut app, 0.5);
        click_at(&mut app, 0.8);
        // Distinguish them by something that travels with the key.
        for (index, key) in app.timeline.keys.iter_mut().enumerate() {
            key.view.camera_distance = 100.0 + index as f32;
        }

        // Pick up the leftmost and haul it past both the others.
        let width = app.timeline.width();
        update(&mut app, Message::TimelineHover(0.2 * width));
        update(&mut app, Message::TimelinePressed);
        assert_eq!(app.timeline.dragged_key(), Some(0));
        update(&mut app, Message::TimelineHover(0.95 * width));
        update(&mut app, Message::TimelineReleased);

        let order: Vec<f32> =
            app.timeline.keys.iter().map(|key| key.view.camera_distance).collect();
        assert_eq!(order, vec![101.0, 102.0, 100.0], "the dragged key did not travel");
        let at: Vec<f32> = app.timeline.keys.iter().map(|key| key.at).collect();
        assert!(at.windows(2).all(|pair| pair[0] <= pair[1]), "the keys came out unsorted: {at:?}");
    }

    #[test]
    fn a_right_click_removes_the_key_under_the_pointer_and_only_that_one() {
        let mut app = app();
        click_at(&mut app, 0.25);
        click_at(&mut app, 0.75);
        app.unsaved = false;

        let width = app.timeline.width();
        update(&mut app, Message::TimelineHover(0.25 * width));
        update(&mut app, Message::TimelineRemoveKey);

        assert_eq!(app.timeline.keys.len(), 1);
        assert!((app.timeline.keys[0].at - 0.75).abs() < 0.01, "the wrong key went");
        assert!(app.unsaved, "removing a key changed the document and did not say so");
    }

    #[test]
    fn a_right_click_on_empty_track_removes_nothing() {
        let mut app = app();
        click_at(&mut app, 0.25);
        app.unsaved = false;

        let width = app.timeline.width();
        update(&mut app, Message::TimelineHover(0.9 * width));
        update(&mut app, Message::TimelineRemoveKey);

        assert_eq!(app.timeline.keys.len(), 1, "a right click on nothing removed a key");
        assert!(!app.unsaved, "nothing changed, so nothing should be marked unsaved");
    }

    #[test]
    fn playing_moves_the_camera_and_leaves_the_brush_alone() {
        // The policy the whole module is built around, and the one that would
        // be maddening to discover by accident: a fly-through must not reach
        // over and change the tool in the user's hand.
        let mut app = app();
        app.camera.yaw = 0.0;
        update(&mut app, Message::BrushRadiusChanged(4.0));
        click_at(&mut app, 0.0);

        app.camera.yaw = 2.0;
        update(&mut app, Message::BrushRadiusChanged(9.0));
        update(&mut app, Message::SymmetryAxisToggled(MirrorAxis::Z));
        click_at(&mut app, 1.0);

        // Back to the start, then play.
        click_at(&mut app, 0.0);
        assert_eq!(app.brush.radius, 4.0, "going to a key should restore the brush");
        update(&mut app, Message::BrushRadiusChanged(9.0));
        update(&mut app, Message::TimelinePlayToggled);
        assert!(app.timeline.playing);

        let started_at = app.camera.yaw;
        for _ in 0..40 {
            update(&mut app, Message::Frame);
        }
        assert_ne!(app.camera.yaw, started_at, "playback did not move the camera");
        assert_eq!(app.brush.radius, 9.0, "playback changed the brush out from under the hand");
    }

    #[test]
    fn playback_stops_itself_at_the_last_key() {
        let mut app = app();
        click_at(&mut app, 0.0);
        app.camera.yaw = 1.5;
        click_at(&mut app, 0.5);
        click_at(&mut app, 0.0);
        update(&mut app, Message::TimelinePlayToggled);

        // In controlled time, for the reason `finish_flight` documents:
        // `Message::Frame` scales by the real clock and consecutive calls in a
        // test are microseconds apart, so a run driven that way never arrives.
        // That `Frame` drives playback at all is covered by the test above.
        //
        // Far more steps than the run needs, so a playhead that failed to stop
        // would have run off the end of the strip.
        for _ in 0..600 {
            if let Some(pose) = app.timeline.advance(16.0) {
                app.fly_camera_to(&pose);
            }
        }
        assert!(!app.timeline.playing, "playback never stopped");
        assert!(app.timeline.playhead <= 1.0, "the playhead ran off the strip");
        assert!(
            (app.timeline.playhead - 0.5).abs() < 0.01,
            "it stopped at {} rather than at the last key",
            app.timeline.playhead
        );
    }

    #[test]
    fn play_does_nothing_with_fewer_than_two_keys() {
        // There is nothing to fly between, and a button that visibly does
        // nothing is worse than one that is plainly unavailable.
        let mut app = app();
        update(&mut app, Message::TimelinePlayToggled);
        assert!(!app.timeline.playing);
        click_at(&mut app, 0.5);
        update(&mut app, Message::TimelinePlayToggled);
        assert!(!app.timeline.playing);
    }

    #[test]
    fn keys_survive_a_save_and_a_reopen_through_the_application() {
        // Through the application rather than the format, because the
        // interesting failures are in the wiring: a state that is built
        // without them, or a load that reads them and drops them on the floor.
        let directory =
            std::env::temp_dir().join(format!("brokkr-timeline-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("keys.brokkr");

        let mut app = app();
        app.camera.yaw = 0.4;
        click_at(&mut app, 0.1);
        app.camera.yaw = 2.4;
        click_at(&mut app, 0.6);
        app.save_project(&path);
        assert!(app.status.starts_with("saved"), "save reported: {}", app.status);

        let mut reopened = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        reopened.open_project(&path);
        assert!(!reopened.status.contains("could not"), "open reported: {}", reopened.status);
        assert_eq!(reopened.timeline.keys.len(), 2, "the keys did not survive the reopen");
        assert!((reopened.timeline.keys[0].view.camera_yaw - 0.4).abs() < 1.0e-6);
        assert!((reopened.timeline.keys[1].view.camera_yaw - 2.4).abs() < 1.0e-6);

        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn opening_a_second_file_does_not_leave_the_first_ones_keys_behind() {
        // `adopt` replaces rather than extends. Getting this wrong leaves keys
        // pointing at a model that is no longer on screen, which looks like
        // the timeline remembering something it should not.
        let directory =
            std::env::temp_dir().join(format!("brokkr-timeline-swap-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let with_keys = directory.join("keys.brokkr");
        let without = directory.join("bare.brokkr");

        let mut app = app();
        click_at(&mut app, 0.3);
        app.save_project(&with_keys);

        let mut bare = Brokkr::with_tablet(crate::tablet::Tablet::inert());
        bare.save_project(&without);

        app.open_project(&without);
        assert!(app.timeline.keys.is_empty(), "the previous model's keys are still on the strip");
        assert!(!app.timeline.playing);

        let _ = std::fs::remove_dir_all(&directory);
    }
}

/// The one rule about what is drawn, checked from both ends.
///
/// Visibility has three inputs -- a body's own eye, every ancestor folder's
/// eye, and solo -- resolved in one function in `brokkr-core` and read by the
/// panel, the pick gate, the plane cut and the renderer. These tests are about
/// the last of those: that what the renderer has been told never disagrees
/// with what the document says. When those two drift, a body is missing from
/// the viewport while its row still reads "visible", and it still raycasts and
/// still carves -- which is not a failure anyone can reproduce from a
/// description.
#[cfg(test)]
mod visibility_tests {
    use super::*;
    use brokkr_core::NodeMeta;
    use iced::Vector;

    fn update(app: &mut Brokkr, message: Message) {
        drop(app.update(message));
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// A second body with something in it, so that hiding it is a question
    /// about geometry rather than about an empty row.
    fn add_body(app: &mut Brokkr, name: &str) -> NodeId {
        let mut volume = Volume::new(app.doc.voxel_size());
        volume.seed_sphere(Vec3::new(60.0, 0.0, 0.0), 10.0);
        volume.mark_everything_dirty();
        app.doc.add_body(name, volume)
    }

    fn set_eye(app: &mut Brokkr, id: NodeId, visible: bool) {
        let meta = NodeMeta { visible, ..app.doc.meta(id).expect("the body is in the document") };
        app.doc.set_meta(&meta);
    }

    /// What the document says is hidden, worked out from the resolver rather
    /// than from `publish_visibility`'s own fold.
    ///
    /// **`app.solo` and not `None`.** The pinned check is that
    /// `hidden_snapshot()` equals `doc.display_visibility(app.solo)` after every
    /// message, so passing `None` here would make every solo message look like
    /// a disagreement -- and, worse, would keep passing if solo ever stopped
    /// reaching the renderer at all.
    fn hidden_by_the_document(app: &Brokkr) -> Vec<NodeId> {
        let mut shown = Vec::new();
        app.doc.display_visibility(app.solo, &mut shown);
        app.doc
            .nodes()
            .iter()
            .zip(&shown)
            .filter(|(node, shown)| !**shown && node.is_body())
            .map(|(node, _)| node.id)
            .collect()
    }

    fn assert_agrees(app: &Brokkr, after: &str) {
        assert_eq!(
            app.shared.hidden_snapshot(),
            hidden_by_the_document(app),
            "the renderer and the document disagree about what is drawn, after {after}"
        );
    }

    /// **After every message**, which is why the check is written as a loop
    /// over messages rather than as one assertion per feature.
    ///
    /// The pass runs in `update` rather than in the arms that change an eye,
    /// precisely because "the arms that change an eye" is a list that goes out
    /// of date silently -- open, reset, import, undo, redo, delete and solo all
    /// belong to it, and increments 9 to 13 add more. If someone moves the pass
    /// back into the arms, this test is what fails.
    #[test]
    fn the_renderers_hidden_set_agrees_with_the_document_after_every_message() {
        let mut app = app();
        let second = add_body(&mut app, "Body 2");
        set_eye(&mut app, second, false);

        // Nothing has been dispatched yet, so this is also the check that the
        // constructor published something rather than leaving the renderer to
        // find out on the first frame.
        update(&mut app, Message::Frame);
        assert_eq!(
            app.shared.hidden_snapshot(),
            vec![second],
            "a hidden body never reached the renderer at all"
        );

        // A stroke, so undo and redo have something to move.
        app.on_pointer(PointerEvent::Moved {
            position: Vector::new(640.0, 360.0),
            size: Vector::new(1280.0, 720.0),
        });
        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: Vector::new(640.0, 360.0),
            size: Vector::new(1280.0, 720.0),
        });
        app.on_pointer(PointerEvent::Released { button: PointerButton::Left });

        let messages = [
            Message::Frame,
            Message::BrushKindChanged(BrushKind::Draw),
            Message::SymmetryAxisToggled(MirrorAxis::X),
            Message::StatsToggled,
            Message::Undo,
            Message::Redo,
            Message::Undo,
            Message::Frame,
        ];
        for message in messages {
            let named = format!("{message:?}");
            update(&mut app, message);
            assert_agrees(&app, &named);
        }

        // Showing it again has to travel the same way, or the eye turns things
        // off and never back on.
        set_eye(&mut app, second, true);
        update(&mut app, Message::Frame);
        assert!(app.shared.hidden_snapshot().is_empty(), "the body never came back");
        assert_agrees(&app, "showing the body again");
    }

    /// A whole-document swap must not leave the renderer holding the previous
    /// document's ids.
    ///
    /// Ids restart from 1 in every new document, so a stale hidden set does not
    /// merely name something absent -- it names a real body in the new
    /// document, and hides it.
    #[test]
    fn opening_and_resetting_do_not_leave_the_previous_documents_ids_hidden() {
        let directory =
            std::env::temp_dir().join(format!("brokkr-visibility-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("two-bodies.brokkr");

        let mut app = app();
        let second = add_body(&mut app, "Body 2");
        set_eye(&mut app, second, false);
        update(&mut app, Message::Frame);
        assert_eq!(app.shared.hidden_snapshot(), vec![second]);
        app.save_project(&path);

        // Reset first: one body, nothing hidden, and the id the old second body
        // used is now free for something else to be given.
        update(&mut app, Message::ResetSphere);
        assert!(app.confirm.is_none(), "the fixture had unsaved work and never reset");
        assert!(
            app.shared.hidden_snapshot().is_empty(),
            "the reset document is still hiding an id from the document before it"
        );
        assert_agrees(&app, "a reset");

        // ...and opening the saved file brings the hidden body back, because
        // the eye is persisted state and a reopen is not a view mode.
        app.open_project(&path);
        assert!(!app.status.contains("could not"), "open reported: {}", app.status);
        assert_eq!(
            app.shared.hidden_snapshot().len(),
            1,
            "the saved eye did not reach the renderer after the open"
        );
        assert_agrees(&app, "an open");

        let _ = std::fs::remove_dir_all(&directory);
    }

    /// **Hiding is a draw-time skip**, so it must not dirty a single brick.
    ///
    /// The other half of this rule -- that no pool space moves either -- is in
    /// `brokkr-gpu`, because `vertices_reserved` and `vertices_watermark` do
    /// not exist outside it. This is the half that can be seen from here, and
    /// it is the one that would be broken by the obvious wrong implementation:
    /// marking the hidden body's bricks dirty so they mesh to nothing.
    #[test]
    fn hiding_and_showing_a_body_marks_no_brick_dirty() {
        let mut app = app();
        let second = add_body(&mut app, "Body 2");
        update(&mut app, Message::Frame);

        // Settle: whatever the new body brought with it is meshed, so anything
        // dirty after this point was dirtied by the eye. The second call is
        // what makes the fixture honest -- `perf.dirty_bricks` is written by
        // `remesh_dirty` and by nothing else, so a stale one from the first
        // call would make every assertion below pass or fail for the wrong
        // reason.
        app.remesh_dirty();
        app.remesh_dirty();
        assert_eq!(app.perf.dirty_bricks, 0, "the fixture did not settle");

        for visible in [false, true, false] {
            set_eye(&mut app, second, visible);
            update(&mut app, Message::Frame);
            app.remesh_dirty();
            assert_eq!(
                app.perf.dirty_bricks,
                0,
                "turning the eye {} marked bricks for a remesh",
                if visible { "on" } else { "off" }
            );
        }
    }

    /// **Hiding a FOLDER moves the active body**, and the reason it has to is
    /// that a folder hides the active body TRANSITIVELY: the row's own eye is
    /// still on, so a rule that read the bit rather than the resolved mask
    /// would see nothing wrong and leave the brush pointing at a body that is
    /// not drawn -- where every press carves, sets `unsaved`, pays a remesh and
    /// changes not one pixel.
    #[test]
    fn hiding_a_folder_moves_the_active_body_out_of_it() {
        let mut app = app();
        let inside = add_body(&mut app, "Inside");
        update(&mut app, Message::BodySelected(inside));
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.parent_of(inside).expect("the new folder");

        update(&mut app, Message::BodyVisibilityToggled(folder));

        assert!(app.doc.node(inside).unwrap().visible, "the folder's eye was written into the row");
        assert_ne!(app.doc.active(), inside, "the active body is inside a hidden folder");
        assert!(app.doc.volume(app.doc.active()).is_some());
        assert_agrees(&app, "hiding a folder");

        // And back: the child's own bit was never touched, so re-showing the
        // folder restores exactly what was there.
        update(&mut app, Message::BodyVisibilityToggled(folder));
        let mut shown = Vec::new();
        app.doc.saved_visibility(&mut shown);
        assert!(shown.iter().all(|shown| *shown), "re-showing the folder left something hidden");
    }
}

/// Solo: a MODE, and everything that follows from it being one.
///
/// The whole increment rests on one property -- **solo is never written
/// anywhere** -- and the tests here are the two halves of it. Leaving the mode
/// restores the hand-set eyes bit for bit because none of them was ever
/// touched, and the file cannot carry a trace of it because `project::write`
/// takes a `&Document` and a `&ProjectState` and solo is a field of neither.
/// The second half is a type-system guarantee, and it is asserted anyway: the
/// obvious "tidy" is to move solo onto `Document` or `View` so a call site
/// loses a parameter, and that change compiles.
#[cfg(test)]
mod solo_tests {
    use super::*;
    use brokkr_core::NodeMeta;

    fn update(app: &mut Brokkr, message: Message) {
        drop(app.update(message));
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// A body with geometry in it, so hiding one is a question about a model
    /// rather than about an empty row. Spaced along X so no two overlap.
    fn add_body(app: &mut Brokkr, name: &str, at: f32) -> NodeId {
        let mut volume = Volume::new(app.doc.voxel_size());
        volume.seed_sphere(Vec3::new(at, 0.0, 0.0), 6.0);
        volume.mark_everything_dirty();
        let id = app.doc.add_body(name, volume);
        app.remesh_dirty();
        id
    }

    /// Which rows are DRAWN, by id, straight from the resolver.
    fn drawn(app: &Brokkr) -> Vec<NodeId> {
        let mut shown = Vec::new();
        app.doc.display_visibility(app.solo, &mut shown);
        app.doc
            .nodes()
            .iter()
            .zip(&shown)
            .filter(|(_, shown)| **shown)
            .map(|(node, _)| node.id)
            .collect()
    }

    fn assert_renderer_agrees(app: &Brokkr, after: &str) {
        let mut shown = Vec::new();
        app.doc.display_visibility(app.solo, &mut shown);
        let expected: Vec<NodeId> = app
            .doc
            .nodes()
            .iter()
            .zip(&shown)
            .filter(|(node, shown)| !**shown && node.is_body())
            .map(|(node, _)| node.id)
            .collect();
        assert_eq!(
            app.shared.hidden_snapshot(),
            expected,
            "the renderer is not being told what solo is doing, after {after}"
        );
    }

    /// **The property the whole design exists for.** An arbitrary sequence of
    /// eye toggles, then in and out of solo, and every bit is exactly where the
    /// user left it.
    ///
    /// Every shipped alternative fails this. Photoshop's alt-click eye restores
    /// the previous set only "if you haven't changed anything else"; Plasticity's
    /// manual says Unisolate "does not step back to the previous hierarchical
    /// isolation layer -- everything becomes visible instead"; Blender's Alt+H
    /// is documented as ruining the scene configuration. They all save a vector
    /// and restore it. This saves nothing, so there is nothing to get wrong.
    #[test]
    fn leaving_solo_restores_every_hand_set_eye_bit_for_bit() {
        let mut app = app();
        let first = app.doc.active();
        let second = add_body(&mut app, "Body 2", 30.0);
        let third = add_body(&mut app, "Body 3", 60.0);
        let fourth = add_body(&mut app, "Body 4", 90.0);

        // An arbitrary sequence, through the real message, including a row
        // toggled twice and one toggled three times. The active body is never
        // in it, so nothing here can move the selection or be refused.
        for id in [second, third, second, fourth, third, third] {
            update(&mut app, Message::BodyVisibilityToggled(id));
        }
        let hand_set: Vec<NodeMeta> = app.doc.outline();
        assert_eq!(
            hand_set.iter().filter(|meta| !meta.visible).count(),
            2,
            "the fixture left no hand-set eyes to restore, so this would pass on anything"
        );
        let unsaved_before = app.unsaved;
        let history_before = app.history.stats();

        update(&mut app, Message::SoloEntered(first));
        assert_eq!(drawn(&app), vec![first], "solo is showing more than the row it names");
        assert_renderer_agrees(&app, "entering solo");

        update(&mut app, Message::SoloExited);

        assert_eq!(
            app.doc.outline(),
            hand_set,
            "leaving solo did not put the hand-set eyes back exactly as they were"
        );
        assert_eq!(app.unsaved, unsaved_before, "a view mode dirtied the document");
        assert_eq!(
            app.history.stats().undo_entries,
            history_before.undo_entries,
            "a view mode pushed an undo entry"
        );
        assert!(app.solo.is_none());
        assert_renderer_agrees(&app, "leaving solo");
    }

    /// **A document saved while soloed is byte-identical to one saved without
    /// it**, and the writer is called directly so that the claim is about the
    /// bytes and not about a round trip that might normalise something.
    ///
    /// This is the assertion that would fail if solo were ever moved onto
    /// `Document`, `ProjectState` or `View` -- which is exactly the tidy-up
    /// somebody reaches for when a call site has one parameter too many, and it
    /// is a change that compiles.
    #[test]
    fn saving_while_soloed_writes_the_same_bytes_as_saving_without_it() {
        let mut app = app();
        let first = app.doc.active();
        let second = add_body(&mut app, "Body 2", 30.0);
        add_body(&mut app, "Body 3", 60.0);
        // Something hand-set in there, so "identical" is not identical because
        // every eye happens to be on.
        update(&mut app, Message::BodyVisibilityToggled(second));

        let mut without = Vec::new();
        brokkr_core::project::write(&mut without, &app.doc, &app.project_state())
            .expect("writing to memory");

        update(&mut app, Message::SoloEntered(first));
        assert!(app.solo.is_some(), "the fixture never entered solo");
        let mut with = Vec::new();
        brokkr_core::project::write(&mut with, &app.doc, &app.project_state())
            .expect("writing to memory");

        assert_eq!(with, without, "solo left a trace in the file");

        // And the file that comes back has no trace of it either: the eye the
        // user set is still off, and solo is not a thing the reader can even
        // name.
        let (reopened, _) = brokkr_core::project::read(&mut with.as_slice()).expect("reading back");
        let mut shown = Vec::new();
        reopened.saved_visibility(&mut shown);
        assert_eq!(
            shown.iter().filter(|shown| !**shown).count(),
            1,
            "the reopened document does not hold exactly the one eye that was turned off"
        );
    }

    /// **Soloing a row whose eye is off turns the eye on**, because the
    /// resolver deliberately will not.
    ///
    /// [`brokkr_core::resolve_visibility`] narrows and never widens, and says so
    /// in its own documentation: rewriting a bit the user set is not its
    /// business. Without this handler doing it, "show me only this" would show
    /// nothing at all, with the header indicator naming a row that is not on
    /// screen. It is an ordinary undoable edit and it sets `unsaved`, exactly as
    /// clicking the eye would.
    #[test]
    fn soloing_a_hidden_body_turns_its_eye_on() {
        let mut app = app();
        let second = add_body(&mut app, "Body 2", 30.0);
        update(&mut app, Message::BodySelected(second));
        update(&mut app, Message::BodyVisibilityToggled(second));
        assert!(!app.doc.node(second).expect("the body").visible, "the fixture never hid it");
        app.unsaved = false;
        let entries_before = app.history.stats().undo_entries;

        update(&mut app, Message::SoloEntered(second));

        assert!(app.doc.node(second).expect("the body").visible, "solo showed nothing at all");
        assert_eq!(drawn(&app), vec![second]);
        assert!(app.unsaved, "turning an eye on is a change to the document");
        assert_eq!(
            app.history.stats().undo_entries,
            entries_before + 1,
            "the eye that was turned on is not undoable"
        );
        assert_renderer_agrees(&app, "soloing a hidden body");
    }

    /// The ancestors go with it, or the row is still masked by the folder above
    /// it and the screen is still empty.
    #[test]
    fn soloing_a_body_inside_a_hidden_folder_opens_the_folders_eye_too() {
        let mut app = app();
        let inside = add_body(&mut app, "Inside", 30.0);
        update(&mut app, Message::BodySelected(inside));
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.parent_of(inside).expect("the new folder");
        // Hiding the folder moves the selection out of it, which is the rule
        // `toggle_visibility` documents -- so the fixture puts it back.
        update(&mut app, Message::BodyVisibilityToggled(folder));
        assert!(!app.doc.node(folder).expect("the folder").visible);

        update(&mut app, Message::SoloEntered(inside));

        assert!(
            app.doc.node(folder).expect("the folder").visible,
            "the folder still masks the row"
        );
        assert_eq!(drawn(&app), vec![inside]);
        assert_renderer_agrees(&app, "soloing inside a hidden folder");
    }

    /// **An eye click on a row solo is not showing is refused**, and nothing
    /// moves.
    ///
    /// Without the refusal the row's open eye is a lie: the click turns a bit
    /// off, the screen does not change because solo was already hiding it, the
    /// user clicks again -- and each of those presses sets `unsaved` and arms an
    /// autosave of a multi-gigabyte document, with the whole effect arriving
    /// later, when they leave the mode.
    #[test]
    fn an_eye_click_outside_the_solo_scope_is_refused_and_changes_nothing() {
        let mut app = app();
        let first = app.doc.active();
        let second = add_body(&mut app, "Body 2", 30.0);
        update(&mut app, Message::SoloEntered(first));
        app.unsaved = false;
        let outline_before = app.doc.outline();
        let entries_before = app.history.stats().undo_entries;

        update(&mut app, Message::BodyVisibilityToggled(second));

        assert_eq!(app.doc.outline(), outline_before, "the refused click wrote an eye anyway");
        assert!(!app.unsaved, "the refused click dirtied the document");
        assert_eq!(
            app.history.stats().undo_entries,
            entries_before,
            "the refused click pushed an entry"
        );
        assert!(app.status.contains("solo"), "the refusal does not name the mode: {}", app.status);
        assert!(app.status.contains("Body 2"), "the refusal does not name the row: {}", app.status);

        // And in scope it goes through, so the guard is not simply off.
        update(&mut app, Message::SoloExited);
        update(&mut app, Message::BodyVisibilityToggled(second));
        assert!(!app.doc.node(second).expect("the body").visible, "the eye stopped working");
    }

    /// **Deleting the soloed row clears solo.** A dangling `Some(id)` shows
    /// nothing at all while the mode is still on, and its only exit is an
    /// indicator with no name to draw.
    #[test]
    fn deleting_the_soloed_body_clears_solo() {
        let mut app = app();
        let second = add_body(&mut app, "Body 2", 30.0);
        update(&mut app, Message::BodySelected(second));
        update(&mut app, Message::SoloEntered(second));
        assert_eq!(app.solo, Some(second));

        update(&mut app, Message::BodyDeleted);

        assert!(app.doc.node(second).is_none(), "the fixture never deleted anything");
        assert_eq!(app.solo, None, "solo is still naming a row that is gone");
        assert!(!drawn(&app).is_empty(), "the document went dark");
        assert_renderer_agrees(&app, "deleting the soloed body");
    }

    /// The same for a folder, and for the routes that remove a row without
    /// anybody calling a delete: undo of a group, and redo of the delete.
    ///
    /// This is why the check rides on the visibility pass rather than sitting in
    /// the delete path. "The places that remove a row" is a list, and a list
    /// goes out of date silently.
    #[test]
    fn every_route_that_removes_the_soloed_row_clears_solo() {
        let mut app = app();
        let inside = add_body(&mut app, "Inside", 30.0);
        update(&mut app, Message::BodySelected(inside));
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.parent_of(inside).expect("the new folder");

        // Dissolving the folder takes the soloed row out from under solo.
        update(&mut app, Message::SoloEntered(folder));
        assert_eq!(app.solo, Some(folder));
        update(&mut app, Message::BodyUngrouped);
        assert!(app.doc.node(folder).is_none(), "the fixture never dissolved the folder");
        assert_eq!(app.solo, None, "dissolving the folder left solo dangling");

        // And the undo of a group, which removes a row with no delete anywhere
        // near it.
        update(&mut app, Message::BodyGrouped);
        let again = app.doc.parent_of(app.doc.active()).expect("the second folder");
        update(&mut app, Message::SoloEntered(again));
        assert_eq!(app.solo, Some(again));
        update(&mut app, Message::Undo);
        assert!(app.doc.node(again).is_none(), "the undo did not remove the folder");
        assert_eq!(app.solo, None, "undoing the group left solo dangling");
    }

    /// **A whole-document swap clears solo outright**, and the reason is
    /// different from the one above: the incoming document numbers its rows from
    /// 1 again, so a stale id does not dangle -- it names a real body and hides
    /// everything else.
    #[test]
    fn resetting_clears_solo_rather_than_soloing_whatever_gets_that_id_next() {
        let mut app = app();
        let second = add_body(&mut app, "Body 2", 30.0);
        update(&mut app, Message::BodySelected(second));
        update(&mut app, Message::SoloEntered(second));
        // `reset_sculpt` rather than the message, so the unsaved-work prompt is
        // not what this test is about.
        app.reset_sculpt();

        assert_eq!(app.solo, None, "the new document is soloing an id from the old one");
        assert_eq!(app.doc.body_count(), 1);
        assert_eq!(drawn(&app).len(), 1, "the reset document is hiding its only body");
        assert_renderer_agrees(&app, "a reset");
    }

    /// **Escape leaves solo, and it does so BEFORE it disarms the cut.**
    ///
    /// Both are modes, and one Escape has to pick. It picks the one whose exit
    /// costs nothing: leaving solo puts every body back on screen and changes
    /// not a byte, where disarming the cut throws away something the user
    /// deliberately armed. The second Escape reaches the cut.
    #[test]
    fn escape_leaves_solo_first_and_the_armed_cut_second() {
        let mut app = app();
        let first = app.doc.active();
        add_body(&mut app, "Body 2", 30.0);
        update(&mut app, Message::CutToggled);
        update(&mut app, Message::SoloEntered(first));
        assert!(app.cut_armed && app.solo.is_some(), "the fixture did not arm both modes");

        update(&mut app, Message::MenuClosed);
        assert_eq!(app.solo, None, "escape did not leave solo");
        assert!(app.cut_armed, "escape disarmed the cut on its way past solo");

        update(&mut app, Message::MenuClosed);
        assert!(!app.cut_armed, "the second escape did not reach the cut");
    }

    /// `ctrl+alt+comma` leaves solo as well as turning every eye on -- a
    /// document with every eye on and solo still on shows one subtree, which is
    /// the opposite of what was asked for.
    ///
    /// It is solo's lesser exit precisely because it IS a document change.
    /// Escape exists so that leaving the mode need not be one.
    #[test]
    fn showing_everything_also_leaves_solo() {
        let mut app = app();
        let first = app.doc.active();
        let second = add_body(&mut app, "Body 2", 30.0);
        update(&mut app, Message::BodyVisibilityToggled(second));
        update(&mut app, Message::SoloEntered(first));

        update(&mut app, Message::EveryBodyShown);

        assert_eq!(app.solo, None, "everything is showing except that solo is still on");
        assert_eq!(drawn(&app).len(), 2, "a body is still missing from the screen");
        assert!(app.status.contains("solo"), "the status does not mention it: {}", app.status);

        // And with nothing hidden it still leaves the mode, which is the arm
        // that returns early.
        update(&mut app, Message::SoloEntered(first));
        update(&mut app, Message::EveryBodyShown);
        assert_eq!(app.solo, None, "the early return skipped the exit");
    }

    /// **Solo never vetoes undo.** The refusal that stops an undo changing a
    /// body the user cannot see is evaluated against `saved_visibility`: it is
    /// about an eye the user set and can see in the panel. A transient view
    /// mode turning ctrl+Z off -- with a message calling a body "hidden" whose
    /// eye is plainly open -- is a different thing wearing the same words.
    #[test]
    fn solo_does_not_veto_an_undo_of_a_body_it_is_hiding() {
        let mut app = app();
        let first = app.doc.active();
        let second = add_body(&mut app, "Body 2", 30.0);

        // A real stroke on the first body, driven the way a user would, so
        // there is a `Change::Bricks` naming it in the history.
        update(&mut app, Message::BodySelected(first));
        let probe = app.surface_under(Vec2::new(640.0, 360.0)).expect("the centre hits the model");
        let before = app.doc.volume(first).expect("a live body").sample_world(probe);
        app.on_pointer(PointerEvent::Moved {
            position: iced::Vector::new(640.0, 360.0),
            size: iced::Vector::new(1280.0, 720.0),
        });
        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: iced::Vector::new(640.0, 360.0),
            size: iced::Vector::new(1280.0, 720.0),
        });
        app.on_pointer(PointerEvent::Released { button: PointerButton::Left });
        let after = app.doc.volume(first).expect("a live body").sample_world(probe);
        assert_ne!(after, before, "the fixture carved nothing");

        // Solo the OTHER body, so the one the undo would change is not drawn.
        update(&mut app, Message::BodySelected(second));
        update(&mut app, Message::SoloEntered(second));
        assert!(!drawn(&app).contains(&first), "the fixture is still drawing the body");

        update(&mut app, Message::Undo);

        assert_eq!(
            app.doc.volume(first).expect("a live body").sample_world(probe),
            before,
            "solo refused an undo: {}",
            app.status
        );
    }

    /// **Solo showing nothing is the one state it must never be left in**, and
    /// it is reachable by one keystroke from two directions, so the guarantee
    /// is an invariant on the visibility pass rather than a check at entry.
    ///
    /// Route A is undo. Turning the soloed row's eye on is an ordinary undoable
    /// change and undo is deliberately not vetoed by solo, so ctrl+Z straight
    /// after soloing a hidden row turns that eye back off underneath the mode.
    /// Route B is the soloed row's own eye: it is inside its own subtree, so
    /// the scope guard passes it, and the "there is another visible body" test
    /// that lets the hide through reads `saved_visibility` -- which is right,
    /// and which knows nothing about solo.
    ///
    /// Both end with every row resolving to false: a black viewport, a SOLO
    /// badge naming a row nobody can see and a status line about something
    /// else. Escape recovers it and nothing on screen says so.
    #[test]
    fn soloing_can_never_leave_the_viewport_empty() {
        // Route A: undo the eye that entering solo turned on.
        {
            let mut app = app();
            let second = add_body(&mut app, "Body 2", 30.0);
            update(&mut app, Message::BodyVisibilityToggled(second));
            update(&mut app, Message::SoloEntered(second));
            assert_eq!(drawn(&app), vec![second], "the fixture never entered solo");

            update(&mut app, Message::Undo);

            assert!(!drawn(&app).is_empty(), "undo emptied the viewport: {}", app.status);
            assert_eq!(app.solo, None, "solo is still on with nothing to show");
            assert!(app.status.contains("solo"), "nothing says why: {}", app.status);
            assert_renderer_agrees(&app, "undoing the eye solo turned on");
        }

        // Route B: turn the soloed row's own eye off.
        {
            let mut app = app();
            let first = app.doc.active();
            add_body(&mut app, "Body 2", 30.0);
            update(&mut app, Message::BodySelected(first));
            update(&mut app, Message::SoloEntered(first));
            assert_eq!(drawn(&app), vec![first], "the fixture never entered solo");

            update(&mut app, Message::BodyVisibilityToggled(first));

            assert!(!drawn(&app).is_empty(), "the eye emptied the viewport: {}", app.status);
            assert_eq!(app.solo, None, "solo is still on with nothing to show");
            assert!(app.status.contains("solo"), "nothing says why: {}", app.status);
            assert_renderer_agrees(&app, "hiding the soloed row");
        }
    }

    /// **The renderer is told what solo is doing, after every solo message.**
    ///
    /// `hidden_snapshot()` equals `doc.display_visibility(app.solo)` or a body
    /// is missing from the viewport while its row reads "visible" -- and it
    /// still raycasts and still carves, which is not a failure anyone can
    /// reproduce from a description.
    #[test]
    fn the_renderer_agrees_with_the_document_after_every_solo_message() {
        let mut app = app();
        let first = app.doc.active();
        let second = add_body(&mut app, "Body 2", 30.0);
        let third = add_body(&mut app, "Body 3", 60.0);

        let messages = [
            Message::SoloEntered(second),
            Message::Frame,
            Message::SoloEntered(third),
            Message::BodyVisibilityToggled(third),
            Message::SoloExited,
            Message::BodyVisibilityToggled(first),
            Message::SoloEntered(second),
            Message::EveryBodyShown,
            Message::SoloEntered(first),
            Message::MenuClosed,
            Message::Frame,
        ];
        for message in messages {
            let named = format!("{message:?}");
            update(&mut app, message);
            assert_renderer_agrees(&app, &named);
        }
    }
}

/// What the four whole-document swap sites owe the renderer.
///
/// Increment 6 deleted eleven lines of stale-coordinate marking from
/// `reset_sculpt`, `open_project`, `adopt_import` and `orient` -- loops that
/// collected the outgoing model's brick coordinates and marked them dirty in
/// the incoming volume so they would mesh to nothing and release their pool
/// slices. All four were already dead, and the reason is entirely in this
/// module: all four call `rebuild_everything`, and that asks the renderer to
/// empty the pool, which drops every slot regardless of what key it was under.
///
/// **The reasoning is only as durable as the reset**, and nothing asserted the
/// reset before this. If `rebuild_everything` ever stops asking for one, all
/// four of these functions start leaving the previous model drawn underneath
/// the new one -- and with the loops gone there is nothing else that would
/// have cleared it.
#[cfg(test)]
mod swap_site_tests {
    use super::*;

    /// One whole-document swap, so the four can be driven by one loop and none
    /// of them can be left out of it by accident.
    type Swap = Box<dyn Fn(&mut Brokkr)>;

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// Every whole-document swap empties the pool, and the swap itself is
    /// enough -- no caller has to remember anything extra.
    #[test]
    fn every_whole_document_swap_asks_the_renderer_to_empty_the_pool() {
        let directory = std::env::temp_dir().join(format!("brokkr-swaps-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("saved.brokkr");

        let mut app = app();
        app.save_project(&path);

        // Each case: clear whatever is pending by standing in for the frame
        // that would consume it, do the swap, and assert the request is back.
        let cases: [(&str, Swap); 4] = [
            ("reset", Box::new(|app: &mut Brokkr| app.reset_sculpt())),
            ("open", Box::new(move |app: &mut Brokkr| app.open_project(&path))),
            (
                "import",
                Box::new(|app: &mut Brokkr| {
                    let mut volume = Volume::new(app.doc.voxel_size());
                    volume.seed_sphere(Vec3::ZERO, 12.0);
                    volume.mark_everything_dirty();
                    app.adopt_import(crate::message::Imported {
                        volume,
                        source: std::path::PathBuf::from("fixture.stl"),
                        report: brokkr_core::voxelise::VoxeliseReport::default(),
                        elapsed_ms: 0.0,
                        resting_up: None,
                    });
                }),
            ),
            (
                "orient",
                Box::new(|app: &mut Brokkr| {
                    app.orient(brokkr_core::AxisRotation::taking(
                        brokkr_core::Facing::Front,
                        brokkr_core::Facing::Up,
                    ));
                }),
            ),
        ];

        for (name, swap) in cases {
            app.shared.take_pool_reset_for_tests();
            swap(&mut app);
            assert!(
                app.shared.take_pool_reset_for_tests(),
                "a {name} left the outgoing model's slots in the pool, so its triangles stay on \
                 screen underneath the new document"
            );
        }

        let _ = std::fs::remove_dir_all(&directory);
    }
}

/// The body panel: primitives, the eye, delete, and the element tree that draws
/// them.
///
/// **`panel.rs` had literally zero coverage before this module**, because
/// `view()` was never called by any test in the crate -- so no assertion at any
/// level could see a row panic, a heading vanish, or a widget tree fail to
/// build. Building an `Element` needs no GPU and no window, which is what makes
/// the smoke test below possible at all.
#[cfg(test)]
mod body_panel_tests {
    use super::*;
    use brokkr_core::{MAX_BODIES, NodeMeta, PrimitiveKind};

    fn update(app: &mut Brokkr, message: Message) {
        drop(app.update(message));
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    /// A body with nothing in it, for the cases that need rows rather than
    /// geometry.
    ///
    /// A real primitive at every one of sixty-four rows is a few hundred
    /// megabytes of bricks to draw a list, and the list does not read a voxel.
    fn cheap_body(app: &mut Brokkr, name: &str) -> NodeId {
        let volume = Volume::new(app.doc.voxel_size());
        app.doc.add_body(name, volume)
    }

    fn names(app: &Brokkr) -> Vec<String> {
        app.doc.nodes().iter().map(|node| node.name.clone()).collect()
    }

    /// **The waterline.** Press `+`, pick Cube, and there is a second body in
    /// the document that does not overlap the first.
    #[test]
    fn adding_a_primitive_puts_a_new_body_clear_of_everything_already_there() {
        let mut app = app();
        let camera_before = (app.camera.yaw, app.camera.pitch, app.camera.distance);

        update(&mut app, Message::PrimitiveMenuToggled);
        assert!(app.adding, "the plus button did not open the menu");
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));

        assert!(!app.adding, "the menu stayed open after a choice");
        assert_eq!(app.doc.body_count(), 2, "the cube did not become a body");
        assert_eq!(names(&app)[1], "Cube");
        assert_eq!(app.doc.active(), app.doc.nodes()[1].id, "the new body was not selected");
        assert!(app.unsaved, "adding a body did not mark the document unsaved");

        // The decision that makes the feature demonstrate itself: clear of the
        // model, not at the origin inside it.
        assert!(
            app.doc.overlaps().is_empty(),
            "the cube interpenetrates the ball: {:?}",
            app.doc.overlaps()
        );
        let ball = app.doc.nodes()[0].volume().unwrap().surface_bounds().unwrap();
        let cube = app.doc.nodes()[1].volume().unwrap().surface_bounds().unwrap();
        assert!(cube.0.x > ball.1.x, "the cube's surface box overlaps the ball's along X");

        // The camera is something the user set. A tool that re-frames on every
        // add makes a twelve-body layout impossible to build.
        assert_eq!(
            (app.camera.yaw, app.camera.pitch, app.camera.distance),
            camera_before,
            "adding a body moved the camera"
        );
    }

    /// Each of the three, so a new kind cannot be added without a shape.
    #[test]
    fn every_primitive_kind_lands_as_its_own_body() {
        let mut app = app();
        for kind in PrimitiveKind::ALL {
            update(&mut app, Message::PrimitiveAdded(kind));
        }
        assert_eq!(app.doc.body_count(), 4);
        assert_eq!(names(&app)[1..], ["Cube", "Sphere", "Cylinder"]);
        assert!(app.doc.overlaps().is_empty(), "two primitives were placed inside each other");
    }

    /// One `Change::NodeAdded`, so one ctrl+Z removes it and one ctrl+shift+Z
    /// brings it back.
    #[test]
    fn adding_a_primitive_is_one_undo_entry() {
        let mut app = app();
        let before = app.history_stats.undo_entries;
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Sphere));
        assert_eq!(
            app.history_stats.undo_entries,
            before + 1,
            "adding a body was not exactly one entry"
        );

        update(&mut app, Message::Undo);
        assert_eq!(app.doc.body_count(), 1, "undo did not remove the primitive");
        update(&mut app, Message::Redo);
        assert_eq!(app.doc.body_count(), 2, "redo did not bring it back");
        assert_eq!(names(&app)[1], "Sphere");
    }

    /// Dynamic radius follows the MODEL changing under the brush, never the
    /// selection changing -- and an add does both at once, which is the case
    /// that catches it.
    #[test]
    fn adding_a_primitive_dirties_only_the_new_body_and_leaves_the_brush_alone() {
        let mut app = app();
        update(&mut app, Message::DynamicRadiusToggled(true));
        let radius = app.brush.radius;
        // Drain whatever the fixture left behind, so what is measured is what
        // the add produced.
        app.remesh_dirty();

        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let added = app.doc.nodes()[1].id;

        // `self.dirty` is the batch the add's own remesh consumed, and it is
        // still there afterwards: every pair in it has to name the new body.
        // The ball's bricks did not change, so not one of them may be in it.
        assert!(!app.dirty.is_empty(), "the add dirtied nothing at all");
        assert!(
            app.dirty.iter().all(|(body, _)| *body == added),
            "the add dirtied bricks belonging to a body it did not touch"
        );
        let mut left = Vec::new();
        app.doc.take_dirty(&mut left);
        assert!(left.is_empty(), "something dirtied bricks after the add: {left:?}");
        assert!(
            app.doc.volume(added).is_some_and(|volume| volume.brick_count() > 0),
            "the cube has no bricks"
        );
        assert_eq!(app.brush.radius, radius, "adding a body resized a Dynamic brush");
    }

    /// A cap is a refusal with a line that names it, never a panic and never a
    /// silent no-op. Nothing the interface builds goes through the reader, so
    /// the reader's clamps do not cover this.
    #[test]
    fn adding_past_the_body_cap_is_refused_and_says_so() {
        let mut app = app();
        while app.doc.body_count() < MAX_BODIES {
            let name = format!("Body {}", app.doc.body_count() + 1);
            cheap_body(&mut app, &name);
        }
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));

        assert_eq!(app.doc.body_count(), MAX_BODIES, "the cap let a body through");
        assert!(
            app.status.contains("could not add") && app.status.contains(&MAX_BODIES.to_string()),
            "the refusal does not name the cap: {}",
            app.status
        );
    }

    /// **The ceiling `MAX_BODIES` is not.** A primitive is sized off the
    /// biggest body in the document, so at a fine voxel the very FIRST one can
    /// be tens of gigabytes; sixty-four rows is not a limit anybody reaches
    /// before the machine does. And the refusal has to land before
    /// `primitive::build`, because `build` is the allocation it is refusing.
    ///
    /// The fixture also pins WHICH pool number is read. The pool here has four
    /// times the vertices this cube needs by `vertices_reserved` and half of
    /// what it needs by the watermark, so a guard that took the reserved figure
    /// -- the substitution `GrowthGuard`'s doc comment exists to forbid -- would
    /// wave it through.
    #[test]
    fn a_primitive_the_pool_cannot_hold_is_refused_before_it_is_built() {
        let mut app = app();
        let (_, half) = brokkr_core::primitive::placement(&app.doc, MODEL_RADIUS_MM);
        let (_, vertices) =
            brokkr_core::primitive::cost(PrimitiveKind::Cube, app.doc.voxel_size(), half);
        app.shared.set_stats_for_tests(brokkr_gpu::PoolStats {
            vertices_reserved: 0,
            vertices_watermark: (vertices * 2.0) as u64,
            vertex_capacity: (vertices * 2.5) as u64,
            ..Default::default()
        });
        let before = app.doc.body_count();
        let bricks = app.doc.totals().dense_bricks;

        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));

        assert_eq!(app.doc.body_count(), before, "the cube was added anyway");
        assert_eq!(app.doc.totals().dense_bricks, bricks, "something was allocated regardless");
        assert!(
            app.status.contains("could not add a cube") && app.status.contains("mesh pool"),
            "the refusal does not say what ran out: {}",
            app.status
        );
        // Established pattern in this codebase: a refusal names the size that
        // WOULD work, rather than leaving the user to bisect.
        assert!(app.status.contains("mm)"), "the refusal names no workable size: {}", app.status);
    }

    /// Clicking one row's eye must leave the active body exactly where it was.
    ///
    /// This is the message-level half. The other half is that the eye's button
    /// captures the press before the row's `mouse_area` can see it, which is a
    /// widget property and is documented on `Brokkr::body_row`.
    #[test]
    fn toggling_another_rows_eye_does_not_change_the_active_body() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let first = app.doc.nodes()[0].id;
        let cube = app.doc.nodes()[1].id;
        update(&mut app, Message::BodySelected(cube));

        update(&mut app, Message::BodyVisibilityToggled(first));
        assert!(!app.doc.node(first).unwrap().visible, "the eye did not go off");
        assert_eq!(app.doc.active(), cube, "hiding another row moved the selection");
    }

    /// Hiding the body edits land on has to move the selection somewhere the
    /// user can see, or every press afterwards reports that it is hidden.
    #[test]
    fn hiding_the_active_body_moves_the_selection_to_one_that_is_visible() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let first = app.doc.nodes()[0].id;
        let cube = app.doc.nodes()[1].id;
        assert_eq!(app.doc.active(), cube);

        update(&mut app, Message::BodyVisibilityToggled(cube));
        assert_eq!(app.doc.active(), first, "the selection stayed on a body nobody can see");
        assert!(app.status.contains("is now selected"), "nothing said why: {}", app.status);
    }

    /// ...and when there is nowhere for it to go, the hide is refused rather
    /// than leaving the application with nothing to sculpt.
    #[test]
    fn hiding_the_last_visible_body_is_refused_and_changes_nothing() {
        let mut app = app();
        let only = app.doc.active();
        let entries = app.history_stats.undo_entries;

        update(&mut app, Message::BodyVisibilityToggled(only));

        assert!(app.doc.node(only).unwrap().visible, "the last visible body was hidden");
        assert!(!app.unsaved, "a refused hide marked the document unsaved");
        assert_eq!(app.history_stats.undo_entries, entries, "a refused hide recorded an entry");
        assert!(app.status.contains("cannot hide"), "the refusal said nothing: {}", app.status);
    }

    /// `ctrl+comma` acts on the active row, and `ctrl+alt+comma` is the way back
    /// from having hidden things and lost track of what.
    #[test]
    fn the_visibility_chords_toggle_the_active_row_and_reveal_everything() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let first = app.doc.nodes()[0].id;
        let cube = app.doc.nodes()[1].id;

        assert!(
            matches!(
                crate::viewport::shortcut(",", true, false, false),
                Some(Message::ActiveBodyVisibilityToggled)
            ),
            "ctrl+comma spells nothing"
        );
        assert!(
            matches!(
                crate::viewport::shortcut(",", true, false, true),
                Some(Message::EveryBodyShown)
            ),
            "ctrl+alt+comma spells nothing"
        );

        // The active body is the cube, and hiding it moves the selection.
        update(&mut app, Message::ActiveBodyVisibilityToggled);
        assert!(!app.doc.node(cube).unwrap().visible);
        assert_eq!(app.doc.active(), first);

        update(&mut app, Message::EveryBodyShown);
        assert!(
            app.doc.nodes().iter().all(|node| node.visible),
            "ctrl+alt+comma left something hidden"
        );
        // One entry for the reveal, so one ctrl+Z puts the hide back.
        update(&mut app, Message::Undo);
        assert!(!app.doc.node(cube).unwrap().visible, "undoing the reveal did not restore the eye");
    }

    /// Nothing to reveal is said out loud rather than pushed onto the stack: an
    /// empty entry costs the user a real undo.
    #[test]
    fn revealing_when_nothing_is_hidden_records_nothing() {
        let mut app = app();
        let entries = app.history_stats.undo_entries;
        update(&mut app, Message::EveryBodyShown);
        assert_eq!(app.history_stats.undo_entries, entries);
        assert!(app.status.contains("already showing"), "{}", app.status);
    }

    /// Delete takes the active body, and one ctrl+Z brings it back whole --
    /// bricks, name, eye and position in the list.
    #[test]
    fn deleting_a_body_is_undoable() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let cube = app.doc.active();
        let bricks = app.doc.volume(cube).unwrap().brick_count();

        update(&mut app, Message::BodyDeleted);
        assert_eq!(app.doc.body_count(), 1, "the delete did nothing");
        assert!(app.status.contains("deleted Cube"), "{}", app.status);

        update(&mut app, Message::Undo);
        assert_eq!(app.doc.body_count(), 2, "undo did not bring the body back");
        assert_eq!(names(&app)[1], "Cube");
        assert_eq!(
            app.doc.volume(cube).map(|volume| volume.brick_count()),
            Some(bricks),
            "the body came back with a different number of bricks"
        );
    }

    /// **A delete owes the renderer two things, and neither is visible on
    /// screen if it is missing.**
    ///
    /// The forget releases the deleted body's slots -- without it a sliver of
    /// it is drawn forever, holding pool space, with no counter moving. The
    /// whole-document remesh is the other half: `MeshPool::forget_body` clears
    /// the pool-full banner when it frees space, and a brick the pool refused
    /// while it was full is long gone from the dirty set, so freeing the space
    /// without re-offering everything takes the warning down and leaves the
    /// geometry missing.
    #[test]
    fn a_delete_tells_the_renderer_to_drop_the_body_and_re_offers_everything_left() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let cube = app.doc.active();
        let survivor = app.doc.nodes()[0].id;
        // Whatever the add queued is not what is being measured.
        let _ = app.shared.take_forgotten_for_tests();

        update(&mut app, Message::BodyDeleted);

        assert_eq!(
            app.shared.take_forgotten_for_tests(),
            vec![cube],
            "the deleted body's slots were left in the pool"
        );
        assert!(
            app.dirty.iter().any(|(body, _)| *body == survivor),
            "the delete did not re-offer the surviving body, so a brick the pool refused while \
             it was full stays missing with the banner gone"
        );
    }

    /// **The cost of placing a primitive off the origin, paid.**
    ///
    /// [`MIRROR_CENTRE`] is the lattice origin, so a dent carved into a body at
    /// x = +80 has its twin written at x = -80: free-floating geometry in empty
    /// space that exports as an extra shell no slicer can print. The first body
    /// a user can create off-origin is a primitive, which makes this the
    /// increment where that stops being hypothetical -- and selecting it is what
    /// turns the mirror off and says why.
    #[test]
    fn adding_a_primitive_clear_of_the_model_turns_off_a_mirror_it_cannot_use() {
        let mut app = app();
        update(&mut app, Message::SymmetryAxisToggled(brokkr_core::MirrorAxis::X));
        assert!(app.symmetry.axis(brokkr_core::MirrorAxis::X), "the ball straddles X");

        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));

        assert!(
            !app.symmetry.axis(brokkr_core::MirrorAxis::X),
            "X mirroring stayed on for a cube sitting entirely to one side of the plane, so \
             every stroke on it would write a twin into empty space"
        );
        assert!(app.status.contains("empty space"), "the refusal said nothing: {}", app.status);
    }

    /// A document always holds one body, so the last one cannot go. Refused
    #[test]
    fn deleting_the_last_body_is_refused() {
        let mut app = app();
        update(&mut app, Message::BodyDeleted);
        assert_eq!(app.doc.body_count(), 1);
        assert!(app.status.contains("cannot delete the last body"), "{}", app.status);
        assert!(app.pending_delete.is_none(), "the last body raised a prompt");
    }

    /// A delete big enough that undo may not hold it asks first, names the size,
    /// and on Cancel changes nothing.
    ///
    /// The threshold is a field rather than the constant read in place
    /// precisely so this test can exist: at the real 512 MB the fixture would
    /// have to allocate half a gigabyte of bricks.
    #[test]
    fn a_large_delete_asks_first_and_cancelling_changes_nothing() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let cube = app.doc.active();
        let bytes = app.doc.volume(cube).unwrap().stats().resident_bytes;
        assert!(bytes > 0, "the fixture body costs nothing, so nothing can be over the threshold");
        app.delete_prompt_bytes = bytes;

        update(&mut app, Message::BodyDeleted);
        let pending = app.pending_delete.as_ref().expect("a large delete did not ask");
        assert_eq!(pending.id, cube);
        assert_eq!(pending.bytes, bytes);
        assert_eq!(app.doc.body_count(), 2, "the prompt deleted the body anyway");
        assert!(app.modal_open(), "the delete prompt is not modal, so a press behind it sculpts");

        update(&mut app, Message::BodyDeleteCancelled);
        assert_eq!(app.doc.body_count(), 2, "cancelling deleted the body");
        assert!(app.pending_delete.is_none());

        // ...and confirming goes through.
        update(&mut app, Message::BodyDeleted);
        update(&mut app, Message::BodyDeleteConfirmed);
        assert_eq!(app.doc.body_count(), 1, "confirming did not delete");
    }

    // --- merge down ------------------------------------------------------------

    /// The active body, the one below it, and the upper of the two selected --
    /// which is the only arrangement merge down does anything in.
    ///
    /// Two primitives rather than two `cheap_body` rows: a merge with no voxels
    /// on either side would pass every assertion about the list while doing
    /// nothing to a field.
    fn two_to_merge() -> (Brokkr, NodeId, NodeId) {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let upper = app.doc.nodes()[0].id;
        let lower = app.doc.nodes()[1].id;
        update(&mut app, Message::BodySelected(upper));
        // Adding the cube dirtied the document; the merge under test is what
        // every `unsaved` assertion below is about.
        app.unsaved = false;
        (app, upper, lower)
    }

    /// **The whole feature from the user's chair**: press the button and the two
    /// bodies are one, where the upper one stood, holding both fields.
    #[test]
    fn merging_down_consumes_the_source_and_leaves_the_target_where_it_stood() {
        let (mut app, upper, lower) = two_to_merge();
        let upper_bricks = app.doc.volume(upper).expect("a field").brick_count();
        let lower_bricks = app.doc.volume(lower).expect("a field").brick_count();

        update(&mut app, Message::BodyMergedDown);

        assert_eq!(app.doc.body_count(), 1, "the merge did nothing: {}", app.status);
        assert!(app.doc.node(upper).is_none(), "the source row survived");
        assert_eq!(app.doc.index_of(lower), Some(0), "the result is not where the source stood");
        assert_eq!(app.doc.active(), lower, "the result was not selected");
        assert!(app.unsaved, "merging did not mark the document unsaved");
        // The two primitives are placed clear of one another, so the union holds
        // every brick of both.
        assert_eq!(
            app.doc.volume(lower).expect("a field").brick_count(),
            upper_bricks + lower_bricks,
            "the merged body is not the union of the two"
        );
        assert!(app.status.contains("bricks changed"), "the merge said nothing: {}", app.status);
    }

    /// One ctrl+Z, and both bodies are back. A merge is one gesture, so half of
    /// it coming back would be a document nothing downstream is written for.
    #[test]
    fn one_undo_takes_a_merge_apart_again() {
        let (mut app, upper, lower) = two_to_merge();
        let before = app.doc.volume(lower).expect("a field").brick_count();
        let entries = app.history_stats.undo_entries;

        update(&mut app, Message::BodyMergedDown);
        assert_eq!(app.history_stats.undo_entries, entries + 1, "a merge is not one entry");

        update(&mut app, Message::Undo);
        assert_eq!(app.doc.body_count(), 2, "undo did not bring the consumed body back");
        assert_eq!(names(&app), ["Body 1", "Cube"], "the row came back in the wrong place");
        assert_eq!(
            app.doc.volume(lower).expect("a field").brick_count(),
            before,
            "undo left the target holding the source's bricks"
        );
        assert!(app.doc.node(upper).is_some(), "the source row did not come back");
    }

    /// The bottom of a list has nothing below it. Refused by name, with the
    /// document untouched -- never a button that silently does nothing.
    #[test]
    fn merging_the_bottom_body_is_refused_by_name_and_changes_nothing() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let entries = app.history_stats.undo_entries;
        app.unsaved = false;

        update(&mut app, Message::BodyMergedDown);

        assert_eq!(app.doc.body_count(), 2, "the bottom body merged into something");
        assert_eq!(app.history_stats.undo_entries, entries, "a refused merge recorded an entry");
        assert!(!app.unsaved, "a refused merge dirtied the document");
        assert!(
            app.status.contains("could not merge Cube") && app.status.contains("no body below"),
            "the refusal does not say why: {}",
            app.status
        );
    }

    /// A folder on the next line is not a body. Merging into a container is
    /// ZBrush's MergeVisible and its universal "the button did nothing"
    /// reaction; naming the folder is the whole difference.
    #[test]
    fn merging_into_a_folder_below_is_refused_and_names_it() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        // Wrap the lower body, so a folder row sits directly below the upper.
        update(&mut app, Message::BodyGrouped);
        let upper = app.doc.nodes()[0].id;
        update(&mut app, Message::BodySelected(upper));

        update(&mut app, Message::BodyMergedDown);

        assert_eq!(app.doc.body_count(), 2, "a body was merged into a folder");
        assert!(
            app.status.contains("Group 1 is a folder"),
            "the refusal does not name the folder: {}",
            app.status
        );
    }

    /// A merge whose one entry is bigger than history can be relied on to keep
    /// asks first, names the size, and on Cancel changes nothing.
    ///
    /// The threshold is a field rather than the constant read in place for the
    /// reason the delete prompt's own test gives: at the real 512 MB the fixture
    /// would have to allocate half a gigabyte of bricks.
    #[test]
    fn a_large_merge_asks_first_and_cancelling_changes_nothing() {
        let (mut app, upper, _) = two_to_merge();
        let plan = app.doc.merge_plan(upper).expect("there is a body below");
        assert!(plan.bytes() > 0, "the fixture costs nothing, so nothing can be over a threshold");
        app.merge_prompt_bytes = plan.bytes();

        update(&mut app, Message::BodyMergedDown);
        let pending = app.pending_merge.as_ref().expect("a large merge did not ask");
        assert_eq!(pending.source, upper);
        assert_eq!(pending.bytes, plan.bytes());
        assert_eq!(pending.stroke_bytes, plan.stroke_bytes);
        assert_eq!(pending.reclaim_bytes, plan.reclaim_bytes);
        assert_eq!(app.doc.body_count(), 2, "the prompt merged anyway");
        assert!(app.modal_open(), "the merge prompt is not modal, so a press behind it sculpts");

        update(&mut app, Message::BodyMergeCancelled);
        assert_eq!(app.doc.body_count(), 2, "cancelling merged the bodies");
        assert!(!app.unsaved, "cancelling dirtied the document");
        assert!(app.pending_merge.is_none());

        // Escape is the other way out, and it means the same harmless thing.
        update(&mut app, Message::BodyMergedDown);
        assert!(app.pending_merge.is_some());
        update(&mut app, Message::MenuClosed);
        assert!(app.pending_merge.is_none(), "escape left the merge prompt up");
        assert_eq!(app.doc.body_count(), 2, "escape merged the bodies");

        // ...and confirming goes through.
        update(&mut app, Message::BodyMergedDown);
        update(&mut app, Message::BodyMergeConfirmed);
        assert_eq!(app.doc.body_count(), 1, "confirming did not merge");
    }

    /// **A merge owes the renderer the same two things a delete does**, and for
    /// the same reasons: the consumed body's slots have to be released, or a
    /// sliver of it is drawn forever holding pool space, and everything left has
    /// to be re-offered, or a brick the pool refused while it was full stays
    /// missing after the banner comes down.
    #[test]
    fn a_merge_tells_the_renderer_to_drop_the_consumed_body_and_re_offers_the_rest() {
        let (mut app, upper, lower) = two_to_merge();
        // Whatever the add and the selection queued is not what is measured.
        let _ = app.shared.take_forgotten_for_tests();

        update(&mut app, Message::BodyMergedDown);

        assert_eq!(
            app.shared.take_forgotten_for_tests(),
            vec![upper],
            "the consumed body's slots were left in the pool"
        );
        assert!(
            app.dirty.iter().any(|(body, _)| *body == lower),
            "the merge did not re-offer the surviving body"
        );
    }

    /// Merging while soloed keeps the mode. **`rebuild_everything` clears solo
    /// and the plan for this increment asked for it; that call was written
    /// before `forget_body` landed and it would drop a view mode a merge has no
    /// business touching.** Solo is cleared only when the row it names stops
    /// existing, which `forget_a_vanished_solo` already does on the update pass.
    #[test]
    fn merging_a_body_that_is_not_soloed_leaves_solo_alone() {
        let (mut app, upper, lower) = two_to_merge();
        update(&mut app, Message::SoloEntered(upper));
        assert_eq!(app.solo, Some(upper), "the fixture never entered solo");

        // Solo the target, then merge the source into it, so the mode names a
        // row that survives.
        update(&mut app, Message::BodySelected(lower));
        update(&mut app, Message::SoloEntered(lower));
        update(&mut app, Message::BodySelected(upper));
        assert_eq!(app.solo, Some(lower), "the fixture is not soloing the survivor");

        update(&mut app, Message::BodyMergedDown);

        assert_eq!(app.doc.body_count(), 1, "the merge did nothing: {}", app.status);
        assert_eq!(app.solo, Some(lower), "the merge dropped solo");
    }

    /// And the other half of the same rule: merging the SOLOED body away leaves
    /// the mode, because a solo naming a row that no longer exists shows nothing
    /// at all while its only exit has no name to draw. It is
    /// `forget_a_vanished_solo` on the update pass that does it, which is where
    /// increment 13 put every "the row stopped existing" case by name --
    /// including this one, before it could be written.
    #[test]
    fn merging_the_soloed_body_away_leaves_solo() {
        let (mut app, upper, _) = two_to_merge();
        update(&mut app, Message::SoloEntered(upper));
        assert_eq!(app.solo, Some(upper), "the fixture never entered solo");

        update(&mut app, Message::BodyMergedDown);

        assert_eq!(app.doc.body_count(), 1, "the merge did nothing: {}", app.status);
        assert!(app.solo.is_none(), "solo still names a row that was merged away");
    }

    /// The thumbnail switch is session state. Nothing about it is written to the
    /// file, so by the rule that governs that it must not dirty the document.
    #[test]
    fn the_thumbnail_switch_does_not_dirty_the_document() {
        let mut app = app();
        assert!(app.thumbnails, "thumbnails default off");
        update(&mut app, Message::ThumbnailsToggled);
        assert!(!app.thumbnails);
        assert!(!app.unsaved, "turning the pictures off marked the sculpt unsaved");
    }

    // --- duplicate -----------------------------------------------------------

    /// **The whole gesture from the user's chair**: the copy appears directly
    /// under the row it came from, carrying its field, and becomes the body
    /// edits land on.
    ///
    /// Every one of the four assertions about naming and selection is ZBrush
    /// inverted. There, duplicating "object" renames the ORIGINAL to "object1",
    /// hands the copy the original's name, and leaves the original selected --
    /// which users had to work out from their own undo history, and which
    /// breaks GoZ round-trips because there the name is the identity key.
    #[test]
    fn a_duplicate_lands_directly_below_its_original_and_becomes_the_active_body() {
        let mut app = app();
        let original = app.doc.active();
        let bricks = app.doc.volume(original).expect("the default body").brick_count();
        // A third row, so "directly below" is a real claim rather than "at the
        // end of the list" wearing a different name.
        cheap_body(&mut app, "Last");

        update(&mut app, Message::BodyDuplicated);

        assert_eq!(app.doc.body_count(), 3, "the copy did not become a body");
        assert_eq!(names(&app), ["Body 1", "Body 1 copy", "Last"]);
        let copy = app.doc.nodes()[1].id;
        assert_ne!(copy, original, "the copy took the original's id");
        assert_eq!(app.doc.active(), copy, "the copy did not become the active body");
        assert!(app.unsaved, "duplicating did not mark the document unsaved");
        assert_eq!(
            app.doc.volume(copy).expect("the copy holds a field").brick_count(),
            bricks,
            "the copy does not hold the original's field"
        );
        assert_eq!(
            app.doc.volume(original).expect("the original is still here").brick_count(),
            bricks,
            "duplicating moved the original's bricks instead of copying them"
        );
    }

    /// **The same button pressed on a row that lives inside a folder**, which
    /// is the case the panel's own smoke test cannot see: there the active row
    /// happens to be at the top level when duplicate runs, so a copy that lands
    /// at depth 0 looks exactly like a copy that lands beside its source.
    ///
    /// It is not the same. A depth-0 copy in the middle of a folder's preorder
    /// run ends that run at the copy, so every sibling below it leaves the
    /// folder -- silently in a release build, and as a failed fold in a debug
    /// one. The sibling is here for precisely that reason: without a row after
    /// the copy the mistake is invisible.
    #[test]
    fn a_duplicate_made_inside_a_folder_stays_inside_it_and_keeps_its_siblings() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let cube = app.doc.active();
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.parent_of(cube).expect("the new folder");
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Sphere));
        let sphere = app.doc.active();
        update(&mut app, Message::BodyMovedToFolder(Some(folder)));
        update(&mut app, Message::BodySelected(cube));
        assert_eq!(
            app.doc.subtree_body_count(folder),
            2,
            "the fixture is not the shape under test"
        );

        update(&mut app, Message::BodyDuplicated);

        let copy = app.doc.active();
        assert_ne!(copy, cube, "nothing was duplicated: {}", app.status);
        assert_eq!(app.doc.parent_of(copy), Some(folder), "the copy landed outside the folder");
        assert_eq!(
            app.doc.parent_of(sphere),
            Some(folder),
            "the sibling below the copy fell out of the folder"
        );
        assert_eq!(app.doc.subtree_body_count(folder), 3);
    }

    /// **A body that never reaches the GPU is in the document, exports
    /// correctly, and is invisible for the rest of the session.**
    ///
    /// `perf.dirty_bricks` and deliberately not `perf.remesh_ms`: the remesh
    /// returns early on an empty dirty set without writing the timing, so a
    /// check written against the milliseconds passes on exactly the broken
    /// build it exists to catch.
    #[test]
    fn the_copy_reaches_the_gpu() {
        let mut app = app();
        update(&mut app, Message::BodyDuplicated);

        let copy = app.doc.active();
        let bricks = app.doc.volume(copy).expect("the copy holds a field").brick_count();
        assert!(bricks > 0, "the fixture body has no bricks, so this asserts nothing");
        assert!(
            app.perf.dirty_bricks >= bricks,
            "the copy holds {bricks} bricks and the remesh saw only {}",
            app.perf.dirty_bricks
        );
    }

    /// Duplicate, then ctrl+Z, and the file that would be written is the file
    /// that would have been written before -- byte for byte.
    ///
    /// Byte-identical rather than "the row is gone", because a row removed
    /// while the document's id counter, its active row or its node order moved
    /// underneath is a document that reads back as something else. The file is
    /// the only place all of that is visible at once.
    ///
    /// `assert!` and not `assert_eq!` throughout: the fixture's brick stream
    /// runs to megabytes, and a failure that prints both copies of it is a
    /// failure nobody can read.
    #[test]
    fn duplicating_and_undoing_leaves_the_document_byte_identical() {
        let mut app = app();
        let before = written(&app);

        update(&mut app, Message::BodyDuplicated);
        assert_eq!(app.doc.body_count(), 2, "the fixture did not actually duplicate");
        assert!(written(&app) != before, "the copy left no trace in the file");

        update(&mut app, Message::Undo);
        let after = written(&app);
        assert!(
            after == before,
            "undoing the duplicate did not restore the document: {} bytes against {}",
            after.len(),
            before.len()
        );
    }

    /// **The residual, said out loud: undoing a duplicate of a row that is not
    /// the last one leaves the selection on the row BELOW, not on the original.**
    ///
    /// It is `Document::remove`'s documented policy and not an accident there:
    /// a row taken out of the list is replaced on screen by the one below it,
    /// which is the right answer for the delete that policy was written for,
    /// and its header says restoring the selection is the business of whatever
    /// built the undo entry. `Change::NodeAdded` carries no room to do that --
    /// it is a position and an id, and the plan's own budget for it is "about
    /// 8 bytes of history".
    ///
    /// Fixing it means widening that variant with the row that was active
    /// before the add, and giving its inverse the symmetric behaviour on redo,
    /// which changes what undoing a DELETE selects as well. That is a change to
    /// the undo model rather than to duplicate, so it is named here instead of
    /// smuggled in: when someone makes it, this test fails and this comment is
    /// what they should read.
    ///
    /// Nothing about the geometry is at stake -- the document is otherwise
    /// restored exactly, which the test above pins.
    #[test]
    fn undoing_a_duplicate_from_the_middle_of_the_list_leaves_the_selection_below_it() {
        let mut app = app();
        let original = app.doc.active();
        let last = cheap_body(&mut app, "Last");

        update(&mut app, Message::BodyDuplicated);
        update(&mut app, Message::Undo);

        assert_eq!(names(&app), ["Body 1", "Last"], "the copy is still in the document");
        assert_eq!(
            app.doc.active(),
            last,
            "the selection no longer lands below the removed row -- read this test's header"
        );
        assert_ne!(app.doc.active(), original);
    }

    /// The document a save would write right now.
    fn written(app: &Brokkr) -> Vec<u8> {
        let mut bytes = Vec::new();
        brokkr_core::project::write(&mut bytes, &app.doc, &brokkr_core::ProjectState::default())
            .expect("writing the sculpt failed");
        bytes
    }

    /// At the cap, duplicate is a refusal with the number in it -- never a
    /// panic and never a silent no-op. The reader's own clamps do not cover
    /// this: nothing built by the interface goes through the reader.
    #[test]
    fn duplicate_at_the_body_cap_is_refused_by_name_and_changes_nothing() {
        let mut app = app();
        while app.doc.body_count() < MAX_BODIES {
            let name = format!("Body {}", app.doc.body_count() + 1);
            cheap_body(&mut app, &name);
        }
        let active = app.doc.active();
        let entries = app.history_stats.undo_entries;

        update(&mut app, Message::BodyDuplicated);

        assert_eq!(app.doc.body_count(), MAX_BODIES, "a body was added past the cap");
        assert_eq!(app.doc.active(), active, "the refusal moved the selection");
        assert_eq!(app.history_stats.undo_entries, entries, "the refusal pushed an undo entry");
        assert!(!app.unsaved, "a refusal marked the document unsaved");
        assert!(
            app.status.contains("could not duplicate") && app.status.contains("64"),
            "the refusal does not name the cap: {}",
            app.status
        );
    }

    /// A name at the file format's limit still duplicates, and what comes back
    /// is a name the file can hold rather than one it will repair to "Body 1"
    /// on the next open.
    ///
    /// The " copy" is what gets cut, and that is the right end to lose: the
    /// alternative is a copy whose name is not a prefix of anything the user
    /// typed. Two rows may share a name -- `NodeId` is the identity and nothing
    /// downstream keys off one.
    #[test]
    fn a_copy_of_a_full_length_name_still_fits_the_file() {
        let mut app = app();
        let id = app.doc.active();
        let full = "0123456789abcdef0123456789abcdef";
        assert_eq!(full.len(), brokkr_core::MAX_NAME_BYTES);
        let meta = NodeMeta { name: full.to_string(), ..app.doc.meta(id).expect("the row") };
        app.doc.set_meta(&meta);

        update(&mut app, Message::BodyDuplicated);

        let copy = app.doc.active();
        let name = app.doc.node(copy).expect("the copy").name.clone();
        assert_eq!(name.len(), brokkr_core::MAX_NAME_BYTES);
        assert_eq!(name, full, "the copy's name is not what the file will hold");
    }

    /// **A copy the pool cannot hold is refused before one brick is copied**,
    /// and the refusal names no size, because a copy has none to offer.
    ///
    /// The 765 MB dragon is 6,120 dense bricks and 1.53 GiB of memory traffic
    /// (`Volume::duplicated`), so a refusal that arrives after the allocation
    /// is not a refusal.
    ///
    /// **The copy counter is what asserts that half, and it is not decoration.**
    /// The document's body count and brick total do NOT distinguish the two
    /// orderings: a copy that is built and then dropped on the refusal never
    /// enters the document either, so both totals hold whichever side of the
    /// guard the allocation happens on. Hoisting `Volume::duplicated` above the
    /// guard -- which reads as a tidy-up, because it would let `bytes` be
    /// measured off the copy -- passes every other assertion here.
    ///
    /// The fixture also pins WHICH pool number the guard reads, in both
    /// directions. The copy needs 4M vertices; the pool has 2M behind its
    /// watermark and 6M behind its live reservation. A guard taking
    /// `capacity - reserved` -- the substitution `GrowthGuard`'s header exists
    /// to forbid, and one this project has shipped twice -- would wave this
    /// through and then overflow, because a duplicate empties nothing and it is
    /// the bump pointer that runs out.
    #[test]
    fn a_copy_the_pool_cannot_hold_is_refused_before_a_brick_is_copied() {
        let mut app = app();
        app.shared.set_stats_for_tests(brokkr_gpu::PoolStats {
            vertices_reserved: 4_000_000,
            vertices_watermark: 8_000_000,
            vertex_capacity: 10_000_000,
            ..Default::default()
        });
        let bodies = app.doc.body_count();
        let bricks = app.doc.totals().dense_bricks;
        let copies = brokkr_core::volume::copies_made_on_this_thread();

        update(&mut app, Message::BodyDuplicated);

        assert_eq!(
            brokkr_core::volume::copies_made_on_this_thread(),
            copies,
            "the field was copied and only then refused -- read this test's header"
        );
        assert_eq!(app.doc.body_count(), bodies, "the copy was made anyway");
        assert_eq!(app.doc.totals().dense_bricks, bricks, "something was allocated regardless");
        assert!(!app.unsaved, "a refusal marked the document unsaved");
        assert!(
            app.status.contains("could not duplicate Body 1")
                && app.status.contains("mesh pool")
                && app.status.contains("2.0M left"),
            "the refusal does not say what ran out and how much is left: {}",
            app.status
        );
        assert!(
            !app.status.contains('%'),
            "duplicate has no size lever, so the refusal must offer no size: {}",
            app.status
        );
    }

    /// **The vertex ceiling judges the BODY being copied, not the document.**
    ///
    /// Every other duplicate test runs on a one-body document, where the body's
    /// share of the document's resident bytes is degenerately 1.0 and the
    /// apportionment in `no_room_to_duplicate` is invisible -- a constant 1.0,
    /// an inverted ratio, or no apportionment at all all pass. This one is the
    /// only thing that reads it, so read its header before touching that line.
    ///
    /// The fixture is two bodies of deliberately different size and ONE pool,
    /// with the headroom set between what the two copies cost. The small one's
    /// copy must be admitted and the large one's refused. The consequences of
    /// each way of getting it wrong:
    ///
    /// * a constant share judges a small body's copy against the WHOLE
    ///   document's reservation and refuses it, quoting a vertex figure many
    ///   times too large -- "it needs about 8M vertices" for a body that needs
    ///   one;
    /// * an inverted share admits a copy that overflows the pool, and an
    ///   overflow is silently missing geometry, which is the whole reason
    ///   `GrowthGuard` exists.
    ///
    /// The large body is refused FIRST and the small one second, because
    /// admitting a copy grows `doc_stats` and every share after it moves. The
    /// order is the fixture, not a preference.
    #[test]
    fn the_vertex_ceiling_is_apportioned_to_the_body_and_not_to_the_document() {
        let mut app = app();
        let mut large = Volume::new(1.0);
        large.seed_sphere(Vec3::ZERO, 24.0);
        let mut small = Volume::new(1.0);
        // Inside a single brick: the point of the fixture is that the two
        // bodies cost visibly different amounts.
        small.seed_sphere(Vec3::new(112.0, 16.0, 16.0), 3.0);
        let mut doc = Document::from_volume(large);
        let small_id = doc.add_body("Small", small);
        let large_id = doc.nodes()[0].id;
        app.doc = doc;
        app.rebuild_everything();

        let bytes = |app: &Brokkr, id| app.doc.volume(id).expect("the body").stats().resident_bytes;
        let large_bytes = bytes(&app, large_id) as f64;
        let small_bytes = bytes(&app, small_id) as f64;
        let total = app.doc_stats.resident_bytes as f64;
        assert!(
            small_bytes * 3.0 < large_bytes,
            "the fixture's two bodies must differ enough to tell their shares apart: \
             {small_bytes} against {large_bytes}"
        );

        // Halfway between the two copies' costs, so that exactly one of them
        // fits and the test says which by arithmetic rather than by a constant
        // someone would have to re-derive.
        let reserved: u64 = 8_000_000;
        let cost = |body: f64| reserved as f64 * (body / total);
        let headroom = (cost(small_bytes) + cost(large_bytes)) / 2.0;
        assert!(cost(small_bytes) < headroom && headroom < cost(large_bytes));
        let capacity: u64 = 20_000_000;
        app.shared.set_stats_for_tests(brokkr_gpu::PoolStats {
            vertices_reserved: reserved,
            vertices_watermark: capacity - headroom as u64,
            vertex_capacity: capacity,
            ..Default::default()
        });

        update(&mut app, Message::BodySelected(large_id));
        update(&mut app, Message::BodyDuplicated);
        assert_eq!(app.doc.body_count(), 2, "the large body's copy did not fit and was made");
        assert!(
            app.status.contains("could not duplicate") && app.status.contains("mesh pool"),
            "the large body's copy was refused for the wrong reason: {}",
            app.status
        );

        update(&mut app, Message::BodySelected(small_id));
        update(&mut app, Message::BodyDuplicated);
        assert_eq!(
            app.doc.body_count(),
            3,
            "the small body's copy fits in the same pool and was refused: {}",
            app.status
        );
    }

    /// A cut drawn across the viewport crosses every body the user can see and
    /// leaves the hidden one bit-identical, in ONE undo entry.
    #[test]
    fn a_cut_leaves_a_hidden_body_it_crosses_untouched() {
        let mut app = app();
        app.camera.yaw = 0.0;
        app.camera.pitch = 0.0;
        app.publish_camera();

        let mut second = Volume::new(app.doc.voxel_size());
        second.seed_sphere(Vec3::new(10.0, 0.0, 0.0), MODEL_RADIUS_MM * 0.5);
        second.mark_everything_dirty();
        let other = app.doc.add_body("Body 2", second);
        let meta = NodeMeta { visible: false, ..app.doc.meta(other).unwrap() };
        app.doc.set_meta(&meta);
        app.publish_visibility();
        app.remesh_dirty();

        let probe = Vec3::new(10.0, 12.0, 0.0);
        let before = app.doc.volume(other).unwrap().sample_world(probe);
        let bricks = app.doc.volume(other).unwrap().brick_count();
        let entries = app.history_stats.undo_entries;

        update(&mut app, Message::CutToggled);
        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: iced::Vector::new(80.0, 300.0),
            size: iced::Vector::new(800.0, 600.0),
        });
        app.on_pointer(PointerEvent::Moved {
            position: iced::Vector::new(720.0, 300.0),
            size: iced::Vector::new(800.0, 600.0),
        });
        app.on_pointer(PointerEvent::Released { button: PointerButton::Left });

        assert_eq!(
            app.doc.volume(other).unwrap().sample_world(probe),
            before,
            "the cut went through a body nobody can see"
        );
        assert_eq!(app.doc.volume(other).unwrap().brick_count(), bricks);
        assert_eq!(
            app.history_stats.undo_entries,
            entries + 1,
            "one gesture became more than one undo entry"
        );
    }

    /// **The panel's smoke test.** Build the element tree after every operation,
    /// at every list length that matters, with the pictures on and off.
    ///
    /// It asserts nothing about what the tree looks like, on purpose: what it
    /// catches is a row that panics, a widget built from an index that is no
    /// longer there, and a `view()` that stops compiling against its own state.
    /// Before this existed, `panel.rs` had no coverage of any kind.
    ///
    /// The folder steps are here for the same reason the rename ones are: a
    /// folder row is a DIFFERENT shape from a body row -- a chevron and a count
    /// and a trash where the thumbnail goes -- and a collapsed one changes
    /// which rows are built at all.
    #[test]
    fn the_widget_tree_builds_after_every_operation_at_every_size() {
        type Step = (&'static str, fn(&mut Brokkr));
        let steps: [Step; 31] = [
            ("add a cube", |app| update(app, Message::PrimitiveAdded(PrimitiveKind::Cube))),
            ("add a sphere", |app| update(app, Message::PrimitiveAdded(PrimitiveKind::Sphere))),
            ("add a cylinder", |app| update(app, Message::PrimitiveAdded(PrimitiveKind::Cylinder))),
            ("open the add menu", |app| update(app, Message::PrimitiveMenuToggled)),
            ("select the first row", |app| {
                let first = app.doc.nodes()[0].id;
                update(app, Message::BodySelected(first));
            }),
            ("toggle an eye", |app| {
                let last = app.doc.nodes().last().unwrap().id;
                update(app, Message::BodyVisibilityToggled(last));
            }),
            ("reveal everything", |app| update(app, Message::EveryBodyShown)),
            // Solo changes three things about the tree: the badge beside the
            // section heading, the circle on the soloable rows, and the dimmed
            // background on every row outside the scope. It stays on for the
            // three rename steps below, so all three are built.
            ("solo the active row", |app| {
                let active = app.doc.active();
                update(app, Message::SoloEntered(active));
            }),
            // Three steps and not one: the tree has to be built with a rename
            // field OPEN, which is a different row shape from every other
            // state here, and then again after both ways out of it.
            ("start a rename", |app| {
                let last = app.doc.nodes().last().unwrap().id;
                update(app, Message::BodyRenameBegan(last));
            }),
            ("type into the rename field", |app| {
                update(app, Message::BodyRenameEdited("Renamed".to_string()));
            }),
            ("commit the rename", |app| update(app, Message::BodyRenameSubmitted)),
            ("leave solo", |app| update(app, Message::SoloExited)),
            ("group", |app| update(app, Message::BodyGrouped)),
            ("group again, to nest", |app| update(app, Message::BodyGrouped)),
            // A FOLDER carrying the circle, which is a different row shape from
            // the body that carried it above -- and it stays on across the
            // select below, which moves the active row OUT of the scope and is
            // the one state where a row is dimmed and marked at once.
            ("solo the folder holding the active row", |app| {
                if let Some(folder) = app.doc.parent_of(app.doc.active()) {
                    update(app, Message::SoloEntered(folder));
                }
            }),
            // Duplicate while the active row is NESTED, which the "duplicate"
            // step near the end of this list cannot cover: by then the row is
            // back at the top level, where a copy at the wrong depth is
            // indistinguishable from a copy at the right one.
            ("duplicate a nested body", |app| {
                let parent = app.doc.parent_of(app.doc.active());
                assert!(parent.is_some(), "the step above did not nest the active row");
                update(app, Message::BodyDuplicated);
                assert_eq!(
                    app.doc.parent_of(app.doc.active()),
                    parent,
                    "the copy left the folder it was made in"
                );
            }),
            ("collapse the folder", |app| {
                let folder = app.doc.parent_of(app.doc.active());
                if let Some(folder) = folder {
                    update(app, Message::FolderCollapseToggled(folder));
                }
            }),
            ("select a row inside a collapsed folder", |app| {
                let first = app.doc.nodes()[0].id;
                update(app, Message::BodySelected(first));
            }),
            ("leave the folder's solo", |app| update(app, Message::SoloExited)),
            ("move a body into a folder", |app| {
                let folder = app.doc.folders().next().map(|folder| folder.id);
                update(app, Message::BodyMovedToFolder(folder));
            }),
            ("move it back out", |app| update(app, Message::BodyMovedToFolder(None))),
            ("hide a folder", |app| {
                let folder = app.doc.folders().next().map(|folder| folder.id);
                if let Some(folder) = folder {
                    update(app, Message::BodyVisibilityToggled(folder));
                }
            }),
            ("delete a folder", |app| {
                let folder = app.doc.folders().next().map(|folder| folder.id);
                if let Some(folder) = folder {
                    update(app, Message::FolderDeleted(folder));
                }
            }),
            ("ungroup", |app| update(app, Message::BodyUngrouped)),
            // At 64 rows this is the refusal and at the others it is the copy,
            // and both have to build a tree: the refusal leaves the list as it
            // was with a status line under it, and the copy adds a row in the
            // middle rather than at the end.
            ("duplicate", |app| update(app, Message::BodyDuplicated)),
            // At one row this is the refusal and at the others it is the
            // merge, and both have to build a tree.
            ("merge down", |app| {
                let first = app.doc.nodes()[0].id;
                update(app, Message::BodySelected(first));
                update(app, Message::BodyMergedDown);
            }),
            // The prompt card, which nothing else here builds. A threshold of
            // zero raises it for any merge that has a target at all.
            ("raise the merge prompt", |app| {
                let first = app.doc.nodes()[0].id;
                update(app, Message::BodySelected(first));
                app.merge_prompt_bytes = 0;
                update(app, Message::BodyMergedDown);
            }),
            ("cancel the merge prompt", |app| {
                app.merge_prompt_bytes = brokkr_core::DEFAULT_RECLAIM_BUDGET;
                update(app, Message::BodyMergeCancelled);
            }),
            ("delete", |app| update(app, Message::BodyDeleted)),
            ("undo", |app| update(app, Message::Undo)),
            ("redo", |app| update(app, Message::Redo)),
        ];

        for rows in [1usize, 2, 12, 64] {
            for thumbnails in [true, false] {
                let mut app = app();
                app.thumbnails = thumbnails;
                while app.doc.node_count() < rows {
                    let name = format!("Body {}", app.doc.node_count() + 1);
                    cheap_body(&mut app, &name);
                }
                app.publish_visibility();
                let _ = app.view();

                for (what, step) in steps {
                    step(&mut app);
                    // The tree is built and dropped. `Element` borrows the
                    // application, so it cannot outlive the loop body -- which
                    // is also what stops a row from caching anything.
                    let _ = app.view();
                    assert!(
                        app.doc.body_count() >= 1,
                        "{what} at {rows} rows left the document with no body"
                    );
                }
            }
        }
    }

    /// The sixth section is FIRST and open, and DETAIL is what paid for it.
    #[test]
    fn the_bodies_section_is_first_and_open_and_detail_now_starts_closed() {
        assert_eq!(PanelSection::ALL[0], PanelSection::Bodies);
        assert_eq!(PanelSection::ALL.len(), 6);
        assert!(PanelSection::Bodies.open_by_default());
        assert!(!PanelSection::Detail.open_by_default());

        let app = app();
        assert!(app.expanded[PanelSection::Bodies as usize], "BODIES did not start open");
        assert!(!app.expanded[PanelSection::Detail as usize], "DETAIL did not start closed");
    }

    // --- folders -------------------------------------------------------------

    /// The shape of the tree, as `(depth, name)` per row, which is what almost
    /// every folder assertion below is really about.
    fn shape(app: &Brokkr) -> Vec<(u8, String)> {
        app.doc.nodes().iter().map(|node| (node.depth(), node.name.clone())).collect()
    }

    /// `ctrl+G` wraps the active row in place, and pressing it again NESTS --
    /// which is the only route to a deep tree before a drag lands, and the one
    /// the depth-seven fixture in the format tests is built with.
    #[test]
    fn ctrl_g_wraps_the_active_row_and_wraps_again_to_nest() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let cube = app.doc.active();

        update(&mut app, Message::BodyGrouped);
        assert_eq!(
            shape(&app),
            vec![(0, "Body 1".into()), (0, "Group 1".into()), (1, "Cube".into())],
            "the folder did not appear where the row was"
        );
        assert_eq!(app.doc.active(), cube, "grouping moved the selection");
        assert!(app.unsaved, "the tree is written to the file, so grouping is a change");

        update(&mut app, Message::BodyGrouped);
        assert_eq!(
            shape(&app),
            vec![
                (0, "Body 1".into()),
                (0, "Group 1".into()),
                (1, "Group 2".into()),
                (2, "Cube".into()),
            ],
            "the second press did not nest"
        );
    }

    /// `ctrl+shift+G` is `ctrl+G` read backwards, outline for outline, and one
    /// ctrl+Z is enough to reverse either of them.
    #[test]
    fn ctrl_shift_g_dissolves_the_folder_the_active_row_is_in() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let was = shape(&app);

        update(&mut app, Message::BodyGrouped);
        update(&mut app, Message::BodyUngrouped);
        assert_eq!(shape(&app), was, "the pair is not each other's inverse");

        // And the pair reversed by hand, one press each.
        update(&mut app, Message::BodyGrouped);
        let grouped = shape(&app);
        update(&mut app, Message::Undo);
        assert_eq!(shape(&app), was, "one ctrl+Z did not take the group apart");
        update(&mut app, Message::Redo);
        assert_eq!(shape(&app), grouped, "one ctrl+shift+Z did not put it back");
    }

    /// A row with no folder above it has nothing to dissolve, and it says so
    /// rather than doing something surprising to the nearest folder it can find.
    #[test]
    fn ungrouping_a_row_that_is_not_in_a_folder_is_refused_with_a_line() {
        let mut app = app();
        let was = shape(&app);
        update(&mut app, Message::BodyUngrouped);
        assert_eq!(shape(&app), was);
        assert!(app.status.contains("not in a folder"), "no line explains it: {}", app.status);
    }

    /// The chevron only folds rows away. It must not change the document's
    /// shape, and it must survive a save -- `collapsed` is a bit in the file.
    #[test]
    fn the_chevron_folds_rows_away_and_changes_nothing_else() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.nodes()[1].id;
        let was = shape(&app);

        update(&mut app, Message::FolderCollapseToggled(folder));
        assert!(app.doc.node(folder).unwrap().collapsed, "the chevron did nothing");
        assert_eq!(shape(&app), was, "collapsing changed the tree");

        update(&mut app, Message::Undo);
        assert!(!app.doc.node(folder).unwrap().collapsed, "collapse is not undoable");
    }

    /// **Deleting a body inside a collapsed folder deletes the body, never the
    /// folder.**
    ///
    /// A user lost an unrecoverable hour to ZBrush doing the other thing, its
    /// own bundled Delete macro had the same hole, and a third-party plugin
    /// exists solely to intercept it. Here it is structural rather than
    /// remembered: the verb row's Delete names `Document::active`, which always
    /// holds a field, and the folder's trash names the folder.
    #[test]
    fn deleting_a_body_in_a_collapsed_folder_never_takes_the_folder() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Sphere));
        let sphere = app.doc.active();
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.parent_of(sphere).expect("the new folder");
        // A second body inside, so the folder has a reason to survive.
        let cube = app.doc.nodes().iter().find(|node| node.name == "Cube").expect("the cube").id;
        update(&mut app, Message::BodyMovedToFolder(Some(folder)));
        update(&mut app, Message::BodySelected(cube));
        update(&mut app, Message::BodyMovedToFolder(Some(folder)));
        update(&mut app, Message::FolderCollapseToggled(folder));
        update(&mut app, Message::BodySelected(sphere));

        update(&mut app, Message::BodyDeleted);
        assert!(app.doc.node(folder).is_some(), "a collapsed folder swallowed a body delete");
        assert!(app.doc.node(cube).is_some(), "the folder's other body went with it");
        assert!(app.doc.node(sphere).is_none(), "the body the user asked about survived");
        assert!(app.doc.node(folder).unwrap().collapsed, "the folder came open");
    }

    /// A folder delete takes its contents, in ONE entry, and one ctrl+Z brings
    /// the whole thing back with the folder above the bodies again.
    #[test]
    fn a_folder_delete_takes_its_bodies_and_one_undo_brings_them_all_back() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.nodes()[1].id;
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Sphere));
        update(&mut app, Message::BodyMovedToFolder(Some(folder)));
        let was = shape(&app);
        assert_eq!(app.doc.subtree_body_count(folder), 2);

        let entries = app.history_stats.undo_entries;
        update(&mut app, Message::FolderDeleted(folder));
        assert_eq!(app.doc.body_count(), 1, "the folder's bodies were left behind");
        assert_eq!(
            app.history_stats.undo_entries,
            entries + 1,
            "three removals became three gestures"
        );
        assert!(app.status.contains("2 bodies"), "the line does not say what went: {}", app.status);

        update(&mut app, Message::Undo);
        assert_eq!(shape(&app), was, "one ctrl+Z did not put the whole folder back");
    }

    /// The folder trash never takes a body row, whatever it is handed. The
    /// message is the folder's own affordance and a body has its own.
    #[test]
    fn the_folder_delete_refuses_a_body_row() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let cube = app.doc.active();
        let was = shape(&app);

        update(&mut app, Message::FolderDeleted(cube));
        assert_eq!(shape(&app), was, "the folder trash deleted a body");
    }

    /// A folder delete over the reclaim allowance prompts with the SUMMED size,
    /// and cancelling changes nothing.
    ///
    /// Folders are what make the prompt the common case: forty modest bodies
    /// pass a per-body threshold every time and then evict each other, which is
    /// the failure the threshold and the allowance being ONE number exists to
    /// catch.
    #[test]
    fn a_folder_delete_over_the_allowance_prompts_with_the_summed_size() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.nodes()[1].id;
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Sphere));
        update(&mut app, Message::BodyMovedToFolder(Some(folder)));

        let summed = app.subtree_bytes(folder);
        let one = app
            .doc
            .node(app.doc.active())
            .and_then(brokkr_core::Node::volume)
            .map_or(0, |volume| volume.stats().resident_bytes);
        assert!(summed > one, "the fixture's two bodies are not big enough to tell apart");
        // Between the two, so a per-body threshold would NOT fire and the
        // summed one must.
        app.delete_prompt_bytes = (one + summed) / 2;

        let was = shape(&app);
        update(&mut app, Message::FolderDeleted(folder));
        let pending = app.pending_delete.as_ref().expect("the prompt did not open");
        assert_eq!(pending.bytes, summed, "the prompt named one body's size, not the folder's");
        assert_eq!(shape(&app), was, "the prompt deleted before it asked");

        update(&mut app, Message::BodyDeleteCancelled);
        assert_eq!(shape(&app), was, "cancelling changed the document");

        update(&mut app, Message::FolderDeleted(folder));
        update(&mut app, Message::BodyDeleteConfirmed);
        assert!(app.doc.node(folder).is_none(), "confirming did not delete the folder");
    }

    /// A press on a folder row selects the first body inside it, because the
    /// active row always holds a field.
    #[test]
    fn pressing_a_folder_row_selects_the_first_body_inside_it() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let cube = app.doc.active();
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.nodes()[1].id;
        let first = app.doc.nodes()[0].id;
        update(&mut app, Message::BodySelected(first));
        assert_ne!(app.doc.active(), cube);

        update(&mut app, Message::BodySelected(folder));
        assert_eq!(app.doc.active(), cube, "pressing a folder did not reach the body inside it");
        assert!(app.doc.volume(app.doc.active()).is_some(), "a folder became the active row");
    }

    /// Move-to-folder is a round trip, and moving the folder's last child out
    /// dissolves the folder in the SAME entry.
    #[test]
    fn moving_the_last_child_out_of_a_folder_dissolves_it_in_one_entry() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        update(&mut app, Message::BodyGrouped);
        let folder = app.doc.nodes()[1].id;
        let was = shape(&app);

        let entries = app.history_stats.undo_entries;
        update(&mut app, Message::BodyMovedToFolder(None));
        assert!(app.doc.node(folder).is_none(), "an empty folder was left behind");
        assert_eq!(
            app.history_stats.undo_entries,
            entries + 1,
            "the move and the dissolve are two gestures"
        );

        update(&mut app, Message::Undo);
        assert_eq!(shape(&app), was, "one ctrl+Z did not restore the folder and the row together");
    }

    /// The row cap is a refusal with a line that names it, exactly as the body
    /// cap is: nothing the interface builds goes through the reader.
    ///
    /// **`MAX_NODES` is twice `MAX_BODIES`**, so the only way to fill the list
    /// is with folders as well as bodies -- which is exactly why the two caps
    /// are separate numbers, and why the fixture has to be built this way.
    #[test]
    fn grouping_past_the_row_cap_is_refused_and_says_so() {
        let mut app = app();
        while app.doc.body_count() < MAX_BODIES {
            let name = format!("Body {}", app.doc.body_count() + 1);
            cheap_body(&mut app, &name);
        }
        // One folder per body, each wrapping only its own row, until the rows
        // run out.
        let bodies: Vec<NodeId> = app.doc.bodies().map(|(id, _)| id).collect();
        for id in bodies {
            if app.doc.node_count() >= brokkr_core::MAX_NODES {
                break;
            }
            update(&mut app, Message::BodySelected(id));
            update(&mut app, Message::BodyGrouped);
        }
        assert_eq!(app.doc.node_count(), brokkr_core::MAX_NODES, "the fixture did not fill up");
        let rows = app.doc.node_count();

        update(&mut app, Message::BodyGrouped);
        assert_eq!(app.doc.node_count(), rows, "the cap let a folder through");
        assert!(
            app.status.contains("could not group")
                && app.status.contains(&brokkr_core::MAX_NODES.to_string()),
            "the refusal does not name the cap: {}",
            app.status
        );
    }

    /// Eight levels is the cap, and the ninth press is refused with a line
    /// rather than clamped into a folder that did nothing.
    #[test]
    fn grouping_past_the_eighth_level_is_refused_and_says_so() {
        let mut app = app();
        for _ in 0..brokkr_core::MAX_DEPTH - 1 {
            update(&mut app, Message::BodyGrouped);
        }
        assert_eq!(
            app.doc.node(app.doc.active()).unwrap().depth(),
            brokkr_core::MAX_DEPTH - 1,
            "seven presses did not reach the deepest legal level"
        );
        let rows = app.doc.node_count();

        update(&mut app, Message::BodyGrouped);
        assert_eq!(app.doc.node_count(), rows, "an eighth level was allowed");
        assert!(
            app.status.contains("as far as the panel goes"),
            "the refusal does not say why: {}",
            app.status
        );
    }
}

/// Inline rename: click a name, type, commit.
///
/// **The two things worth testing here are both about what a rename must NOT
/// do**: it must not let the keyboard shortcuts fire while the field has focus,
/// and it must not put a name in the document that the file cannot hold. The
/// happy path is three lines of state; those two are where the bugs are.
#[cfg(test)]
mod rename_tests {
    use super::*;
    use brokkr_core::{MAX_NAME_BYTES, PrimitiveKind};

    fn update(app: &mut Brokkr, message: Message) {
        drop(app.update(message));
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
    }

    fn name_of(app: &Brokkr, id: NodeId) -> String {
        app.doc.node(id).expect("the row is in the document").name.clone()
    }

    /// Begin a rename on `id` and type `typed` into the field, without
    /// committing.
    fn type_into_the_field(app: &mut Brokkr, id: NodeId, typed: &str) {
        update(app, Message::BodyRenameBegan(id));
        update(app, Message::BodyRenameEdited(typed.to_string()));
    }

    /// Write the document out and read it back, which is the only way to ask
    /// what a name will really be worth tomorrow.
    fn round_trip(app: &Brokkr) -> brokkr_core::Document {
        let mut bytes = Vec::new();
        brokkr_core::project::write(&mut bytes, &app.doc, &brokkr_core::ProjectState::default())
            .expect("writing the sculpt failed");
        let (doc, _) = brokkr_core::project::read(&mut bytes.as_slice()).expect("read failed");
        doc
    }

    /// A window event carrying `character`. The fields beyond `key` and
    /// `modifiers` are what a real winit press fills in and `key_event`
    /// ignores; they are here because the literal does not compile without
    /// them.
    fn digit_event(character: &str) -> iced::Event {
        iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
            key: iced::keyboard::Key::Character(character.into()),
            modified_key: iced::keyboard::Key::Character(character.into()),
            physical_key: iced::keyboard::key::Physical::Unidentified(
                iced::keyboard::key::NativeCode::Unidentified,
            ),
            location: iced::keyboard::Location::Standard,
            modifiers: iced::keyboard::Modifiers::empty(),
            text: None,
            repeat: false,
        })
    }

    /// Opening a rename ISSUES the two widget operations that focus the field.
    ///
    /// **Read what this does NOT say.** A widget operation can only be run
    /// against a live `UserInterface` and a real `Renderer`, neither of which
    /// a headless test can build, so nothing here proves that focus ARRIVES --
    /// only that `begin_rename` hands the runtime two units of work rather
    /// than `Task::none()`. That gap is real and is what the visual pass has
    /// to close: if focus does not land, the field renders unfocused, and
    /// every keystroke meant for it goes to `key_event` as Ignored instead.
    ///
    /// It is still worth pinning, because "tidy the task away" is a one-line
    /// edit that no other test in this file notices: the field would still be
    /// drawn, the text would still commit, and only a human at the keyboard
    /// would see that typing did nothing. The row that has gone is the
    /// control -- without it, `units()` could be reading a constant.
    #[test]
    fn opening_a_rename_hands_the_runtime_the_work_that_focuses_the_field() {
        let mut opened = app();
        let mut missing = app();
        let first = opened.doc.nodes()[0].id;

        let task = opened.update(Message::BodyRenameBegan(first));
        assert_eq!(
            task.units(),
            2,
            "the focus and select-all operations are not being issued, so the field opens dead"
        );

        let gone = missing.update(Message::BodyRenameBegan(NodeId(9_999)));
        assert_eq!(gone.units(), 0, "a rename of a row that is not there still asked for work");
    }

    /// **The check the whole increment 7 plumbing exists for.** A `1` typed
    /// into a focused rename field selects the letter, not the Draw brush.
    ///
    /// The guarantee lives in `key_event`, which drops captured events, and a
    /// focused `text_input` reports its keystrokes captured. This test asserts
    /// both halves against each other: the same key, the same application, the
    /// only difference being who wanted the event. Without the Ignored half the
    /// Captured half would still pass with the shortcut table deleted.
    #[test]
    fn a_digit_typed_into_the_rename_field_does_not_switch_the_brush() {
        let mut app = app();
        let first = app.doc.nodes()[0].id;
        // Something other than what `1` selects, so that "unchanged" is
        // distinguishable from "changed to the default".
        app.brush.kind = BrushKind::ALL[3];
        type_into_the_field(&mut app, first, "Hea");

        let captured =
            key_event(digit_event("1"), iced::event::Status::Captured, iced::window::Id::unique());
        assert!(captured.is_none(), "the field's own keystroke was forwarded as a shortcut");
        assert_eq!(app.brush.kind, BrushKind::ALL[3], "typing a name changed the brush");
        assert!(app.renaming.is_some(), "the field closed on a keystroke it never saw");

        // The control: the same key, wanted by nobody, does select the brush.
        // If this stops firing the assertion above means nothing.
        let ignored =
            key_event(digit_event("1"), iced::event::Status::Ignored, iced::window::Id::unique());
        let Some(message) = ignored else {
            panic!("an ignored digit produced no message, so the check above proves nothing");
        };
        update(&mut app, message);
        assert_eq!(app.brush.kind, BrushKind::ALL[0], "the control never switched the brush");
    }

    /// Enter commits, as ONE undoable change, and undo puts the old name back.
    #[test]
    fn enter_commits_the_typed_name_as_one_undoable_change() {
        let mut app = app();
        let first = app.doc.nodes()[0].id;
        let was = name_of(&app, first);
        let entries = app.history_stats.undo_entries;

        type_into_the_field(&mut app, first, "Left ear");
        assert_eq!(name_of(&app, first), was, "the document changed before the commit");

        update(&mut app, Message::BodyRenameSubmitted);

        assert_eq!(name_of(&app, first), "Left ear");
        assert!(app.renaming.is_none(), "the field stayed open after Enter");
        assert!(app.unsaved, "a rename did not mark the document unsaved");
        assert_eq!(
            app.history_stats.undo_entries,
            entries + 1,
            "one rename was not one undo entry"
        );

        update(&mut app, Message::Undo);
        assert_eq!(name_of(&app, first), was, "undo did not put the old name back");
        update(&mut app, Message::Redo);
        assert_eq!(name_of(&app, first), "Left ear", "redo did not bring the new name back");
    }

    /// Escape reverts, and leaves nothing behind: not the name, not an undo
    /// entry, not the unsaved marker.
    ///
    /// **It is the SECOND Escape in the running application**, because the
    /// focused field captures the first one and unfocuses itself. That is a
    /// property of iced's `text_input` and not of this code, so the test drives
    /// the message the second press produces.
    #[test]
    fn escape_reverts_a_rename_and_leaves_nothing_behind() {
        let mut app = app();
        let first = app.doc.nodes()[0].id;
        let was = name_of(&app, first);
        let entries = app.history_stats.undo_entries;

        type_into_the_field(&mut app, first, "Discarded");
        update(&mut app, Message::MenuClosed);

        assert!(app.renaming.is_none(), "escape left the field open");
        assert_eq!(name_of(&app, first), was, "escape committed the name it was meant to discard");
        assert!(!app.unsaved, "a reverted rename marked the document unsaved");
        assert_eq!(app.history_stats.undo_entries, entries, "a reverted rename cost an undo entry");
    }

    /// Escape against a rename must not also disarm the cut, which is the other
    /// thing Escape does. The early return in the `MenuClosed` arm is what says
    /// so, and without it one press would undo two decisions.
    #[test]
    fn escape_against_a_rename_leaves_an_armed_cut_armed() {
        let mut app = app();
        let first = app.doc.nodes()[0].id;
        update(&mut app, Message::CutToggled);
        assert!(app.cut_armed, "the cut did not arm");

        type_into_the_field(&mut app, first, "Discarded");
        update(&mut app, Message::MenuClosed);

        assert!(app.renaming.is_none(), "escape left the field open");
        assert!(app.cut_armed, "escaping a rename also disarmed the cut");
    }

    /// **What "blur commits" means here.** Every way of leaving the field that
    /// is not Escape keeps what was typed.
    ///
    /// Table-driven over the four routes out, because the guard in `update` is
    /// a single decision covering all of them and one example would not show
    /// that: a press in the viewport, clicking another row, a menu command, and
    /// a keyboard shortcut arriving once the field has been blurred.
    #[test]
    fn every_way_out_of_the_field_except_escape_keeps_what_was_typed() {
        type Exit = (&'static str, fn(&mut Brokkr));
        let exits: [Exit; 5] = [
            // Through `key_event` and not `Message::PressedNothing` directly,
            // because the message is worth nothing unless the subscription
            // raises it: this is the press that used to produce NO message,
            // leaving the field drawn, unfocused and eating the next keystroke
            // as a tool shortcut.
            ("a press on the empty panel below the last row", |app| {
                let message = key_event(
                    iced::Event::Mouse(iced::mouse::Event::ButtonPressed(
                        iced::mouse::Button::Left,
                    )),
                    iced::event::Status::Ignored,
                    iced::window::Id::unique(),
                )
                .expect("a press nobody wanted raised no message, so a blur is invisible");
                update(app, message);
            }),
            ("a press in the viewport", |app| {
                update(
                    app,
                    Message::Pointer(PointerEvent::Pressed {
                        button: PointerButton::Left,
                        position: iced::Vector::new(400.0, 300.0),
                        size: iced::Vector::new(800.0, 600.0),
                    }),
                );
            }),
            ("clicking another row", |app| {
                let other = app.doc.nodes()[1].id;
                update(app, Message::BodySelected(other));
            }),
            ("a menu command", |app| update(app, Message::ThumbnailsToggled)),
            ("a shortcut once the field has been blurred", |app| {
                // Through `KeyPressed` and not the message it decodes to, so
                // this covers the nesting: `on_key` calls `update` again, and
                // the guard has to fire on the INNER message. `KeyPressed`
                // itself is on the keep list, because Escape has to survive it.
                update(
                    app,
                    Message::KeyPressed {
                        key: iced::keyboard::Key::Character("2".into()),
                        modifiers: iced::keyboard::Modifiers::empty(),
                    },
                );
            }),
        ];

        for (what, exit) in exits {
            let mut app = app();
            update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
            let first = app.doc.nodes()[0].id;
            type_into_the_field(&mut app, first, "Kept");

            exit(&mut app);

            assert!(app.renaming.is_none(), "the field survived {what}");
            assert_eq!(name_of(&app, first), "Kept", "{what} threw the typed name away");
        }
    }

    /// The frame tick and a pointer merely moving are not the user leaving.
    ///
    /// Named separately from the exits above because the failure is silent and
    /// total: `Frame` arrives at display rate, so a guard that committed on it
    /// would close the field before the first letter landed and the feature
    /// would look like it was never wired up.
    #[test]
    fn the_frame_tick_and_a_moving_pointer_leave_the_field_alone() {
        let mut app = app();
        let first = app.doc.nodes()[0].id;
        type_into_the_field(&mut app, first, "Still typing");

        update(&mut app, Message::Frame);
        update(
            &mut app,
            Message::Pointer(PointerEvent::Moved {
                position: iced::Vector::new(400.0, 300.0),
                size: iced::Vector::new(800.0, 600.0),
            }),
        );
        update(&mut app, Message::BodyRenameEdited("Still typing more".to_string()));

        let Some((id, typed)) = &app.renaming else {
            panic!("the field closed on a frame tick or a mouse move");
        };
        assert_eq!(*id, first);
        assert_eq!(typed, "Still typing more");
    }

    /// **A multi-byte name at exactly the field's length round-trips
    /// unchanged.** Thirty-two bytes fills the fixed field with no NUL after
    /// it, which is the one name a "must be terminated" reader would destroy.
    #[test]
    fn a_multi_byte_name_at_exactly_the_field_length_round_trips_unchanged() {
        // Eight four-byte characters is thirty-two bytes on the nose.
        let name = "𝄞𝄞𝄞𝄞𝄞𝄞𝄞𝄞";
        assert_eq!(name.len(), MAX_NAME_BYTES);

        let mut app = app();
        let first = app.doc.nodes()[0].id;
        type_into_the_field(&mut app, first, name);
        update(&mut app, Message::BodyRenameSubmitted);

        assert_eq!(name_of(&app, first), name, "the field would not accept a full-length name");
        assert_eq!(
            round_trip(&app).nodes()[0].name,
            name,
            "a full-length name did not survive the file"
        );
    }

    /// **One byte past the field is a valid PREFIX, never "Body 1".** The
    /// field refuses the byte as it is typed, so what is on screen is what the
    /// file will hold.
    ///
    /// Cut on a char boundary or the field writes bytes that are not UTF-8, the
    /// reader repairs the whole name to a default, and the user silently loses
    /// a name this build wrote itself -- eleven characters typed, none kept.
    #[test]
    fn a_name_past_the_field_is_clamped_to_a_valid_prefix_and_never_to_a_default() {
        // Eleven three-byte characters is thirty-three bytes, so the eleventh
        // straddles byte thirty-two.
        let typed = "。。。。。。。。。。。";
        let expected = "。。。。。。。。。。";
        assert_eq!(typed.len(), MAX_NAME_BYTES + 1);
        assert_eq!(expected.len(), 30);

        let mut app = app();
        let first = app.doc.nodes()[0].id;
        type_into_the_field(&mut app, first, typed);

        let Some((_, held)) = &app.renaming else {
            panic!("the field closed while it was being typed into");
        };
        assert_eq!(held, expected, "the field held a name the file cannot store");

        update(&mut app, Message::BodyRenameSubmitted);
        assert_eq!(name_of(&app, first), expected);
        assert_eq!(
            round_trip(&app).nodes()[0].name,
            expected,
            "the name came back repaired to a default rather than as a prefix"
        );
    }

    /// A name that is only whitespace keeps the old one, and says so.
    ///
    /// Committing it would not leave the body nameless: the reader repairs an
    /// empty name field to `Body {n}`, so the body would come back called
    /// something the user never typed.
    #[test]
    fn a_rename_to_nothing_keeps_the_old_name() {
        let mut app = app();
        let first = app.doc.nodes()[0].id;
        let was = name_of(&app, first);
        let entries = app.history_stats.undo_entries;

        type_into_the_field(&mut app, first, "   ");
        update(&mut app, Message::BodyRenameSubmitted);

        assert_eq!(name_of(&app, first), was, "an empty name was committed");
        assert_eq!(app.history_stats.undo_entries, entries, "an empty name cost an undo entry");
        assert!(!app.unsaved, "an empty name marked the document unsaved");
        assert!(app.status.contains(&was), "the refusal did not say which name was kept");
    }

    /// Committing the name that is already there costs nothing at all --
    /// no entry, no unsaved marker. Opening a field and pressing Enter is not
    /// an edit, and charging a real undo press for it is how a history stops
    /// being trustworthy.
    #[test]
    fn committing_an_unchanged_name_costs_no_undo_entry() {
        let mut app = app();
        let first = app.doc.nodes()[0].id;
        let was = name_of(&app, first);
        let entries = app.history_stats.undo_entries;

        update(&mut app, Message::BodyRenameBegan(first));
        // Whitespace either side, so this also pins that the trim happens
        // before the comparison rather than after it.
        update(&mut app, Message::BodyRenameEdited(format!("  {was}  ")));
        update(&mut app, Message::BodyRenameSubmitted);

        assert_eq!(name_of(&app, first), was);
        assert_eq!(app.history_stats.undo_entries, entries, "a no-op rename cost an undo entry");
        assert!(!app.unsaved, "a no-op rename marked the document unsaved");
    }

    /// Beginning a rename on a second row commits the first, rather than
    /// silently swapping the field's owner and losing what was typed.
    #[test]
    fn beginning_a_second_rename_commits_the_first() {
        let mut app = app();
        update(&mut app, Message::PrimitiveAdded(PrimitiveKind::Cube));
        let first = app.doc.nodes()[0].id;
        let second = app.doc.nodes()[1].id;

        type_into_the_field(&mut app, first, "One");
        update(&mut app, Message::BodyRenameBegan(second));

        assert_eq!(name_of(&app, first), "One", "the first rename was thrown away");
        let Some((id, typed)) = &app.renaming else {
            panic!("the second rename did not open a field");
        };
        assert_eq!(*id, second);
        assert_eq!(typed, &name_of(&app, second), "the field did not start from the row's name");
    }

    /// The decode on its own, as a table.
    ///
    /// `keeps_the_rename_open` is inverted on purpose -- an unlisted message
    /// commits -- so what this pins is the short list of exceptions. A message
    /// wrongly ADDED to it leaves a field open over a document that has moved
    /// on; a message wrongly left out closes the field mid-word.
    #[test]
    fn only_the_field_its_keys_and_the_ambient_ticks_keep_a_rename_open() {
        let keeps: [Message; 6] = [
            Message::BodyRenameEdited("half a name".to_string()),
            Message::KeyPressed {
                key: iced::keyboard::Key::Character("1".into()),
                modifiers: iced::keyboard::Modifiers::empty(),
            },
            Message::MenuClosed,
            Message::Frame,
            Message::Pointer(PointerEvent::Moved {
                position: iced::Vector::new(1.0, 1.0),
                size: iced::Vector::new(800.0, 600.0),
            }),
            Message::Pointer(PointerEvent::Released { button: PointerButton::Left }),
        ];
        for message in keeps {
            assert!(keeps_the_rename_open(&message), "{message:?} closed the rename field");
        }

        let commits: [Message; 6] = [
            Message::BodyRenameSubmitted,
            Message::BodyRenameBegan(NodeId(1)),
            Message::BodySelected(NodeId(1)),
            Message::Undo,
            Message::Pointer(PointerEvent::Pressed {
                button: PointerButton::Left,
                position: iced::Vector::new(1.0, 1.0),
                size: iced::Vector::new(800.0, 600.0),
            }),
            // The press that landed on nothing. `Pointer(Pressed)` above is
            // bounds-checked to the viewport and does not cover it.
            Message::PressedNothing,
        ];
        for message in commits {
            assert!(!keeps_the_rename_open(&message), "{message:?} left the rename field open");
        }
    }
}
