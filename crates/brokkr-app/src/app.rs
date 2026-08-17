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
use iced::Subscription;

use crate::camera::OrbitCamera;
use crate::message::{
    ExportFormat, Message, PanelSection, PointerButton, PointerEvent, SpaceMouseSetting,
};
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

/// What a held pointer button is currently doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragKind {
    Orbit,
    Pan,
    Sculpt(BrushDirection),
}

/// A drag in progress, tagged with the button that started it so that
/// releasing a different button does not cancel it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Drag {
    button: PointerButton,
    kind: DragKind,
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
}

impl Brokkr {
    pub fn new() -> Self {
        Self::with_devices(Tablet::start(), SpaceMouse::start())
    }

    /// Build with a given pressure source and an inert puck, which is what
    /// almost every test wants.
    #[cfg(test)]
    fn with_tablet(tablet: Tablet) -> Self {
        Self::with_devices(tablet, SpaceMouse::inert())
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
            perf: Perf::default(),
            volume_stats: VolumeStats::default(),
            history_stats: HistoryStats::default(),
            voxel_size: VOXEL_SIZE_MM,
            status: String::new(),
            expanded: PanelSection::ALL.map(PanelSection::open_by_default),
        };
        app.remesh_dirty();
        // Otherwise the overlay reports a zero byte budget until the first
        // stroke happens to refresh it.
        app.history_stats = app.history.stats();
        app.perf.load_ms = app.perf.remesh_ms;
        app.perf.remesh_ms = 0.0;
        app.perf.dirty_bricks = 0;
        app.publish_camera();
        app
    }

    pub fn title(&self) -> String {
        "BrokkrSculpt".to_string()
    }

    pub fn subscription(&self) -> Subscription<Message> {
        // Drives the frame rate readout and keeps the viewport presenting while
        // a stroke is in flight.
        iced::window::frames().map(|_| Message::Frame)
    }

    fn publish_camera(&self) {
        self.shared.set_camera(self.camera);
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
        let inverted = self.control != self.eraser_in_use();
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

    fn finish_stroke(&mut self) {
        self.stroke.end();
        if let Some(edit) = self.volume.end_stroke() {
            self.history.push(edit);
            self.history_stats = self.history.stats();
        }
    }

    /// Where exported files go.
    ///
    /// A fixed directory rather than a file dialog: Iced has no file picker and
    /// pulling one in is a dependency and a portal setup for something a first
    /// version can do without. The path is shown in the interface so it is never
    /// a guess.
    fn export_directory() -> std::path::PathBuf {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        home.unwrap_or_else(std::env::temp_dir).join("brokkrsculpt")
    }

    /// Weld the sculpt into one mesh and write it out.
    ///
    /// Refuses to write anything that would not print. A slicer given a mesh
    /// with holes either rejects it or fills it wrong, and finding that out
    /// after a failed print is worse than being told here.
    fn export(&mut self, format: ExportFormat) {
        let started = Instant::now();
        let (mesh, report) = self.volume.export_mesh();

        if !report.is_printable() {
            self.status = format!("not exported, {}", report.summary());
            log::error!("refusing to export: {}", report.summary());
            return;
        }

        let directory = Self::export_directory();
        if let Err(error) = std::fs::create_dir_all(&directory) {
            self.status = format!("could not make {}: {error}", directory.display());
            return;
        }
        let path = directory.join(format!("sculpt.{}", format.extension()));

        let write = || -> std::io::Result<()> {
            let file = std::fs::File::create(&path)?;
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
                let bytes = std::fs::metadata(&path).map(|data| data.len()).unwrap_or(0);
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

    fn undo(&mut self) {
        if self.history.undo(&mut self.volume) {
            self.history_stats = self.history.stats();
            self.remesh_dirty();
        }
    }

    fn redo(&mut self) {
        if self.history.redo(&mut self.volume) {
            self.history_stats = self.history.stats();
            self.remesh_dirty();
        }
    }

    fn on_pointer(&mut self, event: PointerEvent) {
        match event {
            PointerEvent::Modifiers { shift, control } => {
                self.shift = shift;
                self.control = control;
            }
            PointerEvent::Pressed { button, position, size } => {
                self.viewport_size = Vec2::new(size.x, size.y);
                let position = Vec2::new(position.x, position.y);
                self.cursor = Some(position);

                let kind = match button {
                    // Left sculpts. Holding control removes instead of adds,
                    // which is the convention every sculpting tool uses.
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
                self.drag = Some(Drag { button, kind });

                if let DragKind::Sculpt(direction) = kind {
                    // One stroke is one undo entry, so recording opens here and
                    // closes when the button comes back up.
                    self.volume.begin_stroke();
                    self.sculpt_to(position, direction, true);
                }
            }
            PointerEvent::Released { button } => {
                if self.drag.is_some_and(|drag| drag.button == button) {
                    if matches!(self.drag.map(|drag| drag.kind), Some(DragKind::Sculpt(_))) {
                        self.finish_stroke();
                    }
                    self.drag = None;
                }
            }
            PointerEvent::Moved { position, size } => {
                self.viewport_size = Vec2::new(size.x, size.y);
                let position = Vec2::new(position.x, position.y);
                let delta = self.cursor.map(|previous| position - previous).unwrap_or(Vec2::ZERO);
                self.cursor = Some(position);

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
                    None => {}
                }
            }
            PointerEvent::Scrolled { amount } => {
                self.camera.zoom(amount);
                self.publish_camera();
            }
        }
    }

    pub fn update(&mut self, message: Message) {
        match message {
            Message::Pointer(event) => self.on_pointer(event),
            Message::Frame => {
                let elapsed_ms = self.perf.record_frame();
                self.drive_from_spacemouse(elapsed_ms);
            }
            Message::BrushKindChanged(kind) => self.brush.kind = kind,
            Message::FalloffChanged(curve) => self.brush.falloff = curve,
            Message::BrushRadiusChanged(radius) => self.brush.radius = radius,
            Message::BrushStrengthChanged(strength) => self.brush.strength = strength,
            Message::SymmetryAxisToggled(axis) => self.symmetry = self.symmetry.toggled(axis),
            Message::PatternChanged(kind) => self.brush.pattern.kind = kind,
            Message::PatternScaleChanged(scale) => self.brush.pattern.scale_mm = scale,
            Message::PatternDepthChanged(depth) => self.brush.pattern.depth = depth,
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
            Message::Export(format) => self.export(format),
            Message::Resample(voxel_size) => self.resample(voxel_size),
            Message::SpaceMouse(setting) => self.configure_spacemouse(setting),
            Message::SectionToggled(section) => {
                let open = &mut self.expanded[section as usize];
                *open = !*open;
            }
            Message::ResetSphere => {
                let mut volume = Volume::new(self.voxel_size);
                volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM);
                volume.mark_everything_dirty();
                // The old bricks must be cleared from the pool too, or their
                // triangles stay on screen. Marking them dirty makes them
                // remesh to nothing, which releases their slices.
                for coord in self.volume.brick_coords() {
                    volume.mark_dirty(coord);
                }
                self.volume = volume;
                // History refers to bricks of the volume that just went away,
                // so keeping it would let undo splice pieces of the discarded
                // model into the new one.
                self.history.clear();
                self.history_stats = self.history.stats();
                self.camera = OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM);
                self.publish_camera();
                self.remesh_dirty();
            }
        }
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
    use crate::tablet::PenState;
    use brokkr_core::BrushKind;
    use iced::Vector;

    const SIZE: Vector = Vector { x: 800.0, y: 600.0 };

    fn centre_of_viewport() -> Vector {
        Vector::new(SIZE.x / 2.0, SIZE.y / 2.0)
    }

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
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

        app.update(Message::Undo);
        assert_eq!(app.volume.sample_world(front), before, "undo did not restore the field");
        assert!(app.perf.dirty_bricks > 0, "undo must schedule a remesh or the screen goes stale");

        app.update(Message::Redo);
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
        app.update(Message::SymmetryAxisToggled(MirrorAxis::X));
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
        app.update(Message::BrushKindChanged(BrushKind::Draw));
        assert_eq!(app.effective_brush().kind, BrushKind::Draw);

        app.update(Message::Pointer(PointerEvent::Modifiers { shift: true, control: false }));
        assert_eq!(app.effective_brush().kind, BrushKind::Smooth, "shift should smooth");
        assert_eq!(
            app.brush.kind,
            BrushKind::Draw,
            "the selection itself must not change, or the tool strip would flicker \
             and a key released out of focus would strand the wrong brush"
        );

        app.update(Message::Pointer(PointerEvent::Modifiers { shift: false, control: false }));
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

        app.update(Message::BrushKindChanged(BrushKind::Draw));
        for _ in 0..4 {
            press(&mut app, centre_of_viewport());
            release(&mut app);
        }
        let raised = app.volume.sample_world(probe);
        assert!(raised < flat, "drawing should have pushed the surface out past the probe");

        // Now the same gesture with shift held. Nothing about the selection
        // changes, but the strokes must smooth rather than pile on more.
        app.update(Message::Pointer(PointerEvent::Modifiers { shift: true, control: false }));
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
            app.update(Message::SymmetryAxisToggled(axis));
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
            app.update(Message::SymmetryAxisToggled(axis));
            assert!(!app.symmetry.axis(axis));
        }
    }

    #[test]
    fn the_radius_keys_stay_inside_the_range_the_slider_offers() {
        let mut app = app();
        for _ in 0..200 {
            app.update(Message::BrushRadiusScaled(1.5));
        }
        assert_eq!(app.brush.radius, MAX_RADIUS_MM);
        for _ in 0..200 {
            app.update(Message::BrushRadiusScaled(1.0 / 1.5));
        }
        assert_eq!(app.brush.radius, MIN_RADIUS_MM);
    }

    #[test]
    fn control_does_not_invert_a_brush_that_has_no_opposite() {
        let mut app = app();
        app.control = true;

        app.update(Message::BrushKindChanged(BrushKind::Draw));
        assert_eq!(app.stroke_direction(), BrushDirection::Subtract);

        app.update(Message::BrushKindChanged(BrushKind::Smooth));
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
        app.update(Message::BrushKindChanged(BrushKind::Smooth));
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

        app.update(Message::TiltToggled(false));
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

        app.update(Message::ResetSphere);
        assert!(!app.history.can_undo(), "reset must clear history");
        assert!(!app.history.can_redo());
    }

    #[test]
    fn every_brush_can_be_driven_from_the_interface_without_panicking() {
        // Cheap breadth: each brush goes through the whole application path
        // once, which is where a bad plane or a zero normal would surface.
        for kind in BrushKind::ALL {
            let mut app = app();
            app.update(Message::BrushKindChanged(kind));
            app.update(Message::BrushStrengthChanged(0.6));
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
    use brokkr_core::BrickCoord;

    fn app() -> Brokkr {
        Brokkr::with_tablet(crate::tablet::Tablet::inert())
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

        app.update(Message::Resample(coarse_voxel / 2.0));

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
        app.update(Message::Resample(app.voxel_size / 2.0));

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
        app.update(Message::Resample(app.voxel_size));
        assert_eq!(app.volume.brick_count(), before);
        assert!(app.status.is_empty(), "nothing happened, so nothing should be reported");
    }
}
