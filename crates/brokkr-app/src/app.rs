// SPDX-License-Identifier: AGPL-3.0-or-later

//! Application state, input handling and the widget tree.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use brokkr_core::{
    BrickCoord, Brush, BrushDirection, BrushKind, BrushScratch, FalloffCurve, History,
    HistoryStats, MeshScratch, Stamp, Stroke, Symmetry, Volume, VolumeStats, lean_normal, raycast,
};
use glam::{Vec2, Vec3};
use iced::widget::{button, checkbox, column, container, pick_list, row, slider, stack, text};
use iced::{Alignment, Element, Length, Subscription};

use crate::camera::OrbitCamera;
use crate::message::{Message, PointerButton, PointerEvent};
use crate::tablet::{Diagnosis, Tablet};
use crate::theme;
use crate::viewport::{SharedFrame, Viewport};

/// World units are millimetres, because the output of this program is meant to
/// be printed.
///
/// A 60 mm ball at a quarter millimetre voxel is 240 voxels across, which is
/// the 256 cubed effective volume the milestones are measured against.
const MODEL_RADIUS_MM: f32 = 30.0;
const VOXEL_SIZE_MM: f32 = 0.25;

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
    fn record_frame(&mut self) {
        let now = Instant::now();
        if let Some(previous) = self.last_frame.replace(now) {
            if self.frame_ms.len() == FRAME_HISTORY {
                self.frame_ms.pop_front();
            }
            self.frame_ms.push_back(now.duration_since(previous).as_secs_f32() * 1000.0);
        }
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
    mesh_scratch: MeshScratch,
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
}

impl Brokkr {
    pub fn new() -> Self {
        Self::with_tablet(Tablet::start())
    }

    /// Build with a given pressure source, so tests do not go looking through
    /// `/dev/input` and spawn reader threads.
    fn with_tablet(tablet: Tablet) -> Self {
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
            symmetry: Symmetry::Off,
            tablet,
            pressure_enabled: true,
            pressure_curve: 1.0,
            tilt_enabled: true,
            stroke: Stroke::new(),
            history: History::default(),
            shared,
            mesh_scratch: MeshScratch::new(),
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
        for &coord in &self.dirty {
            let mut mesh = self.shared.take_mesh();
            self.volume.mesh_brick(coord, &mut self.mesh_scratch, &mut mesh);
            self.shared.publish(coord, mesh);
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
            let spacing = self.brush.spacing(self.volume.voxel_size());
            self.stroke.advance(point, spacing, &mut self.stamp_centres);
        }

        // Sampled once for the whole event rather than per stamp: the pen has
        // not moved between the stamps that one pointer event interpolates, so
        // re-reading it would only add jitter.
        let pressure = self.tablet.stamp_pressure(self.pressure_enabled, self.pressure_curve);
        let lean = self.pen_lean();

        for index in 0..self.stamp_centres.len() {
            let centre = self.stamp_centres[index];
            // Take the normal from the field at each stamp rather than reusing
            // the one from the raycast, so a stroke curving around a form stays
            // oriented to the surface it is actually on.
            // Leaning the pen rotates the direction the brush pushes in, which
            // steers every brush at once because they all read this normal.
            let normal = lean_normal(self.volume.gradient_world(centre), lean);
            let stamp = Stamp::new(centre, normal, direction).with_pressure(pressure);
            self.brush.apply_symmetric(
                &mut self.volume,
                &stamp,
                self.symmetry,
                &mut self.brush_scratch,
            );
        }
        self.perf.stamps = self.stamp_centres.len();
        self.perf.pressure = pressure;
        self.perf.edit_ms = started.elapsed().as_secs_f32() * 1000.0;

        self.remesh_dirty();
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
            Message::Frame => self.perf.record_frame(),
            Message::BrushKindChanged(kind) => self.brush.kind = kind,
            Message::FalloffChanged(curve) => self.brush.falloff = curve,
            Message::BrushRadiusChanged(radius) => self.brush.radius = radius,
            Message::BrushStrengthChanged(strength) => self.brush.strength = strength,
            Message::SymmetryToggled(on) => {
                self.symmetry = if on { Symmetry::X } else { Symmetry::Off };
            }
            Message::PressureToggled(on) => self.pressure_enabled = on,
            Message::PressureCurveChanged(curve) => self.pressure_curve = curve,
            Message::TiltToggled(on) => self.tilt_enabled = on,
            Message::ResetPressurePeak => self.tablet.reset_peak(),
            Message::Undo => self.undo(),
            Message::Redo => self.redo(),
            Message::ResetSphere => {
                let mut volume = Volume::new(VOXEL_SIZE_MM);
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

    pub fn view(&self) -> Element<'_, Message> {
        let viewport = iced::widget::shader(Viewport::new(Arc::clone(&self.shared)))
            .width(Length::Fill)
            .height(Length::Fill);

        let well = container(viewport)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(theme::viewport_well);

        let scene = stack![well, self.overlay()];

        column![
            self.header(),
            row![container(scene).width(Length::Fill).height(Length::Fill), self.tools()]
                .spacing(theme::S3)
        ]
        .spacing(theme::S3)
        .padding(theme::S3)
        .into()
    }

    fn header(&self) -> Element<'_, Message> {
        container(
            row![
                text("BROKKRSCULPT")
                    .size(theme::CAPTION_SIZE)
                    .font(theme::FONT)
                    .color(theme::ACCENT),
                text("M1 brush system").size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S4)
            .align_y(Alignment::Center),
        )
        .padding(theme::PANEL_PADDING)
        .width(Length::Fill)
        .style(theme::panel)
        .into()
    }

    /// The debug overlay: frame rate, frame time, triangles, bricks, resident
    /// memory and what history is holding.
    fn overlay(&self) -> Element<'_, Message> {
        let pool = self.shared.stats();
        let frame_ms = self.perf.average_frame_ms();
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };

        let volume_mb = self.volume_stats.resident_bytes as f64 / (1024.0 * 1024.0);
        let pool_mb =
            (pool.vertices as f64 * 24.0 + pool.triangles as f64 * 12.0) / (1024.0 * 1024.0);
        let history_mb = self.history_stats.bytes as f64 / (1024.0 * 1024.0);

        let mut lines = vec![
            format!(
                "{fps:6.1} fps    {frame_ms:5.2} ms avg   {:5.2} ms worst",
                self.perf.worst_frame_ms()
            ),
            format!(
                "edit {:5.3} ms   remesh {:5.3} ms   {} stamps   {} dirty   (load {:.0} ms)",
                self.perf.edit_ms,
                self.perf.remesh_ms,
                self.perf.stamps,
                self.perf.dirty_bricks,
                self.perf.load_ms
            ),
            format!(
                "{} triangles   {} meshed bricks   pen {}",
                pool.triangles,
                pool.bricks,
                match self.tablet.devices().first() {
                    Some(device) => format!("{:.2} ({})", self.perf.pressure, device.name),
                    None => self.tablet.diagnosis().explain().to_string(),
                }
            ),
            format!(
                "{} dense + {} uniform bricks   {volume_mb:.1} MB volume   {pool_mb:.1} MB mesh",
                self.volume_stats.dense_bricks, self.volume_stats.uniform_bricks
            ),
            format!(
                "history {} undo / {} redo   {history_mb:.1} MB of {} MB{}",
                self.history_stats.undo_entries,
                self.history_stats.redo_entries,
                self.history_stats.budget_bytes / (1024 * 1024),
                if self.history_stats.dropped > 0 {
                    format!("   {} dropped", self.history_stats.dropped)
                } else {
                    String::new()
                }
            ),
        ];
        if pool.overflowed > 0 {
            lines.push(format!("MESH POOL FULL: {} bricks missing from the view", pool.overflowed));
        }

        let readout = lines.into_iter().fold(column![].spacing(2), |stacked, line| {
            stacked.push(text(line).size(theme::TEXT_SIZE_SMALL).font(theme::MONO))
        });

        container(container(readout).padding(theme::S3).style(theme::overlay_card))
            .padding(theme::S4)
            .into()
    }

    fn tools(&self) -> Element<'_, Message> {
        let invert_hint = if self.brush.kind.is_directional() {
            "ctrl drag removes"
        } else {
            "no opposite: ctrl does nothing"
        };

        let radius = column![
            text(format!("Radius  {:.2} mm", self.brush.radius))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.25..=12.0, self.brush.radius, Message::BrushRadiusChanged).step(0.05),
        ]
        .spacing(theme::S2);

        let strength = column![
            text(format!("Strength  {:.2}", self.brush.strength))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.02..=0.80, self.brush.strength, Message::BrushStrengthChanged).step(0.01),
        ]
        .spacing(theme::S2);

        let falloff = column![
            text("Falloff").size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_DIM),
            pick_list(FalloffCurve::ALL, Some(self.brush.falloff), Message::FalloffChanged)
                .text_size(theme::TEXT_SIZE_SMALL)
                .width(Length::Fill),
        ]
        .spacing(theme::S2);

        let history = row![
            button(text("Undo").size(theme::TEXT_SIZE_SMALL))
                .on_press_maybe(self.history.can_undo().then_some(Message::Undo)),
            button(text("Redo").size(theme::TEXT_SIZE_SMALL))
                .on_press_maybe(self.history.can_redo().then_some(Message::Redo)),
        ]
        .spacing(theme::S2);

        container(
            column![
                text("BRUSH").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                pick_list(BrushKind::ALL, Some(self.brush.kind), Message::BrushKindChanged)
                    .text_size(theme::TEXT_SIZE_SMALL)
                    .width(Length::Fill),
                text(invert_hint).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                radius,
                strength,
                falloff,
                checkbox(self.symmetry == Symmetry::X)
                    .label("X symmetry")
                    .on_toggle(Message::SymmetryToggled)
                    .text_size(theme::TEXT_SIZE_SMALL),
                self.pen_panel(),
                text("HISTORY").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                history,
                button(text("Reset sphere").size(theme::TEXT_SIZE_SMALL))
                    .on_press(Message::ResetSphere),
                text(
                    "drag: sculpt\nctrl drag: invert\nright drag: orbit\nshift right drag: pan\nwheel: zoom\nctrl z, ctrl shift z: undo, redo"
                )
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S4),
        )
        .padding(theme::PANEL_PADDING)
        .width(Length::Fixed(240.0))
        .height(Length::Fill)
        .style(theme::panel)
        .into()
    }

    /// Pen controls, plus enough of a live readout that a user can tell whether
    /// their tablet is being seen at all.
    ///
    /// Without this, a tablet that is connected but unreadable looks exactly
    /// like a mouse: strokes work, they just never vary. The device name, the
    /// device's own pressure range and a live peak turn that into something
    /// answerable in a few seconds.
    fn pen_panel(&self) -> Element<'_, Message> {
        let devices = self.tablet.devices();
        let status: Element<'_, Message> = match devices.first() {
            Some(device) => column![
                text(device.name.clone()).size(theme::CAPTION_SIZE).color(theme::OK),
                text(format!("{} levels", device.pressure_max))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE),
            ]
            .spacing(1)
            .into(),
            None => text(self.tablet.diagnosis().explain())
                .size(theme::CAPTION_SIZE)
                .color(match self.tablet.diagnosis() {
                    Diagnosis::PermissionDenied => theme::WARN,
                    _ => theme::TEXT_MUTE,
                })
                .into(),
        };

        let pen = self.tablet.state();
        let live = if pen.in_proximity {
            format!(
                "{} {:.2}  peak {:.2}\ntilt {:+.2} {:+.2}",
                if pen.eraser { "eraser" } else { "tip   " },
                pen.pressure,
                self.tablet.peak(),
                pen.tilt.x,
                pen.tilt.y
            )
        } else {
            format!("pen away   peak {:.2}", self.tablet.peak())
        };

        let capabilities = devices.first().map(|device| {
            let mut parts = Vec::new();
            if device.has_tilt {
                parts.push("tilt");
            }
            if device.has_eraser {
                parts.push("eraser");
            }
            if parts.is_empty() { "pressure only".to_string() } else { parts.join(", ") }
        });

        column![
            text("PEN").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            status,
            checkbox(self.pressure_enabled)
                .label("Pressure")
                .on_toggle(Message::PressureToggled)
                .text_size(theme::TEXT_SIZE_SMALL),
            text(format!("Curve  {:.2}", self.pressure_curve))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.30..=3.00, self.pressure_curve, Message::PressureCurveChanged).step(0.05),
            checkbox(self.tilt_enabled)
                .label("Tilt steers stroke")
                .on_toggle(Message::TiltToggled)
                .text_size(theme::TEXT_SIZE_SMALL),
            text(capabilities.unwrap_or_default())
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            row![
                text(live).size(theme::CAPTION_SIZE).font(theme::MONO).color(theme::TEXT_DIM),
                button(text("reset").size(theme::CAPTION_SIZE))
                    .on_press(Message::ResetPressurePeak),
            ]
            .spacing(theme::S2)
            .align_y(Alignment::Center),
        ]
        .spacing(theme::S2)
        .into()
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
        app.update(Message::SymmetryToggled(true));
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
