// SPDX-License-Identifier: AGPL-3.0-or-later

//! Application state, input handling and the widget tree.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use brokkr_core::{
    BrickCoord, BrushDirection, DrawBrush, MeshScratch, Volume, VolumeStats, raycast,
};
use glam::{Vec2, Vec3};
use iced::widget::{button, column, container, row, slider, stack, text};
use iced::{Alignment, Element, Length, Subscription};

use crate::camera::OrbitCamera;
use crate::message::{Message, PointerButton, PointerEvent};
use crate::theme;
use crate::viewport::{SharedFrame, Viewport};

/// World units are millimetres, because the output of this program is meant to
/// be printed.
///
/// A 60 mm ball at a quarter millimetre voxel is 240 voxels across, which is
/// the 256 cubed effective volume M0 is measured against.
const MODEL_RADIUS_MM: f32 = 30.0;
const VOXEL_SIZE_MM: f32 = 0.25;

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
    brush: DrawBrush,
    shared: Arc<SharedFrame>,
    scratch: MeshScratch,
    dirty: Vec<BrickCoord>,
    drag: Option<Drag>,
    /// Last pointer position in widget pixels, for drag deltas.
    cursor: Option<Vec2>,
    viewport_size: Vec2,
    shift: bool,
    control: bool,
    perf: Perf,
    volume_stats: VolumeStats,
}

impl Brokkr {
    pub fn new() -> Self {
        let shared = SharedFrame::new();
        let mut volume = Volume::new(VOXEL_SIZE_MM);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS_MM);
        // Everything the sphere touches plus a one brick margin, because bricks
        // with no voxels of their own still own the quads on their low faces.
        volume.mark_everything_dirty();

        let mut app = Self {
            volume,
            camera: OrbitCamera::framing(Vec3::ZERO, MODEL_RADIUS_MM),
            brush: DrawBrush { radius: 3.0, strength: 0.25 },
            shared,
            scratch: MeshScratch::new(),
            dirty: Vec::new(),
            drag: None,
            cursor: None,
            viewport_size: Vec2::new(1280.0, 720.0),
            shift: false,
            control: false,
            perf: Perf::default(),
            volume_stats: VolumeStats::default(),
        };
        app.remesh_dirty();
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
            self.volume.mesh_brick(coord, &mut self.scratch, &mut mesh);
            self.shared.publish(coord, mesh);
        }
        self.perf.remesh_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.volume_stats = self.volume.stats();
    }

    /// Cast through a widget pixel position and stamp the brush where it lands.
    fn sculpt_at(&mut self, pixel: Vec2, direction: BrushDirection) {
        let aspect = self.viewport_size.x / self.viewport_size.y.max(1.0);
        let ndc = OrbitCamera::ndc_from_pixels(pixel, self.viewport_size);
        let (origin, ray) = self.camera.ray(ndc, aspect);

        let Some(hit) = raycast(&self.volume, origin, ray, self.camera.far) else {
            log::debug!("raycast missed from {origin:?} along {ray:?}");
            return;
        };
        log::debug!("stamp {direction:?} at {:?}", hit.position);

        let started = Instant::now();
        self.brush.apply(&mut self.volume, hit.position, direction);
        self.perf.edit_ms = started.elapsed().as_secs_f32() * 1000.0;

        self.remesh_dirty();
    }

    fn on_pointer(&mut self, event: PointerEvent) {
        if !matches!(event, PointerEvent::Moved { .. }) {
            log::debug!("pointer {event:?}");
        }
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
                    PointerButton::Left => DragKind::Sculpt(if self.control {
                        BrushDirection::Subtract
                    } else {
                        BrushDirection::Add
                    }),
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
                    self.sculpt_at(position, direction);
                }
            }
            PointerEvent::Released { button } => {
                if self.drag.is_some_and(|drag| drag.button == button) {
                    self.drag = None;
                }
            }
            PointerEvent::Moved { position, size } => {
                self.viewport_size = Vec2::new(size.x, size.y);
                let position = Vec2::new(position.x, position.y);
                let delta = self.cursor.map(|previous| position - previous).unwrap_or(Vec2::ZERO);
                self.cursor = Some(position);

                match self.drag.map(|drag| drag.kind) {
                    // Stroke interpolation is M1 work, so this stamps once per
                    // event. A fast drag will leave gaps until then.
                    Some(DragKind::Sculpt(direction)) => self.sculpt_at(position, direction),
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
            Message::BrushRadiusChanged(radius) => self.brush.radius = radius,
            Message::BrushStrengthChanged(strength) => self.brush.strength = strength,
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
                text("M0 vertical slice").size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S4)
            .align_y(Alignment::Center),
        )
        .padding(theme::PANEL_PADDING)
        .width(Length::Fill)
        .style(theme::panel)
        .into()
    }

    /// The debug overlay: frame rate, frame time, triangles, bricks and
    /// resident memory, as the milestone requires.
    fn overlay(&self) -> Element<'_, Message> {
        let pool = self.shared.stats();
        let frame_ms = self.perf.average_frame_ms();
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };

        let volume_mb = self.volume_stats.resident_bytes as f64 / (1024.0 * 1024.0);
        let pool_mb =
            (pool.vertices as f64 * 24.0 + pool.triangles as f64 * 12.0) / (1024.0 * 1024.0);

        let mut lines = vec![
            format!(
                "{fps:6.1} fps    {frame_ms:5.2} ms avg   {:5.2} ms worst",
                self.perf.worst_frame_ms()
            ),
            format!(
                "edit {:5.3} ms   remesh {:5.3} ms   {} dirty   (load {:.0} ms)",
                self.perf.edit_ms, self.perf.remesh_ms, self.perf.dirty_bricks, self.perf.load_ms
            ),
            format!("{} triangles   {} meshed bricks", pool.triangles, pool.bricks),
            format!(
                "{} dense + {} uniform bricks   {volume_mb:.1} MB volume   {pool_mb:.1} MB mesh",
                self.volume_stats.dense_bricks, self.volume_stats.uniform_bricks
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
        let radius = column![
            text(format!("Radius  {:.2} mm", self.brush.radius))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.5..=12.0, self.brush.radius, Message::BrushRadiusChanged).step(0.05),
        ]
        .spacing(theme::S2);

        let strength = column![
            text(format!("Strength  {:.2}", self.brush.strength))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.02..=0.60, self.brush.strength, Message::BrushStrengthChanged).step(0.01),
        ]
        .spacing(theme::S2);

        container(
            column![
                text("DRAW BRUSH").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                radius,
                strength,
                button(text("Reset sphere").size(theme::TEXT_SIZE_SMALL))
                    .on_press(Message::ResetSphere),
                text("drag: sculpt\nctrl drag: carve\nright drag: orbit\nshift right drag: pan\nwheel: zoom")
                    .size(theme::TEXT_SIZE_SMALL)
                    .color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S5),
        )
        .padding(theme::PANEL_PADDING)
        .width(Length::Fixed(220.0))
        .height(Length::Fill)
        .style(theme::panel)
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
    use iced::Vector;

    /// The whole input to geometry path, with no window and no GPU: a press at
    /// the centre of the viewport must raycast onto the sphere, stamp the
    /// brush, and leave meshed bricks waiting for upload.
    ///
    /// Every piece of this is covered elsewhere, but only in isolation. A
    /// camera whose ray points the wrong way, or a size that never reaches the
    /// projection, breaks nothing that the other tests can see.
    #[test]
    fn a_press_at_the_centre_of_the_viewport_changes_the_model() {
        let mut app = Brokkr::new();
        let size = Vector::new(800.0, 600.0);

        // The camera frames the origin, so the centre of the viewport looks
        // straight at the front of the sphere.
        let front = app.camera.eye().normalize() * MODEL_RADIUS_MM;
        let before = app.volume.sample_world(front);

        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: Vector::new(size.x / 2.0, size.y / 2.0),
            size,
        });

        assert!(app.perf.edit_ms > 0.0, "no edit was timed, so the raycast missed the sphere");
        assert!(app.perf.dirty_bricks > 0, "the stroke dirtied nothing");
        assert!(
            app.volume.sample_world(front) < before,
            "adding clay should have pushed the field negative at the surface"
        );
    }

    #[test]
    fn a_press_that_misses_the_model_changes_nothing() {
        let mut app = Brokkr::new();
        let size = Vector::new(800.0, 600.0);
        let bricks_before = app.volume.brick_count();

        // The far corner of the viewport looks past the sphere into empty space.
        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: Vector::new(2.0, 2.0),
            size,
        });

        assert_eq!(app.volume.brick_count(), bricks_before, "a miss must not allocate");
        assert_eq!(app.perf.dirty_bricks, 0, "a miss must not schedule a remesh");
    }

    #[test]
    fn orbiting_moves_the_camera_without_touching_the_model() {
        let mut app = Brokkr::new();
        let size = Vector::new(800.0, 600.0);
        let yaw = app.camera.yaw;
        let bricks = app.volume.brick_count();

        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Right,
            position: Vector::new(400.0, 300.0),
            size,
        });
        app.on_pointer(PointerEvent::Moved { position: Vector::new(460.0, 300.0), size });

        assert_ne!(app.camera.yaw, yaw, "a right drag should have orbited");
        assert_eq!(app.volume.brick_count(), bricks, "orbiting must not sculpt");
    }

    #[test]
    fn releasing_a_different_button_does_not_cancel_a_stroke() {
        let mut app = Brokkr::new();
        let size = Vector::new(800.0, 600.0);

        app.on_pointer(PointerEvent::Pressed {
            button: PointerButton::Left,
            position: Vector::new(400.0, 300.0),
            size,
        });
        app.on_pointer(PointerEvent::Released { button: PointerButton::Right });
        assert!(app.drag.is_some(), "the left button drag should still be live");

        app.on_pointer(PointerEvent::Released { button: PointerButton::Left });
        assert!(app.drag.is_none());
    }
}
