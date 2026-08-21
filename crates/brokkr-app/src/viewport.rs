// SPDX-License-Identifier: AGPL-3.0-only

//! The 3D viewport, embedded in the Iced widget tree through the `shader`
//! widget.
//!
//! Iced owns the window, the surface and the wgpu device. The `shader` widget
//! hands us that same device inside its `prepare` and `render` callbacks, which
//! is why `wgpu` has to be the exact version `iced_wgpu` depends on. There is
//! no second window and no transparent overlay.
//!
//! The application state cannot be borrowed by the primitive, because a
//! primitive has to be `'static` and `Send`. Instead the two sides share an
//! [`SharedFrame`]: the application pushes finished brick meshes and camera
//! uniforms into it, and `prepare` drains them onto the GPU. Mesh buffers are
//! handed back after upload and reused, so a stroke settles into steady state
//! without allocating.

use std::sync::{Arc, Mutex};

use brokkr_core::{BrickCoord, BrickMesh, BrushKind, MirrorAxis};
use brokkr_gpu::{Frustum, OverlayBatch, PixelRect, PoolStats, SculptRenderer, Uniforms};
use iced::mouse;
use iced::widget::shader;
use iced::{Rectangle, Vector};

use crate::app::SizingTarget;
use crate::camera::OrbitCamera;
use crate::message::{Message, PointerButton, PointerEvent};
use crate::navcube;

/// One brick's mesh on its way to the GPU.
#[derive(Debug)]
pub struct PendingUpload {
    pub coord: BrickCoord,
    pub mesh: BrickMesh,
}

/// The hand off between the application and the render callbacks.
#[derive(Debug, Default)]
pub struct SharedFrame {
    /// The camera, not a finished matrix. The projection depends on the
    /// widget's aspect ratio, which only the render callback knows, so the
    /// matrices are built there and there is no frame of lag after a resize.
    camera: Mutex<OrbitCamera>,
    pending: Mutex<Vec<PendingUpload>>,
    /// Mesh buffers returned after upload, so the next remesh can refill them
    /// instead of allocating. This is what keeps a stroke out of the allocator.
    spare: Mutex<Vec<BrickMesh>>,
    stats: Mutex<PoolStats>,
    /// The navigation cube's geometry, in its own batch because it is drawn in
    /// its own pass with its own matrix.
    cube: Mutex<OverlayBatch>,
    /// Which GPU and backend iced actually chose, recorded the first time the
    /// pipeline is built. The application cannot ask wgpu directly -- iced owns
    /// the adapter -- and it is the first thing a bug report needs.
    adapter: Mutex<Option<String>>,
    /// The brush ring and the mirror planes, rebuilt by the application
    /// whenever something they depend on changes.
    overlay: Mutex<OverlayBatch>,
}

impl SharedFrame {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Take a mesh buffer to fill, reusing a returned one when there is any.
    pub fn take_mesh(&self) -> BrickMesh {
        self.spare.lock().expect("shared frame poisoned").pop().unwrap_or_default()
    }

    /// Queue a filled mesh for upload on the next frame.
    pub fn publish(&self, coord: BrickCoord, mesh: BrickMesh) {
        self.pending.lock().expect("shared frame poisoned").push(PendingUpload { coord, mesh });
    }

    pub fn set_camera(&self, camera: OrbitCamera) {
        *self.camera.lock().expect("shared frame poisoned") = camera;
    }

    /// Hand over this frame's overlay geometry, taking the caller's buffer and
    /// giving back the previous one so neither side allocates.
    pub fn swap_overlay(&self, batch: &mut OverlayBatch) {
        std::mem::swap(&mut *self.overlay.lock().expect("shared frame poisoned"), batch);
    }

    /// The same hand off for the navigation cube.
    pub fn swap_cube(&self, batch: &mut OverlayBatch) {
        std::mem::swap(&mut *self.cube.lock().expect("shared frame poisoned"), batch);
    }

    /// The overlay geometry currently waiting for the renderer.
    ///
    /// The application swaps its buffer in here rather than keeping it, so this
    /// is the only place the current frame's geometry can be read.
    #[cfg(test)]
    pub fn overlay_snapshot(&self) -> OverlayBatch {
        self.overlay.lock().expect("shared frame poisoned").clone()
    }

    /// The GPU iced chose, for diagnostics. "unknown" until the first frame.
    pub fn adapter_summary(&self) -> String {
        self.adapter
            .lock()
            .expect("shared frame poisoned")
            .clone()
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn set_adapter(&self, summary: String) {
        let mut held = self.adapter.lock().expect("shared frame poisoned");
        if held.is_none() {
            *held = Some(summary);
        }
    }

    /// The pool counters as of the last frame, for the debug overlay.
    pub fn stats(&self) -> PoolStats {
        *self.stats.lock().expect("shared frame poisoned")
    }

    /// Tests have no GPU and so no renderer to fill the counters in; this is
    /// how they stand in for one. The resample guard reads these numbers.
    #[cfg(test)]
    pub fn set_stats_for_tests(&self, stats: PoolStats) {
        *self.stats.lock().expect("shared frame poisoned") = stats;
    }
}

/// The `shader` widget program. Cheap to construct: `view` rebuilds it every
/// time the UI is laid out, so it holds nothing but a handle.
#[derive(Debug)]
pub struct Viewport {
    shared: Arc<SharedFrame>,
}

impl Viewport {
    pub fn new(shared: Arc<SharedFrame>) -> Self {
        Self { shared }
    }
}

/// Translate a widget local pixel position and the widget size into a pointer
/// event the application can act on.
fn pointer_position(bounds: Rectangle, cursor: mouse::Cursor) -> Option<(Vector, Vector)> {
    // Deliberately not `position_in`: a drag that leaves the widget must keep
    // orbiting, so positions outside the bounds are wanted, not discarded.
    let position = cursor.position()?;
    Some((
        Vector::new(position.x - bounds.x, position.y - bounds.y),
        Vector::new(bounds.width, bounds.height),
    ))
}

fn button_of(button: mouse::Button) -> Option<PointerButton> {
    match button {
        mouse::Button::Left => Some(PointerButton::Left),
        mouse::Button::Right => Some(PointerButton::Right),
        mouse::Button::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

/// How much one press of the radius keys changes it.
///
/// Multiplicative, because the radius spans fifty to one: a fixed step would
/// crawl at the coarse end and jump in whole multiples at the fine end.
const RADIUS_STEP: f32 = 1.15;

/// The message a key press means, if it means anything.
///
/// Pulled out of `update` so it can be tested without building a widget tree
/// or a window, which on this machine is the only way to test input at all:
/// it is a Wayland session, and XTEST pointer and key synthesis silently does
/// nothing there.
pub(crate) fn shortcut(character: &str, command: bool, shift: bool) -> Option<Message> {
    if command {
        // The only chorded shortcuts. Anything else with control held belongs
        // to the toolkit or the window manager.
        return character.eq_ignore_ascii_case("z").then_some(if shift {
            Message::Redo
        } else {
            Message::Undo
        });
    }

    // Brushes are numbered in the order the tool strip shows them, so the key
    // and the button are always the same thing.
    if let Ok(digit) = character.parse::<usize>()
        && let Some(index) = digit.checked_sub(1)
        && let Some(kind) = BrushKind::ALL.get(index)
    {
        return Some(Message::BrushKindChanged(*kind));
    }

    match character.to_ascii_lowercase().as_str() {
        // ZBrush's own keys for these two sliders, taken rather than invented so
        // muscle memory carries over: S is Draw Size, U is Z Intensity.
        "s" => Some(Message::SizingStarted(SizingTarget::Radius)),
        "u" => Some(Message::SizingStarted(SizingTarget::Strength)),
        "x" => Some(Message::SymmetryAxisToggled(MirrorAxis::X)),
        "y" => Some(Message::SymmetryAxisToggled(MirrorAxis::Y)),
        "z" => Some(Message::SymmetryAxisToggled(MirrorAxis::Z)),
        "[" => Some(Message::BrushRadiusScaled(1.0 / RADIUS_STEP)),
        "]" => Some(Message::BrushRadiusScaled(RADIUS_STEP)),
        _ => None,
    }
}

/// Translate an event into the pointer event the application should see, and
/// whether the viewport may CAPTURE it — which stops every widget after it in
/// traversal order from seeing the event at all.
///
/// # Capture is the whole bug surface here, so the rule is stated once
///
/// **Only events that are bounds-checked may capture.** The viewport
/// deliberately handles cursor moves and button releases wherever the cursor
/// is, because a sculpting drag that leaves the viewport must keep sculpting
/// and its release must still end the stroke. From the day the viewport
/// existed (M0, 2026-08-17) this function captured those too — and `iced`'s
/// `button` is exactly the widget that
/// honours capture (`if shell.is_event_captured() { return; }`), while slider,
/// checkbox and text_input ignore it. The shader traverses before the right
/// panel, so every release anywhere in the window was swallowed before a panel
/// button could see it: presses armed the buttons (that arm is bounds-gated),
/// releases never arrived, and **every button in the properties panel was
/// unclickable** while every slider beside them worked. Verified live on
/// 2026-08-21 with a raw-event probe: ~20 presses over panel buttons, zero
/// messages dispatched.
///
/// Publishing without capturing is safe from misfires by iced's own design: a
/// button only fires when the press STARTED on it, so a sculpt release passing
/// over the panel cannot click anything.
///
/// Keyboard shortcuts used to be captured here too, which was its own bug —
/// the capture stole `1`–`7`, `s`, `u`, `x`, `y`, `z` from every text field in
/// the application, because the shader traverses first. They now live in a
/// subscription over events the widget tree IGNORED (`app.rs`), which is what
/// makes them focus-aware: a focused text input consumes its keystrokes, and
/// the shortcut only fires when nothing wanted the key.
fn route_pointer(
    event: &iced::Event,
    bounds: Rectangle,
    cursor: mouse::Cursor,
) -> Option<(PointerEvent, bool)> {
    let routed = match event {
        iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
            let (position, size) = pointer_position(bounds, cursor)?;
            (PointerEvent::Moved { position, size }, false)
        }
        iced::Event::Mouse(mouse::Event::ButtonPressed(button)) => {
            // Only start a drag that began inside the viewport, so a click
            // on a panel does not sculpt.
            cursor.position_in(bounds)?;
            let (position, size) = pointer_position(bounds, cursor)?;
            (PointerEvent::Pressed { button: button_of(*button)?, position, size }, true)
        }
        iced::Event::Mouse(mouse::Event::ButtonReleased(button)) => {
            // Handled wherever the cursor is: releasing outside the widget
            // still has to end the drag. NOT captured, for the reason above.
            (PointerEvent::Released { button: button_of(*button)? }, false)
        }
        iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
            cursor.position_in(bounds)?;
            let amount = match delta {
                mouse::ScrollDelta::Lines { y, .. } => *y,
                // Pixel deltas are far larger per notch than line deltas.
                mouse::ScrollDelta::Pixels { y, .. } => *y / 40.0,
            };
            (PointerEvent::Scrolled { amount }, true)
        }
        iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(modifiers)) => (
            PointerEvent::Modifiers {
                shift: modifiers.shift(),
                control: modifiers.control(),
                alt: modifiers.alt(),
            },
            // Broadcast state, not an interaction: capturing it would starve
            // every other widget of the same update.
            false,
        ),
        _ => return None,
    };
    Some(routed)
}

impl shader::Program<Message> for Viewport {
    // Drag state lives in the application, not the widget, because sculpting
    // mutates the volume and that is the application's to own.
    type State = ();
    type Primitive = SculptPrimitive;

    fn draw(&self, _state: &(), _cursor: mouse::Cursor, _bounds: Rectangle) -> SculptPrimitive {
        SculptPrimitive { shared: Arc::clone(&self.shared) }
    }

    fn update(
        &self,
        _state: &mut (),
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Message>> {
        let (pointer, captures) = route_pointer(event, bounds, cursor)?;
        let action = shader::Action::publish(Message::Pointer(pointer));
        Some(if captures { action.and_capture() } else { action })
    }

    fn mouse_interaction(
        &self,
        _state: &(),
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if cursor.is_over(bounds) {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::default()
        }
    }
}

/// What the widget draws this frame. Holding only a handle keeps `draw` free of
/// per frame copying.
#[derive(Debug)]
pub struct SculptPrimitive {
    shared: Arc<SharedFrame>,
}

impl shader::Primitive for SculptPrimitive {
    type Pipeline = SculptPipeline;

    fn prepare(
        &self,
        pipeline: &mut SculptPipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        // The depth attachment has to match the colour attachment, which is the
        // whole window rather than just this widget.
        let size = viewport.physical_size();
        pipeline.renderer.resize(device, size.width, size.height);

        let camera = *self.shared.camera.lock().expect("shared frame poisoned");
        let aspect = bounds.width / bounds.height.max(1.0);
        let view = camera.view();
        let view_projection = camera.projection(aspect) * view;
        let uniforms = Uniforms {
            view_projection: view_projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            srgb_target: u32::from(pipeline.renderer.target_is_srgb()),
            padding: [0; 3],
        };
        pipeline.renderer.write_uniforms(queue, &uniforms);
        {
            let overlay = self.shared.overlay.lock().expect("shared frame poisoned");
            pipeline.renderer.write_overlay(device, queue, &overlay, view_projection);
        }
        {
            let cube = self.shared.cube.lock().expect("shared frame poisoned");
            pipeline.renderer.write_cube(device, queue, &cube, navcube::view_projection(&camera));
        }
        // The cube's corner box is defined in logical pixels so it stays the
        // same physical size at any scale factor, but a render pass wants
        // physical ones. Worked out here because `render` sees neither the
        // widget's logical bounds nor the scale factor.
        let scale = viewport.scale_factor();
        let (corner, size) = navcube::corner_rect(glam::Vec2::new(bounds.width, bounds.height));
        pipeline.cube_offset = PixelRect {
            x: (corner.x * scale).max(0.0) as u32,
            y: (corner.y * scale).max(0.0) as u32,
            width: (size.x * scale).max(1.0) as u32,
            height: (size.y * scale).max(1.0) as u32,
        };
        // Culling has to use the same matrix the vertex shader will, or bricks
        // vanish a frame before or after they should.
        pipeline.frustum = Frustum::from_view_projection(view_projection);

        let drained: Vec<PendingUpload> = {
            let mut pending = self.shared.pending.lock().expect("shared frame poisoned");
            std::mem::take(&mut *pending)
        };
        if !drained.is_empty() {
            let mut spare = self.shared.spare.lock().expect("shared frame poisoned");
            for upload in drained {
                pipeline.renderer.upload_brick(queue, upload.coord, &upload.mesh);
                spare.push(upload.mesh);
            }
        }

        *self.shared.stats.lock().expect("shared frame poisoned") = pipeline.renderer.stats();
        self.shared.set_adapter(pipeline.renderer.adapter_summary());
    }

    fn render(
        &self,
        pipeline: &SculptPipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        pipeline.renderer.render(
            encoder,
            target,
            PixelRect {
                x: clip_bounds.x,
                y: clip_bounds.y,
                width: clip_bounds.width,
                height: clip_bounds.height,
            },
            &pipeline.frustum,
        );

        // The cube's box, offset from the widget into the window. Clamped to the
        // clip rect so a viewport narrower than the cube cannot scissor outside
        // its own bounds and paint over the panels.
        let cube = &pipeline.cube_offset;
        let x = clip_bounds.x + cube.x;
        let y = clip_bounds.y + cube.y;
        let width = cube.width.min(clip_bounds.width.saturating_sub(cube.x));
        let height = cube.height.min(clip_bounds.height.saturating_sub(cube.y));
        pipeline.renderer.render_cube(encoder, target, PixelRect { x, y, width, height });
    }
}

/// The GPU state shared by every frame of the viewport. Iced builds this once,
/// the first time it sees a [`SculptPrimitive`].
#[derive(Debug)]
pub struct SculptPipeline {
    renderer: SculptRenderer,
    /// The cube's corner box in physical pixels, relative to the widget's own
    /// origin. Worked out in `prepare`, which is the only place the scale factor
    /// and the logical bounds are both available.
    cube_offset: PixelRect,
    /// Rebuilt in `prepare` from the same matrix the shader gets, and used in
    /// `render`, which is a separate call and has no access to the camera.
    frustum: Frustum,
}

impl shader::Pipeline for SculptPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self {
            renderer: SculptRenderer::new(device, queue, format),
            cube_offset: PixelRect { x: 0, y: 0, width: 0, height: 0 },
            frustum: Frustum::from_view_projection(glam::Mat4::IDENTITY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::widget::shader::Program;
    use iced::{Point, Size};

    /// The capture rule, pinned: **only bounds-checked events may capture.**
    ///
    /// The failure this guards against was live for a year and looked like a
    /// broken panel, not a broken viewport: the shader traverses before the
    /// properties panel, `button` honours capture, and a captured release
    /// anywhere in the window meant no panel button could ever fire. If one of
    /// these assertions starts failing, read `route_pointer`'s comment before
    /// "fixing" the test.
    #[test]
    fn only_bounds_checked_events_capture() {
        let inside = mouse::Cursor::Available(Point::new(500.0, 300.0));
        let outside = mouse::Cursor::Available(Point::new(950.0, 690.0));

        let moved = iced::Event::Mouse(mouse::Event::CursorMoved { position: Point::ORIGIN });
        let pressed = iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left));
        let released = iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left));

        // A move is wanted wherever the cursor is, and must never capture:
        // hover in every widget after the shader depends on seeing it.
        let (_, captures) = route_pointer(&moved, bounds(), outside).expect("moves are routed");
        assert!(!captures, "a cursor move outside the viewport was captured");
        let (_, captures) = route_pointer(&moved, bounds(), inside).expect("moves are routed");
        assert!(!captures, "even inside the viewport, capturing moves starves the panel");

        // A release must arrive wherever the cursor is -- it ends a drag that
        // left the widget -- and must never capture: it is the event a button
        // fires on.
        let (_, captures) =
            route_pointer(&released, bounds(), outside).expect("releases are routed");
        assert!(!captures, "a release outside the viewport was captured");

        // A press inside the viewport is genuinely ours and may capture...
        let (_, captures) = route_pointer(&pressed, bounds(), inside).expect("press inside");
        assert!(captures, "a press inside the viewport should be claimed");
        // ...and a press outside is none of our business at all.
        assert!(
            route_pointer(&pressed, bounds(), outside).is_none(),
            "a press outside the viewport must not reach the sculpt at all"
        );
    }

    /// Keyboard events are none of `route_pointer`'s business: shortcuts fire
    /// from a subscription over IGNORED events, so a focused text field eats
    /// its own keystrokes. This is what keeps `1`-`7`, `s`, `u`, `x`, `y`, `z`
    /// typeable in the print-size field.
    #[test]
    fn keyboard_events_are_not_routed_by_the_viewport() {
        let key = iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
            key: iced::keyboard::Key::Character("s".into()),
            modified_key: iced::keyboard::Key::Character("s".into()),
            physical_key: iced::keyboard::key::Physical::Code(iced::keyboard::key::Code::KeyS),
            location: iced::keyboard::Location::Standard,
            modifiers: iced::keyboard::Modifiers::default(),
            text: None,
            repeat: false,
        });
        let cursor = mouse::Cursor::Available(Point::new(500.0, 300.0));
        assert!(route_pointer(&key, bounds(), cursor).is_none());
    }

    fn bounds() -> Rectangle {
        Rectangle::new(Point::new(100.0, 50.0), Size::new(800.0, 600.0))
    }

    fn press(at: Point) -> (Viewport, iced::Event, mouse::Cursor) {
        (
            Viewport::new(SharedFrame::new()),
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            mouse::Cursor::Available(at),
        )
    }

    /// The routing from an Iced event to an application message has no other
    /// coverage: the offscreen render test drives the brush directly, and a
    /// broken translation here leaves an application that draws perfectly and
    /// does nothing when clicked.
    #[test]
    fn a_press_inside_the_viewport_reports_a_widget_local_position() {
        let (viewport, event, cursor) = press(Point::new(300.0, 200.0));
        let action = viewport
            .update(&mut (), &event, bounds(), cursor)
            .expect("a press inside the viewport must produce a message");

        let (message, _, status) = action.into_inner();
        assert_eq!(status, iced::event::Status::Captured);
        match message {
            Some(Message::Pointer(PointerEvent::Pressed { button, position, size })) => {
                assert_eq!(button, PointerButton::Left);
                // Widget local, so the widget's own origin is subtracted.
                assert_eq!((position.x, position.y), (200.0, 150.0));
                assert_eq!((size.x, size.y), (800.0, 600.0));
            }
            other => panic!("expected a pressed pointer message, got {other:?}"),
        }
    }

    #[test]
    fn a_press_outside_the_viewport_is_ignored() {
        // Otherwise clicking a panel would also carve the model.
        let (viewport, event, cursor) = press(Point::new(10.0, 10.0));
        assert!(viewport.update(&mut (), &event, bounds(), cursor).is_none());
    }

    #[test]
    fn a_release_outside_the_viewport_still_ends_the_drag() {
        // A drag that runs off the edge of the widget has to be finishable.
        let viewport = Viewport::new(SharedFrame::new());
        let event = iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left));
        let action = viewport
            .update(&mut (), &event, bounds(), mouse::Cursor::Available(Point::new(5.0, 5.0)))
            .expect("a release anywhere must end the drag");

        assert!(matches!(
            action.into_inner().0,
            Some(Message::Pointer(PointerEvent::Released { button: PointerButton::Left }))
        ));
    }

    #[test]
    fn a_mesh_buffer_handed_back_after_upload_is_reused() {
        // This is what keeps a stroke out of the allocator, so it is worth
        // pinning rather than assuming.
        let shared = SharedFrame::new();
        let mut mesh = shared.take_mesh();
        mesh.vertices.reserve(4096);
        let capacity = mesh.vertices.capacity();
        assert!(capacity >= 4096);

        shared.publish(BrickCoord::new(0, 0, 0), mesh);
        // Stand in for what prepare does once the upload has happened.
        let drained: Vec<PendingUpload> =
            std::mem::take(&mut *shared.pending.lock().expect("shared frame poisoned"));
        for upload in drained {
            shared.spare.lock().expect("shared frame poisoned").push(upload.mesh);
        }

        let recycled = shared.take_mesh();
        assert_eq!(recycled.vertices.capacity(), capacity, "the buffer was not reused");
    }
}

#[cfg(test)]
mod shortcut_tests {
    use super::*;

    /// Note the test target: `shortcut`, not a synthesised key event. This is
    /// a Wayland session, where XTEST key and pointer synthesis silently does
    /// nothing, so driving the real widget from a test is not an option.
    fn press(character: &str) -> Option<Message> {
        shortcut(character, false, false)
    }

    #[test]
    fn the_digits_select_brushes_in_the_order_the_tool_strip_shows_them() {
        for (index, kind) in BrushKind::ALL.into_iter().enumerate() {
            let key = (index + 1).to_string();
            assert!(
                matches!(press(&key), Some(Message::BrushKindChanged(selected)) if selected == kind),
                "key {key} should select {kind}"
            );
        }
        // One past the last brush, so the mapping cannot silently wrap round.
        let past_the_end = (BrushKind::ALL.len() + 1).to_string();
        assert!(press(&past_the_end).is_none());
        assert!(press("0").is_none(), "there is no brush zero");
    }

    #[test]
    fn xyz_toggle_their_own_mirror_plane() {
        for (key, axis) in [("x", MirrorAxis::X), ("y", MirrorAxis::Y), ("z", MirrorAxis::Z)] {
            assert!(
                matches!(press(key), Some(Message::SymmetryAxisToggled(a)) if a == axis),
                "{key} should toggle {}",
                axis.label()
            );
            // Shift alone must not change what a letter means.
            assert!(matches!(shortcut(key, false, true), Some(Message::SymmetryAxisToggled(_))));
        }
        // Capitals reach the same place.
        assert!(matches!(press("X"), Some(Message::SymmetryAxisToggled(MirrorAxis::X))));
    }

    /// `z` is the collision worth pinning: bare it mirrors, with control it
    /// undoes, and the two must never be confused.
    #[test]
    fn control_z_still_undoes_rather_than_mirroring() {
        assert!(matches!(shortcut("z", true, false), Some(Message::Undo)));
        assert!(matches!(shortcut("z", true, true), Some(Message::Redo)));
        assert!(matches!(shortcut("Z", true, false), Some(Message::Undo)));
        assert!(matches!(press("z"), Some(Message::SymmetryAxisToggled(MirrorAxis::Z))));
    }

    #[test]
    fn the_bracket_keys_scale_the_radius_in_opposite_directions() {
        let Some(Message::BrushRadiusScaled(down)) = press("[") else {
            panic!("[ did not change the radius");
        };
        let Some(Message::BrushRadiusScaled(up)) = press("]") else {
            panic!("] did not change the radius");
        };
        assert!(down < 1.0, "[ should shrink the brush");
        assert!(up > 1.0, "] should grow it");
        // One in each direction returns to where it started.
        assert!((down * up - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn keys_that_mean_nothing_are_left_for_the_toolkit() {
        assert!(press("q").is_none());
        assert!(press("").is_none());
        // Every other control chord belongs to the toolkit or the window
        // manager, so claiming it here would steal it.
        assert!(shortcut("s", true, false).is_none());
        assert!(shortcut("x", true, false).is_none(), "ctrl x is not a mirror toggle");
        assert!(shortcut("1", true, false).is_none());
    }
}
