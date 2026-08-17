// SPDX-License-Identifier: AGPL-3.0-or-later

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

use brokkr_core::{BrickCoord, BrickMesh};
use brokkr_gpu::{Frustum, PixelRect, PoolStats, SculptRenderer, Uniforms};
use iced::mouse;
use iced::widget::shader;
use iced::{Rectangle, Vector};

use crate::camera::OrbitCamera;
use crate::message::{Message, PointerButton, PointerEvent};

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

    /// The pool counters as of the last frame, for the debug overlay.
    pub fn stats(&self) -> PoolStats {
        *self.stats.lock().expect("shared frame poisoned")
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
        use iced::keyboard;

        let pointer = match event {
            iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let (position, size) = pointer_position(bounds, cursor)?;
                PointerEvent::Moved { position, size }
            }
            iced::Event::Mouse(mouse::Event::ButtonPressed(button)) => {
                // Only start a drag that began inside the viewport, so a click
                // on a panel does not sculpt.
                cursor.position_in(bounds)?;
                let (position, size) = pointer_position(bounds, cursor)?;
                PointerEvent::Pressed { button: button_of(*button)?, position, size }
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(button)) => {
                // Handled wherever the cursor is: releasing outside the widget
                // still has to end the drag.
                PointerEvent::Released { button: button_of(*button)? }
            }
            iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                cursor.position_in(bounds)?;
                let amount = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y,
                    // Pixel deltas are far larger per notch than line deltas.
                    mouse::ScrollDelta::Pixels { y, .. } => *y / 40.0,
                };
                PointerEvent::Scrolled { amount }
            }
            iced::Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) => {
                PointerEvent::Modifiers { shift: modifiers.shift(), control: modifiers.control() }
            }
            // Undo and redo are handled here rather than through a global
            // shortcut because the shader widget already receives every event,
            // wherever the cursor happens to be.
            iced::Event::Keyboard(keyboard::Event::KeyPressed { key, modifiers, .. }) => {
                let keyboard::Key::Character(character) = key else {
                    return None;
                };
                if !modifiers.command() || !character.eq_ignore_ascii_case("z") {
                    return None;
                }
                let message = if modifiers.shift() { Message::Redo } else { Message::Undo };
                return Some(shader::Action::publish(message).and_capture());
            }
            _ => return None,
        };

        Some(shader::Action::publish(Message::Pointer(pointer)).and_capture())
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
    }
}

/// The GPU state shared by every frame of the viewport. Iced builds this once,
/// the first time it sees a [`SculptPrimitive`].
#[derive(Debug)]
pub struct SculptPipeline {
    renderer: SculptRenderer,
    /// Rebuilt in `prepare` from the same matrix the shader gets, and used in
    /// `render`, which is a separate call and has no access to the camera.
    frustum: Frustum,
}

impl shader::Pipeline for SculptPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self {
            renderer: SculptRenderer::new(device, queue, format),
            frustum: Frustum::from_view_projection(glam::Mat4::IDENTITY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::widget::shader::Program;
    use iced::{Point, Size};

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
