// SPDX-License-Identifier: AGPL-3.0-or-later

//! Orbit camera.
//!
//! Plain maths with no UI or GPU dependency, so it can be tested without a
//! window.

use glam::{Mat4, Vec2, Vec3};

/// How close the pitch may come to straight up or down.
///
/// Reaching the pole would make the up vector parallel to the view direction
/// and the view matrix would collapse, so stop just short.
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// A camera that orbits a target point.
#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation about the world Y axis, in radians.
    pub yaw: f32,
    /// Elevation above the horizon, in radians, clamped short of the poles.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 4.0,
            yaw: 0.6,
            pitch: 0.35,
            fov_y: 45f32.to_radians(),
            near: 0.01,
            far: 1000.0,
        }
    }
}

impl OrbitCamera {
    /// Frame a sphere so it fills a comfortable part of the view.
    pub fn framing(target: Vec3, radius: f32) -> Self {
        let mut camera = Self { target, ..Default::default() };
        // Far enough that the sphere subtends most of the vertical field, with
        // a little air around it.
        camera.distance = radius / (camera.fov_y * 0.5).sin() * 1.15;
        camera.near = (radius * 0.001).max(1.0e-4);
        camera.far = camera.distance + radius * 20.0;
        camera
    }

    pub fn eye(&self) -> Vec3 {
        self.target + self.forward_from_target() * self.distance
    }

    /// Unit vector from the target toward the eye.
    fn forward_from_target(&self) -> Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Projection for wgpu's clip space.
    ///
    /// wgpu normalises to depth 0 to 1 with Y up, which is glam's `directx`
    /// convention. The `vulkan` one looks like the obvious choice given the
    /// backend, but it is Y down and would render the scene upside down.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(
            self.fov_y,
            aspect.max(1.0e-3),
            self.near,
            self.far,
        )
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// Drag to rotate. Deltas are in pixels.
    pub fn orbit(&mut self, delta: Vec2) {
        const SENSITIVITY: f32 = 0.008;
        self.yaw -= delta.x * SENSITIVITY;
        self.pitch = (self.pitch + delta.y * SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Drag to slide the target across the view plane. Deltas are in pixels.
    ///
    /// Scaled by distance and field of view so the model tracks the cursor at
    /// any zoom level rather than crawling when close and flying when far.
    pub fn pan(&mut self, delta: Vec2, viewport_height: f32) {
        if viewport_height <= 0.0 {
            return;
        }
        let world_per_pixel = 2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height;
        let view = self.view();
        // Rows of a view matrix are the camera basis vectors in world space.
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);
        self.target += (-right * delta.x + up * delta.y) * world_per_pixel;
    }

    /// Wheel to zoom. Positive `amount` moves closer.
    ///
    /// Multiplicative so each notch covers the same visual fraction whether the
    /// camera is near or far.
    pub fn zoom(&mut self, amount: f32) {
        const RATE: f32 = 0.12;
        self.distance = (self.distance * (-amount * RATE).exp()).clamp(self.near * 10.0, self.far);
    }

    /// The world space ray through a point given in normalised device
    /// coordinates, where x and y both run from -1 to 1 and y points up.
    pub fn ray(&self, ndc: Vec2, aspect: f32) -> (Vec3, Vec3) {
        let inverse = self.view_projection(aspect).inverse();
        // wgpu's near plane is at depth 0 and the far plane at depth 1.
        let near = inverse * glam::Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
        let far = inverse * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        (near, (far - near).normalize())
    }

    /// Convert a position in widget pixels to normalised device coordinates.
    ///
    /// Screen y grows downward and clip space y grows upward, hence the flip.
    pub fn ndc_from_pixels(position: Vec2, size: Vec2) -> Vec2 {
        Vec2::new(
            position.x / size.x.max(1.0) * 2.0 - 1.0,
            1.0 - position.y / size.y.max(1.0) * 2.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_eye_sits_at_the_orbit_distance_from_the_target() {
        let camera =
            OrbitCamera { target: Vec3::new(1.0, 2.0, 3.0), distance: 7.0, ..Default::default() };
        assert!((camera.eye().distance(camera.target) - 7.0).abs() < 1.0e-4);
    }

    #[test]
    fn pitch_never_reaches_the_pole() {
        let mut camera = OrbitCamera::default();
        for _ in 0..1000 {
            camera.orbit(Vec2::new(0.0, 100.0));
        }
        assert!(camera.pitch < std::f32::consts::FRAC_PI_2);
        // A collapsed view matrix shows up as a non finite entry.
        assert!(camera.view().to_cols_array().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_centre_ray_points_from_the_eye_at_the_target() {
        let camera = OrbitCamera::framing(Vec3::splat(5.0), 2.0);
        let (origin, direction) = camera.ray(Vec2::ZERO, 16.0 / 9.0);

        let expected = (camera.target - camera.eye()).normalize();
        assert!(direction.dot(expected) > 0.9999, "direction was {direction:?}");
        // The ray starts on the near plane, just in front of the eye.
        assert!((origin.distance(camera.eye()) - camera.near).abs() < 1.0e-3);
    }

    #[test]
    fn a_ray_to_the_right_of_centre_leans_right() {
        let camera = OrbitCamera::framing(Vec3::ZERO, 1.0);
        let (_, centre) = camera.ray(Vec2::ZERO, 1.0);
        let (_, right) = camera.ray(Vec2::new(0.5, 0.0), 1.0);

        let view = camera.view();
        let camera_right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        assert!(
            right.dot(camera_right) > centre.dot(camera_right),
            "a ray at positive ndc x must lean along the camera's right axis"
        );
    }

    #[test]
    fn pixels_map_to_normalised_device_coordinates_with_y_flipped() {
        let size = Vec2::new(800.0, 600.0);
        assert_eq!(OrbitCamera::ndc_from_pixels(Vec2::new(400.0, 300.0), size), Vec2::ZERO);
        // Top left of the widget is ndc (-1, 1).
        assert_eq!(OrbitCamera::ndc_from_pixels(Vec2::ZERO, size), Vec2::new(-1.0, 1.0));
        assert_eq!(OrbitCamera::ndc_from_pixels(size, size), Vec2::new(1.0, -1.0));
    }

    #[test]
    fn zooming_in_and_back_out_returns_to_the_same_distance() {
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 1.0);
        let before = camera.distance;
        camera.zoom(3.0);
        assert!(camera.distance < before, "positive zoom should move closer");
        camera.zoom(-3.0);
        assert!((camera.distance - before).abs() < 1.0e-4);
    }

    #[test]
    fn panning_moves_the_target_across_the_view_not_along_it() {
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 1.0);
        let forward = (camera.target - camera.eye()).normalize();
        camera.pan(Vec2::new(40.0, 0.0), 600.0);
        assert!(camera.target.length() > 0.0, "the target should have moved");
        assert!(
            camera.target.dot(forward).abs() < 1.0e-4,
            "panning must not move the target along the view direction"
        );
    }
}
