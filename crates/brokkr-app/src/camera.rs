// SPDX-License-Identifier: AGPL-3.0-or-later

//! Orbit camera.
//!
//! Plain maths with no UI or GPU dependency, so it can be tested without a
//! window.

use glam::{Mat4, Quat, Vec2, Vec3};

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
    /// Twist about the camera's own view axis, in radians.
    ///
    /// Nothing but the SpaceMouse's twist axis moves this, and it defaults to
    /// zero, so every other feature behaves as though it did not exist.
    pub roll: f32,
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
            roll: 0.0,
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

    /// The camera's right axis in world space.
    ///
    /// The rows of a view matrix are the camera basis vectors, which is why
    /// this reads down a column of each axis rather than along one.
    pub fn right(&self) -> Vec3 {
        let view = self.view();
        Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x)
    }

    /// The camera's up axis in world space.
    pub fn up(&self) -> Vec3 {
        let view = self.view();
        Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y)
    }

    /// The up direction handed to `look_at`, twisted about the view axis by
    /// [`Self::roll`].
    ///
    /// Returns world up unchanged when roll is zero, so a camera nobody has
    /// twisted produces a bit identical view matrix to the one this had before
    /// roll existed.
    ///
    /// An up vector parallel to the view axis collapses `look_at` into a non
    /// finite matrix. `PITCH_LIMIT` is what keeps world up clear of the view
    /// axis, and rotating a vector *about* that axis cannot change its angle
    /// to it, so a rolled camera is exactly as safe as an upright one and
    /// needs no clamp of its own.
    fn up_vector(&self) -> Vec3 {
        if self.roll == 0.0 {
            return Vec3::Y;
        }
        let forward = (self.target - self.eye()).normalize();
        Quat::from_axis_angle(forward, self.roll) * Vec3::Y
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, self.up_vector())
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
        self.orbit_radians(delta * SENSITIVITY);
    }

    /// Rotate by an angle rather than by a drag.
    ///
    /// The SpaceMouse works in radians per millisecond, so it would otherwise
    /// have to convert into pixels only for [`Self::orbit`] to convert them
    /// straight back.
    pub fn orbit_radians(&mut self, delta: Vec2) {
        self.yaw -= delta.x;
        self.pitch = (self.pitch + delta.y).clamp(-PITCH_LIMIT, PITCH_LIMIT);
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
        self.target += (-self.right() * delta.x + self.up() * delta.y) * world_per_pixel;
    }

    /// Wheel to zoom. Positive `amount` moves closer.
    ///
    /// Multiplicative so each notch covers the same visual fraction whether the
    /// camera is near or far.
    pub fn zoom(&mut self, amount: f32) {
        const RATE: f32 = 0.12;
        self.zoom_by((-amount * RATE).exp());
    }

    /// Scale the orbit distance directly. Below 1 moves closer.
    ///
    /// The SpaceMouse works in log zoom per millisecond, so it has the factor
    /// already and would otherwise have to take its logarithm and divide by
    /// `RATE` purely for [`Self::zoom`] to undo both.
    pub fn zoom_by(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(self.near * 10.0, self.far);
    }

    /// The shortest way round from one angle to another, in radians.
    ///
    /// Yaw is unbounded, so two headings that look identical can be many turns
    /// apart as numbers. Interpolating those directly makes the camera spin the
    /// long way — several times round, in the worst case — for a click that
    /// should have moved it a few degrees.
    pub fn shortest_angle_delta(from: f32, to: f32) -> f32 {
        use std::f32::consts::{PI, TAU};
        let delta = (to - from) % TAU;
        if delta > PI {
            delta - TAU
        } else if delta < -PI {
            delta + TAU
        } else {
            delta
        }
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

        let camera_right = camera.right();
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

#[cfg(test)]
mod basis_tests {
    use super::*;

    #[test]
    fn the_basis_axes_are_unit_length_and_mutually_perpendicular() {
        let camera = OrbitCamera { yaw: 0.9, pitch: -0.4, ..OrbitCamera::framing(Vec3::ZERO, 5.0) };
        let right = camera.right();
        let up = camera.up();

        assert!((right.length() - 1.0).abs() < 1.0e-5);
        assert!((up.length() - 1.0).abs() < 1.0e-5);
        assert!(right.dot(up).abs() < 1.0e-5, "right and up were not perpendicular");
    }

    #[test]
    fn the_basis_axes_are_perpendicular_to_the_view_direction() {
        let camera = OrbitCamera::framing(Vec3::new(2.0, -1.0, 4.0), 3.0);
        let forward = (camera.target - camera.eye()).normalize();
        assert!(camera.right().dot(forward).abs() < 1.0e-5);
        assert!(camera.up().dot(forward).abs() < 1.0e-5);
    }
}

#[cfg(test)]
mod roll_tests {
    use super::*;

    /// Roll is only worth having if it costs nothing when unused: every other
    /// feature reads `view()`, so an upright camera has to produce exactly the
    /// matrix it produced before roll existed, not one that merely rounds to
    /// it.
    #[test]
    fn an_unrolled_camera_produces_a_bit_identical_view_matrix() {
        let camera = OrbitCamera { yaw: 0.9, pitch: -0.4, ..OrbitCamera::framing(Vec3::ZERO, 5.0) };
        assert_eq!(camera.roll, 0.0, "roll must default to zero");

        let expected = glam::camera::rh::view::look_at_mat4(camera.eye(), camera.target, Vec3::Y);
        assert_eq!(camera.view().to_cols_array(), expected.to_cols_array());
    }

    #[test]
    fn rolling_turns_the_camera_about_its_own_view_axis() {
        let upright = OrbitCamera::framing(Vec3::ZERO, 5.0);
        let rolled = OrbitCamera { roll: 0.5, ..upright };

        // The eye has not moved, so the view direction is untouched...
        assert_eq!(rolled.eye(), upright.eye());
        let forward = (upright.target - upright.eye()).normalize();
        assert!(rolled.up().dot(forward).abs() < 1.0e-5);
        // ...but the up axis has turned within the plane across the view.
        assert!(rolled.up().dot(upright.up()) < 0.9, "the camera did not actually roll");
    }

    #[test]
    fn a_quarter_turn_of_roll_swaps_the_up_and_right_axes() {
        let upright = OrbitCamera::framing(Vec3::ZERO, 5.0);
        let rolled = OrbitCamera { roll: std::f32::consts::FRAC_PI_2, ..upright };

        // Which of the two it lands on is a sign convention; that they trade
        // places is the thing worth pinning.
        assert!(
            rolled.up().dot(upright.right()).abs() > 0.999,
            "up should have turned onto the old right axis, got {:?}",
            rolled.up()
        );
    }

    /// The one way roll could break the camera is by putting the up vector
    /// along the view axis, which collapses `look_at`. Rotating about that
    /// axis cannot change the angle to it, so this must hold even at the
    /// pitch limit, where world up is as close to the view axis as it ever
    /// gets.
    #[test]
    fn roll_stays_finite_and_orthonormal_even_at_the_pitch_limit() {
        for steps in [0.0, 0.5, 1.0, 2.0, -2.0, std::f32::consts::TAU] {
            let camera = OrbitCamera {
                pitch: PITCH_LIMIT,
                roll: steps,
                ..OrbitCamera::framing(Vec3::ZERO, 5.0)
            };

            assert!(
                camera.view().to_cols_array().iter().all(|v| v.is_finite()),
                "the view matrix collapsed at roll {steps}"
            );
            let (right, up) = (camera.right(), camera.up());
            assert!((right.length() - 1.0).abs() < 1.0e-4, "right was not unit at roll {steps}");
            assert!((up.length() - 1.0).abs() < 1.0e-4, "up was not unit at roll {steps}");
            assert!(right.dot(up).abs() < 1.0e-4, "axes not perpendicular at roll {steps}");
        }
    }

    /// Without this a click on the navigation cube can spin the model several
    /// times round to reach a heading a few degrees away.
    #[test]
    fn the_shortest_way_round_never_takes_the_long_way() {
        use std::f32::consts::{PI, TAU};
        for (from, to, expected) in [
            (0.0, 0.1, 0.1),
            (0.0, -0.1, -0.1),
            // Just past half a turn: the short way is backwards.
            (0.0, PI + 0.2, -(PI - 0.2)),
            (0.0, -(PI + 0.2), PI - 0.2),
            // Many turns apart as numbers, identical as headings.
            (0.0, TAU * 3.0, 0.0),
            (TAU * 5.0 + 0.3, 0.3, 0.0),
        ] {
            let delta = OrbitCamera::shortest_angle_delta(from, to);
            assert!(
                (delta - expected).abs() < 1.0e-4,
                "{from} to {to} gave {delta}, expected {expected}"
            );
            assert!(delta.abs() <= PI + 1.0e-4, "{from} to {to} took the long way: {delta}");
        }
    }

    #[test]
    fn zooming_by_a_factor_agrees_with_zooming_by_an_amount() {
        let mut by_amount = OrbitCamera::framing(Vec3::ZERO, 1.0);
        let mut by_factor = by_amount;

        by_amount.zoom(1.0);
        by_factor.zoom_by((-0.12f32).exp());
        assert!((by_amount.distance - by_factor.distance).abs() < 1.0e-6);
    }

    #[test]
    fn zooming_by_a_factor_honours_the_same_clamp() {
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 1.0);
        camera.zoom_by(1.0e-9);
        assert!(camera.distance >= camera.near * 10.0, "zoom_by went inside the near clamp");
        camera.zoom_by(1.0e9);
        assert!(camera.distance <= camera.far, "zoom_by went past the far clamp");
    }

    #[test]
    fn orbiting_by_radians_agrees_with_orbiting_by_pixels() {
        let mut by_pixels = OrbitCamera::framing(Vec3::ZERO, 1.0);
        let mut by_radians = by_pixels;

        by_pixels.orbit(Vec2::new(10.0, -4.0));
        by_radians.orbit_radians(Vec2::new(10.0, -4.0) * 0.008);
        assert!((by_pixels.yaw - by_radians.yaw).abs() < 1.0e-6);
        assert!((by_pixels.pitch - by_radians.pitch).abs() < 1.0e-6);
    }

    #[test]
    fn orbiting_by_radians_is_clamped_short_of_the_pole_too() {
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 1.0);
        camera.orbit_radians(Vec2::new(0.0, 100.0));
        assert!(camera.pitch <= PITCH_LIMIT);
        assert!(camera.view().to_cols_array().iter().all(|v| v.is_finite()));
    }
}
