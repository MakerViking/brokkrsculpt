// SPDX-License-Identifier: AGPL-3.0-only

//! Orbit camera.
//!
//! Plain maths with no UI or GPU dependency, so it can be tested without a
//! window.

use glam::{Mat4, Quat, Vec2, Vec3};

/// How close the camera may come to what it is looking at, measured in voxels.
///
/// **The unit is the whole point.** The floor this replaces was `near * 10.0`
/// with `near` derived from the model's radius, so it said "a hundredth of the
/// model" -- which is 1.33 mm on a 133 mm figure whatever the lattice under it
/// was, and there is no detail work at 1.33 mm on a 0.0565 mm voxel. Two
/// voxels is the same closeness on a 5 mm model and a 500 mm one, because two
/// voxels is the same amount of *sculpting* on both.
const FLOOR_VOXELS: f32 = 2.0;

/// How far out the camera may go, as a multiple of the content radius.
///
/// [`OrbitCamera::framing`] sits at about three, so this is roughly thirty
/// times framed: far enough that no gesture feels fenced in, close enough that
/// the far plane stays a number a depth buffer can work with and the model is
/// still a visible speck rather than nothing at all.
const CEILING_RADII: f32 = 100.0;

/// A camera that orbits a target point.
#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation about the world Y axis, in radians.
    pub yaw: f32,
    /// Elevation above the horizon, in radians.
    ///
    /// **Unclamped, and free to pass the poles**, which is what the composed
    /// [`OrbitCamera::orientation`] buys. Wrapped into `-pi..pi` by
    /// [`OrbitCamera::orbit_radians`] so a long drag does not accumulate an
    /// angle in the thousands.
    pub pitch: f32,
    /// Twist about the camera's own view axis, in radians.
    ///
    /// Nothing but the SpaceMouse's twist axis moves this, and it defaults to
    /// zero, so every other feature behaves as though it did not exist.
    pub roll: f32,
    pub fov_y: f32,
    /// The lattice the model is stored on. Sets how close the camera may get.
    ///
    /// Kept on the camera rather than asked of the document per gesture
    /// because [`OrbitCamera`] is deliberately free of any dependency on one;
    /// [`OrbitCamera::set_lattice`] is how the application keeps it true.
    pub voxel_size: f32,
    /// Half the diagonal of the content's brick box. Sets the far plane, the
    /// pick reach and how far out the camera may go.
    pub content_radius: f32,
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
            voxel_size: 0.25,
            content_radius: 1.0,
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
        camera.content_radius = radius.max(1.0e-4);
        camera
    }

    /// Tell the camera what it is looking at, in the two figures it measures
    /// itself against.
    ///
    /// Called wherever the document's lattice or extent can change -- an open,
    /// an import, a resample, a body becoming active. Getting it wrong costs a
    /// zoom floor in the wrong unit, which is the failure this replaces.
    pub fn set_lattice(&mut self, voxel_size: f32, content_radius: f32) {
        if voxel_size.is_finite() && voxel_size > 0.0 {
            self.voxel_size = voxel_size;
        }
        if content_radius.is_finite() && content_radius > 0.0 {
            self.content_radius = content_radius;
        }
    }

    /// The closest the camera may come to its target.
    pub fn min_distance(&self) -> f32 {
        (self.voxel_size * FLOOR_VOXELS).max(1.0e-4)
    }

    /// The furthest out it may go.
    pub fn max_distance(&self) -> f32 {
        (self.content_radius * CEILING_RADII).max(self.min_distance() * 10.0)
    }

    /// The near plane, derived rather than stored.
    ///
    /// **A hundredth of the distance, so the target is a hundred near planes
    /// away at every zoom level and cannot be clipped however close the camera
    /// gets.** A stored near plane cannot promise that: it is a fixed number,
    /// the distance is not, and the pair went wrong in both directions at once
    /// -- too coarse to let the camera approach a detail, and, on a file
    /// written by an older build, a floor computed from a body that was not
    /// the one being restored.
    ///
    /// # The floor under it is a depth buffer's, not a geometry one
    ///
    /// A near plane that follows the distance all the way down drives the
    /// far-to-near ratio through the roof exactly where the user is looking
    /// hardest: two voxels from a 133 mm figure at a 0.0565 mm lattice is a
    /// distance of 0.11 mm, and `distance * 0.01` would put the near plane at
    /// a micrometre against a far plane of 920 -- a ratio near a million, on a
    /// `Depth32Float` buffer, which quantises the far side of the model into
    /// millimetres and z-fights it. The floor is a ten-thousandth of the
    /// content radius, which is TEN TIMES SMALLER than the fixed
    /// `radius * 0.001` this replaced, so nothing that used to be visible can
    /// start being clipped -- and it caps the ratio at about forty thousand,
    /// against the twenty thousand the old pair produced at a distance five
    /// times further out. The normal case is far better than before rather
    /// than merely no worse: at a framed distance the ratio is about 230,
    /// because `far` is now `distance + 4r` where it used to be
    /// `distance + 20r`.
    pub fn near(&self) -> f32 {
        (self.distance * 0.01).max(self.content_radius * 1.0e-4).max(1.0e-5)
    }

    /// The far plane, and also the length of every pick ray.
    ///
    /// Eye to target plus a generous allowance for the content around it. The
    /// two uses must not be split apart: a pick ray shorter than the far plane
    /// would fail to find surface the user can plainly see.
    pub fn far(&self) -> f32 {
        self.distance + self.content_radius * 4.0
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

    /// Camera local axes to world.
    ///
    /// The three angle fields are this rotation's YXZ Euler form and stay the
    /// stored representation, so nothing about the file format changes.
    ///
    /// **This is what retires the pitch clamp.** `look_at` builds its basis by
    /// crossing the view direction with a hint, which degenerates when the two
    /// are parallel -- so a camera looking straight down produced a non finite
    /// matrix, and the only defence was to stop short of the pole and tell the
    /// user the camera was "limited by angles". A product of three rotations
    /// is orthonormal at every angle including exactly plus or minus a quarter
    /// turn. Euler storage was never the problem; gimbal lock costs uniqueness
    /// of the *numbers*, not fidelity of the orientation they name.
    ///
    /// The two negative signs are not decoration:
    ///
    /// - `Rx(-pitch)`, because pitch is measured as elevation ABOVE the
    ///   horizon while a rotation about X by a positive angle takes the camera
    ///   the other way. Pinned by `the_composed_forward_matches_the_spherical_one`.
    /// - `Rz(-roll)`, because `Rz` turns about the camera's local +Z, which
    ///   points BACKWARD along the view, where the `up_vector` this replaces
    ///   turned about the view direction itself. Pinned by
    ///   `a_rolled_camera_matches_the_up_vector_it_used_to_build`.
    pub fn orientation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw)
            * Quat::from_rotation_x(-self.pitch)
            * Quat::from_rotation_z(-self.roll)
    }

    /// The camera's right axis in world space.
    pub fn right(&self) -> Vec3 {
        self.orientation() * Vec3::X
    }

    /// The camera's up axis in world space.
    pub fn up(&self) -> Vec3 {
        self.orientation() * Vec3::Y
    }

    pub fn view(&self) -> Mat4 {
        glam::Affine3A::from_rotation_translation(self.orientation(), self.eye()).inverse().into()
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
            self.near(),
            self.far(),
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
    ///
    /// Pitch is wrapped rather than clamped. With the pole open it accumulates
    /// without bound otherwise, and an angle in the thousands of radians both
    /// loses precision and reads as nonsense in a saved file.
    pub fn orbit_radians(&mut self, delta: Vec2) {
        self.yaw = wrap_angle(self.yaw - delta.x);
        self.pitch = wrap_angle(self.pitch + delta.y);
    }

    /// Rotate the whole camera rigidly about a world point.
    ///
    /// The pivot keeps its exact position on screen, which is what makes
    /// orbiting about the surface under the cursor feel like turning the model
    /// in your hand rather than like the model running away from you.
    ///
    /// **The rotation is read out of the basis rather than solved for.** The
    /// obvious route -- work out where the eye should end up and recover yaw
    /// and pitch from it with `atan2` and `asin` -- disagrees with what
    /// [`Self::orbit`] actually did by a little at every step and by a lot near
    /// the poles, and the pivot slides out from under the cursor over a long
    /// drag. Taking `new * old.inverse()` asks the basis what it did and
    /// cannot disagree with it.
    pub fn orbit_about(&mut self, delta: Vec2, pivot: Vec3) {
        let eye_before = self.eye();
        let before = self.orientation();
        self.orbit(delta);
        let turned = self.orientation() * before.inverse();
        let eye = pivot + turned * (eye_before - pivot);
        self.target = eye - self.forward_from_target() * self.distance;
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

    /// The multiplier one wheel amount corresponds to.
    ///
    /// Split out of the `zoom` it replaces because a wheel notch now needs an
    /// anchor as well as an amount, and there must be exactly one place that
    /// knows how many millimetres a notch is worth. Multiplicative, so each
    /// notch covers the same visual fraction whether the camera is near or far.
    pub fn zoom_factor(amount: f32) -> f32 {
        const RATE: f32 = 0.12;
        (-amount * RATE).exp()
    }

    /// Scale the orbit distance directly. Below 1 moves closer.
    ///
    /// The SpaceMouse works in log zoom per millisecond, so it has the factor
    /// already and would otherwise have to take its logarithm and divide by
    /// `RATE` purely for [`Self::zoom`] to undo both.
    pub fn zoom_by(&mut self, factor: f32) {
        self.zoom_by_about(factor, self.target);
    }

    /// Zoom toward a world point, which keeps its position on screen.
    ///
    /// **This, and not the distance floor, is what "I cannot zoom in to work
    /// on a detail" was really about.** The eye travels toward the TARGET, and
    /// after a file is opened the target is the centre of the model -- so
    /// zooming in on an eyelid walks the camera into the skull, and by the time
    /// the eyelid fills the view the eye is inside the head. Lowering the floor
    /// only lets you get further in. Moving what is zoomed toward onto the
    /// surface under the cursor is the fix; the floor is then free to be as low
    /// as the lattice deserves, because the thing it is a floor above is the
    /// surface rather than the centre.
    ///
    /// **The clamp is applied to the distance first and the factor derived back
    /// out of it.** Scaling the target by the requested factor and clamping the
    /// distance separately un-pins the anchor exactly at the floor, which is
    /// where the user is pushing hardest and watching most closely.
    pub fn zoom_by_about(&mut self, factor: f32, anchor: Vec3) {
        if self.distance <= 0.0
            || !self.distance.is_finite()
            || !factor.is_finite()
            || !anchor.is_finite()
        {
            return;
        }
        let clamped = (self.distance * factor).clamp(self.min_distance(), self.max_distance());
        let effective = clamped / self.distance;
        self.target = anchor + (self.target - anchor) * effective;
        self.distance = clamped;
    }

    /// The shortest way round from one angle to another, in radians.
    ///
    /// Yaw is unbounded, so two headings that look identical can be many turns
    /// apart as numbers. Interpolating those directly makes the camera spin the
    /// long way — several times round, in the worst case — for a click that
    /// should have moved it a few degrees.
    pub fn shortest_angle_delta(from: f32, to: f32) -> f32 {
        wrap_angle(to - from)
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

/// Bring an angle back into -pi..pi.
///
/// Moved here from the SpaceMouse, which wanted it for roll, because with the
/// pitch clamp gone pitch accumulates for as long as the user keeps dragging
/// and needs exactly the same treatment. Wrapping does not change the
/// orientation an angle names: a full turn of pitch is the same camera.
pub fn wrap_angle(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let wrapped = angle % TAU;
    if wrapped > PI {
        wrapped - TAU
    } else if wrapped < -PI {
        wrapped + TAU
    } else {
        wrapped
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

    /// **This replaces `pitch_never_reaches_the_pole`, and the behaviour it
    /// pinned is the thing being fixed.** Stopping 0.01 rad short of straight
    /// down was a workaround for `look_at` collapsing there, and the user felt
    /// it as a camera "limited by angles". The property that mattered -- a
    /// finite, orthonormal view matrix -- is now asserted AT the pole and past
    /// it rather than being bought by never going there.
    #[test]
    fn pitch_may_pass_the_pole_and_the_view_matrix_survives_it() {
        use std::f32::consts::{FRAC_PI_2, PI};

        let mut camera = OrbitCamera::default();
        let mut reached_the_far_side = false;
        for _ in 0..1000 {
            camera.orbit(Vec2::new(0.0, 100.0));
            assert!(
                camera.view().to_cols_array().iter().all(|v| v.is_finite()),
                "the view matrix collapsed at pitch {}",
                camera.pitch
            );
            // Upside down: only reachable by having gone over the top.
            reached_the_far_side |= camera.up().y < -0.5;
        }
        assert!(reached_the_far_side, "the camera never got past the pole");
        assert!(camera.pitch.abs() <= PI + 1.0e-4, "pitch was not wrapped: {}", camera.pitch);

        // And exactly at the pole, which the clamp existed to avoid.
        for pitch in [FRAC_PI_2, -FRAC_PI_2] {
            let camera = OrbitCamera { pitch, ..OrbitCamera::framing(Vec3::ZERO, 5.0) };
            let (right, up) = (camera.right(), camera.up());
            assert!(camera.view().to_cols_array().iter().all(|v| v.is_finite()));
            assert!((right.length() - 1.0).abs() < 1.0e-5, "right was not unit at pitch {pitch}");
            assert!((up.length() - 1.0).abs() < 1.0e-5, "up was not unit at pitch {pitch}");
            assert!(right.dot(up).abs() < 1.0e-5, "axes not perpendicular at pitch {pitch}");
        }
    }

    /// The sign of the `-pitch` in [`OrbitCamera::orientation`], which is the
    /// one thing in the composed basis that cannot be derived by reading it.
    #[test]
    fn the_composed_forward_matches_the_spherical_one() {
        for yaw in [-3.0_f32, -1.1, 0.0, 0.7, 2.5, 3.1] {
            for pitch in [-3.0_f32, -1.6, -0.4, 0.0, 0.4, std::f32::consts::FRAC_PI_2, 2.9] {
                let camera = OrbitCamera { yaw, pitch, ..OrbitCamera::framing(Vec3::ZERO, 5.0) };
                let composed = camera.orientation() * Vec3::Z;
                let spherical = camera.forward_from_target();
                assert!(
                    composed.distance(spherical) < 1.0e-5,
                    "at yaw {yaw} pitch {pitch}: composed {composed:?} vs spherical {spherical:?}"
                );
            }
        }
    }

    /// The composed up has to be the one `look_at` would have built, or every
    /// gesture that reads `up()` -- pan above all -- changes direction.
    #[test]
    fn the_composed_up_matches_look_at_at_zero_roll() {
        for yaw in [-2.0_f32, 0.0, 0.9, 2.7] {
            for pitch in [-1.4_f32, -0.3, 0.0, 0.6, 1.4] {
                let camera = OrbitCamera { yaw, pitch, ..OrbitCamera::framing(Vec3::ZERO, 5.0) };
                let expected =
                    glam::camera::rh::view::look_at_mat4(camera.eye(), camera.target, Vec3::Y);
                let expected_up =
                    Vec3::new(expected.x_axis.y, expected.y_axis.y, expected.z_axis.y);
                assert!(
                    camera.up().distance(expected_up) < 1.0e-5,
                    "at yaw {yaw} pitch {pitch}: {:?} vs {expected_up:?}",
                    camera.up()
                );
            }
        }
    }

    #[test]
    fn the_centre_ray_points_from_the_eye_at_the_target() {
        let camera = OrbitCamera::framing(Vec3::splat(5.0), 2.0);
        let (origin, direction) = camera.ray(Vec2::ZERO, 16.0 / 9.0);

        let expected = (camera.target - camera.eye()).normalize();
        assert!(direction.dot(expected) > 0.9999, "direction was {direction:?}");
        // The ray starts on the near plane, just in front of the eye.
        assert!((origin.distance(camera.eye()) - camera.near()).abs() < 1.0e-3);
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
        camera.zoom_by(OrbitCamera::zoom_factor(3.0));
        assert!(camera.distance < before, "positive zoom should move closer");
        camera.zoom_by(OrbitCamera::zoom_factor(-3.0));
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

    /// Roll used to be free when unused because `view()` handed `look_at` a
    /// literal `Vec3::Y` and got back the same matrix it always had.
    ///
    /// **That bit identity is deliberately given up, and this test says so
    /// rather than quietly loosening.** `view()` is no longer built by
    /// `look_at` at all -- it is the inverse of a rotation composed from three
    /// quaternions, which is what lets the camera pass the pole. The two agree
    /// to about a part in a million, which is what a renderer and a picker
    /// need; they cannot agree to the last bit, because they are not the same
    /// arithmetic.
    #[test]
    fn an_unrolled_camera_matches_the_matrix_look_at_would_have_built() {
        let camera = OrbitCamera { yaw: 0.9, pitch: -0.4, ..OrbitCamera::framing(Vec3::ZERO, 5.0) };
        assert_eq!(camera.roll, 0.0, "roll must default to zero");

        let expected = glam::camera::rh::view::look_at_mat4(camera.eye(), camera.target, Vec3::Y);
        for (got, want) in camera.view().to_cols_array().iter().zip(expected.to_cols_array()) {
            assert!((got - want).abs() < 1.0e-5, "{got} vs {want}");
        }
    }

    /// The sign of the `-roll` in [`OrbitCamera::orientation`].
    ///
    /// `Rz` turns about the camera's local +Z, which points backward along the
    /// view; the `up_vector` this replaced turned about the view direction
    /// itself. Get the sign wrong and the SpaceMouse's twist axis reverses.
    #[test]
    fn a_rolled_camera_matches_the_up_vector_it_used_to_build() {
        for roll in [-1.3_f32, -0.4, 0.4, 1.3, 2.9] {
            let camera = OrbitCamera {
                yaw: 0.9,
                pitch: -0.4,
                roll,
                ..OrbitCamera::framing(Vec3::ZERO, 5.0)
            };
            // Exactly what `up_vector` computed, before it was deleted.
            let forward = (camera.target - camera.eye()).normalize();
            let hint = Quat::from_axis_angle(forward, roll) * Vec3::Y;
            let expected = glam::camera::rh::view::look_at_mat4(camera.eye(), camera.target, hint);
            let expected_up = Vec3::new(expected.x_axis.y, expected.y_axis.y, expected.z_axis.y);
            assert!(
                camera.up().distance(expected_up) < 1.0e-5,
                "at roll {roll}: {:?} vs {expected_up:?}",
                camera.up()
            );
        }
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

    /// Roll's one hazard used to be putting the up vector along the view axis,
    /// which collapsed `look_at`. The composed basis has no such case, so this
    /// now sweeps the pole rather than stopping short of it.
    #[test]
    fn roll_stays_finite_and_orthonormal_at_and_past_the_pole() {
        use std::f32::consts::FRAC_PI_2;
        for pitch in [FRAC_PI_2 - 0.01, FRAC_PI_2, FRAC_PI_2 + 0.3, -FRAC_PI_2, 3.0] {
            for steps in [0.0, 0.5, 1.0, 2.0, -2.0, std::f32::consts::TAU] {
                let camera =
                    OrbitCamera { pitch, roll: steps, ..OrbitCamera::framing(Vec3::ZERO, 5.0) };

                assert!(
                    camera.view().to_cols_array().iter().all(|v| v.is_finite()),
                    "the view matrix collapsed at pitch {pitch} roll {steps}"
                );
                let (right, up) = (camera.right(), camera.up());
                assert!((right.length() - 1.0).abs() < 1.0e-4, "right was not unit at {steps}");
                assert!((up.length() - 1.0).abs() < 1.0e-4, "up was not unit at {steps}");
                assert!(right.dot(up).abs() < 1.0e-4, "axes not perpendicular at {steps}");
            }
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

        by_amount.zoom_by(OrbitCamera::zoom_factor(1.0));
        by_factor.zoom_by((-0.12f32).exp());
        assert!((by_amount.distance - by_factor.distance).abs() < 1.0e-6);
    }

    #[test]
    fn zooming_by_a_factor_honours_the_same_clamp() {
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 1.0);
        camera.zoom_by(1.0e-9);
        assert!(camera.distance >= camera.min_distance(), "zoom_by went under the floor");
        camera.zoom_by(1.0e9);
        assert!(camera.distance <= camera.max_distance(), "zoom_by went past the ceiling");
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

    /// Pitch is wrapped now rather than clamped, and it has to be: an
    /// unbounded angle loses precision and reads as nonsense in a saved file.
    #[test]
    fn orbiting_by_radians_wraps_rather_than_accumulating() {
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 1.0);
        camera.orbit_radians(Vec2::new(100.0, 100.0));
        assert!(camera.pitch.abs() <= std::f32::consts::PI + 1.0e-4, "pitch {}", camera.pitch);
        assert!(camera.yaw.abs() <= std::f32::consts::PI + 1.0e-4, "yaw {}", camera.yaw);
        assert!(camera.view().to_cols_array().iter().all(|v| v.is_finite()));
    }
}

#[cfg(test)]
mod anchor_tests {
    use super::*;

    const ASPECT: f32 = 16.0 / 9.0;

    /// Where a world point lands on screen, in normalised device coordinates.
    fn on_screen(camera: &OrbitCamera, point: Vec3) -> Vec2 {
        let clip = camera.view_projection(ASPECT) * point.extend(1.0);
        Vec2::new(clip.x / clip.w, clip.y / clip.w)
    }

    fn model() -> OrbitCamera {
        let mut camera = OrbitCamera::framing(Vec3::new(3.0, -1.0, 2.0), 40.0);
        camera.set_lattice(0.25, 40.0);
        camera
    }

    /// A world point off the view axis, standing in for the surface an
    /// off-centre cursor is hovering.
    ///
    /// Built from the camera's own axes rather than from a screen coordinate,
    /// because the wheel's anchor is now always a picked surface point or the
    /// target -- there is no longer any code that turns a pixel into an anchor,
    /// and a test helper that pretended otherwise would be testing a route
    /// nothing takes. What these tests are about is unchanged: an arbitrary
    /// world point holds its position on screen across a zoom.
    fn off_axis(camera: &OrbitCamera, right: f32, up: f32) -> Vec3 {
        camera.target + camera.right() * right + camera.up() * up
    }

    /// Anchored zoom has to be a strict generalisation of the zoom it
    /// replaces, or every existing gesture changes underneath the user.
    /// Anchoring at the target is the case that must not move at all.
    #[test]
    fn zooming_about_the_target_is_the_old_zoom_exactly() {
        let mut anchored = model();
        let mut plain = model();

        for amount in [1.0_f32, 1.0, -0.5, 3.0, -2.0] {
            let factor = OrbitCamera::zoom_factor(amount);
            anchored.zoom_by_about(factor, anchored.target);
            plain.distance =
                (plain.distance * factor).clamp(plain.min_distance(), plain.max_distance());
            assert_eq!(anchored.distance, plain.distance);
            assert_eq!(anchored.target, plain.target);
        }
    }

    /// The property the whole feature is: the thing under the cursor stays
    /// under the cursor.
    #[test]
    fn the_anchor_stays_under_the_cursor_across_a_zoom() {
        for (right, up) in [(11.0, 7.0), (-18.0, -13.0), (0.0, 0.0)] {
            let mut camera = model();
            let anchor = off_axis(&camera, right, up);
            let before = on_screen(&camera, anchor);

            for _ in 0..20 {
                camera.zoom_by_about(OrbitCamera::zoom_factor(1.0), anchor);
            }
            let after = on_screen(&camera, anchor);
            assert!(
                before.distance(after) < 1.0e-3,
                "the anchor slid from {before:?} to {after:?} over twenty notches"
            );
        }
    }

    /// **The case that catches the obvious wrong implementation.** Scaling the
    /// target by the requested factor and clamping the distance separately
    /// looks right and is right everywhere except at the floor -- which is
    /// where the user is pushing hardest and watching most closely.
    #[test]
    fn the_anchor_stays_pinned_even_against_the_distance_floor() {
        let mut camera = model();
        let anchor = off_axis(&camera, 9.0, -5.0);
        let before = on_screen(&camera, anchor);

        // Far past the floor, so most of these notches are fully clamped.
        for _ in 0..200 {
            camera.zoom_by_about(OrbitCamera::zoom_factor(5.0), anchor);
        }
        assert!(
            (camera.distance - camera.min_distance()).abs() < 1.0e-4,
            "the test did not actually reach the floor: {}",
            camera.distance
        );
        let after = on_screen(&camera, anchor);
        assert!(
            before.distance(after) < 1.0e-3,
            "the anchor un-pinned at the floor: {before:?} to {after:?}"
        );
    }

    /// The floor has to mean the same amount of sculpting whatever the model
    /// is, which is what expressing it in voxels buys and what expressing it
    /// as a fraction of the radius cost.
    #[test]
    fn the_zoom_floor_is_the_same_number_of_voxels_on_a_small_model_and_a_large_one() {
        let mut small = OrbitCamera::framing(Vec3::ZERO, 5.0);
        small.set_lattice(0.01, 5.0);
        let mut large = OrbitCamera::framing(Vec3::ZERO, 500.0);
        large.set_lattice(0.01, 500.0);

        assert!((small.min_distance() - large.min_distance()).abs() < 1.0e-9);
        assert!((small.min_distance() - 0.02).abs() < 1.0e-6, "{}", small.min_distance());

        // And it tracks the lattice, which is the thing detail is measured in.
        let mut fine = large;
        fine.set_lattice(0.0565, 500.0);
        assert!(fine.min_distance() > large.min_distance());
    }

    /// A near plane that does not follow the distance clips the very thing the
    /// camera was moved close to look at.
    #[test]
    fn the_target_is_never_clipped_by_the_near_plane_at_any_distance() {
        let mut camera = model();
        for _ in 0..400 {
            assert!(
                camera.near() < camera.distance * 0.5,
                "near {} against distance {}",
                camera.near(),
                camera.distance
            );
            assert!(camera.far() > camera.distance, "the target is behind the far plane");
            camera.zoom_by_about(OrbitCamera::zoom_factor(1.0), camera.target);
        }
    }

    /// The depth buffer is `Depth32Float` and the projection is not reversed,
    /// so the far-to-near ratio is what decides whether the far side of the
    /// model z-fights. This pins the floor under `near` that keeps it bounded,
    /// including at the closest the camera can get on the finest lattice a
    /// large model is ever imported at.
    #[test]
    fn the_depth_ratio_stays_within_a_float_buffers_reach_at_every_zoom() {
        // The dragon: 133 mm of model at 0.0565 mm, which is 230 mm of loose
        // content radius.
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 230.0);
        camera.set_lattice(0.0565, 230.0);

        let framed = camera.far() / camera.near();
        assert!(framed < 1_000.0, "the ratio at a framed distance is {framed}");

        for _ in 0..400 {
            camera.zoom_by_about(OrbitCamera::zoom_factor(3.0), camera.target);
        }
        assert!((camera.distance - camera.min_distance()).abs() < 1.0e-6);
        let closest = camera.far() / camera.near();
        assert!(closest < 50_000.0, "the ratio at the floor is {closest}");
        // And still nowhere near clipping what the camera came to look at.
        assert!(
            camera.near() * 4.0 < camera.distance,
            "near {} of {}",
            camera.near(),
            camera.distance
        );
    }

    /// `far` is the length of every pick ray as well as the far plane, so a
    /// value that does not cover the content silently stops the cursor working
    /// on the far side of the model.
    #[test]
    fn the_pick_ray_reaches_the_far_corner_from_any_distance() {
        let mut camera = model();
        for _ in 0..60 {
            let furthest = camera.eye().distance(camera.target) + camera.content_radius;
            assert!(camera.far() >= furthest, "far {} against {furthest}", camera.far());
            camera.zoom_by_about(OrbitCamera::zoom_factor(-1.0), camera.target);
        }
    }

    /// Orbiting about the target is what orbiting has always been, so it has
    /// to survive being routed through the pivoted version.
    #[test]
    fn orbiting_about_the_target_is_exactly_what_it_was_before() {
        let mut pivoted = model();
        let mut plain = model();

        for delta in [Vec2::new(30.0, -10.0), Vec2::new(-5.0, 40.0), Vec2::new(0.0, 300.0)] {
            pivoted.orbit_about(delta, pivoted.target);
            plain.orbit(delta);
            assert!((pivoted.yaw - plain.yaw).abs() < 1.0e-6);
            assert!((pivoted.pitch - plain.pitch).abs() < 1.0e-6);
            assert!(
                pivoted.target.distance(plain.target) < 1.0e-3,
                "the target moved: {:?} vs {:?}",
                pivoted.target,
                plain.target
            );
        }
    }

    #[test]
    fn the_pivot_stays_at_the_same_screen_position_through_a_whole_gesture() {
        let mut camera = model();
        let pivot = off_axis(&camera, 8.0, 11.0);
        let before = on_screen(&camera, pivot);

        for step in 0..60 {
            camera.orbit_about(Vec2::new(7.0, if step % 3 == 0 { -5.0 } else { 4.0 }), pivot);
        }
        let after = on_screen(&camera, pivot);
        assert!(
            before.distance(after) < 1.0e-3,
            "the pivot drifted from {before:?} to {after:?} over sixty steps"
        );
    }

    /// A pivot on the view axis is the degenerate case for anything that
    /// recovers angles from a desired eye position; the basis-difference
    /// route has no such case.
    #[test]
    fn orbiting_about_a_pivot_on_the_view_axis_does_not_jump() {
        let mut camera = model();
        let pivot = camera.eye();
        let before = camera.eye();
        camera.orbit_about(Vec2::new(20.0, 15.0), pivot);
        assert!(
            camera.eye().distance(before) < 1.0e-3,
            "the eye left the pivot it was standing on: {:?}",
            camera.eye()
        );
        assert!(camera.target.is_finite());
    }
}
