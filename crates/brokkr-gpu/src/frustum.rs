// SPDX-License-Identifier: AGPL-3.0-only

//! View frustum culling.
//!
//! At the sizes M2 targets a model is several thousand bricks, and issuing a
//! draw call for every one of them costs more CPU time than the whole frame
//! budget allows. Most of them are behind the camera or off the side of the
//! screen at any moment, so the cheapest saving available is not drawing those.

use glam::{Mat4, Vec3, Vec4};

/// The six planes bounding what the camera can see.
///
/// Each plane is stored as `ax + by + cz + d`, positive on the inside.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract the planes from a combined view projection matrix.
    ///
    /// This is the Gribb and Hartmann method: each clip space boundary is a
    /// linear combination of the matrix rows, which falls out of the fact that
    /// clipping tests those same combinations. Reading rows out of a column
    /// major matrix is why the indexing looks inside out.
    ///
    /// The near plane is the third row alone rather than a sum, because wgpu's
    /// clip space runs depth from 0 to 1 rather than -1 to 1.
    pub fn from_view_projection(view_projection: Mat4) -> Self {
        let matrix = view_projection;
        let row = |index: usize| {
            Vec4::new(
                matrix.x_axis[index],
                matrix.y_axis[index],
                matrix.z_axis[index],
                matrix.w_axis[index],
            )
        };
        let (x, y, z, w) = (row(0), row(1), row(2), row(3));

        Self { planes: [w + x, w - x, w + y, w - y, z, w - z] }
    }

    /// Whether an axis aligned box is at least partly inside.
    ///
    /// Conservative: a box straddling a corner of the frustum can be reported
    /// as visible when it is not. That only ever costs a draw call, whereas the
    /// opposite mistake would make geometry disappear.
    pub fn intersects(&self, minimum: Vec3, maximum: Vec3) -> bool {
        for plane in &self.planes {
            let normal = plane.truncate();
            // The corner of the box furthest along this plane's normal. If even
            // that one is outside, every corner is.
            let furthest = Vec3::new(
                if normal.x >= 0.0 { maximum.x } else { minimum.x },
                if normal.y >= 0.0 { maximum.y } else { minimum.y },
                if normal.z >= 0.0 { maximum.z } else { minimum.z },
            );
            if normal.dot(furthest) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at +Z looking back at the origin, of the same convention the
    /// application uses.
    fn camera(distance: f32) -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(
            Vec3::new(0.0, 0.0, distance),
            Vec3::ZERO,
            Vec3::Y,
        );
        let projection = glam::camera::rh::proj::directx::perspective(
            45f32.to_radians(),
            16.0 / 9.0,
            0.1,
            100.0,
        );
        projection * view
    }

    fn cube(centre: Vec3, half: f32) -> (Vec3, Vec3) {
        (centre - Vec3::splat(half), centre + Vec3::splat(half))
    }

    #[test]
    fn a_box_at_the_target_is_visible() {
        let frustum = Frustum::from_view_projection(camera(10.0));
        let (minimum, maximum) = cube(Vec3::ZERO, 1.0);
        assert!(frustum.intersects(minimum, maximum));
    }

    #[test]
    fn a_box_behind_the_camera_is_culled() {
        let frustum = Frustum::from_view_projection(camera(10.0));
        let (minimum, maximum) = cube(Vec3::new(0.0, 0.0, 40.0), 1.0);
        assert!(!frustum.intersects(minimum, maximum));
    }

    #[test]
    fn a_box_far_off_to_the_side_is_culled() {
        let frustum = Frustum::from_view_projection(camera(10.0));
        for offset in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y] {
            let (minimum, maximum) = cube(offset * 60.0, 1.0);
            assert!(!frustum.intersects(minimum, maximum), "not culled at {offset:?}");
        }
    }

    #[test]
    fn a_box_beyond_the_far_plane_is_culled() {
        let frustum = Frustum::from_view_projection(camera(10.0));
        let (minimum, maximum) = cube(Vec3::new(0.0, 0.0, -500.0), 1.0);
        assert!(!frustum.intersects(minimum, maximum));
    }

    #[test]
    fn a_box_large_enough_to_contain_the_camera_is_visible() {
        // The classic culling bug: testing only whether corners are inside
        // makes an enclosing box vanish, taking the ground or the model's
        // interior with it.
        let frustum = Frustum::from_view_projection(camera(10.0));
        let (minimum, maximum) = cube(Vec3::ZERO, 1000.0);
        assert!(frustum.intersects(minimum, maximum));
    }

    #[test]
    fn a_box_straddling_the_edge_of_the_view_is_kept() {
        // Erring toward keeping things costs a draw call. Erring the other way
        // makes geometry pop in and out at the edge of the screen.
        let frustum = Frustum::from_view_projection(camera(10.0));
        // Wide enough that part of it is certainly on screen.
        let minimum = Vec3::new(-30.0, -1.0, -1.0);
        let maximum = Vec3::new(0.5, 1.0, 1.0);
        assert!(frustum.intersects(minimum, maximum));
    }

    #[test]
    fn orbiting_the_camera_moves_what_is_visible() {
        // A sanity check that the planes really follow the camera rather than
        // being fixed in world space.
        let near_side = cube(Vec3::new(0.0, 0.0, 8.0), 0.5);
        let front = Frustum::from_view_projection(camera(10.0));
        assert!(front.intersects(near_side.0, near_side.1), "should be between eye and target");

        // Same box, camera moved to the far side looking back: now behind it.
        let view =
            glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO, Vec3::Y);
        let projection =
            glam::camera::rh::proj::directx::perspective(45f32.to_radians(), 16.0 / 9.0, 0.1, 9.0);
        let behind = Frustum::from_view_projection(projection * view);
        assert!(!behind.intersects(near_side.0, near_side.1));
    }
}
