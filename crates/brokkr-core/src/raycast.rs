// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sphere tracing the volume to find the surface under the cursor.

use glam::Vec3;

use crate::volume::Volume;

/// Where a ray met the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// World space point on the surface.
    pub position: Vec3,
    /// Outward unit normal at that point.
    pub normal: Vec3,
    /// Distance along the ray.
    pub distance: f32,
}

/// Upper bound on marching steps.
///
/// Values are clamped to the narrow band, so far from the surface each step
/// advances by NARROW_BAND voxels rather than the true distance. Crossing a
/// 256 voxel volume therefore costs roughly 90 steps and this bound is
/// comfortable. It is what makes the simple march affordable without a brick
/// level acceleration structure, which belongs with the GPU rewrite.
const MAX_STEPS: u32 = 512;

/// Bisection steps used to refine the crossing once it is bracketed.
const REFINE_STEPS: u32 = 20;

/// March a ray through the volume and return the first surface crossing.
///
/// `direction` must be unit length.
pub fn raycast(volume: &Volume, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<Hit> {
    let voxel_size = volume.voxel_size();
    // Never let a step stall: a grazing ray can sit at a near zero positive
    // distance for a long time otherwise.
    let min_step = 0.25 * voxel_size;

    let field = |t: f32| volume.sample_world(origin + direction * t) * voxel_size;

    let mut previous_t = 0.0_f32;
    let mut previous_d = field(0.0);

    if previous_d <= 0.0 {
        // The ray starts inside the solid. Report the start point so the caller
        // still has somewhere to put the brush.
        return Some(Hit {
            position: origin,
            normal: volume.gradient_world(origin),
            distance: 0.0,
        });
    }

    let mut t = previous_d.max(min_step);
    for _ in 0..MAX_STEPS {
        if t > max_distance {
            return None;
        }
        let d = field(t);
        if d <= 0.0 {
            let hit_t = refine(&field, previous_t, t);
            let position = origin + direction * hit_t;
            return Some(Hit {
                position,
                normal: volume.gradient_world(position),
                distance: hit_t,
            });
        }
        previous_t = t;
        previous_d = d;
        t += previous_d.max(min_step);
    }
    None
}

/// Bisect a bracketed crossing, where `outside` has a positive field value and
/// `inside` a non positive one.
fn refine(field: &impl Fn(f32) -> f32, mut outside: f32, mut inside: f32) -> f32 {
    for _ in 0..REFINE_STEPS {
        let mid = 0.5 * (outside + inside);
        if field(mid) > 0.0 {
            outside = mid;
        } else {
            inside = mid;
        }
    }
    0.5 * (outside + inside)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_volume() -> (Volume, Vec3, f32) {
        let centre = Vec3::splat(64.0);
        let radius = 20.0;
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(centre, radius);
        (volume, centre, radius)
    }

    #[test]
    fn hits_a_sphere_where_geometry_says_it_should() {
        let (volume, centre, radius) = sphere_volume();
        let origin = centre - Vec3::X * 200.0;
        let hit = raycast(&volume, origin, Vec3::X, 1000.0).expect("ray should hit the sphere");

        let expected = centre.x - radius - origin.x;
        assert!(
            (hit.distance - expected).abs() < 0.5,
            "hit at {} but geometry says {expected}",
            hit.distance
        );
        // The normal on the minus X side of a sphere points along minus X.
        assert!(hit.normal.dot(-Vec3::X) > 0.95, "normal was {:?}", hit.normal);
    }

    #[test]
    fn misses_when_the_ray_passes_by() {
        let (volume, centre, radius) = sphere_volume();
        let origin = centre - Vec3::X * 200.0 + Vec3::Y * (radius * 3.0);
        assert!(raycast(&volume, origin, Vec3::X, 1000.0).is_none());
    }

    #[test]
    fn respects_the_maximum_distance() {
        let (volume, centre, radius) = sphere_volume();
        let origin = centre - Vec3::X * 200.0;
        let short = 200.0 - radius - 10.0;
        assert!(raycast(&volume, origin, Vec3::X, short).is_none());
    }

    #[test]
    fn a_ray_starting_inside_reports_its_own_origin() {
        let (volume, centre, _) = sphere_volume();
        let hit = raycast(&volume, centre, Vec3::X, 1000.0).expect("inside counts as a hit");
        assert_eq!(hit.distance, 0.0);
        assert_eq!(hit.position, centre);
    }
}
