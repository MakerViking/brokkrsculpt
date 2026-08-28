// SPDX-License-Identifier: AGPL-3.0-only

//! Sphere tracing the volume to find the surface under the cursor.

use glam::Vec3;

use crate::brick::BrickCoord;
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
/// Values are clamped to the narrow band, so near the surface each step
/// advances by at most NARROW_BAND voxels rather than by the true distance.
/// Empty space is crossed a whole brick at a time instead -- see
/// [`empty_brick_span`] -- so the bound now buys about ten times the reach it
/// used to, and a model thousands of voxels across is within it.
///
/// **It was not, before that skip existed.** Three voxels a step over 512
/// steps is 1536 voxels of travel measured from where the ray ENTERS the
/// body's box, which a 3540 voxel model exceeds: the picker returned `None`
/// where there plainly was surface, and the cursor quietly stopped working on
/// exactly the models big enough to want it. Raising the bound instead would
/// have paid for the emptiness rather than skipping it.
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
        // Empty space costs one step per brick rather than one per three
        // voxels. The bracket moves with it: `previous_t` stays at the near
        // side of the jump because no crossing can lie inside it, so a hit
        // found at the far side is still bracketed by a point known to be
        // outside, and `refine` still sees a single sign change.
        if let Some(jump) = empty_brick_span(volume, origin + direction * t, direction) {
            previous_t = t;
            t += jump;
            if t > max_distance {
                return None;
            }
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

/// How far the ray may advance from `point` without any chance of stepping
/// over a surface, when `point` sits in a brick that is empty everywhere.
///
/// `None` when the brick carries detail, when it is solid, or when the saving
/// would be nothing.
///
/// # Why this cannot skip a crossing
///
/// [`Volume::sample_world`] interpolates between `floor(p / voxel_size)` and
/// the voxel one further on, and nothing else. So every sample taken strictly
/// inside the box from the brick's origin voxel to its LAST voxel reads only
/// voxels belonging to this brick -- one voxel is given up at the high face
/// and none at the low one. A brick whose every voxel is positive therefore
/// samples positive over the whole of that box, and a sign change inside it is
/// not merely unlikely but impossible.
///
/// The half voxel taken off the exit is float safety, not part of the
/// argument: it keeps the next sample clear of the face the argument rests on.
///
/// **A back off measured along the ray would not have been enough**, which is
/// the trap here. A ray running nearly parallel to a side face is metres from
/// its exit face and a thousandth of a voxel from the neighbouring brick the
/// whole way, and that neighbour may be dense. The box is what makes the
/// distance to EVERY face part of the bound; `box_exit` takes the nearest of
/// the six.
fn empty_brick_span(volume: &Volume, point: Vec3, direction: Vec3) -> Option<f32> {
    let voxel_size = volume.voxel_size();
    let coord = BrickCoord::containing((point / voxel_size).floor().as_ivec3());
    // An absent brick answers `OUTSIDE`, which is the common case and the one
    // worth the most: the air in front of a model is not stored at all.
    if volume.brick_fill(coord)? <= 0.0 {
        return None;
    }
    let low = coord.origin().as_vec3() * voxel_size;
    let high = coord.max_voxel().as_vec3() * voxel_size;
    let span = box_exit(point, direction, low, high)? - 0.5 * voxel_size;
    (span > 0.0).then_some(span)
}

/// Distance along the ray from `point` to where it leaves an axis aligned box.
///
/// Negative or `None` when `point` is not inside, which the caller reads as
/// "no saving here" rather than as an error.
fn box_exit(point: Vec3, direction: Vec3, low: Vec3, high: Vec3) -> Option<f32> {
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        let step = direction[axis];
        if step == 0.0 {
            // Parallel to this pair of faces: it never leaves through them,
            // but it also has to already be between them.
            if point[axis] < low[axis] || point[axis] > high[axis] {
                return None;
            }
            continue;
        }
        let near = (low[axis] - point[axis]) / step;
        let far = (high[axis] - point[axis]) / step;
        exit = exit.min(near.max(far));
    }
    exit.is_finite().then_some(exit)
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

    /// The reach the empty brick skip exists for.
    ///
    /// Without it a step is at most [`crate::brick::NARROW_BAND`] voxels, so
    /// `MAX_STEPS` bounds the march at about 1536 voxels of travel and this
    /// ray -- 3540 voxels of empty space before the sphere -- ran out of steps
    /// and reported a miss. The picker backs its start up to the body's
    /// bounding box, which hides the failure for a ray aimed at the near face
    /// and does nothing for one crossing the box's own empty interior.
    #[test]
    fn a_ray_crossing_a_large_empty_region_still_finds_the_far_surface() {
        let voxel_size = 0.0565;
        let centre = Vec3::splat(200.0);
        let radius = 10.0;
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(centre, radius);

        let origin = centre - Vec3::X * 200.0;
        assert!(
            200.0 / voxel_size > 3000.0,
            "the gap has to be past the old bound for this test to mean anything"
        );

        let hit = raycast(&volume, origin, Vec3::X, 1000.0)
            .expect("the surface is right there, 3540 voxels away");
        let expected = 200.0 - radius;
        assert!(
            (hit.distance - expected).abs() < voxel_size * 2.0,
            "hit at {} but geometry says {expected}",
            hit.distance
        );
    }

    /// The skip must not read a neighbouring brick's surface as empty.
    ///
    /// A ray sliding along the inside of a brick face is the case a back off
    /// measured along the ray gets wrong: it is far from its exit face and
    /// touching the neighbour the whole way.
    #[test]
    fn a_ray_grazing_the_face_of_an_empty_brick_still_sees_the_brick_beyond_it() {
        let voxel_size = 1.0;
        let mut volume = Volume::new(voxel_size);
        // Big enough to span many bricks, so the march is mostly skipping.
        volume.seed_sphere(Vec3::splat(256.0), 60.0);

        // Aimed just under the sphere's equator, so the ray runs close to the
        // surface for a long way before meeting it.
        for offset in [0.0_f32, 0.5, 0.9, 1.1, 5.0, 20.0, 55.0] {
            let origin = Vec3::new(0.0, 256.0 + offset, 256.0);
            let hit = raycast(&volume, origin, Vec3::X, 1000.0);
            let expected = 256.0 - (60.0_f32 * 60.0 - offset * offset).sqrt();
            let hit = hit.unwrap_or_else(|| panic!("missed the sphere at offset {offset}"));
            assert!(
                (hit.distance - expected).abs() < 2.0,
                "at offset {offset} hit {} but geometry says {expected}",
                hit.distance
            );
        }
    }

    /// The skip is an optimisation, so it has to agree with the march it
    /// replaces everywhere, not only where it was aimed.
    #[test]
    fn skipping_empty_bricks_finds_the_same_surface_as_crawling_through_them() {
        let (volume, centre, radius) = sphere_volume();
        for (dy, dz) in [(0.0, 0.0), (5.0, -3.0), (-12.0, 8.0), (19.0, 0.0), (0.0, -19.5)] {
            let origin = centre - Vec3::X * 300.0 + Vec3::new(0.0, dy, dz);
            let hit = raycast(&volume, origin, Vec3::X, 1000.0)
                .unwrap_or_else(|| panic!("missed at {dy},{dz}"));
            let sideways = (dy * dy + dz * dz).sqrt();
            let expected = 300.0 - (radius * radius - sideways * sideways).sqrt();
            assert!(
                (hit.distance - expected).abs() < 1.0,
                "at {dy},{dz} hit {} but geometry says {expected}",
                hit.distance
            );
        }
    }
}
