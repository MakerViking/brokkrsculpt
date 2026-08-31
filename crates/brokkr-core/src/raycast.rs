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

/// A run of solid along a ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    /// Distance along the ray where the solid begins.
    pub enter: f32,
    /// Distance along the ray where it ends.
    pub exit: f32,
}

/// What a ray found: the first run of solid, and where the next one starts.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SolidSpans {
    /// The first complete run of solid the ray passed through.
    ///
    /// `None` when the ray met no surface, and also when it entered solid and
    /// never came out within `max_distance` -- an unfinished span has no exit,
    /// and a depth cap placed from a guessed one would cut where nothing was
    /// measured.
    pub first: Option<Span>,
    /// Where the NEXT run of solid begins, if the ray reached one.
    ///
    /// This is the number a depth cap has to respect: it is the surface behind
    /// the thing being cut off, and the wall that a cutter reaching too far
    /// would thin from the front without anybody seeing it.
    pub next_enter: Option<f32>,
}

/// March a ray and report the first solid run and the start of the one behind
/// it.
///
/// `direction` must be unit length.
///
/// # Why this is not [`raycast`] called twice
///
/// Three differences, and each one is a bug if it is left out.
///
/// **The empty-brick skip only works outside.** [`empty_brick_span`] returns
/// `None` for any brick that is not uniformly outside, so it accelerates the
/// air in front of a body and the gap between two bodies, and does nothing at
/// all for the leg from a surface to the far side of the same solid. Applying
/// it while inside would be harmless but pointless; not noticing that it does
/// nothing there is what makes the next point a surprise.
///
/// **The step rule has to flip sign inside.** [`raycast`] advances by
/// `previous_d.max(min_step)`, and inside the solid `previous_d` is negative,
/// so `max` picks the floor and the march crawls a quarter of a voxel at a
/// time -- 512 steps buys 128 voxels of interior, which does not cross a model.
/// Inside, the distance to the surface is `-d`, and stepping by that is what
/// makes the exit reachable.
///
/// **A ray that starts inside contributes nothing.** [`raycast`] reports that
/// as a hit at zero so the brush has somewhere to go. Here it would be a span
/// with a fabricated `enter`, and a far cap placed from it sits at a depth
/// measured from a surface the ray never crossed.
pub fn first_solid_spans(
    volume: &Volume,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> SolidSpans {
    let voxel_size = volume.voxel_size();
    let min_step = 0.25 * voxel_size;
    let field = |t: f32| volume.sample_world(origin + direction * t) * voxel_size;

    let mut previous_t = 0.0_f32;
    let mut previous_d = field(0.0);
    if previous_d <= 0.0 {
        return SolidSpans::default();
    }

    let mut found = SolidSpans::default();
    let mut enter: Option<f32> = None;
    let mut inside = false;
    let mut t = previous_d.max(min_step);

    for _ in 0..MAX_STEPS {
        if t > max_distance {
            break;
        }
        if !inside && let Some(jump) = empty_brick_span(volume, origin + direction * t, direction) {
            previous_t = t;
            t += jump;
            if t > max_distance {
                break;
            }
        }
        let d = field(t);
        if !inside && d <= 0.0 {
            let crossing = refine(&field, previous_t, t);
            match enter {
                None => enter = Some(crossing),
                Some(_) => {
                    // The second run: this is the surface behind, and it is all
                    // the caller needs from it.
                    found.next_enter = Some(crossing);
                    break;
                }
            }
            inside = true;
        } else if inside && d > 0.0 {
            // Arguments swapped: here `t` is the outside end of the bracket and
            // `previous_t` the inside one. `refine` branches only on the sign
            // it samples, so it needs no other change.
            let crossing = refine(&field, t, previous_t);
            if let Some(start) = enter
                && found.first.is_none()
            {
                found.first = Some(Span { enter: start, exit: crossing });
            }
            inside = false;
        }
        previous_t = t;
        previous_d = d;
        t += if inside { (-previous_d).max(min_step) } else { previous_d.max(min_step) };
    }

    found
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

    /// Two balls in a row: the march must report the first as a complete span
    /// and the second as the surface behind it.
    ///
    /// This is the measurement the depth cap is placed from, so getting the
    /// second number wrong is what makes a cutter eat the wall behind a spur.
    #[test]
    fn a_ray_through_two_balls_reports_the_first_and_the_start_of_the_second() {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::new(0.0, 0.0, 0.0), 10.0);
        volume.seed_sphere(Vec3::new(0.0, 0.0, 40.0), 10.0);

        let origin = Vec3::new(0.0, 0.0, -40.0);
        let spans = first_solid_spans(&volume, origin, Vec3::Z, 200.0);

        let first = spans.first.expect("the ray met the near ball");
        // Enters at z = -10 and leaves at z = +10, measured from z = -40.
        assert!((first.enter - 30.0).abs() < 1.0, "entered at {}", first.enter);
        assert!((first.exit - 50.0).abs() < 1.0, "left at {}", first.exit);
        let next = spans.next_enter.expect("the ray met the far ball");
        // The far ball's near face is at z = 30.
        assert!((next - 70.0).abs() < 1.0, "the next surface was reported at {next}");
    }

    /// One ball is one span and nothing behind it, which is what tells the
    /// depth rule there is nothing to spare and the cut may go through.
    #[test]
    fn a_ray_through_one_ball_reports_no_surface_behind_it() {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 10.0);

        let spans = first_solid_spans(&volume, Vec3::new(0.0, 0.0, -40.0), Vec3::Z, 200.0);
        assert!(spans.first.is_some(), "the ray missed the ball");
        assert_eq!(spans.next_enter, None, "a lone ball reported something behind it");
    }

    /// A ray that misses reports nothing at all, rather than a span of zero
    /// length that a cap could be placed from.
    #[test]
    fn a_ray_that_misses_reports_nothing() {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 10.0);

        let spans = first_solid_spans(&volume, Vec3::new(50.0, 0.0, -40.0), Vec3::Z, 200.0);
        assert_eq!(spans, SolidSpans::default());
    }

    /// A ray that starts inside contributes nothing, deliberately: it has no
    /// entry, and a cap placed from a fabricated one sits at a depth measured
    /// from a surface the ray never crossed.
    #[test]
    fn a_ray_that_starts_inside_reports_no_span() {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 10.0);

        let spans = first_solid_spans(&volume, Vec3::ZERO, Vec3::Z, 200.0);
        assert_eq!(spans.first, None, "a ray starting inside invented an entry");
    }

    /// **The interior leg has to step by the distance to the surface, not by
    /// the floor.**
    ///
    /// Inside the solid the field is negative, so a step rule copied from
    /// `raycast` picks `min_step` -- a quarter of a voxel -- and 512 steps buys
    /// 128 voxels. This ball is 400 voxels across, so a march that crawls never
    /// reaches the far side and reports no span at all.
    #[test]
    fn a_ray_crosses_a_body_wider_than_the_step_budget() {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 100.0);

        let spans = first_solid_spans(&volume, Vec3::new(0.0, 0.0, -200.0), Vec3::Z, 600.0);
        let first = spans.first.expect("the march did not cross a 200 mm ball");
        assert!((first.exit - first.enter - 200.0).abs() < 2.0, "span was {first:?}");
    }
}
