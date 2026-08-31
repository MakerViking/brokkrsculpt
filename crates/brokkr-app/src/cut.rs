// SPDX-License-Identifier: AGPL-3.0-only

//! Reading a screen gesture as a convex cutter.
//!
//! Everything here is two dimensional and in widget pixels. It knows nothing
//! about the camera, the document or the field: it turns the path a pointer
//! took into a small convex polygon, and `app` turns that polygon into the
//! [`brokkr_core::ClipPlane`]s that do the work. Keeping the split there is
//! what makes the fiddly part -- which strokes mean what, and what a hull does
//! to a stroke that doubles back -- testable without a camera or a volume.
//!
//! # One gesture, three readings
//!
//! There is one drag and no shape picker, and the shape is inferred from what
//! was drawn. That is a deliberate rejection of the strip of shape buttons
//! every comparable tool ships: the tool card is already at the height where
//! anything further has to displace something, and a stroke that is obviously
//! a straight line does not need to be declared one.
//!
//! * [`CutShape::Line`] -- what ships today. One plane, infinite, through the
//!   whole model. It is the DEFAULT rather than a special case, and the
//!   threshold that leaves it is deliberately generous: a hand-drawn "straight"
//!   drag across most of a window wanders by more pixels than anyone expects,
//!   and turning the one cut people already rely on into a lasso because their
//!   hand shook would be much worse than occasionally reading a gentle arc as a
//!   line.
//! * [`CutShape::Curve`] -- an open stroke. The stroke is extended past both
//!   ends along the direction it was travelling and then hulled, which is what
//!   ZBrush does when it extrapolates a short stroke "to the edge, following
//!   the final path". Without the extension the hull of an arc is a thin
//!   crescent, so a shallow curve across a shoulder -- the first thing anyone
//!   tries -- would gouge a sliver out of the middle and leave everything above
//!   it standing.
//! * [`CutShape::Lasso`] -- a stroke whose ends met. The hull of the points.
//!
//! # The hull, and what it costs
//!
//! **The region removed is always convex, and it is always the hull of what was
//! drawn rather than the stroke itself.** A C-shaped lasso therefore takes the
//! gap in the C as well. That is a real limitation and it is not hidden: the
//! preview draws the decimated hull, so what is shown is exactly what goes, and
//! the divergence between the two is visible before the button comes up rather
//! than discovered afterwards.
//!
//! The alternative -- decomposing a non-convex polygon and taking the union of
//! the pieces -- is not merely harder, it is actively broken on a signed
//! distance field: at a point deep inside the union but on an internal seam the
//! value comes out exactly zero rather than large and positive, so the brick
//! spans zero, classifies as crossing instead of removed, is promoted to a
//! dense 128 KB, and is then refused by the collapse test. The result is a
//! permanently resident brick in the middle of a region the user just deleted,
//! once per seam, with the exported mesh perfectly correct and nothing to see.

use glam::Vec2;

/// Sides a cut hull may have, before the depth caps.
///
/// Every side is a plane, and every plane is a dot product per voxel of every
/// brick it crosses. Sixteen is where the cost stops being free and the shape
/// stops getting visibly better: a hand-drawn loop simplified to sixteen
/// vertices is not distinguishable from the same loop at forty at any zoom the
/// viewport offers.
pub const MAX_CUT_PLANES: usize = 16;

/// How far the pointer must travel before another point is kept.
///
/// The raw event stream is far denser than the shape needs and its density
/// depends on how fast the hand moved, which would make an identical-looking
/// stroke simplify differently depending on speed.
pub const CUT_PATH_SPACING_PX: f32 = 2.0;

/// How far a stroke may wander from its own chord and still be a straight line.
///
/// **Two terms, and the relative one is what matters.** A fixed pixel tolerance
/// is wrong at both ends of the range: eight pixels is a lot of wobble on a
/// short drag and nothing at all on one that crosses the window. The fraction
/// is what makes a long drag stay a line, and the long drag is precisely the
/// case where reading a lasso by accident would be most destructive.
///
/// Both numbers are judgement rather than measurement, and this is the constant
/// most likely to want moving after a day of real use. [`CutShape::Line`] is
/// the safe reading, so they err generous.
pub const LASSO_DEVIATION_PX: f32 = 10.0;

/// The same tolerance as a fraction of the stroke's own chord. See
/// [`LASSO_DEVIATION_PX`].
pub const LASSO_DEVIATION_FRACTION: f32 = 0.06;

/// How close a stroke's ends must come for it to count as closed.
///
/// Generous, because closing a loop exactly is fiddly with a mouse and the
/// failure of reading a closed loop as open is loud -- the extension points
/// send the cutter off across the whole view.
pub const CLOSE_RADIUS_PX: f32 = 28.0;

/// How far past its ends an open stroke is extended, as a multiple of the
/// stroke's own span.
///
/// Large enough that the extended hull covers the viewport for any stroke worth
/// making, small enough that the numbers stay far from where `f32` gets coarse.
const EXTENSION_SPANS: f32 = 8.0;

/// How the stroke was read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutShape {
    /// A straight drag: one infinite plane, exactly as the cut has always
    /// worked.
    Line,
    /// An open stroke, extended past both ends and hulled.
    Curve,
    /// A closed stroke: the hull of what was drawn.
    Lasso,
}

impl CutShape {
    /// What the live tool strip says while this shape is being drawn.
    pub fn label(self) -> &'static str {
        match self {
            CutShape::Line => "PLANE CUT",
            CutShape::Curve => "CURVE CUT",
            CutShape::Lasso => "LASSO CUT",
        }
    }
}

/// A gesture read as a convex region of the screen.
#[derive(Debug, Clone, PartialEq)]
pub struct CutGesture {
    pub shape: CutShape,
    /// The region, in widget pixels, wound counter-clockwise in a y-down
    /// coordinate system.
    ///
    /// Exactly two points for [`CutShape::Line`] -- the ends of the drag, which
    /// name one plane and no region. Three or more, and at most
    /// [`MAX_CUT_PLANES`], for the other two.
    pub hull: Vec<Vec2>,
}

/// Read a pointer path as a cutter, or `None` if it is not a gesture.
///
/// `None` is a click, or a path so short it cannot name a direction. It is not
/// an error -- the caller says so in the status line -- and refusing here is
/// the point: a destructive tool must decline an ambiguous gesture rather than
/// pick an interpretation.
pub fn read_stroke(path: &[Vec2], click_slop_px: f32) -> Option<CutGesture> {
    let first = *path.first()?;
    let last = *path.last()?;

    // The span of the whole path, not the distance between its ends: a stroke
    // that returns to where it started has a zero chord and is still a gesture.
    let span = path_span(path);
    if span < click_slop_px {
        return None;
    }

    let closed = path.len() >= 3 && first.distance(last) <= CLOSE_RADIUS_PX;

    if !closed {
        let chord = first.distance(last);
        let tolerance = LASSO_DEVIATION_PX.max(chord * LASSO_DEVIATION_FRACTION);
        if straightness(path) <= tolerance {
            // Deliberately the raw ends and not a hull. This is today's cut and
            // it must stay today's cut: two points, one plane, one cross
            // product, and the side convention that a test observes rather than
            // derives.
            return Some(CutGesture { shape: CutShape::Line, hull: vec![first, last] });
        }
    }

    let simplified = simplify(path, CUT_PATH_SPACING_PX);
    let points = if closed { simplified } else { extended(&simplified, span) };

    let hull = decimate(convex_hull(&points), MAX_CUT_PLANES);
    if hull.len() < 3 {
        // Collinear after all -- a there-and-back stroke, or one whose hull
        // collapsed. There is no region here and inventing one would remove
        // material along a line the user did not draw a side for.
        return None;
    }

    Some(CutGesture { shape: if closed { CutShape::Lasso } else { CutShape::Curve }, hull })
}

/// The longest distance between any point of the path and its first point.
///
/// Used instead of the chord so that a there-and-back stroke is still measured
/// as having gone somewhere.
fn path_span(path: &[Vec2]) -> f32 {
    let Some(&first) = path.first() else {
        return 0.0;
    };
    path.iter().map(|point| first.distance(*point)).fold(0.0, f32::max)
}

/// The furthest any point strays from the line between the path's ends.
fn straightness(path: &[Vec2]) -> f32 {
    let (Some(&first), Some(&last)) = (path.first(), path.last()) else {
        return 0.0;
    };
    path.iter().map(|point| distance_to_segment(*point, first, last)).fold(0.0, f32::max)
}

/// Perpendicular distance from a point to a segment, clamped to its ends.
fn distance_to_segment(point: Vec2, from: Vec2, to: Vec2) -> f32 {
    let along = to - from;
    let length_squared = along.length_squared();
    if length_squared <= f32::EPSILON {
        return point.distance(from);
    }
    let t = ((point - from).dot(along) / length_squared).clamp(0.0, 1.0);
    point.distance(from + along * t)
}

/// The stroke with a far point added past each end, along the direction the
/// stroke was going when it got there.
///
/// This is what turns an open curve into a region. Without it the hull of an
/// arc is the arc's own thin crescent, and cutting with that removes a sliver
/// from the middle of the stroke and nothing else -- which looks like the tool
/// is broken, because the one thing the user asked for is the part it leaves.
fn extended(path: &[Vec2], span: f32) -> Vec<Vec2> {
    let mut points = path.to_vec();
    let reach = span * EXTENSION_SPANS;

    // Taken from the end SEGMENT rather than from the whole chord: the
    // extension has to continue the direction the stroke was travelling when it
    // stopped, which on a curve is nothing like the direction from end to end.
    if let (Some(&first), Some(&second)) = (path.first(), path.get(1))
        && let Some(away) = (first - second).try_normalize()
    {
        points.push(first + away * reach);
    }
    if path.len() >= 2
        && let (Some(&last), Some(&before)) = (path.last(), path.get(path.len() - 2))
        && let Some(away) = (last - before).try_normalize()
    {
        points.push(last + away * reach);
    }
    points
}

/// Douglas-Peucker: drop every point that lies within `epsilon` of the line
/// its neighbours already describe.
///
/// Iterative rather than recursive. A pointer path can carry thousands of
/// points and the recursion depth of the textbook form is bounded only by the
/// path length, which is a stack overflow driven by how long someone held the
/// button down.
pub fn simplify(path: &[Vec2], epsilon: f32) -> Vec<Vec2> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut keep = vec![false; path.len()];
    keep[0] = true;
    keep[path.len() - 1] = true;

    let mut pending = vec![(0usize, path.len() - 1)];
    while let Some((start, end)) = pending.pop() {
        if end <= start + 1 {
            continue;
        }
        let mut worst = 0.0;
        let mut at = start;
        for (index, point) in path.iter().enumerate().take(end).skip(start + 1) {
            let distance = distance_to_segment(*point, path[start], path[end]);
            if distance > worst {
                worst = distance;
                at = index;
            }
        }
        if worst > epsilon {
            keep[at] = true;
            pending.push((start, at));
            pending.push((at, end));
        }
    }

    path.iter().zip(keep).filter_map(|(point, keep)| keep.then_some(*point)).collect()
}

/// Andrew's monotone chain convex hull.
///
/// Returns the hull wound counter-clockwise in a y-down coordinate system,
/// with no repeated first point. Fewer than three input points come back as
/// they went in.
pub fn convex_hull(points: &[Vec2]) -> Vec<Vec2> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }

    // Positive for a left turn in a y-down system, which is what makes the
    // finished hull counter-clockwise on screen.
    let turn = |o: Vec2, a: Vec2, b: Vec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    let mut hull: Vec<Vec2> = Vec::with_capacity(sorted.len() * 2);
    for &point in &sorted {
        while hull.len() >= 2 && turn(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.0 {
            hull.pop();
        }
        hull.push(point);
    }
    let lower = hull.len() + 1;
    for &point in sorted.iter().rev().skip(1) {
        while hull.len() >= lower && turn(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.0
        {
            hull.pop();
        }
        hull.push(point);
    }
    // The last point is the first one again.
    hull.pop();
    hull
}

/// Drop hull vertices until at most `most` remain: outlier spikes first, then
/// whichever vertex is bending the outline least.
///
/// **Two rules and not one, and the order between them is the whole design.**
///
/// The first rule exists because `min` over the planes is exact inside the
/// region and outside a face, but outside a convex EDGE it over-estimates by
/// `r * (1 - sin(t/2))` at a wedge of interior angle `t` -- 0.88 of a voxel at
/// a right angle and 2.48 at twenty degrees, taken at the edge of the narrow
/// band. An outlier sample on a shaky hand-drawn loop IS an acute hull vertex,
/// so the sharpest corners are both the most expensive to approximate and the
/// least likely to have been meant. Anything sharper than [`SPIKE_ANGLE`] goes
/// before anything else is considered.
///
/// The second rule is Visvalingam-Whyatt: drop the vertex whose triangle with
/// its two neighbours has the least area, which is the vertex whose removal
/// changes the outline least.
///
/// **Angle alone is not enough, and the failure is not obvious.** On a hand-drawn
/// loop -- or any roughly round one -- every vertex has nearly the same angle,
/// so an angle-only rule picks whichever tie it happens to break first, and
/// removing a vertex makes its two neighbours sharper. The next pass therefore
/// picks a neighbour, and the one after that its neighbour, and the decimation
/// eats its way around one arc of the loop and leaves a long chord across it.
/// The hull is still convex, still has the right vertex count, and quietly
/// spares a sixty-degree wedge of everything the user drew a line around. This
/// was caught by `the_region_cut_matches_the_region_drawn` and by nothing else.
///
/// Dropping a hull vertex only ever SHRINKS the region either way, so
/// decimation errs toward removing less material, which is the recoverable
/// direction.
///
/// # The keys are cached, and that is worth a paragraph
///
/// This runs on **every pointer motion event** while a cut is being drawn, over
/// the whole hull accumulated so far, so its cost is what decides whether a
/// long or shaky stroke stays smooth. Recomputing every vertex's key on every
/// removal is `h^2/2` evaluations of two `try_normalize` (two square roots), an
/// `acos` and a cross product: measured at 1.90 ms for a 512-vertex hull,
/// 7.36 ms at 1024 and 28.13 ms at 2048 -- past the frame at the sizes a
/// determined scribble reaches.
///
/// Removing a vertex changes the `before`/`at`/`after` triple of exactly TWO
/// others, its two neighbours. Everything else keeps the key it already had, so
/// caching and repairing those two is not an approximation of the old
/// behaviour, it is the same computation with the redundant part removed: same
/// keys, same `<` comparison, same first-lowest-index tie-break, same output
/// vertex for vertex. Measured at 0.15 / 0.58 / 2.10 ms for the same three
/// sizes -- **12 to 13 times faster**.
pub fn decimate(mut hull: Vec<Vec2>, most: usize) -> Vec<Vec2> {
    let floor = most.max(3);
    if hull.len() <= floor {
        return hull;
    }
    let key_at = |hull: &[Vec2], index: usize| {
        let before = hull[(index + hull.len() - 1) % hull.len()];
        let at = hull[index];
        let after = hull[(index + 1) % hull.len()];
        let angle = interior_angle(before, at, after);
        // The tuple IS the ordering: a spike sorts by its angle in the first
        // slot and beats every non-spike, which all share an infinite first
        // slot and are separated by area.
        if angle < SPIKE_ANGLE {
            (angle, 0.0)
        } else {
            (f32::INFINITY, triangle_area(before, at, after))
        }
    };

    let mut keys: Vec<(f32, f32)> = (0..hull.len()).map(|index| key_at(&hull, index)).collect();

    while hull.len() > floor {
        let mut worst = 0usize;
        let mut best = (f32::INFINITY, f32::INFINITY);
        for (index, key) in keys.iter().enumerate() {
            if *key < best {
                best = *key;
                worst = index;
            }
        }
        hull.remove(worst);
        keys.remove(worst);
        // The two vertices that were either side of the one just removed are
        // now neighbours, so their triples changed and nobody else's did. After
        // the removal those are the entries at `worst` and the one before it,
        // both taken modulo the NEW length.
        let left = (worst + hull.len() - 1) % hull.len();
        let right = worst % hull.len();
        keys[left] = key_at(&hull, left);
        keys[right] = key_at(&hull, right);
    }
    hull
}

/// Interior angle below which a hull vertex is treated as an outlier spike
/// rather than as part of the outline.
///
/// A right angle. Nothing drawn deliberately as part of a loop comes to a
/// corner sharper than this -- a hand tracing round a lump produces obtuse
/// vertices throughout -- while a single stray sample spikes far below it.
const SPIKE_ANGLE: f32 = std::f32::consts::FRAC_PI_2;

/// Twice the area of the triangle three points make. The factor is dropped
/// because only the ordering is used.
fn triangle_area(before: Vec2, at: Vec2, after: Vec2) -> f32 {
    let a = before - at;
    let b = after - at;
    (a.x * b.y - a.y * b.x).abs()
}

/// The interior angle at `at`, in radians, or `TAU` when it is degenerate.
///
/// Degenerate comes back as the largest possible angle so that a coincident
/// vertex is never chosen as the sharpest corner -- it has no angle to be
/// sharp, and picking it would drop a real corner instead.
fn interior_angle(before: Vec2, at: Vec2, after: Vec2) -> f32 {
    let (Some(a), Some(b)) = ((before - at).try_normalize(), (after - at).try_normalize()) else {
        return std::f32::consts::TAU;
    };
    a.dot(b).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLOP: f32 = 4.0;

    fn line(from: Vec2, to: Vec2, steps: usize) -> Vec<Vec2> {
        (0..=steps).map(|step| from.lerp(to, step as f32 / steps as f32)).collect()
    }

    /// The default, and the one that must not move: a straight drag is one
    /// plane and two points, whatever else this module learns to read.
    #[test]
    fn a_straight_drag_is_still_a_line() {
        let path = line(Vec2::new(100.0, 300.0), Vec2::new(900.0, 300.0), 200);
        let gesture = read_stroke(&path, SLOP).expect("a long drag is a gesture");
        assert_eq!(gesture.shape, CutShape::Line);
        assert_eq!(gesture.hull.len(), 2, "a line is two points and one plane");
        assert_eq!(gesture.hull[0], path[0]);
        assert_eq!(gesture.hull[1], *path.last().unwrap());
    }

    /// **The regression this tolerance exists to prevent.** A hand does not
    /// draw a straight line, and a drag across most of a window that wobbles by
    /// a couple of dozen pixels is a line by intent. Reading it as a lasso would
    /// make the one cut people already rely on conditional on hand steadiness.
    #[test]
    fn a_wobbly_long_drag_is_still_a_line() {
        let path: Vec<Vec2> = (0..=200)
            .map(|step| {
                let t = step as f32 / 200.0;
                // +-20 px of wander across an 800 px drag.
                Vec2::new(100.0 + t * 800.0, 300.0 + (t * 9.0).sin() * 20.0)
            })
            .collect();
        let gesture = read_stroke(&path, SLOP).expect("a long drag is a gesture");
        assert_eq!(gesture.shape, CutShape::Line, "a shaky hand turned a line into a region");
    }

    /// And the other side of it: a stroke that is deliberately, obviously bent
    /// must not be flattened into a plane through the middle of the model.
    #[test]
    fn a_deliberate_arc_is_a_curve() {
        let path: Vec<Vec2> = (0..=100)
            .map(|step| {
                let t = step as f32 / 100.0;
                Vec2::new(100.0 + t * 800.0, 400.0 - (t * std::f32::consts::PI).sin() * 250.0)
            })
            .collect();
        let gesture = read_stroke(&path, SLOP).expect("an arc is a gesture");
        assert_eq!(gesture.shape, CutShape::Curve);
        assert!(gesture.hull.len() >= 3, "a curve must name a region");
    }

    /// The failure the extension points exist to fix. Hulling an arc on its own
    /// gives a thin crescent, so the cut would take a sliver out of the middle
    /// of the stroke and leave the thing the user was cutting off standing.
    #[test]
    fn a_curve_covers_more_than_the_crescent_between_its_ends() {
        let path: Vec<Vec2> = (0..=100)
            .map(|step| {
                let t = step as f32 / 100.0;
                Vec2::new(100.0 + t * 800.0, 400.0 - (t * std::f32::consts::PI).sin() * 250.0)
            })
            .collect();
        let gesture = read_stroke(&path, SLOP).expect("an arc is a gesture");

        let reach = gesture
            .hull
            .iter()
            .map(|point| point.distance(Vec2::new(500.0, 400.0)))
            .fold(0.0, f32::max);
        assert!(
            reach > 800.0,
            "the curve's hull stayed inside the stroke, so it is the crescent: reach {reach}"
        );
    }

    /// A closed loop is its own hull, with no extension anywhere.
    #[test]
    fn a_closed_loop_is_a_lasso_and_stays_where_it_was_drawn() {
        let path: Vec<Vec2> = (0..=64)
            .map(|step| {
                let angle = step as f32 / 64.0 * std::f32::consts::TAU;
                Vec2::new(400.0, 300.0) + Vec2::new(angle.cos(), angle.sin()) * 120.0
            })
            .collect();
        let gesture = read_stroke(&path, SLOP).expect("a loop is a gesture");
        assert_eq!(gesture.shape, CutShape::Lasso);
        for point in &gesture.hull {
            let out = point.distance(Vec2::new(400.0, 300.0));
            assert!(
                (100.0..=140.0).contains(&out),
                "a lasso's hull left the loop it was drawn as: {out} from centre"
            );
        }
    }

    /// A click is not a cut, and neither is a twitch.
    #[test]
    fn a_click_is_not_a_gesture() {
        assert!(read_stroke(&[], SLOP).is_none());
        assert!(read_stroke(&[Vec2::new(10.0, 10.0)], SLOP).is_none());
        assert!(read_stroke(&line(Vec2::ZERO, Vec2::new(2.0, 0.0), 4), SLOP).is_none());
    }

    /// A stroke that goes out and comes straight back covers real distance and
    /// still encloses nothing. It must be refused rather than hulled into a
    /// sliver, because a sliver thinner than the lattice removes material in
    /// some places and not others depending on where the voxels happen to fall.
    #[test]
    fn a_there_and_back_stroke_encloses_nothing() {
        let mut path = line(Vec2::new(100.0, 300.0), Vec2::new(500.0, 300.0), 100);
        path.extend(line(Vec2::new(500.0, 300.0), Vec2::new(100.0, 300.0), 100));
        // Reads as closed -- it ends where it started -- and hulls to a line.
        assert!(read_stroke(&path, SLOP).is_none(), "a degenerate loop produced a region");
    }

    /// No stroke may produce more planes than the ceiling, however intricate.
    #[test]
    fn a_hull_never_exceeds_the_plane_ceiling() {
        let path: Vec<Vec2> = (0..=400)
            .map(|step| {
                let angle = step as f32 / 400.0 * std::f32::consts::TAU;
                let wobble = 120.0 + (angle * 17.0).sin() * 30.0;
                Vec2::new(400.0, 300.0) + Vec2::new(angle.cos(), angle.sin()) * wobble
            })
            .collect();
        let gesture = read_stroke(&path, SLOP).expect("a loop is a gesture");
        assert!(
            gesture.hull.len() <= MAX_CUT_PLANES,
            "an intricate loop produced {} planes",
            gesture.hull.len()
        );
    }

    #[test]
    fn the_hull_of_a_square_is_its_corners() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
            // Interior points, which must not survive.
            Vec2::new(5.0, 5.0),
            Vec2::new(2.0, 8.0),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4, "the hull kept an interior point: {hull:?}");
    }

    /// The hull must wind consistently, because the sign of the plane normals
    /// built from it depends on the winding and a hull that sometimes came back
    /// the other way round would cut the wrong side at random.
    #[test]
    fn the_hull_winds_the_same_way_whatever_order_it_is_given() {
        let square = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        let signed_area = |hull: &[Vec2]| {
            (0..hull.len())
                .map(|index| {
                    let a = hull[index];
                    let b = hull[(index + 1) % hull.len()];
                    a.x * b.y - b.x * a.y
                })
                .sum::<f32>()
        };
        let forward = signed_area(&convex_hull(&square));
        let mut backward: Vec<Vec2> = square.to_vec();
        backward.reverse();
        assert!(forward.abs() > 0.0, "the hull has no area");
        assert_eq!(
            forward.is_sign_positive(),
            signed_area(&convex_hull(&backward)).is_sign_positive(),
            "the hull's winding depends on the order it was handed its points"
        );
    }

    /// Decimation drops the sharpest corner, not the smallest one. A long thin
    /// spike off an otherwise round hull is the vertex to lose.
    #[test]
    fn decimation_drops_the_sharpest_corner_first() {
        let mut hull: Vec<Vec2> = (0..8)
            .map(|step| {
                let angle = step as f32 / 8.0 * std::f32::consts::TAU;
                Vec2::new(angle.cos(), angle.sin()) * 100.0
            })
            .collect();
        // A spike: far out, and therefore very acute.
        let spike = Vec2::new(0.0, -600.0);
        hull.insert(2, spike);

        let kept = decimate(hull, 8);
        assert_eq!(kept.len(), 8);
        assert!(!kept.contains(&spike), "the spike survived decimation: {kept:?}");
    }

    /// **Decimation must spread, not eat an arc.**
    ///
    /// On a round loop every vertex has nearly the same angle. An angle-only
    /// rule breaks the tie the same way every pass, and removing a vertex makes
    /// its neighbours sharper -- so it works its way around one arc and leaves
    /// a long chord across it. The hull stays convex and keeps the right vertex
    /// count, and quietly spares a whole wedge of what the user drew a line
    /// around.
    ///
    /// Asserted on the area kept rather than on which vertices survived: what
    /// matters is that the polygon still covers the loop, not which particular
    /// samples represent it.
    #[test]
    fn decimating_a_round_loop_keeps_its_whole_area() {
        let circle: Vec<Vec2> = (0..48)
            .map(|step| {
                let angle = step as f32 / 48.0 * std::f32::consts::TAU;
                Vec2::new(angle.cos(), angle.sin()) * 100.0
            })
            .collect();
        let area = |hull: &[Vec2]| {
            (0..hull.len())
                .map(|index| {
                    let a = hull[index];
                    let b = hull[(index + 1) % hull.len()];
                    a.x * b.y - b.x * a.y
                })
                .sum::<f32>()
                .abs()
                * 0.5
        };

        let full = area(&circle);
        let kept = area(&decimate(circle.clone(), MAX_CUT_PLANES));
        // A regular 16-gon inscribed in a circle keeps 97.4% of its area. Any
        // rule that eats one arc loses far more than that: chopping a 60 degree
        // wedge off alone costs about 9%.
        assert!(
            kept > full * 0.95,
            "decimation lost {:.1}% of the loop, so it ate an arc rather than spreading",
            (1.0 - kept / full) * 100.0
        );
    }

    /// And the two rules together: a spike on a round loop must still go first,
    /// and the rest of the loop must still be spread over.
    #[test]
    fn a_spike_goes_before_the_loop_is_thinned() {
        let mut points: Vec<Vec2> = (0..20)
            .map(|step| {
                let angle = step as f32 / 20.0 * std::f32::consts::TAU;
                Vec2::new(angle.cos(), angle.sin()) * 100.0
            })
            .collect();
        let spike = Vec2::new(0.0, -900.0);
        points.insert(7, spike);

        let kept = decimate(points, MAX_CUT_PLANES);
        assert!(!kept.contains(&spike), "the spike outlived ordinary vertices: {kept:?}");
        let reach = kept.iter().map(|p| p.length()).fold(0.0, f32::max);
        assert!(reach < 200.0, "something far out survived: reach {reach}");
    }

    /// **The cached decimation is the naive one, vertex for vertex.**
    ///
    /// `decimate` caches each vertex's key and repairs only the two neighbours
    /// of whatever it just removed. That is sound because removing a vertex
    /// changes no other triple -- but "sound because I reasoned it through" is
    /// how an off-by-one in the modular index survives, so this runs the
    /// straightforward O(h^2) form beside it and demands the same answer.
    ///
    /// Over awkward shapes on purpose: ties everywhere (a regular polygon),
    /// spikes that must go first, and a jittery loop where the ordering is
    /// decided by small float differences.
    #[test]
    fn caching_the_decimation_keys_changes_nothing_it_produces() {
        /// The form this replaced: every key recomputed on every removal.
        fn naive(mut hull: Vec<Vec2>, most: usize) -> Vec<Vec2> {
            while hull.len() > most.max(3) {
                let mut worst = 0usize;
                let mut best = (f32::INFINITY, f32::INFINITY);
                for index in 0..hull.len() {
                    let before = hull[(index + hull.len() - 1) % hull.len()];
                    let after = hull[(index + 1) % hull.len()];
                    let angle = interior_angle(before, hull[index], after);
                    let key = if angle < SPIKE_ANGLE {
                        (angle, 0.0)
                    } else {
                        (f32::INFINITY, triangle_area(before, hull[index], after))
                    };
                    if key < best {
                        best = key;
                        worst = index;
                    }
                }
                hull.remove(worst);
            }
            hull
        }

        let regular: Vec<Vec2> = (0..40)
            .map(|step| {
                let angle = step as f32 / 40.0 * std::f32::consts::TAU;
                Vec2::new(angle.cos(), angle.sin()) * 100.0
            })
            .collect();

        let mut spiky = regular.clone();
        spiky.insert(9, Vec2::new(0.0, -700.0));
        spiky.insert(23, Vec2::new(640.0, 30.0));

        let jittery: Vec<Vec2> = (0..64)
            .map(|step| {
                let angle = step as f32 / 64.0 * std::f32::consts::TAU;
                let wobble = 100.0 + (angle * 11.0).sin() * 23.0 + (angle * 3.0).cos() * 7.0;
                Vec2::new(angle.cos(), angle.sin()) * wobble
            })
            .collect();

        for (name, points) in [("regular", regular), ("spiky", spiky), ("jittery", jittery)] {
            for most in [3, 5, 8, MAX_CUT_PLANES, 30] {
                assert_eq!(
                    decimate(points.clone(), most),
                    naive(points.clone(), most),
                    "{name} decimated to {most} differs from the straightforward form"
                );
            }
        }
    }

    /// Simplification must not move the ends: they are the line's two points
    /// and the curve's two extension anchors.
    #[test]
    fn simplifying_keeps_both_ends() {
        let path = line(Vec2::new(3.0, 7.0), Vec2::new(103.0, 57.0), 50);
        let simple = simplify(&path, 2.0);
        assert_eq!(simple.first(), path.first());
        assert_eq!(simple.last(), path.last());
        assert!(simple.len() < path.len(), "a straight path did not simplify");
    }

    /// A path of thousands of points must not recurse its way through the
    /// stack. This is the shape a user produces by holding the button down.
    #[test]
    fn simplifying_a_very_long_path_does_not_overflow_the_stack() {
        let path: Vec<Vec2> = (0..20_000)
            .map(|step| Vec2::new(step as f32 * 0.01, (step as f32 * 0.01).sin() * 50.0))
            .collect();
        assert!(!simplify(&path, 0.5).is_empty());
    }
}
