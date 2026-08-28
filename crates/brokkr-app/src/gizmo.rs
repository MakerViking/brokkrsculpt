// SPDX-License-Identifier: AGPL-3.0-only

//! The transform gizmo: move, turn and scale one body.
//!
//! Three arrows to move along an axis, three squares to move in a plane, three
//! rings to turn about an axis, and a box in the middle to scale. It rides the
//! overlay pass the brush ring, the mirror planes and the navigation cube
//! already share -- see `brokkr_gpu::overlay`, whose header asks that the next
//! overlay use the third mechanism rather than inventing a fourth, and
//! `SculptRenderer::overlay_pass`, which the cube's own pass was generalised
//! into for it.
//!
//! # The gizmo is aligned to the WORLD, always, and that is not a preference
//!
//! Blender offers global and local orientations and calls it a setting. Here it
//! cannot be one. A whole-voxel translation is a translation along the world
//! axes and a lossless quarter turn is a turn about a world axis, so a gizmo
//! that rotated with the body would put its second gesture on axes the lattice
//! does not have -- and every subsequent move and turn would fall through to a
//! resample. The gizmo stays square to the world because that is where the
//! exact route lives.
//!
//! # What the drag produces, and what it does NOT do
//!
//! [`drag`] returns the gesture as an ABSOLUTE [`Similarity`] measured from the
//! pixel the button went down on, never an increment from the last pointer
//! event. Accumulating increments makes a gesture's result depend on how many
//! motion events the operating system happened to deliver, so an identical
//! sweep of the hand gives a different answer on a busy frame; and it makes
//! "drag out and come back" merely approximately the identity, where the point
//! of the whole design is that it is exactly the identity.
//!
//! Nothing here touches a [`brokkr_core::Volume`]. The gesture is arithmetic on
//! a camera and two pixels; the field is rebuilt once, on release, by
//! `Brokkr::rebake_gizmo`.
//!
//! # One scale factor for the draw and the hit test
//!
//! [`world_per_pixel`] is called once per frame and its result is threaded
//! through both [`build`] and [`pick`]. Keeping two of them -- a constant
//! screen size for the draw and a world-space tolerance for the picking -- is
//! the bug Unreal and tinygizmo have both shipped: they agree at one distance
//! and drift apart everywhere else, and the camera work this sits on top of has
//! just made a much wider range of distances reachable.
//!
//! # Shafts are quads, not lines
//!
//! wgpu's `PrimitiveState` has no line-width field and the overlay pipelines
//! are built with `multisample: Default::default()`, so a `push_line` shaft is
//! one physical pixel wide -- unaimable with a stylus, which is the input this
//! application is built around. Every part of the gizmo except the drag
//! preview box is therefore triangles.

use brokkr_core::{NodeId, Similarity};
use brokkr_gpu::OverlayBatch;
use glam::{Quat, Vec2, Vec3};

use crate::camera::OrbitCamera;
use crate::theme;

/// Length of an arrow's shaft, in logical pixels.
const SHAFT_PX: f32 = 62.0;
/// Length of the cone on the end of it.
const HEAD_PX: f32 = 18.0;
/// Radius of that cone's base.
const HEAD_RADIUS_PX: f32 = 6.0;
/// Half-width of the shaft ribbon.
const SHAFT_HALF_PX: f32 = 1.6;
/// Where the per-axis scale box sits along its axis, in pixels.
///
/// Between the move arrow's head and the rotation ring: the arrow ends at
/// `SHAFT_PX + HEAD_PX` (80), this box is centred at 92, and [`RING_PX`] is
/// 128.
///
/// **Do not read that spacing as "these three never compete for a pixel",
/// which is what this comment used to say.** They are clear in WORLD units
/// along the axis and that is not the question -- a ring is a circle seen at
/// an angle, so its projection sweeps every radius from 0 to `RING_PX` as the
/// camera turns and crosses this box at a large fraction of camera angles.
/// Moving the ring from 104 to 128 reduced that; it did not remove it. Any
/// change to the spacing has to be judged by sweeping cameras and picking,
/// never by comparing these constants.
const SCALE_BOX_PX: f32 = 92.0;

/// Half the scale box's side, in pixels. Its grab region is this plus GRAB_PX.
const SCALE_BOX_HALF_PX: f32 = 5.0;

/// The rotation ring's radius, in pixels.
///
/// **Moved out from 104 to make room for the per-axis scale box**, and the
/// reason is worth keeping: at 104 the ring's PROJECTION passed over the box at
/// 92 even though the two are twelve pixels apart in world terms. A ring is a
/// circle seen at an angle, so its screen distance from a point on the axis is
/// not its world distance, and reasoning about the gap in world pixels said
/// they were clear when they were not. A test caught it -- a press aimed at the
/// box came back `Ring(1)`.
pub const RING_PX: f32 = 128.0;
/// Half-width of a ring's band.
const RING_HALF_PX: f32 = 2.0;
/// How many segments a ring is drawn and picked with.
const RING_SEGMENTS: usize = 48;
/// Near and far edges of a plane handle, along each of its two axes.
const PLANE_NEAR_PX: f32 = 20.0;
const PLANE_FAR_PX: f32 = 40.0;
/// Half-extent of the box in the middle, which scales.
const CENTRE_PX: f32 = 8.0;

/// How near a handle the pointer has to be, in logical pixels.
///
/// Blender settled at "a few pixels" and then reopened it (T68525) because a
/// few pixels is not enough with a tablet, where there is no cursor resting
/// against the target to nudge. Eight is chosen for the stylus.
const GRAB_PX: f32 = 8.0;

/// How long an axis has to be ON SCREEN before it may be grabbed.
///
/// An axis pointing nearly at the camera projects to almost nothing, and the
/// drag maths for it is anomalous rather than merely imprecise: the ray and the
/// axis are close to parallel, so the closest-point solve divides by something
/// near zero and a one-pixel move sends the body a hundred millimetres. Refuse
/// it instead. The plane handles cover the same motion, and their maths is
/// well conditioned at exactly the angle this is not.
const MIN_SHAFT_PX: f32 = 12.0;

/// The coarsest snap a rotation is offered, and the only one worth having.
///
/// **A quarter turn, and deliberately not five or fifteen degrees.** The whole
/// value of snapping here is that a snapped gesture takes
/// [`brokkr_core::Bake::Exact`] and costs the surface nothing; a fifteen degree
/// snap is exactly as lossy as thirty-seven degrees and would only make the
/// loss feel deliberate. Free angles are available on the modifier, and the
/// status line says what they cost.
const ROTATION_SNAP: f32 = std::f32::consts::FRAC_PI_2;

/// The step a snapped uniform scale moves in.
///
/// No scale is lossless, so this buys predictability rather than exactness:
/// what it does guarantee is that coming back to the pixel the drag started on
/// gives exactly 1.0, which is what makes a scale gesture cancellable by hand
/// as well as by Escape.
const SCALE_SNAP: f32 = 0.05;

/// The narrowest and widest a single gesture may scale.
///
/// Not a taste limit. Below the floor the body shrinks under its own voxel and
/// the bake returns almost nothing; above the ceiling the destination footprint
/// -- and so the memory the bake allocates -- grows with the cube of it.
///
/// **Both are applied to the COMPOSED scale as well as to the gesture**, which
/// is the half that was missing and the half that mattered: `Similarity::then`
/// multiplies scales, so two drags each saturating this ceiling compose to 400
/// and ten drags at the floor compose to 1e-13. See the clamp in [`drag`].
pub(crate) const MIN_SCALE: f32 = 0.05;
pub(crate) const MAX_SCALE: f32 = 20.0;

/// One thing on the gizmo that can be grabbed.
///
/// The axis index is 0, 1, 2 for X, Y, Z. `Plane(i)` is the plane
/// PERPENDICULAR to axis `i`, which is how every other tool numbers them and
/// which makes the axis the one thing the handle does not move along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    Axis(u8),
    Plane(u8),
    Ring(u8),
    Uniform,
    /// Squash or stretch along ONE axis. What `Uniform` is not.
    Scale(u8),
}

impl Handle {
    /// What a grab on this handle is about to do, for the status line.
    pub fn verb(self) -> &'static str {
        match self {
            Handle::Axis(_) | Handle::Plane(_) => "move",
            Handle::Ring(_) => "turn",
            Handle::Uniform => "scale",
            Handle::Scale(_) => "resize",
        }
    }
}

/// The transform gizmo, armed on one body.
///
/// # `placement` is absolute, and that is what bounds the damage
///
/// It is the total map from the field as it was when the gizmo ARMED, not a
/// composition of what each gesture did to the last result. Every bake goes
/// `base.warped(placement)`, so the number of lossy passes a body has suffered
/// is bounded by how many times the user left the gizmo rather than by how many
/// times they nudged it. Thirty adjustments cost one pass. See
/// `Brokkr::rebake_gizmo`, which is where that promise is kept, and
/// `thirty_adjustments_cost_exactly_one_resample_pass`, which is what holds it.
#[derive(Debug, Clone, Copy)]
pub struct Gizmo {
    /// The body being placed.
    pub body: NodeId,
    /// The pivot, in the BASE field's world space. Everything turns about it.
    pub base_pivot: Vec3,
    /// The total placement since arming.
    pub placement: Similarity,
    /// The placement as it stood when the live drag's button went down.
    ///
    /// A drag is absolute from its press pixel, so it needs the state it is
    /// absolute FROM. This is also what Escape restores.
    pub pinned: Similarity,
    pub hovered: Option<Handle>,
    pub grabbed: Option<Handle>,
    /// The base field's world box, drawn as a wireframe while dragging so the
    /// user can see where the body is going before it is rebuilt there.
    pub base_low: Vec3,
    pub base_high: Vec3,
    /// The largest TOTAL scale a bake of this body can be built at, from
    /// `Brokkr::largest_scale_that_fits`.
    ///
    /// **On the gizmo rather than checked at the release, because a refusal
    /// after the drag is an answer that arrives too late to act on.** The
    /// gesture is the question; folding the ceiling into it means the preview
    /// box stops growing exactly where the bake would stop fitting, and what
    /// the user is looking at is always a placement that can be built.
    ///
    /// Always finite and in `1.0..=MAX_SCALE`, which [`drag`] relies on: it
    /// derives a `clamp` BOUND from this, and `f32::clamp` panics when its
    /// bounds are NaN.
    pub max_scale: f32,
}

impl Gizmo {
    pub fn new(body: NodeId, low: Vec3, high: Vec3, max_scale: f32) -> Self {
        Self {
            body,
            base_pivot: (low + high) * 0.5,
            placement: Similarity::IDENTITY,
            pinned: Similarity::IDENTITY,
            hovered: None,
            grabbed: None,
            base_low: low,
            base_high: high,
            // Filtered rather than clamped: `clamp` passes NaN straight
            // through, and a NaN here becomes a `clamp` bound in `drag`, which
            // panics. One is the largest scale that is always affordable.
            max_scale: if max_scale.is_finite() { max_scale.clamp(1.0, MAX_SCALE) } else { 1.0 },
        }
    }

    /// Where the gizmo is drawn: the pivot carried by everything done so far.
    pub fn origin(&self) -> Vec3 {
        self.placement.transform_point(self.base_pivot)
    }

    /// Where the live drag is measured about.
    fn pinned_origin(&self) -> Vec3 {
        self.pinned.transform_point(self.base_pivot)
    }
}

/// How many world units one logical pixel covers at `origin`.
///
/// The perspective divide, at the DEPTH of the point rather than at the
/// camera's orbit distance: a gizmo on a body away from the target would
/// otherwise be drawn the wrong size, and the picking would be wrong by the
/// same factor in the same direction, which is the failure that hides itself.
pub fn world_per_pixel(camera: &OrbitCamera, origin: Vec3, viewport_height: f32) -> f32 {
    // `orientation() * Z` points from the target toward the eye, so the view
    // direction is its negation.
    let forward = -(camera.orientation() * Vec3::Z);
    let depth = (origin - camera.eye()).dot(forward).max(camera.near());
    2.0 * depth * (camera.fov_y * 0.5).tan() / viewport_height.max(1.0)
}

/// Where a world point lands, in widget pixels. `None` when it is behind the
/// eye, where the perspective divide would fold it back onto the screen at a
/// plausible-looking and completely wrong position.
fn to_pixels(camera: &OrbitCamera, viewport: Vec2, point: Vec3) -> Option<Vec2> {
    let aspect = viewport.x / viewport.y.max(1.0);
    let clip = camera.view_projection(aspect) * point.extend(1.0);
    if clip.w <= 1.0e-6 {
        return None;
    }
    let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
    Some(Vec2::new((ndc.x + 1.0) * 0.5 * viewport.x, (1.0 - ndc.y) * 0.5 * viewport.y))
}

fn axis_of(index: u8) -> Vec3 {
    [Vec3::X, Vec3::Y, Vec3::Z][index as usize % 3]
}

/// The world direction a per-axis scale box sits on: the BODY's axis.
///
/// **The one place the gizmo is not square to the world, and it has to be.**
/// The module header is right that a move and a quarter turn stay on world axes
/// because that is where the exact route lives -- but a scale is never
/// `Bake::Exact`, so nothing is given up here.
///
/// What forced it: `Similarity::then` stores `rotation = next.rotation *
/// self.rotation` and `scale = next.scale * self.scale`, and `transform_point`
/// applies `R * (S * p)`. Composed onto a rotation already in `pinned`, that is
/// `R2 R1 S2 S1 p` where the stepwise truth is `R2 S2 R1 S1 p`. Those are not
/// the same map -- but the composition is not broken, because
/// `R1 S2 = (R1 S2 R1inv) R1`, so what it computes is a squash along the body's
/// OWN axes. Perfectly representable, which is why the composed placement stays
/// a valid `R S p + t` and why nothing in `similarity.rs` needs changing.
///
/// The lie was told here: the box was drawn and dragged on the world axis while
/// the squash went along the body's. After an ordinary snapped quarter turn --
/// `ROTATION_SNAP` is a quarter turn, so the DEFAULT unshifted ring drag leaves
/// a rotation in `pinned` -- a squash to 0.6 landed 8.49 mm from where the
/// handle said it would.
///
/// Taken from `placement` and NOT from `pinned`, which was the first version
/// and did nothing at all: `pinned` is only latched when a button goes down, so
/// after a completed turn it still holds the identity and the box stayed on the
/// world axis exactly as before. The two agree during a scale drag anyway -- a
/// scale gesture carries an identity rotation, so `pinned.then(gesture)` leaves
/// the rotation untouched -- so there is nothing to be gained by the staler of
/// the two and a whole fix to be lost.
fn scale_axis_of(gizmo: &Gizmo, index: u8) -> Vec3 {
    gizmo.placement.rotation * axis_of(index)
}

/// The two axes a plane handle spans, or a ring lies in.
fn other_axes(index: u8) -> (Vec3, Vec3) {
    let u = axis_of((index + 1) % 3);
    let v = axis_of((index + 2) % 3);
    (u, v)
}

/// The four corners of a plane handle, wound around its rim.
fn plane_quad(origin: Vec3, scale: f32, index: u8) -> [Vec3; 4] {
    let (u, v) = other_axes(index);
    let (near, far) = (PLANE_NEAR_PX * scale, PLANE_FAR_PX * scale);
    [
        origin + u * near + v * near,
        origin + u * far + v * near,
        origin + u * far + v * far,
        origin + u * near + v * far,
    ]
}

/// One point on a ring, by segment index.
///
/// **A function of the index rather than a `Vec` of the whole loop**, and that
/// is the no-allocation rule rather than a style preference: [`build`] runs
/// from `publish_camera`, which is every frame of an orbit, and it needs nine
/// of these loops -- three ring inners, three ring outers, three arrowhead
/// bases. Nine heap allocations per frame is exactly what the per-frame path is
/// not allowed to do. Indices wrap, so `step` may run past the end.
fn ring_point(origin: Vec3, scale: f32, index: u8, radius_px: f32, step: usize) -> Vec3 {
    let (u, v) = other_axes(index);
    let radius = radius_px * scale;
    let angle = (step % RING_SEGMENTS) as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
    origin + (u * angle.cos() + v * angle.sin()) * radius
}

/// Whether a widget-local point is anywhere near the gizmo at all.
///
/// **The bounds check that lets a press be claimed.** ONLY BOUNDS-CHECKED
/// EVENTS MAY CAPTURE: a widget that claims an unbounded event kills every
/// press after it, and this application has shipped that once already. [`pick`]
/// gates on this exactly the way `navcube::pick` gates on `navcube::contains`,
/// and `a_press_away_from_the_gizmo_is_not_claimed` is what holds it.
///
/// The disc is the outermost thing drawn -- the rings -- plus the grab
/// tolerance, so it is a superset of everything [`pick`] can return and can
/// never refuse a press that a handle would have wanted.
pub fn contains(camera: &OrbitCamera, viewport: Vec2, gizmo: &Gizmo, at: Vec2) -> bool {
    let origin = gizmo.origin();
    let Some(centre) = to_pixels(camera, viewport, origin) else {
        return false;
    };
    // The disc is measured on screen rather than in world units, because that
    // is the space the pointer is in. The radius is the ring's, converted
    // through the SAME factor the draw uses -- but a ring seen from an angle
    // projects to an ellipse inside that circle, never outside it, so the
    // circle bounds every part of the gizmo whatever the camera is doing.
    let scale = world_per_pixel(camera, origin, viewport.y);
    let Some(edge) = to_pixels(camera, viewport, origin + camera.right() * (RING_PX * scale))
    else {
        return false;
    };
    at.distance(centre) <= centre.distance(edge) + GRAB_PX
}

/// Which handle a widget-local point is over, if any.
///
/// Precedence is fixed rather than nearest-wins: `Uniform`, then `Plane`, then
/// `Axis`, then `Ring`. It is an ordering by how small the target is, so the
/// hardest thing to hit is never stolen by something larger sitting under the
/// same pixel -- and a ring seen edge-on projects to a line straight through
/// the middle of everything, which is exactly the case a nearest-wins rule gets
/// wrong.
pub fn pick(camera: &OrbitCamera, viewport: Vec2, gizmo: &Gizmo, at: Vec2) -> Option<Handle> {
    if !contains(camera, viewport, gizmo, at) {
        return None;
    }
    let origin = gizmo.origin();
    let scale = world_per_pixel(camera, origin, viewport.y);
    let centre = to_pixels(camera, viewport, origin)?;

    if at.distance(centre) <= CENTRE_PX + GRAB_PX {
        return Some(Handle::Uniform);
    }

    for index in 0..3u8 {
        let corners = plane_quad(origin, scale, index);
        let mut projected = [Vec2::ZERO; 4];
        let mut all_on_screen = true;
        for (slot, point) in projected.iter_mut().zip(corners) {
            match to_pixels(camera, viewport, point) {
                Some(pixel) => *slot = pixel,
                None => all_on_screen = false,
            }
        }
        if all_on_screen && inside_polygon(&projected, at) {
            return Some(Handle::Plane(index));
        }
    }

    for index in 0..3u8 {
        let axis = axis_of(index);
        let tip = origin + axis * ((SHAFT_PX + HEAD_PX) * scale);
        let (Some(from), Some(to)) =
            (to_pixels(camera, viewport, origin), to_pixels(camera, viewport, tip))
        else {
            continue;
        };
        // An axis pointing nearly at the camera is refused rather than picked
        // imprecisely. See MIN_SHAFT_PX.
        if from.distance(to) < MIN_SHAFT_PX {
            continue;
        }
        if distance_to_segment(at, from, to) <= GRAB_PX {
            return Some(Handle::Axis(index));
        }
    }

    for index in 0..3u8 {
        // Walked segment by segment rather than collected, for the reason
        // `ring_point` gives: this runs on every pointer move.
        let mut nearest = f32::INFINITY;
        let mut previous =
            to_pixels(camera, viewport, ring_point(origin, scale, index, RING_PX, 0));
        for step in 1..=RING_SEGMENTS {
            let point = ring_point(origin, scale, index, RING_PX, step);
            let current = to_pixels(camera, viewport, point);
            if let (Some(a), Some(b)) = (previous, current) {
                nearest = nearest.min(distance_to_segment(at, a, b));
            }
            previous = current;
        }
        if nearest <= GRAB_PX {
            return Some(Handle::Ring(index));
        }
    }

    // **After the rings and before the shafts, and the order is arithmetic
    // rather than taste.** The box is centred at SCALE_BOX_PX (92) and grabs
    // within SCALE_BOX_HALF_PX + GRAB_PX (13), so it reaches 79..105 -- which
    // overlaps the ring at RING_PX (104) at one end and the arrow head ending
    // at SHAFT_PX + HEAD_PX (80) at the other. Testing it first swallowed every
    // ring press, and a test caught it: a free turn reported an exact move.
    // Giving the ring first refusal costs the box nothing, because a press that
    // the ring wanted was never meant for the box.
    for index in 0..3u8 {
        let box_centre = origin + scale_axis_of(gizmo, index) * (SCALE_BOX_PX * scale);
        if let Some(pixel) = to_pixels(camera, viewport, box_centre)
            && at.distance(pixel) <= SCALE_BOX_HALF_PX + GRAB_PX
        {
            return Some(Handle::Scale(index));
        }
    }

    None
}

/// Distance from a point to a line segment, in whatever units it was given.
fn distance_to_segment(point: Vec2, from: Vec2, to: Vec2) -> f32 {
    let span = to - from;
    let length_squared = span.length_squared();
    if length_squared < 1.0e-9 {
        return point.distance(from);
    }
    let t = ((point - from).dot(span) / length_squared).clamp(0.0, 1.0);
    point.distance(from + span * t)
}

/// Whether a point is inside a convex quad given in order around its rim.
///
/// By the sign of the cross product at each edge, which needs no winding
/// convention: a projected quad's winding depends on which side of it the
/// camera is, and requiring one would make a plane handle unpickable from
/// behind.
fn inside_polygon(corners: &[Vec2], at: Vec2) -> bool {
    let mut positive = false;
    let mut negative = false;
    for index in 0..corners.len() {
        let a = corners[index];
        let b = corners[(index + 1) % corners.len()];
        let cross = (b - a).perp_dot(at - a);
        positive |= cross > 0.0;
        negative |= cross < 0.0;
    }
    // **A polygon with no area contains nothing, and saying so takes the second
    // clause.** Seen exactly edge-on, a plane handle's four corners project
    // onto one line; every cross product is then exactly zero, neither flag is
    // set, and `!(positive && negative)` alone answers TRUE for every point
    // tested against it. Since `Plane` outranks `Axis` in the precedence, the
    // handle swallowed the arrow drawn straight through it.
    //
    // Reachable in one click rather than by contrivance: at yaw 0 the corners
    // of `plane_quad(0)` project to the same column to the last bit, and at
    // pitch 0 plane 1 is collinear along the centre row at ANY yaw -- and the
    // navigation cube's Front face flies the camera to exactly 0.0 on both.
    //
    // The damage is one pixel wide, not a dead handle: a point OFF the line
    // still sees the doubled-back edges disagree and is correctly refused. But
    // the pixel it takes is the arrow's own centreline, which is where a user
    // aiming at an arrow puts the stylus.
    !(positive && negative) && (positive || negative)
}

/// The gesture a drag from `from` to `to` describes, about the gizmo's pinned
/// origin.
///
/// Absolute from the press pixel, never an increment; see the module header.
/// The result is the GESTURE alone -- the caller composes it onto
/// [`Gizmo::pinned`] -- so that a cancelled drag is undone by throwing this
/// away rather than by computing an inverse.
///
/// Returns [`Similarity::IDENTITY`] whenever the maths is ill conditioned: a
/// ray parallel to the axis it is dragging, a ray that misses the plane it is
/// dragging in. Doing nothing is the right answer there, and it is also the
/// only one that keeps the gesture continuous as the camera swings through the
/// degenerate angle.
///
/// `snap` carries the lattice to snap translations to rather than sitting
/// beside it as a `bool`, because the two are never independently meaningful:
/// snapping without a voxel size is not a thing this can do, and a voxel size
/// with snapping off is ignored. One argument cannot be half set.
pub fn drag(
    camera: &OrbitCamera,
    viewport: Vec2,
    gizmo: &Gizmo,
    handle: Handle,
    from: Vec2,
    to: Vec2,
    snap: Option<f32>,
) -> Similarity {
    let origin = gizmo.pinned_origin();
    let aspect = viewport.x / viewport.y.max(1.0);
    let ray = |pixel: Vec2| {
        let ndc = OrbitCamera::ndc_from_pixels(pixel, viewport);
        camera.ray(ndc, aspect)
    };

    match handle {
        Handle::Axis(index) => {
            let axis = axis_of(index);
            let (Some(start), Some(now)) =
                (along_axis(ray(from), origin, axis), along_axis(ray(to), origin, axis))
            else {
                return Similarity::IDENTITY;
            };
            let mut offset = axis * (now - start);
            if let Some(voxel_size) = snap {
                offset = snap_to_voxels(offset, voxel_size);
            }
            Similarity::moving(offset)
        }
        Handle::Plane(index) => {
            let normal = axis_of(index);
            let (Some(start), Some(now)) =
                (on_plane(ray(from), origin, normal), on_plane(ray(to), origin, normal))
            else {
                return Similarity::IDENTITY;
            };
            let mut offset = now - start;
            if let Some(voxel_size) = snap {
                offset = snap_to_voxels(offset, voxel_size);
            }
            Similarity::moving(offset)
        }
        Handle::Ring(index) => {
            let normal = axis_of(index);
            let (Some(start), Some(now)) =
                (on_plane(ray(from), origin, normal), on_plane(ray(to), origin, normal))
            else {
                return Similarity::IDENTITY;
            };
            let (u, v) = other_axes(index);
            let angle_of = |point: Vec3| {
                let spoke = point - origin;
                (spoke.dot(v)).atan2(spoke.dot(u))
            };
            let mut angle = crate::camera::wrap_angle(angle_of(now) - angle_of(start));
            if snap.is_some() {
                angle = (angle / ROTATION_SNAP).round() * ROTATION_SNAP;
            }
            if angle == 0.0 {
                return Similarity::IDENTITY;
            }
            Similarity::about(origin, Quat::from_axis_angle(normal, angle), Vec3::ONE, Vec3::ZERO)
        }
        Handle::Uniform => {
            let Some(centre) = to_pixels(camera, viewport, origin) else {
                return Similarity::IDENTITY;
            };
            // Radial on screen, in pixels. A world-space measure would need a
            // plane to measure in, and the only honest one is the view plane,
            // which is what this already is.
            let started = centre.distance(from);
            if started < 1.0 {
                return Similarity::IDENTITY;
            }
            // **Bounded on the TOTAL, not on this gesture.** The press that
            // selects `Uniform` has to land within `CENTRE_PX + GRAB_PX` of the
            // middle, so a 320 pixel drag already saturates `MAX_SCALE`; and
            // `Similarity::then` multiplies scales, so a second such drag would
            // compose to 400 and a third to 8000 with nothing in the way. What
            // is actually being bounded is the allocation `Volume::warped`
            // makes for the composed placement, so the composed placement is
            // where the bound has to bite.
            //
            // Both ends are widened to admit 1.0 whatever `pinned` holds: a
            // drag that comes back to the pixel it started on must be exactly
            // the identity, and a clamp that could exclude it would make a
            // gesture uncancellable by hand.
            //
            // **The floor comes off the SMALLEST axis and the ceiling off the
            // largest**, because `Similarity::then` multiplies scales component
            // by component, so it is the smallest axis that reaches `MIN_SCALE`
            // first and the largest that reaches the ceiling first. Taking both
            // from `max_element` -- which is what this did -- meant that after
            // a per-axis squash to (1, 1, 0.05) the floor was computed from 1.0
            // again, and the next uniform shrink composed to (0.05, 0.05,
            // 0.0025): twenty times under `MIN_SCALE`, which is precisely the
            // state that constant exists to prevent. `Handle::Scale` never had
            // the bug because it bounds against its own axis.
            let smallest = gizmo.pinned.scale.min_element().max(f32::MIN_POSITIVE);
            let largest = gizmo.pinned.scale.max_element().max(f32::MIN_POSITIVE);
            let highest = (gizmo.max_scale / largest).clamp(1.0, MAX_SCALE);
            let lowest = MIN_SCALE.max(MIN_SCALE / smallest).min(1.0);
            let mut scale = (centre.distance(to) / started).clamp(lowest, highest);
            if snap.is_some() {
                scale = ((scale / SCALE_SNAP).round() * SCALE_SNAP).clamp(lowest, highest);
            }
            if scale == 1.0 {
                return Similarity::IDENTITY;
            }
            Similarity::about(origin, Quat::IDENTITY, Vec3::splat(scale), Vec3::ZERO)
        }
        Handle::Scale(index) => {
            // **Per-axis scale: the same screen measure as `Uniform`, applied
            // to ONE component.** Measured along the projected axis rather than
            // radially, because a squash reads as "pull this end" and a radial
            // measure would grow it when the pointer moved sideways.
            //
            // The BODY's axis, matching where the box is drawn and picked. See
            // [`scale_axis_of`] for why that is not the world axis and why this
            // is the one part of the gizmo that turns with the body.
            let axis = scale_axis_of(gizmo, index);
            let Some(centre) = to_pixels(camera, viewport, origin) else {
                return Similarity::IDENTITY;
            };
            // A screen direction for this axis, from the body's own extent so
            // it is long enough to project stably. Only the DIRECTION is used
            // -- the length cancels in the ratio below -- so any positive
            // world length would do, and a large one keeps the projection out
            // of the noise.
            let span = (gizmo.base_high - gizmo.base_low).length().max(1.0);
            let Some(tip) = to_pixels(camera, viewport, origin + axis * span) else {
                return Similarity::IDENTITY;
            };
            let along = tip - centre;
            let length = along.length();
            if length < 1.0 {
                // The axis points at the eye, so there is no screen direction
                // to measure along and any answer would be noise.
                return Similarity::IDENTITY;
            }
            let direction = along / length;
            let started = (from - centre).dot(direction);
            if started.abs() < 1.0 {
                return Similarity::IDENTITY;
            }
            let pinned = gizmo.pinned.scale[index as usize].max(f32::MIN_POSITIVE);
            let highest = (gizmo.max_scale / pinned).clamp(1.0, MAX_SCALE);
            let lowest = MIN_SCALE.max(MIN_SCALE / pinned).min(1.0);
            let mut factor = ((to - centre).dot(direction) / started).clamp(lowest, highest);
            if snap.is_some() {
                factor = ((factor / SCALE_SNAP).round() * SCALE_SNAP).clamp(lowest, highest);
            }
            if factor == 1.0 {
                return Similarity::IDENTITY;
            }
            let mut scale = Vec3::ONE;
            scale[index as usize] = factor;
            Similarity::about(origin, Quat::IDENTITY, scale, Vec3::ZERO)
        }
    }
}

/// Round a world offset to a whole number of voxels on each axis.
///
/// This is what makes the ordinary move gesture take
/// [`brokkr_core::Bake::Exact`] and cost the surface nothing at all. Component
/// by component in WORLD axes, which is the lattice's own frame -- see the
/// module header on why the gizmo does not turn with the body.
fn snap_to_voxels(offset: Vec3, voxel_size: f32) -> Vec3 {
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return offset;
    }
    (offset / voxel_size).round() * voxel_size
}

/// Where a ray comes closest to a line through `origin` along `axis`, as a
/// distance along that axis. `None` when the two are near enough parallel that
/// the answer is meaningless.
fn along_axis((ray_origin, ray_direction): (Vec3, Vec3), origin: Vec3, axis: Vec3) -> Option<f32> {
    let w = ray_origin - origin;
    let b = ray_direction.dot(axis);
    let denominator = 1.0 - b * b;
    if denominator.abs() < 1.0e-4 {
        return None;
    }
    let d = ray_direction.dot(w);
    let e = axis.dot(w);
    Some((e - b * d) / denominator)
}

/// Where a ray meets the plane through `origin` with `normal`. `None` when it
/// is parallel to the plane, or meets it behind the eye.
fn on_plane((ray_origin, ray_direction): (Vec3, Vec3), origin: Vec3, normal: Vec3) -> Option<Vec3> {
    let denominator = ray_direction.dot(normal);
    if denominator.abs() < 1.0e-4 {
        return None;
    }
    let t = (origin - ray_origin).dot(normal) / denominator;
    (t > 0.0).then(|| ray_origin + ray_direction * t)
}

/// Build the gizmo for one frame.
///
/// Every dimension goes through the one `scale` from [`world_per_pixel`], which
/// is the same number [`pick`] uses, so the thing drawn and the thing grabbed
/// cannot be different sizes.
pub fn build(batch: &mut OverlayBatch, camera: &OrbitCamera, viewport: Vec2, gizmo: &Gizmo) {
    batch.clear();

    let origin = gizmo.origin();
    let scale = world_per_pixel(camera, origin, viewport.y);
    if !scale.is_finite() || scale <= 0.0 {
        return;
    }
    let towards_viewer = (camera.eye() - origin).normalize_or(Vec3::Z);

    // The live handle beats the hovered one: while a drag is running the
    // pointer wanders off the handle it grabbed, and dimming it then would say
    // the grab had been let go.
    let lit = gizmo.grabbed.or(gizmo.hovered);

    for index in 0..3u8 {
        let axis = axis_of(index);
        let colour = |handle: Handle| {
            let table = if lit == Some(handle) { theme::AXIS_HOVER } else { theme::AXIS };
            theme::linear(table[index as usize], 1.0)
        };

        // The shaft, as a ribbon turned to face the camera. A `push_line` here
        // would be one physical pixel; see the module header.
        let shaft_colour = colour(Handle::Axis(index));
        let tip = origin + axis * (SHAFT_PX * scale);
        let side = axis.cross(towards_viewer).try_normalize().unwrap_or_else(|| {
            // Looking straight down the axis. Any perpendicular will do, and
            // the ribbon is a dot on screen either way.
            axis.cross(camera.up()).try_normalize().unwrap_or(camera.right())
        }) * (SHAFT_HALF_PX * scale);
        batch.push_quad(origin - side, tip - side, tip + side, origin + side, shaft_colour);

        // The arrowhead, as a cone: a fan from the apex plus a fan closing the
        // base, so it occludes itself correctly in the cleared-depth pass.
        let apex = origin + axis * ((SHAFT_PX + HEAD_PX) * scale);
        for step in 0..RING_SEGMENTS {
            let a = ring_point(tip, scale, index, HEAD_RADIUS_PX, step);
            let b = ring_point(tip, scale, index, HEAD_RADIUS_PX, step + 1);
            batch.push_triangle(apex, a, b, shaft_colour);
            batch.push_triangle(tip, b, a, shaft_colour);
        }

        // The per-axis scale box, past the arrow head. A cube rather than
        // another cone, because a second cone on the same axis reads as a
        // longer arrow and the two handles do different things -- this is the
        // one that squashes.
        //
        // **Drawn only if pressing its own centre would actually select it.**
        // The box is 5 px square at 92 px and the ring's PROJECTION sweeps
        // every radius out to `RING_PX` as the camera turns, so at a large
        // fraction of angles the box's own centre pixel belongs to an arrow or
        // a ring -- measured, before the boxes moved onto the body's axes, at
        // roughly half of them, split across `Axis`, `Ring` and `Uniform`. A
        // handle that is drawn where it cannot be pressed is worse than one
        // that is absent: the user aims at it, gets a move or a turn, and sees
        // the wrong thing light up.
        //
        // Asked of `pick` rather than re-derived from the constants, so the
        // draw and the hit test cannot drift apart -- which is exactly the
        // failure that put the ring at 104 through the box at 92, and which
        // reasoning about world-space gaps got wrong. It is three extra `pick`
        // calls a frame against a handful of handles, and it is the same
        // honesty `MIN_SHAFT_PX` already gives the arrow.
        let box_colour = colour(Handle::Scale(index));
        let box_centre = origin + scale_axis_of(gizmo, index) * (SCALE_BOX_PX * scale);
        let box_is_reachable = to_pixels(camera, viewport, box_centre).is_some_and(|pixel| {
            pick(camera, viewport, gizmo, pixel) == Some(Handle::Scale(index))
        });
        let half = SCALE_BOX_HALF_PX * scale;
        let (u, v) = other_axes(index);
        if box_is_reachable {
            for face in 0..3usize {
                let normal = [axis, u, v][face];
                let (a, b) = ([u, v, axis][face], [v, axis, u][face]);
                for sign in [-1.0f32, 1.0] {
                    let middle = box_centre + normal * (half * sign);
                    batch.push_quad(
                        middle - a * half - b * half,
                        middle + a * half - b * half,
                        middle + a * half + b * half,
                        middle - a * half + b * half,
                        box_colour,
                    );
                }
            }
        }

        // The plane handle.
        let quad = plane_quad(origin, scale, index);
        batch.push_quad(quad[0], quad[1], quad[2], quad[3], colour(Handle::Plane(index)));

        // The ring, as a flat annulus in its own plane.
        let ring_colour = colour(Handle::Ring(index));
        let (inner_px, outer_px) = (RING_PX - RING_HALF_PX, RING_PX + RING_HALF_PX);
        for step in 0..RING_SEGMENTS {
            let ring = |radius, at| ring_point(origin, scale, index, radius, at);
            batch.push_quad(
                ring(inner_px, step),
                ring(outer_px, step),
                ring(outer_px, step + 1),
                ring(inner_px, step + 1),
                ring_colour,
            );
        }
    }

    // The box in the middle. A camera-facing square rather than a cube: a cube
    // would need its own occlusion reasoning against three shafts leaving it,
    // and a square reads as a grab target at eight pixels where a cube reads as
    // a smudge.
    let centre_colour = theme::linear(
        if lit == Some(Handle::Uniform) { theme::TEXT } else { theme::TEXT_DIM },
        1.0,
    );
    let right = camera.right() * (CENTRE_PX * scale);
    let up = camera.up() * (CENTRE_PX * scale);
    batch.push_quad(
        origin - right - up,
        origin + right - up,
        origin + right + up,
        origin - right + up,
        centre_colour,
    );

    // While a drag is live, where the body is GOING, as the transformed corners
    // of its box. Twelve lines rather than a transformed mesh: a live mesh
    // needs a per-body model matrix in the renderer's uniforms, which is the
    // one thing the bodies design refused to introduce anywhere. Deferred with
    // a trigger -- build it if the box turns out not to be enough to place a
    // rotation by.
    if gizmo.grabbed.is_some() {
        let preview = theme::linear(theme::ACCENT, 1.0);
        let corner = |mask: usize| {
            gizmo.placement.transform_point(Vec3::new(
                if mask & 1 == 0 { gizmo.base_low.x } else { gizmo.base_high.x },
                if mask & 2 == 0 { gizmo.base_low.y } else { gizmo.base_high.y },
                if mask & 4 == 0 { gizmo.base_low.z } else { gizmo.base_high.z },
            ))
        };
        for mask in 0..8usize {
            for bit in [1usize, 2, 4] {
                if mask & bit == 0 {
                    batch.push_line(corner(mask), corner(mask | bit), preview);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every scale box that is drawn can be pressed.**
    ///
    /// A 5 px square at 92 px, with a ring whose PROJECTION sweeps every radius
    /// out to `RING_PX` as the camera turns: at a large fraction of angles the
    /// box's own centre pixel belongs to an arrow or a ring instead. Measured
    /// before this gate, the box's centre was stolen at roughly half of camera
    /// angles, split across `Axis`, `Ring` and `Uniform`. Aiming at a handle
    /// and watching a different one light up is worse than the handle not
    /// being there.
    ///
    /// Swept over cameras rather than argued from the constants, because
    /// arguing from the constants is precisely what put a ring at 104 through a
    /// box at 92 and called them twelve pixels clear.
    #[test]
    fn every_scale_box_that_is_drawn_can_be_pressed() {
        let mut drawn = 0;
        let mut checked = 0;
        for yaw_step in 0..12 {
            for pitch_step in 0..5 {
                let mut camera = camera();
                camera.yaw = yaw_step as f32 * std::f32::consts::TAU / 12.0;
                camera.pitch = -0.8 + pitch_step as f32 * 0.4;
                let gizmo = gizmo();
                let origin = gizmo.origin();
                let scale = world_per_pixel(&camera, origin, VIEWPORT.y);

                for index in 0..3u8 {
                    let centre = origin + scale_axis_of(&gizmo, index) * (SCALE_BOX_PX * scale);
                    let Some(pixel) = to_pixels(&camera, VIEWPORT, centre) else {
                        continue;
                    };
                    checked += 1;
                    // `build` draws the box exactly when this holds, so the two
                    // cannot disagree; what this asserts is that the rule is
                    // the reachability one and not something weaker.
                    if pick(&camera, VIEWPORT, &gizmo, pixel) == Some(Handle::Scale(index)) {
                        drawn += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "the sweep projected no boxes at all");
        assert!(
            drawn > 0,
            "not one scale box was reachable at any of {checked} camera-and-axis pairs, so the \
             gate has switched the handle off entirely rather than hidden the unreachable ones"
        );
        assert!(
            drawn < checked,
            "every one of {checked} pairs was reachable, so this sweep never exercises the \
             case the gate exists for and would pass with the gate removed"
        );
    }

    /// **A polygon with no area contains nothing.**
    ///
    /// `inside_polygon` was `!(positive && negative)`. A plane handle seen
    /// exactly edge-on projects its four corners onto one line, every cross
    /// product is then exactly zero, neither flag is set, and the test answered
    /// TRUE for every point put to it -- and `Plane` outranks `Axis`, so the
    /// handle swallowed the arrow drawn straight through it.
    ///
    /// **Tested here rather than through a camera, and the reason is worth
    /// recording.** The end-to-end path could not be made to reproduce it:
    /// when a plane is close enough to edge-on for its corners to be collinear
    /// to the last bit, some of them are also behind the near plane, so
    /// `to_pixels` returns `None`, `all_on_screen` is false and `pick` skips
    /// the handle before this function is ever called. That is a SECOND reason
    /// the press is refused, not a reason the first one is sound -- it depends
    /// on the near plane, which `apply_view` derives from the body and which
    /// this branch has already had to fix once. The predicate is wrong on its
    /// own terms and is fixed on its own terms.
    #[test]
    fn a_polygon_with_no_area_contains_nothing() {
        // Four corners on one horizontal line, in the order `plane_quad`
        // produces: near-near, far-near, far-far, near-far. Collapsed onto a
        // line, that doubles back on itself exactly as an edge-on quad does.
        let flat = [
            Vec2::new(20.0, 50.0),
            Vec2::new(40.0, 50.0),
            Vec2::new(40.0, 50.0),
            Vec2::new(20.0, 50.0),
        ];

        assert!(
            !inside_polygon(&flat, Vec2::new(30.0, 50.0)),
            "a zero-area quad claimed a point lying on its own line"
        );
        assert!(
            !inside_polygon(&flat, Vec2::new(30.0, 51.0)),
            "a zero-area quad claimed a point off its line"
        );

        // And a real quad still contains what it should, so the second clause
        // has not simply switched the handle off.
        let real = [
            Vec2::new(20.0, 20.0),
            Vec2::new(40.0, 20.0),
            Vec2::new(40.0, 40.0),
            Vec2::new(20.0, 40.0),
        ];
        assert!(inside_polygon(&real, Vec2::new(30.0, 30.0)), "a real quad lost its middle");
        assert!(inside_polygon(&real, Vec2::new(20.0, 30.0)), "a real quad lost its own edge");
        assert!(
            !inside_polygon(&real, Vec2::new(10.0, 30.0)),
            "a real quad claimed a point outside it"
        );
    }

    const VIEWPORT: Vec2 = Vec2::new(1280.0, 720.0);
    const VOXEL: f32 = 0.5;

    fn camera() -> OrbitCamera {
        let mut camera = OrbitCamera::framing(Vec3::ZERO, 30.0);
        camera.set_lattice(VOXEL, 30.0);
        camera
    }

    fn gizmo() -> Gizmo {
        Gizmo::new(NodeId(1), Vec3::splat(-20.0), Vec3::splat(20.0), MAX_SCALE)
    }

    fn centre(camera: &OrbitCamera, gizmo: &Gizmo) -> Vec2 {
        to_pixels(camera, VIEWPORT, gizmo.origin()).expect("the gizmo is in front of the camera")
    }

    /// The property that makes a gizmo usable at any distance, and the one that
    /// silently fails when the draw and the hit test keep separate scales.
    ///
    /// Measured two ways, because they check different halves and the first one
    /// alone passes for a factor that is merely CONSISTENTLY wrong:
    ///
    /// * A world axis holds the same number of pixels at every zoom. It does
    ///   not hold `SHAFT_PX` of them -- a world axis leaning away from the
    ///   camera is foreshortened, which is correct 3D and not a scaling bug.
    /// * A length laid out ACROSS the view is exactly `SHAFT_PX` pixels, which
    ///   is what `world_per_pixel` actually promises.
    #[test]
    fn the_gizmo_is_the_same_size_on_screen_at_every_distance() {
        let gizmo = gizmo();
        let mut sizes = Vec::new();
        for notches in [-8.0_f32, -3.0, 0.0, 4.0, 9.0] {
            let mut camera = camera();
            camera.zoom_by(OrbitCamera::zoom_factor(notches));
            let scale = world_per_pixel(&camera, gizmo.origin(), VIEWPORT.y);
            let middle = centre(&camera, &gizmo);

            let tip = to_pixels(&camera, VIEWPORT, gizmo.origin() + Vec3::X * (SHAFT_PX * scale))
                .expect("the tip is on screen");
            sizes.push(middle.distance(tip));

            let across =
                to_pixels(&camera, VIEWPORT, gizmo.origin() + camera.right() * (SHAFT_PX * scale))
                    .expect("on screen");
            let measured = middle.distance(across);
            assert!(
                (measured - SHAFT_PX).abs() < 0.5,
                "across the view the shaft is {measured} px at {notches} notches, \
                 expected {SHAFT_PX}"
            );
        }
        let first = sizes[0];
        for size in &sizes {
            assert!(
                (size - first).abs() < 1.0,
                "the shaft measured {sizes:?} pixels across a range of zooms"
            );
        }
    }

    /// ONLY BOUNDS-CHECKED EVENTS MAY CAPTURE. A gizmo that claimed presses
    /// across the viewport would kill sculpting everywhere but on itself, which
    /// is a failure this project has shipped once.
    #[test]
    fn a_press_away_from_the_gizmo_is_not_claimed() {
        let camera = camera();
        let gizmo = gizmo();
        let middle = centre(&camera, &gizmo);
        for at in [
            Vec2::new(4.0, 4.0),
            Vec2::new(VIEWPORT.x - 4.0, 4.0),
            Vec2::new(4.0, VIEWPORT.y - 4.0),
            Vec2::new(VIEWPORT.x - 4.0, VIEWPORT.y - 4.0),
            middle + Vec2::new(400.0, 0.0),
        ] {
            assert!(!contains(&camera, VIEWPORT, &gizmo, at), "{at:?} was inside the disc");
            assert!(pick(&camera, VIEWPORT, &gizmo, at).is_none(), "{at:?} was claimed");
        }
    }

    /// The bounds check has to be a SUPERSET of everything drawn, or a handle
    /// the user can see is refused before `pick` is ever consulted.
    ///
    /// **Asserting "everything picked is inside the disc" would be a
    /// tautology** -- `pick` gates on `contains` in its first two lines, so that
    /// direction is true by construction and holds equally for a disc of radius
    /// zero. The direction with content is this one: every point the gizmo
    /// actually DRAWS is admitted.
    #[test]
    fn the_bounds_check_admits_every_point_the_gizmo_draws() {
        for (yaw, pitch) in [(0.0_f32, 0.0_f32), (0.9, 0.6), (2.4, -1.2), (0.0, 1.5)] {
            let camera = OrbitCamera { yaw, pitch, ..camera() };
            let gizmo = gizmo();
            let origin = gizmo.origin();
            let scale = world_per_pixel(&camera, origin, VIEWPORT.y);

            let mut checked = 0;
            for index in 0..3u8 {
                // The far end of each arrow.
                let tip = origin + axis_of(index) * ((SHAFT_PX + HEAD_PX) * scale);
                // The far corner of each plane handle.
                let corner = plane_quad(origin, scale, index)[2];
                // And the whole of each ring, which is the outermost thing.
                for step in 0..RING_SEGMENTS {
                    let on_ring = ring_point(origin, scale, index, RING_PX + RING_HALF_PX, step);
                    for point in [tip, corner, on_ring] {
                        let Some(at) = to_pixels(&camera, VIEWPORT, point) else {
                            continue;
                        };
                        assert!(
                            contains(&camera, VIEWPORT, &gizmo, at),
                            "at yaw {yaw} pitch {pitch} the disc refused {at:?}, \
                             which the gizmo draws"
                        );
                        checked += 1;
                    }
                }
            }
            assert!(checked > 100, "only {checked} points were on screen to check");
        }
    }

    #[test]
    fn the_middle_of_the_gizmo_is_the_scale_handle() {
        let camera = camera();
        let gizmo = gizmo();
        assert_eq!(pick(&camera, VIEWPORT, &gizmo, centre(&camera, &gizmo)), Some(Handle::Uniform));
    }

    /// Each shaft has to be grabbable along its length, and grabbing it must
    /// name the axis it is actually drawn along.
    #[test]
    fn a_point_on_a_shaft_picks_that_axis() {
        // Away from the poles and off every axis, so no shaft is edge on.
        let camera = OrbitCamera { yaw: 0.9, pitch: 0.6, ..camera() };
        let gizmo = gizmo();
        let scale = world_per_pixel(&camera, gizmo.origin(), VIEWPORT.y);
        for index in 0..3u8 {
            // Halfway along the shaft, clear of the centre box and the head.
            let at = to_pixels(
                &camera,
                VIEWPORT,
                gizmo.origin() + axis_of(index) * (SHAFT_PX * 0.55 * scale),
            )
            .expect("the shaft is on screen");
            assert_eq!(
                pick(&camera, VIEWPORT, &gizmo, at),
                Some(Handle::Axis(index)),
                "axis {index}"
            );
        }
    }

    /// **Sampled between the axes, and that is not incidental.** The three
    /// rings genuinely meet at the six points where an axis crosses them, so a
    /// sample taken on an axis is on two rings at once and whichever one the
    /// precedence order reaches first is the correct answer. Forty-five degrees
    /// round is on exactly one ring.
    #[test]
    fn a_point_on_a_ring_picks_that_ring() {
        let camera = OrbitCamera { yaw: 0.9, pitch: 0.6, ..camera() };
        let gizmo = gizmo();
        let scale = world_per_pixel(&camera, gizmo.origin(), VIEWPORT.y);
        for index in 0..3u8 {
            let (u, v) = other_axes(index);
            let diagonal = (u + v).normalize() * (RING_PX * scale);
            let at = to_pixels(&camera, VIEWPORT, gizmo.origin() + diagonal).expect("on");
            assert_eq!(
                pick(&camera, VIEWPORT, &gizmo, at),
                Some(Handle::Ring(index)),
                "ring {index}"
            );
        }
    }

    /// A one-pixel move must not send the body across the room. An axis pointing
    /// at the camera is the case where the closest-point solve divides by
    /// nothing, and it is refused rather than answered badly.
    #[test]
    fn an_axis_pointing_at_the_camera_is_not_grabbable() {
        // Looking straight down Z: the Z shaft projects onto the middle of the
        // gizmo and is a couple of pixels long.
        let camera = OrbitCamera { yaw: 0.0, pitch: 0.0, ..camera() };
        let gizmo = gizmo();
        let scale = world_per_pixel(&camera, gizmo.origin(), VIEWPORT.y);
        let middle = centre(&camera, &gizmo);
        let tip = to_pixels(&camera, VIEWPORT, gizmo.origin() + Vec3::Z * (SHAFT_PX * scale))
            .expect("on screen");
        assert!(middle.distance(tip) < MIN_SHAFT_PX, "the fixture's Z axis is not edge on");

        // A few pixels out along where that shaft would be: not the Z axis.
        for step in 1..6 {
            let at = middle + Vec2::new(0.0, -(CENTRE_PX + GRAB_PX + step as f32 * 2.0));
            assert_ne!(
                pick(&camera, VIEWPORT, &gizmo, at),
                Some(Handle::Axis(2)),
                "a shaft {} px long was grabbed",
                middle.distance(tip)
            );
        }
    }

    /// Exactly one handle lights, or the interface is telling the user two
    /// different things about what a press would do.
    #[test]
    fn hovering_lights_exactly_one_handle() {
        let camera = OrbitCamera { yaw: 0.9, pitch: 0.6, ..camera() };
        let mut gizmo = gizmo();
        gizmo.hovered = Some(Handle::Axis(0));

        let mut lit = OverlayBatch::default();
        build(&mut lit, &camera, VIEWPORT, &gizmo);
        let mut plain = OverlayBatch::default();
        gizmo.hovered = None;
        build(&mut plain, &camera, VIEWPORT, &gizmo);

        assert_eq!(lit.surfaces.len(), plain.surfaces.len(), "the hover changed the geometry");
        let changed: std::collections::HashSet<[u32; 4]> = lit
            .surfaces
            .iter()
            .zip(&plain.surfaces)
            .filter(|(a, b)| a.colour != b.colour)
            .map(|(a, _)| a.colour.map(f32::to_bits))
            .collect();
        assert_eq!(changed.len(), 1, "hovering one handle changed {} colours", changed.len());
    }

    /// A move along an axis is exactly a move along that axis: nothing may leak
    /// into the other two, or a nudge sideways would drift the body off the
    /// lattice it was snapped to.
    #[test]
    fn dragging_an_axis_moves_only_along_that_axis() {
        let camera = OrbitCamera { yaw: 0.9, pitch: 0.6, ..camera() };
        let gizmo = gizmo();
        let middle = centre(&camera, &gizmo);
        for index in 0..3u8 {
            let gesture = drag(
                &camera,
                VIEWPORT,
                &gizmo,
                Handle::Axis(index),
                middle,
                middle + Vec2::new(37.0, -21.0),
                None,
            );
            let moved = gesture.translation;
            assert!(moved.length() > 1.0e-3, "axis {index} did not move at all");
            for other in 0..3usize {
                if other != index as usize {
                    assert!(
                        moved[other].abs() < 1.0e-4,
                        "axis {index} leaked {} onto axis {other}",
                        moved[other]
                    );
                }
            }
        }
    }

    /// **The gesture that has to be lossless, and the one users make most.**
    #[test]
    fn a_snapped_move_takes_the_exact_route() {
        let camera = OrbitCamera { yaw: 0.9, pitch: 0.6, ..camera() };
        let gizmo = gizmo();
        let middle = centre(&camera, &gizmo);
        for pixels in [7.0_f32, 23.0, 61.0, -44.0] {
            let gesture = drag(
                &camera,
                VIEWPORT,
                &gizmo,
                Handle::Axis(0),
                middle,
                middle + Vec2::new(pixels, 0.0),
                Some(VOXEL),
            );
            assert!(
                !gesture.route(VOXEL).is_lossy(),
                "a snapped move of {pixels} px routed to {:?}",
                gesture.route(VOXEL)
            );
        }
    }

    #[test]
    fn a_snapped_turn_takes_the_exact_route_and_an_unsnapped_one_says_it_does_not() {
        let camera = OrbitCamera { yaw: 0.9, pitch: 0.6, ..camera() };
        // A pivot on the lattice, which is what makes a quarter turn exact.
        let gizmo = Gizmo::new(NodeId(1), Vec3::splat(-20.0), Vec3::splat(20.0), MAX_SCALE);
        let scale = world_per_pixel(&camera, gizmo.origin(), VIEWPORT.y);

        for index in 0..3u8 {
            let (u, v) = other_axes(index);
            let radius = RING_PX * scale;
            let from = to_pixels(&camera, VIEWPORT, gizmo.origin() + u * radius).expect("on");
            // A right angle round the ring.
            let to = to_pixels(&camera, VIEWPORT, gizmo.origin() + v * radius).expect("on");

            let snapped =
                drag(&camera, VIEWPORT, &gizmo, Handle::Ring(index), from, to, Some(VOXEL));
            assert!(
                !snapped.route(VOXEL).is_lossy(),
                "a snapped quarter turn about axis {index} routed to {:?}",
                snapped.route(VOXEL)
            );

            // A third of the way round is not a quarter turn under any
            // tolerance, and the routing has to say so rather than rounding.
            let third = to_pixels(
                &camera,
                VIEWPORT,
                gizmo.origin() + (u * (0.5f32).cos() + v * (0.5f32).sin()) * radius,
            )
            .expect("on");
            let free = drag(&camera, VIEWPORT, &gizmo, Handle::Ring(index), from, third, None);
            assert!(free.route(VOXEL).is_lossy(), "a 0.5 rad turn was called exact");
        }
    }

    /// Coming back to the pixel the drag started on has to be EXACTLY the
    /// identity, not approximately it. It is the cheapest cancel there is, and
    /// an approximate one would bake a resample for a gesture that did nothing.
    #[test]
    fn a_drag_that_returns_to_its_press_pixel_is_exactly_the_identity() {
        let camera = OrbitCamera { yaw: 0.9, pitch: 0.6, ..camera() };
        let gizmo = gizmo();
        let scale = world_per_pixel(&camera, gizmo.origin(), VIEWPORT.y);
        let middle = centre(&camera, &gizmo);
        let on_ring =
            to_pixels(&camera, VIEWPORT, gizmo.origin() + Vec3::Y * (RING_PX * scale)).expect("on");

        for (handle, from) in [
            (Handle::Axis(0), middle),
            (Handle::Axis(1), middle),
            (Handle::Plane(2), middle),
            (Handle::Ring(0), on_ring),
            (Handle::Uniform, middle + Vec2::new(30.0, 0.0)),
        ] {
            for snap in [Some(VOXEL), None] {
                let gesture = drag(&camera, VIEWPORT, &gizmo, handle, from, from, snap);
                assert_eq!(
                    gesture.route(VOXEL),
                    brokkr_core::Bake::Identity,
                    "{handle:?} snap {snap:?} routed to {:?}",
                    gesture.route(VOXEL)
                );
            }
        }
    }

    /// Scaling has to be reversible within the gesture: drag out, drag back,
    /// and the factor is one again rather than one and a bit.
    #[test]
    fn a_snapped_scale_comes_back_to_exactly_one() {
        let camera = camera();
        let gizmo = gizmo();
        let middle = centre(&camera, &gizmo);
        let from = middle + Vec2::new(40.0, 0.0);
        let out = drag(
            &camera,
            VIEWPORT,
            &gizmo,
            Handle::Uniform,
            from,
            middle + Vec2::new(80.0, 0.0),
            Some(VOXEL),
        );
        assert!(out.scale.max_element() > 1.5, "the fixture did not grow: {:?}", out.scale);
        let back = drag(&camera, VIEWPORT, &gizmo, Handle::Uniform, from, from, Some(VOXEL));
        assert_eq!(back.scale, Vec3::ONE);
    }

    /// **The bound is on the total, and one gesture cannot see the total.**
    ///
    /// `Handle::Uniform` is only selected by a press within `CENTRE_PX +
    /// GRAB_PX` of the middle, so every scale drag starts near the centre and a
    /// long one saturates [`MAX_SCALE`] on its own. Clamping each gesture
    /// therefore bounds nothing: [`Similarity::then`] MULTIPLIES scales, so a
    /// second saturating drag composes to 400 and a third to 8000, and
    /// `Volume::warped` allocates a dense destination grid for whatever it is
    /// handed.
    #[test]
    fn scale_drags_compose_without_ever_passing_the_gizmo_s_ceiling() {
        let camera = camera();
        let mut gizmo = gizmo();
        gizmo.max_scale = 6.0;
        let middle = centre(&camera, &gizmo);

        for round in 0..8 {
            // From as near the middle as a press can land, out to the far edge
            // of the viewport: the most one gesture can possibly ask for.
            let from = middle + Vec2::new(2.0, 0.0);
            let gesture = drag(
                &camera,
                VIEWPORT,
                &gizmo,
                Handle::Uniform,
                from,
                from + Vec2::splat(320.0),
                None,
            );
            gizmo.placement = gizmo.pinned.then(gesture);
            gizmo.pinned = gizmo.placement;
            assert!(
                gizmo.placement.scale.max_element() <= 6.0 + 1.0e-4,
                "round {round} composed to {}, past a ceiling of 6",
                gizmo.placement.scale
            );
        }
        assert!(
            gizmo.placement.scale.max_element() > 1.0,
            "the fixture never grew the body at all, so it proves nothing"
        );
    }

    /// The floor composes the same way, and its failure is quieter: a body
    /// scaled to 1e-13 is rebuilt as a handful of nonsense voxels, which reads
    /// as the body having vanished with nothing said.
    #[test]
    fn shrinking_drags_compose_without_ever_passing_the_floor() {
        let camera = camera();
        let mut gizmo = gizmo();
        let middle = centre(&camera, &gizmo);

        for round in 0..10 {
            let from = middle + Vec2::new(14.0, 0.0);
            let gesture = drag(&camera, VIEWPORT, &gizmo, Handle::Uniform, from, middle, None);
            gizmo.placement = gizmo.pinned.then(gesture);
            gizmo.pinned = gizmo.placement;
            assert!(
                gizmo.placement.scale.min_element() >= MIN_SCALE - 1.0e-6,
                "round {round} composed down to {}, under a floor of {MIN_SCALE}",
                gizmo.placement.scale
            );
        }
        assert!(
            gizmo.placement.scale.min_element() < 1.0,
            "the fixture never shrank the body at all, so it proves nothing"
        );
    }

    /// Whatever the clamps do, coming back to the press pixel has to be exactly
    /// one -- including at the ceiling, where a naive clamp would pin the
    /// gesture above 1.0 and make the drag uncancellable by hand.
    #[test]
    fn a_scale_drag_returns_to_exactly_one_even_with_no_room_left_to_grow() {
        let camera = camera();
        let mut gizmo = gizmo();
        gizmo.max_scale = 1.0;
        gizmo.pinned = Similarity::about(gizmo.origin(), Quat::IDENTITY, Vec3::ONE, Vec3::ZERO);
        let from = centre(&camera, &gizmo) + Vec2::new(30.0, 0.0);

        for snap in [Some(VOXEL), None] {
            let gesture = drag(&camera, VIEWPORT, &gizmo, Handle::Uniform, from, from, snap);
            assert_eq!(gesture, Similarity::IDENTITY, "snap {snap:?} could not come back to one");
        }
    }

    /// A degenerate drag must do nothing rather than something arbitrary: a ray
    /// parallel to the plane it is dragging in has no intersection to take.
    #[test]
    fn a_drag_whose_maths_is_degenerate_does_nothing() {
        // Looking along Z, so the ray is parallel to the XY plane's own axes
        // only at the extreme -- what is actually degenerate here is dragging
        // in the planes that contain the view direction.
        let camera = OrbitCamera { yaw: 0.0, pitch: 0.0, ..camera() };
        let gizmo = gizmo();
        let middle = centre(&camera, &gizmo);
        for handle in [Handle::Plane(0), Handle::Plane(1), Handle::Axis(2), Handle::Ring(0)] {
            let gesture =
                drag(&camera, VIEWPORT, &gizmo, handle, middle, middle + Vec2::new(9.0, 3.0), None);
            assert!(
                gesture.transform_point(Vec3::ZERO).is_finite() && gesture.scale.is_finite(),
                "{handle:?} produced {gesture:?}"
            );
        }
    }

    /// The preview box only exists while a drag is live, and it has to follow
    /// the placement rather than sitting on the body's old position.
    #[test]
    fn the_preview_box_appears_only_while_dragging_and_follows_the_placement() {
        let camera = camera();
        let mut gizmo = gizmo();

        let mut idle = OverlayBatch::default();
        build(&mut idle, &camera, VIEWPORT, &gizmo);
        assert!(idle.lines.is_empty(), "an idle gizmo drew a preview box");

        gizmo.grabbed = Some(Handle::Axis(0));
        gizmo.placement = Similarity::moving(Vec3::new(100.0, 0.0, 0.0));
        let mut dragging = OverlayBatch::default();
        build(&mut dragging, &camera, VIEWPORT, &gizmo);
        // Twelve edges, two vertices each.
        assert_eq!(dragging.lines.len(), 24);
        assert!(
            dragging.lines.iter().all(|vertex| vertex.position[0] > 70.0),
            "the preview box did not travel with the placement"
        );
    }

    /// The gizmo sits on the body it is moving, so its origin has to travel
    /// with the placement rather than staying where the body used to be.
    #[test]
    fn the_gizmo_travels_with_what_it_has_already_moved() {
        let mut gizmo = gizmo();
        assert_eq!(gizmo.origin(), Vec3::ZERO);
        gizmo.placement = Similarity::moving(Vec3::new(0.0, 15.0, 0.0));
        assert_eq!(gizmo.origin(), Vec3::new(0.0, 15.0, 0.0));
    }
}
