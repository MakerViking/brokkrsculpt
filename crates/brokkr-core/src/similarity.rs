// SPDX-License-Identifier: AGPL-3.0-only

//! Rigid placement of a body: turn it, scale it, move it.
//!
//! [`Similarity`] is a rotation, ONE uniform scale and a translation. That is
//! not an arbitrary subset of the affine group, it is **the largest group a
//! signed distance field is closed under**: for a similarity `T` with scale
//! `s`,
//!
//! ```text
//! d'(p) = s * d(T_inverse(p))
//! ```
//!
//! is EXACTLY the signed distance field of the transformed solid, not an
//! approximation of it. Every step of the engine downstream -- the sphere
//! trace in [`crate::raycast`], the surface nets in [`crate::mesh`], the
//! curvature the brushes read -- assumes it is looking at a distance field, and
//! a map outside this group hands them something that merely has the right zero
//! set.
//!
//! # Why per-axis scale is absent from the type rather than refused by a check
//!
//! Squashing one axis is the one ZBrush gizmo affordance this does not offer,
//! and it is worth being precise about why, because the usual reason given is
//! wrong. It is not that a squashed field is inexpressible: `s_min * d(T'(p))`
//! has the correct zero set and *underestimates* the true distance, which is
//! the sound direction for a sphere trace -- it steps short, never through.
//!
//! What is true is that the gradient's length drifts into
//! `[s_min/s_max, 1]`, and **the drift compounds across bakes with nothing to
//! reset it**. [`crate::generate`] clamps at a gradient floor of 0.5 because it
//! already has to cope with the brush's own eikonal residual; a per-axis scale
//! would push past that with no ceiling. The named trigger for revisiting this
//! is a redistancing pass, which would reset the drift and fix the brush's
//! residual at the same time -- one piece of work, two customers.
//!
//! Leaving the axis vector out of the struct means no future caller can smuggle
//! one in through a field that happens to be public.
//!
//! # The routing is the point
//!
//! [`Similarity::route`] is the whole exact-versus-lossy decision, in one
//! place, so that the status line, the bake and the undo entry cannot disagree
//! about which one ran. A whole-voxel move and a quarter turn have exact
//! implementations already ([`crate::Volume::shifted`] and
//! [`crate::Volume::rotated`]); everything else is one trilinear pass and the
//! interface has to say so.

use glam::{IVec3, Quat, Vec3};

use crate::orientation::AxisRotation;

/// How near a quarter turn a rotation has to be to take the exact route.
///
/// Tight on purpose. The gizmo snaps by default, so a snapped gesture arrives
/// with components that are exactly 0 or ±1 and clears this by ten orders of
/// magnitude; a free-angle drag that happens to pass near 90 degrees must NOT
/// be silently snapped onto the lattice, because the user watching the model
/// would see it jump. See `a_rotation_just_off_a_quarter_turn_is_not_called_exact`.
const AXIS_EPSILON: f32 = 1.0e-4;

/// How near a whole voxel a translation has to be to take the exact route,
/// as a fraction of one voxel.
const VOXEL_EPSILON: f32 = 1.0e-3;

/// How near one a scale has to be to take the exact route.
const SCALE_EPSILON: f32 = 1.0e-5;

/// A rotation, one uniform scale and a translation.
///
/// Applied to a point as `rotation * (scale * point) + translation`. See the
/// module documentation for why this is exactly the set of maps a distance
/// field survives, and why per-axis scale is not in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity {
    pub rotation: Quat,
    pub scale: Vec3,
    pub translation: Vec3,
}

impl Default for Similarity {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// The cheapest route that reproduces a [`Similarity`] on the voxel lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bake {
    /// Nothing to do. Not merely cheap -- it is what makes dragging a handle
    /// in a circle and letting go cost the field nothing at all.
    Identity,
    /// Bit-exact: [`crate::Volume::rotated`] followed by a whole-VOXEL shift.
    ///
    /// The two together, and in that order. A quarter turn about the lattice
    /// origin is almost never the turn the user asked for -- theirs is about
    /// the body's centre -- and the shift is what carries it there.
    Exact { turns: AxisRotation, voxel_offset: IVec3 },
    /// One trilinear pass. Lossy, and every interface that reports it says so.
    Resample,
}

impl Bake {
    /// Whether taking this route costs the surface anything.
    ///
    /// The one question the status line, the armed badge and the undo entry all
    /// ask, so it is answered here rather than by three matches that could
    /// drift apart.
    pub fn is_lossy(self) -> bool {
        matches!(self, Bake::Resample)
    }
}

impl Similarity {
    pub const IDENTITY: Self =
        Self { rotation: Quat::IDENTITY, scale: Vec3::ONE, translation: Vec3::ZERO };

    /// A placement about a pivot: turn and scale about `pivot`, then move by
    /// `offset`.
    ///
    /// This is the form every gizmo handle produces, because a handle turns the
    /// body about the gizmo's own origin and not about the world's.
    pub fn about(pivot: Vec3, rotation: Quat, scale: Vec3, offset: Vec3) -> Self {
        // p |-> pivot + offset + rotation * (scale * (p - pivot))
        let translation = pivot + offset - rotation * (scale * pivot);
        Self { rotation, scale, translation }
    }

    /// A pure translation.
    pub fn moving(offset: Vec3) -> Self {
        Self { rotation: Quat::IDENTITY, scale: Vec3::ONE, translation: offset }
    }

    pub fn transform_point(self, point: Vec3) -> Vec3 {
        self.rotation * (self.scale * point) + self.translation
    }

    /// This map followed by `next`.
    ///
    /// `a.then(b)` is `b(a(p))`, which is the order it reads in rather than the
    /// order matrix notation would write it. What needs it is a gizmo drag:
    /// the gesture is measured about the placement as it stood when the button
    /// went down, so the total is that pinned placement THEN the gesture.
    pub fn then(self, next: Self) -> Self {
        Self {
            rotation: next.rotation * self.rotation,
            scale: next.scale * self.scale,
            translation: next.transform_point(self.translation),
        }
    }

    /// The map that undoes this one.
    ///
    /// Exact for the rotation and the translation and a reciprocal for the
    /// scale, so composing the two is the identity to within one float
    /// division. This is the direction the bake actually walks: a destination
    /// voxel asks where it came FROM.
    /// Where a point came FROM, which is the direction the bake walks.
    ///
    /// **This replaced an `inverse()` that returned another `Similarity`, and
    /// it had to.** With a per-axis scale the inverse of `R S p + t` is
    /// `S_inv R_inv (p - t)` -- scale on the OUTSIDE of the rotation -- and
    /// that is not expressible in this struct's `R S p + t` shape at all. For a
    /// uniform scale the two commute and it was, which is why the old form
    /// worked right up until the day the scale became a vector. Returning the
    /// mapped point instead sidesteps the representation entirely and is exact.
    pub fn inverse_transform_point(self, point: Vec3) -> Vec3 {
        (self.rotation.inverse() * (point - self.translation)) / self.scale
    }

    /// The smallest of the three scale factors.
    ///
    /// **What a distance has to be multiplied by after a per-axis scale.**
    /// `s_min * d(T_inverse(p))` keeps the zero set exactly and UNDERESTIMATES
    /// the true distance everywhere else, which is the sound direction for a
    /// sphere trace: it steps short, never through. The overestimate would put
    /// the ray through the surface.
    pub fn min_scale(self) -> f32 {
        self.scale.x.abs().min(self.scale.y.abs()).min(self.scale.z.abs())
    }

    /// Whether all three axes scale alike, so the field stays a true distance
    /// field and [`crate::Volume::redistance`] has nothing to do.
    pub fn is_uniform_scale(self) -> bool {
        let s = self.scale;
        (s.x - s.y).abs() <= SCALE_EPSILON && (s.y - s.z).abs() <= SCALE_EPSILON
    }

    /// Whether this map is close enough to the identity to be worth nothing.
    ///
    /// The tolerance is the ROUTING tolerance and not a display one: a map that
    /// says it is the identity here is one the bake will decline to run, so it
    /// has to be a map that would have moved no lattice point anywhere.
    pub fn is_identity(self, voxel_size: f32) -> bool {
        matches!(self.route(voxel_size), Bake::Identity)
    }

    /// Whether two placements would bake to the same field.
    ///
    /// **Not `==`.** A gizmo drag recomputes its placement through
    /// [`Similarity::then`] on every pointer event, and that arithmetic is not
    /// bit-reproducible: a press and release on one pixel gives a translation
    /// and a scale that are exactly equal and a rotation that differs in the
    /// last bits. An exact comparison therefore never fires, and the caller
    /// that wanted to skip a re-bake pays for one anyway.
    ///
    /// Expressed in the same epsilons [`Similarity::route`] uses, and that is
    /// the point of it living here: the question is "would baking this again
    /// produce the field we already have", which is the routing question, and
    /// an app-side copy with its own tolerances can silently stop agreeing with
    /// the router it exists to short-circuit. They are far below anything a
    /// real gesture produces -- one pixel of drag moves the translation by a
    /// whole `world_per_pixel` -- so this cannot swallow a deliberate nudge.
    pub fn same_bake(self, other: Self, voxel_size: f32) -> bool {
        let voxel = if voxel_size.is_finite() && voxel_size > 0.0 { voxel_size } else { 1.0 };
        (self.translation - other.translation).length() <= voxel * VOXEL_EPSILON
            && (self.scale - other.scale).abs().max_element() <= SCALE_EPSILON
            && self.rotation.dot(other.rotation).abs() >= 1.0 - AXIS_EPSILON
    }

    /// A conservative source box for a destination box: the eight corners
    /// through the inverse.
    ///
    /// **A superset, and that is the sound direction.** It can only make
    /// [`crate::Volume::coverage`] answer `Surface` where `Empty` would have
    /// done, which costs a brick's worth of sampling; a box that was too TIGHT
    /// would drop geometry, silently, and look like a mesher bug rather than a
    /// transform one.
    ///
    /// Eight corners rather than an interval-arithmetic bound because a
    /// similarity is affine: the image of a box under an affine map is a
    /// parallelepiped whose vertices are the images of the box's vertices, so
    /// their bounding box is exact for the parallelepiped and loose only by
    /// the amount the parallelepiped is not axis aligned.
    pub fn inverse_bounds(self, low: Vec3, high: Vec3) -> (Vec3, Vec3) {
        let mut result_low = Vec3::splat(f32::INFINITY);
        let mut result_high = Vec3::splat(f32::NEG_INFINITY);
        for corner in 0..8 {
            let point = Vec3::new(
                if corner & 1 == 0 { low.x } else { high.x },
                if corner & 2 == 0 { low.y } else { high.y },
                if corner & 4 == 0 { low.z } else { high.z },
            );
            let source = self.inverse_transform_point(point);
            result_low = result_low.min(source);
            result_high = result_high.max(source);
        }
        (result_low, result_high)
    }

    /// The cheapest route that reproduces this map on a lattice of
    /// `voxel_size`.
    ///
    /// # The half-voxel trap, which is the whole difficulty here
    ///
    /// [`AxisRotation::apply_voxel`] does not turn about the lattice origin. A
    /// voxel index labels the cell spanning `i..i+1`, so a negated axis sends
    /// `i` to `-i - 1` -- see that function, where the reason is worked
    /// through -- and the turn is therefore about the point half a voxel below
    /// the origin on each negated axis. A gizmo whose ring is drawn on the
    /// body's centroid and whose bake assumed a turn about the origin would
    /// throw the body a whole model-width across the viewport.
    ///
    /// The compensation falls out of asking the rotation itself where the
    /// origin goes. Writing `R` for the quarter turn's linear part and `n` for
    /// its half-voxel bias, `apply_voxel(v) = R v - n`, so
    /// `n = -apply_voxel(0)`; no second derivation of the bias exists to drift
    /// out of step with the first. In world units the exact route therefore
    /// maps `p` to `R p - h n + h k` for a shift of `k` voxels, and matching
    /// that against this map's own translation `t` gives
    ///
    /// ```text
    /// k = t / h + n
    /// ```
    ///
    /// which is the exact route exactly when `k` is a whole number of voxels.
    pub fn route(self, voxel_size: f32) -> Bake {
        if !voxel_size.is_finite() || voxel_size <= 0.0 {
            return Bake::Resample;
        }
        if (self.scale - Vec3::ONE).abs().max_element() > SCALE_EPSILON {
            return Bake::Resample;
        }
        let Some(turns) = quarter_turn(self.rotation) else {
            return Bake::Resample;
        };

        // Where `Volume::rotated` puts the lattice origin, in voxels.
        let bias = -turns.apply_voxel(IVec3::ZERO);
        let wanted = self.translation / voxel_size + bias.as_vec3();
        let rounded = wanted.round();
        if (wanted - rounded).abs().max_element() > VOXEL_EPSILON {
            return Bake::Resample;
        }

        let voxel_offset = rounded.as_ivec3();
        if turns.is_identity() && voxel_offset == IVec3::ZERO {
            return Bake::Identity;
        }
        Bake::Exact { turns, voxel_offset }
    }
}

/// The [`AxisRotation`] a quaternion names, when it names one at all.
///
/// Read off the images of the three basis vectors rather than solved for: a
/// quarter turn sends each axis to a signed axis, and anything that does not is
/// not one. That also makes the tolerance mean something a reader can check --
/// it is how far a basis vector may land from an axis -- rather than being an
/// angle buried in a decomposition.
fn quarter_turn(rotation: Quat) -> Option<AxisRotation> {
    let mut columns = [IVec3::ZERO; 3];
    for (index, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
        let image = rotation * axis;
        let mut column = IVec3::ZERO;
        for component in 0..3 {
            let value = image[component];
            if value.abs() > 1.0 - AXIS_EPSILON {
                column[component] = value.signum() as i32;
            } else if value.abs() > AXIS_EPSILON {
                // Between an axis and nothing: not a quarter turn.
                return None;
            }
        }
        columns[index] = column;
    }
    // `from_columns` re-checks that this is a signed permutation of
    // determinant +1. A quaternion cannot express a reflection, so the check
    // can only fail if the reading above went wrong -- which is exactly when a
    // caller most wants it to.
    AxisRotation::from_columns(columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orientation::Facing;
    use std::f32::consts::FRAC_PI_2;

    const VOXEL: f32 = 0.5;

    #[test]
    fn the_identity_transforms_nothing_and_routes_to_nothing() {
        let point = Vec3::new(3.0, -7.0, 11.0);
        assert_eq!(Similarity::IDENTITY.transform_point(point), point);
        assert_eq!(Similarity::IDENTITY.route(VOXEL), Bake::Identity);
        assert!(!Bake::Identity.is_lossy());
    }

    #[test]
    fn a_placement_about_a_pivot_leaves_the_pivot_where_it_was() {
        let pivot = Vec3::new(12.0, -3.0, 4.5);
        for (rotation, scale) in [
            (Quat::from_rotation_y(0.7), Vec3::ONE),
            (Quat::IDENTITY, Vec3::splat(2.5)),
            (Quat::from_rotation_x(-1.3), Vec3::splat(0.4)),
            // A per-axis scale, which the type could not hold until the
            // redistancing pass made one sound.
            (Quat::from_rotation_z(0.3), Vec3::new(1.0, 0.5, 2.0)),
        ] {
            let placement = Similarity::about(pivot, rotation, scale, Vec3::ZERO);
            assert!(
                placement.transform_point(pivot).distance(pivot) < 1.0e-4,
                "the pivot moved under rotation {rotation:?} scale {scale:?}"
            );
        }
    }

    #[test]
    fn a_placement_and_its_inverse_come_home() {
        let placement = Similarity::about(
            Vec3::new(-2.0, 5.0, 1.0),
            Quat::from_euler(glam::EulerRot::YXZ, 0.9, -0.4, 2.1),
            Vec3::splat(1.7),
            Vec3::new(4.0, 0.5, -9.0),
        );
        for point in [Vec3::ZERO, Vec3::new(30.0, -12.0, 7.0), Vec3::splat(-100.0)] {
            let round_trip = placement.inverse_transform_point(placement.transform_point(point));
            assert!(round_trip.distance(point) < 1.0e-3, "{point:?} came back as {round_trip:?}");
        }
    }

    /// A destination box's source box has to CONTAIN every source point, or the
    /// bake asks `coverage` about a region that misses geometry and drops it.
    #[test]
    fn the_inverse_bounds_contain_every_corner_and_then_some() {
        let placement = Similarity::about(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_euler(glam::EulerRot::YXZ, 0.6, 1.1, -0.3),
            Vec3::splat(0.8),
            Vec3::new(-4.0, 2.0, 6.0),
        );
        let (low, high) = (Vec3::new(-5.0, -6.0, -7.0), Vec3::new(9.0, 4.0, 3.0));
        let (source_low, source_high) = placement.inverse_bounds(low, high);

        // Every corner, which is what the bound is built from...
        for corner in 0..8 {
            let point = Vec3::new(
                if corner & 1 == 0 { low.x } else { high.x },
                if corner & 2 == 0 { low.y } else { high.y },
                if corner & 4 == 0 { low.z } else { high.z },
            );
            let source = placement.inverse_transform_point(point);
            assert!(source.cmpge(source_low).all() && source.cmple(source_high).all());
        }
        // ...and a scatter of interior points, which is the claim that actually
        // matters and which only holds because the map is affine.
        for step in 0..40 {
            let t = step as f32 / 39.0;
            let point = low.lerp(high, t) + Vec3::new(t, 1.0 - t, t * t) * (high - low) * 0.25;
            let point = point.clamp(low, high);
            let source = placement.inverse_transform_point(point);
            assert!(
                source.cmpge(source_low - Vec3::splat(1.0e-4)).all()
                    && source.cmple(source_high + Vec3::splat(1.0e-4)).all(),
                "{source:?} escaped [{source_low:?}, {source_high:?}]"
            );
        }
    }

    /// Composition has to agree with applying the two maps in turn, or a gizmo
    /// drag lands somewhere other than where its preview said it would.
    #[test]
    fn composing_two_placements_is_applying_them_one_after_the_other() {
        let first = Similarity::about(
            Vec3::new(2.0, -1.0, 4.0),
            Quat::from_rotation_y(0.7),
            Vec3::splat(1.4),
            Vec3::new(3.0, 0.0, -2.0),
        );
        let second = Similarity::about(
            Vec3::new(-5.0, 3.0, 0.0),
            Quat::from_rotation_x(-1.1),
            Vec3::splat(0.6),
            Vec3::new(0.0, 8.0, 1.0),
        );
        let composed = first.then(second);
        for point in [Vec3::ZERO, Vec3::new(11.0, -4.0, 7.0), Vec3::splat(-30.0)] {
            let stepwise = second.transform_point(first.transform_point(point));
            assert!(
                composed.transform_point(point).distance(stepwise) < 1.0e-3,
                "{point:?}: {:?} against {stepwise:?}",
                composed.transform_point(point)
            );
        }
    }

    #[test]
    fn composing_with_the_identity_changes_nothing() {
        let placement =
            Similarity::about(Vec3::ONE, Quat::from_rotation_z(0.3), Vec3::splat(2.0), Vec3::X);
        for combined in [placement.then(Similarity::IDENTITY), Similarity::IDENTITY.then(placement)]
        {
            assert!((combined.scale - placement.scale).abs().max_element() < 1.0e-6);
            assert!(combined.translation.distance(placement.translation) < 1.0e-5);
        }
    }

    #[test]
    fn a_whole_voxel_move_takes_the_exact_route() {
        let placement = Similarity::moving(Vec3::new(3.0 * VOXEL, -8.0 * VOXEL, 40.0 * VOXEL));
        match placement.route(VOXEL) {
            Bake::Exact { turns, voxel_offset } => {
                assert!(turns.is_identity(), "a pure move should not turn anything");
                assert_eq!(voxel_offset, IVec3::new(3, -8, 40));
            }
            other => panic!("a whole-voxel move routed to {other:?}"),
        }
    }

    #[test]
    fn a_half_voxel_move_is_honestly_a_resample() {
        let placement = Similarity::moving(Vec3::new(3.5 * VOXEL, 0.0, 0.0));
        assert_eq!(placement.route(VOXEL), Bake::Resample);
        assert!(placement.route(VOXEL).is_lossy());
    }

    /// The half-voxel trap. A quarter turn about the body's centre is the
    /// ordinary gesture, and the bias in [`AxisRotation::apply_voxel`] means
    /// the shift that carries it there is NOT simply the translation in voxels.
    #[test]
    fn a_quarter_turn_about_a_pivot_routes_exactly_and_lands_where_it_says() {
        let pivot = Vec3::new(8.0, 8.0, 8.0);
        for (axis, angle) in [
            (Vec3::Y, FRAC_PI_2),
            (Vec3::Y, -FRAC_PI_2),
            (Vec3::X, FRAC_PI_2),
            (Vec3::Z, std::f32::consts::PI),
        ] {
            let rotation = Quat::from_axis_angle(axis, angle);
            let placement = Similarity::about(pivot, rotation, Vec3::ONE, Vec3::ZERO);
            let Bake::Exact { turns, voxel_offset } = placement.route(VOXEL) else {
                panic!("a quarter turn about a lattice pivot should be exact: {placement:?}");
            };

            // The route's own claim, checked against the map it claims to
            // reproduce: turn a voxel, shift it, and it must land where the
            // similarity would have put it.
            for voxel in [IVec3::ZERO, IVec3::new(5, -9, 13), IVec3::new(-40, 3, 0)] {
                let through_route = (turns.apply_voxel(voxel) + voxel_offset).as_vec3() * VOXEL;
                let through_map = placement.transform_point(voxel.as_vec3() * VOXEL);
                assert!(
                    through_route.distance(through_map) < VOXEL * 1.0e-2,
                    "{axis:?} {angle}: route put {voxel:?} at {through_route:?}, \
                     map says {through_map:?}"
                );
            }
        }
    }

    /// The exact route must never be claimed for a map that does not actually
    /// land the lattice on itself, because taking it would move the model to
    /// somewhere the user did not ask for and report the move as lossless.
    #[test]
    fn a_rotation_just_off_a_quarter_turn_is_not_called_exact() {
        for offset in [0.02_f32, 0.005, 0.001] {
            let rotation = Quat::from_rotation_y(FRAC_PI_2 + offset);
            let placement = Similarity::about(Vec3::ZERO, rotation, Vec3::ONE, Vec3::ZERO);
            assert_eq!(
                placement.route(VOXEL),
                Bake::Resample,
                "a turn {offset} rad off square was called exact"
            );
        }
    }

    /// Whatever the route says is exact really is a lattice map: every lattice
    /// point goes to a lattice point, and to the one the similarity names.
    #[test]
    fn the_exact_route_never_moves_a_lattice_point_off_the_lattice() {
        let mut seed = 0x9e37_79b9_u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };

        let mut exact_seen = 0;
        for _ in 0..2000 {
            // A mixture: mostly snapped, so the exact route is actually
            // exercised, with free values salted in so the refusal is too.
            let snapped = next() < 0.7;
            let rotation = if snapped {
                let quarter = (next() * 4.0) as i32;
                let axis = [Vec3::X, Vec3::Y, Vec3::Z][(next() * 3.0) as usize % 3];
                Quat::from_axis_angle(axis, quarter as f32 * FRAC_PI_2)
            } else {
                Quat::from_rotation_y(next() * 6.0)
            };
            let offset = if snapped {
                ((Vec3::new(next(), next(), next()) - 0.5) * 40.0).round() * VOXEL
            } else {
                (Vec3::new(next(), next(), next()) - 0.5) * 40.0
            };
            let pivot = if snapped {
                ((Vec3::new(next(), next(), next()) - 0.5) * 60.0).round() * VOXEL
            } else {
                (Vec3::new(next(), next(), next()) - 0.5) * 60.0
            };
            let scale = if snapped { 1.0 } else { 0.5 + next() };

            let placement = Similarity::about(pivot, rotation, Vec3::splat(scale), offset);
            let Bake::Exact { turns, voxel_offset } = placement.route(VOXEL) else {
                continue;
            };
            exact_seen += 1;
            for voxel in [IVec3::ZERO, IVec3::new(7, -3, 21), IVec3::new(-64, 64, -1)] {
                let through_route = (turns.apply_voxel(voxel) + voxel_offset).as_vec3() * VOXEL;
                let through_map = placement.transform_point(voxel.as_vec3() * VOXEL);
                assert!(
                    through_route.distance(through_map) < VOXEL * 0.05,
                    "exact route disagreed with its own map by \
                     {} voxels",
                    through_route.distance(through_map) / VOXEL
                );
            }
        }
        assert!(exact_seen > 200, "only {exact_seen} of 2000 took the exact route");
    }

    #[test]
    fn any_scale_at_all_is_a_resample() {
        for scale in [0.5_f32, 0.999, 1.001, 2.0] {
            let placement =
                Similarity::about(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(scale), Vec3::ZERO);
            assert_eq!(placement.route(VOXEL), Bake::Resample, "scale {scale}");
        }
    }

    #[test]
    fn a_quarter_turn_read_off_a_quaternion_matches_the_one_named_by_hand() {
        // Up becomes front: the turn `orientation` builds from two facings, and
        // the same turn expressed as a rotation about X.
        let named = AxisRotation::taking(Facing::Up, Facing::Front);
        let read = quarter_turn(Quat::from_rotation_x(FRAC_PI_2)).expect("a quarter turn");
        for voxel in [IVec3::X, IVec3::Y, IVec3::Z, IVec3::new(3, -5, 7)] {
            assert_eq!(named.apply_ivec(voxel), read.apply_ivec(voxel), "{voxel:?}");
        }
    }

    #[test]
    fn a_free_angle_turn_names_no_axis_rotation() {
        assert!(quarter_turn(Quat::from_rotation_y(0.4)).is_none());
        assert!(quarter_turn(Quat::from_rotation_y(FRAC_PI_2 * 0.5)).is_none());
        assert!(quarter_turn(Quat::from_euler(glam::EulerRot::YXZ, 0.1, 0.2, 0.3)).is_none());
    }

    #[test]
    fn a_nonsense_voxel_size_routes_to_a_resample_rather_than_dividing_by_it() {
        for voxel_size in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(Similarity::IDENTITY.route(voxel_size), Bake::Resample, "{voxel_size}");
        }
    }
}
