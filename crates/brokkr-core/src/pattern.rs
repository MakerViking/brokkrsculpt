// SPDX-License-Identifier: AGPL-3.0-only

//! Surface patterns: hair, scales, weave, cracks and noise.
//!
//! A pattern is not a brush. It is one extra multiply on the weight every
//! brush already computes, so Clay plus Scales and Inflate plus Hair both work
//! and there is no combinatorial pile of brushes to choose between. That is
//! the whole design: the brief asks for something simpler than ZBrush but
//! still flexible enough to make good models, and composing two small controls
//! beats enumerating their product.
//!
//! # The weight must stay in 0..=1
//!
//! Load bearing. Smooth, flatten and clay use the brush weight as a lerp
//! factor toward a target value, and a factor outside the unit range would
//! extrapolate *away* from the target — which is unstable across a stroke in
//! exactly the way [`crate::brush`] documents at length. Carving a pattern in
//! rather than raising it is the invert modifier's job, not a negative weight.
//!
//! # Why the coordinates come from world space
//!
//! The pattern is evaluated from the world position, not from any surface
//! parameterisation. Two consequences, both wanted:
//!
//! * A second stroke over the same place lands on the same features and
//!   deepens them, instead of laying down a fresh set at a new offset and
//!   smearing the two together.
//! * No UV unwrapping is needed, which matters because unwrapping is on the
//!   build spec's do-not-build list.
//!
//! The plane the world position is projected onto is chosen once per stamp
//! from the stamp's normal, so a single stamp is internally seam free. Across
//! a strongly curved surface two stamps can pick different planes and leave a
//! seam between them; blending three projections would fix it at three times
//! the cost, and this runs in the hottest loop in the project. Revisit it if
//! the seam is ever visible in practice.

use glam::{Vec2, Vec3};

/// Which pattern is multiplied into the brush.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PatternKind {
    /// No pattern. The brush behaves exactly as it did before patterns
    /// existed, down to the last bit.
    #[default]
    None,
    /// Unstructured value noise, for roughening a surface.
    Noise,
    /// Overlapping domed scales in offset rows.
    Scales,
    /// Combed strands running along the stroke.
    Hair,
    /// A basket weave of over and under bands.
    Weave,
    /// A thin network of lines, for carving crazing into a surface.
    Cracks,
}

impl PatternKind {
    pub const ALL: [PatternKind; 6] = [
        PatternKind::None,
        PatternKind::Noise,
        PatternKind::Scales,
        PatternKind::Hair,
        PatternKind::Weave,
        PatternKind::Cracks,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PatternKind::None => "None",
            PatternKind::Noise => "Noise",
            PatternKind::Scales => "Scales",
            PatternKind::Hair => "Hair",
            PatternKind::Weave => "Weave",
            PatternKind::Cracks => "Cracks",
        }
    }

    /// Whether this pattern's orientation follows the stroke.
    ///
    /// Only hair does. Everything else is a pure function of world position,
    /// so it reinforces no matter which way the stroke crosses it; hair combs,
    /// which is the point of hair.
    pub fn follows_the_stroke(self) -> bool {
        matches!(self, PatternKind::Hair)
    }
}

impl std::fmt::Display for PatternKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Fewest voxels one pattern feature may span.
///
/// **In voxels, deliberately, not in millimetres.** The same 0.5 mm pattern is
/// comfortable at a 0.06 mm voxel and impossible at a 1 mm one, so a fixed
/// millimetre floor is meaningless across the sixteen to one range the detail
/// control offers.
///
/// Below about three voxels the field cannot carry the feature: the ridges go
/// razor thin and the surface pinches into a non-manifold edge where two
/// sheets meet. Measured, not guessed — at two voxels the seam test finds an
/// edge shared by four triangles, at three it finds none. Four is that with a
/// margin. Such a mesh is still closed, so it renders perfectly, and the
/// export validator then refuses to write it, which is the worst way to find
/// out.
pub const MIN_SCALE_VOXELS: f32 = 4.0;

/// Largest feature size the interface offers, in world units.
pub const MAX_SCALE_MM: f32 = 12.0;

/// A configured pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pattern {
    pub kind: PatternKind,
    /// Size of one feature in world units. Millimetres, like everything else.
    pub scale_mm: f32,
    /// How much of the brush weight the pattern is allowed to take away.
    /// Zero leaves the brush untouched; one lets the pattern cut the weight to
    /// nothing where it is darkest.
    pub depth: f32,
}

impl Default for Pattern {
    fn default() -> Self {
        Self { kind: PatternKind::None, scale_mm: 2.0, depth: 0.7 }
    }
}

impl Pattern {
    /// Whether this pattern changes anything at all.
    #[inline]
    pub fn is_off(&self) -> bool {
        self.kind == PatternKind::None || self.depth <= 0.0 || self.scale_mm <= 0.0
    }

    /// Resolve everything that is constant across one stamp.
    ///
    /// The projection axes and the reciprocal scale do not vary per voxel, and
    /// this is called once per voxel inside the hottest loop in the project,
    /// so working them out here rather than there is the difference between a
    /// pattern that fits the budget and one that does not.
    pub fn prepare(&self, voxel_size: f32, normal: Vec3, tangent: Vec3) -> Prepared {
        let (u_axis, v_axis) = if self.kind.follows_the_stroke() {
            // Comb along the stroke: u runs with the travel direction, v
            // across it, so strands lie the way the brush was dragged.
            let along = tangent.normalize_or(perpendicular_to(normal));
            let across = normal.cross(along).normalize_or(perpendicular_to(along));
            (along, across)
        } else {
            // Triplanar: drop the world axis the normal points along most and
            // keep the other two, so the coordinate stays a pure function of
            // world position and the pattern reinforces itself.
            dominant_plane(normal)
        };

        // Clamped here rather than in the interface, because this is the only
        // place that knows the voxel size. See MIN_SCALE_VOXELS.
        let scale = self.scale_mm.max(voxel_size * MIN_SCALE_VOXELS);

        Prepared {
            kind: self.kind,
            inverse_scale: if scale > 0.0 { 1.0 / scale } else { 0.0 },
            depth: self.depth.clamp(0.0, 1.0),
            u_axis,
            v_axis,
        }
    }
}

/// A pattern with its per stamp constants already worked out.
#[derive(Debug, Clone, Copy)]
pub struct Prepared {
    kind: PatternKind,
    inverse_scale: f32,
    depth: f32,
    u_axis: Vec3,
    v_axis: Vec3,
}

impl Prepared {
    /// A pattern that multiplies every weight by one, for the default path.
    pub const OFF: Prepared = Prepared {
        kind: PatternKind::None,
        inverse_scale: 0.0,
        depth: 0.0,
        u_axis: Vec3::X,
        v_axis: Vec3::Y,
    };

    /// The factor to multiply the brush weight by at a world position. Always
    /// in `0..=1`.
    #[inline]
    pub fn weight(&self, position: Vec3) -> f32 {
        if matches!(self.kind, PatternKind::None) || self.depth <= 0.0 {
            return 1.0;
        }
        let raw = self.raw(position);
        // depth 0 is a no-op and depth 1 is the pattern at full contrast. The
        // clamp is belt and braces: every branch of `raw` is already in 0..=1,
        // and a test holds it there.
        (1.0 - self.depth + self.depth * raw).clamp(0.0, 1.0)
    }

    /// The pattern's own value at a position, in `0..=1`.
    #[inline]
    fn raw(&self, position: Vec3) -> f32 {
        // Each arm builds only the coordinate it needs. Computing both a
        // scaled world position and a surface uv for every pattern cost two
        // dot products per voxel on the arms that never looked at them.
        match self.kind {
            PatternKind::None => 1.0,
            PatternKind::Noise => value_noise(position * self.inverse_scale),
            PatternKind::Cracks => cracks(position * self.inverse_scale),
            PatternKind::Scales => scales(self.uv(position)),
            PatternKind::Hair => hair(self.uv(position)),
            PatternKind::Weave => weave(self.uv(position)),
        }
    }

    /// The surface coordinate, in units of the pattern scale.
    #[inline]
    fn uv(&self, position: Vec3) -> Vec2 {
        Vec2::new(position.dot(self.u_axis), position.dot(self.v_axis)) * self.inverse_scale
    }
}

/// Overlapping domed scales, in rows offset by half a scale.
#[inline]
fn scales(uv: Vec2) -> f32 {
    let row = uv.y.floor();
    // Alternate rows shift by half, which is what stops the scales lining up
    // into a grid and makes them read as scales.
    let shifted = uv.x + if (row as i32) & 1 == 0 { 0.0 } else { 0.5 };
    let across = shifted - shifted.floor() - 0.5;
    let along = uv.y - row - 0.5;
    // Squared distances, with the root taken only inside the dome, which is
    // the only place the value is not already zero.
    let squared = (across * across + along * along) * 4.0;
    if squared >= 1.0 {
        return 0.0;
    }
    1.0 - squared.sqrt()
}

/// Strands running along `u`, parted across `v`.
#[inline]
fn hair(uv: Vec2) -> f32 {
    let lane = uv.y.floor();
    let across = (uv.y - lane - 0.5).abs() * 2.0;
    let ridge = 1.0 - across;
    if ridge <= 0.0 {
        return 0.0;
    }
    // Each strand gets its own offset, or the result reads as corduroy rather
    // than hair. The fractional part of the golden ratio rather than a hash:
    // it walks the unit interval without ever repeating, which is all the
    // variety a strand offset needs, and it is two operations where a hash is
    // eight — in the hottest loop in the project.
    let jitter = fract(lane * 0.618_034);
    // A slow variation along the strand so it tapers rather than running dead
    // straight for ever.
    let taper = 0.75 + 0.25 * wave_01(uv.x * 0.6 + jitter);
    ridge * taper
}

/// The fractional part of a number, in `0..1`.
#[inline]
fn fract(t: f32) -> f32 {
    t - t.floor()
}

/// A smooth periodic wave in `0..=1`, period 1.
///
/// A parabola rather than a sine: the same shape to the eye at this amplitude,
/// and four arithmetic operations against a transcendental call.
#[inline]
fn wave_01(t: f32) -> f32 {
    let x = fract(t) * 2.0 - 1.0;
    1.0 - x * x
}

/// A basket weave: bands crossing over and under one another.
#[inline]
fn weave(uv: Vec2) -> f32 {
    let over = ((uv.x.floor() as i32).wrapping_add(uv.y.floor() as i32) & 1) == 0;
    let band = |t: f32| {
        let across = (t - t.floor() - 0.5).abs() * 2.0;
        (1.0 - across).clamp(0.0, 1.0)
    };
    if over { band(uv.y) } else { band(uv.x) }
}

/// A thin network of lines, following the zero crossing of a noise field.
///
/// High *on* the lines: the pattern says where the brush acts, and for cracks
/// what you want carved is the line itself.
#[inline]
fn cracks(scaled: Vec3) -> f32 {
    /// Half width of a crack, as a fraction of the noise's range.
    ///
    /// Small on purpose. Cubing a full width ridge was the first attempt and
    /// it rendered as lumpy noise rather than crazing -- visibly wrong, and
    /// invisible to every assertion, which is what the offscreen harness is
    /// for. A crack has to be *thin* relative to the space between cracks.
    const HALF_WIDTH: f32 = 0.07;

    let distance = (value_noise(scaled) - 0.5).abs();
    let across = 1.0 - (distance / HALF_WIDTH).min(1.0);
    // Squared, so the line has soft shoulders rather than a hard edge that
    // would alias against the voxel lattice.
    across * across
}

/// Trilinearly interpolated value noise in `0..=1`.
///
/// Value rather than gradient noise, and one octave rather than several,
/// because this is evaluated per voxel inside the sculpt loop. Eight hashes
/// and a smoothstep is about as much as the edit budget will carry.
#[inline]
fn value_noise(p: Vec3) -> f32 {
    let base = p.floor();
    let f = p - base;
    // Smoothstep, so the lattice does not show up as a grid of creases.
    let t = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let (x, y, z) = (base.x as i32, base.y as i32, base.z as i32);

    let corner = |dx: i32, dy: i32, dz: i32| to_unit(hash3(x + dx, y + dy, z + dz));

    let c00 = lerp(corner(0, 0, 0), corner(1, 0, 0), t.x);
    let c10 = lerp(corner(0, 1, 0), corner(1, 1, 0), t.x);
    let c01 = lerp(corner(0, 0, 1), corner(1, 0, 1), t.x);
    let c11 = lerp(corner(0, 1, 1), corner(1, 1, 1), t.x);

    lerp(lerp(c00, c10, t.y), lerp(c01, c11, t.y), t.z)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// An integer hash of a lattice point.
///
/// Hand rolled rather than pulled in: `glam` has no noise, and forty lines is
/// not worth a dependency. The constants are odd so the multiplies are
/// invertible and no lattice point collapses onto another.
#[inline]
fn hash3(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x2974_5b47);
    h ^ (h >> 16)
}

/// A hash mapped onto `0..=1`.
///
/// Uses the top 24 bits, which are the well mixed ones, and divides by an
/// exact power of two so the result can never round above 1.
#[inline]
fn to_unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

/// Any unit vector perpendicular to `v`.
///
/// Picks the world axis `v` is least aligned with before crossing, so the two
/// are never close to parallel and the cross product never collapses.
fn perpendicular_to(v: Vec3) -> Vec3 {
    let a = v.abs();
    let axis = if a.x <= a.y && a.x <= a.z {
        Vec3::X
    } else if a.y <= a.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    v.cross(axis).normalize_or(Vec3::X)
}

/// The two world axes that best span the surface at a normal.
///
/// Dropping the axis the normal points along most is the cheapest projection
/// that keeps the pattern a pure function of world position.
fn dominant_plane(normal: Vec3) -> (Vec3, Vec3) {
    let a = normal.abs();
    if a.x >= a.y && a.x >= a.z {
        (Vec3::Z, Vec3::Y)
    } else if a.y >= a.z {
        (Vec3::X, Vec3::Z)
    } else {
        (Vec3::X, Vec3::Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fine enough that the voxel floor never clamps the scales under test.
    const TEST_VOXEL: f32 = 0.05;

    fn every_kind_prepared(scale: f32, depth: f32) -> Vec<(PatternKind, Prepared)> {
        PatternKind::ALL
            .into_iter()
            .map(|kind| {
                let pattern = Pattern { kind, scale_mm: scale, depth };
                (kind, pattern.prepare(TEST_VOXEL, Vec3::Y, Vec3::X))
            })
            .collect()
    }

    /// The invariant the whole design rests on. Smooth, flatten and clay use
    /// the weight as a lerp factor toward a target, so anything outside the
    /// unit range would extrapolate away from that target and compound across
    /// a stroke.
    #[test]
    fn every_pattern_stays_inside_the_unit_range_and_is_finite() {
        for (kind, prepared) in every_kind_prepared(2.0, 1.0) {
            for i in -60..60 {
                for j in -60..60 {
                    let at = Vec3::new(i as f32 * 0.37, j as f32 * 0.23, i as f32 * -0.11);
                    let w = prepared.weight(at);
                    assert!(w.is_finite(), "{kind} was not finite at {at:?}");
                    assert!((0.0..=1.0).contains(&w), "{kind} left the unit range at {at:?}: {w}");
                }
            }
        }
    }

    /// Patterns off must be free, not nearly free: `None` is the default, so
    /// every existing brush test would otherwise be measuring a slightly
    /// different brush.
    #[test]
    fn no_pattern_multiplies_every_weight_by_exactly_one() {
        let prepared = Pattern::default().prepare(TEST_VOXEL, Vec3::Y, Vec3::X);
        assert!(Pattern::default().is_off());
        for i in 0..500 {
            let at = Vec3::splat(i as f32 * 0.13);
            assert_eq!(prepared.weight(at), 1.0);
        }
        assert_eq!(Prepared::OFF.weight(Vec3::new(3.0, -2.0, 7.0)), 1.0);
    }

    #[test]
    fn zero_depth_is_the_same_as_no_pattern_at_all() {
        for (kind, prepared) in every_kind_prepared(2.0, 0.0) {
            for i in 0..200 {
                let at = Vec3::new(i as f32 * 0.31, 1.0, -0.5);
                assert_eq!(prepared.weight(at), 1.0, "{kind} did something at zero depth");
            }
        }
    }

    /// Depth is the contrast control: half depth must sit halfway between no
    /// pattern and full pattern, or the slider will not behave.
    #[test]
    fn depth_scales_the_pattern_between_off_and_full() {
        let at = Vec3::new(1.3, 0.7, -2.1);
        for kind in PatternKind::ALL {
            if kind == PatternKind::None {
                continue;
            }
            let full =
                Pattern { kind, scale_mm: 2.0, depth: 1.0 }.prepare(TEST_VOXEL, Vec3::Y, Vec3::X);
            let half =
                Pattern { kind, scale_mm: 2.0, depth: 0.5 }.prepare(TEST_VOXEL, Vec3::Y, Vec3::X);
            let expected = 0.5 + 0.5 * full.weight(at);
            assert!(
                (half.weight(at) - expected).abs() < 1.0e-5,
                "{kind} at half depth was {} rather than {expected}",
                half.weight(at)
            );
        }
    }

    /// A pattern that did not vary would be a constant multiplier, which is
    /// what the strength slider already is.
    #[test]
    fn every_pattern_actually_varies_across_a_surface() {
        for (kind, prepared) in every_kind_prepared(2.0, 1.0) {
            if kind == PatternKind::None {
                continue;
            }
            let mut lowest: f32 = 1.0;
            let mut highest: f32 = 0.0;
            for i in 0..400 {
                let at = Vec3::new(i as f32 * 0.11, i as f32 * 0.07, i as f32 * -0.05);
                let w = prepared.weight(at);
                lowest = lowest.min(w);
                highest = highest.max(w);
            }
            assert!(highest - lowest > 0.2, "{kind} barely varied: {lowest} to {highest}");
        }
    }

    #[test]
    fn the_scale_sets_how_big_a_feature_is() {
        // Count how often the pattern crosses its own midpoint along a line.
        // A finer scale has to cross more often.
        let crossings = |scale: f32| {
            let prepared = Pattern { kind: PatternKind::Scales, scale_mm: scale, depth: 1.0 }
                .prepare(TEST_VOXEL, Vec3::Y, Vec3::X);
            let mut count = 0;
            let mut previous = prepared.weight(Vec3::ZERO) > 0.5;
            // Diagonally, and at an awkward ratio. Straight along one axis
            // runs down the seam between two rows of scales and never crosses
            // a centre at all, which reads as "no features" rather than as a
            // degenerate probe.
            for i in 1..2000 {
                let at = Vec3::new(i as f32 * 0.02, 0.0, i as f32 * 0.017);
                let now = prepared.weight(at) > 0.5;
                if now != previous {
                    count += 1;
                }
                previous = now;
            }
            count
        };

        let fine = crossings(1.0);
        let coarse = crossings(4.0);
        assert!(fine > coarse * 2, "a 4x finer scale gave {fine} against {coarse}");
    }

    /// The reason the coordinates are taken from world space: a second stroke
    /// over the same place has to deepen the same features rather than lay
    /// down a fresh set.
    #[test]
    fn a_world_space_pattern_gives_the_same_answer_whatever_the_stroke_direction() {
        for kind in PatternKind::ALL {
            if kind.follows_the_stroke() {
                continue;
            }
            let pattern = Pattern { kind, scale_mm: 2.0, depth: 1.0 };
            let one = pattern.prepare(TEST_VOXEL, Vec3::Y, Vec3::X);
            let other = pattern.prepare(TEST_VOXEL, Vec3::Y, Vec3::new(0.3, 0.0, 0.95).normalize());

            for i in 0..200 {
                let at = Vec3::new(i as f32 * 0.19, 3.0, i as f32 * -0.07);
                assert_eq!(
                    one.weight(at),
                    other.weight(at),
                    "{kind} moved when the stroke direction changed"
                );
            }
        }
    }

    /// ...and the one exception, which is the entire point of hair.
    #[test]
    fn hair_combs_along_the_stroke() {
        let pattern = Pattern { kind: PatternKind::Hair, scale_mm: 2.0, depth: 1.0 };
        assert!(PatternKind::Hair.follows_the_stroke());

        let along_x = pattern.prepare(TEST_VOXEL, Vec3::Y, Vec3::X);
        let along_z = pattern.prepare(TEST_VOXEL, Vec3::Y, Vec3::Z);

        let differences = (0..200)
            .filter(|i| {
                let at = Vec3::new(*i as f32 * 0.13, 0.0, *i as f32 * 0.09);
                (along_x.weight(at) - along_z.weight(at)).abs() > 1.0e-4
            })
            .count();
        assert!(differences > 40, "hair did not rotate with the stroke, {differences} differed");
    }

    #[test]
    fn a_degenerate_tangent_still_produces_a_usable_frame() {
        let pattern = Pattern { kind: PatternKind::Hair, scale_mm: 2.0, depth: 1.0 };
        // A tangent of zero, and one parallel to the normal: both leave the
        // cross product with nothing to work from.
        for tangent in [Vec3::ZERO, Vec3::Y, -Vec3::Y] {
            let prepared = pattern.prepare(TEST_VOXEL, Vec3::Y, tangent);
            assert!(prepared.u_axis.is_finite() && prepared.u_axis.length() > 0.5);
            assert!(prepared.v_axis.is_finite() && prepared.v_axis.length() > 0.5);
            for i in 0..50 {
                let w = prepared.weight(Vec3::splat(i as f32 * 0.21));
                assert!(w.is_finite() && (0.0..=1.0).contains(&w));
            }
        }
    }

    #[test]
    fn the_noise_hash_spreads_evenly_and_never_reaches_one() {
        let mut buckets = [0usize; 8];
        for x in 0..40 {
            for y in 0..40 {
                for z in 0..40 {
                    let v = to_unit(hash3(x, y, z));
                    assert!((0.0..1.0).contains(&v), "hash left the unit range: {v}");
                    buckets[(v * 8.0) as usize] += 1;
                }
            }
        }
        let expected = 40 * 40 * 40 / 8;
        for (index, count) in buckets.iter().enumerate() {
            let ratio = *count as f32 / expected as f32;
            assert!((0.85..1.15).contains(&ratio), "bucket {index} held {count}, ratio {ratio}");
        }
    }

    /// Neighbouring lattice points must not hash to neighbouring values, or
    /// the noise shows the lattice.
    #[test]
    fn adjacent_lattice_points_are_uncorrelated() {
        let mut same_side = 0;
        for x in 0..60 {
            for y in 0..60 {
                let a = to_unit(hash3(x, y, 0));
                let b = to_unit(hash3(x + 1, y, 0));
                if (a > 0.5) == (b > 0.5) {
                    same_side += 1;
                }
            }
        }
        let ratio = same_side as f32 / (60.0 * 60.0);
        assert!((0.42..0.58).contains(&ratio), "neighbours agreed {ratio} of the time");
    }

    #[test]
    fn the_noise_is_continuous_across_a_lattice_boundary() {
        // A step at an integer coordinate would show up as a grid of creases.
        let step = 1.0e-4;
        for axis in 0..3 {
            let mut before = Vec3::new(3.0, 5.0, 7.0);
            before[axis] -= step;
            let mut after = Vec3::new(3.0, 5.0, 7.0);
            after[axis] += step;
            let jump = (value_noise(after) - value_noise(before)).abs();
            assert!(jump < 0.01, "axis {axis} jumped by {jump} across the lattice");
        }
    }
}
