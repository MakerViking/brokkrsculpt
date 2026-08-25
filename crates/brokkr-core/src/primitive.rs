// SPDX-License-Identifier: AGPL-3.0-only

//! Cube, sphere and cylinder, each built as a **fresh** [`Volume`].
//!
//! # Why these are not `seed_sphere`
//!
//! [`Volume::seed_sphere`]'s own documentation says it replaces "anything
//! already in the volume within its bounds": it calls `remove` on every brick of
//! its box that falls outside the band, and it never calls `record_for_undo`.
//! Seeding into an existing body would therefore delete neighbouring material
//! invisibly to history. A primitive is a new body, so it starts from an empty
//! volume and nothing it writes can destroy anything.
//!
//! # Three fields, and why the brick classification generalises
//!
//! All three distance functions are **1-Lipschitz** -- move a point by `t` and
//! the distance changes by at most `t`. So one evaluation at a brick's centre
//! bounds the whole brick, given `h`, half the space diagonal of a brick's
//! sample box:
//!
//! - `d(centre) - h >= band` proves every sample saturates positive, so the
//!   brick is left **absent** (an absent brick reads as [`OUTSIDE`]);
//! - `d(centre) + h <= -band` proves every sample saturates negative, so the
//!   brick becomes `Uniform(INSIDE)`;
//! - anything else is resolved voxel by voxel.
//!
//! Conservative in the safe direction, and never wrong: a brick wrongly called
//! borderline costs 128 KB that is then collapsed again, where a brick wrongly
//! called empty is a hole in the model.
//!
//! **An interior brick must be `Uniform(INSIDE)` and never absent.** Absent
//! reads as empty space, so a cube whose middle went missing would export as a
//! shell around a void -- and that void is watertight, so nothing downstream
//! would refuse it. It would simply print hollow.
//!
//! # `fidget` was considered and is not needed
//!
//! `docs/BUILD-SPEC.md:80` defers a JIT SDF evaluator for "procedural primitives
//! and booleans... Revisit at M4, not before", which is precisely this feature,
//! so someone will reach for it. Three closed-form fields are about fifteen
//! lines each and evaluate only inside the bricks the classification could not
//! decide. A JIT would buy nothing here and cost a compiler.

use glam::{IVec3, Vec3};

use crate::body::Document;
use crate::brick::{
    BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, brick_index,
};
use crate::volume::Volume;

/// Half the space diagonal of one brick's sample box, in VOXELS.
///
/// `32 * sqrt(3) / 2`. A brick spans `BRICK_DIM` sample positions, so the true
/// figure is `31 * sqrt(3) / 2 = 26.846`; the larger number is used because
/// being generous here can only turn an absent brick into a dense one that is
/// then collapsed, while being mean turns a solid brick into a hole.
const BRICK_HALF_DIAGONAL_VOXELS: f32 = 27.712_812;

/// The shapes the `+` button can add.
///
/// Three, and deliberately not a general CSG tree: the brief is "simpler than
/// ZBrush, flexible enough", and a cube, a ball and a rod are what a scan repair
/// actually needs to graft onto a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    Cube,
    Sphere,
    Cylinder,
}

impl PrimitiveKind {
    pub const ALL: [PrimitiveKind; 3] =
        [PrimitiveKind::Cube, PrimitiveKind::Sphere, PrimitiveKind::Cylinder];

    /// What the menu entry and the new body are called.
    ///
    /// One string for both, so the row a user sees says the same word they
    /// pressed.
    pub fn label(self) -> &'static str {
        match self {
            PrimitiveKind::Cube => "Cube",
            PrimitiveKind::Sphere => "Sphere",
            PrimitiveKind::Cylinder => "Cylinder",
        }
    }

    /// Signed distance in MILLIMETRES from a point given relative to the
    /// primitive's centre, with `half` the half-extent in millimetres.
    ///
    /// Every one of these is the standard closed form and every one is
    /// 1-Lipschitz, which is what [`build`]'s brick classification rests on.
    /// The cylinder stands on Y, because that is up in sculpt space.
    fn distance(self, at: Vec3, half: f32) -> f32 {
        match self {
            PrimitiveKind::Sphere => at.length() - half,
            PrimitiveKind::Cube => {
                let q = at.abs() - Vec3::splat(half);
                // Outside contributes the length of the positive part; inside,
                // where every component is negative, the largest one is the
                // distance to the nearest face.
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
            PrimitiveKind::Cylinder => {
                let radial = Vec3::new(at.x, 0.0, at.z).length() - half;
                let axial = at.y.abs() - half;
                let outside = Vec3::new(radial.max(0.0), axial.max(0.0), 0.0).length();
                outside + radial.max(axial).min(0.0)
            }
        }
    }

    /// Square millimetres of surface for a half-extent of `half`.
    ///
    /// What both cost estimates are made of, because the whole cost of a
    /// primitive is a shell over its surface. Exact closed forms -- a cube is
    /// six faces of `2h` square, a ball is `4 pi h^2`, and a rod of radius `h`
    /// and height `2h` is `4 pi h^2` of side plus `2 pi h^2` of caps.
    fn surface_area(self, half: f64) -> f64 {
        match self {
            PrimitiveKind::Cube => 24.0 * half * half,
            PrimitiveKind::Sphere => 4.0 * std::f64::consts::PI * half * half,
            PrimitiveKind::Cylinder => 6.0 * std::f64::consts::PI * half * half,
        }
    }
}

/// What a primitive of this size WOULD cost, as bytes of voxel data and mesh
/// vertices, without building it.
///
/// Fed to [`crate::body::GrowthGuard::no_room_for`] so an add that cannot fit
/// is refused **before** [`build`] allocates it. That ordering is the whole
/// point: the refusal is worth nothing if the allocation it is refusing has
/// already happened, and a big primitive at a fine voxel runs to tens of
/// gigabytes -- a 133 mm cube at 0.05 mm is about 54 GB, which is not a
/// message, it is an OOM kill.
///
/// The estimators are [`crate::voxelise`]'s, shared rather than reinvented, so
/// the import path and the add path predict the same shell for the same
/// surface. Both are deliberately generous; a refusal that is slightly too
/// eager costs a smaller cube, and one that is slightly too keen costs the
/// session.
///
/// **The byte figure takes the larger of that estimate and a brick count**,
/// because an area-based prediction is asymptotic and comes up SHORT below
/// about two bricks across: a body occupies whole bricks whatever its area, and
/// an 8 mm cube at a 0.5 mm voxel allocates 1.0 MB against a predicted 0.5 --
/// measured. It cannot matter at the sizes the guard refuses, and an estimator
/// documented as generous that is quietly short in one corner is the kind of
/// thing someone later scales up.
pub fn cost(kind: PrimitiveKind, voxel_size: f32, half: f32) -> (f64, f64) {
    let area = kind.surface_area(half as f64);
    let bytes = crate::voxelise::estimated_bytes(area, voxel_size)
        .max(shell_bricks(voxel_size, half) * BRICK_VOXELS as f64 * 4.0);
    (bytes, crate::voxelise::estimated_vertices(area, voxel_size))
}

/// Bricks the shell of a solid primitive can occupy, laid out the way [`build`]
/// lays it out: the bricks its box spans, less the ones two deep inside it,
/// which come out `Uniform` and cost nothing.
///
/// Generous by a brick a side, because where the surface falls within a brick
/// depends on where the centre landed on the lattice.
fn shell_bricks(voxel_size: f32, half: f32) -> f64 {
    let brick_mm = BRICK_DIM as f64 * voxel_size as f64;
    let span = 2.0 * (half + NARROW_BAND * voxel_size) as f64;
    let across = (span / brick_mm).ceil() + 2.0;
    across.powi(3) - (across - 2.0).max(0.0).powi(3)
}

/// The inclusive range of brick coordinates [`build`] walks for a primitive.
///
/// Everything that could hold anything but saturated [`OUTSIDE`], with a
/// brick's reach on top so the classification is asked about every brick it
/// could answer "borderline" for.
///
/// A function rather than four lines inside `build` because the test that
/// checks the classification against the formula has to walk **the same**
/// range: walking the bricks that were actually stored instead cannot see a
/// brick that was wrongly left out, which is the one failure the check exists
/// for.
fn brick_range(voxel_size: f32, centre: Vec3, half: f32) -> (IVec3, IVec3) {
    let band = NARROW_BAND * voxel_size;
    let reach = BRICK_HALF_DIAGONAL_VOXELS * voxel_size;
    let extent = half + band + reach;
    let low_voxel = ((centre - extent) / voxel_size).floor().as_ivec3();
    let high_voxel = ((centre + extent) / voxel_size).ceil().as_ivec3();
    (BrickCoord::containing(low_voxel).0, BrickCoord::containing(high_voxel).0)
}

/// Build one primitive as a whole new volume, on the given lattice.
///
/// `half` is the half-extent in millimetres: half a cube's side, a sphere's
/// radius, and both the radius and the half-height of a cylinder.
///
/// The volume comes back with **every** brick marked dirty, because nothing
/// else will mark them: `Document::add_body` does not, and a body whose bricks
/// were never meshed is a body that is in the document, exports correctly, and
/// is invisible on screen for the rest of the session. This project has shipped
/// that class of bug twice.
pub fn build(kind: PrimitiveKind, voxel_size: f32, centre: Vec3, half: f32) -> Volume {
    let mut volume = Volume::new(voxel_size);
    let band = NARROW_BAND * voxel_size;
    let reach = BRICK_HALF_DIAGONAL_VOXELS * voxel_size;
    let (low_brick, high_brick) = brick_range(voxel_size, centre, half);

    for bz in low_brick.z..=high_brick.z {
        for by in low_brick.y..=high_brick.y {
            for bx in low_brick.x..=high_brick.x {
                let coord = BrickCoord::new(bx, by, bz);
                let origin = coord.origin();
                let low = origin.as_vec3() * voxel_size;
                let high = coord.max_voxel().as_vec3() * voxel_size;
                let middle = kind.distance((low + high) * 0.5 - centre, half);

                if middle - reach >= band {
                    // Proven saturated positive throughout. Absent already reads
                    // as OUTSIDE, so this brick costs nothing at all.
                    continue;
                }
                if middle + reach <= -band {
                    volume.insert_brick(coord, Brick::Uniform(INSIDE));
                    continue;
                }

                let mut brick = Brick::dense_filled(OUTSIDE);
                let data = brick.make_dense();
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let at = (origin + IVec3::new(x as i32, y as i32, z as i32)).as_vec3()
                                * voxel_size;
                            // Stored in VOXELS, which is what the field holds.
                            // Millimetres here would scale the whole shape by
                            // the voxel size.
                            let distance = kind.distance(at - centre, half) / voxel_size;
                            data[brick_index(x, y, z)] = distance.clamp(INSIDE, OUTSIDE);
                        }
                    }
                }
                // The classification is conservative, so a brick it called
                // borderline can still come out constant -- a corner of the
                // bounding box that the sphere never reaches, say. Collapsing
                // releases the 128 KB rather than storing 32,768 copies of one
                // number.
                match brick.is_collapsible() {
                    Some(value) if value >= OUTSIDE => {}
                    Some(value) => volume.insert_brick(coord, Brick::Uniform(value)),
                    None => volume.insert_brick(coord, brick),
                }
            }
        }
    }

    volume.mark_everything_dirty();
    volume
}

/// Where the next primitive goes and how big it is: a centre and a half-extent
/// in millimetres, both snapped to a whole voxel.
///
/// **A new primitive is placed CLEAR of the model rather than at the world
/// origin, and that is a decision rather than a convenience.** Sized to a third
/// of the model, a primitive dropped at the origin lands inside the body the
/// user is looking at and cannot be seen -- so the first press of the feature
/// that this whole increment exists for would appear to do nothing. Its low
/// face sits one whole brick clear of the document's box along +X instead,
/// which is far enough that the two surfaces cannot share a brick and therefore
/// cannot share an apron.
///
/// **The box is over EVERY body, visible or not.** A primitive placed clear of
/// only the visible ones is a primitive inside a hidden one, which is the same
/// invisible-first-press failure one reveal later.
///
/// **The size is measured against the BIGGEST SINGLE BODY, never against the
/// document's overall extent.** The plan said "one third of the document's
/// content radius" and that is what shipped, and it compounds: the union box
/// spans the gaps between bodies as well as the bodies, and every primitive is
/// placed in a new gap, so each press produced a bigger cube further out --
/// 37, 48, 66, 93, 128, 181, 249, 342 mm across on a 30 mm ball at a 0.25 mm
/// voxel, unbounded. See [`Document::largest_body_radius`], which is a fixed
/// point instead: a third of the biggest body is smaller than that body.
///
/// `fallback_radius_mm` is what an empty document is sized against. It is a
/// parameter because a default model size is the application's idea of how big
/// a sculpt is, not the engine's -- `brokkr-core` has no opinion about it and
/// should not grow one.
pub fn placement(doc: &Document, fallback_radius_mm: f32) -> (Vec3, f32) {
    let voxel_size = doc.voxel_size();
    let brick_mm = BRICK_DIM as f32 * voxel_size;
    let snap = |value: f32| (value / voxel_size).round() * voxel_size;

    let radius = doc.largest_body_radius().unwrap_or(fallback_radius_mm);
    // At least two voxels, or the field has no room to hold a surface at all
    // and the "primitive" is an empty body.
    let half = snap(radius / 3.0).max(2.0 * voxel_size);

    let Some((low, high)) = doc.world_bounds() else {
        // Nothing in the document to be clear of.
        return (Vec3::ZERO, half);
    };

    // Ceiling rather than nearest, so that rounding the face onto the lattice
    // can only ever add clearance.
    let low_face = (high.x + brick_mm) / voxel_size;
    let low_face = low_face.ceil() * voxel_size;
    let centre =
        Vec3::new(low_face + half, snap((low.y + high.y) * 0.5), snap((low.z + high.z) * 0.5));
    (centre, half)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOXEL: f32 = 0.25;

    /// The headline promise: what comes out is something a slicer will take.
    ///
    /// `is_printable` and deliberately not "manifold" -- it is no boundary edges
    /// and no winding disagreements, which is what `export.rs` checks and what
    /// OrcaSlicer actually refuses a model for.
    #[test]
    fn every_primitive_exports_printable() {
        for kind in PrimitiveKind::ALL {
            let volume = build(kind, VOXEL, Vec3::ZERO, 6.0);
            let (mesh, report) = volume.export_mesh();
            assert!(
                report.is_printable(),
                "{kind:?} is not printable: {} ({} triangles)",
                report.summary(),
                mesh.triangles.len()
            );
        }
    }

    /// A solid primitive must allocate only its shell, and its middle must be a
    /// uniform tile rather than nothing.
    ///
    /// **Measured at a 1.0 mm voxel rather than the 0.25 mm the design named**,
    /// and the property is scale-free: what is being checked is that the
    /// classification calls interior bricks `Uniform(INSIDE)` and never absent,
    /// which does not depend on the lattice. A 200 mm cube at 0.25 mm is 3,458
    /// dense bricks -- 442 MB and 113 million field evaluations -- in a test
    /// process that runs alongside every other test in this crate.
    #[test]
    fn a_solid_cube_allocates_its_shell_and_a_uniform_interior() {
        const VOXEL_MM: f32 = 1.0;
        const HALF_MM: f32 = 100.0;

        let volume = build(PrimitiveKind::Cube, VOXEL_MM, Vec3::ZERO, HALF_MM);
        let stats = volume.stats();

        // 200 mm at 1 mm is 200 voxels, which is 6.25 bricks, so the cube spans
        // seven bricks along each axis once it is placed on the lattice. The
        // shell is that cube of bricks minus the one two smaller inside it, and
        // the bound is generous by one brick a side because where the surface
        // falls inside a brick depends on the offset.
        let across = (2.0 * HALF_MM / (BRICK_DIM as f32 * VOXEL_MM)).ceil() as usize + 2;
        let shell = across.pow(3) - (across - 2).pow(3);
        assert!(
            stats.dense_bricks <= shell,
            "the cube allocated {} dense bricks, more than the {shell} a shell can hold",
            stats.dense_bricks
        );

        // **The clause that does the work.** The bound above is satisfied by a
        // cube that stores its whole interior densely -- at this scale that is
        // 360 dense bricks against a bound of 386 -- and `uniform_bricks > 0`
        // is satisfied too, because a borderline brick that came out constant
        // is collapsed to `Uniform` anyway. So the interior is asserted
        // directly: every brick the Lipschitz bound PROVES is inside must be a
        // uniform tile, neither dense nor absent. Replacing the
        // `Uniform(INSIDE)` in `build` with `dense_filled(INSIDE)` used to
        // leave the whole workspace green; it fails here now, and at the
        // 0.25 mm voxel the design names it was 1.5 GB of dense storage in
        // place of a few tens of kilobytes.
        let reach = (BRICK_HALF_DIAGONAL_VOXELS + NARROW_BAND) * VOXEL_MM;
        let inner = HALF_MM - reach;
        let low = BrickCoord::containing((Vec3::splat(-inner) / VOXEL_MM).ceil().as_ivec3()).0;
        let high = BrickCoord::containing((Vec3::splat(inner) / VOXEL_MM).floor().as_ivec3()).0;
        let mut proven = 0;
        for z in low.z..=high.z {
            for y in low.y..=high.y {
                for x in low.x..=high.x {
                    let coord = BrickCoord::new(x, y, z);
                    // Only the bricks wholly within the proven-inside box: one
                    // on the edge of it may legitimately be dense.
                    let origin = coord.origin().as_vec3() * VOXEL_MM;
                    let corner = coord.max_voxel().as_vec3() * VOXEL_MM;
                    if origin.min_element() < -inner || corner.max_element() > inner {
                        continue;
                    }
                    proven += 1;
                    assert!(
                        matches!(volume.brick(coord), Some(Brick::Uniform(_))),
                        "brick {coord:?} is inside the cube and is not a uniform tile"
                    );
                }
            }
        }
        assert!(proven > 0, "the fixture proved no brick interior, so it checked nothing");

        // The middle is the part that must not be absent: absent reads as empty
        // space, and a cube hollow in the middle still validates as watertight.
        assert!(
            volume.sample_world(Vec3::ZERO) < 0.0,
            "the middle of a solid cube reads as empty space"
        );
    }

    /// A budget assertion, pinned so that a change to the classification which
    /// quietly doubles what a small primitive costs shows up as a failure rather
    /// than as a slower session.
    ///
    /// A 5 mm sphere at a 0.25 mm voxel is 40 voxels across, straddling the
    /// lattice origin, so it falls into eight bricks and every one of them holds
    /// surface: 8 x 128 KB of voxels, near enough 1.00 MB.
    #[test]
    fn a_small_sphere_costs_eight_bricks_and_about_a_megabyte() {
        let volume = build(PrimitiveKind::Sphere, VOXEL, Vec3::ZERO, 5.0);
        let stats = volume.stats();

        assert_eq!(stats.dense_bricks + stats.uniform_bricks, 8, "a 5 mm sphere is eight bricks");
        let voxels = 8.0 * BRICK_VOXELS as f64 * 4.0;
        let megabyte = 1024.0 * 1024.0;
        assert!(
            (stats.resident_bytes as f64 - voxels).abs() < 0.1 * megabyte,
            "a 5 mm sphere costs {:.2} MB, not the 1.00 MB it is budgeted at",
            stats.resident_bytes as f64 / megabyte
        );
    }

    /// The classification is an optimisation over evaluating every voxel, so it
    /// has to produce exactly what evaluating every voxel would.
    ///
    /// The same check `clip.rs` keeps over its own brick classification, and for
    /// the same reason: this is the part that can silently skip a brick that
    /// mattered, and nothing else here would notice.
    ///
    /// **It walks [`brick_range`], not `volume.brick_coords()`.** The latter is
    /// `self.bricks.keys()` -- the bricks that were STORED -- so a build that
    /// wrongly left a brick out was compared against nothing, and this test
    /// reported ok on a volume that came out completely empty. Over the range
    /// `build` itself walks, an absent brick reads [`OUTSIDE`] through
    /// `sample_voxel` and fails the comparison at the first voxel whose true
    /// distance is inside the band.
    ///
    /// **The primitive is deliberately bigger than a brick.** At the 4 mm half
    /// extent this started with, the shape is half a brick across and the
    /// classification's margin is never the thing under test: shrinking `reach`
    /// by a fifth -- which culls bricks with surface in the corner -- passed. At
    /// 12 mm it fails, and the whole range is still 216 bricks a kind.
    #[test]
    fn the_classification_matches_evaluating_every_voxel() {
        let centre = Vec3::new(1.1, -0.7, 0.3);
        let half = 12.0;
        let (low_brick, high_brick) = brick_range(VOXEL, centre, half);
        for kind in PrimitiveKind::ALL {
            let volume = build(kind, VOXEL, centre, half);
            for bz in low_brick.z..=high_brick.z {
                for by in low_brick.y..=high_brick.y {
                    for bx in low_brick.x..=high_brick.x {
                        let origin = BrickCoord::new(bx, by, bz).origin();
                        for z in 0..BRICK_DIM {
                            for y in 0..BRICK_DIM {
                                for x in 0..BRICK_DIM {
                                    let voxel = origin + IVec3::new(x as i32, y as i32, z as i32);
                                    let at = voxel.as_vec3() * VOXEL;
                                    let wanted = (kind.distance(at - centre, half) / VOXEL)
                                        .clamp(INSIDE, OUTSIDE);
                                    let got = volume.sample_voxel(voxel);
                                    assert!(
                                        (got - wanted).abs() < 1.0e-4,
                                        "{kind:?} at {voxel:?}: field says {got}, the formula \
                                         says {wanted}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The sizes have to mean what they say, or a "10 mm cube" is a surprise.
    ///
    /// The tolerance is the narrow band and not slack: `surface_bounds` reports
    /// every voxel carrying a distance, and the field carries one for
    /// [`NARROW_BAND`] voxels either side of the surface, so a solid always
    /// measures a little over. What this pins is that it is not measuring a
    /// different shape.
    #[test]
    fn a_primitive_is_the_size_it_was_asked_for() {
        let half = 8.0;
        let slack = (2.0 * NARROW_BAND + 1.0) * VOXEL;
        for kind in PrimitiveKind::ALL {
            let volume = build(kind, VOXEL, Vec3::ZERO, half);
            let (low, high) = volume.surface_bounds().expect("a primitive has a surface");
            let measured = (high - low).max_element();
            assert!(
                measured >= 2.0 * half - VOXEL && measured <= 2.0 * half + slack,
                "{kind:?} measures {measured} mm across, not {}",
                2.0 * half
            );
        }
    }

    /// Placing off the origin is the whole point, so the arithmetic gets its own
    /// check rather than being inferred from the application test above it.
    #[test]
    fn a_primitive_is_placed_one_brick_clear_of_the_document() {
        let mut seed = Volume::new(VOXEL);
        seed.seed_sphere(Vec3::ZERO, 20.0);
        let doc = Document::from_volume(seed);

        let (centre, half) = placement(&doc, 30.0);
        let (_, high) = doc.world_bounds().expect("the seeded document has bricks");
        let clearance = (centre.x - half) - high.x;
        let brick_mm = BRICK_DIM as f32 * VOXEL;
        assert!(
            clearance >= brick_mm - 1.0e-4,
            "the primitive's low face is {clearance} mm clear, less than one {brick_mm} mm brick"
        );
        assert!(half > 0.0, "the primitive has no size");
    }

    /// An empty document has no box to be clear of, and must not produce a
    /// centre made of infinities.
    #[test]
    fn an_empty_document_places_a_primitive_at_the_origin() {
        let doc = Document::new(VOXEL);
        let (centre, half) = placement(&doc, 30.0);
        assert_eq!(centre, Vec3::ZERO);
        assert!(half.is_finite() && half > 0.0, "an empty document sized a primitive at {half}");
    }

    /// **Pressing `+` again and again must not make the primitive bigger every
    /// time**, and the shipped version did exactly that.
    ///
    /// Sized against the document's union box, eight presses on a 30 mm ball at
    /// a 0.25 mm voxel gave cubes of 37, 48, 66, 93, 128, 181, 249 and 342 mm --
    /// measured, by adding each one to the document and asking again. The sixth
    /// is some 460 MB of bricks allocated inside one `build` call, and there is
    /// no press at which it stops: `MAX_BODIES` is 64 and the machine gives out
    /// a very long way before the sixty-fourth row.
    ///
    /// Bodies are added for real rather than faked, because the property is
    /// about what `placement` reads back out of a document it has grown. At a
    /// 1.0 mm voxel, for the same reason the shell test uses one: eight cubes at
    /// the design's 0.25 mm would be a quarter of a gigabyte of test fixture.
    #[test]
    fn adding_primitive_after_primitive_does_not_grow_them() {
        const VOXEL_MM: f32 = 1.0;
        let mut seed = Volume::new(VOXEL_MM);
        seed.seed_sphere(Vec3::ZERO, 30.0);
        let mut doc = Document::from_volume(seed);

        let mut first = None;
        for press in 0..8 {
            let (centre, half) = placement(&doc, 30.0);
            let first = *first.get_or_insert(half);
            assert!(
                half <= first + VOXEL_MM,
                "press {press} sized a cube {half} mm, up from the first at {first} mm"
            );
            doc.add_body("Cube", build(PrimitiveKind::Cube, VOXEL_MM, centre, half));
        }
    }

    /// The refusal is only worth having if it fires BEFORE the allocation, so
    /// what it is fed must not be an underestimate of what `build` then does.
    ///
    /// Both estimators are [`crate::voxelise`]'s, which were tuned against
    /// imported meshes rather than against boxes and balls, so this is the check
    /// that they transfer. Generous is fine and is what they are for; short is
    /// the failure, because a guard that says a 54 GB cube fits is not a guard.
    #[test]
    fn the_cost_estimate_is_never_less_than_the_bricks_that_appear() {
        const VOXEL_MM: f32 = 0.5;
        for kind in PrimitiveKind::ALL {
            for half in [4.0, 16.0, 40.0] {
                let (bytes, vertices) = cost(kind, VOXEL_MM, half);
                let volume = build(kind, VOXEL_MM, Vec3::ZERO, half);
                let actual = volume.stats().resident_bytes as f64;
                assert!(
                    bytes >= actual,
                    "{kind:?} at half {half} mm: predicted {:.1} MB, allocated {:.1} MB",
                    bytes / (1024.0 * 1024.0),
                    actual / (1024.0 * 1024.0)
                );
                let (mesh, _) = volume.export_mesh();
                assert!(
                    vertices >= mesh.positions.len() as f64,
                    "{kind:?} at half {half} mm: predicted {vertices:.0} vertices, meshed {}",
                    mesh.positions.len()
                );
            }
        }
    }
}
