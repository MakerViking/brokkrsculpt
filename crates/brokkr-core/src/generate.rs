// SPDX-License-Identifier: AGPL-3.0-only

//! Masks generated from the geometry rather than painted onto it.
//!
//! Four recipes, one entry point -- [`Volume::generated_mask`] -- and one
//! prediction of what each would cost before it runs,
//! [`Volume::generated_mask_demand`]. Every one of them REPLACES the body's
//! mask outright rather than blending into it, and every one of them lands at
//! normal polarity: a generator produces protection directly, so carrying an
//! inversion into it would read the values it wrote back upside down. Invert
//! afterwards if that is what was wanted; it is still one bool.
//!
//! # Why a generated mask is the ordinary case for memory, not the exceptional
//! one
//!
//! A hand-painted mask touches a few percent of the surface bricks and collapses
//! to tiles almost everywhere. [`MaskRecipe::Cavity`] does not: measured on the
//! reference dragon it writes 7,306,957 non-zero voxels spread across all 6,806
//! of its bricks, so essentially every band brick comes out dense and the +25%
//! of [`crate::mask`]'s arithmetic is what you actually pay. That is why the
//! refusal in front of these is part of the feature rather than a nicety, and
//! why [`Volume::generated_mask_demand`] assumes a dense mask brick for every
//! field brick instead of hoping for collapse.
//!
//! # Cavity and smoothness are one pass, and the pass is seven reads
//!
//! The stored field is an exact Euclidean signed distance **in voxels** as
//! imported (`voxelise` pins that with its own test), so for a true distance
//! field the Laplacian is the sum of the principal curvatures and a 7-point
//! stencil over an apron [`Volume::gather_apron`] already produces is the whole
//! algorithm. The field is negative inside, so a positive Laplacian is CONVEX
//! and a negative one is CONCAVE: a solid sphere of radius `r` reads a uniform
//! `+2/r` and must mask nothing, a spherical void reads `-2/r` and must mask
//! fully. Both of those are one character away from each other and both produce
//! plausible-looking output, which is why each has its own test below.
//!
//! ## The premise degrades where the user has been working, and by how much
//!
//! `brush.rs` records that none of the brushes preserves the eikonal property:
//! after many overlapping stamps the gradient magnitude drifts from 1. Writing
//! the field as `d = f(s)` over the true distance `s` gives
//! `laplacian(d) = f'' + |grad d| * (k1 + k2)`, so dividing the stencil by
//! `|grad d|` -- which the same six samples already give, at the cost of one
//! square root -- makes the curvature term exact again and leaves only `f''`.
//! **That residual is real and is not corrected here**: a region the user has
//! stamped thirty times over can read a curvature that is wrong by the second
//! derivative of its own remapping. Clamping to the narrow band bounds it, so
//! the error is a curvature read low or high rather than a sign flipped, and
//! the sculpted-fixture test below is what holds that claim up.
//!
//! ## The slider is in millimetres, and that is where this beats ZBrush
//!
//! ZBrush's cavity masking is resolution-relative, which is its most documented
//! cavity complaint: the same model at a different subdivision level masks
//! differently. A body here has one fixed voxel size, so "mask features narrower
//! than 1.5 mm" is stable, means the same thing at every voxel size (the
//! threshold and the measured curvature both scale with the voxel, and the ratio
//! does not), and is directly meaningful to somebody choosing a nozzle. One
//! cavity mask ships and not two; ZBrush has two that behave differently and
//! that is itself a documented confusion.

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use crate::apron::ApronBuffer;
use crate::brick::{BRICK_DIM, BRICK_VOXELS, BrickCoord, NARROW_BAND, brick_index};
use crate::cavity::thin_voxels;
use crate::clip::ClipPlane;
use crate::mask::{MaskBrick, MaskField, MaskFilter, PROTECTED, UNMASKED};
use crate::volume::Volume;

/// How close to the surface the curvature stencil is trusted, in voxels.
///
/// **The clamp, and it is the second of the two one-character bugs.** The field
/// saturates at `+/- NARROW_BAND`, so the second derivative has a step at the
/// band edge and a naive Laplacian reads the entire iso-shell as maximum
/// curvature -- a plausible-looking mask that is really a picture of the band.
/// Nothing further out than this is measured, and nothing within one voxel of
/// the shell is masked.
const CURVATURE_GATE: f32 = 2.0;

/// Where the generated protection starts fading out toward that gate, in
/// voxels.
///
/// A hard edge at [`CURVATURE_GATE`] would be a step in the mask, which is a
/// fold in the geometry under Move -- see [`crate::mask`]'s "soft, not a
/// bitmask". The fade makes the mask continuous across the band as well as
/// along the surface.
const FADE_FROM: f32 = 1.0;

/// The narrowest and widest gradient magnitude the normalisation will believe.
///
/// A sculpted field can flatten to a plateau, where dividing by the measured
/// gradient turns a rounding error into an enormous curvature; and it can
/// steepen, where dividing over-damps. Both are clamped rather than trusted,
/// which is what keeps the eikonal drift a wrong magnitude instead of a wrong
/// sign.
const GRADIENT_FLOOR: f32 = 0.5;
const GRADIENT_CEILING: f32 = 2.0;

/// The thickest wall a thickness mask can be asked about, in voxels.
///
/// **Hard, and the interface has to say so.** A seed voxel has to sit at least
/// half the wall below the surface, and past [`NARROW_BAND`] every interior
/// sample reads [`crate::INSIDE`] and carries no depth at all. Six voxels is
/// 3 mm at a 0.5 mm voxel but 0.34 mm at the 0.0565 mm dragon -- right at the
/// size of the strands the feature exists for -- so the number is expressed in
/// voxels with the millimetres beside it and never the other way round.
pub const MAX_THICKNESS_VOXELS: u32 = 2 * NARROW_BAND as u32;

/// How far a six-connected dilation has to run to cover a Euclidean radius.
///
/// See [`Volume::generated_mask`]'s thickness arm for what this buys and what
/// it costs.
const SQRT_3: f32 = 1.732_050_8;

/// The smallest feature a cavity or smoothness mask can be asked about, in
/// voxels.
///
/// Below two voxels the curvature the stencil can represent is bounded by the
/// lattice rather than by the model, so asking for it would return the same
/// answer with a more confident-looking number on the slider.
const FINEST_FEATURE_VOXELS: f32 = 2.0;

/// One way of deriving a mask from the geometry it will protect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaskRecipe {
    /// Protect what is concave at a scale narrower than `feature_mm`.
    ///
    /// Crevices, pits, the inside of a fold -- the places a print will lose
    /// detail and the places a sculptor wants held still while the form around
    /// them moves.
    Cavity {
        /// The width of the narrowest feature that comes out fully protected.
        feature_mm: f32,
    },
    /// Protect what is FLAT at that scale: the complement of curvature, either
    /// sign.
    ///
    /// The same pass read differently, which is why they ship together. Where
    /// cavity holds the detail still, this holds the smooth ground still and
    /// leaves the detail free.
    Smoothness {
        /// The width below which a feature counts as detail rather than ground.
        feature_mm: f32,
    },
    /// Protect solid material thinner than `voxels` across.
    ///
    /// The selection problem behind hair strands, fins and lattices, answered
    /// as a selection rather than as a repair.
    Thickness {
        /// Full wall thickness, in voxels, clamped to [`MAX_THICKNESS_VOXELS`].
        voxels: u32,
    },
    /// Protect one side of a plane, feathered across it.
    ///
    /// No new gesture and no new geometry code: the plane is the one the cut
    /// tool already builds from a drag, and the brick classification is the
    /// cut's own -- so an interior tile is never promoted to dense and the cost
    /// is proportional to the boundary rather than to the volume.
    Halfspace {
        /// The plane, in the cut's own orientation: positive distance is the
        /// side a cut would take away, and that is the side this protects.
        plane: ClipPlane,
        /// How far the protection ramps from nothing to full, in millimetres.
        feather_mm: f32,
    },
}

impl MaskRecipe {
    /// The verb as it reads inside a refusal, which is a lower-case sentence.
    pub fn verb(self) -> &'static str {
        match self {
            MaskRecipe::Cavity { .. } => "mask by cavity",
            MaskRecipe::Smoothness { .. } => "mask by smoothness",
            MaskRecipe::Thickness { .. } => "mask by thickness",
            MaskRecipe::Halfspace { .. } => "mask that half",
        }
    }

    /// What the status line says once it has run.
    pub fn done(self) -> &'static str {
        match self {
            MaskRecipe::Cavity { .. } => "masked the cavities",
            MaskRecipe::Smoothness { .. } => "masked the smooth areas",
            MaskRecipe::Thickness { .. } => "masked the thin material",
            MaskRecipe::Halfspace { .. } => "masked that half",
        }
    }

    /// Whether this recipe needs a copy of the field to run.
    ///
    /// Only the thickness walk does, and it is the difference between a
    /// prediction of 1.25 R and one of 2.25 R -- which on a document near the
    /// ceiling is the difference between a refusal and an out-of-memory kill.
    fn copies_the_field(self) -> bool {
        matches!(self, MaskRecipe::Thickness { .. })
    }
}

/// Which way one curvature pass is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Curvature {
    /// Concave, and nothing else.
    Cavity,
    /// Neither concave nor convex.
    Smoothness,
}

impl Volume {
    /// A whole new mask derived from this body's geometry.
    ///
    /// Replaces whatever the body carries -- see the module documentation for
    /// why it also replaces the polarity. The result is already feathered, so
    /// nothing downstream has to soften it, and it is safe to hand straight to
    /// [`Volume::replace_mask`].
    ///
    /// **Ask [`Volume::generated_mask_demand`] first.** This allocates a mask
    /// over essentially every band brick and, for
    /// [`MaskRecipe::Thickness`], a copy of the field besides.
    pub fn generated_mask(&self, recipe: MaskRecipe) -> MaskField {
        match recipe {
            MaskRecipe::Cavity { feature_mm } => self.curvature_mask(Curvature::Cavity, feature_mm),
            MaskRecipe::Smoothness { feature_mm } => {
                self.curvature_mask(Curvature::Smoothness, feature_mm)
            }
            MaskRecipe::Thickness { voxels } => self.thickness_mask(voxels),
            MaskRecipe::Halfspace { plane, feather_mm } => self.halfspace_mask(plane, feather_mm),
        }
    }

    /// Bytes a generated mask would add at its peak, worked out from the brick
    /// census alone.
    ///
    /// **Deliberately pessimistic about collapse.** A generated mask fills
    /// essentially every band brick, so assuming the tiles a hand-painted mask
    /// collapses to would under-predict by four times at exactly the moment the
    /// document is largest. What is counted is a dense mask brick for every
    /// field brick, the map that holds them, and -- for the one recipe that
    /// needs it -- the copy of the field the walk makes.
    ///
    /// The mask the body already carries is NOT subtracted: it is still
    /// resident while the new one is built, and history holds it afterwards.
    pub fn generated_mask_demand(&self, recipe: MaskRecipe) -> usize {
        let stats = self.stats();
        let bricks = self.brick_count();
        let entry = size_of::<BrickCoord>() + size_of::<MaskBrick>();
        // The map is grown to the brick count and a hash map keeps headroom
        // over that, so the entry count is doubled rather than taken at face
        // value -- and a floor besides, because a SMALL map rounds its capacity
        // up to a bucket count that has nothing to do with what is in it, which
        // is how a prediction of a nearly-empty body comes out under the truth
        // by a few dozen bytes.
        let mask = bricks * BRICK_VOXELS + bricks * entry * 2 + 4096;
        if recipe.copies_the_field() {
            // A dense copy of every dense brick, plus 4 KB of reach bits each.
            mask + stats.dense_bricks * BRICK_VOXELS * size_of::<f32>() + bricks * BRICK_VOXELS / 8
        } else {
            mask
        }
    }

    // ----------------------------------------------- cavity and smoothness

    /// One curvature pass, read as cavity or as smoothness.
    fn curvature_mask(&self, mode: Curvature, feature_mm: f32) -> MaskField {
        let voxel_size = self.voxel_size();
        let feature_mm = feature_mm.max(FINEST_FEATURE_VOXELS * voxel_size);
        // The curvature magnitude that means "exactly this feature size": a
        // feature `w` mm wide has radius `w / 2` mm, which is
        // `w / (2 * voxel)` voxels, and a sphere of radius `R` voxels reads
        // `2 / R`. The voxel size cancels out of the ratio below, which is what
        // makes a millimetre slider resolution-independent.
        let full = 4.0 * voxel_size / feature_mm;

        let coords: Vec<BrickCoord> = self.brick_coords().collect();
        let built: Vec<(BrickCoord, MaskBrick)> = coords
            .par_iter()
            // One apron per WORKER. It is 157 KB, and a body of the size this
            // is written for is tens of thousands of bricks.
            .map_init(ApronBuffer::new, |apron, coord| {
                self.gather_apron(*coord, apron);
                curvature_brick(apron, mode, full).map(|brick| (*coord, brick))
            })
            .flatten()
            .collect();
        self.mask().generated(built)
    }

    // --------------------------------------------------------- thickness

    /// Protection over solid material thinner than `voxels` across.
    fn thickness_mask(&self, voxels: u32) -> MaskField {
        let voxels = voxels.clamp(1, MAX_THICKNESS_VOXELS);
        // A wall of full thickness `t` has a voxel `t / 2` deep at its centre,
        // so the seed test is that depth. The REACH is not that same number,
        // and getting it wrong is what calls the skin of a solid ball thin: the
        // dilation is six-connected, so it measures Manhattan steps, and a wall
        // lying at 45 degrees to every axis gains only `1 / sqrt(3)` of depth
        // per step. Rounding the reach up by `sqrt(3)` is what makes "thicker
        // than this is never selected" true at every orientation.
        //
        // The price is anisotropy in the other direction: along an axis the
        // reach is up to `sqrt(3)` times the half-thickness, so a genuinely thin
        // fin growing out of thick material stays unselected for that far from
        // the join. That is the ZBrush behaviour too, and it is the side to err
        // on -- a selection that misses the root of a strand is a nuisance, one
        // that includes the skin of everything is unusable.
        let half = voxels as f32 / 2.0;
        let thin = thin_voxels(self, -half, (half * SQRT_3).ceil() as usize);

        let built: Vec<(BrickCoord, MaskBrick)> = thin
            .into_par_iter()
            .map(|(coord, thin)| {
                let mut brick = MaskBrick::dense_filled(UNMASKED);
                let data = brick.make_dense();
                for (voxel, value) in data.iter_mut().enumerate() {
                    if thin.has(voxel) {
                        *value = PROTECTED;
                    }
                }
                (coord, brick)
            })
            .collect();

        // **The feather is one pass of the existing Blur and not a second
        // kernel.** A reach test answers yes or no, and a step in the mask is a
        // fold in the geometry under Move; `filtered` is already the 3x3x3 box
        // over the soft mask, already tested, and already skips the bricks
        // where protection does not change.
        self.mask().generated(built).filtered(MaskFilter::Blur, 1.0)
    }

    // --------------------------------------------------------- half-space

    /// Protection over the side of `plane` a cut would have removed.
    ///
    /// The classification is the cut's own -- wholly one side, wholly the
    /// other, or crossing -- so only the crossing bricks are ever promoted to
    /// dense, and a mask over half a dragon costs its boundary rather than its
    /// volume.
    fn halfspace_mask(&self, plane: ClipPlane, feather_mm: f32) -> MaskField {
        let voxel_size = self.voxel_size();
        // A feather narrower than a voxel is a step by another name, so the
        // floor is one voxel however small the caller asked for.
        let feather = feather_mm.max(voxel_size);
        let brick_mm = BRICK_DIM as f32 * voxel_size;
        // A brick spans `BRICK_DIM` voxel positions, so the distance from its
        // first to its last sample is one voxel short of its nominal size --
        // the same half-extent `Volume::clip` uses.
        let half = Vec3::splat(0.5 * (brick_mm - voxel_size));

        let coords: Vec<BrickCoord> = self.brick_coords().collect();
        let built: Vec<(BrickCoord, MaskBrick)> = coords
            .par_iter()
            .filter_map(|coord| {
                let origin = coord.origin();
                let centre = origin.as_vec3() * voxel_size + half;
                let (nearest, farthest) = plane.range_over_box(centre, half);
                if farthest <= -feather {
                    // Wholly on the kept side: no entry at all, which is what
                    // keeps this proportional to the boundary.
                    return None;
                }
                if nearest >= feather {
                    return Some((*coord, MaskBrick::Uniform(PROTECTED)));
                }
                let mut brick = MaskBrick::dense_filled(UNMASKED);
                let data = brick.make_dense();
                for z in 0..BRICK_DIM {
                    for y in 0..BRICK_DIM {
                        for x in 0..BRICK_DIM {
                            let at = (origin + IVec3::new(x as i32, y as i32, z as i32)).as_vec3()
                                * voxel_size;
                            data[brick_index(x, y, z)] = feathered(plane.distance(at) / feather);
                        }
                    }
                }
                Some((*coord, brick))
            })
            .collect();
        self.mask().generated(built)
    }
}

/// One brick of a curvature mask, or `None` when nothing in it is protected.
fn curvature_brick(apron: &ApronBuffer, mode: Curvature, full: f32) -> Option<MaskBrick> {
    let mut brick = MaskBrick::dense_filled(UNMASKED);
    let data = brick.make_dense();
    let mut any = false;

    for z in 0..BRICK_DIM {
        for y in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                // The brick's own voxels start at (1, 1, 1) of the apron, so
                // every neighbour below is inside it by construction.
                let (ax, ay, az) = (x + 1, y + 1, z + 1);
                let here = apron.get(ax, ay, az);
                // `is_finite` first, so a NaN out of a damaged file is skipped
                // rather than masked: every comparison against a NaN is false,
                // so a bare `>=` test would let it through.
                if !here.is_finite() || here.abs() >= CURVATURE_GATE {
                    continue;
                }
                let px = apron.get(ax + 1, ay, az);
                let nx = apron.get(ax - 1, ay, az);
                let py = apron.get(ax, ay + 1, az);
                let ny = apron.get(ax, ay - 1, az);
                let pz = apron.get(ax, ay, az + 1);
                let nz = apron.get(ax, ay, az - 1);
                // A stencil with one foot on the saturated shell reads the
                // band's own edge as curvature. The gate above bounds the
                // centre; this bounds the arms, which a sculpted field can
                // saturate a voxel sooner than an exact one would.
                if [px, nx, py, ny, pz, nz]
                    .iter()
                    .any(|value| !value.is_finite() || value.abs() >= NARROW_BAND)
                {
                    continue;
                }

                let laplacian = px + nx + py + ny + pz + nz - 6.0 * here;
                let gradient = 0.5 * Vec3::new(px - nx, py - ny, pz - nz);
                let scale = gradient.length().clamp(GRADIENT_FLOOR, GRADIENT_CEILING);
                // The field is negative inside, so this is positive on a convex
                // surface and negative in a cavity.
                let curvature = laplacian / scale;

                let strength = match mode {
                    Curvature::Cavity => (-curvature / full).clamp(0.0, 1.0),
                    Curvature::Smoothness => 1.0 - (curvature.abs() / full).clamp(0.0, 1.0),
                };
                let across =
                    ((CURVATURE_GATE - here.abs()) / (CURVATURE_GATE - FADE_FROM)).clamp(0.0, 1.0);
                let protection = smoothstep(strength) * smoothstep(across);
                let byte = (protection * PROTECTED as f32).round() as u8;
                if byte != UNMASKED {
                    data[brick_index(x, y, z)] = byte;
                    any = true;
                }
            }
        }
    }

    if !any {
        return None;
    }
    Some(match brick.is_collapsible() {
        Some(byte) => MaskBrick::Uniform(byte),
        None => brick,
    })
}

/// One voxel's protection from its signed distance to a plane, in feather
/// widths.
///
/// Full protection a whole feather past the plane, none a whole feather behind
/// it, and a smoothstep across -- so the boundary carries real intermediate
/// values rather than a step, which is the rule every write to a mask is under.
#[inline]
fn feathered(distance: f32) -> u8 {
    let across = (distance * 0.5 + 0.5).clamp(0.0, 1.0);
    (smoothstep(across) * PROTECTED as f32).round() as u8
}

/// The usual cubic ease, on an input already in `0..=1`.
#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::{Brick, INSIDE, OUTSIDE};
    use crate::brush::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp, Symmetry};
    use crate::mask::MaskField;

    const VOXEL: f32 = 0.5;

    /// A solid sphere, which is convex at every point of its surface.
    fn sphere(voxel: f32, radius: f32) -> Volume {
        let mut volume = Volume::new(voxel);
        volume.seed_sphere(Vec3::ZERO, radius);
        volume
    }

    /// A solid block with a spherical void hollowed out of its middle, which is
    /// concave at every point of the void's wall.
    ///
    /// The block reaches well past the void on every side, so the field near
    /// the wall is exactly the void's own signed distance and the curvature
    /// there is exactly `-2 / radius`. That is what makes the sign assertion an
    /// assertion about the algorithm rather than about a fixture.
    fn block_with_a_void(voxel: f32, half_mm: f32, radius: f32) -> Volume {
        let mut volume = Volume::new(voxel);
        let reach = half_mm + NARROW_BAND * voxel * 2.0;
        let lo = (Vec3::splat(-reach) / voxel).floor().as_ivec3();
        let hi = (Vec3::splat(reach) / voxel).ceil().as_ivec3();
        volume.edit_voxels(lo, hi, |_, position, _| {
            let outside_the_block =
                (position.abs() - Vec3::splat(half_mm)).max(Vec3::ZERO).length()
                    + (position.abs() - Vec3::splat(half_mm)).max_element().min(0.0);
            let inside_the_void = radius - position.length();
            (outside_the_block.max(inside_the_void) / voxel).clamp(INSIDE, OUTSIDE)
        });
        volume
    }

    /// Every lattice cell of `volume` whose stored distance is within `band`
    /// voxels of the surface.
    fn cells_near_the_surface(volume: &Volume, band: f32) -> Vec<IVec3> {
        let mut cells = Vec::new();
        for coord in volume.brick_coords() {
            let origin = coord.origin();
            for z in 0..BRICK_DIM as i32 {
                for y in 0..BRICK_DIM as i32 {
                    for x in 0..BRICK_DIM as i32 {
                        let cell = origin + IVec3::new(x, y, z);
                        if volume.sample_voxel(cell).abs() <= band {
                            cells.push(cell);
                        }
                    }
                }
            }
        }
        cells
    }

    // --------------------------------------------------------- the two signs

    /// The first of the two one-character bugs: a sign flip here masks exactly
    /// the wrong half of every model and the output still looks plausible.
    #[test]
    fn a_solid_sphere_is_convex_everywhere_and_the_cavity_mask_leaves_it_free() {
        let volume = sphere(VOXEL, 6.0);
        let mask = volume.generated_mask(MaskRecipe::Cavity { feature_mm: 16.0 });
        assert!(
            mask.is_free(),
            "a solid sphere reads a uniform +2/r, which is convex, and must mask nothing"
        );
    }

    /// The other half of that pair, and it has to be a separate fixture: a
    /// cavity mask that masked nothing at all would pass the sphere test.
    #[test]
    fn a_spherical_void_is_concave_everywhere_and_the_cavity_mask_protects_its_wall() {
        let volume = block_with_a_void(VOXEL, 10.0, 4.0);
        let mask = volume.generated_mask(MaskRecipe::Cavity { feature_mm: 16.0 });
        assert!(!mask.is_free(), "a spherical void reads -2/r, which is concave, and must mask");

        let wall: Vec<IVec3> = cells_near_the_surface(&volume, 0.5)
            .into_iter()
            .filter(|cell| {
                // The void's own wall and not the block's outer faces, which
                // are flat and must stay free.
                (cell.as_vec3() * VOXEL).length() < 8.0
            })
            .collect();
        assert!(wall.len() > 100, "the fixture has no void wall to measure: {} cells", wall.len());
        let weakest = wall.iter().map(|cell| mask.at(*cell)).min().expect("the wall is not empty");
        assert_eq!(weakest, PROTECTED, "every voxel of the void wall has to come out fully masked");
    }

    /// The block's own flat faces are the control for the test above: they sit
    /// in the same volume, in the same pass, and must come out untouched.
    #[test]
    fn the_flat_faces_around_a_void_stay_free_while_the_void_is_masked() {
        let volume = block_with_a_void(VOXEL, 10.0, 4.0);
        let mask = volume.generated_mask(MaskRecipe::Cavity { feature_mm: 16.0 });
        let faces: Vec<IVec3> = cells_near_the_surface(&volume, 0.5)
            .into_iter()
            .filter(|cell| {
                let at = cell.as_vec3() * VOXEL;
                // Well inside one face and well away from the convex edges.
                at.z.abs() > 9.0 && at.x.abs() < 5.0 && at.y.abs() < 5.0
            })
            .collect();
        assert!(faces.len() > 50, "the fixture has no flat face to measure: {}", faces.len());
        let strongest = faces.iter().map(|cell| mask.at(*cell)).max().expect("not empty");
        assert_eq!(strongest, UNMASKED, "a flat face has no curvature and must not be masked");
    }

    // ------------------------------------------------------------- the clamp

    /// The second one-character bug. Without the gate the second derivative's
    /// step at the band edge reads as maximum curvature everywhere, and the
    /// result is a picture of the narrow band wearing a cavity mask's name.
    ///
    /// **The bound below is [`NARROW_BAND`] and deliberately not
    /// [`CURVATURE_GATE`].** Written against the gate this assertion is
    /// circular -- the same constant decides both which voxels are measured and
    /// which voxels are checked, so setting the gate to `NARROW_BAND` (exactly
    /// the failure the doc above describes) leaves it green while 312 voxels of
    /// this fixture past `|d| = 2` come out masked. `NARROW_BAND` is a property
    /// of the FIELD -- where the stored distance saturates -- so an assertion
    /// in terms of it is an assertion about the output, and it is the plan's
    /// sentence read literally: nothing within one voxel of the shell.
    #[test]
    fn nothing_within_one_voxel_of_the_saturated_shell_is_ever_masked() {
        let volume = block_with_a_void(VOXEL, 10.0, 4.0);
        for recipe in
            [MaskRecipe::Cavity { feature_mm: 16.0 }, MaskRecipe::Smoothness { feature_mm: 16.0 }]
        {
            let mask = volume.generated_mask(recipe);
            for coord in volume.brick_coords() {
                let origin = coord.origin();
                for z in 0..BRICK_DIM as i32 {
                    for y in 0..BRICK_DIM as i32 {
                        for x in 0..BRICK_DIM as i32 {
                            let cell = origin + IVec3::new(x, y, z);
                            let distance = volume.sample_voxel(cell);
                            if distance.abs() >= NARROW_BAND - 1.0 {
                                assert_eq!(
                                    mask.at(cell),
                                    UNMASKED,
                                    "{recipe:?} masked {cell:?}, which is at {distance} and so \
                                     within one voxel of the saturated shell"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The rule every write to a mask is under, checked on the path that writes
    /// the most of them: protection has to arrive at a feathered edge.
    #[test]
    fn a_generated_cavity_mask_arrives_feathered_and_not_as_a_step() {
        let volume = block_with_a_void(VOXEL, 10.0, 4.0);
        let mask = volume.generated_mask(MaskRecipe::Cavity { feature_mm: 16.0 });
        let mut middling = 0;
        for cell in cells_near_the_surface(&volume, NARROW_BAND) {
            let protection = mask.at(cell);
            if protection > UNMASKED && protection < PROTECTED {
                middling += 1;
            }
        }
        assert!(
            middling > 100,
            "a mask that only ever writes 0 or 255 is a bitmask: {middling} intermediate voxels"
        );
    }

    // ------------------------------------------- the millimetre is the point

    /// The strongest and the average protection over the void's own wall, over
    /// the cells the block's flat faces cannot reach.
    ///
    /// The mean is here as well as the peak because a peak is one voxel's worth
    /// of lattice luck; both numbers moving together is what says the threshold
    /// landed in the same place rather than that one cell happened to.
    fn wall_protection(volume: &Volume, mask: &MaskField) -> (u8, u8) {
        let wall: Vec<IVec3> = cells_near_the_surface(volume, 0.5)
            .into_iter()
            .filter(|cell| (cell.as_vec3() * volume.voxel_size()).length() < 5.0)
            .collect();
        assert!(wall.len() > 100, "the fixture has no void wall to measure: {}", wall.len());
        let peak = wall.iter().map(|cell| mask.at(*cell)).max().expect("the wall is not empty");
        let total: u64 = wall.iter().map(|cell| mask.at(*cell) as u64).sum();
        (peak, (total / wall.len() as u64) as u8)
    }

    /// What makes the slider beat ZBrush's resolution-relative one: the same
    /// physical model at two voxel sizes has to give the same millimetre mask.
    ///
    /// **The feature size is chosen to land mid-range and the test asserts on
    /// the protection VALUE, and both of those are the point.** Written at a
    /// feature size that saturates -- 12 mm against this 3 mm void gives a
    /// strength ratio of 2.0, which clamps -- both runs come out at 255 on the
    /// wall, the only thing left to compare is where the narrow band ends, and
    /// the `voxel_size` factor in `full` becomes invisible: deleting it (the
    /// literal `0.5`, so the threshold is right at 0.5 mm and wrong by 2x at
    /// 0.25 mm) leaves a saturated test bit-identically green. At `feature_mm`
    /// 3.0 the ratio is `feature / (2 * radius)` = 0.5 and the wall reads
    /// 141/125 coarse against 136/127 fine, while that same deletion drops the
    /// fine run to 43/39.
    #[test]
    fn the_same_model_at_two_voxel_sizes_gives_the_same_millimetre_mask() {
        let recipe = MaskRecipe::Cavity { feature_mm: 3.0 };
        let coarse = block_with_a_void(0.5, 8.0, 3.0);
        let fine = block_with_a_void(0.25, 8.0, 3.0);
        let coarse_mask = coarse.generated_mask(recipe);
        let fine_mask = fine.generated_mask(recipe);

        let (coarse_peak, coarse_mean) = wall_protection(&coarse, &coarse_mask);
        let (fine_peak, fine_mean) = wall_protection(&fine, &fine_mask);
        for (peak, which) in [(coarse_peak, "0.5 mm"), (fine_peak, "0.25 mm")] {
            assert!(
                peak > UNMASKED && peak < PROTECTED,
                "the {which} run saturated at {peak}, so this fixture can no longer see the \
                 threshold at all -- pick a feature size that lands mid-range"
            );
        }
        // Twelve out of 255 is under 5%, which is lattice noise between a 0.5 mm
        // and a 0.25 mm sampling of the same sphere. A missing voxel-size factor
        // is a factor of two and lands nowhere near it.
        const AGREEMENT: i32 = 12;
        assert!(
            (coarse_peak as i32 - fine_peak as i32).abs() <= AGREEMENT
                && (coarse_mean as i32 - fine_mean as i32).abs() <= AGREEMENT,
            "the same model at 0.5 mm and 0.25 mm protected its wall to {coarse_peak}/\
             {coarse_mean} and {fine_peak}/{fine_mean}"
        );

        // And the protection has to reach the same distance in millimetres, not
        // just the same height: the radius at which it crosses half, walked out
        // from the void's centre along an axis.
        let crossing = |volume: &Volume, mask: &MaskField| {
            let voxel = volume.voxel_size();
            let mut last = 0.0_f32;
            let mut at = 0.0_f32;
            while at < 6.0 {
                let cell = (Vec3::new(at, 0.0, 0.0) / voxel).round().as_ivec3();
                if mask.at(cell) >= 128 {
                    last = at;
                }
                at += voxel * 0.5;
            }
            last
        };
        let a = crossing(&coarse, &coarse_mask);
        let b = crossing(&fine, &fine_mask);
        assert!(a > 0.0 && b > 0.0, "neither run masked anything: {a} and {b}");
        assert!(
            (a - b).abs() <= 0.5,
            "the same model at 0.5 mm and 0.25 mm masked out to {a} mm and {b} mm"
        );
    }

    // --------------------------------------------------- the eikonal residual

    /// The premise degrades on sculpted geometry and the module doc says by how
    /// much. This is what holds that claim up: the solid-sphere case is an
    /// exact distance field and cannot see the drift at all.
    ///
    /// **A groove and not a pit, and that is a finding rather than a
    /// convenience.** Thirty stamps at ONE point drive the field from `INSIDE`
    /// to `+2.45` across a single voxel -- a gradient of 2.7 per voxel where an
    /// exact field has 1 -- and leave no sample inside the trusted band at all,
    /// so the curvature there is not read low, it is not read. That is the far
    /// end of the residual the module documents, it is reached by a gesture no
    /// stroke makes, and the honest fixture is the one a stroke does make.
    #[test]
    fn a_heavily_sculpted_sphere_still_tracks_the_groove_carved_into_it() {
        const RADIUS: f32 = 8.0;
        let mut volume = sphere(VOXEL, RADIUS);
        let brush =
            Brush { kind: BrushKind::Draw, radius: 2.5, strength: 0.25, ..Brush::default() };
        let mut scratch = BrushScratch::default();
        // Thirty overlapping stamps walked across the +Z cap, which is what a
        // real stroke lays down and what takes the gradient magnitude off 1.
        for step in 0..30 {
            let x = -4.0 + step as f32 * (8.0 / 29.0);
            let at = Vec3::new(x, 0.0, (RADIUS * RADIUS - x * x).sqrt());
            let stamp =
                Stamp::new(at, at.normalize(), BrushDirection::Subtract).with_tangent(Vec3::X);
            brush.apply_symmetric(
                &mut volume,
                &stamp,
                Symmetry::default(),
                Vec3::ZERO,
                &mut scratch,
            );
        }

        let mask = volume.generated_mask(MaskRecipe::Cavity { feature_mm: 6.0 });
        assert!(!mask.is_free(), "a carved groove has to read as a cavity");

        // The strongest protection anywhere on the +Z cap against the strongest
        // anywhere on the untouched -Z cap, which is still the exact convex
        // sphere and is the control inside the same volume.
        let cap = |sign: f32| {
            cells_near_the_surface(&volume, 1.0)
                .into_iter()
                .filter(|cell| {
                    let at = cell.as_vec3() * VOXEL;
                    at.z * sign > RADIUS * 0.6 && at.x.abs() < 3.0 && at.y.abs() < 3.0
                })
                .map(|cell| mask.at(cell))
                .max()
                .unwrap_or(0)
        };
        assert!(cap(1.0) > 128, "the carved groove was barely masked: {}", cap(1.0));
        assert_eq!(cap(-1.0), UNMASKED, "the untouched convex cap must stay free");
    }

    // ------------------------------------------------------------ smoothness

    /// The other reading of the same pass: it protects the ground and frees the
    /// detail, which is the opposite selection from the same seven reads.
    #[test]
    fn the_smoothness_mask_protects_what_the_cavity_mask_leaves_free() {
        let volume = block_with_a_void(VOXEL, 10.0, 4.0);
        let cavity = volume.generated_mask(MaskRecipe::Cavity { feature_mm: 16.0 });
        let smooth = volume.generated_mask(MaskRecipe::Smoothness { feature_mm: 16.0 });

        let face: Vec<IVec3> = cells_near_the_surface(&volume, 0.5)
            .into_iter()
            .filter(|cell| {
                let at = cell.as_vec3() * VOXEL;
                at.z.abs() > 9.0 && at.x.abs() < 5.0 && at.y.abs() < 5.0
            })
            .collect();
        assert!(face.len() > 50, "the fixture has no flat face to measure");
        let cavity_on_the_face = face.iter().map(|cell| cavity.at(*cell)).max().expect("not empty");
        let smooth_on_the_face = face.iter().map(|cell| smooth.at(*cell)).min().expect("not empty");
        assert_eq!(cavity_on_the_face, UNMASKED);
        assert_eq!(smooth_on_the_face, PROTECTED, "a flat face is what smoothness means");
    }

    // ------------------------------------------------------------- thickness

    /// The claim on the slider: everything thinner than N voxels comes out
    /// selected and everything thicker does not.
    #[test]
    fn mask_by_thickness_selects_a_thin_wall_and_leaves_a_thick_block_alone() {
        // Two slabs in one volume, far enough apart to be independent: one four
        // voxels thick and one twenty.
        let mut volume = Volume::new(VOXEL);
        let lo = IVec3::new(-40, -40, -40);
        let hi = IVec3::new(40, 40, 40);
        volume.edit_voxels(lo, hi, |_, at, _| {
            let thin = (at.z - 8.0).abs() - 1.0;
            let thick = (at.z + 8.0).abs() - 5.0;
            (thin.min(thick) / VOXEL).clamp(INSIDE, OUTSIDE)
        });

        let mask = volume.generated_mask(MaskRecipe::Thickness { voxels: 6 });
        let inside_thin = (Vec3::new(0.0, 0.0, 8.0) / VOXEL).round().as_ivec3();
        let inside_thick = (Vec3::new(0.0, 0.0, -8.0) / VOXEL).round().as_ivec3();
        assert!(
            mask.at(inside_thin) > 200,
            "a 4-voxel wall is thinner than 6 and must be selected, got {}",
            mask.at(inside_thin)
        );
        assert_eq!(
            mask.at(inside_thick),
            UNMASKED,
            "a 20-voxel block is not thin and must not be selected"
        );
    }

    /// The ceiling is the field's, not a preference, so asking past it has to
    /// behave as asking for the ceiling rather than as an error or a wider
    /// selection.
    #[test]
    fn mask_by_thickness_clamps_to_the_band_it_can_actually_measure() {
        let mut volume = Volume::new(VOXEL);
        volume.edit_voxels(IVec3::splat(-40), IVec3::splat(40), |_, at, _| {
            ((at.z.abs() - 1.0) / VOXEL).clamp(INSIDE, OUTSIDE)
        });
        let at_the_ceiling =
            volume.generated_mask(MaskRecipe::Thickness { voxels: MAX_THICKNESS_VOXELS });
        let far_past_it = volume.generated_mask(MaskRecipe::Thickness { voxels: 4000 });
        let cells = cells_near_the_surface(&volume, NARROW_BAND);
        for cell in cells {
            assert_eq!(
                at_the_ceiling.at(cell),
                far_past_it.at(cell),
                "asking for 4000 voxels of thickness answered differently at {cell:?}"
            );
        }
    }

    /// A reach test answers yes or no, and a step in the mask is a fold in the
    /// geometry under Move. The feather is one pass of the existing Blur, and
    /// this is what says it ran.
    #[test]
    fn a_thickness_mask_is_feathered_by_the_blur_it_ends_with() {
        let mut volume = Volume::new(VOXEL);
        volume.edit_voxels(IVec3::splat(-40), IVec3::splat(40), |_, at, _| {
            ((at.z.abs() - 1.0) / VOXEL).clamp(INSIDE, OUTSIDE)
        });
        let mask = volume.generated_mask(MaskRecipe::Thickness { voxels: 6 });
        let middling = cells_near_the_surface(&volume, NARROW_BAND)
            .into_iter()
            .filter(|cell| {
                let protection = mask.at(*cell);
                protection > UNMASKED && protection < PROTECTED
            })
            .count();
        assert!(middling > 100, "a thickness mask that never blurred: {middling} soft voxels");
    }

    // ------------------------------------------------------------ half-space

    /// The whole reason this recipe reuses the cut's classification: a mask over
    /// half a model must cost its boundary and not its volume.
    ///
    /// A solid block of brick tiles and not a sphere, because that is what
    /// isolates the claim: every one of these 512 bricks is an interior tile
    /// costing no heap at all, so any that comes back dense came back dense
    /// because of the classification and not because of a surface passing
    /// through it.
    #[test]
    fn the_half_space_mask_promotes_the_boundary_and_never_the_interior() {
        const SIDE: i32 = 8;
        let mut volume = Volume::new(VOXEL);
        for z in 0..SIDE {
            for y in 0..SIDE {
                for x in 0..SIDE {
                    volume.insert_brick(BrickCoord::new(x, y, z), Brick::Uniform(INSIDE));
                }
            }
        }
        let total = volume.brick_count();
        assert_eq!(total, (SIDE * SIDE * SIDE) as usize);

        // Through the middle of the block, well away from any brick boundary.
        let middle = (SIDE as f32 / 2.0) * BRICK_DIM as f32 * VOXEL;
        let plane =
            ClipPlane::new(Vec3::new(0.0, 0.0, middle), Vec3::Z).expect("a unit normal is a plane");
        let mask = volume.generated_mask(MaskRecipe::Halfspace { plane, feather_mm: 1.0 });
        volume.replace_mask(mask);

        let stats = volume.stats();
        assert!(stats.mask_dense_bricks > 0, "the boundary has to be resolved per voxel");
        // One layer of the eight is the boundary, so a third of the block is
        // already far more generous than the geometry needs and still fails
        // outright if an interior tile is ever promoted.
        assert!(
            stats.mask_dense_bricks * 3 < total,
            "{} of {total} bricks went dense, which is a volume and not a boundary",
            stats.mask_dense_bricks,
        );
    }

    /// Feathered, per the rule in `crate::mask`: the boundary carries real
    /// intermediate values rather than one step from free to protected.
    #[test]
    fn the_half_space_boundary_carries_at_least_three_distinct_values_across_it() {
        let volume = sphere(VOXEL, 12.0);
        let plane = ClipPlane::new(Vec3::ZERO, Vec3::Z).expect("a unit normal is a plane");
        let mask = volume.generated_mask(MaskRecipe::Halfspace { plane, feather_mm: 2.0 });

        let mut seen: Vec<u8> = Vec::new();
        let mut z = -6.0_f32;
        while z <= 6.0 {
            let cell = (Vec3::new(0.0, 0.0, z) / VOXEL).round().as_ivec3();
            let protection = mask.at(cell);
            if !seen.contains(&protection) {
                seen.push(protection);
            }
            z += VOXEL;
        }
        assert!(seen.contains(&UNMASKED), "the kept side has to be free: {seen:?}");
        assert!(seen.contains(&PROTECTED), "the cut side has to be protected: {seen:?}");
        assert!(seen.len() >= 3, "the feather is a step: {seen:?}");
    }

    // ----------------------------------------------------- polarity and cost

    /// A generator computes protection directly, so it cannot inherit an
    /// inversion: a cavity mask asked for on top of Mask All would otherwise
    /// protect precisely the flat ground it was told to leave free.
    #[test]
    fn a_generated_mask_replaces_the_polarity_it_found() {
        let mut volume = block_with_a_void(VOXEL, 10.0, 4.0);
        let all = volume.mask().cleared(true);
        volume.replace_mask(all);
        assert!(volume.mask().protects_everything());

        let mask = volume.generated_mask(MaskRecipe::Cavity { feature_mm: 16.0 });
        assert!(!mask.inverted(), "a generated mask has to arrive at normal polarity");
        let strongest_on_the_wall = cells_near_the_surface(&volume, 0.5)
            .into_iter()
            .filter(|cell| (cell.as_vec3() * VOXEL).length() < 8.0)
            .map(|cell| mask.at(cell))
            .max()
            .expect("the wall is not empty");
        assert_eq!(strongest_on_the_wall, PROTECTED);
    }

    /// The refusal in front of these is part of the feature, and it can only be
    /// as good as the prediction it is made from: a generated mask fills
    /// essentially every band brick, so the prediction must not hope for
    /// collapse.
    #[test]
    fn a_generated_mask_never_costs_more_than_it_was_predicted_to() {
        let volume = block_with_a_void(VOXEL, 10.0, 4.0);
        for recipe in [
            MaskRecipe::Cavity { feature_mm: 16.0 },
            MaskRecipe::Smoothness { feature_mm: 16.0 },
            MaskRecipe::Thickness { voxels: 4 },
        ] {
            let predicted = volume.generated_mask_demand(recipe);
            let real = volume.generated_mask(recipe).bytes();
            assert!(
                real <= predicted,
                "{recipe:?} cost {real} bytes against a prediction of {predicted}"
            );
        }
    }

    /// The thickness walk copies the field and the other recipes do not, and on
    /// a document near the ceiling that difference is a refusal against an
    /// out-of-memory kill.
    #[test]
    fn only_the_thickness_walk_predicts_the_copy_of_the_field_it_makes() {
        let volume = block_with_a_void(VOXEL, 10.0, 4.0);
        let cavity = volume.generated_mask_demand(MaskRecipe::Cavity { feature_mm: 16.0 });
        let thickness = volume.generated_mask_demand(MaskRecipe::Thickness { voxels: 4 });
        assert!(
            thickness > cavity * 2,
            "thickness predicted {thickness} against cavity's {cavity}, which cannot include a \
             copy of the field"
        );
    }

    /// An empty body is a real thing to press a button on, and every recipe has
    /// to answer it with a mask nothing shows a card for.
    #[test]
    fn every_recipe_on_an_empty_body_produces_a_mask_that_is_free() {
        let volume = Volume::new(VOXEL);
        let plane = ClipPlane::new(Vec3::ZERO, Vec3::Z).expect("a unit normal is a plane");
        for recipe in [
            MaskRecipe::Cavity { feature_mm: 2.0 },
            MaskRecipe::Smoothness { feature_mm: 2.0 },
            MaskRecipe::Thickness { voxels: 3 },
            MaskRecipe::Halfspace { plane, feather_mm: 1.0 },
        ] {
            assert!(volume.generated_mask(recipe).is_free(), "{recipe:?} invented a mask");
        }
    }
}
