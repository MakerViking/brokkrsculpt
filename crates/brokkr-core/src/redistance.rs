// SPDX-License-Identifier: AGPL-3.0-only

//! Put the distances back, so the field is a distance field again.
//!
//! Every operation in this engine that is not a similarity leaves a volume
//! whose zero set is right and whose *distances* are wrong. The brushes do it a
//! little -- [`crate::brush`] warps the domain and the gradient's length drifts
//! away from one across many overlapping stamps. A per-axis scale does it a
//! lot: squashing one axis by `s_min` and another by `s_max` drives the
//! gradient into `[s_min/s_max, 1]` everywhere at once. Nothing in the engine
//! reset either of them, so the drift accumulated with no ceiling, and that --
//! not any difficulty in expressing the transform -- is the whole reason
//! [`crate::Similarity`] refused a per-axis scale.
//!
//! This is the pass [`crate::similarity`]'s header names as its trigger. It has
//! the two customers that header predicted, and a third that turned up on the
//! way: the brush residual, per-axis scale, and any future domain warp -- a
//! bend, a twist, a taper -- which is the same problem with a different shape.
//!
//! # The method, and why this one
//!
//! Fast sweeping, restricted to the narrow band. The field is already clamped
//! to `+/-NARROW_BAND` voxels either side of the surface, so there is nothing
//! to solve in the interior or the exterior -- both are saturated constants and
//! stay exactly as they are. What is left is a shell three voxels thick, which
//! is a very small fraction of any real body and the reason this is affordable
//! at all.
//!
//! Inside that shell the Eikonal equation `|grad d| = 1` is solved by Godunov
//! upwind differencing, sweeping the volume in all eight diagonal directions.
//! Sweeping alternately with and against each axis is what lets information
//! travel outward from the surface in one pass per direction rather than
//! propagating one voxel per iteration; the classical result is that eight
//! sweeps suffice in three dimensions, and the band here is thin enough that
//! the first two do nearly all of it.
//!
//! **Fast marching was rejected, and not on grounds of accuracy.** It is the
//! better-known method and it converges in one pass rather than eight, but it
//! needs a priority queue over the whole band, which is an allocation
//! proportional to the surface area and a heap operation per voxel. Sweeping
//! needs no allocation beyond the brick scratch this crate already keeps, and
//! it is trivially parallel per brick row. At the sizes here -- a 133 mm body
//! at 0.079 mm is tens of millions of voxels, of which the band is a few
//! million -- the constant factor decides it.
//!
//! # What is NOT touched, and why that matters more than what is
//!
//! **The sign is never changed.** Redistancing moves magnitudes; a voxel that
//! was inside stays inside. That is not a nicety: the sign is what the mesher
//! reads, so a pass that could flip one could move the surface, and a
//! correction that can move the surface is not a correction. Every write here
//! goes through [`keep_sign`].
//!
//! **A saturated voxel stays saturated.** `+/-NARROW_BAND` means "further than
//! the band reaches" and solving for it would invent a distance the field
//! cannot hold anyway.
//!
//! **Uniform bricks are skipped entirely**, which is what makes this cheap on a
//! real model: a brick that is all interior or all exterior has no band in it.

use glam::Vec3;

use crate::brick::{BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE};
use crate::volume::Volume;

/// Copy the scratch back into a brick-sized array.
fn solved_array(solved: &[f32]) -> [f32; BRICK_VOXELS] {
    let mut out = [0.0f32; BRICK_VOXELS];
    out.copy_from_slice(solved);
    out
}

/// How near the gradient has to be to one before a body is left alone.
///
/// Measured against the worst voxel rather than the average: an average hides
/// exactly the local defect this exists to find. The value is not arbitrary --
/// [`crate::generate`] clamps a sampled gradient into `[0.5, 2.0]` and treats
/// anything inside that as usable, so a body whose worst voxel is within 10% of
/// one is comfortably inside what every other reader already tolerates, and
/// redistancing it would be work with no customer.
pub const GRADIENT_TOLERANCE: f32 = 0.1;

/// Sweeps per pass. Eight is every combination of direction along three axes.
const SWEEPS: usize = 8;

/// What one redistancing did, for the status line and for the tests.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RedistanceReport {
    /// Bricks that held some band and were solved.
    pub bricks: usize,
    /// Voxels written. Zero means the field was already a distance field.
    pub corrected: usize,
    /// The worst `||grad d| - 1|` found before the pass, and after it.
    pub worst_before: f32,
    pub worst_after: f32,
}

impl Volume {
    /// Measure how far this volume has drifted from being a distance field.
    ///
    /// The worst deviation of `|grad d|` from one, over every band voxel. Cheap
    /// enough to gate on: it is one pass with no writes, and it is what lets
    /// [`Volume::redistance`] decline to touch a body that does not need it.
    pub fn gradient_drift(&self) -> f32 {
        let mut worst: f32 = 0.0;
        let mut apron = crate::apron::ApronBuffer::new();
        for coord in self.brick_coords().collect::<Vec<_>>() {
            if !matches!(self.brick(coord), Some(Brick::Dense(_))) {
                continue;
            }
            self.gather_apron(coord, &mut apron);
            for index in 0..BRICK_VOXELS {
                let (x, y, z) = unindex(index);
                if !measurable(&apron, x, y, z) {
                    continue;
                }
                let length = central_gradient(&apron, x, y, z).length();
                worst = worst.max((length - 1.0).abs());
            }
        }
        worst
    }

    /// Solve `|grad d| = 1` across the narrow band, leaving every sign alone.
    ///
    /// Returns `None` when the field is already within [`GRADIENT_TOLERANCE`],
    /// which is the common case and costs one read-only pass. A body that has
    /// only ever been sculpted and moved rigidly never needs this.
    pub fn redistance(&mut self) -> Option<RedistanceReport> {
        let worst_before = self.gradient_drift();
        if worst_before <= GRADIENT_TOLERANCE {
            return None;
        }

        let mut report = RedistanceReport { worst_before, ..Default::default() };
        let coords: Vec<BrickCoord> = self
            .brick_coords()
            .filter(|coord| matches!(self.brick(*coord), Some(Brick::Dense(_))))
            .collect();

        // **Two phases, because a block-wise Eikonal solve needs its halo.**
        //
        // Phase one writes the interface seeds -- and only those -- into the
        // volume, everything else saturated. Phase two sweeps each brick
        // repeatedly, reading its neighbours through the apron, so a correction
        // crosses a brick boundary one voxel per outer iteration. The band is
        // NARROW_BAND voxels deep and the apron is one thick, so the loop runs
        // that many times plus one and no further: past that there is nothing
        // left inside the band for a neighbour to tell it.
        let mut apron = crate::apron::ApronBuffer::new();
        let mut solved = vec![0.0f32; BRICK_VOXELS];

        for coord in &coords {
            self.gather_apron(*coord, &mut apron);
            for (index, value) in solved.iter_mut().enumerate() {
                let (x, y, z) = unindex(index);
                *value = apron.get(x + 1, y + 1, z + 1);
            }
            let seeded = seed_from_interface(&solved, &apron);
            for index in 0..BRICK_VOXELS {
                let was = solved[index];
                solved[index] = if in_band(was) && seeded[index].is_finite() {
                    keep_sign(was, seeded[index])
                } else if in_band(was) {
                    keep_sign(was, OUTSIDE)
                } else {
                    was
                };
            }
            self.insert_brick(*coord, Brick::Dense(Box::new(solved_array(&solved))));
        }

        let outer = NARROW_BAND as usize + 1;
        for _ in 0..outer {
            for coord in &coords {
                self.gather_apron(*coord, &mut apron);
                let mut working = vec![0.0f32; BRICK_VOXELS];
                for (index, value) in working.iter_mut().enumerate() {
                    let (x, y, z) = unindex(index);
                    *value = apron.get(x + 1, y + 1, z + 1).abs();
                }
                for sweep in 0..SWEEPS {
                    sweep_once(&mut working, &apron, sweep);
                }
                for index in 0..BRICK_VOXELS {
                    let (x, y, z) = unindex(index);
                    let was = apron.get(x + 1, y + 1, z + 1);
                    solved[index] = keep_sign(was, working[index]);
                }
                self.insert_brick(*coord, Brick::Dense(Box::new(solved_array(&solved))));
            }
        }
        report.bricks = coords.len();
        report.corrected = coords.len() * BRICK_VOXELS;

        report.worst_after = self.gradient_drift();
        Some(report)
    }
}

/// Whether a value is inside the band and therefore solvable.
///
/// Strict on both sides. A voxel sitting exactly at `+/-NARROW_BAND` is
/// saturated -- it means "further than the band reaches" -- and solving for it
/// would invent a distance the field has no room to hold.
fn in_band(value: f32) -> bool {
    value > INSIDE && value < OUTSIDE
}

/// Give `magnitude` the sign of `was`, and clamp back into the band.
///
/// **The one place a redistanced value is written**, so that "this pass cannot
/// move the surface" is a property of one function rather than a claim about
/// every call site. `total_cmp` rather than `< 0.0` because the voxeliser
/// biases an exact zero to the inside as `-0.0`, and `-0.0 < 0.0` is false --
/// the same trap `fill_sealed_cavities` documents, and reading it wrong here
/// would flip the sign of every on-surface voxel.
fn keep_sign(was: f32, magnitude: f32) -> f32 {
    let negative = was.total_cmp(&0.0).is_le();
    let signed = if negative { -magnitude } else { magnitude };
    signed.clamp(INSIDE, OUTSIDE)
}

/// Whether a voxel's central-difference stencil is free of saturation.
///
/// A voxel one step inside the band has a neighbour pinned at `+/-NARROW_BAND`,
/// so its central difference measures the clamp rather than the field and reads
/// far from one on a PERFECTLY good ball. Measuring those was the first version
/// of this and it reported a clean seeded sphere as drifting by 0.497.
fn measurable(apron: &crate::apron::ApronBuffer, x: usize, y: usize, z: usize) -> bool {
    let (x, y, z) = (x + 1, y + 1, z + 1);
    if !in_band(apron.get(x, y, z)) {
        return false;
    }
    [
        apron.get(x + 1, y, z),
        apron.get(x - 1, y, z),
        apron.get(x, y + 1, z),
        apron.get(x, y - 1, z),
        apron.get(x, y, z + 1),
        apron.get(x, y, z - 1),
    ]
    .iter()
    .all(|v| in_band(*v))
}

/// Distances for the voxels that touch the surface, and infinity for the rest.
///
/// The seed is a magnitude grid: sign is carried separately and restored on the
/// way out, so the sweep never has to think about it.
fn seed_from_interface(solved: &[f32], apron: &crate::apron::ApronBuffer) -> Vec<f32> {
    let mut seed = vec![f32::INFINITY; BRICK_VOXELS];
    for index in 0..BRICK_VOXELS {
        let (x, y, z) = unindex(index);
        let here = solved[index];
        if !in_band(here) {
            continue;
        }
        let negative = here.total_cmp(&0.0).is_le();
        let mut best = f32::INFINITY;
        for (dx, dy, dz) in
            [(1isize, 0, 0), (-1, 0, 0), (0, 1isize, 0), (0, -1, 0), (0, 0, 1isize), (0, 0, -1)]
        {
            let there = apron.get(
                (x as isize + dx + 1) as usize,
                (y as isize + dy + 1) as usize,
                (z as isize + dz + 1) as usize,
            );
            if there.total_cmp(&0.0).is_le() == negative {
                continue;
            }
            // The crossing along this edge, as a fraction of a voxel. Invariant
            // under a rescaling of the whole field, which is the point.
            let span = here - there;
            if span.abs() > f32::EPSILON {
                best = best.min((here / span).abs());
            }
        }
        if best.is_finite() {
            seed[index] = best;
        }
    }
    seed
}

/// One Godunov sweep in one of the eight diagonal directions, over magnitudes.
///
/// Monotone DECREASING, which is what makes eight sweeps enough: every voxel
/// starts at infinity except the interface seeds, and information flows outward
/// from them. The first version updated in place from the existing values and
/// could only ever lower them, so it could not repair a field whose distances
/// were too SMALL -- which is precisely the drift this pass exists to remove.
fn sweep_once(grid: &mut [f32], apron: &crate::apron::ApronBuffer, sweep: usize) {
    let back_x = sweep & 1 != 0;
    let back_y = sweep & 2 != 0;
    let back_z = sweep & 4 != 0;

    for step_z in 0..BRICK_DIM {
        let z = if back_z { BRICK_DIM - 1 - step_z } else { step_z };
        for step_y in 0..BRICK_DIM {
            let y = if back_y { BRICK_DIM - 1 - step_y } else { step_y };
            for step_x in 0..BRICK_DIM {
                let x = if back_x { BRICK_DIM - 1 - step_x } else { step_x };
                let index = x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM;
                let a = axis_min(grid, apron, x, y, z, 0);
                let b = axis_min(grid, apron, x, y, z, 1);
                let c = axis_min(grid, apron, x, y, z, 2);
                let candidate = godunov(a, b, c);
                if candidate < grid[index] {
                    grid[index] = candidate;
                }
            }
        }
    }
}

/// The nearer neighbour along one axis, inside this brick.
///
/// Outside the brick reads infinity rather than the apron: the apron holds the
/// OLD drifted values, and seeding the sweep from them would carry the error
/// back in across every boundary. A brick is 32 voxels and the band is three,
/// so the interface seeds inside the brick reach every band voxel in it.
fn axis_min(
    grid: &[f32],
    apron: &crate::apron::ApronBuffer,
    x: usize,
    y: usize,
    z: usize,
    axis: usize,
) -> f32 {
    let (dx, dy, dz) = match axis {
        0 => (1isize, 0, 0),
        1 => (0, 1isize, 0),
        _ => (0, 0, 1isize),
    };
    let mut best = f32::INFINITY;
    for sign in [-1isize, 1] {
        let (nx, ny, nz) = (x as isize + dx * sign, y as isize + dy * sign, z as isize + dz * sign);
        if (0..BRICK_DIM as isize).contains(&nx)
            && (0..BRICK_DIM as isize).contains(&ny)
            && (0..BRICK_DIM as isize).contains(&nz)
        {
            let at = nx as usize + ny as usize * BRICK_DIM + nz as usize * BRICK_DIM * BRICK_DIM;
            best = best.min(grid[at]);
        } else {
            // Out of this brick: the apron carries the neighbour's magnitude
            // from the previous outer iteration. This is the halo exchange that
            // makes a block-wise solve equal to a global one -- without it each
            // brick solves in isolation, every brick face becomes a
            // discontinuity, and the pass makes the field WORSE than it found
            // it, which is exactly what the first version did.
            best =
                best.min(apron.get((nx + 1) as usize, (ny + 1) as usize, (nz + 1) as usize).abs());
        }
    }
    best
}

/// The Godunov solution of `|grad d| = 1` from three upwind neighbours.
///
/// Distances are in VOXELS here, so the grid spacing is one and drops out of
/// the arithmetic. Solve with one neighbour, then two, then three, taking the
/// first that stays consistent -- a candidate is only valid while it is at
/// least as large as every neighbour it was built from.
fn godunov(a: f32, b: f32, c: f32) -> f32 {
    let mut sorted = [a, b, c];
    sorted.sort_by(f32::total_cmp);
    let [first, second, third] = sorted;

    if !first.is_finite() {
        return f32::INFINITY;
    }
    let one = first + 1.0;
    if one <= second || !second.is_finite() {
        return one;
    }
    // Two neighbours: (d-a)^2 + (d-b)^2 = 1.
    let sum = first + second;
    let discriminant = 2.0 - (first - second) * (first - second);
    if discriminant < 0.0 {
        return one;
    }
    let two = 0.5 * (sum + discriminant.sqrt());
    if two <= third || !third.is_finite() {
        return two;
    }
    // Three: (d-a)^2 + (d-b)^2 + (d-c)^2 = 1.
    let sum3 = first + second + third;
    let squares = first * first + second * second + third * third;
    let discriminant3 = sum3 * sum3 - 3.0 * (squares - 1.0);
    if discriminant3 < 0.0 {
        return two;
    }
    (sum3 + discriminant3.sqrt()) / 3.0
}

/// Central-difference gradient at a brick voxel, in voxel units.
fn central_gradient(apron: &crate::apron::ApronBuffer, x: usize, y: usize, z: usize) -> Vec3 {
    let (x, y, z) = (x + 1, y + 1, z + 1);
    Vec3::new(
        0.5 * (apron.get(x + 1, y, z) - apron.get(x - 1, y, z)),
        0.5 * (apron.get(x, y + 1, z) - apron.get(x, y - 1, z)),
        0.5 * (apron.get(x, y, z + 1) - apron.get(x, y, z - 1)),
    )
}

fn unindex(index: usize) -> (usize, usize, usize) {
    (index % BRICK_DIM, (index / BRICK_DIM) % BRICK_DIM, index / (BRICK_DIM * BRICK_DIM))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field straight out of the voxeliser is already a distance field, so
    /// the pass declines it. That `None` is the whole reason this is
    /// affordable: an ordinary sculpt never pays for it.
    #[test]
    fn a_clean_field_is_left_alone() {
        let mut volume = Volume::new(0.25);
        volume.seed_sphere(Vec3::ZERO, 12.0);
        let drift = volume.gradient_drift();
        assert!(drift <= GRADIENT_TOLERANCE, "a seeded ball already drifts by {drift}");
        assert_eq!(volume.redistance(), None, "it redistanced a field that did not need it");
    }

    /// **The one that matters: a squashed field is put back.**
    ///
    /// Multiplying every distance by 0.4 is exactly what a per-axis scale of
    /// 0.4 does to the gradient -- the zero set is untouched and every
    /// magnitude is wrong by the same factor, which is the drift
    /// `similarity.rs` refuses to let compound. If this does not converge there
    /// is no per-axis scale.
    #[test]
    fn a_squashed_field_is_restored_to_unit_gradient() {
        let mut volume = squashed_ball(0.25, 8.0, 0.5);

        let before = volume.gradient_drift();
        assert!(before > 0.2, "the fixture did not actually squash anything: {before}");

        let report = volume.redistance().expect("a squashed field needs the pass");
        assert!(
            report.worst_after < report.worst_before,
            "redistancing made it worse: {} -> {}",
            report.worst_before,
            report.worst_after
        );
        assert!(
            report.worst_after <= before * 0.6,
            "the gradient is still {} from one after the pass",
            report.worst_after
        );
    }

    /// **A correction that can move the surface is not a correction.**
    ///
    /// The sign is what the mesher reads, so this asserts the pass moved
    /// magnitudes and nothing else. Checked voxel for voxel rather than by
    /// triangle count, which would pass on a surface that had moved and stayed
    /// the same size.
    #[test]
    fn redistancing_never_changes_a_sign() {
        let mut volume = squashed_ball(0.25, 8.0, 0.5);

        let signs = signs_of(&volume);
        volume.redistance().expect("the fixture needs the pass");
        assert_eq!(signs, signs_of(&volume), "a voxel changed side");
    }

    /// Build the field a per-axis scale actually produces.
    ///
    /// **Not "multiply the stored values", which was the first fixture and was
    /// wrong.** Scaling only the in-band values leaves them beside untouched
    /// saturated neighbours, a cliff no real transform makes, and the pass was
    /// then judged on repairing an impossible field. A per-axis scale resamples
    /// the whole thing: squashing z by `s` gives
    /// `d(p) = (|p / diag(1,1,s)| - r) * s`, which has the right zero set and a
    /// gradient of `s` along the squashed axis -- the exact drift
    /// `similarity.rs` refuses to let compound.
    fn squashed_ball(voxel_size: f32, radius: f32, squash: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        let reach = (radius / voxel_size).ceil() as i32 + 8;
        let bricks = reach / BRICK_DIM as i32 + 1;
        for bz in -bricks..=bricks {
            for by in -bricks..=bricks {
                for bx in -bricks..=bricks {
                    let coord = BrickCoord(glam::IVec3::new(bx, by, bz));
                    let mut data = [0.0f32; BRICK_VOXELS];
                    for (index, slot) in data.iter_mut().enumerate() {
                        let (x, y, z) = unindex(index);
                        let voxel = glam::IVec3::new(
                            bx * BRICK_DIM as i32 + x as i32,
                            by * BRICK_DIM as i32 + y as i32,
                            bz * BRICK_DIM as i32 + z as i32,
                        );
                        let p = voxel.as_vec3() * voxel_size;
                        let stretched = Vec3::new(p.x, p.y, p.z / squash);
                        let d = (stretched.length() - radius) * squash;
                        *slot = (d / voxel_size).clamp(INSIDE, OUTSIDE);
                    }
                    let first = data[0];
                    if data.iter().all(|v| *v == first) {
                        volume.insert_brick(coord, Brick::Uniform(first));
                    } else {
                        volume.insert_brick(coord, Brick::Dense(Box::new(data)));
                    }
                }
            }
        }
        volume
    }

    fn signs_of(volume: &Volume) -> Vec<bool> {
        let mut out = Vec::new();
        for coord in volume.brick_coords().collect::<Vec<_>>() {
            if let Some(Brick::Dense(data)) = volume.brick(coord) {
                out.extend(data.iter().map(|v| v.total_cmp(&0.0).is_le()));
            }
        }
        out
    }
}
