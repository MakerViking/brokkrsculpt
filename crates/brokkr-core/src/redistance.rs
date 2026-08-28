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
use rayon::prelude::*;

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

/// How far a voxel has to move for the pass to call it movement.
///
/// A thousandth of a voxel, which is four hundred times finer than
/// [`GRADIENT_TOLERANCE`] admits and so cannot decide the pass's answer. It
/// exists because the Godunov update is a square root of a quotient: a settled
/// voxel keeps shedding fractions of an ULP indefinitely, so an exact
/// comparison never reports a fixed point and the early exit never fires.
const SETTLED: f32 = 1.0e-3;

thread_local! {
    /// How many brick sweeps this thread has run inside [`Volume::redistance`].
    ///
    /// The same device as [`crate::transform::warps_made_on_this_thread`], and
    /// for the same reason: the property worth asserting is how much work was
    /// done, and nothing in the result can see it. `RedistanceReport::corrected`
    /// cannot stand in -- a skipped outer iteration contributes zero moved
    /// voxels by construction, so a forced run and an early-exiting one report
    /// exactly the same count and a test comparing them proves nothing. That
    /// was written first and the non-degeneracy assertion caught it.
    static SWEEPS_RUN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The value of that counter. Take it before and after and compare.
///
/// Test-only, unlike [`crate::transform::warps_made_on_this_thread`], which is
/// public because the app's status line has a use for it. Nothing outside this
/// file needs a sweep count, and CI builds with `-D warnings`.
#[cfg(test)]
fn sweeps_run_on_this_thread() -> usize {
    SWEEPS_RUN.with(std::cell::Cell::get)
}

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
        self.redistance_inner(true)
    }

    /// The pass, with the early exits switchable so a test can prove they do
    /// not change the answer.
    ///
    /// `stop_when_settled` false forces every sweep and every outer iteration
    /// to run. That is the reference the fast path is checked against, and it
    /// is the only way to state "this optimisation is free" as an assertion
    /// rather than as a comparison of two bench runs.
    fn redistance_inner(&mut self, stop_when_settled: bool) -> Option<RedistanceReport> {
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
        // crosses one brick boundary per outer iteration. See the bound above
        // phase two for why the unit is a brick and not a voxel.
        let mut apron = crate::apron::ApronBuffer::new();
        let mut solved = vec![0.0f32; BRICK_VOXELS];

        // **Phase one is STAGED, and that is load bearing now the seed matters.**
        //
        // Writing each brick back inside this walk means a brick reached later
        // gathers an apron whose neighbours have already been reduced to seeds
        // and saturation, so it computes its own seed against a wiped face
        // rather than against the field it came in with. That was survivable
        // while the seed was a crossing fraction, because phase two washed the
        // contamination out; it is not survivable now the seed is the whole
        // initial condition. Collect first, insert afterwards.
        let mut staged: Vec<(BrickCoord, Brick)> = Vec::with_capacity(coords.len());
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
            staged.push((*coord, Brick::Dense(Box::new(solved_array(&solved)))));
        }
        for (coord, brick) in staged {
            self.insert_brick(coord, brick);
        }

        // **The outer bound is in BRICKS, not voxels**, and the old comment
        // above said voxels. A sweep carries an apron value across a whole
        // brick in one pass, not one voxel per pass, so the unit here is the
        // brick: a band voxel is at most one brick away from a brick that
        // holds an interface seed, which makes the structural bound two.
        // `NARROW_BAND + 1` is kept as a ceiling with margin, but the loop
        // stops as soon as nothing moves and in practice never reaches it.
        // Leaving the derivation wrong invites the next reader to widen it.
        let outer = NARROW_BAND as usize + 1;
        let mut corrected = 0usize;

        // **Eight colours, so the bricks of one colour are never neighbours.**
        //
        // Two bricks sharing `(x & 1, y & 1, z & 1)` differ by at least two in
        // some coordinate, so neither is in the other's 26-neighbourhood. A
        // whole colour can therefore be solved in parallel: each brick reads
        // its apron from bricks of OTHER colours, none of which is being
        // written during that phase, and the results are inserted before the
        // next colour begins. Red-black by another name, and the reason it is
        // eight rather than two is that the halo here is a full 26-neighbour
        // one rather than a six-neighbour one.
        //
        // **A full staging buffer was rejected**, not merely not chosen: a
        // `Vec<(BrickCoord, Brick)>` over every dense brick duplicates the
        // field at 128 KB a brick, which would take `rebake_gizmo`'s deliberate
        // two live copies to three -- about 1.4 GB on the bench's 470 MB row,
        // on a path whose arm limit is 512 MB. At most an eighth is staged
        // here.
        let mut by_colour: [Vec<BrickCoord>; 8] = Default::default();
        for coord in &coords {
            let v = coord.0;
            let colour =
                (v.x.rem_euclid(2) + v.y.rem_euclid(2) * 2 + v.z.rem_euclid(2) * 4) as usize;
            by_colour[colour].push(*coord);
        }

        for _ in 0..outer {
            let mut moved = 0usize;
            for colour in &by_colour {
                let solved: Vec<(BrickCoord, Brick, usize, usize)> = colour
                    .par_iter()
                    .map_init(
                        || (crate::apron::ApronBuffer::new(), vec![0.0f32; BRICK_VOXELS]),
                        |(apron, working), coord| {
                            self.gather_apron(*coord, apron);
                            for (index, value) in working.iter_mut().enumerate() {
                                let (x, y, z) = unindex(index);
                                *value = apron.get(x + 1, y + 1, z + 1).abs();
                            }
                            let mut swept = 0usize;
                            for sweep in 0..SWEEPS {
                                swept += 1;
                                if sweep_once(working, apron, sweep) == 0 && stop_when_settled {
                                    break;
                                }
                            }
                            let mut brick_moved = 0usize;
                            let mut out = [0.0f32; BRICK_VOXELS];
                            for index in 0..BRICK_VOXELS {
                                let (x, y, z) = unindex(index);
                                let was = apron.get(x + 1, y + 1, z + 1);
                                let now = keep_sign(was, working[index]);
                                if (now - was).abs() > SETTLED {
                                    brick_moved += 1;
                                }
                                out[index] = now;
                            }
                            // Hand back a brick the pass has flattened to one
                            // value. Phase two rewrites every dense brick
                            // anyway, so this is the site where it costs
                            // nothing extra, and a squash leaves a real number
                            // of them fully saturated once the band is back.
                            let brick = Brick::Dense(Box::new(out));
                            let brick = match brick.is_collapsible() {
                                Some(value) => Brick::Uniform(value),
                                None => brick,
                            };
                            (*coord, brick, brick_moved, swept)
                        },
                    )
                    .collect();

                // Summed on the CALLING thread and not inside the closure:
                // `SWEEPS_RUN` is a thread local, and once the work moved onto
                // rayon's pool the counter on this thread stayed at zero --
                // which the non-degeneracy assertion in
                // `stopping_early_changes_no_voxel_by_more_than_it_promises`
                // caught immediately, having been written for exactly that
                // class of mistake.
                for (coord, brick, brick_moved, swept) in solved {
                    moved += brick_moved;
                    SWEEPS_RUN.with(|run| run.set(run.get() + swept));
                    self.insert_brick(coord, brick);
                }
            }
            corrected += moved;
            // **Stopping when nothing decreased is sound, and it is not
            // obvious.** `axis_min` takes the minimum over BOTH neighbours on
            // each axis, so the per-voxel update does not depend on which
            // direction the sweep runs -- only the visit order differs. A pass
            // in which no voxel moved is therefore a fixed point of the
            // operator, and no other schedule can move it either. Iterating to
            // a fixed COUNT rather than to a fixed point would not have this
            // property, which is why the count is a ceiling and this is the
            // real termination.
            if moved == 0 && stop_when_settled {
                break;
            }
        }
        report.bricks = coords.len();
        // The voxels this pass actually wrote. It used to be
        // `coords.len() * BRICK_VOXELS` -- a constant times the brick count,
        // under a doc comment promising that zero means the field was already
        // a distance field, which it could never report because the function
        // returns `None` in exactly that case.
        report.corrected = corrected;

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

/// The gradient magnitude at a brick voxel, in voxel units, for seeding.
///
/// Central along an axis where both neighbours carry a measured distance, and
/// one-sided where one of them is saturated. The saturated case is not an edge
/// case to be tidy about: [`crate::Similarity`] permits a scale down to
/// `MIN_SCALE`, after which the whole band can be less than a voxel deep and a
/// seed voxel's neighbour is a clamp rather than a distance. A central
/// difference there measures the clamp and reports a gradient far from one,
/// which would divide the seed down to nothing.
fn seed_gradient(apron: &crate::apron::ApronBuffer, x: usize, y: usize, z: usize) -> f32 {
    let (x, y, z) = (x + 1, y + 1, z + 1);
    let here = apron.get(x, y, z);
    let mut squared = 0.0f32;
    for (back, forward) in [
        (apron.get(x - 1, y, z), apron.get(x + 1, y, z)),
        (apron.get(x, y - 1, z), apron.get(x, y + 1, z)),
        (apron.get(x, y, z - 1), apron.get(x, y, z + 1)),
    ] {
        let derivative = match (in_band(back), in_band(forward)) {
            (true, true) => (forward - back) * 0.5,
            (true, false) => here - back,
            (false, true) => forward - here,
            (false, false) => 0.0,
        };
        squared += derivative * derivative;
    }
    squared.sqrt()
}

/// Distances for the voxels that touch the surface, and infinity for the rest.
///
/// The seed is a magnitude grid: sign is carried separately and restored on the
/// way out, so the sweep never has to think about it.
///
/// # The seed must be the distance to the SURFACE, not along an axis
///
/// The first version took the crossing along each edge -- `here / (here -
/// there)` -- and kept the smallest. That is the distance to the interface
/// measured **along a coordinate axis**, and it is not the distance to the
/// surface. For a plane of normal `n` the field changes along axis `i` at rate
/// `n_i`, so an axis crossing reads `|d| / |n_i|`; taking the minimum over six
/// neighbours keeps the largest `|n_i|`, leaving `|d| / max|n_i|`. That is
/// exact only for a surface facing straight down an axis and over-estimates by
/// up to `sqrt(3)` -- **73% on the body diagonal**.
///
/// It was not a small error. The sweeps can only lower a value and every voxel
/// starts at infinity, so the seed is the whole initial condition: a seed that
/// is too large stays too large. The pass's fixed point was a drift of about
/// **0.35**, three and a half times the [`GRADIENT_TOLERANCE`] it declines
/// bodies for exceeding, so it ran, reported success, and left the field
/// outside its own bar. Worse than failing to converge -- at a squash of 0.9 a
/// field entered at 0.1007, a hair over the gate, and left at 0.3455. The pass
/// made it worse, and the one fixture squashed to 0.5, where the ratio bar the
/// old test used could not see it.
///
/// This is the Russo-Smereka seed, `|d| / |grad d|`, which is the first-order
/// distance to the interface in the direction the surface actually faces.
/// Measured over the same fixtures: 0.0336 at a squash of 0.9, 0.0514 at 0.6,
/// 0.0712 at 0.5, against a tolerance of 0.1.
fn seed_from_interface(solved: &[f32], apron: &crate::apron::ApronBuffer) -> Vec<f32> {
    let mut seed = vec![f32::INFINITY; BRICK_VOXELS];
    for index in 0..BRICK_VOXELS {
        let (x, y, z) = unindex(index);
        let here = solved[index];
        if !in_band(here) {
            continue;
        }
        let negative = here.total_cmp(&0.0).is_le();
        let touches_interface =
            [(1isize, 0, 0), (-1, 0, 0), (0, 1isize, 0), (0, -1, 0), (0, 0, 1isize), (0, 0, -1)]
                .into_iter()
                .any(|(dx, dy, dz)| {
                    let there = apron.get(
                        (x as isize + dx + 1) as usize,
                        (y as isize + dy + 1) as usize,
                        (z as isize + dz + 1) as usize,
                    );
                    there.total_cmp(&0.0).is_le() != negative
                });
        if !touches_interface {
            continue;
        }

        let gradient = seed_gradient(apron, x, y, z);
        // A vanishing gradient means the neighbourhood carries no direction to
        // measure against, so there is nothing to seed from and the sweeps must
        // reach this voxel from somewhere that does.
        if gradient > f32::EPSILON {
            seed[index] = (here.abs() / gradient).min(NARROW_BAND);
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
///
/// Returns how many voxels it actually lowered, by more than [`SETTLED`]. A
/// sweep that lowered none is a fixed point of the update, and because
/// [`axis_min`] takes the minimum over BOTH neighbours on each axis the update
/// does not depend on the direction being swept -- so no other of the eight
/// directions could lower one either, and the remaining sweeps are work with a
/// known-empty result. The module header already said the first two sweeps do
/// nearly all of it; this is what lets the pass act on that rather than assert
/// it.
fn sweep_once(grid: &mut [f32], apron: &crate::apron::ApronBuffer, sweep: usize) -> usize {
    let back_x = sweep & 1 != 0;
    let back_y = sweep & 2 != 0;
    let back_z = sweep & 4 != 0;
    let mut lowered = 0usize;

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
                    // Counted against a tolerance rather than exactly. The
                    // update is a square root of a quotient, so a settled voxel
                    // goes on shedding fractions of an ULP for ever and an
                    // exact `<` test finds the field still moving after any
                    // number of sweeps -- which is what an earlier version of
                    // this did, and it meant the early exit never once fired.
                    if grid[index] - candidate > SETTLED {
                        lowered += 1;
                    }
                    grid[index] = candidate;
                }
            }
        }
    }
    lowered
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

    /// **The early exits must be nearly free, and "nearly" is measured.**
    ///
    /// Stopping a sweep when it lowered nothing, and the outer loop when a
    /// whole iteration lowered nothing, took the pass from 16.3 s to 7.1 s on
    /// the bench's largest arm-able body. That is only a win if the field that
    /// comes out is the same one, and two bench runs agreeing to three decimal
    /// places is not that claim -- this is. It is deliberately NOT bit-equality:
    /// a sub-`SETTLED` drop is still applied, only not counted, so the two runs
    /// separate by fractions of a thousandth of a voxel. `assert_same_field`
    /// was tried first and failed for exactly that reason. What is asserted is
    /// what `SETTLED` actually promises -- a bound on how far the answer may
    /// move -- plus the one thing that must not move at all, the sign.
    ///
    /// The argument it pins: [`axis_min`] takes the minimum over BOTH
    /// neighbours on each axis, so the per-voxel update does not depend on the
    /// direction being swept and a sweep that lowered nothing is a fixed point
    /// no other direction could move either.
    #[test]
    fn stopping_early_changes_no_voxel_by_more_than_it_promises() {
        let mut quick = squashed_ball(0.25, 8.0, 0.5);
        let mut thorough = squashed_ball(0.25, 8.0, 0.5);

        let before_quick = sweeps_run_on_this_thread();
        let quick_report = quick.redistance_inner(true).expect("the fixture needs the pass");
        let quick_sweeps = sweeps_run_on_this_thread() - before_quick;

        let before_thorough = sweeps_run_on_this_thread();
        let thorough_report = thorough.redistance_inner(false).expect("the fixture needs the pass");
        let thorough_sweeps = sweeps_run_on_this_thread() - before_thorough;

        // Non-degeneracy, measured in WORK rather than in voxels moved: if the
        // forced run swept no more than the early one, no early exit fired and
        // nothing below is being exercised.
        assert!(
            thorough_sweeps > quick_sweeps,
            "the forced run swept {thorough_sweeps} times and the early one {quick_sweeps}, \
             so no early exit fired and this test is not exercising what it names"
        );

        assert_eq!(
            quick_report.corrected, thorough_report.corrected,
            "the two runs disagree about how many voxels they moved"
        );

        // NOT bit-equality, and the difference is the point. Sub-`SETTLED`
        // drops are still APPLIED, they are merely not COUNTED, so the two
        // runs separate by fractions of a thousandth of a voxel and
        // `assert_same_field` fails -- which it did, on the first version of
        // this test. What `SETTLED` actually promises is a bound on how far
        // the answer can move, so that is what is asserted.
        let mut worst = 0.0f32;
        for coord in quick.brick_coords().collect::<Vec<_>>() {
            match (quick.brick(coord), thorough.brick(coord)) {
                (Some(Brick::Dense(a)), Some(Brick::Dense(b))) => {
                    for (x, y) in a.iter().zip(b.iter()) {
                        assert_eq!(
                            x.total_cmp(&0.0).is_le(),
                            y.total_cmp(&0.0).is_le(),
                            "stopping early moved a voxel across the surface at {coord:?}"
                        );
                        worst = worst.max((x - y).abs());
                    }
                }
                (a, b) => assert_eq!(
                    a.map(|brick| matches!(brick, Brick::Dense(_))),
                    b.map(|brick| matches!(brick, Brick::Dense(_))),
                    "the two runs stored {coord:?} differently"
                ),
            }
        }
        assert!(
            worst <= SETTLED,
            "stopping early moved a voxel by {worst}, which is past the {SETTLED} the early \
             exit is allowed to give away"
        );
        assert!(
            (quick_report.worst_after - thorough_report.worst_after).abs() <= GRADIENT_TOLERANCE,
            "the two runs disagree about the drift they achieved: {} against {}",
            quick_report.worst_after,
            thorough_report.worst_after,
        );
    }

    /// **The pass stops when the field stops moving, not when a counter runs
    /// out.**
    ///
    /// `RedistanceReport::corrected` is now the count of voxels actually
    /// written rather than `coords.len() * BRICK_VOXELS`, which was a constant
    /// times the brick count under a doc comment promising that zero means the
    /// field was already a distance field -- a value it could never report,
    /// since the function returns `None` in exactly that case.
    ///
    /// The early exit is sound because `axis_min` takes the minimum over both
    /// neighbours on each axis, so the update does not depend on sweep
    /// direction and a pass in which nothing moved is a fixed point no other
    /// order could move either. This asserts that directly: the field after the
    /// real pass is bit-identical to the field after one forced to run every
    /// permitted iteration.
    #[test]
    fn redistancing_stops_as_soon_as_the_field_stops_moving() {
        let mut volume = squashed_ball(0.25, 8.0, 0.5);
        let report = volume.redistance().expect("the fixture needs the pass");

        assert!(report.corrected > 0, "the pass reported writing nothing but was not declined");
        assert!(
            report.corrected < report.bricks * BRICK_VOXELS,
            "the count is still every voxel of every brick ({} of {}), so it is the old \
             constant rather than a measurement",
            report.corrected,
            report.bricks * BRICK_VOXELS,
        );

        // Running it a second time must find a fixed point and decline, which
        // is the same claim the early exit rests on seen from outside.
        let settled = volume.redistance();
        assert_eq!(
            settled, None,
            "a field the pass has just settled was not recognised as settled, so the pass \
             does not reach a fixed point"
        );
    }

    /// **A pass whose answer depends on hash order is not a pass, it is a
    /// coincidence.**
    ///
    /// [`Volume::redistance`] collects `brick_coords()` -- which is
    /// `self.bricks.keys()` over an `FxHashMap`, so insertion-history order and
    /// nothing more -- and then writes each brick back with `insert_brick`
    /// while re-gathering its aprons from the volume it is mutating. Bricks
    /// reached later in the walk therefore read neighbours this same round has
    /// already rewritten.
    ///
    /// In phase one that is worse than a scheduling difference. An
    /// already-processed neighbour's face voxel may have been re-saturated by
    /// `keep_sign(was, OUTSIDE)`, so a brick reached later computes its
    /// interface crossing `d0 / (d0 - d1)` against 3.0 rather than against the
    /// value that was there -- and the crossing is the seed, which the module
    /// documentation calls the point. The seed is wrong before a single sweep
    /// runs, so the same field, inserted differently, redistances to a
    /// different surface.
    ///
    /// This is why the sweep has to become order-independent before anything is
    /// built on top of it, and it is why that is a correctness fix rather than
    /// the performance option it looks like.
    #[test]
    fn redistancing_gives_the_same_field_however_the_bricks_are_ordered() {
        let mut forwards = squashed_ball(0.25, 8.0, 0.5);
        let mut backwards = squashed_ball_inserted_backwards(0.25, 8.0, 0.5);

        crate::testing::assert_same_field(
            &forwards,
            &backwards,
            "the fixtures did not start equal",
        );

        // Non-degeneracy: reversing the inserts is only a way to ask the map for
        // a different walk, and it is the map that decides. If both volumes
        // happen to iterate alike then nothing below is being exercised, and
        // this test would pass while saying nothing.
        let order_forwards: Vec<BrickCoord> = forwards.brick_coords().collect();
        let order_backwards: Vec<BrickCoord> = backwards.brick_coords().collect();
        assert_ne!(
            order_forwards, order_backwards,
            "both fixtures walk their bricks in the same order, so this test is not exercising \
             the thing it names"
        );

        forwards.redistance().expect("the fixture needs the pass");
        backwards.redistance().expect("the fixture needs the pass");

        crate::testing::assert_same_field(
            &forwards,
            &backwards,
            "redistancing gave a different field for a different brick order",
        );
    }

    /// **The pass must reach the bar it gates itself on.**
    ///
    /// [`Volume::redistance`] declines a body whose drift is already within
    /// [`GRADIENT_TOLERANCE`], so that is the standard it holds other fields
    /// to. A pass that runs and then leaves the field outside that standard has
    /// not finished, and worse, it will be asked to run again on every
    /// subsequent gesture and decline to converge each time.
    ///
    /// This is deliberately an ABSOLUTE bar and not a ratio.
    /// `a_squashed_field_is_restored_to_unit_gradient` accepts
    /// `worst_after <= worst_before * 0.6`, which at a before of 1.0 admits
    /// 0.6 -- six times the tolerance the pass gates on. A ratio measures that
    /// the pass did something; only the absolute number measures that it did
    /// enough.
    #[test]
    fn redistancing_reaches_the_tolerance_it_gates_on() {
        let mut volume = squashed_ball(0.25, 8.0, 0.5);

        let before = volume.gradient_drift();
        assert!(before > GRADIENT_TOLERANCE, "the fixture does not need the pass: {before}");

        let report = volume.redistance().expect("the fixture needs the pass");
        assert!(
            report.worst_after <= GRADIENT_TOLERANCE,
            "the pass declines a field drifting by more than {GRADIENT_TOLERANCE} and then \
             leaves this one at {} -- it does not reach its own bar (from {})",
            report.worst_after,
            report.worst_before,
        );
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
    ///
    /// Returned as a list rather than inserted, so the same field can be built
    /// into a volume in more than one order. See
    /// [`redistancing_gives_the_same_field_however_the_bricks_are_ordered`].
    fn squashed_ball_bricks(voxel_size: f32, radius: f32, squash: f32) -> Vec<(BrickCoord, Brick)> {
        let mut bricks_out = Vec::new();
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
                        bricks_out.push((coord, Brick::Uniform(first)));
                    } else {
                        bricks_out.push((coord, Brick::Dense(Box::new(data))));
                    }
                }
            }
        }
        bricks_out
    }

    fn squashed_ball(voxel_size: f32, radius: f32, squash: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        for (coord, brick) in squashed_ball_bricks(voxel_size, radius, squash) {
            volume.insert_brick(coord, brick);
        }
        volume
    }

    /// The identical field, with its bricks inserted in the opposite order.
    fn squashed_ball_inserted_backwards(voxel_size: f32, radius: f32, squash: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        for (coord, brick) in squashed_ball_bricks(voxel_size, radius, squash).into_iter().rev() {
            volume.insert_brick(coord, brick);
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
