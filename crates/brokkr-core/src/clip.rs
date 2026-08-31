// SPDX-License-Identifier: AGPL-3.0-only

//! Cutting the model with a plane.
//!
//! This is where the engine's shape pays off. On a signed distance field, a
//! half-space cut is `max(field, plane)` -- exact, watertight by construction,
//! and it leaves a flat closed face where it cut without any repair step. A mesh
//! sculptor runs boolean remeshing for the same result, and the result is
//! fragile on exactly the input this is for.
//!
//! And this is what BrokkrSculpt is *for*. Scans arrive with defects, and the
//! first thing anyone wants to do is cut the bad part off and keep the rest
//! printable. So the cut has to leave a solid, not a hole: deleting geometry
//! would leave an opening that no slicer will accept, whereas `max` against a
//! plane closes the cut face as a side effect of the arithmetic.
//!
//! # Why this is not `edit_voxels`
//!
//! A cut passes through the whole model, and `edit_voxels` promotes every brick
//! it touches to dense. Most of a scan is solid interior stored as one number
//! per brick, so running the cut through that path would turn a 40 MB model into
//! a gigabyte before it had removed anything.
//!
//! Instead each brick is classified against the plane first, and only the ones
//! the plane actually passes through are ever made dense. The other two cases
//! cost nothing: a brick wholly on the keep side is not touched at all, and one
//! wholly on the cut side is dropped whatever it contained. That is the same
//! shape `resample` uses, and for the same reason.
//!
//! # The mask cuts too
//!
//! A cut is direct manipulation -- a line drawn across what is on screen -- so
//! it answers to the mask exactly as a brush does, and for the same reason a
//! brush does: the user said this part is not to change.
//!
//! That costs the classification one arm. **[`Cut::Removes`] is all-or-nothing
//! and cannot be masked**, because the way it removes is to drop the brick out
//! of the map entirely, and a dropped brick has no voxels left to protect. So a
//! brick the plane would take whole is downgraded to [`Cut::Crosses`] whenever
//! its resolved mask fill is not uniformly free, which sends it through the per
//! voxel path where a protected voxel gets its own value written straight back.
//! Fully protected, that path is skipped in turn and the brick stays exactly
//! where it is -- the whole point, since `remove_brick` would otherwise delete a
//! masked brick and report the field bit-identical only because there is nothing
//! left of it to compare.
//!
//! An unmasked body never reaches any of this: the mask is resolved once for the
//! whole volume, and a body that protects nothing anywhere takes the same
//! branches, and writes the same bits, as it did before masks existed.
//!
//! # What a cut actually costs
//!
//! Measured by `benches/budget.rs`'s `measure_the_cut`, on a 256³ effective
//! volume: a ball of 423 bricks with thirty spurs unioned onto its equator,
//! 41.9 MB resident. **The machine was not idle**, so the timings are upper
//! bounds -- each row is the fastest of five runs, since contention only ever
//! adds time -- and the counts are exact.
//!
//! A cut is three costs, and the interesting thing is which one is biggest.
//! For a plane through the middle of the fixture, before any of the work below:
//!
//! | | ms | share |
//! |---|---|---|
//! | remesh | 5.17 | 44% |
//! | undo encode | 3.62 | 31% |
//! | the cut proper | 2.89 | 25% |
//!
//! **The classification was never any of it.** Seventeen planes against a body
//! they miss costs 0.013 ms, because the `Keeps` early exit rejects almost
//! every brick on the first plane it tries. Every obvious optimisation aims
//! there, and there was nothing there to get.
//!
//! ## Where it went instead
//!
//! | | before | after |
//! |---|---|---|
//! | plane, midline: cut / encode / remesh | 2.89 / 3.62 / 5.17 | **2.02 / 1.05 / 2.41** |
//! | 17-plane prism over a spur | 0.83 / 0.15 / 1.19 | **0.62 / 0.17 / 0.81** |
//! | 17-plane prism straight through | 2.39 / 0.25 / 2.37 | **1.68 / 0.23 / 1.65** |
//! | `Document::clip`, midline | 11.9 | **7.4** |
//!
//! **81% of the remesh was building nothing.** A plane through the middle
//! dirties 669 bricks and 540 of them mesh to zero triangles -- a cut removes
//! material, so most of what it dirties ends up empty on one side or solid on
//! the other -- and each still paid a 34-cubed apron gather and a full
//! surface-nets pass to say so. Two gates in `Volume::mesh_brick` now answer
//! from the neighbourhood's stored form and from a linear scan of the gathered
//! apron. See `Volume::neighbourhood_has_no_surface`.
//!
//! **The undo encode was serial and is embarrassingly parallel.** Every brick
//! run-length encodes independently and `StrokeEdit::from_recording` did them
//! one at a time, next door to a `par_iter` doing the identical shape of job.
//!
//! **Half the recorded bricks were copied for nothing.** A dropped brick IS the
//! prior undo needs, and `record_for_undo` + a bare remove cloned 128 KB and
//! then discarded the original. See `Volume::remove_brick_recording`.
//!
//! **Thirty cuts in a row do not degrade here**, and resident bytes go *down*
//! (41.9 → 39.9 MB). Whatever the repeated-cut risk is, it is not in this
//! crate -- it is the mesh pool's bump allocator, which lives in `brokkr-gpu`
//! and is measured there: thirty trims take it from 1.11x to 1.45x watermark
//! over live.
//!
//! # What a SHAPED cut costs, and where THAT cost was
//!
//! A sixteen-sided prism -- the ceiling hull decimation allows -- plus a depth
//! cap, so seventeen planes. Naively that is 20 million dot products for 37
//! crossing bricks, and it measured **17.8 ms against a 4 ms edit budget**.
//! Three changes, all bit-exact, took it to 1.68 ms:
//!
//! 1. **Per-brick plane pruning** (`classify_active`). A plane already
//!    saturated across a brick cannot decide any voxel in it. Most bricks a
//!    hull crosses are straddled by two or three faces, not seventeen.
//! 2. **Specialised write loops** for one to four planes, which is nearly every
//!    crossing brick. Each holds its planes as values the compiler keeps in
//!    registers and makes ONE pass over the brick.
//! 3. **Plane-major writing** above that (`write_cut_brick_plane_major`): one
//!    plane per pass over a scratch buffer, so it is still in a register, where
//!    holding them all at once is no longer possible.
//!
//! `several_planes_write_the_minimum_of_their_distances_exactly` compares the
//! result bit for bit against the definition written out through `edit_voxels`,
//! because "this optimisation cannot change the answer" is the kind of claim
//! that is convincing and occasionally wrong.
//!
//! The specialisation was found by accident and is worth more than it looks:
//! routing the SINGLE-plane path through a slice, which is the obvious way to
//! write this, cost **9.6 ms against 2.2 ms** on the plane through the middle
//! -- a four-fold regression in the cut that ships today, invisible to every
//! test, caused by nothing but the plane no longer being a value the compiler
//! could keep in a register.

use glam::{IVec3, Vec3};

use crate::body::{Document, NodeId};
use crate::brick::{BRICK_DIM, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, brick_index};
use crate::mask::{MaskField, UNMASKED};
use crate::undo::{Change, Entry};
use crate::volume::{Freedom, Volume};

/// A half-space to cut away.
///
/// Everything on the side the `normal` points toward is removed; everything
/// behind it is kept. The plane is infinite and the cut passes through the
/// entire model, which is what ZBrush's clip brushes do and what lopping a
/// defect off a scan needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipPlane {
    /// Any point on the plane.
    pub point: Vec3,
    /// Unit normal. The side it points to is the side that goes.
    pub normal: Vec3,
}

impl ClipPlane {
    /// Build from a point and a direction, normalising the direction.
    ///
    /// Returns `None` for a direction with no length, because a plane with no
    /// normal has no sides and the caller has almost certainly derived it from
    /// a degenerate drag -- a click rather than a stroke.
    pub fn new(point: Vec3, normal: Vec3) -> Option<Self> {
        let normal = normal.try_normalize()?;
        point.is_finite().then_some(Self { point, normal })
    }

    /// Signed distance from a world point to the plane, in millimetres.
    /// Positive on the side that gets cut away.
    #[inline]
    pub fn distance(&self, at: Vec3) -> f32 {
        (at - self.point).dot(self.normal)
    }

    /// The smallest and largest signed distance over an axis aligned box.
    ///
    /// The distance is linear, so the extremes are at opposite corners and can
    /// be had from the centre and the half extent without visiting all eight.
    ///
    /// `pub(crate)` for [`crate::generate`]'s half-space mask, which is the
    /// same classification writing protection instead of distance -- and
    /// sharing it is the point: a second copy of this arithmetic would let the
    /// mask and the cut disagree about which bricks a plane touches.
    #[inline]
    pub(crate) fn range_over_box(&self, centre: Vec3, half: Vec3) -> (f32, f32) {
        let middle = self.distance(centre);
        let reach = half.x * self.normal.x.abs()
            + half.y * self.normal.y.abs()
            + half.z * self.normal.z.abs();
        (middle - reach, middle + reach)
    }
}

/// What a plane does to one brick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cut {
    /// Wholly behind the plane by at least the band: nothing changes.
    Keeps,
    /// Wholly past the plane by at least the band: the brick goes entirely.
    Removes,
    /// The plane passes through it, so every voxel has to be resolved.
    Crosses,
}

/// What a plane cut did to ONE body's bricks.
///
/// **Two counts and not one**, because a brick the mask kept whole and a brick
/// the plane never reached are the same zero to the caller and completely
/// different things to the user: the first means "your mask stopped this", the
/// second means "your line missed". [`Document::clip`] carries both up so
/// the status line can tell them apart.
///
/// The three behind them -- `classified`, `crossed`, `removed` -- are not for
/// the user at all. They are the classification's own census, and they exist
/// because "the shaped cut is no more expensive than the plane" is a claim
/// about which arm each brick took, which no count of what *changed* can
/// settle: a plane and a loop that remove the same bricks can reach that
/// result having promoted wildly different numbers of them to dense.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClipCounts {
    /// Bricks whose contents changed.
    pub changed: usize,
    /// Bricks the mask kept WHOLE that the cut would otherwise have changed.
    ///
    /// Only fully protected bricks are counted, and only after checking that
    /// the cut really had something to do in them -- see
    /// [`cut_would_change_a_voxel`]. A brick the mask merely thinned is in
    /// [`ClipCounts::changed`], where it belongs, and a fully protected brick
    /// the plane only grazed is in neither, so "the mask blocked the cut" is
    /// never said about a cut that had nothing to remove anyway.
    pub spared_by_mask: usize,
    /// Bricks that reached the plane arithmetic at all.
    ///
    /// Today that is every brick in the body, because the classification loop
    /// walks the whole map. It is counted rather than inferred from
    /// [`crate::VolumeStats`] because it is about to stop being every brick:
    /// a bounded cutter can be rejected against a brick's integer coordinate
    /// before any float touches it, and the only way to say whether such a
    /// filter is working is to have measured what it filtered.
    pub classified: usize,
    /// Bricks the cut had to promote dense and resolve one voxel at a time.
    ///
    /// **The cost number.** Everything expensive about a cut is proportional to
    /// this: the dense promotion, the 128 KB undo record, the per voxel loop,
    /// and -- through [`crate::Volume::mark_dirty_voxel_range`] -- most of the
    /// remesh that follows. A cut whose `crossed` grows without its `changed`
    /// growing is doing work it throws away.
    pub crossed: usize,
    /// Bricks dropped whole, without ever being made dense.
    ///
    /// The cheap arm, and worth counting separately from
    /// [`ClipCounts::changed`] (which includes it) because it is the one a
    /// shape can win: a loop drawn *around* a lump takes bricks whole that a
    /// plane covering the same silhouette would have had to cross.
    pub removed: usize,
}

/// The removed region's signed distance at a point, in millimetres.
///
/// The region is the INTERSECTION of the half-spaces, so a point is inside it
/// only where every plane says so and the distance to the whole is the smallest
/// of the distances to the parts: `-max_i(-d_i)` is `min_i d_i`.
///
/// **The single-plane path is the plane, bit for bit.** The first distance is
/// taken outright and only the rest go through `min`, so with one plane no
/// comparison happens at all and the value is character-for-character what
/// `plane.distance(at)` returned before this function existed. That is not an
/// optimisation; it is what makes today's cut provably unchanged, and it is why
/// this is not written as a fold from `f32::INFINITY`.
///
/// # Where it over-estimates
///
/// Exactly right inside the region and outside a single face, and too large
/// outside a convex EDGE, where the true distance is to the edge itself and
/// this reports the distance to the nearer face's plane. At a wedge of interior
/// angle `t`, at radius `r`, it over-estimates by `r * (1 - sin(t/2))` -- 0.88
/// of a voxel at a right angle and 2.48 at 20 degrees, taken at the band edge
/// where `r` is three voxels.
///
/// The direction is the good one. Over-estimating the removed region's distance
/// makes `max` take slightly LESS at the corner, so a cut corner comes out
/// rounded rather than sharpened, and a knife edge is the printability failure
/// no watertightness check can see. It is still worth bounding, which is why
/// the caller drops the sharpest hull vertices first rather than the smallest.
///
/// `pub(crate)` for [`crate::generate`]'s cutter mask, which is this same
/// region written as protection instead of removal -- and sharing it is the
/// point, for the reason [`ClipPlane::range_over_box`] gives about the
/// classification: two copies of the arithmetic would let the mask and the cut
/// disagree about where the region is.
#[inline]
pub(crate) fn cut_distance(planes: &[ClipPlane], at: Vec3) -> f32 {
    let Some((first, rest)) = planes.split_first() else {
        // No planes is no region. Returning the identity for `min` here would
        // be `INFINITY`, which `max` would write over every voxel in the model.
        // Callers refuse an empty set before reaching this; the value below is
        // the harmless answer if one ever does not.
        return f32::NEG_INFINITY;
    };
    let mut smallest = first.distance(at);
    for plane in rest {
        let (held, now) = (smallest, plane.distance(at));
        debug_assert!(now.is_finite());
        smallest = if held < now { held } else { now };
    }
    smallest
}

/// What a convex cutter does to one brick, and which of its planes still
/// matter there.
///
/// `half` is the caller's, and must be: it is one voxel SHORT of the nominal
/// brick size, because a brick spans `BRICK_DIM` sample positions and the
/// distance from its first to its last is one voxel less than its extent.
/// Recomputing it here from `BRICK_DIM * voxel_size` would widen every brick by
/// a voxel and quietly move which ones classify as crossing.
///
/// Both verdicts are sound rather than conservative, and each for its own
/// reason:
///
/// * **Keeps** on the first plane whose farthest corner is a full band behind
///   it. `min_i d_i <= d_j` everywhere, so if one plane puts the whole brick
///   behind itself the minimum is behind it too, whatever the other planes say.
///   This is the early exit, and it is the common case: most bricks in a body
///   are nowhere near the cutter, and most of those are rejected by the first
///   plane tested.
/// * **Removes** only when EVERY plane puts the whole brick a full band past
///   it, which is exactly saturation inside the polyhedron.
///
/// The two are tested in the opposite order from the single-plane code this
/// replaces, which checked `Removes` first. For one plane they are mutually
/// exclusive -- `nearest <= farthest` and the band is positive -- so the
/// reorder cannot change a single-plane verdict. For several it must be this
/// way round: `Keeps` is decidable from one plane and `Removes` is not.
///
/// # Why it also narrows the plane set, and why that is exact
///
/// **This is where the shaped cut's real cost lives.** A plane whose nearest
/// corner is already a full band past it contributes at least `OUTSIDE` at
/// every point of the brick, in voxels. [`cut_voxel`] saturates there:
/// `max(old, OUTSIDE)` clamped is `OUTSIDE` whatever `old` was, and whatever
/// the mask lets through, since the blend runs toward that same target. So for
/// any voxel in this brick either some other plane is smaller -- in which case
/// the minimum is that one and this plane never mattered -- or none is, in
/// which case the minimum is still at least `OUTSIDE` and the voxel saturates
/// identically. Dropping it cannot change one written bit.
///
/// The measurement is what made this worth doing rather than the parallel
/// classification that was originally planned. Classifying a sixteen-plane hull
/// against a body it misses costs 0.008 ms where one plane costs 0.002 -- the
/// classification was never the problem. The per voxel minimum was: 33 crossing
/// bricks at eighteen planes is 19.5 million dot products, and it measured
/// 17.8 ms against a 4 ms edit budget. Most bricks a hull crosses are straddled
/// by two or three of its faces, not eighteen.
///
/// `active` is cleared and refilled rather than returned, so a cut over
/// thousands of bricks allocates once. It is meaningful only for
/// [`Cut::Crosses`]; the other two verdicts leave it in whatever state the scan
/// reached, and nothing reads it.
///
/// **This is where the shaped cut's real cost lives, and dropping a plane here
/// is exact rather than approximate.** A plane whose nearest corner is already
/// a full band past it contributes at least `OUTSIDE` at every point of the
/// brick, in voxels. `cut_voxel` saturates there: `max(old, OUTSIDE)` clamped is
/// `OUTSIDE` whatever `old` was, and whatever the mask lets through, since the
/// blend runs toward that same target. So for any voxel in this brick either
/// some other plane is smaller -- in which case the minimum is that one and this
/// plane never mattered -- or none is, in which case the minimum is still at
/// least `OUTSIDE` and the voxel saturates identically. Removing it from the
/// slice cannot change one written bit.
///
/// The measurement is what made this worth doing rather than the parallelism
/// the plan proposed. Classification of a sixteen-plane hull over a body it
/// misses costs 0.008 ms against one plane's 0.002 -- it was never the problem.
/// The per voxel minimum was: 33 crossing bricks at eighteen planes is 19.5
/// million dot products, and it measured 17.8 ms against a 4 ms edit budget.
/// Most bricks a hull crosses are straddled by two or three of its faces, not
/// eighteen.
///
/// `active` is cleared and refilled rather than returned, so a cut over
/// thousands of bricks allocates once. It is left in an unspecified state
/// unless the verdict is [`Cut::Crosses`]; nothing else reads it.
fn classify_active(
    planes: &[ClipPlane],
    centre: Vec3,
    half: Vec3,
    band_mm: f32,
    active: &mut Vec<ClipPlane>,
) -> Cut {
    active.clear();
    for plane in planes {
        let (nearest, farthest) = plane.range_over_box(centre, half);
        if farthest <= -band_mm {
            return Cut::Keeps;
        }
        if nearest < band_mm {
            active.push(*plane);
        }
    }
    // Every plane saturated positive across the brick, which is exactly the
    // condition `classify` calls `Removes`.
    if active.is_empty() { Cut::Removes } else { Cut::Crosses }
}

/// Whether the cut would move any voxel of a brick it is not allowed to touch.
///
/// A read-only pass, and it is worth its cost precisely because it is the
/// alternative to guessing: without it a fully protected brick that the plane
/// merely grazed inside the band -- one where `max` would have changed nothing
/// even unmasked -- would be reported as spared, and the status line would tell
/// the user their mask stopped a cut that had nothing to stop.
///
/// It costs one brick's worth of reads and no allocation, against the dense
/// promotion plus 128 KB undo record that actually writing the brick would
/// cost, and it returns at the first voxel that moves.
fn cut_would_change_a_voxel(
    brick: &Brick,
    origin: IVec3,
    voxel_size: f32,
    planes: &[ClipPlane],
) -> bool {
    for z in 0..BRICK_DIM {
        for y in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                let at = (origin + IVec3::new(x as i32, y as i32, z as i32)).as_vec3() * voxel_size;
                let old = brick.get(x, y, z);
                if cut_voxel(old, cut_distance(planes, at) / voxel_size, 1.0) != old {
                    return true;
                }
            }
        }
    }
    false
}

/// Write one dense brick's worth of the cut.
///
/// **Generic over how the cut distance is found, and that is the whole point.**
/// It is instantiated twice -- once with a closure holding a single `Copy`
/// [`ClipPlane`], once with one that takes the minimum over a slice -- so the
/// single-plane instance compiles to a loop with the plane's point and normal
/// live in registers across all 32,768 voxels, which is exactly the code that
/// shipped before the cut took a shape.
///
/// Writing it once and calling it twice is not tidiness. Fusing the two into
/// one loop that reads the plane out of a slice measured **9.6 ms against
/// 2.2 ms** on a plane through the middle of the bench fixture: the compiler
/// cannot keep a slice element in a register across a write to `data`, so it
/// reloaded the plane for every voxel. Duplicating the loop body instead would
/// have worked equally well and left two copies of the mask blend, the clamp
/// and the millimetre-to-voxel conversion to keep in step.
///
/// `distance_mm` returns MILLIMETRES, as [`ClipPlane::distance`] does. The
/// conversion to voxels happens here, once, because the field stores voxels and
/// mixing the two moves the cut by a factor of the voxel size.
#[inline]
fn write_cut_brick(
    data: &mut [f32; BRICK_DIM * BRICK_DIM * BRICK_DIM],
    origin: IVec3,
    voxel_size: f32,
    freedom: &Freedom,
    uniform: Option<f32>,
    distance_mm: impl Fn(Vec3) -> f32,
) {
    for z in 0..BRICK_DIM {
        for y in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                let at = (origin + IVec3::new(x as i32, y as i32, z as i32)).as_vec3() * voxel_size;
                let slot = brick_index(x, y, z);
                let cut = distance_mm(at) / voxel_size;
                let free = match uniform {
                    Some(free) => free,
                    None => freedom.at(slot),
                };
                data[slot] = cut_voxel(data[slot], cut, free);
            }
        }
    }
}

/// Write one dense brick's worth of the cut, taking the minimum PLANE by plane
/// rather than voxel by voxel.
///
/// Same result as calling [`write_cut_brick`] with a closure that minimises over
/// the slice, and a different shape of loop. Each pass holds exactly one plane,
/// so its point and normal stay in registers across all 32,768 voxels and the
/// arithmetic is a straight-line dot product over contiguous memory; the
/// voxel-major form has to read a different plane out of a slice on every
/// iteration and cannot keep any of them.
///
/// It costs a 128 KB scratch buffer, reused across every brick of the cut, and
/// it reads and writes that buffer once per plane. That trade only pays when
/// there are several planes, which is why the single-plane path does not come
/// through here -- there it would be two passes over 128 KB where one fused
/// loop does.
///
/// The minimum is taken in MILLIMETRES and converted once at the end, so this
/// and the voxel-major form round identically: neither divides before the min.
fn write_cut_brick_plane_major(
    data: &mut [f32; BRICK_DIM * BRICK_DIM * BRICK_DIM],
    scratch: &mut [f32; BRICK_DIM * BRICK_DIM * BRICK_DIM],
    origin: IVec3,
    voxel_size: f32,
    freedom: &Freedom,
    uniform: Option<f32>,
    planes: &[ClipPlane],
) {
    let Some((first, rest)) = planes.split_first() else {
        return;
    };

    // The first plane fills rather than minimises, which is what makes this
    // agree with `cut_distance` to the bit: that also takes the first distance
    // outright instead of folding from an infinity.
    for z in 0..BRICK_DIM {
        for y in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                let at = (origin + IVec3::new(x as i32, y as i32, z as i32)).as_vec3() * voxel_size;
                scratch[brick_index(x, y, z)] = first.distance(at);
            }
        }
    }
    for plane in rest {
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let at =
                        (origin + IVec3::new(x as i32, y as i32, z as i32)).as_vec3() * voxel_size;
                    let slot = brick_index(x, y, z);
                    let (held, now) = (scratch[slot], plane.distance(at));
                    debug_assert!(now.is_finite());
                    scratch[slot] = if held < now { held } else { now };
                }
            }
        }
    }

    for slot in 0..data.len() {
        let free = match uniform {
            Some(free) => free,
            None => freedom.at(slot),
        };
        data[slot] = cut_voxel(data[slot], scratch[slot] / voxel_size, free);
    }
}

/// One voxel's new distance: the cut, admitted by as much of itself as the mask
/// lets through.
///
/// `free` is the mask factor in `0..=1` -- 1 is unprotected and 0 is fully
/// protected -- and the blend is toward `old.max(cut)`, so the result always
/// lies between the voxel's own value and the value an unmasked cut would have
/// given it. A cut can therefore only ever remove less through a mask, never
/// more and never something else.
///
/// **The clamp is not optional, and it has to happen BEFORE the blend.** `cut`
/// is `cut_distance(planes, at) / voxel_size` and is unbounded -- a plane a
/// metre away from a quarter millimetre voxel gives four thousand -- so without
/// it a `free` of 1 writes a distance far outside the narrow band into the
/// field, which the project loader then refuses on the next open.
///
/// # Why the order matters, and what it cost
///
/// Clamping afterwards instead is correct for an unmasked cut and **silently
/// wrong for every partially masked one**, which is how it shipped. The blend
/// runs toward `target`, so with the clamp at the end the target is that raw
/// four thousand: a voxel at `-3` under half protection, with the cut twenty
/// voxels past it, came out as `-3 + (20 - -3) * 0.5 = 8.5`, clamped to
/// `OUTSIDE`. It should have come out at `0.0` -- half way from `-3` to the
/// most a cut can ever write.
///
/// So a half-protected voxel was removed as completely as an unprotected one,
/// and the mask's feathered edge -- the thing `mask.rs` requires every writer
/// to produce, so that protection is a gradient and not a step -- did nothing
/// at all except exactly where `free` was zero. The further the cutter was from
/// a voxel, the more completely the protection was overwhelmed, which is the
/// opposite of what anyone would predict and is why it survived: the case that
/// looks correct while you test it, a cut right at the mask's edge, is the one
/// case where the two orders nearly agree.
///
/// Clamping first makes the blend a convex combination of two values that are
/// both already in the band, so the result is in the band by construction. The
/// trailing clamp is kept anyway: it costs nothing, it makes the `free >= 1.0`
/// arm's output obviously in range, and a `free` outside `0..=1` would
/// otherwise extrapolate.
///
/// # Why `free == 1` takes the plain `max` instead of the blend
///
/// Byte identity, not speed. `old + (m - old) * 1.0` is not `m` in binary
/// floating point whenever `m - old` rounds: over five million random pairs
/// with `old` in the band and the cut within a few voxels of it, 65,040 of them
/// came out one bit different AFTER the clamp. Running an unmasked cut through
/// the blend would therefore change the last bit of voxels in bricks nothing was
/// masking, in every model, on the first build that shipped masking.
#[inline]
fn cut_voxel(old: f32, cut: f32, free: f32) -> f32 {
    let target = old.max(cut).clamp(INSIDE, OUTSIDE);
    let new = if free >= 1.0 { target } else { old + (target - old) * free };
    new.clamp(INSIDE, OUTSIDE)
}

impl Volume {
    /// Cut away everything on the normal side of `plane`.
    ///
    /// Returns how many bricks changed and how many the mask kept whole. Both
    /// are zero when the plane misses the model entirely -- worth checking,
    /// because a cut that did nothing should not become an undo entry.
    ///
    /// Bracket a call in [`Volume::begin_stroke`] and [`Volume::end_stroke`] to
    /// make it undoable, exactly as a brush stroke is. Only bricks that really
    /// change are recorded, so cutting a corner off a large model costs an undo
    /// entry proportional to the corner rather than to the model.
    pub fn clip(&mut self, plane: ClipPlane) -> ClipCounts {
        self.clip_convex(std::slice::from_ref(&plane))
    }

    /// Cut away the convex region every plane in `planes` agrees on.
    ///
    /// The removed region is the INTERSECTION of the half-spaces -- material
    /// goes only where every plane says it should. One plane is therefore
    /// [`Volume::clip`]'s infinite half-space, which is why that is a wrapper
    /// over this rather than a second implementation: `min` over one element is
    /// the element, so the arithmetic is not merely equivalent, it is the same
    /// arithmetic.
    ///
    /// **An empty slice removes nothing**, and that is a decision rather than a
    /// fallback. The intersection of no half-spaces is all of space, so the
    /// mathematically faithful answer is to delete the entire model -- which is
    /// never what a caller with an empty list meant. It is what one would get
    /// from a gesture whose plane construction failed, and there is no gesture
    /// for which erasing the document is the right recovery.
    ///
    /// Bracket a call in [`Volume::begin_stroke`] and [`Volume::end_stroke`] to
    /// make it undoable, exactly as a brush stroke is. Only bricks that really
    /// change are recorded, so cutting a corner off a large model costs an undo
    /// entry proportional to the corner rather than to the model.
    pub fn clip_convex(&mut self, planes: &[ClipPlane]) -> ClipCounts {
        if planes.is_empty() {
            return ClipCounts::default();
        }
        self.with_mask_lifted(|volume, mask| volume.clip_convex_masked(planes, mask))
    }

    /// The cut proper, with the mask already lifted off the volume.
    ///
    /// Split out only so that [`Volume::with_mask_lifted`] can hold the mask
    /// across it; everything the cut does is here.
    fn clip_convex_masked(&mut self, planes: &[ClipPlane], mask: Option<&MaskField>) -> ClipCounts {
        let voxel_size = self.voxel_size();
        let band_mm = NARROW_BAND * voxel_size;
        let brick_mm = BRICK_DIM as f32 * voxel_size;
        // A brick spans `BRICK_DIM` voxel positions, so the distance from its
        // first to its last sample is one voxel short of its nominal size.
        let half = Vec3::splat(0.5 * (brick_mm - voxel_size));

        let coords: Vec<BrickCoord> = self.brick_coords().collect();
        let mut counts = ClipCounts::default();
        let mut touched: Vec<BrickCoord> = Vec::new();
        // Reused across every brick: see `classify_active`.
        let mut active: Vec<ClipPlane> = Vec::with_capacity(planes.len());
        // 128 KB, and only ever allocated by a cut that really has more than
        // one plane crossing a brick -- so a plane cut, which is every cut that
        // ships today, never pays for it. See `write_cut_brick_plane_major`.
        let mut scratch: Option<Box<[f32; BRICK_DIM.pow(3)]>> = None;

        for coord in coords {
            counts.classified += 1;
            let origin = coord.origin();
            let centre = origin.as_vec3() * voxel_size + half;
            let mut verdict = classify_active(planes, centre, half, band_mm, &mut active);

            // Removing is dropping the brick, which cannot be done by halves, so
            // anything the mask has to say about this brick sends it down the
            // per voxel path instead. `protection_fill` is the RESOLVED
            // protection and not the stored byte, so an empty map under
            // inversion -- which is exactly what Mask All is -- downgrades here
            // as it must.
            if verdict == Cut::Removes
                && mask.is_some_and(|mask| mask.protection_fill(coord) != Some(UNMASKED))
            {
                verdict = Cut::Crosses;
            }

            match verdict {
                // Every plane's contribution is already saturated positive
                // across the whole brick, so their minimum is too and `max`
                // would make every voxel OUTSIDE. An absent brick reads as
                // OUTSIDE, so dropping it is both correct and free.
                Cut::Removes => {
                    // One call, because the brick being dropped IS the prior
                    // undo needs: recording and then removing copies 128 KB
                    // and immediately throws the original away. See
                    // `Volume::remove_brick_recording`.
                    self.remove_brick_recording(coord);
                    counts.changed += 1;
                    counts.removed += 1;
                    touched.push(coord);
                }
                // `max(field, cutter)` is `field` everywhere in here. Touching
                // it would cost a dense promotion and an undo entry for no
                // change.
                Cut::Keeps => {}
                Cut::Crosses => {
                    let Some(present) = self.brick(coord) else {
                        // Absent already reads as OUTSIDE, and no `free` can
                        // lower it, so there is nothing a cut can do here.
                        continue;
                    };
                    // Resolved once for the whole brick, exactly as the brush's
                    // `write_voxels` resolves it, so an unmasked body and a body
                    // whose mask collapsed this brick to a tile both pay nothing
                    // per voxel.
                    let freedom = Freedom::resolve(mask, coord);
                    let uniform = freedom.uniform();
                    if uniform.is_some_and(|free| free <= 0.0) {
                        // Fully protected. The loop below would write every
                        // voxel back exactly as it found it, after promoting the
                        // brick to dense and recording 128 KB of undo for it, so
                        // the brick is left alone entirely -- which is also what
                        // keeps a downgraded `Removes` brick present instead of
                        // deleted.
                        // **The full plane set, not the pruned one.** A brick
                        // the mask downgraded from `Removes` arrives here with
                        // an EMPTY active set -- empty is precisely how
                        // `classify_active` says "every plane saturates across
                        // this brick" -- and the minimum over no planes is
                        // negative infinity, which reports that the cut would
                        // have changed nothing. That is the opposite of the
                        // truth for a brick the cutter would have taken whole,
                        // and it drops the brick out of `bricks_spared_by_mask`
                        // -- so the status line says "the cut missed the model"
                        // over a cut a mask had just blocked completely, which
                        // is the one message that sends a user off to redraw a
                        // gesture that was never the problem.
                        //
                        // Only fully protected bricks reach this, so paying the
                        // full plane count here costs nothing anywhere else.
                        if cut_would_change_a_voxel(present, origin, voxel_size, planes) {
                            counts.spared_by_mask += 1;
                        }
                        continue;
                    }
                    // Recorded before the brick is taken, because the recorder
                    // reads the prior contents out of the map. That copy is the
                    // one undo needs; taking rather than cloning avoids making a
                    // second one of every brick along the cut.
                    self.record_for_undo(coord);
                    let Some(mut brick) = self.take_brick(coord) else {
                        continue;
                    };
                    // Counted here rather than at the verdict, because the two
                    // arms above reach this match with a `Crosses` verdict and
                    // cost nothing dense: an absent brick has nothing to
                    // promote, and a fully protected one is deliberately left
                    // where it is. What this number is for is the price of the
                    // promotion, so it counts promotions.
                    counts.crossed += 1;
                    let data = brick.make_dense();
                    // Specialised on the plane count, and it is worth the two
                    // call sites. See `write_cut_brick`: with one plane the
                    // closure captures a `Copy` `ClipPlane` that stays in
                    // registers for all 32,768 voxels, where reading it back
                    // out of a slice each time measured four times slower.
                    match active.as_slice() {
                        [only] => {
                            let only = *only;
                            write_cut_brick(data, origin, voxel_size, &freedom, uniform, |at| {
                                only.distance(at)
                            });
                        }
                        many => {
                            let scratch =
                                scratch.get_or_insert_with(|| Box::new([0.0; BRICK_DIM.pow(3)]));
                            write_cut_brick_plane_major(
                                data, scratch, origin, voxel_size, &freedom, uniform, many,
                            );
                        }
                    }
                    // A brick the cutter merely grazed can come out entirely
                    // empty, and one it barely entered can come out unchanged.
                    // Collapsing releases the 128 KB either way.
                    // Already recorded and already taken out of the map, so
                    // these only decide what goes back in.
                    match brick.is_collapsible() {
                        Some(value) if value >= OUTSIDE => {}
                        Some(value) => self.insert_brick(coord, Brick::Uniform(value)),
                        None => self.insert_brick(coord, brick),
                    }
                    counts.changed += 1;
                    touched.push(coord);
                }
            }
        }

        // Every brick that changed, plus its neighbours: a brick's apron reads
        // one voxel into each of the 26 around it, so remeshing only what was
        // written would leave a seam along the cut face.
        //
        // The dedup that keeps this from costing 27 remeshes per changed brick
        // is the `FxHashSet` inside `mark_dirty_voxel_range`, not anything
        // here: `touched` is a plain `Vec` and deliberately so, since a brick
        // reaches it at most once. Measured at 3.03 dirty bricks per brick
        // changed rather than 27 -- see the module doc.
        for coord in touched {
            self.mark_dirty_voxel_range(coord.origin(), coord.max_voxel());
        }
        counts
    }
}

/// What one plane cut did, across the whole document.
///
/// **Both counts, because the status line has to tell "the line missed
/// everything" apart from "it crossed three bodies and changed two".** Those
/// two produce the same brick count and mean completely different things: the
/// first is a gesture that went nowhere near the model, the second is a cut
/// that passed through empty space inside bodies it really did cross.
pub struct CutOutcome {
    /// Bricks changed, summed over every body.
    pub bricks: usize,
    /// Bodies at least one brick of which changed.
    /// The bodies at least one brick of which changed, in node order.
    ///
    /// A list rather than a count, for the same reason
    /// [`CutOutcome::bodies_spared_by_mask`] is one: the caller has real work
    /// that is proportional to WHICH bodies changed, not how many. Reporting
    /// loose pieces means a connectivity walk, and walking a body the cut never
    /// reached costs as much as walking one it did and can only ever find what
    /// was already there.
    ///
    /// The alternative -- asking each body afterwards whether it has anything
    /// dirty -- reads the same answer off a set that happens not to have been
    /// drained yet, and would go quietly wrong the day a remesh moved above the
    /// status line.
    pub bodies_cut: Vec<NodeId>,
    /// Bodies the half-space reached at all, whether or not it found anything
    /// there to remove. Always at least `bodies_cut.len()`.
    pub bodies_crossed: usize,
    /// Bricks the mask kept whole, summed over every body.
    ///
    /// **This is what stops a fully masked body reporting "the cut missed the
    /// model".** A cut the mask blocked entirely produces `bricks == 0` exactly
    /// as a cut that went nowhere near anything does, and those two read
    /// completely differently to whoever drew the line: one of them is a mask
    /// doing its job and the other is a gesture to try again.
    pub bricks_spared_by_mask: usize,
    /// The bodies that spared at least one brick, in node order.
    ///
    /// A list rather than a count because the message names the body -- "the
    /// mask on Left Ear blocked the cut" is actionable and "a mask blocked the
    /// cut" leaves the user hunting through the panel for which one. Empty
    /// whenever nothing was spared, so it allocates only on a cut a mask really
    /// did block.
    pub bodies_spared_by_mask: Vec<NodeId>,
    /// **ONE entry for the whole gesture**, or `None` when nothing changed.
    ///
    /// It is built here rather than handed back as a list of changes for the
    /// caller to wrap, so that "one gesture is one undo entry" is a property of
    /// this type rather than of everybody who calls it remembering. Undoing a
    /// cut that crossed four bodies has to put all four back or none of them:
    /// half a cut is not a document state anything downstream is written
    /// against.
    pub entry: Option<Entry>,
}

impl Document {
    /// Cut every body that is DRAWN with one plane, as one gesture.
    ///
    /// `visible` is indexed by node position and comes from
    /// [`Document::display_visibility`], which is where solo is applied -- so
    /// solo narrows the cut, and a hidden body a line passes over comes out
    /// bit-identical. That is the decision: **a cut is a line the user draws
    /// across what they can see**, so it acts on what is drawn and nothing else.
    /// Cutting a body that is not on screen would set `unsaved`, push history,
    /// pay a remesh and an upload, and change not one pixel.
    ///
    /// # Why this is not a loop over `Volume::clip` at the call site
    ///
    /// Two things that a call-site loop gets wrong and this does not. The undo
    /// entry is ONE [`Entry`] of N `Change::Bricks` rather than N entries, so
    /// one ctrl+Z undoes the whole cut. And the box gate below skips a body the
    /// half-space cannot reach without walking its brick map at all, which is
    /// what keeps a cut across a two-body document from costing a full scan of
    /// the dragon sitting behind it.
    ///
    /// [`Volume::clip`] itself still cuts one body; this sums what each of them
    /// reports, both the bricks that changed and the bricks a mask kept whole.
    pub fn clip(&mut self, plane: ClipPlane, visible: &[bool]) -> CutOutcome {
        self.clip_convex(std::slice::from_ref(&plane), visible)
    }

    /// Cut every body that is DRAWN with a convex cutter, as one gesture.
    ///
    /// [`Document::clip`] is this with one plane, and everything its
    /// documentation says holds here: the gesture acts on what is drawn, one
    /// gesture is one [`Entry`], and a body the cutter cannot reach is skipped
    /// without walking its brick map.
    ///
    /// # The body gate generalises by reusing the arm the plane version threw
    ///   away
    ///
    /// The single-plane gate binds `let (_, farthest)` and rejects a body lying
    /// wholly BEHIND the plane. That is the same test as [`classify`]'s
    /// `Cut::Keeps`, run over a body's bounds instead of a brick's, and it
    /// generalises the same way: `min_i d_i <= d_j` everywhere, so a body that
    /// any single plane puts wholly behind it cannot be reached by the
    /// intersection either.
    ///
    /// What is deliberately NOT done here is the mirror of `Cut::Removes` --
    /// noticing that the cutter swallows a body whole and dropping every brick
    /// without looking. It would be sound, but a body that small is already
    /// cheap to walk, and the arm would be a second place where "the mask can
    /// veto a whole-brick drop" has to be remembered.
    pub fn clip_convex(&mut self, planes: &[ClipPlane], visible: &[bool]) -> CutOutcome {
        debug_assert_eq!(
            visible.len(),
            self.nodes().len(),
            "the visibility mask is indexed by node position"
        );

        let band_mm = NARROW_BAND * self.voxel_size();
        // Resolved up front, so the loop below can take each body mutably in
        // turn without holding a borrow of the node list.
        let crossed: Vec<NodeId> = if planes.is_empty() {
            // No cutter reaches nothing. Returning every visible body here
            // would report "the cut crossed 3 bodies and found nothing", which
            // describes a cut that happened.
            Vec::new()
        } else {
            self.nodes()
                .iter()
                .enumerate()
                .filter(|(index, _)| visible.get(*index).copied().unwrap_or(false))
                .filter_map(|(_, node)| {
                    let (low, high) = node.bounds()?;
                    let centre = (low + high) * 0.5;
                    let half = (high - low) * 0.5;
                    // Wholly behind ANY ONE plane by at least the band: every
                    // brick in it would classify as `Keeps`, so there is
                    // nothing to do and the gesture did not reach this body at
                    // all.
                    let reached =
                        planes.iter().all(|plane| plane.range_over_box(centre, half).1 > -band_mm);
                    reached.then_some(node.id)
                })
                .collect()
        };

        let mut outcome = CutOutcome {
            bricks: 0,
            bodies_cut: Vec::new(),
            bodies_crossed: crossed.len(),
            bricks_spared_by_mask: 0,
            bodies_spared_by_mask: Vec::new(),
            entry: None,
        };
        let mut changes = Vec::new();

        for body in crossed {
            let Some(volume) = self.volume_mut(body) else {
                continue;
            };
            volume.begin_stroke();
            let counts = volume.clip_convex(planes);
            let edit = volume.end_stroke();
            // Counted whatever else happened in this body: a cut that removed
            // half a body and was blocked on the other half spared bricks just
            // as much as one that was blocked outright.
            if counts.spared_by_mask > 0 {
                outcome.bricks_spared_by_mask += counts.spared_by_mask;
                outcome.bodies_spared_by_mask.push(body);
            }
            // The recorder is the authority on whether anything changed: a
            // count with no edit behind it would push an entry that restores
            // nothing.
            if let Some(edit) = edit.filter(|edit| !edit.is_empty()) {
                outcome.bricks += counts.changed;
                outcome.bodies_cut.push(body);
                changes.push(Change::Bricks { body, edit });
            }
        }

        if !changes.is_empty() {
            outcome.entry = Some(Entry::new(changes));
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOXEL: f32 = 0.5;
    const RADIUS: f32 = 20.0;

    fn ball() -> Volume {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, RADIUS);
        volume.mark_everything_dirty();
        volume
    }

    /// The headline behaviour: everything on the normal side goes, everything
    /// behind it stays.
    #[test]
    fn a_cut_removes_the_side_the_normal_points_at() {
        let mut volume = ball();
        let plane = ClipPlane::new(Vec3::ZERO, Vec3::X).expect("a unit normal");
        assert!(volume.clip(plane).changed > 0, "the cut did nothing");

        assert!(
            volume.sample_world(Vec3::new(10.0, 0.0, 0.0)) > 0.0,
            "material survived on the cut side"
        );
        assert!(
            volume.sample_world(Vec3::new(-10.0, 0.0, 0.0)) < 0.0,
            "the kept side was removed as well"
        );
    }

    /// The whole reason for `max` rather than deleting geometry: the cut face
    /// has to be closed, or nothing downstream will print it.
    #[test]
    fn a_cut_model_is_still_watertight_and_manifold() {
        let mut volume = ball();
        volume.clip(ClipPlane::new(Vec3::new(3.0, 0.0, 0.0), Vec3::X).unwrap());

        let (mesh, report) = volume.export_mesh();
        assert!(
            report.is_printable(),
            "a cut left the model unprintable: {} ({} triangles)",
            report.summary(),
            mesh.triangles.len()
        );
    }

    /// An off-axis plane is the normal case for lopping a defect off a scan,
    /// and it is where a sloppy brick classification shows up.
    #[test]
    fn an_oblique_cut_is_also_watertight() {
        let mut volume = ball();
        let plane = ClipPlane::new(Vec3::new(2.0, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5)).unwrap();
        assert!(volume.clip(plane).changed > 0);

        let (_, report) = volume.export_mesh();
        assert!(report.is_printable(), "an oblique cut is not printable: {}", report.summary());
    }

    /// The cut goes through the entire model, front and back, rather than
    /// stopping at the first surface it meets.
    #[test]
    fn a_cut_passes_through_the_whole_model() {
        let mut volume = Volume::new(VOXEL);
        // Two separate balls, one either side of the origin along X.
        volume.seed_sphere(Vec3::new(-25.0, 0.0, 0.0), 8.0);
        volume.seed_sphere(Vec3::new(25.0, 0.0, 0.0), 8.0);
        volume.mark_everything_dirty();

        // A plane at the origin facing +X should take the far ball entirely.
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap());

        assert!(volume.sample_world(Vec3::new(-25.0, 0.0, 0.0)) < 0.0, "the near ball was removed");
        assert!(
            volume.sample_world(Vec3::new(25.0, 0.0, 0.0)) > 0.0,
            "the far ball survived, so the cut did not pass through the model"
        );
    }

    /// A cut that misses must be free and must not become an undo entry.
    #[test]
    fn a_plane_that_misses_the_model_changes_nothing() {
        let mut volume = ball();
        let before: Vec<BrickCoord> = volume.brick_coords().collect();
        let plane = ClipPlane::new(Vec3::new(500.0, 0.0, 0.0), Vec3::X).unwrap();

        assert_eq!(volume.clip(plane).changed, 0, "a plane far outside the model changed bricks");
        assert_eq!(volume.brick_coords().count(), before.len());
    }

    /// The memory point. A cut through a solid model must not promote its
    /// interior tiles to dense: that is what would turn a 40 MB scan into a
    /// gigabyte, and it is invisible in any test that only looks at geometry.
    #[test]
    fn a_cut_does_not_promote_untouched_interior_bricks_to_dense() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 40.0);
        volume.mark_everything_dirty();
        let before = volume.stats();
        assert!(before.uniform_bricks > 0, "the fixture has no interior tiles to protect");

        // Cut a thin sliver off one end, so almost every interior tile is
        // untouched.
        volume.clip(ClipPlane::new(Vec3::new(36.0, 0.0, 0.0), Vec3::X).unwrap());
        let after = volume.stats();

        assert!(
            after.uniform_bricks >= before.uniform_bricks - 4,
            "interior tiles were promoted to dense: {} before, {} after",
            before.uniform_bricks,
            after.uniform_bricks
        );
        assert!(
            after.resident_bytes < before.resident_bytes * 2,
            "a sliver cut doubled the resident memory: {} -> {}",
            before.resident_bytes,
            after.resident_bytes
        );
    }

    /// One cut is one undo entry, and undoing it puts the model back exactly.
    #[test]
    fn a_cut_is_undoable_and_restores_the_field_exactly() {
        let mut volume = ball();
        let probe = Vec3::new(10.0, 0.0, 0.0);
        let before = volume.sample_world(probe);
        let bricks_before = volume.brick_coords().count();

        volume.begin_stroke();
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap());
        let edit = volume.end_stroke().expect("a cut that changed bricks should record an entry");

        assert_ne!(volume.sample_world(probe), before, "the cut did not change the probe");

        volume.apply_edit(edit);
        assert_eq!(volume.sample_world(probe), before, "undo did not restore the field");
        assert_eq!(volume.brick_coords().count(), bricks_before, "undo lost or added bricks");
    }

    /// Cutting the whole model away is legal and must not panic or leave a
    /// half-state.
    #[test]
    fn cutting_everything_away_leaves_an_empty_volume() {
        let mut volume = ball();
        volume.clip(ClipPlane::new(Vec3::new(-RADIUS - 5.0, 0.0, 0.0), Vec3::X).unwrap());
        assert!(
            volume.sample_world(Vec3::ZERO) > 0.0,
            "material survived a plane placed beyond the whole model"
        );
    }

    /// One oblique plane passing is luck; a spread of them passing is the
    /// property.
    ///
    /// This test is the reason the cut is a plain `max` and not something
    /// cleverer. A smooth max was tried first, to round the rim where the cut
    /// face meets the old surface, because that rim is where the mesher leaves
    /// four-way edges. It does not converge: one voxel of rounding left 1 bad
    /// edge, two left 0 on ONE plane but 12 on the (1,1,1) diagonal, three left
    /// 1, four left 6, six left 2. Rounding just moves which cells straddle the
    /// rim. The real answer was that those edges are harmless -- OrcaSlicer
    /// reports `manifold = yes` on a model carrying them -- so the validator
    /// changed instead of the geometry.
    #[test]
    fn oblique_cuts_at_many_angles_are_all_printable() {
        let normals = [
            Vec3::new(1.0, 2.0, -0.5),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-3.0, 1.0, 2.0),
            Vec3::new(0.3, -1.0, 0.7),
            Vec3::new(5.0, -2.0, -1.0),
            Vec3::new(-1.0, -1.0, 4.0),
            Vec3::new(2.0, 0.1, 0.0),
        ];
        for (index, normal) in normals.into_iter().enumerate() {
            // Offset the plane a little differently each time, so the cut does
            // not always land in the same place relative to the lattice.
            let point = Vec3::splat(index as f32 * 0.7 - 2.0);
            let mut volume = ball();
            let plane = ClipPlane::new(point, normal).unwrap();
            assert!(volume.clip(plane).changed > 0, "plane {index} cut nothing");

            let (_, report) = volume.export_mesh();
            assert!(
                report.is_printable(),
                "cut {index} with normal {normal:?} is not printable: {}",
                report.summary()
            );
        }
    }

    #[test]
    fn a_degenerate_drag_produces_no_plane() {
        assert!(ClipPlane::new(Vec3::ZERO, Vec3::ZERO).is_none());
        assert!(ClipPlane::new(Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)).is_none());
    }

    /// Two cuts in a row compose, which is how a real defect gets trimmed back.
    #[test]
    fn cuts_compose() {
        let mut volume = ball();
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap());
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap());

        assert!(volume.sample_world(Vec3::new(-10.0, -10.0, 0.0)) < 0.0, "the kept quadrant went");
        assert!(volume.sample_world(Vec3::new(10.0, -10.0, 0.0)) > 0.0);
        assert!(volume.sample_world(Vec3::new(-10.0, 10.0, 0.0)) > 0.0);

        let (_, report) = volume.export_mesh();
        assert!(report.is_printable(), "two cuts left it unprintable: {}", report.summary());
    }
}

/// The cut across a whole document, which is where the decision that a cut
/// crosses every VISIBLE body actually lives.
#[cfg(test)]
mod across_the_document {
    use super::*;
    use crate::body::Document;
    use crate::undo::{History, UndoOutcome};

    const VOXEL: f32 = 0.5;

    /// Two balls side by side along X, both straddling the Y = 0 plane, so one
    /// plane facing +Y takes the top off both.
    fn two_bodies() -> Document {
        let mut first = Volume::new(VOXEL);
        first.seed_sphere(Vec3::new(-15.0, 0.0, 0.0), 8.0);
        first.mark_everything_dirty();
        let mut doc = Document::from_volume(first);

        let mut second = Volume::new(VOXEL);
        second.seed_sphere(Vec3::new(15.0, 0.0, 0.0), 8.0);
        second.mark_everything_dirty();
        doc.add_body("Body 2", second);
        doc
    }

    fn shown(doc: &Document) -> Vec<bool> {
        let mut out = Vec::new();
        doc.display_visibility(None, &mut out);
        out
    }

    /// The headline: a cut acts on everything on screen, not on the active body.
    #[test]
    fn a_cut_crosses_every_visible_body() {
        let mut doc = two_bodies();
        let above = [Vec3::new(-15.0, 5.0, 0.0), Vec3::new(15.0, 5.0, 0.0)];
        let ids: Vec<_> = doc.bodies().map(|(id, _)| id).collect();
        for (id, probe) in ids.iter().zip(above) {
            assert!(doc.volume(*id).unwrap().sample_world(probe) < 0.0, "the fixture has a hole");
        }

        let visible = shown(&doc);
        let outcome = doc.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap(), &visible);

        assert_eq!(outcome.bodies_cut.len(), 2, "the cut reached {:?}", outcome.bodies_cut);
        assert_eq!(outcome.bodies_crossed, 2);
        assert!(outcome.bricks > 0);
        for (id, probe) in ids.iter().zip(above) {
            assert!(
                doc.volume(*id).unwrap().sample_world(probe) > 0.0,
                "the cut left {id:?} whole"
            );
        }
    }

    /// One gesture is ONE undo entry, however many bodies it touched. Undoing it
    /// has to put all of them back, because half a cut is not a state anything
    /// downstream is written against.
    #[test]
    fn a_cut_across_two_bodies_is_one_undo_entry_that_restores_both() {
        let mut doc = two_bodies();
        let ids: Vec<_> = doc.bodies().map(|(id, _)| id).collect();
        let above = [Vec3::new(-15.0, 5.0, 0.0), Vec3::new(15.0, 5.0, 0.0)];
        let before: Vec<f32> = ids
            .iter()
            .zip(above)
            .map(|(id, at)| doc.volume(*id).unwrap().sample_world(at))
            .collect();

        let visible = shown(&doc);
        let outcome = doc.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap(), &visible);
        let mut history = History::new(64 * 1024 * 1024);
        history.push(outcome.entry.expect("a cut that changed bricks records an entry"));
        assert_eq!(history.stats().undo_entries, 1, "one gesture became more than one entry");

        let visible = shown(&doc);
        assert!(matches!(history.undo(&mut doc, &visible), UndoOutcome::Applied(_)));
        for ((id, at), was) in ids.iter().zip(above).zip(before) {
            assert_eq!(
                doc.volume(*id).unwrap().sample_world(at),
                was,
                "one undo did not restore {id:?}"
            );
        }
        assert!(!history.can_undo(), "the cut left a second entry behind");
    }

    /// A body the user cannot see must come back bit-identical: hiding is a
    /// draw-time skip, so cutting one would cost a remesh and an upload and
    /// change not one pixel.
    #[test]
    fn a_hidden_body_the_line_passes_over_is_left_untouched() {
        let mut doc = two_bodies();
        let ids: Vec<_> = doc.bodies().map(|(id, _)| id).collect();
        let hidden = ids[1];
        let mut meta = doc.meta(hidden).expect("the second body");
        meta.visible = false;
        doc.set_meta(&meta);

        let probe = Vec3::new(15.0, 5.0, 0.0);
        let before = doc.volume(hidden).unwrap().sample_world(probe);
        let bricks_before = doc.volume(hidden).unwrap().brick_count();

        let visible = shown(&doc);
        let outcome = doc.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap(), &visible);

        assert_eq!(outcome.bodies_cut.len(), 1, "the cut reached the hidden body");
        assert_eq!(outcome.bodies_crossed, 1);
        assert_eq!(
            doc.volume(hidden).unwrap().sample_world(probe),
            before,
            "the hidden body was cut"
        );
        assert_eq!(doc.volume(hidden).unwrap().brick_count(), bricks_before);
    }

    /// A plane nowhere near the model produces no entry at all -- an undo entry
    /// for a no-op is worse than none, because it costs the user a real one.
    #[test]
    fn a_line_that_misses_everything_records_nothing() {
        let mut doc = two_bodies();
        let visible = shown(&doc);
        let outcome =
            doc.clip(ClipPlane::new(Vec3::new(0.0, 500.0, 0.0), Vec3::Y).unwrap(), &visible);

        assert_eq!(outcome.bricks, 0);
        assert_eq!(outcome.bodies_cut.len(), 0);
        assert_eq!(outcome.bodies_crossed, 0, "a plane 500 mm away counted as crossing a body");
        assert!(outcome.entry.is_none(), "a cut that changed nothing recorded an entry");
    }

    /// The two counts have to be able to differ, or the status line cannot tell
    /// "the line missed" from "it crossed something and found nothing there".
    ///
    /// The gate is a body's world BOX, and a box has corners the body does not
    /// fill: two blobs on a diagonal leave the other two corners of their shared
    /// box empty. A plane through one of those corners reaches the body and
    /// removes nothing.
    #[test]
    fn a_body_the_plane_reaches_but_finds_nothing_in_counts_as_crossed_and_not_cut() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::new(-20.0, -20.0, 0.0), 8.0);
        volume.seed_sphere(Vec3::new(20.0, 20.0, 0.0), 8.0);
        volume.mark_everything_dirty();
        let mut doc = Document::from_volume(volume);

        // Through the empty +X, -Y corner of the shared box, facing out of it.
        let plane = ClipPlane::new(Vec3::new(24.0, -24.0, 0.0), Vec3::new(1.0, -1.0, 0.0)).unwrap();

        let visible = shown(&doc);
        let outcome = doc.clip(plane, &visible);
        assert_eq!(outcome.bodies_crossed, 1, "the plane did not reach the body's box");
        assert_eq!(outcome.bodies_cut.len(), 0, "the plane found something in an empty corner");
        assert_eq!(outcome.bricks, 0);
        assert!(outcome.entry.is_none());
    }
}

#[cfg(test)]
mod provenance {
    use super::*;

    /// The brick classification must produce EXACTLY what touching every voxel
    /// would, or the cut is quietly wrong in the bricks it decided to skip.
    ///
    /// This started as a question -- was the oblique cut's non-manifold rim my
    /// classification or the lattice? -- and the answer was the lattice: both
    /// paths gave an identical 8 non manifold edges. It is kept as a permanent
    /// check because the classification is the part that can silently skip a
    /// brick that needed work, and nothing else here would notice.
    #[test]
    fn the_classification_matches_touching_every_voxel() {
        let plane = ClipPlane::new(Vec3::new(2.0, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5)).unwrap();

        let mut fast = Volume::new(0.5);
        fast.seed_sphere(Vec3::ZERO, 20.0);
        fast.mark_everything_dirty();
        fast.clip(plane);
        let (_, fast_report) = fast.export_mesh();

        let mut naive = Volume::new(0.5);
        naive.seed_sphere(Vec3::ZERO, 20.0);
        naive.mark_everything_dirty();
        let voxel_size = naive.voxel_size();
        let (lo, hi) = naive.voxel_bounds(Vec3::splat(-30.0), Vec3::splat(30.0));
        naive.edit_voxels(lo, hi, |_, position, value| {
            value.max(plane.distance(position) / voxel_size).clamp(INSIDE, OUTSIDE)
        });
        let (_, naive_report) = naive.export_mesh();

        eprintln!("clip:  {}", fast_report.summary());
        eprintln!("naive: {}", naive_report.summary());
        assert_eq!(
            fast_report.non_manifold_edges, naive_report.non_manifold_edges,
            "the brick classification changed the result, so it skipped a brick that mattered"
        );
        assert_eq!(fast_report.boundary_edges, naive_report.boundary_edges);
        assert_eq!(fast_report.triangles, naive_report.triangles);
    }

    /// An unmasked cut writes EXACTLY what the plain `max` wrote, voxel for
    /// voxel.
    ///
    /// The regression this exists for is the one masking creates. The write is
    /// now `old + (max(old, cut) - old) * free`, and at a `free` of 1 that is
    /// **not** `max(old, cut)` in binary floating point: `max - old` rounds, and
    /// adding `old` back does not always recover it. Measured before the branch
    /// that avoids it was written -- over five million random pairs with `old`
    /// in the band and the cut within a few voxels of it, 65,040 came out one
    /// bit different after the clamp. That is a change to every model anyone
    /// ever cut, in bricks nothing was masking, and the mesh-report comparison
    /// above would not see a bit of it.
    ///
    /// Compared against `edit_voxels` running the old expression rather than
    /// against a stored fixture, because the claim is about the arithmetic and
    /// a fixture would also pin the brick layout, the seeding and the mesher.
    #[test]
    fn an_unmasked_cut_writes_exactly_what_the_plain_max_wrote() {
        /// A cell's value however it is stored. An absent brick reads as
        /// [`OUTSIDE`], which is what makes the two sides comparable at all:
        /// the cut drops a brick the naive pass fills with `OUTSIDE`.
        fn voxel(volume: &Volume, cell: IVec3) -> f32 {
            let coord = BrickCoord::containing(cell);
            let local = cell - coord.origin();
            volume.brick(coord).map_or(OUTSIDE, |brick| {
                brick.get(local.x as usize, local.y as usize, local.z as usize)
            })
        }

        // Oblique, so the plane's distance lands on no lattice value twice and
        // the rounding this is about actually happens.
        let plane = ClipPlane::new(Vec3::new(2.0, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5)).unwrap();

        let mut cut = Volume::new(0.5);
        cut.seed_sphere(Vec3::ZERO, 20.0);
        cut.mark_everything_dirty();
        assert!(cut.clip(plane).changed > 0, "the fixture was not cut");

        let mut naive = Volume::new(0.5);
        naive.seed_sphere(Vec3::ZERO, 20.0);
        naive.mark_everything_dirty();
        let voxel_size = naive.voxel_size();
        let (lo, hi) = naive.voxel_bounds(Vec3::splat(-30.0), Vec3::splat(30.0));
        naive.edit_voxels(lo, hi, |_, position, value| {
            value.max(plane.distance(position) / voxel_size).clamp(INSIDE, OUTSIDE)
        });

        let mut differing = 0usize;
        let mut worst = None;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let cell = IVec3::new(x, y, z);
                    let (theirs, ours) = (voxel(&naive, cell), voxel(&cut, cell));
                    if theirs.to_bits() != ours.to_bits() {
                        differing += 1;
                        worst.get_or_insert((cell, theirs, ours));
                    }
                }
            }
        }
        assert_eq!(
            differing, 0,
            "an unmasked cut no longer writes what the plain max wrote: {differing} voxels \
             differ, first at {worst:?}"
        );
    }

    /// One plane through [`Volume::clip_convex`] is [`Volume::clip`], to the
    /// bit.
    ///
    /// The reason this is a test rather than an argument is that the argument
    /// is *almost* airtight and the gap is exactly where a mistake would live.
    /// `min` over one element is that element, so no comparison happens and no
    /// rounding can differ -- provided the single-plane path really does take
    /// the first distance outright rather than folding from `f32::INFINITY`.
    /// Folding would be equivalent for every finite value and would silently
    /// stop being equivalent the day a `NaN` reached it, and nothing else in
    /// this file would notice.
    ///
    /// Oblique on all three axes, because an axis-aligned plane lands its
    /// distances on lattice values and would pass this test even if the
    /// arithmetic had changed.
    #[test]
    fn a_one_plane_convex_cut_is_the_plane_cut_bit_for_bit() {
        use crate::testing::assert_same_field;

        let plane = ClipPlane::new(Vec3::new(2.0, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5)).unwrap();

        let mut singular = Volume::new(0.5);
        singular.seed_sphere(Vec3::ZERO, 20.0);
        singular.mark_everything_dirty();
        let one = singular.clip(plane);

        let mut plural = Volume::new(0.5);
        plural.seed_sphere(Vec3::ZERO, 20.0);
        plural.mark_everything_dirty();
        let many = plural.clip_convex(&[plane]);

        assert!(one.changed > 0, "the fixture was not cut");
        assert_eq!(one, many, "the two paths disagreed about what they did");
        assert_same_field(&singular, &plural, "one plane through clip_convex");
    }

    /// **A half-protected voxel must land half way, not all the way.**
    ///
    /// This is the regression test for a bug that shipped: the blend ran toward
    /// the RAW cut distance, which is unbounded, and only the finished value
    /// was clamped. So a voxel at `-3` under half protection with the cutter
    /// twenty voxels past it blended toward 20 rather than toward `OUTSIDE`,
    /// landed at 8.5, and clamped to 3.0 -- removed exactly as completely as an
    /// unprotected voxel.
    ///
    /// **The effect got WORSE the further the cutter was**, which is why it
    /// survived: a cut drawn right at the edge of a mask, which is how anyone
    /// would test this by hand, is the one case where the two orders nearly
    /// agree. Everything further out was silently unprotected.
    ///
    /// The far end of the sweep is the part that bites. `free` of 0.5 with the
    /// cutter one voxel away is nearly right under either order; at twenty
    /// voxels the old form is off by the whole band.
    #[test]
    fn a_partly_protected_voxel_blends_toward_outside_and_not_beyond_it() {
        for free in [0.25_f32, 0.5, 0.75] {
            for cut in [1.0_f32, 4.0, 20.0, 4000.0] {
                for old in [INSIDE, -1.0, 0.0, 1.0] {
                    let got = cut_voxel(old, cut, free);
                    // The most a cut can ever write is OUTSIDE, so a fraction
                    // `free` of the way there is the answer.
                    let ceiling = old.max(cut).clamp(INSIDE, OUTSIDE);
                    let want = old + (ceiling - old) * free;
                    assert!(
                        (got - want).abs() < 1.0e-6,
                        "old {old}, cut {cut}, free {free}: got {got}, wanted {want}"
                    );
                    assert!(
                        got < OUTSIDE || ceiling >= OUTSIDE && free >= 1.0,
                        "a partly protected voxel was driven fully outside: old {old}, cut \
                         {cut}, free {free} gave {got}"
                    );
                }
            }
        }
    }

    /// And the unmasked path is untouched by that fix, which is the property
    /// that let it be made at all: clamping a value that is already clamped is
    /// the identity, so `free >= 1` writes exactly what it always wrote.
    #[test]
    fn clamping_before_the_blend_does_not_move_an_unmasked_cut() {
        for cut in [-4000.0_f32, -3.5, -1.0, 0.0, 1.0, 3.5, 4000.0] {
            for old in [INSIDE, -1.5, 0.0, 1.5, OUTSIDE] {
                let got = cut_voxel(old, cut, 1.0);
                let was = old.max(cut).clamp(INSIDE, OUTSIDE);
                assert_eq!(got.to_bits(), was.to_bits(), "old {old}, cut {cut}");
            }
        }
    }

    /// A cutter of no planes removes nothing.
    ///
    /// The intersection of no half-spaces is all of space, so the faithful
    /// answer is to erase the model. This pins the unfaithful one, because
    /// there is no gesture whose failure should be answered by deleting
    /// everything -- and an empty slice is exactly what a gesture whose plane
    /// construction failed would hand over.
    #[test]
    fn a_cutter_with_no_planes_removes_nothing() {
        use crate::testing::assert_same_field;

        let mut cut = Volume::new(0.5);
        cut.seed_sphere(Vec3::ZERO, 20.0);
        cut.mark_everything_dirty();

        let mut untouched = Volume::new(0.5);
        untouched.seed_sphere(Vec3::ZERO, 20.0);
        untouched.mark_everything_dirty();

        assert_eq!(cut.clip_convex(&[]).changed, 0, "an empty cutter removed bricks");
        assert_same_field(&cut, &untouched, "an empty cutter");
    }
}

/// The cut as a convex region rather than a half-space.
///
/// What these pin is the one property a plane cannot have: **the cutter stops**.
/// Every test here is built around material the cutter must leave alone that a
/// plane covering the same silhouette would have taken, because that is the
/// entire difference and nothing in the single-plane suite can see it.
#[cfg(test)]
mod as_a_convex_region {
    use super::*;

    const VOXEL: f32 = 0.5;

    /// An axis-aligned box as six half-spaces, normals pointing INWARD.
    ///
    /// Inward is the direction that makes the box's interior the region that
    /// goes: a plane's normal points at the side being removed, so for the
    /// intersection to be the inside of the box each face must point in.
    /// Getting this backwards produces the complement -- an unbounded region
    /// with a box-shaped hole in it -- which removes the entire model except a
    /// box, and is a failure worth being able to recognise on sight.
    fn box_cutter(low: Vec3, high: Vec3) -> Vec<ClipPlane> {
        let mut planes = Vec::with_capacity(6);
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            planes.push(ClipPlane::new(low, axis).expect("a unit axis"));
            planes.push(ClipPlane::new(high, -axis).expect("a unit axis"));
        }
        planes
    }

    /// Two balls, far enough apart that a brick belongs to at most one.
    fn a_ball_and_a_distant_one() -> Volume {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        volume.seed_sphere(Vec3::new(60.0, 0.0, 0.0), 8.0);
        volume.mark_everything_dirty();
        volume
    }

    /// **The headline property.** A bounded cutter takes what is inside it and
    /// leaves what is behind it, where the plane that removes the same near
    /// material would go on to remove the far ball as well.
    ///
    /// The far ball is not decoration. `a_cut_passes_through_the_whole_model`
    /// asserts the opposite behaviour on the same shape of fixture, and it must
    /// keep passing: an infinite half-space is still what one plane means. The
    /// two tests together are what say the boundedness comes from the extra
    /// planes and not from a change of heart about what a cut is.
    #[test]
    fn a_bounded_cutter_stops_where_a_plane_would_carry_on() {
        let mut volume = a_ball_and_a_distant_one();
        let counts = volume
            .clip_convex(&box_cutter(Vec3::new(10.0, -40.0, -40.0), Vec3::new(40.0, 40.0, 40.0)));

        assert!(counts.changed > 0, "the cutter did nothing");
        assert!(
            volume.sample_world(Vec3::new(15.0, 0.0, 0.0)) > 0.0,
            "material survived inside the cutter"
        );
        assert!(
            volume.sample_world(Vec3::new(-15.0, 0.0, 0.0)) < 0.0,
            "the cutter reached behind itself"
        );
        assert!(
            volume.sample_world(Vec3::new(60.0, 0.0, 0.0)) < 0.0,
            "the cutter carried on past its far face and took the distant ball"
        );
    }

    /// A cutter that encloses material drops bricks whole rather than resolving
    /// them one voxel at a time. This is the shape's own payoff: a plane
    /// covering the same silhouette has to cross every brick along an infinite
    /// sheet, where a loop drawn around a lump saturates inside it.
    ///
    /// Asserted on `removed` directly rather than inferred from a drop in
    /// resident bytes, because there are TWO ways a brick leaves the map -- the
    /// whole-brick drop, and a crossed brick that collapses to `OUTSIDE` and is
    /// reinserted as nothing -- and a bytes measurement cannot tell them apart.
    #[test]
    fn a_cutter_that_encloses_material_drops_whole_bricks() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 30.0);
        volume.mark_everything_dirty();

        // Comfortably larger than a brick on every axis (a brick is 32 voxels,
        // 16 mm here), and wholly inside the ball, so bricks in the middle of
        // it are saturated past the band on all six faces.
        let counts = volume.clip_convex(&box_cutter(Vec3::splat(-25.0), Vec3::splat(25.0)));

        assert!(counts.removed > 0, "nothing was dropped whole: {counts:?}");
    }

    /// The memory point, mirrored from the plane's own version of it: a cutter
    /// must not promote interior tiles it never reaches.
    ///
    /// This is the failure mode the plan singles out as the reason v1 is convex
    /// only -- a decomposed non-convex region evaluates to exactly zero on its
    /// internal seams, deep inside the removed region, which forces `Crosses`
    /// where `Removes` was correct and leaves a 128 KB brick resident in a
    /// space the user just deleted. A convex cutter cannot do that, and this is
    /// what says so.
    #[test]
    fn a_cutter_does_not_promote_untouched_interior_bricks_to_dense() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 40.0);
        volume.mark_everything_dirty();
        let before = volume.stats();
        assert!(before.uniform_bricks > 0, "the fixture has no interior tiles to protect");

        // A small box at one end, so almost every interior tile is untouched.
        volume.clip_convex(&box_cutter(Vec3::new(30.0, -10.0, -10.0), Vec3::new(50.0, 10.0, 10.0)));
        let after = volume.stats();

        assert!(
            after.uniform_bricks >= before.uniform_bricks - 4,
            "interior tiles were promoted to dense: {} before, {} after",
            before.uniform_bricks,
            after.uniform_bricks
        );
        assert!(
            after.resident_bytes < before.resident_bytes * 2,
            "a small cut doubled the resident bytes: {} before, {} after",
            before.resident_bytes,
            after.resident_bytes
        );
    }

    /// A convex cutter leaves a printable model, which is not free: it adds
    /// cut-cut edges where two faces of the cutter meet inside the material,
    /// and those are new geometry the plane never produced.
    #[test]
    fn a_box_cut_model_is_still_printable() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        volume.mark_everything_dirty();
        volume.clip_convex(&box_cutter(Vec3::new(8.0, -30.0, -30.0), Vec3::new(30.0, 30.0, 30.0)));

        let (mesh, report) = volume.export_mesh();
        assert!(
            report.is_printable(),
            "a box cut left the model unprintable: {} ({} triangles)",
            report.summary(),
            mesh.triangles.len()
        );
    }

    /// **A convex cutter must not leave the model more non-manifold than the
    /// plane that covers the same silhouette.**
    ///
    /// `is_printable` is the obvious gate and it is structurally blind to what
    /// this adds. It is `boundary_edges == 0 && inconsistent_edges == 0 &&
    /// triangles > 0` -- it counts holes and winding, and does NOT count
    /// `non_manifold_edges` at all. A shaped cut adds sixteen sharp cut-cut
    /// edges running the full depth of the cutter, plus their corner junctions,
    /// which is exactly the geometry `is_printable` cannot see; and this file
    /// already records four-way edges at cut rims as a problem that was
    /// accepted rather than solved.
    ///
    /// So the assertion is against a BASELINE on the same fixture rather than
    /// against zero. Against zero it would fail on the plane cut too, which
    /// ships; against `is_printable` it would pass on a model with sixteen
    /// non-manifold seams.
    #[test]
    fn a_shaped_cut_is_no_less_manifold_than_the_plane_that_covers_it() {
        let mut worst_extra = 0i64;
        for step in 0..6 {
            let angle = step as f32 / 6.0 * std::f32::consts::TAU;
            let (sin, cos) = angle.sin_cos();
            // A prism whose axis is oblique to the lattice at every step, which
            // is where a sloppy classification shows up.
            let axis = Vec3::new(cos, 0.35, sin).normalize();

            let mut planed = Volume::new(VOXEL);
            planed.seed_sphere(Vec3::ZERO, 20.0);
            planed.mark_everything_dirty();
            planed.clip_convex(&[ClipPlane::new(axis * 8.0, axis).unwrap()]);
            let (_, plane_report) = planed.export_mesh();

            let mut shaped = Volume::new(VOXEL);
            shaped.seed_sphere(Vec3::ZERO, 20.0);
            shaped.mark_everything_dirty();
            shaped.clip_convex(&prism(axis * 8.0, axis, 9.0, 16));
            let (_, shaped_report) = shaped.export_mesh();

            assert!(
                shaped_report.is_printable(),
                "a shaped cut at {angle:.2} rad is not printable: {}",
                shaped_report.summary()
            );
            let extra =
                shaped_report.non_manifold_edges as i64 - plane_report.non_manifold_edges as i64;
            worst_extra = worst_extra.max(extra);
        }
        // Sixteen side planes meeting the surface make more rim than one plane
        // does, so a small positive number is expected and zero would be a
        // surprise. What this catches is the shape adding non-manifold edges
        // out of proportion to the rim it draws -- an order of magnitude, not a
        // handful.
        assert!(
            worst_extra < 64,
            "a shaped cut added {worst_extra} non-manifold edges over the plane covering the \
             same silhouette"
        );
    }

    /// A regular prism, inward normals, as the app builds one from a hull.
    fn prism(centre: Vec3, axis: Vec3, radius: f32, sides: usize) -> Vec<ClipPlane> {
        let axis = axis.normalize();
        let helper = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let u = axis.cross(helper).normalize();
        let v = axis.cross(u);
        (0..sides)
            .map(|side| {
                let angle = side as f32 / sides as f32 * std::f32::consts::TAU;
                let (sin, cos) = angle.sin_cos();
                let outward = u * cos + v * sin;
                ClipPlane::new(centre + outward * radius, -outward).expect("a unit normal")
            })
            .collect()
    }

    /// **The absolute cost assertion, and it is absolute on purpose.**
    ///
    /// The tempting form -- "a lasso dirties no more bricks than the plane that
    /// covers it" -- gates nothing. A plane pushes every removed brick into the
    /// touched set, which is half the body, against a small loop's handful of
    /// wall bricks; it passes by three orders of magnitude and would keep
    /// passing if a shaped cut promoted every brick it crossed twice over.
    ///
    /// So these are numbers, measured on this fixture, that a regression has to
    /// move. They are generous -- the point is to catch something going badly
    /// wrong, not to freeze the classification.
    #[test]
    fn a_shaped_cut_promotes_a_bounded_number_of_bricks() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        volume.mark_everything_dirty();
        volume.take_dirty(&mut Vec::new());

        let counts = volume.clip_convex(&prism(Vec3::new(14.0, 0.0, 0.0), Vec3::X, 6.0, 16));

        assert!(counts.changed > 0, "the cutter did nothing to measure");
        assert!(
            counts.crossed <= 48,
            "a small loop promoted {} bricks to dense, which is more than the region it covers",
            counts.crossed
        );
        assert!(
            volume.dirty_count() <= 200,
            "a small loop dirtied {} bricks",
            volume.dirty_count()
        );
    }

    /// **The mask must be reported as sparing exactly what the cut would have
    /// taken -- including the bricks it would have taken WHOLE.**
    ///
    /// This is the regression test for a bug the per-brick plane pruning
    /// introduced. A brick the cutter saturates classifies as `Removes`, and
    /// `classify_active` says so by leaving the active plane set EMPTY. The
    /// mask then downgrades it to `Crosses`, and the fully-protected arm asked
    /// "would the cut have changed anything here?" using that empty set -- over
    /// which the minimum is negative infinity, so the answer came back "no".
    ///
    /// Those bricks fell out of `bricks_spared_by_mask`: on this fixture 152
    /// were reported where 216 were really spared, the 64 missing ones being
    /// exactly those the cutter enclosed. A cut a mask had blocked entirely
    /// could then report **"the cut missed the model"** -- the single most
    /// misleading thing the status line can say, because it sends the user off
    /// to redraw a gesture that was never the problem. Nothing else could see
    /// it: the field really is untouched, the bricks really are all still
    /// there, and no undo entry really is recorded. Only the count was wrong.
    ///
    /// **Asserted against the same cut run unmasked**, rather than against a
    /// number written down here. "The mask spared what the cut would have
    /// taken" is the property; a hard-coded 216 would pass just as well if the
    /// fixture changed underneath it and would say nothing about why.
    #[test]
    fn a_mask_spares_exactly_what_the_cut_would_have_taken() {
        // Big enough that bricks sit wholly INSIDE the cutter with room to
        // spare, which is what makes them classify `Removes` rather than
        // merely crossing. A cutter that only ever grazes bricks exercises the
        // other arm and would pass either way.
        let cutter = box_cutter(Vec3::splat(-40.0), Vec3::splat(40.0));

        let mut unmasked = Volume::new(VOXEL);
        unmasked.seed_sphere(Vec3::ZERO, 60.0);
        unmasked.mark_everything_dirty();
        let would_take = unmasked.clip_convex(&cutter).changed;
        assert!(would_take > 0, "the fixture is not cut at all");

        let mut masked = Volume::new(VOXEL);
        masked.seed_sphere(Vec3::ZERO, 60.0);
        masked.mark_everything_dirty();
        // Mask All: an empty map read inverted protects every voxel there is.
        masked.mask_mut().set_inverted(true);
        let counts = masked.clip_convex(&cutter);

        assert_eq!(counts.changed, 0, "a fully masked body was cut");
        assert_eq!(
            counts.spared_by_mask, would_take,
            "the mask spared {} bricks but the same cut takes {would_take} unmasked, so the \
             bricks it would have dropped whole are being reported as spared by nobody",
            counts.spared_by_mask
        );
    }

    /// A cutter the model is nowhere near costs the classification and nothing
    /// else, and records no undo entry.
    #[test]
    fn a_cutter_that_misses_changes_nothing() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        volume.mark_everything_dirty();
        let before = volume.brick_coords().count();

        let counts = volume.clip_convex(&box_cutter(Vec3::splat(500.0), Vec3::splat(540.0)));

        assert_eq!(counts.changed, 0, "a cutter far outside the model changed bricks");
        assert_eq!(counts.crossed, 0, "a cutter far outside the model promoted bricks");
        assert_eq!(volume.brick_coords().count(), before);
    }

    /// **The multi-plane path writes what the definition says, to the bit.**
    ///
    /// This is the test that keeps two optimisations honest at once, and
    /// without it neither is checkable. A brick crossed by one plane is written
    /// by a fused voxel-major loop and a brick crossed by several by a
    /// plane-major one over a scratch buffer; no input takes both, so a
    /// disagreement between them is invisible. On top of that the plane set is
    /// PRUNED per brick -- planes that saturate across a brick are dropped from
    /// its minimum -- and the argument that this is exact rather than
    /// approximate is exactly the kind that is convincing and occasionally
    /// wrong.
    ///
    /// So the reference here is neither path: it is `min` over ALL the planes
    /// at every voxel in a box, through `edit_voxels`, which is the definition
    /// written out. Compared bit for bit, because the claim is bit-identity.
    ///
    /// Oblique planes at deliberately awkward angles, so no distance lands on a
    /// lattice value and the rounding this is about actually happens.
    #[test]
    fn several_planes_write_the_minimum_of_their_distances_exactly() {
        let planes = [
            ClipPlane::new(Vec3::new(2.0, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5)).unwrap(),
            ClipPlane::new(Vec3::new(-3.0, 4.0, 1.0), Vec3::new(-0.7, 1.0, 0.3)).unwrap(),
            ClipPlane::new(Vec3::new(1.0, 1.0, -2.0), Vec3::new(0.2, -0.4, 1.0)).unwrap(),
        ];

        let mut cut = Volume::new(VOXEL);
        cut.seed_sphere(Vec3::ZERO, 20.0);
        cut.mark_everything_dirty();
        assert!(cut.clip_convex(&planes).crossed > 0, "no brick took the multi-plane path");

        let mut reference = Volume::new(VOXEL);
        reference.seed_sphere(Vec3::ZERO, 20.0);
        reference.mark_everything_dirty();
        let voxel_size = reference.voxel_size();
        let (lo, hi) = reference.voxel_bounds(Vec3::splat(-30.0), Vec3::splat(30.0));
        reference.edit_voxels(lo, hi, |_, position, value| {
            // The minimum in millimetres and the division afterwards, which is
            // the order both real paths use. Dividing first would round each
            // distance separately and this test would fail for a reason that
            // has nothing to do with what it is checking.
            let smallest =
                planes.iter().map(|plane| plane.distance(position)).fold(f32::INFINITY, f32::min);
            value.max(smallest / voxel_size).clamp(INSIDE, OUTSIDE)
        });

        let mut differing = 0usize;
        let mut worst = None;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let cell = IVec3::new(x, y, z);
                    let coord = BrickCoord::containing(cell);
                    let local = cell - coord.origin();
                    let read = |volume: &Volume| {
                        volume.brick(coord).map_or(OUTSIDE, |brick| {
                            brick.get(local.x as usize, local.y as usize, local.z as usize)
                        })
                    };
                    let (theirs, ours) = (read(&reference), read(&cut));
                    if theirs.to_bits() != ours.to_bits() {
                        differing += 1;
                        worst.get_or_insert((cell, theirs, ours));
                    }
                }
            }
        }
        assert_eq!(
            differing, 0,
            "the convex cut is not the minimum of its planes: {differing} voxels differ, first \
             at {worst:?}"
        );
    }
}

/// The cut through a mask, which is where "direct manipulation acts on what is
/// drawn" meets "the user said this part is not to change".
///
/// **Every test here is written new**, because not one of the cut's existing
/// tests can fail under a mask bug: they all run on fixtures carrying no mask,
/// where an inverted sense, a slab fetched for the wrong brick and the mask
/// ignored entirely all produce identical output.
#[cfg(test)]
mod through_a_mask {
    use super::*;
    use crate::mask::PROTECTED;
    use crate::testing::assert_same_field;

    const VOXEL: f32 = 0.5;
    const RADIUS: f32 = 20.0;

    fn ball() -> Volume {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, RADIUS);
        volume.mark_everything_dirty();
        volume
    }

    /// Mask All: an empty map read inverted, which protects every voxel there
    /// is in O(1) time and no memory at all.
    ///
    /// The most-used masking state there is, and the one an "absent means free"
    /// reading of the mask cuts straight through -- so it is the state worth
    /// testing the cut against rather than a hand-painted region.
    fn mask_everything(volume: &mut Volume) {
        volume.mask_mut().set_inverted(true);
    }

    /// Protection feathered along X: free below -2 mm, fully protected above
    /// +2 mm, and a ramp in between.
    ///
    /// **Feathered and not a step**, which is the rule every path that writes a
    /// mask is held to -- see [`crate::mask`]. The box covers every voxel that
    /// can hold material, so no part of the model is protected only by accident
    /// of where the writing stopped.
    fn protect_the_positive_x_half(volume: &mut Volume) {
        let voxel_size = volume.voxel_size();
        let reach = RADIUS + 4.0;
        let (lo, hi) = volume.voxel_bounds(Vec3::splat(-reach), Vec3::splat(reach));
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let ramp = ((x as f32 * voxel_size + 2.0) / 4.0).clamp(0.0, 1.0);
                    let protection = (ramp * PROTECTED as f32).round() as u8;
                    volume.mask_mut().write(IVec3::new(x, y, z), protection);
                }
            }
        }
        volume.mask_mut().collapse();
    }

    /// The headline. A cut straight through a fully masked body leaves the
    /// field bit-identical AND leaves every brick where it was.
    ///
    /// The second half is the one the classification can get wrong on its own:
    /// `Cut::Removes` drops the brick out of the map, and a dropped brick would
    /// make "the field is unchanged" vacuously true over the bricks that were
    /// left. The `Removes` to `Crosses` downgrade is what stops it.
    #[test]
    fn a_cut_through_a_fully_masked_body_leaves_the_field_and_every_brick_alone() {
        let plane = ClipPlane::new(Vec3::ZERO, Vec3::X).expect("a unit normal");

        // The control. Without it a mask that did nothing and a cut that did
        // nothing would look the same from here.
        let mut unmasked = ball();
        let plain = unmasked.clip(plane);
        assert!(plain.changed > 0, "the fixture is not cut by this plane at all");
        assert!(
            unmasked.brick_count() < ball().brick_count(),
            "the fixture has no brick this plane removes WHOLE, so the downgrade is untested"
        );
        assert_eq!(plain.spared_by_mask, 0, "an unmasked cut reported bricks spared by a mask");

        let mut volume = ball();
        mask_everything(&mut volume);
        let counts = volume.clip(plane);

        assert_eq!(counts.changed, 0, "a fully masked body was cut anyway");
        assert!(counts.spared_by_mask > 0, "the mask blocked the cut and reported nothing spared");
        assert_eq!(
            volume.brick_count(),
            ball().brick_count(),
            "a masked brick was removed rather than spared"
        );
        assert_same_field(&volume, &ball(), "a cut straight through a fully masked body");
    }

    /// A blocked cut records no undo entry, because it promoted nothing and
    /// wrote nothing.
    ///
    /// Separate from the field check on purpose: writing every voxel back as it
    /// was would pass that one while costing a dense promotion and 128 KB of
    /// undo per brick along the plane.
    #[test]
    fn a_fully_masked_cut_records_no_undo_entry_and_promotes_no_brick() {
        let mut volume = ball();
        let dense_before = volume.stats().dense_bricks;
        mask_everything(&mut volume);

        volume.begin_stroke();
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap());
        let edit = volume.end_stroke();

        assert!(
            edit.is_none_or(|edit| edit.is_empty()),
            "a cut that changed nothing recorded an undo entry"
        );
        assert_eq!(
            volume.stats().dense_bricks,
            dense_before,
            "a blocked cut promoted bricks to dense"
        );
    }

    /// The mask spares nothing from a cut that had nothing left to remove.
    ///
    /// This is what [`cut_would_change_a_voxel`] buys, and it is worth a test
    /// because the cheap version -- count every fully protected brick the plane
    /// crosses -- passes every other test here and then tells the user their
    /// mask blocked a cut that would have done nothing anyway. A plane applied
    /// twice is the exact case: `max` is idempotent, so the second cut has
    /// nothing to do in any brick, and the bricks along the cut face are still
    /// there for it to cross.
    #[test]
    fn a_mask_spares_nothing_from_a_cut_that_had_nothing_left_to_remove() {
        let plane = ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap();
        let mut volume = ball();
        assert!(volume.clip(plane).changed > 0, "the first cut did nothing");

        mask_everything(&mut volume);
        let again = volume.clip(plane);

        assert_eq!(again.changed, 0, "cutting twice with the same plane is not idempotent");
        assert_eq!(
            again.spared_by_mask, 0,
            "the mask claimed to have blocked a cut that had nothing to remove"
        );
    }

    /// Half masked: the cut removes the unmasked half and leaves the other one,
    /// and what it leaves behind still prints.
    ///
    /// The cut face is no longer a plane -- it is the plane where the mask is
    /// free, the old surface where the mask is solid, and a curve between the
    /// two through the feather -- which is exactly the rim that makes this worth
    /// checking. `is_printable` and not manifoldness: an oblique cut rim
    /// legitimately leaves a handful of four-way edges, a feathered one is more
    /// dihedral still, and OrcaSlicer reports both as manifold.
    #[test]
    fn a_cut_through_a_half_masked_body_removes_only_the_unmasked_half() {
        let mut volume = ball();
        protect_the_positive_x_half(&mut volume);
        // Facing +Y, so the top goes -- except where the mask says otherwise.
        let counts = volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap());
        assert!(counts.changed > 0, "the cut did nothing at all");

        assert!(
            volume.sample_world(Vec3::new(-10.0, 10.0, 0.0)) > 0.0,
            "the unmasked half survived the cut"
        );
        assert!(
            volume.sample_world(Vec3::new(10.0, 10.0, 0.0)) < 0.0,
            "the mask did not protect its half"
        );
        assert!(
            volume.sample_world(Vec3::new(0.0, -10.0, 0.0)) < 0.0,
            "the kept side of the plane went as well"
        );

        let (mesh, report) = volume.export_mesh();
        assert!(
            report.is_printable(),
            "a feathered cut rim is not printable: {} ({} triangles)",
            report.summary(),
            mesh.triangles.len()
        );

        // All three formats, because the rim is the input each writer is least
        // likely to have been tried on.
        let mut stl = Vec::new();
        crate::export::stl::write(&mesh, &mut stl).expect("the STL writer refused a masked cut");
        let mut obj = Vec::new();
        crate::export::obj::write(&mesh, &mut obj).expect("the OBJ writer refused a masked cut");
        let mut threemf = Vec::new();
        crate::export::threemf::write(&mesh, &mut threemf)
            .expect("the 3MF writer refused a masked cut");
        for (format, bytes) in [("STL", stl), ("OBJ", obj), ("3MF", threemf)] {
            assert!(!bytes.is_empty(), "the {format} writer produced nothing");
        }
    }

    /// The mask multiplies the cut rather than replacing it, so a partly
    /// protected voxel ends up between where it was and where the cut would
    /// have put it -- never past either.
    ///
    /// The guard against the arithmetic drifting into something that
    /// extrapolates, which is the same rule the brush weight is held to and for
    /// the same reason.
    #[test]
    fn a_masked_voxel_lands_between_its_old_value_and_the_unmasked_cut() {
        for old in [INSIDE, -1.5, 0.0, 1.5, OUTSIDE] {
            for cut in [-9000.0, -2.0, 0.0, 2.0, 9000.0] {
                let unmasked = cut_voxel(old, cut, 1.0);
                for step in 0..=255u32 {
                    let free = step as f32 / 255.0;
                    let got = cut_voxel(old, cut, free);
                    let (low, high) = (old.min(unmasked), old.max(unmasked));
                    assert!(
                        (low..=high).contains(&got),
                        "old {old}, cut {cut}, free {free} left the band: {got} is outside \
                         {low}..={high}"
                    );
                    assert!(
                        (INSIDE..=OUTSIDE).contains(&got),
                        "old {old}, cut {cut}, free {free} wrote {got} outside the narrow band"
                    );
                }
            }
        }
    }

    /// Across the document: a fully masked body is CROSSED and not cut, and its
    /// spared bricks are reported against it by name.
    #[test]
    fn the_document_reports_which_body_a_mask_blocked_the_cut_on() {
        let mut first = Volume::new(VOXEL);
        first.seed_sphere(Vec3::new(-15.0, 0.0, 0.0), 8.0);
        first.mark_everything_dirty();
        let mut doc = Document::from_volume(first);

        let mut second = Volume::new(VOXEL);
        second.seed_sphere(Vec3::new(15.0, 0.0, 0.0), 8.0);
        second.mark_everything_dirty();
        doc.add_body("Body 2", second);

        let ids: Vec<NodeId> = doc.bodies().map(|(id, _)| id).collect();
        mask_everything(doc.volume_mut(ids[1]).expect("the second body"));

        let mut visible = Vec::new();
        doc.display_visibility(None, &mut visible);
        let outcome = doc.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap(), &visible);

        assert_eq!(outcome.bodies_crossed, 2);
        assert_eq!(outcome.bodies_cut.len(), 1, "the masked body was cut");
        assert!(outcome.bricks > 0, "the unmasked body was spared as well");
        assert!(outcome.bricks_spared_by_mask > 0, "the masked body reported nothing spared");
        assert_eq!(
            outcome.bodies_spared_by_mask,
            vec![ids[1]],
            "the wrong body was named, or more than one was"
        );
    }
}
