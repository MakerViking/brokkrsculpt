// SPDX-License-Identifier: AGPL-3.0-or-later

//! Turning a triangle mesh into a narrow band signed distance field.
//!
//! This is the back half of importing a mesh. [`crate::import`] parses bytes
//! into triangles; this module decides what those triangles mean as a solid.
//!
//! It lives in `brokkr-core` because it has to: [`Volume::insert_brick`] and
//! [`Volume::brick`] are `pub(crate)`, and widening them would leak the storage
//! representation into the shell, which is what keeps the Iced choice
//! reversible. `edit_voxels` is the wrong tool for the job as well -- it
//! promotes every brick it touches to dense, and most of a model's bricks are
//! solid interior that must stay uniform.
//!
//! # The invariant the sparse structure rests on
//!
//! A brick is *binned* if and only if some triangle intersects its world box
//! expanded by `NARROW_BAND * voxel_size`. If a brick is not binned then no
//! triangle is within the band of any of its voxels; therefore no surface
//! crosses it, every voxel's true distance saturates the clamp, and the brick is
//! homogeneous -- either absent (which reads as `OUTSIDE`) or `Uniform(INSIDE)`.
//! And for any voxel inside a binned brick that lies within the band of some
//! triangle T, T is in that brick's bin. So no brick ever needs a neighbour's
//! triangle list, and the band is never truncated at a brick face.
//!
//! Violating that is what produces a flat facet aligned to the 32 voxel grid,
//! which renders perfectly and is invisible to every numeric test.
//!
//! # Why the sign does not come from the nearest triangle's normal
//!
//! On a mesh with a hole, a self intersection, or inconsistent winding, a
//! nearest-normal test flips in pockets, and the resulting negative islands mesh
//! into internal bubbles that [`crate::MeshReport::is_printable`] happily calls
//! watertight. The user finds them in the slicer.
//!
//! The sign here comes from a winding number swept along X, under the **nonzero**
//! rule rather than even-odd. Even-odd calls a duplicated coincident shell --
//! routine output of CSG and of "export both halves" workflows -- completely
//! empty, and would swap a blank volume over the user's sculpt from a file that
//! renders fine in every viewer. Nonzero reads that as `w = 2` and gets it
//! right, and handles a globally inverted mesh for free as `w = -1`.
//!
//! Neither rule survives a *flipped interior island*. That is a generalised
//! winding number's territory, and the honest trade is recorded here: parity and
//! flood fill fail globally and silently -- one hole inverts a whole scan line,
//! or leaks and empties the model -- whereas a generalised winding number fails
//! locally and proportionally. The sweep ships because it is an order of
//! magnitude cheaper and exact on closed meshes; GWN is the escalation if real
//! files demand it, not a strawman.

use glam::{IVec3, Vec3};
use rayon::prelude::*;

use crate::brick::{
    BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, brick_index,
};
use crate::export::ExportMesh;
use crate::import::ImportError;
use crate::volume::Volume;

/// Longest dimension, in millimetres, below which a model is assumed to be in
/// the wrong units rather than genuinely tiny.
const IMPLAUSIBLY_SMALL_MM: f32 = 0.5;
/// ...and above which the same is assumed at the other end.
const IMPLAUSIBLY_LARGE_MM: f32 = 500.0;
/// What an implausible model is rescaled to, longest dimension.
const REFIT_LONGEST_MM: f32 = 60.0;

/// Vertices the GPU mesh pool can hold.
///
/// Past this the pool logs an error and returns, leaving the model silently
/// incomplete on screen, so an import that would exceed it is refused up front
/// with a voxel size that would fit instead.
const VERTEX_CAPACITY: f64 = 8_000_000.0;

/// Rough dense bricks per unit of surface area, used only by the preflight.
///
/// Deliberately conservative, and a measured number rather than a derived one:
/// the shell of a closed surface is more than one brick thick wherever the
/// surface is not axis aligned, and the band adds more.
const SHELL_FACTOR: f64 = 2.5;
/// Rough vertices per unit of surface area in voxel units, same purpose.
const VERTEX_FACTOR: f64 = 2.0;

/// How much a single import may allocate for voxel data.
const MAX_IMPORT_BYTES: f64 = 2.0 * 1024.0 * 1024.0 * 1024.0;

/// How an imported mesh should be placed on the lattice.
#[derive(Debug, Clone, Copy)]
pub struct VoxeliseOptions {
    /// World size of one voxel, in millimetres.
    pub voxel_size: f32,
    /// Move the model's bounding box centre onto the world origin.
    ///
    /// Effectively always wanted: the camera frames the origin, the mirror
    /// planes pass through it, and `bounding_radius` is measured from it.
    pub centre: bool,
    /// Rescale a model whose longest dimension is outside
    /// [`IMPLAUSIBLY_SMALL_MM`]..[`IMPLAUSIBLY_LARGE_MM`], which almost always
    /// means the file was written in metres or in inches.
    ///
    /// Only then. A sculpting tool whose output is a printed part must not
    /// quietly resize a model that states a real size -- the millimetre figure
    /// is the one number that matters in the whole file.
    pub refit_if_implausible: bool,
}

impl VoxeliseOptions {
    pub fn at(voxel_size: f32) -> Self {
        Self { voxel_size, centre: true, refit_if_implausible: true }
    }
}

/// What voxelising a mesh turned out to involve.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoxeliseReport {
    pub triangles: usize,
    pub dense_bricks: usize,
    pub uniform_bricks: usize,
    /// The factor the model was rescaled by, or 1.0 if it was left alone.
    pub rescaled_by: f32,
    /// Longest dimension after placement, in millimetres.
    pub longest_mm: f32,
    /// Input triangles whose surface fell entirely between voxels and was lost.
    pub lost_triangles: usize,
    /// Voxels whose sign disagrees with all six of their in-brick neighbours.
    /// The cheapest available signal for hole-leak or duplicate-face corruption.
    pub isolated_sign_flips: usize,
}

impl VoxeliseReport {
    pub fn summary(&self) -> String {
        let mut out = format!(
            "{} triangles, {:.1} mm across, {} dense + {} uniform bricks",
            self.triangles, self.longest_mm, self.dense_bricks, self.uniform_bricks
        );
        if (self.rescaled_by - 1.0).abs() > 1.0e-6 {
            out.push_str(&format!(", rescaled {:.4}x", self.rescaled_by));
        }
        if self.lost_triangles > 0 {
            let percent = 100.0 * self.lost_triangles as f64 / self.triangles.max(1) as f64;
            out.push_str(&format!(
                ", {percent:.1}% of the surface was thinner than a voxel and was lost"
            ));
        }
        if self.isolated_sign_flips > 0 {
            out.push_str(&format!(", {} isolated sign flips", self.isolated_sign_flips));
        }
        out
    }
}

/// Build a volume from a mesh.
///
/// Returns a volume with every brick already marked dirty, exactly as
/// `project::read` does, so a caller can never forget to and get an invisible
/// model.
pub fn voxelise(
    mesh: &ExportMesh,
    options: &VoxeliseOptions,
) -> Result<(Volume, VoxeliseReport), ImportError> {
    if mesh.triangles.is_empty() {
        return Err(ImportError::Empty);
    }
    let voxel_size = options.voxel_size;
    if !(voxel_size.is_finite() && voxel_size > 0.0) {
        return Err(ImportError::Malformed("the voxel size is not a positive number".into()));
    }

    let mut report = VoxeliseReport { triangles: mesh.triangles.len(), ..Default::default() };
    let corners = place(mesh, options, &mut report)?;

    let band_mm = NARROW_BAND * voxel_size;
    let (aabb_min, aabb_max) = bounds(&corners);

    preflight(&corners, aabb_min, aabb_max, voxel_size)?;

    // The grid is snapped to whole bricks, not to the expanded voxel box. This
    // is the one alignment that matters: brick origins are multiples of 32
    // because `BrickCoord::containing` floors, while the voxel box is expanded
    // by only about six voxels -- so a brick origin can sit up to 31 voxels
    // BELOW the voxel minimum, and a sign probe there would index before the
    // start of the grid. Snapping to the brick box makes every probe in range by
    // construction, on the low corner bricks of essentially every model.
    let padded_min = aabb_min - Vec3::splat(2.0 * band_mm);
    let padded_max = aabb_max + Vec3::splat(2.0 * band_mm);
    let v_min = (padded_min / voxel_size).floor().as_ivec3();
    let v_max = (padded_max / voxel_size).ceil().as_ivec3();
    let b_min = BrickCoord::containing(v_min);
    let b_max = BrickCoord::containing(v_max);
    let lattice_min = b_min.origin();
    let lattice_max = b_max.max_voxel();

    let sign = SignField::build(&corners, lattice_min, lattice_max, voxel_size);
    let bins = bin_triangles(&corners, lattice_min, lattice_max, voxel_size);

    let flips = std::sync::atomic::AtomicUsize::new(0);
    let mut built: Vec<(BrickCoord, Brick)> = bins
        .runs()
        .par_iter()
        .filter_map(|(coord, triangles)| {
            fill_brick(*coord, triangles, &corners, &sign, voxel_size, &flips)
        })
        .collect();
    report.isolated_sign_flips = flips.into_inner();

    // Everything the triangles did not reach is homogeneous, and which of the
    // two it is has to be decided or the model comes out hollow.
    built.extend(homogeneous_bricks(&bins, &sign, b_min, b_max));

    let mut volume = Volume::new(voxel_size);
    let mut any_inside = false;
    for (coord, brick) in built {
        match &brick {
            Brick::Uniform(value) => {
                report.uniform_bricks += 1;
                any_inside |= *value < 0.0;
            }
            Brick::Dense(data) => {
                report.dense_bricks += 1;
                any_inside |= data.iter().any(|value| *value < 0.0);
            }
        }
        volume.insert_brick(coord, brick);
    }
    if !any_inside {
        // Not one negative voxel anywhere. That is an open sheet -- a scan
        // patch, a plane, a surface with no thickness -- rather than a solid.
        // Note it still produces plenty of dense bricks, because a distance
        // field forms around the sheet; only the sign says it encloses nothing.
        // Refuse by name rather than swapping an empty model over the user's
        // sculpt with a success message, and note that `is_printable` on the
        // result would report "0 triangles, NOT watertight: 0 holes", which is
        // actively misleading as an import failure.
        return Err(ImportError::Malformed(
            "there is no enclosed volume in it, only an open surface".into(),
        ));
    }

    // One call, over everything. It does the plus or minus one expansion across
    // all 26 neighbours, and an absent brick still owns the quads on its low
    // faces, so marking only what was written loses a ring of quads at the outer
    // boundary.
    volume.mark_everything_dirty();

    report.lost_triangles = count_lost_surface(&volume, &corners, voxel_size);
    Ok((volume, report))
}

// --- placement ---------------------------------------------------------------

/// Every triangle as three world space corners, after placement.
type Corners = Vec<[Vec3; 3]>;

fn bounds(corners: &Corners) -> (Vec3, Vec3) {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for triangle in corners {
        for corner in triangle {
            minimum = minimum.min(*corner);
            maximum = maximum.max(*corner);
        }
    }
    (minimum, maximum)
}

fn place(
    mesh: &ExportMesh,
    options: &VoxeliseOptions,
    report: &mut VoxeliseReport,
) -> Result<Corners, ImportError> {
    let mut corners: Corners = mesh
        .triangles
        .iter()
        .map(|[a, b, c]| {
            [mesh.positions[*a as usize], mesh.positions[*b as usize], mesh.positions[*c as usize]]
        })
        .collect();

    let (minimum, maximum) = bounds(&corners);
    let extent = maximum - minimum;
    let longest = extent.max_element();
    if !longest.is_finite() || longest <= 0.0 {
        return Err(ImportError::Malformed("the mesh has no size in any direction".into()));
    }

    let scale = if options.refit_if_implausible
        && !(IMPLAUSIBLY_SMALL_MM..=IMPLAUSIBLY_LARGE_MM).contains(&longest)
    {
        REFIT_LONGEST_MM / longest
    } else {
        1.0
    };
    let centre = if options.centre { (minimum + maximum) * 0.5 } else { Vec3::ZERO };

    if scale != 1.0 || centre != Vec3::ZERO {
        for triangle in &mut corners {
            for corner in triangle.iter_mut() {
                *corner = (*corner - centre) * scale;
            }
        }
    }

    report.rescaled_by = scale;
    report.longest_mm = longest * scale;
    Ok(corners)
}

/// Refuse an import that would not fit, before anything is allocated.
///
/// The only backstop that exists otherwise is an `assert!` deep in
/// `FieldRegion::resize`, which is a panic rather than a message.
fn preflight(
    corners: &Corners,
    minimum: Vec3,
    maximum: Vec3,
    voxel_size: f32,
) -> Result<(), ImportError> {
    let area: f64 =
        corners.par_iter().map(|[a, b, c]| 0.5 * (*b - *a).cross(*c - *a).length() as f64).sum();

    let brick_mm = BRICK_DIM as f64 * voxel_size as f64;
    let shell_bricks = area / (brick_mm * brick_mm) * SHELL_FACTOR;
    let bytes = shell_bricks * BRICK_VOXELS as f64 * 4.0;
    let vertices = area / (voxel_size as f64 * voxel_size as f64) * VERTEX_FACTOR;

    // What voxel size would fit? Both limits scale as the square of it, so the
    // factor to grow by is the square root of the overshoot.
    let over = (bytes / MAX_IMPORT_BYTES).max(vertices / VERTEX_CAPACITY);
    if over > 1.0 {
        let workable = voxel_size as f64 * over.sqrt();
        let extent = maximum - minimum;
        return Err(ImportError::Malformed(format!(
            "it is {:.0} x {:.0} x {:.0} mm, which at {:.3} mm would need about {:.1} GB -- \
             resample to {:.3} mm or coarser first",
            extent.x,
            extent.y,
            extent.z,
            voxel_size,
            bytes / (1024.0 * 1024.0 * 1024.0),
            workable,
        )));
    }
    Ok(())
}

// --- the sign field ----------------------------------------------------------

/// One crossing of a grid line by a triangle.
#[derive(Debug, Clone, Copy)]
struct Crossing {
    line: u32,
    x: f32,
    delta: i8,
}

/// Winding numbers along every lattice line parallel to X.
///
/// Stored as a CSR: one sorted run of crossings per line. A per-line `Vec` would
/// be millions of tiny allocations, and this project has already measured a
/// twenty percent cost from that shape of structure once.
struct SignField {
    /// Voxel coordinate of the grid's minimum corner.
    origin: IVec3,
    /// Grid line counts along Y and Z.
    span_y: usize,
    span_z: usize,
    voxel_size: f32,
    offsets: Vec<u32>,
    crossings: Vec<(f32, i8)>,
}

impl SignField {
    #[inline]
    fn line_of(&self, y: i32, z: i32) -> Option<usize> {
        // Checked before the subtraction, not after: `(y - origin.y) as usize`
        // on a negative difference wraps to something enormous rather than
        // failing, which is the shape of bug that reads as a wild index far
        // from here.
        if y < self.origin.y || z < self.origin.z {
            return None;
        }
        let ly = (y - self.origin.y) as usize;
        let lz = (z - self.origin.z) as usize;
        if ly >= self.span_y || lz >= self.span_z {
            return None;
        }
        Some(lz * self.span_y + ly)
    }

    fn build(corners: &Corners, lattice_min: IVec3, lattice_max: IVec3, voxel_size: f32) -> Self {
        let span_y = (lattice_max.y - lattice_min.y + 1) as usize;
        let span_z = (lattice_max.z - lattice_min.z + 1) as usize;

        // Scatter, not gather. A "for each line, for each triangle covering it"
        // loop is O(lines x triangles) and reaches ten billion point-in-triangle
        // tests at this project's own stated scale case.
        let mut found: Vec<Crossing> = corners
            .par_iter()
            .fold(Vec::new, |mut out: Vec<Crossing>, triangle| {
                emit_crossings(triangle, lattice_min, span_y, span_z, voxel_size, &mut out);
                out
            })
            .reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                a
            });

        found.par_sort_unstable_by(|a, b| {
            a.line.cmp(&b.line).then(a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
        });

        let lines = span_y * span_z;
        let mut offsets = vec![0u32; lines + 1];
        for crossing in &found {
            offsets[crossing.line as usize + 1] += 1;
        }
        for index in 0..lines {
            offsets[index + 1] += offsets[index];
        }
        let crossings = found.iter().map(|c| (c.x, c.delta)).collect();

        Self { origin: lattice_min, span_y, span_z, voxel_size, offsets, crossings }
    }

    /// Winding number at a world x along one line, under the nonzero rule.
    #[inline]
    fn inside_at(&self, line: usize, world_x: f32) -> bool {
        let from = self.offsets[line] as usize;
        let to = self.offsets[line + 1] as usize;
        let mut winding = 0i32;
        for (x, delta) in &self.crossings[from..to] {
            if *x > world_x {
                break;
            }
            winding += *delta as i32;
        }
        winding != 0
    }

    /// Whether a lattice voxel is inside the solid.
    #[inline]
    fn inside(&self, voxel: IVec3) -> bool {
        match self.line_of(voxel.y, voxel.z) {
            Some(line) => self.inside_at(line, voxel.x as f32 * self.voxel_size),
            // Outside the grid entirely, which is outside the padded bounds of
            // the mesh, so it cannot be inside the solid.
            None => false,
        }
    }
}

/// Emit this triangle's crossings of every lattice line it covers.
///
/// The ray runs along +X at a fixed (y, z), so the triangle is projected onto
/// the (y, z) plane and tested there.
fn emit_crossings(
    triangle: &[Vec3; 3],
    lattice_min: IVec3,
    span_y: usize,
    span_z: usize,
    voxel_size: f32,
    out: &mut Vec<Crossing>,
) {
    let [a, b, c] = *triangle;
    let normal = (b - a).cross(c - a);
    // A triangle seen edge-on from +X crosses no ray along X. Its projected area
    // is zero, so it contributes nothing and must be skipped -- dividing by that
    // area is how it would otherwise become a NaN crossing.
    if normal.x == 0.0 || !normal.is_finite() {
        return;
    }

    // Project to (u, v) = (y, z).
    let p = [glam::Vec2::new(a.y, a.z), glam::Vec2::new(b.y, b.z), glam::Vec2::new(c.y, c.z)];
    let area2 = (p[1] - p[0]).perp_dot(p[2] - p[0]);
    if area2 == 0.0 {
        return;
    }
    // Normalise the projected triangle to positive area so one tie-break rule
    // covers both windings, carrying the 3D corners along in the same order so
    // the barycentric weights can be used directly. The winding DELTA still
    // comes from the original orientation, below.
    let (q, world) =
        if area2 > 0.0 { ([p[0], p[1], p[2]], [a, b, c]) } else { ([p[0], p[2], p[1]], [a, c, b]) };
    let area2 = area2.abs();

    // A face whose outward normal points against the ray is an entry into the
    // solid, so it increments the winding number.
    let delta: i8 = if normal.x < 0.0 { 1 } else { -1 };

    let min_uv = q[0].min(q[1]).min(q[2]);
    let max_uv = q[0].max(q[1]).max(q[2]);
    let y_from = ((min_uv.x / voxel_size).ceil() as i32).max(lattice_min.y);
    let y_to = ((max_uv.x / voxel_size).floor() as i32).min(lattice_min.y + span_y as i32 - 1);
    let z_from = ((min_uv.y / voxel_size).ceil() as i32).max(lattice_min.z);
    let z_to = ((max_uv.y / voxel_size).floor() as i32).min(lattice_min.z + span_z as i32 - 1);

    for z in z_from..=z_to {
        for y in y_from..=y_to {
            let sample = glam::Vec2::new(y as f32 * voxel_size, z as f32 * voxel_size);
            let Some(bary) = inside_projected(&q, sample, area2) else {
                continue;
            };
            // Barycentric interpolation of x across the triangle, using the
            // same corner order the weights were computed against.
            let x = bary[0] * world[0].x + bary[1] * world[1].x + bary[2] * world[2].x;
            if !x.is_finite() {
                continue;
            }
            let line = ((z - lattice_min.z) as usize) * span_y + (y - lattice_min.y) as usize;
            out.push(Crossing { line: line as u32, x, delta });
        }
    }
}

/// Barycentric weights if `sample` is inside the projected triangle `q`, which
/// is known to wind positively.
///
/// # The tie-break, and why it is not optional
///
/// A sample landing exactly on a shared edge belongs to exactly one of the two
/// triangles that share it, or the winding number double counts and the sign
/// inverts for the whole rest of that line. This is not a rare floating point
/// coincidence: it is *systematic* for any CAD box whose faces sit on lattice
/// planes, which is most of them.
///
/// The top-left rule settles it consistently in both directions. Two triangles
/// sharing an interior edge traverse it in opposite directions, so after
/// normalising both to positive area exactly one of them claims a point on it.
/// Two sharing a silhouette edge traverse it in the same direction, so both
/// claim it or neither does -- which is the correct answer for a tangential
/// touch, since the two contributions cancel.
#[inline]
fn inside_projected(q: &[glam::Vec2; 3], sample: glam::Vec2, area2: f32) -> Option<[f32; 3]> {
    let mut weights = [0.0f32; 3];
    for corner in 0..3 {
        let from = q[corner];
        let to = q[(corner + 1) % 3];
        let edge = to - from;
        let value = edge.perp_dot(sample - from);
        if value < 0.0 {
            return None;
        }
        if value == 0.0 && !is_top_left(edge) {
            return None;
        }
        // The edge opposite corner `corner + 2`.
        weights[(corner + 2) % 3] = value / area2;
    }
    Some(weights)
}

/// Whether an edge is a "top" or "left" edge, for the fill rule above.
#[inline]
fn is_top_left(edge: glam::Vec2) -> bool {
    edge.y > 0.0 || (edge.y == 0.0 && edge.x < 0.0)
}

// --- binning -----------------------------------------------------------------

/// Triangle indices grouped by the brick they can reach.
///
/// Two parallel arrays rather than a `Vec` of pairs, so a brick's triangle list
/// is a real `&[u32]` slice; and not an `FxHashMap<BrickCoord, Vec<u32>>`,
/// which would be millions of tiny allocations -- this project has already
/// measured a twenty percent cost from hashbrown tombstones under exactly that
/// shape.
struct Bins {
    /// Sorted brick keys, one per (brick, triangle) pair.
    keys: Vec<u64>,
    /// Triangle index for the pair at the same position.
    indices: Vec<u32>,
    /// Where each run of equal keys begins, plus a final sentinel.
    starts: Vec<usize>,
}

/// Pack a brick coordinate into a sortable key.
///
/// Ordered z, then y, then x, matching `BrickCoord`'s own `Ord`, which gives the
/// fill sequential access through memory rather than a random walk.
#[inline]
fn pack(coord: IVec3) -> u64 {
    const BIAS: i64 = 1 << 20;
    let x = (coord.x as i64 + BIAS) as u64;
    let y = (coord.y as i64 + BIAS) as u64;
    let z = (coord.z as i64 + BIAS) as u64;
    (z << 42) | (y << 21) | x
}

#[inline]
fn unpack(key: u64) -> BrickCoord {
    const BIAS: i64 = 1 << 20;
    const MASK: u64 = (1 << 21) - 1;
    BrickCoord(IVec3::new(
        ((key & MASK) as i64 - BIAS) as i32,
        (((key >> 21) & MASK) as i64 - BIAS) as i32,
        (((key >> 42) & MASK) as i64 - BIAS) as i32,
    ))
}

impl Bins {
    fn from_pairs(mut pairs: Vec<(u64, u32)>) -> Self {
        pairs.par_sort_unstable();
        let mut keys = Vec::with_capacity(pairs.len());
        let mut indices = Vec::with_capacity(pairs.len());
        let mut starts = Vec::new();
        for (at, (key, index)) in pairs.into_iter().enumerate() {
            if keys.last() != Some(&key) {
                starts.push(at);
            }
            keys.push(key);
            indices.push(index);
        }
        starts.push(keys.len());
        Self { keys, indices, starts }
    }

    /// One entry per binned brick: its coordinate and its triangle list.
    fn runs(&self) -> Vec<(BrickCoord, &[u32])> {
        self.starts
            .windows(2)
            .map(|pair| (unpack(self.keys[pair[0]]), &self.indices[pair[0]..pair[1]]))
            .collect()
    }

    fn contains(&self, coord: BrickCoord) -> bool {
        self.keys.binary_search(&pack(coord.0)).is_ok()
    }
}

fn bin_triangles(
    corners: &Corners,
    lattice_min: IVec3,
    lattice_max: IVec3,
    voxel_size: f32,
) -> Bins {
    let band_mm = NARROW_BAND * voxel_size;
    let b_min = BrickCoord::containing(lattice_min).0;
    let b_max = BrickCoord::containing(lattice_max).0;
    let brick_mm = BRICK_DIM as f32 * voxel_size;

    let pairs: Vec<(u64, u32)> = corners
        .par_iter()
        .enumerate()
        .fold(Vec::new, |mut out: Vec<(u64, u32)>, (index, triangle)| {
            let low = triangle[0].min(triangle[1]).min(triangle[2]) - Vec3::splat(band_mm);
            let high = triangle[0].max(triangle[1]).max(triangle[2]) + Vec3::splat(band_mm);
            let from = (low / brick_mm).floor().as_ivec3().max(b_min);
            let to = (high / brick_mm).floor().as_ivec3().min(b_max);

            for bz in from.z..=to.z {
                for by in from.y..=to.y {
                    for bx in from.x..=to.x {
                        let coord = IVec3::new(bx, by, bz);
                        let centre = (coord.as_vec3() + Vec3::splat(0.5)) * brick_mm
                            - Vec3::splat(0.5 * voxel_size);
                        let half = Vec3::splat(0.5 * brick_mm + band_mm);
                        // Exact, not just the bounding box overlap the loop
                        // above gives: one long diagonal triangle otherwise
                        // bins into every brick its box touches, which for a
                        // large flat face is most of the model.
                        if triangle_hits_box(triangle, centre, half) {
                            out.push((pack(coord), index as u32));
                        }
                    }
                }
            }
            out
        })
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });

    Bins::from_pairs(pairs)
}

/// Separating axis test between a triangle and an axis aligned box.
///
/// The standard thirteen axes: three box normals, one triangle normal, and the
/// nine cross products of their edge directions.
fn triangle_hits_box(triangle: &[Vec3; 3], centre: Vec3, half: Vec3) -> bool {
    let v = [triangle[0] - centre, triangle[1] - centre, triangle[2] - centre];
    let edges = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];

    // Three box normals.
    for axis in 0..3 {
        let (low, high) = min_max(v[0][axis], v[1][axis], v[2][axis]);
        if low > half[axis] || high < -half[axis] {
            return false;
        }
    }

    // The triangle's own plane.
    let normal = edges[0].cross(edges[1]);
    let distance = normal.dot(v[0]);
    let reach = half.x * normal.x.abs() + half.y * normal.y.abs() + half.z * normal.z.abs();
    if distance.abs() > reach {
        return false;
    }

    // Nine edge cross products, written out per axis pair.
    for edge in &edges {
        let candidates = [
            Vec3::new(0.0, -edge.z, edge.y),
            Vec3::new(edge.z, 0.0, -edge.x),
            Vec3::new(-edge.y, edge.x, 0.0),
        ];
        for axis in candidates {
            if axis.length_squared() == 0.0 {
                continue;
            }
            let p = [axis.dot(v[0]), axis.dot(v[1]), axis.dot(v[2])];
            let (low, high) = min_max(p[0], p[1], p[2]);
            let reach = half.x * axis.x.abs() + half.y * axis.y.abs() + half.z * axis.z.abs();
            if low > reach || high < -reach {
                return false;
            }
        }
    }
    true
}

#[inline]
fn min_max(a: f32, b: f32, c: f32) -> (f32, f32) {
    (a.min(b).min(c), a.max(b).max(c))
}

// --- filling -----------------------------------------------------------------

fn fill_brick(
    coord: BrickCoord,
    triangles: &[u32],
    corners: &Corners,
    sign: &SignField,
    voxel_size: f32,
    flips: &std::sync::atomic::AtomicUsize,
) -> Option<(BrickCoord, Brick)> {
    let origin = coord.origin();
    let band_mm = NARROW_BAND * voxel_size;

    let mut squared = vec![f32::INFINITY; BRICK_VOXELS];

    // Scatter: each triangle writes only to the voxels it can actually reach.
    // A gather -- every voxel against every triangle in the brick -- visits a
    // strict superset of these pairs, so this is never worse and is far better
    // for a scan whose triangles are smaller than a voxel.
    for index in triangles {
        let triangle = &corners[*index as usize];
        let low = triangle[0].min(triangle[1]).min(triangle[2]) - Vec3::splat(band_mm);
        let high = triangle[0].max(triangle[1]).max(triangle[2]) + Vec3::splat(band_mm);
        let from = (low / voxel_size).floor().as_ivec3().max(origin);
        let to = (high / voxel_size).ceil().as_ivec3().min(coord.max_voxel());

        for z in from.z..=to.z {
            for y in from.y..=to.y {
                for x in from.x..=to.x {
                    let world = IVec3::new(x, y, z).as_vec3() * voxel_size;
                    let d2 = point_triangle_distance_squared(world, triangle);
                    let slot = brick_index(
                        (x - origin.x) as usize,
                        (y - origin.y) as usize,
                        (z - origin.z) as usize,
                    );
                    if d2 < squared[slot] {
                        squared[slot] = d2;
                    }
                }
            }
        }
    }

    let mut brick = Brick::dense_filled(OUTSIDE);
    let data = brick.make_dense();
    for z in 0..BRICK_DIM {
        for y in 0..BRICK_DIM {
            let voxel_y = origin.y + y as i32;
            let voxel_z = origin.z + z as i32;
            let line = sign.line_of(voxel_y, voxel_z);
            for x in 0..BRICK_DIM {
                let slot = brick_index(x, y, z);
                let world_x = (origin.x + x as i32) as f32 * voxel_size;
                let inside = match line {
                    Some(line) => sign.inside_at(line, world_x),
                    None => false,
                };
                let distance_mm = squared[slot].sqrt();
                let mut signed = if inside { -distance_mm } else { distance_mm };
                // Bias an exact zero to the inside, after the sign decision and
                // in the same units as the value. `-0.0 < 0.0` is false in Rust
                // and `fast-surface-nets` classifies with `d < 0.0`, so a `-0.0`
                // reads as outside to the mesher while `raycast` reads `<= 0.0`
                // as inside: the cursor would hit a surface the mesh does not
                // have, and a flat plate whose samples all land at exactly zero
                // would emit no triangles at all, silently.
                if inside && signed == 0.0 {
                    signed = -1.0e-5 * voxel_size;
                }
                data[slot] = (signed / voxel_size).clamp(INSIDE, OUTSIDE);
            }
        }
    }

    let found = count_isolated_flips(data);
    if found > 0 {
        flips.fetch_add(found, std::sync::atomic::Ordering::Relaxed);
    }

    match brick.is_collapsible() {
        Some(value) if value >= OUTSIDE => None,
        Some(value) => Some((coord, Brick::Uniform(value))),
        None => Some((coord, brick)),
    }
}

/// Voxels whose sign disagrees with all six of their neighbours.
///
/// Free to compute here, and the cheapest signal available for the two ways the
/// sign sweep can be corrupted: a ray leaking through a hole in the mesh, and a
/// duplicated face miscounting a crossing. A solid has no isolated voxels, so
/// any count above a handful means the input is not what it claims to be.
///
/// Only interior voxels are examined -- a brick's own faces would need the
/// neighbouring brick's data, and this is a quality signal rather than a
/// measurement, so the edge is not worth gathering an apron for.
fn count_isolated_flips(data: &[f32; BRICK_VOXELS]) -> usize {
    let mut found = 0;
    for z in 1..BRICK_DIM - 1 {
        for y in 1..BRICK_DIM - 1 {
            for x in 1..BRICK_DIM - 1 {
                let inside = data[brick_index(x, y, z)] < 0.0;
                let alone = [
                    brick_index(x - 1, y, z),
                    brick_index(x + 1, y, z),
                    brick_index(x, y - 1, z),
                    brick_index(x, y + 1, z),
                    brick_index(x, y, z - 1),
                    brick_index(x, y, z + 1),
                ]
                .iter()
                .all(|slot| (data[*slot] < 0.0) != inside);
                if alone {
                    found += 1;
                }
            }
        }
    }
    found
}

/// Bricks no triangle reached, which are therefore homogeneous.
///
/// Probed at nine points -- the eight lattice corners and the centre -- rather
/// than one. A single probe turns any one wrong grid line into a 32 cubed solid
/// cube of material that meshes cleanly, uses every edge exactly twice, and
/// passes `is_printable`: precisely the plausible looking corruption this whole
/// module is arranged to avoid. Disagreement between the nine means the sign
/// field is not to be trusted there, so the brick is resolved voxel by voxel
/// instead.
fn homogeneous_bricks(
    bins: &Bins,
    sign: &SignField,
    b_min: BrickCoord,
    b_max: BrickCoord,
) -> Vec<(BrickCoord, Brick)> {
    let mut out = Vec::new();
    for bz in b_min.0.z..=b_max.0.z {
        for by in b_min.0.y..=b_max.0.y {
            for bx in b_min.0.x..=b_max.0.x {
                let coord = BrickCoord::new(bx, by, bz);
                if bins.contains(coord) {
                    continue;
                }
                let origin = coord.origin();
                let far = coord.max_voxel();
                let mut probes = Vec::with_capacity(9);
                for z in [origin.z, far.z] {
                    for y in [origin.y, far.y] {
                        for x in [origin.x, far.x] {
                            probes.push(sign.inside(IVec3::new(x, y, z)));
                        }
                    }
                }
                probes.push(sign.inside((origin + far) / 2));

                if probes.iter().all(|inside| *inside) {
                    out.push((coord, Brick::Uniform(INSIDE)));
                } else if probes.iter().any(|inside| *inside) {
                    // The nine disagree. Resolve every voxel from its own probe
                    // rather than trusting either answer.
                    let mut brick = Brick::dense_filled(OUTSIDE);
                    let data = brick.make_dense();
                    for z in 0..BRICK_DIM {
                        for y in 0..BRICK_DIM {
                            for x in 0..BRICK_DIM {
                                let voxel = origin + IVec3::new(x as i32, y as i32, z as i32);
                                if sign.inside(voxel) {
                                    data[brick_index(x, y, z)] = INSIDE;
                                }
                            }
                        }
                    }
                    match brick.is_collapsible() {
                        Some(value) if value >= OUTSIDE => {}
                        Some(value) => out.push((coord, Brick::Uniform(value))),
                        None => out.push((coord, brick)),
                    }
                }
                // Unanimously outside: leave it absent, which reads as OUTSIDE.
            }
        }
    }
    out
}

/// Squared distance from a point to a triangle.
///
/// Ericson's closest-point-on-triangle, by Voronoi region. Written out rather
/// than pulled in, because it is thirty lines and `brokkr-core` adds no
/// dependency it can avoid.
fn point_triangle_distance_squared(point: Vec3, triangle: &[Vec3; 3]) -> f32 {
    let [a, b, c] = *triangle;
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }

    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = if d1 - d3 != 0.0 { d1 / (d1 - d3) } else { 0.0 };
        return (ap - ab * t).length_squared();
    }

    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = if d2 - d6 != 0.0 { d2 / (d2 - d6) } else { 0.0 };
        return (ap - ac * t).length_squared();
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denominator = (d4 - d3) + (d5 - d6);
        let t = if denominator != 0.0 { (d4 - d3) / denominator } else { 0.0 };
        return (point - (b + (c - b) * t)).length_squared();
    }

    let denominator = va + vb + vc;
    if denominator == 0.0 {
        return ap.length_squared();
    }
    let v = vb / denominator;
    let w = vc / denominator;
    (point - (a + ab * v + ac * w)).length_squared()
}

/// How many input triangles left no trace in the field.
///
/// Sampling the finished field just off each face, on both sides: if both read
/// as outside, that triangle's surface was thinner than the lattice can carry
/// and was lost. This reports what actually happened rather than estimating it
/// from an edge length percentile, and it is sign agnostic, so it survives a
/// mesh with inconsistent winding.
fn count_lost_surface(volume: &Volume, corners: &Corners, voxel_size: f32) -> usize {
    corners
        .par_iter()
        .filter(|[a, b, c]| {
            let normal = (*b - *a).cross(*c - *a);
            let Some(normal) = normal.try_normalize() else {
                return false;
            };
            let centroid = (*a + *b + *c) / 3.0;
            let step = normal * (0.5 * voxel_size);
            volume.sample_world(centroid + step) > 0.0 && volume.sample_world(centroid - step) > 0.0
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickCoord;

    const VOXEL: f32 = 0.5;

    /// An axis aligned cube as twelve triangles, wound counter-clockwise seen
    /// from outside.
    ///
    /// `low` and `high` are the opposite corners. Deliberately buildable at
    /// coordinates that land exactly on lattice planes, because that is the case
    /// the tie-break rule exists for and the one that fails loudly without it.
    fn cube(low: Vec3, high: Vec3) -> ExportMesh {
        let c = [
            Vec3::new(low.x, low.y, low.z),
            Vec3::new(high.x, low.y, low.z),
            Vec3::new(high.x, high.y, low.z),
            Vec3::new(low.x, high.y, low.z),
            Vec3::new(low.x, low.y, high.z),
            Vec3::new(high.x, low.y, high.z),
            Vec3::new(high.x, high.y, high.z),
            Vec3::new(low.x, high.y, high.z),
        ];
        let faces: [[usize; 3]; 12] = [
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        ExportMesh {
            positions: c.to_vec(),
            normals: Vec::new(),
            triangles: faces.iter().map(|[a, b, cc]| [*a as u32, *b as u32, *cc as u32]).collect(),
        }
    }

    fn unit_cube() -> ExportMesh {
        cube(Vec3::splat(-5.0), Vec3::splat(5.0))
    }

    fn build(mesh: &ExportMesh) -> (Volume, VoxeliseReport) {
        voxelise(mesh, &VoxeliseOptions::at(VOXEL)).expect("this fixture should voxelise")
    }

    /// How solid the result is, as a fraction of the voxels inside the shape's
    /// own bounding box that came back negative.
    fn solidity(volume: &Volume, low: Vec3, high: Vec3) -> f64 {
        let mut inside = 0usize;
        let mut total = 0usize;
        let step = VOXEL;
        let mut z = low.z + step;
        while z < high.z - step {
            let mut y = low.y + step;
            while y < high.y - step {
                let mut x = low.x + step;
                while x < high.x - step {
                    total += 1;
                    if volume.sample_world(Vec3::new(x, y, z)) < 0.0 {
                        inside += 1;
                    }
                    x += step;
                }
                y += step;
            }
            z += step;
        }
        inside as f64 / total.max(1) as f64
    }

    // --- the conventions, each of which is silent when wrong -----------------

    /// A half voxel offset, or millimetres stored where voxels belong, both mesh
    /// perfectly and put the surface in the wrong place. This pins both.
    #[test]
    fn stored_values_are_signed_distance_in_voxels_with_no_half_voxel_offset() {
        let (volume, _) = build(&unit_cube());
        // Walk out along +X from the centre. The cube face is at x = 5.
        for step in 0..8 {
            let x = 5.0 - step as f32 * VOXEL;
            let expected = ((x - 5.0) / VOXEL).clamp(INSIDE, OUTSIDE);
            let got = volume.sample_world(Vec3::new(x, 0.0, 0.0));
            assert!(
                (got - expected).abs() < 0.25,
                "at x = {x} the field reads {got}, expected about {expected} -- \
                 a value in millimetres or a half voxel offset would look like this"
            );
        }
    }

    /// An x/z swap in a fill loop mirrors the model about the diagonal. It
    /// meshes cleanly, exports watertight and passes the seam test, so extents
    /// per axis are the only cheap thing that catches it.
    ///
    /// Measured by walking out from the centre until the field turns positive,
    /// NOT from `world_bounds`: those are brick extents, granular to 16 mm here,
    /// and a model this size lands inside one brick on every axis and reports a
    /// perfect cube whatever its real shape.
    #[test]
    fn the_axes_are_not_transposed() {
        let mesh = cube(Vec3::new(-2.0, -5.0, -11.0), Vec3::new(2.0, 5.0, 11.0));
        let (volume, _) = build(&mesh);

        let reach = |direction: Vec3| {
            let mut distance = 0.0f32;
            while distance < 40.0 {
                if volume.sample_world(direction * distance) >= 0.0 {
                    return distance;
                }
                distance += VOXEL * 0.5;
            }
            distance
        };
        let x = reach(Vec3::X);
        let y = reach(Vec3::Y);
        let z = reach(Vec3::Z);
        assert!(
            (x - 2.0).abs() < 1.0 && (y - 5.0).abs() < 1.0 && (z - 11.0).abs() < 1.0,
            "the half extents read {x}, {y}, {z} but the cube is 2, 5, 11 -- \
             an axis swap in the fill loop looks exactly like this"
        );
    }

    /// Interiors left absent read as OUTSIDE, so the model comes back hollow
    /// with an internal void that `is_printable` still calls watertight. The
    /// uniform brick count is what proves the interior was filled.
    #[test]
    fn the_interior_is_solid_rather_than_a_hollow_shell() {
        let mesh = cube(Vec3::splat(-20.0), Vec3::splat(20.0));
        let (volume, report) = build(&mesh);
        assert!(
            report.uniform_bricks > 0,
            "no interior tiles were written, so the model is a hollow shell"
        );
        assert!(
            volume.sample_world(Vec3::ZERO) <= INSIDE,
            "the very centre of a 40 mm cube is not solid"
        );
        assert!(solidity(&volume, Vec3::splat(-20.0), Vec3::splat(20.0)) > 0.98);
    }

    // --- the adversarial fixtures --------------------------------------------
    //
    // Each is the same cube, mutated one way, and each must still come back a
    // solid cube. They exist to make the choice of sign method falsifiable
    // rather than argued.

    #[test]
    fn a_cube_whose_faces_land_exactly_on_lattice_planes_is_solid() {
        // Every coordinate is a multiple of the voxel size, so every grid line
        // meets shared edges exactly. Without the top-left tie-break these
        // double count and whole rows invert.
        let mesh = cube(Vec3::splat(-5.0), Vec3::splat(5.0));
        let (volume, _) = build(&mesh);
        assert!(
            solidity(&volume, Vec3::splat(-5.0), Vec3::splat(5.0)) > 0.98,
            "a lattice aligned cube came back with holes, which is the tie-break failing"
        );
    }

    #[test]
    fn one_flipped_triangle_does_not_change_the_solid() {
        let mut mesh = unit_cube();
        mesh.triangles[3].swap(0, 1);
        let (volume, _) = build(&mesh);
        assert!(
            solidity(&volume, Vec3::splat(-5.0), Vec3::splat(5.0)) > 0.95,
            "one reversed triangle broke the solid, which a nearest-normal sign would do"
        );
    }

    /// A duplicated coincident shell is routine output of CSG and of "export
    /// both halves" workflows. Even-odd parity calls it completely empty.
    #[test]
    fn a_doubled_coincident_shell_is_still_solid() {
        let mut mesh = unit_cube();
        let doubled = mesh.triangles.clone();
        mesh.triangles.extend(doubled);
        let (volume, _) = build(&mesh);
        assert!(
            solidity(&volume, Vec3::splat(-5.0), Vec3::splat(5.0)) > 0.98,
            "a doubled shell came back empty, which is even-odd parity rather than nonzero"
        );
    }

    /// A globally inverted mesh reads as winding -1, which is still nonzero.
    #[test]
    fn a_completely_reversed_mesh_is_still_solid() {
        let mut mesh = unit_cube();
        for triangle in &mut mesh.triangles {
            triangle.swap(0, 1);
        }
        let (volume, _) = build(&mesh);
        assert!(
            solidity(&volume, Vec3::splat(-5.0), Vec3::splat(5.0)) > 0.98,
            "reversing every triangle emptied the model"
        );
    }

    /// Two overlapping cubes must union into one solid with no internal
    /// membrane where they meet.
    #[test]
    fn two_overlapping_cubes_union_without_an_internal_bubble() {
        let mut mesh = cube(Vec3::new(-6.0, -4.0, -4.0), Vec3::new(2.0, 4.0, 4.0));
        let second = cube(Vec3::new(-2.0, -4.0, -4.0), Vec3::new(6.0, 4.0, 4.0));
        let offset = mesh.positions.len() as u32;
        mesh.positions.extend(second.positions);
        mesh.triangles
            .extend(second.triangles.iter().map(|[a, b, c]| [a + offset, b + offset, c + offset]));

        let (volume, _) = build(&mesh);
        // The overlap region is solid, and so is each cube's exclusive part.
        for probe in [Vec3::new(0.0, 0.0, 0.0), Vec3::new(-4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)]
        {
            assert!(
                volume.sample_world(probe) < 0.0,
                "the union is not solid at {probe:?}, so the two shells cancelled"
            );
        }
        let (exported, report) = volume.export_mesh();
        assert!(
            report.is_printable(),
            "the union is not printable: {} ({} triangles)",
            report.summary(),
            exported.triangles.len()
        );
    }

    /// STL writes every corner separately, so an unwelded cube is the normal
    /// case rather than the exception. It must produce the identical field.
    #[test]
    fn an_unwelded_cube_produces_the_same_field_as_a_welded_one() {
        let welded = unit_cube();
        let mut unwelded = ExportMesh::default();
        for [a, b, c] in &welded.triangles {
            let base = unwelded.positions.len() as u32;
            for index in [a, b, c] {
                unwelded.positions.push(welded.positions[*index as usize]);
            }
            unwelded.triangles.push([base, base + 1, base + 2]);
        }

        let (from_welded, _) = build(&welded);
        let (from_unwelded, _) = build(&unwelded);
        for probe in [Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::new(0.0, 4.9, 0.0)] {
            assert_eq!(
                from_welded.sample_world(probe),
                from_unwelded.sample_world(probe),
                "welding changed the field at {probe:?}"
            );
        }
    }

    // --- the keystone --------------------------------------------------------

    /// Round trip a shape the engine can build analytically: seed a sphere,
    /// export it, voxelise the export at the same size, and require the two
    /// fields to agree.
    ///
    /// This is the test that catches a sign convention, a unit error and a
    /// hollow interior all at once, because the reference is the engine's own
    /// `seed_sphere` rather than an expectation written by hand.
    #[test]
    fn a_seeded_sphere_survives_being_exported_and_voxelised_back() {
        let mut original = Volume::new(VOXEL);
        original.seed_sphere(Vec3::ZERO, 40.0);
        original.mark_everything_dirty();
        let (mesh, report) = original.export_mesh();
        assert!(report.is_printable(), "the fixture is not printable: {}", report.summary());

        let (rebuilt, rebuilt_report) = voxelise(
            &mesh,
            &VoxeliseOptions { voxel_size: VOXEL, centre: false, refit_if_implausible: false },
        )
        .expect("a sphere should voxelise");

        // A radius of 40 mm is not arbitrary. A brick is 32 voxels, so 16 mm
        // here, and a brick counts as interior only when its furthest CORNER
        // clears the surface by the band -- so the sphere has to be wider than a
        // brick's space diagonal, about 27.7 mm, before any brick is purely
        // inside at all. At 24 mm every brick within the sphere still has a
        // corner poking through it, and asserting on tiles fails for a reason
        // that has nothing to do with the code being tested.
        assert!(rebuilt_report.uniform_bricks > 0, "the rebuilt sphere is hollow");
        assert_eq!(
            rebuilt.sample_world(Vec3::ZERO),
            INSIDE,
            "the centre of the rebuilt sphere is not fully inside"
        );

        // Every stored value is a clamped voxel distance. A field in millimetres
        // would still put the zero crossing in the right place, so this is what
        // actually catches it.
        for coord in rebuilt.brick_coords() {
            let origin = coord.origin();
            for step in 0..BRICK_DIM {
                let value = rebuilt.sample_voxel(origin + IVec3::new(step as i32, 0, 0));
                assert!(
                    (INSIDE..=OUTSIDE).contains(&value),
                    "a stored value of {value} is outside the narrow band"
                );
            }
        }

        // And the surfaces agree: sample both fields on a shell of directions.
        let mut worst: f32 = 0.0;
        for index in 0..64 {
            let angle = index as f32 * 0.61;
            let direction =
                Vec3::new(angle.cos(), (angle * 1.7).sin(), (angle * 0.9).cos()).normalize();
            for radius in [20.0f32, 38.0, 40.0, 42.0, 55.0] {
                let probe = direction * radius;
                let difference = (original.sample_world(probe) - rebuilt.sample_world(probe)).abs();
                worst = worst.max(difference);
            }
        }
        assert!(worst < 1.2, "the rebuilt field differs from the original by {worst} voxels");
    }

    /// An imported model has to survive the thing every exported model must:
    /// meshing to a closed, manifold surface.
    #[test]
    fn an_imported_model_still_exports_watertight() {
        for mesh in [unit_cube(), cube(Vec3::new(-3.0, -7.0, -2.0), Vec3::new(9.0, 1.0, 6.0))] {
            let (volume, _) = build(&mesh);
            let (_, report) = volume.export_mesh();
            assert!(
                report.is_printable(),
                "an imported cube is not printable: {}",
                report.summary()
            );
        }
    }

    // --- placement and refusal ------------------------------------------------

    #[test]
    fn a_model_is_centred_on_the_origin() {
        let mesh = cube(Vec3::new(40.0, 40.0, 40.0), Vec3::new(50.0, 50.0, 50.0));
        let (volume, _) = build(&mesh);
        assert!(
            volume.sample_world(Vec3::ZERO) < 0.0,
            "a cube far from the origin was not centred, so the camera would frame empty space"
        );
    }

    /// A model that states a real size must keep it. Rescaling every import
    /// would destroy the one number that matters in a tool whose output is a
    /// printed part.
    #[test]
    fn a_plausibly_sized_model_is_not_rescaled() {
        let mesh = cube(Vec3::splat(-15.0), Vec3::splat(15.0));
        let (_, report) = build(&mesh);
        assert_eq!(report.rescaled_by, 1.0, "a 30 mm model was resized");
        assert!((report.longest_mm - 30.0).abs() < 0.01);
    }

    /// A file written in metres arrives a thousand times too small, which is a
    /// unit error rather than a design choice.
    #[test]
    fn an_implausibly_tiny_model_is_refitted() {
        let mesh = cube(Vec3::splat(-0.01), Vec3::splat(0.01));
        let (_, report) = build(&mesh);
        assert!(report.rescaled_by > 1.0, "a 0.02 mm model was left unusably small");
        assert!((report.longest_mm - REFIT_LONGEST_MM).abs() < 0.01);
    }

    /// The only backstop otherwise is an assert deep in `FieldRegion::resize`,
    /// which is a panic rather than something a user can act on.
    #[test]
    fn a_model_too_large_for_the_lattice_is_refused_with_a_workable_size() {
        let mesh = cube(Vec3::splat(-2500.0), Vec3::splat(2500.0));
        let outcome = voxelise(
            &mesh,
            &VoxeliseOptions { voxel_size: 0.05, centre: true, refit_if_implausible: false },
        );
        // `Volume` has no `Debug`, so the success case cannot be unwrapped into
        // a message; match instead.
        let Err(error) = outcome else {
            panic!("a five metre cube at 0.05 mm was accepted");
        };
        let text = error.to_string();
        assert!(text.contains("resample"), "the refusal does not say what to do: {text}");
    }

    #[test]
    fn an_open_sheet_is_refused_rather_than_imported_as_an_empty_model() {
        let mesh = ExportMesh {
            positions: vec![
                Vec3::new(-5.0, 0.0, -5.0),
                Vec3::new(5.0, 0.0, -5.0),
                Vec3::new(5.0, 0.0, 5.0),
                Vec3::new(-5.0, 0.0, 5.0),
            ],
            normals: Vec::new(),
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        assert!(
            voxelise(&mesh, &VoxeliseOptions::at(VOXEL)).is_err(),
            "a flat sheet produced a model rather than an error"
        );
    }

    #[test]
    fn an_empty_mesh_is_refused() {
        assert!(voxelise(&ExportMesh::default(), &VoxeliseOptions::at(VOXEL)).is_err());
    }

    /// Rayon splits the work differently from run to run, so a `min` that
    /// depended on order would show up here and nowhere else.
    #[test]
    fn voxelising_the_same_mesh_twice_gives_the_same_field() {
        let mesh = cube(Vec3::new(-4.3, -6.1, -2.7), Vec3::new(5.9, 3.3, 7.1));
        let (first, _) = build(&mesh);
        let (second, _) = build(&mesh);

        let mut first_coords: Vec<BrickCoord> = first.brick_coords().collect();
        let mut second_coords: Vec<BrickCoord> = second.brick_coords().collect();
        first_coords.sort();
        second_coords.sort();
        assert_eq!(first_coords, second_coords, "a different set of bricks was produced");

        for coord in first_coords {
            let origin = coord.origin();
            for step in 0..BRICK_DIM {
                let voxel = origin + IVec3::new(step as i32, (step % 7) as i32, (step % 5) as i32);
                assert_eq!(
                    first.sample_voxel(voxel),
                    second.sample_voxel(voxel),
                    "the field differs at {voxel:?} between two runs"
                );
            }
        }
    }
}
