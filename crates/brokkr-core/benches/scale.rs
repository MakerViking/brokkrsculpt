// SPDX-License-Identifier: AGPL-3.0-only

//! What the CPU engine does at M2 scale.
//!
//! M2's target is ten million triangles staying interactive. This measures the
//! current CPU path at that size so the work of moving to compute shaders is
//! aimed at whatever actually breaks, rather than at whatever seems likely.
//!
//! Deliberately separate from the budget harness, which has to stay quick
//! enough to run on every change. This one allocates most of a gigabyte and
//! takes a while.
//!
//! Run with `cargo bench -p brokkr-core --bench scale`.

use std::time::Instant;

use brokkr_core::{
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, DEFAULT_RECLAIM_BUDGET,
    Similarity, Stamp, Volume, redistance::GRADIENT_TOLERANCE,
};
use glam::{Quat, Vec3};

/// The model, in millimetres, matching what the application seeds.
const MODEL_RADIUS: f32 = 30.0;

/// Brush radius in millimetres, also matching the application's default. Held
/// in world units on purpose: a finer voxel size means the same brush covers
/// cubically more voxels, which is the whole point of the measurement.
const BRUSH_RADIUS: f32 = 3.0;

fn millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn megabytes(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Mesh everything that can carry geometry and report the triangle count.
/// What one full mesh of the model came to.
struct FullMesh {
    triangles: usize,
    /// Vertices as the renderer must actually store them, which is more than a
    /// welded count: each brick meshes independently and so duplicates the
    /// vertices along its seams.
    vertices: usize,
    indices: usize,
    /// Bricks that carry any geometry, which is the draw call count before any
    /// culling or batching.
    drawn_bricks: usize,
    milliseconds: f64,
}

fn mesh_all(volume: &mut Volume) -> FullMesh {
    volume.mark_everything_dirty();
    let mut dirty = Vec::new();
    volume.take_dirty(&mut dirty);

    let mut meshes = vec![BrickMesh::default(); dirty.len()];
    let started = Instant::now();
    volume.mesh_bricks(&dirty, &mut meshes);
    let milliseconds = millis(started);

    FullMesh {
        triangles: meshes.iter().map(BrickMesh::triangle_count).sum(),
        vertices: meshes.iter().map(|mesh| mesh.vertices.len()).sum(),
        indices: meshes.iter().map(|mesh| mesh.indices.len()).sum(),
        drawn_bricks: meshes.iter().filter(|mesh| !mesh.is_empty()).count(),
        milliseconds,
    }
}

/// What one release-time bake cost, for a body of a given size.
struct BakePasses {
    warp_ms: f64,
    redistance_ms: f64,
    /// The worst `||grad d| - 1|` before the warp, after it, and after the
    /// repair. The middle number is the damage a per-axis scale does; the last
    /// is whether `redistance` actually undid it.
    drift_before: f32,
    drift_warped: f32,
    drift_repaired: f32,
    dense_before: usize,
    dense_after: usize,
    resident_before: usize,
    resident_after: usize,
}

/// A per-axis scale and the repair that follows it, which is what every gizmo
/// release already pays for and what a deformer will pay again.
///
/// The scale is deliberately non-uniform: a uniform one leaves the field a true
/// distance field, `Similarity::is_uniform_scale` says so, and `rebake_gizmo`
/// skips the repair entirely. This measures the branch that does not skip.
fn bake_passes(volume: &Volume) -> BakePasses {
    let before = volume.stats();
    let drift_before = volume.gradient_drift();

    let squash =
        Similarity::about(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 0.6, 1.0), Vec3::ZERO);
    let warp_start = Instant::now();
    let mut warped = volume.warped(squash);
    let warp_ms = millis(warp_start);
    let drift_warped = warped.gradient_drift();

    let redistance_start = Instant::now();
    let report = warped.redistance();
    let redistance_ms = millis(redistance_start);

    let after = warped.stats();
    BakePasses {
        warp_ms,
        redistance_ms,
        drift_before,
        drift_warped,
        // `redistance` returns None when it declined, which means it judged the
        // field already within tolerance -- so the drift it would have reported
        // is the one measured going in.
        drift_repaired: report.map_or(drift_warped, |report| report.worst_after),
        dense_before: before.dense_bricks,
        dense_after: after.dense_bricks,
        resident_before: before.resident_bytes,
        resident_after: after.resident_bytes,
    }
}

/// The two passes that run on every gizmo release, neither of which had ever
/// been timed.
///
/// Deliberately not in `budget.rs`: every constant there is derived from the
/// 16 ms frame, and a release-time bake has no defensible frame budget. Putting
/// these rows beside a budget would invite the next reader to invent one.
///
/// **The sweep stops at the size the gizmo can actually arm on.** `arm_gizmo`
/// refuses outright when `size_of::<Volume>() + resident` exceeds
/// [`DEFAULT_RECLAIM_BUDGET`], because a move it could not undo is worse than a
/// move it will not make. Measuring past that measures a regime a deformer can
/// never enter -- the reference dragon at 765 MB is one such body, and it is
/// why the rows below print whether the gizmo would take them.
fn report_release_passes() {
    println!(
        "The two passes every gizmo release pays for: one trilinear warp, then the repair.\n\
         A {MODEL_RADIUS} mm sphere squashed to 0.6 along Y. The repair runs on EVERY resample\n\
         now, uniform scales included -- a similarity leaves the measured distances exact but\n\
         not the band, and this is the pass that puts the band back.\n"
    );

    for voxel_size in [0.25_f32, 0.18, 0.125, 0.09, 0.07] {
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);

        let passes = bake_passes(&volume);
        let entry_bytes = size_of::<Volume>() + passes.resident_before;
        let armable = entry_bytes <= DEFAULT_RECLAIM_BUDGET;

        println!(
            "voxel {voxel_size:.4} mm   {:>7.1} MB in {} dense bricks   {}",
            megabytes(passes.resident_before),
            passes.dense_before,
            if armable {
                "the gizmo would arm on this"
            } else {
                "OVER the 512 MB arm limit, unreachable through the gizmo"
            },
        );
        println!(
            "  warp {:>9.1} ms   redistance {:>9.1} ms   together {:>9.1} ms",
            passes.warp_ms,
            passes.redistance_ms,
            passes.warp_ms + passes.redistance_ms,
        );
        println!(
            "  drift {:.3} before, {:.3} warped, {:.3} repaired   {}",
            passes.drift_before,
            passes.drift_warped,
            passes.drift_repaired,
            if passes.drift_repaired <= GRADIENT_TOLERANCE {
                "within tolerance"
            } else {
                "STILL OUT OF TOLERANCE"
            },
        );
        println!(
            "  {} dense bricks after, {:>7.1} MB   {:+.1}% resident",
            passes.dense_after,
            megabytes(passes.resident_after),
            (passes.resident_after as f64 / passes.resident_before.max(1) as f64 - 1.0) * 100.0,
        );
        println!();
    }

    println!(
        "Read the redistance column first. It is serial where every other heavy pass in this\n\
         crate is rayon-parallel, it runs NARROW_BAND + 1 outer iterations of eight Godunov\n\
         sweeps over every dense brick, and it allocates a fresh 128 KB vector inside the outer\n\
         loop. If it is seconds at the arm limit, then committing a deformation freezes the\n\
         window and the repair has to be made parallel before anything is built on top of it.\n\n\
         Read the resident column second. A warp that leaves more dense bricks than it found has\n\
         not grown the model -- it has failed to collapse bricks back to tiles, which costs\n\
         128 KB each and is invisible to any growth ceiling measured in stretch.\n"
    );
}

fn main() {
    println!("BrokkrSculpt scale report: what the CPU path does as the voxel size shrinks.\n");
    println!(
        "A {MODEL_RADIUS} mm sphere, brush radius {BRUSH_RADIUS} mm held constant in world units.\n"
    );

    // Each step roughly doubles the triangle count. The last is around the M2
    // target of ten million.
    for voxel_size in [0.25_f32, 0.18, 0.125, 0.09, 0.0625, 0.055] {
        let effective = (MODEL_RADIUS * 2.0 / voxel_size) as u32;
        let brush_voxels = BRUSH_RADIUS / voxel_size;

        let mut volume = Volume::new(voxel_size);
        let seed_start = Instant::now();
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
        let seed_ms = millis(seed_start);

        let full = mesh_all(&mut volume);
        let stats = volume.stats();

        // One stamp of the brush, which is the per event cost that has a budget.
        let mut brush_scratch = BrushScratch::new();
        let brush = Brush {
            kind: BrushKind::Draw,
            radius: BRUSH_RADIUS,
            strength: 0.5,
            ..Brush::default()
        };
        let at = Vec3::new(MODEL_RADIUS, 0.0, 0.0);
        let normal = volume.gradient_world(at);

        // Warm the bricks so the first stamp's allocation is not counted alone.
        brush.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut brush_scratch);
        let mut dirty: Vec<BrickCoord> = Vec::new();
        volume.take_dirty(&mut dirty);

        let stamp_start = Instant::now();
        for _ in 0..5 {
            brush.apply(
                &mut volume,
                &Stamp::new(at, normal, BrushDirection::Add),
                &mut brush_scratch,
            );
        }
        let stamp_ms = millis(stamp_start) / 5.0;

        volume.take_dirty(&mut dirty);
        let mut remesh_buffers = vec![BrickMesh::default(); dirty.len()];
        let remesh_start = Instant::now();
        volume.mesh_bricks(&dirty, &mut remesh_buffers);
        let remesh_ms = millis(remesh_start);

        // Mesh memory as the renderer actually holds it, from the real counts.
        let mesh_mb = megabytes(full.vertices * 24 + full.indices * 4);

        println!(
            "voxel {voxel_size:.4} mm, {effective} cubed effective, brush {brush_voxels:.0} voxels across"
        );
        println!(
            "  {:>10} triangles   seed {seed_ms:>8.1} ms   full mesh {:>9.1} ms",
            full.triangles, full.milliseconds
        );
        println!(
            "  {:>10} vertices   {:>11} indices   {} bricks to draw",
            full.vertices, full.indices, full.drawn_bricks
        );
        println!(
            "  volume {:>7.1} MB in {} dense bricks   mesh {:>7.1} MB   total {:>7.1} MB",
            megabytes(stats.resident_bytes),
            stats.dense_bricks,
            mesh_mb,
            megabytes(stats.resident_bytes) + mesh_mb
        );
        println!(
            "  one stamp {stamp_ms:>8.3} ms {}   remesh {} bricks {remesh_ms:>7.3} ms {}",
            if stamp_ms > 4.0 { "OVER 4 ms" } else { "under budget" },
            dirty.len(),
            if remesh_ms > 8.0 { "OVER 8 ms" } else { "under budget" },
        );
        println!();
    }

    println!(
        "The brush covers a fixed world radius, so halving the voxel size costs eight times as\n\
         much per stamp while the remesh only grows with the number of bricks touched. Whichever\n\
         of those two crosses its budget first is what has to move to the GPU."
    );

    println!();
    report_release_passes();
}
