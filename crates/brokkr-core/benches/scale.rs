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
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, Stamp, Volume,
};
use glam::Vec3;

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
}
