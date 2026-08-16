// SPDX-License-Identifier: AGPL-3.0-or-later

//! Performance budget harness.
//!
//! The budgets in the build spec are meant to be tests, not aspirations, so
//! this runs a realistic stroke over a 256 cubed effective volume and reports
//! each phase against its budget. It exits non zero if a budget is blown, which
//! is what makes it usable as a gate.
//!
//! Run it with `cargo bench -p brokkr-core`. It deliberately uses no benchmark
//! framework: the numbers that matter here are worst case per stroke step
//! latencies against a fixed budget, not throughput distributions.

use std::time::{Duration, Instant};

use brokkr_core::{BrickCoord, BrickMesh, BrushDirection, DrawBrush, MeshScratch, Volume};
use glam::Vec3;

/// Total frame budget at 60 fps.
const FRAME_BUDGET: Duration = Duration::from_micros(16_000);
/// Applying the brush to the voxel field.
const EDIT_BUDGET: Duration = Duration::from_micros(4_000);
/// Remeshing whatever the edit dirtied.
const REMESH_BUDGET: Duration = Duration::from_micros(8_000);

/// Voxels across the model, which is the M0 target size.
const EFFECTIVE_RESOLUTION: f32 = 256.0;
/// Stroke steps to time.
const STEPS: usize = 240;

struct Samples(Vec<Duration>);

impl Samples {
    fn new() -> Self {
        Self(Vec::with_capacity(STEPS))
    }

    fn sorted(&mut self) -> &[Duration] {
        self.0.sort_unstable();
        &self.0
    }

    fn median(&mut self) -> Duration {
        let sorted = self.sorted();
        sorted[sorted.len() / 2]
    }

    fn percentile(&mut self, fraction: f64) -> Duration {
        let sorted = self.sorted();
        let index = ((sorted.len() as f64 - 1.0) * fraction).round() as usize;
        sorted[index]
    }

    fn max(&mut self) -> Duration {
        *self.sorted().last().expect("at least one sample")
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Print one line of the report and say whether it fits its budget.
fn report(label: &str, samples: &mut Samples, budget: Duration) -> bool {
    let median = samples.median();
    let p95 = samples.percentile(0.95);
    let worst = samples.max();
    let ok = p95 <= budget;
    println!(
        "  {:<22} median {:>7.3} ms   p95 {:>7.3} ms   max {:>7.3} ms   budget {:>5.1} ms  {}",
        label,
        millis(median),
        millis(p95),
        millis(worst),
        millis(budget),
        if ok { "pass" } else { "OVER" }
    );
    ok
}

fn main() {
    let voxel_size = 1.0_f32;
    let centre = Vec3::splat(EFFECTIVE_RESOLUTION * voxel_size * 0.5);
    let radius = EFFECTIVE_RESOLUTION * voxel_size * 0.47;

    println!(
        "BrokkrSculpt budget harness: {}^3 effective volume, voxel size {voxel_size}, {STEPS} stroke steps",
        EFFECTIVE_RESOLUTION as u32
    );

    let mut volume = Volume::new(voxel_size);
    let seed_start = Instant::now();
    volume.seed_sphere(centre, radius);
    let seed_time = seed_start.elapsed();

    // Initial full mesh. This is a one off at load, not a per frame cost, so it
    // has no budget. It is reported because it bounds how long opening a model
    // takes and it gives the baseline triangle count.
    let mut scratch = MeshScratch::new();
    let mut mesh = BrickMesh::default();
    let mut dirty = Vec::new();
    volume.take_dirty(&mut dirty);
    // Absent bricks bordering the shell own some boundary quads, so mesh a
    // margin around the stored set as the renderer does.
    let mut initial = dirty.clone();
    expand(&mut initial);

    let mesh_start = Instant::now();
    let mut triangles = 0usize;
    for &coord in &initial {
        volume.mesh_brick(coord, &mut scratch, &mut mesh);
        triangles += mesh.triangle_count();
    }
    let mesh_time = mesh_start.elapsed();

    let stats = volume.stats();
    println!(
        "  seed {:.1} ms, initial mesh of {} bricks {:.1} ms, {} triangles",
        millis(seed_time),
        initial.len(),
        millis(mesh_time),
        triangles
    );
    println!(
        "  {} dense bricks, {} uniform bricks, {:.1} MB resident",
        stats.dense_bricks,
        stats.uniform_bricks,
        stats.resident_bytes as f64 / (1024.0 * 1024.0)
    );
    println!();

    // A stroke that drags across the surface, which is what a user actually
    // does and what the per frame budget has to cover.
    let brush = DrawBrush { radius: 12.0 * voxel_size, strength: 0.25 };
    let mut edit_samples = Samples::new();
    let mut remesh_samples = Samples::new();
    let mut combined_samples = Samples::new();
    let mut dirty_total = 0usize;
    let mut remeshed_triangles = 0usize;

    for step in 0..STEPS {
        // Walk the brush around a great circle on the sphere.
        let angle = step as f32 / STEPS as f32 * std::f32::consts::TAU;
        let tilt = (step as f32 * 0.11).sin() * 0.8;
        let point = centre
            + Vec3::new(angle.cos() * tilt.cos(), tilt.sin(), angle.sin() * tilt.cos()) * radius;

        let edit_start = Instant::now();
        brush.apply(&mut volume, point, BrushDirection::Add);
        let edit_time = edit_start.elapsed();

        volume.take_dirty(&mut dirty);
        dirty_total += dirty.len();

        let remesh_start = Instant::now();
        for &coord in &dirty {
            volume.mesh_brick(coord, &mut scratch, &mut mesh);
            remeshed_triangles += mesh.triangle_count();
        }
        let remesh_time = remesh_start.elapsed();

        edit_samples.0.push(edit_time);
        remesh_samples.0.push(remesh_time);
        combined_samples.0.push(edit_time + remesh_time);
    }

    println!("  average {:.1} bricks remeshed per step", dirty_total as f64 / STEPS as f64);
    println!("  {remeshed_triangles} triangles rebuilt over the stroke");
    println!();

    let mut passed = true;
    passed &= report("brush edit", &mut edit_samples, EDIT_BUDGET);
    passed &= report("dirty remesh", &mut remesh_samples, REMESH_BUDGET);
    passed &= report("edit plus remesh", &mut combined_samples, FRAME_BUDGET);

    let after = volume.stats();
    println!();
    println!(
        "  after the stroke: {} dense bricks, {:.1} MB resident",
        after.dense_bricks,
        after.resident_bytes as f64 / (1024.0 * 1024.0)
    );

    if passed {
        println!("\nall budgets met");
    } else {
        eprintln!("\nBUDGET EXCEEDED: see the lines marked OVER above");
        std::process::exit(1);
    }
}

/// Grow a brick list by one in every direction, the way the renderer must so
/// that absent bricks holding boundary quads are meshed too.
fn expand(coords: &mut Vec<BrickCoord>) {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<BrickCoord> = coords.iter().copied().collect();
    for coord in coords.iter() {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    set.insert(BrickCoord::new(
                        coord.0.x + dx,
                        coord.0.y + dy,
                        coord.0.z + dz,
                    ));
                }
            }
        }
    }
    coords.clear();
    coords.extend(set);
}
