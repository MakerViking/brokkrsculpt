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

use brokkr_core::{
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, FalloffCurve, History,
    MeshScratch, Stamp, Stroke, Symmetry, Volume,
};
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

    let mut passed = true;

    // A stroke that drags across the surface, which is what a user actually
    // does and what the per frame budget has to cover.
    //
    // Run with everything M1 added switched on: stroke interpolation, so one
    // pointer event can produce several stamps, X symmetry, so each of those is
    // applied twice, and undo recording, so the first touch of each brick also
    // copies it. Measuring a bare single stamp would flatter the result and
    // measure something the user never does.
    let brush = Brush {
        kind: BrushKind::Draw,
        radius: 12.0 * voxel_size,
        strength: 0.25,
        falloff: FalloffCurve::Smooth,
    };
    let spacing = brush.spacing(voxel_size);

    let mut brush_scratch = BrushScratch::new();
    let mut meshes: Vec<BrickMesh> = Vec::new();
    let mut history = History::default();
    let mut centres: Vec<Vec3> = Vec::new();

    // Slow and fast drags over the same path. The fast one is the case the
    // per event budget really has to survive: fewer pointer samples means more
    // interpolated stamps per event.
    for (label, events) in [("slow drag", STEPS), ("fast drag", STEPS / 5)] {
        let mut edit_samples = Samples::new();
        let mut remesh_samples = Samples::new();
        let mut combined_samples = Samples::new();
        let mut dirty_total = 0usize;
        let mut stamp_total = 0usize;

        let mut stroke = Stroke::new();
        volume.begin_stroke();

        for step in 0..events {
            // Walk the brush around a great circle on the sphere.
            let angle = step as f32 / events as f32 * std::f32::consts::TAU;
            let tilt = (step as f32 * 0.11).sin() * 0.8;
            let point = centre
                + Vec3::new(angle.cos() * tilt.cos(), tilt.sin(), angle.sin() * tilt.cos())
                    * radius;

            let edit_start = Instant::now();
            centres.clear();
            stroke.advance(point, spacing, &mut centres);
            for &at in &centres {
                let normal = volume.gradient_world(at);
                brush.apply_symmetric(
                    &mut volume,
                    &Stamp::new(at, normal, BrushDirection::Add),
                    Symmetry::X,
                    &mut brush_scratch,
                );
            }
            let edit_time = edit_start.elapsed();
            stamp_total += centres.len();

            volume.take_dirty(&mut dirty);
            dirty_total += dirty.len();

            let remesh_start = Instant::now();
            while meshes.len() < dirty.len() {
                meshes.push(BrickMesh::default());
            }
            volume.mesh_bricks(&dirty, &mut meshes[..dirty.len()]);
            let remesh_time = remesh_start.elapsed();

            edit_samples.0.push(edit_time);
            remesh_samples.0.push(remesh_time);
            combined_samples.0.push(edit_time + remesh_time);
        }

        let undo_start = Instant::now();
        if let Some(edit) = volume.end_stroke() {
            print!(
                "  undo entry covers {} bricks, {:.1} MB",
                edit.len(),
                edit.bytes() as f64 / (1024.0 * 1024.0)
            );
            history.push(edit);
        }
        println!(", recorded in {:.2} ms", millis(undo_start.elapsed()));

        println!(
            "  {label}: {events} pointer events, {:.1} stamps and {:.1} bricks remeshed per event",
            stamp_total as f64 / events as f64,
            dirty_total as f64 / events as f64
        );
        passed &= report("  brush edit", &mut edit_samples, EDIT_BUDGET);
        passed &= report("  dirty remesh", &mut remesh_samples, REMESH_BUDGET);
        passed &= report("  edit plus remesh", &mut combined_samples, FRAME_BUDGET);
        println!();
    }

    // Undo and redo are not per frame work, so they carry no budget, but a
    // multi second undo would still be unusable and is worth watching.
    let undo_start = Instant::now();
    let undone = history.undo(&mut volume);
    let undo_time = undo_start.elapsed();
    volume.take_dirty(&mut dirty);
    let restore_start = Instant::now();
    while meshes.len() < dirty.len() {
        meshes.push(BrickMesh::default());
    }
    volume.mesh_bricks(&dirty, &mut meshes[..dirty.len()]);
    println!(
        "  undo of a whole stroke: {:.2} ms to restore {} bricks, {:.1} ms to remesh {} of them (no budget, not per frame)",
        millis(undo_time),
        if undone { "the recorded" } else { "no" },
        millis(restore_start.elapsed()),
        dirty.len()
    );
    println!("  history holds {:.1} MB", history.stats().bytes as f64 / (1024.0 * 1024.0));
    println!();

    // Per brush comparison. Smooth, pinch and flatten read a neighbourhood, so
    // they cost more than the ones that are a pure function of the voxel.
    println!("  cost of one stamp by brush, radius {} voxels:", (brush.radius / voxel_size) as u32);
    for kind in BrushKind::ALL {
        let each = Brush { kind, ..brush };
        let mut samples = Samples::new();
        for step in 0..60 {
            let angle = step as f32 / 60.0 * std::f32::consts::TAU;
            let at = centre + Vec3::new(angle.cos(), 0.0, angle.sin()) * radius;
            let normal = volume.gradient_world(at);
            let started = Instant::now();
            each.apply(
                &mut volume,
                &Stamp::new(at, normal, BrushDirection::Add),
                &mut brush_scratch,
            );
            samples.0.push(started.elapsed());
            volume.take_dirty(&mut dirty);
        }
        passed &= report(&format!("  {kind}"), &mut samples, EDIT_BUDGET);
    }
    println!();

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
                    set.insert(BrickCoord::new(coord.0.x + dx, coord.0.y + dy, coord.0.z + dz));
                }
            }
        }
    }
    coords.clear();
    coords.extend(set);
}
