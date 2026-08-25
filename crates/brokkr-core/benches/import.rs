// SPDX-License-Identifier: AGPL-3.0-only

//! Import cost harness, and a gate on it.
//!
//! `budget.rs` gates the per-frame budgets, which is what protects the feel of
//! sculpting. Nothing gated import until this existed: reading a mesh and
//! voxelising it is seconds of work off the update loop, so it cannot stutter a
//! stroke, but it can still take a minute or exhaust memory, and both of those
//! read to a user as the application having hung.
//!
//! The ceilings here are deliberately generous. This is a backstop against an
//! order-of-magnitude regression -- an accidental gather where a scatter was
//! intended, a bin that stops culling, an O(lines x triangles) sweep -- not a
//! tuning target. A change that makes import twice as slow should be noticed by
//! a human reading these numbers; a change that makes it twenty times slower
//! should fail the build.
//!
//! Run it with `cargo bench -p brokkr-core --bench import`. As with the other
//! two, **kill any running instance of the application first**: a live
//! BrokkrSculpt at 60 fps has polluted a measurement on this machine badly
//! enough to look like a 10x regression.

use std::time::{Duration, Instant};

use brokkr_core::Volume;
use brokkr_core::voxelise::{VoxeliseOptions, voxelise};
use glam::Vec3;

/// The reference model: a sphere the size of the one the application seeds.
const MODEL_RADIUS_MM: f32 = 30.0;

/// Wall clock ceiling for reading a mesh off disk.
///
/// Parsing is memcpy and a hash lookup per corner, so this is only ever blown
/// by an accidental quadratic -- a linear scan of the welded set, say.
const READ_CEILING: Duration = Duration::from_millis(4_000);

/// Wall clock ceiling for voxelising the reference model at the default voxel
/// size.
const VOXELISE_CEILING: Duration = Duration::from_millis(20_000);

/// Resident ceiling for the same, which is the number that decides whether an
/// import is usable rather than merely possible.
const RESIDENT_CEILING_MB: f64 = 1_500.0;

fn seeded_sphere(voxel_size: f32, radius: f32) -> Volume {
    let mut volume = Volume::new(voxel_size);
    volume.seed_sphere(Vec3::ZERO, radius);
    volume.mark_everything_dirty();
    volume
}

/// Time one closure, returning its result and how long it took.
fn timed<T>(work: impl FnOnce() -> T) -> (T, Duration) {
    let started = Instant::now();
    let value = work();
    (value, started.elapsed())
}

fn report(label: &str, taken: Duration, ceiling: Duration, passed: &mut bool) {
    let over = taken > ceiling;
    if over {
        *passed = false;
    }
    println!(
        "  {label:<38} {:>8.1} ms  (ceiling {:>6.0} ms) {}",
        taken.as_secs_f64() * 1000.0,
        ceiling.as_secs_f64() * 1000.0,
        if over { "OVER" } else { "ok" }
    );
}

fn main() {
    let mut passed = true;

    println!("import cost, sphere of radius {MODEL_RADIUS_MM} mm\n");

    // Build the reference mesh once, through this project's own exporter, so
    // the bench measures the readers against bytes they will really see.
    let source = seeded_sphere(0.25, MODEL_RADIUS_MM);
    let (mesh, mesh_report) = source.export_mesh();
    assert!(mesh_report.is_printable(), "the fixture is not printable");
    println!("  fixture: {} triangles, {} vertices\n", mesh.triangles.len(), mesh.positions.len());

    let mut stl = Vec::new();
    brokkr_core::export::stl::write(&mesh, &mut stl).expect("writing the fixture STL");
    let mut obj = Vec::new();
    brokkr_core::export::obj::write(&mesh, &mut obj).expect("writing the fixture OBJ");
    let mut threemf = Vec::new();
    brokkr_core::export::threemf::write(&mesh, &mut threemf).expect("writing the fixture 3MF");
    println!(
        "  encoded: STL {:.1} MB, OBJ {:.1} MB, 3MF {:.1} MB\n",
        stl.len() as f64 / (1024.0 * 1024.0),
        obj.len() as f64 / (1024.0 * 1024.0),
        threemf.len() as f64 / (1024.0 * 1024.0),
    );

    println!("reading");
    let (read_stl, taken) = timed(|| brokkr_core::import::stl::read(&stl));
    let read_stl = read_stl.expect("the fixture STL should read");
    report("STL", taken, READ_CEILING, &mut passed);

    let (read_obj, taken) = timed(|| brokkr_core::import::obj::read(&obj));
    let read_obj = read_obj.expect("the fixture OBJ should read");
    report("OBJ", taken, READ_CEILING, &mut passed);

    let (read_3mf, taken) = timed(|| brokkr_core::import::threemf::read(&threemf));
    let read_3mf = read_3mf.expect("the fixture 3MF should read");
    report("3MF", taken, READ_CEILING, &mut passed);

    // A reader that silently dropped most of a file would otherwise look fast.
    for (name, parsed) in [("STL", &read_stl), ("OBJ", &read_obj), ("3MF", &read_3mf)] {
        assert_eq!(
            parsed.triangles.len(),
            mesh.triangles.len(),
            "{name} came back with a different triangle count, so its timing means nothing"
        );
    }

    println!("\nvoxelising, by voxel size");
    for voxel_size in [0.5f32, 0.25, 0.125] {
        let options = VoxeliseOptions {
            voxel_size,
            centre: false,
            refit_if_implausible: false,
            fill_sealed_cavities: true,
            repair_broken_scan_lines: true,
            coarsen_to_fit: false,
            refine_to_resolve: false,
            already_reserved: 0.0,
        };
        let (built, taken) = timed(|| voxelise(&read_stl, &options));
        let (volume, voxel_report) = built.expect("the fixture should voxelise");
        let stats = volume.stats();
        let resident_mb = stats.resident_bytes as f64 / (1024.0 * 1024.0);

        let over_time = taken > VOXELISE_CEILING;
        let over_memory = resident_mb > RESIDENT_CEILING_MB;
        if over_time || over_memory {
            passed = false;
        }
        println!(
            "  {voxel_size:>5.3} mm  {:>8.1} ms  {:>7.1} MB  {} dense + {} uniform bricks  {}",
            taken.as_secs_f64() * 1000.0,
            resident_mb,
            voxel_report.dense_bricks,
            voxel_report.uniform_bricks,
            if over_time || over_memory { "OVER" } else { "ok" }
        );

        // The preflight's estimate is what refuses an import before it is
        // attempted, so print it beside what actually happened. Drift here is
        // the early warning that the refusal threshold is becoming wrong.
        assert!(
            voxel_report.uniform_bricks > 0,
            "the sphere came back hollow at {voxel_size} mm, so this timing is not of real work"
        );
        assert!(
            voxel_report.lost_triangles * 20 < voxel_report.triangles,
            "more than 5% of the surface was lost at {voxel_size} mm"
        );
    }

    println!("\nround trip");
    let options = VoxeliseOptions {
        voxel_size: 0.25,
        centre: false,
        refit_if_implausible: false,
        fill_sealed_cavities: true,
        repair_broken_scan_lines: true,
        coarsen_to_fit: false,
        refine_to_resolve: false,
        already_reserved: 0.0,
    };
    let (rebuilt, _) = voxelise(&read_stl, &options).expect("the fixture should voxelise");
    let (again, again_report) = rebuilt.export_mesh();
    println!(
        "  re-exported {} triangles, printable: {}",
        again.triangles.len(),
        again_report.is_printable()
    );
    assert!(
        again_report.is_printable(),
        "an imported model did not re-export cleanly: {}",
        again_report.summary()
    );

    if passed {
        println!("\nall import ceilings met");
    } else {
        eprintln!("\nCEILING EXCEEDED: see the lines marked OVER above");
        std::process::exit(1);
    }
}
