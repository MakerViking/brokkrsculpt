// SPDX-License-Identifier: AGPL-3.0-only

//! Diagnose what a real file does on the way in.
//!
//! Not a gate and not part of the suite: an investigation tool for when an
//! import comes out looking wrong. It answers the question the status line
//! cannot, which is **whether a defect arrived in the file or was made here**.
//! That distinction decides everything about the fix, and guessing it has
//! already cost this project a session.
//!
//! ```fish
//! env BROKKR_DIAGNOSE=/path/to/model.obj \
//!   cargo test -p brokkr-core --release --test diagnose_import -- --ignored --nocapture
//! ```
//!
//! Set `BROKKR_DIAGNOSE_VOXEL` to try another voxel size; it defaults to the
//! 0.25 mm the application imports at.

use brokkr_core::export::ExportMesh;
use brokkr_core::voxelise::{VoxeliseOptions, voxelise};

fn read(path: &std::path::Path) -> ExportMesh {
    let bytes = std::fs::read(path).expect("reading the model");
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let mesh = match ext.as_str() {
        "obj" => brokkr_core::import::obj::read(&bytes),
        "stl" => brokkr_core::import::stl::read(&bytes),
        "3mf" => brokkr_core::import::threemf::read(&bytes),
        other => panic!("no reader for .{other}"),
    };
    mesh.expect("the file should read")
}

/// Longest side of the mesh's bounding box, and the box itself.
fn extent(mesh: &ExportMesh) -> (glam::Vec3, glam::Vec3) {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for p in &mesh.positions {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

#[test]
#[ignore = "an investigation tool; needs BROKKR_DIAGNOSE=<path>"]
fn diagnose() {
    let Ok(path) = std::env::var("BROKKR_DIAGNOSE") else {
        panic!("set BROKKR_DIAGNOSE to the model to look at");
    };
    let path = std::path::PathBuf::from(path);
    let voxel: f32 =
        std::env::var("BROKKR_DIAGNOSE_VOXEL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);

    println!("\n=== {} ===", path.display());

    let started = std::time::Instant::now();
    let mesh = read(&path);
    println!("read in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0);

    // --- what arrived -------------------------------------------------------
    //
    // The reader welds, so this is real topology rather than an artefact of
    // however the file happened to index its corners.
    let source = mesh.validate();
    let (min, max) = extent(&mesh);
    let size = max - min;
    println!("\nTHE FILE ITSELF");
    println!("  {}", source.summary());
    println!("  non-manifold edges: {}", source.non_manifold_edges);
    // Scale-dependent, and the check that produces it compares a cross product
    // against f32::EPSILON. A model authored at 0.1 mm has triangles whose
    // cross products are ~1e-8 and every one of them counts as "zero area",
    // which says nothing about the model. Read it only alongside the box.
    println!(
        "  zero-area triangles: {} (meaningless if the box below is tiny)",
        source.zero_area_triangles
    );
    println!(
        "  bounding box: {:.3} x {:.3} x {:.3} mm (longest {:.3})",
        size.x,
        size.y,
        size.z,
        size.max_element()
    );
    println!(
        "  -> the file is {}",
        if source.boundary_edges == 0 {
            "CLOSED. Any hole below was made here.".to_string()
        } else {
            format!("ALREADY OPEN: {} boundary edges arrived in it.", source.boundary_edges)
        }
    );

    // --- what the import does with it ---------------------------------------
    //
    // `VoxeliseOptions::at` is exactly what the application uses, refit and
    // centring included. A diagnostic that quietly used different options would
    // be answering a question nobody asked -- the first version of this did,
    // and reported a refusal the application never hits.
    for (label, cavities, repair) in
        [("as the application imports it", true, true), ("with both repairs OFF", false, false)]
    {
        let options = VoxeliseOptions {
            fill_sealed_cavities: cavities,
            repair_broken_scan_lines: repair,
            ..VoxeliseOptions::at(voxel)
        };
        let started = std::time::Instant::now();
        let Ok((volume, report)) = voxelise(&mesh, &options) else {
            println!("\nAT {voxel} mm, {label}: REFUSED");
            continue;
        };
        let took = started.elapsed().as_secs_f64() * 1000.0;

        println!("\nAT {voxel} mm, {label} ({took:.0} ms)");
        println!("  {}", report.summary());
        println!(
            "  surface lost to thin walls: {} of {} triangles ({:.2}%)",
            report.lost_triangles,
            report.triangles,
            100.0 * report.lost_triangles as f64 / report.triangles.max(1) as f64
        );
        println!("  scan lines repaired:      {}", report.repaired_scan_lines);
        println!("  filament voxels erased:   {}", report.erased_filament_voxels);
        println!("  isolated sign flips:      {}", report.isolated_sign_flips);
        println!("  cavity voxels filled:     {}", report.filled_voxels);
        println!(
            "  bricks: {} dense + {} uniform  <- uniform 0 means NOTHING is solid inside",
            report.dense_bricks, report.uniform_bricks
        );

        let (out, out_report) = volume.export_mesh();
        println!("  back out: {}", out_report.summary());
        println!(
            "  -> {}",
            if out_report.boundary_edges == 0 {
                "closed, printable".to_string()
            } else {
                format!("{} holes in the result", out_report.boundary_edges)
            }
        );
        drop(out);

        // Where do the visible streaks come from?
        //
        // A strand is solid material with almost nothing solid beside it. The
        // question that decides the fix is whether those strands are SATURATED:
        //
        //   saturated  -> no source triangle within the narrow band, so the
        //                 material was invented here and the filament eraser
        //                 is the right place to deal with it.
        //   not        -> a real surface is right there, so the spike is IN THE
        //                 FILE and no amount of sweeping the field will remove
        //                 it without also removing genuine thin features.
        //
        // Getting this backwards means tuning a repair pass that was never
        // going to help.
        if std::env::var("BROKKR_DIAGNOSE_STRANDS").is_ok() {
            // Which AXIS does the surviving thin material run along?
            //
            // This is the question that identifies a sweep leak, and it is the
            // one to ask first. `voxelise` signs voxels by a winding number
            // swept along X, and its own header warns that one hole "inverts a
            // whole scan line, or leaks". So thin material that runs
            // overwhelmingly along X is material the SWEEP invented, however
            // clean the source is -- and the saturation test will not see it,
            // because a leaked line passing near the model stays inside the
            // narrow band of real triangles for most of its length.
            //
            // Runs along Y or Z in similar numbers would mean the opposite:
            // real thin geometry, which has no reason to prefer an axis.
            let (mut along_x, mut along_y, mut along_z) = (0usize, 0usize, 0usize);
            let (mut saturated, mut near_surface) = (0usize, 0usize);
            for coord in volume.brick_coords().collect::<Vec<_>>() {
                let origin = coord.origin();
                for dz in 0..32 {
                    for dy in 0..32 {
                        for dx in 0..32 {
                            let v = origin + glam::IVec3::new(dx, dy, dz);
                            let d = volume.sample_voxel(v);
                            if d >= 0.0 {
                                continue;
                            }
                            let solid = |s: glam::IVec3| volume.sample_voxel(v + s) < 0.0;
                            let x = solid(glam::IVec3::X) || solid(glam::IVec3::NEG_X);
                            let y = solid(glam::IVec3::Y) || solid(glam::IVec3::NEG_Y);
                            let z = solid(glam::IVec3::Z) || solid(glam::IVec3::NEG_Z);
                            // A strand: continues along exactly one axis and
                            // has nothing beside it on the other two.
                            match (x, y, z) {
                                (true, false, false) => along_x += 1,
                                (false, true, false) => along_y += 1,
                                (false, false, true) => along_z += 1,
                                _ => continue,
                            }
                            if d <= -3.0 {
                                saturated += 1;
                            } else {
                                near_surface += 1;
                            }
                        }
                    }
                }
            }
            let total = along_x + along_y + along_z;
            println!("  STRAND VOXELS by the axis they run along:");
            println!("    along X (the sweep axis): {along_x}");
            println!("    along Y:                  {along_y}");
            println!("    along Z:                  {along_z}");
            println!("    of those: {saturated} saturated, {near_surface} within the band");
            if total > 0 && along_x > 4 * (along_y + along_z).max(1) {
                println!("  -> SWEEP LEAK. The sign sweep invented these; the source is innocent.");
            } else if total > 0 {
                println!("  -> no axis preference, so this is real thin geometry from the file.");
            }
        }
    }
}
