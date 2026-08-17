// SPDX-License-Identifier: AGPL-3.0-or-later

//! Writing the three export formats to real files.
//!
//! The format modules check the bytes they produce. This checks the step after
//! that: that a sculpt can be written to disk, and that what lands there is the
//! size and shape expected. It also leaves the files behind so they can be
//! opened in a slicer or picked apart by another tool, which is the only way to
//! find out whether a reader that is not ours accepts them.

use std::fs;
use std::path::PathBuf;

use brokkr_core::export::{obj, stl, threemf};
use brokkr_core::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp, Volume};
use glam::Vec3;

/// A sculpted ball, not just a seeded one, so the exported geometry has been
/// through the brushes.
fn sculpted() -> Volume {
    let mut volume = Volume::new(0.5);
    volume.seed_sphere(Vec3::ZERO, 15.0);

    let mut scratch = BrushScratch::new();
    for (kind, at) in [
        (BrushKind::Draw, Vec3::new(15.0, 0.0, 0.0)),
        (BrushKind::Clay, Vec3::new(0.0, 15.0, 0.0)),
        (BrushKind::Inflate, Vec3::new(0.0, 0.0, 15.0)),
        (BrushKind::Flatten, Vec3::new(-15.0, 0.0, 0.0)),
    ] {
        let brush = Brush { kind, radius: 5.0, strength: 0.7, ..Brush::default() };
        let normal = volume.gradient_world(at);
        brush.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
    }
    volume
}

fn output_directory() -> PathBuf {
    let directory = std::env::temp_dir().join("brokkr-export-test");
    fs::create_dir_all(&directory).expect("could not make an output directory");
    directory
}

#[test]
fn all_three_formats_write_to_disk() {
    let volume = sculpted();
    let (mesh, report) = volume.export_mesh();

    assert!(
        report.is_printable(),
        "refusing to write a model that would not print: {}",
        report.summary()
    );
    println!("exporting {}", report.summary());

    let directory = output_directory();

    let stl_path = directory.join("sculpt.stl");
    let mut file = fs::File::create(&stl_path).expect("could not create the STL");
    stl::write(&mesh, &mut file).expect("could not write the STL");
    drop(file);
    let stl_size = fs::metadata(&stl_path).expect("the STL exists").len() as usize;
    assert_eq!(stl_size, stl::size_of(&mesh), "the STL on disk is not the expected size");

    let obj_path = directory.join("sculpt.obj");
    let mut file = fs::File::create(&obj_path).expect("could not create the OBJ");
    obj::write(&mesh, &mut file).expect("could not write the OBJ");
    drop(file);
    let obj_text = fs::read_to_string(&obj_path).expect("the OBJ is text");
    assert_eq!(
        obj_text.lines().filter(|line| line.starts_with("f ")).count(),
        mesh.triangles.len()
    );

    let threemf_path = directory.join("sculpt.3mf");
    let mut file = fs::File::create(&threemf_path).expect("could not create the 3MF");
    threemf::write(&mesh, &mut file).expect("could not write the 3MF");
    drop(file);
    let threemf_bytes = fs::read(&threemf_path).expect("the 3MF exists");
    assert!(threemf_bytes.starts_with(b"PK\x03\x04"), "the 3MF is not a ZIP");
    // The end of central directory record has to be findable from the end,
    // which is how every ZIP reader starts.
    assert!(
        threemf_bytes.windows(4).any(|window| window == b"PK\x05\x06"),
        "the 3MF has no end of central directory record"
    );

    println!("wrote:");
    for path in [&stl_path, &obj_path, &threemf_path] {
        println!("  {} ({} bytes)", path.display(), fs::metadata(path).unwrap().len());
    }
}

#[test]
fn an_unprintable_mesh_is_something_a_caller_can_refuse() {
    // The report exists so that writing a broken file is a decision rather than
    // an accident.
    let volume = sculpted();
    let (mut mesh, report) = volume.export_mesh();
    assert!(report.is_printable());

    mesh.triangles.truncate(mesh.triangles.len() - 1);
    assert!(!mesh.validate().is_printable());
}
