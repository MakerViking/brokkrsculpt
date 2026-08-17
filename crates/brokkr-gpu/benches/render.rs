// SPDX-License-Identifier: AGPL-3.0-or-later

//! What it costs to draw a model at M2 scale.
//!
//! The engine side of M2 is measured by `brokkr-core`'s scale bench. This is
//! the other half: a model of several thousand bricks is several thousand draw
//! calls, and encoding those is CPU work that comes straight out of the frame
//! budget. It reports the cost from a few camera positions, because how much
//! culling saves depends entirely on how much of the model is on screen.
//!
//! Run with `cargo bench -p brokkr-gpu --bench render`.

use std::time::{Duration, Instant};

use brokkr_core::{BrickCoord, BrickMesh, Volume};
use brokkr_gpu::{Frustum, PixelRect, SculptRenderer, Uniforms};
use glam::{Mat4, Vec3};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The model, in millimetres.
const MODEL_RADIUS: f32 = 30.0;
/// Chosen to land above the ten million triangles M2 targets.
const VOXEL_SIZE: f32 = 0.055;

/// Frames timed per camera position. The first is discarded: it pays for
/// pipeline warm up and first touch of the buffers.
const FRAMES: usize = 24;

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn view_projection(distance: f32) -> Mat4 {
    let eye = Vec3::new(0.45, 0.35, 1.0).normalize() * distance;
    let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
    let projection = glam::camera::rh::proj::directx::perspective(
        45f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        MODEL_RADIUS * 0.001,
        distance + MODEL_RADIUS * 4.0,
    );
    projection * view
}

fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) else {
        eprintln!("no usable wgpu adapter, skipping the render bench");
        return;
    };
    let Ok((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
    else {
        eprintln!("could not open a device, skipping the render bench");
        return;
    };

    println!("BrokkrSculpt render bench on {}", adapter.get_info().name);
    println!("{WIDTH} by {HEIGHT}, {MODEL_RADIUS} mm sphere at a {VOXEL_SIZE} mm voxel\n");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render bench target"),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = SculptRenderer::new(&device, &queue, TARGET_FORMAT);
    renderer.resize(&device, WIDTH, HEIGHT);

    // Build and upload the model.
    let build_start = Instant::now();
    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    volume.mark_everything_dirty();
    let mut coords: Vec<BrickCoord> = Vec::new();
    volume.take_dirty(&mut coords);
    let mut meshes = vec![BrickMesh::default(); coords.len()];
    volume.mesh_bricks(&coords, &mut meshes);
    let mesh_ms = millis(build_start.elapsed());

    let upload_start = Instant::now();
    for (coord, mesh) in coords.iter().zip(meshes.iter()) {
        renderer.upload_brick(&queue, *coord, mesh);
    }
    let upload_ms = millis(upload_start.elapsed());

    let stats = renderer.stats();
    println!(
        "  built and meshed in {mesh_ms:.0} ms, uploaded in {upload_ms:.0} ms\n  {} triangles in {} bricks, {} vertices",
        stats.triangles, stats.bricks, stats.vertices
    );
    println!(
        "  vertices: {:.0} MB in use, {:.0} MB reserved of {:.0} MB   ({:.0}% padding)",
        stats.vertices as f64 * 24.0 / (1024.0 * 1024.0),
        stats.vertices_reserved as f64 * 24.0 / (1024.0 * 1024.0),
        stats.vertex_capacity as f64 * 24.0 / (1024.0 * 1024.0),
        (stats.vertices_reserved as f64 / stats.vertices.max(1) as f64 - 1.0) * 100.0,
    );
    println!(
        "  indices:  {:.0} MB in use, {:.0} MB reserved of {:.0} MB   ({:.0}% padding)",
        stats.indices as f64 * 4.0 / (1024.0 * 1024.0),
        stats.indices_reserved as f64 * 4.0 / (1024.0 * 1024.0),
        stats.index_capacity as f64 * 4.0 / (1024.0 * 1024.0),
        (stats.indices_reserved as f64 / stats.indices.max(1) as f64 - 1.0) * 100.0,
    );
    if stats.overflowed > 0 {
        eprintln!(
            "\n  MESH POOL OVERFLOWED on {} bricks: the pool is too small for this model, so \
             every number below is measuring an incomplete scene.",
            stats.overflowed
        );
    }
    println!();

    // The frame budget at 60 fps.
    let budget = Duration::from_micros(16_000);
    let mut all_passed = true;

    for (label, distance) in [
        ("whole model in view", MODEL_RADIUS * 3.0),
        ("filling the view", MODEL_RADIUS * 1.6),
        ("close up detail", MODEL_RADIUS * 0.6),
    ] {
        let matrix = view_projection(distance);
        let frustum = Frustum::from_view_projection(matrix);
        let uniforms = Uniforms {
            view_projection: matrix.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            srgb_target: 1,
            padding: [0; 3],
        };
        renderer.write_uniforms(&queue, &uniforms);

        let mut samples: Vec<Duration> = Vec::with_capacity(FRAMES);
        for frame in 0..FRAMES {
            let started = Instant::now();
            let mut encoder = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
            renderer.render(
                &mut encoder,
                &view,
                PixelRect { x: 0, y: 0, width: WIDTH, height: HEIGHT },
                &frustum,
            );
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");
            if frame > 0 {
                samples.push(started.elapsed());
            }
        }
        samples.sort_unstable();

        let median = samples[samples.len() / 2];
        let worst = *samples.last().expect("at least one frame");
        let stats = renderer.stats();
        let passed = median <= budget;
        all_passed &= passed;

        println!(
            "  {label:<22} {} drawn, {} culled   median {:>6.2} ms   worst {:>6.2} ms   budget 16.0 ms  {}",
            stats.drawn,
            stats.culled,
            millis(median),
            millis(worst),
            if passed { "pass" } else { "OVER" }
        );
    }

    println!(
        "\n  Culling is per brick against the view frustum. The first row is the worst case for\n\
         draw call count, because nothing is off screen to skip."
    );

    if !all_passed {
        eprintln!("\nRENDER BUDGET EXCEEDED");
        std::process::exit(1);
    }
    println!("\nall render budgets met");
}
