// SPDX-License-Identifier: AGPL-3.0-only

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

use brokkr_core::{BrickCoord, BrickMesh, ClipPlane, Volume};
use brokkr_gpu::{
    Frustum, PixelRect, SculptRenderer, SlotKey, THE_ONLY_BODY, THUMBNAIL_SIZE, Uniforms,
};
use glam::{Mat4, Vec3};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// What one thumbnail of the largest possible body is allowed to cost.
///
/// 8 rather than 15, because a thumbnail render rides on top of an ordinary
/// frame inside a 16 ms budget, and a 15 ms threshold would permit a picture
/// that consumed the whole frame on its own.
const THUMBNAIL_BUDGET_MS: f64 = 8.0;

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

/// Add a sphere to whatever is already there.
///
/// **Not `seed_sphere`, which OVERWRITES**: seeding clears every brick its box
/// touches before writing, so seeding a small sphere onto a large one carves a
/// moat around it instead of adding a lump to it.
fn union_sphere(volume: &mut Volume, centre: Vec3, radius: f32) {
    let voxel_size = volume.voxel_size();
    let band = brokkr_core::NARROW_BAND * voxel_size;
    let (lo, hi) =
        volume.voxel_bounds(centre - (radius + band * 2.0), centre + (radius + band * 2.0));
    volume.edit_voxels(lo, hi, |_, position, value| {
        let outside = (position.distance(centre) - radius) / voxel_size;
        value.min(outside).clamp(brokkr_core::INSIDE, brokkr_core::OUTSIDE)
    });
}

/// **Thirty cuts in a row, watching the mesh pool's bump pointer.**
///
/// This is the one failure mode the cut tool's design names as a live risk, and
/// it is invisible from `brokkr-core`: the pool's `BlockAllocator` never splits
/// or merges blocks, a freed block is reusable only by a request rounding to
/// the same granule count, and it is the bump `watermark` -- not `live` -- that
/// fails an allocation. A cut only ever SHRINKS the meshes it touches, so every
/// changed brick can orphan a block in a large granule class and take a fresh
/// one off the bump. `MeshPool::reset` is the only cure and, in the whole
/// application, is reached from `rebuild_everything` and from nothing else --
/// no editing operation calls it.
///
/// So: trim thirty spurs, re-uploading what changed each time exactly as the
/// application does, and report whether the watermark climbs while `live`
/// stays flat. A watermark that ends near `live` means the allocator is
/// reusing; one that ends far above it is the shape of `MESH POOL FULL` with
/// most of the pool empty.
fn measure_repeated_cuts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut SculptRenderer,
) {
    const SPURS: usize = 30;
    const RADIUS: f32 = 30.0;
    // Coarser than the render rows above: this measures allocator behaviour
    // over thirty edits, and the thirty edits are the expensive part.
    const VOXEL: f32 = 0.25;

    let mut volume = Volume::new(VOXEL);
    volume.seed_sphere(Vec3::ZERO, RADIUS);
    let mut directions = Vec::with_capacity(SPURS);
    for index in 0..SPURS {
        let angle = index as f32 / SPURS as f32 * std::f32::consts::TAU;
        let tilt = (index as f32 * 0.37).sin() * 0.7;
        let direction =
            Vec3::new(angle.cos() * tilt.cos(), tilt.sin(), angle.sin() * tilt.cos()).normalize();
        union_sphere(&mut volume, direction * RADIUS, RADIUS * 0.12);
        directions.push(direction);
    }
    volume.mark_everything_dirty();

    let mut coords: Vec<BrickCoord> = Vec::new();
    let mut meshes: Vec<BrickMesh> = Vec::new();
    let publish = |renderer: &mut SculptRenderer,
                   volume: &mut Volume,
                   coords: &mut Vec<BrickCoord>,
                   meshes: &mut Vec<BrickMesh>| {
        coords.clear();
        volume.take_dirty(coords);
        while meshes.len() < coords.len() {
            meshes.push(BrickMesh::default());
        }
        volume.mesh_bricks(coords, &mut meshes[..coords.len()]);
        for (coord, mesh) in coords.iter().zip(meshes.iter()) {
            renderer.upload_brick(
                device,
                queue,
                SlotKey { body: THE_ONLY_BODY, coord: *coord },
                mesh,
            );
        }
        coords.len()
    };

    publish(renderer, &mut volume, &mut coords, &mut meshes);
    let seeded = renderer.stats();
    println!(
        "
thirty cuts in a row, mesh pool"
    );
    println!(
        "  after seeding:  {:>9} live, {:>9} watermark",
        seeded.vertices, seeded.vertices_watermark
    );

    let mut republished = 0usize;
    for direction in &directions {
        let plane =
            ClipPlane::new(*direction * (RADIUS * 0.94), *direction).expect("a unit normal");
        // Bounded, as the shaped cut is: a cap a little BEYOND the spur, so
        // this measures thirty trims rather than thirty cuts through the model.
        // The two normals face each other and the region removed is the shell
        // between them -- putting the cap INSIDE the first plane instead makes
        // the intersection empty and the whole loop a no-op, which reports a
        // perfectly flat watermark for entirely the wrong reason.
        let cap = ClipPlane::new(*direction * (RADIUS * 1.25), -*direction).expect("a unit normal");
        volume.clip_convex(&[plane, cap]);
        republished += publish(renderer, &mut volume, &mut coords, &mut meshes);
    }

    let after = renderer.stats();
    println!(
        "  after {SPURS} cuts: {:>9} live, {:>9} watermark   ({republished} brick re-uploads)",
        after.vertices, after.vertices_watermark
    );
    let headroom = after.vertices_watermark as f64 / after.vertices.max(1) as f64;
    println!(
        "  watermark is {headroom:.2}x live, and the pool fails at {:.0} MB of watermark",
        after.vertex_capacity as f64 * 24.0 / (1024.0 * 1024.0)
    );
    if after.overflowed > 0 {
        eprintln!("  MESH POOL OVERFLOWED during thirty ordinary cuts");
    }
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
        // One body, so one bucket per buffer pair. Increment 2 is what makes
        // the body half of the key vary.
        let key = SlotKey { body: THE_ONLY_BODY, coord: *coord };
        renderer.upload_brick(&device, &queue, key, mesh);
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
            // Tinted, at full strength, because the whole point of the row is
            // what a frame costs in the state the application ships in -- and
            // the tint is on by default. A bench that switched it off would be
            // measuring the branch it is meant to be measuring the cost of.
            mask_inverted: 0,
            mask_tint: 1.0,
            padding: [0; 1],
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

    all_passed &= thumbnail_row(&device, &queue, &renderer, &volume, stats.bricks);

    // Its own model and its own pool state, so it cannot be read as a comment
    // on the rows above. Report only: this is the first measurement of the
    // allocator under repeated editing, and a pass mark drawn on the same day
    // as the first measurement is not a budget.
    let mut cut_renderer = SculptRenderer::new(&device, &queue, TARGET_FORMAT);
    cut_renderer.resize(&device, WIDTH, HEIGHT);
    measure_repeated_cuts(&device, &queue, &mut cut_renderer);

    if !all_passed {
        eprintln!("\nRENDER BUDGET EXCEEDED");
        std::process::exit(1);
    }
    println!("\nall render budgets met");
}

/// Bricks in the model this application is built for. See `handoff.md`.
const DRAGON_BRICKS: usize = 45_567;

/// What one live thumbnail costs, and whether the feature is affordable at all.
///
/// **This is a go/no-go, not a regression check, and it is deliberately a
/// comparison rather than an absolute.** The threshold is 8 ms rather than the
/// frame's 16, because a thumbnail render rides ON TOP of an ordinary frame: a
/// 15 ms threshold would permit a picture that consumed the whole budget on its
/// own and still called itself a pass.
///
/// The extrapolation is the number that decides it. The bench model is a
/// sculpted sphere of a few thousand bricks; the model the mesh pool is sized
/// for is 45,567, which is 8.4x the draw calls at the same 84 x 84 target.
/// **Two things that number does NOT cover, and they belong beside it:** an
/// imported scan carries several times more geometry per brick than a sculpted
/// sphere does, and this machine's GPU is not the slowest one the application
/// will run on. If the extrapolated figure is over 8 ms, the answer is the
/// panel's `Thumbnails` switch, which exists for exactly this.
fn thumbnail_row(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &SculptRenderer,
    volume: &Volume,
    bricks: usize,
) -> bool {
    let Some(bounds) = volume.world_bounds() else {
        eprintln!("  the bench model has no bricks, so there is no thumbnail to time");
        return false;
    };

    let mut samples: Vec<Duration> = Vec::with_capacity(FRAMES);
    for frame in 0..FRAMES {
        let started = Instant::now();
        renderer.render_thumbnail(device, queue, 0, THE_ONLY_BODY, bounds);
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");
        if frame > 0 {
            samples.push(started.elapsed());
        }
    }
    samples.sort_unstable();

    let median = millis(samples[samples.len() / 2]);
    let worst = millis(*samples.last().expect("at least one frame"));
    let scaled = median * DRAGON_BRICKS as f64 / bricks.max(1) as f64;
    let passed = scaled <= THUMBNAIL_BUDGET_MS;

    println!(
        "\n  thumbnail {THUMBNAIL_SIZE} x {THUMBNAIL_SIZE}   median {median:>6.2} ms   \
         worst {worst:>6.2} ms   over {bricks} bricks"
    );
    println!(
        "  extrapolated to {DRAGON_BRICKS} bricks: {scaled:>6.2} ms   budget \
         {THUMBNAIL_BUDGET_MS:.1} ms  {}",
        if passed { "pass" } else { "OVER" }
    );
    if !passed {
        eprintln!(
            "\n  A thumbnail of the largest model this pool is sized for would cost {scaled:.1} ms \
             on top of an ordinary frame. Live pictures are not affordable here: leave the \
             placeholder, or default the panel's Thumbnails switch to off."
        );
    }
    println!(
        "  Sculpted geometry on this machine's GPU. An imported scan carries several times more\n\
         geometry per brick, and this is not the slowest GPU the application runs on."
    );
    passed
}
