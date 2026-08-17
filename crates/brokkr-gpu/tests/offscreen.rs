// SPDX-License-Identifier: AGPL-3.0-or-later

//! Offscreen end to end test of the sculpt loop.
//!
//! This runs the whole path with no window: seed a sphere, mesh every brick,
//! upload to the pool, render to a texture, read the pixels back and check
//! them. Then it sculpts and checks the picture actually changed where the
//! brush went and nowhere else.
//!
//! It exists because the parts that break here are the ones a unit test cannot
//! see: a projection with the wrong depth convention, a uniform buffer whose
//! layout disagrees with the shader, a mesh uploaded at the wrong offset. All
//! three produce code that compiles and tests that pass, and a blank window.
//!
//! Set `BROKKR_DUMP_FRAMES=<directory>` to also write the frames out as binary
//! PPM for a human to look at.

use brokkr_core::{
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, MeshScratch, Pattern,
    PatternKind, Stamp, Volume,
};
use brokkr_gpu::{Frustum, OverlayBatch, PixelRect, SculptRenderer, Uniforms};
use glam::{IVec3, Vec3};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 360;
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The model, in millimetres, matching what the application seeds.
const MODEL_RADIUS: f32 = 30.0;
const VOXEL_SIZE: f32 = 0.5;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row_bytes: u32,
    /// The matrix the frames are rendered with, so culling and the shader agree.
    view_projection: glam::Mat4,
}

impl Harness {
    /// Returns `None` when the machine has no usable adapter, so the test can
    /// skip instead of failing for a reason that is not about this code.
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        eprintln!("offscreen test adapter: {:?}", adapter.get_info());

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        // Buffer to texture copies need rows aligned to 256 bytes.
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row_bytes = (WIDTH * 4).div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen readback"),
            size: (padded_row_bytes * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Some(Self {
            device,
            queue,
            target,
            view,
            readback,
            padded_row_bytes,
            view_projection: view_projection(MODEL_RADIUS * 3.0),
        })
    }

    /// Clear to a known background, draw the sculpt, then read the pixels back
    /// as tightly packed RGBA.
    ///
    /// Culling uses the same matrix the shader gets, which is what the
    /// application does too. A frustum that disagreed with the projection would
    /// show up here as missing geometry.
    fn frame(&self, renderer: &SculptRenderer) -> Vec<u8> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("offscreen") });

        // The renderer loads rather than clears, because in the application the
        // UI has already drawn underneath. Stand in for that here.
        encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("offscreen clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            })
            .forget_lifetime();

        renderer.render(
            &mut encoder,
            &self.view,
            PixelRect { x: 0, y: 0, width: WIDTH, height: HEIGHT },
            &Frustum::from_view_projection(self.view_projection),
        );

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row_bytes),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        );

        self.queue.submit([encoder.finish()]);

        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("readback map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
        for row in 0..HEIGHT {
            let start = (row * self.padded_row_bytes) as usize;
            pixels.extend_from_slice(&mapped[start..start + (WIDTH * 4) as usize]);
        }
        drop(mapped);
        self.readback.unmap();
        pixels
    }
}

/// Pixels that are not the black background, and their bounding box.
fn coverage(pixels: &[u8]) -> (usize, [u32; 4]) {
    let mut count = 0;
    let mut bounds = [u32::MAX, u32::MAX, 0, 0];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let index = ((y * WIDTH + x) * 4) as usize;
            let lit = pixels[index] as u32 + pixels[index + 1] as u32 + pixels[index + 2] as u32;
            if lit > 24 {
                count += 1;
                bounds[0] = bounds[0].min(x);
                bounds[1] = bounds[1].min(y);
                bounds[2] = bounds[2].max(x);
                bounds[3] = bounds[3].max(y);
            }
        }
    }
    (count, bounds)
}

fn dump(name: &str, pixels: &[u8]) {
    let Ok(directory) = std::env::var("BROKKR_DUMP_FRAMES") else {
        return;
    };
    let mut ppm = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
    for pixel in pixels.chunks_exact(4) {
        ppm.extend_from_slice(&pixel[..3]);
    }
    let path = std::path::Path::new(&directory).join(format!("{name}.ppm"));
    std::fs::write(&path, ppm).expect("could not write frame dump");
    eprintln!("wrote {}", path.display());
}

/// Mesh every brick that can carry geometry and hand it to the renderer.
fn upload_all(renderer: &mut SculptRenderer, queue: &wgpu::Queue, volume: &mut Volume) -> usize {
    volume.mark_everything_dirty();
    upload_dirty(renderer, queue, volume)
}

/// Mesh only what the volume says is dirty. Returns how many bricks that was.
fn upload_dirty(renderer: &mut SculptRenderer, queue: &wgpu::Queue, volume: &mut Volume) -> usize {
    let mut scratch = MeshScratch::new();
    let mut mesh = BrickMesh::default();
    let mut dirty: Vec<BrickCoord> = Vec::new();
    volume.take_dirty(&mut dirty);
    for &coord in &dirty {
        volume.mesh_brick(coord, &mut scratch, &mut mesh);
        renderer.upload_brick(queue, coord, &mesh);
    }
    dirty.len()
}

/// A camera looking at the origin from a fixed direction, matching the
/// conventions the application uses.
fn view_matrix(distance: f32) -> glam::Mat4 {
    let eye = Vec3::new(0.45, 0.35, 1.0).normalize() * distance;
    glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y)
}

fn view_projection(distance: f32) -> glam::Mat4 {
    let projection = glam::camera::rh::proj::directx::perspective(
        45f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        distance * 4.0,
    );
    projection * view_matrix(distance)
}

fn uniforms(distance: f32) -> Uniforms {
    let view = view_matrix(distance);
    Uniforms {
        view_projection: view_projection(distance).to_cols_array_2d(),
        view: view.to_cols_array_2d(),
        // The offscreen target is an sRGB format, so the shader must not encode
        // a second time.
        srgb_target: 1,
        padding: [0; 3],
    }
}

#[test]
fn the_sculpt_loop_renders_and_responds_to_a_stroke() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the offscreen render test");
        return;
    };

    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);

    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    let bricks = upload_all(&mut renderer, &harness.queue, &mut volume);
    assert!(bricks > 50, "a 60 mm sphere should span many bricks, got {bricks}");

    let stats = renderer.stats();
    assert!(stats.triangles > 10_000, "expected a dense sphere, got {} triangles", stats.triangles);
    assert_eq!(stats.overflowed, 0, "the mesh pool overflowed, so the picture is incomplete");

    // Frame the sphere so it fills a good part of the view.
    let distance = MODEL_RADIUS * 3.0;
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let before = harness.frame(&renderer);
    dump("before", &before);

    let (lit, bounds) = coverage(&before);
    let total = (WIDTH * HEIGHT) as usize;

    // A blank frame is the failure this test exists to catch: it is what a bad
    // projection, a mismatched uniform layout or a misplaced vertex offset all
    // look like.
    assert!(
        lit > total / 20,
        "only {lit} of {total} pixels were drawn, so almost nothing reached the screen"
    );
    assert!(
        lit < total * 3 / 4,
        "{lit} of {total} pixels were drawn, so the model fills the frame rather than sitting in it"
    );

    // The sphere is centred on the origin and the camera looks at the origin,
    // so the drawn region must straddle the middle of the frame. A projection
    // with the wrong handedness or depth range puts it somewhere else, or
    // flips it.
    let [min_x, min_y, max_x, max_y] = bounds;
    let centre_x = (min_x + max_x) / 2;
    let centre_y = (min_y + max_y) / 2;
    assert!(
        centre_x.abs_diff(WIDTH / 2) < WIDTH / 10,
        "the model is off centre horizontally: bounds {bounds:?}"
    );
    assert!(
        centre_y.abs_diff(HEIGHT / 2) < HEIGHT / 10,
        "the model is off centre vertically: bounds {bounds:?}"
    );

    // A sphere is as wide as it is tall.
    let drawn_width = max_x - min_x;
    let drawn_height = max_y - min_y;
    assert!(
        drawn_width.abs_diff(drawn_height) < drawn_width / 8,
        "the silhouette is {drawn_width} by {drawn_height}, which is not round"
    );

    // Now sculpt. The stroke is placed on the front of the sphere by hand
    // rather than by raycasting, so this test stays about rendering.
    let brush = Brush { kind: BrushKind::Draw, radius: 9.0, strength: 0.5, ..Brush::default() };
    let mut brush_scratch = BrushScratch::new();
    let front = Vec3::new(0.45, 0.35, 1.0).normalize() * MODEL_RADIUS;
    for step in 0..6 {
        let along = Vec3::new(0.0, 1.0, 0.0) * (step as f32 - 2.5) * 3.0;
        let at = front + along;
        let normal = volume.gradient_world(at);
        brush.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut brush_scratch);
    }

    let dirty = upload_dirty(&mut renderer, &harness.queue, &mut volume);
    assert!(dirty > 0, "the stroke dirtied nothing");
    assert!(
        dirty < bricks / 2,
        "the stroke dirtied {dirty} of {bricks} bricks, which is not proportional to the brush"
    );

    let after = harness.frame(&renderer);
    dump("after", &after);

    let changed =
        before.chunks_exact(4).zip(after.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
    assert!(
        changed > 200,
        "only {changed} pixels changed after a stroke, so the edit never reached the screen"
    );
    assert!(
        changed < lit,
        "{changed} pixels changed but only {lit} were ever drawn, so the whole model moved rather than the brushed part"
    );

    // Adding clay must not carve a hole in the silhouette.
    let (lit_after, _) = coverage(&after);
    assert!(
        lit_after >= lit,
        "the silhouette shrank from {lit} to {lit_after} pixels after adding material"
    );
}

#[test]
fn an_empty_volume_draws_nothing() {
    // The counterpart to the test above: if a blank frame is the failure signal,
    // then an empty volume must actually produce one, or the coverage check
    // proves nothing.
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the offscreen render test");
        return;
    };

    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(MODEL_RADIUS * 3.0));

    let mut volume = Volume::new(VOXEL_SIZE);
    // Touch a brick without putting a surface in it.
    volume.mark_dirty(BrickCoord(IVec3::ZERO));
    upload_dirty(&mut renderer, &harness.queue, &mut volume);

    let (lit, _) = coverage(&harness.frame(&renderer));
    assert_eq!(lit, 0, "an empty volume drew {lit} pixels");
}

#[test]
fn every_brush_leaves_a_visible_mark() {
    // Field level tests pin down what each brush does to the numbers. This
    // checks the other half: that the result reaches the screen at all, and by
    // a plausible amount. A brush that dirties bricks but produces geometry
    // identical to what was there would pass every unit test in the engine.
    //
    // Run with BROKKR_DUMP_FRAMES set to look at the six results.
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the offscreen render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let front = Vec3::new(0.45, 0.35, 1.0).normalize() * MODEL_RADIUS;

    for kind in BrushKind::ALL {
        let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
        renderer.resize(&harness.device, WIDTH, HEIGHT);
        renderer.write_uniforms(&harness.queue, &uniforms(distance));

        let mut volume = Volume::new(VOXEL_SIZE);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);

        // Give smooth, pinch and flatten something to work on, otherwise they
        // are operating on an already smooth sphere and correctly do very
        // little.
        let mut brush_scratch = BrushScratch::new();
        let ridge = Brush { kind: BrushKind::Draw, radius: 6.0, strength: 0.9, ..Brush::default() };
        for step in -3..=3 {
            let at = front + Vec3::new(0.0, step as f32 * 4.0, 0.0);
            let normal = volume.gradient_world(at);
            ridge.apply(
                &mut volume,
                &Stamp::new(at, normal, BrushDirection::Add),
                &mut brush_scratch,
            );
        }
        upload_all(&mut renderer, &harness.queue, &mut volume);
        let before = harness.frame(&renderer);

        let brush = Brush { kind, radius: 10.0, strength: 0.7, ..Brush::default() };
        for step in -3..=3 {
            let at = front + Vec3::new(0.0, step as f32 * 3.0, 0.0);
            let normal = volume.gradient_world(at);
            for _ in 0..4 {
                brush.apply(
                    &mut volume,
                    &Stamp::new(at, normal, BrushDirection::Add),
                    &mut brush_scratch,
                );
            }
        }

        let dirty = upload_dirty(&mut renderer, &harness.queue, &mut volume);
        assert!(dirty > 0, "{kind} dirtied nothing");

        let after = harness.frame(&renderer);
        dump(&format!("brush-{}", kind.label().to_lowercase()), &after);

        let changed = before
            .chunks_exact(4)
            .zip(after.chunks_exact(4))
            .filter(|(a, b)| a[..3] != b[..3])
            .count();
        assert!(
            changed > 200,
            "{kind} changed only {changed} pixels, so it did not reach the screen"
        );

        let (lit, _) = coverage(&after);
        assert!(lit > 0, "{kind} left nothing on screen at all");
        assert_eq!(renderer.stats().overflowed, 0, "{kind} overflowed the mesh pool");
    }
}

#[test]
fn a_leaned_stroke_looks_different_from_an_upright_one() {
    // Tilt steering is checked numerically in the engine and the application,
    // but brush behaviour in this project has twice looked wrong on screen
    // while every number passed, so it gets a picture too.
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the offscreen render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let front = Vec3::new(0.45, 0.35, 1.0).normalize() * MODEL_RADIUS;
    // Lean across the stroke, which runs up and down the screen.
    let sideways = Vec3::new(0.45, 0.35, 1.0).normalize().cross(Vec3::Y).normalize();

    let mut frames = Vec::new();
    for (label, lean) in [("tilt-upright", Vec3::ZERO), ("tilt-leaned", sideways * 0.9)] {
        let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
        renderer.resize(&harness.device, WIDTH, HEIGHT);
        renderer.write_uniforms(&harness.queue, &uniforms(distance));

        let mut volume = Volume::new(VOXEL_SIZE);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);

        let mut brush_scratch = BrushScratch::new();
        let brush = Brush { kind: BrushKind::Draw, radius: 9.0, strength: 0.9, ..Brush::default() };
        for step in -4..=4 {
            let at = front + Vec3::new(0.0, step as f32 * 2.5, 0.0);
            let normal = brokkr_core::lean_normal(volume.gradient_world(at), lean);
            for _ in 0..6 {
                brush.apply(
                    &mut volume,
                    &Stamp::new(at, normal, BrushDirection::Add),
                    &mut brush_scratch,
                );
            }
        }

        upload_all(&mut renderer, &harness.queue, &mut volume);
        let frame = harness.frame(&renderer);
        dump(label, &frame);
        assert!(coverage(&frame).0 > 0, "{label} rendered nothing");
        frames.push(frame);
    }

    let changed = frames[0]
        .chunks_exact(4)
        .zip(frames[1].chunks_exact(4))
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    assert!(
        changed > 1_000,
        "leaning the pen changed only {changed} pixels, so it is not steering the stroke"
    );
}

/// Every pattern, rendered so a human can look at it.
///
/// This is the whole reason the offscreen harness exists. Twice in this
/// project a brush has been numerically fine, passed every assertion and been
/// visibly wrong, and a pattern is exactly that class of thing: "is this
/// hair or is it corduroy" is not a question any number answers.
///
///     env BROKKR_DUMP_FRAMES=/tmp/patterns cargo test -p brokkr-gpu --test offscreen
///
/// then convert the PPMs and look at them.
#[test]
fn every_pattern_leaves_a_visible_and_distinct_mark() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the pattern render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let front = Vec3::new(0.45, 0.35, 1.0).normalize() * MODEL_RADIUS;
    let mut frames: Vec<(PatternKind, Vec<u8>)> = Vec::new();

    for kind in PatternKind::ALL {
        let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
        renderer.resize(&harness.device, WIDTH, HEIGHT);
        renderer.write_uniforms(&harness.queue, &uniforms(distance));

        let mut volume = Volume::new(VOXEL_SIZE);
        volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
        upload_all(&mut renderer, &harness.queue, &mut volume);
        let before = harness.frame(&renderer);

        let mut brush_scratch = BrushScratch::new();
        let brush = Brush {
            kind: BrushKind::Draw,
            radius: 12.0,
            strength: 0.8,
            pattern: Pattern { kind, scale_mm: 2.5, depth: 1.0 },
            ..Brush::default()
        };

        // Drag across the front of the sphere, so a combing pattern has a
        // direction to comb along.
        for step in -4..=4 {
            let at = front + Vec3::new(step as f32 * 2.5, 0.0, 0.0);
            let at = at.normalize() * MODEL_RADIUS;
            let normal = volume.gradient_world(at);
            for _ in 0..3 {
                brush.apply(
                    &mut volume,
                    &Stamp::new(at, normal, BrushDirection::Add).with_tangent(Vec3::X),
                    &mut brush_scratch,
                );
            }
        }

        let dirty = upload_dirty(&mut renderer, &harness.queue, &mut volume);
        assert!(dirty > 0, "{kind} dirtied nothing");

        let after = harness.frame(&renderer);
        dump(&format!("pattern-{}", kind.label().to_lowercase()), &after);

        if kind != PatternKind::None {
            let changed = before
                .chunks_exact(4)
                .zip(after.chunks_exact(4))
                .filter(|(a, b)| a[..3] != b[..3])
                .count();
            assert!(changed > 200, "{kind} left almost nothing on screen: {changed} pixels");
        }
        frames.push((kind, after));
    }

    // Two patterns that rendered identically would mean one of them is not
    // reaching the field at all -- which is precisely the bug that looks fine
    // in every numeric test.
    for (index, (kind, frame)) in frames.iter().enumerate() {
        for (other_kind, other) in &frames[index + 1..] {
            let differing = frame
                .chunks_exact(4)
                .zip(other.chunks_exact(4))
                .filter(|(a, b)| a[..3] != b[..3])
                .count();
            assert!(
                differing > 100,
                "{kind} and {other_kind} rendered almost identically: {differing} pixels differ"
            );
        }
    }
}

/// The overlay pipeline: does it draw, and does it respect depth?
///
/// Depth and blending are the whole difficulty of the overlay — the geometry is
/// circles and quads. Three things have to hold, and all three are invisible to
/// any test that only checks "some pixels changed":
///
///   * geometry in front of the model draws,
///   * geometry behind the model does not,
///   * a translucent surface tints the model rather than erasing it.
#[test]
fn overlay_geometry_draws_in_front_and_is_hidden_behind() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the overlay render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    upload_all(&mut renderer, &harness.queue, &mut volume);

    let bare = harness.frame(&renderer);

    // The camera sits along this direction, so "toward the eye" is +normal and
    // "behind the model" is -normal.
    let toward_eye = Vec3::new(0.45, 0.35, 1.0).normalize();
    let quad = |centre: Vec3, half: f32, colour: [f32; 4]| {
        // A quad facing the camera, spanned by two axes across the view.
        let right = toward_eye.cross(Vec3::Y).normalize();
        let up = right.cross(toward_eye).normalize();
        let mut batch = OverlayBatch::default();
        batch.push_quad(
            centre - right * half - up * half,
            centre + right * half - up * half,
            centre + right * half + up * half,
            centre - right * half + up * half,
            colour,
        );
        batch
    };

    // --- in front of the model: must draw --------------------------------
    let in_front = quad(toward_eye * (MODEL_RADIUS + 10.0), 12.0, [1.0, 0.2, 0.05, 1.0]);
    renderer.write_overlay(&harness.device, &harness.queue, &in_front, view_projection(distance));
    let front = harness.frame(&renderer);
    dump("overlay-in-front", &front);

    let changed =
        bare.chunks_exact(4).zip(front.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
    assert!(changed > 2_000, "an overlay in front of the model barely drew: {changed} pixels");

    // --- behind the model: must be hidden --------------------------------
    let behind = quad(-toward_eye * (MODEL_RADIUS + 10.0), 12.0, [1.0, 0.2, 0.05, 1.0]);
    renderer.write_overlay(&harness.device, &harness.queue, &behind, view_projection(distance));
    let hidden = harness.frame(&renderer);
    dump("overlay-behind", &hidden);

    let leaked =
        bare.chunks_exact(4).zip(hidden.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
    // Not zero: the quad is wider than the sphere is at that depth, so its rim
    // legitimately shows past the silhouette. What must not happen is it
    // painting over the model itself.
    assert!(
        leaked < changed / 4,
        "an overlay behind the model was not occluded: {leaked} pixels against {changed} in front"
    );

    // --- translucent: must tint, not erase -------------------------------
    // In front of the model, and barely opaque. Note a plane at the model's
    // CENTRE would be correctly hidden by the model's front half, which is the
    // depth test doing its job, not a bug — so translucency has to be measured
    // where the surface is genuinely in front.
    let veil = quad(toward_eye * (MODEL_RADIUS + 10.0), MODEL_RADIUS, [0.2, 0.5, 1.0, 0.25]);
    renderer.write_overlay(&harness.device, &harness.queue, &veil, view_projection(distance));
    let tinted = harness.frame(&renderer);
    dump("overlay-translucent", &tinted);

    let middle = (((HEIGHT / 2) * WIDTH + WIDTH / 2) * 4) as usize;
    let (before, after) = (&bare[middle..middle + 3], &tinted[middle..middle + 3]);
    assert_ne!(before, after, "a translucent plane over the model changed nothing");

    // Tinted, not replaced: at a quarter alpha the model underneath still
    // dominates. An opaque draw would land near the quad's own colour.
    let opaque = quad(toward_eye * (MODEL_RADIUS + 10.0), MODEL_RADIUS, [0.2, 0.5, 1.0, 1.0]);
    renderer.write_overlay(&harness.device, &harness.queue, &opaque, view_projection(distance));
    let solid = harness.frame(&renderer);
    let drift = |a: &[u8], b: &[u8]| -> i32 {
        a.iter().zip(b).map(|(x, y)| (*x as i32 - *y as i32).abs()).sum()
    };
    assert!(
        drift(after, &bare[middle..middle + 3]) < drift(after, &solid[middle..middle + 3]),
        "a quarter alpha plane should stay nearer the model than the opaque one: \
         model {:?}, tinted {after:?}, opaque {:?}",
        &bare[middle..middle + 3],
        &solid[middle..middle + 3]
    );

    // --- lines ------------------------------------------------------------
    let mut ring = OverlayBatch::default();
    let centre = toward_eye * MODEL_RADIUS;
    let right = toward_eye.cross(Vec3::Y).normalize();
    let up = right.cross(toward_eye).normalize();
    for step in 0..64 {
        let a = step as f32 / 64.0 * std::f32::consts::TAU;
        let b = (step + 1) as f32 / 64.0 * std::f32::consts::TAU;
        let at = |t: f32| centre + (right * t.cos() + up * t.sin()) * 14.0;
        ring.push_line(at(a), at(b), [1.0, 0.48, 0.24, 1.0]);
    }
    renderer.write_overlay(&harness.device, &harness.queue, &ring, view_projection(distance));
    let ringed = harness.frame(&renderer);
    dump("overlay-ring", &ringed);

    let ring_pixels =
        bare.chunks_exact(4).zip(ringed.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
    assert!(ring_pixels > 200, "the ring barely drew: {ring_pixels} pixels");
    // A ring is a rim, not a disc: it must cover far less than a filled quad.
    assert!(ring_pixels < changed, "the ring covered as much as a filled quad");
}
