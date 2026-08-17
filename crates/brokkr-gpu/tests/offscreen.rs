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
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, MeshScratch, Stamp,
    Volume,
};
use brokkr_gpu::{PixelRect, SculptRenderer, Uniforms};
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

        Some(Self { device, queue, target, view, readback, padded_row_bytes })
    }

    /// Clear to a known background, draw the sculpt, then read the pixels back
    /// as tightly packed RGBA.
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
fn uniforms(distance: f32) -> Uniforms {
    let eye = Vec3::new(0.45, 0.35, 1.0).normalize() * distance;
    let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
    let projection = glam::camera::rh::proj::directx::perspective(
        45f32.to_radians(),
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        distance * 4.0,
    );
    Uniforms {
        view_projection: (projection * view).to_cols_array_2d(),
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
