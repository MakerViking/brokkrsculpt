// SPDX-License-Identifier: AGPL-3.0-only

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
    BrickCoord, BrickMesh, Brush, BrushDirection, BrushKind, BrushScratch, INSIDE, MeshScratch,
    NARROW_BAND, OUTSIDE, Pattern, PatternKind, Stamp, Volume,
};
use brokkr_gpu::{
    Frustum, NodeId, OverlayBatch, PixelRect, SculptRenderer, SlotKey, THE_ONLY_BODY,
    THUMBNAIL_BACKGROUND, THUMBNAIL_SIZE, Uniforms, background_texel,
};
use glam::{IVec3, Vec3};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 360;

/// The format the picture tests render in.
///
/// **The thumbnail tests deliberately do not use it**, and run over
/// [`THUMBNAIL_FORMATS`] instead. Both GPU harnesses in this workspace pinned
/// this one format, which is exactly why a hardcoded thumbnail atlas format
/// would have been green here and fatal in the application: iced picks the
/// first sRGB format the surface reports, which on Linux/Vulkan is normally
/// `Bgra8UnormSrgb`, and binding the sculpt pipeline against a mismatched
/// colour attachment is a validation error that wgpu's default handler turns
/// into a dead process.
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The two formats iced realistically hands the renderer.
const THUMBNAIL_FORMATS: [wgpu::TextureFormat; 2] =
    [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Bgra8UnormSrgb];

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
        Self::in_format(TARGET_FORMAT)
    }

    fn in_format(format: wgpu::TextureFormat) -> Option<Self> {
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
            format,
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
        self.frame_inner(renderer, false)
    }

    /// The same, plus the gizmo's own always-on-top pass.
    ///
    /// A second entry point rather than a flag on every call site, because the
    /// gizmo pass is the only thing in the renderer that CLEARS depth and so
    /// the only thing whose presence changes what a frame means.
    fn frame_with_gizmo(&self, renderer: &SculptRenderer) -> Vec<u8> {
        self.frame_inner(renderer, true)
    }

    fn frame_inner(&self, renderer: &SculptRenderer, gizmo: bool) -> Vec<u8> {
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

        if gizmo {
            renderer.render_gizmo(
                &mut encoder,
                &self.view,
                PixelRect { x: 0, y: 0, width: WIDTH, height: HEIGHT },
            );
        }

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

    /// One cell of the thumbnail atlas, tightly packed, in the texture's own
    /// channel order.
    ///
    /// There is no other way to see a thumbnail from a test, and "the picture
    /// is there" is the whole claim the feature makes.
    fn cell(&self, renderer: &SculptRenderer, cell: u32) -> Vec<u8> {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = (THUMBNAIL_SIZE * 4).div_ceil(align) * align;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thumbnail readback"),
            size: u64::from(padded * THUMBNAIL_SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thumbnail") });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: renderer.thumbnails().texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: cell },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(THUMBNAIL_SIZE),
                },
            },
            wgpu::Extent3d {
                width: THUMBNAIL_SIZE,
                height: THUMBNAIL_SIZE,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("readback map failed"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4) as usize);
        for row in 0..THUMBNAIL_SIZE {
            let start = (row * padded) as usize;
            pixels.extend_from_slice(&mapped[start..start + (THUMBNAIL_SIZE * 4) as usize]);
        }
        drop(mapped);
        readback.unmap();
        pixels
    }
}

/// Summed RGB above which a pixel counts as drawn rather than background.
///
/// It can be this low because the matcap has an ambient floor -- "so nothing is
/// pure black" -- and the darkest colour it can produce sums to around 200
/// against a background that sums to exactly 0. That margin is what makes an
/// exact mask comparison safe: no drawn pixel is near the threshold, so which
/// body wins the depth test at a given pixel cannot change whether it counts as
/// drawn.
const LIT: u32 = 24;

/// Pixels that are not the black background, and their bounding box.
fn coverage(pixels: &[u8]) -> (usize, [u32; 4]) {
    let mut count = 0;
    let mut bounds = [u32::MAX, u32::MAX, 0, 0];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let index = ((y * WIDTH + x) * 4) as usize;
            let lit = pixels[index] as u32 + pixels[index + 1] as u32 + pixels[index + 2] as u32;
            if lit > LIT {
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

/// One bool per pixel: was anything drawn there at all.
fn mask(pixels: &[u8]) -> Vec<bool> {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32 > LIT)
        .collect()
}

fn dump(name: &str, pixels: &[u8]) {
    let Ok(directory) = std::env::var("BROKKR_DUMP_FRAMES") else {
        return;
    };
    let mut ppm = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
    for pixel in pixels.as_chunks::<4>().0.iter() {
        ppm.extend_from_slice(&pixel[..3]);
    }
    let path = std::path::Path::new(&directory).join(format!("{name}.ppm"));
    std::fs::write(&path, ppm).expect("could not write frame dump");
    eprintln!("wrote {}", path.display());
}

/// Mesh every brick that can carry geometry and hand it to the renderer.
fn upload_all(
    renderer: &mut SculptRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &mut Volume,
) -> usize {
    volume.mark_everything_dirty();
    upload_dirty(renderer, device, queue, volume)
}

/// Mesh only what the volume says is dirty. Returns how many bricks that was.
fn upload_dirty(
    renderer: &mut SculptRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &mut Volume,
) -> usize {
    upload_dirty_as(renderer, device, queue, volume, THE_ONLY_BODY).0
}

/// Mesh everything and upload it as `body`. Returns how many bricks actually
/// produced geometry, which is how many pool SLOTS the upload created.
///
/// Not the same number as the dirty count: a brick the surface has left meshes
/// to nothing and takes no slot at all. The distinction is what lets a test
/// compare `PoolStats::bricks` against what it uploaded.
fn upload_body(
    renderer: &mut SculptRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &mut Volume,
    body: NodeId,
) -> usize {
    volume.mark_everything_dirty();
    upload_dirty_as(renderer, device, queue, volume, body).1
}

/// Returns the number of bricks meshed and the number that produced geometry.
fn upload_dirty_as(
    renderer: &mut SculptRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    volume: &mut Volume,
    body: NodeId,
) -> (usize, usize) {
    let mut scratch = MeshScratch::new();
    let mut mesh = BrickMesh::default();
    let mut dirty: Vec<BrickCoord> = Vec::new();
    let mut filled = 0;
    volume.take_dirty(&mut dirty);
    for &coord in &dirty {
        volume.mesh_brick(coord, &mut scratch, &mut mesh);
        if !mesh.indices.is_empty() {
            filled += 1;
        }
        renderer.upload_brick(device, queue, SlotKey { body, coord }, &mesh);
    }
    (dirty.len(), filled)
}

/// Seed an axis aligned cube centred on the origin, the way `seed_sphere`
/// seeds a ball: the exact signed distance in voxels, clamped into the narrow
/// band by `edit_voxels`.
///
/// There is no `seed_cube` in the engine and this does not add one -- a box is
/// four lines of `edit_box`, and giving `brokkr-core` a new primitive so that a
/// render test can have a second shape would be a wider change than the one
/// being tested. The `max_element().min(0.0)` term is what makes it exact
/// inside the box as well as outside it.
fn seed_cube(volume: &mut Volume, half: f32) {
    let voxel_size = volume.voxel_size();
    // Two bands past the face, so the exterior half of the band is written and
    // the surface has a gradient to be meshed from.
    let reach = Vec3::splat(half + NARROW_BAND * voxel_size * 2.0);
    volume.edit_box(-reach, reach, move |position, _| {
        let outside = position.abs() - Vec3::splat(half);
        let distance = outside.max(Vec3::ZERO).length() + outside.max_element().min(0.0);
        (distance / voxel_size).clamp(INSIDE, OUTSIDE)
    });
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

/// The camera, and the mask drawn the way the application ships it: tint on,
/// polarity normal. Tests that are about the tint override with
/// `Uniforms { mask_tint: .., ..uniforms(distance) }`.
fn uniforms(distance: f32) -> Uniforms {
    let view = view_matrix(distance);
    Uniforms {
        view_projection: view_projection(distance).to_cols_array_2d(),
        view: view.to_cols_array_2d(),
        // The offscreen target is an sRGB format, so the shader must not encode
        // a second time.
        srgb_target: 1,
        mask_inverted: 0,
        mask_tint: 1.0,
        padding: [0; 1],
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
    let bricks = upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
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

    let dirty = upload_dirty(&mut renderer, &harness.device, &harness.queue, &mut volume);
    assert!(dirty > 0, "the stroke dirtied nothing");
    assert!(
        dirty < bricks / 2,
        "the stroke dirtied {dirty} of {bricks} bricks, which is not proportional to the brush"
    );

    let after = harness.frame(&renderer);
    dump("after", &after);

    let changed = before
        .as_chunks::<4>()
        .0
        .iter()
        .zip(after.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    assert!(
        changed > 200,
        "only {changed} pixels changed after a stroke, so the edit never reached the screen"
    );
    assert!(
        changed < lit,
        "{changed} pixels changed but only {lit} were ever drawn, so the whole model moved rather than the brushed part"
    );

    // Adding clay must not carve a HOLE in the silhouette, which is what this
    // is for -- not that the outline is pixel for pixel monotonic, which it
    // cannot be. Adding material moves the surface, the surface moves the
    // silhouette edge by fractions of a pixel, and which edge pixels clear the
    // coverage threshold changes in both directions. macOS shrank by 17 of
    // 73,779 (0.02%) and Linux does not; both are the same picture.
    //
    // A thousandth, so a hole is still caught by three orders of magnitude: the
    // stroke covers hundreds of pixels and losing one would show here long
    // before this tolerance did.
    let (lit_after, _) = coverage(&after);
    let floor = lit - lit / 1000;
    assert!(
        lit_after >= floor,
        "the silhouette shrank from {lit} to {lit_after} pixels after adding material, past the \
         {floor} that edge rasterisation alone can account for"
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
    upload_dirty(&mut renderer, &harness.device, &harness.queue, &mut volume);

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
    // Run with BROKKR_DUMP_FRAMES set to look at the results.
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
        upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
        let before = harness.frame(&renderer);

        let brush = Brush { kind, radius: 10.0, strength: 0.7, ..Brush::default() };
        for step in -3..=3 {
            let at = front + Vec3::new(0.0, step as f32 * 3.0, 0.0);
            let normal = volume.gradient_world(at);
            for _ in 0..4 {
                brush.apply(
                    &mut volume,
                    // The drag that move follows. Every other brush ignores it
                    // unless it is running a comb pattern, and none is here.
                    &Stamp::new(at, normal, BrushDirection::Add).with_tangent(Vec3::Y),
                    &mut brush_scratch,
                );
            }
        }

        let dirty = upload_dirty(&mut renderer, &harness.device, &harness.queue, &mut volume);
        assert!(dirty > 0, "{kind} dirtied nothing");

        let after = harness.frame(&renderer);
        dump(&format!("brush-{}", kind.label().to_lowercase()), &after);

        let changed = before
            .as_chunks::<4>()
            .0
            .iter()
            .zip(after.as_chunks::<4>().0.iter())
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

        upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
        let frame = harness.frame(&renderer);
        dump(label, &frame);
        assert!(coverage(&frame).0 > 0, "{label} rendered nothing");
        frames.push(frame);
    }

    let changed = frames[0]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(frames[1].as_chunks::<4>().0.iter())
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
        upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
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

        let dirty = upload_dirty(&mut renderer, &harness.device, &harness.queue, &mut volume);
        assert!(dirty > 0, "{kind} dirtied nothing");

        let after = harness.frame(&renderer);
        dump(&format!("pattern-{}", kind.label().to_lowercase()), &after);

        if kind != PatternKind::None {
            let changed = before
                .as_chunks::<4>()
                .0
                .iter()
                .zip(after.as_chunks::<4>().0.iter())
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
                .as_chunks::<4>()
                .0
                .iter()
                .zip(other.as_chunks::<4>().0.iter())
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
    upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);

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

    let changed = bare
        .as_chunks::<4>()
        .0
        .iter()
        .zip(front.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    assert!(changed > 2_000, "an overlay in front of the model barely drew: {changed} pixels");

    // --- behind the model: must be hidden --------------------------------
    let behind = quad(-toward_eye * (MODEL_RADIUS + 10.0), 12.0, [1.0, 0.2, 0.05, 1.0]);
    renderer.write_overlay(&harness.device, &harness.queue, &behind, view_projection(distance));
    let hidden = harness.frame(&renderer);
    dump("overlay-behind", &hidden);

    let leaked = bare
        .as_chunks::<4>()
        .0
        .iter()
        .zip(hidden.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
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

    let ring_pixels = bare
        .as_chunks::<4>()
        .0
        .iter()
        .zip(ringed.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    assert!(ring_pixels > 200, "the ring barely drew: {ring_pixels} pixels");
    // A ring is a rim, not a disc: it must cover far less than a filled quad.
    assert!(ring_pixels < changed, "the ring covered as much as a filled quad");
}

/// An imported model has to actually be visible.
///
/// Every other check on the voxeliser is numeric, and the class of bug that
/// compiles, passes every numeric test and still looks wrong has already caught
/// this project three times. A field can be geometrically correct and render as
/// nothing -- bricks written but never marked dirty, or marked and never
/// uploaded -- and only pixels say so.
///
/// It also compares against the sculpted sphere it was made from. Two frames
/// that are identically blank would satisfy a bare "did it draw" check, so the
/// assertion is that the import covers a comparable area of the screen.
#[test]
fn an_imported_mesh_renders() {
    use brokkr_core::voxelise::{VoxeliseOptions, voxelise};

    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the offscreen render test");
        return;
    };

    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(MODEL_RADIUS * 3.0));

    // The reference: a seeded sphere, drawn the ordinary way.
    let mut seeded = Volume::new(VOXEL_SIZE);
    seeded.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    seeded.mark_everything_dirty();
    upload_all(&mut renderer, &harness.device, &harness.queue, &mut seeded);
    let (seeded_lit, _) = coverage(&harness.frame(&renderer));
    assert!(seeded_lit > 0, "the reference sphere drew nothing, so this test proves nothing");
    dump("import-reference", &harness.frame(&renderer));

    // The same sphere, put through export and voxelise, and drawn again.
    let (mesh, report) = seeded.export_mesh();
    assert!(report.is_printable(), "the fixture is not printable: {}", report.summary());
    let (mut imported, _) = voxelise(
        &mesh,
        &VoxeliseOptions {
            voxel_size: VOXEL_SIZE,
            centre: false,
            refit_if_implausible: false,
            fill_sealed_cavities: true,
            repair_broken_scan_lines: true,
            coarsen_to_fit: false,
            refine_to_resolve: false,
            already_reserved: 0.0,
        },
    )
    .expect("the exported sphere should voxelise");

    let mut fresh = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    fresh.resize(&harness.device, WIDTH, HEIGHT);
    fresh.write_uniforms(&harness.queue, &uniforms(MODEL_RADIUS * 3.0));
    let uploaded = upload_all(&mut fresh, &harness.device, &harness.queue, &mut imported);
    assert!(uploaded > 0, "the imported volume produced no mesh to upload");

    let pixels = harness.frame(&fresh);
    let (imported_lit, _) = coverage(&pixels);
    dump("import-voxelised", &pixels);

    assert!(imported_lit > 0, "the imported model drew nothing at all");
    let ratio = imported_lit as f64 / seeded_lit as f64;
    assert!(
        (0.9..=1.1).contains(&ratio),
        "the imported sphere covers {imported_lit} pixels against the seeded sphere's \
         {seeded_lit} ({ratio:.2}x) -- it is on screen but it is not the same shape"
    );
}

/// Two bodies that share brick coordinates must both reach the screen whole.
///
/// This is the picture behind [`brokkr_gpu::SlotKey`]. Every `Volume` sits on
/// the same lattice -- there is no world origin held anywhere, voxel (0,0,0) is
/// world (0,0,0) in all of them -- so two bodies near the origin occupy the
/// same brick coordinates, which is the normal case rather than a corner one.
/// Keyed on the coordinate alone, the second body's upload took over the first
/// body's slice, or handed it back to the free list outright, and nothing
/// logged it.
///
/// **The check is an exact mask and deliberately not a lit-pixel count.** Two
/// bodies stacked on the origin have nearly the same silhouette, so a count
/// goes green on a badly corrupted picture: lose every brick of the sphere on
/// one side and the cube behind it still lights most of those pixels. Rendering
/// each alone and requiring `both == a | b` pixel for pixel is the assertion
/// that cannot be satisfied by accident.
///
/// `PoolStats::bricks` is asserted alongside, so a pool-side eviction (a slot
/// that was never kept) and a render-side one (a slot kept but not drawn) are
/// distinguishable rather than both arriving as "the picture is wrong".
///
/// **The plan asked for a sphere of radius 20 against a cube of half-extent
/// 20, on the grounds that each reaches outside the other. It does not:** a
/// sphere of radius 20 is inscribed in that cube, touching the six face centres
/// and never leaving it, so `mask_a` would be a subset of `mask_b` and the
/// union test would hold no matter what happened to the sphere. Half-extent 14
/// is the geometry that has the property the plan wanted: the cube's corners
/// reach 24.2 mm and stand outside the sphere, and the sphere at 20 mm bulges
/// out through the middle of every face. Measured from this camera, that
/// leaves 3,901 pixels the sphere alone lights and 1,455 the cube alone does.
/// The two `only_in_*` assertions below pin both, so the test cannot quietly
/// degenerate into a tautology again. 12 and 13 were measured too and are
/// worse: 12 puts the cube entirely inside the silhouette (0 pixels of its
/// own) and 13 leaves it only 275.
#[test]
fn two_bodies_sharing_bricks_each_render_whole() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the two body render test");
        return;
    };

    /// Radius of body A.
    const SPHERE_RADIUS: f32 = 20.0;
    /// Half-extent of body B. See the note above about why this is not 20.
    const CUBE_HALF: f32 = 14.0;

    const BODY_A: NodeId = NodeId(1);
    const BODY_B: NodeId = NodeId(2);

    let distance = MODEL_RADIUS * 3.0;
    let make_renderer = || {
        let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
        renderer.resize(&harness.device, WIDTH, HEIGHT);
        renderer.write_uniforms(&harness.queue, &uniforms(distance));
        renderer
    };

    let mut sphere = Volume::new(VOXEL_SIZE);
    sphere.seed_sphere(Vec3::ZERO, SPHERE_RADIUS);
    let mut cube = Volume::new(VOXEL_SIZE);
    seed_cube(&mut cube, CUBE_HALF);

    // The premise of the whole test: these two really do want the same slots.
    // Every brick the cube occupies is also one of the sphere's, which is what
    // "sharing bricks near the origin" means and is the normal case for two
    // bodies on one lattice -- not a contrived overlap.
    let cube_bricks: std::collections::HashSet<BrickCoord> = cube.brick_coords().collect();
    let shared = sphere.brick_coords().filter(|coord| cube_bricks.contains(coord)).count();
    assert_eq!(
        shared,
        cube_bricks.len(),
        "the cube occupies {} bricks and only {shared} of them are also the sphere's, so the two \
         bodies are not really competing for slots",
        cube_bricks.len()
    );
    assert!(shared >= 8, "only {shared} bricks are contested, which proves little");

    // --- each body on its own ------------------------------------------------
    let mut alone_a = make_renderer();
    let slots_a = upload_body(&mut alone_a, &harness.device, &harness.queue, &mut sphere, BODY_A);
    let frame_a = harness.frame(&alone_a);
    dump("two-bodies-a", &frame_a);
    let mask_a = mask(&frame_a);
    assert_eq!(alone_a.stats().bricks, slots_a, "body A did not keep every slot it filled");

    let mut alone_b = make_renderer();
    let slots_b = upload_body(&mut alone_b, &harness.device, &harness.queue, &mut cube, BODY_B);
    let frame_b = harness.frame(&alone_b);
    dump("two-bodies-b", &frame_b);
    let mask_b = mask(&frame_b);
    assert_eq!(alone_b.stats().bricks, slots_b, "body B did not keep every slot it filled");

    // Each has to be visible outside the other, or the union below is a
    // tautology and would pass over a body that never drew at all.
    let only_in_a = mask_a.iter().zip(&mask_b).filter(|(a, b)| **a && !**b).count();
    let only_in_b = mask_a.iter().zip(&mask_b).filter(|(a, b)| !**a && **b).count();
    assert!(only_in_a > 200, "the sphere only reaches outside the cube on {only_in_a} pixels");
    assert!(only_in_b > 200, "the cube only reaches outside the sphere on {only_in_b} pixels");

    // --- both bodies in one pool --------------------------------------------
    let mut together = make_renderer();
    let both_a = upload_body(&mut together, &harness.device, &harness.queue, &mut sphere, BODY_A);
    let both_b = upload_body(&mut together, &harness.device, &harness.queue, &mut cube, BODY_B);
    assert_eq!(both_a, slots_a, "body A meshed differently the second time");
    assert_eq!(both_b, slots_b, "body B meshed differently the second time");

    let stats = together.stats();
    assert_eq!(stats.overflowed, 0, "the pool overflowed, so the picture is incomplete");
    assert_eq!(
        stats.bricks,
        slots_a + slots_b,
        "the pool holds {} slots for {slots_a} + {slots_b} uploaded bricks: the second body took \
         the first body's slices",
        stats.bricks
    );

    let frame_ab = harness.frame(&together);
    dump("two-bodies-both", &frame_ab);
    let mask_ab = mask(&frame_ab);

    let wrong = mask_ab
        .iter()
        .zip(mask_a.iter().zip(&mask_b))
        .filter(|(both, (a, b))| **both != (**a || **b))
        .count();
    assert_eq!(
        wrong,
        0,
        "{wrong} pixels disagree with the union of the two bodies drawn alone, out of {} \
         ({only_in_a} belong to the sphere alone, {only_in_b} to the cube alone)",
        mask_ab.len()
    );
}

/// **Hiding a body takes it out of the picture and does nothing else.**
///
/// This is the only test in the workspace that can see the actual consequence
/// of the hidden set, and it is worth having because every cheaper way of
/// implementing visibility passes a bookkeeping test and fails here: dropping
/// the hidden body's slots would empty the pool as well as the picture, and
/// meshing its bricks to nothing would do the same by a longer route. The
/// assertion that says it is a DRAW-time skip is the pair "the picture lost
/// exactly one body" and "the pool held on to every brick of it".
///
/// The two bodies are the same sphere and cube as the test above, for the same
/// reason: they contest bricks near the origin, so a hidden body that took a
/// visible one's slice with it would show up as a hole in what remains rather
/// than as a clean removal.
#[test]
fn hiding_a_body_removes_exactly_that_body_from_the_picture() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the hidden body render test");
        return;
    };

    const SPHERE_RADIUS: f32 = 20.0;
    const CUBE_HALF: f32 = 14.0;
    const BODY_A: NodeId = NodeId(1);
    const BODY_B: NodeId = NodeId(2);

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let mut sphere = Volume::new(VOXEL_SIZE);
    sphere.seed_sphere(Vec3::ZERO, SPHERE_RADIUS);
    let mut cube = Volume::new(VOXEL_SIZE);
    seed_cube(&mut cube, CUBE_HALF);

    // Body A alone first, in its own renderer, so there is something to
    // compare the hidden frame against that was never near body B at all.
    let mut alone = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    alone.resize(&harness.device, WIDTH, HEIGHT);
    alone.write_uniforms(&harness.queue, &uniforms(distance));
    upload_body(&mut alone, &harness.device, &harness.queue, &mut sphere, BODY_A);
    let mask_alone = mask(&harness.frame(&alone));

    let slots_a = upload_body(&mut renderer, &harness.device, &harness.queue, &mut sphere, BODY_A);
    let slots_b = upload_body(&mut renderer, &harness.device, &harness.queue, &mut cube, BODY_B);
    let both = mask(&harness.frame(&renderer));
    let full = renderer.stats();
    assert_eq!(full.bricks, slots_a + slots_b);
    // The fixture asserts nothing unless the cube is visible outside the
    // sphere: hiding a body that draws no pixels of its own would pass anything.
    let only_in_b = both.iter().zip(&mask_alone).filter(|(both, a)| **both && !**a).count();
    assert!(only_in_b > 200, "the cube only reaches outside the sphere on {only_in_b} pixels");

    renderer.set_hidden(&[BODY_B]);
    let hidden_frame = harness.frame(&renderer);
    dump("hidden-body", &hidden_frame);
    let hidden = mask(&hidden_frame);
    let wrong = hidden.iter().zip(&mask_alone).filter(|(one, other)| one != other).count();
    assert_eq!(
        wrong, 0,
        "{wrong} pixels differ from body A drawn on its own, so hiding body B either left some \
         of it on screen or took some of body A with it"
    );

    // ...and the pool did not give up a single byte for it, which is what makes
    // this a draw-time skip and why the overflow message tells the user that
    // hiding frees nothing.
    let while_hidden = renderer.stats();
    assert_eq!(while_hidden.bricks, full.bricks, "hiding dropped slots");
    assert_eq!(while_hidden.vertices_reserved, full.vertices_reserved);
    assert_eq!(while_hidden.vertices_watermark, full.vertices_watermark);
    assert_eq!(renderer.body_bricks(BODY_B), slots_b, "the hidden body lost bricks");
    assert_eq!(while_hidden.hidden, slots_b, "the skipped bricks were not counted as hidden");
    assert_eq!(while_hidden.drawn + while_hidden.culled + while_hidden.hidden, full.bricks);

    // Showing it again needs no remesh and no upload: the same frame comes
    // back from the slices that were there all along.
    renderer.set_hidden(&[]);
    let shown = mask(&harness.frame(&renderer));
    let wrong = shown.iter().zip(&both).filter(|(one, other)| one != other).count();
    assert_eq!(wrong, 0, "{wrong} pixels did not come back when the body was shown again");
}

/// How cool a pixel is: blue minus red, in sRGB levels.
///
/// The one number the mask's tint moves and the matcap barely does. Its whole
/// gamut on the clay runs from a rim at -28 to a fill-lit cavity at +17, which
/// is why the tint was designed to leave that range rather than to be a shade
/// of it -- and why this, rather than luminance, is what these tests measure.
fn chroma(pixels: &[u8]) -> Vec<i32> {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| i32::from(pixel[2]) - i32::from(pixel[0]))
        .collect()
}

/// Fully protect every voxel of the model, or the half of it at x below zero.
fn protect(volume: &mut Volume, half: bool) {
    let reach = (MODEL_RADIUS / VOXEL_SIZE) as i32 + 4;
    let high = IVec3::new(if half { 0 } else { reach }, reach, reach);
    volume.edit_mask(IVec3::splat(-reach), high, |_, _, _| brokkr_core::PROTECTED);
}

/// A masked body is tinted where it is masked, and is the same SHAPE either
/// way.
///
/// Three claims. The second is the one that makes the first safe to make: the
/// mask must never move a vertex. It is a read-only multiplier on what a brush
/// may do, so a silhouette that changed would mean the attribute had reached
/// the geometry. The third is that the tint STOPS at the mask, bounded above
/// and below over the body's own pixels -- see the comment on it for why a
/// one-sided count taken over the whole frame proves nothing at all.
///
/// **The tint is measured against the DISTRIBUTION of the unmasked pixels of
/// the same body**, not against one of them. The matcap's own luminance swing
/// is 102 levels, so a tint compared with a single reference pixel proves
/// nothing about whether a human can separate it from form; compared with the
/// whole spread of what the clay can be, "outside that range" is exactly the
/// claim the hue was chosen to make.
#[test]
fn a_masked_body_is_tinted_beyond_anything_the_unmasked_one_can_be() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the mask tint render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
    let bare = harness.frame(&renderer);
    dump("mask-none", &bare);

    // Half of it, so one frame carries both answers and the boundary between
    // them is on screen.
    protect(&mut volume, true);
    let remeshed = upload_dirty(&mut renderer, &harness.device, &harness.queue, &mut volume);
    assert!(remeshed > 0, "painting the mask marked no brick for remesh");
    let masked = harness.frame(&renderer);
    dump("mask-half", &masked);

    let before = mask(&bare);
    let after = mask(&masked);
    let moved = before.iter().zip(&after).filter(|(one, other)| one != other).count();
    assert_eq!(moved, 0, "{moved} pixels changed shape, so the mask reached the geometry");

    // What the clay itself can be, over the drawn pixels of this very body.
    let bare_chroma = chroma(&bare);
    let masked_chroma = chroma(&masked);
    let clay_coolest = bare_chroma
        .iter()
        .zip(&before)
        .filter_map(|(value, drawn)| drawn.then_some(*value))
        .max()
        .expect("the fixture drew nothing");

    let outside_the_gamut = masked_chroma
        .iter()
        .zip(&after)
        .filter(|(value, drawn)| **drawn && **value > clay_coolest)
        .count();
    let drawn = after.iter().filter(|drawn| **drawn).count();
    assert!(
        outside_the_gamut * 5 > drawn,
        "only {outside_the_gamut} of {drawn} drawn pixels are cooler than the coolest the \
         unmasked body reaches ({clay_coolest}), so the tint is not separable from the form"
    );

    // And the tint stops where the mask does: the masked part of the body
    // changed, the rest of it did not change by one byte.
    //
    // **Counted over the DRAWN pixels only, and bounded on BOTH sides.** Over
    // the whole frame neither bound would mean anything: the background is
    // 98,659 of this 172,800-pixel frame and the body shader never writes it,
    // so an "unchanged" count taken over the frame is dominated by pixels
    // neither picture ever drew, and a tint that washed the entire body would
    // still leave it far above any share of the drawn pixels. That is the
    // shape of a GPU-side leak -- a wrong `@location`, a stride slip, a floor
    // ORed in -- and it is invisible to the core tests, which is exactly why
    // it has to be caught here.
    //
    // The band is a real geometric quantity, not a formality. The fixture
    // protects the half of the ball at x below zero, but the camera sits at
    // +x, so the masked half is the one turned AWAY: with the view direction's
    // x component at 0.391, an orthographic camera would put 30.5% of the
    // silhouette under the mask ((1 - 0.391) / 2, integrating the visible cap
    // against the plane x = 0), and this perspective camera at three radii
    // measures 23.6%. Anything below 15% means the tint is failing to reach
    // most of the mask; anything above 35% means it is bleeding off it.
    let unchanged = bare
        .as_chunks::<4>()
        .0
        .iter()
        .zip(masked.as_chunks::<4>().0.iter())
        .zip(&before)
        .filter(|((one, other), drawn)| **drawn && one == other)
        .count();
    let changed = drawn - unchanged;
    assert!(
        changed * 100 > drawn * 15,
        "only {changed} of {drawn} drawn pixels changed when half the body was masked, so the \
         tint is not following the attribute onto the masked half"
    );
    assert!(
        changed * 100 < drawn * 35,
        "{changed} of {drawn} drawn pixels changed, far past the third of the silhouette the \
         mask is on, so the tint spread past the mask and washed the body it is on"
    );
}

/// **Switching the tint off draws a masked body exactly as an unmasked one**,
/// byte for byte, with no remesh and no upload.
///
/// This is the whole of what makes a `show mask` toggle safe to ship: it is a
/// uniform and reaches nothing else. The comparison is against the frame the
/// body drew BEFORE it was ever masked, which is a stronger claim than "it
/// changed back" -- a tint that had leaked into the mesh would come back as a
/// difference here and nowhere else.
///
/// What is NOT asserted here, because it cannot be seen from this crate: that
/// the standing mask card stays up with its percentage unchanged. That half
/// lives beside the card, in `brokkr-app`.
#[test]
fn switching_the_tint_off_draws_a_masked_body_exactly_as_an_unmasked_one() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the mask tint toggle test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
    let bare = harness.frame(&renderer);

    protect(&mut volume, false);
    upload_dirty(&mut renderer, &harness.device, &harness.queue, &mut volume);
    let tinted = harness.frame(&renderer);
    assert_ne!(tinted, bare, "the fixture masked nothing");
    let slots = renderer.stats();

    // The toggle, and nothing but the toggle: one uniform write, no upload.
    renderer.write_uniforms(&harness.queue, &Uniforms { mask_tint: 0.0, ..uniforms(distance) });
    let untinted = harness.frame(&renderer);
    dump("mask-tint-off", &untinted);
    assert_eq!(
        untinted, bare,
        "a body with the tint switched off must be pixel-identical to an unmasked one"
    );
    assert_eq!(renderer.stats().bricks, slots.bricks, "the toggle moved pool slots");
    assert_eq!(renderer.stats().vertices_watermark, slots.vertices_watermark);
}

/// **The tint is continuous across a brick seam.**
///
/// The attribute is sampled by the vertex's lattice CELL, which two bricks
/// either side of a seam derive from the same world coordinate, so both look
/// the same voxel up and get the same byte. Sampling by position instead --
/// the obvious alternative -- splits that pair in the last bits, and what the
/// user would see is a hard line of mismatched tint every 32 voxels.
///
/// A previous revision of the plan said no numeric test could see this. It can:
/// mask the body UNIFORMLY, so the tint owes the same amount everywhere, and
/// walk a row of pixels across several brick boundaries taking the difference
/// between the tinted frame and the untinted one. A seam that dropped to the
/// unmasked byte is a step of tens of levels in a signal that otherwise moves
/// by ones.
#[test]
fn the_tint_is_continuous_across_a_brick_seam() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the mask seam render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
    let bare = harness.frame(&renderer);

    // The whole body, so any variation left in the difference below belongs to
    // the mesher and not to the mask.
    protect(&mut volume, false);
    upload_dirty(&mut renderer, &harness.device, &harness.queue, &mut volume);
    let tinted = harness.frame(&renderer);
    dump("mask-uniform", &tinted);

    // A 60 mm ball at 0.5 mm voxels is 120 voxels across, so a row through its
    // middle crosses three brick boundaries at least.
    let drawn = mask(&bare);
    let bare_chroma = chroma(&bare);
    let tinted_chroma = chroma(&tinted);
    let row = (HEIGHT / 2) as usize;

    let mut samples = 0;
    let mut worst = 0;
    let mut previous: Option<i32> = None;
    for x in 0..WIDTH as usize {
        let index = row * WIDTH as usize + x;
        if !drawn[index] {
            previous = None;
            continue;
        }
        let lift = tinted_chroma[index] - bare_chroma[index];
        if let Some(last) = previous {
            worst = worst.max((lift - last).abs());
            samples += 1;
        }
        previous = Some(lift);
    }

    assert!(samples > 100, "the row crossed only {samples} pixels of the model");
    // Chosen against what a broken seam would produce, not against the noise:
    // a vertex that fell back to the unmasked byte moves this by tens, because
    // the tint's own lift on lit clay is around 70 levels.
    assert!(
        worst < 20,
        "the tint jumps by {worst} levels between two neighbouring pixels of a uniformly \
         masked body, which is a brick seam showing through"
    );
}

/// Flipping the polarity changes the picture and marks NOTHING dirty.
///
/// The payoff of resolving the polarity in the shader rather than baking it
/// into the attribute: Invert and Mask All are one word, where baking makes
/// them a remesh of the whole body -- 71 ms on the dragon and roughly 475 ms at
/// the brick count the pool is sized for.
#[test]
fn flipping_the_polarity_changes_the_picture_and_dirties_no_brick() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the mask polarity render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    protect(&mut volume, true);
    upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);
    let normal = harness.frame(&renderer);
    let before = renderer.stats();

    // The engine's half of the flip...
    volume.mask_mut().set_inverted(true);
    let mut dirty: Vec<BrickCoord> = Vec::new();
    volume.take_dirty(&mut dirty);
    assert!(dirty.is_empty(), "{} bricks were marked for remesh by a bool", dirty.len());

    // ...and the viewport's, which is one word.
    renderer.write_uniforms(&harness.queue, &Uniforms { mask_inverted: 1, ..uniforms(distance) });
    let flipped = harness.frame(&renderer);
    dump("mask-inverted", &flipped);
    assert_ne!(flipped, normal, "the polarity did not reach the shader");
    assert_eq!(renderer.stats().bricks, before.bricks, "a flip moved pool slots");

    // The halves swapped rather than the whole body changing: what was masked
    // is now free and what was free is now masked.
    let drawn = mask(&normal);
    let normal_chroma = chroma(&normal);
    let flipped_chroma = chroma(&flipped);
    let cooler = normal_chroma
        .iter()
        .zip(&flipped_chroma)
        .zip(&drawn)
        .filter(|((one, other), drawn)| **drawn && other > one)
        .count();
    let warmer = normal_chroma
        .iter()
        .zip(&flipped_chroma)
        .zip(&drawn)
        .filter(|((one, other), drawn)| **drawn && other < one)
        .count();
    assert!(
        cooler > 100 && warmer > 100,
        "inverting tinted {cooler} pixels and untinted {warmer}, so it was not an inversion"
    );
}

/// **Two bodies that disagree about polarity are each drawn with their own.**
///
/// Increment 24's review finding, in the only place a pixel exists. The mask is
/// per BODY and `Uniforms::mask_inverted` is one word for the whole draw, so
/// when the application published only the ACTIVE body's, every other body was
/// drawn through it. The harmless direction of that is an unmasked body coming
/// out fully tinted; the direction that matters is the other one -- a body under
/// Mask All, whose stored bytes are all zero, drawn as free. A fully protected
/// body with no tint on it is the failure the whole masking design is arranged
/// around.
///
/// The fixture isolates the bind-group choice and nothing else: BOTH masks are
/// empty, so both bodies' vertex attributes are zero and the two meshes are
/// identical in everything the shader reads. The only difference between the
/// frames is which uniform buffer each bucket is bound against.
///
/// The last frame is the control. Without it, "the unmasked body was not
/// tinted" would also pass if the tint had simply stopped working, so the same
/// document is drawn once more with an EMPTY opposite set -- the old behaviour
/// -- and the unmasked body has to come out tinted there.
#[test]
fn two_bodies_that_disagree_about_polarity_are_each_drawn_with_their_own() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the split polarity render test");
        return;
    };

    const BODY_A: NodeId = NodeId(1);
    const BODY_B: NodeId = NodeId(2);
    // Small enough and close enough in that both land well inside a frustum
    // built for one model of `MODEL_RADIUS`, and far enough apart that no pixel
    // belongs to both.
    const RADIUS: f32 = 11.0;
    const OFFSET: f32 = 18.0;

    let distance = MODEL_RADIUS * 3.0;
    let mut left = Volume::new(VOXEL_SIZE);
    left.seed_sphere(Vec3::new(-OFFSET, 0.0, 0.0), RADIUS);
    let mut right = Volume::new(VOXEL_SIZE);
    right.seed_sphere(Vec3::new(OFFSET, 0.0, 0.0), RADIUS);

    // Which pixels are whose, from each body drawn on its own.
    let where_it_is = |volume: &mut Volume, body: NodeId| {
        let mut alone = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
        alone.resize(&harness.device, WIDTH, HEIGHT);
        alone.write_uniforms(&harness.queue, &uniforms(distance));
        upload_body(&mut alone, &harness.device, &harness.queue, volume, body);
        mask(&harness.frame(&alone))
    };
    let is_a = where_it_is(&mut left, BODY_A);
    let is_b = where_it_is(&mut right, BODY_B);
    let a_pixels = is_a.iter().filter(|drawn| **drawn).count();
    let b_pixels = is_b.iter().filter(|drawn| **drawn).count();
    assert!(a_pixels > 500 && b_pixels > 500, "the fixture drew {a_pixels} and {b_pixels} pixels");
    assert_eq!(
        is_a.iter().zip(&is_b).filter(|(one, other)| **one && **other).count(),
        0,
        "the two bodies overlap on screen, so no pixel can be attributed to either"
    );

    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));
    upload_body(&mut renderer, &harness.device, &harness.queue, &mut left, BODY_A);
    upload_body(&mut renderer, &harness.device, &harness.queue, &mut right, BODY_B);
    let agreed = harness.frame(&renderer);

    // Mask all on body A: an empty map read inverted. Nothing is remeshed and
    // nothing is re-uploaded, which is the point of resolving polarity at read.
    left.mask_mut().set_inverted(true);
    let mut dirty: Vec<BrickCoord> = Vec::new();
    left.take_dirty(&mut dirty);
    assert!(dirty.is_empty(), "{} bricks were marked for remesh by a bool", dirty.len());

    renderer.write_uniforms(&harness.queue, &Uniforms { mask_inverted: 1, ..uniforms(distance) });
    renderer.set_opposite_polarity(&[BODY_B]);
    assert_eq!(renderer.opposite_polarity_bodies(), &[BODY_B], "the set did not reach the pool");
    let split = harness.frame(&renderer);
    dump("polarity-split", &split);

    let agreed_chroma = chroma(&agreed);
    let split_chroma = chroma(&split);
    let average_lift = |drawn: &[bool], after: &[i32]| {
        let total: i32 = drawn
            .iter()
            .zip(agreed_chroma.iter().zip(after))
            .filter(|(drawn, _)| **drawn)
            .map(|(_, (before, after))| after - before)
            .sum();
        total / drawn.iter().filter(|drawn| **drawn).count().max(1) as i32
    };

    // The masked body is tinted. The threshold is against what a broken tint
    // would produce and not against the noise: the tint's lift on lit clay is
    // around 70 levels, where the matcap's whole chroma gamut is 45.
    let lifted = average_lift(&is_a, &split_chroma);
    assert!(lifted > 30, "the fully masked body lifted by only {lifted}");

    // ...and the unmasked one is untouched, pixel for pixel. This is the
    // assertion the finding is about.
    let changed = is_b
        .iter()
        .enumerate()
        .filter(|(index, drawn)| {
            let texel = index * 4..index * 4 + 3;
            **drawn && agreed[texel.clone()] != split[texel]
        })
        .count();
    assert_eq!(
        changed, 0,
        "{changed} pixels of the UNMASKED body changed when the other body was masked, so it was \
         drawn through the active body's polarity"
    );

    // The control: the same document with an empty opposite set is the old
    // behaviour, and there the unmasked body IS tinted.
    renderer.set_opposite_polarity(&[]);
    let one_word = harness.frame(&renderer);
    let lift = average_lift(&is_b, &chroma(&one_word));
    assert!(
        lift > 30,
        "the control did not reproduce the defect: the unmasked body lifted by {lift} with one \
         polarity for the whole draw, so this test would pass with a dead tint"
    );
}

/// The thumbnail atlas, end to end: a real body drawn into a real cell, read
/// back, and looked at.
///
/// **Run over both formats iced realistically picks, and that is the whole
/// point of the parameter.** The sculpt pipeline is built against the target
/// format it was handed, and binding it in a pass whose colour attachment has a
/// different format is an `IncompatibleColorAttachment` validation error at
/// `set_pipeline` -- which, under wgpu's default uncaptured-error handler,
/// kills the process. A hardcoded atlas format would therefore have passed
/// every test in this file, which pins `Rgba8UnormSrgb`, and killed the
/// application on the common Linux/Vulkan configuration 200 ms after the first
/// stroke.
///
/// Four claims, and each one fails differently:
///
///  * a rendered cell holds a PICTURE -- more than one colour. A blank cell is
///    what a wrong camera, a wrong matrix buffer or a body with no slots all
///    look like, and it is indistinguishable from "not rendered yet";
///  * that picture is not just the background with the sphere missing, so the
///    lit fraction is bounded from both sides;
///  * rendering another cell leaves this one BYTE IDENTICAL. The atlas is one
///    texture and a layer index off by anything at all would show here;
///  * a cell nothing has drawn into is exactly the placeholder colour, in the
///    texture's own channel order. That is what makes the free placeholder
///    free, and getting the swizzle wrong makes every unrendered cell blue on
///    exactly the configuration iced picks.
#[test]
fn a_body_renders_into_its_own_cell_and_leaves_the_others_alone() {
    /// The cell the body is drawn into, and a second one to prove the first is
    /// not simply "wherever the atlas happens to write".
    const DRAWN: u32 = 3;
    const OTHER: u32 = 5;
    /// A cell nothing ever touches.
    const UNTOUCHED: u32 = 0;
    const BODY: NodeId = NodeId(1);
    const SECOND: NodeId = NodeId(2);

    for format in THUMBNAIL_FORMATS {
        let Some(harness) = Harness::in_format(format) else {
            eprintln!("no usable wgpu adapter, skipping the thumbnail render test");
            return;
        };

        let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, format);
        let mut sphere = Volume::new(VOXEL_SIZE);
        sphere.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
        let bounds = sphere.world_bounds().expect("a seeded sphere has bricks");
        upload_body(&mut renderer, &harness.device, &harness.queue, &mut sphere, BODY);

        let background = background_texel(format);
        let placeholder = harness.cell(&renderer, UNTOUCHED);
        for (index, texel) in placeholder.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                *texel, background,
                "{format:?}: texel {index} of an unrendered cell is {texel:?}, not the \
                 placeholder colour {background:?}"
            );
        }

        renderer.render_thumbnail(&harness.device, &harness.queue, DRAWN, BODY, bounds);
        let drawn = harness.cell(&renderer, DRAWN);

        let lit =
            drawn.as_chunks::<4>().0.iter().filter(|texel| texel[..3] != background[..3]).count();
        let total = (THUMBNAIL_SIZE * THUMBNAIL_SIZE) as usize;
        assert!(
            lit > total / 8,
            "{format:?}: only {lit} of {total} texels differ from the background, so the body \
             barely reached its cell"
        );
        assert!(
            lit < total * 9 / 10,
            "{format:?}: {lit} of {total} texels are lit, so the body fills the cell rather than \
             sitting in it -- the framing is too close"
        );

        // A picture, not a flat fill: the matcap has to be shading it.
        let shades: std::collections::HashSet<[u8; 3]> = drawn
            .as_chunks::<4>()
            .0
            .iter()
            .map(|texel| [texel[0], texel[1], texel[2]])
            .filter(|texel| texel[..] != background[..3])
            .collect();
        assert!(
            shades.len() > 8,
            "{format:?}: the cell holds {} distinct colours, so it is a silhouette rather than a \
             rendered body",
            shades.len()
        );

        // Every other cell is untouched, including one that has never been
        // written and one that is about to be.
        let before_other = harness.cell(&renderer, OTHER);
        assert_eq!(
            harness.cell(&renderer, UNTOUCHED),
            placeholder,
            "{format:?}: drawing cell {DRAWN} changed cell {UNTOUCHED}"
        );

        let mut cube = Volume::new(VOXEL_SIZE);
        seed_cube(&mut cube, MODEL_RADIUS * 0.5);
        let cube_bounds = cube.world_bounds().expect("a seeded cube has bricks");
        upload_body(&mut renderer, &harness.device, &harness.queue, &mut cube, SECOND);
        renderer.render_thumbnail(&harness.device, &harness.queue, OTHER, SECOND, cube_bounds);

        assert_eq!(
            harness.cell(&renderer, DRAWN),
            drawn,
            "{format:?}: rendering cell {OTHER} changed cell {DRAWN}"
        );
        assert_ne!(
            harness.cell(&renderer, OTHER),
            before_other,
            "{format:?}: rendering into cell {OTHER} left it as it was"
        );
    }
}

/// A request naming a body the pool has never heard of leaves a correctly
/// cleared cell rather than the last body drawn there, and a cell past the end
/// of the atlas does nothing at all.
///
/// Both are states the application can genuinely reach for a frame: the cell
/// bookkeeping and the pool's contents are separate pieces of state, and a
/// delete that lands between the request and the drain is exactly the gap.
#[test]
fn a_request_for_a_body_the_pool_does_not_hold_clears_the_cell_rather_than_keeping_the_last_one() {
    const CELL: u32 = 7;
    const REAL: NodeId = NodeId(1);
    const GHOST: NodeId = NodeId(99);

    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the thumbnail clearing test");
        return;
    };

    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    let mut sphere = Volume::new(VOXEL_SIZE);
    sphere.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    let bounds = sphere.world_bounds().expect("a seeded sphere has bricks");
    upload_body(&mut renderer, &harness.device, &harness.queue, &mut sphere, REAL);

    renderer.render_thumbnail(&harness.device, &harness.queue, CELL, REAL, bounds);
    let background = background_texel(TARGET_FORMAT);
    let with_body = harness.cell(&renderer, CELL);
    assert!(
        with_body.as_chunks::<4>().0.iter().any(|texel| texel[..3] != background[..3]),
        "the fixture drew nothing, so this test proves nothing"
    );

    renderer.render_thumbnail(&harness.device, &harness.queue, CELL, GHOST, bounds);
    let after = harness.cell(&renderer, CELL);
    for (index, texel) in after.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(
            *texel, background,
            "texel {index} still holds the previous body: the pass loaded instead of clearing, \
             so a deleted body's picture would stay in a cell that now belongs to another one"
        );
    }

    // Past the end of the atlas: no panic, no submission, nothing changed.
    let far = harness.cell(&renderer, 1);
    renderer.render_thumbnail(&harness.device, &harness.queue, u32::MAX, REAL, bounds);
    assert_eq!(harness.cell(&renderer, 1), far, "an out of range cell wrote somewhere else");
}

/// The atlas is exactly as big as a document may be, and no bigger.
///
/// Two constants in two crates, and a mismatch is either a body with no picture
/// or 1.72 MiB of VRAM nothing can ever address.
#[test]
fn the_atlas_has_one_cell_per_body_a_document_is_allowed() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the atlas size test");
        return;
    };
    let renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    assert_eq!(renderer.thumbnails().cells(), brokkr_core::MAX_BODIES as u32);
    // And the placeholder constant is opaque, or every cell the model does not
    // cover blends against whatever the panel left there -- while an RGB dump
    // of the texture looks perfect.
    assert_eq!(THUMBNAIL_BACKGROUND[3], 0xff);
}

/// **The blit, which is the only part of a thumbnail the user actually looks
/// at, and the only part no other test touches.**
///
/// `render_thumbnail` puts pixels in the atlas; everything between there and
/// the panel is the twenty-five lines of `blit.wgsl` and one pipeline built
/// with `depth_stencil: None`. A flipped `uv`, a triangle that misses a corner,
/// a bind group naming the wrong layer or an sRGB round trip that encodes twice
/// all leave every other assertion in this file green and the running
/// application visibly wrong -- and the running application is the one thing a
/// test cannot open.
///
/// Blitted at 1:1, so the linear sampler lands on texel centres and the answer
/// is the atlas cell itself. That makes the assertion an equality rather than a
/// resemblance, and it is what pins the sRGB round trip: the atlas decodes on
/// sample and the target encodes on write, so the two have to cancel exactly.
/// Run over both formats, because that cancellation is a property of the
/// format and not of the shader.
#[test]
fn a_cell_blits_into_a_row_unflipped_and_with_the_srgb_round_trip_cancelling() {
    const CELL: u32 = 6;
    const BODY: NodeId = NodeId(1);

    for format in THUMBNAIL_FORMATS {
        let Some(harness) = Harness::in_format(format) else {
            eprintln!("no usable wgpu adapter, skipping the thumbnail blit test");
            return;
        };

        let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, format);
        // A cube rather than a sphere: a sphere is symmetric about the
        // horizontal axis from this camera, so a vertical flip would be
        // invisible. A cube seen from above one corner is not.
        let mut cube = Volume::new(VOXEL_SIZE);
        seed_cube(&mut cube, MODEL_RADIUS * 0.6);
        let bounds = cube.world_bounds().expect("a seeded cube has bricks");
        upload_body(&mut renderer, &harness.device, &harness.queue, &mut cube, BODY);
        renderer.render_thumbnail(&harness.device, &harness.queue, CELL, BODY, bounds);
        let cell = harness.cell(&renderer, CELL);
        // The control, without which a comparison of two flat fills would pass
        // over a blit that drew nothing at all.
        let background = background_texel(format);
        let lit =
            cell.as_chunks::<4>().0.iter().filter(|texel| texel[..3] != background[..3]).count();
        assert!(lit > 400, "{format:?}: the fixture cell holds only {lit} lit texels");

        // A row, one to one with the cell, drawn the way iced draws one: an
        // already-open pass with NO depth attachment, viewport and scissor
        // already set to the row's bounds.
        let row = harness.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("thumbnail row"),
            size: wgpu::Extent3d {
                width: THUMBNAIL_SIZE,
                height: THUMBNAIL_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let row_view = row.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = harness
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("row") });
        let mut drawn = false;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("row pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &row_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Magenta: nothing the blit can produce, so an
                        // untouched pixel is unmistakable.
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(0.0, 0.0, THUMBNAIL_SIZE as f32, THUMBNAIL_SIZE as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
            drawn = renderer.blit_thumbnail(&mut pass, CELL) || drawn;
        }
        harness.queue.submit([encoder.finish()]);
        assert!(drawn, "{format:?}: the blit returned false, so iced would open a second pass");

        let blitted = read_back(&harness, &row);
        let close = blitted
            .as_chunks::<4>()
            .0
            .iter()
            .zip(cell.as_chunks::<4>().0.iter())
            .filter(|(a, b)| a[..3].iter().zip(&b[..3]).all(|(x, y)| x.abs_diff(*y) <= 1))
            .count();
        let total = (THUMBNAIL_SIZE * THUMBNAIL_SIZE) as usize;
        assert!(
            close * 100 >= total * 99,
            "{format:?}: only {close} of {total} row pixels match the cell they came from. A \
             flipped uv, a triangle that misses the corners, or an sRGB round trip that encodes \
             twice all land here."
        );

        // And the corners specifically, because a triangle one unit too small
        // fails only there and would still pass a 99% match.
        let last = THUMBNAIL_SIZE - 1;
        for (x, y) in [(0, 0), (last, 0), (0, last), (last, last)] {
            let at = ((y * THUMBNAIL_SIZE + x) * 4) as usize;
            assert_eq!(
                blitted[at..at + 3],
                cell[at..at + 3],
                "{format:?}: corner ({x}, {y}) of the row is not the corner of the cell"
            );
        }
    }
}

/// One texture read back tightly packed, for the blit test's row target.
fn read_back(harness: &Harness, texture: &wgpu::Texture) -> Vec<u8> {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = (THUMBNAIL_SIZE * 4).div_ceil(align) * align;
    let buffer = harness.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("row readback"),
        size: u64::from(padded * THUMBNAIL_SIZE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = harness
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("row readback") });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(THUMBNAIL_SIZE),
            },
        },
        wgpu::Extent3d { width: THUMBNAIL_SIZE, height: THUMBNAIL_SIZE, depth_or_array_layers: 1 },
    );
    harness.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("readback map failed"));
    harness.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");
    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4) as usize);
    for row in 0..THUMBNAIL_SIZE {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + (THUMBNAIL_SIZE * 4) as usize]);
    }
    drop(mapped);
    buffer.unmap();
    pixels
}

/// **The property the transform gizmo exists on, on a real device.**
///
/// A gizmo sits on the body it moves, which means its origin is INSIDE the
/// mesh. Every one of `overlay.rs`'s three pipelines compares `Less` against
/// the sculpt's depth, so geometry there is buried and unreachable -- correct
/// for a brush ring lying on a surface, useless for a control the user has to
/// grab. `SculptRenderer::overlay_pass` clears depth and keeps `Less`, which is
/// what makes the gizmo visible while still occluding itself.
///
/// Both halves are asserted, and the first is what gives the second meaning:
/// the same geometry through the ordinary overlay has to be hidden, or the test
/// would pass just as well if the depth clear did nothing.
#[test]
fn a_gizmo_inside_the_model_is_drawn_over_it_where_an_overlay_would_be_buried() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the gizmo render test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let mut volume = Volume::new(VOXEL_SIZE);
    volume.seed_sphere(Vec3::ZERO, MODEL_RADIUS);
    upload_all(&mut renderer, &harness.device, &harness.queue, &mut volume);

    let bare = harness.frame(&renderer);
    let differing = |a: &[u8], b: &[u8]| {
        a.as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0.iter())
            .filter(|(x, y)| x[..3] != y[..3])
            .count()
    };

    // A quad at the model's CENTRE, buried under 30 mm of sphere.
    let toward_eye = Vec3::new(0.45, 0.35, 1.0).normalize();
    let right = toward_eye.cross(Vec3::Y).normalize();
    let up = right.cross(toward_eye).normalize();
    let half = 10.0;
    let mut batch = OverlayBatch::default();
    batch.push_quad(
        -right * half - up * half,
        right * half - up * half,
        right * half + up * half,
        -right * half + up * half,
        [0.94, 0.35, 0.46, 1.0],
    );

    // Through the ordinary overlay: buried, which is the behaviour the gizmo
    // could not live with.
    renderer.write_overlay(&harness.device, &harness.queue, &batch, view_projection(distance));
    let buried = harness.frame(&renderer);
    dump("gizmo-through-the-overlay", &buried);
    let leaked = differing(&bare, &buried);

    // Through the gizmo's own pass: drawn.
    renderer.write_overlay(
        &harness.device,
        &harness.queue,
        &OverlayBatch::default(),
        view_projection(distance),
    );
    renderer.write_gizmo(&harness.device, &harness.queue, &batch, view_projection(distance));
    let shown = harness.frame_with_gizmo(&renderer);
    dump("gizmo-through-its-own-pass", &shown);
    let drawn = differing(&bare, &shown);

    assert!(drawn > 2_000, "the gizmo pass drew almost nothing over the model: {drawn} pixels");
    assert!(
        leaked < drawn / 10,
        "the ordinary overlay was not buried after all, so this test proves nothing: \
         {leaked} pixels against the gizmo pass's {drawn}"
    );
}

/// The cleared-depth pass still depth-tests WITHIN itself, which is what gives
/// an arrowhead over its own shaft and three crossing rings for free.
///
/// Without it the answer would be draw order, and the alternative -- a
/// `depth_compare: Always` pipeline -- would need the geometry sorted back to
/// front on the CPU every time the camera moved.
#[test]
fn the_gizmo_pass_occludes_its_own_far_side() {
    let Some(harness) = Harness::new() else {
        eprintln!("no usable wgpu adapter, skipping the gizmo occlusion test");
        return;
    };

    let distance = MODEL_RADIUS * 3.0;
    let mut renderer = SculptRenderer::new(&harness.device, &harness.queue, TARGET_FORMAT);
    renderer.resize(&harness.device, WIDTH, HEIGHT);
    renderer.write_uniforms(&harness.queue, &uniforms(distance));

    let toward_eye = Vec3::new(0.45, 0.35, 1.0).normalize();
    let right = toward_eye.cross(Vec3::Y).normalize();
    let up = right.cross(toward_eye).normalize();
    let quad = |batch: &mut OverlayBatch, centre: Vec3, half: f32, colour: [f32; 4]| {
        batch.push_quad(
            centre - right * half - up * half,
            centre + right * half - up * half,
            centre + right * half + up * half,
            centre - right * half + up * half,
            colour,
        );
    };

    // A big near quad and a small far one directly behind it, submitted FAR
    // FIRST so that draw order alone would leave the near one on top anyway...
    let mut near_first = OverlayBatch::default();
    quad(&mut near_first, toward_eye * 20.0, 14.0, [0.1, 0.9, 0.3, 1.0]);
    quad(&mut near_first, toward_eye * -20.0, 6.0, [0.9, 0.1, 0.1, 1.0]);
    renderer.write_gizmo(&harness.device, &harness.queue, &near_first, view_projection(distance));
    let near_first_frame = harness.frame_with_gizmo(&renderer);

    // ...and the same pair submitted the other way round. If depth is working,
    // the two frames are identical; if it is draw order, the small red quad
    // paints over the green one in exactly one of them.
    let mut far_first = OverlayBatch::default();
    quad(&mut far_first, toward_eye * -20.0, 6.0, [0.9, 0.1, 0.1, 1.0]);
    quad(&mut far_first, toward_eye * 20.0, 14.0, [0.1, 0.9, 0.3, 1.0]);
    renderer.write_gizmo(&harness.device, &harness.queue, &far_first, view_projection(distance));
    let far_first_frame = harness.frame_with_gizmo(&renderer);
    dump("gizmo-self-occlusion", &far_first_frame);

    let differing = near_first_frame
        .as_chunks::<4>()
        .0
        .iter()
        .zip(far_first_frame.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    assert_eq!(
        differing, 0,
        "the gizmo pass is resolving overlap by draw order rather than by depth: \
         {differing} pixels changed when the two quads were submitted the other way round"
    );
}
