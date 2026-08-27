// SPDX-License-Identifier: AGPL-3.0-only

//! The sculpt viewport renderer: pipeline, uniforms, depth buffer and the draw
//! call over the mesh pool.
//!
//! This crate knows nothing about the UI toolkit. It is handed a device, a
//! queue and a target format, and it draws into whatever texture view it is
//! given. The Iced glue that satisfies the `shader` widget's traits lives in
//! `brokkr-app`.

use brokkr_core::{BrickMesh, NodeId, Vertex};
use bytemuck::{Pod, Zeroable};

use crate::frustum::Frustum;
use crate::matcap;
use crate::mesh_pool::{MaskPolarity, MeshPool, PoolStats, SlotKey};
use crate::overlay::{OverlayBatch, OverlayRenderer};
use crate::thumbnail::ThumbnailAtlas;

/// Depth format. Depth32Float is universally supported and precise enough that
/// a sculpt at arm's length shows no z fighting.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The depth format overlay pipelines have to be built against, since every
/// pipeline in a pass must agree with the pass's depth attachment.
pub const OVERLAY_DEPTH_FORMAT: wgpu::TextureFormat = DEPTH_FORMAT;

/// Per frame shader constants.
///
/// The layout has to match the `Uniforms` struct in `sculpt.wgsl` byte for
/// byte. Uniform address space rounds a struct up to its largest member
/// alignment, which the two matrices set to 16, and it aligns a `vec3` to 16
/// as well. That is why the tail is four scalars on both sides rather than a
/// vector: a `vec3<u32>` there would make the shader struct 160 bytes against
/// this type's 144, and wgpu rejects the mismatch only at draw time.
///
/// **The tail is four NAMED scalars and not a `[u32; 3]` padding array**, and
/// the naming is what makes the mask flags reachable. `mask_tint` is a float
/// while the array was `u32`, so parking it in the array would have meant a
/// bit-cast on both sides; worse, four construction sites hardcoded
/// `padding: [0; 3]`, so a flag put there would have been silently zero at
/// every one of them until all four were found. Splitting the array made that
/// a compile error instead.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub view_projection: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub srgb_target: u32,
    /// Non zero when protection is read inverted, which the shader applies to
    /// the vertex attribute.
    ///
    /// **Resolved here rather than baked into the mesh**, which is what makes
    /// Invert and Mask All one write of this word instead of a remesh of the
    /// whole body -- 71 ms on the dragon and roughly 475 ms at the brick count
    /// the pool is sized for.
    ///
    /// **The mask is per BODY, so one word cannot answer for a whole document,
    /// and it does not have to.** The application publishes the ACTIVE body's
    /// polarity here and the SET of bodies that disagree with it separately;
    /// [`MeshPool::draw`] already walks one bucket per body, so it binds a
    /// second group that differs only in this word for those. See
    /// [`MaskPolarity`] and [`MeshPool::set_opposite_polarity`]. Before that
    /// existed, a Mask All on a body that was not active drew that body's
    /// stored zeros as free -- a fully protected body with no tint on it, which
    /// is the failure the whole masking design is arranged around.
    ///
    /// The thumbnail pass draws at `mask_tint: 0.0` and so is untouched by any
    /// of this.
    pub mask_inverted: u32,
    /// How strongly the mask is tinted, 0..1. Zero draws the body exactly as an
    /// unmasked one.
    ///
    /// A VIEW strength and never a protection strength: 3D-Coat has to warn in
    /// its own documentation that its Freeze Opacity "does not affect the
    /// freezing strength of the current stroke", which is a documented
    /// confusion in a shipping professional tool. Nothing downstream of this
    /// word can change what a stroke does.
    ///
    /// It reaches zero, because the `show mask` toggle drives it there. What
    /// keeps that safe is that the toggle governs the tint and nothing else --
    /// the application's standing mask card is unconditional -- so "a mask is
    /// active and nothing on screen says so" is still unreachable.
    pub mask_tint: f32,
    pub padding: [u32; 1],
}

const _: () = assert!(
    size_of::<Uniforms>() == 144,
    "Uniforms must stay 144 bytes to match the WGSL struct in sculpt.wgsl"
);

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_projection: glam::Mat4::IDENTITY.to_cols_array_2d(),
            view: glam::Mat4::IDENTITY.to_cols_array_2d(),
            srgb_target: 1,
            mask_inverted: 0,
            // Tinted, not untinted. The default has to fail in the direction
            // where a mask that is there is visible: the failure this whole
            // design is arranged around is a masked surface reading as a broken
            // brush, and a defaulted-to-zero tint is exactly that.
            mask_tint: 1.0,
            padding: [0; 1],
        }
    }
}

/// A depth buffer sized to the current render target.
#[derive(Debug)]
struct DepthBuffer {
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl DepthBuffer {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brokkr viewport depth"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            width: width.max(1),
            height: height.max(1),
        }
    }
}

/// Draws the sculpted mesh with matcap shading.
#[derive(Debug)]
pub struct SculptRenderer {
    pipeline: wgpu::RenderPipeline,
    overlay: OverlayRenderer,
    /// A second overlay for the navigation cube, which needs its own matrix and
    /// its own geometry in its own pass.
    ///
    /// A whole second instance rather than two slots inside one: it costs three
    /// extra pipeline objects at startup and saves plumbing a slot index through
    /// every call. Nothing here is hot enough for that trade to matter.
    cube: OverlayRenderer,
    /// A third overlay for the transform gizmo.
    ///
    /// A whole instance again, for the reason `cube` gives, and it earns it for
    /// a second reason: the gizmo is drawn in the SCULPT's matrix but over the
    /// sculpt's depth, so it can share neither the cube's matrix nor the
    /// ring's pass. What it does share is the mechanism -- see
    /// [`SculptRenderer::overlay_pass`], which the cube's own pass was
    /// generalised into rather than copied.
    gizmo: OverlayRenderer,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    /// The same uniforms with [`Uniforms::mask_inverted`] complemented, and the
    /// group that binds them.
    ///
    /// **A whole second buffer rather than a dynamic offset into one**, because
    /// the layout declares `has_dynamic_offset: false` and a 144-byte struct
    /// would have to be padded to the 256-byte minimum alignment to use one.
    /// Two buffers of 144 bytes is the cheaper and the plainer answer; they are
    /// written together in [`SculptRenderer::write_uniforms`] so they cannot
    /// drift apart. [`MeshPool::draw`] picks between them one body at a time --
    /// see [`MaskPolarity`].
    opposite_uniform_buffer: wgpu::Buffer,
    opposite_bind_group: wgpu::BindGroup,
    depth: DepthBuffer,
    pool: MeshPool,
    /// The offscreen pictures the body panel's rows blit. See
    /// [`crate::thumbnail`], whose header is where the design lives.
    thumbnails: ThumbnailAtlas,
    /// **The format, kept and not merely reduced to `srgb_target`.** The
    /// thumbnail atlas has to be created in exactly the format this renderer's
    /// pipeline was built against: binding the pipeline in a pass whose colour
    /// attachment differs is an `IncompatibleColorAttachment` error at
    /// `set_pipeline`, and wgpu's default handler turns that into a dead
    /// process. iced normally picks `Bgra8UnormSrgb` on Linux, so a hardcoded
    /// atlas format would have been green on every test machine here and fatal
    /// in the application.
    format: wgpu::TextureFormat,
    srgb_target: bool,
}

impl SculptRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let matcap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brokkr matcap"),
            size: wgpu::Extent3d {
                width: matcap::MATCAP_SIZE,
                height: matcap::MATCAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &matcap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &matcap::clay(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(matcap::MATCAP_SIZE * 4),
                rows_per_image: Some(matcap::MATCAP_SIZE),
            },
            wgpu::Extent3d {
                width: matcap::MATCAP_SIZE,
                height: matcap::MATCAP_SIZE,
                depth_or_array_layers: 1,
            },
        );

        let matcap_view = matcap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brokkr matcap sampler"),
            // Clamping matters: a normal pointing exactly sideways lands on the
            // very edge of the disc, and wrapping there would fetch the colour
            // from the opposite side.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brokkr viewport uniforms"),
            size: size_of::<Uniforms>() as u64,
            // `COPY_SRC` so that a test can read back what is actually in here.
            // See `SculptRenderer::read_viewport_uniforms`: the failure it
            // guards against -- a thumbnail's camera landing in the viewport's
            // buffer -- is invisible to every other kind of check.
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let opposite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brokkr viewport uniforms, opposite polarity"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brokkr viewport bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brokkr viewport bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&matcap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let opposite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brokkr viewport bind group, opposite polarity"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: opposite_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&matcap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brokkr sculpt shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sculpt.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brokkr viewport pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brokkr sculpt pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    },
                    // The pool's third buffer, at the same block offsets as the
                    // first. `Unorm8x4` because a vertex buffer's stride must be
                    // a multiple of four, so one byte would cost four anyway;
                    // byte 0 is the mask and the rest are reserved for colour.
                    wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![2 => Unorm8x4],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Sculpting means looking into cavities and through openings
                // that are mid cut, so back faces have to stay visible. The
                // fragment shader flips their normals instead. Culling belongs
                // with the M2 batching work, once the mesh is reliably closed.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            overlay: OverlayRenderer::new(device, target_format, DEPTH_FORMAT),
            cube: OverlayRenderer::new(device, target_format, DEPTH_FORMAT),
            gizmo: OverlayRenderer::new(device, target_format, DEPTH_FORMAT),
            bind_group,
            uniform_buffer,
            opposite_uniform_buffer,
            opposite_bind_group,
            depth: DepthBuffer::new(device, 1, 1),
            pool: MeshPool::new(device),
            thumbnails: ThumbnailAtlas::new(
                device,
                queue,
                target_format,
                &layout,
                &matcap_view,
                &sampler,
            ),
            format: target_format,
            srgb_target: target_format.is_srgb(),
        }
    }

    /// A one line description of the target this renderer draws into.
    ///
    /// wgpu's adapter belongs to iced here, so the exact GPU name is not
    /// reachable; what is reachable is what the pipeline was built against,
    /// which is still the first thing a bug report needs.
    pub fn adapter_summary(&self) -> String {
        format!(
            "wgpu {}, {} target",
            env!("CARGO_PKG_VERSION"),
            if self.srgb_target { "sRGB" } else { "linear" }
        )
    }

    /// Whether the target format encodes sRGB itself, which the shader needs to
    /// know so it does not encode twice.
    pub fn target_is_srgb(&self) -> bool {
        self.srgb_target
    }

    /// Throw away every brick in the pool and reset its allocator.
    ///
    /// For a whole-model rebuild only, and the caller must re-upload
    /// everything afterwards. See [`MeshPool::reset`].
    pub fn reset_pool(&mut self) {
        self.pool.reset();
    }

    /// Replace one brick's mesh in the pool.
    ///
    /// The key names the body as well as the brick coordinate, because two
    /// bodies near the world origin share brick coordinates -- see [`SlotKey`].
    /// Takes the device as well as the queue because the pool grows itself a
    /// buffer when the ones it has are full.
    pub fn upload_brick(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: SlotKey,
        mesh: &BrickMesh,
    ) {
        self.pool.upload(device, queue, key, mesh);
    }

    /// Drop every brick one body owns and give its pool space back.
    ///
    /// For a body that has left the document. The caller must have made sure
    /// no upload for that body is still queued -- see `SharedFrame::apply` in
    /// the application, which drains the forget list before the uploads and
    /// throws away any pending upload that names a forgotten body. Applying
    /// them in the other order re-uploads the meshes that were just released,
    /// which draws a sliver of a deleted body forever with no counter moving.
    ///
    /// See [`MeshPool::forget_body`] for what it does to the allocator and to
    /// the overflow banner, and for the remesh the caller owes it afterwards.
    pub fn forget_body(&mut self, body: NodeId) -> usize {
        self.pool.forget_body(body)
    }

    /// Replace the set of bodies the sculpt pass must not draw.
    ///
    /// Wholesale, from the application's one visibility pass. See
    /// [`MeshPool::set_hidden`].
    pub fn set_hidden(&mut self, hidden: &[NodeId]) {
        self.pool.set_hidden(hidden);
    }

    /// Replace the set of bodies drawn with the opposite of
    /// [`Uniforms::mask_inverted`].
    ///
    /// Wholesale, from the same visibility pass that publishes the hidden set
    /// and the uniform this is relative to. See
    /// [`MeshPool::set_opposite_polarity`].
    pub fn set_opposite_polarity(&mut self, opposite: &[NodeId]) {
        self.pool.set_opposite_polarity(opposite);
    }

    /// How many of one body's bricks the pool holds, for tests that have to
    /// tell "gone" from "still there but not drawn".
    pub fn body_bricks(&self, body: NodeId) -> usize {
        self.pool.body_bricks(body)
    }

    /// What the renderer was last told not to draw, for tests that have to
    /// tell "the hidden set arrived" from "the hidden set was published".
    /// See [`MeshPool::hidden_bodies`].
    pub fn hidden_bodies(&self) -> &[NodeId] {
        self.pool.hidden_bodies()
    }

    /// What the renderer was last told draws with the opposite polarity, for
    /// the reason [`SculptRenderer::hidden_bodies`] exists.
    /// See [`MeshPool::opposite_polarity_bodies`].
    pub fn opposite_polarity_bodies(&self) -> &[NodeId] {
        self.pool.opposite_polarity_bodies()
    }

    /// Write the frame's constants, into BOTH polarity buffers.
    ///
    /// The second differs in exactly one word and is written from the same
    /// value, here and nowhere else, so the two can never disagree about the
    /// camera. See [`SculptRenderer::opposite_uniform_buffer`] for why it is a
    /// second buffer and not an offset.
    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &Uniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
        let opposite =
            Uniforms { mask_inverted: u32::from(uniforms.mask_inverted == 0), ..*uniforms };
        queue.write_buffer(&self.opposite_uniform_buffer, 0, bytemuck::bytes_of(&opposite));
    }

    /// What the viewport's uniform buffer actually holds, read back off the GPU.
    ///
    /// **This exists for one test and it is worth the public method.** The
    /// failure it guards against is that a thumbnail render writes its own
    /// camera over the viewport's -- iced runs every `prepare` before any
    /// `render`, so the main viewport would spend that frame drawing at an 84
    /// pixel thumbnail's camera. Nothing else can see it: it is not a panic,
    /// not a validation error and not a wrong pixel in any headless harness,
    /// only one visibly wrong frame in the running application. The test lives
    /// in `brokkr-app`, next to the drain that would cause it, so a
    /// `#[cfg(test)]` here would not reach it.
    ///
    /// Blocks on the device. Never call it from a frame.
    pub fn read_viewport_uniforms(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Uniforms {
        let size = size_of::<Uniforms>() as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brokkr uniform readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brokkr uniform readback"),
        });
        encoder.copy_buffer_to_buffer(&self.uniform_buffer, 0, &readback, 0, size);
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("uniform readback failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");
        let uniforms = *bytemuck::from_bytes::<Uniforms>(&slice.get_mapped_range());
        readback.unmap();
        uniforms
    }

    /// The thumbnail atlas, for the readback in the offscreen tests.
    pub fn thumbnails(&self) -> &ThumbnailAtlas {
        &self.thumbnails
    }

    /// Draw one body into its cell of the thumbnail atlas.
    ///
    /// **Its own encoder and its own `queue.submit`, because `prepare` is handed
    /// no encoder at all.** iced builds one encoder per frame, runs every
    /// `prepare`, then every `render`, then submits once
    /// (`iced_wgpu-0.14.0/src/lib.rs:140-147`, `:175`), so a submission issued
    /// from inside `prepare` executes strictly before the frame's own commands
    /// and the row's blit later in that same frame samples fresh pixels
    /// whatever the layer order.
    ///
    /// `bounds` is the body's world box; the framing is worked out from it in
    /// [`crate::thumbnail`]. Nothing here reads the user's camera, and nothing
    /// here touches the viewport's uniform buffer, the frustum, the overlay or
    /// the `drawn`/`culled` counters -- see [`MeshPool::draw_body`], which
    /// takes no frustum precisely so that the viewport's cannot be passed to it
    /// by accident.
    ///
    /// A cell out of range, or a body with nothing in the pool, leaves a
    /// correctly cleared empty picture rather than a panic: the caller's cell
    /// bookkeeping and the pool's contents are separate pieces of state and
    /// they are allowed to disagree for a frame.
    pub fn render_thumbnail(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell: u32,
        body: NodeId,
        bounds: (glam::Vec3, glam::Vec3),
    ) {
        debug_assert_eq!(
            self.thumbnails.format(),
            self.format,
            "the thumbnail atlas and the sculpt pipeline disagree about the target format, which \
             is a validation error at set_pipeline and a dead process in the application"
        );
        if cell >= self.thumbnails.cells() {
            return;
        }

        self.thumbnails.write_uniforms(queue, bounds, self.srgb_target);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brokkr thumbnail"),
        });
        {
            let Some(mut pass) = self.thumbnails.begin_cell(&mut encoder, cell) else {
                return;
            };
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.thumbnails.bind_group(), &[]);
            self.pool.draw_body(&mut pass, body);
        }
        queue.submit([encoder.finish()]);
    }

    /// Blit one cell into iced's already-open render pass.
    ///
    /// Returns what `Primitive::draw` has to return: `true` when the row was
    /// drawn inside the existing pass, which is the whole point -- zero extra
    /// render passes per row.
    pub fn blit_thumbnail(&self, pass: &mut wgpu::RenderPass<'_>, cell: u32) -> bool {
        self.thumbnails.blit(pass, cell)
    }

    /// Replace this frame's overlay geometry: the brush ring and the mirror
    /// planes. Drawn at the end of the sculpt pass, so it shares the sculpt's
    /// viewport, scissor and depth buffer.
    pub fn write_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &OverlayBatch,
        view_projection: glam::Mat4,
    ) {
        self.overlay.upload(device, queue, batch, view_projection);
    }

    /// Overlay vertices drawn last frame, for the debug readout.
    pub fn overlay_vertices(&self) -> usize {
        self.overlay.vertex_count() + self.cube.vertex_count() + self.gizmo.vertex_count()
    }

    /// Replace the navigation cube's geometry and the matrix it is drawn with.
    pub fn write_cube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &OverlayBatch,
        view_projection: glam::Mat4,
    ) {
        self.cube.upload(device, queue, batch, view_projection);
    }

    /// Replace the transform gizmo's geometry.
    ///
    /// The SCULPT's matrix, not one of its own: the gizmo sits on the body it
    /// moves, in world space, and only its SIZE is held constant on screen --
    /// which `brokkr-app` does by scaling the geometry it builds, so that the
    /// draw and the hit test cannot disagree about how big it is.
    pub fn write_gizmo(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &OverlayBatch,
        view_projection: glam::Mat4,
    ) {
        self.gizmo.upload(device, queue, batch, view_projection);
    }

    /// Draw an overlay OVER the sculpt, in a pass with a depth buffer of its
    /// own.
    ///
    /// **The one always-on-top mechanism, and the reason it is one rather than
    /// two.** All three of `overlay.rs`'s pipelines compare `Less` against the
    /// sculpt's depth, which is right for a ring lying on a surface and wrong
    /// for anything that has to be reachable while buried: the navigation cube
    /// is drawn over whatever is behind it, and a gizmo on a body's centroid is
    /// *inside* the mesh. Clearing depth and keeping `Less` is what buys both,
    /// and it buys correct self-occlusion with it -- an arrowhead over its own
    /// shaft, three rings crossing -- for nothing. A `depth_compare: Always`
    /// variant would need the geometry sorted back to front on the CPU every
    /// time the camera moved.
    ///
    /// The cube's pass was generalised into this rather than copied for it,
    /// which is what `overlay.rs`'s "do not add a fourth mechanism for the next
    /// overlay" asks for: the next overlay uses the third one.
    ///
    /// The depth clear covers the whole attachment rather than the scissor
    /// rect, which is how wgpu defines it. That is a full screen clear for a 92
    /// pixel box, and it is still cheaper and far less fragile than the
    /// alternatives (a second depth texture cannot be used, since every
    /// attachment in a pass must share the colour attachment's size).
    ///
    /// **Correct only while nothing depth-tested is scheduled after it.** Two
    /// of these run per frame now, and a future pass added below them would
    /// draw against a cleared buffer with no test to catch it.
    fn overlay_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip: PixelRect,
        label: &str,
        overlay: &OverlayRenderer,
        solid_surfaces: bool,
    ) {
        if clip.width == 0 || clip.height == 0 || overlay.vertex_count() == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_viewport(
            clip.x as f32,
            clip.y as f32,
            clip.width as f32,
            clip.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(clip.x, clip.y, clip.width, clip.height);
        overlay.draw(&mut pass, solid_surfaces);
    }

    /// Draw the navigation cube into its corner.
    pub fn render_cube(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip: PixelRect,
    ) {
        // Solid: the cube is opaque and convex, so it must occlude its own far
        // faces rather than blending with them.
        self.overlay_pass(encoder, target, clip, "brokkr navigation cube pass", &self.cube, true);
    }

    /// Draw the transform gizmo over the whole viewport.
    ///
    /// Solid for the same reason the cube is: an arrowhead is an opaque cone
    /// and a ring is an opaque band, and both have to occlude their own far
    /// side rather than blending through it.
    pub fn render_gizmo(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip: PixelRect,
    ) {
        self.overlay_pass(encoder, target, clip, "brokkr gizmo pass", &self.gizmo, true);
    }

    /// Make sure the depth buffer matches the render target.
    ///
    /// Every attachment in a render pass must share a size, and the colour
    /// attachment is the whole window rather than just the widget, so this is
    /// sized to the window and the draw is confined with a scissor instead.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.depth.width != width || self.depth.height != height {
            self.depth = DepthBuffer::new(device, width, height);
        }
    }

    pub fn stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// Draw the sculpt into `target`, confined to `clip` in physical pixels.
    ///
    /// The colour attachment loads rather than clears, because the UI has
    /// already drawn into this texture and the viewport is only one region of
    /// it. The viewport's own background comes from the clear coloured quad the
    /// UI draws underneath.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip: PixelRect,
        frustum: &Frustum,
    ) {
        if clip.width == 0 || clip.height == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("brokkr sculpt pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_viewport(
            clip.x as f32,
            clip.y as f32,
            clip.width as f32,
            clip.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(clip.x, clip.y, clip.width, clip.height);
        self.pool.draw(
            &mut pass,
            frustum,
            MaskPolarity { as_published: &self.bind_group, opposite: &self.opposite_bind_group },
        );

        // Last, and inside the same pass: the ring has to depth test against
        // the model it is lying on, and a mirror plane against the model it
        // passes through.
        self.overlay.draw(&mut pass, false);
    }
}

/// A rectangle in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
