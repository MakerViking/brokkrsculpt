// SPDX-License-Identifier: AGPL-3.0-only

//! The sculpt viewport renderer: pipeline, uniforms, depth buffer and the draw
//! call over the mesh pool.
//!
//! This crate knows nothing about the UI toolkit. It is handed a device, a
//! queue and a target format, and it draws into whatever texture view it is
//! given. The Iced glue that satisfies the `shader` widget's traits lives in
//! `brokkr-app`.

use brokkr_core::{BrickCoord, BrickMesh, Vertex};
use bytemuck::{Pod, Zeroable};

use crate::frustum::Frustum;
use crate::matcap;
use crate::mesh_pool::{MeshPool, PoolStats};
use crate::overlay::{OverlayBatch, OverlayRenderer};

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
/// as well. That is why the tail padding is three scalars on both sides rather
/// than one vector: a `vec3<u32>` there would make the shader struct 160 bytes
/// against this type's 144, and wgpu rejects the mismatch only at draw time.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub view_projection: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub srgb_target: u32,
    pub padding: [u32; 3],
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
            padding: [0; 3],
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
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    depth: DepthBuffer,
    pool: MeshPool,
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
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
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
            bind_group,
            uniform_buffer,
            depth: DepthBuffer::new(device, 1, 1),
            pool: MeshPool::new(device),
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
    pub fn upload_brick(&mut self, queue: &wgpu::Queue, coord: BrickCoord, mesh: &BrickMesh) {
        self.pool.upload(queue, coord, mesh);
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, uniforms: &Uniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
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
        self.overlay.vertex_count() + self.cube.vertex_count()
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

    /// Draw the navigation cube into its corner.
    ///
    /// A pass of its own, because the cube has to be depth tested against
    /// itself and nothing else: it is drawn over the model wherever it sits, and
    /// the sculpt's depth values in that corner would otherwise clip it.
    ///
    /// The depth clear covers the whole attachment rather than the scissor rect,
    /// which is how wgpu defines it. That is a full screen clear for a 92 pixel
    /// box, and it is still cheaper and far less fragile than the alternatives
    /// (a second depth texture cannot be used, since every attachment in a pass
    /// must share the colour attachment's size).
    pub fn render_cube(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip: PixelRect,
    ) {
        if clip.width == 0 || clip.height == 0 || self.cube.vertex_count() == 0 {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("brokkr navigation cube pass"),
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
        // Solid: the cube is opaque and convex, so it must occlude its own far
        // faces rather than blending with them.
        self.cube.draw(&mut pass, true);
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
        self.pool.draw(&mut pass, frustum);

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
