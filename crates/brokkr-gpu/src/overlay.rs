// SPDX-License-Identifier: AGPL-3.0-only

//! Flat coloured geometry drawn over the sculpt.
//!
//! Three consumers share this one pipeline: the brush cursor ring, the mirror
//! planes, and the navigation cube. Building a mechanism each would have been
//! three places to get depth and blending wrong, which is the part of this that
//! is actually difficult — the geometry is a handful of circles and quads.
//!
//! # Where the geometry comes from
//!
//! `brokkr-app` builds it, every frame, and hands it over through
//! `SharedFrame` exactly the way brick meshes already travel. It is a few
//! hundred vertices of interface, so there is nothing to gain from generating
//! it here, and `brokkr-core` must not know that a screen exists at all.
//!
//! # Why three pipelines rather than one
//!
//! They differ only in how they treat depth, and each difference is load
//! bearing:
//!
//! * Lines — the brush ring lies *on* the surface it measures, so it needs a
//!   depth bias pulling it toward the viewer or it is z-fought into dashes.
//! * Surfaces — a mirror plane is translucent and passes through the model, so
//!   it tests depth but must **not write** it, or it would occlude whatever is
//!   drawn after it and self-occlude where it folds.
//! * Solids — the navigation cube is opaque and convex and must occlude its own
//!   far faces, so it both tests and writes depth in its own pass.

use glam::Mat4;

/// One overlay vertex: a world position and a colour of its own.
///
/// The colour is **linear**, not sRGB. The shader encodes it only when the
/// target does not, which is the same rule `sculpt.wgsl` follows, so an overlay
/// and the model beside it cannot end up in different colour spaces.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayVertex {
    pub position: [f32; 3],
    pub colour: [f32; 4],
}

impl OverlayVertex {
    pub fn new(position: glam::Vec3, colour: [f32; 4]) -> Self {
        Self { position: position.to_array(), colour }
    }
}

/// A frame's worth of overlay geometry.
#[derive(Debug, Default, Clone)]
pub struct OverlayBatch {
    /// Vertex pairs, one line each.
    pub lines: Vec<OverlayVertex>,
    /// Vertex triples, one translucent triangle each.
    pub surfaces: Vec<OverlayVertex>,
}

impl OverlayBatch {
    pub fn clear(&mut self) {
        self.lines.clear();
        self.surfaces.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty() && self.surfaces.is_empty()
    }

    pub fn push_line(&mut self, from: glam::Vec3, to: glam::Vec3, colour: [f32; 4]) {
        self.lines.push(OverlayVertex::new(from, colour));
        self.lines.push(OverlayVertex::new(to, colour));
    }

    /// A quad as two triangles, wound `a b c d` around its rim.
    pub fn push_quad(
        &mut self,
        a: glam::Vec3,
        b: glam::Vec3,
        c: glam::Vec3,
        d: glam::Vec3,
        colour: [f32; 4],
    ) {
        for point in [a, b, c, a, c, d] {
            self.surfaces.push(OverlayVertex::new(point, colour));
        }
    }
}

/// Per frame constants. Must match `Uniforms` in `overlay.wgsl` byte for byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub srgb_target: u32,
    pub padding: [u32; 3],
}

const _: () = assert!(
    size_of::<OverlayUniforms>() == 80,
    "OverlayUniforms must stay 80 bytes to match the WGSL struct in overlay.wgsl"
);

/// A vertex buffer that grows to fit and never shrinks.
///
/// Overlay geometry is rebuilt every frame but its size barely changes, so
/// after the first few frames this stops reallocating entirely and the per
/// frame path is a single `write_buffer`.
#[derive(Debug)]
struct GrowBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
    len: usize,
    label: &'static str,
}

impl GrowBuffer {
    fn new(device: &wgpu::Device, label: &'static str, capacity: usize) -> Self {
        Self { buffer: Self::allocate(device, label, capacity), capacity, len: 0, label }
    }

    fn allocate(device: &wgpu::Device, label: &'static str, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (capacity.max(1) * size_of::<OverlayVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, vertices: &[OverlayVertex]) {
        self.len = vertices.len();
        if vertices.is_empty() {
            return;
        }
        if vertices.len() > self.capacity {
            // Doubling rather than exact fit, so a slowly growing overlay does
            // not reallocate every frame.
            self.capacity = vertices.len().next_power_of_two();
            self.buffer = Self::allocate(device, self.label, self.capacity);
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(vertices));
    }
}

/// Draws [`OverlayBatch`] geometry over an existing render pass.
#[derive(Debug)]
pub struct OverlayRenderer {
    lines_pipeline: wgpu::RenderPipeline,
    surfaces_pipeline: wgpu::RenderPipeline,
    solids_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    lines: GrowBuffer,
    surfaces: GrowBuffer,
    srgb_target: bool,
}

impl OverlayRenderer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brokkr overlay uniforms"),
            size: size_of::<OverlayUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brokkr overlay bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brokkr overlay bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brokkr overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brokkr overlay pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let build = |label: &str,
                     topology: wgpu::PrimitiveTopology,
                     blend: Option<wgpu::BlendState>,
                     depth_write: bool,
                     bias: wgpu::DepthBiasState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: size_of::<OverlayVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend,
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    // Overlays are looked at from both sides: a mirror plane
                    // seen edge on, a cube face from behind while the camera
                    // swings past.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: depth_write,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias,
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };

        // A ring drawn exactly on the surface it measures is coplanar with it,
        // so without a bias toward the viewer it is z-fought into dashes.
        let toward_viewer = wgpu::DepthBiasState { constant: -4, slope_scale: -1.0, clamp: 0.0 };

        Self {
            lines_pipeline: build(
                "brokkr overlay lines",
                wgpu::PrimitiveTopology::LineList,
                Some(wgpu::BlendState::ALPHA_BLENDING),
                false,
                toward_viewer,
            ),
            surfaces_pipeline: build(
                "brokkr overlay surfaces",
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::BlendState::ALPHA_BLENDING),
                // Translucent and passing through the model: writing depth
                // would occlude whatever is drawn next and self-occlude.
                false,
                Default::default(),
            ),
            solids_pipeline: build(
                "brokkr overlay solids",
                wgpu::PrimitiveTopology::TriangleList,
                None,
                // Opaque and convex: it has to occlude its own far faces.
                true,
                Default::default(),
            ),
            bind_group,
            uniform_buffer,
            lines: GrowBuffer::new(device, "brokkr overlay lines buffer", 256),
            surfaces: GrowBuffer::new(device, "brokkr overlay surfaces buffer", 256),
            srgb_target: target_format.is_srgb(),
        }
    }

    /// Replace this frame's geometry and the matrix it is drawn with.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &OverlayBatch,
        view_projection: Mat4,
    ) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&OverlayUniforms {
                view_projection: view_projection.to_cols_array_2d(),
                srgb_target: u32::from(self.srgb_target),
                padding: [0; 3],
            }),
        );
        self.lines.write(device, queue, &batch.lines);
        self.surfaces.write(device, queue, &batch.surfaces);
    }

    /// Draw into an already configured pass, whose viewport and scissor are the
    /// caller's business.
    ///
    /// Surfaces before lines, so a ring reads over a plane rather than under it.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, solid_surfaces: bool) {
        pass.set_bind_group(0, &self.bind_group, &[]);

        if self.surfaces.len > 0 {
            pass.set_pipeline(if solid_surfaces {
                &self.solids_pipeline
            } else {
                &self.surfaces_pipeline
            });
            pass.set_vertex_buffer(0, self.surfaces.buffer.slice(..));
            pass.draw(0..self.surfaces.len as u32, 0..1);
        }

        if self.lines.len > 0 {
            pass.set_pipeline(&self.lines_pipeline);
            pass.set_vertex_buffer(0, self.lines.buffer.slice(..));
            pass.draw(0..self.lines.len as u32, 0..1);
        }
    }

    /// Vertices drawn last frame, for the debug overlay.
    pub fn vertex_count(&self) -> usize {
        self.lines.len + self.surfaces.len
    }
}
