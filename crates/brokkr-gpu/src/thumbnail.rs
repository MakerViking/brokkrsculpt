// SPDX-License-Identifier: AGPL-3.0-only

//! The thumbnail atlas: one 84 x 84 layer per body, rendered offscreen and
//! blitted into a panel row.
//!
//! # Why the pictures live here and not in a second renderer
//!
//! `iced_wgpu` stores one pipeline per PRIMITIVE type -- `Storage::has::<P>()`
//! and `store::<P, _>` key on `TypeId::of::<T>()` where `T` is the primitive
//! (`iced_wgpu-0.14.0/src/primitive.rs:133`, `:218`). A separate
//! `ThumbnailPrimitive`, even one declaring `type Pipeline = SculptPipeline`,
//! therefore gets its **own** [`crate::SculptRenderer`] with its own
//! [`crate::MeshPool`], and `MeshPool::new` eagerly creates buffer pair 0 --
//! half a gigabyte of VRAM reserved for a pool with nothing in it, rendering
//! blank thumbnails forever. "Surely the thumbnail wants its own primitive" is
//! exactly the tidy-up a future reader attempts; it does not, and this is why.
//!
//! So there is one primitive type with two arms, one renderer, one pool, and
//! this atlas hanging off the side of it.
//!
//! # Why the 3D render is offscreen and the row is a flat blit
//!
//! iced's render pass is built with `depth_stencil_attachment: None`
//! (`iced_wgpu-0.14.0/src/lib.rs:452`, `:526`, `:619`), so depth-tested 3D can
//! never ride it. The real render goes into a layer of this atlas from a
//! command encoder of its own, submitted from inside `prepare`; the row then
//! draws a textured quad inside iced's existing pass, which costs no extra
//! render pass per row. iced submits one encoder per frame after every
//! `prepare` has run (`lib.rs:140-147`, `:175`), so the blit always samples
//! pixels the thumbnail pass has already written, whatever the layer order.
//!
//! # The format is taken from the pipeline, never named
//!
//! [`crate::SculptRenderer`]'s pipeline is built against the target format it
//! was handed, and binding it in a pass whose colour attachment has a different
//! format is an `IncompatibleColorAttachment` validation error at
//! `set_pipeline` -- which, under wgpu's default uncaptured-error handler,
//! kills the process. iced picks the first sRGB format the surface reports,
//! which on Linux/Vulkan is normally `Bgra8UnormSrgb`, while both GPU harnesses
//! in this workspace used to pin `Rgba8UnormSrgb`. A hardcoded atlas format
//! would therefore have passed every test here and killed the application on
//! the common Linux configuration 200 ms after the first stroke. The atlas is
//! built from the stored format, `render_thumbnail` asserts they still agree,
//! and `offscreen.rs` runs its thumbnail tests over both formats.

use brokkr_core::MAX_BODIES;
use glam::{Mat4, Vec3};

use crate::renderer::Uniforms;

/// The side of one cell, in physical pixels.
///
/// **A compile-time constant, deliberately not derived from the scale factor.**
/// The scale factor is visible only inside `Primitive::prepare`, and there is
/// no channel by which the application could learn the atlas had been
/// reallocated -- so a scale-derived cell would blank all 64 pictures on a
/// monitor change with nothing anywhere marking them stale. 84 is 2x
/// supersampled at a 1.5 scale factor and 3x at 1.0, resolved down by a linear
/// sampler. Blender's own data-block icon renderer uses 32.
pub const THUMBNAIL_SIZE: u32 = 84;

/// What an unrendered cell holds, in sRGB bytes, red first.
///
/// This is `brokkr-app`'s `theme::BG_DEEP`, the inset-well colour the
/// placeholder container already draws, repeated here because `brokkr-gpu` must
/// not depend on the application. If the panel's well colour changes, change
/// this with it -- a mismatch shows as a picture in a slightly different hole
/// from the rows either side of it.
///
/// **Opaque, and that matters.** The sculpt pipeline is built with
/// `write_mask: ColorWrites::COLOR` -- red, green, blue, no alpha -- so a cell
/// cleared transparent would keep whatever alpha the clear left everywhere the
/// model drew, and the thumbnail would vanish while an RGB dump of the texture
/// looked perfect. Filling every layer with this at startup is also what
/// delivers the placeholder free: an unrendered cell blits a flat swatch with
/// no bitmask and no shader branch.
pub const THUMBNAIL_BACKGROUND: [u8; 4] = [0x0b, 0x0d, 0x10, 0xff];

/// Depth format for the thumbnail pass. Must match the sculpt pipeline's, since
/// every pipeline in a pass has to agree with the pass's depth attachment.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The direction a thumbnail is viewed from.
///
/// The same one `offscreen.rs` and `benches/render.rs` already use, so a
/// thumbnail and every existing render test agree about what the model looks
/// like. **Never the user's camera:** following it would stale all 64 cells on
/// every orbit frame.
const EYE: Vec3 = Vec3::new(0.45, 0.35, 1.0);

/// How far back the camera sits, as a multiple of the body's bounding radius.
///
/// At a 45 degree vertical field of view and a square frame, the whole sphere
/// of that radius is inside the frustum from `1.0 / tan(22.5 deg)` = 2.41
/// radii. 3.0 is the same framing the render bench calls "whole model in view",
/// and the margin is what keeps a long thin body off the edge of its cell.
const FRAMING: f32 = 3.0;

/// The matrices that frame one body's box in its cell.
///
/// Returns `(view_projection, view)`, because the matcap is indexed by
/// view-space normal and so the shader needs both.
fn framing(bounds: (Vec3, Vec3)) -> (Mat4, Mat4) {
    let (low, high) = bounds;
    let centre = (low + high) * 0.5;
    // Clamped away from zero so a body one voxel across still gets a camera
    // rather than a division by nothing.
    let radius = ((high - low).length() * 0.5).max(1.0e-3);
    let distance = radius * FRAMING;

    let view =
        glam::camera::rh::view::look_at_mat4(centre + EYE.normalize() * distance, centre, Vec3::Y);
    let projection = glam::camera::rh::proj::directx::perspective(
        45f32.to_radians(),
        1.0,
        radius * 0.01,
        distance + radius * 4.0,
    );
    (projection * view, view)
}

/// One channel of [`THUMBNAIL_BACKGROUND`] as the clear value wgpu wants.
///
/// A clear value is given in the texture's colour space **without** the
/// transfer function applied, so an sRGB target encodes it on the way in and a
/// byte handed over raw would come out visibly lighter than the same byte
/// written with `write_texture`. Decoding here is what makes a rendered cell
/// and an unrendered one the same colour where the model does not cover them.
fn clear_channel(channel: u8, srgb: bool) -> f64 {
    let value = f64::from(channel) / 255.0;
    if !srgb {
        return value;
    }
    if value <= 0.04045 { value / 12.92 } else { ((value + 0.055) / 1.055).powf(2.4) }
}

/// [`THUMBNAIL_BACKGROUND`] in the byte order this format stores.
///
/// `write_texture` copies bytes straight in, so a BGRA texture needs them
/// swapped or every placeholder comes out blue. Public because the readback
/// test has to un-swap them to compare against the constant.
pub fn background_texel(format: wgpu::TextureFormat) -> [u8; 4] {
    let [r, g, b, a] = THUMBNAIL_BACKGROUND;
    match format {
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => [b, g, r, a],
        _ => [r, g, b, a],
    }
}

/// The offscreen pictures, one layer per body.
#[derive(Debug)]
pub struct ThumbnailAtlas {
    format: wgpu::TextureFormat,
    texture: wgpu::Texture,
    /// One view per layer. The same view serves as the render attachment when
    /// the cell is redrawn and as the sampled texture when a row blits it --
    /// legal because the two happen in different passes on different
    /// submissions, and cheaper than keeping two sets of 64.
    layers: Vec<wgpu::TextureView>,
    /// The blit's bind group per layer, built once. The alternative -- one bind
    /// group over the array plus a uniform naming the layer -- would be a
    /// buffer write per row per frame, which is exactly what
    /// "nothing in a panel row may compute per frame" forbids.
    blits: Vec<wgpu::BindGroup>,
    /// Shared by every cell: the pass is 84 x 84 whichever layer it targets.
    depth: wgpu::TextureView,
    blit_pipeline: wgpu::RenderPipeline,
    /// **The viewport's uniform buffer must not be shared, and this is
    /// invisible to every headless test.** iced runs `prepare` for every
    /// primitive before `render` for any of them
    /// (`iced_wgpu-0.14.0/src/lib.rs:146-147`), so a thumbnail that wrote its
    /// matrix into the one 144-byte buffer would leave the MAIN VIEWPORT
    /// drawing at the thumbnail's camera for that frame. A second buffer and a
    /// second bind group over the same layout, not a dynamic offset: the sculpt
    /// layout declares `has_dynamic_offset: false`, so an offset would be a
    /// validation error, and flipping the flag would invalidate the viewport's
    /// own offset-free `set_bind_group` instead.
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ThumbnailAtlas {
    /// Build the atlas, and fill every layer with the placeholder colour.
    ///
    /// `sculpt_layout`, `matcap` and `matcap_sampler` are the viewport's own,
    /// passed in rather than rebuilt: the second bind group has to be over the
    /// *same* layout as the sculpt pipeline, and the matcap is what makes a
    /// thumbnail look like the viewport rather than like a different renderer.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        sculpt_layout: &wgpu::BindGroupLayout,
        matcap: &wgpu::TextureView,
        matcap_sampler: &wgpu::Sampler,
    ) -> Self {
        let layer_count = MAX_BODIES as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brokkr thumbnail atlas"),
            size: wgpu::Extent3d {
                width: THUMBNAIL_SIZE,
                height: THUMBNAIL_SIZE,
                depth_or_array_layers: layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // The placeholder, written once. 64 layers of 84 x 84 is 1.72 MiB, so
        // one staging copy of the lot is cheaper than any cleverness.
        let texel = background_texel(format);
        let filled: Vec<u8> = texel
            .iter()
            .copied()
            .cycle()
            .take((THUMBNAIL_SIZE * THUMBNAIL_SIZE * layer_count * 4) as usize)
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &filled,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(THUMBNAIL_SIZE * 4),
                rows_per_image: Some(THUMBNAIL_SIZE),
            },
            wgpu::Extent3d {
                width: THUMBNAIL_SIZE,
                height: THUMBNAIL_SIZE,
                depth_or_array_layers: layer_count,
            },
        );

        let depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("brokkr thumbnail depth"),
                size: wgpu::Extent3d {
                    width: THUMBNAIL_SIZE,
                    height: THUMBNAIL_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brokkr thumbnail uniforms"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brokkr thumbnail bind group"),
            layout: sculpt_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(matcap),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(matcap_sampler),
                },
            ],
        });

        // --- the blit ---------------------------------------------------------
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brokkr thumbnail sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            // 84 px resolved down to a 28 px row: without this the row would
            // point-sample one texel in nine and a thin limb would flicker as
            // the list scrolled.
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brokkr thumbnail blit layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mut layers = Vec::with_capacity(layer_count as usize);
        let mut blits = Vec::with_capacity(layer_count as usize);
        for layer in 0..layer_count {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("brokkr thumbnail cell"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            blits.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("brokkr thumbnail blit bind group"),
                layout: &blit_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            }));
            layers.push(view);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brokkr thumbnail blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brokkr thumbnail blit pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("brokkr thumbnail blit pipeline layout"),
                bind_group_layouts: &[&blit_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            // **`None`, and it is not an oversight.** iced's pass has no depth
            // attachment, and a pipeline that declared one could not be bound
            // in it at all.
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self { format, texture, layers, blits, depth, blit_pipeline, uniform_buffer, bind_group }
    }

    /// The format the atlas was built in, for the assertion in
    /// `SculptRenderer::render_thumbnail`.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// How many cells there are. Fixed at [`MAX_BODIES`].
    pub fn cells(&self) -> u32 {
        self.layers.len() as u32
    }

    /// The atlas texture, so a test can copy a cell back and look at it.
    ///
    /// There is no other way to see a thumbnail from a test, and "the picture
    /// is there" is the whole claim this feature makes.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Open the pass that draws one body into `cell`, cleared to the
    /// placeholder colour. `None` when the cell is out of range.
    pub(crate) fn begin_cell<'pass>(
        &'pass self,
        encoder: &'pass mut wgpu::CommandEncoder,
        cell: u32,
    ) -> Option<wgpu::RenderPass<'pass>> {
        let view = self.layers.get(cell as usize)?;
        let srgb = self.format.is_srgb();
        let [r, g, b, _] = THUMBNAIL_BACKGROUND;
        Some(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("brokkr thumbnail pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Cleared, never loaded: the previous picture of this body
                    // must not show through the gaps in the new one.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear_channel(r, srgb),
                        g: clear_channel(g, srgb),
                        b: clear_channel(b, srgb),
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        }))
    }

    /// The bind group the sculpt pipeline reads while drawing a thumbnail.
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Put one body's camera in the thumbnail's OWN uniform buffer.
    ///
    /// **The mask is deliberately not tinted here.** A row's picture is an
    /// identity -- "which body is this" -- and the tint is a view preference of
    /// the viewport, switchable and adjustable there; a thumbnail that followed
    /// it would change what a body looks like in the list because of something
    /// the user did to the model on screen. The off-screen mask signal is the
    /// standing card's `+2 masked, hidden` count, and the badge the plan
    /// reserves for a cell, neither of which is this.
    pub(crate) fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        bounds: (Vec3, Vec3),
        srgb_target: bool,
    ) {
        let (view_projection, view) = framing(bounds);
        let uniforms = Uniforms {
            view_projection: view_projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            srgb_target: u32::from(srgb_target),
            mask_inverted: 0,
            mask_tint: 0.0,
            padding: [0; 1],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw one cell into iced's already-open pass. `false` when the cell is
    /// out of range, which asks iced for a `render` call this primitive does
    /// not want -- so the range check is also the "draw nothing" answer.
    pub(crate) fn blit(&self, pass: &mut wgpu::RenderPass<'_>, cell: u32) -> bool {
        let Some(bind_group) = self.blits.get(cell as usize) else {
            return false;
        };
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole of a body's box has to land inside its cell, or a thumbnail is
    /// a close-up of the middle of the model.
    ///
    /// Checked on the eight corners of the box rather than on its centre,
    /// because the failure mode is a corner poking out of a frame whose centre
    /// is perfectly placed.
    #[test]
    fn every_corner_of_a_bodys_box_lands_inside_its_cell() {
        // A box that is neither cubic nor centred on the world origin: bodies
        // are placed off origin deliberately, and a framing that measured from
        // the origin would put this one off the edge.
        for bounds in [
            (Vec3::new(-30.0, -30.0, -30.0), Vec3::new(30.0, 30.0, 30.0)),
            (Vec3::new(90.0, -5.0, 12.0), Vec3::new(140.0, 3.0, 18.0)),
            (Vec3::splat(-0.1), Vec3::splat(0.1)),
        ] {
            let (view_projection, _) = framing(bounds);
            let (low, high) = bounds;
            for corner in 0..8 {
                let point = Vec3::new(
                    if corner & 1 == 0 { low.x } else { high.x },
                    if corner & 2 == 0 { low.y } else { high.y },
                    if corner & 4 == 0 { low.z } else { high.z },
                );
                let clip = view_projection * point.extend(1.0);
                assert!(clip.w > 0.0, "corner {corner} of {bounds:?} is behind the camera");
                let ndc = clip.truncate() / clip.w;
                assert!(
                    ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
                    "corner {corner} of {bounds:?} lands at {ndc:?}, outside the cell"
                );
                assert!(
                    (0.0..=1.0).contains(&ndc.z),
                    "corner {corner} of {bounds:?} is outside the depth range at {}",
                    ndc.z
                );
            }
        }
    }

    /// A body one voxel across still gets a camera rather than a division by
    /// nothing. An empty body is filtered out before a request is ever made,
    /// but a body that has been carved down to a single brick is not.
    #[test]
    fn a_degenerate_box_still_produces_a_finite_camera() {
        let (view_projection, view) = framing((Vec3::ZERO, Vec3::ZERO));
        for value in view_projection.to_cols_array().iter().chain(view.to_cols_array().iter()) {
            assert!(value.is_finite(), "the framing produced {value}");
        }
    }

    /// The placeholder byte order follows the texture, or every unrendered cell
    /// is blue on the configuration iced actually picks on Linux.
    #[test]
    fn the_placeholder_is_swapped_for_a_bgra_texture_and_not_for_an_rgba_one() {
        let [r, g, b, a] = THUMBNAIL_BACKGROUND;
        assert_eq!(background_texel(wgpu::TextureFormat::Rgba8UnormSrgb), [r, g, b, a]);
        assert_eq!(background_texel(wgpu::TextureFormat::Bgra8UnormSrgb), [b, g, r, a]);
    }

    /// An sRGB clear has to be decoded and a linear one must not be, or a
    /// rendered cell and an unrendered one are two different greys.
    #[test]
    fn the_clear_value_is_decoded_only_for_an_srgb_target() {
        // Untouched on a linear target, whatever the value.
        assert!((clear_channel(0x0b, false) - 11.0 / 255.0).abs() < 1.0e-9);
        assert!((clear_channel(0x02, false) - 2.0 / 255.0).abs() < 1.0e-9);

        // 0x02 is under the 0.04045 knee, so the linear segment applies...
        assert!((clear_channel(0x02, true) - (2.0 / 255.0) / 12.92).abs() < 1.0e-9);
        // ...and 0x0b, which is what the panel's well actually is, is just over
        // it, so the gamma segment does. Getting that boundary the wrong way
        // round is a barely visible shade, which is why it is pinned by number.
        let expected = ((11.0 / 255.0 + 0.055) / 1.055f64).powf(2.4);
        assert!((clear_channel(0x0b, true) - expected).abs() < 1.0e-9);

        // Decoding always darkens, and the ends are fixed points either way.
        assert!(clear_channel(0x0b, true) < clear_channel(0x0b, false));
        assert!((clear_channel(0, true) - 0.0).abs() < 1.0e-9);
        assert!((clear_channel(255, true) - 1.0).abs() < 1.0e-6);
    }
}
