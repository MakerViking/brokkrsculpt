// SPDX-License-Identifier: AGPL-3.0-only
//
// Matcap shading for the sculpt viewport. The whole lighting model is a lookup
// into a small image by view space normal, which is why there are no lights,
// no shadows and no material parameters here.

struct Uniforms {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    // Non zero when the render target encodes sRGB itself, in which case the
    // fragment must stay linear. iced picks the surface format, so this cannot
    // be assumed.
    srgb_target: u32,
    // Non zero when protection is read inverted. Resolved here rather than
    // baked into the mesh, which is what makes Invert one uniform write instead
    // of a remesh of the whole body.
    mask_inverted: u32,
    // How strongly the mask is tinted, 0..1. A VIEW strength: nothing about it
    // changes what a stroke does. Zero is the `show mask` toggle switched off,
    // and it draws the body exactly as an unmasked one.
    mask_tint: f32,
    // Non zero when painted slots are drawn in their filament colours. A VIEW
    // toggle like the tint: zero draws every body as bare clay.
    paint_shown: u32,
    // Linear RGB per filament slot, indexed by the stored slot byte. Slot 0 is
    // never read. Sixteen entries of 16 bytes, which keeps the struct's size a
    // multiple of 16 without a padding word; the Rust type asserts the total,
    // and `renderer::tests::the_shader_palette_is_sized_to_palette_slots`
    // reads this literal and the clamp below out of the source, because a
    // Rust-side change to PALETTE_SLOTS would otherwise pass validation with
    // the shader indexing past what it was given.
    palette: array<vec4<f32>, 16>,
}

// Luminance of the matcap's BASE clay colour in linear light, which is what a
// painted pixel divides its shading by: `luma / MATCAP_BASE_LUMA` recovers the
// lighting term the matcap multiplied into the clay, near enough, and the
// filament colour is then lit the way the clay was. One fact in two places;
// `matcap::tests::the_base_luma_the_shader_divides_by_is_the_clays` pins it.
const MATCAP_BASE_LUMA: f32 = 0.384;
// Filament colours are drawn a little under their full value. The lit side of
// the matcap runs to about 1.4x BASE's luminance, so a white filament still
// clips at the specular peak -- as white plastic does -- while the rest of the
// lit quarter keeps its gradient rather than flattening to one value.
const PAINT_EXPOSURE: f32 = 0.8;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var matcap_texture: texture_2d<f32>;
@group(0) @binding(2) var matcap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
    // The STORED protection at this vertex's own lattice cell, 0..1. Byte 0 of
    // the pool's attribute buffer.
    @location(1) mask: f32,
    // The painted filament slot at that cell, byte 1 of the attribute buffer.
    // FLAT, never interpolated: a slot is categorical, and a triangle whose
    // three corners disagree prints in ONE filament. Interpolating would draw
    // a blend of two that no printer can make. On Vulkan, Metal and DX12 the
    // flat value is the FIRST vertex's, which is the rule the 3MF export uses
    // for the same triangle, so what is drawn is what will be printed; wgpu's
    // GL backend takes the LAST vertex, so on that one backend a boundary
    // triangle can draw in the other filament. One triangle wide, at most.
    @location(2) @interpolate(flat) slot: u32,
}

@vertex
fn vertex_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) attributes: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_projection * vec4<f32>(position, 1.0);
    // The view matrix is a rigid transform, so its upper 3x3 rotates normals
    // correctly without an inverse transpose.
    out.view_normal = (uniforms.view * vec4<f32>(normal, 0.0)).xyz;
    out.mask = attributes.r;
    // `Unorm8x4` hands the byte back as `n / 255`; the round undoes it exactly.
    out.slot = u32(round(attributes.g * 255.0));
    return out;
}

@fragment
fn fragment_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var normal = normalize(in.view_normal);

    // Back faces are visible inside cavities and through openings. Flipping the
    // normal there shades them as surfaces rather than as whatever the matcap
    // holds at the mirrored coordinate, which reads as an inside out dent.
    if (!front_facing) {
        normal = -normal;
    }

    // Matcap lookup: the view space normal's xy maps onto the unit disc. The v
    // axis is flipped because texture row 0 is the top of the sphere.
    let uv = vec2<f32>(normal.x * 0.5 + 0.5, 0.5 - normal.y * 0.5);
    var colour = textureSample(matcap_texture, matcap_sampler, uv).rgb;

    // The paint, BEFORE the mask so protection still reads on a painted
    // surface: the clay's albedo is swapped for the filament's and the matcap's
    // lighting kept. A slot past the palette clamps to the last entry, which
    // the application fills with a colour no filament is.
    if (uniforms.paint_shown != 0u && in.slot != 0u) {
        let filament = uniforms.palette[min(in.slot, 15u)].rgb;
        let luma = dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722));
        // Not clamped at 1: the lighting term is allowed to exceed the clay's
        // own, or every painted pixel on the key-lit side collapses to one
        // flat value and the form under the paint is gone.
        colour = filament * (luma / MATCAP_BASE_LUMA) * PAINT_EXPOSURE;
    }

    // The mask, as a CHROMA shift toward a cool blue the matcap cannot produce,
    // with luminance barely touched so the form stays readable underneath.
    //
    // Measured against the matcap's own constants rather than chosen by taste.
    // Its whole luminance swing is 102 levels and its coolest pixel is a
    // fill-lit cavity at b - r = +17, so darkening (ZBrush's answer) merges the
    // masked shadow into the viewport background and takes the silhouette with
    // it, and a gentle cool desaturation moves a lit pixel by only 36 levels --
    // a fully masked lit pixel would come out brighter than an unmasked
    // shadowed one. The hue below is theme::MASK, with blue driven past 1.0 so
    // the result sits outside anything the clay can be.
    let m = select(in.mask, 1.0 - in.mask, uniforms.mask_inverted != 0u);
    if (m > 0.0) {
        // The floor is what makes a first pass at brush strength visible at
        // all. It is a discontinuity at exactly zero and nowhere else, and that
        // is deliberate: it says "there is a mask here", and every value above
        // it is shown un-thresholded, which is the whole thing eight bits of
        // protection bought.
        let s = (0.30 + 0.70 * m) * uniforms.mask_tint;
        let luma = dot(colour, vec3<f32>(0.2126, 0.7152, 0.0722));
        colour = mix(colour, luma * vec3<f32>(0.345, 0.529, 1.15), s);
    }

    if (uniforms.srgb_target == 0u) {
        // Plain unorm target: encode here, since nothing downstream will.
        colour = pow(colour, vec3<f32>(1.0 / 2.2));
    }

    return vec4<f32>(colour, 1.0);
}
