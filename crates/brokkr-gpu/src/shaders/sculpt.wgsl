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
    // Separate scalars, not a vec3<u32>. A vec3 aligns to 16 bytes in uniform
    // address space, which would push the struct to 160 bytes while the
    // matching Rust type is 144, and wgpu rejects the bind group at draw time.
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var matcap_texture: texture_2d<f32>;
@group(0) @binding(2) var matcap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
    // The STORED protection at this vertex's own lattice cell, 0..1. Byte 0 of
    // the pool's attribute buffer; bytes 1 and 2 are reserved for the filament
    // slots and are deliberately not read here.
    @location(1) mask: f32,
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
