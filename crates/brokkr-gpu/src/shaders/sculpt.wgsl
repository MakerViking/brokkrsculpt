// SPDX-License-Identifier: AGPL-3.0-or-later
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
    // Three separate scalars, not a vec3<u32>. A vec3 aligns to 16 bytes in
    // uniform address space, which would push the struct to 160 bytes while the
    // matching Rust type is 144, and wgpu rejects the bind group at draw time.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var matcap_texture: texture_2d<f32>;
@group(0) @binding(2) var matcap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_normal: vec3<f32>,
}

@vertex
fn vertex_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_projection * vec4<f32>(position, 1.0);
    // The view matrix is a rigid transform, so its upper 3x3 rotates normals
    // correctly without an inverse transpose.
    out.view_normal = (uniforms.view * vec4<f32>(normal, 0.0)).xyz;
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

    if (uniforms.srgb_target == 0u) {
        // Plain unorm target: encode here, since nothing downstream will.
        colour = pow(colour, vec3<f32>(1.0 / 2.2));
    }

    return vec4<f32>(colour, 1.0);
}
