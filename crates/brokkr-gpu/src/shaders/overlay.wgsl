// SPDX-License-Identifier: AGPL-3.0-only
//
// Flat coloured geometry drawn over the sculpt: the brush cursor ring, the
// mirror planes, and the navigation cube. No lighting and no textures — every
// vertex carries its own colour, because these are interface, not surface.

struct Uniforms {
    view_projection: mat4x4<f32>,
    // Non zero when the render target encodes sRGB itself, in which case the
    // fragment must stay linear. Matches sculpt.wgsl, so an overlay colour and
    // the model beside it agree whichever surface format iced picked.
    srgb_target: u32,
    // Three separate scalars, not a vec3<u32>: a vec3 aligns to 16 bytes in
    // uniform address space and would silently disagree with the Rust type's
    // size, which wgpu only catches at draw time.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) colour: vec4<f32>,
}

@vertex
fn vertex_main(
    @location(0) position: vec3<f32>,
    @location(1) colour: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_projection * vec4<f32>(position, 1.0);
    out.colour = colour;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var colour = in.colour.rgb;
    if (uniforms.srgb_target == 0u) {
        // Plain unorm target: encode here, since nothing downstream will.
        colour = pow(colour, vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(colour, in.colour.a);
}
