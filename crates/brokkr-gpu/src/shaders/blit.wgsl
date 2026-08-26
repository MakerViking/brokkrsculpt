// SPDX-License-Identifier: AGPL-3.0-only

// One thumbnail cell, blitted into a panel row.
//
// This runs INSIDE iced's own render pass, which has no depth attachment at
// all, so nothing here may depend on depth. The pass has already set its
// viewport and scissor to the row's bounds, so a triangle covering the whole
// of clip space covers exactly that row's square and nothing else.
//
// One oversized triangle rather than two: it costs one fewer vertex, has no
// diagonal seam, and needs no vertex buffer -- the three corners come out of
// the vertex index. uv (0,0) is the top left of both the cell and the row,
// because in WebGPU clip y = +1 is the top of the framebuffer and texel row 0
// is the top of a texture, so there is no flip to undo.

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var cell: texture_2d<f32>;
@group(0) @binding(1) var cell_sampler: sampler;

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOut;
    out.clip = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fragment_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Opaque, always. The cell was cleared opaque and drawn opaque; a row that
    // carried the cell's alpha through would blend against whatever the panel
    // left behind, and the sculpt pipeline does not write alpha at all.
    return vec4<f32>(textureSample(cell, cell_sampler, in.uv).rgb, 1.0);
}
