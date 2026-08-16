# BrokkrSculpt

A native desktop 3D sculpting application: voxel and SDF based, GPU accelerated,
Linux first. Output is meant to be printed, so watertight and manifold geometry
is a hard requirement rather than a nice to have.

Sibling to [SindriCAD](https://tinkeratlas.com/sindricad). Brokkr and Sindri
forged Mjolnir together. SindriCAD does parametric CAD; BrokkrSculpt does clay.

Nomad Sculpt ships for Windows and macOS but not Linux. That gap is why this
exists, and it is why Linux comes first and stays first.

Full design and milestone plan: [docs/BUILD-SPEC.md](docs/BUILD-SPEC.md).

## Status

**M0, the vertical slice.** A sphere deforms under the cursor at frame rate.
Not yet a usable sculpting tool: one brush, no undo, no export.

Measured on a Radeon RX 6900 XT, at a 256 cubed effective volume:

| | measured | budget |
| --- | --- | --- |
| brush edit, p95 | 0.072 ms | 4 ms |
| dirty brick remesh, p95 | 0.919 ms | 8 ms |
| render, 543k triangles | vsync capped at 6.94 ms | 16 ms |
| resident volume memory | 40 MB | beat 3 GB at 15M vertices |

An average stroke step remeshes 5.9 bricks out of 408, which is the property the
whole design exists to protect: work is proportional to what the brush touched,
never to the size of the model.

## Building

Needs a recent stable Rust toolchain and a Vulkan capable GPU.

```fish
cargo run --release -p brokkr-app
```

Controls:

| input | action |
| --- | --- |
| left drag | add clay |
| ctrl left drag | carve |
| right or middle drag | orbit |
| shift right drag | pan |
| wheel | zoom |

## Checking it

```fish
cargo test --workspace
cargo bench -p brokkr-core
```

The tests worth knowing about:

- `crates/brokkr-core/tests/seams.rs` asserts the union of the per brick meshes
  is closed, every edge shared by exactly two triangles. This is the crack test.
  It ships with a control that proves it can detect a gap.
- `crates/brokkr-gpu/tests/offscreen.rs` renders the sculpt to a texture with no
  window and checks the pixels, then sculpts and checks they changed. It catches
  the class of bug that compiles, passes every unit test, and shows a blank
  window.
- `cargo bench` is a budget gate, not a benchmark report. It exits non zero when
  a budget is blown.

To look at what the offscreen test rendered:

```fish
env BROKKR_DUMP_FRAMES=/tmp cargo test -p brokkr-gpu --test offscreen
```

## Layout

```
crates/brokkr-core/   volume, bricks, brushes, meshing. No UI, no windowing, no GPU.
crates/brokkr-gpu/    wgpu resources, buffer pools, the sculpt pipeline. No UI.
crates/brokkr-app/    iced shell, input, tools, viewport.
```

`brokkr-core` staying free of UI and GPU dependencies is what keeps the shell
choice reversible. CI fails the build if that ever stops being true.

## Licence

AGPL-3.0-or-later. See [LICENSE](LICENSE).
