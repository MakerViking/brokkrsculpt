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

**M1, the brush system.** Six brushes, stroke interpolation, X symmetry,
falloff curves and undo. Still no export, no masking, no multiresolution.

Measured on a Radeon RX 6900 XT, at a 256 cubed effective volume, with
everything switched on at once: interpolation producing several stamps per
pointer event, symmetry applying each of them twice, and undo recording
snapshotting each brick on first touch.

| | slow drag | fast drag | budget |
| --- | --- | --- | --- |
| brush edit, p95 | 2.15 ms | 3.32 ms | 4 ms |
| dirty remesh, p95 | 1.53 ms | 2.18 ms | 8 ms |
| both, p95 | 3.64 ms | 5.30 ms | 16 ms |

Rendering 543k triangles fits inside a 6.94 ms vsync interval, and the volume
holds 40 MB against a 3 GB design ceiling.

The fast drag column is the one to watch: it is 4.8 stamps per event doubled by
symmetry, and it is the only figure without a large margin. Draw and pinch cost
roughly three times what the other four do, because both resample the field
rather than adding to it. Moving edits to compute shaders in M2 is the answer,
not shaving this.

An average stroke step remeshes about 13 bricks out of 408, which is the
property the whole design exists to protect: work is proportional to what the
brush touched, never to the size of the model.

## Brushes

| brush | what it does to the field |
| --- | --- |
| Draw | translates the patch along the stroke normal |
| Clay | blends toward a plane held just outside the surface, adding only |
| Smooth | blends toward the average of the neighbouring values |
| Inflate | offsets the level set, moving every point along its own normal |
| Pinch | squeezes the field toward the brush axis, sharpening ridges |
| Flatten | blends toward the tangent plane under the cursor |

Smooth and flatten have no opposite, so inverting them does nothing and the
interface says so.

Draw and pinch were both first written the obvious way, as a value the brush
adds or amplifies, and both had to be rewritten. Anything that multiplies a
displacement by the local gradient, or amplifies the difference from a local
average, has gain above one somewhere and turns its own rounding error into
visible crust over a stroke. Both now resample the field from a shifted
position instead, which cannot introduce detail that was not already there.

## Building

Needs a recent stable Rust toolchain and a Vulkan capable GPU.

```fish
cargo run --release -p brokkr-app
```

Controls:

| input | action |
| --- | --- |
| left drag | sculpt |
| ctrl left drag | invert the brush |
| right or middle drag | orbit |
| shift right drag | pan |
| wheel | zoom |
| ctrl z, ctrl shift z | undo, redo |

Stylus pressure is plumbed through as a first class value on every stamp, but
nothing supplies it: iced 0.14 drops winit's `force` field, so no pressure
reaches the application. Wiring it up means patching iced or bypassing it for
pointer input.

## Checking it

```fish
cargo test --workspace
cargo bench -p brokkr-core
```

The tests worth knowing about:

- `crates/brokkr-core/tests/seams.rs` asserts the union of the per brick meshes
  is closed, every edge shared by exactly two triangles, after every brush has
  been dragged across brick corners. This is the crack test. It ships with a
  control that proves it can detect a gap.
- `crates/brokkr-gpu/tests/offscreen.rs` renders the sculpt to a texture with no
  window and checks the pixels, then sculpts and checks they changed. It catches
  the class of bug that compiles, passes every unit test, and shows a blank
  window. It also renders each brush in turn, which is how the crust in draw and
  pinch was found: every value was inside the narrow band and every field level
  assertion passed.
- `no_brush_carves_a_pit_where_it_should_raise_a_bump` in `brush.rs` is the
  regression test for that crust, measuring how far the surface height jumps
  between neighbouring samples after a stroke.
- `cargo bench` is a budget gate, not a benchmark report. It exits non zero when
  a budget is blown.

To look at what the offscreen tests rendered:

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
