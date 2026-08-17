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

**M2, scale.** Six brushes, stroke interpolation, X symmetry, falloff curves,
stylus pressure with tilt and an eraser, undo, and eleven million triangles
staying interactive. Still no export, no masking, no multiresolution.

All figures on a Radeon RX 6900 XT and a Ryzen 9 7900X.

At M1 sizes, a 240 cubed effective volume, with everything on at once:
interpolation producing several stamps per pointer event, symmetry applying each
of them twice, and undo snapshotting each brick on first touch.

| | slow drag | fast drag | budget |
| --- | --- | --- | --- |
| brush edit, p95 | 2.24 ms | 3.53 ms | 4 ms |
| dirty remesh, p95 | 0.58 ms | 0.59 ms | 8 ms |
| both, p95 | 2.82 ms | 4.03 ms | 16 ms |

At M2's target, a 30 mm sphere at a 0.055 mm voxel, which is 1090 cubed
effective:

| | measured | budget |
| --- | --- | --- |
| triangles | 11,216,268 | 10 million |
| brush edit, one stamp | 2.63 ms | 4 ms |
| dirty remesh, 64 bricks | 0.86 ms | 8 ms |
| render, all 5435 bricks drawn | 1.13 ms | 16 ms |
| volume plus mesh | 1027 MB | well under 3 GB |

An average stroke step remeshes about 13 bricks out of 6056, which is the
property the whole design exists to protect: work is proportional to what the
brush touched, never to the size of the model.

### M2 was met without compute shaders

The build spec's plan for M2 was to move both the SDF edits and the meshing to
wgpu compute, with a GPU brick pool and a page table. Measuring first showed
that was not what stood in the way, and it is worth recording why the code does
not look like the plan.

At the target size the CPU path had exactly one problem: a brush edit took
13.9 ms against a 4 ms budget, because a brush covers a fixed world radius and
so touches cubically more voxels as the voxel size shrinks. Meshing and memory
were already inside their budgets. Two changes fixed it:

- Meshing across cores took a full mesh from 2547 ms to 233 ms and a stroke's
  remesh from 5.6 ms to under 1 ms. Bricks are independent and meshing only
  reads, so this is close to free parallelism.
- Editing across cores, above a work threshold, took one stamp from 13.9 ms to
  2.6 ms.

Two things the plan called for turned out not to be needed. Drawing all 5435
bricks individually costs 1.13 ms, so per brick batching would buy nothing. And
16 bit distance values would halve a volume that already fits in a third of its
ceiling.

There is also an argument against moving only the edits: with meshing on the CPU,
GPU edits would need a read back every stamp, and waiting for the GPU costs more
than the edit saved. The compute route is close to all or nothing, and its one
justification has gone. It remains the right answer for scales far beyond this,
and the CPU path is a working reference to check it against when that time comes.

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

## Stylus pressure

Pressure works with any tablet the Linux kernel has a driver for. There is no
vendor list and no per device configuration: a stylus is any input device that
reports both `ABS_PRESSURE` and `BTN_TOOL_PEN`, which every tablet driver sets.
Each device's own pressure range is read from the device, so a Huion reporting
8191 levels and a Wacom reporting 2047 both normalise correctly.

It is read straight from the kernel's evdev interface rather than from the
window system. That is not a shortcut: iced 0.14's touch events carry only a
position and drop winit's `force` field, and winit 0.30 never had force for pens
at all. Reading evdev also means one code path covers X11, XWayland and Wayland,
because it sits below all three.

**Reading `/dev/input` needs the `input` group on most distributions.** Without
it the tablet is simply invisible and the brush runs at full strength, exactly
as it does for a mouse. To fix that, then log out and back in:

```fish
sudo usermod -aG input $USER
```

### The eraser end

Flipping the stylus over inverts the brush, the same as holding the modifier
key. The two combine rather than override, so holding the modifier while using
the eraser gives the additive brush back. Brushes with no opposite, smooth and
flatten, ignore both.

The kernel reports the tip and the eraser as separate tools that are never in
range at the same time, which is worth knowing: anything checking only for the
tip treats every eraser stroke as a mouse and quietly runs it at full pressure.

### Tilt

Leaning the pen rotates the direction the brush pushes in, up to sixty degrees.
That steers every brush at once, because they all work from the same stroke
normal: draw pushes clay sideways, the clay and flatten planes tip over so a
surface can be flattened at an angle, and pinch's axis leans with the pen.

Leaning also reduces how far the brush pushes outward, by the cosine of the
angle, which is what makes a leaned stroke feel like a glancing one.

Tilt arrives in the tablet's frame, which lines up with the screen, and is
carried into world space through the camera. Positive tilt on the second axis
is taken to mean the pen is leaning toward you, and so toward the bottom of the
screen. If steering comes out mirrored on some tablet, that convention is a
single sign in `pen_lean`.

### Checking a tablet

The **PEN** panel names the tablet it found, shows the device's pressure range
and whether it has tilt and an eraser, and gives a live reading of pressure,
tilt and which end of the pen is in range. That is enough to confirm a tablet
is working in a few seconds rather than guessing from how the brush feels. If
it says no tablet was found, this prints every input device and why each was
accepted or rejected:

```fish
cargo run --release -p brokkr-app -- --tablets
```

The **Curve** slider shapes the response: below 1 makes light touches bite
harder, above 1 gives finer control at the light end.

Windows and macOS fall back to full pressure. Those need Pointer Input or Wintab
and `NSEvent` respectively, and both are milestones away.

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
| pen pressure | scales the brush |
| pen eraser end | inverts the brush |
| pen tilt | steers which way the brush pushes |

See [Stylus pressure](#stylus-pressure) for what a tablet adds and how to check
one is being seen.

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
- `tablet.rs` builds a synthetic tablet with uinput, lets the ordinary scanner
  find it through `/dev/input` like any other hardware, and checks that
  pressure, tilt and both ends of the pen come out the far end. Nothing in it is
  a mock, so the same path runs for real hardware. It skips loudly when
  `/dev/uinput` is not writable.
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
