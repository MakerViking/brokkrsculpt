# BrokkrSculpt — build spec

> Commit this as `docs/BUILD-SPEC.md`. The `/goal` condition is deliberately short and points here for the detail.

---

## LOCKED DECISIONS

1. **Name.** Product name **BrokkrSculpt**, short name **Brokkr**. Parallel construction to SindriCAD (Sindri plus CAD). Brokkr is Sindri's brother in the myth; the two of them forged Mjolnir together. Crates are prefixed `brokkr-`. Canonical URL will be `tinkeratlas.com/brokkrsculpt`, matching `tinkeratlas.com/sindricad`.
2. **License.** AGPL, matching SindriCAD. Not open-core. Add `LICENSE` and SPDX headers from the first commit.
3. **Repo.** New standalone private repo. Not a SindriCAD subfolder, and no shared code between them. See "Repo visibility" below for when this has to change.
4. **UI shell.** **Iced**, themed to match SindriCAD. See "UI and design" below.

### Repo visibility

Private is correct for the build phase, but it cannot stay private past release, for two independent reasons:

- AGPL obligations trigger on distribution. Once installers ship, recipients are entitled to the complete corresponding source, so the repo goes public at release.
- SignPath Foundation's free signing requires a public open-source project. A private repo means no free Windows signing for direct downloads.

Note that the Microsoft Store MSIX path does not care about repo visibility, so Store distribution would work either way. Plan on public at release, same as SindriCAD.

---

## What we are building

A native desktop 3D sculpting application in the spirit of ZBrush and Nomad Sculpt. Voxel/SDF based, GPU accelerated, Linux first, built to stay interactive at high detail. Output is print ready geometry, so watertight and manifold meshes are a hard requirement, not a nice to have.

**Primary target platform: Linux.** Nomad Sculpt ships desktop builds for Windows and macOS but not Linux. That gap is the reason this project exists, and it does not stop being the reason now that all three ship.

**Status, 2026-08-29: all three build and are published.** Linux is still where the work happens and still sets the bar; Windows and macOS compile from the same commit, run the same suite, and are held to "must not compromise the Linux build" exactly as written. What they do NOT yet have is time on real hardware -- see the platform table in the README, which says specifically what is unproven on each.

## What we are explicitly NOT building

Do not start any of these. Do not scaffold for them. Do not add abstractions "in case" we want them later. If one seems necessary to make something else work, stop and ask.

- Dynamic topology (live mesh retriangulation). Blender's own developers have moved away from it. We avoid it by being SDF based.
- Auto retopology (ZRemesher style)
- UV unwrapping
- Polypaint or vertex colors
- Sculpt layers
- IMM / insert mesh brushes
- Any web, cloud, or account features
- Any parametric CAD features. That is SindriCAD's job.

### What this list does and does not forbid, clarified 2026-08-25

The asking this list invites has happened, for the bodies, primitives and
masking arc. The answer was that **none of that arc is on this list**, and the
list stays as it is. Written down because three of the entries above sit close
enough to the new work to be misread as forbidding it:

- **"Sculpt layers"** means ZBrush's Layers: recorded per-layer deltas over one
  mesh, with an intensity slider to dial a recording up and down. Not built, not
  planned. **Bodies are ZBrush's SubTools** — several independent sculpts in one
  document — and **a mask is a per-voxel protection value**, not a recording.
  Neither replays an edit at reduced strength, which is the thing this entry
  exists to keep out.
- **"IMM / insert mesh brushes"** means stamping a stored mesh onto a surface
  along a stroke. Not built, not planned. **Adding a cube as a new body** is not
  that: no mesh library, no stroke placement, no surface conforming.
- **"Polypaint or vertex colors"** stands, and the planned colour feature is
  deliberately not it — colour is a **filament slot number** resolved at use
  time, not an RGB value per vertex.

The plan those decisions live in is the maintainer's local note
`~/.claude/plans/brokkr-bodies-and-primitives.md`, which will not resolve for a
visitor; everything load-bearing from it that constrains this repository is
restated in `handoff.md`.

## Stack

Pinned choices. Do not substitute without asking.

- **Rust**, 2021 edition or later
- **wgpu** for rendering and compute, targeting Vulkan on Linux. Native surface via **winit**. No webview anywhere in the render or sculpt path.
- **Iced** (`iced_wgpu`) for tool panels and chrome. The 3D viewport is embedded using Iced's `shader` widget, which exists precisely to render custom wgpu content inside the UI tree. Do not try to composite a separate native window under a transparent overlay.
- **glam** for math
- **fast-surface-nets** for CPU meshing in M0 and M1, and as the reference implementation we port to GPU compute in M2
- **rayon** for CPU parallelism where it helps

Deliberately deferred: `fidget` (JIT evaluated closed-form implicit surfaces) is a good fit for procedural primitives and booleans, not for the sculpted voxel field itself. Revisit it at M4, not before.

## Core architecture

### Volume representation

A **sparse brick grid**, VDB inspired, at a **fixed world-space voxel size**. Resolution is uniform and independent of object size. Getting finer detail means resampling the whole volume to a smaller voxel size, which is a deliberate explicit operation, the same model Nomad's voxel remesher uses.

- Brick = 32x32x32 voxels. Start here; the size is a tuning knob, so keep it a constant, not a magic number sprinkled through the code. Larger bricks mean fewer allocations but more wasted work on small edits.
- Bricks live in a hash map keyed by integer brick coordinate. Empty space costs nothing. Only bricks the surface passes through are allocated.
- Each voxel stores a **narrow band** signed distance value. Only store distances within roughly plus or minus 3 voxels of the surface; clamp beyond that.
- Memory budget is the real ceiling here, not compute. Nomad users hit crashes above roughly 15M vertices and 3GB. Beating that is a design goal, so track allocated brick count and resident bytes from day one and surface both in a debug overlay.

### Sculpt loop

Each brush stroke runs this pipeline:

1. **Raycast.** Sphere trace the SDF from the cursor to find the surface point and normal.
2. **Resolve affected bricks.** Compute the brush AABB in world space, find the bricks it touches, and allocate any that do not exist yet. This allocation step is the hard part and the reason we are not simply using NanoVDB: its tooling is built for reading and interpolating, and modification is largely limited to changing values of already active voxels. We need to activate new voxels constantly as clay is added, so the authoritative edit structure stays on our side.
3. **Apply the edit.** Falloff weighted modification of the SDF values in the affected bricks.
4. **Mark bricks dirty.**
5. **Remesh only the dirty bricks.** Never remesh the whole volume per frame. This is the single most important performance property in the entire project. Work must be proportional to what the brush touched, not to total model size.
6. **Splice and render.** Each brick owns a slice of a large vertex buffer.

### The apron rule (read this twice)

Meshing a brick requires a **one voxel halo of data from its neighbors**. Without it you get visible cracks and seams at every brick boundary. This will be the first thing that goes wrong. Build the halo into the brick sampling API from the very first commit so that no call site can accidentally mesh a brick without it.

### Stroke continuity

Mouse and stylus samples arrive far apart during fast strokes. Interpolate along the stroke path and apply the brush at spaced intervals, or fast strokes will leave a dotted trail instead of a continuous cut.

### Undo

One stroke equals one undo entry. On the first modification of a brick within a stroke, snapshot that brick's prior contents. Cap history by a memory budget rather than by entry count. Design this in at M1; retrofitting undo onto a GPU resident volume later is painful.

## Crate layout

```
crates/
  brokkr-core/   # volume, bricks, brushes, meshing. NO UI, NO winit, NO iced deps.
  brokkr-gpu/    # wgpu device, compute pipelines, buffer pools
  brokkr-app/    # iced shell, input, tools, viewport
```

`brokkr-core` must stay free of UI and windowing dependencies. This is what keeps the UI shell decision reversible. Iced is Elm-style and retained-mode, which means more boilerplate than an immediate-mode toolkit, and transient tool state (brush popups, gizmos, drag handles) is genuinely quicker to build in immediate mode. If Iced turns out to be fighting us by the end of M1, swapping to `egui` touches only `brokkr-app` and leaves the engine untouched. Flag it if that starts happening rather than pushing through.

## UI and design

Match SindriCAD's design, not its stack. SindriCAD is Tauri with a web frontend; the sculpt loop cannot route through a webview at the framerates this needs, so the stacks diverge on purpose.

Before building any panels, extract SindriCAD's design tokens (color palette, spacing scale, typography, corner radii, icon set) into a single `theme.rs` in `brokkr-app`, and build every widget from those tokens. The goal is family resemblance, not a pixel-perfect match. Do not hand-tune colors per widget.

## Milestones

Complete each one and stop for review before starting the next. Do not work ahead.

### M0: Vertical slice — "I can push clay"

The goal is the moment where a sphere deforms under the cursor at a solid framerate. Everything else in the project depends on this loop being right.

- winit window, wgpu device, orbit camera
- Sparse brick volume, CPU only, seeded with a sphere SDF
- One brush: draw / pull, with a smooth falloff
- Raycast to surface under cursor
- Dirty brick tracking, CPU surface nets meshing of dirty bricks only, with the apron handled correctly
- Matcap shading (no PBR, no lights, no shadows)
- Debug overlay: FPS, frame time, triangle count, allocated bricks, resident MB

Done when: a sphere at a 256-cubed effective volume can be sculpted continuously at 60fps with no visible seams at brick boundaries.

### M1: A real brush system

- Brushes: draw, clay, smooth, inflate, pinch, flatten
- Radius and strength, with pressure support if a stylus is present. Read below the toolkit on every platform -- evdev on Linux, Raw Input on Windows, IOKit on macOS -- because iced drops winit's `force` field and winit has no desktop pen pressure to drop.
- Falloff curve control
- X axis symmetry
- Stroke interpolation
- Undo and redo per the design above

### M2: Scale it

This is where the "fast at high detail" requirement is actually met.

- Move SDF edits to wgpu compute shaders
- Move surface nets meshing to compute
- GPU brick pool with an atomic free list, and a page table so shaders can address bricks
- Narrow band storage, consider f16 for distance values
- Frustum culling and per-brick draw batching

Done when: 10M or more triangles remain interactive at 60fps, with resident memory well under the roughly 3GB and 15M vertex point where comparable tools fall over.

### M3: Print ready export

- STL, OBJ, 3MF
- **Watertight and manifold output is mandatory.** Validate it, do not assume it. Add a test that checks exported meshes for non-manifold edges and degenerate triangles. SindriCAD's texture work produced non-manifold exports from degenerate triangles; do not repeat that.
- Correct unit scale on export
- Resample volume to a different voxel size (the explicit "increase detail" operation)

### M4: Polish

Masking, material and matcap selection, multiresolution levels, procedural primitives via fidget. Packaging and signing per the distribution notes below.

## Performance budget

Treat these as tests, not aspirations. Add a benchmark harness at M0 and keep it running.

- Frame: 16ms total
- Brush edit dispatch: under 4ms
- Dirty brick remesh: under 8ms
- Never allocate in the per frame path
- Never remesh a brick that was not marked dirty

## Distribution

**No longer context only: an unsigned rolling `beta` prerelease ships all three
platforms from `.github/workflows/release.yml`, rebuilt on every push to
`main`.** Signing is the part still waiting on funding, and the notes below are
what it will cost when it happens.

- **Linux:** first and primary. A tarball ships today; AppImage is still wanted, matching SindriCAD's approach.
- **Windows:** package as **MSIX for the Microsoft Store**. Store registration is now free for both individual and company accounts, and Microsoft re-signs MSIX packages server side, so Store users never see a SmartScreen warning and no certificate purchase is needed. Note that a plain EXE or MSI submitted to the Store is NOT re-signed. Direct downloads from tinkeratlas.com still need separate signing, free via SignPath Foundation if the project is open source.
- **macOS:** ships unsigned as a `.app` in a zip, built with `ditto` to keep the bundle structure. Requires the 99 USD per year Apple Developer Program for notarization, and is the only platform with an unavoidable recurring cost. The old right-click-to-open Gatekeeper bypass was removed in macOS Sequoia, so unsigned builds force users through System Settings, Privacy and Security, "Open Anyway", and an admin password -- **and an app with no signature at all may not appear in that list to be excused**, which is why the release notes give `xattr -dr com.apple.quarantine` as the primary instruction rather than the Apple-supported route.

## Working conventions

- **Shell is Fish, not bash.** Write any commands accordingly.
- **No em-dashes** in any prose, docs, comments, commit messages, or README content.
- **Put important context, warnings, and required values BEFORE any command block**, never after.
- Write anything worth remembering to Mimir as you go: architecture decisions, gotchas, tuning constants that turned out to matter, and anything that cost more than an hour to figure out.
- Small, reviewable commits. CI from the start, and a failing end-to-end run blocks a release.

## First actions

1. Confirm the four DECISIONS above are filled in.
2. Scaffold the workspace and the three crates.
3. Get a window with a wgpu clear color and an orbit camera on screen.
4. Then start M0 proper.

Ask before deviating from any pinned choice or before starting anything in the NOT building list.
