<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Third-Party Notices

BrokkrSculpt is built on Rust crates. This file records their licensing and
satisfies their attribution requirements.

**Nothing here imposes a copyleft obligation on BrokkrSculpt.** The project is
AGPL-3.0-only by its own choice, not because a dependency forced it. Every one
of the **305 crates** compiled into a Linux build is permissively licensed —
MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib, Unlicense, CC0-1.0,
CDLA-Permissive-2.0, or Unicode-3.0. There is no GPL, no LGPL, no MPL and no
EUPL in the graph. That is worth stating plainly because it is unusual: it
means nobody taking this code on inherits a linking question, a written offer,
or a relinking obligation from us.

Full license texts are not vendored here. Each crate ships its own under
`~/.cargo/registry/`, and every project below is linked to its source.

## How this was compiled, and how to re-check it

From `cargo metadata`, filtered to `x86_64-unknown-linux-gnu`, walking the
`normal` and `build` dependency edges from the three workspace crates —
`brokkr-core`, `brokkr-gpu` and `brokkr-app` — and **excluding
`dev-dependencies`**, which are used to test the software and are not
distributed in a build of it. Re-run it whenever `Cargo.lock` changes.

Counts by declared license expression, over those 305 crates:

| Crates | Expression |
| ---: | --- |
| 136 | `MIT OR Apache-2.0` |
| 77 | `MIT` |
| 33 | `Apache-2.0 OR MIT` |
| 10 | `Apache-2.0` |
| 9 | `MIT/Apache-2.0` |
| 7 | `MIT OR Apache-2.0 OR Zlib` |
| 5 | `Unlicense OR MIT` |
| 4 | `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` |
| 3 | `Zlib OR Apache-2.0 OR MIT` |
| 3 | `Zlib` |
| 3 | `ISC` |
| 3 | `BSD-3-Clause` |
| 2 | `Apache-2.0/MIT` |
| 2 | `BSD-2-Clause OR Apache-2.0 OR MIT` |
| 1 each | `BSD-2-Clause`, `Apache-2.0 AND ISC`, `Apache-2.0 AND MIT`, `Apache-2.0 OR ISC OR MIT`, `Apache-2.0 OR GPL-2.0-only`, `(MIT OR Apache-2.0) AND Unicode-3.0`, `CDLA-Permissive-2.0`, `CC0-1.0` |

## The one line that mentions the GPL, and why it is not a problem

**`self_cell` 1.3.0** — `Apache-2.0 OR GPL-2.0-only` —
<https://github.com/Voultapher/self_cell>. This is a *disjunction*: a
downstream user picks one. **BrokkrSculpt takes it under Apache-2.0**, so no
GPL-2.0 term applies to this project or to anything built from it. It is
recorded here rather than left for someone to find in a license scan and worry
about. It arrives transitively through the text stack — `iced` → `iced_wgpu` →
`cryoglyph` → `cosmic-text` → `self_cell` — not by direct choice.

Two other expressions are conjunctions rather than choices, so *both* terms
apply and both are permissive: **`ring`** (`Apache-2.0 AND ISC`) and
**`unicode-ident`** (`(MIT OR Apache-2.0) AND Unicode-3.0`).

## The dependencies that carry the project

Direct dependencies of the workspace, and the few transitive ones large enough
to be worth naming:

### Interface and rendering

- **iced** and **iced_wgpu** 0.14 — MIT —
  <https://github.com/iced-rs/iced>. The application shell and the `shader`
  widget the viewport is drawn into.
- **wgpu** 27 and **naga** — MIT OR Apache-2.0 —
  <https://github.com/gfx-rs/wgpu>. Vulkan is the backend in practice.
- **winit** — Apache-2.0 — <https://github.com/rust-windowing/winit>. Reached
  through iced; the application does not use it directly.
- **cosmic-text** — MIT OR Apache-2.0 —
  <https://github.com/pop-os/cosmic-text>. Text shaping, via iced.
- **tiny-skia** and **tiny-skia-path** — BSD-3-Clause —
  <https://github.com/RazrFalcon/tiny-skia>. Vector rasterisation, via iced.
- **bytemuck** — Zlib OR Apache-2.0 OR MIT —
  <https://github.com/Lokathor/bytemuck>. Taken under Apache-2.0. Getting Rust
  structs safely to the GPU as bytes.
- **glam** — MIT OR Apache-2.0 — <https://github.com/bitshifter/glam-rs>.
  Vectors and matrices.

### Engine

- **fast-surface-nets** — MIT OR Apache-2.0 —
  <https://github.com/bonsairobo/fast-surface-nets-rs>. The isosurface
  extraction each brick's mesh comes out of.
- **rayon** — MIT OR Apache-2.0 — <https://github.com/rayon-rs/rayon>. The
  parallel path in `edit_voxels`.
- **rustc-hash** — Apache-2.0 OR MIT —
  <https://github.com/rust-lang/rustc-hash>. The sparse brick map's hasher.

### Files

- **roxmltree** 0.20 — MIT OR Apache-2.0 —
  <https://github.com/RazrFalcon/roxmltree>. 3MF is XML.
- **yazi** 0.2.1 — Apache-2.0 OR MIT — <https://github.com/dfrg/yazi>. Raw
  DEFLATE; a 3MF is a ZIP and the ZIP container is read by hand.
- **serde_json** — MIT OR Apache-2.0 — <https://github.com/serde-rs/json>.
- **rfd** 0.17.2 — MIT — <https://github.com/PolyMeilex/rfd>. Native file
  dialogs. The version is pinned for dependency cost, not compatibility.

### Devices

- **evdev** — Apache-2.0 OR MIT — <https://github.com/cmr/evdev>. Stylus
  pressure and tilt, and the SpaceMouse, read below the display server.

### Network

The bug reporter is the only outbound HTTPS in the application, and TLS is
where most of the dependency count went.

- **ureq** — MIT OR Apache-2.0 — <https://github.com/algesten/ureq>. Built
  with `default-features = false` and `rustls` only.
- **rustls** — Apache-2.0 OR ISC OR MIT — <https://github.com/rustls/rustls>.
- **ring** — Apache-2.0 AND ISC — <https://github.com/briansmith/ring>. The
  cryptography under rustls. Both terms apply; both are permissive. Contains
  code derived from BoringSSL, which carries its own permissive notices in the
  crate's own `LICENSE`.
- **rustls-webpki** and **untrusted** — ISC — certificate path validation.
- **webpki-roots** — CDLA-Permissive-2.0 —
  <https://github.com/rustls/webpki-roots>. Mozilla's CA trust store, packaged
  as data. It is redistributed unmodified; the CDLA-Permissive-2.0 terms cover
  it as a *data* set rather than as code.

### Logging

- **log** and **env_logger** — MIT OR Apache-2.0 — <https://github.com/rust-lang/log>
  and <https://github.com/rust-cli/env_logger>.
- **pollster** — Apache-2.0/MIT — <https://github.com/zesterer/pollster>.

## Assets

**Nothing here is third-party.** Every image and mark in this repository was
made for it, and there are no third-party fonts, icons, textures or sample
models to attribute.

- `docs/images/*.jpg` — screenshots of the application, taken from it.
- `docs/images/brokkr-dwarf.png` — the Brokkr character illustration,
  commissioned for this project and generated with Google Gemini. Copyright
  holder as for the rest of the repository.
- `assets/brand/*.svg` — the BrokkrSculpt lockup and mark, hand-written SVG.
  They deliberately share the molten gradient of SindriCAD's mark, which is the
  same author's.
- `assets/icons/*.svg` — the user-interface icon set, **generated** from
  `crates/brokkr-app/src/icon.rs`, which is where the drawings live. Original
  work, drawn for this project. They follow the same house style as SindriCAD's
  icons — a 24 by 24 grid, 1.6 stroke, round caps, `currentColor` throughout —
  because that is the same author's set too; no third-party icon library is
  used, vendored or traced.

**No font files are shipped.** The lockups name a font *stack* and fall back
through the system's own faces, so nothing here embeds a typeface. Converting
the wordmark to outlines before it goes anywhere it must render identically is
noted in the files themselves.

The clay matcap the model is shaded with is *generated in code*
(`brokkr-gpu/src/matcap.rs`) rather than shipped, which keeps the largest asset
the application needs out of the repository entirely — that was the reasoning
before any of the above existed, and it still holds for anything the renderer
consumes.
