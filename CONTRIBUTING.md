<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Contributing to BrokkrSculpt

Thanks for your interest in BrokkrSculpt! A few ground rules keep the project
healthy and its licensing clean.

## License of your contributions

BrokkrSculpt is released under **AGPL-3.0-only** (see [`LICENSE`](LICENSE)). By
submitting a contribution (a pull request, patch, or any code or content), you
agree that:

1. Your contribution is licensed to the project and its users under
   **AGPL-3.0-only**; and
2. You grant the project maintainer (the copyright holder) a perpetual,
   irrevocable, worldwide, royalty-free right to **relicense your contribution
   under other terms**, including a commercial license.

This dual-licensing grant is what lets BrokkrSculpt stay fully open-source under
the AGPL while the maintainer can also offer a commercial license to
organizations that can't use AGPL software, the revenue that keeps the project
maintained. It's the same inbound-relicensable model used by projects like
GitLab and Qt. Its sibling, [SindriCAD](https://tinkeratlas.com/sindricad), uses
the same one.

You confirm you have the right to grant this (the work is yours, or your
employer has authorized it).

> This is a lightweight contributor agreement, not legal advice; a formal
> CLA/DCO document may replace this note later.

Every source file carries an `SPDX-License-Identifier: AGPL-3.0-only` header.
Add one to any new file.

## Development

You need a Linux machine with a Vulkan-capable GPU, and a Rust toolchain at
least as new as the `rust-version` in the workspace manifest. See the
**Building** section of [`README.md`](README.md) to get started.

Then [`docs/`](docs/), which is where the reasoning lives. These are long on
purpose: they record what was tried and rejected as well as what shipped, which
is usually the faster way in than the code.

- [`BUILD-SPEC.md`](docs/BUILD-SPEC.md): what the application is meant to be,
  and the performance budgets it is held to.
- [`DRIVING-THE-APP.md`](docs/DRIVING-THE-APP.md): launching, driving and
  screenshotting the running application, and the traps in doing so.
- [`CUT-TOOL-PLAN.md`](docs/CUT-TOOL-PLAN.md): the cut tool's design, and a
  worked example of the detail a plan here gets before any code does.
- [`AUTOUPDATE-PLAN.md`](docs/AUTOUPDATE-PLAN.md): the updater, its trust model,
  and what could not be deferred.

Before opening a PR:

- `cargo fmt --all -- --check` — clean.
- `cargo clippy --workspace --all-targets` — **zero** warnings. CI adds
  `-D warnings`, so a warning is a failed build there.
- `cargo test --workspace` — all green.
- `cargo bench -p brokkr-core` if you touched the sculpt loop. It is a **gate,
  not a report**: it exits non-zero when a frame-time budget is blown. Close
  BrokkrSculpt and any other GPU or CPU hog first — the fast-drag case sits at
  roughly 7.7 ms of its 8 ms budget and a busy desktop alone can fail it.

CI runs a stricter toolchain than a typical local one, and has caught lints
that did not exist locally. Local green does not guarantee CI green.

**If you changed anything you can see or click, drive it.** Handle picking and
widget hit-testing both need a compositor, so no test in the suite can tell you
whether a gizmo handle actually grabs or a preview actually draws. This project
has shipped a green-but-dead interaction twice.
[`docs/DRIVING-THE-APP.md`](docs/DRIVING-THE-APP.md) is how to launch, drive
and screenshot the running application, and the traps section is worth reading
before you script anything.

## Things that will get a PR sent back

These are not style preferences. Each one cost real debugging time, and each is
documented where it lives:

- **`brokkr-core` must stay free of UI, windowing and GPU dependencies.** CI
  fails the build if `iced`, `winit`, `wgpu`, `egui`, `tauri` or
  `raw-window-handle` appear in its tree. That is what keeps the toolkit choice
  reversible.
- **Meshing goes through `Volume::mesh_brick`.** A brick needs a one-voxel halo
  from its 26 neighbours; any path that meshes without it puts a crack at every
  brick boundary. There is deliberately no public route from a `Brick` to a mesh.
- **Nothing allocates in the per-frame path**, and nothing remeshes a brick that
  was not marked dirty. Work scales with what the brush touched, never with the
  size of the model.
- **Anything touching the filesystem goes through an iced `Task`.** The event
  loop is the thread that draws; a blocking dialog or write freezes the window.
- **Anything that parses bytes from disk gets a mutation fuzz, and every fuzz
  ships a control** proving the mutants actually reached the parser rather than
  bouncing off a magic-number check.
- **Measure before you optimise, and put the number in the commit message.**
  Several plausible optimisations in this codebase were implemented, measured,
  and reverted for making things worse; they are recorded so nobody spends the
  day again.

## Reporting bugs

Use the in-app reporter — the bug button in the corner of the viewport, or
**Help > Report a bug**. It shows you the exact payload before it sends, and it
is anonymous unless you have signed in to TinkerAtlas, which the dialog says
either way. Or open a GitHub issue. For anything security-sensitive, see
[`SECURITY.md`](SECURITY.md) instead.

**On Windows or macOS, include the PEN and PUCK lines from the diagnostics.**
Those backends are written and compiled but have never met hardware, so a
report saying which of "listening" or "reading" they show is the most useful
thing anyone with a tablet or a SpaceMouse can send.
