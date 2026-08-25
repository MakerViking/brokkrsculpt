<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# `.brokkr` compatibility fixtures

`container-v1.brokkr` (156 bytes) and `container-v2.brokkr` (246 bytes) stand in
for the `.brokkr` files already on users' disks. Every test that reads them lives
in `src/project.rs`, because the volume in them is built with `insert_brick`,
which is `pub(crate)` — an integration test in this directory could not have
made them.

They are **manufactured, not cut down from a real project**. A real file cannot
be truncated and stay valid: the `u64` at byte 63 says how many bricks follow, so
shortening one means rewriting that count, which needs a tool that already reads
the format. Real files are also dense — even a hundred bricks is about 13 MB,
which does not belong in git. Five `Uniform` bricks is a few hundred bytes and is
byte for byte what the writer of the day produced for the same volume.

Layout, confirmed with `od` against both these files and the real ones:

| offset | bytes | what |
|---|---|---|
| 0 | 8 | `BROKKR\0\1` |
| 8 | 2 | container version — `1` here, `2` in the other file |
| 10 | 2 | field version |
| 12 | 8 | the lattice: `BRICK_DIM` then `NARROW_BAND` |
| 20 | 4 | `voxel_size` — the byte that refuses with the same error a corrupt brick does |
| 24 | 39 | the view |
| 63 | 8 | `u64` brick count, `5` here |
| 71 | 12 | the first brick's coordinate, `(-2, -3, -4)` — the same one the real v1 files carry |
| 83 | 1 | brick tag, `0` for uniform (the real files hold `1`, dense) |
| … | | four more bricks |
| end | 0 or 4+ | nothing at all in v1; the key trailer in v2 |

**If a test says these no longer match what this build writes, do not
regenerate them.** They represent files that do not change when this code does.
The one legitimate regeneration is when the header has grown a section that
`manufactured_fixture` already knows how to strip, which by construction leaves
the output identical anyway. Everything else is a question about what the reader
now does to the real projects.

For that one case, regenerating takes two deliberate steps — the `#[ignore]` and
an environment variable:

```sh
BROKKR_REGENERATE_FIXTURES=1 cargo test -p brokkr-core --lib -- --ignored regenerate_the_committed_fixtures
```

The variable is not decoration. `--ignored` is a sweep, not a per-test opt-in,
and the other ignored test in `src/project.rs` is one the maintainer is meant to
run by hand — so without the gate, `cargo test -p brokkr-core -- --ignored`
would quietly replace these files with whatever the current build emits, and the
byte-for-byte guard would then be comparing the build against itself.
