---
name: drive-the-app
description: Launch, drive and screenshot the running BrokkrSculpt window on this KDE Wayland session. Use when asked to run or start the app, to verify an interaction end to end, to reproduce a reported bug, or to check that a gizmo handle, camera gesture or panel control is actually live — anything the test suite cannot see because 3D and widget hit-testing both need a compositor.
---

# Driving BrokkrSculpt

Two `#[ignore]`-free facts to start from: **this project has shipped a
green-but-dead interaction twice**, and no test in the suite can catch a third,
because both 3D handle picking and widget picking need a compositor. That is
what this skill is for. It is not a substitute for `cargo test`; it is the only
way to answer "does the handle actually grab".

## The three tools

| | |
|---|---|
| `scripts/drive.py` | virtual pointer and keyboard over `/dev/uinput` |
| `scripts/shot.sh` | screenshot the window, refusing anything it cannot prove is ours |
| `scripts/drive.py --geometry` | where the window is right now |

## Always, in this order

**1. Build, then check the binary is the one you built.** A GUI fix means
nothing until the process is the one you just compiled, and the mistake is
silent — a stale process looks exactly like a failed fix. This has wasted a
session here before.

```bash
cargo build --release -p brokkr-app
ls -l --time-style=+%H:%M:%S target/release/brokkrsculpt; date +%H:%M:%S
pgrep -x brokkrsculpt || ./target/release/brokkrsculpt &
```

**2. Drive it.** Coordinates are WINDOW-LOCAL; `drive.py` reads the window
position from KWin every run, so nothing is hard-coded to where the window
happened to be.

```bash
cat > /tmp/t.txt <<'EOF'
move 420 418        # viewport centre of a 1024x768 window
key w               # arm the gizmo
sleep 700
EOF
scripts/drive.py /tmp/t.txt
```

**3. Screenshot, and LOOK at it.** A blank frame is a failure to launch, and a
screenshot you did not read is not evidence.

```bash
./scripts/shot.sh /tmp/out.png
```

## Converting a capture position into a drive coordinate

`shot.sh` captures at the output's scale factor, **1.5** on this desktop: a
1024x768 window comes back 1536x1152. So a feature you can see at image
`(ix, iy)` is driven at `(ix / 1.5, iy / 1.5)`.

Verified: the brush ring drawn at the pointer lands at exactly
`window_local * 1.5` in the capture, with no offset on either axis. Do not
assume that survives a change of scale or a second monitor — re-check it by
moving to a point over the model and confirming the ring is where you aimed.

## Controls worth knowing before you script anything

- **Left** press picks a gizmo handle, or sculpts.
- **Middle or right drag** orbits; **shift** with them pans. Right *click*
  opens the brush menu, so use middle for orbiting.
- **Wheel** zooms about the surface under the cursor.
- `w` arms the transform gizmo, `1`-`7` pick a brush, `x`/`y`/`z` toggle a
  mirror plane, `m` masks, `esc` cancels a live drag, `ctrl+z` undoes.
- A **release with no motion is deliberately a no-op** — it will not re-bake.
  If you are testing a drag, move at least a few pixels.

## The traps, all of which have produced confident wrong conclusions here

**Focus.** Everything typed goes to whatever the compositor thinks is focused.
`drive.py` activates the window and **refuses to run** if it did not come
forward, because the alternative is typing into the user's terminal — that is
how a capture once showed a brush changed and a mirror plane on that no code
had touched. If it refuses, something took focus back; do not work around it.

**Absolute, never relative.** libinput accelerates relative motion, so a delta
of 500 does not travel 500 pixels. The device is absolute over the logical
desktop union, read from `kscreen-doctor`.

**A drag is many small steps.** An application tells a drag from a click by
travel; one jump reads as neither. `drag` emits 24 steps by default.

**`shot.sh` may refuse, and that is it working.** It checks focus before *and*
after the grab and compares the image's aspect ratio to the window's, then
deletes anything it cannot prove is ours — because it once captured a media
player and a private browsing session straight into an agent's context. Never
work around it with `spectacle -f`.

**Do not kill the app to rebuild without asking.** The title bar shows `*` for
unsaved work. The autosave in `$XDG_STATE_HOME/brokkrsculpt/autosave.brokkr` is
only written every two minutes, and File > Recover is the way back.

## Reading the status line is most of the value

The status line names what actually happened, and it is usually a better oracle
than the picture. Real examples, each of which confirmed a whole code path:

- `moved 16.75 mm — exact, 14 ms, not one voxel recomputed` — the lossless
  `Bake::Exact` route.
- `resized to 100/45/100% — resampled in 122 ms` — a per-axis scale on the Y
  axis alone, and it names the cost.
- `cancelled the drag` — a gesture abandoned rather than committed.

If a gesture produced no status change at all, the press probably missed the
handle. Screenshot before assuming the feature is broken.

## What has been driven this way, and worked

Arming the gizmo; dragging an axis arrow; dragging a per-axis scale box;
abandoning a live drag with a second button; scrolling while holding a handle;
orbiting; adding a primitive from the BODIES panel; and `ctrl+z`. Six of those
are behaviours no test in the suite can reach.
