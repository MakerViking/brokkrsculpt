# Changelog

What changed in each BrokkrSculpt build, newest first. Everything under
**Unreleased** ships in the next rolling beta, and the release notes on the
[beta release](https://github.com/MakerViking/brokkrsculpt/releases/tag/beta)
are generated from that section.

Every green `main` produces a build, so most entries land under Unreleased and
stay there until a milestone is worth naming. To draw that line, rename the
heading to `## 0.1.NN (YYYY-MM-DD)` and open a fresh `## Unreleased` above it.
Cutting in the same commit as the last change is tidiest but not required: when
Unreleased is empty the release job falls back to the newest named section, so
the build carrying a cut still publishes real notes rather than replacing good
notes with none.

Written for someone deciding whether to download, not for someone reading the
diff. A change nobody can notice does not belong here; a change that alters
what the application does to your model does, even when it is a fix.

This file starts on 2026-08-29, the day of the first public build. For anything
before it, see the
[commit history](https://github.com/MakerViking/brokkrsculpt/commits/main).

## Unreleased

### The first public build

BrokkrSculpt is now downloadable for **Linux, Windows and macOS**. Nothing is
signed yet, so both Windows and macOS will warn you about the download — the
release notes quote each warning and say what to do about it.

The three are not equally finished, and
[the README says specifically what is unproven on each](https://github.com/MakerViking/brokkrsculpt#what-works-on-which-platform).
Linux is where this is built and used every day. Windows and macOS compile from
the same commit and pass the same suite, but have had far less time in front of
real hardware.

### Added

- **Sign in to TinkerAtlas, optionally.** It exists for one reason: an
  anonymous bug report cannot be answered. Signed out, a report carries no
  account and no credential; signed in, it goes under your name — and the line
  above the Send button says which before you press it.
- **A bug button in the corner of the viewport**, so reporting something does
  not require finding a menu.
- **Stylus and SpaceMouse on Windows and macOS.** Written, compiled on each
  platform, and never yet connected to hardware by anyone involved. The PEN and
  PUCK panels say which state they are in — listening, or actually reading — so
  a single screenshot from someone with a tablet settles it. If it never starts
  reading, strokes run at full pressure exactly as a mouse does.

### Changed

- **An imported model now arrives at a size you can work with.** A generative
  export carries no units and lands about two millimetres across, which is far
  too small to sculpt and was being taken at face value. It is now recognised
  and brought in at a real size.
- **An import chooses its own resolution from the mesh** rather than inheriting
  whatever the last thing on screen was built at. The same file imported twice
  used to give two different results depending on what you had open before.
- **The brush follows the model.** A 3 mm brush is a brush on a 60 mm figure and
  a pin on a 200 mm one; the radius is now a proportion of what you are working
  on rather than a fixed number.
- **Undo holds about four times as many steps** in the same memory. Nothing
  about undo behaves differently — there is simply more of it.

### Fixed

- **A mask no longer hangs in the air waiting to catch moving material.**
  Dragging an unmasked part of a model towards a masked one made the moving
  part arrive masked, and stretched it on the way in. Protection now travels
  with the material it protects.
- **The tool strip stopped clipping its own labels** once it grew a scrollbar.

### Known problems

- **Masking renders wrong on macOS.** The tint spreads past the mask and shows a
  seam between bricks. The mask itself is correct, so brushes stop where they
  should and exports are unaffected — but you cannot trust the blue to tell you
  where it is. Linux and Windows are unaffected.
- **Nothing is signed** on Windows or macOS, which is a cost rather than an
  oversight and is what the beta is partly for.
