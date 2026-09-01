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
- **Updates, checked and applied.** The application asks GitHub which build is
  published, and can download and install it for you. Three things are worth
  knowing before you decide whether you want that.

  It is **on by default and there is a tick to turn it off** — on the welcome
  screen, and in the Help menu so it stays reachable once you have turned the
  welcome screen off. Turning it off stops the check as well as the offer;
  nothing phones home to display nothing.

  Every update is **signed**, and the signature is checked before anything is
  written. What is signed is a manifest carrying the length and the SHA-256 of
  each download, so the bytes are bound to the release they claim to belong to
  rather than merely being someone's bytes. The public key is compiled into the
  application; the private half has never been on a build server.

  **Linux is the only platform where the swap itself has been run.** Windows and
  macOS use the same verified download and the same checks, and their
  replacement step has been tested against everything except real hardware —
  which, on those two, is the part that can go wrong. Each keeps a copy of the
  build it replaced and writes a `RECOVER-BROKKRSCULPT.txt` beside the
  application saying how to put it back. Set `BROKKR_NO_SELF_UPDATE=1` and the
  update is downloaded and verified but never installed, which is what macOS did
  before this and what to fall back to if the swap ever misbehaves.
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

- **Selecting with the cut tool now tells you where the button actually is.**
  Ctrl and a cut gesture protects what you circled instead of removing it, and
  the message afterwards sent you to the BODIES panel, which has no such
  button. The mask card in the corner now carries a **split off** verb beside
  invert and clear, and the message names it. The same message also named
  whichever body happened to be selected rather than the one you had just
  masked.
- **A failed update now says why, somewhere you can read it.** The install
  button on the welcome screen reported its outcome into the status line
  behind that screen, so a refused install left nothing on screen at all.
  Refusals appear on the welcome screen itself, and every update outcome is
  written to `update-log.txt` beside the crash report and included in
  Help > Report a bug — on Windows there is no console, so until now a failed
  update left no trace anywhere once the window was closed.
- **A newly installed build that will not start is retried** for a couple of
  seconds before giving up, and the message says how many attempts it took.
  Windows antivirus can hold a file it has just seen appear.
- **The "mesh pool full" warning now has a button instead of bad advice.**
  After a lot of cutting the pool can be fragmented rather than actually
  full — there is room, it is just stranded. The warning used to tell you to
  reopen the file, which meant closing a sculpt to fix a display problem.
  It now offers **Rebuild view**, which packs the pool in place. It is a
  button and not something a cut decides for you: it is a full remesh, and
  a few seconds of one arriving uninvited mid-stroke is worse than the
  warning it silences.
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
