# Auto-update

Status: plan only. Nothing described below is built. Line references are to the tree at `bce5f18`; they will drift, so trust the symbol names over the numbers.

The hard part of this is not downloading a file. It is that every mechanism we would use to make an update safe — a version that orders, a signature, a rollback, a supervisor — does not exist in this project yet, and three of the four must be got right in the first shipped binary because they cannot be retrofitted onto copies already in the field. A public key list, a manifest format and an anti-rollback floor are all v1 decisions. Get one wrong and the remedy travels down the same channel the mistake is in.

The second hard part is that we can build and test on Linux only. Windows and macOS behaviour will be inferred from CI runners, which are not desktops. Where that gap matters it is named, and Phase 4 has a human gate written into it rather than discovered in the field.

---

## What is already here

`crates/brokkr-app/src/update_check.rs` (153 lines) already ships a notify-only check. It GETs `https://api.github.com/repos/MakerViking/brokkrsculpt/releases/tags/beta`, scans out `target_commitish`, shortens it to seven characters and compares it to `build_commit()`. It refuses to run on a `-dirty` or unstamped build, so a developer's machine never makes the call. It is wired into `Brokkr::new` (`app.rs` ~1618-1632) via `Task::perform`, unconditionally — the comment there argues deliberately that the person who most needs the signal is the one who turned the welcome screen off. The answer lands in `self.newer_build`, is rendered on the welcome card (`app/panel.rs` ~767-778) and, when the card is down, in the status line (`app.rs`, `Message::UpdateChecked`).

So "notice that a different build is published" is done. What is missing is ordering, authenticity, and doing anything about it.

### A defect to fix regardless of any of this

Confirmed by reading the code, not by running it:

- `panel.rs:778` dispatches `Message::LinkOpened(update_check::RELEASE_PAGE)`.
- `app.rs` routes `LinkOpened` through `articles::open_in_browser`.
- `articles.rs:377` is `if !leads_to_tinkeratlas(link) { return Err(...) }`, and `leads_to_tinkeratlas` (`articles.rs:367`) is `link.starts_with("https://tinkeratlas.com/")`.
- `RELEASE_PAGE` is `https://github.com/MakerViking/brokkrsculpt/releases/tag/beta`.

The "get it" button therefore always fails and sets the status line to "that link does not lead to TinkerAtlas". The allowlist test at `articles.rs:471-472` pins the Join and Visit buttons and not this one, which is why nothing caught it. Independently, `open_in_browser` spawns `xdg-open`, which does not exist on Windows or macOS, so the button is dead there for a second reason.

---

## What SindriCAD does today, and why we cannot copy it

SindriCAD's updater is two keys of configuration and about 130 lines of TypeScript. CI rewrites `tauri.conf.json`'s version to `0.1.<github.run_number>`, Tauri's bundler signs each artefact with minisign, a jq script assembles `latest.json` on a rolling `beta` tag, and the client calls `check()` 8 seconds after load, raises a two-button modal, and calls `downloadAndInstall()`. Publish ordering is load-bearing and well thought through: binaries first with `--clobber`, manifest last, tag moved rather than deleted.

Five things stop us lifting it.

**It is a Tauri app and we are not.** The updater plugin, the bundler that signs, the installer formats it swaps and the semver comparison are all Tauri's. We have a plain tarball on Linux, a plain zip on Windows and a hand-built `.app` in a zip on macOS, with no installer on any platform, so we do not even know where the binary lives.

**It signs the artefacts and not the manifest.** The signature proves the bytes are Thomas's; nothing binds them to the version `latest.json` claims. Anyone who controls the endpoint can serve a genuinely-signed old build under a new version number. We sign the manifest instead.

**It has exactly one public key and no rotation path.** `handoff.md:4771` records the backup location and nothing else. If that key leaks, every installed copy is unreachable for ever and the only recovery is telling users by hand — which is precisely what GitHub had to do after its 2023 GHES key exposure.

**It has no rollback of any kind.** Nothing keeps the previous binary, nothing detects a bad update, and the release sweep deletes the previous version's assets, so a user who takes a broken build cannot even re-download the one they had.

**Its Linux applicability gate is `$APPIMAGE.is_some()`.** A `.deb`, an `.rpm`, a dev run and an extracted AppDir are indistinguishable to it. We have no AppImage at all, so that discriminator does not exist for us.

One more thing worth saying plainly: SindriCAD's own notes record that the version-N-to-N+1 hop has never been human-verified on any platform. The publish side is verified; the install side is not. That is the mistake most worth not repeating, and it is why the phase plan below spends its verification budget on the hop rather than on the manifest.

---

## The four hard problems

### 1. Version identity on a rolling tag

`CARGO_PKG_VERSION` is a frozen `0.0.1` and every push to `main` republishes the same `beta` tag. The only per-build discriminator is a git short SHA, which has no order. So today we can only ask "is the published build a different commit from mine", never "is it newer" — which is exactly what `update_check.rs` does, and why its wording says *different*.

Worse, the tag is not a build identity. `git rev-list -n1 beta` gives `ec4713c` while the assets were built from a later commit, because `gh release edit --target` updates the release record and leaves the existing tag where it is. Resolving downloads by tag *name* works and returns current assets; resolving the tag to a commit gives stale source. That is an AGPL corresponding-source problem today, independent of any updater.

**Decision: a monotonic build ordinal, not semver and not a commit.** `build.rs` already has the escape hatch — it honours an explicit `BROKKR_COMMIT` over git. Add a sibling `BROKKR_BUILD`, set in `release.yml` to `${{ github.run_number }}`, read as `option_env!("BROKKR_BUILD")` parsed to `u64`. `None` on any local or source build, and `None` means the updater is structurally inert. That is better than SindriCAD's arrangement, which needs an explicit dev-gate because its dev builds report `0.1.0` and therefore sort below every published build for ever.

Rejected: bumping `CARGO_PKG_VERSION` per release. It needs a commit per release and every push to `main` republishes, so the tree would churn continuously; SindriCAD's in-place rewrite also dirties the tree, which in our `build.rs` would append `-dirty` to every published build's commit stamp.

Two traps to write down. `github.run_number` restarts at 1 if `release.yml` is renamed or recreated, so the workflow must compute `BASE + run_number` with `BASE` a literal recorded in the workflow, bumped by hand if the identity ever changes. And `build.rs` returns early when `BROKKR_COMMIT` is set — which CI sets — so the `BROKKR_BUILD` emission and its `cargo:rerun-if-env-changed=BROKKR_BUILD` must go **before** that early return, or `Swatinem/rust-cache` will serve a warm build script output and publish a binary that reports the previous ordinal. That failure is silent, self-perpetuating and would have survived every verification step anyone would naturally write, so CI must read the ordinal back out of the binary it just built and fail if it disagrees.

### 2. Verification, with no OS code signing

We have no Apple Developer ID and no Authenticode certificate, so the OS will not vouch for anything. On Linux there is no gate at all. A compiled-in public key is the only available root of trust, and the first install remains trust-on-first-use no matter what we do.

The manifest is the thing that must be signed, because the manifest is what carries the ordinal, the length and the digest. One minisign signature over one manifest authenticates all three in a single act, for every platform at once, rather than three per-artefact signatures that authenticate bytes and nothing about which release they belong to.

The runner-up design proposed putting `version:… target:… len:…` in minisign's trusted comment, which is covered by the second (global) Ed25519 check, and keeping the per-artefact signature. That is clever and it is cheap, but it is one signature per platform per release with three chances for a missing `.sig` to silently drop a platform — SindriCAD's `WARNING:`-and-continue failure mode. The signed manifest wins.

### 3. Replacement per platform

Linux is the easy one and it is genuinely easy: `rename(2)` acts on the directory entry, not the inode, so writing `brokkrsculpt.new` in the *same directory* and renaming it over the target is atomic and does not disturb the running process, which keeps its old inode until it exits. The permission that matters is write and execute on the **containing directory**, not on the file.

Windows is the hard one: the kernel refuses to unlink a running image but permits renaming it, so the swap is rename-aside then rename-into-place, with a real-time AV minifilter holding handles on freshly written executables and on-access scan timeouts documented in the tens of seconds.

macOS is the one we should not attempt. The unit of replacement is the whole `.app`; any edit inside a signed bundle invalidates the signature, which on Apple Silicon is SIGKILL at exec; Sequoia removed the Control-click bypass and a *fully unsigned* app does not appear in the Privacy & Security pane to be excused at all. Self-replacing there takes a working install and can make it unlaunchable. macOS gets notify-and-stage, permanently, until there is a Developer ID.

### 4. Failure recovery

There is no supervisor and there will not be one. The failure the user cannot recover from alone is "the new build crashes before it draws anything" — the app that would fetch the fix is the app that is dead. `crash.rs` already solves the shape of this problem (leave a report, say so on the next launch) and the same shape solves this one: a marker written before the restart, cleared once the process has been alive long enough, and a launch that finds it still there reverts.

Everything before the final `rename` must be non-destructive, and the `rename` itself must be the only irreversible step.

---

## The design, component by component

### Build ordinal — `crates/brokkr-app/build.rs`, `.github/workflows/release.yml`, `app.rs`

`BROKKR_BUILD` emitted before the `BROKKR_COMMIT` early return, with `cargo:rerun-if-env-changed=BROKKR_BUILD`. A `build_number() -> Option<u64>` beside `build_commit()`. CI sets `BROKKR_BUILD=$((BASE + github.run_number))` and asserts the built binary reports it.

The updater refuses to do anything when: `build_number()` is `None`; `build_commit()` is `unknown` or ends `-dirty`; or the resolved executable sits under a `target/` directory. The first alone would cover an honest local build, but the other two are free and they are the difference between "inert" and "one Enter key away from overwriting the developer's own binary with a CI artefact". `update_check.rs` already refuses on the dirty/unknown grounds and explains why; that refusal must survive the rewrite.

### Payload shape — raw executables, no archive reader

**This is taken from the runner-up design and it beats the winning one.** The winning design proposed a hand-rolled tar reader in `update/tar.rs`; the reviewers then pointed out that the Windows and macOS assets are zips, so a second hand-rolled extractor would be needed too, and neither was in the estimate. Both disappear if the update payload is not an archive.

`release.yml` publishes, alongside the existing human-facing `tar.gz`/`zip` assets, build-stamped raw payloads:

- `brokkrsculpt-<build>-linux-x86_64` (the stripped-of-nothing release ELF, 33 MB measured)
- `brokkrsculpt-<build>-windows-x86_64.exe` (18.8 MB measured)
- `brokkrsculpt-<build>-macos-arm64.zip` (the existing `ditto` bundle zip, unchanged — macOS is download-and-hand-over, so we never open it)

Consequences: no tar parser, no zip parser, no DEFLATE bomb bound, no path-traversal defence written from scratch, and — the one that matters most — the signed SHA-256 is the digest of the exact bytes we make executable. There is no window between "verified the archive" and "installed whatever came out of it".

We do not strip. `crash.rs` calls `Backtrace::force_capture()` and those reports are uploaded and fingerprinted; stripping the updater payload only would give self-updated users worse crash reports than everyone else. 33 MB per update is accepted. If that becomes a complaint, gzip alone (no tar) is available at zero new crates — `flate2` is already in the graph via `png` → `image` — and would need an added signed uncompressed-size field as the decompressor's cap. Trigger: a user says the download is too slow.

### Signed manifest — two release assets

`latest.conf` and `latest.conf.minisig`, fetched as plain release-download URLs rather than through the GitHub API. Today's API poll is rate-limited to 60 requests per hour per address; one makerspace behind one NAT breaks it for everyone there, silently, because failure is deliberately silence. Asset URLs go through the CDN with no such limit.

Format is the flat `key = value` this project already reads, parsed by `paths::entries` — no new parser, no nesting rules, and `serde_json` (already a direct dependency at `Cargo.toml:41`) is not needed. `expires` is an integer Unix timestamp, because there is no date crate in the graph and adding one to parse RFC 3339 would be absurd.

```
seq = 118
build = 118
key_epoch = 0
expires = 1790000000
minimum_build = 101
linux-x86_64.name = brokkrsculpt-118-linux-x86_64
linux-x86_64.size = 33408944
linux-x86_64.sha256 = <64 hex>
windows-x86_64.name = brokkrsculpt-118-windows-x86_64.exe
...
```

The manifest supplies a **filename, never a URL**. The prefix `https://github.com/MakerViking/brokkrsculpt/releases/download/beta/` is a literal in the binary, and the name is rejected if it contains `/`, `\`, `..`, or anything outside `[A-Za-z0-9._-]`. This is what bounds a key leak to "downgrade to one of my own past builds" rather than arbitrary code execution: an attacker would need the key *and* write access to a GitHub release they do not control. Two independent things must fail.

Parsing is bounded explicitly: manifest capped at 64 KiB and 200 lines, signature at 4 KiB, every field length capped, and a non-numeric, overflowing, duplicated or missing key is a **named rejection**, never a default or a saturate. These are attacker-controlled bytes at a point where nothing is trusted yet.

### `crates/brokkr-app/src/update.rs` — replaces `update_check.rs`

`update_check.rs`, `Message::UpdateChecked`, the `newer_build` field and the welcome-card button are deleted in the same change that lands the new module. Two startup connections with two disagreeing "something is newer" surfaces would be worse than what we have.

Note what that does to the privacy story, because it is the opposite of what it first looks like. `articles.rs:24-34` argues that network access happens only while the welcome screen is up, so one visible tick is an honest master switch. That property is *already broken today* by `update_check.rs`, which fires whether the card is up or not and has no tick at all. Replacing it with one connection that has a setting makes the situation better than today, not worse.

Shape:

- `pub fn fetch(base: &str) -> Result<Option<Newer>, String>` — base URL as a **function parameter**, not `env!` and not a global. `report.rs:525-530` records that making it a global broke exactly the ability to point tests at a local socket and run them in parallel.
- One `ureq::Agent` with `timeout_global(Some(20s))` and **`https_only(true)`**. I read `ureq-3.3.0/src/config.rs:868`: the default is `false`, so a redirect to plain `http` is currently followed, and GitHub asset URLs do redirect.
- Redirects capped at 3 (default is 10) and the host allowlist re-applied on **every hop**, not just the first. GitHub 302s asset downloads to `objects.githubusercontent.com`, so redirects must be followed and the allowlist must therefore be a set of permitted hosts checked per hop. Without this the signature still protects the payload's integrity, but not the client from being pointed at an arbitrary host for an attacker-chosen number of bytes.
- **`const MAX_UPDATE_BYTES: u64 = 128 * 1024 * 1024`**, compiled in, enforced first. The signed `size` is a *secondary tightening*, applied after the signature verifies. Using the declared size as the only cap would let anyone who can answer the request make us stream 4 GB into the user's install directory before the digest check fails. `MAX_THUMB_BYTES` in `articles.rs:302-308` is the right precedent precisely because it is a constant the far end cannot set.
- SHA-256 from **`ring::digest`**. `ring` 0.17 is already compiled as rustls' backend (`Cargo.lock:3040`), so this costs zero new crates and beats `sha2`. Hand-rolling SHA-256 here is forbidden — the tree has form for hand-rolling (FNV-1a in `report.rs:84`, the 3MF ZIP writer, the IOKit externs) and this is the single worst place in the codebase to be clever.

Verification order, all of it before anything is made executable:

1. Fetch the manifest and its signature under the absolute caps.
2. Verify the signature with `minisign-verify`, `allow_legacy: false`, against `TRUSTED_KEYS[manifest.key_epoch]` — the epoch selects the key, so a leaked key can only ever claim its own epoch.
3. Reject if `key_epoch < persisted floor_epoch`.
4. Reject if `key_epoch == floor_epoch && seq <= floor_seq`. A higher `key_epoch` that verifies resets `floor_seq` to the running build's ordinal.
5. Reject if `expires` has passed.
6. Reject if `seq > running_build + 10_000` (implausible; bounds a poisoned floor).
7. Pick the platform entry; an absent key means "no update for this platform", not an error.
8. Validate the filename; construct the URL from the compiled-in prefix.
9. Download with the byte cap set to the signed `size` exactly, digesting while streaming; assert bytes-written equals `size`; compare digests.

### `minisign-verify 0.2.5` — `crates/brokkr-app/Cargo.toml`

One crate, zero transitive dependencies, Ed25519 and Blake2b implemented inline, ~2.5k SLoC, MIT. Its key-id check rejects a wrong-key signature before any crypto runs. Pin the exact version and read the lockfile diff by hand: cargo has no release-age gate, so the machine-wide npm/bun cooldown does not cover this.

Record the measured count in the comment the way `ureq`'s 17 and iced canvas's 6 are recorded — and **re-measure the baseline first**. `NOTICE.md` quotes 305 crates; a reviewer measured 394; running `cargo tree -p brokkr-app --edges normal --target x86_64-unknown-linux-gnu --prefix none | awk '{print $1" "$2}' | sort -u | wc -l` today gives **314**. Three numbers means the command matters more than the figure. Write the command down next to the number.

`ed25519-dalek` is rejected: it gives a bare primitive and we would reinvent the container format, the key-id check and the prehash, badly.

### Trusted key list and epochs — `crates/brokkr-app/src/update.rs`

```rust
/// Index is the key epoch. Never reorder. Append only.
const TRUSTED_KEYS: &[&str] = &[PRIMARY, STANDBY];
```

Epoch is the index. The manifest names an epoch; that key and only that key may verify it; the client refuses an epoch below the highest it has ever accepted. This is what makes rotation actually *revoke* rather than merely accumulate: the recovery build signed by the standby is accepted, its higher epoch is persisted, and the leaked key is refused from that moment on every install that took the recovery build. New binaries drop the compromised key from the list entirely, since only the latest manifest is ever fetched and nothing needs it any more.

This corrects both source designs. The winning design said "never remove a key", which leaves a leaked key trusted for ever. The runner-up's two-key list without epochs is poisonable: an attacker with the leaked key signs a manifest with an enormous `seq`, every client that merely *sees* it writes that floor, and the honest recovery build signed by the standby is then refused by the client's own anti-rollback rule — turning a recoverable compromise into exactly the permanent orphaning the second key exists to prevent.

### `update.state` — `paths::state_file`

Flat `key = value`, written with the temp-and-rename shape from `account.rs:169-205`:

```
floor_epoch = 0
floor_seq = 118
last_check = 1790000000
last_outcome = installed build 118
```

`floor_seq` advances **only on a successful apply**, never on merely seeing an offer. Seeded at first run to the running binary's own ordinal, so a fresh install refuses any manifest older than itself out of the box — otherwise a brand-new install has no floor at all and will accept the oldest still-unexpired signed manifest, and new testers are exactly the people least able to notice they are on a build with a known bug.

When an offer is refused by the floor, the status line says so **with the number**. A silent permanent refusal is indistinguishable from a network that stopped answering, and the user has no terminal. The documented reset is "delete `update.state`", and it goes in SECURITY.md.

`last_check` exists for freshness only. It drives a line in the Help panel — "no successful update check in 34 days" — and nothing else. That is deliberately visibility rather than enforcement: TUF's expiry-lapse trap is that a maintainer who goes quiet breaks the channel for everyone, and nothing here can expire on the user.

`last_outcome` is surfaced in the About panel and included in the bug-report payload. Both already exist. It costs almost nothing and turns every future "it stopped updating" into a one-message diagnosis.

### `update.conf` — `paths::config_file`

```
check_for_updates = never | welcome | always
skip_build = 117
```

`skip_build` means a *newer* build still notifies — the thing SindriCAD lacks, where declining is not remembered at all and the modal returns 8 seconds into every session. The predicate is stated exactly, because the two source designs disagreed about it:

> notify when `manifest.build != running_build` **and** `manifest.build != skip_build`

`!=` rather than `>` because a deliberate downgrade must be expressible. The honest consequence, which must be in the message text: during a rollback the user is offered a build with a *lower* ordinal, and the UI must not call that an upgrade.

`BROKKR_UPDATE_EXPLANATION` is the packager kill switch, Zed's pattern: set it and the entire update UI is replaced by the packager's text.

The default for `check_for_updates` is an open question below, not a decision inherited from this document.

### The prompt — `app.rs`, `app/panel.rs`

It does **not** go through `guard()`. `guard()` shows the confirm dialog only `if self.would_lose_work(&action)`, and `would_lose_work` is `self.unsaved || (Import && body_count > 1)`. Routing a restart through it means a user with a *saved* document gets no prompt at all and the app restarts under them. The "free" reuse is the bug.

So: the update prompt is its own always-shown, two-button modal, and `would_lose_work` is stacked on top of it as a second condition, not as the trigger. Default focus is **Later**, not the install button — SindriCAD focuses install first, on a channel that republishes on every push to `main`, which means a stray Enter restarts the app.

The restart is specified, because it is the most user-visible step in the whole feature: spawn the **cached** executable path, then exit. If the spawn fails, do not exit — the swap has already landed, the running process is simply the old build, and say exactly that in the status line. A user who clicked restart and got a closed window with nothing coming back has no terminal and no way to tell whether their install survived.

### Cross-platform browser opener — `crates/brokkr-app/src/articles.rs`

Fixes the dead button, and does it properly rather than bolting a second prefix onto a function called `leads_to_tinkeratlas`. Rename it to `may_be_opened`, make the rule "starts with one of an exact list of prefixes, each ending in `/`" an invariant in its doc comment, add `https://github.com/MakerViking/brokkrsculpt/releases/`, and extend the test at `articles.rs:465-473` to pin `RELEASE_PAGE` the way it already pins `JOIN_URL` and `VISIT_URL`. That test fails today and passes after the fix, which is the cheapest possible proof of the defect.

For Windows, **do not use `cmd /c start`**. `cmd.exe` parses `&`, `|`, `^` and `%` off the command line before argv splitting, and Rust's `Command` quoting does not save you; the 1.77.2 BatBadBut fix covers `.bat`/`.cmd` targets, not `cmd.exe` invoked directly. `open_in_browser` is already called with externally-supplied strings — `panel.rs:713` passes `article.link` straight from the fetched RSS feed — and the guard is a prefix check, so `https://tinkeratlas.com/x&calc` would pass it and reach a shell. Use `ShellExecuteW` via `windows-sys`, which is already in the graph, and reject shell metacharacters before the call anyway. macOS uses `open`. Both arms are compile-checked by the existing cross matrix and unverified in behaviour until someone runs them.

A failed open reports the URL in a copyable status line, never on stdout. Printing it is the correct diagnosis with the wrong destination: the user this path exists for has no terminal.

### `crates/brokkr-app/src/update/apply.rs` — Linux

The executable path is resolved once at startup with `current_exe()` + `canonicalize()` and cached, before anything can move. This is an invariant with a test, not a note: after a swap, `current_exe` resolves through the running inode, which is no longer at the target name, so a second update in one session would otherwise overwrite the rollback copy and never touch the real binary. `current_exe` is used nowhere in the tree today, so all of its caveats are new here.

Refusals, before anything else:

- the containing directory is not writable by this user (this one test covers `.deb`, `.rpm`, `/usr/local` and system installs with no path list and no attempt at elevation);
- the containing directory is group- or world-writable (a loose `/opt` or a shared `/usr/local` is exactly what a bare writability test papers over);
- the path is under `target/`, or the build is dirty/unstamped;
- free space in that directory is under twice the payload size.

Then:

1. Create `.brokkrsculpt.<random>.part` in the **destination directory** with `O_EXCL | O_NOFOLLOW`, mode **0600**. Never `/tmp`: a cross-filesystem rename degrades to copy, which is the operation that fails against a running image. The random suffix and `O_EXCL` also handle two instances of the app both deciding to update.
2. Download into it under the caps, digesting as we go. Verify.
3. `fchmod` 0755 — **after** verification, not at creation. `write_private` in `account.rs` sets 0600 at creation to *narrow* exposure; creating at 0755 and then filling it with unverified network bytes inverts the point of that rule, and `~/.local/bin` is on `PATH` on most distributions.
4. `fsync`.
5. Record the current binary's SHA-256 into `update.state`, then **hard-link** it to `.brokkrsculpt.old`.
6. `rename(part, target)`.

Step 5 is a hard link, not a rename-aside. The runner-up design renames the target aside and then renames the new file in, and defends it as "two renames back to back, the window is microseconds" — but the window is real, a SIGKILL or power loss inside it leaves *nothing* at the target path, and on Unix the aside is not needed at all: `rename(new, target)` is already atomic and the running process keeps its inode. A hard link creates a second name for the same inode and removes nothing, so there is no window whatsoever. This is a place where the winning design is right and the runner-up imported a Windows constraint onto Linux.

One reviewer asked for `.old` to be mode 0600 while parked. It cannot be: a hard link shares the inode's mode with the running binary, so changing one changes both. The exposure is a known-old build of our own app in a directory we have just refused to touch unless it is user-private, which is not a new exposure. `.old` is deleted on the next successful **launch**, not the next successful update, so a known-vulnerable executable does not sit there indefinitely if no further update ever arrives.

Stale `.part` files from failed attempts are swept at startup.

### Auto-revert — `update/apply.rs` + `paths::state_file`

Before restarting, write `update-pending` containing the canonical executable path, the build ordinal just installed, and an attempt count. Keying it to the path matters: two installs on one machine (one system, one under `~/.local`) share a state directory, and an unkeyed marker means one install clears the other's.

On launch:

- marker absent → nothing to do;
- marker present, path matches, `attempt == 0` → set `attempt = 1`, continue; clear the marker once the app has drawn a frame **or** has been alive 10 seconds, whichever comes first;
- marker present, `attempt >= 1` → the previous launch died before either of those. If `.brokkrsculpt.old` exists **and its SHA-256 matches the digest recorded before the update**, rename it back, write `skip_build = <the failed build>`, clear the marker, and say so on screen.

The digest check is not ceremony: reverting means executing a file with a predictable name in a directory, and a revert path that runs whatever is sitting there is a code-execution path. Writing `skip_build` is what stops revert → notify → update → crash → revert becoming a loop with a network fetch per cycle.

The honest cost: a session that never draws and is killed inside 10 seconds — a broken driver, a headless run, someone closing the window instantly — will revert a perfectly good update on the following launch. That is the trade, and it is the right way round: being stuck on an older build beats being stuck with an app that will not start.

### `update/apply.rs` — Windows (Phase 4)

Same staging rules. The sequence is restructured so the long retry budget sits **outside** the window where the app has no binary at its own path:

1. Stage, verify, `fsync`, close.
2. Open the staged file for exclusive access and close it again, retrying with backoff for up to 60 seconds. This is where AV interference is absorbed — McAfee's documented default on-access scan timeout is 45 seconds. If the staged file has *vanished*, that is quarantine, not a lock: report it as such and stop. A self-rewriting `.exe` is a textbook Defender heuristic and deletion is at least as likely as a sharing violation.
3. Write `RECOVER-BROKKRSCULPT.txt` beside the binary: "if BrokkrSculpt has disappeared, rename brokkrsculpt.old back to brokkrsculpt.exe".
4. `rename(target, target.old)` — short budget, ~2 seconds. On failure, abort; nothing has changed.
5. `rename(part, target)` — short budget, ~5 seconds. On failure, rename `.old` back immediately.
6. Hash the resulting file rather than trusting the return code. IBM documents AV-contended installs that report success and leave corrupted files.
7. Delete the recovery note. Spawn and exit — Windows has no `exec`.

Never `MOVEFILE_COPY_ALLOWED`: Microsoft documents it as CopyFile plus DeleteFile, which is exactly what fails against a locked running image. Never `MOVEFILE_DELAY_UNTIL_REBOOT`: it needs administrators-group or LocalSystem, and its return value only tells you the registry write succeeded.

The bounded window in steps 4-5 with an automatic restore, plus a recovery note a user can act on in Explorer, is the answer to "the process dies mid-swap". It is not a perfect answer. It is the one available without a supervisor.

### `update/apply.rs` — macOS

There isn't one. Download, verify, leave the zip in the state directory, tell the user where it is and that this install cannot update itself. The message says that plainly rather than inventing a package manager, which is the falsehood SindriCAD's first copy shipped and which stranded one field report on a build with three already-fixed bugs for a week.

### Release pipeline — `.github/workflows/release.yml`

- `concurrency: { group: release, cancel-in-progress: false }`. There is none today, and two pushes to `main` in quick succession produce racing publish jobs that `--clobber` each other's fixed-name assets. Reachable states include a new binary beside an old manifest.
- Build-stamped payload names, so `latest.conf` is the **only** mutable object in the release. The risk register cannot claim "GitHub asset immutability is the trust anchor" while the workflow destroys it with `--clobber` on every push.
- The `beta` tag is pushed from the workflow. `gh release edit --target` does not move an existing tag; that is *why* the tag has been frozen since the first publish while assets have been replaced on every push since. State the mechanism in the fix or it will re-freeze.
- Sweep to the last three build-stamped payload sets (~75 MB retained on Linux), ordered **after** the manifest upload, with the keep list read back out of the published manifest so the sweep can never delete the build the signed manifest names. Retention is what makes rollback possible at all; SindriCAD's sweep deletes the previous set, which is why it has no rollback.
- During a rollback, re-upload the fixed-name human assets from the rolled-back build, after the manifest. Otherwise the download page serves build 119 while the manifest names 112, and a user who installs today is immediately offered the known-bad build as if it were an update.
- `permissions: contents: read` on the build job (it has none today, so it inherits repo defaults) and `persist-credentials: false` on checkout. The build job compiles the entire dependency tree with build scripts and proc macros; it has no business holding a write token.
- `actions/attest-build-provenance` on the publish job with `id-token: write`. Free, and it is an independent trust root that a locally-held signing key cannot provide.
- A post-publish check that re-fetches `latest.conf` over the public URL and warns if its `build` lags the newest published payload by more than one. Signing locally means a release is not live until Thomas signs it, which is deliberate, but it is also a step that will eventually be forgotten.

### `scripts/sign-release.sh` — on Thomas's machine

Roughly 60 lines. It **downloads the payload bytes and hashes them locally**. It does not read digests back from `gh release view --json assets`.

This is the correction to the winning design's biggest overclaim. That design said "no build reaches any user until a human signs a manifest for it; nothing spreads on autopilot" — but if the digests come from GitHub, the maintainer never holds the artefact, and a compromised runner or token uploads a backdoored payload which the script then faithfully signs. Keeping the key off CI protects the **key**, not the **artefact**. So: `gh release download`, hash locally with `sha256sum`, and `gh attestation verify` each payload against the expected workflow and commit SHA before writing the manifest.

Signing uses the distribution's `minisign` binary with `-H` (prehashed), not `cargo install rsign2`. The tool default produces legacy `Ed` signatures, which `verify_stream` rejects unconditionally with `UnsupportedLegacyMode` — get that wrong once and every client silently refuses the release. And `cargo install` in the presence of the key means building an external crate tree, with arbitrary build-script execution, adjacent to the signing material.

### `update-selftest` — a separate binary, not a hidden flag

The winning design proposed a hidden `--update-selftest <base>` flag in the shipped binary, citing `main.rs`'s `--tablets` and `--spacemouse` as precedent. Those are ungated runtime `std::env::args()` parsing, so by that precedent the flag ships — and then either the CI job signs with the production key (destroying the whole point of keeping it off CI) or a test public key is trusted by every shipped binary with its private half in a public repo. Either way, anyone who can launch the binary with an argument installs a binary of their choosing.

Instead:

- The selftest is a separate binary behind `#[cfg(feature = "update-selftest")]`, built only by that CI job, with its own test key list. A feature flag alone is not enough (feature unification can enable it from any workspace member), so a test asserts that a default release build contains neither the test key nor any way to override the base location.
- It takes a **local directory** containing a manifest, a signature and a payload — not a base URL. That removes the arbitrary-endpoint channel *and* the second problem, which is that `report.rs:606-611`'s `serve_once` harness serves plain `http://127.0.0.1:<port>` and `https_only(true)` would refuse it. We do not add a scheme-relaxing knob to satisfy a test.
- Consequently `update.rs` is shaped as pure functions with no HTTP in them: `verify_manifest(body: &[u8], sig: &[u8]) -> Result<Manifest>` and `check_payload(reader, size, digest)`. Every negative case — wrong key, legacy signature, expired, seq below floor, epoch below floor, size cap tripped, digest mismatch, `/` or `..` in the filename, duplicate key, non-numeric ordinal — is testable without a socket. One thin networked `fetch` remains, which the unit tests do not touch and which the real endpoint exercises.

---

## Signing and key custody, plainly

Two minisign keypairs are generated **before the first signed release ships**, on Thomas's machine, with `minisign -G`.

Both public halves are compiled into `TRUSTED_KEYS` in that order: index 0 is the live key, index 1 is the standby. This is a v1 format decision with no retrofit and it is the one thing in this design that must not be deferred.

Private halves: `minisign` cannot generate a keypair without writing the secret key to disk, so the claim "the standby's private half never touches disk" — which the winning design made in one section and contradicted in another — is not achievable and should not be written down. What is achievable: the live key stays at `~/.minisign/brokkrsculpt-0.key` on Thomas's machine, and the standby is moved off that machine after generation, to two places (a password manager entry and a printed copy), with no copy remaining on any machine that builds or publishes. Record both locations in `handoff.md` the way SindriCAD records its backup path, and say which is which.

Neither key ever enters CI. Runner-memory credential theft is a live attack, not theory — in May 2026 every tag on `actions-cool/issues-helper` was moved to imposter commits that stole CI credentials out of runner memory. A key that authorises pushing code to every tester's machine has no business in a runner. If it ever has to move there, it must be an *environment* secret behind required reviewers, and note that environment branch filters do not constrain which tags may trigger a signing build.

### If the key leaks

The blast radius is deliberately small, and the reason is worth writing into SECURITY.md because it is the strongest single argument for the filename-not-URL shape:

An attacker with the private key can sign a manifest. They cannot serve it. The endpoint is a GitHub release they cannot write to, and the URL prefix is compiled into the binary rather than supplied by the manifest. Two independent things must fail. Even if both do, the worst outcome is a downgrade to one of our own previously-published builds — every payload digest in a manifest they sign still has to match bytes we published — not arbitrary code execution.

Recovery: sign a manifest with the standby at `key_epoch = 1`. Every client that takes it persists epoch 1 and refuses epoch 0 for ever after. Ship a build whose `TRUSTED_KEYS` drops the leaked key and appends a freshly generated standby at index 2. Installs that never take the recovery build stay exposed to a downgrade; nothing can reach them, which is why the epoch mechanism has to exist from day one rather than being added when it is needed.

### Stopping a bad build

Four mechanisms, in the order they should be reached for:

1. **Delete `latest.conf` from the release.** No key, no CLI, no build — a browser and thirty seconds. Clients 404 and fall silent. This is the emergency stop, it is the only one a co-maintainer could ever perform, and it is the only one available when Thomas is unreachable. It belongs in `handoff.md` as well as here.
2. **Publish `seq+1` with `build = <the last good one>`.** Everyone walks backwards. This works only because payloads are build-stamped and retained.
3. **`minimum_build`.** Clients at or below it get a stronger "this build has a known problem" message. It never blocks launching — refusing to start a sculpting application someone already has installed is worse than the bug.
4. **The human signing gate.** Nothing reaches a user until Thomas signs, so nothing spreads on autopilot. With the local-hashing and provenance-verification fix above, this genuinely constrains the artefact and not only the key.

---

## Phased delivery

Effort figures are working days for one person who already knows this tree. Where calendar time differs materially, it is given separately, because the binding constraint on Windows is not effort — it is that every iteration is a 10-20 minute CI round trip against an MSVC toolchain we cannot reproduce locally (the local cross target is `x86_64-pc-windows-gnu`, a different ABI from the one CI ships).

The earlier drafts of this plan costed phases 0-4 at about 5.5 and about 14.5 days. Both were roughly half. The figures below are the honest ones.

### Phase 0 — fix what is already broken · 1-2 days

Deliverable: the welcome card's "get it" button opens a browser on all three platforms (`ShellExecuteW` on Windows, `open` on macOS, `xdg-open` on Linux, with metacharacter rejection); the allowlist function renamed and its test extended to pin `RELEASE_PAGE`; the `beta` tag pushed from the workflow so it tracks the commit the assets were built from; `concurrency` added to `release.yml`; `permissions: contents: read` and `persist-credentials: false` on the build job; SECURITY.md's "there are no releases, no packaged build, no installer and no updater" replaced with the truth.

Verified by: a unit test asserting `may_be_opened(RELEASE_PAGE)` — it fails before the fix and passes after, which is the whole proof. `scripts/drive.py` clicks the real button on Linux. `gh api repos/.../git/ref/tags/beta` confirms the tag matches the release target after the next publish.

Unverified: the Windows and macOS spawn arms are compile-checked only. Say so.

### Phase 1 — ordinal and signed manifest, still notify-only · 4-6 days

Deliverable: `BROKKR_BUILD` stamped and read back out of the built binary in CI; build-stamped payloads published; two keypairs generated and custody documented; `scripts/sign-release.sh`; `update.rs` with signature, epoch, floor, expiry and `minimum_build` all verified; `update_check.rs`, `Message::UpdateChecked`, `newer_build` and the old card button deleted; `update.conf` and its tick; `update.state`; the API poll gone.

Shippable on its own. It turns "a different beta exists" into "build 118 is out, you are on 112", from an authenticated source, and it reduces launch-time network connections from one unticked to one ticked.

Verified by: the pure-function test set (wrong key, legacy `Ed`, expired, seq below floor, epoch below floor, malformed, duplicate key, bad filename), all socket-free. A test asserting exactly one outbound connection at launch. A real end-to-end here: publish two builds, sign both, watch a running copy notice. The whole phase is Linux-testable.

Not in this phase, contrary to earlier drafts: "confirm a PR build produces no signature". `release.yml` has no `pull_request` trigger and the publish job additionally guards `github.event_name != 'pull_request'`, so that step is unrunnable — and since signing happens locally rather than in CI, the fork-secret question it was meant to answer does not arise.

### Phase 2 — verified download, hand over · 3-5 days

Deliverable: download the payload into the state directory under the absolute cap and then the signed size, digest while streaming, then stop and tell the user where the file is. No replacement anywhere. This is the permanent macOS answer and the interim Windows one.

Verified by: the `update-selftest` binary on ubuntu-latest, windows-latest and macos-latest, in download-and-verify mode, against a local directory — real filesystems, real digest check, headless. Negative cases: truncated body, one flipped byte, a size that disagrees with the bytes, a manifest naming a file that does not exist.

Caveat on "all three": `ci.yml`'s cross matrix marks macos-latest `informational: true` with `continue-on-error`, so a macOS-only failure goes green today. The selftest job must **not** be informational on macOS, or its evidence is worthless.

### Phase 3 — Linux in-place replacement · 5-7 days

Deliverable: cached executable path with its invariant test, the directory gates, `O_EXCL` staging at 0600, verify, chmod, hard-link aside, atomic rename, the restart prompt with its own modal, the auto-revert marker, stale-`.part` sweeping, `.old` deletion on next launch.

Verified by: `update-selftest` on ubuntu-latest asserting the ordinal changes across the hop. Fault injection: kill between stage and rename, kill after rename, corrupt the staged file, make the directory read-only, make it group-writable, fill the disk (a sized loopback or tmpfs — this one costs half a day to set up and was missing from earlier estimates). Deliberately publish a build that panics before its first frame and confirm the next launch reverts itself and writes `skip_build`. `scripts/drive.py` for the GUI. Needs no hardware anyone lacks.

### Phase 4 — Windows in-place replacement · 4-6 working days, 1.5-3 weeks calendar, plus a blocking human gate

Deliverable: the restructured sequence above, the exclusive-open wait, the bounded swap window with automatic restore, the recovery note, hash-after-write, spawn-and-exit.

Verified by: `update-selftest` on windows-latest, which is real Windows and does prove the rename dance, the retry loop and the relaunch. It does **not** prove SmartScreen or Smart App Control behaviour: runners write no Mark-of-the-Web and run in a different security context, so a green run there transfers nothing about a home machine.

**This phase does not ship until a person with a Windows desktop has taken the hop once.** Written into the phase, not discovered later. And note what that gate does not cover: one machine with Smart App Control presumably off. SAC blocks unsigned executables regardless of internet origin, and it blocks the process before any of our code runs — the auto-revert marker cannot help, because nothing of ours executes. If that turns out to bite, the answer is to hold Windows at Phase 2 until there is a certificate, not to iterate on it.

### Phase 5 — macOS · unscheduled, blocked

Apple Developer ID (\$99/yr), hardened runtime, sign inner-to-outer, `ditto -c -k --keepParent`, `notarytool submit --wait`, `stapler staple`. Only then: whole-bundle swap, translocation detection (refuse, and say "move it in Finder first" — `mv` and `NSFileManager` do not clear translocation, only a Finder move does), quarantine stripped from the staged bundle, swap performed by the **old** process.

Until every one of those exists, macOS stays on Phase 2 and says so. macos-latest runners can exercise a bundle swap mechanically and can tell you nothing about Gatekeeper, and an unsigned bundle swap is the one change in this whole plan that can turn a working install into an unlaunchable one. No macOS replacement ships on CI evidence alone.

### Totals

Phases 0-4: **17-26 working days**, four to six weeks of calendar time, plus the wait for a Windows human. No single phase claims a week and takes a month; the aggregate is where earlier estimates went wrong.

---

## What cannot be verified before it reaches users

Stated here so it is not discovered as a surprise, and repeated in `handoff.md`:

- **SmartScreen and Smart App Control at launch on Windows.** Reputation is per-file-hash and starts at zero for every new unsigned version; EV certificates no longer buy instant reputation (Microsoft's current guidance says so outright). Our own download writes no Zone.Identifier, so the SmartScreen *download* check should not fire on a self-updated exe — but SAC is not gated on Mark-of-the-Web. Unmeasurable from here, either way.
- **Gatekeeper, quarantine and translocation on macOS.** All of Phase 5, and the specific unknown of whether a user's `xattr -dr` exemption survives a bundle replacement.
- **The release pipeline itself.** `sign-release.sh`, the CDN redirect chain under `https_only(true)` and the claim that release-asset URLs have no rate limit are all first exercised on the day the first signed release ships. Rehearse the whole thing against a scratch repository first; "a real end-to-end on this machine" does not cover the GitHub half.
- **The N-to-N+1 hop as a human experiences it.** CI selftests reduce this a great deal. They are not a person double-clicking an icon. The first real update is an event, not a routine.
- **Whether the `-dirty`/`target/` refusals are complete.** They cover the cases we thought of.

---

## What we will deliberately not do

**`self_update`, at any version.** Measured at 120 crates against a workspace of ~314, and it pulls `reqwest` even with the ureq feature requested — a second HTTP stack alongside the pinned `ureq 3.3`, which breaks the stated rule that ureq stays the only network dependency (`articles.rs:17-22`). Its 1.0 and 1.1 are days old with zero field adopters; every real user found is still on 0.42. Its defaults print to stdout and block on stdin, which in a GUI with no terminal is a hang. Its genuinely unique value is macOS bundle mode, which is the platform we are deferring anyway.

**`self-replace`.** Solves self-*deletion*; we need self-*replacement*. The hard-link-and-rename shape avoids its `FILE_FLAG_DELETE_ON_CLOSE` helper-process dance and its `.__selfdelete__.exe` filename footgun. No release since 2024-09.

**`axoupdater`.** Needs a `dist` install receipt we do not write, works by downloading and executing `{app}-installer.sh`/`.ps1`, and `dist` emits no tarball, `.app` or zip target. Its PowerShell installer has no hash verification at all across 673 lines.

**Velopack.** The only option with delta updates and an official Rust+iced sample, and still rejected: its packaging CLI is a .NET application, so adopting it means a .NET toolchain in a release pipeline whose stated principle is "the matrix IS the toolchain". Its Linux output is AppImage only. Its verification is checksum-only from the same feed as the package — integrity, not authenticity.

**`cargo-packager-updater`.** Mandatory minisign is genuinely attractive and it is the only crate here that gets authenticity right by default. Rejected because adopting it means adopting its packaging system wholesale, and the updater crate has had no release since 2025-07-21. We take its best idea and none of its code.

**TUF / `tough`.** Timestamp metadata must be re-signed on a cadence for ever, and a solo maintainer who gets busy for three months breaks the channel for everyone. Threshold signing is meaningless with one signer. AWS shipped CVE-2025-2885 and CVE-2025-2886 in exactly the metadata-validation machinery a hand-roll would reproduce badly. Two ideas are taken — a signed length used as a download cap, and a monotonic floor — and the rest left.

**Per-artefact signatures.** SindriCAD's shape: no binding between the bytes and the version claimed, three signatures per release, and three chances for a missing `.sig` to drop a platform silently.

**URLs inside the manifest.** A filename, always. This is what bounds a key leak, and it is free.

**Hand-rolled tar and zip readers.** Removed entirely by the raw-executable payload. Path traversal in an update path, on bytes that arrived over the network, is the worst possible place to be writing a parser from scratch.

**Hand-rolled SHA-256.** `ring` is already compiled. This prohibition is written down because the tree has four precedents for hand-rolling and this is the one place it must not happen.

**`cmd /c start`.** Shell metacharacter injection, reachable from RSS feed content today.

**A hidden flag in the shipped binary that points the updater somewhere else.** See the selftest section.

**Signing in CI.** See key custody.

**Auto-install, and a startup modal that installs by default.** SindriCAD raises a blocking modal 8 seconds into every session, focuses the install button and does not remember a refusal. We notify on surfaces that already exist, default the focus to Later, and persist `skip_build`.

**Delta updates.** 19-33 MB payloads. No.

**Staged rollout, cohorts, update telemetry.** No server, no cohort mechanism, no analytics. The rollback that is worth having is client-side and local.

**Downloading to `/tmp`.** Cross-filesystem rename degrades to copy on both Linux and Windows, which is precisely the operation that fails against a running image.

---

## Open questions for Thomas

**1. The default for `check_for_updates`: `welcome` or `always`?**
`welcome` preserves `articles.rs:24-34`'s property — one visible tick, one honest master switch. `always` matches what `app.rs:1618-1624` argues today: the user who most needs to hear their build is stale is the one who turned the welcome screen off. Both source designs noticed this contradiction and then quietly inherited a default anyway. My recommendation is **`always`**, with the tick on the welcome screen and the docstring in `articles.rs` rewritten to say "every outbound connection has exactly one visible switch" rather than "network only while the welcome screen is up" — that is the property that was actually load-bearing, and it survives. But it is a real trade and it is yours.

**2. Does Phase 4 ship at all before there is a Windows certificate?**
The human gate proves one machine. Smart App Control is the failure mode that gate cannot sample, and it is unrecoverable in-app because nothing of ours runs. Holding Windows at Phase 2 indefinitely is a defensible answer.

**3. Where does the standby private key live?**
Two places, off the build machine. Which two? A password manager entry plus a printed copy is my suggestion; it needs to be a decision with a written procedure, not a location that gets chosen at 11 p.m. on release night.

**4. Is Apple Developer Program enrolment actually open to you?**
Apple routes individual enrolment through the Developer app on iOS or macOS. Phase 5's "\$99 plus one to two weeks" assumes the \$99 is spendable at all. Worth confirming the web enrolment path before treating Phase 5 as a scheduling question rather than a hardware one.

**5. Retention depth for build-stamped payloads.**
Three sets is ~75 MB on Linux and is what makes rollback possible. Deeper costs only storage. Shallower costs rollback range.

**6. Does the ordinal base get recorded now?**
`BASE + github.run_number` needs a starting value chosen against the current run number. Cheap to do at Phase 1, impossible to fix cleanly once clients have persisted a floor.

---

## Completeness review

A reviewer was asked what a reader would still be blocked by. Kept verbatim
rather than folded in, so the gaps stay visible.

Verified against the tree at `bce5f18` where cheap; findings marked **[verified]** were checked, the rest are read-of-the-plan.

## Wrong (would ship broken)

- **The redirect host is wrong, and it is compiled into every shipped binary. [verified]** The plan says GitHub 302s assets to `objects.githubusercontent.com`. A `curl -I` against the real release right now returns `302 → https://release-assets.githubusercontent.com/...`. A per-hop allowlist containing only the named host refuses every download, on every install, unfixable through the channel it broke — GitHub has already moved this host once. This belongs in the "v1 decisions that cannot be retrofitted" list at the top and is not there. Allowlist by suffix (`github.com`, `*.githubusercontent.com`) and say so. Also note the 302 target carries a ~1-hour signed expiry (`se=` in the query), so a staged-then-retried download needs a fresh 302, not a stored URL.
- **`timeout_global(Some(20s))` on a 33 MB payload.** ureq 3's global timeout covers the entire call including body read. One agent shared between the manifest fetch and the payload download means every user under ~13 Mbit/s fails, permanently and silently. Needs a separate agent or per-request timeout, plus a stall-based timeout rather than a wall-clock one.
- **`floor_seq` and the build ordinal are different number spaces, and the plan mixes them.** Step 4 says "a higher `key_epoch` that verifies resets `floor_seq` to the running build's ordinal", and `update.state` is "seeded at first run to the running binary's own ordinal". After any rollback `seq > build` by construction (the plan's own mechanism 2: publish `seq+1` with `build = <last good>`). So both rules set an anti-rollback floor *below* the seq of known-bad manifests. Pick one space and state it: is `floor_seq` set to `manifest.seq` or `manifest.build` on apply? The plan never says.
- **Nothing generates `seq`.** `build` is `BASE + run_number`. `seq` has no source, no monotonicity guarantee, and the rollback path re-signs by hand off a machine that may hold a stale copy of the last manifest. A repeated or lowered `seq` bricks every client that already persisted the floor — the exact unrecoverable state the epoch mechanism exists to avoid. Specify: `sign-release.sh` fetches the live `latest.conf`, asserts `new_seq > live_seq`, refuses otherwise.
- **Key selection requires parsing attacker bytes before verification.** Step 2 verifies "against `TRUSTED_KEYS[manifest.key_epoch]`" — but `key_epoch` comes out of the unverified manifest. The naive implementation parses first, which is the ordering the plan forbids everywhere else. The fix is available and free: minisign's signature carries the key id, so select the key by key id, verify, *then* parse, then assert `TRUSTED_KEYS[parsed_epoch]` is the key that verified. Say this, or someone builds the wrong order.
- **`expires` reproduces the TUF trap the plan rejects TUF for.** "Nothing here can expire on the user" is contradicted three sections earlier by a hard `expires` rejection. No re-sign cadence is defined, no expiry window is chosen, no clock-skew allowance exists, and a machine with a dead CMOS battery or a fresh VM refuses all updates silently for ever. Either drop `expires` or define who re-signs on what cadence and what the client says when it trips.
- **`.old` is deleted on the next successful launch, so the rollback window is exactly one launch.** The plan justifies this against "crashes before drawing anything", but the more common failure is "launches fine empty, dies on the user's document / second GPU init / a specific tablet". Those users get past the 10-second marker on launch 1, lose `.old`, and have no recovery at all. Keep `.old` until the next successful *update*, or until N clean launches.

## Missing

- **No mechanism for CI to read the ordinal back out of the binary.** The plan calls the warm-build-script failure "silent, self-perpetuating" and names the readback as the only defence — then never says how. `main.rs` parses only `--tablets` and `--spacemouse` **[verified]**; there is no `--version`. Add it in Phase 1 and cost it.
- **The manifest format cannot express anything but a single executable.** Raw-payload is the right call, but a future release needing a companion file (shader, icon, an updated `.desktop`) has no route through the updater and no `requires_reinstall` flag to fall back on. That is a v1 format decision of exactly the class the plan's opening paragraph says must not be got wrong, and it is unlisted.
- **No restore drill for the standby key.** A printed key you have never read back and signed with is not a backup. This is the single highest-consequence untested item in the plan and it is absent from both Phase 1 and the "cannot be verified" list. Generate, park, then *before shipping v1* restore from each location and sign a throwaway manifest.
- **The free-space gate has no mechanism.** There is no `statvfs` in the tree and no std API. `libc` and `rustix` are both in the lock **[verified]** but neither is a direct dependency of `brokkr-app`; this is new FFI or a new dep, and it is uncosted.
- **`windows-sys` is transitive only, in three versions [verified].** "Already in the graph" does not let you `use` it — `ShellExecuteW` needs a direct dependency declaration with `Win32_UI_Shell`, and a version choice. Minor, but it is stated as free and isn't.
- **The restart drops the user's open document.** Spawn-and-exit is specified; re-opening what they had is not, and neither is the save path (only "`would_lose_work` stacked as a second condition"). This is the most visible step in the feature.
- **Concurrency is handled for staging only.** `O_EXCL` + random suffix covers two instances racing on `.part`. Two instances writing `update.state`, both hard-linking `.brokkrsculpt.old` (EEXIST? clobber?), and both writing one path-keyed `update-pending` marker are unspecified.
- **`minimum_build` off-by-one.** "Clients at or below it get a stronger message" contradicts the name — a client *at* the minimum meets it. State the comparison.

## Answers to your six

- **Version identity: solved, not hand-waved.** `BASE + run_number` with the pre-early-return emission and the `rerun-if-env-changed` trap is correct and the `build.rs` early return is real **[verified]** — but the readback that guards it has no mechanism (above), and the `seq` half of the identity is missing entirely.
- **Replacement per platform: all three have a concrete answer, and macOS's answer is honestly "no".** Linux's `rename(2)`/hard-link reasoning is right and better than the alternative it rejects. Windows' sequence is the most credible part of the document. Nothing is glossed here.
- **Signature verification: nearly tight enough, two holes.** Key format, prehash, `allow_legacy: false`, and what is signed are all pinned. Missing: the key-selection ordering above, and an explicit "the trusted comment is never read" (it is attacker-influenced text that verifies).
- **Kill switch and rollback: yes, genuinely — four server-side plus a client-side revert.** The `latest.conf` deletion as an emergency stop a co-maintainer can perform is the best idea in the plan. The client-side revert is weakened by the one-launch `.old` window.
- **Estimates: credible in shape, incomplete in scope.** The per-phase numbers are plausible for someone who knows this tree. Uncosted and named as prerequisites: the scratch-repo pipeline rehearsal (a day at least, and it is the only rehearsal of the half that has never run), SECURITY.md and `handoff.md` updates, the crate-count re-measure, the "a release build contains no test key" test, the `--version` readback, and the standby restore drill. Call it +2-3 days, so 19-29.
- **Admits what it cannot test: yes, unusually well.** SmartScreen/SAC, Gatekeeper, the pipeline, and the human hop are all named, and the macos-latest `continue-on-error` catch is a genuine save. The one omission from that list is the key-restore drill.
