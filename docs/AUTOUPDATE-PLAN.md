# Auto-update

Status: **every code deliverable in this plan is implemented as of 2026-08-30.** What remains is not code — see *What is left, and why none of it is code* below.

### What is actually built, as of 2026-08-30

Verified by running it, not by reading it. **667 tests pass, 69 of them the updater's own**, clippy is clean at `-D warnings` with no `allow(dead_code)` anywhere, and the crate count went 314 → 315 exactly as predicted.

- **Phase 0.** `may_be_opened` renamed with its prefix invariant written down; `RELEASE_PAGE` moved into `articles.rs` (deleting `update_check.rs` without moving it first does not compile); `articles.rs`'s "no account" docstring corrected; `release.yml` gained `concurrency`, `permissions: contents: read` and `persist-credentials: false` on the build job, and now moves the `beta` tag to the built commit through the API; SECURITY.md's "there are no releases" replaced. The dead "get it" button and the `xdg-open`-only opener were **already fixed** in `2ce7d2c` — see the section above.
- **Phase 1.** `BROKKR_BUILD` with `BASE = 1000`, emitted before `build.rs`'s early return and read back by CI; `--version`, whose formatting lives in `update::version_report` so it is unit-testable in a binary-only crate; `update.rs` with verify-then-parse, epoch-carrying `TRUSTED_KEYS`, the anti-rollback floor, bounded parsing with named rejections, unknown keys ignored and `requires_reinstall` required; `update.conf` and `update.state`; the tick on the welcome card **and** in Help; `update_check.rs` and the `api.github.com` poll deleted; `scripts/sign-release.sh`.
- **Phase 2.** `check_payload` streaming a `ring` SHA-256 under the absolute cap and then the signed size, `download` behind a stall-timeout agent with the redirect loop followed by hand, and a status line naming where the file landed.
- **Phase 3, Linux.** `update/apply.rs`: the executable path resolved once at startup, the four gates (unstamped/dirty, under `target/`, not writable, group- or world-writable, wrong owner), `O_EXCL` staging at 0600 with `O_NOFOLLOW`, chmod and fsync only after verification, the hard-link-then-rename aside, the permanent recovery note, the atomic swap, an exclusive `update.lock` that is never waited on, the per-install `update-pending-<hex>` marker with its `resume` line, the startup sweep of `.part` and `.link`, install routed through `guard(PendingAction::Restart)`, and the launch-time auto-revert with its digest check and `skip_build` write.

  One defect found and fixed while wiring it: the download originally landed in the state directory while `install` renamed from there onto the binary — **a cross-filesystem rename, which degrades to a copy and is exactly the operation that fails against a running image**. The destination is now chosen by whether the install is replaceable, and the gates answer that before 33 MB is transferred rather than after.

**Verified end to end in the running GUI:** both tick surfaces render, the tick writes `check_for_updates = never`, a relaunch reads it back, and no outbound update call is made when it is off. **Verified against the failure the ordinal exists to catch:** a build with `BROKKR_COMMIT` set — the case that triggers the early return — still reports its ordinal.

- **Phase 4, Windows — written, compile-checked, sequence-tested, NOT shipped.** `swap_windows` implements the restructured order: wait for exclusive access with backoff *before* anything moves (60 s, where antivirus interference is absorbed), the recovery note, rename aside on a 2 s budget, rename into place on a 5 s budget with an immediate restore on failure, hash the result rather than trusting the return code, and keep the note. Never `MOVEFILE_COPY_ALLOWED`, never `MOVEFILE_DELAY_UNTIL_REBOOT`.

  The syscalls are Windows-only; **the ordering is not, and the ordering is the part that can lose an install**, so the probe is a parameter and the whole sequence is exercised on Linux: the AV wait, quarantine as a terminal outcome, the failed-second-rename restore, and the post-swap hash. That caught a real bug — an early `?` on the post-swap hash returned without restoring, leaving nothing usable at the target path, which is the exact state the sequence exists to prevent. Reached through the error handling rather than through the error.

  It stays unshipped regardless: `cargo check --target x86_64-pc-windows-gnu` is a different ABI from the MSVC toolchain CI ships, and a green runner proves nothing about SmartScreen or Smart App Control.

- **macOS is refused the swap, and the refusal is a path test rather than a `cfg`.** A defect found while wiring Phase 4: `install` fell through to the Unix path on macOS and would have replaced the executable *inside the `.app` bundle* — which invalidates the signature, and on Apple Silicon an invalid signature is SIGKILL at exec. The plan says outright that self-replacing there can turn a working install into an unlaunchable one, and the code did it anyway. Now `is_in_app_bundle` matches `.../Something.app/Contents/MacOS/exe` on **every** platform, so the refusal is reachable and testable on Linux; a `cfg` that silently stopped matching would have turned "macOS never self-replaces" into "macOS self-replaces" with nothing objecting.

- **`last_check`, `installed_build` and the Help panel's freshness line.** The only clock value stored, displayed and never compared against a rule — which is the whole difference from the `expires` field this plan rejected. A wrong clock costs a sentence, not the ability to update. Verified in the running app: a fresh install reads "no update check has succeeded yet".

- **The crash-driven offered revert, `last_outcome` in the diagnostics, provenance attestation, and the post-publish freshness warning.** The offered revert is the answer to the failure the automatic path structurally cannot catch: a build that starts fine and dies later legitimately cleared its own marker, so the signal has to be the crash report, which already existed and was not being used. Offered rather than performed — the app drew a frame, the user is at a window, and the automatic path exists for the user who never got one.

## What is left, and why none of it is code

Four items, none of which can be closed by writing more:

1. ~~The production keys have never signed anything this code verified.~~ **Done 2026-08-30.** Both secret halves signed a manifest, and both are checked through `verify_manifest` against `TRUSTED_KEYS` as compiled, by a permanent test. A future edit to that list now fails in CI rather than in the field.
2. **The standby is backed up and drilled — from one location.** Its restored copy signed a manifest the compiled-in epoch 1 key accepted, which proves the bytes and that the passphrase reproduces. Copies exist in more than one place; the others are **not yet drilled**, and until they are, the original stays on the build machine. Deleting the only proven copy while the rest are unverified is how a recovery key is lost for good. Locations and the drill are in `handoff.md`.
3. **Nothing has been published**, so the fetch path has never run against the real endpoint, and `sign-release.sh` has never run at all. Publishing is outward-facing.
4. **Phase 4 ships unproven, by decision.** Its human gate cannot be met — there is no Windows desktop on this project — and Thomas chose to ship rather than hold. `BROKKR_NO_SELF_UPDATE=1` is the escape hatch. **macOS now replaces itself too, and the plan's reason for refusing turned out not to apply.** This document ruled it out because "any edit inside a signed bundle invalidates the signature, which on Apple Silicon is SIGKILL at exec" — and that reasoning assumed a signed bundle. `release.yml` runs no `codesign`; the app is unsigned. There is no signature to invalidate, and the whole `.app` is replaced rather than edited, so the ad-hoc arm64 signature the linker applies travels inside the payload because the payload *is* the built bundle.

The real macOS obstacle is a different one this document names in passing and never acted on: **translocation**. An app launched from a quarantined location runs from a randomised read-only path, where every write fails. That is now a named refusal with the only instruction that works — move it in Finder, since neither `mv` nor `NSFileManager` clears it.

One respect in which this is *better* than a manual update: our download writes no `com.apple.quarantine`, so the replaced bundle does not earn the "is damaged and can't be opened" dialog that a browser download does. Quarantine is stripped from the staged copy anyway, best effort, because `ditto` preserves extended attributes.

Expanded with `/usr/bin/ditto -x -k`, matching how `release.yml` creates it — `unzip` flattens bundle structure, which is why the archive is made with `ditto` in the first place. No zip reader is written, so the prohibition on hand-rolled archive parsers in the update path still holds. Neither is a matter of effort, and neither is a matter of code.

**Not verified, and the list matters more than the list above.** No signed manifest has been published, so the fetch path has never run against the real endpoint; the production keys have never signed anything this code has checked; the standby key is still on the build machine and has never been restore-drilled; and there has been no scratch-repo rehearsal of the publish half.

The hard part of this is not downloading a file. It is that every mechanism we would use to make an update safe — a version that orders, a signature, a rollback, a supervisor — does not exist in this project yet, and three of the four must be got right in the first shipped binary because they cannot be retrofitted onto copies already in the field. The v1 decisions, each unfixable through the channel it breaks, are: the public key list and its epochs; the anti-rollback floor and which number space it lives in; **unknown manifest keys being ignored rather than rejected**, without which the format can never gain a field — together with the one field that must be *read* by the first binary, `requires_reinstall`, because ignoring unknown keys makes the format extensible for future readers and does nothing whatever for the copies already installed; and **the redirect host allowlist**, which is an exact match on `github.com` or any `.githubusercontent.com` subdomain — never a bare suffix test, since `evilgithub.com` ends with `github.com`, and never a single named host, since GitHub has already moved this one once and a binary that names the wrong one refuses every download on every install for ever. Get one wrong and the remedy travels down the same channel the mistake is in.

The second hard part is that we can build and test on Linux only. Windows and macOS behaviour will be inferred from CI runners, which are not desktops. Where that gap matters it is named, and Phase 4 has a human gate written into it rather than discovered in the field.

---

## What is already here

`crates/brokkr-app/src/update_check.rs` (153 lines) already ships a notify-only check. It GETs `https://api.github.com/repos/MakerViking/brokkrsculpt/releases/tags/beta`, scans out `target_commitish`, shortens it to seven characters and compares it to `build_commit()`. It refuses to run on a `-dirty` or unstamped build, so a developer's machine never makes the call. It is wired into `Brokkr::new` (`app.rs` ~1618-1632) via `Task::perform`, unconditionally — the comment there argues deliberately that the person who most needs the signal is the one who turned the welcome screen off. The answer lands in `self.newer_build`, is rendered on the welcome card (`app/panel.rs` ~767-778) and, when the card is down, in the status line (`app.rs`, `Message::UpdateChecked`).

So "notice that a different build is published" is done. What is missing is ordering, authenticity, and doing anything about it.

### A defect that was fixed while this document was being written

This section described a live bug. It is no longer one, and the correction is
kept rather than deleted because the shape of the mistake is the reusable part.

The claim was that the welcome card's "get it" button always failed: `panel.rs`
dispatches `Message::LinkOpened(update_check::RELEASE_PAGE)`, `app.rs` routes
that through `articles::open_in_browser`, and the guard there was
`link.starts_with("https://tinkeratlas.com/")`, which `RELEASE_PAGE` does not
satisfy. A second defect was stacked on it: the opener spawned `xdg-open`, which
does not exist on Windows or macOS.

Both were fixed in `2ce7d2c` — the same commit that added this document.
Verified at HEAD rather than reasoned about: `articles.rs:368` now reads
`link.starts_with("https://tinkeratlas.com/") || link == crate::update_check::RELEASE_PAGE`;
the test at `articles.rs:496` already pins `RELEASE_PAGE` the way it pins
`JOIN_URL` and `VISIT_URL`; and `open_in_browser` already selects `open` on
macOS, `rundll32.exe url.dll,FileProtocolHandler` on Windows and `xdg-open` on
Linux, with a comment arguing explicitly against putting `cmd /C start` and
therefore a shell between the app and a URL that came out of an RSS feed.

What is left of the original item is small and is what Phase 0 now carries: the
rename of `leads_to_tinkeratlas`, whose name no longer describes what it does now
that it admits a GitHub URL.

The reusable part is the failure mode, not the bug. This document asserted a
defect **[verified]**, in a section headed "confirmed by reading the code", and
the assertion was already false when it was written — because reading a file is
not the same as reading the file at the commit you are shipping from. The
document's own status line compounded it by naming a base commit, `bce5f18`, that
predates half the files it cites. A finding is a hypothesis until it is run.

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

Worse, the tag is not a build identity, and the real state is sharper than an earlier draft recorded. Measured 2026-08-30: `git ls-remote origin refs/tags/beta` gives **`c5323fd`**, while the release record targets **`eabe9cb`** — and `git merge-base --is-ancestor c5323fd origin/main` **fails**, so the tag names a commit that is not reachable from `main` at all. (An earlier draft quoted `ec4713c`, which is a stale *local* tag; check the remote, not your checkout.) The mechanism is that `gh release edit --target` updates the release record and leaves the existing tag where it is, so the tag has been frozen at the first publish while the assets have been replaced on every push since. Resolving downloads by tag *name* works and returns current assets; resolving the tag to a commit gives source that no branch contains. That is an AGPL corresponding-source problem today, independent of any updater — someone exercising their right to the source for the binary they are running is handed a commit that is not the one it was built from and cannot be reached from the repository's history.

**Decision: a monotonic build ordinal, not semver and not a commit.** `build.rs` already has the escape hatch — it honours an explicit `BROKKR_COMMIT` over git. Add a sibling `BROKKR_BUILD`, set in `release.yml` to `${{ github.run_number }}`, read as `option_env!("BROKKR_BUILD")` parsed to `u64`. `None` on any local or source build, and `None` means the updater is structurally inert. That is better than SindriCAD's arrangement, which needs an explicit dev-gate because its dev builds report `0.1.0` and therefore sort below every published build for ever.

Rejected: bumping `CARGO_PKG_VERSION` per release. It needs a commit per release and every push to `main` republishes, so the tree would churn continuously. SindriCAD's in-place rewrite avoids the commit but is worse: it dirties the tree, `build.rs` appends `-dirty` to the commit stamp, and `update_check.rs:73` **refuses to run on a dirty build** — so bumping the version in CI would make the updater structurally inert on every single shipped build. That is a loop, not a style preference. `0.1.<run_number>` is in any case an ordinal wearing a semver costume: the major and minor never move and nothing about it communicates compatibility.

Nothing here forecloses a human-facing version later. When there is a real release cut, `Cargo.toml` gets `0.2.0`, About and the changelog show it, and the updater goes on ordering by `build` — the way Chrome and Firefox carry a marketing version and a build ordinal side by side. Two numbers answering two questions. Today there is only one heading in `CHANGELOG.md` (`## Unreleased`) and `Cargo.toml:8` is frozen at `0.0.1`, so semver would have nothing to say even if it were free.

Two traps to write down. `github.run_number` restarts at 1 if `release.yml` is renamed or recreated, so the workflow must compute `BASE + run_number` with `BASE` a literal recorded in the workflow, bumped by hand if the identity ever changes. **`BASE = 1000`, settled 2026-08-30 against a `run_number` of 4, so the first stamped build is 1005**; the rename rule is "bump to the next multiple of 1000 above the highest published ordinal". A four-digit ordinal is also unmistakably not a run number and not a version, which is what makes a stale-ordinal bug visible by eye — and that failure mode is otherwise silent. And `build.rs` returns early when `BROKKR_COMMIT` is set — which CI sets — so the `BROKKR_BUILD` emission and its `cargo:rerun-if-env-changed=BROKKR_BUILD` must go **before** that early return, or `Swatinem/rust-cache` will serve a warm build script output and publish a binary that reports the previous ordinal. That failure is silent, self-perpetuating and would have survived every verification step anyone would naturally write, so CI must read the ordinal back out of the binary it just built and fail if it disagrees.

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

`BROKKR_BUILD` emitted before the `BROKKR_COMMIT` early return, with `cargo:rerun-if-env-changed=BROKKR_BUILD`. A `build_number() -> Option<u64>` beside `build_commit()`. CI sets `BROKKR_BUILD=$((BASE + github.run_number))` and reads it back out of the binary it just built.

That readback needs a channel out of the binary, and there is none today: `main.rs` scans `std::env::args()` for `--tablets` and `--spacemouse` and nothing else. So add a third arm, `--version`, in the same shape — print, then `return Ok(())`, before `crash::install()` and before `iced::application(…).run()`, which is what makes it runnable on a runner with no compositor. It prints the flat `key = value` this project already parses with `paths::entries`, so the update files, the manifest and this output all have one shape:

```
version = 0.0.1
build = 1005
commit = 2ce7d2c
```

`build = none` when `build_number()` is `None`, never `0`: a stamp that failed must not compare as a number.

The check runs in the **build** job on each of the three runners, immediately after `cargo build --release` and before `upload-artifact`, against `target/release/brokkrsculpt` directly — `brokkrsculpt.exe` on Windows. Nothing is cross compiled here, the matrix is the toolchain, so every runner can execute what it just built. Name `shell: bash` explicitly, because the Windows job's steps are written in `pwsh`. The assertion is `grep -qx "build = ${BROKKR_BUILD:?}"`: `-x` so `1005` cannot match inside `10050`, and `:?` so an unset variable fails the step rather than silently comparing the empty string against the empty string. That last detail is the whole point — the failure being guarded against is a cache serving a stale build script output, and a check that quietly compares nothing to nothing reproduces it exactly.

Cost: about eight lines in `main.rs`, four in `release.yml`, and one unit test asserting the printed ordinal equals `build_number()`. Half a day, inside Phase 1's existing figure.

This is not the hidden flag rejected under *What we will deliberately not do*. That prohibition, and the selftest section's argument against reasoning from `--tablets`, are both about what a flag can *do* — point the updater at an endpoint of the caller's choosing — not about parsing `std::env::args()`. `--version` takes no argument, opens no socket and writes nothing. Printing the commit is also the AGPL corresponding-source answer, which `build_commit`'s own doc comment already argues for.

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

Format is the flat `key = value` this project already reads, parsed by `paths::entries` — no new parser, no nesting rules, and `serde_json` (already a direct dependency at `Cargo.toml:41`) is not needed. There is no `expires` field, and no step of the verification order reads the user's clock; the reasoning is under *What we will deliberately not do*. Note in passing that availability was never the objection: `jiff 0.2.35` is already compiled as a normal dependency of `brokkr-app` through `env_logger` (`Cargo.toml:20`), so a date crate would have cost zero new crates the way `ring` does — it would still need a direct declaration to be usable, which is the `windows-sys` trap the completeness review names, but that is a line of `Cargo.toml`, not an absurdity. An earlier draft claimed there was no date crate in the graph at all. It was wrong on the fact and, more importantly, wrong on the design.

```
seq = 7
build = 1018
key_epoch = 0
minimum_build = 1012
requires_reinstall = 0
linux-x86_64.name = brokkrsculpt-1018-linux-x86_64
linux-x86_64.size = 33543336
linux-x86_64.sha256 = <64 hex>
windows-x86_64.name = brokkrsculpt-1018-windows-x86_64.exe
...
```

Note that `seq` and `build` are deliberately unequal in that sample, and small versus four-digit. They are different number spaces — `seq` counts manifests and is generated by `sign-release.sh`, `build` counts CI runs and is `BASE + run_number` — and a sample showing them equal teaches the mixing *Decisions* item 6 exists to stop. A manifest whose `build` is lower than a previous manifest's is a rollback and is legal; one whose `seq` is lower is refused.

The manifest supplies a **filename, never a URL**. The prefix `https://github.com/MakerViking/brokkrsculpt/releases/download/beta/` is a literal in the binary, and the name is rejected if it contains `/`, `\`, `..`, or anything outside `[A-Za-z0-9._-]`. This is what bounds a key leak to "downgrade to one of my own past builds" rather than arbitrary code execution: an attacker would need the key *and* write access to a GitHub release they do not control. Two independent things must fail.

**A signature that does not verify is a transient outcome, not an alarm.** `latest.conf` and `latest.conf.minisig` are two release assets replaced separately with `--clobber`, and no publish order removes the window in which a client fetches one new and one old — either order leaves one. So a verification failure writes nothing to `floor_epoch` or `floor_seq`, raises nothing on the status line, and is simply retried at the next check. Only a failure that persists across checks is worth a word to the user. Getting this wrong turns every routine publish into a scary message for whoever polled during the gap.

Parsing is bounded explicitly: manifest capped at 64 KiB and 200 lines, signature at 4 KiB, every field length capped, and a non-numeric, overflowing, duplicated or missing key is a **named rejection**, never a default or a saturate. These are attacker-controlled bytes at a point where nothing is trusted yet.

**Unknown keys are ignored, not rejected.** This is a v1 format decision with no retrofit, and it is the opposite of the rule above on purpose. Rejecting a key it does not recognise means the format can never gain a field — a `display_version`, a second digest algorithm beside `sha256` — without breaking every client already in the field, and the remedy would have to travel down the channel the mistake is in. Ignoring them costs nothing here because the manifest is signed as a whole: an attacker cannot smuggle meaning into a field no code reads, and the fields that *are* read are each validated by name.

But ignoring is forward compatibility for the *reader*, not for the *fleet*. A field added in 2027 is honoured only by binaries built after it; every copy already installed ignores it exactly as designed. So any field whose job is to *stop* an old client doing something has to ship in v1, before there is anything to stop. There is one such field.

**`requires_reinstall` is in v1 even though nothing will set it for years.** A single executable is a complete update today, and that is a measured fact rather than an assumption: the running binary reads nothing from its install directory. All three WGSL shaders are `include_str!` in `brokkr-gpu` — the only three `include_str!` anywhere in the tree; every icon is vector code in `icon.rs`, and the SVGs under `assets/icons/` are *generated out of it* by a test rather than read by the app; `logo.rs` is drawn geometry. The files that ship beside the binary — `packaging/brokkrsculpt.desktop`, `README.md`, `LICENSE`, and the macOS `Info.plist`, whose `CFBundleVersion` is a frozen `0.1.0` literal rather than anything per-build — are install-time only and none of them changes from release to release.

That holds today. It is not a property to bet the format on. A `.desktop` gaining a MIME association for a `.brokkr` document, an icon the compositor reads out of a theme directory, a shader too large to embed — any one of those makes the tarball rather than the executable the unit of the release, and a client that copies one file over another produces a half-updated install that nothing detects. `requires_reinstall = 1` is the manifest saying **this build cannot be reached by replacing one file**. The client then does what macOS does, minus the download: it names the build, opens the release page, stages nothing and swaps nothing. Steps 7 and 8 of the verification order are skipped and the payload is never fetched, because a bare executable is not what that user needs. Be honest about what that costs — the human `tar.gz`/`zip` they then download is not named by the manifest and so is not covered by the signature, so this path trades a verified download for a manual one. On the one release in years that needs it, that is the right way round, and it is written here rather than discovered.

One flag for the whole manifest, not one per platform. The finer grain cannot be added later for exactly the reason the flag itself cannot: an old client would ignore a `linux-x86_64.requires_reinstall` and self-update anyway. So the choice is made now and it is the coarse one — a needless manual reinstall for Windows users on the release where a Linux `.desktop` changed is cheaper than a required sub-key inside blocks whose existing rule is that an absent key simply means no update for that platform.

So: **required in every manifest**, written as `0` by `sign-release.sh` on every ordinary release, branching into a path Phase 2 builds anyway. Required rather than optional-defaulting-to-zero for the same reason as `minimum_build` — a field that silently disables itself when omitted is discovered on the release where it mattered. Forgetting to set the bit is the whole failure mode, so `sign-release.sh`, the one place a human is already in the loop, takes the value as an explicit argument and refuses to run without it. It cannot compute the answer: the human assets carry fixed names and are re-uploaded with `--clobber` on every push, so the previous release's companion files are gone by the time the manifest is signed. A conscious keystroke is the guard that is actually available.

### `crates/brokkr-app/src/update.rs` — replaces `update_check.rs`

`update_check.rs`, `Message::UpdateChecked`, the `newer_build` field and the welcome-card button are deleted in the same change that lands the new module. Two startup connections with two disagreeing "something is newer" surfaces would be worse than what we have.

Note what that does to the privacy story, because it is the opposite of what it first looks like. `articles.rs:24-34` argues that network access happens only while the welcome screen is up, so one visible tick is an honest master switch. That property is *already broken today* by `update_check.rs`, which fires whether the card is up or not and has no tick at all. Replacing it with one connection that has a setting makes the situation better than today, not worse.

Shape:

- `pub fn fetch(base: &str) -> Result<Option<Newer>, String>` — base URL as a **function parameter**, not `env!` and not a global. `report.rs:525-530` records that making it a global broke exactly the ability to point tests at a local socket and run them in parallel.
- **Two agents, and the payload one sets no wall-clock limit at all.** ureq documents `timeout_global` as "end-to-end, from DNS lookup to finishing reading the response body" — it covers the body. One shared agent at 20 s over a 33,543,336-byte payload demands **13.4 Mbit/s sustained**, so every user below that fails on every attempt, for ever, and silently, because failure here is deliberately silence.

So `meta()` keeps `timeout_global(Some(20s))` and serves the manifest and its signature, both capped at 64 KiB and with no business taking longer. `payload()` sets `timeout_resolve(10s)`, `timeout_connect(10s)`, `timeout_send_request(20s)` and `timeout_recv_body(30s)`, and **no `timeout_global` and no `timeout_recv_response`**. The download then has no total-time limit, only a stall limit — which is the right shape: a slow line should be allowed to finish, a dead socket should not be allowed to hang.

That rests on two ureq behaviours the names do not give away, both measured against 3.3.0 rather than read off the documentation:

- **`timeout_recv_body` is a stall timeout, not a deadline.** `CallTimings::next_timeout` recomputes it as `now + recv_body` on every read poll, so a body that keeps trickling under the budget completes however long it takes in total. An 18-second trickle under a 2-second `recv_body` succeeds; one 6-second gap fails at exactly 2.0 s with `Timeout(RecvBody)`; a half-open socket fails at 3.5 s under a 3-second budget. The review asked for a stall-based timeout and ureq already has one.
- **`timeout_recv_response` caps the body as well, so it must be left unset here.** ureq checks the *preceding* phase's deadline while reading the body, so once `headers_done + recv_response` has passed, the per-poll budget collapses to one second and the next ordinary network hiccup aborts the download reporting `Timeout(RecvResponse)`. Measured: a body arriving in 1.5-second chunks dies at 2.0 s under `recv_response(2s) + recv_body(60s)`, and completes under `recv_body(60s)` alone. Bound the header wait with `timeout_send_request` instead — it is checked while awaiting headers and **not** while reading the body: a server that accepts the connection and never answers fails at 20.1 s with `Timeout(SendRequest)`, and a slow body under that same agent is untouched.

Two agents rather than ureq's per-request override (`request.rs:349`), which does exist, because the two profiles are opposites — one bounds total time, the other must not — and one agent plus an override is one forgotten `.config()` away from the failure above.

What remains true: a stalled connection blocks its thread for up to `recv_body`. That is survivable rather than a frozen app because the fetch runs on a pool thread via `Task::perform`, so the window keeps drawing, and because we own the read loop anyway — we are digesting as we stream — so the byte counter drives a progress line and the UI can abandon the attempt.

Also **`https_only(true)`**. I read `ureq-3.3.0/src/config.rs:868`: the default is `false`, so a redirect to plain `http` is currently followed, and GitHub asset URLs do redirect.
- Redirects followed **by hand**, capped at 3 hops, with the host checked on every one. Measured 2026-08-30, the real chain is a single hop: `github.com` 302s to **`release-assets.githubusercontent.com`**. Not `objects.githubusercontent.com`, which is where GitHub used to send these and which an earlier draft of this document was about to compile into every shipped binary.

  State the rule exactly, because "match by suffix" is itself a footgun: the test is `host == "github.com" || host.ends_with(".githubusercontent.com")`, lowercased, and rejected outright if non-ASCII. `github.com` is an **equality** test and not a suffix — `evilgithub.com` ends with `github.com` — and the leading dot on the other arm does the same work, since `evil-githubusercontent.com` does not end with `.githubusercontent.com`.

  By hand, because the obvious mechanism silently does not work: ureq's middleware hook wraps `run()` (`agent.rs:228`) and the redirect loop lives *inside* `run()`, so a middleware allowlist sees the first request and no other while looking entirely correct, and `save_redirect_history` only tells you where you went once the bytes have already arrived. So `max_redirects(0)`, which returns the 3xx as-is rather than erroring — verified against 3.3.0: `Ok`, status 302, `Location` readable, even with the default `http_status_as_error` — then check the host, then issue the next request. Each hop being its own call is also what makes `https_only(true)` bite per hop rather than once, since ureq applies it at the top of every call.

  The 302 target is a short-lived signed URL — the observed `skt`/`se` pair spanned `22:31:46Z` to `23:32:27Z`, so about an hour — which is why nothing anywhere stores it; see step 8. Without this the signature still protects the payload's integrity, but not the client from being pointed at an arbitrary host for an attacker-chosen number of bytes.
- **`const MAX_UPDATE_BYTES: u64 = 128 * 1024 * 1024`**, compiled in, enforced first. The signed `size` is a *secondary tightening*, applied after the signature verifies. Using the declared size as the only cap would let anyone who can answer the request make us stream 4 GB into the user's install directory before the digest check fails. `MAX_THUMB_BYTES` in `articles.rs:302-308` is the right precedent precisely because it is a constant the far end cannot set.
- SHA-256 from **`ring::digest`**. `ring` 0.17 is already compiled as rustls' backend (`Cargo.lock:3040`), so this costs zero new crates and beats `sha2`. Hand-rolling SHA-256 here is forbidden — the tree has form for hand-rolling (FNV-1a in `report.rs:84`, the 3MF ZIP writer, the IOKit externs) and this is the single worst place in the codebase to be clever.

Verification order, all of it before anything is made executable:

1. Fetch the manifest and its signature under the absolute caps. **Neither is parsed yet.** At this point the manifest is at most 64 KiB of bytes with no fields in it, and the only thing done to the signature is minisign's own container parse, which has to happen first because the key id lives inside it.
2. Verify, **then** parse — in that order, which is the entire point of this step, so it is spelled out rather than left to whoever writes it. `key_epoch` is a manifest field, so `TRUSTED_KEYS[manifest.key_epoch]` means parsing attacker bytes to choose the key that decides whether they are attacker bytes. It is also not implementable as written: **`minisign-verify` exposes no key-id accessor** — `key_id` is private on both `Signature` and `PublicKey` — so selection is done by trying the list.

    a. `Signature::decode(sig_text)`. It takes `&str`, so the `&[u8]` that arrived off the wire gets a UTF-8 check first and a non-UTF-8 `.minisig` is a named rejection here rather than a panic later.

    b. For each entry of `TRUSTED_KEYS` in order, `PublicKey::from_base64(k)?.verify(manifest_bytes, &sig, /* allow_legacy */ false)`, stopping at the first `Ok`. This is not two signature checks: `verify` compares the 8-byte key id as its first statement and returns `UnexpectedKeyId` before it hashes anything, so a wrong key costs a 42-byte base64 decode and an integer compare — no Blake2b, no Ed25519. Every key failing is a named rejection. The variant for a legacy `Ed` signature on this path is `UnexpectedAlgorithm`; `UnsupportedLegacyMode` is what `verify_stream` returns, and we do not use `verify_stream` because the manifest is small and verified in one shot. The negative test must assert the variant this call actually produces, and must do it with `matches!` — `Error` derives `Debug` and not `PartialEq`.

    c. Keep the entry that verified, and take the epoch **from that entry**, never from its position in the slice. The two are the same number today and will not stay that way: *Trusted key list and epochs* says a new binary drops the compromised key from the list, and *If the key leaks* says the fresh standby is appended at **epoch 2** — after a drop, a bare `&[&str]` indexed by position cannot express both. So `TRUSTED_KEYS` carries the epoch beside the key and the loop returns the epoch it read out of our own binary. Either way the number came from the compiled-in list rather than off the wire, which is the property this step exists for.

    d. Only now parse the manifest, and reject unless `parsed.key_epoch` equals the epoch of the key that verified. The signature already proves who signed; this catches a release signed with one key and declared as another, which is not an attack — it is a release-night mistake, and it would otherwise persist a `floor_epoch` that no key can ever satisfy again.

    Verification answers *who signed this*. It does not answer *may they still* — a leaked key stays compiled in until a new binary ships, so its signatures go on verifying, and step 3 is what refuses them. Do not merge the two by skipping keys below the floor before trying them: it saves nothing measurable and it makes the code read as though a passing signature were an acceptance.
3. Reject if `key_epoch < persisted floor_epoch`.
4. Reject if `key_epoch == floor_epoch && seq <= floor_seq`. A higher `key_epoch` that verifies resets `floor_seq` to **zero**, not to the running build's ordinal — the ordinal is build space and `floor_seq` is manifest space, and an earlier draft mixed them here. Zero is also the right value on its own terms: a rotation is the recovery path, the whole point is that the new key's first manifest must be accepted whatever the old key persuaded this install to persist, and the epoch bump is itself the anti-rollback guarantee, since `floor_epoch` has already advanced and no epoch below it will ever verify again.
5. Reject if `seq > MAX_SEQ`, a compiled-in `1_000_000` (implausible; bounds a poisoned floor).

    An earlier draft wrote this as `seq > running_build + 10_000`, which compares manifest space to build space — the exact mixing *Decisions* item 6 settles against, and harmless today only because `BASE = 1000` happens to keep both numbers in the same range. That is precisely the kind of bug that survives review. The bound it needs is in seq space and against nothing at all: an absolute constant in the binary. What it defends is real — the floor advances on apply, so an attacker holding the key can sign an enormous `seq` over a genuinely published payload, and a client that takes it persists a floor no honest release will ever clear. A fresh install is the *most* exposed to that, having seeded `floor_seq = 0`, which is why the bound cannot be relative to a floor either. At the measured cadence — four releases in one afternoon — `1_000_000` is well over a century.
6. Pick the platform entry; an absent key means "no update for this platform", not an error.
7. Validate the filename; construct the URL from the compiled-in prefix.
8. Download with the byte cap set to the signed `size` exactly, digesting while streaming; assert bytes-written equals `size`; compare digests.

**A retry is a fresh attempt from byte zero, and no URL is ever stored.** The 302 lands on a signed URL good for about an hour — the observed `skt`/`se` pair spanned `22:31:46Z` to `23:32:27Z` — so a URL cached across a paused prompt, a failed attempt or a restart is a bug with a one-hour fuse: green in every test, failing the first time someone leaves the dialog open over lunch. Re-running step 7 costs one string concatenation and fetches a fresh 302 as a side effect, which removes the failure mode rather than documenting it. So there is no `Range` request, no resume and no partial-`.part` reuse: a failed download is repeated in full, the existing stale-`.part` sweep stays the only cleanup path, and the digest keeps meaning what it says, which a partly-filled file's digest does not until its last byte. Repeating 33 MB is the cheaper mistake than a wrong byte offset in the one code path whose output gets made executable. Resume can be added later without touching the manifest format or breaking a deployed client; the trigger is a user reporting a download that never completes.

No step reads the user's clock, and that is a property of the list rather than an accident of it. Every rejection above is decidable from exactly three things: bytes fetched under the caps, constants compiled into the binary, and the floor this install wrote for itself on its own last successful apply. None of them is the wall clock. The one clock that still matters belongs to `rustls`, which checks certificate validity during the handshake — so a machine whose clock is wrong by more than a certificate lifetime fails to connect at all, which the status line can name, rather than being silently refused an update it never mentions having fetched. An earlier draft had an `expires` rejection here; it is gone, and *What we will deliberately not do* says why, so nobody re-adds it.

### `minisign-verify 0.2.5` — `crates/brokkr-app/Cargo.toml`

One crate, zero transitive dependencies, Ed25519 and Blake2b implemented inline, ~2.5k SLoC, MIT. Its key-id check rejects a wrong-key signature before any crypto runs — `verify` compares the 8-byte key id as its first statement, ahead of the Blake2b prehash and the Ed25519 check. What it does *not* offer is a way to read that key id: `key_id` is private on both `Signature` and `PublicKey` with no accessor, so key selection is "try each entry of `TRUSTED_KEYS` until one returns `Ok`", at a cost per miss of one 42-byte base64 decode and one integer compare against a two-entry list. The verification order above depends on this and states it step by step.

**Neither comment on the signature is ever read.** `untrusted_comment()` is the easy half — `Signature::decode` accepts whatever is on the first line, unsigned, so it is straightforwardly attacker text. `trusted_comment()` is the trap, because it *does* verify: the global signature covers `signature || trusted_comment`, which is why the runner-up design wanted to carry `version:` and `len:` in it. Verified is not validated. That field is written by whatever tool signed, none of the manifest's length caps or named rejections reach it, and reading it would create a second channel of meaning running beside `latest.conf` with nothing binding the two together — a trusted comment saying `build:118` next to a manifest saying `build:112` is a state no code we write would notice. We read `latest.conf` and only `latest.conf`. Neither accessor is called anywhere in the tree, and keeping that true is a grep, cheapest as one more assertion in the test that already checks a default release build carries no test key. Pin the exact version and read the lockfile diff by hand: cargo has no release-age gate, so the machine-wide npm/bun cooldown does not cover this.

Record the measured count in the comment the way `ureq`'s 17 and iced canvas's 6 are recorded — and **re-measure the baseline first**. `NOTICE.md` quotes 305 crates; a reviewer measured 394; running `cargo tree -p brokkr-app --edges normal --target x86_64-unknown-linux-gnu --prefix none | awk '{print $1" "$2}' | sort -u | wc -l` today gives **314**. Three numbers means the command matters more than the figure. Write the command down next to the number.

`ed25519-dalek` is rejected: it gives a bare primitive and we would reinvent the container format, the key-id check and the prehash, badly.

### Trusted key list and epochs — `crates/brokkr-app/src/update.rs`

```rust
/// The epoch is carried beside the key, never inferred from position.
/// Append only; entries may be dropped, and the numbers do not close up.
const TRUSTED_KEYS: &[(u32, &str)] = &[(0, PRIMARY), (1, STANDBY)];
```

The epoch is a field, not an index, and that is load-bearing rather than fussy. An earlier draft wrote `&[&str]` with "index is the epoch, never reorder", which cannot survive this section's own recovery story: dropping the compromised key renumbers everything after it, so the standby that was epoch 1 silently becomes epoch 0 and a client that has persisted `floor_epoch = 1` refuses the very build sent to rescue it. Verification step 2c takes the epoch out of the entry that verified for exactly this reason. After a rotation the list reads `&[(1, STANDBY), (2, FRESH)]` — no epoch 0, and no confusion about which key that is.

The manifest names an epoch; the key that verifies decides which epoch that actually was; the client refuses an epoch below the highest it has ever accepted. This is what makes rotation actually *revoke* rather than merely accumulate: the recovery build signed by the standby is accepted, its higher epoch is persisted, and the leaked key is refused from that moment on every install that took the recovery build. New binaries drop the compromised key from the list entirely, since only the latest manifest is ever fetched and nothing needs it any more.

This corrects both source designs. The winning design said "never remove a key", which leaves a leaked key trusted for ever. The runner-up's two-key list without epochs is poisonable: an attacker with the leaked key signs a manifest with an enormous `seq`, every client that merely *sees* it writes that floor, and the honest recovery build signed by the standby is then refused by the client's own anti-rollback rule — turning a recoverable compromise into exactly the permanent orphaning the second key exists to prevent.

### `update.state` — `paths::state_file`

Flat `key = value`, written with the temp-and-rename shape from `account.rs:169-205`:

```
floor_epoch = 0
floor_seq = 7
installed_build = 1018
previous_sha256 = <64 hex>
last_check = 1790000000
last_outcome = installed build 1018
```

`floor_seq` advances **only on a successful apply**, never on merely seeing an offer. Seeded at first run to **zero**, per *Decisions* item 6 — not to the running binary's own ordinal, which an earlier draft specified and which is the space-mixing that breaks rollback. A fresh install therefore has no floor, deliberately: `floor_seq` is manifest space and the running ordinal is build space, and seeding one from the other on a `BASE = 1000` build would set a floor of 1005 against a `seq` counter that starts near 1, refusing every manifest that will ever be published. Silently, for ever, on every new install. The worry that justified the old seeding — a new tester landing on a build with a known bug — is `minimum_build`'s job, in build space, where it belongs.

When an offer is refused by the floor, the status line says so **with the number**. A silent permanent refusal is indistinguishable from a network that stopped answering, and the user has no terminal. The documented reset is "delete `update.state`", and it goes in SECURITY.md.

`last_check` exists for freshness only. It drives a line in the Help panel — "no successful update check in 34 days" — and nothing else. That is deliberately visibility rather than enforcement, and after the decision recorded under *What we will deliberately not do* the claim is literal rather than aspirational: there is no `expires` field in the manifest and no step of the verification order reads the clock, so nothing here can expire on the user. TUF's expiry-lapse trap is that a maintainer who goes quiet breaks the channel for everyone; a line in the Help panel says the same thing and refuses nothing. This is also the only clock value the design stores — a Unix timestamp, displayed, never compared against a rule — and it is all we have against a freeze: a client held on an old-but-valid manifest by someone who controls the endpoint looks exactly like a client whose maintainer has been quiet for a month, and this design cannot tell those apart. It does not try. It bounds the damage instead and puts the staleness where a human will see it.

`last_outcome` is surfaced in the About panel and included in the bug-report payload. Both already exist. It costs almost nothing and turns every future "it stopped updating" into a one-message diagnosis.

### `update.conf` — `paths::config_file`

```
check_for_updates = never | welcome | always
skip_build = 1017
```

`skip_build` means a *newer* build still notifies — the thing SindriCAD lacks, where declining is not remembered at all and the modal returns 8 seconds into every session. Both predicates are stated exactly, because the two source designs disagreed about the first:

> **offer** when `manifest.build != running_build` **and** `manifest.build != skip_build`
>
> **warn** whenever the running build is below `minimum_build`, whatever `skip_build` says — the comparison itself is stated once, under *Stopping a bad build*

`!=` rather than `>` because a deliberate downgrade must be expressible. The honest consequence, which must be in the message text: during a rollback the user is offered a build with a *lower* ordinal, and the UI must not call that an upgrade.

`BROKKR_UPDATE_EXPLANATION` is the packager kill switch, Zed's pattern: set it and the entire update UI is replaced by the packager's text. It suppresses the **check** as well as the UI — a distro build that keeps polling GitHub to display nothing is a connection nobody asked for and nobody can see, which is the failure `check_for_updates` exists to avoid. It is a Phase 1 deliverable; a kill switch specified in a design section and absent from every phase list is exactly the omission *Totals* warns about.

The default for `check_for_updates` is **`always`** — settled 2026-08-30, with the reasoning under *Decisions*. The tick must appear in Help/About as well as on the welcome screen, or it is a switch only reachable from a screen the user turned off.

### The prompt — `app.rs`, `app/panel.rs`

The prompt itself is **not** `guard()`'s. `guard()` shows a dialog only `if self.would_lose_work(&action)`, and `would_lose_work` is `self.unsaved || (Import && body_count > 1)`, so a user with a *saved* document would get no prompt at all and the app would restart under them. The update question has to be asked always. So: the update prompt is its own always-shown, two-button modal, with default focus on **Later** rather than the install button — SindriCAD focuses install first, on a channel that republishes on every push to `main`, which means a stray Enter restarts the app.

The order is: Install closes the update modal and calls `guard()`; `guard()` raises the unsaved-work prompt if there is work to lose; and `run_pending(PendingAction::Restart(window))` then does download → verify → swap → marker → spawn → exit. Nothing touches the disk until the work question is settled, so Cancel and a failed Save both leave the install exactly as it was — which is the property that makes routing through `guard()` safe rather than merely tidy. The *consequence* of pressing Install does go through `guard()`, as `PendingAction::Restart(window)`. The update question is asked always; the work question only when there is work, which is exactly what that gate is for. One new variant buys the whole answer path: `answer_confirm`'s Save-As detour for a document with no path (`pick_project_to_save` → `Message::SavedThenContinue`) and its rule that a write which failed leaves the prompt standing rather than proceeding on a file that was never written. The card already reads "Unsaved changes" over "*{describe}* will discard the changes to *{document_name}*", and for any non-Import action `would_lose_work` is exactly `unsaved`, so that heading is true for this variant untouched. `describe()` returns `&'static str` and has one call site, so the new arm is a fixed phrase — "Restarting into the new build", never the ordinal. And Install must take the update modal down as it calls `guard()`: `guard()` clears `top_menu` and `menu` and knows nothing about a second modal, `panel.rs` picks its overlay from `self.confirm` alone, and two modal layers over each other is a bug this tree has had before.

The restart is specified, because it is the most user-visible step in the whole feature: spawn the **cached** executable path, then exit. If the spawn fails, do not exit — the swap has already landed, the running process is simply the old build, and say exactly that in the status line. A user who clicked restart and got a closed window with nothing coming back has no terminal and no way to tell whether their install survived.

**And the document comes back.** It cannot today: `main.rs` scans argv for `--tablets` and `--spacemouse` and nothing else, and `packaging/brokkrsculpt.desktop` is a bare `Exec=brokkrsculpt` with no `%f` and no `MimeType`, so nothing in this project can be handed a file at launch. Adding that is the wrong fix here anyway — mechanism 2 publishes `seq+1` naming an *older* build, so the process being spawned can be older than the one spawning it, and an older `main.rs` ignores an argument it has never heard of and comes up empty with no error at all. Use the channel that already crosses the restart: the `update-pending` marker gains a fourth line, `resume = <absolute path>`, beside the three named under *Auto-revert*, and the launch that reads that marker for the revert check reads this too, opens it, and leaves the marker's own lifecycle alone. That reader ignores keys it does not know, for the same reason the manifest does and with the same rollback behind it — the build reading this file can be older than the build that wrote it. No argv, no session format, and nothing outside the app can name a file for it to open.

Per case this is closed rather than partial, and the reason is that `would_lose_work` is `unsaved || …`:

- **Saved and clean.** The file on disk *is* the document. The marker names `project_path` and `open_project` puts it back with its camera, brush, mirror planes and timeline keys, because `project::read` carries all of that in `ProjectState`. A reopen is not a lesser thing than a resume.
- **Unsaved.** Cannot reach the restart without the prompt, by construction: dirty implies `would_lose_work`. Save writes the file and the clean case then applies. Discard means what it means at every other gate — the marker still names the path, so the user gets the last saved state back rather than an empty window.
- **Never saved.** Save is Save As, after which there is a path. Discard loses it exactly as it does on Quit today, and the crash net is the backstop it already is: `write_autosave` is untouched by any of this and the next launch still says the autosave is in File > Recover. We deliberately do not clear it across an update restart — `clear_autosave` fires from `save_project` and nowhere else, and that is the right rule.

`recover_autosave` is **not** the mechanism, for the reason its own doc comment gives: it drops `project_path` and marks the document unsaved so that Save behaves as Save As. That is correct after a crash and wrong after a restart the user asked for, where the file they had open is still theirs.

A resume that fails is a status line and nothing more — `open_project` already reports a file that has moved and forgets it from the recent list — and a marker whose executable path does not match the cached `current_exe()` is not honoured, so two installs sharing a state directory cannot reopen each other's documents. The status line is the one thing a resume does not fight for: a revert message and a pending crash report both outrank it, on the existing rule in `Brokkr::new` that the rarer message with something to do about it wins the line, and the reopened document names itself in the title bar anyway. After a revert there is no marker and so no resume; the document is one entry down the recent list.

One side effect worth naming, because it cuts both ways — and it is narrower than it first looks. The reopen happens inside the window where the marker is still set, but that window closes on the first frame *or* at 10 seconds, whichever comes first, and a document opens fast enough that the frame usually wins. So the automatic revert catches a build that dies opening the document only when it dies **before** that first frame; past it the offered, crash-report-driven revert is the mechanism, not this one. Within that narrow window a build that dies on *the user's document* rather than on an empty one is caught by the revert — which is the failure the one-launch `.old` window otherwise misses. The price is the mirror image: a document that would crash any build takes a good update down with it and writes `skip_build`. Same trade as the 10-second rule, and the same way round.

Cost: one `PendingAction` variant, one `describe` arm, one `run_pending` arm, one marker key and one open at launch — inside the restart-prompt line Phase 3 already carries.

macOS never restarts — it is download-and-hand-over — so none of this applies there.

### Cross-platform browser opener — `crates/brokkr-app/src/articles.rs`

The dead button is already fixed (see *A defect that was fixed while this document was being written*); what is left here is doing it properly rather than leaving a second prefix bolted onto a function called `leads_to_tinkeratlas`. Rename it to `may_be_opened`, make the rule "starts with one of an exact list of prefixes, each ending in `/`" an invariant in its doc comment, add `https://github.com/MakerViking/brokkrsculpt/releases/`, and extend the test at `articles.rs:465-473` to pin `RELEASE_PAGE` the way it already pins `JOIN_URL` and `VISIT_URL`. That test fails today and passes after the fix, which is the cheapest possible proof of the defect.

On Windows the tree already does the right thing, and this section used to prescribe the wrong one. `open_in_browser` selects `rundll32.exe url.dll,FileProtocolHandler` and passes the URL as a separate argument to `std::process::Command` — no shell anywhere. An earlier draft argued at length for `ShellExecuteW` on the grounds that the alternative was `cmd /C start`, which would indeed be dangerous: `cmd.exe` parses `&`, `|`, `^` and `%` off the command line before argv splitting, Rust's `Command` quoting does not save you, the 1.77.2 BatBadBut fix covers `.bat`/`.cmd` targets rather than `cmd.exe` invoked directly, and `open_in_browser` really is called with externally-supplied strings — `panel.rs:713` passes `article.link` straight from the fetched RSS feed. But that was never the mechanism in this tree, so the danger was never live and no metacharacter rejection is needed: with no shell in the path there are no metacharacters to reject.

`ShellExecuteW` remains available and is not obviously better here. It would want `Win32_UI_Shell` added to the `windows-sys` pin — `windows-sys` 0.61 is **already a direct target dependency** of `brokkr-app`, declared for the Raw Input stylus and used by `raw_input.rs`, `tablet.rs` and `spacemouse.rs`, so that is one feature string and zero new crates, despite three `windows-sys` versions sitting in the lockfile, of which 0.52 comes in through `winit` and 0.59 through `rustix` and neither is ours to call. Against that it brings two traps that both fail quietly: it takes a `PCWSTR`, a UTF-16 buffer with an explicit trailing nul that `OsStr::encode_wide` does not append, and it returns an `HINSTANCE` that is an error code in disguise, where **success is a value greater than 32** rather than merely non-zero. Two unsafe footguns to replace a spawn that already works and is already testable. Left alone; the trigger to revisit is a Windows user reporting that a link does nothing.

A failed open reports the URL in a copyable status line, never on stdout. Printing it is the correct diagnosis with the wrong destination: the user this path exists for has no terminal.

### `crates/brokkr-app/src/update/apply.rs` — Linux

The executable path is resolved once at startup with `current_exe()` + `canonicalize()` and cached, before anything can move. This is an invariant with a test, not a note: after a swap, `current_exe` resolves through the running inode, which is no longer at the target name, so a second update in one session would otherwise overwrite the rollback copy and never touch the real binary. `current_exe` is used nowhere in the tree today, so all of its caveats are new here.

The whole sequence runs under one exclusive lock: `update.lock` in the state directory, taken with `std::fs::File::try_lock`. Stable since 1.89 and so inside this workspace's 1.90 MSRV, `flock(LOCK_EX | LOCK_NB)` on Unix and `LockFileEx` with `LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY` on Windows, zero new crates — which matters, because there is no locking of any kind in this tree today and neither `libc` nor `rustix` is a direct dependency, so the FFI answer would add supply chain for something std now does. Compiled and run on the pinned 1.98.0: a second open file description on the same path returns `TryLockError::WouldBlock`. The kernel drops the lock when the handle closes, including on process death, which is the whole reason to prefer it to an `O_EXCL` lock file — that shape needs a staleness policy and a PID check, and an updater killed mid-apply leaves one behind that wedges every future update on the machine with nothing on screen to say why. The Windows sequence in Phase 4 takes the same lock, for the same reasons.

**One lock, not one per install, and it covers every read-modify-write of `update.state` as well as the apply.** The two are not separable: `floor_seq` lives in manifest space and is a property of the channel, so a system install and one under `~/.local` share it correctly — and two instances that each read the file, change a line and rename their copy over it lose whichever floor advance landed first. That is a floor going silently backwards, which is the failure the epoch mechanism exists to make impossible; it is not a lost line of text. The check path takes the same lock for the microseconds it needs to update `last_check`. `.old` and the pending marker are per-install because of where they sit and how they are named, not because of the lock.

Three rules keep it honest. **It is never waited on**: an instance that cannot take it says "another copy of BrokkrSculpt is installing an update" and does nothing, because the user is looking at a modal and a blocking wait behind someone else's 33 MB download is a hung window. `WouldBlock` is that message; a real I/O error from the lock — a filesystem that does not implement one — refuses the apply and says so, leaving Phase 2's download-and-hand-over, which needs no lock. **It is released explicitly, with `unlock`, before the restart spawn.** Not because the child inherits it — `std::fs::File` is opened close-on-exec, so it does not — but because the child races the parent's exit: it opens its own handle, and while the parent is still winding down that handle gets `WouldBlock`, which under the rule above means the newly installed build silently skips its own first state write. Measured, not reasoned: a child process launched from a parent still holding the lock gets `WouldBlock` even though the descriptor was never inherited. And **the lock file is created once and never deleted**: unlinking a file another process holds flocked lets the next process create a fresh inode and lock that instead — ran it, both handles held a lock at once — so a tidy-up sweep that removes it removes the mutual exclusion while appearing to work. Open it `.read(true).write(true)`; the std docs are explicit that on Windows a handle opened only for append cannot be locked.

Refusals, before anything else:

- the containing directory is not writable by this user (this one test covers `.deb`, `.rpm`, `/usr/local` and system installs with no path list and no attempt at elevation);
- the containing directory is group- or world-writable (a loose `/opt` or a shared `/usr/local` is exactly what a bare writability test papers over);
- the path is under `target/`, or the build is dirty/unstamped.

There is deliberately **no free-space gate**; an earlier draft's "under twice the payload size" is removed rather than implemented, because it has no mechanism and does not earn one. There is no `statvfs` in std, and `libc` is a direct dependency of nothing in this workspace — it arrives transitively through `evdev`, `getrandom` and `errno`, which does not let you call it. So the gate is either a new dependency edge or hand-written FFI. Hand-written is not the trade `raw_hid.rs` made for IOKit: those ten externs pass opaque pointers, whereas `struct statvfs` has a layout that differs between glibc and musl and between 32- and 64-bit, with glibc redirecting the symbol to its LFS variant, so getting it wrong produces a plausible wrong number rather than a link error. Naming `libc = "0.2"` on the Linux target would cost zero new crates and get the layout right — and Phase 4 would then owe a second implementation, on the platform we are least able to test, for the same non-benefit.

It is a non-benefit because the gate predicts what the write measures. Every step that consumes space happens **before** the rename: the payload streams into `.part`, so `ENOSPC` arrives partway through the download rather than at the end of it, and the hard link in step 5 is a directory entry that also fails before the point of no return. Nothing after the rename needs space. A full disk therefore costs a wasted partial download and an error, the `.part` is swept at the next startup, and the installed binary is never touched. A gate could not do better and could not even be reliably right: space can vanish between the check and the write, and reserved blocks, per-user quotas and copy-on-write filesystems all make the number it reads a guess.

What replaces it is one line of error handling: report `ENOSPC` by name — "not enough space in `<directory>`" — rather than as a bare I/O failure, so the user gets the diagnosis the gate would have given them, one download later. Phase 3's disk-fill fault injection is what verifies this, and it was already costed; it now exercises the path that actually runs instead of a gate in front of it.

Then:

1. Create `.brokkrsculpt.<random>.part` in the **destination directory** with `O_EXCL | O_NOFOLLOW`, mode **0600**. Never `/tmp`: a cross-filesystem rename degrades to copy, which is the operation that fails against a running image. The random suffix and `O_EXCL` also handle two instances of the app both deciding to update.
2. Download into it under the caps, digesting as we go. Verify.
3. `fchmod` 0755 — **after** verification, not at creation. `write_private` in `account.rs` sets 0600 at creation to *narrow* exposure; creating at 0755 and then filling it with unverified network bytes inverts the point of that rule, and `~/.local/bin` is on `PATH` on most distributions.
4. `fsync`.
5. Record the current binary's SHA-256 into `update.state` as `previous_sha256` — the key the revert reads, named here so the two ends cannot drift — then **hard-link** it to `.brokkrsculpt.old` — as `link(target, .brokkrsculpt.<random>.link)` followed by `rename` over `.brokkrsculpt.old`. The two steps are not ceremony. `std::fs::hard_link` is `link(2)`, which fails `EEXIST` against an existing name rather than replacing it (ran it on the pinned 1.98.0: `AlreadyExists`, os error 17), so now that `.old` persists a bare link would fail on every update after the first — on the one step whose entire job is to make the next failure survivable. Unlinking `.old` first would compile and would open a window in which there is no rollback copy at all; `rename` has none, for the same reason it is used at step 6. If step 6 then fails, `.old` and the target name the same inode, which is the correct state and not a hazard. The stale-file sweep at the end of this section covers `.link` as well as `.part`: a kill between the link and the rename otherwise leaves a name on the superseded inode that nothing ever reclaims, which is one leaked 33 MB binary per crashed attempt.
6. Write `RECOVER-BROKKRSCULPT.txt` beside the binary, naming `.brokkrsculpt.old` and how to see it (`ls -a`, or Ctrl-H in a file manager), then `rename(part, target)`.

The recovery note is not Windows-only, and an earlier draft had it there alone. Every step of the Linux auto-revert — read the marker, bump `attempt`, check `.old`'s digest, rename it back — is code **inside the new build**, so it runs only for a payload that starts and then dies. A payload that cannot `exec` at all executes none of it, and that is not hypothetical on Linux: `release.yml` builds on `ubuntu-latest` against whatever glibc that image ships, so a runner image bump can produce a binary a user on an older distribution cannot start. The symptom is `version GLIBC_2.xx not found` from a shell, or nothing happening at all from a desktop icon. The marker cannot help, for exactly the reason Smart App Control cannot be helped on Windows: nothing of ours runs. One text file is the whole remedy, and a Linux user is the one most likely to have a terminal to read it in.

Step 5 is a hard link, not a rename-aside. The runner-up design renames the target aside and then renames the new file in, and defends it as "two renames back to back, the window is microseconds" — but the window is real, a SIGKILL or power loss inside it leaves *nothing* at the target path, and on Unix the aside is not needed at all: `rename(new, target)` is already atomic and the running process keeps its inode. A hard link creates a second name for the same inode and removes nothing, so there is no window whatsoever. This is a place where the winning design is right and the runner-up imported a Windows constraint onto Linux.

One reviewer asked for `.old` to be mode 0600 while parked. It cannot be: a hard link shares the inode's mode with the running binary, so changing one changes both. The exposure is a known-old build of our own app in a directory we have just refused to touch unless it is user-private, which is not a new exposure. `.old` lives until the next successful **apply** replaces it, not until the next successful launch. Deleting it a launch after the swap was meant to bound how long a superseded binary sits on disk, and it bought that for far too much. It protects nothing the auto-revert needs — a build that dies before drawing still finds `.old` in place, because the deletion was conditioned on a *successful* launch — while it removes the only recovery from the commoner failure shape: launches fine empty, dies opening the user's document, on a second GPU init, on one particular tablet. Those users clear the marker on launch 1 and would then have nothing to go back to.

Worse, the short window made the Windows recovery note a lie. `RECOVER-BROKKRSCULPT.txt` is permanent by decision and tells the user to rename `brokkrsculpt.old` back; under the one-launch rule that file was already gone by the time anyone read the note. Both cannot be right, and the note is the one that matters, because it is the entire remedy for the one Windows failure nothing of ours can catch.

The exposure just dismissed shrinks further against a decision made under *Release pipeline*: twenty build-stamped payload sets are retained on the release precisely so a rollback has something to name, so anyone who wants an old build of ours has twenty of them one click away. Be exact about what `.old` is, though, because two of the comforting things one wants to say about it are untrue: it is executable, and `~/.local/bin` is on `PATH`. What bounds it is that nothing invokes a leading-dot name, that the gates above have already refused any directory that is not user-private, and that the revert path checks its digest before renaming it back.

The steady cost is one extra binary on disk — 33.5 MB — and it is permanent rather than transient now. It begins at the swap: while `.old` and the target still name one inode the link costs no blocks at all, and the blocks start counting when `rename` gives the target a new inode. There is no free-space gate to change — it was deleted a few paragraphs above — and the `ENOSPC` path that replaced it is unaffected either way: the superseded inode's blocks were already spent before the update began.

Stale `.part` **and `.link`** files from failed attempts are swept at startup.

### Auto-revert — `update/apply.rs` + `paths::state_file`

Before restarting, write a marker in the state directory named `update-pending-<16 hex>`, where the hex is the first eight bytes of the SHA-256 of the canonical executable path. It carries the build ordinal just installed, an attempt count, and the full path — the path is there for a person reading the state directory, not for the code, which identifies the install by the filename alone.

The key is in the name rather than in the contents on purpose. Two installs on one machine — one system, one under `~/.local` — share a state directory: `state_directory` is keyed by platform and by the application name and by nothing else. One shared marker with a path field inside means each install must read the other's file, compare paths and decide not to touch it, and that comparison is a thing someone can get wrong; getting it wrong means one install reverts itself on the other's evidence. One file per install makes the collision impossible instead of merely checked, which is the same argument that keeps `.old` beside the binary. Writing it goes through the temp-and-rename shape from `account.rs`, but with a **random** temp suffix rather than that file's fixed `.json.tmp` — one fixed temp name shared by two instances is the race the rename was supposed to remove.

On launch:

- marker absent → nothing to do;
- marker present, `attempt == 0` → set `attempt = 1`, continue; clear the marker once the app has drawn a frame **or** has been alive 10 seconds, whichever comes first. There is no path comparison here: the install is identified by the marker's filename, so a marker belonging to the other install is a file this one never opens.
- marker present, `attempt >= 1` → the previous launch died before either of those. If `.brokkrsculpt.old` exists **and its SHA-256 matches `previous_sha256` in `update.state`**, rename it back over the target, write `skip_build = <the failed build>`, clear the marker, and say on screen *which build is now running* rather than only that something was undone. The rename consumes `.old`; there is no rollback copy again until the next apply makes one, and that is correct — the copy exists to undo one update, and it has.

All of the above only ever catches a launch that dies before it draws. It cannot catch the commoner shape — starts fine empty, dies opening the user's document, on a second GPU init, on one particular tablet — because that build *did* draw and legitimately cleared its marker, and no amount of keeping `.old` longer changes that. The signal for that case already exists and is not being used: `crash.rs` leaves a report and the next launch announces it. So when a crash report is pending **and** the running ordinal equals `installed_build` in `update.state` — one line written by the apply beside `floor_seq`, because `last_outcome` is prose for the About panel and must not quietly become a machine-readable flag — **and** `.brokkrsculpt.old` is present with a matching digest, the launch offers "go back to build N", performing the same rename and the same `skip_build` write. Offered and not performed: the app drew a frame, the user is at a window and can decide, and the automatic path exists precisely for the user who never got one. The offer carries no counter and no lifetime of its own — it is available exactly as long as `.old` is, which is until the next update.

Two things that shape does not get for free. The crash notice today is a status-line string and the report path goes deliberately to `log::warn!`, so there is no button to hang this on; the offer is the update prompt's own two-button modal with different text, focus on **Keep this build**, and `crash::take_pending` stays the single consumer of the report — it is taken rather than read, so the offer must be raised from that one call site or the report is announced twice.

The digest check is not ceremony: reverting means executing a file with a predictable name in a directory, and a revert path that runs whatever is sitting there is a code-execution path. Writing `skip_build` is what stops revert → notify → update → crash → revert becoming a loop with a network fetch per cycle.

The honest cost: a session that never draws and is killed inside 10 seconds — a broken driver, a headless run, someone closing the window instantly — will revert a perfectly good update on the following launch. That is the trade, and it is the right way round: being stuck on an older build beats being stuck with an app that will not start.

### `update/apply.rs` — Windows (Phase 4)

Same staging rules. The sequence is restructured so the long retry budget sits **outside** the window where the app has no binary at its own path:

1. Stage, verify, `fsync`, close.
2. Open the staged file for exclusive access and close it again, retrying with backoff for up to 60 seconds. This is where AV interference is absorbed — McAfee's documented default on-access scan timeout is 45 seconds. If the staged file has *vanished*, that is quarantine, not a lock: report it as such and stop. A self-rewriting `.exe` is a textbook Defender heuristic and deletion is at least as likely as a sharing violation.
3. Write `RECOVER-BROKKRSCULPT.txt` beside the binary: "if BrokkrSculpt has disappeared, rename brokkrsculpt.exe.old back to brokkrsculpt.exe". **The note and step 4 must name the same file.** Step 4 renames the target aside, which on Windows produces `brokkrsculpt.exe.old` and not `brokkrsculpt.old`; an earlier draft said the latter, sending the one user who needs this file looking for a name that is not on disk.
4. `rename(target, target.old)` — short budget, ~2 seconds. On failure, abort; nothing has changed.
5. `rename(part, target)` — short budget, ~5 seconds. On failure, rename `.old` back immediately.
6. Hash the resulting file rather than trusting the return code. IBM documents AV-contended installs that report success and leave corrupted files.
7. **Leave `RECOVER-BROKKRSCULPT.txt` in place**, rewritten to name the current `.old`. Spawn and exit — Windows has no `exec`.

Step 7 deleted that note in the first draft of this plan, which quietly destroyed the only recovery route for the one Windows failure mode nothing of ours can catch: Smart App Control declining to execute the new binary. Nothing of ours runs, so the auto-revert marker cannot fire; a text file beside the binary telling the user what to rename in Explorer is the entire remedy, and it costs one file.

Never `MOVEFILE_COPY_ALLOWED`: Microsoft documents it as CopyFile plus DeleteFile, which is exactly what fails against a locked running image. Never `MOVEFILE_DELAY_UNTIL_REBOOT`: it needs administrators-group or LocalSystem, and its return value only tells you the registry write succeeded.

The bounded window in steps 4-5 with an automatic restore, plus a recovery note a user can act on in Explorer, is the answer to "the process dies mid-swap". It is not a perfect answer. It is the one available without a supervisor.

### `update/apply.rs` — macOS

There isn't one. Download, verify, leave the zip in the state directory, tell the user where it is and that this install cannot update itself. The message says that plainly rather than inventing a package manager, which is the falsehood SindriCAD's first copy shipped and which stranded one field report on a build with three already-fixed bugs for a week.

### Release pipeline — `.github/workflows/release.yml`

- `concurrency: { group: release, cancel-in-progress: false }`. There is none today, and two pushes to `main` in quick succession produce racing publish jobs that `--clobber` each other's fixed-name assets. Reachable states include a new binary beside an old manifest.
- Build-stamped payload names, so `latest.conf` is the **only** mutable object in the release. Nothing may claim "GitHub asset immutability is the trust anchor" while the workflow destroys it with `--clobber` on every push.
- The `beta` tag is pushed from the workflow. `gh release edit --target` does not move an existing tag; that is *why* the tag has been frozen since the first publish while assets have been replaced on every push since. State the mechanism in the fix or it will re-freeze.
- Sweep to the last **twenty** build-stamped payload sets (~1.2 GB retained; GitHub caps neither total release size nor bandwidth, and the binding limit is 1000 assets per release), ordered **after** the manifest upload, with the keep list read back out of the published manifest so the sweep can never delete the build the signed manifest names. Retention is what makes rollback possible at all; SindriCAD's sweep deletes the previous set, which is why it has no rollback. Three sets — this document's first answer — was less than one afternoon at the measured cadence of four releases in under four hours, so mechanism 2 would routinely have had no good build left to name.
- During a rollback, re-upload the fixed-name human assets from the rolled-back build, after the manifest. Otherwise the download page serves build 119 while the manifest names 112, and a user who installs today is immediately offered the known-bad build as if it were an update.
- `permissions: contents: read` on the build job (it has none today, so it inherits repo defaults) and `persist-credentials: false` on checkout. The build job compiles the entire dependency tree with build scripts and proc macros; it has no business holding a write token.
- `actions/attest-build-provenance` on the publish job with `id-token: write`. Free, and it is an independent trust root that a locally-held signing key cannot provide.
- A post-publish check that re-fetches `latest.conf` over the public URL and warns if its `build` lags the newest published payload by more than one. Signing locally means a release is not live until Thomas signs it, which is deliberate, but it is also a step that will eventually be forgotten.

### `scripts/sign-release.sh` — on Thomas's machine

Roughly 60 lines. It **downloads the payload bytes and hashes them locally**. It does not read digests back from `gh release view --json assets`.

This is the correction to the winning design's biggest overclaim. That design said "no build reaches any user until a human signs a manifest for it; nothing spreads on autopilot" — but if the digests come from GitHub, the maintainer never holds the artefact, and a compromised runner or token uploads a backdoored payload which the script then faithfully signs. Keeping the key off CI protects the **key**, not the **artefact**. So: `gh release download`, hash locally with `sha256sum`, and `gh attestation verify` each payload against the expected workflow and commit SHA before writing the manifest.

Signing uses the distribution's `minisign` binary with `-H` (prehashed), not `cargo install rsign2`. **Correction, measured 2026-08-30 against the installed `minisign 0.12`: `-H` is no longer load-bearing.** An earlier draft warned that "the tool default produces legacy `Ed` signatures, which the client refuses", and that was true of older minisign but is not true of this one — signing with and without `-H` both produce `ED` (prehashed); the algorithm bytes were decoded from both to check. Pass `-H` anyway, because it is free, explicit, and correct against an older tool on someone else's machine, but do not build a release procedure around a danger that is not there. The real client-side variant, if a legacy signature ever does arrive, is `UnexpectedAlgorithm` from `verify` — `UnsupportedLegacyMode` is what `verify_stream` returns and the client does not call it. And `cargo install` in the presence of the key means building an external crate tree, with arbitrary build-script execution, adjacent to the signing material.

A consequence worth writing down: because the tool cannot emit a legacy signature, the negative test for one has to be **constructed** rather than generated. `update.rs`'s fixture is a genuine prehashed signature with its algorithm bytes edited from `ED` to `Ed`, which is enough to prove the client refuses it.

### `update-selftest` — a separate binary, not a hidden flag

The winning design proposed a hidden `--update-selftest <base>` flag in the shipped binary, citing `main.rs`'s `--tablets` and `--spacemouse` as precedent. Those are ungated runtime `std::env::args()` parsing, so by that precedent the flag ships — and then either the CI job signs with the production key (destroying the whole point of keeping it off CI) or a test public key is trusted by every shipped binary with its private half in a public repo. Either way, anyone who can launch the binary with an argument installs a binary of their choosing.

Instead:

- The selftest is a separate binary behind `#[cfg(feature = "update-selftest")]`, built only by that CI job, with its own test key list. A feature flag alone is not enough (feature unification can enable it from any workspace member), so a test asserts that a default release build contains neither the test key nor any way to override the base location.
- It takes a **local directory** containing a manifest, a signature and a payload — not a base URL. That removes the arbitrary-endpoint channel *and* the second problem, which is that `report.rs:606-611`'s `serve_once` harness serves plain `http://127.0.0.1:<port>` and `https_only(true)` would refuse it. We do not add a scheme-relaxing knob to satisfy a test.
- Consequently `update.rs` is shaped as pure functions with no HTTP in them: `verify_manifest(body: &[u8], sig: &[u8]) -> Result<Manifest>` and `check_payload(reader, size, digest)`. Every negative case — wrong key, legacy signature, seq below floor, epoch below floor, size cap tripped, digest mismatch, `/` or `..` in the filename, duplicate key, non-numeric ordinal — is testable without a socket. One thin networked `fetch` remains, which the unit tests do not touch and which the real endpoint exercises.

---

## Signing and key custody, plainly

Two minisign keypairs are generated **before the first signed release ships**, on Thomas's machine, with `minisign -G`.

Both public halves are compiled into `TRUSTED_KEYS` in that order: epoch 0 is the live key, epoch 1 is the standby. This is a v1 format decision with no retrofit and it is the one thing in this design that must not be deferred.

Private halves: `minisign` cannot generate a keypair without writing the secret key to disk, so the claim "the standby's private half never touches disk" — which the winning design made in one section and contradicted in another — is not achievable and should not be written down. What is achievable: the live key stays at `~/.minisign/brokkrsculpt-0.key` on Thomas's machine, and the standby is moved off that machine after generation, to two places, with no copy remaining on any machine that builds or publishes. Record both locations in `handoff.md` the way SindriCAD records its backup path, and say which is which.

Settled 2026-08-30, the two places are **a password manager entry and a printed copy held at a physically separate address**. Both carry the encrypted `.key` *and* the passphrase. Splitting the passphrase from the key across the two locations is the tempting mistake: it means losing either location loses the key, which is the wrong failure mode for the one artefact whose whole job is surviving a bad day. For a backup the threat is loss, not theft. The printed sheet carries a `sha256` line beside the key so a mistyped base64 character is caught on the page rather than on release night.

**The restore drill is a Phase 1 deliverable.** Generate, park, then restore from *each* location independently and sign a throwaway manifest with the result. A printed key that has never been read back is not a backup, it is a belief about a piece of paper. This is the highest-consequence untested item in the plan and it was absent from both the phase list and the "cannot be verified" list.

Neither key ever enters CI. Runner-memory credential theft is a live attack, not theory — in May 2026 every tag on `actions-cool/issues-helper` was moved to imposter commits that stole CI credentials out of runner memory. A key that authorises pushing code to every tester's machine has no business in a runner. If it ever has to move there, it must be an *environment* secret behind required reviewers, and note that environment branch filters do not constrain which tags may trigger a signing build.

**One exception, named now so Phase 5 does not read as impossible.** Apple code signing and notarisation require macOS tooling (`codesign`, `notarytool`, `stapler`). With no Mac, Phase 5 means an Apple certificate on a `macos-latest` runner. That is a genuine exception to the rule above and it is defensible on its own terms: an Apple certificate is scoped to Apple platforms, is revocable by Apple, and **cannot sign a manifest that installs code on Linux and Windows**. The minisign key can. They are not the same risk and this prohibition should not be read as though they are. The minisign key stays off CI unconditionally.

### If the key leaks

The blast radius is deliberately small, and the reason is worth writing into SECURITY.md because it is the strongest single argument for the filename-not-URL shape:

An attacker with the private key can sign a manifest. They cannot serve it. The endpoint is a GitHub release they cannot write to, and the URL prefix is compiled into the binary rather than supplied by the manifest. Two independent things must fail. Even if both do, the worst outcome is a downgrade to one of our own previously-published builds — every payload digest in a manifest they sign still has to match bytes we published — not arbitrary code execution.

Recovery: sign a manifest with the standby at `key_epoch = 1`. Every client that takes it persists epoch 1 and refuses epoch 0 for ever after. Ship a build whose `TRUSTED_KEYS` drops the leaked key and appends a freshly generated standby at **epoch 2** — epoch, not index: after the drop the list is `&[(1, STANDBY), (2, FRESH)]`, so the fresh key sits at index 1. Installs that never take the recovery build stay exposed to a downgrade; nothing can reach them, which is why the epoch mechanism has to exist from day one rather than being added when it is needed.

### Stopping a bad build

Four mechanisms, in the order they should be reached for:

1. **Delete `latest.conf` from the release.** No key, no CLI, no build — a browser and thirty seconds. Clients 404 and fall silent. This is the emergency stop, it is the only one a co-maintainer could ever perform, and it is the only one available when Thomas is unreachable. It belongs in `handoff.md` as well as here.
2. **Publish `seq+1` with `build = <the last good one>`.** Everyone walks backwards. This works only because payloads are build-stamped and retained.
3. **`minimum_build`.** The comparison is `build_number() < manifest.minimum_build` — **strictly below**, so a client *at* the minimum meets it and is left alone, which is what the name says. It is evaluated in build space against the running binary's own ordinal, never against `seq`; after a rollback the two have diverged by construction (mechanism 2 raises `seq` while lowering `build`) and comparing the wrong one warns the whole fleet or none of it. Required in every manifest and written as `0` when nothing is known-bad, because a field that silently means "no floor" when omitted is discovered on the release where it mattered — and Decision 5 leans on this field being *used*, since it is the only bound on a leaked key's downgrade surface across twenty retained builds. A client below the floor gets a stronger message naming both numbers — "build 1012 has a known problem; build 1031 fixes it" — and gets it **even when `skip_build` names that build**, because "not now" is an answer to an offer and not to a warning. It never blocks launching: refusing to start a sculpting application someone already has installed is worse than the bug.
4. **The human signing gate.** Nothing reaches a user until Thomas signs, so nothing spreads on autopilot. With the local-hashing and provenance-verification fix above, this genuinely constrains the artefact and not only the key.

---

## Phased delivery

Effort figures are working days for one person who already knows this tree. Where calendar time differs materially, it is given separately, because the binding constraint on Windows is not effort — it is that every iteration is a 10-20 minute CI round trip against an MSVC toolchain we cannot reproduce locally (the local cross target is `x86_64-pc-windows-gnu`, a different ABI from the one CI ships).

This is the third costing. The earlier drafts came to about 5.5 and about 14.5 days, and the figures that replaced them — 17-26 — were themselves set before roughly a dozen deliverables were written into the prose above. Each pass has been wrong in the same direction and for the same reason: work described in a paragraph but never given a number. So the figures below are rebuilt bottom-up from the deliverable lists as they now stand, and where a phase moved, the sentence after its heading says which items moved it. What is deliberately excluded is named in *Totals*.

### Phase 0 — fix what is already broken · 2-3 days

The day this gained over its first figure is entirely the Windows arm: `Win32_UI_Shell` added to the existing `windows-sys` dependency, a PCWSTR that needs its trailing nul written by hand, a success test that is *greater than 32* rather than non-zero, and the metacharacter rejection in front of all of it — four ways to be wrong on the one platform that answers only through a 10-20 minute round trip. The docstring correction, the SECURITY.md rewrite and the workflow's `concurrency` and `permissions` blocks are minutes each.

Deliverable: `leads_to_tinkeratlas` renamed to `may_be_opened`, with the rule "starts with one of an exact list of prefixes, each ending in `/`" written into its doc comment as an invariant; the `beta` tag pushed from the workflow so it tracks the commit the assets were built from; `concurrency` added to `release.yml`; `permissions: contents: read` and `persist-credentials: false` on the build job; SECURITY.md's "There are no releases. BrokkrSculpt has no packaged build, no installer and no updater yet; the only way to run it is to build `main` from source" replaced with the truth, which is a public beta on three platforms; and `articles.rs:30`'s "this application has deliberately no account and no stored credential" corrected — `app.rs:1614` calls `account::load()` and `app.rs:7337-7343` fetches an avatar over the network, so that docstring is wrong today, independent of the updater, and the *Decisions* section leans on it being right.

This is a smaller phase than it was. The browser opener, the cross-platform spawn and the `RELEASE_PAGE` allowlist entry all landed in `2ce7d2c`; see *A defect that was fixed while this document was being written*. No metacharacter rejection is listed because nothing in the path reaches a shell: all three arms are a direct `Command` with the URL as one argument.

Verified by: `cargo test -p brokkr-app` on the renamed function, which already asserts `RELEASE_PAGE`, `JOIN_URL` and `VISIT_URL` are admitted and two lookalike GitHub URLs are not. `scripts/drive.py` clicks the real button on Linux. `gh api repos/.../git/ref/tags/beta` confirms the tag matches the release target after the next publish.

Unverified: the rename touches no platform-specific code, so nothing here is compile-checked-only. That was true of the previous draft of this phase, which owned the browser opener; it is not true of this one.

### Phase 1 — ordinal and signed manifest, still notify-only · 9-13 days

This is the largest phase by a distance and the one most likely to slip. The 4-6 it replaces costed the client and nothing around it. What is also in here: the key ceremony and its restore drill from *both* locations; the scratch-repo rehearsal, which is the only rehearsal the publish half will ever get before it runs for real; the `seq` generator, which has to fetch the live manifest, refuse to go backwards, and still work on the release where there is no live manifest to fetch; the `--version` arm and its readback asserted on all three runners; the tick in Help/About as a second settings surface inside a 20,912-line `app.rs`; the redirect loop, which is now client code rather than one middleware line; and SECURITY.md, `handoff.md` and the crate re-measure. If it has to be split, the seam is the signing ceremony — the ordinal, the payloads and the pipeline are all publishable before a key exists.

Deliverable: `BROKKR_BUILD` stamped with `BASE = 1000` and read back out of the built binary in CI; build-stamped payloads published; two keypairs generated, custody documented, **and restored from each of the two locations to sign a throwaway manifest**; `scripts/sign-release.sh`, including the `seq` generator that fetches the live `latest.conf` and refuses to sign unless `new_seq > live_seq`; `update.rs` with signature, epoch, floor and `minimum_build` all verified; `update_check.rs`, `Message::UpdateChecked`, `newer_build` and the old card button deleted; `update.conf` defaulting to `always`, with its tick on the welcome screen **and** in Help/About; `BROKKR_UPDATE_EXPLANATION`, suppressing the check as well as the UI; `update.state`; the API poll gone.

Shippable on its own. It turns "a different beta exists" into "build 1018 is out, you are on 1012", from an authenticated source, and it reduces launch-time network connections from one unticked to one ticked.

Verified by: the pure-function test set (wrong key, legacy `Ed`, seq below floor, epoch below floor, malformed, duplicate key, bad filename), all socket-free. A test scoped to the update path rather than to the process: no request at all when `check_for_updates = never`, and otherwise exactly the manifest and its signature. Not "one connection at launch" — with the welcome screen up there are already three (articles, avatar, update check), and the check itself is two fetches, each of which is re-issued by hand after its 302. A real end-to-end here: publish two builds, sign both, watch a running copy notice. The whole phase is Linux-testable.

Not in this phase, contrary to earlier drafts: "confirm a PR build produces no signature". `release.yml` has no `pull_request` trigger and the publish job additionally guards `github.event_name != 'pull_request'`, so that step is unrunnable — and since signing happens locally rather than in CI, the fork-secret question it was meant to answer does not arise.

### Phase 2 — verified download, hand over · 4-6 days

The download is the cheap half: with Phase 1's caps, digest and agent already in place it is a second named agent, four timeouts, and a `.part` that always starts at byte zero. The day this gained is the harness — a feature-gated second binary with its own key list, a job on three runners, and the test that a default release build carries neither the test key nor any way to move the base location. That test has to inspect a built artefact rather than call a function, so it is a CI step with its own round-trip loop.

Deliverable: download the payload into the state directory under the absolute cap and then the signed size, digest while streaming, then stop and tell the user where the file is. No replacement anywhere. This is the permanent macOS answer and the interim Windows one.

Verified by: the `update-selftest` binary on ubuntu-latest, windows-latest and macos-latest, in download-and-verify mode, against a local directory — real filesystems, real digest check, headless. Negative cases: truncated body, one flipped byte, a size that disagrees with the bytes, a manifest naming a file that does not exist.

Caveat on "all three": `ci.yml`'s cross matrix marks macos-latest `informational: true` with `continue-on-error`, so a macOS-only failure goes green today. The selftest job must **not** be informational on macOS, or its evidence is worthless.

### Phase 3 — Linux in-place replacement · 6-9 days

The free-space gate's deletion gives half a day back — it was `statvfs` FFI on two platforms for a diagnosis a named `ENOSPC` already provides. Against that, four things arrived after the first figure was set: the exclusive `update.lock` around the apply and every read-modify-write of `update.state`; the link-to-a-random-name step with the startup sweep taught the new `.link` name; the `resume =` line in the marker and the launch that opens it; and the crash-driven offered revert, which is the largest because it is new UI, a new `PendingAction` variant and a new field in the crash report rather than a reordering of steps that already exist. The fault-injection budget is unchanged and remains the expensive half.

Deliverable: cached executable path with its invariant test, the directory gates, `O_EXCL` staging at 0600, verify, chmod, hard-link aside, atomic rename, the restart prompt with its own modal, the auto-revert marker, stale `.part` **and `.link`** sweeping, the `update.lock` that serialises the whole apply and every write to `update.state`, `.old` kept until the next apply replaces it, and the crash-report-driven offer to go back.

Verified by: `update-selftest` on ubuntu-latest asserting the ordinal changes across the hop. Fault injection: kill between stage and rename, kill after rename, corrupt the staged file, make the directory read-only, make it group-writable, fill the disk (a sized loopback or tmpfs — this one costs half a day to set up and was missing from earlier estimates). Deliberately publish a build that panics before its first frame and confirm the next launch reverts itself and writes `skip_build`. `scripts/drive.py` for the GUI. Needs no hardware anyone lacks.

### Phase 4 — Windows in-place replacement · 5-7 working days, 2-3.5 weeks calendar, plus a blocking human gate

Effort barely moves: the `update.lock`, the marker's `resume` line and the `.old` lifetime are cross-platform and are written in Phase 3, so this phase inherits the code and pays only to prove it. Calendar moves, because proving it is three more things that must go green at 10-20 minutes an iteration before the human gate can start — and the gate is a person's availability on top of that, which is why it stays outside the number.

Deliverable: the restructured sequence above, the exclusive-open wait, the bounded swap window with automatic restore, the recovery note, hash-after-write, spawn-and-exit.

Verified by: `update-selftest` on windows-latest, which is real Windows and does prove the rename dance, the retry loop and the relaunch. It does **not** prove SmartScreen or Smart App Control behaviour: runners write no Mark-of-the-Web and run in a different security context, so a green run there transfers nothing about a home machine.

**This phase does not ship until a person with a Windows desktop has taken the hop once.** Written into the phase, not discovered later. And note what that gate does not cover: one machine with Smart App Control presumably off. SAC blocks unsigned executables regardless of internet origin, and it blocks the process before any of our code runs — the auto-revert marker cannot help, because nothing of ours executes. If that turns out to bite, the answer is to hold Windows at Phase 2 until there is a certificate, not to iterate on it.

### Phase 5 — macOS · unscheduled, blocked

Apple Developer ID (\$99/yr), hardened runtime, sign inner-to-outer, `ditto -c -k --keepParent`, `notarytool submit --wait`, `stapler staple`. Only then: whole-bundle swap, translocation detection (refuse, and say "move it in Finder first" — `mv` and `NSFileManager` do not clear translocation, only a Finder move does), quarantine stripped from the staged bundle, swap performed by the **old** process.

Until every one of those exists, macOS stays on Phase 2 and says so. macos-latest runners can exercise a bundle swap mechanically and can tell you nothing about Gatekeeper, and an unsigned bundle swap is the one change in this whole plan that can turn a working install into an unlaunchable one. No macOS replacement ships on CI evidence alone.

### Totals

Phases 0-4: **26-38 working days**, six to ten weeks of calendar time, plus the wait for a Windows human. That is the sum of the revised phases, not a multiplier on the old total, and it is about half again the 17-26 it replaces. A reviewer priced the same gap at +2-3 days; that was arrived at by costing the missing items without rebuilding the phases they fall into, and nine to twelve of them land inside Phase 1 alone.

Excluded, deliberately: Phase 5, which is unscheduled and stays unpriced until there is a certificate; and the Windows human's own time, which is a gate rather than a task. The failure mode this document keeps hitting is not a phase that claims a week and takes a month — it is a deliverable written into a paragraph and never carried into a number. If a further deliverable is added above and this line does not move with it, the line is wrong again.

---

## What cannot be verified before it reaches users

Stated here so it is not discovered as a surprise, and repeated in `handoff.md`:

- **SmartScreen and Smart App Control at launch on Windows.** Reputation is per-file-hash and starts at zero for every new unsigned version; EV certificates no longer buy instant reputation (Microsoft's current guidance says so outright). Our own download writes no Zone.Identifier, so the SmartScreen *download* check should not fire on a self-updated exe — but SAC is not gated on Mark-of-the-Web. Unmeasurable from here, either way.
- **Gatekeeper, quarantine and translocation on macOS.** All of Phase 5, and the specific unknown of whether a user's `xattr -dr` exemption survives a bundle replacement.
- **The release pipeline itself.** `sign-release.sh` and the claim that release-asset URLs have no rate limit are first exercised on the day the first signed release ships. The redirect chain has come off this list: it was measured on 2026-08-30 — one 302 from `github.com` to `release-assets.githubusercontent.com`, landing on a signed URL good for about an hour — and the host rule and the no-stored-URL rule are both written against that measurement. What is still unknown is the chain under *our* client rather than `curl`, since the manual redirect loop has been exercised against a local 302 and never against GitHub's, and whether GitHub adds a hop — which is what the host rule is for. Rehearse the whole thing against a scratch repository first; "a real end-to-end on this machine" does not cover the GitHub half.
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

**A manifest `expires` field.** An earlier draft of this plan had one, and it was the same trap in miniature. Expiry defends against a *freeze*: someone who holds the endpoint but not the key replays an old, genuinely signed manifest for ever, so the client never learns a fix exists. Three things make it a bad trade here. First, on any install that has ever applied an update the floor already blocks the half of that attack worth blocking — such a client can be held still but cannot be walked backwards. (A fresh install seeds `floor_seq = 0` and has no floor at all; that gap is `minimum_build`'s job, in build space, and an expiry field would not have covered it either.) What is left is "you stay on the build you have", which on a one-person beta channel is indistinguishable from Thomas being busy for a month — and enforcing against an input you cannot tell from the normal case is precisely how TUF's expiry lapse breaks a channel. Second, it would be the only rule in the entire design that depends on the user's clock, and the clock is the one input an attacker never has to touch: a fresh VM before NTP has synced, or a machine with a dead CMOS battery, gets a silent permanent refusal of every update, with no message worth reading and no terminal to read it in. Third, it needs a re-sign cadence — a recurring chore owned by the one person whose absence is the failure being modelled — and when it trips, the remedy has to travel down the channel it just closed.

Say the cost rather than pretend the refusal is free. Against someone who genuinely holds the endpoint, nothing in this design *detects* a freeze; `last_check` in the Help panel is the entire answer, and it is surfaced rather than enforced. Deleting `latest.conf` is the emergency stop for the failure that actually happens — a bad build we published — and it beats expiry there on every axis: immediate, deliberate, reversible, and performable by someone who is not Thomas. Note finally that this is one of the few format decisions here that is *not* a v1 commitment. Unknown keys are ignored, so an `expires` field could be added later without breaking a single client in the field; old clients would skip it rather than reject it, so the protection would reach new installs only. It is left out on the design, not deferred for want of a slot.

**Per-artefact signatures.** SindriCAD's shape: no binding between the bytes and the version claimed, three signatures per release, and three chances for a missing `.sig` to drop a platform silently.

**URLs inside the manifest.** A filename, always. This is what bounds a key leak, and it is free.

**Hand-rolled tar and zip readers.** Removed entirely by the raw-executable payload. Path traversal in an update path, on bytes that arrived over the network, is the worst possible place to be writing a parser from scratch.

**A companion-file list in the manifest.** The obvious answer to a release that needs a new shader, an icon or a changed `.desktop`, and it is a package manager wearing a `key = value` costume: an install location per file, an ownership rule for files a later release drops, a partial-apply state where the executable landed and the rest did not, and network-supplied path components back in the code path the raw payload was chosen to empty out. `requires_reinstall = 1` sends that user to the full download instead — one manual step, on the one release in years that needs it, in exchange for none of the above.

**Hand-rolled SHA-256.** `ring` is already compiled. This prohibition is written down because the tree has four precedents for hand-rolling and this is the one place it must not happen.

**`cmd /c start`.** Shell metacharacter injection, reachable from RSS feed content today.

**A hidden flag in the shipped binary that points the updater somewhere else.** See the selftest section.

**Signing in CI.** See key custody.

**Auto-install, and a startup modal that installs by default.** SindriCAD raises a blocking modal 8 seconds into every session, focuses the install button and does not remember a refusal. We notify on surfaces that already exist, default the focus to Later, and persist `skip_build`.

**Delta updates.** 19-33 MB payloads. No.

**Staged rollout, cohorts, update telemetry.** No server, no cohort mechanism, no analytics. The rollback that is worth having is client-side and local.

**Downloading to `/tmp`.** Cross-filesystem rename degrades to copy on both Linux and Windows, which is precisely the operation that fails against a running image.

---

## Decisions — settled 2026-08-30

The six open questions are answered. Three came out against this document's own
recommendation, because the evidence contradicted it. Each answer records what
was measured, so a future reader can tell a decision from a preference.

**1. `check_for_updates` defaults to `always`.**
Not for the reason this document originally gave. `articles.rs:30`'s premise —
"this application has deliberately no account and no stored credential" — is
**already false**: `app.rs:1614` calls `account::load()` and `app.rs:7337-7343`
fetches `account.avatar_url` over the network. The welcome tick therefore
already gates two connections, and the update check is a third with no tick at
all. `welcome` would not preserve the one-switch property; it would restore a
property the account feature broke, and pay for it by cutting update reach below
today's unconditional check. `always` matches today and adds a switch where
there is none.

Consequence that must ship with it: **the tick appears in Help/About as well as
on the welcome screen.** A switch reachable only from a screen the user turned
off is not a switch. And `articles.rs:30` is corrected as part of Phase 0 — it is
wrong today, independent of any of this.

**2. Phase 4 ships. The Windows risk is the other way round.**
Mark-of-the-Web is written by the downloader, not by the filesystem, so a payload
written by our own process carries none. SmartScreen therefore fires on every
manual re-download and **not** on a self-updated binary: self-update removes a
dialog rather than adding one.

Smart App Control cannot strand a user who was previously fine, with one narrow
exception. SAC blocks unsigned executables outright, so an enforcing machine
cannot run today's beta at all and never reaches the updater. The exception is
that SAC also permits unsigned files its cloud model predicts safe, per file
hash — so a predicted-safe build can update to one that is not. That dice roll
already happens on every manual download of every new build; self-update does not
add it, it changes only who is holding a working install when it comes up bad.
SAC is also becoming *more* common, not less: since KB5083769 (April 2026) it can
be enabled without a clean install, and there is no per-app bypass — only "turn
SAC off" or "sign it".

**Amended twice on 2026-08-30. Final answer: Phase 4 ships, unproven, by
decision.** The evidence above still holds — self-update genuinely removes a
SmartScreen dialog rather than adding one, and SAC cannot strand a user who was
previously fine. What could not be satisfied is this phase's human gate: there
is no Windows desktop on this project, so nobody can take the hop, and a green
`windows-latest` runner is not a substitute (different security context, no
Mark-of-the-Web written, silent about SmartScreen and Smart App Control).

An intermediate revision held Windows at hand-over for exactly that reason.
Thomas overrode it: all platforms that can self-replace, do. **That is a
deliberate acceptance of an unverified path, recorded here so it is not later
read as an oversight.** What is being accepted, precisely: a Windows build that
Smart App Control declines to execute leaves an application that will not start
and no code of ours running to fix it, because nothing of ours is allowed to
execute. `RECOVER-BROKKRSCULPT.txt` beside the binary is the entire remedy for
that case — which is why it is written *before* the swap and never deleted, and
why deleting it was the first defect this document had to correct.

What ships alongside it: `BROKKR_NO_SELF_UPDATE=1`, an opt-OUT that drops back
to download-and-hand-over without also silencing the check. `check_for_updates
= never` is the bigger hammer and a worse fit — a user in trouble wants the
swap to stop, not the news.

The first Windows user through this path is still the first human through it.
That is now a known cost rather than an unknown one.

The narrow case is unrecoverable in-app, and this document defeated its own fix:
**Windows step 7 deleted `RECOVER-BROKKRSCULPT.txt` on success**, which is
precisely the artefact that rescues a user whose new binary will not execute.
The note is now permanent. The human gate stays.

**3. Standby key: password manager entry plus a printed copy at a separate
address.** Both copies carry the encrypted `.key` *and* the passphrase. Splitting
the passphrase from the key across the two locations means losing either one
loses the key, which is the wrong failure mode for a thing that exists to survive
a bad day — for a backup the threat is loss, not theft. The printed sheet carries
a `sha256` line beside the key so a mistyped base64 character is caught on the
page rather than on release night.

**The restore drill is part of Phase 1, not a nice-to-have.** Generate, park,
then restore from *each* location and sign a throwaway manifest before v1 ships.
A printed key that has never been read back is not a backup, and this is the
highest-consequence untested item in the plan.

**4. Apple enrolment is open; Phase 5 is a money question, not a hardware one.**
Web enrolment for individuals still exists and needs no Apple hardware — an Apple
Account with 2FA, a legal name matching government ID, and your own payment card.
The route that requires an iPhone/iPad with biometrics or a T2/Apple Silicon Mac
is the *Developer app* route, which is also region-limited; that is the path this
document was worried about.

The sting, which was not in the original question: **signing and notarising
require macOS.** `codesign`, `notarytool` and `stapler` are Xcode tools. With no
Mac, Phase 5 means an Apple certificate on a `macos-latest` runner — which
violates the "signing in CI" prohibition under *Key custody*. The exception is
defensible and is written down there rather than left for someone to trip over:
an Apple certificate is scoped and revocable by Apple and cannot sign a manifest
that installs code on Linux and Windows. The minisign key can. They are not the
same risk and the prohibition should not read as though they are.

**5. Retain 20 build-stamped sets, not three.**
GitHub documents **no limit on total release size and no bandwidth limit**; the
binding constraint is 1000 assets per release, which at three payloads per build
is ~330 builds. Meanwhile the measured cadence is **four releases in under four
hours** (`release.yml` runs 1-4, all on 2026-08-29, 15:45→19:26). Three sets is
therefore less than one afternoon of rollback range — push three times after a
bad build and mechanism 2, "publish `seq+1` naming the last good build", has
nothing left to name. Retention is the whole reason rollback exists; three nearly
deletes it. Twenty sets is ~1.2 GB and about five days at that cadence.

Correct the arithmetic while you are here: the Linux raw ELF measures
**33,543,336 bytes** (`target/release/brokkrsculpt`, 2026-08-30), so three sets
was ~100 MB on Linux and ~178 MB across all three platforms. Twenty sets is the
~1.2 GB now stated under *Release pipeline*.

The one real cost of depth is that it widens a leaked key's downgrade surface —
an attacker who can sign may offer any retained build. `minimum_build` is the
bound on that, which means `minimum_build` has to be *used* and not merely
specified.

**6. `BASE = 1000`, recorded now. And `seq` gets a generator in the same change.**
`release.yml`'s `github.run_number` is **4** as of 2026-08-30, so the next release
is build 1005. An ordinal that cannot be mistaken for a run number or a version
is what makes a stale-ordinal bug visible by eye, which matters because the
warm-build-script failure mode is otherwise silent. Rename rule: **bump to the
next multiple of 1000 above the highest published ordinal.**

This question could not be answered cleanly without settling the number-space
confusion the completeness review flagged, so it is settled here:

- **`build`** = `BASE + run_number`, emitted by CI. Payload identity.
- **`seq`** = manifest counter, generated by `sign-release.sh`, which fetches the
  live `latest.conf` and asserts `new_seq > live_seq` or refuses to sign.
  Manifest identity.
- **The floor is compared against `seq` only, never `build`.** On a successful
  apply, `floor_seq = manifest.seq`.
- **A fresh install seeds `floor_seq = 0`.** Seeding it from the running binary's
  ordinal is exactly the space-mixing that breaks rollback. The "a new tester
  lands on a stale build" worry that justified that seeding is `minimum_build`'s
  job, and `minimum_build` compares in build space, where it belongs.

---
## Completeness review — resolved 2026-08-30

A reviewer was asked what a reader would still be blocked by. Its findings were
kept verbatim for a while so the gaps stayed visible; they have now been resolved
into the body above and the verbatim list is gone, because a list headed "Wrong
(would ship broken)" describing defects that no longer exist misleads a reader
worse than it ever informed one. What the review found, and where each answer
now lives:

| Finding | Resolved in |
| --- | --- |
| Redirect host was `objects.githubusercontent.com`; real host is `release-assets.` | *update.rs*, measured, plus the exact host rule in the v1 list |
| `timeout_global` starves any connection under 13.4 Mbit/s | *update.rs* — two agents, `timeout_recv_body` as the stall timer |
| `floor_seq` and `build` are different number spaces and the plan mixed them | *Decisions* item 6, and verification step 5's `MAX_SEQ` |
| Nothing generates `seq` | *Decisions* item 6 — `sign-release.sh`, refusing unless `new_seq > live_seq` |
| Key selected by parsing unverified bytes | Verification step 2a-2d — verify by trying the list, then parse |
| `expires` reproduces the TUF trap | Dropped entirely; *What we will deliberately not do* |
| `.old` deleted on the next launch | *apply.rs* — kept until the next apply replaces it |
| No mechanism for CI to read the ordinal back | *Build ordinal* — a `--version` arm and `grep -qx` |
| Manifest cannot express a companion file | `requires_reinstall`, required in v1; the file list is rejected |
| No restore drill for the standby key | *Signing and key custody*, and Phase 1's deliverable |
| Free-space gate has no mechanism | Deleted; a named `ENOSPC` does the same job |
| `windows-sys` is transitive only | False — it is already a direct target dependency; one feature string is the real cost |
| The restart drops the open document | *The prompt* — `guard()` plus a `resume =` line in the marker |
| Concurrency handled for staging only | *apply.rs* — one `update.lock`, and a per-install marker filename |
| `minimum_build` off-by-one | *Stopping a bad build* — strictly below, stated once |
| Estimates incomplete in scope | *Phased delivery*, rebuilt bottom-up: 26-38 days |

Two of the review's own claims did not survive checking, and both are worth
recording because they were marked confident. `windows-sys` is **not** transitive
only — `brokkr-app` declares it directly for the Raw Input stylus, so the cost is
one feature string rather than a new dependency and a version choice. And the
remedy attached to the timeout finding — "needs a stall-based timeout rather than
a wall-clock one" — implies ureq has no stall timeout; it has one, and
`timeout_recv_body` is it. The diagnosis was right and the prescription was not.
A finding is a hypothesis until someone runs it, which is the same rule this
document applies to itself everywhere else.

What is still genuinely open is not in this section. It is in *What cannot be
verified before it reaches users*, which is the honest list, and it is shorter
than it was: the redirect chain has come off it, measured.
