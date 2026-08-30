<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Security Policy

## Reporting a vulnerability

Please use GitHub's private reporting: the **Report a vulnerability** button
under the repository's
[Security tab](https://github.com/MakerViking/brokkrsculpt/security/advisories/new).
Do not open a public issue for anything security-sensitive.

BrokkrSculpt is a solo project. I read every report, and I aim to acknowledge
within a few days. If a report is valid, the fix lands on `main` and the
advisory is credited to you (unless you prefer otherwise).

## Supported versions

There is a **public open beta** on Linux, Windows and macOS: the rolling
[`beta` release](https://github.com/MakerViking/brokkrsculpt/releases), rebuilt
and republished on every push to `main`. So there is exactly one supported
version and it is whatever `beta` currently points at. Fixes land on `main` and
reach that release on the next push; there is nothing older to backport to.
Everything in the tree still says version `0.0.1`.

Nothing in that release is signed by Apple or Microsoft. That is a cost
decision rather than an oversight, and the release notes say what each
operating system will show you and what to do about it. Building `main` from
source remains supported and is the same code.

## Scope

In scope:

- the desktop application: `brokkr-app` (iced shell, input), `brokkr-gpu`
  (wgpu renderer), `brokkr-core` (engine, file formats)
- **parsing untrusted files** — `.brokkr` project containers and imported
  STL, OBJ and 3MF. This is the largest attack surface here: a 3MF is a ZIP of
  XML, and a project file is raw binary that describes its own allocations.
  The readers are mutation-fuzzed and the growth paths are bounded, but a file
  that panics, hangs, exhausts memory or reads out of bounds is a valid report.
- the outbound bug report to `tinkeratlas.com` over HTTPS. Anonymous unless
  you have signed in, and the dialog names which before you send. The payload
  is shown in full first and passes through path redaction.
- **the TinkerAtlas sign-in token, when you have one.** Optional, and it exists
  so a bug report can be replied to. Obtained through the system browser and a
  loopback callback guarded by a nonce, never by the application handling your
  password; stored in the per-user application data directory, mode `0600` on
  Linux and macOS. Windows has no mode: there the profile's own ACLs are what
  exclude other users, which is the same set `0600` excludes and admits an
  administrator exactly as `0600` admits root. Sign out deletes the file.
- **the update check on startup.** Two unauthenticated GETs to ordinary
  release-asset URLs on `github.com` — a signed manifest and its signature —
  which together say which build is currently published.

  **If one names a newer build and you accept it, the payload is downloaded,
  its SHA-256 checked against the signed manifest, and the application replaced
  with it** — the executable on Linux and Windows, the whole `.app` on macOS.
  Nothing is made executable before the digest matches. `BROKKR_NO_SELF_UPDATE=1`
  downloads and verifies without installing.

  It has **its own switch**, on the welcome screen and in the Help menu, and it
  is on by default. Two surfaces on purpose: the default is to check on every
  launch, so a control reachable only from a screen you can turn off would not
  be a control. Setting `BROKKR_UPDATE_EXPLANATION` — the packager kill switch
  — suppresses the check as well as the interface.

  It is skipped entirely for a build made from a dirty tree, made outside a
  checkout, or carrying no build ordinal, so a development machine never makes
  the call.

  This replaced a single GET to `api.github.com` that asked which commit the
  published beta was built from. The API is rate limited to 60 requests an hour
  per address, which one makerspace behind one NAT is enough to exhaust for
  everyone in it, and failure was deliberately silent — so the symptom was an
  update check that simply stopped working with nothing said. Asset URLs have
  no such limit. See **Updates, signing and trust** below. Worth stating
  plainly either way, because it is a third
  party: GitHub learns that an address is running BrokkrSculpt, and the
  request names the application in its user agent rather than disguising
  itself.
- the read-only Moonraker query to a printer you configure on your own LAN.
  This leg is plain HTTP by necessity and is deliberately kept separate from
  the HTTPS one; the host in `printer.conf` is validated when it is read, not
  only when it is written.
- reading input devices directly, below the window system, on all three
  platforms: `evdev` (`/dev/input`) on Linux, which requires membership of the
  `input` group; Raw Input on Windows; IOKit on macOS, which requires Input
  Monitoring. A pen and a six-axis puck are the only devices matched, and only
  their own usages are read
- files the application writes without being asked: the autosave and recent
  list, under `$XDG_STATE_HOME` / `$XDG_CONFIG_HOME` on Linux,
  `%LOCALAPPDATA%` / `%APPDATA%` on Windows, and Application Support on macOS

Out of scope:

- vulnerabilities that require an already-compromised machine or an attacker
  who can already write to the user's config directory
- reports from automated scanners with no plausible impact
- the `input` group requirement itself. Reading a tablet or a SpaceMouse means
  reading `/dev/input`; that is a documented prerequisite, not a defect.
- geometry that is *wrong* rather than unsafe. A model that imports with holes
  in it is a bug — please file it as one — but it is not a security issue.

Issues in the tinkeratlas.com website itself can go through the same private
channel; they reach the same person.

## Updates, signing and trust

**This describes what the shipped beta does, as of build 1020.** It checks for
updates on launch, downloads them, verifies them, and — on Linux and Windows —
replaces its own executable; on macOS it replaces the whole `.app` bundle. The
check has a tick on the welcome screen and in the Help menu and is on by
default. `BROKKR_NO_SELF_UPDATE=1` downloads and verifies without installing,
and `check_for_updates = never` in `update.conf` stops it entirely.

One thing to weigh, stated here rather than buried: **the swap itself has only
been performed for real on Linux.** Windows and macOS share the same signed
manifest, the same download and the same digest check; what has not been
exercised on real hardware is the replacement step. Both keep the build they
replaced and write a `RECOVER-BROKKRSCULPT.txt` beside the application saying
how to restore it. The full reasoning, with what was measured and what was
assumed, is in `docs/AUTOUPDATE-PLAN.md`.

### The trust model

There is no operating-system code signing anywhere in this: no Apple Developer
ID, no Authenticode certificate. macOS and Windows will not vouch for these
builds, and Linux has no such gate at all. The only root of trust is a minisign
public key **compiled into the binary**, which means the first install is
trust-on-first-use however careful you are — if the copy you first downloaded
was not ours, nothing that happens afterwards can tell you so.

What gets signed is the **manifest**, not the individual downloads: one
signature over one small file carrying the build ordinal, the byte length and
the SHA-256 of each platform's payload. That binds the bytes to the release
they claim to belong to. Signing the bytes alone would not — anyone controlling
the endpoint could then serve a genuinely signed *old* build under a new
version number.

Each trusted key carries an **epoch** beside it, and a client refuses any epoch
below the highest it has ever accepted. That is what lets a leaked key actually
be revoked, rather than merely joined on the list by its replacement. There are
two keys from the first signed release onwards — a live one and a standby whose
secret half is kept off the machine that builds and publishes — because a
rotation path cannot be added to copies that are already installed. Signing a
manifest with the standby is the recovery path, and every client that takes it
refuses the old epoch from then on.

### What a leaked signing key can and cannot do

Someone holding the private key can **sign** a manifest. They cannot **serve**
it. The download location is a GitHub release they cannot write to, and the URL
prefix is compiled into the binary — the manifest supplies a *filename*, never
a URL, and a name containing `/`, `\`, `..` or anything outside
`[A-Za-z0-9._-]` is rejected. Two independent things have to fail.

Even if both do, the worst available outcome is a **downgrade to one of our own
previously published builds**, because every digest in a manifest they sign
still has to match bytes we published. It is not arbitrary code execution. That
bound is the reason the design is shaped this way, and it costs nothing.

The signing key is not in CI and will not be. The signing script downloads each
payload and hashes it locally rather than reading digests back from the GitHub
API, because keeping the key off CI protects the *key* and not the *artefact*:
if the digests came from the API, a compromised runner could upload a
backdoored payload that then got faithfully signed.

### Stopping a bad build

Four mechanisms, in the order they should be reached for:

1. **Delete `latest.conf` from the release.** No key, no command line, no
   build — a browser and thirty seconds. Clients get a 404 and fall silent.
   This is the emergency stop, it is the only one a co-maintainer can perform,
   and it is the only one available when I am unreachable.
2. **Publish the next manifest naming the last good build.** Everyone walks
   backwards. This works only because payloads are stamped with their build
   ordinal and twenty sets of them are retained.
3. **`minimum_build`.** Warns every client below a named build, naming both
   numbers. It never blocks launching: refusing to start a sculpting
   application that someone already has installed is worse than the bug.
4. **The human signing gate.** Nothing reaches anyone until a manifest is
   signed by hand, so nothing spreads on autopilot.

### If it refuses to update

An update can be refused permanently and entirely correctly — that is what the
anti-rollback floor is for — and when it happens the status line says so, with
the number. The documented reset is to **delete `update.state`** from the
application's state directory (`$XDG_STATE_HOME` on Linux, `%LOCALAPPDATA%` on
Windows, Application Support on macOS). That clears the floor this install
recorded for itself. It disables no signature check.
