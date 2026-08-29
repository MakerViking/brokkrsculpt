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

**There are no releases.** BrokkrSculpt has no packaged build, no installer and
no updater yet; the only way to run it is to build `main` from source. So
`main` is what is supported — fixes land there, and there is nothing older to
backport to. Everything in the tree still says version `0.0.1`.

This will change once the project decides what it ships as, and this section
will change with it.

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
- **the update check to `api.github.com` on startup.** One unauthenticated
  GET asking which commit the published beta was built from, so the
  application can say whether yours differs. It downloads nothing and executes
  nothing. It is skipped entirely for a build made from a dirty tree or
  outside a checkout, so a development machine never makes the call. Worth
  stating plainly because it is a new third party: GitHub learns that an
  address is running BrokkrSculpt, and the request names the application in
  its user agent rather than disguising itself.
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
