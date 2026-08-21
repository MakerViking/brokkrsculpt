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
- the outbound bug report to `tinkeratlas.com` over HTTPS (anonymous; no
  account, no stored credential; the payload is shown before it is sent and
  passes through path redaction first)
- the read-only Moonraker query to a printer you configure on your own LAN.
  This leg is plain HTTP by necessity and is deliberately kept separate from
  the HTTPS one; the host in `printer.conf` is validated when it is read, not
  only when it is written.
- reading input devices through `evdev` (`/dev/input`), which requires
  membership of the `input` group
- files the application writes without being asked: the autosave and recent
  list under `$XDG_STATE_HOME` / `$XDG_CONFIG_HOME`

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
