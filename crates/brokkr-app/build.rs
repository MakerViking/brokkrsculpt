// SPDX-License-Identifier: AGPL-3.0-only

//! Stamp the build with the commit it was built from.
//!
//! `build_commit()` reads `BROKKR_COMMIT`, which nothing set, so every build
//! reported itself as `unknown`. That was cosmetic while diagnostics were a
//! string on the clipboard. It stopped being cosmetic when reports started
//! being uploaded: TinkerAtlas groups repeat reports by a fingerprint taken
//! over the version and the first line, so with a frozen version every report
//! of "it went black" from every build for all time collapses into one row —
//! including the ones filed after the bug was supposedly fixed.
//!
//! No dependency: `git` is asked directly, and if it is not there (a source
//! tarball, a sandboxed builder) the variable is simply not set and
//! `build_commit()` falls back exactly as it did before. A build must not fail
//! because it happened outside a checkout.

use std::process::Command;

fn main() {
    // Only rerun when the checked-out commit actually moves. Without this the
    // whole crate rebuilds on every `cargo build`, because a build script with
    // no declared inputs is assumed to depend on everything.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-env-changed=BROKKR_COMMIT");

    // **The build ordinal, and it MUST be emitted before the early return
    // below.** A git SHA has no order, so `BROKKR_COMMIT` can answer "is the
    // published build different from mine" and never "is it newer". The release
    // workflow sets this to `1000 + github.run_number`; nothing sets it locally,
    // and `None` is what makes the updater structurally inert on a developer's
    // machine rather than one keystroke away from overwriting their own binary.
    //
    // The ordering is not stylistic. CI sets `BROKKR_COMMIT`, so an emission
    // placed after that guard would never run in exactly the case that needs
    // it, and `Swatinem/rust-cache` would then serve a warm build-script output
    // and publish a binary reporting the PREVIOUS run's ordinal -- silently,
    // and self-perpetuating from then on. `release.yml` reads the ordinal back
    // out of the built binary for the same reason: this comment is a request,
    // and that check is the enforcement.
    println!("cargo:rerun-if-env-changed=BROKKR_BUILD");
    if let Some(build) = std::env::var_os("BROKKR_BUILD") {
        // Passed through as text and parsed at the far end. A build script
        // cannot fail usefully here -- `panic!` would break `cargo build` for a
        // typo in an environment variable -- and `build_number` refusing to
        // parse it has the same effect as never setting it: an inert updater.
        println!("cargo:rustc-env=BROKKR_BUILD={}", build.to_string_lossy());
    }

    // An explicit value wins, so a release pipeline that knows better than git
    // -- a tag, a build number -- can say so.
    if std::env::var_os("BROKKR_COMMIT").is_some() {
        return;
    }

    let Ok(output) = Command::new("git").args(["rev-parse", "--short", "HEAD"]).output() else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let Ok(commit) = String::from_utf8(output.stdout) else {
        return;
    };
    let commit = commit.trim();
    if commit.is_empty() {
        return;
    }

    // A tree with uncommitted changes is not the commit it names, and a report
    // that says it is sends someone reading it to code that never ran.
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .is_ok_and(|out| out.status.success() && !out.stdout.is_empty());

    println!("cargo:rustc-env=BROKKR_COMMIT={commit}{}", if dirty { "-dirty" } else { "" });
}
