// SPDX-License-Identifier: AGPL-3.0-or-later

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
