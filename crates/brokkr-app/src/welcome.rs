// SPDX-License-Identifier: AGPL-3.0-only

//! The welcome screen: what to do first, and how to get back to what you did
//! last.
//!
//! Structure ported from SindriCAD's `src/ui/welcome.ts` -- a modal over a
//! scrim, the lockup at the head, local actions and the recent list down the
//! left, a second column beside them, and "show this on startup" along the
//! foot. **The content is ported and the code is not**, which is the standing
//! rule between these two applications: that one is a Tauri webview and this is
//! a Rust binary drawing through iced, and pretending otherwise is how a port
//! turns into a rewrite of the wrong half.
//!
//! # Two things deliberately did not come across
//!
//! **The remote pane.** SindriCAD's right column is an `<iframe>` of
//! `tinkeratlas.com/sindricad/welcome`, probed for reachability from Rust first
//! because a cross-origin frame never reports its own load failures. There is
//! no webview here to put a frame in, and the dependency policy would refuse
//! one long before the design question came up -- `iced`'s `image` feature
//! alone is seventy-one crates against this workspace's lockfile. So the second
//! column carries what a first-time user actually needs and cannot get
//! anywhere else yet: there is no manual, and a beta is the moment strangers
//! meet the keys for the first time.
//!
//! **The account row.** SindriCAD offers "Sign in with TinkerAtlas" and shows
//! an avatar. This application has no sign-in at all, and that is a decision
//! rather than an omission: the bug reporter is anonymous precisely so the
//! workspace never holds a credential. A welcome screen is not the place to
//! introduce one.
//!
//! # The preference
//!
//! A flat `key = value` file in the config directory, which is the shape
//! `printer.conf` and `spacemouse.conf` already use. **Absent means show it**,
//! matching SindriCAD's `localStorage.getItem(...) !== "false"`: a first run
//! has no file, and the screen a first run most needs is this one.

use std::path::PathBuf;

/// The file the preference lives in.
const FILE: &str = "welcome.conf";
const KEY: &str = "show_on_startup";

/// Where the preference is kept, so the tests and the writer agree.
fn file() -> Option<PathBuf> {
    crate::paths::config_file(FILE)
}

/// Whether to open the welcome screen when the application starts.
///
/// Anything that is not an explicit `false` means yes, including a missing
/// file, an unreadable one and a corrupt one. The failure this guards is
/// silently *hiding* the screen from someone who never asked to hide it --
/// which on a first run would mean meeting an empty sculpt with no idea what
/// the keys are.
pub fn on_startup() -> bool {
    read_from(file().as_deref())
}

fn read_from(path: Option<&std::path::Path>) -> bool {
    let Some(path) = path else {
        return true;
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return true;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((name, value)) = line.split_once('=')
            && name.trim() == KEY
        {
            return value.trim() != "false";
        }
    }
    true
}

/// Remember the answer. A write that fails is dropped: the preference is worth
/// less than the session, and there is nothing useful to say about a read-only
/// config directory in the middle of closing a dialog.
pub fn set_on_startup(show: bool) {
    if let Some(path) = file() {
        write_to(&path, show);
    }
}

fn write_to(path: &std::path::Path, show: bool) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(
        path,
        format!(
            "# BrokkrSculpt welcome screen.\n\
             # Delete this file to see it again on startup.\n\
             {KEY} = {show}\n"
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("brokkr-welcome-{name}-{}", std::process::id()))
    }

    /// **A missing file means show it**, which is the case a first run is in.
    ///
    /// The other direction -- defaulting to hidden -- fails silently and in the
    /// worst place: the one person who most needs the screen is the one who has
    /// never seen it, and they would never know it existed.
    #[test]
    fn a_missing_preference_shows_the_screen() {
        let path = scratch("missing");
        let _ = std::fs::remove_file(&path);
        assert!(read_from(Some(&path)));
        assert!(read_from(None), "no config directory at all must still show it");
    }

    /// Only an explicit `false` turns it off, and it survives a round trip.
    #[test]
    fn the_preference_survives_being_written_and_read_back() {
        let path = scratch("roundtrip");
        write_to(&path, false);
        assert!(!read_from(Some(&path)), "false did not stick");
        write_to(&path, true);
        assert!(read_from(Some(&path)), "true did not stick");
        let _ = std::fs::remove_file(&path);
    }

    /// A corrupt file shows the screen rather than hiding it.
    ///
    /// Same argument as the missing one: of the two ways to be wrong, showing
    /// a screen someone did not want is a click, and hiding one they did want
    /// is invisible.
    #[test]
    fn a_corrupt_preference_shows_the_screen() {
        let path = scratch("corrupt");
        std::fs::write(&path, "\u{0}not a config at all\n[section]\n").expect("temp is writable");
        assert!(read_from(Some(&path)));

        // And a key that is not ours does not answer for it.
        std::fs::write(&path, "something_else = false\n").expect("temp is writable");
        assert!(read_from(Some(&path)));
        let _ = std::fs::remove_file(&path);
    }
}
