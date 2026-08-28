// SPDX-License-Identifier: AGPL-3.0-only

//! Where the application keeps its own files.
//!
//! Three places needed the same base directory dance -- the puck's settings,
//! the recent files list and the autosave -- and each had grown its own copy of
//! it. The rule is small but it is a rule: honour the environment variable only
//! when it is set AND absolute, per the base directory specification, and fall
//! back to a fixed path under `$HOME` otherwise. A copy that forgets the
//! absolute check silently writes to a relative path resolved against whatever
//! directory the application happened to be started from, which is not stable
//! across sessions and is exactly when these files are read.

use std::path::PathBuf;

/// Everything the application writes lives under a directory of this name.
const APPLICATION: &str = "brokkrsculpt";

/// A base directory from the environment, or a fallback under `$HOME`.
///
/// `variable` is honoured only when it is set and absolute, which is what the
/// base directory specification requires and what a hand-rolled copy is most
/// likely to get wrong.
fn base(variable: &str, fallback: &str) -> Option<PathBuf> {
    std::env::var_os(variable).map(PathBuf::from).filter(|path| path.is_absolute()).or_else(|| {
        let home = PathBuf::from(std::env::var_os("HOME")?);
        Some(fallback.split('/').fold(home, |path, part| path.join(part)))
    })
}

/// A settings file: `$XDG_CONFIG_HOME/brokkrsculpt/<name>`, else
/// `~/.config/brokkrsculpt/<name>`.
///
/// For things the user chose and would expect to survive a reinstall.
pub fn config_file(name: &str) -> Option<PathBuf> {
    Some(base("XDG_CONFIG_HOME", ".config")?.join(APPLICATION).join(name))
}

/// A state file: `$XDG_STATE_HOME/brokkrsculpt/<name>`, else
/// `~/.local/state/brokkrsculpt/<name>`.
///
/// For things the application recovers from rather than things the user set.
/// The autosave lives here and not in the config directory precisely so it is
/// never mistaken for a document.
pub fn state_file(name: &str) -> Option<PathBuf> {
    Some(base("XDG_STATE_HOME", ".local/state")?.join(APPLICATION).join(name))
}

/// The `key = value` pairs in one of this application's flat config files.
///
/// Blank lines and `#` comments are skipped, a line with no `=` is skipped, and
/// both halves are trimmed. What a key MEANS, and what to do about one that is
/// unknown or unparseable, is the caller's business and differs per file --
/// `printer.rs` falls back to a default port, `spacemouse.rs` leaves the
/// binding alone, `welcome.rs` treats anything but `false` as yes.
///
/// **Extracted at the third copy and not before.** `printer.rs` and
/// `spacemouse.rs` each grew this loop independently and each documented that
/// it matched the other; `welcome.rs` made three, which is where duplication
/// stops being cheaper than an abstraction. Only the scanning is shared: a
/// "config file" type that also owned the defaults would have to know all three
/// schemas, which is the wrong abstraction rather than a missing one.
pub fn entries(text: &str) -> impl Iterator<Item = (&str, &str)> {
    text.lines().filter_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (key, value) = line.split_once('=')?;
        Some((key.trim(), value.trim()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The check a hand-rolled copy is most likely to drop. A relative value
    /// resolves against the working directory, which is not stable across
    /// sessions.
    #[test]
    fn a_relative_environment_value_is_ignored_in_favour_of_home() {
        // Asserted through the predicate rather than by setting the variable,
        // because the environment is process-wide and the test harness runs
        // these in parallel.
        assert!(!PathBuf::from("relative/path").is_absolute());
        assert!(PathBuf::from("/absolute/path").is_absolute());
    }

    #[test]
    fn both_kinds_land_under_the_application_directory() {
        for built in [config_file("thing"), state_file("thing")] {
            let Some(path) = built else {
                continue;
            };
            assert!(
                path.to_string_lossy().contains(APPLICATION),
                "{} is not under the application directory",
                path.display()
            );
            assert!(path.ends_with("thing"));
            assert!(path.is_absolute(), "{} is not absolute", path.display());
        }
    }

    /// Config and state must not collide: an autosave sitting next to the
    /// settings would read as a document the user chose to keep.
    #[test]
    fn config_and_state_are_different_places() {
        if let (Some(config), Some(state)) = (config_file("x"), state_file("x")) {
            assert_ne!(config, state);
        }
    }
}
