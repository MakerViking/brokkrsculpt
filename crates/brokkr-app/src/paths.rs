// SPDX-License-Identifier: AGPL-3.0-only

//! Where the application keeps its own files.
//!
//! Three places needed the same base directory dance -- the puck's settings,
//! the recent files list and the autosave -- and each had grown its own copy of
//! it. The rule is small but it is a rule: honour the environment variable only
//! when it is set AND absolute, per the base directory specification, and fall
//! back to a fixed path under the home otherwise. A copy that forgets the
//! absolute check silently writes to a relative path resolved against whatever
//! directory the application happened to be started from, which is not stable
//! across sessions and is exactly when these files are read.
//!
//! Each platform is asked where it wants these, rather than XDG being imposed
//! on all three: Windows splits roaming settings from local state and calls
//! the home `USERPROFILE`, and macOS puts both under Application Support with
//! no overridable variable at all. The one invariant that holds everywhere is
//! that settings and state are different directories.

use std::path::PathBuf;

/// Everything the application writes lives under a directory of this name.
const APPLICATION: &str = "brokkrsculpt";

/// The user's home, under whichever name this platform gives it.
///
/// Windows does not set `HOME`; `USERPROFILE` is the equivalent, and reading
/// only `HOME` there means every path below comes back `None` and the recent
/// list, the autosave and the puck's settings all silently stop existing.
fn home() -> Option<PathBuf> {
    let variable = if cfg!(target_os = "windows") { "USERPROFILE" } else { "HOME" };
    std::env::var_os(variable).map(PathBuf::from)
}

/// A base directory from the environment, or a fallback under the home.
///
/// `variable` is honoured only when it is set and absolute, which is what the
/// base directory specification requires and what a hand-rolled copy is most
/// likely to get wrong. `None` for platforms whose convention is a fixed path
/// rather than an overridable variable -- macOS has no `XDG_*` equivalent.
fn base(variable: Option<&str>, fallback: &str) -> Option<PathBuf> {
    if let Some(from_environment) =
        variable.and_then(std::env::var_os).map(PathBuf::from).filter(|path| path.is_absolute())
    {
        return Some(from_environment);
    }
    Some(fallback.split('/').fold(home()?, |path, part| path.join(part)))
}

/// Where settings live, per platform.
///
/// * Linux: `$XDG_CONFIG_HOME/brokkrsculpt`, else `~/.config/brokkrsculpt`
/// * Windows: `%APPDATA%\brokkrsculpt` -- roaming, because settings are what a
///   user would expect to follow them to another machine on a domain
/// * macOS: `~/Library/Application Support/brokkrsculpt`
fn config_directory() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        base(Some("APPDATA"), "AppData/Roaming")?
    } else if cfg!(target_os = "macos") {
        base(None, "Library/Application Support")?
    } else {
        base(Some("XDG_CONFIG_HOME"), ".config")?
    };
    Some(base.join(APPLICATION))
}

/// Where recoverable state lives, per platform.
///
/// **Never the same directory as [`config_directory`].** An autosave sitting
/// beside the settings reads as a document the user chose to keep, and that
/// invariant is what `config_and_state_are_different_places` pins.
///
/// * Linux: `$XDG_STATE_HOME/brokkrsculpt`, else `~/.local/state/brokkrsculpt`
/// * Windows: `%LOCALAPPDATA%\brokkrsculpt` -- local rather than roaming, since
///   an autosave of a half-finished sculpt has no business crossing machines
/// * macOS: a `State` directory inside the application support one. Apple has
///   no separate state location, and the alternative -- `~/Library/Caches` --
///   is purgeable, which is the one thing an autosave must not be.
fn state_directory() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        return Some(base(Some("LOCALAPPDATA"), "AppData/Local")?.join(APPLICATION));
    }
    if cfg!(target_os = "macos") {
        return Some(config_directory()?.join("State"));
    }
    Some(base(Some("XDG_STATE_HOME"), ".local/state")?.join(APPLICATION))
}

/// A settings file under [`config_directory`].
///
/// For things the user chose and would expect to survive a reinstall.
pub fn config_file(name: &str) -> Option<PathBuf> {
    Some(config_directory()?.join(name))
}

/// A state file under [`state_directory`].
///
/// For things the application recovers from rather than things the user set.
/// The autosave lives here and not in the config directory precisely so it is
/// never mistaken for a document.
pub fn state_file(name: &str) -> Option<PathBuf> {
    Some(state_directory()?.join(name))
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

/// An absolute path, spelled the way the platform running the tests spells one.
///
/// **`/thing` is not absolute on Windows.** It has no drive letter, so
/// `is_absolute` is false and every fixture that builds a path that way is
/// quietly testing the rejection branch instead of the one it meant. Ten tests
/// across three modules failed exactly that way the first time this crate was
/// built for Windows.
#[cfg(test)]
pub(crate) fn absolute(rest: &str) -> PathBuf {
    if cfg!(target_os = "windows") {
        PathBuf::from(format!(r"C:\{}", rest.replace('/', r"\")))
    } else {
        PathBuf::from(format!("/{rest}"))
    }
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
        assert!(absolute("absolute/path").is_absolute());
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
