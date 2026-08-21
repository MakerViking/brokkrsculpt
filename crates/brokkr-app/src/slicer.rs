// SPDX-License-Identifier: AGPL-3.0-only

//! Handing a sculpt to OrcaSlicer.
//!
//! BrokkrSculpt does not slice, and should not: slicing is a large, well solved
//! problem with several good implementations that a user of a 3D printer
//! already has installed and already has their machine profile in. What this
//! does is export the model and open it in that slicer, which is the step
//! between "the sculpt is finished" and "the printer is printing".
//!
//! Ported from SindriCAD's `src-tauri/src/slicer.rs`, and the candidate table
//! is the valuable part. Every path in it was earned by a real report from a
//! machine nobody here owns, which is why it comes over verbatim rather than
//! being rewritten from a guess about where a program installs.
//!
//! # Why nothing here talks to the printer
//!
//! SindriCAD can also upload a sliced `.gcode` to a Snapmaker U1 over
//! Moonraker. That flow starts from a `.gcode`, and this application cannot
//! produce one -- so reproducing it here would mean: export, open Orca, slice,
//! save the gcode, come back, pick it, upload. OrcaSlicer's own "Upload and
//! print" button does the last three steps against the same machine, from the
//! window the user is already looking at. See `printer.rs`, which does the part
//! that is genuinely worth having from inside this application: telling you
//! whether the print is finished.

use std::path::{Path, PathBuf};

/// OrcaSlicer's flatpak app ids, current first.
///
/// `io.github.softfever.OrcaSlicer` is the pre-2.3.2 id. It is gone from
/// Flathub but is still installed on machines that never rebased, and listing
/// only that one is why a flatpak Orca installed today was found in no scope at
/// all. Ported along with the bug.
const ORCA_FLATPAK_IDS: [&str; 2] = ["com.orcaslicer.OrcaSlicer", "io.github.softfever.OrcaSlicer"];

/// `$XDG_DATA_HOME`, falling back to `~/.local/share`.
///
/// Where flatpak puts a `--user` installation, which is Fedora's GNOME Software
/// default -- so looking only at the system-wide export finds nothing on a very
/// ordinary setup. A relative value is refused rather than joined onto, because
/// joining a relative path onto a home directory silently produces a path that
/// is not where anything is.
fn xdg_data_home(home: Option<&PathBuf>, env: &dyn Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    env("XDG_DATA_HOME")
        .filter(|path| path.is_absolute())
        .or_else(|| home.map(|home| home.join(".local/share")))
}

/// Where OrcaSlicer usually installs, most likely first.
///
/// Pure, and takes the platform and the environment as parameters, so the whole
/// table can be tested on a machine that is none of these platforms. That
/// property is why it was worth porting rather than rewriting.
fn candidates_for(
    os: &str,
    home: Option<&PathBuf>,
    env: &dyn Fn(&str) -> Option<PathBuf>,
) -> Vec<PathBuf> {
    let under_home = |rel: &str| home.map(|home| home.join(rel));
    let under_env = |key: &str, rel: &str| env(key).map(|base| base.join(rel));

    let candidates = match os {
        "windows" => vec![
            under_env("ProgramFiles", "OrcaSlicer/orca-slicer.exe"),
            under_env("LOCALAPPDATA", "Programs/OrcaSlicer/orca-slicer.exe"),
            under_env("ProgramFiles(x86)", "OrcaSlicer/orca-slicer.exe"),
        ],
        "macos" => vec![
            Some(PathBuf::from("/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer")),
            under_home("Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"),
        ],
        _ => {
            let mut linux = vec![
                under_home("Applications/OrcaSlicer.AppImage"),
                Some(PathBuf::from("/usr/bin/orca-slicer")),
                Some(PathBuf::from("/usr/local/bin/orca-slicer")),
            ];
            // Flatpak exports after the native installs. Per-user scope first,
            // then system wide, current app id before the legacy one in both.
            // The export is a `#!/bin/sh` wrapper that execs `flatpak run … "$@"`,
            // so spawning it with the file path is enough -- no `flatpak run`
            // special casing, and no scope for this to guess wrong.
            let user_bin = xdg_data_home(home, env).map(|dir| dir.join("flatpak/exports/bin"));
            for id in ORCA_FLATPAK_IDS {
                linux.push(user_bin.as_ref().map(|dir| dir.join(id)));
            }
            for id in ORCA_FLATPAK_IDS {
                linux.push(Some(PathBuf::from("/var/lib/flatpak/exports/bin").join(id)));
            }
            linux
        }
    };
    candidates.into_iter().flatten().collect()
}

/// The first OrcaSlicer that is actually installed.
///
/// A versioned AppImage is looked for by glob rather than by name, because the
/// name carries the version and hardcoding one -- as SindriCAD does, with
/// `OrcaSlicer_V2.4.0-alpha.AppImage` -- stops finding it the day the user
/// updates.
pub(crate) fn find() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let env = |key: &str| std::env::var_os(key).map(PathBuf::from);
    let mut candidates = candidates_for(std::env::consts::OS, home.as_ref(), &env);

    // Any AppImage in ~/Applications whose name starts with OrcaSlicer, newest
    // name last so the highest version wins a lexical sort.
    if let Some(home) = &home
        && let Ok(entries) = std::fs::read_dir(home.join("Applications"))
    {
        let mut images: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name().and_then(|name| name.to_str()).is_some_and(|name| {
                    name.starts_with("OrcaSlicer") && name.ends_with(".AppImage")
                })
            })
            .collect();
        images.sort();
        // Ahead of the fixed table: an explicitly installed AppImage is a
        // stronger signal than a package that may be a leftover.
        for image in images.into_iter().rev() {
            candidates.insert(0, image);
        }
    }

    candidates.into_iter().find(|path| path.is_file())
}

/// Open a file in OrcaSlicer.
///
/// Detached and unwaited: the slicer outlives this application quite properly,
/// and a user who closes it has not failed at anything. The exit code is
/// deliberately not checked, because there is nothing useful to do with it.
///
/// # What this is careful about
///
/// This is the first `std::process::Command` in the workspace, so the rules it
/// establishes matter. The binary comes from [`find`] and never from a document,
/// a status string or anything a file could influence. The argument is a single
/// path this application has just written, passed as one argument and never
/// through a shell. And the extension is checked against what was exported,
/// which is the same guard SindriCAD ships.
pub(crate) fn open(slicer: &Path, model: &Path) -> Result<(), String> {
    let extension = model
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if !matches!(extension.as_str(), "3mf" | "stl" | "obj") {
        return Err(format!("{extension} is not a model this can hand over"));
    }

    std::process::Command::new(slicer)
        .arg(model)
        .spawn()
        .map(|_| ())
        .map_err(|why| format!("{} would not start ({why})", slicer.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<PathBuf> {
        None
    }

    fn home() -> PathBuf {
        PathBuf::from("/home/someone")
    }

    #[test]
    fn the_linux_table_looks_in_every_place_orca_is_actually_installed() {
        let found = candidates_for("linux", Some(&home()), &no_env);
        let shown: Vec<String> = found.iter().map(|path| path.display().to_string()).collect();

        assert!(shown.iter().any(|p| p == "/usr/bin/orca-slicer"));
        assert!(shown.iter().any(|p| p == "/usr/local/bin/orca-slicer"));
        assert!(shown.iter().any(|p| p.ends_with("Applications/OrcaSlicer.AppImage")));

        // Both flatpak ids, in both scopes. The current id first in each: the
        // legacy one is still installed on machines that never rebased, and
        // listing ONLY it is the bug this table was fixed for.
        for id in ORCA_FLATPAK_IDS {
            assert!(
                shown
                    .iter()
                    .any(|p| p == &format!("/home/someone/.local/share/flatpak/exports/bin/{id}")),
                "the per-user flatpak export for {id} is not looked for"
            );
            assert!(
                shown.iter().any(|p| p == &format!("/var/lib/flatpak/exports/bin/{id}")),
                "the system flatpak export for {id} is not looked for"
            );
        }

        let user =
            shown.iter().position(|p| p.contains(".local/share/flatpak")).expect("user scope");
        let system =
            shown.iter().position(|p| p.contains("/var/lib/flatpak")).expect("system scope");
        assert!(
            user < system,
            "per-user scope must be searched first: it is GNOME Software's default"
        );
    }

    #[test]
    fn a_relative_xdg_data_home_is_refused_rather_than_joined_onto() {
        // Joining a relative value onto the home directory produces a path
        // that is not where anything is, and then reports "not installed".
        let relative = |key: &str| (key == "XDG_DATA_HOME").then(|| PathBuf::from("relative/bits"));
        let found = candidates_for("linux", Some(&home()), &relative);
        assert!(
            found.iter().all(|path| !path.display().to_string().contains("relative/bits")),
            "a relative XDG_DATA_HOME was joined onto instead of being refused"
        );
        assert!(
            found.iter().any(|path| path.display().to_string().contains(".local/share/flatpak")),
            "refusing it should fall back to ~/.local/share, not give up"
        );
    }

    #[test]
    fn windows_and_macos_look_where_those_platforms_install() {
        let with_program_files = |key: &str| match key {
            "ProgramFiles" => Some(PathBuf::from(r"C:\Program Files")),
            _ => None,
        };
        let windows = candidates_for("windows", None, &with_program_files);
        assert!(
            windows.iter().any(|p| p.ends_with("OrcaSlicer/orca-slicer.exe")),
            "the Program Files install is not looked for"
        );

        let macos = candidates_for("macos", Some(&home()), &no_env);
        assert!(
            macos.iter().any(|p| p.starts_with("/Applications")),
            "the system Applications folder is not looked for"
        );
        assert!(
            macos.iter().any(|p| p.starts_with("/home/someone/Applications")),
            "a per-user install is not looked for"
        );
    }

    #[test]
    fn nothing_is_looked_for_under_a_home_that_is_not_known() {
        // Every entry has to be absolute and real; a `None` home must drop the
        // entries that needed it rather than producing relative paths.
        let found = candidates_for("linux", None, &no_env);
        assert!(found.iter().all(|path| path.is_absolute()), "a relative candidate: {found:?}");
        assert!(!found.is_empty(), "the system paths do not need a home directory");
    }

    #[test]
    fn only_a_model_file_is_handed_over() {
        // The guard that keeps this from becoming a way to open an arbitrary
        // file with an arbitrary program.
        let slicer = PathBuf::from("/bin/true");
        assert!(open(&slicer, Path::new("/tmp/x.brokkr")).is_err());
        assert!(open(&slicer, Path::new("/tmp/x")).is_err());
        assert!(open(&slicer, Path::new("/tmp/x.sh")).is_err());
        // And the ones it will: `/bin/true` exists on this machine, so these
        // exercise the spawn as well as the check.
        assert!(open(&slicer, Path::new("/tmp/x.stl")).is_ok());
        assert!(open(&slicer, Path::new("/tmp/x.3MF")).is_ok(), "the check is case insensitive");
    }

    /// Not a unit test of the table but a check against this machine, and it
    /// skips loudly rather than failing where OrcaSlicer is not installed.
    ///
    /// It is here because the table is only worth anything if it finds a real
    /// install, and the one on this machine is a versioned AppImage --
    /// `OrcaSlicer_V2.4.0-alpha.AppImage` -- which is exactly the case
    /// SindriCAD hardcoded by name and would stop finding on the next update.
    #[test]
    fn the_installed_slicer_is_found_where_it_actually_is() {
        let Some(found) = find() else {
            println!("skipping: no OrcaSlicer installed here");
            return;
        };
        assert!(found.is_file(), "{} is not a file", found.display());
        let name = found.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        assert!(
            name.to_lowercase().contains("orca"),
            "found something that is not OrcaSlicer: {}",
            found.display()
        );
        println!("found {}", found.display());
    }
}
