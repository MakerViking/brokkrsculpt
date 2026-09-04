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

/// This user's home directory.
///
/// **`HOME` is not set on Windows** -- it is `USERPROFILE` there, and `HOME`
/// only exists if a shell like Git Bash put it there. Every home-relative path
/// in this module therefore resolves to nothing on Windows unless both are
/// consulted, which is the quiet half of the failure SindriCAD's own notes
/// describe: nobody exercises the non-Linux branch, so a path that silently
/// resolves to nowhere looks exactly like "the slicer is not installed".
fn home_directory() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
}

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
    let home = home_directory();
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

// --- Where the plate is ------------------------------------------------------
//
// A 3MF opened as a project is placed exactly where the file says, so the
// writer has to know where the middle of the bed is. The printer itself cannot
// answer: the U1's Moonraker reports `axis_maximum [271, 335, 275]`, which is
// the motion range including the toolchanger dock, not the printable area. The
// slicer's own machine preset does know, and it is the same preset the user
// will slice with, so that is what is read here.

/// The flatpak app id this binary belongs to, if it is a flatpak export.
fn flatpak_id(slicer: &Path) -> Option<&'static str> {
    let text = slicer.to_str()?;
    ORCA_FLATPAK_IDS.into_iter().find(|id| text.ends_with(id))
}

/// OrcaSlicer's own data directory, where its presets live.
///
/// **A flatpak Orca never reads `~/.config/OrcaSlicer`**: inside the sandbox
/// `XDG_CONFIG_HOME` is `~/.var/app/<app-id>/config`, so on the host its
/// presets sit at `~/.var/app/<app-id>/config/OrcaSlicer`. The datadir follows
/// the binary [`find`] resolved, so it describes the install we would actually
/// launch. Ported from SindriCAD, which learned it the hard way.
fn datadir_for(
    os: &str,
    home: Option<&PathBuf>,
    env: &dyn Fn(&str) -> Option<PathBuf>,
    slicer: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(id) = slicer.and_then(flatpak_id) {
        return home.map(|home| home.join(".var/app").join(id).join("config/OrcaSlicer"));
    }
    match os {
        "windows" => env("APPDATA").map(|base| base.join("OrcaSlicer")),
        "macos" => home.map(|home| home.join("Library/Application Support/OrcaSlicer")),
        _ => home.map(|home| home.join(".config/OrcaSlicer")),
    }
}

/// Where a slicer of this lineage keeps its presets, wherever it was installed
/// from.
///
/// # Why this is found by SHAPE rather than by name
///
/// `OrcaSlicer` is not the only directory worth reading. Snapmaker ships its own
/// OrcaSlicer fork -- and Snapmaker owners are exactly the people this feature
/// is for -- BambuStudio is the same lineage again, and every one of them can
/// arrive as a native package, an AppImage, a flatpak or a snap, each with a
/// different root. A hardcoded list of names times a list of packaging formats
/// is a table that is wrong the day somebody forks again.
///
/// So a directory qualifies if it LOOKS like one: a `system/` tree, a `user/`
/// tree, and a `<name>.conf` beside them. Verified against two real, unrelated
/// installations on this machine -- a native OrcaSlicer and a flatpak
/// BambuStudio -- which have byte-identical layouts and different names.
fn is_preset_dir(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(std::ffi::OsStr::to_str) else { return false };
    path.join("system").is_dir()
        && path.join("user").is_dir()
        && path.join(format!("{name}.conf")).is_file()
}

/// The slicer's own configuration file, named after its directory.
///
/// **Not hardcoded to `OrcaSlicer.conf`.** A fork's is named after the fork, so
/// hardcoding it finds a Snapmaker Orca installation and then reads nothing out
/// of it -- which looks like "no printer configured" rather than like a bug.
fn conf_path(datadir: &Path) -> Option<PathBuf> {
    let name = datadir.file_name().and_then(std::ffi::OsStr::to_str)?;
    Some(datadir.join(format!("{name}.conf")))
}

/// Every directory a slicer of this lineage could keep its presets under.
///
/// Pure, so the whole table is exercisable from one platform. The flatpak and
/// snap roots are globs over *any* application id rather than the two Orca ones,
/// because that is what catches a fork nobody here has heard of.
fn config_roots(
    os: &str,
    home: Option<&PathBuf>,
    env: &dyn Fn(&str) -> Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    match os {
        "windows" => {
            roots.extend(env("APPDATA"));
            roots.extend(env("LOCALAPPDATA"));
        }
        "macos" => {
            if let Some(home) = home {
                roots.push(home.join("Library/Application Support"));
            }
        }
        _ => {
            roots.extend(env("XDG_CONFIG_HOME").filter(|path| path.is_absolute()));
            if let Some(home) = home {
                roots.push(home.join(".config"));
                // Every flatpak's sandboxed config, and every snap's.
                for (base, tail) in
                    [(home.join(".var/app"), "config"), (home.join("snap"), "current/.config")]
                {
                    if let Ok(entries) = std::fs::read_dir(&base) {
                        for entry in entries.flatten() {
                            roots.push(entry.path().join(tail));
                        }
                    }
                }
            }
        }
    }
    roots
}

/// Every preset directory found under these roots, most recently used first.
///
/// Ordered by the modification time of the slicer's own config file, which is
/// touched on exit -- so when somebody has both OrcaSlicer and a vendor fork
/// installed, the one they actually work in is preferred over the one they
/// opened once.
fn discover(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(root) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if !is_preset_dir(&path) {
                continue;
            }
            let used = conf_path(&path)
                .and_then(|conf| std::fs::metadata(conf).ok())
                .and_then(|data| data.modified().ok())
                .unwrap_or(std::time::UNIX_EPOCH);
            found.push((used, path));
        }
    }
    found.sort_by_key(|(used, _)| std::cmp::Reverse(*used));
    found.into_iter().map(|(_, path)| path).collect()
}

/// An explicit datadir, for an installation nothing here can find.
///
/// The escape hatch that makes the rest of this safe to be wrong about: a
/// packaging format or a fork that no probe here knows is one line in
/// `slicer.conf`, rather than a bug report and a wait for a release.
fn configured_datadir() -> Option<PathBuf> {
    let path = crate::paths::config_file("slicer.conf")?;
    let text = std::fs::read_to_string(path).ok()?;
    crate::paths::entries(&text)
        .find(|(key, _)| *key == "datadir")
        .map(|(_, value)| PathBuf::from(value))
        .filter(|path| path.is_absolute() && path.is_dir())
}

/// The preset directory to read, if there is one.
///
/// In order: what the user wrote down, then the conventional location for the
/// binary we would actually launch, then anything that looks like one anywhere.
pub(crate) fn datadir() -> Option<PathBuf> {
    if let Some(configured) = configured_datadir() {
        return Some(configured);
    }
    let home = home_directory();
    let env = |key: &str| std::env::var_os(key).map(PathBuf::from);
    let conventional = datadir_for(std::env::consts::OS, home.as_ref(), &env, find().as_deref());
    if let Some(path) = conventional.filter(|path| is_preset_dir(path)) {
        return Some(path);
    }
    discover(&config_roots(std::env::consts::OS, home.as_ref(), &env)).into_iter().next()
}

/// The name of the machine preset OrcaSlicer last had selected.
///
/// Pure, so the parsing is testable without an OrcaSlicer installation.
fn active_machine(conf: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(conf).ok()?;
    let name = parsed.get("presets")?.get("machine")?.as_str()?;
    if name.is_empty() { None } else { Some(name.to_string()) }
}

/// The middle of a preset's `printable_area`.
///
/// The field is a list of corners spelled `"XxY"` -- the U1's is
/// `["0.5x1", "270.5x1", "270.5x271", "0.5x271"]`, so the bed is not quite
/// square and does not start at the origin, which is exactly why this is read
/// rather than assumed. Any corner that does not parse is skipped rather than
/// failing the lot; fewer than two leaves no area to speak of.
fn plate_centre_from_area(area: &[String]) -> Option<(f32, f32)> {
    let mut low = (f32::INFINITY, f32::INFINITY);
    let mut high = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut corners = 0;
    for corner in area {
        let Some((x, y)) = corner.split_once('x') else { continue };
        let (Ok(x), Ok(y)) = (x.trim().parse::<f32>(), y.trim().parse::<f32>()) else { continue };
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        low = (low.0.min(x), low.1.min(y));
        high = (high.0.max(x), high.1.max(y));
        corners += 1;
    }
    if corners < 2 {
        return None;
    }
    Some(((low.0 + high.0) / 2.0, (low.1 + high.1) / 2.0))
}

/// Every preset of one kind on disk, by the name Orca knows it as.
///
/// System presets first and the user's own last, so a user preset shadows a
/// system one of the same name -- which is the order Orca resolves them in.
/// One level of vendor subdirectory is descended, because filament libraries
/// nest by brand.
fn index_presets(datadir: &Path, kind: &str) -> std::collections::HashMap<String, PathBuf> {
    let mut directories = Vec::new();
    if let Ok(vendors) = std::fs::read_dir(datadir.join("system")) {
        for vendor in vendors.flatten() {
            let root = vendor.path().join(kind);
            if !root.is_dir() {
                continue;
            }
            if let Ok(nested) = std::fs::read_dir(&root) {
                for entry in nested.flatten() {
                    if entry.path().is_dir() {
                        directories.push(entry.path());
                    }
                }
            }
            directories.push(root);
        }
    }
    directories.push(datadir.join("user/default").join(kind));

    let mut found = std::collections::HashMap::new();
    for directory in directories {
        let Ok(entries) = std::fs::read_dir(&directory) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
                continue;
            }
            let name = std::fs::read_to_string(&path)
                .ok()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| {
                    value.get("name").and_then(serde_json::Value::as_str).map(str::to_string)
                });
            if let Some(name) = name {
                found.insert(name, path);
            }
        }
    }
    found
}

/// The machine presets, which is all [`printable_area`] needs.
fn machine_presets(datadir: &Path) -> std::collections::HashMap<String, PathBuf> {
    index_presets(datadir, "machine")
}

/// Walk a preset's `inherits` chain until something states a `printable_area`.
///
/// **Only the one key is resolved, not the whole preset.** A child overrides
/// its parent, so the first definition found walking upward is the effective
/// one -- which makes merging all 159 keys of a real machine preset needless
/// work for a question about the bed.
fn printable_area(
    presets: &std::collections::HashMap<String, PathBuf>,
    start: &str,
) -> Option<Vec<String>> {
    let mut seen = std::collections::HashSet::new();
    let mut current = Some(start.to_string());
    while let Some(name) = current {
        if !seen.insert(name.clone()) {
            return None; // A cycle. Orca would not load this either.
        }
        let path = presets.get(&name)?;
        let text = std::fs::read_to_string(path).ok()?;
        let value: serde_json::Value = serde_json::from_str(&text).ok()?;
        if let Some(area) = value.get("printable_area").and_then(serde_json::Value::as_array) {
            let corners: Vec<String> =
                area.iter().filter_map(serde_json::Value::as_str).map(str::to_string).collect();
            if !corners.is_empty() {
                return Some(corners);
            }
        }
        current = value
            .get("inherits")
            .and_then(serde_json::Value::as_str)
            .filter(|parent| !parent.is_empty())
            .map(str::to_string);
    }
    None
}

/// The middle of the plate the user's active OrcaSlicer machine describes.
///
/// `None` whenever any link in that chain is missing, and the caller then
/// exports a model that sits on the bed without being centred. **That happens
/// more often than it looks**: opening a project whose settings name no printer
/// makes Orca invent a project-custom preset named after the file and write
/// *that* name into its config, so the last-selected machine can be a preset
/// that exists nowhere on disk.
pub(crate) fn plate_centre() -> Option<(f32, f32)> {
    let datadir = datadir()?;
    let conf = std::fs::read_to_string(conf_path(&datadir)?).ok()?;
    let machine = active_machine(&conf)?;
    let area = printable_area(&machine_presets(&datadir), &machine)?;
    plate_centre_from_area(&area)
}

/// Per-file bookkeeping that is not effective configuration, dropped once an
/// `inherits` chain is merged. Mirrors Orca's own preset model.
const META_KEYS: [&str; 10] = [
    "inherits",
    "from",
    "name",
    "setting_id",
    "filament_id",
    "renamed_from",
    "is_custom_defined",
    "version",
    "upward_compatible_machine",
    "instantiation",
];

/// Whether a preset key carries something that must not leave this machine.
///
/// **A machine preset holds the printer's address and its credentials** --
/// `print_host`, `printhost_apikey`, `printhost_password`, `printhost_user`
/// and friends were all present in the real preset this was written against.
/// An exported 3MF is a file people hand to other people, so embedding the
/// preset verbatim would publish a LAN address and an API key to whoever
/// receives the model.
///
/// **SindriCAD does exactly that**, and its own doc comment celebrates the
/// result ("Orca then selects the U1, with its print_host"). This is the one
/// place the port deliberately diverges. Matched by shape rather than by an
/// exact list, so a key a future Orca adds is excluded by default rather than
/// leaking until somebody notices.
///
/// Dropping these does not stop Orca binding the preset: it matches on the
/// preset *name*, and the user's own copy already holds their host.
fn is_secret(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key.starts_with("printhost_")
        || key.contains("print_host")
        || key.contains("apikey")
        || key.contains("api_key")
        || key.contains("password")
        || key.contains("access_code")
        || key.contains("token")
        || key.contains("secret")
}

/// Flatten a preset by walking its `inherits` chain, child overriding parent.
///
/// Returns the merged configuration and every name in the chain, which is what
/// a compatibility check matches against.
fn resolve_chain(
    presets: &std::collections::HashMap<String, PathBuf>,
    name: &str,
) -> Option<(serde_json::Map<String, serde_json::Value>, std::collections::HashSet<String>)> {
    let mut chain = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut current = Some(name.to_string());
    while let Some(name) = current {
        if !names.insert(name.clone()) {
            break; // A cycle. Orca would not load this either.
        }
        let path = presets.get(&name)?;
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
        current = value
            .get("inherits")
            .and_then(serde_json::Value::as_str)
            .filter(|parent| !parent.is_empty())
            .map(str::to_string);
        chain.push(value);
    }

    let mut merged = serde_json::Map::new();
    for value in chain.into_iter().rev() {
        if let Some(object) = value.as_object() {
            for (key, value) in object {
                if !is_secret(key) {
                    merged.insert(key.clone(), value.clone());
                }
            }
        }
    }
    for key in META_KEYS {
        merged.remove(key);
    }
    Some((merged, names))
}

/// Whether a preset is compatible with the machine whose chain this is.
///
/// The candidate's own chain is resolved first, so a sparse user override
/// inherits `compatible_printers` from its system parent rather than looking
/// compatible with nothing.
fn is_compatible(
    presets: &std::collections::HashMap<String, PathBuf>,
    name: &str,
    chain: &std::collections::HashSet<String>,
) -> bool {
    resolve_chain(presets, name)
        .and_then(|(config, _)| {
            config.get("compatible_printers").and_then(serde_json::Value::as_array).cloned()
        })
        .is_some_and(|list| {
            list.iter().filter_map(serde_json::Value::as_str).any(|name| chain.contains(name))
        })
}

/// Whether a preset file is one the user made, rather than a shipped one.
///
/// **Matches a path COMPONENT, not a `"/user/"` substring**: on Windows the
/// separator is `\`, so a substring test never matches, every user preset is
/// filed as a system one, and the "prefer the user's own" rule silently
/// inverts there. Ported along with the reason.
fn is_user_preset(path: &Path) -> bool {
    path.components().any(|part| part.as_os_str() == "user")
}

/// The best preset of a kind for a machine: compatible with it, preferring the
/// user's own, then a name hint, then alphabetical.
fn pick_preset(
    presets: &std::collections::HashMap<String, PathBuf>,
    chain: &std::collections::HashSet<String>,
    hint: &str,
) -> Option<String> {
    let (mut mine, mut shipped) = (Vec::new(), Vec::new());
    for (name, path) in presets {
        if !is_compatible(presets, name, chain) {
            continue;
        }
        if is_user_preset(path) { mine.push(name.clone()) } else { shipped.push(name.clone()) }
    }
    for mut pool in [mine, shipped] {
        pool.sort();
        if let Some(hinted) = pool.iter().find(|name| name.contains(hint)) {
            return Some(hinted.clone());
        }
        if let Some(first) = pool.first() {
            return Some(first.clone());
        }
    }
    None
}

/// Every machine preset describing the same physical printer as this one.
///
/// **A process preset declares `compatible_printers` naming the SYSTEM machine**
/// -- `["Snapmaker U1 (0.4 nozzle)"]` -- while the machine a user actually has
/// selected is very often their own copy of it, `"Snapmaker U1 (0.4 nozzle) -
/// Tuned"`, saved standalone with `inherits` empty. Its chain is then just
/// itself, nothing declares compatibility with that name, and every process and
/// filament preset is judged incompatible: the export names a printer correctly
/// and still leaves Orca inventing a project process and four project filaments.
/// Observed exactly that way on 2026-09-02.
///
/// Matching on model AND variant, not model alone: a 0.8 nozzle profile is not
/// compatible with a 0.4 machine, and offering one would be worse than offering
/// none.
fn sibling_machines(
    presets: &std::collections::HashMap<String, PathBuf>,
    model: Option<&str>,
    variant: Option<&str>,
) -> Vec<String> {
    let Some(model) = model else { return Vec::new() };
    presets
        .iter()
        .filter(|(_, path)| {
            let Ok(text) = std::fs::read_to_string(path) else { return false };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else { return false };
            let field = |key: &str| value.get(key).and_then(serde_json::Value::as_str);
            field("printer_model") == Some(model)
                && (variant.is_none() || field("printer_variant") == variant)
        })
        .map(|(name, _)| name.clone())
        .collect()
}

/// The `project_settings.config` body that makes OrcaSlicer bind the user's own
/// presets instead of inventing ones named after the file.
///
/// **A machine-only config is not enough**, and neither is the three-key
/// minimum: given those, Orca creates a project-custom machine, process and
/// filament all named `(yourfile.3mf)` and writes that machine name into its
/// own configuration, so the printer the user had selected is displaced by
/// opening a model. All three kinds are therefore flattened and embedded.
///
/// `slots` is how many filament entries the per-slot arrays get -- Orca expects
/// one entry per tool head, not one per preset.
///
/// Returns `None` whenever the chain cannot be followed, and the caller then
/// falls back to the minimal settings: worse binding, but never a wrong one.
pub(crate) fn project_settings(slots: usize) -> Option<serde_json::Map<String, serde_json::Value>> {
    let datadir = datadir()?;
    let conf = std::fs::read_to_string(conf_path(&datadir)?).ok()?;
    let machine = active_machine(&conf)?;

    let machines = index_presets(&datadir, "machine");
    let (mut config, mut chain) = resolve_chain(&machines, &machine)?;

    // Widen the chain to the same printer's other presets before anything is
    // judged compatible against it. See `sibling_machines` for why.
    let model = config.get("printer_model").and_then(serde_json::Value::as_str).map(str::to_string);
    let variant =
        config.get("printer_variant").and_then(serde_json::Value::as_str).map(str::to_string);
    chain.extend(sibling_machines(&machines, model.as_deref(), variant.as_deref()));

    // The process, which is what stops Orca showing a blank project profile.
    let processes = index_presets(&datadir, "process");
    if let Some(name) = pick_preset(&processes, &chain, "0.20")
        && let Some((process, _)) = resolve_chain(&processes, &name)
    {
        config.extend(process);
        config.insert("print_settings_id".into(), name.into());
    }

    // The filament, spread across the slots as per-slot arrays.
    let slots = slots.max(1);
    let filaments = index_presets(&datadir, "filament");
    if let Some(name) = pick_preset(&filaments, &chain, "PLA")
        && let Some((filament, _)) = resolve_chain(&filaments, &name)
    {
        for (key, value) in filament {
            let one = match &value {
                serde_json::Value::Array(entries) if !entries.is_empty() => entries[0].clone(),
                other => other.clone(),
            };
            config.insert(key, serde_json::Value::Array(vec![one; slots]));
        }
        config.insert(
            "filament_settings_id".into(),
            serde_json::Value::Array(vec![serde_json::Value::String(name); slots]),
        );
    }

    config.entry("printer_settings_id").or_insert_with(|| machine.into());
    // The palette owns the colours, and a compatibility list means nothing in a
    // project. Both are re-added or dropped by the caller.
    config.remove("filament_colour");
    config.remove("compatible_printers");
    Some(config)
}

/// The whole `project_settings.config` body for an export, palette included.
///
/// `None` when the user's OrcaSlicer presets cannot be read at all, and the
/// writer then falls back to its own minimal three keys -- fewer colours bound
/// and no printer, but nothing wrong.
///
/// **The palette is applied last and wins.** The flattened filament preset
/// carries a `filament_colour` of its own, and letting that through would show
/// the user the preset's colours rather than the ones their sculpt declares.
pub(crate) fn project_settings_body(palette: &crate::palette::Palette) -> Option<String> {
    let mut config = project_settings(palette.slots.len())?;
    let filaments = palette.as_filaments();
    config.insert("filament_colour".into(), filaments.colours.into());
    config.insert("filament_type".into(), filaments.materials.into());
    config.insert("from".into(), "project".into());
    serde_json::to_string_pretty(&serde_json::Value::Object(config)).ok()
}

/// Open a file in OrcaSlicer.///
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

    /// The real U1's printable area, copied off the machine preset on this
    /// machine on 2026-09-02. It is neither square nor at the origin, which is
    /// the whole reason this is read rather than assumed -- a 220x220 guess
    /// would have been wrong by 50 mm in both directions.
    #[test]
    fn the_plate_centre_comes_from_the_corners_the_preset_names() {
        let u1: Vec<String> =
            ["0.5x1", "270.5x1", "270.5x271", "0.5x271"].map(str::to_string).to_vec();
        let (x, y) = plate_centre_from_area(&u1).expect("four corners are an area");
        assert!((x - 135.5).abs() < 1.0e-3, "x was {x}");
        assert!((y - 136.0).abs() < 1.0e-3, "y was {y}");
    }

    #[test]
    fn a_plate_that_does_not_parse_is_no_plate_rather_than_a_guess() {
        assert_eq!(plate_centre_from_area(&[]), None);
        assert_eq!(plate_centre_from_area(&["nonsense".to_string()]), None);
        // One good corner is a point, not an area.
        assert_eq!(plate_centre_from_area(&["0x0".to_string(), "junk".to_string()]), None);
        // A corner that does not parse is skipped, not fatal.
        let mixed = ["0x0".to_string(), "oops".to_string(), "200x100".to_string()];
        assert_eq!(plate_centre_from_area(&mixed), Some((100.0, 50.0)));
    }

    #[test]
    fn the_active_machine_is_read_from_orcas_own_config() {
        assert_eq!(
            active_machine(r#"{"presets":{"machine":"Snapmaker U1 (0.4 nozzle) - Tuned"}}"#),
            Some("Snapmaker U1 (0.4 nozzle) - Tuned".to_string())
        );
        // Absent, empty, and not-even-JSON are all simply "no machine", because
        // the caller's fallback -- sit on the bed without centring -- is safe.
        assert_eq!(active_machine(r#"{"presets":{}}"#), None);
        assert_eq!(active_machine(r#"{"presets":{"machine":""}}"#), None);
        assert_eq!(active_machine("not json at all"), None);
    }

    /// A flatpak Orca keeps its presets inside the sandbox's config directory,
    /// and reading the host's would find a different install's settings or none.
    #[test]
    fn a_flatpak_orca_looks_inside_its_own_sandbox() {
        let exported = PathBuf::from("/var/lib/flatpak/exports/bin/com.orcaslicer.OrcaSlicer");
        let found = datadir_for("linux", Some(&home()), &no_env, Some(&exported));
        assert_eq!(
            found,
            Some(PathBuf::from(
                "/home/someone/.var/app/com.orcaslicer.OrcaSlicer/config/OrcaSlicer"
            ))
        );
        // A native install is the plain config directory.
        let native = PathBuf::from("/usr/bin/orca-slicer");
        assert_eq!(
            datadir_for("linux", Some(&home()), &no_env, Some(&native)),
            Some(PathBuf::from("/home/someone/.config/OrcaSlicer"))
        );
    }

    /// The datadir on the two platforms nobody here can run.
    ///
    /// The house rule in this module: `os` and `env` are parameters rather than
    /// `cfg!`, so every branch is exercisable from the Linux runner. These two
    /// were added without it, which is precisely how a path that resolves to
    /// nowhere ships looking like "the slicer is not installed".
    #[test]
    fn the_datadir_is_right_on_windows_and_macos_too() {
        let appdata = |key: &str| {
            (key == "APPDATA").then(|| PathBuf::from(r"C:\Users\someone\AppData\Roaming"))
        };
        // The expectation is joined rather than spelled out: this suite runs on
        // Linux, where `join` uses `/`, so a hand-written `\` would be testing
        // the runner's separator instead of the branch.
        let roaming = PathBuf::from(r"C:\Users\someone\AppData\Roaming");
        assert_eq!(
            datadir_for("windows", None, &appdata, None),
            Some(roaming.join("OrcaSlicer")),
            "Windows does not use HOME, so this must come from APPDATA alone"
        );
        // No APPDATA is no datadir, not a path relative to nothing.
        assert_eq!(datadir_for("windows", Some(&home()), &no_env, None), None);

        assert_eq!(
            datadir_for("macos", Some(&home()), &no_env, None),
            Some(PathBuf::from("/home/someone/Library/Application Support/OrcaSlicer"))
        );
        // And no home is no datadir on the platforms that need one.
        assert_eq!(datadir_for("macos", None, &no_env, None), None);
        assert_eq!(datadir_for("linux", None, &no_env, None), None);
    }

    /// A preset tree like the real one: a user preset inheriting from a system
    /// one, with only the parent stating the bed.
    ///
    /// This is the link the real-install test above cannot reach on a machine
    /// whose last-selected machine is a project-custom preset, which is exactly
    /// the state opening one of our own exports leaves OrcaSlicer in.
    #[test]
    fn a_preset_inherits_its_plate_from_its_parent() {
        let root = std::env::temp_dir().join(format!("brokkr-orca-{}", std::process::id()));
        let system = root.join("system/Snapmaker/machine");
        let user = root.join("user/default/machine");
        std::fs::create_dir_all(&system).expect("a temp tree");
        std::fs::create_dir_all(&user).expect("a temp tree");
        std::fs::write(
            system.join("base.json"),
            r#"{"name":"Snapmaker U1 0.4","printable_area":["0.5x1","270.5x1","270.5x271","0.5x271"]}"#,
        )
        .unwrap();
        // The child says nothing about the bed, which is the normal shape of a
        // user preset that only tweaks temperatures.
        std::fs::write(
            user.join("tuned.json"),
            r#"{"name":"U1 Tuned","inherits":"Snapmaker U1 0.4","nozzle_temperature":"230"}"#,
        )
        .unwrap();

        let presets = machine_presets(&root);
        assert!(presets.contains_key("U1 Tuned"), "the user preset was not indexed");
        assert!(presets.contains_key("Snapmaker U1 0.4"), "the system preset was not indexed");

        let area = printable_area(&presets, "U1 Tuned").expect("the parent states a plate");
        assert_eq!(plate_centre_from_area(&area), Some((135.5, 136.0)));

        // A preset that is not there at all is no plate, not a panic.
        assert_eq!(printable_area(&presets, "(sculpt.3mf)"), None);

        std::fs::remove_dir_all(&root).ok();
    }

    /// The keys that must never leave this machine.
    ///
    /// Every name here was present in the real Snapmaker U1 machine preset this
    /// was written against. An exported 3MF is a file people send to other
    /// people; a printer's address and API key have no business travelling with
    /// a model.
    #[test]
    fn a_printers_address_and_credentials_are_never_embedded() {
        for key in [
            "print_host",
            "print_host_webui",
            "printhost_apikey",
            "printhost_password",
            "printhost_user",
            "printhost_cafile",
            "printhost_port",
            "printhost_authorization_type",
            "printer_access_code",
            "bbl_auth_token",
            "PRINTHOST_APIKEY",
        ] {
            assert!(is_secret(key), "{key} would have been embedded in an exported model");
        }
        // And ordinary configuration is not swept up with them. `host_type`
        // names a protocol, not an address, and the gcode hooks are the user's
        // own machine setup.
        for key in [
            "printable_area",
            "printer_model",
            "machine_start_gcode",
            "nozzle_diameter",
            "host_type",
        ] {
            assert!(!is_secret(key), "{key} was stripped, which breaks the binding");
        }
    }

    #[test]
    fn a_flattened_preset_carries_the_configuration_and_not_the_credentials() {
        let root = std::env::temp_dir().join(format!("brokkr-orca-flat-{}", std::process::id()));
        let system = root.join("system/Snapmaker/machine");
        let user = root.join("user/default/machine");
        std::fs::create_dir_all(&system).expect("a temp tree");
        std::fs::create_dir_all(&user).expect("a temp tree");
        std::fs::write(
            system.join("base.json"),
            r#"{"name":"Base","printer_model":"Snapmaker U1","nozzle_diameter":["0.4"],
                "print_host":"192.0.2.46","printhost_apikey":"SUPERSECRET"}"#,
        )
        .unwrap();
        std::fs::write(
            user.join("tuned.json"),
            r#"{"name":"Tuned","inherits":"Base","nozzle_temperature":"230",
                "printhost_password":"ALSOSECRET"}"#,
        )
        .unwrap();

        let presets = index_presets(&root, "machine");
        let (config, chain) = resolve_chain(&presets, "Tuned").expect("the chain resolves");

        // The child overrode nothing it should not, and the parent's real
        // configuration came through.
        assert_eq!(config.get("printer_model").and_then(|v| v.as_str()), Some("Snapmaker U1"));
        assert_eq!(config.get("nozzle_temperature").and_then(|v| v.as_str()), Some("230"));
        // Both names are in the chain, which is what a compatibility check needs.
        assert!(chain.contains("Base") && chain.contains("Tuned"));
        // Meta keys are bookkeeping, not configuration.
        assert!(!config.contains_key("inherits") && !config.contains_key("name"));

        // The point of the test.
        let rendered = serde_json::to_string(&serde_json::Value::Object(config)).unwrap();
        for leaked in ["SUPERSECRET", "ALSOSECRET", "192.0.2.46"] {
            assert!(!rendered.contains(leaked), "{leaked} reached an exportable config");
        }

        std::fs::remove_dir_all(&root).ok();
    }

    /// The sculpt's own palette beats the filament preset's colours.
    #[test]
    fn the_palette_owns_the_colours_in_the_project_settings() {
        let mut palette = crate::palette::Palette::default();
        palette.slots[0].colour = "#ABCDEF".to_string();
        let Some(body) = project_settings_body(&palette) else {
            println!("skipping: no OrcaSlicer presets on this machine");
            return;
        };
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        let colours = parsed.get("filament_colour").and_then(|v| v.as_array()).expect("colours");
        assert_eq!(colours[0].as_str(), Some("#ABCDEF"), "a preset colour beat the palette");
        assert_eq!(colours.len(), palette.slots.len(), "one colour per slot");
        assert_eq!(parsed.get("from").and_then(|v| v.as_str()), Some("project"));
    }

    /// Which presets the real installation actually binds.
    ///
    /// Printing them is the point: a machine that binds while the process and
    /// filament do not is exactly the state that looked fixed and was not.
    #[test]
    fn the_real_install_binds_a_process_and_a_filament_too() {
        let Some(config) = project_settings(4) else {
            println!("skipping: no OrcaSlicer presets on this machine");
            return;
        };
        let named = |key: &str| match config.get(key) {
            Some(serde_json::Value::String(name)) => Some(name.clone()),
            Some(serde_json::Value::Array(names)) => {
                names.first().and_then(|v| v.as_str()).map(str::to_string)
            }
            _ => None,
        };
        let printer = named("printer_settings_id");
        let process = named("print_settings_id");
        let filament = named("filament_settings_id");
        println!("  printer:  {printer:?}");
        println!("  process:  {process:?}");
        println!("  filament: {filament:?}");
        assert!(printer.is_some(), "no printer was named at all");
        // These two are what a widened compatibility chain buys. Without them
        // Orca invents `(yourfile.3mf)` for each, which is the reported bug.
        assert!(process.is_some(), "no process preset was compatible with the active machine");
        assert!(filament.is_some(), "no filament preset was compatible with the active machine");
    }

    /// Against the real installation, and this one is a privacy check rather
    /// than a behaviour check: whatever this machine's presets hold, none of it
    /// may reach a file the user might share.
    #[test]
    fn the_real_project_settings_carry_no_address_and_no_credential() {
        let palette = crate::palette::Palette::default();
        let Some(body) = project_settings_body(&palette) else {
            println!("skipping: no OrcaSlicer presets on this machine");
            return;
        };
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        let object = parsed.as_object().expect("an object");
        for key in object.keys() {
            assert!(!is_secret(key), "the export would have carried {key}");
        }
        // And the configured printer's own address is not in there under some
        // key this test did not think of. Never printed, only checked.
        if let Some(printer) = crate::printer::configured(crate::printer::config_path().as_deref())
        {
            assert!(
                !body.contains(&printer.host),
                "the printer's address reached the exported project settings"
            );
        }
        println!("{} keys, no secrets", object.len());
    }

    /// Every preset directory on this machine, whatever it is called.
    ///
    /// This box has a native OrcaSlicer and a flatpak BambuStudio, which is a
    /// genuinely useful pair: two different names, two different packaging
    /// formats, one layout. A Snapmaker Orca would come through the same way.
    #[test]
    fn discovery_finds_the_slicers_installed_here() {
        let roots = config_roots(std::env::consts::OS, home_directory().as_ref(), &|key| {
            std::env::var_os(key).map(PathBuf::from)
        });
        let found = discover(&roots);
        if found.is_empty() {
            println!("skipping: no slicer presets on this machine");
            return;
        }
        for path in &found {
            println!("  {}", path.display());
            // The shape test and the conf name have to agree, or a fork is
            // found and then read out of the wrong file.
            assert!(is_preset_dir(path));
            assert!(conf_path(path).is_some_and(|conf| conf.is_file()));
        }
        // Most recently used first, so the slicer somebody actually works in
        // wins over one they opened once.
        let times: Vec<_> = found
            .iter()
            .filter_map(|path| conf_path(path))
            .filter_map(|conf| std::fs::metadata(conf).ok()?.modified().ok())
            .collect();
        assert!(times.windows(2).all(|pair| pair[0] >= pair[1]), "not ordered by last use");
    }

    /// A fork nobody here has heard of, in a root nobody here probes.
    #[test]
    fn a_directory_is_a_preset_directory_because_of_its_shape() {
        let root = std::env::temp_dir().join(format!("brokkr-shape-{}", std::process::id()));
        let fork = root.join("SnapmakerOrca");
        std::fs::create_dir_all(fork.join("system")).unwrap();
        std::fs::create_dir_all(fork.join("user")).unwrap();
        assert!(!is_preset_dir(&fork), "no config file yet, so not a preset directory");
        std::fs::write(fork.join("SnapmakerOrca.conf"), "{}").unwrap();
        assert!(is_preset_dir(&fork), "a fork was not recognised");
        assert_eq!(conf_path(&fork), Some(fork.join("SnapmakerOrca.conf")));
        assert_eq!(discover(std::slice::from_ref(&root)), vec![fork.clone()]);

        // A directory that merely sits alongside is not one.
        let cache = root.join("SomethingElse");
        std::fs::create_dir_all(cache.join("system")).unwrap();
        assert!(!is_preset_dir(&cache));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_roots_cover_every_way_a_slicer_is_packaged() {
        let with_appdata = |key: &str| match key {
            "APPDATA" => Some(PathBuf::from("C:/Users/someone/AppData/Roaming")),
            "LOCALAPPDATA" => Some(PathBuf::from("C:/Users/someone/AppData/Local")),
            _ => None,
        };
        let windows = config_roots("windows", None, &with_appdata);
        assert_eq!(windows.len(), 2, "Windows keeps per-user config in both AppData roots");

        let macos = config_roots("macos", Some(&home()), &no_env);
        assert_eq!(macos, vec![PathBuf::from("/home/someone/Library/Application Support")]);

        // On Linux the plain config directory is always a root; the flatpak and
        // snap roots only appear when those trees exist, which they do not for
        // a fabricated home.
        let linux = config_roots("linux", Some(&home()), &no_env);
        assert!(linux.contains(&PathBuf::from("/home/someone/.config")));

        // An XDG override is honoured, and a relative one is refused rather
        // than joined onto something.
        let xdg = |key: &str| (key == "XDG_CONFIG_HOME").then(|| PathBuf::from("/somewhere/else"));
        assert!(
            config_roots("linux", Some(&home()), &xdg).contains(&PathBuf::from("/somewhere/else"))
        );
        let relative = |key: &str| (key == "XDG_CONFIG_HOME").then(|| PathBuf::from("relative"));
        assert!(
            !config_roots("linux", Some(&home()), &relative).contains(&PathBuf::from("relative"))
        );
    }

    /// Against whatever OrcaSlicer is really installed. Skips loudly otherwise,
    /// the way the printer test does.
    #[test]
    fn the_real_orca_install_answers_about_its_plate() {
        let Some(datadir) = datadir() else {
            println!("skipping: no slicer presets found anywhere");
            return;
        };
        println!("reading presets from {}", datadir.display());
        match plate_centre() {
            Some((x, y)) => {
                println!("plate centre: {x} x {y}");
                assert!(x > 0.0 && y > 0.0, "a plate centre behind the origin");
                assert!(x < 2000.0 && y < 2000.0, "a plate two metres across is not a printer");
            }
            // Entirely legitimate: Orca's last-selected machine can be a
            // project-custom preset that exists nowhere on disk.
            None => println!("no plate centre; the export will sit on the bed uncentred"),
        }
    }

    #[test]
    // `candidates_for` takes the platform as an argument, but the `PathBuf` it
    // builds them with does not: `join` uses the HOST separator and
    // `is_absolute` the host rule, so on Windows this table comes back
    // `\`-joined and `/usr/bin/orca-slicer` is not absolute. Asserting the
    // Linux table there would be asserting Windows `PathBuf` semantics. It is
    // also dead code there -- a Windows build only ever asks for the Windows
    // table, which `windows_and_macos_look_where_those_platforms_install`
    // covers from any host because it only checks prefixes.
    #[cfg(unix)]
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
    // Linux table, host `PathBuf` semantics -- see the note above.
    #[cfg(unix)]
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
    // Linux table, host `PathBuf` semantics -- see the note above.
    #[cfg(unix)]
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
        //
        // `/bin/echo` rather than `/bin/true`: macOS ships no `/bin/true`, and
        // that alone failed this test the first time the crate was built there
        // -- the guard was fine, the fixture was not. Windows gets `cmd.exe`
        // for the same reason. All three exist and all three spawn, which is
        // the only property the passing cases need.
        let slicer = if cfg!(target_os = "windows") {
            PathBuf::from(r"C:\Windows\System32\cmd.exe")
        } else {
            PathBuf::from("/bin/echo")
        };
        assert!(open(&slicer, Path::new("/tmp/x.brokkr")).is_err());
        assert!(open(&slicer, Path::new("/tmp/x")).is_err());
        assert!(open(&slicer, Path::new("/tmp/x.sh")).is_err());
        // And the ones it will: the program above exists on this machine, so
        // these exercise the spawn as well as the check.
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
