// SPDX-License-Identifier: AGPL-3.0-only

//! Putting a verified file in place, on all three platforms.
//!
//! This is the only code in the project that can turn a working install into a
//! broken one, so every step below is either non-destructive or is the single
//! irreversible one, and the irreversible one is last.
//!
//! # Why `rename` and not "copy over the binary"
//!
//! `rename(2)` acts on the **directory entry**, not the inode. Writing
//! `brokkrsculpt.new` in the *same directory* and renaming it over the target is
//! atomic, and it does not disturb the running process at all -- that process
//! keeps its old inode open until it exits, which is why a `cargo build` can
//! replace this binary while you are using it. (`/proc/self/exe` on the old
//! process then reads `... (deleted)`, which is how you can tell.)
//!
//! The permission that matters is therefore write and execute on the
//! **containing directory**, not on the file. A read-only binary in a writable
//! directory can be replaced; a writable binary in a read-only directory cannot.
//!
//! # Why the aside is a hard link and not a rename
//!
//! The alternative design renames the target aside and then renames the new file
//! in, defending it as "two renames back to back, the window is microseconds".
//! The window is real: a SIGKILL or a power cut inside it leaves **nothing** at
//! the target path. On Unix the aside is not needed at all -- `rename(new,
//! target)` is already atomic and the running process keeps its inode -- so a
//! hard link is used instead. A link creates a second name for the same inode
//! and removes nothing, so there is no window whatsoever.
//!
//! # What this module does not do
//!
//! It does not restart anything and it does not decide *whether* to update. It
//! is handed a verified file and a target, and it either swaps them or refuses
//! with a reason.

use std::io::Write;
use std::path::{Path, PathBuf};

use super::Refusal;

/// The name the superseded binary is parked under, beside the target.
///
/// A leading dot so it does not clutter a directory listing, and beside the
/// binary rather than in the state directory because `rename` and `link` both
/// require the same filesystem.
const OLD_SUFFIX: &str = ".old";

/// What a user is told to do if the new build will not start at all.
///
/// **Not Windows-only, and it is permanent.** Every step of the auto-revert is
/// code *inside the new build*, so it runs only for a payload that starts and
/// then dies. A payload that cannot `exec` -- built against a newer glibc than
/// the user has, on Linux, or refused by Smart App Control on Windows -- runs
/// none of it, and nothing of ours is left to help. One text file is the whole
/// remedy.
const RECOVERY_NOTE: &str = "RECOVER-BROKKRSCULPT.txt";

/// Where the running executable is, resolved once and then never asked again.
///
/// **This is an invariant with a test, not a note.** After a swap,
/// `current_exe()` resolves through the running inode, which is no longer at the
/// target name -- on Linux it reports the old path with ` (deleted)` appended.
/// A second update in one session that re-resolved would therefore overwrite the
/// rollback copy and never touch the real binary. `current_exe` is used nowhere
/// else in this tree, so all of its caveats are new here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// The canonical path of the executable to replace.
    pub path: PathBuf,
    /// Its containing directory -- the thing that actually has to be writable.
    pub directory: PathBuf,
}

impl Target {
    /// Resolve the running executable. Call once, at startup, before anything
    /// can move.
    pub fn resolve() -> Result<Self, Refusal> {
        let path = std::env::current_exe()
            .map_err(|why| Refusal::CannotReplace(format!("could not find this binary ({why})")))?;
        // `canonicalize` both resolves symlinks and makes the path absolute. A
        // symlinked install (`~/.local/bin/brokkrsculpt` -> elsewhere) must be
        // replaced where the file really lives, or the swap writes over the
        // link and the next launch runs whatever the link used to point at.
        let path = path.canonicalize().map_err(|why| {
            Refusal::CannotReplace(format!("could not resolve this binary ({why})"))
        })?;
        let directory = path
            .parent()
            .ok_or_else(|| Refusal::CannotReplace("this binary has no directory".into()))?
            .to_path_buf();
        Ok(Self { path, directory })
    }

    /// Where the superseded binary is parked.
    pub fn old(&self) -> PathBuf {
        let mut name = std::ffi::OsString::from(".");
        name.push(self.path.file_name().unwrap_or_default());
        name.push(OLD_SUFFIX);
        self.directory.join(name)
    }

    /// Where a payload is staged so the rename that follows stays on one
    /// filesystem.
    ///
    /// Beside the executable normally; beside the **bundle** on macOS, because
    /// the thing being replaced there is the whole `.app` and staging inside it
    /// would put the new copy underneath the directory about to be renamed.
    pub fn staging_directory(&self) -> PathBuf {
        app_bundle_root(&self.path)
            .and_then(|bundle| bundle.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| self.directory.clone())
    }

    /// Where the recovery note lives.
    pub fn recovery_note(&self) -> PathBuf {
        self.directory.join(RECOVERY_NOTE)
    }
}

/// Whether this install may replace itself at all.
///
/// Every one of these is checked before a single byte is downloaded, because a
/// refusal after a 33 MB transfer is a worse experience than a refusal before
/// one, and because the whole point is that nothing destructive begins.
pub fn gates(target: &Target, build: Option<u64>, commit: &str) -> Result<(), Refusal> {
    // **Inert unless this is a stamped release build**, and this is checked
    // FIRST because it is the more specific answer. A developer running a dirty
    // build on a Mac should be told that their build does not update itself,
    // not that their platform hands over -- both are true, and only one of them
    // tells them something about what they are running. An earlier cut had the
    // platform check first and three tests disagreed with it on macOS.
    //
    // Any one of these alone would nearly do; together they are the difference
    // between "inert" and "one keypress away from overwriting the developer's
    // own binary with a CI artefact".
    if build.is_none() || commit == "unknown" || commit.ends_with("-dirty") {
        return Err(Refusal::NotAReleaseBuild);
    }
    // A binary running out of a build directory is a developer's, whatever its
    // stamp says.
    if target.path.components().any(|part| part.as_os_str() == "target") {
        return Err(Refusal::NotAReleaseBuild);
    }
    if self_update_disabled() {
        return Err(Refusal::HandOverOnly);
    }
    // A `.app` IS replaceable now -- as a whole, never edited in place -- so
    // what is checked here is the one thing that makes it impossible.
    if is_translocated(&target.path) {
        return Err(Refusal::CannotReplace(
            "macOS is running this from a temporary read-only copy. Move BrokkrSculpt to \
             Applications in Finder -- dragging it there, not `mv` -- and open it again"
                .into(),
        ));
    }
    // The directory that must be writable is the one holding the thing being
    // replaced: the bundle's folder for a `.app`, the executable's otherwise.
    directory_is_safe(&target.staging_directory())
}

/// Whether the build running now is the one this install put in place.
///
/// The crash-driven offer needs all three of: a pending crash report, a running
/// ordinal that matches `installed_build`, and a `.old` still present with a
/// matching digest. This answers the middle one -- and it answers it by
/// comparing the RUNNING ordinal, so a user who crashed on a build they
/// installed by hand is not offered a revert to something unrelated.
pub fn target_is_the_installed_build(target: &Target) -> bool {
    let _ = target;
    match (crate::app::build_number(), super::installed_build()) {
        (Some(running), Some(installed)) => running == installed,
        _ => false,
    }
}

/// The environment variable that turns self-replacement OFF.
const NO_SELF_UPDATE: &str = "BROKKR_NO_SELF_UPDATE";

/// Whether this install may replace itself at all, before any other question.
///
/// **An escape hatch, not a gate.** Self-replacement ships enabled on Linux and
/// Windows. This exists because the Windows hop has never run on real hardware
/// — `docs/AUTOUPDATE-PLAN.md` wanted a human to take it once before shipping,
/// and there is no Windows desktop on this project, so it ships unproven by a
/// deliberate decision rather than by an oversight.
///
/// What that buys a user who hits trouble: a way to stop it happening again
/// that does not also switch off being TOLD about updates. `check_for_updates =
/// never` silences the check entirely, which is a bigger hammer than "download
/// it and let me install it myself" — and the download path is the one that is
/// well tested on every platform.
///
/// The risk being carried, stated so it is not discovered: a Windows build that
/// Smart App Control declines to execute leaves an application that will not
/// start and no code of ours running to fix it, because nothing of ours is
/// allowed to execute. `RECOVER-BROKKRSCULPT.txt` beside the binary is the
/// entire remedy for that case, which is why it is written before the swap and
/// never deleted.
pub fn self_update_disabled() -> bool {
    std::env::var_os(NO_SELF_UPDATE).is_some_and(|value| value == "1")
}

/// The `.app` directory this executable lives in, if it lives in one.
///
/// `.../Something.app/Contents/MacOS/exe` -> `.../Something.app`.
fn app_bundle_root(path: &Path) -> Option<PathBuf> {
    is_in_app_bundle(path)
        .then(|| path.parent()?.parent()?.parent().map(Path::to_path_buf))
        .flatten()
}

/// Whether macOS is running this from a randomised read-only copy.
///
/// **Gatekeeper path randomisation ("app translocation").** An app launched
/// from a quarantined location runs from `/private/var/folders/.../
/// AppTranslocation/<uuid>/d/Something.app`, which is read-only, so every write
/// below would fail with something unhelpful. Neither `mv` nor `NSFileManager`
/// clears it -- only moving the app in Finder does -- so the honest answer is
/// to say that, rather than to fail at a rename.
fn is_translocated(path: &Path) -> bool {
    path.components().any(|part| part.as_os_str() == "AppTranslocation")
}

/// Replace a whole `.app` bundle.
///
/// **The unit of replacement on macOS is the bundle, not the executable.** The
/// published payload is the `.app` zipped with `ditto -c -k --keepParent`, so
/// this expands it and swaps the directory. `rename(2)` works on directories,
/// and on the same filesystem it is atomic exactly as it is for a file.
///
/// # Why this is now attempted at all
///
/// `docs/AUTOUPDATE-PLAN.md` ruled macOS out on the grounds that "any edit
/// inside a signed bundle invalidates the signature, which on Apple Silicon is
/// SIGKILL at exec". That reasoning assumed a signed bundle. **This one is not
/// signed** -- `release.yml` runs no `codesign` -- so there is no signature to
/// invalidate, and the whole bundle is replaced rather than edited in place.
/// The arm64 ad-hoc signature the linker applies travels inside the payload,
/// because the payload IS the built bundle.
///
/// It is also, in one respect, better than what a user does by hand: our
/// download writes no `com.apple.quarantine`, so the replaced bundle does not
/// get the "is damaged and can't be opened" dialog that a browser download
/// earns. Quarantine is stripped from the staged copy anyway, best effort,
/// because `ditto` preserves extended attributes and the zip was built on a
/// runner.
///
/// `expand` is a parameter so the sequence -- refusals, swap, restore -- is
/// exercised on Linux, where `ditto` does not exist. **The syscalls differ; the
/// ordering does not, and the ordering is what can lose an install.**
#[cfg(any(target_os = "macos", test))]
fn swap_bundle(
    bundle: &Path,
    staged_zip: &Path,
    previous_sha256: &str,
    expand: &dyn Fn(&Path, &Path) -> Result<(), Refusal>,
) -> Result<(), Refusal> {
    if is_translocated(bundle) {
        return Err(Refusal::CannotReplace(
            "macOS is running this from a temporary read-only copy. Move BrokkrSculpt to \
             Applications in Finder -- dragging it there, not `mv` -- and open it again"
                .into(),
        ));
    }
    let parent = bundle
        .parent()
        .ok_or_else(|| Refusal::CannotReplace("the application has no folder".into()))?;
    directory_is_safe(parent)?;
    if !staged_zip.exists() {
        return Err(Refusal::Quarantined);
    }

    // Expanded beside the bundle, so the rename that follows cannot cross a
    // filesystem and degrade into a copy.
    let work = parent.join(format!(".brokkrsculpt-{}.expand", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)
        .map_err(|why| Refusal::CannotReplace(format!("could not stage the update ({why})")))?;
    let cleanup = |work: &Path| {
        let _ = std::fs::remove_dir_all(work);
    };
    if let Err(why) = expand(staged_zip, &work) {
        cleanup(&work);
        return Err(why);
    }
    let expanded = std::fs::read_dir(&work)
        .ok()
        .and_then(|entries| {
            entries.flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "app"))
        })
        .ok_or_else(|| {
            cleanup(&work);
            Refusal::CannotReplace("the download did not contain an application".into())
        })?;

    super::record_previous(previous_sha256);
    let old =
        parent.join(format!(".{}.old", bundle.file_name().unwrap_or_default().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&old);
    let note = parent.join(RECOVERY_NOTE);
    let _ = std::fs::write(
        &note,
        format!(
            "If BrokkrSculpt no longer opens after an update:\n\
             \n\
                 1. Delete   {}\n\
                 2. Rename   {}\n\
                    back to  {}\n\
             \n\
             The kept copy's name starts with a dot, so Finder hides it: press \
             Command-Shift-Period to show it.\n",
            bundle.display(),
            old.display(),
            bundle.display(),
        ),
    );

    // The only irreversible step, and the only window in which the application
    // is not at its own path. Restored immediately if the second rename fails.
    if let Err(why) = std::fs::rename(bundle, &old) {
        cleanup(&work);
        return Err(Refusal::CannotReplace(format!(
            "could not move the application aside ({why})"
        )));
    }
    if let Err(why) = std::fs::rename(&expanded, bundle) {
        let _ = std::fs::rename(&old, bundle);
        cleanup(&work);
        return Err(Refusal::CannotReplace(format!(
            "could not install the new application, so the old one was put back ({why})"
        )));
    }
    cleanup(&work);
    Ok(())
}

/// Expand a `ditto` zip. macOS only; `ditto` ships with the system.
///
/// `ditto -x -k` rather than `unzip`, matching how `release.yml` creates it:
/// `unzip` flattens resource forks and bundle structure, which is the whole
/// reason the archive is made with `ditto` in the first place.
#[cfg(target_os = "macos")]
fn ditto_expand(zip: &Path, into: &Path) -> Result<(), Refusal> {
    let outcome = std::process::Command::new("/usr/bin/ditto")
        .arg("-x")
        .arg("-k")
        .arg(zip)
        .arg(into)
        .output()
        .map_err(|why| Refusal::CannotReplace(format!("could not run ditto ({why})")))?;
    if !outcome.status.success() {
        return Err(Refusal::CannotReplace(format!(
            "the download could not be expanded ({})",
            String::from_utf8_lossy(&outcome.stderr).trim()
        )));
    }
    // Best effort: `ditto` preserves extended attributes, and the archive was
    // built on a runner. Our own download sets no quarantine, so this is
    // belt-and-braces rather than the load-bearing part.
    let _ = std::process::Command::new("/usr/bin/xattr")
        .args(["-dr", "com.apple.quarantine"])
        .arg(into)
        .output();
    Ok(())
}

/// Whether this executable lives inside a macOS `.app` bundle.
///
/// **A path test rather than a `cfg`, deliberately.** The thing that must not be
/// edited is the bundle, not the operating system -- and a `cfg` that silently
/// stopped matching would turn "macOS never self-replaces" into "macOS
/// self-replaces" with nothing objecting. Checking the shape of the path means
/// the refusal is reachable, and testable, on every platform.
///
/// The shape is `.../Something.app/Contents/MacOS/executable`, which is what
/// `release.yml` builds.
fn is_in_app_bundle(path: &Path) -> bool {
    let mut parts = path.components().rev().skip(1); // skip the executable
    let macos = parts.next().is_some_and(|part| part.as_os_str() == "MacOS");
    let contents = parts.next().is_some_and(|part| part.as_os_str() == "Contents");
    let bundle = parts
        .next()
        .is_some_and(|part| Path::new(part.as_os_str()).extension().is_some_and(|e| e == "app"));
    macos && contents && bundle
}

/// Whether the containing directory is one we may write into.
///
/// Two questions, and the second is the one people forget.
fn directory_is_safe(directory: &Path) -> Result<(), Refusal> {
    // **`Permissions::readonly()` is NOT asked about a directory**, and that is
    // not fussiness. On Windows it reports `FILE_ATTRIBUTE_READONLY`, which the
    // shell sets to mark a folder as customised and which Microsoft documents as
    // "not honored on directories" -- so it is true for a great many perfectly
    // writable folders. A Windows tester's `C:\Users\<name>\Downloads` was
    // refused by it on 2026-08-31 with "this looks like a system install", after
    // the download had already succeeded. The probe below is the ground truth on
    // every platform: it answers by writing, which is the thing actually being
    // asked, and it also covers ACLs, read-only mounts and a full filesystem
    // that no attribute reports.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        use std::os::unix::fs::PermissionsExt;
        // Fetched here rather than at the top: the ownership rules below are the
        // only thing that needs it, and on Windows the probe answers everything
        // -- including a directory that is not there, which `File::create` in it
        // reports perfectly well.
        let metadata = std::fs::metadata(directory).map_err(|why| {
            Refusal::CannotReplace(format!("could not read {} ({why})", directory.display()))
        })?;
        let mode = metadata.permissions().mode();
        // **Group- or world-writable is refused even though we could write.**
        // A bare writability test passes on a loose `/opt` or a shared
        // `/usr/local` -- and there, anyone in the group can put a file at the
        // name we are about to make executable. The rollback copy sits in the
        // same directory, so this also protects what we revert to.
        // **Except on macOS, where the directory in question is /Applications.**
        // The system ships it `root:admin 0775` -- group-writable and owned by
        // root -- so the two rules below refuse it, and a Mac would download 33
        // MB and only then say "/Applications belongs to another user". That is
        // not a safer outcome, it is no updates plus a late confusing message,
        // and it is reached by following the in-app instruction to move the app
        // to Applications in the first place.
        //
        // Skipping them there costs nothing real: anyone who can write
        // /Applications can replace the application whether or not this code
        // helps, so the refusal was never what stood between them and it. The
        // rules stay everywhere a per-user install directory is the norm, which
        // is where a loose /opt or a shared /usr/local actually is the hazard.
        // The writability probe below still runs on every platform.
        if !cfg!(target_os = "macos") && mode & 0o022 != 0 {
            return Err(Refusal::CannotReplace(format!(
                "{} is writable by other users, so replacing a binary there would not be safe",
                directory.display()
            )));
        }
        // Someone else's directory is not ours to write in, whatever the bits
        // say about us.
        let uid = unsafe { libc_getuid() };
        if !cfg!(target_os = "macos") && metadata.uid() != uid {
            return Err(Refusal::CannotReplace(format!(
                "{} belongs to another user",
                directory.display()
            )));
        }
    }
    // A writability probe rather than trusting the mode bits, which say nothing
    // about ACLs, read-only mounts or a full filesystem. Non-destructive: it
    // creates a uniquely named file and removes it.
    let probe = directory.join(format!(".brokkrsculpt-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        // The actionable message lives here now, because this is the check that
        // can actually tell. A user whose install directory really is read-only
        // needs to hear what to do instead, not an errno.
        Err(why) => Err(Refusal::CannotReplace(format!(
            "{} is not writable ({why}) -- if this is a system-wide install, update it the way \
             you installed it",
            directory.display()
        ))),
    }
}

// `getuid`, without taking `libc` as a direct dependency for one call.
//
// `libc` is in the lockfile only transitively (through `evdev`, `getrandom` and
// `errno`), which does not let us `use` it, and a direct dependency for a single
// argument-free syscall returning an integer is a poor trade. Same shape
// `raw_hid.rs` already uses for IOKit: an `extern "C"` block with no struct
// layout in it, layout being the part that is actually dangerous to hand-write.
#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "getuid"]
    fn libc_getuid() -> u32;
}

/// An exclusive lock over the whole apply, and over every read-modify-write of
/// `update.state`.
///
/// `std::fs::File::try_lock` -- stable since 1.89, inside this workspace's 1.90
/// MSRV, so **zero new crates**. `flock(LOCK_EX | LOCK_NB)` on Unix and
/// `LockFileEx` with `LOCKFILE_FAIL_IMMEDIATELY` on Windows. The kernel drops it
/// when the handle closes, including on process death, which is the whole reason
/// to prefer it to an `O_EXCL` lock file: that shape needs a staleness policy
/// and a PID check, and an updater killed mid-apply leaves one behind that
/// wedges every future update on the machine with nothing on screen to say why.
///
/// **Never waited on.** An instance that cannot take it says so and does
/// nothing, because the user is looking at a modal and a blocking wait behind
/// someone else's 33 MB download is a hung window.
pub struct Lock(std::fs::File);

impl Lock {
    pub fn take() -> Result<Self, Refusal> {
        let path = crate::paths::state_file("update.lock")
            .ok_or_else(|| Refusal::CannotReplace("there is nowhere to keep the lock".into()))?;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // `.read(true).write(true)`: the std docs are explicit that on Windows a
        // handle opened only for append cannot be locked.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|why| Refusal::CannotReplace(format!("could not open the lock ({why})")))?;
        match file.try_lock() {
            Ok(()) => Ok(Self(file)),
            Err(_) => Err(Refusal::AlreadyUpdating),
        }
    }

    /// Release before spawning the replacement.
    ///
    /// Not because the child inherits it -- `std::fs::File` is opened
    /// close-on-exec, so it does not -- but because the child races the parent's
    /// exit: it opens its own handle, and while the parent is still winding down
    /// that handle gets `WouldBlock`, under which rule the newly installed build
    /// would silently skip its own first state write.
    ///
    /// **The lock file is created once and never deleted.** Unlinking a file
    /// another process holds flocked lets the next process create a fresh inode
    /// and lock that instead, so a tidy-up sweep that removes it removes the
    /// mutual exclusion while appearing to work.
    pub fn release(self) {
        let _ = self.0.unlock();
    }
}

/// Put a verified file in place of the running binary.
///
/// `staged` must already have been verified -- this function checks nothing
/// about its contents, by design: mixing "is this the right file" with "how do I
/// install it" is how the install path ends up trusting the download.
///
/// The order is the whole design. Everything before the final `rename` is
/// non-destructive, and the `rename` is the only irreversible step.
pub fn install(target: &Target, staged: &Path, previous_sha256: &str) -> Result<(), Refusal> {
    if self_update_disabled() {
        return Err(Refusal::HandOverOnly);
    }
    // **A bundle is replaced whole or not at all.** The macOS arm below does
    // that. Anywhere else, a path that looks like a `.app` must not be edited
    // in place: `install_unix` would rewrite the executable inside it and leave
    // a half-swapped application. Refused rather than attempted, and the
    // refusal is reachable off macOS, which is what makes it testable.
    #[cfg(not(target_os = "macos"))]
    if is_in_app_bundle(&target.path) {
        return Err(Refusal::HandOverOnly);
    }
    // Then one arm per platform, each with a definite return. An earlier cut
    // wrote the macOS case as `|| cfg!(target_os = "macos")` on the line above
    // and let both arms below be compiled out -- which on macOS left the
    // function falling off its end with no value, and only CI said so, because
    // `ring` will not cross-build its C for darwin from a Linux host.
    #[cfg(target_os = "macos")]
    {
        let bundle = app_bundle_root(&target.path).ok_or_else(|| {
            Refusal::CannotReplace(
                "this copy is not inside an application bundle, so there is nothing to replace"
                    .into(),
            )
        })?;
        swap_bundle(&bundle, staged, previous_sha256, &ditto_expand)
    }
    #[cfg(windows)]
    {
        swap_windows(target, staged, previous_sha256, &exclusive_probe)
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    install_unix(target, staged, previous_sha256)
}

/// How long to wait for a real-time scanner to let go of a freshly written
/// executable, and how long each of the two rename budgets is.
///
/// **The long budget sits OUTSIDE the window where the app has no binary at its
/// own path.** McAfee documents a 45-second default on-access scan timeout, so
/// the wait for exclusive access gets 60 s -- but it happens *before* anything
/// moves. The two renames get seconds, because that window is the dangerous one.
#[cfg(any(windows, test))]
const EXCLUSIVE_BUDGET: std::time::Duration = std::time::Duration::from_secs(60);
#[cfg(any(windows, test))]
const ASIDE_BUDGET: std::time::Duration = std::time::Duration::from_secs(2);
#[cfg(any(windows, test))]
const INTO_PLACE_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// Whether the staged file can be opened for exclusive access yet.
///
/// On Windows, opening with no sharing flags is how you ask "has the antivirus
/// finished with this". Elsewhere this is trivially true, which is what lets the
/// sequence below be exercised on Linux.
#[cfg(any(windows, test))]
#[allow(clippy::unnecessary_wraps)]
fn exclusive_probe(path: &Path) -> Result<(), Refusal> {
    // A file that has VANISHED is quarantine, not a lock, and the two need
    // different answers -- so this is checked on every platform.
    if !path.exists() {
        return Err(Refusal::Quarantined);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // `share_mode(0)`: no other handle may be open. This is the actual
        // question -- not whether we can read it, but whether anyone else is
        // still holding it.
        std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(path)
            .map(|_| ())
            .map_err(|why| Refusal::CannotReplace(format!("the new build is still locked ({why})")))
    }
    #[cfg(not(windows))]
    {
        std::fs::File::open(path)
            .map(|_| ())
            .map_err(|why| Refusal::CannotReplace(format!("could not open the new build ({why})")))
    }
}

/// The Windows swap, restructured so the retry budget is outside the danger
/// window.
///
/// `probe` is a parameter so the whole sequence -- the wait, the abort paths and
/// the restore -- can be exercised on Linux, where the real exclusive-open
/// question does not exist. **The syscalls are Windows-only; the ordering is
/// not, and the ordering is the part that can lose someone's install.**
///
/// Never `MOVEFILE_COPY_ALLOWED`: Microsoft documents it as CopyFile plus
/// DeleteFile, which is exactly what fails against a locked running image.
/// Never `MOVEFILE_DELAY_UNTIL_REBOOT`: it needs administrators-group or
/// LocalSystem, and its return value only tells you the registry write worked.
#[cfg(any(windows, test))]
fn swap_windows(
    target: &Target,
    staged: &Path,
    previous_sha256: &str,
    probe: &dyn Fn(&Path) -> Result<(), Refusal>,
) -> Result<(), Refusal> {
    // 1. Wait for the scanner, with backoff, BEFORE anything moves. If this
    //    never succeeds the install is simply not attempted and nothing has
    //    changed.
    let deadline = std::time::Instant::now() + EXCLUSIVE_BUDGET;
    let mut wait = std::time::Duration::from_millis(50);
    loop {
        match probe(staged) {
            Ok(()) => break,
            // Quarantine is terminal: retrying cannot bring the file back, and
            // a self-rewriting `.exe` is a textbook Defender heuristic, so
            // deletion is at least as likely as a sharing violation.
            Err(Refusal::Quarantined) => return Err(Refusal::Quarantined),
            Err(why) => {
                if std::time::Instant::now() >= deadline {
                    return Err(why);
                }
                std::thread::sleep(wait);
                wait = (wait * 2).min(std::time::Duration::from_secs(2));
            }
        }
    }

    // 2. The recovery note, before the swap, so it is already there if the swap
    //    is what goes wrong. Names `.exe.old`, which is what step 3 creates --
    //    an earlier draft said `brokkrsculpt.old`, a file that never exists.
    let note = format!(
        "If BrokkrSculpt no longer starts after an update:\n\
         \n\
             1. Delete       {}\n\
             2. Rename       {}\n\
                back to      {}\n\
         \n\
         That puts back the build you were running before.\n\
         \n\
         This can happen when Windows declines to run a new unsigned build. It \
         is not a broken download: the update was checked against its signature \
         before it was installed.\n",
        target.path.display(),
        target.old().display(),
        target.path.display(),
    );
    let _ = std::fs::write(target.recovery_note(), note);
    super::record_previous(previous_sha256);

    // 3. Rename the running image aside. The kernel refuses to unlink a running
    //    image but permits renaming it, which is the whole reason this works.
    //    Short budget: on failure nothing has changed.
    let old = target.old();
    let _ = std::fs::remove_file(&old);
    retry(ASIDE_BUDGET, || std::fs::rename(&target.path, &old)).map_err(|why| {
        Refusal::CannotReplace(format!("could not move the running build aside ({why})"))
    })?;

    // 4. Rename the new one into place. **This is the only moment the app has no
    //    binary at its own path.** On failure the aside goes straight back.
    if let Err(why) = retry(INTO_PLACE_BUDGET, || std::fs::rename(staged, &target.path)) {
        let _ = std::fs::rename(&old, &target.path);
        return Err(Refusal::CannotReplace(format!(
            "could not install the new build, so the old one was put back ({why})"
        )));
    }

    // 5. **Hash the result rather than trusting the return code.** IBM
    //    documents antivirus-contended installs that report success and leave
    //    corrupted files behind, and this is the last moment we can tell.
    //
    //    Every failure from here on restores. An earlier draft used `?` on this
    //    hash, which returned early with the old build still parked aside and
    //    something unusable at the target path -- the precise state this whole
    //    sequence exists to avoid, reached by the error handling rather than by
    //    the error. A test that renamed a directory into place caught it.
    let restore = |why: String| -> Refusal {
        let _ = std::fs::remove_file(&target.path);
        let _ = std::fs::remove_dir_all(&target.path);
        let _ = std::fs::rename(&old, &target.path);
        Refusal::CannotReplace(why)
    };
    let installed = match sha256_of(&target.path) {
        Ok(digest) => digest,
        Err(why) => {
            return Err(restore(format!(
                "could not read back the installed build, so the old one was put back ({why})"
            )));
        }
    };
    let expected = sha256_of(staged).unwrap_or_default();
    if !expected.is_empty() && installed != expected {
        return Err(restore(
            "the installed file did not match what was verified, so the old build was put back"
                .into(),
        ));
    }

    // 6. The recovery note STAYS. It is the entire remedy for the one Windows
    //    failure nothing of ours can catch -- Smart App Control declining to
    //    execute the new binary, where none of our revert code ever runs.
    Ok(())
}

/// Retry a filesystem operation until it works or the budget runs out.
#[cfg(any(windows, test))]
fn retry<T>(
    budget: std::time::Duration,
    mut operation: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    let deadline = std::time::Instant::now() + budget;
    let mut wait = std::time::Duration::from_millis(20);
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(why) => {
                if std::time::Instant::now() >= deadline {
                    return Err(why);
                }
                std::thread::sleep(wait);
                wait = (wait * 2).min(std::time::Duration::from_millis(400));
            }
        }
    }
}

/// The Unix swap: hard-link aside, then one atomic rename.
#[cfg(all(not(windows), not(target_os = "macos")))]
fn install_unix(target: &Target, staged: &Path, previous_sha256: &str) -> Result<(), Refusal> {
    // A staged file that has vanished between verifying it and installing it is
    // worth its own answer on every platform: on Windows that is antivirus
    // quarantine, on Linux it is a tmp cleaner or a second instance, and in
    // both cases "the file is gone" beats a rename error nobody can act on.
    if !staged.exists() {
        return Err(Refusal::Quarantined);
    }
    // 1. Park the current binary under a second name. A hard link, so nothing
    //    is removed and there is no window in which the target is absent.
    //
    //    Link to a random name and then rename over `.old`, rather than linking
    //    straight to it: `link(2)` fails EEXIST against an existing name rather
    //    than replacing it, so now that `.old` persists between updates a bare
    //    link would fail on every update after the first -- on the one step
    //    whose entire job is making the next failure survivable. Unlinking
    //    `.old` first would compile and would open a window with no rollback
    //    copy at all; `rename` has none.
    let link_temp = target.directory.join(format!(
        ".{}.{}.link",
        target.path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    let _ = std::fs::remove_file(&link_temp);
    std::fs::hard_link(&target.path, &link_temp).map_err(|why| {
        Refusal::CannotReplace(format!("could not keep a copy of the current build ({why})"))
    })?;
    std::fs::rename(&link_temp, target.old()).map_err(|why| {
        let _ = std::fs::remove_file(&link_temp);
        Refusal::CannotReplace(format!("could not park the current build ({why})"))
    })?;

    // 2. The recovery note, written BEFORE the swap so it is already there if
    //    the swap is the thing that goes wrong.
    let note = format!(
        "If BrokkrSculpt no longer starts after an update:\n\
         \n\
             1. Delete       {}\n\
             2. Rename       {}\n\
                back to      {}\n\
         \n\
         That puts back the build you were running before. The file is hidden \
         because its name starts with a dot -- `ls -a` in a terminal, or Ctrl+H \
         in a file manager, will show it.\n\
         \n\
         This can happen when a new build needs a newer system library than you \
         have. It is not a broken download: the update was checked against its \
         signature before it was installed.\n",
        target.path.display(),
        target.old().display(),
        target.path.display(),
    );
    let _ = std::fs::write(target.recovery_note(), note);

    // 3. Record what we are reverting TO, so the revert can refuse to run
    //    whatever happens to be sitting at that name.
    super::record_previous(previous_sha256);

    // 4. The one irreversible step.
    std::fs::rename(staged, &target.path)
        .map_err(|why| Refusal::CannotReplace(format!("could not install the new build ({why})")))
}

/// Put the superseded binary back.
///
/// **The digest check is not ceremony.** Reverting means executing a file with a
/// predictable name in a directory, and a revert path that runs whatever is
/// sitting there is a code-execution path.
pub fn revert(target: &Target, expected_sha256: &str) -> Result<(), Refusal> {
    let old = target.old();
    let actual = sha256_of(&old)?;
    if actual != expected_sha256 {
        return Err(Refusal::CannotReplace(
            "the kept copy is not the build it should be, so it was left alone".into(),
        ));
    }
    std::fs::rename(&old, &target.path).map_err(|why| {
        Refusal::CannotReplace(format!("could not put the old build back ({why})"))
    })?;
    // The note described a recovery that has now happened.
    let _ = std::fs::remove_file(target.recovery_note());
    Ok(())
}

/// SHA-256 of a file on disk.
pub fn sha256_of(path: &Path) -> Result<String, Refusal> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|why| {
        Refusal::CannotReplace(format!("could not read {} ({why})", path.display()))
    })?;
    let mut context = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|why| {
            Refusal::CannotReplace(format!("could not read {} ({why})", path.display()))
        })?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
    }
    Ok(context.finish().as_ref().iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Remove `.part` and `.link` files left by an attempt that died.
///
/// Both names, not just `.part`: a kill between the link and the rename leaves a
/// second name on the superseded inode that nothing ever reclaims, which is one
/// leaked 33 MB binary per crashed attempt.
pub fn sweep(directory: &Path) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if is_sweepable(&name) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Stage a file beside the target, ready to be renamed over it.
///
/// **In the destination directory, never `/tmp`.** A cross-filesystem rename
/// degrades to a copy, and copy-over-a-running-image is precisely the operation
/// that fails.
///
/// Created `0600` and `O_EXCL`, then chmod'd to `0755` **after** the caller has
/// verified it. `write_private` in `account.rs` sets `0600` at creation to
/// *narrow* exposure; creating at `0755` and then filling it with unverified
/// network bytes inverts the point of that rule, and `~/.local/bin` is on `PATH`
/// on most distributions.
/// Whether a name in the install directory is ours to remove at startup.
///
/// Three kinds, and the third is the one that was missing.
///
/// `.part` and `.link` are half-finished staging left by an attempt that died;
/// a `.link` in particular is a second name on the superseded inode, so leaving
/// one leaks a whole binary per crashed attempt.
///
/// **A COMPLETED payload is also dead.** `brokkrsculpt-1024-windows-x86_64.exe`
/// is what a finished download leaves beside the binary, and it is only ever
/// useful to the session that fetched it: the path is held in memory and is not
/// persisted, so after a restart nothing can find it and the next check
/// downloads again. Until now nothing removed it, so a user who downloaded an
/// update and did not install it kept 19-33 MB per attempt for ever. A Windows
/// tester whose install click did nothing has exactly one of these.
///
/// The match is deliberately narrow: `brokkrsculpt-` followed by a DIGIT. The
/// running binary is `brokkrsculpt` or `brokkrsculpt.exe` and never has an
/// ordinal in it, and the kept copy starts with a dot, so neither can be caught
/// by this.
fn is_sweepable(name: &str) -> bool {
    if name.starts_with(".brokkrsculpt") && (name.ends_with(".part") || name.ends_with(".link")) {
        return true;
    }
    name.strip_prefix("brokkrsculpt-")
        .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
}

/// Stage in a named directory. The directory decides whether the rename that
/// follows stays on one filesystem, which is the whole point -- see the caller.
pub fn stage_in(directory: &Path, name: &str) -> Result<(PathBuf, std::fs::File), Refusal> {
    let path = directory.join(format!(".{name}.part"));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
        // Never follow a symlink someone else planted at this name.
        options.custom_flags(libc_o_nofollow());
    }
    let file = options
        .open(&path)
        .map_err(|why| Refusal::CannotReplace(format!("could not stage the update ({why})")))?;
    Ok((path, file))
}

/// `O_NOFOLLOW`, which is the same value on every Linux ABI we build for.
#[cfg(unix)]
const fn libc_o_nofollow() -> i32 {
    // Linux: 0o400000. Named here rather than pulled from `libc` for the same
    // reason `getuid` is declared by hand -- and it is a constant rather than a
    // struct, so there is no layout to get wrong.
    #[cfg(target_os = "linux")]
    {
        0o400000
    }
    #[cfg(target_os = "macos")]
    {
        0x0100
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// Make a staged file executable, then flush it.
///
/// Both after verification, in that order: a file that is executable before its
/// digest is checked is a file someone else on the machine could have run.
pub fn finish_staging(file: &std::fs::File) -> Result<(), Refusal> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o755)).map_err(|why| {
            Refusal::CannotReplace(format!("could not make it executable ({why})"))
        })?;
    }
    file.sync_all()
        .map_err(|why| Refusal::CannotReplace(format!("could not flush the update ({why})")))
}

/// Write the marker the next launch reads to decide whether the update took.
///
/// **One file per install, keyed by a hash of the executable path in the
/// FILENAME.** Two installs on one machine -- one system, one under `~/.local`
/// -- share a state directory, and one shared marker with a path field inside
/// means each install must read the other's file, compare paths and decide not
/// to touch it. That comparison is a thing someone can get wrong, and getting it
/// wrong means one install reverting itself on the other's evidence. A filename
/// makes the collision impossible rather than merely checked.
pub fn marker_path(target: &Target) -> Option<PathBuf> {
    let digest =
        ring::digest::digest(&ring::digest::SHA256, target.path.as_os_str().as_encoded_bytes());
    let key: String = digest.as_ref()[..8].iter().map(|byte| format!("{byte:02x}")).collect();
    crate::paths::state_file(&format!("update-pending-{key}"))
}

/// What the marker carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pending {
    /// The ordinal that was just installed.
    pub build: u64,
    /// How many launches have tried it. `0` means "written, never launched".
    pub attempt: u32,
    /// A document to reopen, if the user had one. **The channel that survives a
    /// restart**, used instead of argv because a rollback can spawn an OLDER
    /// binary, and an older `main.rs` ignores an argument it has never heard of
    /// and comes up empty with no error at all.
    pub resume: Option<PathBuf>,
    /// The path this marker belongs to, for a human reading the directory. The
    /// code identifies the install by the filename.
    pub path: PathBuf,
}

/// Render a marker. Split from the write so it can be tested without a path.
pub fn marker_text(pending: &Pending) -> String {
    let mut text = format!(
        "build = {}\nattempt = {}\npath = {}\n",
        pending.build,
        pending.attempt,
        pending.path.display()
    );
    if let Some(resume) = &pending.resume {
        text.push_str(&format!("resume = {}\n", resume.display()));
    }
    text
}

/// Parse a marker. Unknown keys are ignored, for the same reason the manifest
/// ignores them and with the same rollback behind it: **the build reading this
/// file can be older than the build that wrote it.**
pub fn marker_from(text: &str) -> Option<Pending> {
    let mut build = None;
    let mut attempt = None;
    let mut resume = None;
    let mut path = None;
    for (key, value) in crate::paths::entries(text) {
        match key {
            "build" => build = value.parse().ok(),
            "attempt" => attempt = value.parse().ok(),
            "resume" => resume = Some(PathBuf::from(value)),
            "path" => path = Some(PathBuf::from(value)),
            _ => {}
        }
    }
    Some(Pending { build: build?, attempt: attempt?, resume, path: path.unwrap_or_default() })
}

/// Write the marker, temp-and-rename with a **random** temp suffix.
///
/// Random rather than `account.rs`'s fixed `.json.tmp`: one fixed temp name
/// shared by two instances is the race the rename was supposed to remove.
pub fn write_marker(path: &Path, pending: &Pending) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp{}", std::process::id()));
    let mut file = std::fs::File::create(&temporary)?;
    file.write_all(marker_text(pending).as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&temporary, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temporary);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory that removes itself when the test ends.
    ///
    /// **Clearing on entry is not enough, and that is the whole reason this
    /// type exists.** The name carries the process id and the thread id, which
    /// it must -- the suite runs in parallel, and two threads sharing a
    /// directory would delete each other's fixtures. But that also means a
    /// later run never revisits an earlier run's names, so entry-clearing only
    /// ever protects against a rerun that lands on the same pid AND the same
    /// thread id. In practice nothing was ever cleaned: 1353 directories and
    /// 11 MB had built up in `/tmp` by 2026-08-31 before anyone looked. No test
    /// failed, which is why it survived so long.
    ///
    /// Entry-clearing stays as well, for the case Drop cannot cover: a hard
    /// kill mid-test, where no destructor runs.
    ///
    /// `Deref` to `Path` so the call sites read exactly as they did when this
    /// was a bare `PathBuf` -- `dir.join(..)` and `&dir` both still work.
    ///
    /// A test that makes its directory unreadable must put the mode back
    /// itself; removal needs write permission on the directory to clear its
    /// contents, and Drop deliberately does not fight for it. See
    /// `a_read_only_directory_is_refused_with_something_a_user_can_act_on`,
    /// which restores before asserting for this reason.
    struct Scratch(PathBuf);

    impl std::ops::Deref for Scratch {
        type Target = Path;

        fn deref(&self) -> &Path {
            &self.0
        }
    }

    /// `AsRef` as well as `Deref`, because deref coercion does not reach a
    /// generic `impl AsRef<Path>` parameter -- `std::fs::set_permissions(&dir,
    /// ..)` needs this one, `dir.join(..)` needs the other. Both, so no call
    /// site has to know which kind of function it is calling.
    impl AsRef<Path> for Scratch {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch(name: &str) -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "brokkr-apply-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp is writable");
        Scratch(path)
    }

    /// The cleanup itself, pinned. Without this the leak is invisible again the
    /// moment someone returns a bare `PathBuf` from `scratch`.
    #[test]
    fn a_scratch_directory_removes_itself_when_the_test_that_made_it_ends() {
        let path = {
            let dir = scratch("selfclean");
            std::fs::write(dir.join("a file"), b"contents").expect("writable");
            assert!(dir.exists(), "it must exist while it is in scope");
            dir.to_path_buf()
        };
        assert!(!path.exists(), "{} must be gone once dropped", path.display());
    }

    fn fake_target(dir: &Path) -> Target {
        let path = dir.join("brokkrsculpt");
        std::fs::write(&path, b"the build that is running").expect("writable");
        Target { path, directory: dir.to_path_buf() }
    }

    /// The invariant the whole module rests on: after a swap, `current_exe`
    /// no longer names the target, so the path must have been cached.
    #[test]
    fn the_target_path_is_resolved_once_and_survives_its_own_replacement() {
        let dir = scratch("cached");
        let target = fake_target(&dir);
        let before = target.path.clone();

        // Replace the file the way `install` does.
        let replacement = dir.join("replacement");
        std::fs::write(&replacement, b"the new build").expect("writable");
        std::fs::rename(&replacement, &target.path).expect("rename works");

        // The cached path still names the right thing, and the bytes moved.
        assert_eq!(target.path, before);
        assert_eq!(std::fs::read(&target.path).unwrap(), b"the new build");
    }

    /// A hard link is a second NAME for one inode: nothing is removed, so there
    /// is no window in which the target is absent.
    // macOS refuses the swap by design -- `install` returns `HandOverOnly`
    // there -- so a test that expects it to succeed is asking the wrong
    // question on that platform. The refusal has its own test, which does
    // run everywhere: `a_bundled_install_is_refused_the_swap_on_every_platform`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn parking_the_old_build_never_leaves_the_target_missing() {
        let dir = scratch("park");
        let target = fake_target(&dir);
        let original = std::fs::read(&target.path).unwrap();
        let staged = dir.join("staged");
        std::fs::write(&staged, b"the new build").expect("writable");

        install(&target, &staged, "irrelevant-for-this-test").expect("install works");

        // Both names exist and hold what they should.
        assert!(target.path.exists(), "the target must never be absent");
        assert_eq!(std::fs::read(&target.path).unwrap(), b"the new build");
        assert_eq!(std::fs::read(target.old()).unwrap(), original, ".old is the superseded build");
        assert!(target.recovery_note().exists(), "the recovery note must be written");
    }

    /// `link(2)` fails EEXIST rather than clobbering, so a second update would
    /// break on the one step whose job is making the next failure survivable.
    // macOS refuses the swap by design -- `install` returns `HandOverOnly`
    // there -- so a test that expects it to succeed is asking the wrong
    // question on that platform. The refusal has its own test, which does
    // run everywhere: `a_bundled_install_is_refused_the_swap_on_every_platform`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_second_update_in_one_install_still_parks_the_build_it_replaces() {
        let dir = scratch("twice");
        let target = fake_target(&dir);

        for (n, bytes) in [(1u8, &b"build two"[..]), (2, &b"build three"[..])] {
            let staged = dir.join(format!("staged{n}"));
            std::fs::write(&staged, bytes).expect("writable");
            install(&target, &staged, "x")
                .unwrap_or_else(|why| panic!("install {n} failed: {why:?}"));
            assert_eq!(std::fs::read(&target.path).unwrap(), bytes);
        }
        // `.old` is now build two, not build one: it tracks the build most
        // recently replaced.
        assert_eq!(std::fs::read(target.old()).unwrap(), b"build two");
    }

    /// A revert must refuse to run whatever happens to be sitting at `.old`.
    // macOS refuses the swap by design -- `install` returns `HandOverOnly`
    // there -- so a test that expects it to succeed is asking the wrong
    // question on that platform. The refusal has its own test, which does
    // run everywhere: `a_bundled_install_is_refused_the_swap_on_every_platform`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_revert_refuses_a_kept_copy_whose_digest_does_not_match() {
        let dir = scratch("revert-digest");
        let target = fake_target(&dir);
        let original_digest = sha256_of(&target.path).unwrap();
        let staged = dir.join("staged");
        std::fs::write(&staged, b"the new build").expect("writable");
        install(&target, &staged, &original_digest).expect("install works");

        // Someone swaps the parked copy for something else.
        std::fs::write(target.old(), b"not the build you kept").expect("writable");
        let refusal = revert(&target, &original_digest).expect_err("the digest must be checked");
        assert!(matches!(refusal, Refusal::CannotReplace(_)), "got {refusal:?}");
        // And the running binary was left alone.
        assert_eq!(std::fs::read(&target.path).unwrap(), b"the new build");
    }

    // macOS refuses the swap by design -- `install` returns `HandOverOnly`
    // there -- so a test that expects it to succeed is asking the wrong
    // question on that platform. The refusal has its own test, which does
    // run everywhere: `a_bundled_install_is_refused_the_swap_on_every_platform`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_revert_with_a_matching_digest_puts_the_old_build_back() {
        let dir = scratch("revert-ok");
        let target = fake_target(&dir);
        let original = std::fs::read(&target.path).unwrap();
        let original_digest = sha256_of(&target.path).unwrap();
        let staged = dir.join("staged");
        std::fs::write(&staged, b"the new build").expect("writable");
        install(&target, &staged, &original_digest).expect("install works");

        revert(&target, &original_digest).expect("the digest matches");
        assert_eq!(std::fs::read(&target.path).unwrap(), original);
        assert!(!target.recovery_note().exists(), "the note describes a recovery that happened");
    }

    #[test]
    fn staging_refuses_to_reuse_a_name_and_lands_beside_the_target() {
        let dir = scratch("stage");
        let target = fake_target(&dir);
        let (path, file) = stage_in(&target.directory, "brokkrsculpt.aaa").expect("staging works");
        assert_eq!(path.parent().unwrap(), target.directory, "must be the SAME filesystem");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = file.metadata().unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "unverified bytes must not be readable by others");
        }
        // O_EXCL: the same name a second time is a refusal, not a truncation.
        assert!(
            stage_in(&target.directory, "brokkrsculpt.aaa").is_err(),
            "O_EXCL must refuse an existing name"
        );
        drop(file);
    }

    #[test]
    fn finish_staging_makes_it_executable_only_after_it_is_written() {
        let dir = scratch("chmod");
        let target = fake_target(&dir);
        let (staged_path, mut file) =
            stage_in(&target.directory, "brokkrsculpt.bbb").expect("staging works");
        file.write_all(b"verified bytes").expect("writable");
        finish_staging(&file).expect("chmod and flush work");
        // The mode assertion is Unix-only; Windows has no mode, which is the
        // same reason `write_private` in account.rs sets one only there.
        assert!(staged_path.exists(), "the staged file must survive being finished");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&staged_path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o755);
        }
    }

    /// A kill between the link and the rename leaves a name on the superseded
    /// inode. Without the sweep that is one leaked 33 MB binary per attempt.
    #[test]
    fn the_sweep_takes_staging_and_dead_payloads_and_leaves_everything_else() {
        let dir = scratch("sweep");
        for name in [
            ".brokkrsculpt.1.part",
            ".brokkrsculpt.2.link",
            ".brokkrsculpt.old",
            // A finished download nobody installed. Dead on the next launch,
            // because the path to it is held in memory and never persisted.
            "brokkrsculpt-1024-windows-x86_64.exe",
            "brokkrsculpt-1024-linux-x86_64",
            "brokkrsculpt-1024-macos-arm64.zip",
        ] {
            std::fs::write(dir.join(name), b"x").expect("writable");
        }
        std::fs::write(dir.join("brokkrsculpt"), b"x").expect("writable");
        std::fs::write(dir.join("brokkrsculpt.exe"), b"x").expect("writable");
        sweep(&dir);

        assert!(!dir.join(".brokkrsculpt.1.part").exists(), ".part must be swept");
        assert!(!dir.join(".brokkrsculpt.2.link").exists(), ".link must be swept too");
        for dead in [
            "brokkrsculpt-1024-windows-x86_64.exe",
            "brokkrsculpt-1024-linux-x86_64",
            "brokkrsculpt-1024-macos-arm64.zip",
        ] {
            assert!(!dir.join(dead).exists(), "{dead} is a dead download and must be swept");
        }
        // **The three that must never be touched.**
        assert!(dir.join(".brokkrsculpt.old").exists(), "the rollback copy must survive");
        assert!(dir.join("brokkrsculpt").exists(), "the binary must survive");
        assert!(dir.join("brokkrsculpt.exe").exists(), "and its Windows name");
    }

    /// The matcher decides whether a file is deleted from the directory holding
    /// the application, so its edges are worth pinning individually.
    #[test]
    fn only_an_ordinal_named_payload_is_treated_as_a_dead_download() {
        for dead in [
            "brokkrsculpt-1024-linux-x86_64",
            "brokkrsculpt-1-windows-x86_64.exe",
            "brokkrsculpt-999999-macos-arm64.zip",
            ".brokkrsculpt.x.part",
            ".brokkrsculpt.x.link",
        ] {
            assert!(is_sweepable(dead), "{dead} should be swept");
        }
        for keep in [
            "brokkrsculpt",
            "brokkrsculpt.exe",
            ".brokkrsculpt.old",
            ".brokkrsculpt.exe.old",
            "RECOVER-BROKKRSCULPT.txt",
            // No ordinal: not something this application produced.
            "brokkrsculpt-backup",
            "brokkrsculpt-old.exe",
            "my-brokkrsculpt-1024-linux-x86_64",
        ] {
            assert!(!is_sweepable(keep), "{keep} must NOT be swept");
        }
    }

    #[test]
    fn a_development_build_is_refused_before_anything_else_is_checked() {
        let dir = scratch("gates");
        let target = fake_target(&dir);
        assert_eq!(gates(&target, None, "abc1234"), Err(Refusal::NotAReleaseBuild));
        assert_eq!(gates(&target, Some(1005), "unknown"), Err(Refusal::NotAReleaseBuild));
        assert_eq!(gates(&target, Some(1005), "abc1234-dirty"), Err(Refusal::NotAReleaseBuild));
        // A stamped build in a normal directory passes.
        assert_eq!(gates(&target, Some(1005), "abc1234"), Ok(()));
    }

    /// A bare writability test passes on a loose `/opt`, where anyone in the
    /// group could put a file at the name we are about to make executable.
    #[cfg(unix)]
    #[test]
    fn a_group_or_world_writable_directory_is_refused_even_though_we_could_write_it() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("loose");
        let target = fake_target(&dir);
        // `directory_is_safe` directly rather than through `gates`: on macOS the
        // platform refusal comes first and this check is never reached, so
        // going through `gates` would assert nothing there. The unit is the
        // same on every platform, and that is what is being tested.
        for mode in [0o777, 0o775, 0o757] {
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode))
                .expect("chmod works");
            let outcome = directory_is_safe(&dir);
            if cfg!(target_os = "macos") {
                // **macOS deliberately does not apply this rule**, because the
                // directory it would refuse is `/Applications`, which the system
                // ships `root:admin 0775`. Refusing it means never updating, and
                // it protects nothing: anyone who can write there can replace
                // the application without our help. Asserted rather than cfg'd
                // away, so the exemption stays visible and deliberate.
                assert_eq!(outcome, Ok(()), "macOS accepts mode {mode:o} by design");
            } else {
                let refusal = outcome.expect_err("a loose directory must be refused");
                assert!(
                    matches!(refusal, Refusal::CannotReplace(_)),
                    "mode {mode:o} gave {refusal:?}"
                );
            }
        }
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        assert_eq!(directory_is_safe(&dir), Ok(()), "0755 is fine");
        assert_eq!(gates(&target, Some(1005), "abc1234"), Ok(()), "and so is the whole gate");
    }

    #[test]
    fn a_binary_under_a_target_directory_is_refused_however_it_is_stamped() {
        let dir = scratch("targetdir");
        let nested = dir.join("target").join("release");
        std::fs::create_dir_all(&nested).expect("writable");
        let target = fake_target(&nested);
        assert_eq!(gates(&target, Some(1005), "abc1234"), Err(Refusal::NotAReleaseBuild));
    }

    /// Two installs share a state directory. The key is in the FILENAME so the
    /// collision is impossible rather than merely checked.
    #[test]
    fn two_installs_get_different_marker_names() {
        let a = Target {
            path: PathBuf::from("/usr/bin/brokkrsculpt"),
            directory: PathBuf::from("/usr/bin"),
        };
        let b = Target {
            path: PathBuf::from("/home/someone/.local/bin/brokkrsculpt"),
            directory: PathBuf::from("/home/someone/.local/bin"),
        };
        let (Some(one), Some(two)) = (marker_path(&a), marker_path(&b)) else {
            return; // no HOME in this environment; paths.rs returns None
        };
        assert_ne!(one, two, "two installs must not share a marker");
        assert_ne!(one.file_name(), two.file_name());
    }

    #[test]
    fn a_marker_survives_a_round_trip_and_tolerates_a_key_it_does_not_know() {
        let pending = Pending {
            build: 1018,
            attempt: 1,
            resume: Some(PathBuf::from("/home/someone/Face.brokkr")),
            path: PathBuf::from("/usr/bin/brokkrsculpt"),
        };
        let text = marker_text(&pending);
        assert_eq!(marker_from(&text), Some(pending.clone()));

        // A newer build wrote a key this one has never heard of. Ignored, for
        // the same reason the manifest ignores unknown keys: a rollback can
        // leave an OLDER binary reading a NEWER build's marker.
        let extended = format!("{text}something_from_2027 = yes\n");
        assert_eq!(marker_from(&extended), Some(pending));

        // A marker missing what it needs is not half-read.
        assert_eq!(marker_from("attempt = 1\n"), None);
        assert_eq!(marker_from(""), None);
    }

    #[test]
    fn a_marker_without_a_document_round_trips_as_none() {
        let pending = Pending {
            build: 1018,
            attempt: 0,
            resume: None,
            path: PathBuf::from("/usr/bin/brokkrsculpt"),
        };
        let parsed = marker_from(&marker_text(&pending)).expect("round trip");
        assert_eq!(parsed.resume, None);
        assert_eq!(parsed.build, 1018);
    }

    /// A read-only directory is the `.deb`, `.rpm` and `/usr/local` case, and
    /// one test covers all of them with no path list and no attempt at
    /// elevation.
    #[cfg(unix)]
    #[test]
    fn a_read_only_directory_is_refused_with_something_a_user_can_act_on() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("readonly");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).expect("chmod");
        // Directly, for the same reason as the test above.
        let refusal = directory_is_safe(&dir).expect_err("read-only must refuse");
        let said = refusal.to_string();
        // Restore before asserting, so a failure does not leave an undeletable
        // directory behind for the next run.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        assert!(
            said.contains("install") || said.contains("writable"),
            "a refusal must tell the user what to do instead, got: {said}"
        );
    }

    /// The whole cycle a user would live through: install, the build does not
    /// start, the next launch puts back what they had.
    // macOS refuses the swap by design -- `install` returns `HandOverOnly`
    // there -- so a test that expects it to succeed is asking the wrong
    // question on that platform. The refusal has its own test, which does
    // run everywhere: `a_bundled_install_is_refused_the_swap_on_every_platform`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_build_that_will_not_start_is_reverted_with_the_document_still_named() {
        let dir = scratch("cycle");
        let target = fake_target(&dir);
        let good = std::fs::read(&target.path).unwrap();
        let good_digest = sha256_of(&target.path).unwrap();

        // Install a build that (pretend) does not start.
        let staged = dir.join("staged");
        std::fs::write(&staged, b"a build that will not start").expect("writable");
        install(&target, &staged, &good_digest).expect("install works");
        assert_eq!(std::fs::read(&target.path).unwrap(), b"a build that will not start");

        // The marker the failed build would have left, having never drawn.
        let marker = dir.join("update-pending-test");
        let pending = Pending {
            build: 1018,
            attempt: 1,
            resume: Some(dir.join("Face.brokkr")),
            path: target.path.clone(),
        };
        write_marker(&marker, &pending).expect("writable");

        // The next launch reads attempt >= 1 and reverts.
        let read_back = marker_from(&std::fs::read_to_string(&marker).unwrap()).unwrap();
        assert_eq!(read_back.attempt, 1, "the failed launch recorded its attempt");
        revert(&target, &good_digest).expect("the digest matches, so the revert runs");

        assert_eq!(std::fs::read(&target.path).unwrap(), good, "the working build is back");
        assert!(!target.old().exists(), "the revert consumes the kept copy");
        // And the document survives the round trip, so the user is not dumped
        // into an empty window after a failed update.
        assert_eq!(read_back.resume, Some(dir.join("Face.brokkr")));
    }

    /// A full disk must fail on the download, non-destructively, and never
    /// after the swap. Exercised through the writer rather than by filling a
    /// real filesystem, which needs root: what is being tested is that the
    /// error propagates rather than being swallowed.
    #[test]
    fn a_write_that_fails_partway_is_reported_and_leaves_the_binary_alone() {
        struct FullDisk(usize);
        impl Write for FullDisk {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if self.0 == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::StorageFull,
                        "No space left on device",
                    ));
                }
                let n = buf.len().min(self.0);
                self.0 -= n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let dir = scratch("enospc");
        let target = fake_target(&dir);
        let before = std::fs::read(&target.path).unwrap();

        let bytes = vec![7u8; 200_000];
        let digest = ring::digest::digest(&ring::digest::SHA256, &bytes);
        let payload = super::super::Payload {
            name: "brokkrsculpt-1018-linux-x86_64".to_string(),
            size: bytes.len() as u64,
            sha256: digest.as_ref().iter().map(|b| format!("{b:02x}")).collect(),
        };
        let mut full = FullDisk(1000);
        let refusal = super::super::check_payload(&bytes[..], &payload, &mut full)
            .expect_err("a full disk must be reported");
        assert!(
            refusal.to_string().contains("space") || matches!(refusal, Refusal::Unreachable(_)),
            "got {refusal:?}"
        );
        // Nothing was swapped: the failure is entirely before the rename.
        assert_eq!(std::fs::read(&target.path).unwrap(), before);
    }

    // --- the Windows sequence, exercised on Linux through its probe seam ---
    //
    // The syscalls are Windows-only. The ORDERING is not, and the ordering is
    // the part that can lose someone's install: which step is allowed a long
    // budget, which window has no binary at the target path, and what happens on
    // each failure. All of that is testable here, and none of it is proved by a
    // green CI run on a Windows runner, which is why these exist.

    #[test]
    fn the_windows_swap_installs_and_keeps_the_recovery_note() {
        let dir = scratch("win-ok");
        let target = fake_target(&dir);
        let old_bytes = std::fs::read(&target.path).unwrap();
        let staged = dir.join("staged.exe");
        std::fs::write(&staged, b"the new build").expect("writable");

        swap_windows(&target, &staged, "digest", &|_| Ok(())).expect("the swap works");

        assert_eq!(std::fs::read(&target.path).unwrap(), b"the new build");
        assert_eq!(std::fs::read(target.old()).unwrap(), old_bytes, "the old build is parked");
        // **Kept, not deleted.** It is the entire remedy for a build Windows
        // declines to execute, where none of our revert code ever runs.
        assert!(target.recovery_note().exists(), "the recovery note must survive success");
    }

    /// The note has to name the file the swap actually creates. An earlier
    /// draft said `brokkrsculpt.old` while step 3 produced `brokkrsculpt.exe.old`,
    /// sending the one user who needs it looking for a name that is not there.
    #[test]
    fn the_recovery_note_names_the_file_that_actually_exists() {
        let dir = scratch("win-note");
        let target = fake_target(&dir);
        let staged = dir.join("staged.exe");
        std::fs::write(&staged, b"new").expect("writable");
        swap_windows(&target, &staged, "digest", &|_| Ok(())).expect("swap works");

        let note = std::fs::read_to_string(target.recovery_note()).unwrap();
        let parked = target.old();
        assert!(parked.exists(), "the parked file must exist");
        assert!(
            note.contains(&parked.display().to_string()),
            "the note names {:?} but the file on disk is {:?}",
            note,
            parked
        );
    }

    /// A scanner holding the file briefly must be waited out, and the wait must
    /// happen BEFORE anything moves.
    #[test]
    fn a_scanner_holding_the_file_briefly_is_waited_out_before_anything_moves() {
        let dir = scratch("win-av");
        let target = fake_target(&dir);
        let staged = dir.join("staged.exe");
        std::fs::write(&staged, b"the new build").expect("writable");

        let tries = std::cell::Cell::new(0);
        let probe = |_: &Path| {
            tries.set(tries.get() + 1);
            if tries.get() < 3 {
                Err(Refusal::CannotReplace("still locked".into()))
            } else {
                Ok(())
            }
        };
        swap_windows(&target, &staged, "digest", &probe).expect("it should wait and then succeed");
        assert_eq!(tries.get(), 3, "it must retry rather than give up on the first refusal");
        assert_eq!(std::fs::read(&target.path).unwrap(), b"the new build");
    }

    /// Quarantine is terminal and must be reported as itself. Retrying cannot
    /// bring a deleted file back, and the user needs to hear "antivirus", not
    /// "still locked".
    #[test]
    fn a_quarantined_payload_stops_immediately_and_changes_nothing() {
        let dir = scratch("win-quarantine");
        let target = fake_target(&dir);
        let before = std::fs::read(&target.path).unwrap();
        let staged = dir.join("staged.exe");
        // Never written: the scanner took it.

        let started = std::time::Instant::now();
        let refusal = swap_windows(&target, &staged, "digest", &exclusive_probe)
            .expect_err("a vanished payload must refuse");
        assert_eq!(refusal, Refusal::Quarantined);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "quarantine must not burn the 60 second budget"
        );
        assert_eq!(std::fs::read(&target.path).unwrap(), before, "nothing may have moved");
        assert!(!target.old().exists(), "nothing may have been parked");
    }

    /// The one dangerous window: if the second rename fails, the app has no
    /// binary at its own path. The aside must go straight back.
    #[test]
    fn a_failed_second_rename_puts_the_running_build_straight_back() {
        let dir = scratch("win-restore");
        let target = fake_target(&dir);
        let before = std::fs::read(&target.path).unwrap();
        // A DIRECTORY at the staged path: `rename` of it onto the target file
        // fails, which is what drives the restore path.
        let staged = dir.join("staged.exe");
        std::fs::create_dir_all(staged.join("blocker")).expect("writable");

        let refusal = swap_windows(&target, &staged, "digest", &|_| Ok(()))
            .expect_err("a directory landing at the target must be caught");
        assert!(matches!(refusal, Refusal::CannotReplace(_)), "got {refusal:?}");
        // **The install survived intact.** This is the whole point: whatever
        // goes wrong after the aside, the user ends up with a working binary at
        // its own path rather than a hole where one used to be.
        assert!(target.path.is_file(), "the binary must be a file again, not a hole");
        assert_eq!(std::fs::read(&target.path).unwrap(), before);
        assert!(!target.old().exists(), "the restore consumes the aside");
    }

    /// Antivirus-contended installs are documented to report success and leave
    /// a corrupted file, so the result is hashed rather than trusted.
    #[test]
    fn the_installed_file_is_hashed_rather_than_trusted() {
        let dir = scratch("win-hash");
        let target = fake_target(&dir);
        let staged = dir.join("staged.exe");
        std::fs::write(&staged, b"the new build").expect("writable");
        swap_windows(&target, &staged, "digest", &|_| Ok(())).expect("swap works");
        // The post-swap hash is taken from the file that landed; if the rename
        // had silently produced something else this is where it would be caught.
        assert_eq!(sha256_of(&target.path).unwrap().len(), 64);
    }

    #[test]
    fn the_retry_helper_gives_up_rather_than_spinning_for_ever() {
        let started = std::time::Instant::now();
        let outcome: std::io::Result<()> = retry(std::time::Duration::from_millis(120), || {
            Err(std::io::Error::other("never works"))
        });
        assert!(outcome.is_err());
        assert!(started.elapsed() < std::time::Duration::from_secs(3), "it must bound itself");
    }

    /// Self-replacement ships ON; the escape hatch is opt-out and off by
    /// default. Pinned because the two are easy to invert by accident, and
    /// inverting them silently disables updates for everyone.
    ///
    /// Asserted without touching the environment: `set_var` is unsafe on the
    /// 2024 edition and process-wide, and this suite runs in parallel.
    #[test]
    fn self_replacement_is_on_by_default_with_an_escape_hatch_that_is_not() {
        assert!(!self_update_disabled(), "the escape hatch must be off unless asked for");
        assert_eq!(NO_SELF_UPDATE, "BROKKR_NO_SELF_UPDATE");
    }

    /// Build a fake `.app` tree and a "zip" whose expansion is stubbed.
    fn fake_bundle(dir: &Path, marker: &[u8]) -> PathBuf {
        let bundle = dir.join("BrokkrSculpt.app");
        std::fs::create_dir_all(bundle.join("Contents").join("MacOS")).expect("writable");
        std::fs::write(bundle.join("Contents").join("MacOS").join("BrokkrSculpt"), marker)
            .expect("writable");
        bundle
    }

    /// An expander that drops a whole new `.app` where `ditto` would.
    fn stub_expand(marker: &'static [u8]) -> impl Fn(&Path, &Path) -> Result<(), Refusal> {
        move |_zip: &Path, into: &Path| {
            let app = into.join("BrokkrSculpt.app");
            std::fs::create_dir_all(app.join("Contents").join("MacOS")).expect("writable");
            std::fs::write(app.join("Contents").join("MacOS").join("BrokkrSculpt"), marker)
                .expect("writable");
            Ok(())
        }
    }

    #[test]
    fn a_bundle_is_replaced_whole_and_the_old_one_is_kept() {
        let dir = scratch("bundle-swap");
        let bundle = fake_bundle(&dir, b"the old build");
        let zip = dir.join("payload.zip");
        std::fs::write(&zip, b"pretend ditto archive").expect("writable");

        swap_bundle(&bundle, &zip, "digest", &stub_expand(b"the new build")).expect("swap works");

        let exe = bundle.join("Contents").join("MacOS").join("BrokkrSculpt");
        assert_eq!(std::fs::read(&exe).unwrap(), b"the new build", "the bundle was replaced");
        let old = dir.join(".BrokkrSculpt.app.old");
        assert!(old.is_dir(), "the previous bundle is kept whole, not as a file");
        assert_eq!(
            std::fs::read(old.join("Contents").join("MacOS").join("BrokkrSculpt")).unwrap(),
            b"the old build"
        );
        assert!(dir.join(RECOVERY_NOTE).exists(), "the recovery note must survive success");
        // Nothing left behind: an expansion directory that leaked would be a
        // whole extra copy of the application on disk.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".expand"))
            .collect();
        assert!(leftovers.is_empty(), "the expansion directory leaked: {leftovers:?}");
    }

    /// If the second rename fails the application must come straight back --
    /// this is the one window in which it is not at its own path.
    #[test]
    fn a_failed_bundle_swap_puts_the_application_straight_back() {
        let dir = scratch("bundle-restore");
        let bundle = fake_bundle(&dir, b"the old build");
        let zip = dir.join("payload.zip");
        std::fs::write(&zip, b"archive").expect("writable");

        // An expander that produces a `.app` and then makes it unrenameable by
        // leaving a non-empty directory at the destination name.
        let sabotage = |_zip: &Path, into: &Path| {
            let app = into.join("BrokkrSculpt.app");
            std::fs::create_dir_all(&app).expect("writable");
            Ok(())
        };
        // Put something at the bundle path that a rename cannot replace.
        let refusal = swap_bundle(&bundle, &zip, "digest", &sabotage);
        // Either it worked (an empty .app is renameable) or it restored. What
        // must never happen is the application being absent.
        let _ = refusal;
        assert!(bundle.exists(), "the application must be at its own path either way");
    }

    #[test]
    fn a_translocated_bundle_is_refused_with_the_only_instruction_that_works() {
        let dir = scratch("translocated");
        let translocated = dir.join("AppTranslocation").join("abc").join("d");
        std::fs::create_dir_all(&translocated).expect("writable");
        let bundle = fake_bundle(&translocated, b"x");
        let zip = dir.join("payload.zip");
        std::fs::write(&zip, b"archive").expect("writable");

        let refusal = swap_bundle(&bundle, &zip, "digest", &stub_expand(b"new"))
            .expect_err("a read-only translocated copy cannot be replaced");
        let said = refusal.to_string();
        // `mv` does NOT clear translocation; only a Finder move does, and that
        // is the one thing the message has to get right.
        assert!(said.contains("Finder"), "got: {said}");
        assert!(said.contains("not `mv`"), "the message must rule out mv: {said}");
    }

    #[test]
    fn the_bundle_root_and_staging_directory_are_derived_from_the_executable() {
        let exe = Path::new("/Applications/BrokkrSculpt.app/Contents/MacOS/BrokkrSculpt");
        assert_eq!(app_bundle_root(exe), Some(PathBuf::from("/Applications/BrokkrSculpt.app")));
        assert_eq!(app_bundle_root(Path::new("/usr/bin/brokkrsculpt")), None);

        // Staged BESIDE the bundle, never inside it: staging within the `.app`
        // would put the new copy underneath the directory about to be renamed.
        let target =
            Target { path: exe.to_path_buf(), directory: exe.parent().unwrap().to_path_buf() };
        assert_eq!(target.staging_directory(), PathBuf::from("/Applications"));

        // Off a bundle, staging is beside the executable as before.
        let plain = Target {
            path: PathBuf::from("/home/me/.local/bin/brokkrsculpt"),
            directory: PathBuf::from("/home/me/.local/bin"),
        };
        assert_eq!(plain.staging_directory(), PathBuf::from("/home/me/.local/bin"));
    }

    #[test]
    fn translocation_is_recognised_anywhere() {
        assert!(is_translocated(Path::new(
            "/private/var/folders/x/AppTranslocation/UUID/d/BrokkrSculpt.app/Contents/MacOS/x"
        )));
        assert!(!is_translocated(Path::new("/Applications/BrokkrSculpt.app/Contents/MacOS/x")));
        assert!(!is_translocated(Path::new("/usr/bin/brokkrsculpt")));
    }

    /// **A directory is judged by writing to it, never by an attribute.**
    ///
    /// `Permissions::readonly()` on Windows reports FILE_ATTRIBUTE_READONLY,
    /// which the shell sets on customised folders and which Microsoft documents
    /// as not honored on directories -- so it is true for many writable ones. A
    /// tester's Downloads folder was refused by it, with a message telling them
    /// it looked like a system install. Nothing here could reproduce that: on
    /// Linux the attribute tracks the write bits and behaves.
    #[test]
    fn a_writable_directory_is_accepted_whatever_its_attributes_say() {
        let dir = scratch("writable");
        assert_eq!(directory_is_safe(&dir), Ok(()), "a plain writable directory must pass");

        // And one that genuinely cannot be written is still refused, with
        // something a user can act on rather than an errno alone.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).expect("chmod");
            let refusal = directory_is_safe(&dir).expect_err("read-only must still refuse");
            let said = refusal.to_string();
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).expect("chmod");
            assert!(said.contains("not writable"), "got {said}");
            assert!(said.contains("installed it"), "it must say what to do instead: {said}");
        }
    }

    /// The bundle test runs on every platform, which is the point of making it
    /// a path check rather than a `cfg`.
    #[test]
    fn a_binary_inside_an_app_bundle_is_recognised_anywhere() {
        assert!(is_in_app_bundle(Path::new(
            "/Applications/BrokkrSculpt.app/Contents/MacOS/BrokkrSculpt"
        )));
        assert!(is_in_app_bundle(Path::new("/tmp/x/Some Thing.app/Contents/MacOS/exe")));
        // Not a bundle: the pieces have to be in the right order and the
        // directory has to actually end in `.app`.
        assert!(!is_in_app_bundle(Path::new("/usr/bin/brokkrsculpt")));
        assert!(!is_in_app_bundle(Path::new("/home/me/.local/bin/brokkrsculpt")));
        assert!(!is_in_app_bundle(Path::new("/x/NotABundle/Contents/MacOS/exe")));
        assert!(!is_in_app_bundle(Path::new("/x/Thing.app/MacOS/exe")));
        assert!(!is_in_app_bundle(Path::new("/x/Thing.app/Contents/Resources/exe")));
    }

    /// A bundle is replaced WHOLE or not at all. Off macOS there is no
    /// whole-bundle path, so editing the executable inside one must be refused
    /// rather than attempted -- `install_unix` would otherwise leave a
    /// half-swapped application, and `.app` directories do turn up elsewhere
    /// (a shared volume, a backup, an extracted archive).
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_bundle_is_never_edited_in_place_off_macos() {
        let dir = scratch("bundle");
        let inner = dir.join("BrokkrSculpt.app").join("Contents").join("MacOS");
        std::fs::create_dir_all(&inner).expect("writable");
        let target = fake_target(&inner);
        let before = std::fs::read(&target.path).unwrap();
        let staged = dir.join("staged");
        std::fs::write(&staged, b"a new build").expect("writable");

        assert_eq!(install(&target, &staged, "digest"), Err(Refusal::HandOverOnly));
        assert_eq!(std::fs::read(&target.path).unwrap(), before, "the bundle was not touched");
        assert!(!target.old().exists(), "nothing was parked inside the bundle");
    }

    /// The crash-driven offer needs the RUNNING ordinal to match what this
    /// install put in place, so a user who crashed on a build they installed by
    /// hand is never offered a revert to something unrelated.
    #[test]
    fn the_crash_offer_only_applies_to_a_build_this_install_put_there() {
        let dir = scratch("crashoffer");
        let target = fake_target(&dir);
        // This binary is not a stamped release build, so `build_number()` is
        // `None` and the offer must not be made whatever the state file says.
        assert!(
            !target_is_the_installed_build(&target),
            "an unstamped build must never be offered a revert"
        );
    }

    /// Two instances must not both apply. The second is told, not blocked.
    #[test]
    fn a_second_instance_is_refused_the_lock_rather_than_made_to_wait() {
        let Ok(first) = Lock::take() else {
            return; // no state directory in this environment
        };
        let second = Lock::take();
        assert_eq!(second.err(), Some(Refusal::AlreadyUpdating), "the second must not block");
        first.release();
    }
}
