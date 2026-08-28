// SPDX-License-Identifier: AGPL-3.0-only

//! What happens when the application panics, and how the next launch says so.
//!
//! Before this, a panic was invisible. The window vanished, the sculpt went
//! with it, and the only trace was a backtrace on a terminal nobody launched
//! the application from -- so on a machine that is not this one, the report
//! that reached the maintainer was "it closed". That is a fine state of affairs
//! while the only user builds from source; it is not one to open a beta with.
//!
//! # What this deliberately does NOT do
//!
//! **It does not save the document.** The obvious next thought is to write an
//! emergency copy of the field from the hook, and it is a trap: the panic may
//! be *because* the document is inconsistent, the hook has no access to it
//! anyway -- iced owns the state -- and a corrupt emergency file written over a
//! good autosave turns a recoverable crash into a lost afternoon. The sculpt is
//! already covered by the two-minute autosave and `File > Recover`; this covers
//! the thing that had nothing at all.
//!
//! # Rules a panic hook has to keep
//!
//! **It must not panic.** A panic inside a panic hook aborts the process
//! immediately, losing the very report it was writing, so every fallible step
//! here is `let _ =` or `if let Ok`. There is no `unwrap` in this file and
//! there must not be one.
//!
//! **It must chain to the hook it replaced.** Otherwise the terminal output
//! disappears for the developer who *is* running from a terminal, and the fix
//! for "I cannot see crashes" would have removed the one place they were
//! visible.
//!
//! **Everything written goes through [`crate::report::redact_user_paths`]**, by
//! the same argument the bug reporter makes: a backtrace is full of build paths
//! and a panic message can quote a filename. The file lands in the state
//! directory beside the autosave, where a user can read it before it goes
//! anywhere -- nothing here sends anything.

use std::fmt::Write;

/// Where the last crash is left for the next launch to find.
///
/// A state file rather than a config one: it is something the application
/// recovers from, not something the user set. See [`crate::paths::state_file`].
const CRASH_FILE: &str = "last-crash.txt";

/// How much of a report is kept.
///
/// A deep recursion produces a backtrace measured in megabytes, and the point
/// of this file is to be read -- by a person first and the bug reporter second,
/// where it has to fit in a payload the dialog shows in full.
const MAX_BYTES: usize = 64 * 1024;

/// Install the panic hook. Call once, before the window opens.
pub fn install() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        write_report(info);
        // Chained last, so the report exists even if the default hook decides
        // to abort.
        previous(info);
    }));
}

/// Compose the report and put it beside the autosave. Never fails loudly.
fn write_report(info: &std::panic::PanicHookInfo<'_>) {
    // `payload_as_str` is not stable, so this is the two-cast dance every panic
    // hook does. A payload that is neither is not worth guessing at.
    let payload = info
        .payload()
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
        .unwrap_or("(a panic payload that was neither &str nor String)");
    let where_ = info.location().map(|at| (at.file().to_string(), at.line()));
    let text = compose(payload, where_.as_ref().map(|(f, l)| (f.as_str(), *l)));
    if let Some(path) = crate::paths::state_file(CRASH_FILE) {
        let _ = put(&path, &text);
    }
}

/// The report's text, from the two things a panic actually carries.
///
/// **Split out from the hook so it can be tested without installing one.** A
/// test that sets the global panic hook and then panics inside `catch_unwind`
/// is not a local act: the suite runs tests in parallel, so any OTHER test's
/// failing assertion during that window would be swallowed by this hook instead
/// of printing. Testing the composition directly costs nothing and cannot mask
/// a real failure.
fn compose(payload: &str, at: Option<(&str, u32)>) -> String {
    let mut text = String::with_capacity(4096);
    text.push_str("BrokkrSculpt crashed.\n\n");
    let _ = writeln!(text, "version: {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(text, "target:  {}", std::env::consts::ARCH);
    let _ = writeln!(text, "os:      {}", std::env::consts::OS);
    if let Some((file, line)) = at {
        let _ = writeln!(text, "at:      {file}:{line}");
    }
    let _ = writeln!(text, "\nmessage:\n{payload}\n");
    let _ = writeln!(text, "backtrace:\n{}", std::backtrace::Backtrace::force_capture());

    let mut redacted = crate::report::redact_user_paths(&text);
    if redacted.len() > MAX_BYTES {
        // On a char boundary, because the result is written as UTF-8 and read
        // back as a String.
        let mut cut = MAX_BYTES;
        while cut > 0 && !redacted.is_char_boundary(cut) {
            cut -= 1;
        }
        redacted.truncate(cut);
        redacted.push_str("\n\n[truncated]\n");
    }
    redacted
}

/// Write a report to a given path, creating the directory. Errors are dropped.
fn put(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)
}

/// Take a report from a given path. The path is a parameter so a test never
/// touches the real one -- the suite runs in parallel, and two tests sharing
/// one file race, while clobbering the real file would swallow a crash report
/// the developer had not read yet.
fn take_from(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let _ = std::fs::remove_file(path);
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// The report a previous run left, if there is one, removing it as it goes.
///
/// **Taken and not merely read**, so a crash is announced once. Left in place
/// it would greet the user on every launch for ever, and a notice that is
/// always there is one nobody reads -- which is the state the autosave notice
/// was deliberately designed out of.
pub fn take_pending() -> Option<String> {
    take_from(&crate::paths::state_file(CRASH_FILE)?)
}

/// Where a crash report would be written, for the message that mentions it.
pub fn report_path() -> Option<std::path::PathBuf> {
    crate::paths::state_file(CRASH_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("brokkr-crash-test-{name}-{}", std::process::id()))
    }

    /// **A crash is announced once, not for ever.**
    ///
    /// The autosave notice is shown whenever an autosave exists, which is right
    /// for a file that stays useful. A crash report is not that: left in place
    /// it would greet every launch until the file were deleted by hand, and a
    /// notice that is always there is one nobody reads.
    #[test]
    fn a_pending_report_is_taken_rather_than_read() {
        let path = scratch("taken");
        put(&path, "a crash from a previous run").expect("the temp dir is writable");

        assert_eq!(take_from(&path).as_deref(), Some("a crash from a previous run"));
        assert_eq!(take_from(&path), None, "the report was announced a second time");
        assert!(!path.exists(), "the report file outlived being taken");
    }

    /// **A report carries the panic and not the user's home directory.**
    ///
    /// A backtrace is full of build paths and a panic message can quote a
    /// filename, which is the whole reason this goes through
    /// `redact_user_paths` -- the same argument the bug reporter makes about
    /// everything it collects.
    #[test]
    fn a_report_carries_the_panic_and_no_home_directory() {
        let text = compose(
            "the field went missing at /home/somebody/Models/secret.brokkr",
            Some(("crates/brokkr-app/src/volume.rs", 42)),
        );

        assert!(text.contains("the field went missing"), "the message is missing: {text}");
        assert!(text.contains("volume.rs:42"), "the panic location is missing: {text}");
        assert!(text.contains("backtrace:"), "no backtrace was captured");
        assert!(
            !text.contains("somebody"),
            "the report carries an unredacted home directory: {text}"
        );
    }

    /// An empty or whitespace-only file is not a crash.
    ///
    /// A zero-byte file is what a run killed mid-write leaves, and announcing
    /// "the last session crashed" over nothing would train the user to ignore
    /// the message that matters.
    #[test]
    fn an_empty_report_file_is_not_announced() {
        let path = scratch("empty");
        put(&path, "   \n\t\n ").expect("the temp dir is writable");
        assert_eq!(take_from(&path), None);
    }
}
