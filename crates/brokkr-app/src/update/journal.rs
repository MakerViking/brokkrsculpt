// SPDX-License-Identifier: AGPL-3.0-only

//! What the updater tried, and how it went, written where the one person who
//! needs it can read it.
//!
//! A Windows release build is linked for the windows subsystem, so there is no
//! console and `eprintln!` has nowhere to go. Until this existed, a failed
//! update reached exactly one place -- `self.status` -- and the welcome card,
//! which carries an install button of its own, draws over the line that shows
//! it. So the failure was invisible while it happened and gone afterwards, and
//! that is the whole reason the Windows relaunch error has been reported more
//! than once and never once read. `crash.rs` is the model and its rules are
//! inherited: everything goes through `redact_user_paths`, every fallible step
//! is `let _ =`, there is no `unwrap` in this file, and the readers and writers
//! take a path so the suite never touches the real one.
//!
//! # Append, do not overwrite
//!
//! `update.state` is overwritten because it holds CURRENT FACTS -- a floor, a
//! digest, an ordinal. This holds HISTORY, and the failures worth diagnosing
//! are sequences: refused, retried, succeeded; or installed, then reverted on
//! the next launch. Overwriting erases the record that says *why* with the
//! record that says what happened next, which is the wrong one to lose.
//!
//! # It records, it does not announce
//!
//! **Nothing here is read at startup, and that is deliberate rather than
//! unfinished.** The obvious extra -- write a `started` record, and have the
//! next launch call a dangling one an unfinished update -- was designed and
//! then dropped, because the pending marker written by `apply::marker_path`
//! ALREADY is that record and `Brokkr::new` already acts on it. A second
//! announcer reading a second file gets it wrong in a way the first does not:
//! the child process writes its record while the parent is still finishing,
//! so "the last record" depends on which process won a race, and the first
//! draft of this module announced a failure on every SUCCESSFUL update.
//!
//! This file is evidence for a human and for the bug report. The decisions
//! stay where they already are.
//!
//! # The check is not recorded
//!
//! Failure there is deliberately silence (see the parent module), it runs on
//! every launch, and one record per failed check is a week of somebody's flaky
//! wifi pushing the record that matters past the cap. A check that SUCCEEDS
//! already leaves `last_check` in `update.state`.
//!
//! # Not shared with `crash.rs`
//!
//! This is the SECOND copy of "write a file in the state directory", and the
//! third is where extracting stops costing more than it saves -- particularly
//! here, where the two differ in the part an abstraction would have to own: a
//! crash report is taken and deleted, this is appended and kept.

use std::fmt::Write as _;
use std::path::Path;

/// Where the log lives.
///
/// `.txt` and not `.log` for the reason `last-crash.txt` is: the reader is a
/// Windows user with no console, and a double click has to open something.
const LOG_FILE: &str = "update-log.txt";

/// How much history to keep, trimmed from the front on whole records.
///
/// A byte cap rather than a record count, for the reason `crash.rs` gives for
/// its own: the file has to stay readable by a person and fit inside a bug
/// report, and that is a byte question. A count cap plus one pathological
/// `io::Error` string would satisfy the count and fail the requirement.
const MAX_BYTES: usize = 32 * 1024;

/// How much of one reason to keep.
///
/// Capped separately so no single record can dominate the file and push every
/// other one out. By characters and not by bytes, because the result is written
/// as UTF-8 and read back as a `String` -- the same rule `report.rs` states.
const MAX_DETAIL_CHARS: usize = 400;

/// Which part of an update a record is about.
///
/// `Relaunch` is separate from `Install` and that distinction is load bearing:
/// by the time the new build is spawned the swap has completed, the
/// anti-rollback floor has already moved and the file at the target path IS the
/// new build. Recording a failed relaunch as a failed install would tell a
/// maintainer the install did not happen in the one state where recovery
/// depends on knowing that it did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Fetching and verifying the payload.
    Download,
    /// Putting the verified file in place.
    Install,
    /// Starting the build that was just installed.
    Relaunch,
    /// Going back to the kept copy.
    Revert,
}

impl Step {
    fn as_str(self) -> &'static str {
        match self {
            Step::Download => "download",
            Step::Install => "install",
            Step::Relaunch => "relaunch",
            Step::Revert => "revert",
        }
    }
}

/// How it went.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    Ok,
    Failed,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Failed => "failed",
        }
    }
}

/// One thing that happened, ready to be written down.
pub struct Entry<'a> {
    pub step: Step,
    pub outcome: Outcome,
    /// The build the record is about, when there is one.
    pub build: Option<u64>,
    /// Why it failed, in the words the user was shown.
    pub detail: Option<&'a str>,
}

/// Seconds since the epoch, or zero if the clock refuses.
///
/// The same shape the parent module uses for `last_check`, and the same rule
/// applies: **nothing compares this to make a decision**, so a wrong clock
/// makes `at` wrong and changes nothing else. Ordering is position in the file.
fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

/// One record, as the flat `key = value` block every other file here uses.
///
/// `crate::paths::entries` parses this with no new parser. Newlines in `detail`
/// are replaced with spaces before anything else: a blank line is the only
/// thing separating records, so a reason containing one could otherwise forge a
/// record boundary. `split_once('=')` keeps everything after the first `=`, so
/// a reason containing `=` needs no escaping.
fn compose(entry: &Entry<'_>, at: u64) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "at = {at}");
    let _ = writeln!(out, "step = {}", entry.step.as_str());
    let _ = writeln!(out, "outcome = {}", entry.outcome.as_str());
    if let Some(build) = entry.build {
        let _ = writeln!(out, "build = {build}");
    }
    if let Some(detail) = entry.detail {
        let flat: String = detail
            .chars()
            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
            .take(MAX_DETAIL_CHARS)
            .collect();
        let _ = writeln!(out, "detail = {}", flat.trim());
    }
    out.push('\n');
    // Once, here, over the whole record: the details carry the target path and
    // the staged payload path, which on Windows are under `C:\Users\<name>`.
    // Redaction is idempotent, so the second pass when this reaches a bug
    // report is a no-op.
    crate::report::redact_user_paths(&out)
}

/// Keep the file under `max` bytes by dropping whole records from the front.
///
/// Oldest first, and on record boundaries rather than byte ones: half a record
/// left at the top of the file would parse as a record with whatever keys
/// survived the cut.
fn trim_to(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let blocks: Vec<&str> = text.split("\n\n").filter(|block| !block.trim().is_empty()).collect();
    let mut kept: Vec<&str> = Vec::new();
    let mut total = 0usize;
    for block in blocks.iter().rev().copied() {
        let cost = block.len() + 2;
        if total + cost > max {
            break;
        }
        total += cost;
        kept.push(block);
    }
    kept.reverse();
    let mut out = String::with_capacity(total);
    for block in kept {
        out.push_str(block.trim_end());
        out.push_str("\n\n");
    }
    out
}

/// Append one record to a given path, creating the directory.
///
/// The path is a parameter for the reason `crash.rs` states for its own
/// readers: the suite runs in parallel, so two tests sharing one file race, and
/// writing to the real one would put test noise into a user's update history.
///
/// Read-trim-write rather than a bare append. An append is one syscall and this
/// is three, but the cap has to be enforced by the writer -- nothing else
/// prunes the state directory -- and an update outcome happens at most a
/// handful of times in a session.
fn append_to(path: &Path, entry: &Entry<'_>, at: u64) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = std::fs::read_to_string(path).unwrap_or_default();
    text.push_str(&compose(entry, at));
    std::fs::write(path, trim_to(&text, MAX_BYTES))
}

/// Write down one update outcome at a stated path, dropping any failure.
///
/// **This is where the `let _ =` discipline actually lives**, so it is what the
/// tests drive. [`record`] is the two lines that resolve the real state
/// directory and call this; a test driving those would append to the
/// developer's own update history, in parallel and unsynchronised, mixed in
/// with records a user might need. `crash.rs` states the same rule for its own
/// readers.
fn record_to(path: &Path, entry: &Entry<'_>, at: u64) {
    let _ = append_to(path, entry, at);
}

/// Write down one update outcome. Failure to write is dropped.
///
/// **Every call happens AFTER the thing it records**, so a log that cannot be
/// written can never cause the failure it would have described.
///
/// Resolves the real state directory and does nothing else, which is why it is
/// inert under `cfg(test)`: the app-level tests that exercise an update outcome
/// dispatch real messages, and without this they would append to the
/// developer's own update history, in parallel and unsynchronised, mixed in
/// with records a user might still need. What it delegates to, [`record_to`],
/// is driven directly by the tests below at scratch paths, so the discipline
/// this exists to guarantee is still pinned.
pub fn record(entry: &Entry<'_>) {
    if cfg!(test) {
        return;
    }
    let Some(path) = crate::paths::state_file(LOG_FILE) else {
        return;
    };
    record_to(&path, entry, now());
}

/// Write down a refusal. Failure to write is dropped.
pub fn failed(step: Step, build: Option<u64>, why: &str) {
    record(&Entry { step, outcome: Outcome::Failed, build, detail: Some(why) });
}

/// Write down a step that worked. Failure to write is dropped.
///
/// Successes are recorded as well as failures, because the sequence is the
/// diagnosis: "installed, then the next launch reverted" is a different report
/// from "install refused", and only the successful record tells them apart.
pub fn ok(step: Step, build: Option<u64>) {
    record(&Entry { step, outcome: Outcome::Ok, build, detail: None });
}

/// The whole log, for the bug report, or empty when there is none.
///
/// **This is the route by which a failed update reaches the maintainer.** A
/// user with no console cannot be asked to go and find a file, and the one
/// place they already press is Help > Report a bug. `assemble_report` redacts
/// again on the way out, which is a no-op over already-redacted text.
pub fn for_report() -> String {
    read_from(crate::paths::state_file(LOG_FILE).as_deref())
}

/// [`for_report`], against a stated path.
fn read_from(path: Option<&Path>) -> String {
    let Some(path) = path else {
        return String::new();
    };
    std::fs::read_to_string(path).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!("brokkr-journal-test-{name}-{}", std::process::id()))
            .join(LOG_FILE)
    }

    fn clean(path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::remove_dir_all(parent);
        }
    }

    fn entry<'a>(step: Step, outcome: Outcome, detail: Option<&'a str>) -> Entry<'a> {
        Entry { step, outcome, build: Some(1042), detail }
    }

    /// The record parses with the parser every other file here uses, rather
    /// than needing one of its own.
    #[test]
    fn a_record_is_readable_by_the_config_parser_the_rest_of_the_tree_uses() {
        let text = compose(&entry(Step::Install, Outcome::Failed, Some("access is denied")), 1788);
        let pairs: Vec<(&str, &str)> = crate::paths::entries(&text).collect();

        assert!(pairs.contains(&("at", "1788")), "no timestamp: {pairs:?}");
        assert!(pairs.contains(&("step", "install")), "no step: {pairs:?}");
        assert!(pairs.contains(&("outcome", "failed")), "no outcome: {pairs:?}");
        assert!(pairs.contains(&("build", "1042")), "no build: {pairs:?}");
        assert!(pairs.contains(&("detail", "access is denied")), "no detail: {pairs:?}");
    }

    /// **A relaunch failure is not an install failure.**
    ///
    /// By the time the spawn is attempted the swap has completed and the
    /// anti-rollback floor has moved, so a record saying the install failed
    /// would be wrong in the one state where a maintainer needs it to be right.
    #[test]
    fn a_failed_relaunch_is_recorded_as_its_own_step_and_not_as_a_failed_install() {
        let text = compose(&entry(Step::Relaunch, Outcome::Failed, Some("os error 5")), 1788);
        let pairs: Vec<(&str, &str)> = crate::paths::entries(&text).collect();

        assert!(pairs.contains(&("step", "relaunch")), "the step was not relaunch: {pairs:?}");
        assert!(
            !pairs.contains(&("step", "install")),
            "a failed relaunch was filed as a failed install: {pairs:?}"
        );
    }

    /// A reason spanning lines must not be able to forge a record boundary:
    /// the blank line is the only thing separating one record from the next.
    #[test]
    fn a_reason_containing_a_blank_line_cannot_forge_a_second_record() {
        let nasty = "it failed\n\nat = 999\nstep = install\noutcome = ok";
        let text = compose(&entry(Step::Install, Outcome::Failed, Some(nasty)), 1788);

        let records = text.split("\n\n").filter(|block| !block.trim().is_empty()).count();
        assert_eq!(records, 1, "a reason split itself into two records:\n{text}");
        let pairs: Vec<(&str, &str)> = crate::paths::entries(&text).collect();
        assert_eq!(
            pairs.iter().filter(|(key, _)| *key == "outcome").count(),
            1,
            "two outcomes in one record: {pairs:?}"
        );
        assert!(!pairs.contains(&("outcome", "ok")), "the forged outcome won: {pairs:?}");
    }

    /// A path in a reason is redacted before it is ever written down.
    #[test]
    fn a_home_directory_in_a_reason_is_redacted_before_it_reaches_the_file() {
        let text = compose(
            &entry(Step::Install, Outcome::Failed, Some("could not write /home/somebody/bin/x")),
            1788,
        );
        assert!(!text.contains("somebody"), "the user's name went into the log:\n{text}");
        assert!(text.contains("REDACTED"), "nothing was redacted:\n{text}");
    }

    /// The cap drops whole records from the front, and keeps the newest.
    ///
    /// Asserts the SURVIVORS rather than only the size: a trim that emptied the
    /// file would satisfy a size assertion perfectly, and lose the record that
    /// matters.
    #[test]
    fn the_cap_drops_the_oldest_whole_records_and_keeps_the_newest() {
        let one = compose(&entry(Step::Install, Outcome::Failed, Some("first")), 1);
        let two = compose(&entry(Step::Install, Outcome::Failed, Some("second")), 2);
        let three = compose(&entry(Step::Install, Outcome::Failed, Some("third")), 3);
        let all = format!("{one}{two}{three}");

        // Room for two records and not three.
        let trimmed = trim_to(&all, one.len() * 2 + 4);

        assert!(trimmed.contains("third"), "the newest record was dropped:\n{trimmed}");
        assert!(trimmed.contains("second"), "the second record was dropped:\n{trimmed}");
        assert!(!trimmed.contains("first"), "the oldest record survived the cap:\n{trimmed}");
        // Whole records only: every survivor still parses.
        for block in trimmed.split("\n\n").filter(|b| !b.trim().is_empty()) {
            let pairs: Vec<(&str, &str)> = crate::paths::entries(block).collect();
            assert!(
                pairs.iter().any(|(key, _)| *key == "at"),
                "a record was cut in half:\n{block}"
            );
        }
    }

    /// A short file is left exactly as it was, rather than being rewritten.
    #[test]
    fn a_log_under_the_cap_is_not_touched_by_the_trim() {
        let one = compose(&entry(Step::Download, Outcome::Ok, None), 7);
        assert_eq!(trim_to(&one, MAX_BYTES), one);
    }

    /// Appending twice leaves two records, in the order they happened.
    #[test]
    fn two_outcomes_leave_two_records_in_the_order_they_happened() {
        let path = scratch("order");
        clean(&path);

        append_to(&path, &entry(Step::Download, Outcome::Ok, None), 10).expect("first append");
        append_to(&path, &entry(Step::Relaunch, Outcome::Failed, Some("os error 5")), 20)
            .expect("second append");

        let text = std::fs::read_to_string(&path).expect("the log was not written");
        let first = text.find("download").expect("the download record is missing");
        let second = text.find("relaunch").expect("the relaunch record is missing");
        assert!(first < second, "the records are out of order:\n{text}");
        clean(&path);
    }

    /// **A log that cannot be written never breaks an update.**
    ///
    /// The writer is driven at a path it cannot possibly write -- a directory
    /// standing where the file should be -- and the assertion is that the
    /// PUBLIC entry point returns normally anyway. That is the claim: every
    /// call site does `let _ =` on this, and an `unwrap` slipped in here would
    /// turn a logging failure into a panic in the middle of an update.
    ///
    /// Asserting only that `append_to` returns `Err` would be the weaker test
    /// the name does not promise: it would stay green if `record_to` were
    /// changed to unwrap that very `Err`.
    #[test]
    fn a_log_that_cannot_be_written_is_dropped_rather_than_breaking_the_update() {
        let blocked = scratch("blocked");
        clean(&blocked);
        if let Some(parent) = blocked.parent() {
            std::fs::create_dir_all(parent).expect("scratch directory");
        }
        // A directory where the file should be: every write to it fails.
        std::fs::create_dir_all(&blocked).expect("a directory in the file's place");

        assert!(
            append_to(&blocked, &entry(Step::Install, Outcome::Failed, None), 1).is_err(),
            "the fixture no longer blocks writes, so this test proves nothing"
        );

        // The claim: the public writer swallows that and returns.
        record_to(&blocked, &entry(Step::Relaunch, Outcome::Failed, Some("os error 5")), 2);

        assert!(blocked.is_dir(), "the blocking directory should still be in the way");
        clean(&blocked);
    }

    /// No log is an empty report section, not a missing file error.
    #[test]
    fn a_report_from_a_machine_that_has_never_updated_carries_an_empty_log() {
        let missing = scratch("never");
        clean(&missing);
        assert!(read_from(Some(&missing)).is_empty());
        assert!(read_from(None).is_empty());
    }
}
