// SPDX-License-Identifier: AGPL-3.0-only

//! What the session did before it went wrong.
//!
//! A bug report that says "it went black" is not diagnosable. One that says "it
//! went black, and here are the last twenty things the application did" usually
//! is. This is the trail, and the point of it is that it costs nothing to
//! collect: the status line the user already reads doubles as the record, the
//! way SindriCAD's toast layer does.
//!
//! # Two kinds of crumb, and why the difference is load bearing
//!
//! **Ordinary crumbs** are things that happened, and they carry a `HH:MM:SS `
//! prefix. Only the last [`MAX_ORDINARY`] are kept.
//!
//! **Sticky facts** are things that are true — the graphics adapter, whether a
//! tablet was found — recorded once at startup. They carry no timestamp and
//! they are never dropped to make room for ordinary crumbs.
//!
//! That timestamp is the whole discriminator, and
//! [`crate::report::trim_breadcrumbs`] is the other half of the contract: it
//! partitions on exactly this prefix. A sticky fact that gained one would
//! silently become droppable, which is the bug SindriCAD shipped — its startup
//! facts sat at the front of one flat list, so keeping "the last twenty" threw
//! them away in every session busy enough to be worth reporting.

use std::collections::VecDeque;

/// Ordinary crumbs held. The report sends at most twenty, so holding a few more
/// costs nothing and leaves room to widen that without touching this.
const MAX_ORDINARY: usize = 32;
/// Sticky facts held.
const MAX_STICKY: usize = 24;
/// Longest crumb kept, in characters.
///
/// The server truncates at 300 per crumb. Cutting here as well means what the
/// dialog shows is what arrives, rather than a longer version of it.
const MAX_LEN: usize = 300;

/// The session's trail.
#[derive(Debug, Default)]
pub(crate) struct Breadcrumbs {
    sticky: Vec<String>,
    ordinary: VecDeque<String>,
}

impl Breadcrumbs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record something that happened.
    ///
    /// Deduplicated against the previous crumb, because the status line is set
    /// on a great many messages and a trail of twenty identical entries says
    /// less than one does.
    pub fn crumb(&mut self, what: &str) {
        let what = trim_to(what.trim(), MAX_LEN);
        if what.is_empty() {
            return;
        }
        let line = format!("{} {what}", clock());
        // Compared past the timestamp, so the same event a second apart still
        // counts as a repeat.
        if self.ordinary.back().is_some_and(|last| last[9..] == line[9..]) {
            return;
        }
        if self.ordinary.len() == MAX_ORDINARY {
            self.ordinary.pop_front();
        }
        self.ordinary.push_back(line);
    }

    /// Record something that is true of this session.
    ///
    /// No timestamp, deliberately: that is what keeps it out of the ordinary
    /// trail's twenty-slot budget. See the module docs.
    pub fn sticky_fact(&mut self, what: &str) {
        let what = trim_to(what.trim(), MAX_LEN);
        if what.is_empty() || self.sticky.len() == MAX_STICKY {
            return;
        }
        if self.sticky.contains(&what) {
            return;
        }
        self.sticky.push(what);
    }

    /// The whole trail, sticky facts first, for a report or for the clipboard.
    pub fn all(&self) -> Vec<String> {
        self.sticky.iter().chain(self.ordinary.iter()).cloned().collect()
    }
}

/// `HH:MM:SS`, UTC, exactly eight characters.
///
/// UTC rather than local time, and without a date. A crumb is read against the
/// crumbs around it — what happened just before the thing broke — so the offset
/// from anybody's wall clock does not matter, and getting a local offset in
/// `std` alone is not possible without a timezone database.
///
/// The width is not cosmetic. `report::trim_breadcrumbs` recognises an ordinary
/// crumb by exactly nine leading bytes, so this and that must agree.
fn clock() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let day = seconds % 86_400;
    format!("{:02}:{:02}:{:02}", day / 3600, (day % 3600) / 60, day % 60)
}

/// Cut to `limit` characters, never mid character.
fn trim_to(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    text.chars().take(limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::trim_breadcrumbs;

    #[test]
    fn the_clock_is_exactly_the_shape_the_report_recognises() {
        let now = clock();
        assert_eq!(now.len(), 8, "the prefix width is a contract with trim_breadcrumbs");
        let bytes = now.as_bytes();
        assert!(bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit());
        assert_eq!(bytes[2], b':');
        assert_eq!(bytes[5], b':');
    }

    /// The contract, asserted across both modules rather than described in
    /// each. This is what would catch someone giving sticky facts a timestamp.
    #[test]
    fn a_sticky_fact_outlives_a_busy_session_and_an_ordinary_crumb_does_not() {
        let mut trail = Breadcrumbs::new();
        trail.sticky_fact("[wgpu] Radeon RX 6900 XT, Vulkan");
        trail.crumb("opened a file");
        for index in 0..MAX_ORDINARY * 2 {
            trail.crumb(&format!("did thing {index}"));
        }

        let sent = trim_breadcrumbs(&trail.all());
        assert!(sent[0].starts_with("[wgpu]"), "the sticky fact did not lead: {sent:?}");
        assert!(!sent.iter().any(|c| c.contains("opened a file")), "an old crumb survived");
        assert!(sent.iter().any(|c| c.contains("did thing 63")), "the newest crumb was dropped");
    }

    #[test]
    fn a_repeated_status_is_recorded_once() {
        // The status line is set on a great many messages, so without this the
        // trail is twenty copies of whatever was last shown.
        let mut trail = Breadcrumbs::new();
        trail.crumb("remeshed");
        trail.crumb("remeshed");
        trail.crumb("remeshed");
        assert_eq!(trail.all().len(), 1);
        trail.crumb("saved");
        trail.crumb("remeshed");
        assert_eq!(trail.all().len(), 3, "only ADJACENT repeats collapse");
    }

    #[test]
    fn a_sticky_fact_is_recorded_once_however_often_it_is_offered() {
        let mut trail = Breadcrumbs::new();
        for _ in 0..10 {
            trail.sticky_fact("[tablet] no tablet found");
        }
        assert_eq!(trail.all(), vec!["[tablet] no tablet found".to_string()]);
    }

    #[test]
    fn nothing_empty_is_recorded() {
        let mut trail = Breadcrumbs::new();
        trail.crumb("");
        trail.crumb("   ");
        trail.sticky_fact("");
        assert!(trail.all().is_empty());
    }

    #[test]
    fn a_long_crumb_is_cut_by_characters_and_not_by_bytes() {
        let mut trail = Breadcrumbs::new();
        trail.crumb(&"é".repeat(MAX_LEN * 2));
        let only = &trail.all()[0];
        // The nine-byte prefix, then exactly MAX_LEN characters.
        assert_eq!(only.chars().skip(9).count(), MAX_LEN);
    }

    #[test]
    fn the_trail_is_bounded_however_long_the_session_runs() {
        let mut trail = Breadcrumbs::new();
        for index in 0..10_000 {
            trail.crumb(&format!("thing {index}"));
            trail.sticky_fact(&format!("fact {index}"));
        }
        assert_eq!(trail.all().len(), MAX_STICKY + MAX_ORDINARY);
    }
}
