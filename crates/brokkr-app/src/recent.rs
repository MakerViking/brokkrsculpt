// SPDX-License-Identifier: AGPL-3.0-only

//! The list of recently opened sculpts.
//!
//! Deliberately the same shape as [`crate::spacemouse::Config`]: a flat text
//! file under `$XDG_CONFIG_HOME/brokkrsculpt`, hand parsed, where anything
//! missing or unreadable falls back to a working default rather than being an
//! error. A recent files list is a convenience, and no convenience may stop the
//! application starting or interrupt what the user was doing.

use std::path::{Path, PathBuf};

/// How many entries to keep.
///
/// Long enough to span a few sessions, short enough that the File menu does not
/// become a scrolling list of its own.
const CAPACITY: usize = 10;

/// Sculpts opened or saved recently, most recent first.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Recent {
    paths: Vec<PathBuf>,
    /// The file this was read from and will be written back to.
    ///
    /// Carried rather than recomputed so a test can point an instance at a
    /// temporary directory. Without it every test run would rewrite the real
    /// `~/.config/brokkrsculpt/recent`, which is someone's actual list.
    file: Option<PathBuf>,
}

impl Recent {
    /// Where the list lives, beside the puck's settings.
    pub fn path() -> Option<PathBuf> {
        crate::paths::config_file("recent")
    }

    /// Read the list from the standard location. A missing or unreadable file
    /// is an empty list, never an error.
    pub fn load() -> Recent {
        Recent::load_from(Self::path())
    }

    /// Read the list from a given file, for tests and for anything that must
    /// not touch the user's own.
    pub fn load_from(file: Option<PathBuf>) -> Recent {
        let Some(file) = file else {
            return Recent::default();
        };
        let Ok(text) = std::fs::read_to_string(&file) else {
            return Recent { paths: Vec::new(), file: Some(file) };
        };
        let mut recent = Recent::parse(&text);
        recent.file = Some(file);
        recent
    }

    /// One path per line. Blank lines and anything that is not an absolute path
    /// are dropped rather than rejected, so a hand edited or truncated file
    /// still yields whatever of it was good.
    fn parse(text: &str) -> Recent {
        let mut paths: Vec<PathBuf> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let candidate = PathBuf::from(line);
            if candidate.is_absolute() && !paths.contains(&candidate) {
                paths.push(candidate);
            }
        }
        paths.truncate(CAPACITY);
        Recent { paths, file: None }
    }

    /// Write the list out, complaining to the log and nowhere else if it fails.
    ///
    /// Never touches `status`: the user is reading that line for the save or
    /// open they just asked for, and a bookkeeping failure has no business
    /// taking it over.
    pub fn save(&self) {
        let Some(path) = self.file.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            log::warn!("could not create {}: {error}", parent.display());
            return;
        }
        let body: String = self.paths.iter().map(|path| format!("{}\n", path.display())).collect();
        if let Err(error) = std::fs::write(path, body) {
            log::warn!("could not write {}: {error}", path.display());
        }
    }

    /// Put `path` at the front, and persist.
    ///
    /// Absolute paths only: a relative one would resolve against whatever
    /// directory the application happened to be started from, which is not
    /// stable across sessions and is exactly when a recent list is read.
    pub fn record(&mut self, path: &Path) {
        let Ok(absolute) = std::path::absolute(path) else {
            return;
        };
        self.paths.retain(|existing| existing != &absolute);
        self.paths.insert(0, absolute);
        self.paths.truncate(CAPACITY);
        self.save();
    }

    /// Drop one entry, for a file that has since gone away.
    pub fn forget(&mut self, path: &Path) {
        self.paths.retain(|existing| existing != path);
        self.save();
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_puts_the_newest_first_and_removes_the_duplicate() {
        let mut recent = Recent::load_from(None);
        recent.record(Path::new("/a.brokkr"));
        recent.record(Path::new("/b.brokkr"));
        recent.record(Path::new("/a.brokkr"));

        assert_eq!(
            recent.paths(),
            [PathBuf::from("/a.brokkr"), PathBuf::from("/b.brokkr")],
            "reopening a file should move it to the top rather than list it twice"
        );
    }

    #[test]
    fn a_recorded_file_can_be_forgotten_when_it_goes_away() {
        let mut recent = Recent::load_from(None);
        recent.record(Path::new("/gone.brokkr"));
        recent.forget(Path::new("/gone.brokkr"));
        assert!(recent.is_empty());
    }

    #[test]
    fn the_list_is_capped() {
        let text: String = (0..40).map(|index| format!("/sculpt{index}.brokkr\n")).collect();
        let recent = Recent::parse(&text);
        assert_eq!(recent.paths().len(), CAPACITY);
        assert_eq!(recent.paths()[0], PathBuf::from("/sculpt0.brokkr"));
    }

    #[test]
    fn garbage_parses_to_whatever_of_it_was_good() {
        let recent = Recent::parse("\n\nnot/absolute\n/good.brokkr\n   \n/good.brokkr\n");
        assert_eq!(
            recent.paths(),
            [PathBuf::from("/good.brokkr")],
            "a relative path, a blank line and a duplicate should all be dropped"
        );
    }

    #[test]
    fn an_empty_file_is_an_empty_list_rather_than_an_error() {
        assert!(Recent::parse("").is_empty());
    }

    #[test]
    fn a_round_trip_through_a_real_file_survives() {
        let directory = std::env::temp_dir().join(format!("brokkr-recent-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let file = directory.join("recent");

        let mut written = Recent::load_from(Some(file.clone()));
        assert!(written.is_empty(), "a file that does not exist yet is an empty list");
        written.record(Path::new("/one.brokkr"));
        written.record(Path::new("/two.brokkr"));

        let read = Recent::load_from(Some(file.clone()));
        assert_eq!(
            read.paths(),
            [PathBuf::from("/two.brokkr"), PathBuf::from("/one.brokkr")],
            "the most recently recorded should come back first"
        );

        // And recording one it already has moves it up rather than repeating it.
        let mut again = read;
        again.record(Path::new("/one.brokkr"));
        assert_eq!(
            Recent::load_from(Some(file)).paths(),
            [PathBuf::from("/one.brokkr"), PathBuf::from("/two.brokkr")]
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A list with nowhere to write must not panic or invent a location. This
    /// is what stops a test run touching the real file.
    #[test]
    fn a_list_with_no_file_silently_does_not_persist() {
        let mut orphan = Recent::load_from(None);
        orphan.record(Path::new("/somewhere.brokkr"));
        assert_eq!(
            orphan.paths(),
            [PathBuf::from("/somewhere.brokkr")],
            "it still tracks in memory"
        );
        orphan.save();
    }
}
