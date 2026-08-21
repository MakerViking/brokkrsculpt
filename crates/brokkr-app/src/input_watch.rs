// SPDX-License-Identifier: AGPL-3.0-only

//! Scanning `/dev/input` for devices worth reading.
//!
//! The stylus and the SpaceMouse are found the same way: walk the event nodes,
//! open each one, ask whether its capabilities make it interesting, and hand
//! the interesting ones to a reader thread. Rescan on a timer, so hardware
//! switched on after the application still works — a pen display routinely is.
//!
//! This is one module rather than a copy per device because the part that is
//! easy to get wrong is the bookkeeping, not the capability test. Two of the
//! rules below were paid for once already and must not be lost:
//!
//! * A verdict of "not interesting" is **remembered**. There are typically
//!   thirty event nodes on a desktop and almost none of them will ever become
//!   a tablet, so without this the scanner reopens every device on the machine
//!   twice a second for the life of the process.
//! * That verdict is **forgotten when the node goes away**, because the same
//!   `eventN` name can come back as different hardware.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use evdev::Device;

/// What a watcher does with the devices it finds.
pub trait Adopter: Send + Sync + 'static {
    /// Whether this device's capabilities make it worth reading.
    ///
    /// Asked once per node, and the answer is cached, so it must depend only
    /// on what the device *is* and never on changing state.
    fn wants(&self, device: &Device) -> bool;

    /// Take ownership of a wanted device and start reading it.
    ///
    /// `false` means the attempt failed rather than that the device was
    /// unsuitable, so the node is left to be tried again on the next scan.
    fn adopt(&self, path: &Path, device: Device) -> bool;

    /// Whether a device this adopter took is still being read.
    ///
    /// A reader that has exited releases the node to be picked up again, which
    /// is what makes unplugging and replugging work without a restart.
    fn still_reading(&self, path: &Path) -> bool;

    /// Record whether the scan could open nothing at all.
    ///
    /// Only that case says anything about the user's group membership. A
    /// single unreadable device among many readable ones is ordinary and must
    /// not be reported as a permissions problem.
    fn set_permission_denied(&self, denied: bool);
}

/// Every `/dev/input/event*` node, unsorted.
///
/// Shared with the diagnostic reports, which list the same set the scanner
/// walks so that "the report does not show my device" and "the scanner does
/// not find my device" can never disagree.
pub fn event_nodes() -> Vec<PathBuf> {
    std::fs::read_dir("/dev/input")
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name().is_some_and(|name| name.to_string_lossy().starts_with("event"))
        })
        .collect()
}

/// Spawn a thread that scans `/dev/input` every `rescan` until the process ends.
pub fn spawn(thread_name: &str, rescan: Duration, adopter: impl Adopter) {
    let name = thread_name.to_string();
    std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || scan_loop(rescan, adopter))
        .map_err(|error| log::warn!("could not start the {name} scanner: {error}"))
        .ok();
}

fn scan_loop(rescan: Duration, adopter: impl Adopter) {
    let mut open: HashSet<PathBuf> = HashSet::new();
    let mut rejected: HashSet<PathBuf> = HashSet::new();

    loop {
        // Forget devices whose reader has exited, so hardware that is
        // unplugged and plugged back in is picked up again.
        open.retain(|path| adopter.still_reading(path));

        let mut denied = false;
        let mut anything_readable = false;
        let mut present: HashSet<PathBuf> = HashSet::new();

        for path in event_nodes() {
            present.insert(path.clone());
            if open.contains(&path) || rejected.contains(&path) {
                anything_readable = true;
                continue;
            }

            match Device::open(&path) {
                Ok(device) => {
                    anything_readable = true;
                    if !adopter.wants(&device) {
                        rejected.insert(path);
                        continue;
                    }
                    if adopter.adopt(&path, device) {
                        open.insert(path);
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                    denied = true;
                }
                Err(_) => {}
            }
        }

        // A node that has gone away may come back as different hardware under
        // the same name, so its verdict must not outlive it.
        rejected.retain(|path| present.contains(path));

        adopter.set_permission_denied(denied && !anything_readable);

        std::thread::sleep(rescan);
    }
}
