// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stylus pressure.
//!
//! # Why this reads the kernel directly
//!
//! Nothing in the windowing stack can supply pen pressure. iced 0.14's
//! `touch::Event` carries only a position and drops winit's `force` field, and
//! winit 0.30 never had force for pens at all, only for touch. Short of
//! forking iced, the pressure has to come from somewhere else.
//!
//! It comes from evdev, the kernel's input interface. That sits below the
//! display server, so a single code path covers X11, XWayland and Wayland, and
//! it works with any tablet the kernel has a driver for. No vendor list: a
//! Huion, a Wacom and an XP-Pen all publish `ABS_PRESSURE` on a device that
//! also reports `BTN_TOOL_PEN`, and that pair is the whole detection rule.
//!
//! # How it fits with the pointer
//!
//! The tablet driver already turns pen contact into ordinary pointer motion and
//! button presses, so iced sees a normal drag and the application needs no
//! separate positioning path. This module only answers "how hard is the pen
//! being pressed right now", which the sculpt loop samples when it builds a
//! stamp. Position and pressure come from the same physical event stream
//! microseconds apart, so sampling the latest value is accurate enough.
//!
//! # Permissions
//!
//! Reading `/dev/input/event*` needs membership of the `input` group on most
//! distributions. Without it the devices are simply invisible and the
//! application falls back to full pressure, which is exactly how it behaves for
//! a mouse. [`Tablet::diagnosis`] explains which of those two happened, because
//! "my tablet does nothing" otherwise looks identical to "I am using a mouse".

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

/// How long after the last pen event the pen is still considered present.
///
/// Some drivers do not send a clean `BTN_TOOL_PEN` release when the device is
/// unplugged mid stroke. Without this the application would keep believing a
/// pen is hovering and scale every mouse stroke by a stale pressure.
const PEN_TIMEOUT_MS: u64 = 1_500;

/// What the pen is doing right now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenState {
    /// A stylus is in range of a tablet, so its pressure is meaningful.
    pub in_proximity: bool,
    /// Normalised to 0 through 1 using the device's own reported range.
    pub pressure: f32,
}

impl PenState {
    /// What a mouse looks like: no pen, so strokes run at full strength.
    pub const NONE: Self = Self { in_proximity: false, pressure: 0.0 };
}

/// A tablet the application has found and is listening to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabletDevice {
    pub name: String,
    pub path: String,
    /// Raw units at full press. Wildly device specific: 8191 on a Huion Kamvas,
    /// 2047 or 4095 on various Wacom tablets, 1023 on older hardware. Shown in
    /// the interface because it is the quickest way to tell a working tablet
    /// from a misdetected one.
    pub pressure_max: i32,
}

/// Why no pressure is arriving, for the interface to explain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagnosis {
    /// A tablet is open and being read.
    Listening,
    /// Nothing that looks like a stylus is connected.
    NoTabletFound,
    /// Devices exist but could not be opened. Almost always group membership.
    PermissionDenied,
    /// This platform has no implementation yet.
    Unsupported,
}

impl Diagnosis {
    pub fn explain(self) -> &'static str {
        match self {
            Diagnosis::Listening => "listening",
            Diagnosis::NoTabletFound => "no tablet found, using full pressure",
            Diagnosis::PermissionDenied => {
                "cannot read /dev/input, add your user to the input group"
            }
            Diagnosis::Unsupported => "pen pressure is Linux only so far",
        }
    }
}

/// State shared with the reader threads.
#[derive(Debug)]
struct Shared {
    /// Latest normalised pressure, as `f32` bits.
    pressure: AtomicU32,
    /// Highest pressure seen since the last reset, so a user can confirm their
    /// tablet reaches full range without having to watch a number flicker.
    peak: AtomicU32,
    in_proximity: AtomicBool,
    /// Milliseconds since `started` at the last pen event.
    last_event_ms: AtomicU64,
    started: Instant,
    devices: Mutex<Vec<TabletDevice>>,
    permission_denied: AtomicBool,
}

impl Shared {
    fn new() -> Self {
        Self {
            pressure: AtomicU32::new(0),
            peak: AtomicU32::new(0),
            in_proximity: AtomicBool::new(false),
            last_event_ms: AtomicU64::new(0),
            started: Instant::now(),
            devices: Mutex::new(Vec::new()),
            permission_denied: AtomicBool::new(false),
        }
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    fn touch(&self) {
        self.last_event_ms.store(self.now_ms(), Ordering::Relaxed);
    }

    fn set_pressure(&self, value: f32) {
        self.pressure.store(value.to_bits(), Ordering::Relaxed);
        self.peak.fetch_max(value.to_bits(), Ordering::Relaxed);
        self.touch();
    }

    fn set_proximity(&self, present: bool) {
        self.in_proximity.store(present, Ordering::Relaxed);
        if !present {
            self.pressure.store(0.0f32.to_bits(), Ordering::Relaxed);
        }
        self.touch();
    }

    fn state(&self) -> PenState {
        self.state_at(self.now_ms())
    }

    /// Split out from `state` so the timeout can be tested without waiting for
    /// a wall clock, and without the result depending on how long the process
    /// happened to have been running.
    fn state_at(&self, now_ms: u64) -> PenState {
        let last = self.last_event_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) > PEN_TIMEOUT_MS {
            return PenState::NONE;
        }
        PenState {
            in_proximity: self.in_proximity.load(Ordering::Relaxed),
            pressure: f32::from_bits(self.pressure.load(Ordering::Relaxed)),
        }
    }
}

/// Watches for tablets and tracks the current pen pressure.
///
/// Cheap to clone: everything lives behind an `Arc` shared with the reader
/// threads.
#[derive(Debug, Clone)]
pub struct Tablet {
    shared: Arc<Shared>,
}

impl Tablet {
    /// Start watching. Never fails: a machine with no tablet, or no permission
    /// to read one, simply reports [`PenState::NONE`] for ever.
    pub fn start() -> Self {
        let shared = Arc::new(Shared::new());
        backend::spawn(Arc::clone(&shared));
        Self { shared }
    }

    /// A tablet that will never report anything, so tests do not go looking
    /// through `/dev/input` or spawn reader threads.
    #[cfg(test)]
    pub fn inert() -> Self {
        Self { shared: Arc::new(Shared::new()) }
    }

    pub fn state(&self) -> PenState {
        self.shared.state()
    }

    pub fn devices(&self) -> Vec<TabletDevice> {
        self.shared.devices.lock().expect("tablet state poisoned").clone()
    }

    /// Highest pressure seen since the last reset.
    pub fn peak(&self) -> f32 {
        f32::from_bits(self.shared.peak.load(Ordering::Relaxed))
    }

    pub fn reset_peak(&self) {
        self.shared.peak.store(0, Ordering::Relaxed);
    }

    pub fn diagnosis(&self) -> Diagnosis {
        if !backend::SUPPORTED {
            return Diagnosis::Unsupported;
        }
        if !self.devices().is_empty() {
            return Diagnosis::Listening;
        }
        if self.shared.permission_denied.load(Ordering::Relaxed) {
            return Diagnosis::PermissionDenied;
        }
        Diagnosis::NoTabletFound
    }

    /// The pressure to apply to a stamp right now.
    ///
    /// Returns 1 whenever there is no pen, so a mouse sculpts at full strength.
    /// That fallback is the whole reason proximity is tracked separately from
    /// pressure: a hovering or absent pen reads zero, and treating that as the
    /// stroke strength would make the application appear to do nothing at all.
    pub fn stamp_pressure(&self, enabled: bool, curve: f32) -> f32 {
        if !enabled {
            return 1.0;
        }
        let pen = self.state();
        if !pen.in_proximity {
            return 1.0;
        }
        shape(pen.pressure, curve)
    }
}

/// Map raw pressure through a response curve.
///
/// An exponent below 1 makes light touches bite harder, which is what most
/// people want on a tablet whose raw response is close to linear in force but
/// not in feel. Above 1 gives finer control at the light end.
pub fn shape(raw: f32, curve: f32) -> f32 {
    raw.clamp(0.0, 1.0).powf(curve.max(0.05))
}

/// Normalise a raw axis value against the range the device reports.
///
/// Devices differ by more than an order of magnitude here, so nothing may
/// assume a fixed maximum. A device reporting a degenerate range is treated as
/// full pressure rather than dividing by zero.
fn normalise(value: i32, minimum: i32, maximum: i32) -> f32 {
    if maximum <= minimum {
        return 1.0;
    }
    ((value - minimum) as f32 / (maximum - minimum) as f32).clamp(0.0, 1.0)
}

/// Print what every input device looks like to the scanner, and why each was
/// accepted or rejected.
///
/// A tablet that does not work is otherwise a black box: the interface can say
/// "no tablet found", but not whether the device is missing, unreadable, or
/// present but reporting no pressure because it is in mouse mode. This answers
/// that without needing a debugger or a second tool.
pub fn report() -> String {
    backend::report()
}

#[cfg(target_os = "linux")]
mod backend {
    use super::{Shared, TabletDevice, normalise};
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use evdev::{AbsoluteAxisCode, Device, EventType, KeyCode};

    pub const SUPPORTED: bool = true;

    /// How often to look for a tablet that was plugged in after startup.
    ///
    /// A pen display is routinely switched on after the application, so
    /// scanning once at startup would mean restarting to use it.
    const RESCAN: Duration = Duration::from_secs(2);

    /// A device is a stylus if it can report pressure and identifies a pen tool.
    ///
    /// Pressure alone is not enough: plenty of touchscreens report
    /// `ABS_PRESSURE` for finger contact, and some pressure sensitive pads do
    /// too. `BTN_TOOL_PEN` is what the kernel uses to mean "this is a pen", and
    /// every tablet driver sets it.
    fn is_stylus(device: &Device) -> bool {
        let has_pressure = device
            .supported_absolute_axes()
            .is_some_and(|axes| axes.contains(AbsoluteAxisCode::ABS_PRESSURE));
        let has_pen =
            device.supported_keys().is_some_and(|keys| keys.contains(KeyCode::BTN_TOOL_PEN));
        has_pressure && has_pen
    }

    pub fn report() -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out, "Input devices, as the tablet scanner sees them.\n");

        let mut entries: Vec<PathBuf> = std::fs::read_dir("/dev/input")
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name().is_some_and(|name| name.to_string_lossy().starts_with("event"))
            })
            .collect();
        entries.sort();

        if entries.is_empty() {
            let _ = writeln!(out, "No /dev/input/event* nodes at all.");
            return out;
        }

        let mut found = 0usize;
        let mut unreadable = 0usize;
        for path in &entries {
            match Device::open(path) {
                Ok(device) => {
                    let name = device.name().unwrap_or("unnamed").to_string();
                    let pressure = device
                        .supported_absolute_axes()
                        .is_some_and(|axes| axes.contains(AbsoluteAxisCode::ABS_PRESSURE));
                    let pen = device
                        .supported_keys()
                        .is_some_and(|keys| keys.contains(KeyCode::BTN_TOOL_PEN));

                    if !pressure && !pen {
                        // Keyboards, mice and gamepads. Listing every one buries
                        // the interesting lines.
                        continue;
                    }

                    let range = device
                        .get_absinfo()
                        .ok()
                        .and_then(|mut axes| {
                            axes.find(|(axis, _)| *axis == AbsoluteAxisCode::ABS_PRESSURE)
                        })
                        .map(|(_, info)| format!("{} to {}", info.minimum(), info.maximum()))
                        .unwrap_or_else(|| "unknown".into());

                    let verdict = match (pressure, pen) {
                        (true, true) => {
                            found += 1;
                            "STYLUS, pressure will be read from this"
                        }
                        (true, false) => {
                            "ignored: reports pressure but no pen tool, so it is a touch device"
                        }
                        (false, true) => "ignored: identifies a pen but reports no pressure",
                        (false, false) => unreachable!("filtered above"),
                    };
                    let _ = writeln!(
                        out,
                        "{}\n  {name}\n  pressure range {range}\n  {verdict}\n",
                        path.display()
                    );
                }
                Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                    unreadable += 1;
                }
                Err(_) => {}
            }
        }

        let _ = writeln!(out, "{found} stylus device(s) usable.");
        if unreadable > 0 {
            let _ = writeln!(
                out,
                "{unreadable} device(s) could not be opened. If your tablet is missing above, \n\
                 add your user to the input group and log back in:\n\
                 \n    sudo usermod -aG input $USER\n"
            );
        }
        if found == 0 && unreadable == 0 {
            let _ = writeln!(
                out,
                "No stylus found. Check the tablet is plugged in and not in a mouse only mode,\n\
                 and that a driver bound to it: look for hid-uclogic (Huion, XP-Pen) or wacom\n\
                 in the output of lsmod."
            );
        }
        out
    }

    pub fn spawn(shared: Arc<Shared>) {
        std::thread::Builder::new()
            .name("brokkr-tablet-scan".into())
            .spawn(move || scan_loop(shared))
            .map_err(|error| log::warn!("could not start the tablet scanner: {error}"))
            .ok();
    }

    fn scan_loop(shared: Arc<Shared>) {
        let mut open: HashSet<PathBuf> = HashSet::new();
        // Keyboards, mice and the rest never become styluses, and there are
        // typically thirty of them. Remembering the verdict keeps the scan from
        // reopening every input device on the machine twice a second for the
        // life of the process.
        let mut rejected: HashSet<PathBuf> = HashSet::new();
        loop {
            // Forget devices whose reader has exited, so a tablet that is
            // unplugged and plugged back in is picked up again.
            {
                let live = shared.devices.lock().expect("tablet state poisoned");
                open.retain(|path| live.iter().any(|device| device.path == path.to_string_lossy()));
            }

            let mut denied = false;
            let mut anything_readable = false;
            let mut present: HashSet<PathBuf> = HashSet::new();

            for entry in std::fs::read_dir("/dev/input").into_iter().flatten().flatten() {
                let path = entry.path();
                if !path.file_name().is_some_and(|name| name.to_string_lossy().starts_with("event"))
                {
                    continue;
                }
                present.insert(path.clone());
                if open.contains(&path) || rejected.contains(&path) {
                    anything_readable = true;
                    continue;
                }

                match Device::open(&path) {
                    Ok(device) => {
                        anything_readable = true;
                        if !is_stylus(&device) {
                            rejected.insert(path);
                            continue;
                        }
                        if let Some(info) = adopt(&shared, &path, device) {
                            open.insert(path);
                            log::info!(
                                "tablet: reading {} at {} with {} pressure levels",
                                info.name,
                                info.path,
                                info.pressure_max
                            );
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                        denied = true;
                    }
                    Err(_) => {}
                }
            }

            // A node that has gone away may come back as different hardware
            // under the same name, so its verdict must not be remembered.
            rejected.retain(|path| present.contains(path));

            // Only report a permissions problem when nothing at all could be
            // opened. A single unreadable device among many readable ones is
            // normal and says nothing about the user's groups.
            shared.permission_denied.store(denied && !anything_readable, Ordering::Relaxed);

            std::thread::sleep(RESCAN);
        }
    }

    /// Take ownership of a stylus device and start a thread reading it.
    fn adopt(shared: &Arc<Shared>, path: &Path, mut device: Device) -> Option<TabletDevice> {
        let (minimum, maximum) = device
            .get_absinfo()
            .ok()?
            .find(|(axis, _)| *axis == AbsoluteAxisCode::ABS_PRESSURE)
            .map(|(_, info)| (info.minimum(), info.maximum()))?;

        let info = TabletDevice {
            name: device.name().unwrap_or("unnamed tablet").to_string(),
            path: path.to_string_lossy().into_owned(),
            pressure_max: maximum,
        };

        shared.devices.lock().expect("tablet state poisoned").push(info.clone());

        let shared = Arc::clone(shared);
        let registered = info.clone();
        std::thread::Builder::new()
            .name("brokkr-tablet-read".into())
            .spawn(move || {
                read_loop(&shared, &mut device, minimum, maximum);
                // Unplugged, or the read failed. Drop the registration so the
                // scanner can pick the device up again if it comes back.
                shared
                    .devices
                    .lock()
                    .expect("tablet state poisoned")
                    .retain(|device| *device != registered);
                shared.set_proximity(false);
                log::info!("tablet: {} disconnected", registered.name);
            })
            .ok()?;

        Some(info)
    }

    fn read_loop(shared: &Arc<Shared>, device: &mut Device, minimum: i32, maximum: i32) {
        loop {
            let events = match device.fetch_events() {
                Ok(events) => events,
                Err(error) => {
                    log::debug!("tablet read ended: {error}");
                    return;
                }
            };
            for event in events {
                match event.event_type() {
                    EventType::ABSOLUTE if event.code() == AbsoluteAxisCode::ABS_PRESSURE.0 => {
                        shared.set_pressure(normalise(event.value(), minimum, maximum));
                    }
                    EventType::KEY if event.code() == KeyCode::BTN_TOOL_PEN.0 => {
                        shared.set_proximity(event.value() != 0);
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod backend {
    use super::Shared;
    use std::sync::Arc;

    pub const SUPPORTED: bool = false;

    /// Windows would use Pointer Input or Wintab, macOS `NSEvent` pressure.
    /// Both are milestones away, and the application degrades to full pressure
    /// in the meantime, which is what it already does for a mouse.
    pub fn spawn(_shared: Arc<Shared>) {}

    pub fn report() -> String {
        "Pen pressure is implemented for Linux only so far.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_is_normalised_against_the_device_range() {
        // The ranges real tablets actually report.
        for maximum in [1023, 2047, 4095, 8191, 65535] {
            assert_eq!(normalise(0, 0, maximum), 0.0, "zero should be zero at max {maximum}");
            assert_eq!(normalise(maximum, 0, maximum), 1.0, "full should be one at max {maximum}");
            let half = normalise(maximum / 2, 0, maximum);
            assert!((half - 0.5).abs() < 0.001, "half of {maximum} came out at {half}");
        }
    }

    #[test]
    fn a_non_zero_minimum_is_honoured() {
        assert_eq!(normalise(100, 100, 1100), 0.0);
        assert_eq!(normalise(600, 100, 1100), 0.5);
        assert_eq!(normalise(1100, 100, 1100), 1.0);
    }

    #[test]
    fn out_of_range_values_are_clamped_rather_than_trusted() {
        assert_eq!(normalise(-50, 0, 1000), 0.0);
        assert_eq!(normalise(9999, 0, 1000), 1.0);
    }

    #[test]
    fn a_degenerate_range_reads_as_full_pressure() {
        // Better to sculpt at full strength than to divide by zero or leave the
        // user with a brush that silently does nothing.
        assert_eq!(normalise(0, 0, 0), 1.0);
        assert_eq!(normalise(5, 10, 3), 1.0);
    }

    #[test]
    fn the_response_curve_keeps_its_end_points() {
        for curve in [0.4, 1.0, 2.5] {
            assert_eq!(shape(0.0, curve), 0.0);
            assert_eq!(shape(1.0, curve), 1.0);
        }
        // Below one lifts the middle, above one lowers it.
        assert!(shape(0.5, 0.5) > 0.5);
        assert!((shape(0.5, 1.0) - 0.5).abs() < 1.0e-6);
        assert!(shape(0.5, 2.0) < 0.5);
    }

    #[test]
    fn the_response_curve_never_leaves_the_unit_range() {
        for curve in [0.05, 0.5, 1.0, 3.0] {
            for step in 0..=20 {
                let value = shape(step as f32 / 20.0, curve);
                assert!((0.0..=1.0).contains(&value), "shape({step}, {curve}) was {value}");
            }
        }
        // A zero or negative exponent would make everything full pressure.
        assert!(shape(0.5, 0.0).is_finite());
    }

    #[test]
    fn a_mouse_sculpts_at_full_strength() {
        // The failure this guards against is the worst possible one: a user
        // with no tablet finding that the brush does nothing.
        let tablet = Tablet::inert();
        assert_eq!(tablet.state(), PenState::NONE);
        assert_eq!(tablet.stamp_pressure(true, 1.0), 1.0);
        assert_eq!(tablet.stamp_pressure(false, 1.0), 1.0);
    }

    #[test]
    fn a_pen_in_proximity_drives_the_stamp_pressure() {
        let tablet = Tablet::inert();
        tablet.shared.set_proximity(true);
        tablet.shared.set_pressure(0.25);

        assert_eq!(tablet.stamp_pressure(true, 1.0), 0.25);
        // Turning the feature off must ignore the pen entirely.
        assert_eq!(tablet.stamp_pressure(false, 1.0), 1.0);
    }

    #[test]
    fn lifting_the_pen_out_of_range_returns_to_full_strength() {
        let tablet = Tablet::inert();
        tablet.shared.set_proximity(true);
        tablet.shared.set_pressure(0.5);
        assert_eq!(tablet.stamp_pressure(true, 1.0), 0.5);

        tablet.shared.set_proximity(false);
        assert_eq!(tablet.state().pressure, 0.0, "leaving proximity must clear the pressure");
        assert_eq!(tablet.stamp_pressure(true, 1.0), 1.0);
    }

    #[test]
    fn a_pen_that_stops_reporting_is_forgotten() {
        // Some drivers do not send a clean release when the tablet is
        // unplugged. Without the timeout every later mouse stroke would be
        // scaled by whatever pressure the pen last had.
        let tablet = Tablet::inert();
        tablet.shared.set_proximity(true);
        tablet.shared.set_pressure(0.5);

        let last = tablet.shared.last_event_ms.load(Ordering::Relaxed);
        // Still present a moment later.
        assert!(tablet.shared.state_at(last + PEN_TIMEOUT_MS).in_proximity);
        // Gone once the timeout has passed with nothing further from the pen.
        assert_eq!(tablet.shared.state_at(last + PEN_TIMEOUT_MS + 1), PenState::NONE);
    }

    #[test]
    fn the_peak_records_the_hardest_press_seen() {
        let tablet = Tablet::inert();
        tablet.shared.set_pressure(0.3);
        tablet.shared.set_pressure(0.9);
        tablet.shared.set_pressure(0.4);
        assert_eq!(tablet.peak(), 0.9);

        tablet.reset_peak();
        assert_eq!(tablet.peak(), 0.0);
    }

    #[test]
    fn an_inert_tablet_reports_why_it_is_silent() {
        let tablet = Tablet::inert();
        let diagnosis = tablet.diagnosis();
        if backend::SUPPORTED {
            assert_eq!(diagnosis, Diagnosis::NoTabletFound);
        } else {
            assert_eq!(diagnosis, Diagnosis::Unsupported);
        }
        assert!(!diagnosis.explain().is_empty());
    }
}

/// End to end tests against a synthetic tablet.
///
/// These build a real device with `uinput`, let the ordinary scanner find it
/// through `/dev/input` like any other tablet, and check that pressure comes
/// out the far end. Nothing here is a mock: the same code path runs for a Huion
/// Kamvas, and the only thing the test supplies is the hardware.
///
/// They skip when `/dev/uinput` is not writable, which is the case on most
/// build machines. A skip is printed rather than passing quietly, because a
/// green tick from a test that never ran is worse than no test.
#[cfg(all(test, target_os = "linux"))]
mod uinput_tests {
    use super::*;
    use evdev::uinput::VirtualDevice;
    use evdev::{
        AbsInfo, AbsoluteAxisCode, AttributeSet, EventType, InputEvent, KeyCode, UinputAbsSetup,
    };
    use std::time::Duration;

    /// What a Huion Kamvas 13 reports. Chosen so the test would catch a
    /// hard coded 0 to 1 or 0 to 255 assumption anywhere in the chain.
    const PRESSURE_MAX: i32 = 8191;

    fn build(name: &str, pen_tool: bool) -> Option<VirtualDevice> {
        let mut keys = AttributeSet::<KeyCode>::new();
        keys.insert(KeyCode::BTN_TOUCH);
        if pen_tool {
            keys.insert(KeyCode::BTN_TOOL_PEN);
        }

        let axis = |code, maximum| UinputAbsSetup::new(code, AbsInfo::new(0, 0, maximum, 0, 0, 0));

        VirtualDevice::builder()
            .ok()?
            .name(name)
            .with_keys(&keys)
            .ok()?
            .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_X, 1920))
            .ok()?
            .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_Y, 1080))
            .ok()?
            .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_PRESSURE, PRESSURE_MAX))
            .ok()?
            .build()
            .ok()
    }

    fn emit(device: &mut VirtualDevice, events: &[InputEvent]) {
        device.emit(events).expect("could not emit to the virtual tablet");
    }

    fn pressure_event(raw: i32) -> InputEvent {
        InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_PRESSURE.0, raw)
    }

    fn pen_tool_event(down: bool) -> InputEvent {
        InputEvent::new(EventType::KEY.0, KeyCode::BTN_TOOL_PEN.0, i32::from(down))
    }

    /// Poll until `predicate` holds, up to roughly twelve seconds. The scanner
    /// rescans every two, and udev takes a moment to apply permissions to a
    /// freshly created node.
    fn wait_for(mut predicate: impl FnMut() -> bool) -> bool {
        for _ in 0..120 {
            if predicate() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        false
    }

    #[test]
    fn a_virtual_tablet_is_found_and_its_pressure_reaches_the_brush() {
        let Some(mut stylus) = build("BrokkrSculpt test stylus", true) else {
            eprintln!(
                "skipping: cannot create a uinput device. Needs write access to /dev/uinput, \
                 usually via the input group."
            );
            return;
        };
        // A touchscreen reports pressure too but is not a stylus. If this got
        // adopted, every touch would drive the brush.
        let _touchscreen = build("BrokkrSculpt test touchscreen", false);

        let tablet = Tablet::start();

        let found = wait_for(|| {
            tablet.devices().iter().any(|device| device.name == "BrokkrSculpt test stylus")
        });
        assert!(found, "the scanner never picked up the virtual stylus");

        let devices = tablet.devices();
        let stylus_info = devices
            .iter()
            .find(|device| device.name == "BrokkrSculpt test stylus")
            .expect("just found it");
        assert_eq!(
            stylus_info.pressure_max, PRESSURE_MAX,
            "the device's own pressure range was not read from it"
        );
        assert!(
            !devices.iter().any(|device| device.name == "BrokkrSculpt test touchscreen"),
            "a device with pressure but no pen tool was mistaken for a stylus"
        );
        assert_eq!(tablet.diagnosis(), Diagnosis::Listening);

        // Pen enters range and presses to half of its range.
        emit(&mut stylus, &[pen_tool_event(true), pressure_event(PRESSURE_MAX / 2)]);
        assert!(
            wait_for(|| {
                let pen = tablet.state();
                pen.in_proximity && (pen.pressure - 0.5).abs() < 0.01
            }),
            "half pressure never arrived, got {:?}",
            tablet.state()
        );
        // The number the brush would actually use.
        assert!((tablet.stamp_pressure(true, 1.0) - 0.5).abs() < 0.01);

        // Pressed all the way down.
        emit(&mut stylus, &[pressure_event(PRESSURE_MAX)]);
        assert!(
            wait_for(|| tablet.state().pressure > 0.99),
            "full pressure never arrived, got {:?}",
            tablet.state()
        );
        assert!(tablet.peak() > 0.99, "the peak did not record the hardest press");

        // A light touch, which is the end of the range that matters most for
        // sculpting and the one a wrong normalisation would ruin.
        emit(&mut stylus, &[pressure_event(PRESSURE_MAX / 100)]);
        assert!(
            wait_for(|| tablet.state().pressure < 0.02 && tablet.state().pressure > 0.0),
            "a light touch did not come through as a small pressure, got {:?}",
            tablet.state()
        );

        // Pen lifted out of range: the brush must go back to full strength so a
        // mouse still works.
        emit(&mut stylus, &[pen_tool_event(false)]);
        assert!(
            wait_for(|| !tablet.state().in_proximity),
            "the pen never left proximity, got {:?}",
            tablet.state()
        );
        assert_eq!(
            tablet.stamp_pressure(true, 1.0),
            1.0,
            "with the pen away the brush must run at full strength"
        );

        // Unplugging must deregister the device rather than leaving a ghost.
        drop(stylus);
        assert!(
            wait_for(|| { !tablet.devices().iter().any(|d| d.name == "BrokkrSculpt test stylus") }),
            "the device was still listed after being removed"
        );
    }
}
