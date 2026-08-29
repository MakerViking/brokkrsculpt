// SPDX-License-Identifier: AGPL-3.0-only

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

use glam::Vec2;

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
    ///
    /// True for either end of the pen. The eraser and the tip are reported as
    /// separate tools by the kernel and are never in range at the same time, so
    /// checking only for the tip would make every eraser stroke look like a
    /// mouse and quietly run at full pressure.
    pub in_proximity: bool,
    /// Normalised to 0 through 1 using the device's own reported range.
    pub pressure: f32,
    /// The eraser end of the stylus is the one in range.
    pub eraser: bool,
    /// Tilt away from vertical, each axis normalised to -1 through 1 against
    /// the device's own reported range. Zero is upright.
    pub tilt: Vec2,
}

impl PenState {
    /// What a mouse looks like: no pen, so strokes run at full strength.
    pub const NONE: Self =
        Self { in_proximity: false, pressure: 0.0, eraser: false, tilt: Vec2::ZERO };
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
    /// Whether the device reports tilt. Plenty of pens do not, and saying so is
    /// better than leaving the user to wonder why leaning does nothing.
    pub has_tilt: bool,
    /// Whether the device has an eraser end.
    pub has_eraser: bool,
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
    /// The two pen ends, tracked separately because proximity is either of
    /// them and the kernel reports them as different tools.
    tool_tip: AtomicBool,
    tool_eraser: AtomicBool,
    tilt_x: AtomicU32,
    tilt_y: AtomicU32,
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
            tool_tip: AtomicBool::new(false),
            tool_eraser: AtomicBool::new(false),
            tilt_x: AtomicU32::new(0),
            tilt_y: AtomicU32::new(0),
            last_event_ms: AtomicU64::new(0),
            started: Instant::now(),
            devices: Mutex::new(Vec::new()),
            permission_denied: AtomicBool::new(false),
        }
    }

    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    // Written only by the evdev backend, which is Linux only. The fields and
    // the readers are shared, so these stay compiled everywhere rather than
    // being cfg'd into two shapes -- but with no producer on Windows or macOS
    // they are dead there, and `-D warnings` is right to say so. Allowed with
    // a reason rather than silenced: when those platforms grow a Pointer
    // Input, Wintab or IOKit backend, it calls exactly these.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn touch(&self) {
        self.last_event_ms.store(self.now_ms(), Ordering::Relaxed);
    }

    fn set_pressure(&self, value: f32) {
        self.pressure.store(value.to_bits(), Ordering::Relaxed);
        self.peak.fetch_max(value.to_bits(), Ordering::Relaxed);
        self.touch();
    }

    fn set_tilt(&self, tilt: Vec2) {
        self.tilt_x.store(tilt.x.to_bits(), Ordering::Relaxed);
        self.tilt_y.store(tilt.y.to_bits(), Ordering::Relaxed);
        self.touch();
    }

    /// Record which end of the stylus is in range.
    ///
    /// Proximity is either end. Pressure and tilt are cleared when both are
    /// gone, so a lifted pen cannot leave a stale value scaling later strokes.
    fn set_tool(&self, tip: Option<bool>, eraser: Option<bool>) {
        if let Some(down) = tip {
            self.tool_tip.store(down, Ordering::Relaxed);
        }
        if let Some(down) = eraser {
            self.tool_eraser.store(down, Ordering::Relaxed);
        }
        let present =
            self.tool_tip.load(Ordering::Relaxed) || self.tool_eraser.load(Ordering::Relaxed);
        self.in_proximity.store(present, Ordering::Relaxed);
        if !present {
            self.pressure.store(0.0f32.to_bits(), Ordering::Relaxed);
            self.tilt_x.store(0.0f32.to_bits(), Ordering::Relaxed);
            self.tilt_y.store(0.0f32.to_bits(), Ordering::Relaxed);
        }
        self.touch();
    }

    /// Convenience for tests and for dropping every tool at once.
    fn set_proximity(&self, present: bool) {
        self.set_tool(Some(present), Some(false));
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
            eraser: self.tool_eraser.load(Ordering::Relaxed),
            tilt: Vec2::new(
                f32::from_bits(self.tilt_x.load(Ordering::Relaxed)),
                f32::from_bits(self.tilt_y.load(Ordering::Relaxed)),
            ),
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

    /// Feed a pen state in directly, so code that consumes the pen can be
    /// tested without a device.
    #[cfg(test)]
    pub fn simulate(&self, state: PenState) {
        // Which end is in range only means anything while the pen is in range
        // at all, or an "eraser away" state would report as present.
        self.shared.set_tool(
            Some(state.in_proximity && !state.eraser),
            Some(state.in_proximity && state.eraser),
        );
        if state.in_proximity {
            self.shared.set_pressure(state.pressure);
            self.shared.set_tilt(state.tilt);
        }
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
// Written only by the evdev backend, which is Linux only. The fields and
// the readers are shared, so these stay compiled everywhere rather than
// being cfg'd into two shapes -- but with no producer on Windows or macOS
// they are dead there, and `-D warnings` is right to say so. Allowed with
// a reason rather than silenced: when those platforms grow a Pointer
// Input, Wintab or IOKit backend, it calls exactly these.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn normalise(value: i32, minimum: i32, maximum: i32) -> f32 {
    if maximum <= minimum {
        return 1.0;
    }
    ((value - minimum) as f32 / (maximum - minimum) as f32).clamp(0.0, 1.0)
}

/// Normalise a signed axis such as tilt to -1 through 1.
///
/// Scaled against the larger half of the range rather than mapped across the
/// whole of it, so that an upright pen reading zero comes out as exactly zero
/// even on a device whose range is slightly lopsided, like -64 to 63.
// Written only by the evdev backend, which is Linux only. The fields and
// the readers are shared, so these stay compiled everywhere rather than
// being cfg'd into two shapes -- but with no producer on Windows or macOS
// they are dead there, and `-D warnings` is right to say so. Allowed with
// a reason rather than silenced: when those platforms grow a Pointer
// Input, Wintab or IOKit backend, it calls exactly these.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn normalise_signed(value: i32, minimum: i32, maximum: i32) -> f32 {
    let extent = minimum.abs().max(maximum.abs());
    if extent == 0 {
        return 0.0;
    }
    (value as f32 / extent as f32).clamp(-1.0, 1.0)
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
    use super::{Shared, TabletDevice, normalise, normalise_signed};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use evdev::{AbsInfo, AbsoluteAxisCode, Device, EventType, KeyCode};
    use glam::Vec2;

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
    /// Range of an absolute axis, if the device has it.
    fn axis_range(device: &Device, axis: AbsoluteAxisCode) -> Option<(i32, i32)> {
        device
            .get_absinfo()
            .ok()?
            .find(|(code, _)| *code == axis)
            .map(|(_, info): (_, AbsInfo)| (info.minimum(), info.maximum()))
    }

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

        // The same set the scanner walks, so the report and the scanner can
        // never disagree about which devices exist.
        let mut entries: Vec<PathBuf> = crate::input_watch::event_nodes();
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

                    let mut extras = Vec::new();
                    if device
                        .supported_absolute_axes()
                        .is_some_and(|axes| axes.contains(AbsoluteAxisCode::ABS_TILT_X))
                    {
                        extras.push("tilt");
                    }
                    if device
                        .supported_keys()
                        .is_some_and(|keys| keys.contains(KeyCode::BTN_TOOL_RUBBER))
                    {
                        extras.push("eraser");
                    }

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
                    let also = if extras.is_empty() {
                        String::new()
                    } else {
                        format!(", {}", extras.join(", "))
                    };
                    let _ = writeln!(
                        out,
                        "{}\n  {name}\n  pressure range {range}{also}\n  {verdict}\n",
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
        crate::input_watch::spawn("brokkr-tablet-scan", RESCAN, Stylus { shared });
    }

    /// The tablet's half of the shared `/dev/input` scanner. The scanning,
    /// caching and permission bookkeeping live in [`crate::input_watch`],
    /// because the SpaceMouse needs exactly the same thing.
    struct Stylus {
        shared: Arc<Shared>,
    }

    impl crate::input_watch::Adopter for Stylus {
        fn wants(&self, device: &Device) -> bool {
            is_stylus(device)
        }

        fn adopt(&self, path: &Path, device: Device) -> bool {
            let Some(info) = adopt(&self.shared, path, device) else {
                return false;
            };
            log::info!(
                "tablet: reading {} at {} with {} pressure levels{}{}",
                info.name,
                info.path,
                info.pressure_max,
                if info.has_tilt { ", tilt" } else { "" },
                if info.has_eraser { ", eraser" } else { "" },
            );
            true
        }

        fn still_reading(&self, path: &Path) -> bool {
            let live = self.shared.devices.lock().expect("tablet state poisoned");
            live.iter().any(|device| device.path == path.to_string_lossy())
        }

        fn set_permission_denied(&self, denied: bool) {
            self.shared.permission_denied.store(denied, Ordering::Relaxed);
        }
    }

    /// Everything about one device's axes that the reader needs.
    #[derive(Debug, Clone, Copy)]
    struct Axes {
        pressure: (i32, i32),
        tilt_x: Option<(i32, i32)>,
        tilt_y: Option<(i32, i32)>,
    }

    /// Take ownership of a stylus device and start a thread reading it.
    fn adopt(shared: &Arc<Shared>, path: &Path, mut device: Device) -> Option<TabletDevice> {
        let axes = Axes {
            pressure: axis_range(&device, AbsoluteAxisCode::ABS_PRESSURE)?,
            tilt_x: axis_range(&device, AbsoluteAxisCode::ABS_TILT_X),
            tilt_y: axis_range(&device, AbsoluteAxisCode::ABS_TILT_Y),
        };

        let info = TabletDevice {
            name: device.name().unwrap_or("unnamed tablet").to_string(),
            path: path.to_string_lossy().into_owned(),
            pressure_max: axes.pressure.1,
            has_tilt: axes.tilt_x.is_some() && axes.tilt_y.is_some(),
            has_eraser: device
                .supported_keys()
                .is_some_and(|keys| keys.contains(KeyCode::BTN_TOOL_RUBBER)),
        };

        shared.devices.lock().expect("tablet state poisoned").push(info.clone());

        let shared = Arc::clone(shared);
        let registered = info.clone();
        std::thread::Builder::new()
            .name("brokkr-tablet-read".into())
            .spawn(move || {
                read_loop(&shared, &mut device, axes);
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

    fn read_loop(shared: &Arc<Shared>, device: &mut Device, axes: Axes) {
        // Tilt arrives one axis at a time, so the other half has to be
        // remembered in order to publish a complete vector.
        let mut tilt = Vec2::ZERO;

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
                        let (minimum, maximum) = axes.pressure;
                        shared.set_pressure(normalise(event.value(), minimum, maximum));
                    }
                    EventType::ABSOLUTE if event.code() == AbsoluteAxisCode::ABS_TILT_X.0 => {
                        if let Some((minimum, maximum)) = axes.tilt_x {
                            tilt.x = normalise_signed(event.value(), minimum, maximum);
                            shared.set_tilt(tilt);
                        }
                    }
                    EventType::ABSOLUTE if event.code() == AbsoluteAxisCode::ABS_TILT_Y.0 => {
                        if let Some((minimum, maximum)) = axes.tilt_y {
                            tilt.y = normalise_signed(event.value(), minimum, maximum);
                            shared.set_tilt(tilt);
                        }
                    }
                    // The two ends of the stylus are separate tools and are
                    // never in range together, so each only updates its own.
                    EventType::KEY if event.code() == KeyCode::BTN_TOOL_PEN.0 => {
                        shared.set_tool(Some(event.value() != 0), None);
                    }
                    EventType::KEY if event.code() == KeyCode::BTN_TOOL_RUBBER.0 => {
                        shared.set_tool(None, Some(event.value() != 0));
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
    fn signed_axes_keep_zero_at_zero() {
        // A lopsided range like -64 to 63 is normal for tilt. Mapping it across
        // the whole span would leave an upright pen reporting a small lean, and
        // every stroke would drift.
        assert_eq!(normalise_signed(0, -64, 63), 0.0);
        assert_eq!(normalise_signed(-64, -64, 63), -1.0);
        assert_eq!(normalise_signed(64, -64, 63), 1.0);
        assert!((normalise_signed(32, -64, 63) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn signed_axes_clamp_and_survive_a_degenerate_range() {
        assert_eq!(normalise_signed(999, -60, 60), 1.0);
        assert_eq!(normalise_signed(-999, -60, 60), -1.0);
        assert_eq!(normalise_signed(5, 0, 0), 0.0);
    }

    #[test]
    fn the_eraser_end_counts_as_a_pen_being_present() {
        // The bug this guards against: the kernel reports the tip and the
        // eraser as separate tools that are never in range together, so
        // checking only for the tip made every eraser stroke look like a mouse
        // and silently run at full pressure.
        let tablet = Tablet::inert();
        tablet.shared.set_tool(None, Some(true));
        tablet.shared.set_pressure(0.4);

        let pen = tablet.state();
        assert!(pen.in_proximity, "the eraser end must count as a pen in range");
        assert!(pen.eraser);
        assert!(
            (tablet.stamp_pressure(true, 1.0) - 0.4).abs() < 1.0e-6,
            "eraser strokes must be pressure sensitive too"
        );
    }

    #[test]
    fn the_pen_is_only_gone_once_both_ends_are_out_of_range() {
        let tablet = Tablet::inert();
        tablet.shared.set_tool(Some(true), None);
        assert!(tablet.state().in_proximity);

        // Flipping the pen over: the tip leaves and the eraser arrives.
        tablet.shared.set_tool(None, Some(true));
        tablet.shared.set_tool(Some(false), None);
        assert!(tablet.state().in_proximity, "the eraser is still in range");
        assert!(tablet.state().eraser);

        tablet.shared.set_tool(None, Some(false));
        assert!(!tablet.state().in_proximity);
    }

    #[test]
    fn lifting_the_pen_clears_the_tilt_as_well_as_the_pressure() {
        // A stale tilt would keep steering strokes after the pen was put down.
        let tablet = Tablet::inert();
        tablet.shared.set_tool(Some(true), None);
        tablet.shared.set_tilt(Vec2::new(0.7, -0.3));
        assert_eq!(tablet.state().tilt, Vec2::new(0.7, -0.3));

        tablet.shared.set_tool(Some(false), None);
        assert_eq!(tablet.state().tilt, Vec2::ZERO);
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
    use glam::Vec2;
    use std::time::Duration;

    /// What a Huion Kamvas 13 reports. Chosen so the test would catch a
    /// hard coded 0 to 1 or 0 to 255 assumption anywhere in the chain.
    const PRESSURE_MAX: i32 = 8191;

    /// The tilt range a real tablet reports: lopsided, which is exactly the
    /// case that would leave an upright pen reporting a small lean if the
    /// normalisation mapped across the whole span.
    const TILT_MIN: i32 = -64;
    const TILT_MAX: i32 = 63;

    fn build(name: &str, pen_tool: bool) -> Option<VirtualDevice> {
        let mut keys = AttributeSet::<KeyCode>::new();
        keys.insert(KeyCode::BTN_TOUCH);
        if pen_tool {
            keys.insert(KeyCode::BTN_TOOL_PEN);
            keys.insert(KeyCode::BTN_TOOL_RUBBER);
        }

        let axis = |code, maximum| UinputAbsSetup::new(code, AbsInfo::new(0, 0, maximum, 0, 0, 0));
        let signed_axis =
            |code| UinputAbsSetup::new(code, AbsInfo::new(0, TILT_MIN, TILT_MAX, 0, 0, 0));

        let mut builder = VirtualDevice::builder()
            .ok()?
            .name(name)
            .with_keys(&keys)
            .ok()?
            .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_X, 1920))
            .ok()?
            .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_Y, 1080))
            .ok()?
            .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_PRESSURE, PRESSURE_MAX))
            .ok()?;
        if pen_tool {
            builder = builder
                .with_absolute_axis(&signed_axis(AbsoluteAxisCode::ABS_TILT_X))
                .ok()?
                .with_absolute_axis(&signed_axis(AbsoluteAxisCode::ABS_TILT_Y))
                .ok()?;
        }
        builder.build().ok()
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

    fn eraser_tool_event(down: bool) -> InputEvent {
        InputEvent::new(EventType::KEY.0, KeyCode::BTN_TOOL_RUBBER.0, i32::from(down))
    }

    fn tilt_event(axis: AbsoluteAxisCode, raw: i32) -> InputEvent {
        InputEvent::new(EventType::ABSOLUTE.0, axis.0, raw)
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

    /// Whether this process can actually OPEN the node of a device it created.
    ///
    /// This is what tells "the scanner is broken" apart from "this machine
    /// cannot do the test", and the two are indistinguishable at the assertion
    /// itself. Two things have to go right and only the first is obvious:
    ///
    /// 1. Write access to `/dev/uinput`, which is enough to CREATE a device --
    ///    so `build` succeeds and the earlier skip never fires.
    /// 2. Read access to the `/dev/input/eventN` node udev then makes for it,
    ///    which is what the scanner opens. On a CI runner the node appears and
    ///    is owned by root, so opening it fails and no scanner however correct
    ///    could ever have seen the device. (Checking only that the node EXISTS
    ///    is not enough, and was measured not to be: it came back true on the
    ///    runner and the test failed anyway.)
    ///
    /// **A node this process cannot open is a fact about the machine and is
    /// worth skipping over; a node it CAN open that the scanner still missed is
    /// a real failure and must stay one.**
    fn can_read_own_node(device: &mut VirtualDevice) -> bool {
        device
            .enumerate_dev_nodes_blocking()
            .is_ok_and(|nodes| nodes.flatten().any(|path| std::fs::File::open(path).is_ok()))
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
        if !found {
            if !can_read_own_node(&mut stylus) {
                eprintln!(
                    "skipping: the virtual stylus was created but its /dev/input node cannot be opened by \
                     this process, so nothing could have seen it. That needs udev and \
                     membership of the input group, which a CI container has not got."
                );
                return;
            }
            panic!("the scanner never picked up the virtual stylus");
        }

        let devices = tablet.devices();
        let stylus_info = devices
            .iter()
            .find(|device| device.name == "BrokkrSculpt test stylus")
            .expect("just found it");
        assert_eq!(
            stylus_info.pressure_max, PRESSURE_MAX,
            "the device's own pressure range was not read from it"
        );
        assert!(stylus_info.has_tilt, "tilt axes were not detected");
        assert!(stylus_info.has_eraser, "the eraser end was not detected");
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

        // Tilt. A lopsided device range must still put an upright pen at zero.
        emit(&mut stylus, &[tilt_event(AbsoluteAxisCode::ABS_TILT_X, 0)]);
        emit(&mut stylus, &[tilt_event(AbsoluteAxisCode::ABS_TILT_Y, 0)]);
        assert!(
            wait_for(|| tablet.state().tilt == Vec2::ZERO),
            "an upright pen should report no tilt, got {:?}",
            tablet.state().tilt
        );

        emit(
            &mut stylus,
            &[
                tilt_event(AbsoluteAxisCode::ABS_TILT_X, TILT_MAX),
                tilt_event(AbsoluteAxisCode::ABS_TILT_Y, TILT_MIN / 2),
            ],
        );
        assert!(
            wait_for(|| {
                let tilt = tablet.state().tilt;
                tilt.x > 0.9 && (tilt.y + 0.5).abs() < 0.02
            }),
            "tilt did not come through normalised, got {:?}",
            tablet.state().tilt
        );

        // Flip to the eraser end. The kernel reports the two ends as separate
        // tools that are never in range together, so this is the case where
        // only checking for the tip would silently drop back to full pressure.
        emit(&mut stylus, &[pen_tool_event(false), eraser_tool_event(true)]);
        emit(&mut stylus, &[pressure_event(PRESSURE_MAX / 4)]);
        assert!(
            wait_for(|| {
                let pen = tablet.state();
                pen.in_proximity && pen.eraser && (pen.pressure - 0.25).abs() < 0.01
            }),
            "the eraser end did not report as a pen with pressure, got {:?}",
            tablet.state()
        );
        assert!(
            (tablet.stamp_pressure(true, 1.0) - 0.25).abs() < 0.01,
            "eraser strokes must be pressure sensitive, not full strength"
        );

        emit(&mut stylus, &[eraser_tool_event(false), pen_tool_event(true)]);
        assert!(
            wait_for(|| !tablet.state().eraser && tablet.state().in_proximity),
            "flipping back to the tip did not register"
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
        assert_eq!(tablet.state().tilt, Vec2::ZERO, "a lifted pen must not leave a stale tilt");

        // Unplugging must deregister the device rather than leaving a ghost.
        drop(stylus);
        assert!(
            wait_for(|| { !tablet.devices().iter().any(|d| d.name == "BrokkrSculpt test stylus") }),
            "the device was still listed after being removed"
        );
    }
}
