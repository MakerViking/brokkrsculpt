// SPDX-License-Identifier: AGPL-3.0-only

//! 3Dconnexion SpaceMouse: a six degree of freedom puck for driving the camera.
//!
//! # Why evdev rather than hidraw
//!
//! SindriCAD reads the same hardware over hidraw, which needs a udev rule
//! shipped and installed before the device can be opened at all. evdev needs
//! only membership of the `input` group, which the stylus already requires, so
//! the puck works on this machine with no setup. The device is found through
//! the same `/dev/input` scanner the tablet uses.
//!
//! # Detection is a capability rule, not a vendor list
//!
//! A device is a puck if it reports **all six** of `REL_X` through `REL_RZ`.
//! Measured across every input device on the development machine: mice report
//! `[X, Y, HWHEEL, WHEEL, WHEEL_HI_RES, HWHEEL_HI_RES]`, keyboards report a
//! wheel pair, and only the SpaceNavigator reports the six.
//!
//! This is strictly stronger than matching a vendor id, and deliberately so.
//! 3Dconnexion's older devices carry Logitech's `0x046d`, which Logitech also
//! puts on every mouse it ships — SindriCAD shipped a version that took an MX
//! Anywhere for a puck and rotated the world whenever the mouse moved.
//!
//! # Two things about relative axes that are not obvious
//!
//! Both look like tuning problems and are not:
//!
//! * **The values are absolute deflection, not deltas.** The puck streams its
//!   current stick position in every report and the kernel forwards each one as
//!   a relative event. Keep the latest value per axis; accumulating them would
//!   make a steady push accelerate without limit.
//! * **Letting go produces silence, not zeroes.** The kernel's input core drops
//!   relative events whose value is zero, so a puck returning to centre emits
//!   nothing at all. Without [`Config::stale_ms`] the last deflection would
//!   steer the camera forever. SindriCAD needed the same timeout against
//!   hidraw for the same reason, which is why 120 ms is a proven number rather
//!   than a guess.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use glam::Vec2;

use crate::camera::OrbitCamera;

/// The puck's six raw axes, in the order the kernel reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// `REL_X`, sliding the cap left and right.
    Tx,
    /// `REL_Y`, pushing the cap away and pulling it back.
    Ty,
    /// `REL_Z`, lifting the cap and pressing it down.
    Tz,
    /// `REL_RX`, tipping the cap forward and back.
    Rx,
    /// `REL_RY`, tipping the cap sideways.
    Ry,
    /// `REL_RZ`, twisting the cap.
    Rz,
}

impl Axis {
    pub const ALL: [Axis; 6] = [Axis::Tx, Axis::Ty, Axis::Tz, Axis::Rx, Axis::Ry, Axis::Rz];

    /// The short name used in the config file.
    pub fn key(self) -> &'static str {
        match self {
            Axis::Tx => "tx",
            Axis::Ty => "ty",
            Axis::Tz => "tz",
            Axis::Rx => "rx",
            Axis::Ry => "ry",
            Axis::Rz => "rz",
        }
    }

    /// What the hand actually does, which is what the settings panel shows.
    /// A raw axis name means nothing to someone holding the puck.
    pub fn label(self) -> &'static str {
        match self {
            Axis::Tx => "Slide ←→",
            Axis::Ty => "Push / pull",
            Axis::Tz => "Lift ↑↓",
            Axis::Rx => "Tip forward",
            Axis::Ry => "Tip sideways",
            Axis::Rz => "Twist",
        }
    }

    fn parse(text: &str) -> Option<Axis> {
        Axis::ALL.into_iter().find(|axis| axis.key() == text)
    }
}

/// A camera movement an axis can be bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    PanX,
    PanY,
    Zoom,
    OrbitAz,
    OrbitPolar,
    Roll,
}

impl Action {
    pub const ALL: [Action; 6] = [
        Action::PanX,
        Action::PanY,
        Action::Zoom,
        Action::OrbitAz,
        Action::OrbitPolar,
        Action::Roll,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Action::PanX => "pan_x",
            Action::PanY => "pan_y",
            Action::Zoom => "zoom",
            Action::OrbitAz => "orbit_az",
            Action::OrbitPolar => "orbit_polar",
            Action::Roll => "roll",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Action::PanX => "Pan ←→",
            Action::PanY => "Pan ↑↓",
            Action::Zoom => "Zoom",
            Action::OrbitAz => "Rotate ←→",
            Action::OrbitPolar => "Rotate ↑↓",
            Action::Roll => "Roll",
        }
    }
}

/// What pressing one of the puck's buttons does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonAction {
    None,
    Undo,
    Redo,
    FrameModel,
    ResetView,
    ToggleSymmetry,
}

impl ButtonAction {
    pub const ALL: [ButtonAction; 6] = [
        ButtonAction::None,
        ButtonAction::Undo,
        ButtonAction::Redo,
        ButtonAction::FrameModel,
        ButtonAction::ResetView,
        ButtonAction::ToggleSymmetry,
    ];

    pub fn key(self) -> &'static str {
        match self {
            ButtonAction::None => "none",
            ButtonAction::Undo => "undo",
            ButtonAction::Redo => "redo",
            ButtonAction::FrameModel => "frame",
            ButtonAction::ResetView => "reset_view",
            ButtonAction::ToggleSymmetry => "symmetry",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ButtonAction::None => "Nothing",
            ButtonAction::Undo => "Undo",
            ButtonAction::Redo => "Redo",
            ButtonAction::FrameModel => "Frame model",
            ButtonAction::ResetView => "Reset view",
            ButtonAction::ToggleSymmetry => "Toggle symmetry",
        }
    }

    fn parse(text: &str) -> Option<ButtonAction> {
        ButtonAction::ALL.into_iter().find(|action| action.key() == text)
    }
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::fmt::Display for ButtonAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Which raw axis drives an action, and whether its sign is flipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisBinding {
    pub source: Axis,
    pub invert: bool,
}

/// Whether the puck moves the model or the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The cap moves the model. 3Dconnexion's own default, and the inverse of
    /// [`Mode::Camera`] on pan and orbit.
    Object,
    /// The cap flies the camera.
    Camera,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Object => "Object",
            Mode::Camera => "Camera",
        }
    }

    fn sign(self) -> f32 {
        match self {
            Mode::Object => -1.0,
            Mode::Camera => 1.0,
        }
    }
}

/// How many buttons can be bound. The SpaceNavigator has exactly two; a puck
/// with more has the extras logged and ignored rather than silently dropped.
pub const BUTTON_COUNT: usize = 2;

/// Everything tunable about the puck.
///
/// The defaults are SindriCAD's, ported rather than rediscovered: they were
/// tuned against this same SpaceNavigator and its `staleMs`, deadzone and
/// per millisecond sensitivities are all known good.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub mode: Mode,
    /// Ignore any axis whose magnitude is below this. The cap does not sit
    /// perfectly still and its rest reading wanders by a few counts.
    pub deadzone: f32,
    /// Fraction of the view height per axis unit per millisecond.
    pub pan_sens: f32,
    /// Natural log of the zoom factor per axis unit per millisecond.
    pub zoom_sens: f32,
    /// Radians per axis unit per millisecond.
    pub orbit_sens: f32,
    /// Treat the puck as centred after this long with no event. See the module
    /// documentation: this is not a nicety, it is what stops the last
    /// deflection steering forever.
    pub stale_ms: u64,
    /// One binding per [`Action`], indexed in `Action::ALL` order.
    pub bind: [AxisBinding; 6],
    pub buttons: [ButtonAction; BUTTON_COUNT],
}

impl Default for Config {
    fn default() -> Self {
        let bind = |source, invert| AxisBinding { source, invert };
        Self {
            mode: Mode::Object,
            deadzone: 24.0,
            pan_sens: 6.0e-7,
            zoom_sens: 7.0e-7,
            orbit_sens: 2.2e-6,
            stale_ms: 120,
            // Which axis drives what is SindriCAD's map, verbatim. The invert
            // flags are NOT: every one of them is the opposite of SindriCAD's,
            // and deliberately so.
            //
            // SindriCAD drives `camera-controls`, whose `truck` and `tumble`
            // move the camera and leave it to the caller to negate for
            // object-style motion — which is what its `modeSign` is for.
            // `OrbitCamera` already bakes that negation in, because a mouse
            // drag has to carry the model with the cursor. Porting the sign
            // convention verbatim on top of that double-negates, and every
            // axis comes out backwards on the real device.
            //
            // These are a starting point to tune from, not a finished answer:
            // the two conventions differ per axis rather than uniformly, so
            // the flags below will not all be right. Flip them in the
            // settings panel, then the file at `Config::path()` is what
            // should be pasted back in here to lock the defaults in.
            bind: [
                bind(Axis::Tx, true),  // pan x
                bind(Axis::Tz, true),  // pan y
                bind(Axis::Ty, true),  // zoom
                bind(Axis::Rz, false), // orbit azimuth
                bind(Axis::Rx, true),  // orbit polar
                bind(Axis::Ry, true),  // roll
            ],
            buttons: [ButtonAction::Undo, ButtonAction::Redo],
        }
    }
}

impl Config {
    pub fn binding(&self, action: Action) -> AxisBinding {
        self.bind[action as usize]
    }

    pub fn set_binding(&mut self, action: Action, binding: AxisBinding) {
        self.bind[action as usize] = binding;
    }

    /// Flip every action at once.
    ///
    /// Worth a button of its own because the two camera conventions this was
    /// ported between differ by a sign on most axes, so "everything is
    /// backwards" is the single most likely thing to be wrong, and fixing it
    /// six checkboxes at a time is six chances to lose track of which have
    /// been done.
    pub fn invert_all(&mut self) {
        for binding in &mut self.bind {
            binding.invert = !binding.invert;
        }
    }

    /// The deadzoned, sign corrected value of whatever axis drives `action`.
    fn value(&self, action: Action, motion: &Motion) -> f32 {
        let binding = self.binding(action);
        let raw = motion.axes[binding.source as usize];
        if raw.abs() < self.deadzone {
            return 0.0;
        }
        if binding.invert { -raw } else { raw }
    }

    /// Drive the camera from one sample of the puck. Returns whether anything
    /// actually moved, so the caller can skip republishing an unchanged camera.
    ///
    /// `elapsed_ms` scales everything, which is what makes the puck feel the
    /// same at any frame rate.
    pub fn apply(
        &self,
        motion: &Motion,
        elapsed_ms: f32,
        camera: &mut OrbitCamera,
        viewport_height: f32,
    ) -> bool {
        // A stalled frame must not fling the camera across the model.
        let elapsed_ms = elapsed_ms.clamp(0.0, 50.0);
        if elapsed_ms <= 0.0 {
            return false;
        }
        let sign = self.mode.sign();
        let mut moved = false;

        let (pan_x, pan_y) = (self.value(Action::PanX, motion), self.value(Action::PanY, motion));
        if pan_x != 0.0 || pan_y != 0.0 {
            // pan_sens is a fraction of the view height, and `pan` takes
            // pixels over that same height, so the two cancel and the puck
            // moves the view by a constant fraction of what is on screen at
            // any zoom. Fixed world unit steps are exactly what made
            // SindriCAD's puck feel a hundred times too fast at mm scale.
            let fraction = Vec2::new(pan_x, pan_y) * (self.pan_sens * elapsed_ms * sign);
            camera.pan(fraction * viewport_height, viewport_height);
            moved = true;
        }

        // Zoom direction is a preference of its own rather than a consequence
        // of object versus camera mode, so it is not multiplied by `sign`. A
        // positive axis zooms in, matching the wheel.
        let zoom = self.value(Action::Zoom, motion);
        if zoom != 0.0 {
            camera.zoom_by((-zoom * self.zoom_sens * elapsed_ms).exp());
            moved = true;
        }

        let azimuth = self.value(Action::OrbitAz, motion);
        let polar = self.value(Action::OrbitPolar, motion);
        if azimuth != 0.0 || polar != 0.0 {
            camera.orbit_radians(Vec2::new(azimuth, polar) * (self.orbit_sens * elapsed_ms * sign));
            moved = true;
        }

        let roll = self.value(Action::Roll, motion);
        if roll != 0.0 {
            camera.roll =
                crate::camera::wrap_angle(camera.roll + roll * self.orbit_sens * elapsed_ms * sign);
            moved = true;
        }

        moved
    }
}

/// One sample of the puck's six axes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Motion {
    /// Deflection per axis in `Axis::ALL` order, raw device units. Zero on
    /// every axis once the puck has gone quiet.
    pub axes: [f32; 6],
    /// Whether the puck reported anything within [`Config::stale_ms`].
    pub live: bool,
}

impl Motion {
    pub fn axis(&self, axis: Axis) -> f32 {
        self.axes[axis as usize]
    }
}

/// A puck the scanner adopted, for the settings panel and the diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PuckDevice {
    pub name: String,
    pub path: String,
    /// How many of its buttons are bound.
    pub buttons: usize,
}

struct Shared {
    axes: [AtomicI32; 6],
    /// Monotonic press count per button. A live pressed/released mask would
    /// lose a press that began and ended between two frames, because the
    /// application only samples this once per frame.
    presses: [AtomicU32; BUTTON_COUNT],
    /// Milliseconds since `started` at the last axis event.
    last_event_ms: AtomicU64,
    /// Whether any axis event has ever arrived, so that "no puck yet" is not
    /// confused with "an event at time zero".
    seen: AtomicBool,
    /// Largest magnitude seen on any axis. Relative axes carry no range to
    /// read, so full scale has to be learnt from use.
    full_scale: AtomicI32,
    started: Instant,
    devices: Mutex<Vec<PuckDevice>>,
    permission_denied: AtomicBool,
}

/// A floor for the learnt full scale, so the readout bar is not wildly
/// oversensitive before the puck has been pushed properly once.
const FULL_SCALE_FLOOR: i32 = 128;

impl Shared {
    fn new() -> Self {
        Self {
            axes: std::array::from_fn(|_| AtomicI32::new(0)),
            presses: std::array::from_fn(|_| AtomicU32::new(0)),
            last_event_ms: AtomicU64::new(0),
            seen: AtomicBool::new(false),
            full_scale: AtomicI32::new(FULL_SCALE_FLOOR),
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
    fn set_axis(&self, index: usize, value: i32) {
        self.axes[index].store(value, Ordering::Relaxed);
        self.full_scale.fetch_max(value.saturating_abs(), Ordering::Relaxed);
        self.last_event_ms.store(self.now_ms(), Ordering::Relaxed);
        self.seen.store(true, Ordering::Relaxed);
    }

    // Same as `set_axis` above: written only by the evdev backend.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn press(&self, index: usize) {
        self.presses[index].fetch_add(1, Ordering::Relaxed);
    }

    // Written only by the evdev backend, which is Linux only. The fields and
    // the readers are shared, so these stay compiled everywhere rather than
    // being cfg'd into two shapes -- but with no producer on Windows or macOS
    // they are dead there, and `-D warnings` is right to say so. Allowed with
    // a reason rather than silenced: when those platforms grow a Pointer
    // Input, Wintab or IOKit backend, it calls exactly these.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn clear_axes(&self) {
        for axis in &self.axes {
            axis.store(0, Ordering::Relaxed);
        }
        self.seen.store(false, Ordering::Relaxed);
    }

    fn motion(&self, stale_ms: u64) -> Motion {
        if !self.seen.load(Ordering::Relaxed) {
            return Motion::default();
        }
        let idle = self.now_ms().saturating_sub(self.last_event_ms.load(Ordering::Relaxed));
        if idle > stale_ms {
            return Motion::default();
        }
        Motion {
            axes: std::array::from_fn(|index| self.axes[index].load(Ordering::Relaxed) as f32),
            live: true,
        }
    }

    fn press_counts(&self) -> [u32; BUTTON_COUNT] {
        std::array::from_fn(|index| self.presses[index].load(Ordering::Relaxed))
    }
}

/// Why the puck is not doing anything, in terms the user can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagnosis {
    /// Reading it is implemented, a device was found, and it is reporting.
    Working,
    /// A device was found but has sent nothing yet. Normal until it is touched.
    Idle,
    /// Nothing in `/dev/input` could be opened at all.
    PermissionDenied,
    /// No device with six relative axes is present.
    NoDevice,
    /// Not Linux.
    Unsupported,
}

impl Diagnosis {
    pub fn explain(self) -> &'static str {
        match self {
            Diagnosis::Working => "Reading the puck.",
            Diagnosis::Idle => "Found, waiting for it to be moved.",
            Diagnosis::PermissionDenied => {
                "No input device could be opened. Add your user to the input group \
                 and log back in: sudo usermod -aG input $USER"
            }
            Diagnosis::NoDevice => {
                "No six axis device found. Run with --spacemouse to see every input \
                 device and why each was rejected."
            }
            Diagnosis::Unsupported => "The SpaceMouse is implemented for Linux only so far.",
        }
    }
}

/// The application's handle on the puck.
pub struct SpaceMouse {
    shared: Arc<Shared>,
    pub config: Config,
    /// Press counts already acted on, so each press fires exactly once.
    acknowledged: [u32; BUTTON_COUNT],
}

impl SpaceMouse {
    pub fn start() -> Self {
        let shared = Arc::new(Shared::new());
        backend::spawn(Arc::clone(&shared));
        Self { acknowledged: shared.press_counts(), shared, config: Config::load() }
    }

    /// A handle with no scanner behind it, so tests do not go looking through
    /// `/dev/input` or spawn reader threads.
    ///
    /// Deliberately uses the default settings rather than [`Config::load`]:
    /// a test that read the developer's own config file would pass or fail
    /// depending on how they had tuned their puck.
    #[cfg(test)]
    pub fn inert() -> Self {
        let shared = Arc::new(Shared::new());
        Self { acknowledged: shared.press_counts(), shared, config: Config::default() }
    }

    /// The latest deflection, already zeroed if the puck has gone quiet.
    pub fn motion(&self) -> Motion {
        self.shared.motion(self.config.stale_ms)
    }

    pub fn devices(&self) -> Vec<PuckDevice> {
        self.shared.devices.lock().expect("spacemouse state poisoned").clone()
    }

    /// The largest deflection seen so far, which is what the readout scales
    /// against.
    pub fn full_scale(&self) -> f32 {
        self.shared.full_scale.load(Ordering::Relaxed) as f32
    }

    /// Every button press since this was last called, in press order.
    ///
    /// Draining a counter rather than reading a mask is what makes a press and
    /// release inside a single frame still count.
    pub fn take_presses(&mut self) -> Vec<ButtonAction> {
        let counts = self.shared.press_counts();
        let mut fired = Vec::new();
        for ((count, acknowledged), action) in
            counts.into_iter().zip(&mut self.acknowledged).zip(self.config.buttons)
        {
            let pending = count.wrapping_sub(*acknowledged);
            *acknowledged = count;
            if action == ButtonAction::None {
                continue;
            }
            // Capped so that a device spewing key events cannot turn one frame
            // into thousands of undos.
            for _ in 0..pending.min(8) {
                fired.push(action);
            }
        }
        fired
    }

    pub fn diagnosis(&self) -> Diagnosis {
        if !backend::SUPPORTED {
            return Diagnosis::Unsupported;
        }
        if !self.devices().is_empty() {
            if self.shared.seen.load(Ordering::Relaxed) {
                return Diagnosis::Working;
            }
            return Diagnosis::Idle;
        }
        if self.shared.permission_denied.load(Ordering::Relaxed) {
            return Diagnosis::PermissionDenied;
        }
        Diagnosis::NoDevice
    }

    /// Feed a sample straight in, for tests and for the offscreen harness.
    #[cfg(test)]
    pub fn simulate(&self, axes: [i32; 6]) {
        for (index, value) in axes.into_iter().enumerate() {
            self.shared.set_axis(index, value);
        }
    }

    /// Fire one button, for tests that need the application's own button path
    /// rather than the action it happens to be bound to.
    #[cfg(test)]
    pub fn simulate_press(&self, button: usize) {
        self.shared.press(button);
    }
}

/// One line per input device and whether it is a puck, for `--spacemouse`.
///
/// A puck that is plugged in but silent is otherwise very hard to tell from
/// one that was never found, and the answer is usually a group membership.
pub fn report() -> String {
    backend::report()
}

// --- configuration file ----------------------------------------------------

impl Config {
    /// Where the puck's settings live.
    pub fn path() -> Option<std::path::PathBuf> {
        crate::paths::config_file("spacemouse.conf")
    }

    /// Read the settings, falling back to the defaults for anything missing or
    /// unparseable.
    ///
    /// A broken config file must never stop the application starting, and must
    /// never leave an action unbound: an unbound action would be a hole in the
    /// motion loop rather than a setting.
    pub fn load() -> Config {
        let Some(path) = Self::path() else {
            return Config::default();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Config::default();
        };
        let mut config = Config::default();
        config.merge(&text);
        config
    }

    fn merge(&mut self, text: &str) {
        for (key, value) in crate::paths::entries(text) {
            // An unknown key or an unparseable value leaves the default in
            // place rather than failing the whole file, so a config written by
            // a newer version still mostly works.
            match key {
                "mode" => match value {
                    "object" => self.mode = Mode::Object,
                    "camera" => self.mode = Mode::Camera,
                    _ => {}
                },
                "deadzone" => {
                    if let Ok(parsed) = value.parse::<f32>() {
                        self.deadzone = parsed.max(0.0);
                    }
                }
                "pan_sens" => set_positive(&mut self.pan_sens, value),
                "zoom_sens" => set_positive(&mut self.zoom_sens, value),
                "orbit_sens" => set_positive(&mut self.orbit_sens, value),
                "stale_ms" => {
                    if let Ok(parsed) = value.parse::<u64>() {
                        // Zero would make every sample stale and the puck dead.
                        self.stale_ms = parsed.max(1);
                    }
                }
                _ => {
                    if let (Some(action), Some(binding)) =
                        (Action::ALL.into_iter().find(|a| a.key() == key), parse_binding(value))
                    {
                        self.set_binding(action, binding);
                    } else if let (Some(index), Some(action)) =
                        (button_index(key), ButtonAction::parse(value))
                    {
                        self.buttons[index] = action;
                    }
                }
            }
        }
    }

    pub fn to_text(self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out, "# BrokkrSculpt SpaceMouse settings.");
        let _ = writeln!(out, "# An axis may be prefixed with - to invert it.\n");
        let _ = writeln!(
            out,
            "mode = {}",
            match self.mode {
                Mode::Object => "object",
                Mode::Camera => "camera",
            }
        );
        let _ = writeln!(out, "deadzone = {}", self.deadzone);
        let _ = writeln!(out, "pan_sens = {:e}", self.pan_sens);
        let _ = writeln!(out, "zoom_sens = {:e}", self.zoom_sens);
        let _ = writeln!(out, "orbit_sens = {:e}", self.orbit_sens);
        let _ = writeln!(out, "stale_ms = {}\n", self.stale_ms);
        for action in Action::ALL {
            let binding = self.binding(action);
            let sign = if binding.invert { "-" } else { "" };
            let _ = writeln!(out, "{} = {sign}{}", action.key(), binding.source.key());
        }
        let _ = writeln!(out);
        for (index, action) in self.buttons.iter().enumerate() {
            let _ = writeln!(out, "button_{} = {}", index + 1, action.key());
        }
        out
    }

    /// Write the settings out. Best effort: failing to save a preference must
    /// not interrupt sculpting.
    pub fn save(&self) {
        let Some(path) = Self::path() else {
            return;
        };
        if let Some(parent) = path.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            log::warn!("could not create {}: {error}", parent.display());
            return;
        }
        if let Err(error) = std::fs::write(&path, self.to_text()) {
            log::warn!("could not save the SpaceMouse settings to {}: {error}", path.display());
        }
    }
}

fn set_positive(field: &mut f32, value: &str) {
    if let Ok(parsed) = value.parse::<f32>()
        && parsed.is_finite()
        && parsed > 0.0
    {
        *field = parsed;
    }
}

fn parse_binding(value: &str) -> Option<AxisBinding> {
    let (invert, name) = match value.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, value),
    };
    Axis::parse(name).map(|source| AxisBinding { source, invert })
}

fn button_index(key: &str) -> Option<usize> {
    let number: usize = key.strip_prefix("button_")?.parse().ok()?;
    // The file counts buttons from one, the array from zero.
    number.checked_sub(1).filter(|index| *index < BUTTON_COUNT)
}

#[cfg(target_os = "linux")]
mod backend {
    use super::{BUTTON_COUNT, PuckDevice, Shared};
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use evdev::{Device, EventType, RelativeAxisCode};

    pub const SUPPORTED: bool = true;

    /// A puck may be switched on after the application, like any other device.
    const RESCAN: Duration = Duration::from_secs(2);

    /// The six axes that define a 6DOF controller, in `Axis::ALL` order.
    const AXES: [RelativeAxisCode; 6] = [
        RelativeAxisCode::REL_X,
        RelativeAxisCode::REL_Y,
        RelativeAxisCode::REL_Z,
        RelativeAxisCode::REL_RX,
        RelativeAxisCode::REL_RY,
        RelativeAxisCode::REL_RZ,
    ];

    /// A device is a puck if it reports **all six** relative axes.
    ///
    /// See the module documentation for why this is the rule and a vendor id
    /// is not.
    fn is_puck(device: &Device) -> bool {
        device
            .supported_relative_axes()
            .is_some_and(|axes| AXES.iter().all(|axis| axes.contains(*axis)))
    }

    /// Position of a relative axis code in `AXES`, or `None` if it is one we
    /// do not use.
    fn axis_index(code: u16) -> Option<usize> {
        AXES.iter().position(|axis| axis.0 == code)
    }

    /// The device's buttons in ascending code order, which is the order the
    /// settings panel numbers them in.
    fn buttons_of(device: &Device) -> Vec<u16> {
        let mut keys: Vec<u16> =
            device.supported_keys().into_iter().flatten().map(|key| key.0).collect();
        keys.sort_unstable();
        keys
    }

    pub fn spawn(shared: Arc<Shared>) {
        crate::input_watch::spawn("brokkr-spacemouse-scan", RESCAN, Puck { shared });
    }

    struct Puck {
        shared: Arc<Shared>,
    }

    impl crate::input_watch::Adopter for Puck {
        fn wants(&self, device: &Device) -> bool {
            is_puck(device)
        }

        fn adopt(&self, path: &Path, device: Device) -> bool {
            adopt(&self.shared, path, device)
        }

        fn still_reading(&self, path: &Path) -> bool {
            let live = self.shared.devices.lock().expect("spacemouse state poisoned");
            live.iter().any(|device| device.path == path.to_string_lossy())
        }

        fn set_permission_denied(&self, denied: bool) {
            self.shared.permission_denied.store(denied, Ordering::Relaxed);
        }
    }

    fn adopt(shared: &Arc<Shared>, path: &Path, mut device: Device) -> bool {
        let all_buttons = buttons_of(&device);
        let mut buttons = [None; BUTTON_COUNT];
        for (slot, code) in buttons.iter_mut().zip(all_buttons.iter()) {
            *slot = Some(*code);
        }

        let info = PuckDevice {
            name: device.name().unwrap_or("unnamed 6 axis device").to_string(),
            path: path.to_string_lossy().into_owned(),
            buttons: all_buttons.len().min(BUTTON_COUNT),
        };
        if all_buttons.len() > BUTTON_COUNT {
            log::info!(
                "spacemouse: {} has {} buttons, only the first {BUTTON_COUNT} are bindable",
                info.name,
                all_buttons.len()
            );
        }

        shared.devices.lock().expect("spacemouse state poisoned").push(info.clone());

        let for_reader = Arc::clone(shared);
        let registered = info.clone();
        let started = std::thread::Builder::new()
            .name("brokkr-spacemouse-read".into())
            .spawn(move || {
                read_loop(&for_reader, &mut device, buttons);
                // Unplugged, or the read failed. Drop the registration so the
                // scanner can pick the device up again if it comes back, and
                // zero the axes so a puck that vanished mid push does not
                // leave the camera drifting.
                for_reader
                    .devices
                    .lock()
                    .expect("spacemouse state poisoned")
                    .retain(|device| *device != registered);
                for_reader.clear_axes();
                log::info!("spacemouse: {} disconnected", registered.name);
            })
            .is_ok();

        if started {
            log::info!(
                "spacemouse: reading {} at {} with {} bindable button(s)",
                info.name,
                info.path,
                info.buttons
            );
        } else {
            shared.devices.lock().expect("spacemouse state poisoned").retain(|d| *d != info);
        }
        started
    }

    fn read_loop(shared: &Arc<Shared>, device: &mut Device, buttons: [Option<u16>; BUTTON_COUNT]) {
        loop {
            let events = match device.fetch_events() {
                Ok(events) => events,
                Err(error) => {
                    log::debug!("spacemouse read ended: {error}");
                    return;
                }
            };
            for event in events {
                match event.event_type() {
                    // Deflection, not a delta: keep the latest value. Only
                    // these refresh the staleness clock, because a button
                    // press must not keep a stale axis alive.
                    EventType::RELATIVE => {
                        if let Some(index) = axis_index(event.code()) {
                            shared.set_axis(index, event.value());
                        }
                    }
                    // 1 is a press; 2 is autorepeat, which a button held down
                    // must not turn into a stream of undos.
                    EventType::KEY if event.value() == 1 => {
                        if let Some(index) =
                            buttons.iter().position(|button| *button == Some(event.code()))
                        {
                            shared.press(index);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn report() -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out, "Input devices, as the SpaceMouse scanner sees them.\n");

        let mut entries = crate::input_watch::event_nodes();
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
                    let present: Vec<&str> = AXES
                        .iter()
                        .zip(super::Axis::ALL)
                        .filter(|(code, _)| {
                            device
                                .supported_relative_axes()
                                .is_some_and(|axes| axes.contains(**code))
                        })
                        .map(|(_, axis)| axis.key())
                        .collect();

                    if present.is_empty() {
                        // Keyboards and the like. Listing every one buries the
                        // interesting lines.
                        continue;
                    }

                    let verdict = if present.len() == AXES.len() {
                        found += 1;
                        "SPACEMOUSE, the camera will be driven from this".to_string()
                    } else {
                        format!(
                            "ignored: has {} of the 6 relative axes, so it is an ordinary \
                             pointing device",
                            present.len()
                        )
                    };
                    let _ = writeln!(
                        out,
                        "{}\n  {name}\n  relative axes: {}\n  buttons: {}\n  {verdict}\n",
                        path.display(),
                        present.join(", "),
                        buttons_of(&device).len(),
                    );
                }
                Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                    unreadable += 1;
                }
                Err(_) => {}
            }
        }

        let _ = writeln!(out, "{found} SpaceMouse device(s) usable.");
        if unreadable > 0 {
            let _ = writeln!(
                out,
                "{unreadable} device(s) could not be opened. If your puck is missing above,\n\
                 add your user to the input group and log back in:\n\
                 \n    sudo usermod -aG input $USER\n"
            );
        }
        if found == 0 && unreadable == 0 {
            let _ = writeln!(
                out,
                "No 6 axis device found. A SpaceMouse reports all six of REL_X, REL_Y,\n\
                 REL_Z, REL_RX, REL_RY and REL_RZ. If yours is listed above with fewer,\n\
                 check that spacenavd or the 3Dconnexion driver is not holding it."
            );
        }
        out
    }
}

#[cfg(not(target_os = "linux"))]
mod backend {
    use super::Shared;
    use std::sync::Arc;

    pub const SUPPORTED: bool = false;

    /// Windows and macOS both have 3Dconnexion SDKs, and neither is a
    /// milestone away. The application simply has no puck there, which is the
    /// same thing that happens on Linux without one.
    pub fn spawn(_shared: Arc<Shared>) {}

    pub fn report() -> String {
        "The SpaceMouse is implemented for Linux only so far.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn motion(axes: [f32; 6]) -> Motion {
        Motion { axes, live: true }
    }

    /// A push on one raw axis, at a magnitude well clear of the deadzone.
    fn push(axis: Axis, value: f32) -> Motion {
        let mut axes = [0.0; 6];
        axes[axis as usize] = value;
        motion(axes)
    }

    fn camera() -> OrbitCamera {
        OrbitCamera::framing(glam::Vec3::ZERO, 30.0)
    }

    /// The sensitivities, deadzone and staleness are a port of SindriCAD's
    /// tuned values, not fresh guesses, and the axis each action reads is its
    /// map. Pinning them here means a well meaning tweak has to be deliberate.
    /// The invert flags are the deliberate exception -- see
    /// `the_invert_flags_are_the_opposite_of_sindricads_on_purpose`.
    #[test]
    fn the_defaults_are_the_values_sindricad_tuned() {
        let config = Config::default();
        assert_eq!(config.deadzone, 24.0);
        assert_eq!(config.pan_sens, 6.0e-7);
        assert_eq!(config.zoom_sens, 7.0e-7);
        assert_eq!(config.orbit_sens, 2.2e-6);
        assert_eq!(config.stale_ms, 120);
        assert_eq!(config.mode, Mode::Object);

        // Which axis drives what, which is the part taken verbatim.
        for (action, source) in [
            (Action::PanX, Axis::Tx),
            (Action::PanY, Axis::Tz),
            (Action::Zoom, Axis::Ty),
            (Action::OrbitAz, Axis::Rz),
            (Action::OrbitPolar, Axis::Rx),
            (Action::Roll, Axis::Ry),
        ] {
            assert_eq!(config.binding(action).source, source, "{} moved axis", action.key());
        }
        assert_eq!(config.buttons, [ButtonAction::Undo, ButtonAction::Redo]);
    }

    /// A camera in a known pose, so "which way did it go" has one answer.
    ///
    /// Target at the origin, no yaw or pitch, so the eye sits on +Z looking
    /// down -Z with right = +X and up = +Y.
    fn known_pose() -> OrbitCamera {
        OrbitCamera {
            target: glam::Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            ..OrbitCamera::framing(glam::Vec3::ZERO, 30.0)
        }
    }

    /// What one action does to the camera, in world terms, at full deflection.
    fn drive(action: Action, config: &Config) -> OrbitCamera {
        let binding = config.binding(action);
        let mut axes = [0.0; 6];
        axes[binding.source as usize] = 300.0;
        let mut camera = known_pose();
        config.apply(&motion(axes), 16.0, &mut camera, 720.0);
        camera
    }

    /// Pins which way every action moves the camera.
    ///
    /// These directions were WRONG on the real device on first contact: the
    /// bindings are a port from SindriCAD, whose `camera-controls` primitives
    /// move the camera and leave the object-style negation to the caller,
    /// while `OrbitCamera` already bakes it in. The two conventions differ per
    /// axis rather than uniformly, so this is the test that stops a future
    /// change flipping one back without anyone noticing.
    ///
    /// Read the assertions as "what the user sees". Moving the target left
    /// carries the view left, so the model appears to move RIGHT.
    ///
    /// These record what the defaults DO, not what they ought to do -- the
    /// directions are still being tuned against the real device. When a
    /// default is changed on purpose, change the matching line here; when one
    /// changes by accident, this is what says so.
    #[test]
    fn each_action_moves_the_camera_the_documented_way() {
        let config = Config::default();

        // Slide the cap right: the target goes left, which carries the view
        // left, so the model appears to move RIGHT with the hand.
        assert!(drive(Action::PanX, &config).target.x < 0.0, "pan x direction changed");

        // Lift the cap: the target rises, so the model appears to move DOWN.
        assert!(drive(Action::PanY, &config).target.y > 0.0, "pan y direction changed");

        // Push the cap: further away.
        assert!(
            drive(Action::Zoom, &config).distance > known_pose().distance,
            "zoom direction changed"
        );

        assert!(drive(Action::OrbitAz, &config).yaw > 0.0, "azimuth direction changed");
        assert!(drive(Action::OrbitPolar, &config).pitch > 0.0, "polar direction changed");
        assert!(drive(Action::Roll, &config).roll > 0.0, "roll direction changed");
    }

    /// The single most likely thing to be wrong on a new device, so it gets a
    /// button and a test rather than six checkboxes and a hope.
    #[test]
    fn inverting_everything_reverses_every_action_and_is_its_own_undo() {
        let mut config = Config::default();
        let before: Vec<OrbitCamera> = Action::ALL.map(|a| drive(a, &config)).into();

        config.invert_all();
        for (action, was) in Action::ALL.into_iter().zip(&before) {
            let now = drive(action, &config);
            match action {
                Action::PanX => assert!(now.target.x * was.target.x < 0.0, "pan x did not flip"),
                Action::PanY => assert!(now.target.y * was.target.y < 0.0, "pan y did not flip"),
                Action::Zoom => assert!(
                    (now.distance - known_pose().distance) * (was.distance - known_pose().distance)
                        < 0.0,
                    "zoom did not flip"
                ),
                Action::OrbitAz => assert!(now.yaw * was.yaw < 0.0, "azimuth did not flip"),
                Action::OrbitPolar => {
                    assert!(now.pitch * was.pitch < 0.0, "polar did not flip")
                }
                Action::Roll => assert!(now.roll * was.roll < 0.0, "roll did not flip"),
            }
        }

        // Twice is the identity, so a mis-click costs nothing.
        config.invert_all();
        assert_eq!(config, Config::default());
    }

    /// The defaults deliberately differ from SindriCAD's invert flags. If
    /// someone "restores" them to match, every axis goes backwards again, so
    /// the divergence is pinned here with the reason.
    #[test]
    fn the_invert_flags_are_the_opposite_of_sindricads_on_purpose() {
        let config = Config::default();
        for (action, sindricad_invert) in [
            (Action::PanX, false),
            (Action::PanY, false),
            (Action::Zoom, false),
            (Action::OrbitAz, true),
            (Action::OrbitPolar, false),
            (Action::Roll, false),
        ] {
            assert_ne!(
                config.binding(action).invert,
                sindricad_invert,
                "{} matches SindriCAD's flag, which double-negates against OrbitCamera",
                action.key()
            );
        }
    }

    #[test]
    fn every_action_and_axis_has_a_distinct_key() {
        let mut keys: Vec<&str> = Action::ALL.iter().map(|a| a.key()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), Action::ALL.len());

        let mut keys: Vec<&str> = Axis::ALL.iter().map(|a| a.key()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), Axis::ALL.len());
    }

    #[test]
    fn a_deflection_inside_the_deadzone_moves_nothing() {
        let config = Config::default();
        let mut moved = camera();
        let before = moved;

        // The cap's rest reading wanders by a few counts; 23 is inside the 24
        // count deadzone and must be treated as still.
        assert!(!config.apply(&push(Axis::Tx, 23.0), 16.0, &mut moved, 720.0));
        assert_eq!(moved.target, before.target);
        assert_eq!(moved.distance, before.distance);
    }

    #[test]
    fn a_deflection_past_the_deadzone_pans() {
        let config = Config::default();
        let mut moved = camera();
        assert!(config.apply(&push(Axis::Tx, 300.0), 16.0, &mut moved, 720.0));
        assert!(moved.target.length() > 0.0, "the view did not pan");
    }

    /// The whole reason pan goes through `OrbitCamera::pan`: a puck deflection
    /// should move the view by the same fraction of what is on screen at any
    /// zoom. Fixed world unit steps made SindriCAD's puck feel about a hundred
    /// times too fast once zoomed into mm scale detail.
    #[test]
    fn panning_is_proportional_to_how_much_is_on_screen() {
        let config = Config::default();
        let sample = push(Axis::Tx, 300.0);

        let mut near = camera();
        near.distance = 10.0;
        config.apply(&sample, 16.0, &mut near, 720.0);

        let mut far = camera();
        far.distance = 20.0;
        config.apply(&sample, 16.0, &mut far, 720.0);

        let ratio = far.target.length() / near.target.length();
        assert!((ratio - 2.0).abs() < 0.01, "twice as far should pan twice as much, got {ratio}");
    }

    /// Which way "positive" zooms is a default under tuning, so this asserts
    /// only that the axis does something and that it undoes itself. The
    /// direction lives in `each_action_moves_the_camera_the_documented_way`,
    /// where it is meant to be edited when a default changes.
    #[test]
    fn the_zoom_axis_is_reversible() {
        let config = Config::default();
        let mut moved = camera();
        let before = moved.distance;

        config.apply(&push(Axis::Ty, 300.0), 16.0, &mut moved, 720.0);
        assert!((moved.distance - before).abs() > 1.0e-4, "the zoom axis did nothing");

        config.apply(&push(Axis::Ty, -300.0), 16.0, &mut moved, 720.0);
        assert!((moved.distance - before).abs() < 1.0e-3, "zoom was not reversible");
    }

    #[test]
    fn inverting_a_binding_reverses_that_action_only() {
        let mut config = Config::default();
        config.set_binding(Action::Zoom, AxisBinding { source: Axis::Ty, invert: true });

        let mut moved = camera();
        let before = moved.distance;
        config.apply(&push(Axis::Ty, 300.0), 16.0, &mut moved, 720.0);
        assert!(moved.distance > before, "inverting zoom should now move away");
    }

    #[test]
    fn rebinding_an_action_moves_which_axis_drives_it() {
        let mut config = Config::default();
        config.set_binding(Action::Zoom, AxisBinding { source: Axis::Rx, invert: false });

        let mut moved = camera();
        let before = moved.distance;
        // The axis zoom used to be on now does nothing to the distance.
        config.apply(&push(Axis::Ty, 300.0), 16.0, &mut moved, 720.0);
        assert_eq!(moved.distance, before);

        config.apply(&push(Axis::Rx, 300.0), 16.0, &mut moved, 720.0);
        assert!(moved.distance < before, "the newly bound axis did not zoom");
    }

    #[test]
    fn object_and_camera_mode_are_opposite_on_pan_but_not_on_zoom() {
        let sample = push(Axis::Tx, 300.0);
        let mut object = camera();
        let mut camera_mode = camera();

        Config { mode: Mode::Object, ..Config::default() }.apply(&sample, 16.0, &mut object, 720.0);
        Config { mode: Mode::Camera, ..Config::default() }.apply(
            &sample,
            16.0,
            &mut camera_mode,
            720.0,
        );
        assert!(
            (object.target + camera_mode.target).length() < 1.0e-5,
            "the two modes should pan opposite ways"
        );

        let zoom = push(Axis::Ty, 300.0);
        let mut object = camera();
        let mut camera_mode = camera();
        Config { mode: Mode::Object, ..Config::default() }.apply(&zoom, 16.0, &mut object, 720.0);
        Config { mode: Mode::Camera, ..Config::default() }.apply(
            &zoom,
            16.0,
            &mut camera_mode,
            720.0,
        );
        assert!(
            (object.distance - camera_mode.distance).abs() < 1.0e-6,
            "zoom direction is its own preference, not a consequence of the mode"
        );
    }

    #[test]
    fn twisting_the_cap_rolls_the_camera_and_the_angle_stays_readable() {
        let config = Config::default();
        let mut moved = camera();
        config.apply(&push(Axis::Ry, 300.0), 16.0, &mut moved, 720.0);
        assert_ne!(moved.roll, 0.0, "the twist axis did not roll the camera");

        // Held for a very long time, the angle must wrap rather than grow
        // without bound and lose precision.
        for _ in 0..4000 {
            config.apply(&push(Axis::Ry, 300.0), 50.0, &mut moved, 720.0);
        }
        assert!(
            moved.roll.abs() <= std::f32::consts::PI + 1.0e-4,
            "roll ran away to {}",
            moved.roll
        );
        assert!(moved.roll.is_finite());
    }

    /// A frame that took a second (a resample, an export) must not fling the
    /// camera across the model when it finally arrives.
    #[test]
    fn a_stalled_frame_is_capped_rather_than_flinging_the_camera() {
        let config = Config::default();
        let sample = push(Axis::Tx, 300.0);

        let mut capped = camera();
        config.apply(&sample, 1000.0, &mut capped, 720.0);

        let mut expected = camera();
        config.apply(&sample, 50.0, &mut expected, 720.0);
        assert!((capped.target - expected.target).length() < 1.0e-5);
    }

    #[test]
    fn a_puck_that_has_never_reported_is_centred() {
        let puck = SpaceMouse::inert();
        let sample = puck.motion();
        assert!(!sample.live);
        assert_eq!(sample.axes, [0.0; 6]);
        // Off Linux there is no backend to find a device with, so the honest
        // answer is `Unsupported` rather than "looked and found none" -- see
        // `backend::SUPPORTED`. The centring above is what this test is for and
        // holds everywhere; only the reason for the silence differs.
        let expected =
            if cfg!(target_os = "linux") { Diagnosis::NoDevice } else { Diagnosis::Unsupported };
        assert_eq!(puck.diagnosis(), expected);
    }

    /// The kernel drops zero valued relative events, so a puck returning to
    /// centre sends nothing at all. Silence is the only signal that it was let
    /// go, and without this the last deflection would steer forever.
    #[test]
    fn going_quiet_is_read_as_returning_to_centre() {
        let mut puck = SpaceMouse::inert();
        puck.config.stale_ms = 1;

        puck.simulate([300, 0, 0, 0, 0, 0]);
        let sample = puck.motion();
        assert!(sample.live);
        assert_eq!(sample.axis(Axis::Tx), 300.0);

        std::thread::sleep(std::time::Duration::from_millis(20));
        let sample = puck.motion();
        assert!(!sample.live, "a silent puck should read as centred");
        assert_eq!(sample.axes, [0.0; 6]);

        // ...and therefore moves the camera no further.
        let mut moved = camera();
        let before = moved.target;
        assert!(!puck.config.apply(&sample, 16.0, &mut moved, 720.0));
        assert_eq!(moved.target, before);
    }

    #[test]
    fn the_learnt_full_scale_grows_to_the_hardest_push_seen() {
        let puck = SpaceMouse::inert();
        let floor = puck.full_scale();
        assert!(floor > 0.0, "the readout would divide by zero before the first push");

        puck.simulate([0, 0, 0, 0, 0, -350]);
        assert_eq!(puck.full_scale(), 350.0, "magnitude, not sign, sets full scale");

        puck.simulate([10, 0, 0, 0, 0, 0]);
        assert_eq!(puck.full_scale(), 350.0, "full scale must not shrink back");
    }

    /// The application samples once per frame, so a press and release inside a
    /// single frame would vanish from a live pressed/released mask.
    #[test]
    fn every_press_fires_exactly_once_even_between_two_frames() {
        let mut puck = SpaceMouse::inert();
        assert!(puck.take_presses().is_empty(), "nothing was pressed yet");

        puck.shared.press(0);
        puck.shared.press(0);
        puck.shared.press(1);
        assert_eq!(
            puck.take_presses(),
            vec![ButtonAction::Undo, ButtonAction::Undo, ButtonAction::Redo]
        );
        assert!(puck.take_presses().is_empty(), "a press fired twice");
    }

    #[test]
    fn a_button_bound_to_nothing_fires_nothing() {
        let mut puck = SpaceMouse::inert();
        puck.config.buttons[0] = ButtonAction::None;
        puck.shared.press(0);
        assert!(puck.take_presses().is_empty());
    }

    // --- configuration ------------------------------------------------------

    #[test]
    fn the_settings_survive_a_round_trip_through_the_file_format() {
        let mut written = Config {
            mode: Mode::Camera,
            deadzone: 30.0,
            pan_sens: 1.5e-6,
            stale_ms: 200,
            buttons: [ButtonAction::ToggleSymmetry, ButtonAction::FrameModel],
            ..Config::default()
        };
        written.set_binding(Action::Roll, AxisBinding { source: Axis::Rz, invert: true });

        let mut read = Config::default();
        read.merge(&written.to_text());
        assert_eq!(read, written);
    }

    /// A broken config must never stop the application starting, and must
    /// never leave an action unbound — an unbound action would be a hole in
    /// the motion loop rather than a setting.
    #[test]
    fn a_corrupt_config_falls_back_to_the_defaults_without_unbinding_anything() {
        let mut config = Config::default();
        config.merge(
            "this line has no equals sign\n\
             mode = sideways\n\
             deadzone = banana\n\
             pan_sens = -3\n\
             zoom_sens = 0\n\
             stale_ms = 0\n\
             orbit_az = qq\n\
             pan_x =\n\
             button_1 = explode\n\
             button_9 = undo\n\
             unknown_key = tx\n\
             # a comment\n",
        );

        assert_eq!(config.mode, Config::default().mode);
        assert_eq!(config.deadzone, Config::default().deadzone);
        assert_eq!(config.pan_sens, Config::default().pan_sens, "a negative sensitivity");
        assert_eq!(config.zoom_sens, Config::default().zoom_sens, "a zero sensitivity");
        assert!(config.stale_ms >= 1, "a zero timeout would make the puck permanently stale");
        assert_eq!(config.buttons, Config::default().buttons);
        for action in Action::ALL {
            assert_eq!(
                config.binding(action),
                Config::default().binding(action),
                "{} lost its binding",
                action.key()
            );
        }
    }

    #[test]
    fn an_inverted_axis_is_written_and_read_back_with_its_sign() {
        assert_eq!(parse_binding("-rz"), Some(AxisBinding { source: Axis::Rz, invert: true }));
        assert_eq!(parse_binding("rz"), Some(AxisBinding { source: Axis::Rz, invert: false }));
        assert_eq!(parse_binding("nonsense"), None);
        // Whichever action happens to be inverted in the defaults must survive
        // being written out and read back.
        let text = Config::default().to_text();
        for action in Action::ALL {
            let binding = Config::default().binding(action);
            let sign = if binding.invert { "-" } else { "" };
            assert!(
                text.contains(&format!("{} = {sign}{}", action.key(), binding.source.key())),
                "{} did not round trip, file said:\n{text}",
                action.key()
            );
        }
    }

    #[test]
    fn button_numbering_runs_from_one_in_the_file_and_from_zero_in_the_array() {
        assert_eq!(button_index("button_1"), Some(0));
        assert_eq!(button_index("button_2"), Some(1));
        assert_eq!(button_index("button_0"), None, "there is no button zero");
        assert_eq!(button_index("button_3"), None, "beyond what can be bound");
        assert_eq!(button_index("buttons"), None);
    }
}

#[cfg(all(test, target_os = "linux"))]
mod uinput_tests {
    use super::*;
    use evdev::uinput::VirtualDevice;
    use evdev::{AttributeSet, EventType, InputEvent, KeyCode, RelativeAxisCode};
    use std::time::Duration;

    /// Roughly what a real SpaceNavigator reports at a firm push.
    const FULL_PUSH: i32 = 350;

    /// The six axes a puck has, and the two a mouse has. Building both is the
    /// point of the test: the decoy is what would break if the filter ever
    /// weakened to "has some relative axes" or to a vendor id.
    fn build(name: &str, axes: &[RelativeAxisCode], buttons: usize) -> Option<VirtualDevice> {
        let mut relative = AttributeSet::<RelativeAxisCode>::new();
        for axis in axes {
            relative.insert(*axis);
        }
        let mut keys = AttributeSet::<KeyCode>::new();
        for index in 0..buttons {
            keys.insert(KeyCode::new(KeyCode::BTN_0.0 + index as u16));
        }

        VirtualDevice::builder()
            .ok()?
            .name(name)
            .with_relative_axes(&relative)
            .ok()?
            .with_keys(&keys)
            .ok()?
            .build()
            .ok()
    }

    fn puck_axes() -> Vec<RelativeAxisCode> {
        vec![
            RelativeAxisCode::REL_X,
            RelativeAxisCode::REL_Y,
            RelativeAxisCode::REL_Z,
            RelativeAxisCode::REL_RX,
            RelativeAxisCode::REL_RY,
            RelativeAxisCode::REL_RZ,
        ]
    }

    fn emit(device: &mut VirtualDevice, events: &[InputEvent]) {
        device.emit(events).expect("could not emit to the virtual puck");
    }

    fn axis_event(axis: RelativeAxisCode, value: i32) -> InputEvent {
        InputEvent::new(EventType::RELATIVE.0, axis.0, value)
    }

    fn button_event(index: u16, down: bool) -> InputEvent {
        InputEvent::new(EventType::KEY.0, KeyCode::BTN_0.0 + index, i32::from(down))
    }

    /// The scanner rescans every two seconds and udev takes a moment to apply
    /// permissions to a freshly created node.
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
    fn a_virtual_puck_is_found_and_drives_the_camera() {
        let Some(mut device) = build("BrokkrSculpt test puck", &puck_axes(), 2) else {
            eprintln!(
                "skipping: cannot create a uinput device. Needs write access to /dev/uinput, \
                 usually via the input group."
            );
            return;
        };
        // A mouse. If this were adopted, every mouse movement would fly the
        // camera — which is exactly the bug SindriCAD shipped by matching on a
        // Logitech vendor id.
        let _decoy = build(
            "BrokkrSculpt test mouse",
            &[RelativeAxisCode::REL_X, RelativeAxisCode::REL_Y],
            3,
        );

        let mut puck = SpaceMouse::start();

        if !wait_for(|| puck.devices().iter().any(|d| d.name == "BrokkrSculpt test puck")) {
            if !can_read_own_node(&mut device) {
                eprintln!(
                    "skipping: the virtual puck was created but its /dev/input node cannot be opened by \
                     this process, so nothing could have seen it. That needs udev and \
                     membership of the input group, which a CI container has not got."
                );
                return;
            }
            panic!("the scanner never picked up the virtual puck");
        }
        assert!(
            !puck.devices().iter().any(|d| d.name == "BrokkrSculpt test mouse"),
            "a two axis mouse was mistaken for a 6DOF puck"
        );
        assert_eq!(
            puck.devices().iter().find(|d| d.name == "BrokkrSculpt test puck").unwrap().buttons,
            2
        );

        // Push the cap sideways. The axis is bound to pan by default.
        //
        // Held down rather than emitted once, because that is what the hardware
        // does: a real puck streams its current deflection every 8 ms for as
        // long as it is deflected, which is the whole reason the reader has a
        // 120 ms staleness timeout. A single event goes stale in less time than
        // `wait_for`'s 100 ms poll interval, so the one-shot version of this
        // failed about half the time on a loaded machine -- and read as a
        // SpaceMouse bug rather than as a test that did not model the device.
        let mut arrived = false;
        for _ in 0..250 {
            emit(&mut device, &[axis_event(RelativeAxisCode::REL_X, FULL_PUSH)]);
            if puck.motion().axis(Axis::Tx) == FULL_PUSH as f32 {
                arrived = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(8));
        }
        assert!(arrived, "the deflection never arrived, got {:?}", puck.motion());
        assert_eq!(puck.diagnosis(), Diagnosis::Working);
        assert_eq!(puck.full_scale(), FULL_PUSH as f32, "full scale was not learnt from use");

        let mut camera = OrbitCamera::framing(glam::Vec3::ZERO, 30.0);
        let before = camera.target;
        assert!(puck.config.apply(&puck.motion(), 16.0, &mut camera, 720.0));
        assert_ne!(camera.target, before, "a bound axis did not move the camera");

        // Deflection is absolute, not a delta: a second identical report must
        // read as the same push, not as twice the push.
        emit(&mut device, &[axis_event(RelativeAxisCode::REL_X, FULL_PUSH)]);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            puck.motion().axis(Axis::Tx),
            FULL_PUSH as f32,
            "repeated reports were accumulated instead of replacing each other"
        );

        // Let go. The kernel sends nothing at all, so only the staleness
        // timeout can notice, and the camera has to stop.
        assert!(
            wait_for(|| !puck.motion().live),
            "a puck that went quiet was still reported as deflected"
        );
        assert_eq!(puck.motion().axes, [0.0; 6]);

        // Buttons, including a press and release inside what would be one
        // frame — the case a live pressed/released mask would lose entirely.
        emit(&mut device, &[button_event(0, true), button_event(0, false)]);
        let mut fired = Vec::new();
        assert!(
            wait_for(|| {
                fired.extend(puck.take_presses());
                fired.contains(&ButtonAction::Undo)
            }),
            "the first button never fired"
        );

        emit(&mut device, &[button_event(1, true), button_event(1, false)]);
        let mut fired = Vec::new();
        assert!(
            wait_for(|| {
                fired.extend(puck.take_presses());
                fired.contains(&ButtonAction::Redo)
            }),
            "the second button never fired"
        );
    }
}
