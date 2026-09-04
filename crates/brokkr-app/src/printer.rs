// SPDX-License-Identifier: AGPL-3.0-only

//! Watching a Snapmaker U1 print, over Moonraker.
//!
//! Ported from SindriCAD's `src-tauri/src/printer.rs`, and deliberately only
//! the reading half of it. SindriCAD can also upload a sliced `.gcode` and
//! start a job with a filament remap; this application cannot produce a
//! `.gcode` at all, so that flow would be: export, open OrcaSlicer, slice,
//! save, come back here, pick the file, upload. Orca's own "Upload and print"
//! does the last three steps against the same machine from the window the user
//! is already in. What is worth having from in here is the other thing — how
//! far along the print is, without alt-tabbing to find out.
//!
//! # What is pinned to the machine
//!
//! Verified against the U1 on this network, firmware `1.3.0.168_20260414155825`,
//! which is the 1.3.0 SindriCAD's notes are written against:
//!
//! - **The LAN is fully trusted: there is no API key.** A plain GET works.
//! - Moonraker answers on **port 7125**.
//! - `GET /printer/info` carries `result.state` and `result.hostname`.
//! - `GET /printer/objects/query?print_stats&virtual_sdcard` carries
//!   `result.status.print_stats.state` — `standby`, `printing`, `paused`,
//!   `complete`, `error` — and `result.status.virtual_sdcard.progress`, a
//!   fraction from 0 to 1.
//! - `GET /printer/objects/query?print_task_config` carries what filament is
//!   loaded, as **parallel arrays** — see [`filaments`].
//!
//! # Why this is its own HTTP call and not the report sender's
//!
//! A printer is plain HTTP on a local network; TinkerAtlas is HTTPS and could
//! one day carry a credential. Sharing a request builder between them is how a
//! token ends up going out in clear text to a machine on the LAN, so they are
//! separate modules with separate constructors, and this one refuses anything
//! that is not `http://` at a validated bare host.

use std::time::Duration;

/// Moonraker's default port, and the U1's.
pub(crate) const MOONRAKER_PORT: u16 = 7125;

/// How long to wait for a printer that may be asleep or off.
///
/// Short: this runs while someone is sculpting, and a printer that is not there
/// should be reported as not there quickly rather than holding a task open.
const TIMEOUT: Duration = Duration::from_secs(4);

/// A host is a bare name or address, and nothing else.
///
/// Ported verbatim. It is what makes `format!("http://{host}:{port}")` safe
/// against a host that came from a settings file: without it a value like
/// `evil.example/x?` or `user@elsewhere` turns the format string into a URL
/// pointing somewhere else entirely.
///
/// **Validated on read as well as on write**, which is a deliberate departure
/// from SindriCAD -- it checks on upsert and then trusts its registry, so a
/// hand-edited file carries whatever it likes into that format string.
pub(crate) fn valid_host(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 253
        && host.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
}

/// What the printer is doing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Status {
    /// Moonraker's own word: `standby`, `printing`, `paused`, `complete`,
    /// `error`, or whatever a later firmware invents. Kept as text rather than
    /// an enum because it is a firmware string and this is a readout, so an
    /// unknown value should be shown rather than swallowed.
    pub state: String,
    /// How far through, 0 to 1, when there is a job.
    pub progress: f32,
    /// The file being printed, if any.
    pub filename: Option<String>,
}

impl Status {
    /// One line for the status bar.
    pub fn summary(&self) -> String {
        match self.state.as_str() {
            "printing" | "paused" => {
                let percent = (self.progress * 100.0).clamp(0.0, 100.0);
                match &self.filename {
                    Some(name) => format!("printer: {} {percent:.0}% — {name}", self.state),
                    None => format!("printer: {} {percent:.0}%", self.state),
                }
            }
            other => format!("printer: {other}"),
        }
    }
}

/// Ask a printer what it is doing.
///
/// Blocking, and called from inside a `Task` like every other network and
/// filesystem call here.
pub(crate) fn status(host: &str, port: u16) -> Result<Status, String> {
    parse_status(&query(host, port, "print_stats&virtual_sdcard")?)
}

/// One `printer/objects/query` against Moonraker, returning the raw body.
///
/// **The objects to ask for are a parameter rather than being folded into one
/// call**, because the two callers here run at completely different rates: the
/// status readout polls while someone is sculpting, and the filament list is
/// read when a button is pressed. Dragging five filament arrays through every
/// status poll would pay for a thing that changes when somebody walks over to
/// the machine and changes it.
fn query(host: &str, port: u16, objects: &str) -> Result<String, String> {
    if !valid_host(host) {
        return Err(format!("{host:?} is not a host name"));
    }
    let url = format!("http://{host}:{port}/printer/objects/query?{objects}");

    ureq::get(&url)
        .config()
        .timeout_global(Some(TIMEOUT))
        .build()
        .call()
        .map_err(|why| format!("could not reach {host} ({why})"))?
        .body_mut()
        .read_to_string()
        .map_err(|why| format!("{host} answered something unreadable ({why})"))
}

/// Pull the three interesting values out of Moonraker's reply.
///
/// Tolerant on purpose. A field that is absent means the printer has not been
/// asked to print since it booted, not that the reply is broken -- `print_stats`
/// exists with no `filename` in exactly that case -- so a missing value becomes
/// a default rather than an error. Only a reply that is not JSON at all, or has
/// no `result`, is a failure.
fn parse_status(body: &str) -> Result<Status, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(body).map_err(|why| format!("not JSON ({why})"))?;
    let status = parsed
        .get("result")
        .and_then(|result| result.get("status"))
        .ok_or_else(|| "no result.status in the reply".to_string())?;

    let stats = status.get("print_stats");
    Ok(Status {
        state: stats
            .and_then(|stats| stats.get("state"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        progress: status
            .get("virtual_sdcard")
            .and_then(|card| card.get("progress"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32,
        filename: stats
            .and_then(|stats| stats.get("filename"))
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty())
            .map(str::to_string),
    })
}

/// What is loaded in one of the printer's tool heads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Filament {
    pub vendor: String,
    pub material: String,
    pub sub_type: String,
    /// `#RRGGBB`, or empty when the machine did not say.
    pub colour: String,
    /// Whether there is anything in this head at all.
    pub present: bool,
}

impl Filament {
    /// What to call this in the panel.
    ///
    /// Vendor and material, falling back through material alone to the slot
    /// number -- the U1 answers with a blank vendor more often than not, and
    /// "Slot 2" is a better row label than a leading space.
    pub fn label(&self, slot: usize) -> String {
        let joined = format!("{} {}", self.vendor.trim(), self.material.trim());
        let joined = joined.trim();
        if joined.is_empty() { format!("Slot {slot}") } else { joined.to_string() }
    }
}

/// Ask a printer what filament is loaded.
///
/// `Ok(None)` means **the printer answered and does not report this**, which is
/// not a failure: `print_task_config` is a Snapmaker object, so a plain Klipper
/// or a Qidi on the same protocol simply has no such key. Reporting that as an
/// error would put "could not reach your printer" in front of someone whose
/// printer just answered.
pub(crate) fn filaments(host: &str, port: u16) -> Result<Option<Vec<Filament>>, String> {
    parse_filaments(&query(host, port, "print_task_config")?)
}

/// Pull the filament slots out of Moonraker's reply.
///
/// **The five fields are parallel arrays, not an array of objects**, so the
/// slot count is the longest of them rather than the length of any one -- a
/// firmware that omits `filament_vendor` for an empty head must not shorten the
/// list and renumber every slot after it.
///
/// `filament_color_rgba` is `RRGGBBAA` **despite the name**: the alpha is last,
/// so the colour is the leading six characters. That detail came from
/// SindriCAD's notes rather than from this codebase, which is why
/// `the_real_printer_answers_the_way_the_module_says_it_does` checks it against
/// the machine.
fn parse_filaments(body: &str) -> Result<Option<Vec<Filament>>, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(body).map_err(|why| format!("not JSON ({why})"))?;
    let status = parsed
        .get("result")
        .and_then(|result| result.get("status"))
        .ok_or_else(|| "no result.status in the reply".to_string())?;
    let Some(config) = status.get("print_task_config") else {
        return Ok(None);
    };

    let array = |key: &str| config.get(key).and_then(serde_json::Value::as_array);
    let text = |key: &str, at: usize| -> String {
        array(key)
            .and_then(|values| values.get(at))
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string()
    };

    const FIELDS: [&str; 5] = [
        "filament_vendor",
        "filament_type",
        "filament_sub_type",
        "filament_color_rgba",
        "filament_exist",
    ];
    let count = FIELDS.iter().filter_map(|key| array(key)).map(Vec::len).max().unwrap_or(0);
    if count == 0 {
        // The object is there but names no slots. Unknown, not empty -- a
        // machine that says nothing must not be reported as having no filament.
        return Ok(None);
    }

    Ok(Some(
        (0..count)
            .map(|at| Filament {
                vendor: text("filament_vendor", at),
                material: text("filament_type", at),
                sub_type: text("filament_sub_type", at),
                colour: rgb_from_rgba(&text("filament_color_rgba", at)),
                // Absent means loaded: a firmware that does not track presence
                // has filament in every head it names, and treating silence as
                // "empty" would refuse to sync anything at all.
                present: array("filament_exist")
                    .and_then(|values| values.get(at))
                    .is_none_or(truthy),
            })
            .collect(),
    ))
}

/// `RRGGBBAA` (with or without a leading `#`) to `#RRGGBB`.
///
/// Anything that is not six or more hex digits becomes empty rather than a
/// guess, so the palette keeps the colour it already had.
///
/// **Counted in characters, not bytes.** `len()` is bytes and `&text[..6]`
/// needs a character boundary, so a reply like `"aaaaaé"` is long enough to
/// pass a byte-length check and then panics inside the guard that was supposed
/// to reject it -- before the hex test can run. The body of this reply is
/// arbitrary UTF-8 off the network, and this runs inside the task the sync
/// button spawns, so that panic is reachable from anything answering on the
/// printer's port.
fn rgb_from_rgba(text: &str) -> String {
    let body = text.strip_prefix('#').unwrap_or(text);
    let hex: Vec<char> = body.chars().take(6).collect();
    if hex.len() < 6 || !hex.iter().all(char::is_ascii_hexdigit) {
        return String::new();
    }
    let hex: String = hex.into_iter().collect();
    format!("#{}", hex.to_ascii_uppercase())
}

/// Whether a JSON value means yes.
///
/// The U1 has been seen to answer `filament_exist` as booleans, and Moonraker
/// objects elsewhere use `0`/`1` integers for the same idea. Accepting both is
/// cheaper than discovering which one a firmware update chose.
fn truthy(value: &serde_json::Value) -> bool {
    value.as_bool().unwrap_or_else(|| value.as_i64().is_some_and(|number| number != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_host_is_a_bare_name_and_nothing_else() {
        assert!(valid_host("192.0.2.46"));
        assert!(valid_host("printer.local"));
        assert!(valid_host("my-printer"));
        // Every one of these would turn `http://{host}:{port}` into a URL
        // pointing somewhere other than the printer.
        assert!(!valid_host(""));
        assert!(!valid_host("http://192.0.2.46"));
        assert!(!valid_host("192.0.2.46/evil"));
        assert!(!valid_host("user@elsewhere.example"));
        assert!(!valid_host("host:1234"));
        assert!(!valid_host("a host"));
        assert!(!valid_host(&"x".repeat(254)));
    }

    #[test]
    fn a_bad_host_is_refused_before_a_request_is_built() {
        let why = status("192.0.2.1/evil", MOONRAKER_PORT).expect_err("that is not a host");
        assert!(why.contains("is not a host"), "it tried to connect anyway: {why}");
    }

    /// The exact shape the U1 on this network answers with, firmware 1.3.0.
    #[test]
    fn a_real_reply_is_read_the_way_the_machine_sends_it() {
        let body = r#"{"result":{"eventtime":2299854.1,"status":{
            "print_stats":{"filename":"dragon.gcode","state":"printing","print_duration":842.0},
            "virtual_sdcard":{"progress":0.4213,"is_active":true}}}}"#;
        let status = parse_status(body).expect("a real reply should parse");
        assert_eq!(status.state, "printing");
        assert_eq!(status.filename.as_deref(), Some("dragon.gcode"));
        assert!((status.progress - 0.4213).abs() < 1.0e-6);
        assert_eq!(status.summary(), "printer: printing 42% — dragon.gcode");
    }

    #[test]
    fn an_idle_printer_that_has_never_printed_is_not_an_error() {
        // What the machine actually answers from a cold boot: the objects are
        // there, the filename is an empty string, progress is absent.
        let body = r#"{"result":{"status":{"print_stats":{"filename":"","state":"standby"}}}}"#;
        let status = parse_status(body).expect("an idle printer is not a failure");
        assert_eq!(status.state, "standby");
        assert_eq!(status.filename, None, "an empty filename is no filename");
        assert_eq!(status.progress, 0.0);
        assert_eq!(status.summary(), "printer: standby");
    }

    #[test]
    fn a_state_this_build_has_never_heard_of_is_shown_rather_than_swallowed() {
        let body = r#"{"result":{"status":{"print_stats":{"state":"cancelling"}}}}"#;
        let status = parse_status(body).expect("an unknown state is still a state");
        assert_eq!(status.summary(), "printer: cancelling");
    }

    #[test]
    fn a_reply_that_is_not_moonraker_is_refused() {
        assert!(parse_status("<html>not a printer</html>").is_err());
        assert!(parse_status("{}").is_err(), "no result.status is not a status");
        assert!(parse_status(r#"{"result":{}}"#).is_err());
    }

    /// The shape SindriCAD's notes say the U1 answers with, firmware 1.3.0.
    ///
    /// Deliberately lower case and `RRGGBBAA`, which is the pair of details
    /// most likely to be wrong in a second-hand fact.
    #[test]
    fn the_filament_arrays_are_read_as_parallel_columns() {
        let body = r#"{"result":{"status":{"print_task_config":{
            "filament_vendor":["Snapmaker","Polymaker","",""],
            "filament_type":["PLA","PETG","",""],
            "filament_sub_type":["Basic","Tough","",""],
            "filament_color_rgba":["e8e8e8ff","202020ff","",""],
            "filament_exist":[true,true,false,false]}}}}"#;
        let slots = parse_filaments(body).expect("a real reply parses").expect("it names slots");
        assert_eq!(slots.len(), 4);
        assert_eq!(slots[0].colour, "#E8E8E8", "the alpha was not dropped, or case was kept");
        assert_eq!(slots[0].label(1), "Snapmaker PLA");
        assert_eq!(slots[1].sub_type, "Tough");
        assert!(!slots[2].present);
        assert_eq!(slots[2].label(3), "Slot 3", "a blank head produced a blank label");
    }

    #[test]
    fn a_printer_that_does_not_report_filament_is_unknown_rather_than_broken() {
        // A plain Klipper or a Qidi has no `print_task_config` at all. Saying
        // "could not reach your printer" about a printer that just answered is
        // the failure this guards.
        let body = r#"{"result":{"status":{}}}"#;
        assert_eq!(parse_filaments(body).expect("still a valid reply"), None);
        // Present but naming nothing is equally unknown, not "no filament".
        let empty = r#"{"result":{"status":{"print_task_config":{}}}}"#;
        assert_eq!(parse_filaments(empty).expect("still a valid reply"), None);
        // Not a Moonraker reply at all is still a failure.
        assert!(parse_filaments("<html>nope</html>").is_err());
    }

    #[test]
    fn a_ragged_reply_keeps_every_slot_in_its_own_place() {
        // A firmware that omits a column for an empty head must not shorten the
        // list and renumber every slot after it.
        let body = r#"{"result":{"status":{"print_task_config":{
            "filament_vendor":["A"],
            "filament_type":["PLA","PETG","ABS"],
            "filament_color_rgba":["ff0000ff"]}}}}"#;
        let slots = parse_filaments(body).expect("parses").expect("names slots");
        assert_eq!(slots.len(), 3, "the longest column sets the slot count");
        assert_eq!(slots[1].material, "PETG");
        assert_eq!(slots[1].colour, "", "a missing colour became a guess");
        assert!(slots[2].present, "an absent presence column was read as empty");
    }

    #[test]
    fn presence_is_accepted_as_a_boolean_or_as_a_number() {
        let body = r#"{"result":{"status":{"print_task_config":{
            "filament_type":["PLA","PLA","PLA"],
            "filament_exist":[1,0,true]}}}}"#;
        let slots = parse_filaments(body).expect("parses").expect("names slots");
        assert!(slots[0].present);
        assert!(!slots[1].present);
        assert!(slots[2].present);
    }

    #[test]
    fn a_colour_that_is_not_one_is_left_empty_rather_than_guessed() {
        assert_eq!(rgb_from_rgba("e8e8e8ff"), "#E8E8E8");
        assert_eq!(rgb_from_rgba("#E8E8E8"), "#E8E8E8");
        assert_eq!(rgb_from_rgba("abc"), "");
        assert_eq!(rgb_from_rgba(""), "");
        assert_eq!(rgb_from_rgba("zzzzzzff"), "");
        // Byte six falls INSIDE a multi-byte character here. Slicing the string
        // by byte offset panics on this, inside the very guard meant to reject
        // it -- and the value comes off the network.
        assert_eq!(rgb_from_rgba("aaaaaé"), "");
        assert_eq!(rgb_from_rgba("ééééééé"), "");
        assert_eq!(rgb_from_rgba("#aaaaaé"), "");
    }

    #[test]
    fn a_bad_host_is_refused_before_a_filament_request_is_built() {
        let why = filaments("192.0.2.1/evil", MOONRAKER_PORT).expect_err("that is not a host");
        assert!(why.contains("is not a host"), "it tried to connect anyway: {why}");
    }

    /// The config file this application names is the one it reads.
    ///
    /// `printer` was the odd name out among four `.conf` files, and a correct
    /// `printer.conf` sat unread on the author's own machine for a fortnight
    /// while the application reported no printer at all.
    #[test]
    fn the_printer_config_is_named_like_every_other_config() {
        let Some(path) = config_path() else {
            println!("skipping: no config directory on this machine");
            return;
        };
        let name = path.file_name().and_then(std::ffi::OsStr::to_str).unwrap_or_default();
        assert!(
            name == "printer.conf" || name == "printer",
            "the printer config resolved to {name:?}, which is neither name"
        );
        // When neither file exists, the name offered is the modern one -- that
        // is the string a "no printer set" message tells someone to create.
        if !path.exists() {
            assert_eq!(name, "printer.conf", "an absent config offered the legacy name");
        }
    }
    /// Against the real machine, when there is one. Skips loudly otherwise, the
    /// way the uinput device tests do, because a test that needs hardware must
    /// not fail on a machine that does not have it.
    ///
    /// Point it somewhere with `BROKKR_TEST_PRINTER=192.0.2.46`. This is what
    /// turns every fact in the module docs from something read in another
    /// codebase into something observed here.
    #[test]
    fn the_real_printer_answers_the_way_the_module_says_it_does() {
        let Ok(host) = std::env::var("BROKKR_TEST_PRINTER") else {
            println!("skipping: set BROKKR_TEST_PRINTER to a Moonraker host to run this");
            return;
        };
        let status = status(&host, MOONRAKER_PORT)
            .unwrap_or_else(|why| panic!("{host} did not answer: {why}"));
        println!("{host}: {}", status.summary());
        assert!(!status.state.is_empty(), "a printer with no state at all");
        assert!(
            (0.0..=1.0).contains(&status.progress),
            "progress is a fraction, not a percentage: {}",
            status.progress
        );

        // The filament half. Everything asserted here was read in SindriCAD's
        // notes rather than observed from this codebase, and this is the one
        // place that turns it into an observation -- above all the claim that
        // `filament_color_rgba` puts the alpha LAST.
        let slots = filaments(&host, MOONRAKER_PORT)
            .unwrap_or_else(|why| panic!("{host} did not answer about filament: {why}"));
        let Some(slots) = slots else {
            println!("{host} does not report print_task_config; nothing more to check");
            return;
        };
        assert!(!slots.is_empty(), "print_task_config named zero slots");
        for (index, slot) in slots.iter().enumerate() {
            println!(
                "  slot {}: {} [{}] {} present={}",
                index + 1,
                slot.label(index + 1),
                slot.material,
                slot.colour,
                slot.present
            );
            assert!(
                slot.colour.is_empty() || slot.colour.len() == 7,
                "slot {} colour {:?} is not #RRGGBB -- is the alpha really last?",
                index + 1,
                slot.colour
            );
        }
    }
}

/// Where the printer's address is kept.
///
/// **`printer.conf` first, then the extensionless `printer`.** The original
/// name was `printer`, and it is the odd one out: `spacemouse.conf`,
/// `welcome.conf`, `update.conf` and now `filaments.conf` all carry the
/// extension, so a user writing this file by hand writes `printer.conf` by
/// analogy and the application never looks at it. Found on the author's own
/// machine, where a correct `printer.conf` had sat unread since 2026-08-19 and
/// every "check the printer" said "no printer set".
///
/// The old name is still read so nobody who followed the old error message
/// loses their setting; the new name is what a missing-file message names,
/// because that is the one to write today.
pub(crate) fn config_path() -> Option<std::path::PathBuf> {
    let preferred = crate::paths::config_file("printer.conf")?;
    if preferred.exists() {
        return Some(preferred);
    }
    let legacy = crate::paths::config_file("printer")?;
    if legacy.exists() {
        return Some(legacy);
    }
    Some(preferred)
}

/// Where the printer's address is kept, and what is in it.
///
/// A flat `key = value` file, the same shape `spacemouse.conf` uses, because
/// this workspace already has a parser-shaped-like-that and adding a config
/// format is a dependency and a second thing to get wrong. Two keys:
///
/// ```text
/// host = 192.0.2.46
/// port = 7125
/// ```
///
/// **Seeded empty, and there is no discovery.** SindriCAD's registry ships with
/// its author's own LAN addresses in it, which is a machine nobody else has. A
/// printer here is something the user writes down once.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Printer {
    pub host: String,
    pub port: u16,
}

/// Read the configured printer, if there is one and it is usable.
///
/// The host is validated **on read** as well as on write. SindriCAD validates
/// on upsert and then trusts its registry, so a hand-edited file carries
/// whatever it likes into a URL; this is a file a user is expected to edit by
/// hand, which makes that gap the normal case rather than the exotic one.
pub(crate) fn configured(from: Option<&std::path::Path>) -> Option<Printer> {
    let text = std::fs::read_to_string(from?).ok()?;
    let mut host = None;
    let mut port = MOONRAKER_PORT;
    for (key, value) in crate::paths::entries(&text) {
        match key {
            "host" => host = Some(value.to_string()),
            // An unparseable port leaves the default rather than failing the
            // whole file: the same forgiveness the spacemouse config gives.
            "port" => port = value.parse().unwrap_or(MOONRAKER_PORT),
            _ => {}
        }
    }
    let host = host.filter(|host| valid_host(host))?;
    Some(Printer { host, port })
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use std::io::Write;

    /// A temp file per call. Named by a counter rather than by its contents:
    /// a body containing a slash makes a path that is not a file name, which
    /// is a confusing way for a test to fail.
    fn write(body: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "brokkr-printer-{}-{}.conf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::File::create(&path).expect("a temp file");
        file.write_all(body.as_bytes()).expect("write");
        path
    }

    #[test]
    fn a_written_down_printer_is_read_back() {
        let path = write(
            "# my machine
host = 192.0.2.46
port = 7125
",
        );
        assert_eq!(
            configured(Some(&path)),
            Some(Printer { host: "192.0.2.46".into(), port: 7125 })
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_port_is_optional_and_defaults() {
        let path = write(
            "host = printer.local
",
        );
        assert_eq!(configured(Some(&path)).map(|p| p.port), Some(MOONRAKER_PORT));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_hand_edited_file_cannot_smuggle_a_url_in_as_a_host() {
        // The gap SindriCAD leaves: it validates on write and trusts on read,
        // and this is a file a user is meant to edit.
        let path = write(
            "host = evil.example/x?a=b
",
        );
        assert_eq!(configured(Some(&path)), None, "a non-host was accepted from the file");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn nonsense_leaves_the_defaults_rather_than_failing_the_file() {
        let path = write(
            "host = printer.local
port = banana
nonsense
unknown = 3
",
        );
        assert_eq!(
            configured(Some(&path)),
            Some(Printer { host: "printer.local".into(), port: MOONRAKER_PORT })
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn no_file_and_no_host_are_both_simply_no_printer() {
        assert_eq!(configured(None), None);
        assert_eq!(configured(Some(std::path::Path::new("/nowhere/at/all"))), None);
        let path = write(
            "port = 7125
",
        );
        assert_eq!(configured(Some(&path)), None);
        let _ = std::fs::remove_file(path);
    }
}
