// SPDX-License-Identifier: AGPL-3.0-or-later

//! The bug report: what is collected, what is stripped out of it, and the exact
//! bytes that go to TinkerAtlas.
//!
//! Ported from SindriCAD's `src-tauri/src/tinkeratlas.rs`, which is the same
//! report against the same endpoint. The pure parts came over nearly verbatim
//! along with their tests -- there is no reason for two applications to disagree
//! about what a home directory looks like -- and the parts that were Tauri's
//! (the `AppHandle`, the webview version, the Python sidecar's log) are gone or
//! replaced with this application's equivalents.
//!
//! # The privacy rule
//!
//! `Brokkr::diagnostics` was written as something the user reads and pastes
//! themselves, and its doc says so: "the user can read exactly what they are
//! sending before they send it". Uploading it changes that bargain, so the
//! same text now goes through [`redact_user_paths`] first, and the dialog shows
//! the finished payload before anything is sent.
//!
//! **The server re-scrubs everything.** TinkerAtlas's `/api/desktop/bug-report`
//! strips ANSI, control, zero-width, bidi and Unicode-tag characters and redacts
//! paths again, deliberately, because a client's privacy guarantee is not one.
//! That is defence in depth and not a reason to send less carefully: it never
//! sees what we chose not to collect.

/// Longest first line that becomes the report's title.
///
/// Counted in `char`s and not bytes. The server caps the title at 200 bytes,
/// so a shorter char limit here cannot overflow it, and cutting a multi-byte
/// character in half would not survive the trip as valid UTF-8.
const TITLE_CHARS: usize = 120;

/// Ordinary breadcrumbs kept, most recent last.
const MAX_TAIL_CRUMBS: usize = 20;
/// Sticky facts kept. See [`trim_breadcrumbs`].
const MAX_STICKY_CRUMBS: usize = 24;

/// Replace the segment after a home-directory marker with `[REDACTED]`.
///
/// Ported verbatim from SindriCAD, whose tests came with it. It handles the
/// Windows drive form, the UNC form with the host preserved, and both unix
/// forms, and a segment ends at a separator, whitespace or a quote — which is
/// what makes it work on log lines rather than only on bare paths.
///
/// It is deliberately not clever. It knows about home directories and nothing
/// else: not emails, not tokens, not hostnames, not the name of a file that
/// happens to sit outside a home directory. Anything collected here has to be
/// worth sending on that basis, rather than on the basis that this will catch
/// what should not have been collected.
pub(crate) fn redact_user_paths(text: &str) -> String {
    fn redact_after(hay: &str, needle_lower: &str, separators: &[char]) -> String {
        let mut out = String::with_capacity(hay.len());
        let lower = hay.to_lowercase();
        let mut i = 0;
        while let Some(rel) = lower[i..].find(needle_lower) {
            let start = i + rel + needle_lower.len();
            out.push_str(&hay[i..start]);
            let rest = &hay[start..];
            let end = rest
                .find(|c: char| {
                    separators.contains(&c) || c.is_whitespace() || c == '"' || c == '\''
                })
                .unwrap_or(rest.len());
            if end > 0 {
                out.push_str("[REDACTED]");
            }
            i = start + end;
        }
        out.push_str(&hay[i..]);
        out
    }

    let text = redact_after(text, "\\users\\", &['\\', '/']);
    let text = redact_after(&text, "/home/", &['/', '\\']);
    redact_after(&text, "/users/", &['/', '\\'])
}

/// FNV-1a 64 bit, as lowercase hex.
///
/// The server groups repeat reports by this. A grouping key and not a security
/// hash, which is the whole reason it is not sha2: collisions merely merge two
/// unrelated reports, and avoiding them is not worth a dependency.
pub(crate) fn fnv1a64_hex(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// True when a crumb carries the `HH:MM:SS ` prefix an ordinary one gets.
fn is_timestamped(crumb: &str) -> bool {
    let bytes = crumb.as_bytes();
    bytes.len() >= 9
        && bytes[0].is_ascii_digit()
        && bytes[1].is_ascii_digit()
        && bytes[2] == b':'
        && bytes[3].is_ascii_digit()
        && bytes[4].is_ascii_digit()
        && bytes[5] == b':'
        && bytes[6].is_ascii_digit()
        && bytes[7].is_ascii_digit()
        && bytes[8] == b' '
}

/// Keep the last twenty ordinary crumbs, but never at the cost of the sticky
/// facts recorded at startup.
///
/// Ported along with the bug it fixes, which is worth repeating because the
/// obvious implementation has it. SindriCAD's was a plain "keep the last
/// twenty", and the facts captured once at startup — the device inventory, the
/// graphics adapter — sit at the front of the trail, so that dropped exactly
/// them the moment the session did anything. In other words it dropped them in
/// every session busy enough to be worth reporting, paying the privacy cost of
/// collecting an inventory and delivering none of its diagnostic value.
///
/// **The timestamp is the discriminator.** `crumb` always prefixes one and
/// `sticky_fact` never does, and a sticky fact that gained one would silently
/// become droppable again. See `breadcrumbs.rs`, which owns the other half of
/// that contract.
pub(crate) fn trim_breadcrumbs(crumbs: &[String]) -> Vec<String> {
    let (ordinary, sticky): (Vec<&String>, Vec<&String>) =
        crumbs.iter().partition(|crumb| is_timestamped(crumb));
    sticky
        .into_iter()
        .take(MAX_STICKY_CRUMBS)
        .chain(ordinary.into_iter().rev().take(MAX_TAIL_CRUMBS).rev())
        .cloned()
        .collect()
}

/// Everything one report sends.
///
/// Assembled and redacted before it is shown to the user, so what the dialog
/// displays is what leaves the machine rather than a description of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Report {
    pub title: String,
    pub description: String,
    pub app_version: String,
    pub os: String,
    pub os_build: String,
    pub arch: String,
    /// The graphics adapter, backend and driver.
    ///
    /// Sent under the key SindriCAD uses for the WebView version, because on
    /// both applications it answers the same question — which platform
    /// component is the story — and because a second key would need the server
    /// to learn about it. That it is a lie about the field's name is the cost
    /// of not changing a schema for one field; see the handoff.
    pub renderer: String,
    pub breadcrumbs: Vec<String>,
    pub log_tail: Option<String>,
    pub fingerprint: String,
}

impl Report {
    /// Build a report from what the user typed and what the application knows.
    ///
    /// Every free-text field is redacted here rather than at the call sites, so
    /// there is one place to look when asking what leaves the machine.
    pub fn new(
        description: &str,
        diagnostics: &str,
        breadcrumbs: &[String],
        log_tail: Option<String>,
        version: &str,
        renderer: &str,
    ) -> Self {
        let description = redact_user_paths(description.trim());
        // The first line becomes the title, and the fingerprint is taken over
        // it, so a user who reports the same thing twice gets one row rather
        // than two. Taken over the REDACTED text, so two people whose paths
        // differ but whose problem does not still group together.
        let first_line = description.lines().next().unwrap_or("bug report").to_string();
        let title: String = first_line.chars().take(TITLE_CHARS).collect();

        // The version is in the fingerprint so that a fixed bug reopening in a
        // later build is a new row rather than a bump on a closed one.
        let fingerprint = fnv1a64_hex(&format!("{version}|{first_line}"));

        Self {
            title,
            // The diagnostics block rides along with the description, because
            // the server has no field for it and it is the half of the report
            // that is actually diagnosable.
            description: format!("{description}\n\n---\n{}", redact_user_paths(diagnostics)),
            app_version: version.to_string(),
            os: std::env::consts::OS.to_string(),
            os_build: os_build(),
            arch: std::env::consts::ARCH.to_string(),
            renderer: redact_user_paths(renderer),
            breadcrumbs: trim_breadcrumbs(breadcrumbs)
                .into_iter()
                .map(|crumb| redact_user_paths(&crumb))
                .collect(),
            log_tail: log_tail.map(|tail| redact_user_paths(&tail)),
            fingerprint,
        }
    }

    /// The exact JSON body the endpoint receives.
    ///
    /// Hand written rather than through `serde_json`, which measured at three
    /// new crates for one fixed twelve-key object. The escaping is the only
    /// part that can be got wrong, so it is one function with its own tests and
    /// there is a golden test over the whole body. If a second thing in this
    /// workspace ever needs to write JSON, take the dependency instead of
    /// growing this.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push('{');
        push_string(&mut out, "product", PRODUCT);
        out.push(',');
        push_string(&mut out, "title", &self.title);
        out.push(',');
        push_string(&mut out, "description", &self.description);
        out.push(',');
        push_string(&mut out, "app_version", &self.app_version);
        out.push(',');
        push_string(&mut out, "os", &self.os);
        out.push(',');
        push_string(&mut out, "os_version", &self.os_build);
        out.push(',');
        push_string(&mut out, "arch", &self.arch);
        out.push(',');
        push_string(&mut out, "webview_version", &self.renderer);
        out.push(',');
        // No sidecar in this application. Sent anyway, and always false, because
        // the field is not optional on the other side.
        out.push_str("\"sidecar_connected\":false,");
        push_string(&mut out, "fingerprint", &self.fingerprint);
        out.push_str(",\"breadcrumbs\":[");
        for (index, crumb) in self.breadcrumbs.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            push_escaped(&mut out, crumb);
        }
        out.push(']');
        if let Some(tail) = &self.log_tail {
            out.push(',');
            push_string(&mut out, "log_tail", tail);
        }
        out.push('}');
        out
    }
}

/// Which application this report came from.
///
/// TinkerAtlas files desktop reports with a `source` column, and its
/// `/api/desktop/bug-report` route sets it to the literal `"sindricad"`. Until
/// the route reads this field a BrokkrSculpt report lands in the SindriCAD
/// queue -- which is why it is sent from the first version, so that the day the
/// server learns about it, every client already speaks it.
const PRODUCT: &str = "brokkrsculpt";

fn push_string(out: &mut String, key: &str, value: &str) {
    push_escaped(out, key);
    out.push(':');
    push_escaped(out, value);
}

/// A JSON string literal, escaped per RFC 8259.
///
/// Control characters below 0x20 have to be escaped or the body is not JSON,
/// and a log tail is exactly where one turns up.
fn push_escaped(out: &mut String, value: &str) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Which OS build this is, for the reports where the distribution is the story.
///
/// SindriCAD added this after a report it could not diagnose: the only platform
/// facts it collected were "linux" and "x86_64", neither of which says which
/// desktop or driver stack is installed. Linux reads `/etc/os-release` rather
/// than shelling out; the other platforms return nothing, which is honest.
fn os_build() -> String {
    if cfg!(target_os = "linux")
        && let Ok(text) = std::fs::read_to_string("/etc/os-release")
    {
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("PRETTY_NAME=") {
                return value.trim_matches('"').to_string();
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SindriCAD's own cases, brought over unchanged. Each of these is a shape
    /// that turned up in a real field report.
    #[test]
    fn user_paths_are_redacted_in_every_form_a_log_line_takes() {
        // Windows drive, and the exact shape a spawn line has.
        assert_eq!(
            redact_user_paths(r"spawn: C:\Users\jrabi\AppData\Local\BrokkrSculpt\app.exe"),
            r"spawn: C:\Users\[REDACTED]\AppData\Local\BrokkrSculpt\app.exe"
        );
        // UNC, where the HOST is not a username and has to survive.
        assert_eq!(
            redact_user_paths(r"\\HOST\Users\someone\share"),
            r"\\HOST\Users\[REDACTED]\share"
        );
        // Both unix forms, including a name that is not this machine's.
        assert_eq!(
            redact_user_paths("path /home/alice/proj/x.stl and /Users/bob/y"),
            "path /home/[REDACTED]/proj/x.stl and /Users/[REDACTED]/y"
        );
        // A segment ends at a quote, not only at a separator.
        assert_eq!(
            redact_user_paths(r#"log at "C:\Users\name" end"#),
            r#"log at "C:\Users\[REDACTED]" end"#
        );
        // Text with nothing to redact comes back byte identical.
        assert_eq!(redact_user_paths("no paths here"), "no paths here");
    }

    /// The case this application will actually hit: its own status line.
    #[test]
    fn the_status_line_this_application_writes_is_redacted() {
        assert_eq!(
            redact_user_paths("saved /home/thomash/Downloads/sculpt2.brokkr"),
            "saved /home/[REDACTED]/Downloads/sculpt2.brokkr"
        );
    }

    #[test]
    fn the_fingerprint_is_stable_hex_and_moves_with_the_build() {
        let a = fnv1a64_hex("0.0.1+abc123|the window went black");
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
        assert_eq!(a, fnv1a64_hex("0.0.1+abc123|the window went black"));
        // A different build is a different group, so a bug that comes back
        // after a fix opens a new row rather than bumping a closed one.
        assert_ne!(a, fnv1a64_hex("0.0.1+def456|the window went black"));
    }

    #[test]
    fn sticky_facts_survive_a_busy_session_and_lead_the_trail() {
        // The regression this function exists for: a plain "keep the last
        // twenty" drops exactly the startup facts, and only once the session
        // has been busy enough to be worth reporting.
        let mut crumbs = vec![
            "[wgpu] AMD Radeon RX 6900 XT, Vulkan".to_string(),
            "[tablet] no tablet found".to_string(),
        ];
        for index in 0..60 {
            crumbs.push(format!("12:00:0{} noise {index}", index % 10));
        }

        let out = trim_breadcrumbs(&crumbs);
        assert!(out[0].starts_with("[wgpu]"), "sticky facts must lead: {out:?}");
        assert!(out.iter().any(|c| c.contains("Radeon")), "the adapter was dropped");
        assert!(out.iter().any(|c| c.contains("noise 59")), "the newest crumb was dropped");
        assert!(!out.iter().any(|c| c.contains("noise 0")), "an old crumb survived");
        assert_eq!(out.len(), 2 + MAX_TAIL_CRUMBS);
    }

    #[test]
    fn a_short_plain_trail_is_left_alone() {
        let crumbs: Vec<String> =
            (0..5).map(|index| format!("12:00:0{index} step {index}")).collect();
        assert_eq!(trim_breadcrumbs(&crumbs), crumbs);
    }

    #[test]
    fn sticky_facts_are_themselves_bounded() {
        let crumbs: Vec<String> = (0..100).map(|index| format!("[fact] {index}")).collect();
        assert_eq!(trim_breadcrumbs(&crumbs).len(), MAX_STICKY_CRUMBS);
    }

    #[test]
    fn a_title_is_cut_by_characters_and_not_by_bytes() {
        // Cutting by bytes would split a multi-byte character and the body
        // would not be valid UTF-8. Every one of these is four bytes.
        let description = "🔨".repeat(200);
        let report = Report::new(&description, "", &[], None, "0.0.1", "");
        assert_eq!(report.title.chars().count(), TITLE_CHARS);
        assert!(report.title.chars().all(|c| c == '🔨'));
    }

    #[test]
    fn json_escapes_everything_that_would_otherwise_break_the_body() {
        let mut out = String::new();
        push_escaped(&mut out, "a \"quote\" and a \\ and a \nnewline\tand \u{7} a bell");
        assert_eq!(out, r#""a \"quote\" and a \\ and a \nnewline\tand \u0007 a bell""#);
    }

    /// The golden test that makes a hand written body safe to ship.
    ///
    /// It is deliberately an exact string. A field renamed, reordered, dropped
    /// or left unescaped fails here rather than at the server, which answers a
    /// 400 with no detail.
    #[test]
    fn the_wire_body_is_exactly_this() {
        let report = Report {
            title: "it went black".to_string(),
            description: "it went black\n\n---\nBrokkrSculpt 0.0.1".to_string(),
            app_version: "0.0.1+abc".to_string(),
            os: "linux".to_string(),
            os_build: "CachyOS".to_string(),
            arch: "x86_64".to_string(),
            renderer: "Radeon RX 6900 XT (Vulkan)".to_string(),
            breadcrumbs: vec!["[wgpu] Vulkan".to_string(), "12:00:01 opened".to_string()],
            log_tail: None,
            fingerprint: "0123456789abcdef".to_string(),
        };
        assert_eq!(
            report.to_json(),
            concat!(
                r#"{"product":"brokkrsculpt","title":"it went black","#,
                r#""description":"it went black\n\n---\nBrokkrSculpt 0.0.1","#,
                r#""app_version":"0.0.1+abc","os":"linux","os_version":"CachyOS","#,
                r#""arch":"x86_64","webview_version":"Radeon RX 6900 XT (Vulkan)","#,
                r#""sidecar_connected":false,"fingerprint":"0123456789abcdef","#,
                r#""breadcrumbs":["[wgpu] Vulkan","12:00:01 opened"]}"#,
            )
        );
    }

    #[test]
    fn a_log_tail_is_included_only_when_there_is_one() {
        let with = Report { log_tail: Some("boom".into()), ..sample() };
        assert!(with.to_json().contains(r#""log_tail":"boom""#));
        assert!(!sample().to_json().contains("log_tail"));
    }

    fn sample() -> Report {
        Report::new("x", "", &[], None, "0.0.1", "")
    }

    #[test]
    fn the_description_carries_the_diagnostics_and_both_are_redacted() {
        let report = Report::new(
            "crashed after saving to /home/thomash/x.brokkr",
            "last message: saved /home/thomash/x.brokkr",
            &[],
            None,
            "0.0.1",
            "",
        );
        assert!(!report.description.contains("thomash"), "a home directory reached the payload");
        assert!(report.description.contains("last message:"), "the diagnostics were dropped");
        assert!(report.title.contains("[REDACTED]"), "the title is taken from redacted text");
    }
}

/// Where reports go.
///
/// A parameter rather than a constant, which is a deliberate departure from
/// SindriCAD's compile-time `option_env!`: there, a shipped binary cannot be
/// repointed and no test can reach the sender at all. It is passed rather than
/// read from the environment because the tests below run in parallel and a
/// process-global would have them pointing each other at the wrong server --
/// which is exactly what happened when it was one.
pub(crate) const TINKERATLAS: &str = "https://tinkeratlas.com";

fn endpoint(base: &str) -> String {
    format!("{}/api/desktop/bug-report", base.trim_end_matches('/'))
}

/// How long to wait before giving up.
///
/// The send runs inside a `Task`, so a slow server stalls the report and not
/// the window. Fifteen seconds is long enough for a cold serverless start and
/// short enough that a user does not conclude the application has hung.
const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Post a report, blocking.
///
/// Called from inside `Task::perform`, which is where every other blocking
/// filesystem call in this application already lives. An async client would
/// mean an async runtime, and the whole request is one round trip.
///
/// **No account, no token, no identifier.** SindriCAD attaches a bearer token
/// when one happens to be cached and sends the report anonymously otherwise,
/// because "the application is broken" usually comes before signing in. This
/// application has no sign-in at all, so every report is anonymous and there is
/// no credential in this workspace to store, leak or get wrong. The only thing
/// tying two reports together is the fingerprint, which is a hash of the build
/// and the first line.
pub(crate) fn send(report: Report, base: &str) -> Result<String, String> {
    let body = report.to_json();
    let response = ureq::post(&endpoint(base))
        .config()
        .timeout_global(Some(TIMEOUT))
        .build()
        .header("content-type", "application/json")
        .send(&body);

    match response {
        Ok(_) => Ok("report sent — thank you".to_string()),
        Err(ureq::Error::StatusCode(429)) => {
            Err("too many reports from here; try again in an hour".to_string())
        }
        Err(ureq::Error::StatusCode(413)) => {
            Err("the report is too large; untick the details".to_string())
        }
        Err(ureq::Error::StatusCode(code)) => Err(format!("the server answered {code}")),
        // Everything else is the network: no route, DNS, TLS, timeout. Worth
        // distinguishing from a rejection, because the answer is "try again"
        // rather than "change the report".
        Err(other) => Err(format!("could not reach tinkeratlas.com ({other})")),
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Serve exactly one request and hand back what arrived.
    ///
    /// A real socket rather than a mocked client: it exercises the bytes this
    /// actually puts on the wire, which is the half a unit test of `to_json`
    /// cannot reach. No new dependency -- `std::net` is enough for one
    /// request and one canned reply.
    fn serve_once(
        status: &'static str,
        body: &'static str,
    ) -> (String, std::thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let base = format!("http://{}", listener.local_addr().expect("an address"));
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("one connection");
            // Read until the whole declared body has arrived, not once. A
            // single `read` returns whatever one TCP segment held, which for
            // this request is the headers and none of the body -- and a test
            // that asserts over the headers alone silently stops checking the
            // payload, which is the only interesting part.
            let mut seen: Vec<u8> = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let read = stream.read(&mut chunk).unwrap_or(0);
                if read == 0 {
                    break;
                }
                seen.extend_from_slice(&chunk[..read]);
                let text = String::from_utf8_lossy(&seen);
                let Some(head_end) = text.find("\r\n\r\n") else {
                    continue;
                };
                let declared = text
                    .to_lowercase()
                    .split("content-length:")
                    .nth(1)
                    .and_then(|rest| rest.split("\r\n").next())
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if seen.len() >= head_end + 4 + declared {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&seen).to_string();
            let reply = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(reply.as_bytes());
            let _ = stream.flush();
            request
        });
        (base, handle)
    }

    fn sample_report() -> Report {
        Report::new("the window went black", "wgpu: vulkan", &[], None, "0.0.1", "")
    }

    #[test]
    fn the_endpoint_is_built_without_a_double_slash() {
        assert_eq!(endpoint("https://x.test"), "https://x.test/api/desktop/bug-report");
        assert_eq!(endpoint("https://x.test/"), "https://x.test/api/desktop/bug-report");
    }

    #[test]
    fn a_report_arrives_as_a_json_post_carrying_the_payload() {
        let (base, server) = serve_once("200 OK", r#"{"success":true}"#);
        let outcome = send(sample_report(), &base);
        let request = server.join().expect("the stub server panicked");

        assert!(outcome.is_ok(), "a 200 was not treated as success: {outcome:?}");
        assert!(request.starts_with("POST /api/desktop/bug-report"), "wrong route: {request}");
        assert!(
            request.to_lowercase().contains("content-type: application/json"),
            "the body was not declared as JSON"
        );
        assert!(request.contains(r#""product":"brokkrsculpt""#), "the product tag was not sent");
        assert!(request.contains("the window went black"), "the description was not sent");
        assert!(
            !request.to_lowercase().contains("authorization"),
            "an anonymous report must carry no credential"
        );
    }

    #[test]
    fn a_rate_limit_says_so_rather_than_reporting_a_generic_failure() {
        let (base, server) = serve_once("429 Too Many Requests", r#"{"error":"slow down"}"#);
        let outcome = send(sample_report(), &base);
        let _ = server.join();
        let why = outcome.expect_err("a 429 is not a success");
        assert!(why.contains("try again in an hour"), "unhelpful message: {why}");
    }

    #[test]
    fn an_oversized_report_is_told_what_to_do_about_it() {
        let (base, server) = serve_once("413 Payload Too Large", "{}");
        let outcome = send(sample_report(), &base);
        let _ = server.join();
        let why = outcome.expect_err("a 413 is not a success");
        assert!(why.contains("untick"), "the message does not say what to do: {why}");
    }

    #[test]
    fn an_unreachable_server_reads_as_a_network_problem() {
        // Port 1 on loopback: nothing listens, and the connection is refused
        // immediately rather than hanging.
        let why =
            send(sample_report(), "http://127.0.0.1:1").expect_err("nothing is listening there");
        assert!(why.contains("could not reach"), "unhelpful message: {why}");
    }
}
