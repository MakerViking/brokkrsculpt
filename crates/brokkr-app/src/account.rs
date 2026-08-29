// SPDX-License-Identifier: AGPL-3.0-only

//! Signing in to TinkerAtlas, so a bug report can be replied to.
//!
//! **Why there is a sign-in at all.** Every report this application sends was
//! anonymous by design, and anonymous is the half that matters least: a report
//! nobody can answer is one that often cannot be acted on, and asking the
//! reporter a single question is usually the difference between a fix and a
//! guess. Signing in is optional and stays optional -- signed out, nothing
//! about the old behaviour changes.
//!
//! # The flow, which is SindriCAD's and RFC 8252's
//!
//! There is no webview here to log in inside, and there does not need to be:
//!
//! 1. Bind a listener on `127.0.0.1:0` and let the kernel pick the port.
//! 2. Mint a random nonce and open the real browser at
//!    `tinkeratlas.com/brokkrsculpt/authorize?port=…&state=<nonce>`.
//! 3. The user logs in on the site as they normally would -- with a password
//!    manager, with social login, with all of it -- and clicks Authorize.
//! 4. The site redirects the browser to `http://127.0.0.1:<port>/callback`
//!    carrying a desktop token and the nonce back.
//! 5. This module checks the nonce, exchanges the token for a profile at
//!    `/api/desktop/me`, and writes both to a mode-600 file.
//!
//! **The nonce is the whole security of step 4.** The listener is on loopback,
//! but loopback is not private: any process on this machine can connect to it
//! and post a token of its choosing. What it cannot do is guess a nonce it
//! never saw, so a callback that does not carry ours is dropped unread.
//!
//! # The token
//!
//! It is a credential, and this repository is public. It is never logged,
//! never put in a status line, never included in a crash report, and never
//! rendered by [`std::fmt::Debug`] -- [`Account`] has a hand-written one for
//! exactly that reason, because a derived `Debug` on a struct that holds a
//! secret is one `dbg!` away from a leak. The only place it is written is the
//! file below, and that file is created 0600 before anything is written into
//! it.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Where the site is asked to send the browser.
const AUTHORIZE_URL: &str = "https://tinkeratlas.com/brokkrsculpt/authorize";

/// Where a token is exchanged for the profile it belongs to.
const ME_URL: &str = "https://tinkeratlas.com/api/desktop/me";

/// How long to wait for the user to finish in the browser.
///
/// Generous on purpose: this covers reading a consent screen, and often a
/// login and a password manager prompt before it. Too short and the token
/// arrives at a listener that has already given up, which reads to the user as
/// "signing in is broken" rather than "that took a while".
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(180);

/// How long the profile lookup may take. Short: it is one round trip, and the
/// user is watching a spinner by now.
const HTTP_TIMEOUT: Duration = Duration::from_secs(15);

/// A ceiling on the callback request line.
///
/// The connection is from an unknown local process until the nonce says
/// otherwise, so what it sends is read with a limit rather than to EOF.
const MAX_REQUEST_LINE: u64 = 8 * 1024;

/// The signed-in user, and the token that proves it.
#[derive(Clone)]
pub struct Account {
    pub username: String,
    pub display_name: String,
    /// Empty when the profile has no picture. Not fetched by this module.
    pub avatar_url: String,
    /// **Private, and it stays private.** Read it through
    /// [`Account::authorization`] so that every use is a deliberate one.
    token: String,
}

impl Account {
    /// The `Authorization` header value for an authenticated request.
    pub fn authorization(&self) -> String {
        format!("Bearer {}", self.token)
    }

    /// What to call this person on screen, preferring what they chose.
    pub fn label(&self) -> &str {
        if !self.display_name.is_empty() { &self.display_name } else { &self.username }
    }
}

/// Hand-written so that no accidental `{:?}` can ever print the token.
///
/// The derived form would put a live credential into whatever it was written
/// to -- a log line, a panic message, a crash report. This one names the field
/// without its value, so a `Debug` that was added for debugging stays that.
impl std::fmt::Debug for Account {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Account")
            .field("username", &self.username)
            .field("display_name", &self.display_name)
            .field("avatar_url", &self.avatar_url)
            .field("token", &"<redacted>")
            .finish()
    }
}

/// Where the signed-in account is kept.
///
/// State and not config: it is something the application recovers, not
/// something the user set, and it must never be mistaken for a preference
/// worth syncing or committing.
fn account_file() -> Option<PathBuf> {
    crate::paths::state_file("tinkeratlas-account.json")
}

/// Read the stored account, if there is one.
///
/// Every failure is the same answer -- signed out. A corrupt or unreadable
/// file is not worth an error on the welcome screen: the remedy is to sign in
/// again, which is the button already on screen.
pub fn load() -> Option<Account> {
    let text = std::fs::read_to_string(account_file()?).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let field = |name: &str| value.get(name).and_then(|v| v.as_str()).unwrap_or_default();

    let token = field("token").to_string();
    if token.is_empty() {
        return None;
    }
    Some(Account {
        username: field("username").to_string(),
        display_name: field("display_name").to_string(),
        avatar_url: field("avatar_url").to_string(),
        token,
    })
}

/// Write the account, replacing whatever was there.
///
/// **0600 before any bytes, and a rename rather than a write in place.** The
/// permissions are set when the temporary file is created, so the token is
/// never briefly world-readable; the rename is atomic, so a crash halfway
/// through leaves either the old account or the new one and never half of a
/// token.
fn store(account: &Account) -> Result<(), String> {
    let path = account_file().ok_or("there is nowhere to keep the sign-in")?;
    let parent = path.parent().ok_or("the account path has no directory")?;
    std::fs::create_dir_all(parent).map_err(|why| format!("could not make {parent:?} ({why})"))?;

    let body = serde_json::json!({
        "token": account.token,
        "username": account.username,
        "display_name": account.display_name,
        "avatar_url": account.avatar_url,
    })
    .to_string();

    let temporary = path.with_extension("json.tmp");
    write_private(&temporary, body.as_bytes())
        .map_err(|why| format!("could not write the sign-in ({why})"))?;
    std::fs::rename(&temporary, &path).map_err(|why| {
        let _ = std::fs::remove_file(&temporary);
        format!("could not save the sign-in ({why})")
    })
}

/// Create a file readable only by this user and write to it.
///
/// `OpenOptions::mode` applies the bits at creation, which is the point: a
/// `write` followed by `set_permissions` leaves a window in which the token is
/// on disk with the default mask.
///
/// # Windows has no mode, and that is not an oversight
///
/// `std::os::unix` does not exist there, so the bits are `cfg`'d off -- and a
/// bare `cfg` is exactly what a later reader would take for a forgotten
/// platform, so: **the token is protected by the per-user app data directory
/// instead.** `%LOCALAPPDATA%` sits under a profile whose ACLs already exclude
/// other standard users, which is the same set `0o600` excludes. What both let
/// through is an administrator, precisely as `0o600` lets through root, so the
/// threat models line up rather than one being a downgrade of the other.
///
/// SindriCAD reached the same answer for the same file -- see
/// `src-tauri/src/tinkeratlas.rs`, which wraps its own `set_permissions` in
/// `cfg(unix)` -- and one behaviour across the two apps is worth more here than
/// a Windows-only ACL call that would need a Windows machine to test.
///
/// macOS is unix and keeps the mode. Gating on `unix` and not on
/// `target_os = "linux"` is what makes that true.
fn write_private(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Forget the account. Signed out is the absence of the file.
pub fn sign_out() -> Result<(), String> {
    let Some(path) = account_file() else {
        return Ok(());
    };
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        // Already gone is the state that was asked for.
        Err(why) if why.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(why) => Err(format!("could not sign out ({why})")),
    }
}

/// Sixteen random bytes as hex, from the kernel.
///
/// `getrandom` rather than `/dev/urandom`, which is where this started and
/// which does not exist on Windows -- the nonce test was the single thing that
/// failed there once the token file compiled.
///
/// **Not a new dependency.** iced pulls winit pulls ahash pulls exactly this
/// crate at exactly this version, so naming it directly adds no code to the
/// build and no second supply chain to audit. The original note here argued
/// that a dependency for thirty-two characters was not worth a supply chain;
/// that was right, and it is why this is the one crate that costs nothing.
///
/// The server requires `[a-f0-9]{16,64}`, which this satisfies at 32.
fn nonce() -> Result<String, String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|why| format!("no randomness available ({why})"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Run the whole sign-in, blocking. Call it from `Task::perform`.
///
/// Blocking for the same reason every other network call here is: the work is
/// one browser round trip and one request, and an async runtime to avoid two
/// blocking waits would be a large dependency for a button nobody presses
/// twice.
pub fn sign_in() -> Result<Account, String> {
    // Port 0 asks the kernel for a free one, which is what keeps two copies of
    // the application from fighting over a fixed port -- and means no port has
    // to be reserved or documented anywhere.
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .map_err(|why| format!("could not listen for the browser ({why})"))?;
    let port = listener
        .local_addr()
        .map_err(|why| format!("could not read the listening port ({why})"))?
        .port();
    listener
        .set_nonblocking(true)
        .map_err(|why| format!("could not set the listener up ({why})"))?;

    let state = nonce()?;
    let url = format!("{AUTHORIZE_URL}?port={port}&state={state}");
    // Through the same one-host check every article link goes through, so this
    // cannot become a second way to open an arbitrary URL.
    crate::articles::open_in_browser(&url)?;

    let token = wait_for_callback(&listener, &state)?;
    let account = fetch_profile(token)?;
    store(&account)?;
    Ok(account)
}

/// Wait for the browser to come back, and take the token out of it.
///
/// Polls rather than blocking in `accept`, so the deadline is real: a blocking
/// accept that is never satisfied is a thread parked for the life of the
/// process, and a user who closed the browser tab instead of clicking
/// Authorize is an ordinary way to get there.
///
/// A connection that does not carry our nonce is answered and dropped, and the
/// wait CONTINUES. Treating a wrong nonce as the end of the attempt would let
/// any local process cancel a sign-in by connecting first.
fn wait_for_callback(listener: &TcpListener, state: &str) -> Result<String, String> {
    let deadline = Instant::now() + CALLBACK_TIMEOUT;
    while Instant::now() < deadline {
        match listener.accept() {
            Ok((stream, _)) => match read_callback(stream, state) {
                Ok(Some(token)) => return Ok(token),
                // Not ours, or unreadable. Keep waiting.
                Ok(None) | Err(_) => continue,
            },
            Err(why) if why.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(120));
            }
            Err(why) => return Err(format!("the browser could not connect ({why})")),
        }
    }
    Err("signing in timed out — nothing came back from the browser".to_string())
}

/// Read one callback request, answer it, and return the token if it is ours.
///
/// `Ok(None)` means the request was well formed but not the one we are waiting
/// for; the caller keeps listening.
fn read_callback(mut stream: TcpStream, state: &str) -> Result<Option<String>, String> {
    stream.set_read_timeout(Some(Duration::from_secs(5))).map_err(|why| format!("{why}"))?;

    // Only the request line is needed, and only a bounded amount of it.
    let mut line = String::new();
    BufReader::new(stream.try_clone().map_err(|why| format!("{why}"))?)
        .take(MAX_REQUEST_LINE)
        .read_line(&mut line)
        .map_err(|why| format!("{why}"))?;

    let target = line.split_whitespace().nth(1).unwrap_or_default();
    let query = target.split_once('?').map(|(_, q)| q).unwrap_or_default();
    let mut token = String::new();
    let mut echoed = String::new();
    for pair in query.split('&') {
        match pair.split_once('=') {
            Some(("token", value)) => token = percent_decode(value),
            Some(("state", value)) => echoed = percent_decode(value),
            _ => {}
        }
    }

    // **Checked before the token is looked at, let alone kept.** Anything on
    // this machine can reach this port; only the browser we sent knows the
    // nonce.
    let ours = !echoed.is_empty() && echoed == state && !token.is_empty();
    respond(&mut stream, ours);
    Ok(ours.then_some(token))
}

/// Tell the person in the browser what happened. Deliberately plain: this is
/// the last thing they see before going back to the application.
fn respond(stream: &mut TcpStream, ours: bool) {
    let body = if ours {
        "<!doctype html><meta charset=utf-8><title>Signed in</title>\
         <body style=\"font-family:system-ui;padding:3rem;text-align:center\">\
         <h1>Signed in</h1><p>You can close this tab and go back to BrokkrSculpt.</p>"
    } else {
        "<!doctype html><meta charset=utf-8><title>Sign-in failed</title>\
         <body style=\"font-family:system-ui;padding:3rem;text-align:center\">\
         <h1>That did not come from BrokkrSculpt</h1>\
         <p>Start the sign-in again from the application.</p>"
    };
    let status = if ours { "200 OK" } else { "400 Bad Request" };
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    // Best effort: the token is already in hand, and a browser that hung up
    // early must not fail the sign-in.
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// Undo the `%XX` escaping a query string arrives with.
///
/// Hand-rolled because the alternative is a URL crate for one loop: the only
/// values read here are a hex nonce and a token of `[A-Za-z0-9_]`, neither of
/// which needs escaping, so this exists to be correct about input rather than
/// because anything is expected to use it. `+` is a space in a query string.
fn percent_decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or_default();
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        index += 3;
                    }
                    Err(_) => {
                        out.push(b'%');
                        index += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Exchange a token for the profile it belongs to.
///
/// This is also what proves the token works: a sign-in that stored a token
/// without ever using it would report success and then fail on the first bug
/// report, which is the worst moment to discover it.
fn fetch_profile(token: String) -> Result<Account, String> {
    let mut response = ureq::get(ME_URL)
        .config()
        .timeout_global(Some(HTTP_TIMEOUT))
        .build()
        .header("authorization", &format!("Bearer {token}"))
        .call()
        .map_err(|why| match why {
            ureq::Error::StatusCode(401) => "TinkerAtlas refused the sign-in".to_string(),
            ureq::Error::StatusCode(code) => format!("tinkeratlas.com answered {code}"),
            other => format!("could not reach tinkeratlas.com ({other})"),
        })?;

    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|why| format!("could not read the reply ({why})"))?;
    profile_from_json(&text, token)
}

/// Pull the account out of what `/api/desktop/me` answered.
///
/// Split from the request so the wire shape can be pinned by a test. It is
/// worth pinning: this cost a working sign-in once already.
fn profile_from_json(text: &str, token: String) -> Result<Account, String> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|why| format!("the reply was not JSON ({why})"))?;
    // **`{ "user": { … } }`, which is what the route actually returns.**
    // Read off production rather than assumed: the first version of this
    // looked for `data` and fell back to the whole object, and neither finds
    // `username` in the real reply -- so every sign-in would have got as far
    // as a valid token and then failed with "TinkerAtlas did not say who that
    // is". `data` and the bare profile stay as fallbacks so a change of
    // wrapper does not break sign-in a second time.
    let profile = value.get("user").or_else(|| value.get("data")).unwrap_or(&value);
    let field = |name: &str| profile.get(name).and_then(|v| v.as_str()).unwrap_or_default();

    let username = field("username").to_string();
    let display_name = field("display_name").to_string();
    if username.is_empty() && display_name.is_empty() {
        return Err("TinkerAtlas did not say who that is".to_string());
    }
    Ok(Account { username, display_name, avatar_url: field("avatar_url").to_string(), token })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The token must not be printable by accident.** A derived `Debug` here
    /// would put a live credential into any log line or panic message that
    /// formatted an `Account`, and this repository is public.
    #[test]
    fn debug_never_prints_the_token() {
        let account = Account {
            username: "maker".to_string(),
            display_name: "Maker".to_string(),
            avatar_url: String::new(),
            token: "ta_bskt_supersecretvalue".to_string(),
        };
        let shown = format!("{account:?}");
        assert!(!shown.contains("supersecret"), "the token was printed: {shown}");
        assert!(shown.contains("redacted"), "the field vanished instead of being masked");
        // And the header is still built from the real thing.
        assert_eq!(account.authorization(), "Bearer ta_bskt_supersecretvalue");
    }

    #[test]
    fn a_nonce_is_hex_and_long_enough_for_the_server() {
        let a = nonce().expect("the kernel has randomness");
        assert_eq!(a.len(), 32, "the server wants 16 to 64 hex characters");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
        assert_ne!(a, nonce().expect("twice"), "the nonce is not random");
    }

    #[test]
    fn percent_decoding_handles_what_a_query_string_carries() {
        assert_eq!(percent_decode("ta_bskt_abc123"), "ta_bskt_abc123");
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("a+b"), "a b");
        // A stray or truncated escape is passed through rather than eaten.
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
    }

    /// **The exact shape `/api/desktop/me` returns**, copied from a live
    /// response rather than from the route's source, because the source is
    /// what I read the first time and still got this wrong.
    ///
    /// `NextResponse.json({ user })` wraps the profile in `user`. Looking for
    /// `data` and falling back to the whole object finds neither, so a
    /// perfectly good token produced "TinkerAtlas did not say who that is" at
    /// the last step of every sign-in.
    #[test]
    fn the_profile_is_read_out_of_the_shape_production_actually_sends() {
        let body = r#"{"user":{"id":"70aec187","username":"MakerViking",
            "display_name":"MakerViking","avatar_url":"https://example.test/a.jpg"}}"#;
        let account = profile_from_json(body, "ta_bskt_x".to_string()).expect("a profile");
        assert_eq!(account.username, "MakerViking");
        assert_eq!(account.display_name, "MakerViking");
        assert_eq!(account.avatar_url, "https://example.test/a.jpg");
    }

    #[test]
    fn a_bare_profile_or_a_data_wrapper_still_works() {
        for body in [
            r#"{"username":"m","display_name":"M"}"#,
            r#"{"data":{"username":"m","display_name":"M"}}"#,
        ] {
            let account = profile_from_json(body, "t".to_string()).expect("a profile");
            assert_eq!(account.username, "m", "failed on {body}");
        }
    }

    /// A reply naming nobody is a failure, not an account with a blank name:
    /// storing it would show an empty row and send reports under nothing.
    #[test]
    fn a_reply_that_names_nobody_is_refused() {
        assert!(profile_from_json(r#"{"user":{"id":"x"}}"#, "t".to_string()).is_err());
        assert!(profile_from_json("not json", "t".to_string()).is_err());
    }

    const STATE: &str = "0123456789abcdef0123456789abcdef";

    /// A listener set up exactly as [`sign_in`] sets one up.
    fn listener_on_a_free_port() -> (TcpListener, u16) {
        let listener =
            TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("a free port");
        let port = listener.local_addr().expect("an address").port();
        listener.set_nonblocking(true).expect("nonblocking");
        (listener, port)
    }

    /// Connect and send one callback, the way the browser's redirect does,
    /// then read the reply so the answering path is exercised too.
    fn send_callback(port: u16, query: &str) {
        if let Ok(mut stream) = TcpStream::connect((Ipv4Addr::LOCALHOST, port)) {
            let _ = stream.write_all(
                format!("GET /callback?{query} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").as_bytes(),
            );
            let mut reply = Vec::new();
            let _ = stream.read_to_end(&mut reply);
        }
    }

    /// **The half nothing else proves.** The site's part is one redirect;
    /// everything that can actually go wrong lives on this side -- binding,
    /// reading the request line, matching the nonce, answering so the tab does
    /// not hang -- and a test of the parser alone would touch none of it.
    #[test]
    fn a_callback_carrying_our_nonce_hands_the_token_over() {
        let (listener, port) = listener_on_a_free_port();
        std::thread::spawn(move || {
            send_callback(port, &format!("token=ta_bskt_realtoken&state={STATE}"));
        });
        let token = wait_for_callback(&listener, STATE).expect("the callback was accepted");
        assert_eq!(token, "ta_bskt_realtoken");
    }

    /// **A stranger must not be able to end the wait.** Anything on this
    /// machine can reach the port. If the first connection decided the
    /// outcome, any local process could cancel a sign-in -- or land its own
    /// token -- by racing the browser.
    #[test]
    fn a_callback_with_the_wrong_nonce_is_ignored_and_the_wait_goes_on() {
        let (listener, port) = listener_on_a_free_port();
        std::thread::spawn(move || {
            send_callback(port, "token=ta_bskt_stolen&state=ffffffffffffffffffffffffffffffff");
            std::thread::sleep(Duration::from_millis(150));
            send_callback(port, &format!("token=ta_bskt_genuine&state={STATE}"));
        });
        let token = wait_for_callback(&listener, STATE).expect("the real one arrived");
        assert_eq!(token, "ta_bskt_genuine", "the impostor's token was taken");
    }

    /// A callback with a good nonce but no token is not a sign-in. Accepting
    /// it would store an empty credential, report success, and fail on the
    /// first bug report -- the worst moment to find out.
    #[test]
    fn a_callback_with_no_token_does_not_count_as_signing_in() {
        let (listener, port) = listener_on_a_free_port();
        std::thread::spawn(move || {
            send_callback(port, &format!("state={STATE}"));
            std::thread::sleep(Duration::from_millis(150));
            send_callback(port, &format!("token=ta_bskt_genuine&state={STATE}"));
        });
        let token = wait_for_callback(&listener, STATE).expect("the real one arrived");
        assert_eq!(token, "ta_bskt_genuine", "an empty token was accepted");
    }

    #[test]
    fn label_prefers_the_name_the_user_chose() {
        let mut account = Account {
            username: "maker".to_string(),
            display_name: "Maker Viking".to_string(),
            avatar_url: String::new(),
            token: "t".to_string(),
        };
        assert_eq!(account.label(), "Maker Viking");
        account.display_name = String::new();
        assert_eq!(account.label(), "maker", "an empty display name hid the user");
    }
}
