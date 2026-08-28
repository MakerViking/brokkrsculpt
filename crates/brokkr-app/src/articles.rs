// SPDX-License-Identifier: AGPL-3.0-only

//! The TinkerAtlas articles shown on the welcome screen.
//!
//! SindriCAD embeds `tinkeratlas.com/sindricad/welcome` in an `<iframe>` and
//! lets the page render itself. There is no webview here to put a frame in, so
//! this fetches the same site's articles as DATA and draws them with the same
//! widgets as everything else.
//!
//! # Why RSS and not the JSON API
//!
//! `GET /api/articles?limit=5` returns everything an article has, including its
//! full rendered HTML body and its search vector: **105 KB for five titles**.
//! The RSS feed is 43 KB for the whole list and carries exactly what a link
//! needs -- title, link, description, date, author.
//!
//! The deciding argument is the dependency policy rather than the bytes.
//! Parsing the JSON would mean `serde_json`; `roxmltree` is already a compiled
//! dependency of this workspace for the 3MF importer, so RSS costs **zero new
//! crates**. `ureq` is likewise already here for the bug reporter, and the line
//! that workspace holds is that it stays the only network dependency -- this
//! adds no second one.
//!
//! # When this runs, and when it does not
//!
//! **Only while the welcome screen is actually on screen.** Everything else
//! this application does is local: the bug reporter talks to the network when a
//! user asks it to, and `printer.rs` talks to a machine on the LAN. A fetch on
//! every launch would be a new outbound connection nobody asked for, and this
//! application has deliberately no account and no stored credential.
//!
//! Tying it to the screen makes the control honest: **turning the welcome
//! screen off also turns this off**, with no second setting to find, and the
//! tick that does it is on the screen itself.

use std::time::Duration;

/// Where the feed lives.
const FEED_URL: &str = "https://tinkeratlas.com/api/rss/articles";

/// How long to wait before deciding the site is unreachable.
///
/// Short on purpose. This is decoration on a screen the user is already
/// looking at, so the cost of waiting is worse than the cost of missing it --
/// and the offline state says what happened and offers a retry.
const TIMEOUT: Duration = Duration::from_secs(6);

/// How many to show. The column holds about this many before the card grows
/// taller than a 768-high window, which is the size `tool_strip`'s own header
/// records as the one that must keep working.
pub const MAX_SHOWN: usize = 4;

/// One article, reduced to what a link needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Article {
    pub title: String,
    pub link: String,
    pub summary: String,
    /// As the feed wrote it, trimmed to the date. Not parsed into a calendar
    /// type: nothing here does arithmetic on it, and a date format that fails
    /// to parse should still be shown rather than swallowed.
    pub date: String,
}

/// What the welcome screen has to draw.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Feed {
    /// Not asked for yet.
    #[default]
    Idle,
    /// Asked, nothing back.
    Loading,
    Ready(Vec<Article>),
    /// Unreachable, with what to say about it.
    Failed(String),
}

/// Fetch and parse the feed. Blocking: call it off the interface thread.
pub fn fetch() -> Result<Vec<Article>, String> {
    // Same shape as `report.rs`'s send, which is the only other place this
    // workspace touches the network.
    let mut response =
        ureq::get(FEED_URL).config().timeout_global(Some(TIMEOUT)).build().call().map_err(
            |why| match why {
                ureq::Error::StatusCode(code) => format!("tinkeratlas.com answered {code}"),
                // Everything else is the network: no route, DNS, TLS, timeout.
                // Worth saying differently from a rejection, because the answer is
                // "try again" rather than "something is wrong with the request".
                other => format!("could not reach tinkeratlas.com ({other})"),
            },
        )?;
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|why| format!("the feed could not be read ({why})"))?;
    parse(&body)
}

/// Pull the items out of an RSS document.
///
/// Deliberately tolerant: an item missing a title or a link is skipped rather
/// than failing the whole feed, because one malformed entry on the site should
/// not blank the panel. A document that is not XML at all IS an error -- that
/// is a captive portal or an error page, and saying "no articles" for it would
/// be a lie about what happened.
pub fn parse(xml: &str) -> Result<Vec<Article>, String> {
    let document =
        roxmltree::Document::parse(xml).map_err(|why| format!("the feed is not XML ({why})"))?;

    let mut articles = Vec::new();
    for item in document.descendants().filter(|node| node.has_tag_name("item")) {
        let child = |name: &str| {
            item.children()
                .find(|node| node.has_tag_name(name))
                .and_then(|node| node.text())
                .unwrap_or_default()
                .trim()
                .to_string()
        };
        let title = child("title");
        let link = child("link");
        if title.is_empty() || link.is_empty() {
            continue;
        }
        // Only ever https, and only ever this site. The link is handed to the
        // browser, so a feed that had been tampered with -- or a future entry
        // pointing somewhere else entirely -- must not be able to send the user
        // anywhere the welcome screen did not promise.
        if !link.starts_with("https://tinkeratlas.com/") {
            continue;
        }
        articles.push(Article {
            title,
            link,
            summary: shorten(&child("description")),
            date: date_only(&child("pubDate")),
        });
        if articles.len() == MAX_SHOWN {
            break;
        }
    }
    Ok(articles)
}

/// The day out of an RFC-822 date, without the time or the zone.
///
/// **Not a string split on `" 0"`, which is what this was first**, and the
/// screenshot is what caught it: that cuts at the first space-then-zero, so
/// `Tue, 18 Aug 2026 22:29:33 GMT` kept its whole time because the hour starts
/// with a two, and `Sun, 03 Aug 2026 …` was cut down to `Sun,` because the DAY
/// starts with a zero. Two different wrong answers from one clever line.
///
/// Taking the first four whitespace-separated words is what the format
/// actually promises -- `Thu, 27 Aug 2026 00:38:32 GMT` -- and anything that
/// is not that shape is passed through untouched rather than mangled, because
/// a date we cannot read is still better shown than blanked.
fn date_only(pub_date: &str) -> String {
    let words: Vec<&str> = pub_date.split_whitespace().collect();
    if words.len() < 4 {
        return pub_date.trim().to_string();
    }
    words[..4].join(" ")
}

/// Cut a summary to something that fits beside the actions without growing the
/// card. See `panel.rs`'s `shorten`, which does the same job for paths and
/// records why a fixed-width column does not clip its own children.
fn shorten(text: &str) -> String {
    const MAX_CHARS: usize = 110;
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= MAX_CHARS {
        return flat;
    }
    let cut: String = flat.chars().take(MAX_CHARS - 1).collect();
    // Back off to a word boundary so the ellipsis does not land mid-word.
    let trimmed = cut.rsplit_once(' ').map(|(head, _)| head).unwrap_or(&cut);
    format!("{trimmed}…")
}

/// Hand a link to whatever the desktop opens links with.
///
/// `xdg-open` rather than a crate: this is Linux-first, `slicer.rs` already
/// spawns a process the same way, and a browser-opening dependency would be a
/// second one to audit for the sake of one command.
pub fn open_in_browser(link: &str) -> Result<(), String> {
    if !link.starts_with("https://tinkeratlas.com/") {
        return Err("that link does not lead to TinkerAtlas".to_string());
    }
    std::process::Command::new("xdg-open")
        .arg(link)
        .spawn()
        .map(|_| ())
        .map_err(|why| format!("nothing would open the link ({why})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FEED: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>TinkerAtlas Articles</title>
    <item>
      <title>Your Sketch Went Red</title>
      <link>https://tinkeratlas.com/site-posts/your-sketch-went-red</link>
      <description>A constraint that turns your sketch red and refuses to say which line caused it is the same bug as a save that quietly truncates your file, and it took ten days to fix properly.</description>
      <pubDate>Thu, 27 Aug 2026 00:38:32 GMT</pubDate>
    </item>
    <item>
      <title>Second</title>
      <link>https://tinkeratlas.com/site-posts/second</link>
      <description>Short one.</description>
      <pubDate>Wed, 26 Aug 2026 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"#;

    /// **The dates that broke the first version, kept as the test.**
    ///
    /// Both came off the real feed and both were wrong on screen: an hour that
    /// does not start with zero kept the whole time, and a day that does was
    /// cut down to the weekday alone.
    #[test]
    fn a_date_keeps_its_day_and_drops_its_time_whatever_the_digits_are() {
        assert_eq!(date_only("Thu, 27 Aug 2026 00:38:32 GMT"), "Thu, 27 Aug 2026");
        assert_eq!(
            date_only("Tue, 18 Aug 2026 22:29:33 GMT"),
            "Tue, 18 Aug 2026",
            "an hour not starting with zero kept its time"
        );
        assert_eq!(
            date_only("Sun, 03 Aug 2026 09:00:00 GMT"),
            "Sun, 03 Aug 2026",
            "a day starting with zero was cut down to the weekday"
        );
        // Anything not of that shape is shown rather than mangled.
        assert_eq!(date_only("yesterday"), "yesterday");
        assert_eq!(date_only(""), "");
    }

    #[test]
    fn the_feed_yields_the_fields_a_link_needs() {
        let articles = parse(FEED).expect("the fixture is valid RSS");
        assert_eq!(articles.len(), 2);
        assert_eq!(articles[0].title, "Your Sketch Went Red");
        assert_eq!(articles[0].link, "https://tinkeratlas.com/site-posts/your-sketch-went-red");
        assert_eq!(articles[0].date, "Thu, 27 Aug 2026", "the time of day is not worth the width");
        assert!(articles[0].summary.ends_with('…'), "a long summary was not cut");
        assert_eq!(articles[1].summary, "Short one.", "a short summary was cut anyway");
    }

    /// **A link is handed to the browser, so it may only ever go one place.**
    ///
    /// The feed is fetched over TLS from a site we control, which is most of
    /// the answer -- but "most" is not the standard for a URL that a click
    /// hands to the user's browser. An entry pointing anywhere else is dropped
    /// rather than shown, and `open_in_browser` refuses it a second time in
    /// case a link ever reaches it by another route.
    #[test]
    fn an_item_pointing_off_the_site_is_dropped_and_refused() {
        let hostile =
            FEED.replace("https://tinkeratlas.com/site-posts/second", "https://evil.example/phish");
        let articles = parse(&hostile).expect("still valid RSS");
        assert_eq!(articles.len(), 1, "an off-site item was shown: {articles:?}");

        assert!(open_in_browser("https://evil.example/phish").is_err());
        assert!(open_in_browser("http://tinkeratlas.com/site-posts/x").is_err(), "plain http");
    }

    /// One bad entry must not blank the panel.
    #[test]
    fn an_item_missing_a_title_or_link_is_skipped_rather_than_fatal() {
        let broken = FEED.replace("<title>Second</title>", "");
        let articles = parse(&broken).expect("still valid RSS");
        assert_eq!(articles.len(), 1);
        assert_eq!(articles[0].title, "Your Sketch Went Red");
    }

    /// **A page that is not the feed is an error, not an empty list.**
    ///
    /// A captive portal or a 502 page answers with HTML, and reporting "no
    /// articles" for it would say the site had nothing to show when what
    /// actually happened is that we never reached it.
    #[test]
    fn a_page_that_is_not_the_feed_is_an_error() {
        assert!(parse("<html><body>not the feed</body>").is_err());
        assert!(parse("").is_err());
    }

    /// The cap is the cap, however long the feed is.
    #[test]
    fn no_more_than_the_column_holds_is_returned() {
        let mut many =
            String::from(r#"<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>"#);
        for index in 0..20 {
            many.push_str(&format!(
                "<item><title>Article {index}</title>\
                 <link>https://tinkeratlas.com/site-posts/{index}</link>\
                 <description>d</description><pubDate>Thu, 27 Aug 2026 00:00:00 GMT</pubDate></item>"
            ));
        }
        many.push_str("</channel></rss>");
        assert_eq!(parse(&many).expect("valid").len(), MAX_SHOWN);
    }
}
