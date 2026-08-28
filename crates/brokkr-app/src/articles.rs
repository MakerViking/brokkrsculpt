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

use std::io::Read;
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
const MAX_SHOWN: usize = 4;

/// One article, reduced to what a link needs.
#[derive(Debug, Clone)]
pub struct Article {
    pub title: String,
    pub link: String,
    /// The opening of the article, cut to fit the card. See [`shorten`].
    ///
    /// **Shown again, and the card is a grid because of it.** An earlier pass
    /// dropped this on the grounds that four summaries stacked in a 340-pixel
    /// column read as a wall of text. That was true of the column; the answer
    /// was the layout rather than the words, and the page this mirrors has
    /// carried a summary on every card all along.
    pub summary: String,
    /// As the feed wrote it, trimmed to the date. Not parsed into a calendar
    /// type: nothing here does arithmetic on it, and a date format that fails
    /// to parse should still be shown rather than swallowed.
    pub date: String,
    /// The picture beside it, when there was one and it decoded.
    ///
    /// **An iced handle and not the pixels**, built once here rather than in
    /// the view. `Handle::from_rgba` stamps a fresh `Id::unique()` on every
    /// call, and the renderer's atlas is keyed on that id -- so building one
    /// per frame is a cache miss per frame, which uploads every thumbnail to
    /// the GPU again and drops last frame's. Four 240x135 images is 518 KB of
    /// memcpy and 518 KB of upload, sixty times a second, for pictures that
    /// never change. Cloning a handle is a refcount bump and keeps the id, so
    /// the atlas entry survives.
    ///
    /// `None` is an ordinary outcome and not an error: an article may have no
    /// image, the host may be slow, the bytes may be a format this does not
    /// decode. The row is drawn either way -- a missing picture must not cost
    /// the headline.
    pub thumbnail: Option<iced::widget::image::Handle>,
}

/// What the welcome screen has to draw.
#[derive(Debug, Clone, Default)]
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
    // **One agent for all five requests.** `ureq::get` is documented as running
    // on a use-once agent, so a bare call per request builds a fresh rustls
    // config, opens a fresh connection and pays a fresh TLS handshake -- and
    // all four thumbnails come from one host, so four of those five handshakes
    // would be pure waste.
    let agent: ureq::Agent =
        ureq::Agent::config_builder().timeout_global(Some(TIMEOUT)).build().into();

    let mut response = agent.get(FEED_URL).call().map_err(|why| match why {
        ureq::Error::StatusCode(code) => format!("tinkeratlas.com answered {code}"),
        // Everything else is the network: no route, DNS, TLS, timeout.
        // Worth saying differently from a rejection, because the answer is
        // "try again" rather than "something is wrong with the request".
        other => format!("could not reach tinkeratlas.com ({other})"),
    })?;
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|why| format!("the feed could not be read ({why})"))?;

    // Pictures after the list, so a slow image host costs the panel time and
    // never the headlines: a thumbnail that does not arrive leaves its article
    // drawn without one.
    Ok(parse_items(&body)?
        .into_iter()
        .map(|(mut article, picture)| {
            article.thumbnail = picture.as_deref().and_then(|url| fetch_thumbnail(&agent, url));
            article
        })
        .collect())
}

/// Test-only, and it wraps the real parser rather than duplicating it: what
/// production runs is [`parse_items`], and a test that exercised anything else
/// would be checking a second implementation nobody uses. CI builds with
/// `-D warnings`, so this cannot simply be left `pub` and unused.
#[cfg(test)]
pub fn parse(xml: &str) -> Result<Vec<Article>, String> {
    Ok(parse_items(xml)?.into_iter().map(|(article, _)| article).collect())
}

/// Pull the items out of an RSS document, each with the URL of its picture if
/// it has a usable one.
///
/// Deliberately tolerant: an item missing a title or a link is skipped rather
/// than failing the whole feed, because one malformed entry on the site should
/// not blank the panel. A document that is not XML at all IS an error -- that
/// is a captive portal or an error page, and saying "no articles" for it would
/// be a lie about what happened.
///
/// **It returns the picture URL rather than fetching it, so that parsing stays
/// pure.** Fetching thumbnails from inside the parser meant the unit tests only
/// avoided the network by accident -- their fixtures happen to carry no
/// `<enclosure>`, and the first one that did would have started making real
/// requests from `cargo test`.
fn parse_items(xml: &str) -> Result<Vec<(Article, Option<String>)>, String> {
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
        let picture = item
            .children()
            .find(|node| node.has_tag_name("enclosure"))
            .and_then(|node| node.attribute("url"))
            .and_then(thumbnail_url);
        articles.push((
            Article {
                title,
                link,
                summary: shorten(&child("description")),
                date: date_only(&child("pubDate")),
                thumbnail: None,
            },
            picture,
        ));
        if articles.len() == MAX_SHOWN {
            break;
        }
    }
    Ok(articles)
}

/// How wide a thumbnail is asked for, in pixels.
///
/// Sized for the grid card, which draws the picture at the card's full width
/// rather than beside the text -- roughly 320 logical pixels on the 960-wide
/// welcome card. The panel draws them at that width, so on a HiDPI output the
/// renderer is upscaling slightly; asking for double would quadruple the bytes
/// for a picture that sits behind a headline. See [`thumbnail_url`] for why the
/// size is asked of the server at all.
const THUMB_WIDTH: u32 = 320;
const THUMB_HEIGHT: u32 = 180;

/// Turn a stored-object URL into a resized one, or refuse it.
///
/// **The size is asked of the server rather than taken locally**, and it is the
/// difference between a panel that costs 17 KB and one that costs half a
/// megabyte: the article images are full size, 113 KB each, and Supabase's
/// render endpoint returns the same picture at 240 wide for 4 KB. Downloading
/// the big one to shrink it here would be four times the bytes and all of the
/// decode.
///
/// Only ever the storage host, over TLS. These bytes are fed to a decoder, so
/// where they come from is a security question and not a tidiness one.
fn thumbnail_url(enclosure: &str) -> Option<String> {
    const PREFIX: &str = "https://api.tinkeratlas.com/storage/v1/object/public/";
    const RENDER: &str = "https://api.tinkeratlas.com/storage/v1/render/image/public/";
    let rest = enclosure.strip_prefix(PREFIX)?;
    // A query of its own would be appended to ours and could override the size.
    if rest.contains('?') {
        return None;
    }
    Some(format!(
        "{RENDER}{rest}?width={THUMB_WIDTH}&height={THUMB_HEIGHT}&resize=cover&quality=70"
    ))
}

/// Fetch and decode one thumbnail. `None` on any failure, which is ordinary.
///
/// A picture that will not arrive or will not decode must not cost the article
/// its headline, so every failure here is a `None` and never an error that
/// propagates. The one thing it will not do is decode something enormous: the
/// server was asked for 240 pixels, and anything far larger than that is not
/// the picture that was requested.
fn fetch_thumbnail(agent: &ureq::Agent, url: &str) -> Option<iced::widget::image::Handle> {
    let mut response = agent.get(url).call().ok()?;
    let mut bytes = Vec::new();
    response.body_mut().as_reader().take(MAX_THUMB_BYTES).read_to_end(&mut bytes).ok()?;

    let decoded = image::load_from_memory(&bytes).ok()?;
    let (width, height) = (decoded.width(), decoded.height());
    if width == 0 || height == 0 || width > THUMB_WIDTH * 4 || height > THUMB_HEIGHT * 4 {
        return None;
    }
    Some(iced::widget::image::Handle::from_rgba(width, height, decoded.into_rgba8().into_raw()))
}

/// A ceiling on what will be read for one thumbnail.
///
/// The request asks for a 240-pixel picture and the answer should be a few
/// kilobytes. This is not a tuning constant, it is a refusal to let a
/// misbehaving or hostile host stream unbounded bytes into memory while a modal
/// waits on it.
const MAX_THUMB_BYTES: u64 = 512 * 1024;

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

/// Cut a summary to two lines on a card, without breaking a word.
///
/// A character count and not a pixel measurement: the text is laid out by the
/// renderer at a size and font this module knows nothing about, and a count
/// that is roughly right everywhere beats a measurement that is exactly right
/// on one machine. `MAX_CHARS` is two lines of the grid card at
/// `TEXT_SIZE_SMALL`, with the ellipsis counted.
///
/// See `panel.rs`'s own `shorten`, which does the same job for paths and
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

/// Where the two buttons on the welcome screen lead.
///
/// Here rather than in `panel.rs` so they sit beside the rule that admits them:
/// a URL and the policy deciding whether it may be opened are one thing, and
/// splitting them is how one quietly stops satisfying the other.
pub const JOIN_URL: &str = "https://tinkeratlas.com/signup";
pub const VISIT_URL: &str = "https://tinkeratlas.com/";

/// Whether a link may be handed to the user's browser.
///
/// **Split out of [`open_in_browser`] so acceptance can be tested.** That one
/// spawns a process the moment it approves, so a test proving a URL is allowed
/// would open a browser on whoever ran the suite -- which is why the existing
/// test only ever checks refusals. This is the same question with no side
/// effect.
pub(crate) fn leads_to_tinkeratlas(link: &str) -> bool {
    link.starts_with("https://tinkeratlas.com/")
}

/// Hand a link to whatever the desktop opens links with.
///
/// `xdg-open` rather than a crate: this is Linux-first, `slicer.rs` already
/// spawns a process the same way, and a browser-opening dependency would be a
/// second one to audit for the sake of one command.
pub fn open_in_browser(link: &str) -> Result<(), String> {
    if !leads_to_tinkeratlas(link) {
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

    /// **The welcome screen's two buttons must survive their own guard.**
    ///
    /// They are constants, so a typo in one is not a compile error and not a
    /// panic -- the button simply reports "that link does not lead to
    /// TinkerAtlas" when pressed, which nobody would find until a user did.
    /// A wrong scheme is the likely slip, and it is exactly what the guard
    /// rejects.
    #[test]
    fn the_welcome_buttons_lead_somewhere_the_guard_allows() {
        assert!(leads_to_tinkeratlas(JOIN_URL), "the Join button would be refused: {JOIN_URL}");
        assert!(leads_to_tinkeratlas(VISIT_URL), "the Visit button would be refused: {VISIT_URL}");
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
