// SPDX-License-Identifier: AGPL-3.0-only

//! Noticing that a newer beta exists.
//!
//! **This is a check, not an updater.** It downloads nothing and executes
//! nothing; it compares two short strings and, if they differ, puts one line
//! in front of the user with a link. The real thing -- verified download,
//! replacing a running binary on three platforms, rollback -- is a milestone
//! of its own and is planned separately. Shipping the check first is not a
//! placeholder for that work: it is the half that carries no risk and removes
//! most of the pain, because the rolling `beta` tag means a user who
//! downloaded last week has no way of knowing they are behind.
//!
//! # Why a commit and not a version
//!
//! `CARGO_PKG_VERSION` does not move. Every push to `main` republishes the
//! same `beta` tag with the same version number, so "is there something newer"
//! cannot be answered by comparing versions -- SindriCAD sidesteps this by
//! stamping `0.1.<run number>` into each build, which its updater then
//! compares as semver. This does the equivalent with what is already stamped:
//! `build_commit`, which `build.rs` takes from git.
//!
//! Different, not newer. Comparing commits tells you the release was built
//! from something else, and cannot tell you which came first -- so the wording
//! the user sees says exactly that and nothing stronger.
//!
//! # What it will not do
//!
//! A build from a dirty tree, or one with no commit stamped at all, is not
//! behind anything: it is somebody's working copy, and telling them to
//! download the beta over it would be wrong. Both are skipped before any
//! request is made, which also means a developer's machine never makes this
//! call at all.

use std::time::Duration;

/// Where the current beta is published.
///
/// The API rather than the HTML page: the page is a moving target and parsing
/// it would break on a redesign, while this field is part of a documented
/// contract. Unauthenticated, which GitHub rate limits to sixty an hour per
/// address -- one call per launch is nowhere near it.
const RELEASE_API: &str =
    "https://api.github.com/repos/MakerViking/brokkrsculpt/releases/tags/beta";

/// Where a user is sent to get it.
pub const RELEASE_PAGE: &str = "https://github.com/MakerViking/brokkrsculpt/releases/tag/beta";

/// Short enough that a slow or captive network cannot delay anything the user
/// is doing. Nothing waits on this -- it resolves into a message whenever it
/// resolves, and a failure is silence.
const TIMEOUT: Duration = Duration::from_secs(10);

/// A newer beta than the one running, if there is one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Newer {
    /// The commit the published beta was built from, shortened to match the
    /// form `build_commit` reports.
    pub commit: String,
}

/// Ask whether the published beta was built from a different commit.
///
/// `None` for every uninteresting answer, and they are deliberately not
/// distinguished: no network, a rate limit, a malformed reply, a development
/// build, or genuinely being up to date all mean "say nothing". An update
/// check that reports its own failures is a check that interrupts people to
/// tell them about GitHub.
pub fn check(running: &str) -> Option<Newer> {
    // A working copy is not behind the beta, and neither is a build that was
    // made outside a checkout. Skipped before the request, so a development
    // machine never makes this call.
    if running == "unknown" || running.ends_with("-dirty") {
        return None;
    }

    let agent = ureq::Agent::config_builder().timeout_global(Some(TIMEOUT)).build().new_agent();

    // GitHub refuses a request with no user agent. Naming the application is
    // also the honest thing: this call tells GitHub an address is running
    // BrokkrSculpt, and pretending to be a browser would hide that from
    // anyone reading a proxy log on their own network.
    let body = agent
        .get(RELEASE_API)
        .header("User-Agent", "BrokkrSculpt")
        .header("Accept", "application/vnd.github+json")
        .call()
        .ok()?
        .body_mut()
        .read_to_string()
        .ok()?;

    let published = target_commitish(&body)?;
    (!published.starts_with(running) && !running.starts_with(&published))
        .then_some(Newer { commit: published })
}

/// The `target_commitish` field, shortened.
///
/// Scanned rather than deserialised: one field out of a reply with about sixty
/// of them, and a `serde` derive for it would be a struct that has to be kept
/// in step with an API nobody here controls. The field holds a full forty
/// character SHA; `build_commit` reports git's short form, so this is cut to
/// match rather than the comparison being made clever.
fn target_commitish(body: &str) -> Option<String> {
    let at = body.find("\"target_commitish\"")?;
    let rest = &body[at..];
    let open = rest.find(':')?;
    let quote = rest[open..].find('"')? + open + 1;
    let close = rest[quote..].find('"')? + quote;
    let value = rest[quote..close].trim();
    // A branch name is a legitimate value for this field and is not a commit.
    // Refusing anything that is not hex means a release published from a
    // branch reads as "nothing to say" rather than as a permanent nag.
    if value.len() < 7 || !value.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(value[..7].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_development_build_is_never_behind() {
        // Both skip before any request, so these do not touch the network.
        assert_eq!(check("unknown"), None);
        assert_eq!(check("abc1234-dirty"), None);
    }

    #[test]
    fn the_commit_is_read_out_of_the_shape_the_api_really_sends() {
        let body = r#"{"url":"https://api.github.com/x","assets_url":"y",
            "tag_name":"beta","target_commitish":"85ac0e1bfa91d9bcf0993f4e71617c7aa25b1899",
            "name":"BrokkrSculpt open beta","draft":false,"prerelease":true}"#;
        assert_eq!(target_commitish(body).as_deref(), Some("85ac0e1"));
    }

    /// `target_commitish` holds a branch name when a release was published
    /// from one, and a branch is not a commit to compare against.
    #[test]
    fn a_branch_name_is_not_mistaken_for_a_commit() {
        assert_eq!(target_commitish(r#"{"target_commitish":"main"}"#), None);
        assert_eq!(target_commitish(r#"{"target_commitish":""}"#), None);
    }

    #[test]
    fn a_reply_without_the_field_says_nothing() {
        assert_eq!(target_commitish(r#"{"message":"Not Found"}"#), None);
        assert_eq!(target_commitish("not json at all"), None);
    }
}
