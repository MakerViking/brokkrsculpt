#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Sign the update manifest for a build that has already been published.
#
# This runs on Thomas's machine and nowhere else. The minisign secret key never
# enters CI, because a key that authorises pushing code to every tester's
# machine has no business in a runner -- runner-memory credential theft is a
# live attack, not a theory.
#
# But keeping the key local protects the KEY, not the ARTEFACT. If the digests
# in the manifest came back from `gh release view --json assets`, the maintainer
# would never hold the bytes, and a compromised runner or token could upload a
# backdoored payload that this script then faithfully signed. So every payload
# is DOWNLOADED and hashed here, locally, with sha256sum. That is the whole
# reason this file exists rather than four lines in release.yml.
#
# Usage:
#   scripts/sign-release.sh <build> --requires-reinstall <0|1> [options]
#
#   <build>                     the build ordinal to publish, e.g. 1005
#   --requires-reinstall 0|1    REQUIRED, see below
#   --minimum-build N           warn every client below N (default 0, no floor)
#   --key-epoch N               sign with ~/.minisign/brokkrsculpt-N.key
#                               and declare epoch N (default 0, the live key)
#   --no-attestation            skip `gh attestation verify` -- see below
#
# `--requires-reinstall` has no default on purpose. It is the manifest field
# that says "this build cannot be reached by replacing one file", and forgetting
# to set it is the entire failure mode: the client would copy one executable
# over another and produce a half-updated install that nothing detects. The
# script cannot work the answer out for itself -- the human-facing archives
# carry fixed names and are re-uploaded with --clobber on every push, so the
# previous release's companion files are already gone by the time we get here.
# A conscious keystroke is the guard that is actually available, and this is the
# one place in the pipeline where a human is already in the loop.
#
# `BROKKR_RELEASE_REPO` overrides the repository, for the scratch-repo
# rehearsal Phase 1 owes: the publish half of this design is otherwise first
# exercised on the day the first signed release ships.
set -euo pipefail

REPO="${BROKKR_RELEASE_REPO:-MakerViking/brokkrsculpt}"
TAG=beta

# The client rejects any seq above this (a compiled-in constant, so that it
# bounds a poisoned floor even on a fresh install). Signing one above it would
# publish a manifest that every copy in the field refuses, silently.
MAX_SEQ=1000000

die() { echo "$*" >&2; exit 1; }
is_number() { [[ "$1" =~ ^[0-9]+$ ]]; }

usage() {
    cat <<'USAGE'
scripts/sign-release.sh <build> --requires-reinstall <0|1> [options]

  <build>                     the build ordinal to publish, e.g. 1005
  --requires-reinstall 0|1    required; no default, deliberately
  --minimum-build N           warn every client below N (default 0, no floor)
  --key-epoch N               sign with ~/.minisign/brokkrsculpt-N.key
                              and declare epoch N (default 0, the live key)
  --no-attestation            skip `gh attestation verify`
USAGE
}

# --- arguments ---------------------------------------------------------------

BUILD=""
REQUIRES_REINSTALL=""
MINIMUM_BUILD=0
KEY_EPOCH=0
ATTEST=1

while [ $# -gt 0 ]; do
    case "$1" in
        --requires-reinstall) REQUIRES_REINSTALL="${2:-}"; shift 2 ;;
        --minimum-build)      MINIMUM_BUILD="${2:-}"; shift 2 ;;
        --key-epoch)          KEY_EPOCH="${2:-}"; shift 2 ;;
        --no-attestation)     ATTEST=0; shift ;;
        -h|--help)            usage; exit 0 ;;
        -*)                   usage >&2; die "unknown option: $1" ;;
        *)
            [ -z "$BUILD" ] || die "expected exactly one build ordinal"
            BUILD="$1"; shift ;;
    esac
done

[ -n "$BUILD" ] || die "no build ordinal given: scripts/sign-release.sh <build> --requires-reinstall <0|1>"
is_number "$BUILD" || die "build ordinal must be a number, got '$BUILD'"

if [ -z "$REQUIRES_REINSTALL" ]; then
    die "refusing to sign without --requires-reinstall 0|1.
It has no default because forgetting to set it is the failure mode: a build
that needs a full reinstall, published with the flag clear, produces a
half-updated install on every machine that takes it. Pass 0 for an ordinary
release -- one executable is a complete update today."
fi
case "$REQUIRES_REINSTALL" in
    0|1) ;;
    *) die "--requires-reinstall must be 0 or 1, got '$REQUIRES_REINSTALL'" ;;
esac
is_number "$MINIMUM_BUILD" || die "--minimum-build must be a number, got '$MINIMUM_BUILD'"
is_number "$KEY_EPOCH" || die "--key-epoch must be a number, got '$KEY_EPOCH'"

# The key file name carries its own epoch, so the two cannot disagree. A
# manifest signed by one key and declaring another is refused by the client at
# step 2d -- not as an attack, but because it would otherwise persist a
# floor_epoch that no key can ever satisfy again.
SECRET_KEY="$HOME/.minisign/brokkrsculpt-${KEY_EPOCH}.key"
PUBLIC_KEY="$HOME/.minisign/brokkrsculpt-${KEY_EPOCH}.pub"

# --- preflight ---------------------------------------------------------------

for tool in gh minisign sha256sum; do
    command -v "$tool" >/dev/null || die "$tool is not installed"
done

# The DISTRIBUTION minisign, not `cargo install rsign2`. `cargo install` in the
# presence of the signing key means building an external crate tree, with
# arbitrary build-script execution, next to the secret material.
[ -f "$SECRET_KEY" ] || die "no secret key at $SECRET_KEY"

# A signing script that cannot ask a question must not sign.
[ -t 0 ] || die "not a terminal: this script requires a typed confirmation"

work=$(mktemp -d /tmp/brokkr-sign-XXXXXX)
trap 'rm -rf "$work"' EXIT

# --- the seq, read out of the live manifest ----------------------------------
#
# `seq` counts MANIFESTS. `build` counts CI runs. They are different number
# spaces and are deliberately allowed to diverge: a rollback publishes seq+1
# naming an older build, which is legal and is mechanism 2 for stopping a bad
# build. The client refuses a lower seq and accepts a lower build.

live_seq=0
if gh release download "$TAG" --repo "$REPO" --pattern latest.conf \
        --dir "$work/live" >/dev/null 2>&1; then
    # Only a numeric seq is accepted. A latest.conf that exists but does not
    # parse must NOT fall back to "there is no live manifest, start at 1" --
    # that would publish a seq below the floor every installed copy has already
    # persisted, and every one of them would refuse it for ever, silently.
    live_seq=$(sed -n 's/^[[:space:]]*seq[[:space:]]*=[[:space:]]*\([0-9][0-9]*\)[[:space:]]*$/\1/p' \
        "$work/live/latest.conf" | head -1)
    [ -n "$live_seq" ] || die "the live latest.conf has no readable 'seq' line; refusing to guess one"
    echo "live manifest: seq = $live_seq"
else
    echo "no live latest.conf on $REPO@$TAG -- treating this as the first signed release"
fi

new_seq=$((live_seq + 1))
[ "$new_seq" -gt "$live_seq" ] || die "new seq $new_seq would not be greater than the live $live_seq"
[ "$new_seq" -le "$MAX_SEQ" ] || die "new seq $new_seq exceeds MAX_SEQ ($MAX_SEQ); every client would refuse it"

# --- payloads: download, attest, hash locally --------------------------------

manifest="$work/latest.conf"
{
    printf 'seq = %s\n' "$new_seq"
    printf 'build = %s\n' "$BUILD"
    printf 'key_epoch = %s\n' "$KEY_EPOCH"
    printf 'minimum_build = %s\n' "$MINIMUM_BUILD"
    printf 'requires_reinstall = %s\n' "$REQUIRES_REINSTALL"
} > "$manifest"

found=0
for slug in linux-x86_64 windows-x86_64 macos-arm64; do
    case "$slug" in
        linux-x86_64)   name="brokkrsculpt-${BUILD}-linux-x86_64" ;;
        windows-x86_64) name="brokkrsculpt-${BUILD}-windows-x86_64.exe" ;;
        macos-arm64)    name="brokkrsculpt-${BUILD}-macos-arm64.zip" ;;
    esac

    # An absent payload means "no update for this platform", not an error: the
    # release matrix has fail-fast off precisely so a two-platform release is
    # publishable while the third is fixed. The client reads a missing block
    # the same way.
    if ! gh release download "$TAG" --repo "$REPO" --pattern "$name" \
            --dir "$work" >/dev/null 2>&1; then
        echo "no $name in the release -- omitting $slug"
        MISSING="${MISSING:-} $slug"
        continue
    fi

    file="$work/$name"

    # Provenance, checked before the bytes are blessed. This is the independent
    # trust root a locally-held signing key cannot provide: it says the payload
    # came out of our workflow in our repository, not merely that someone with
    # push access uploaded it.
    #
    # `actions/attest-build-provenance` landed on release.yml's publish job on
    # 2026-08-30, with `id-token: write` and `attestations: write`, attesting
    # `dist/payload/*` BEFORE the upload. So this check is live for anything
    # published from that run onwards, and `--no-attestation` is needed only for
    # payloads built before it -- which includes every build currently on the
    # release. Keep the flag explicit rather than falling back silently: a
    # verification step that quietly succeeds when there is nothing to verify is
    # worse than no step at all.
    #
    # The commit SHA is deliberately not pinned. During a rollback this script
    # signs a manifest naming an OLDER build, whose payload was built from a
    # commit that is not the release's current target.
    if [ "$ATTEST" -eq 1 ]; then
        echo "verifying provenance of $name"
        gh attestation verify "$file" --repo "$REPO" \
            --signer-workflow "$REPO/.github/workflows/release.yml" \
            || die "provenance check failed for $name -- refusing to sign it"
    fi

    size=$(stat -c %s "$file")
    digest=$(sha256sum "$file" | cut -d' ' -f1)

    {
        printf '%s.name = %s\n' "$slug" "$name"
        printf '%s.size = %s\n' "$slug" "$size"
        printf '%s.sha256 = %s\n' "$slug" "$digest"
    } >> "$manifest"

    found=$((found + 1))
done

[ "$found" -gt 0 ] || die "no payloads for build $BUILD are attached to $REPO@$TAG; there is nothing to sign"

# **A missing platform is legitimate; a missing platform going UNSAID is not.**
# A `continue` above treats an absent payload exactly as the client treats an
# absent block -- "no update for this platform" -- which is right when a runner
# genuinely failed and is silent in the case that matters: someone renames one
# platform's artefact in release.yml, this signs a correct manifest without it,
# and every client on that platform reads an ordinary no-op for ever. Nothing on
# either end says a word.
#
# There is no way to tell those two apart from here, so this does not refuse.
# It makes the operator say it out loud, which is the same reason
# --requires-reinstall has no default.
if [ "$found" -lt 3 ]; then
    echo
    echo "WARNING: only $found of 3 platforms have a payload for build $BUILD."
    echo "Missing:${MISSING:- none}"
    echo "That is correct if a runner failed. If a payload was RENAMED, this"
    echo "will publish a manifest that silently never updates that platform."
    printf 'type "yes" if that is expected: '
    read -r sure
    [ "$sure" = "yes" ] || die "aborted; nothing was signed"
fi

# --- show it, then ask -------------------------------------------------------

echo
echo "about to sign and publish this manifest to $REPO@$TAG:"
echo
sed 's/^/    /' "$manifest"
echo
echo "  key:                $SECRET_KEY"
echo "  seq:                $live_seq -> $new_seq"
echo "  platforms:          $found of 3"
[ "$ATTEST" -eq 1 ] || echo "  provenance:         NOT CHECKED (--no-attestation)"
[ "$REQUIRES_REINSTALL" -eq 0 ] || echo "  requires_reinstall: 1 -- clients will NOT self-update to this build"
[ "$MINIMUM_BUILD" -eq 0 ] || echo "  minimum_build:      $MINIMUM_BUILD -- every client below this is warned"
echo
printf 'type "sign" to sign and upload, anything else to abort: '
read -r reply
[ "$reply" = "sign" ] || die "aborted; nothing was uploaded"

# --- sign --------------------------------------------------------------------
#
# `-H` is prehashed, and it is passed explicitly rather than relied on. The
# client passes allow_legacy = false and refuses a legacy `Ed` signature, and
# getting that wrong once means every release is rejected on every install --
# reported by the client as a verification failure it deliberately treats as
# transient, so it would look like a flaky network for as long as it lasted.
#
# Measured on this machine, 2026-08-30: minisign 0.12's default is ALREADY
# prehashed, so `-H` changes nothing here today. The plan says the tool default
# is legacy; that is true of `cargo install rsign2` and of older minisign, not
# of the distribution binary at this version. Passing it explicitly is what
# makes the output independent of which of those is on PATH -- which is the
# reason to keep both the flag and the readback below.
minisign -S -H -s "$SECRET_KEY" -m "$manifest" -x "$manifest.minisig"

# Read the algorithm back out of the signature we just wrote rather than
# trusting the flag. The two bytes at the head of the base64 payload are the
# algorithm: "ED" is prehashed, "Ed" is the legacy mode the client refuses.
# Verified against both variants produced by minisign 0.12.
alg=$(sed -n 2p "$manifest.minisig" | base64 -d 2>/dev/null | head -c 2 || true)
[ "$alg" = "ED" ] || die "the signature is '$alg', not the prehashed 'ED' -- every client would refuse it"

# And verify it here, with the public half, before it goes anywhere. A release
# night is the wrong time to discover that the key pair does not match.
if [ -f "$PUBLIC_KEY" ]; then
    minisign -V -p "$PUBLIC_KEY" -m "$manifest" -x "$manifest.minisig" \
        || die "the signature does not verify against $PUBLIC_KEY"
else
    echo "no $PUBLIC_KEY, so the signature was not verified locally" >&2
fi

# --- upload ------------------------------------------------------------------
#
# Both assets in one call, and the order genuinely does not matter: they are
# replaced separately, so no publish order removes the window in which a client
# fetches one new and one old. That is why a failed verification is a transient
# outcome on the client and not an alarm -- it writes no floor, says nothing,
# and retries at the next check.
gh release upload "$TAG" "$manifest" "$manifest.minisig" --repo "$REPO" --clobber

echo "published seq $new_seq for build $BUILD"
