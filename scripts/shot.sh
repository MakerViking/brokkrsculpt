#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Screenshot the running BrokkrSculpt window on a KDE Wayland session.
#
# `spectacle -a` captures the *active* window, and on this two screen desktop
# that is usually the terminal that launched the build -- which is how a
# screenshot of the terminal ends up standing in for a screenshot of the
# application. So activate the window first, through KWin's scripting
# interface, since there is no Wayland input injection here to click on it
# with.
#
# Usage: scripts/shot.sh [output.png]
set -euo pipefail

OUT="${1:-/tmp/brokkr-shot.png}"
CLASS="${BROKKR_WINDOW_CLASS:-brokkrsculpt}"

if ! pgrep -x brokkrsculpt >/dev/null; then
    echo "brokkrsculpt is not running" >&2
    exit 1
fi

SCRIPT=$(mktemp /tmp/kwin-activate-XXXXXX.js)
trap 'rm -f "$SCRIPT"' EXIT
cat >"$SCRIPT" <<EOF
const wanted = "${CLASS}";
for (const window of workspace.windowList()) {
    const klass = (window.resourceClass || "").toString().toLowerCase();
    const name = (window.resourceName || "").toString().toLowerCase();
    if (klass.includes(wanted) || name.includes(wanted)) {
        workspace.activeWindow = window;
        break;
    }
}
EOF

# The script name has to be unique per load: KWin refuses to load a second
# script under a name it already knows, and silently keeps the old one.
NAME="brokkr-activate-$$-$RANDOM"
ID=$(qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$SCRIPT" "$NAME")
qdbus6 org.kde.KWin "/Scripting/Script${ID}" org.kde.kwin.Script.run
qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "$NAME" >/dev/null

# The compositor needs a moment to raise and refocus before the grab.
sleep 1.5

# Ask the compositor a question and read the answer back out of the journal.
#
# A KWin script has no return channel -- `print` goes to the journal -- so each
# question carries a unique marker and finds its own answer by it. Reusing one
# marker would read a previous run's reply, which is the same class of mistake
# as reading a stale window position out of `journalctl`.
kwin_ask() {
    local body="$1" marker script name id
    marker="brokkr-ask-$$-$RANDOM"
    script=$(mktemp /tmp/kwin-ask-XXXXXX.js)
    printf 'const MARKER = "%s";\n%s\n' "$marker" "$body" >"$script"
    name="brokkr-ask-$$-$RANDOM"
    id=$(qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$script" "$name")
    qdbus6 org.kde.KWin "/Scripting/Script${id}" org.kde.kwin.Script.run
    qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "$name" >/dev/null
    rm -f "$script"
    journalctl --user -b --since "-20s" 2>/dev/null |
        grep -o "${marker} .*" | tail -1 | cut -d' ' -f2- | tr -d '\r'
}

active_class() {
    kwin_ask 'const a = workspace.activeWindow;
              print(MARKER + " " + (a ? a.resourceClass : "none"));'
}

# CHECK the activation worked, because activation is a request and not a
# guarantee.
#
# `spectacle -a` captures whatever is active at the moment it runs, so if the
# activation above lost a race -- a busy desktop, a video player taking focus
# back, a window that was still mapping -- the capture silently becomes a
# picture of somebody else's window. That is not a wasted screenshot: on
# 2026-08-19 it captured a media player mid-playback, and on 2026-08-23 a
# browser showing a private session, both straight into an agent's context.
# Whatever is on this desktop is nobody's business but its owner's.
ACTIVE=$(active_class)

if [[ -z "$ACTIVE" ]]; then
    echo "could not ask the compositor what is focused, so not taking a screenshot" >&2
    exit 1
fi
if [[ "${ACTIVE,,}" != *"${CLASS,,}"* ]]; then
    echo "refusing to capture: '${ACTIVE}' has focus, not '${CLASS}'" >&2
    echo "the window did not come forward -- raise it by hand and try again" >&2
    exit 1
fi

# What shape the window we are aiming at actually is, asked before the grab so
# the image can be checked against it afterwards.
GEOM=$(kwin_ask 'const wanted = "'"${CLASS}"'";
                 for (const w of workspace.windowList()) {
                     const k = (w.resourceClass || "").toString().toLowerCase();
                     const n = (w.resourceName || "").toString().toLowerCase();
                     if (k.includes(wanted) || n.includes(wanted)) {
                         print(MARKER + " " + Math.round(w.frameGeometry.width)
                               + "x" + Math.round(w.frameGeometry.height));
                         break;
                     }
                 }')

# -b background, -n no notification, -a active window. NOT -f: full screen
# captures the whole desktop, including whatever else is on the other monitor.
spectacle -b -n -a -o "$OUT" >/dev/null 2>&1
sleep 1.5

if [[ ! -s "$OUT" ]]; then
    echo "no screenshot was written to $OUT" >&2
    exit 1
fi

# --- and now check what was ACTUALLY captured --------------------------------
#
# Everything above happens BEFORE the grab, which is precisely the hole: focus
# can move in the moment between the check passing and spectacle running, and
# then a guard that "passed" hands back a picture of another window. That is
# not theoretical -- it is how the private-session capture above happened, on a
# run whose focus check had just succeeded.
#
# So the file is now guilty until proven innocent, and a capture that cannot be
# shown to be the right window is DELETED rather than left on disk for the next
# thing to read.
refuse() {
    rm -f "$OUT"
    echo "refusing the capture and deleting it: $1" >&2
    exit 1
}

AFTER=$(active_class)
if [[ "${AFTER,,}" != *"${CLASS,,}"* ]]; then
    refuse "focus moved to '${AFTER}' during the grab"
fi

SIZE=$(identify -format "%wx%h" "$OUT" 2>/dev/null || true)
if [[ -z "$SIZE" ]]; then
    refuse "could not measure $OUT"
fi

# Compare SHAPE, not size. The image is the window scaled by the output's
# scale factor and padded by the compositor's drop shadow, so its pixel size
# is not the window's -- but its aspect ratio survives both. A window that is
# 4:3 and an image that is 2.47:1 are not the same window, which is exactly
# the case that got through.
if [[ -n "$GEOM" ]]; then
    if ! awk -v g="$GEOM" -v s="$SIZE" '
        BEGIN {
            split(g, a, "x"); split(s, b, "x");
            if (a[1] <= 0 || a[2] <= 0 || b[1] <= 0 || b[2] <= 0) exit 1;
            want = a[1] / a[2]; got = b[1] / b[2];
            exit (got > want * 1.15 || got < want * 0.85) ? 1 : 0;
        }'; then
        refuse "the image is ${SIZE}, the wrong shape for a ${GEOM} window"
    fi
fi

echo "$SIZE"
