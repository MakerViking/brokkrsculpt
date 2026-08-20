#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
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

# Then CHECK it worked, because activation is a request and not a guarantee.
#
# `spectacle -a` captures whatever is active at the moment it runs, so if the
# activation above lost a race -- a busy desktop, a video player taking focus
# back, a window that was still mapping -- the capture silently becomes a
# picture of somebody else's window. That is not a wasted screenshot: on
# 2026-08-19 it captured a media player mid-playback and put it in an agent's
# context. Whatever is on this desktop is nobody's business but its owner's, so
# this asks the compositor what is actually focused and refuses rather than
# guessing.
MARKER="brokkr-active-$$-$RANDOM"
PROBE=$(mktemp /tmp/kwin-probe-XXXXXX.js)
cat >"$PROBE" <<EOF
const active = workspace.activeWindow;
print("${MARKER} " + (active ? active.resourceClass : "none"));
EOF
PROBE_NAME="brokkr-probe-$$-$RANDOM"
PROBE_ID=$(qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$PROBE" "$PROBE_NAME")
qdbus6 org.kde.KWin "/Scripting/Script${PROBE_ID}" org.kde.kwin.Script.run
qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript "$PROBE_NAME" >/dev/null
rm -f "$PROBE"

ACTIVE=$(journalctl --user -b --since "-20s" 2>/dev/null |
    grep -o "${MARKER} .*" | tail -1 | cut -d' ' -f2- | tr -d '\r')

if [[ -z "$ACTIVE" ]]; then
    echo "could not ask the compositor what is focused, so not taking a screenshot" >&2
    exit 1
fi
if [[ "${ACTIVE,,}" != *"${CLASS,,}"* ]]; then
    echo "refusing to capture: '${ACTIVE}' has focus, not '${CLASS}'" >&2
    echo "the window did not come forward -- raise it by hand and try again" >&2
    exit 1
fi

# -b background, -n no notification, -a active window. NOT -f: full screen
# captures the whole desktop, including whatever else is on the other monitor.
spectacle -b -n -a -o "$OUT" >/dev/null 2>&1
sleep 1.5

if [[ ! -s "$OUT" ]]; then
    echo "no screenshot was written to $OUT" >&2
    exit 1
fi

# Gate on the size. A capture of the terminal instead of the application is the
# failure this whole script exists to prevent, and it is invisible unless
# something checks -- so report the geometry and let the caller judge.
identify -format "%wx%h" "$OUT" 2>/dev/null || true
echo
