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
