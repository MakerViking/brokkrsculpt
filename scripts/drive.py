#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Drive the running BrokkrSculpt window with a virtual input device.

There is no ydotool, wtype or dotool on this machine and XTEST silently does
nothing under KWin Wayland, which is why `scripts/shot.sh` says there is no
input injection here. That was true of the routes tried; it is not true of
`/dev/uinput`, which is writable by the `input` group and creates a device the
kernel and the compositor cannot tell from real hardware.

# Two things that are not obvious and cost a session each

**Absolute, never relative.** libinput accelerates relative motion, so a delta
of 500 does not travel 500 pixels and nothing ever lands where it was aimed.
The ABS range is set to the logical desktop union, read from `kscreen-doctor`,
so a coordinate here is a coordinate on screen.

**Coordinates are WINDOW-LOCAL.** The window moves between sessions and the
desktop is two outputs at an offset, so a desktop coordinate written down today
aims at nothing tomorrow. The window's position comes from KWin at run time and
local coordinates are converted against it. Local (0, 0) is the top-left of the
window, in the same logical pixels the application lays out in -- so a position
read off a `shot.sh` capture converts by dividing by the capture's scale
(1.5 on this desktop: a 1024x768 window captures at 1536x1152).

This device deliberately does NOT advertise `ABS_PRESSURE` or `BTN_TOOL_PEN`
(how the application recognises a tablet) nor all six of `REL_X`..`REL_RZ` (how
it recognises a SpaceMouse), so the application under test will not adopt it as
either and the pen and puck paths stay untouched.

Usage
-----
    scripts/drive.py <script-file>          # one command per line
    scripts/drive.py --geometry             # print the window rectangle

Commands, all coordinates window-local:

    move X Y                    click [left|right|middle]
    down [button]               up [button]
    drag X1 Y1 X2 Y2 [steps]    mdrag X1 Y1 X2 Y2 [steps]   # middle: orbits
    key NAME                    keydown NAME / keyup NAME
    scroll N                    sleep MS

`#` starts a comment. Key names are in KEYS below; `ctrl+z` is
`keydown ctrl`, `key z`, `keyup ctrl`.
"""
import fcntl
import os
import re
import struct
import subprocess
import sys
import time

UINPUT = "/dev/uinput"
# Which window to drive. Overridable the way `shot.sh` already takes
# `BROKKR_WINDOW_CLASS`, because the application is not the only window a test
# has to reach: an export puts up a native save dialog, which is its own window
# with its own class (`org.freedesktop.impl.portal.desktop.kde` here), and
# without this the driver activates the application behind the dialog and every
# keystroke lands in the wrong place. Geometry is read for whichever window
# this names, so coordinates stay window-local either way.
WINDOW_CLASS = os.environ.get("BROKKR_WINDOW_CLASS", "brokkrsculpt")

EV_SYN, EV_KEY, EV_REL, EV_ABS = 0, 1, 2, 3
SYN_REPORT = 0
ABS_X, ABS_Y = 0, 1
REL_WHEEL = 8
BTN = {"left": 0x110, "right": 0x111, "middle": 0x112}
KEYS = {
    "esc": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8,
    "q": 16, "w": 17, "e": 18, "r": 19, "t": 20, "y": 21, "u": 22,
    "s": 31, "f": 33, "z": 44, "x": 45, "c": 46, "v": 47, "m": 50,
    "ctrl": 29, "shift": 42, "alt": 56, "space": 57, "tab": 15, "enter": 28,
}


def _ioc(direction, typ, nr, size):
    return (direction << 30) | (size << 16) | (typ << 8) | nr


UI_SET_EVBIT = _ioc(1, 0x55, 100, 4)
UI_SET_KEYBIT = _ioc(1, 0x55, 101, 4)
UI_SET_RELBIT = _ioc(1, 0x55, 102, 4)
UI_SET_ABSBIT = _ioc(1, 0x55, 103, 4)
UI_DEV_CREATE = _ioc(0, 0x55, 1, 0)
UI_DEV_DESTROY = _ioc(0, 0x55, 2, 0)


def desktop_union():
    """The logical rectangle every output sits inside, from `kscreen-doctor`.

    The ABS range maps onto this, so it has to cover both monitors or the far
    one is unreachable.
    """
    try:
        out = subprocess.run(
            ["kscreen-doctor", "-o"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return 3840, 2160
    width = height = 0
    # Geometry lines look like "Geometry:  786,0 1536x864", with colour codes.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    for x, y, w, h in re.findall(r"Geometry:\s*(\d+),(\d+)\s+(\d+)x(\d+)", plain):
        width = max(width, int(x) + int(w))
        height = max(height, int(y) + int(h))
    return (width or 3840), (height or 2160)


def kwin_ask(body):
    """Ask KWin a question and read the answer out of the journal.

    A KWin script has no return channel -- `print` goes to the journal -- so
    each question carries a unique marker and finds its own answer by it.
    Reusing one marker reads a previous run's reply. Same device as
    `scripts/shot.sh`, which explains it at greater length.
    """
    marker = f"drive-{os.getpid()}-{time.time_ns() % 100000}"
    path = f"/tmp/kwin-drive-{marker}.js"
    with open(path, "w") as handle:
        handle.write(f'const MARKER = "{marker}";\n{body}\n')
    name = f"drive-{marker}"
    try:
        ident = subprocess.run(
            ["qdbus6", "org.kde.KWin", "/Scripting",
             "org.kde.kwin.Scripting.loadScript", path, name],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        subprocess.run(["qdbus6", "org.kde.KWin", f"/Scripting/Script{ident}",
                        "org.kde.kwin.Script.run"], capture_output=True, timeout=15)
        subprocess.run(["qdbus6", "org.kde.KWin", "/Scripting",
                        "org.kde.kwin.Scripting.unloadScript", name],
                       capture_output=True, timeout=15)
        time.sleep(0.3)
        log = subprocess.run(
            ["journalctl", "--user", "-b", "--since", "-20s"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        hits = [line for line in log.splitlines() if marker in line]
        if not hits:
            return ""
        return hits[-1].split(marker, 1)[1].strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def window_rect():
    """Where the application's window is, in logical desktop pixels."""
    answer = kwin_ask(
        'for (const w of workspace.windowList()) {'
        f'  const k = (w.resourceClass || "").toString().toLowerCase();'
        f'  if (k.includes("{WINDOW_CLASS}")) {{'
        '    print(MARKER + " " + Math.round(w.frameGeometry.x) + " "'
        '          + Math.round(w.frameGeometry.y) + " "'
        '          + Math.round(w.frameGeometry.width) + " "'
        '          + Math.round(w.frameGeometry.height)); break; }'
        '}'
    )
    parts = answer.split()
    if len(parts) != 4:
        raise SystemExit(
            f"could not find a {WINDOW_CLASS} window through KWin (got {answer!r}) -- "
            "is the application running?"
        )
    return tuple(int(p) for p in parts)


def activate():
    """Bring the window forward, and CHECK that it came.

    Activation is a request, not a guarantee. Every key below goes to whatever
    is focused, so a failed activation types into the user's terminal instead --
    which is how a screenshot once showed a brush changed and a mirror plane on
    that no code had touched.
    """
    active = ""
    # Asked up to three times: raising a window is a request to the compositor
    # and it loses to a window that is still mapping, or to one that takes focus
    # back. One attempt reads as "the application is not there".
    for _ in range(3):
        kwin_ask(
            'for (const w of workspace.windowList()) {'
            f'  const k = (w.resourceClass || "").toString().toLowerCase();'
            f'  if (k.includes("{WINDOW_CLASS}")) {{ workspace.activeWindow = w;'
            '    print(MARKER + " ok"); break; }}'
            '}'
        )
        time.sleep(0.8)
        active = kwin_ask('const a = workspace.activeWindow;'
                          'print(MARKER + " " + (a ? a.resourceClass : "none"));')
        if WINDOW_CLASS in active.lower():
            return
    raise SystemExit(
        f"refusing to drive: '{active}' has focus, not {WINDOW_CLASS}. "
        "Every keystroke would land in it."
    )


class Device:
    def __init__(self, origin):
        self.ox, self.oy = origin
        self.width, self.height = desktop_union()
        self.fd = os.open(UINPUT, os.O_WRONLY | os.O_NONBLOCK)
        for ev in (EV_KEY, EV_ABS, EV_REL, EV_SYN):
            fcntl.ioctl(self.fd, UI_SET_EVBIT, ev)
        for code in list(BTN.values()) + list(KEYS.values()):
            fcntl.ioctl(self.fd, UI_SET_KEYBIT, code)
        for axis in (ABS_X, ABS_Y):
            fcntl.ioctl(self.fd, UI_SET_ABSBIT, axis)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_WHEEL)

        absmax = [0] * 64
        absmax[ABS_X], absmax[ABS_Y] = self.width - 1, self.height - 1
        os.write(self.fd, struct.pack(
            "80sHHHHi" + "64i" * 4, b"brokkr-drive", 0x03, 0x1234, 0x5678, 1, 0,
            *absmax, *([0] * 64), *([0] * 64), *([0] * 64),
        ))
        fcntl.ioctl(self.fd, UI_DEV_CREATE)
        # The compositor has to notice the device before it will route to it.
        time.sleep(0.5)

    def _emit(self, typ, code, value):
        os.write(self.fd, struct.pack("llHHi", 0, 0, typ, code, value))

    def _syn(self):
        self._emit(EV_SYN, SYN_REPORT, 0)

    def move(self, lx, ly):
        x = max(0, min(self.width - 1, int(round(self.ox + lx))))
        y = max(0, min(self.height - 1, int(round(self.oy + ly))))
        self._emit(EV_ABS, ABS_X, x)
        self._emit(EV_ABS, ABS_Y, y)
        self._syn()
        time.sleep(0.012)

    def button(self, name, down):
        self._emit(EV_KEY, BTN[name], 1 if down else 0)
        self._syn()
        time.sleep(0.05)

    def key(self, name, down):
        if name not in KEYS:
            raise SystemExit(f"no key code for {name!r}; add it to KEYS")
        self._emit(EV_KEY, KEYS[name], 1 if down else 0)
        self._syn()
        time.sleep(0.04)

    def scroll(self, notches):
        step = 1 if notches > 0 else -1
        for _ in range(abs(notches)):
            self._emit(EV_REL, REL_WHEEL, step)
            self._syn()
            time.sleep(0.06)

    def drag(self, x1, y1, x2, y2, steps=24, button="left"):
        # Many small steps and never one jump: an application tells a drag from
        # a click by travel, and a single leap reads as neither.
        self.move(x1, y1)
        time.sleep(0.15)
        self.button(button, True)
        time.sleep(0.12)
        for i in range(1, steps + 1):
            t = i / steps
            self.move(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
        time.sleep(0.15)
        self.button(button, False)
        time.sleep(0.25)

    def close(self):
        time.sleep(0.25)
        fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        os.close(self.fd)


def run(path):
    activate()
    x, y, w, h = window_rect()
    print(f"window at {x},{y} size {w}x{h}", flush=True)
    dev = Device((x, y))
    try:
        for raw in open(path).read().splitlines():
            line = raw.split("#")[0].strip()
            if not line:
                continue
            cmd, *args = line.split()
            if cmd == "move":
                dev.move(float(args[0]), float(args[1]))
            elif cmd == "down":
                dev.button(args[0] if args else "left", True)
            elif cmd == "up":
                dev.button(args[0] if args else "left", False)
            elif cmd == "click":
                name = args[0] if args else "left"
                dev.button(name, True)
                dev.button(name, False)
            elif cmd in ("drag", "mdrag"):
                steps = int(args[4]) if len(args) > 4 else 24
                dev.drag(*[float(a) for a in args[:4]], steps=steps,
                         button="middle" if cmd == "mdrag" else "left")
            elif cmd == "key":
                dev.key(args[0], True)
                dev.key(args[0], False)
            elif cmd == "keydown":
                dev.key(args[0], True)
            elif cmd == "keyup":
                dev.key(args[0], False)
            elif cmd == "scroll":
                dev.scroll(int(args[0]))
            elif cmd == "sleep":
                time.sleep(float(args[0]) / 1000.0)
            else:
                raise SystemExit(f"unknown command: {cmd}")
            print(f"ok: {line}", flush=True)
    finally:
        dev.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--geometry":
        print("%d %d %d %d" % window_rect())
    else:
        run(sys.argv[1])
