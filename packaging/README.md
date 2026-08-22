<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Packaging bits

Not a packaged build — see the README's "Get it", which says to build from
source and means it. These are the two files a Linux desktop needs in order to
show BrokkrSculpt as an application rather than as an unnamed window.

## Why the icon lives here rather than in the application

The application does **not** set a window icon, and could not usefully: there
is no Wayland protocol for a client to hand its compositor one, and `winit`'s
Wayland backend implements `set_window_icon` as an empty function. The
compositor matches the surface's `app_id` — already `brokkrsculpt` — against an
installed `.desktop` file and takes the icon from the icon theme.

This is also what gives the application a menu entry with a name, which is the
larger part of what it is for. SindriCAD ships the same pair from its bundler.

## Installing for the current user

```fish
install -Dm644 packaging/brokkrsculpt.desktop ~/.local/share/applications/brokkrsculpt.desktop
install -Dm644 assets/brand/brokkrsculpt-mark.svg ~/.local/share/icons/hicolor/scalable/apps/brokkrsculpt.svg
update-desktop-database ~/.local/share/applications 2>/dev/null; gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor 2>/dev/null; true
```

`Exec=brokkrsculpt` assumes the binary is on `PATH`. Running it from
`target/release` instead is fine — the icon matches on `StartupWMClass`, not on
the command.
