// SPDX-License-Identifier: AGPL-3.0-only

//! BrokkrSculpt: a voxel and SDF based 3D sculpting application.

// **No console window on Windows.** Without this the binary is built for the
// console subsystem, so double-clicking it opens a black terminal beside the
// application and leaves it there for the session. A Windows tester's first
// screenshot on 2026-08-30 had exactly that in it.
//
// Redirected output still works: the subsystem flag governs whether a console
// is ALLOCATED, not whether inherited handles are written to. So
// `brokkrsculpt.exe --version | grep ...` -- which is how `release.yml` reads
// the build ordinal back out of what it just built -- keeps working, because
// its stdout is a pipe. What is lost is running `--version` interactively in
// an existing `cmd` window, where the output goes nowhere: attaching to a
// parent console needs `AttachConsole` and is not worth it for a flag whose
// only consumer is CI.
//
// Release builds only. A debug build keeps its console, because that is where
// `env_logger` output goes while developing.
//
// If this is wrong, CI fails loudly at the readback rather than shipping
// something broken -- which is the safe direction for a change that cannot be
// tested from this machine.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod account;
mod app;
mod articles;
mod breadcrumbs;
mod camera;
mod crash;
mod cursor;
mod gizmo;
mod icon;
// `/dev/input` and the evdev crate are Linux only, and so is everything that
// reads them: the stylus in `tablet` and the puck in `spacemouse` both sit
// behind the same gate. Windows and macOS need Pointer Input, Wintab or IOKit
// instead, which is separate work rather than a port of this.
#[cfg(target_os = "linux")]
mod input_watch;
mod logo;
mod message;
mod navcube;
mod paths;
mod printer;
// Reading HID devices on macOS and on Windows, the way `input_watch` reads
// them on Linux. Both the pen and the puck come through whichever applies.
#[cfg(target_os = "macos")]
mod raw_hid;
#[cfg(target_os = "windows")]
mod raw_input;
mod recent;
mod report;
mod slicer;
mod spacemouse;
mod tablet;
mod theme;
mod thumbnails;
mod timeline;
mod update;
mod viewport;
mod welcome;

use app::Brokkr;

/// A named function rather than a closure: iced needs this to be callable for
/// any lifetime of the borrowed state, and a closure gets inferred at one
/// specific lifetime instead.
fn app_theme(_state: &Brokkr) -> iced::Theme {
    theme::theme()
}

fn main() -> iced::Result {
    // **The channel CI reads the build ordinal back out of.**
    //
    // `release.yml` stamps `BROKKR_BUILD` and then greps this output for it,
    // because the way that stamp goes wrong is silent: `build.rs` returns early
    // when `BROKKR_COMMIT` is set, and a cached build-script output would ship a
    // binary reporting the previous run's ordinal with nothing anywhere saying
    // so. Compiling proves nothing; asking the artefact does.
    //
    // Flat `key = value`, the same shape `paths::entries` parses, so the update
    // files, the signed manifest and this output all read alike.
    //
    // This is not the "hidden flag pointing the updater elsewhere" that the plan
    // forbids: it takes no argument, opens no socket and writes nothing. Before
    // `crash::install()`, like its two neighbours, so keep it to `option_env!`
    // reads and printing -- there is no panic handler installed yet. Printing
    // the commit is also the AGPL corresponding-source answer.
    if std::env::args().any(|argument| argument == "--version") {
        // Formatted in `update::version_report` rather than here, so it can be
        // unit tested: this crate is binary-only, so nothing in `main.rs` is
        // reachable from a test, and CI greps this exact output for the build
        // ordinal. A reworded line would break the release pipeline silently.
        print!("{}", update::version_report());
        return Ok(());
    }
    // A tablet that does not work is hard to diagnose from inside a running
    // application, so the same detection logic is available without one.
    if std::env::args().any(|argument| argument == "--tablets") {
        print!("{}", tablet::report());
        return Ok(());
    }
    // Likewise for the puck: "it is plugged in and nothing happens" is
    // otherwise indistinguishable from "it was never found".
    if std::env::args().any(|argument| argument == "--spacemouse") {
        print!("{}", spacemouse::report());
        return Ok(());
    }

    // Before anything can panic, and before the window opens. A panic used to
    // take the window and leave nothing behind but a backtrace on a terminal
    // the user had not launched from.
    crash::install();

    // `brokkrsculpt`, not `brokkr_app`: env_logger filters on the MODULE PATH,
    // and the module path is the crate root name, which for a binary target is
    // the `[[bin]] name` rather than the package name. It read `brokkr_app`
    // until 2026-08-21, matched nothing, and so every `log::info!` in this
    // crate had been invisible since the day it was written -- which is why an
    // import or a resample left no trace in a log people were being told to
    // read. `brokkr_gpu` IS correct: that one is a library.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,brokkrsculpt=info,brokkr_gpu=info"),
    )
    .init();

    iced::application(Brokkr::new, Brokkr::update, Brokkr::view)
        .title(Brokkr::title)
        .subscription(Brokkr::subscription)
        .theme(app_theme)
        .default_font(theme::FONT)
        .antialiasing(true)
        // The close button must not end the process on its own: unsaved work
        // needs a prompt first. `Brokkr::subscription` picks the request up as
        // `Message::CloseRequested` and is the only thing that closes the
        // window from here on, so the two must be changed together.
        .exit_on_close_request(false)
        // No compositor title bar: the application draws its own, the way
        // SindriCAD does, so there is one bar instead of two. `panel.rs`'s
        // `header` carries the move, maximise, minimise and close that the
        // decoration would have provided.
        .window(iced::window::Settings { decorations: false, ..Default::default() })
        .run()
}
