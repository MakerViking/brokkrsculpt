// SPDX-License-Identifier: AGPL-3.0-only

//! BrokkrSculpt: a voxel and SDF based 3D sculpting application.

mod app;
mod breadcrumbs;
mod camera;
mod cursor;
#[cfg(target_os = "linux")]
mod icon;
mod input_watch;
mod logo;
mod message;
mod navcube;
mod paths;
mod printer;
mod recent;
mod report;
mod slicer;
mod spacemouse;
mod tablet;
mod theme;
mod timeline;
mod viewport;

use app::Brokkr;

/// A named function rather than a closure: iced needs this to be callable for
/// any lifetime of the borrowed state, and a closure gets inferred at one
/// specific lifetime instead.
fn app_theme(_state: &Brokkr) -> iced::Theme {
    theme::theme()
}

fn main() -> iced::Result {
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
