// SPDX-License-Identifier: AGPL-3.0-or-later

//! BrokkrSculpt: a voxel and SDF based 3D sculpting application.

mod app;
mod camera;
mod cursor;
#[cfg(target_os = "linux")]
mod input_watch;
mod message;
mod spacemouse;
mod tablet;
mod theme;
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

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,brokkr_app=info,brokkr_gpu=info"),
    )
    .init();

    iced::application(Brokkr::new, Brokkr::update, Brokkr::view)
        .title(Brokkr::title)
        .subscription(Brokkr::subscription)
        .theme(app_theme)
        .default_font(theme::FONT)
        .antialiasing(true)
        .run()
}
