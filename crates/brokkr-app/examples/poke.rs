// SPDX-License-Identifier: AGPL-3.0-or-later

//! Drive the running application with a synthetic pointer.
//!
//! There is no Wayland input injection on this machine -- no ydotool, wtype or
//! dotool, and XTEST does nothing under this compositor -- which is why every
//! pointer gesture in this project has been verified by test rather than by
//! hand. But `/dev/uinput` *is* writable here, which is how the tablet and
//! SpaceMouse tests already build synthetic devices, and a device made that way
//! is indistinguishable from real hardware to the compositor. So this is the
//! injection that was said not to exist.
//!
//! Absolute rather than relative, deliberately: libinput applies pointer
//! acceleration to relative motion, so a delta of 500 does not move 500 pixels
//! and no amount of care makes it land somewhere exact. An absolute device is
//! told where to be.
//!
//! ```text
//! cargo run --release -p brokkr-app --example poke -- move 964 1000
//! cargo run --release -p brokkr-app --example poke -- click 964 1000
//! cargo run --release -p brokkr-app --example poke -- rightclick 964 1000
//! cargo run --release -p brokkr-app --example poke -- drag 964 1000 1200 1000
//! cargo run --release -p brokkr-app --example poke -- rdrag 900 1300 1100 1300
//! ```
//!
//! A drag is emitted as a run of small steps rather than one jump, because
//! every gesture in this application distinguishes a drag from a click by how
//! far the pointer travelled, and a single leap looks like neither.
//!
//! Coordinates are logical desktop pixels, the ones KWin's `frameGeometry`
//! reports. `DESKTOP_W`/`DESKTOP_H` below have to match the desktop this runs
//! on, because an absolute device's range is what maps onto it.

use std::thread::sleep;
use std::time::Duration;

use evdev::uinput::VirtualDevice;
use evdev::{
    AbsInfo, AbsoluteAxisCode, AttributeSet, EventType, InputEvent, KeyCode, UinputAbsSetup,
};

/// Logical size of the whole desktop, which is what an absolute device's range
/// is mapped onto.
///
/// From `kscreen-doctor -o`: this desktop is two outputs, HDMI-A-1 at 786,0
/// sized 1536x864 and DP-1 at 0,864 sized 2294x960, so the union runs to
/// 2322x1824. Change these when the monitors change; nothing here can find out
/// on its own.
const DESKTOP_W: i32 = 2322;
const DESKTOP_H: i32 = 1824;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (action, x, y, to) = match args.as_slice() {
        [action, x, y] => (action, number(x), number(y), None),
        [action, x, y, to_x, to_y] => {
            (action, number(x), number(y), Some((number(to_x), number(to_y))))
        }
        _ => {
            eprintln!(
                "usage: poke <move|click|rightclick|press|release> <x> <y>\n       \
                 poke <drag|rdrag> <x> <y> <to x> <to y>"
            );
            std::process::exit(2);
        }
    };

    let mut device = build().expect(
        "could not create a uinput device -- needs write access to /dev/uinput, \
         which means membership of the `input` group",
    );

    // The compositor needs a moment to notice a device appearing, and a click
    // sent before it has is delivered nowhere.
    sleep(Duration::from_millis(400));

    move_to(&mut device, x, y);
    sleep(Duration::from_millis(120));

    match action.as_str() {
        "move" => {}
        "click" => {
            button(&mut device, KeyCode::BTN_LEFT, true);
            sleep(Duration::from_millis(60));
            button(&mut device, KeyCode::BTN_LEFT, false);
        }
        "rightclick" => {
            button(&mut device, KeyCode::BTN_RIGHT, true);
            sleep(Duration::from_millis(60));
            button(&mut device, KeyCode::BTN_RIGHT, false);
        }
        "press" => button(&mut device, KeyCode::BTN_LEFT, true),
        "release" => button(&mut device, KeyCode::BTN_LEFT, false),
        "drag" | "rdrag" => {
            let (to_x, to_y) = to.expect("a drag needs a destination");
            let held = if action == "rdrag" { KeyCode::BTN_RIGHT } else { KeyCode::BTN_LEFT };
            button(&mut device, held, true);
            sleep(Duration::from_millis(60));
            // Enough steps that the travel reads as a drag at every gesture's
            // slop threshold, and slow enough that a frame is drawn between
            // them -- an application that only samples the pointer on redraw
            // would otherwise see the start and the end and nothing between.
            const STEPS: i32 = 24;
            for step in 1..=STEPS {
                let t = step as f32 / STEPS as f32;
                move_to(
                    &mut device,
                    x + ((to_x - x) as f32 * t) as i32,
                    y + ((to_y - y) as f32 * t) as i32,
                );
                sleep(Duration::from_millis(16));
            }
            sleep(Duration::from_millis(60));
            button(&mut device, held, false);
        }
        other => {
            eprintln!("unknown action {other}");
            std::process::exit(2);
        }
    }

    // Held open a moment: dropping the device immediately can remove it before
    // the compositor has processed the events it just sent.
    sleep(Duration::from_millis(250));
    println!("poked {action} at {x},{y}");
}

fn number(raw: &str) -> i32 {
    raw.parse().unwrap_or_else(|_| panic!("{raw} is not a number"))
}

fn build() -> Option<VirtualDevice> {
    let mut keys = AttributeSet::<KeyCode>::new();
    // BTN_LEFT with absolute axes and no BTN_TOOL_* is what libinput reads as
    // an absolute pointer rather than as a tablet or a touchscreen.
    keys.insert(KeyCode::BTN_LEFT);
    keys.insert(KeyCode::BTN_RIGHT);

    let axis = |code, maximum| UinputAbsSetup::new(code, AbsInfo::new(0, 0, maximum, 0, 0, 0));

    VirtualDevice::builder()
        .ok()?
        .name("brokkr synthetic pointer")
        .with_keys(&keys)
        .ok()?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_X, DESKTOP_W))
        .ok()?
        .with_absolute_axis(&axis(AbsoluteAxisCode::ABS_Y, DESKTOP_H))
        .ok()?
        .build()
        .ok()
}

fn move_to(device: &mut VirtualDevice, x: i32, y: i32) {
    emit(
        device,
        &[
            InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_X.0, x),
            InputEvent::new(EventType::ABSOLUTE.0, AbsoluteAxisCode::ABS_Y.0, y),
        ],
    );
}

fn button(device: &mut VirtualDevice, code: KeyCode, down: bool) {
    emit(device, &[InputEvent::new(EventType::KEY.0, code.0, i32::from(down))]);
}

fn emit(device: &mut VirtualDevice, events: &[InputEvent]) {
    device.emit(events).expect("could not emit to the synthetic pointer");
}
