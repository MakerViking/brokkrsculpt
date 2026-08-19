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
//! cargo run --release -p brokkr-app --example poke -- type 964 1000 "it went black"
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
    let mut to_text: Option<String> = None;
    let (action, x, y, to) = match args.as_slice() {
        [action, x, y] => (action, number(x), number(y), None),
        // `type` takes text where a drag takes a destination.
        [action, x, y, text] if action == "type" => {
            to_text = Some(text.clone());
            (action, number(x), number(y), None)
        }
        [action, x, y, to_x, to_y] => {
            (action, number(x), number(y), Some((number(to_x), number(to_y))))
        }
        _ => {
            eprintln!(
                "usage: poke <move|click|rightclick|press|release> <x> <y>\n       \
                 poke <drag|rdrag> <x> <y> <to x> <to y>\n       \
                 poke type <x> <y> <text>"
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
        "type" => {
            let text = to_text.as_deref().unwrap_or_default();
            for character in text.chars() {
                let Some((code, shifted)) = key_for(character) else {
                    eprintln!("poke: no key for {character:?}, skipping");
                    continue;
                };
                if shifted {
                    button(&mut device, KeyCode::KEY_LEFTSHIFT, true);
                }
                button(&mut device, code, true);
                button(&mut device, code, false);
                if shifted {
                    button(&mut device, KeyCode::KEY_LEFTSHIFT, false);
                }
                // Slower than a real typist. A key press and release in the
                // same millisecond is within one frame, and a widget that
                // samples on redraw can miss it.
                sleep(Duration::from_millis(12));
            }
        }
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

/// The keys `type` can send: character, key code, and whether shift is held.
///
/// Deliberately a table rather than a computed mapping. A keyboard layout is
/// not a function of the character -- this assumes the compositor is on a US
/// layout, which is what this machine runs, and a wrong assumption shows up as
/// the wrong letter rather than as an error.
const TYPEABLE: &[(char, KeyCode, bool)] = &[
    ('a', KeyCode::KEY_A, false),
    ('b', KeyCode::KEY_B, false),
    ('c', KeyCode::KEY_C, false),
    ('d', KeyCode::KEY_D, false),
    ('e', KeyCode::KEY_E, false),
    ('f', KeyCode::KEY_F, false),
    ('g', KeyCode::KEY_G, false),
    ('h', KeyCode::KEY_H, false),
    ('i', KeyCode::KEY_I, false),
    ('j', KeyCode::KEY_J, false),
    ('k', KeyCode::KEY_K, false),
    ('l', KeyCode::KEY_L, false),
    ('m', KeyCode::KEY_M, false),
    ('n', KeyCode::KEY_N, false),
    ('o', KeyCode::KEY_O, false),
    ('p', KeyCode::KEY_P, false),
    ('q', KeyCode::KEY_Q, false),
    ('r', KeyCode::KEY_R, false),
    ('s', KeyCode::KEY_S, false),
    ('t', KeyCode::KEY_T, false),
    ('u', KeyCode::KEY_U, false),
    ('v', KeyCode::KEY_V, false),
    ('w', KeyCode::KEY_W, false),
    ('x', KeyCode::KEY_X, false),
    ('y', KeyCode::KEY_Y, false),
    ('z', KeyCode::KEY_Z, false),
    ('0', KeyCode::KEY_0, false),
    ('1', KeyCode::KEY_1, false),
    ('2', KeyCode::KEY_2, false),
    ('3', KeyCode::KEY_3, false),
    ('4', KeyCode::KEY_4, false),
    ('5', KeyCode::KEY_5, false),
    ('6', KeyCode::KEY_6, false),
    ('7', KeyCode::KEY_7, false),
    ('8', KeyCode::KEY_8, false),
    ('9', KeyCode::KEY_9, false),
    (' ', KeyCode::KEY_SPACE, false),
    ('.', KeyCode::KEY_DOT, false),
    (',', KeyCode::KEY_COMMA, false),
    ('-', KeyCode::KEY_MINUS, false),
    ('/', KeyCode::KEY_SLASH, false),
    ('\n', KeyCode::KEY_ENTER, false),
    ('?', KeyCode::KEY_SLASH, true),
    ('!', KeyCode::KEY_1, true),
    (':', KeyCode::KEY_SEMICOLON, true),
    (';', KeyCode::KEY_SEMICOLON, false),
];

fn key_for(character: char) -> Option<(KeyCode, bool)> {
    let lower = character.to_ascii_lowercase();
    TYPEABLE
        .iter()
        .find(|(c, _, _)| *c == lower)
        .map(|(_, code, shifted)| (*code, *shifted || character.is_ascii_uppercase()))
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
    // Every key `type` can emit, declared up front: a uinput device's
    // capabilities are fixed when it is created, and a key that was not
    // declared is silently dropped rather than refused.
    for key in TYPEABLE.iter().map(|(_, code, _)| *code) {
        keys.insert(key);
    }
    keys.insert(KeyCode::KEY_LEFTSHIFT);

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
