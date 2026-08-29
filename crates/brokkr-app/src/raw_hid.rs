// SPDX-License-Identifier: AGPL-3.0-only

//! Reading a HID device on macOS, the way `raw_input` reads one on Windows and
//! `input_watch` reads one on Linux.
//!
//! Both the stylus and the puck arrive here, and for the same reason they do on
//! the other two: iced 0.14 drops winit's `force` field and winit has no
//! desktop pen pressure at all, so the pen has to be read below the toolkit.
//!
//! # IOKit directly, and no crate for it
//!
//! `io-kit-sys` exists and is not in this graph. Ten `extern` declarations are,
//! and this is the same trade the nonce made in `account`: a dependency is
//! worth it when it does something hard, and declaring the handful of functions
//! a caller actually uses is not hard. `core-foundation` IS used, because it is
//! already in the graph via winit and because hand-rolling retain and release
//! is exactly the kind of hard that earns a crate.
//!
//! # Elements, not reports
//!
//! This is the one place where macOS is simpler than Windows. Raw Input hands
//! over a report and leaves the parsing to `HidP_*`; IOKit has already split it
//! and calls back per ELEMENT, with the usage and the logical range attached.
//! So there is no report layout to reason about here at all -- the callback is
//! handed "this usage, on this page, is now this value, out of this range".
//!
//! # One run loop per device kind
//!
//! Same argument as Windows: matching costs nothing once registered, so each
//! backend owns a manager and a run loop rather than sharing one and needing a
//! protocol to add matchers to it. A `CFRunLoopRun` thread is a blocked thread.

use std::ffi::c_void;

use core_foundation::base::{CFRelease, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::runloop::{CFRunLoop, kCFRunLoopDefaultMode};
use core_foundation::string::CFString;

/// HID usage pages this application reads from. The numbers are the HID
/// specification's own and match `raw_input`'s, which is not a coincidence
/// worth hiding: it is the same table underneath both platforms.
pub const PAGE_GENERIC: u32 = 0x01;
pub const PAGE_DIGITIZER: u32 = 0x0D;
pub const PAGE_BUTTON: u32 = 0x09;

type IOHIDManagerRef = *mut c_void;
type IOHIDValueRef = *mut c_void;
type IOHIDElementRef = *mut c_void;

/// What IOKit hands back for one element that changed.
pub struct Sample {
    pub page: u32,
    pub usage: u32,
    pub value: i64,
    /// The logical range the element declares, for normalising against.
    pub minimum: i64,
    pub maximum: i64,
}

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOHIDManagerCreate(allocator: *const c_void, options: u32) -> IOHIDManagerRef;
    fn IOHIDManagerSetDeviceMatching(manager: IOHIDManagerRef, matching: *const c_void);
    fn IOHIDManagerRegisterInputValueCallback(
        manager: IOHIDManagerRef,
        callback: extern "C" fn(*mut c_void, i32, *mut c_void, IOHIDValueRef),
        context: *mut c_void,
    );
    fn IOHIDManagerScheduleWithRunLoop(
        manager: IOHIDManagerRef,
        run_loop: *mut c_void,
        mode: *const c_void,
    );
    fn IOHIDManagerOpen(manager: IOHIDManagerRef, options: u32) -> i32;
    fn IOHIDValueGetElement(value: IOHIDValueRef) -> IOHIDElementRef;
    fn IOHIDValueGetIntegerValue(value: IOHIDValueRef) -> isize;
    fn IOHIDElementGetUsagePage(element: IOHIDElementRef) -> u32;
    fn IOHIDElementGetUsage(element: IOHIDElementRef) -> u32;
    fn IOHIDElementGetLogicalMin(element: IOHIDElementRef) -> isize;
    fn IOHIDElementGetLogicalMax(element: IOHIDElementRef) -> isize;
}

/// Everything the callback needs, kept alive for the life of the run loop.
///
/// Leaked on purpose. The run loop below never returns, so this lives until the
/// process ends and there is no drop to arrange; boxing it and reclaiming it
/// would mean proving IOKit holds no pointer to it after a return that does not
/// happen.
struct Sink {
    on_sample: Box<dyn Fn(Sample)>,
}

/// Read one kind of device until the process ends, handing every changed
/// element to `on_sample`.
///
/// Returns only on failure, and silently, for the reason the other two
/// backends do: a device that cannot be read must never stop the application
/// starting. Each caller says the difference between "matching" and
/// "receiving" in its own panel instead.
pub fn pump(usage_page: u32, usage: u32, on_sample: impl Fn(Sample) + 'static) {
    let manager = unsafe { IOHIDManagerCreate(std::ptr::null(), 0) };
    if manager.is_null() {
        return;
    }

    // The matching dictionary is by DEVICE usage, not element usage: it selects
    // which devices to open, and every element on them is then delivered.
    let matching = CFDictionary::from_CFType_pairs(&[
        (
            CFString::new("DeviceUsagePage").as_CFType(),
            CFNumber::from(usage_page as i32).as_CFType(),
        ),
        (CFString::new("DeviceUsage").as_CFType(), CFNumber::from(usage as i32).as_CFType()),
    ]);
    unsafe { IOHIDManagerSetDeviceMatching(manager, matching.as_CFTypeRef() as *const c_void) };

    let sink = Box::into_raw(Box::new(Sink { on_sample: Box::new(on_sample) }));
    unsafe {
        IOHIDManagerRegisterInputValueCallback(manager, on_value, sink as *mut c_void);
        IOHIDManagerScheduleWithRunLoop(
            manager,
            CFRunLoop::get_current().as_CFTypeRef() as *mut c_void,
            kCFRunLoopDefaultMode as *const c_void,
        );
    }
    // kIOHIDOptionsTypeNone. A non-zero return is a device this process is not
    // allowed to open, which on a Mac means Input Monitoring has not been
    // granted -- silent here, and visible in the panel as "none has reported".
    if unsafe { IOHIDManagerOpen(manager, 0) } != 0 {
        unsafe { CFRelease(manager as *const c_void) };
        return;
    }

    CFRunLoop::run_current();
}

/// IOKit's callback, once per element that changed.
extern "C" fn on_value(
    context: *mut c_void,
    _result: i32,
    _sender: *mut c_void,
    value: IOHIDValueRef,
) {
    if context.is_null() || value.is_null() {
        return;
    }
    let element = unsafe { IOHIDValueGetElement(value) };
    if element.is_null() {
        return;
    }
    let sample = Sample {
        page: unsafe { IOHIDElementGetUsagePage(element) },
        usage: unsafe { IOHIDElementGetUsage(element) },
        value: unsafe { IOHIDValueGetIntegerValue(value) } as i64,
        minimum: unsafe { IOHIDElementGetLogicalMin(element) } as i64,
        maximum: unsafe { IOHIDElementGetLogicalMax(element) } as i64,
    };
    let sink = unsafe { &*(context as *const Sink) };
    (sink.on_sample)(sample);
}
