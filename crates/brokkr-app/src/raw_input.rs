// SPDX-License-Identifier: AGPL-3.0-only

//! Reading a HID device on Windows, the way `input_watch` reads one on Linux.
//!
//! Both the stylus and the puck arrive here. Neither goes through the window
//! system: iced 0.14 drops winit's `force` field and winit has no pen pressure
//! on the desktop at all, so the pen has to be read below the toolkit -- and
//! once one device is being read that way the other may as well be, rather
//! than having two unrelated input paths to reason about.
//!
//! # Why this shares helpers and not a thread
//!
//! `input_watch` on Linux is shared because SCANNING is expensive: thirty
//! event nodes, reopened twice a second, is real work and doing it twice would
//! be twice the work. Raw Input has no scan. The OS pushes reports at a window
//! and a blocked `GetMessageW` costs a thread and nothing else, so each backend
//! owns its own window and pump and there is no shared mutable state, no
//! registration protocol and no ordering between them. What is shared is the
//! part that is genuinely identical: creating the sink, and asking Windows what
//! a report says.
//!
//! # Why `HidP_*` and not a report descriptor parser
//!
//! HID reports are vendor specific and the descriptor that explains them is a
//! small language. Windows has already parsed it. Asking by usage --
//! [`value`], [`button`] -- means a tablet and a puck are read by the same four
//! functions, and a device with an unusual report layout is the OS's problem
//! rather than ours.

use windows_sys::Win32::Devices::HumanInterfaceDevice::{
    HIDP_STATUS_SUCCESS, HIDP_VALUE_CAPS, HidP_GetUsageValue, HidP_GetUsages, HidP_GetValueCaps,
    HidP_Input, PHIDP_PREPARSED_DATA,
};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::UI::Input::{
    GetRawInputData, GetRawInputDeviceInfoW, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER,
    RID_INPUT, RIDEV_INPUTSINK, RIDI_PREPARSEDDATA, RegisterRawInputDevices,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, HWND_MESSAGE, MSG,
    RegisterClassW, WM_INPUT, WNDCLASSW,
};

/// HID usage pages this application reads from.
pub const PAGE_GENERIC: u16 = 0x01;
pub const PAGE_DIGITIZER: u16 = 0x0D;

/// Read one kind of device until the process ends, handing every report to
/// `on_report` along with the parsed descriptor it should be read against.
///
/// Returns only on failure, and silently: a device that cannot be read must
/// never stop the application starting. Both callers report the difference
/// between "set up" and "receiving" in their own panel instead, which is the
/// thing a user can actually act on.
///
/// `class` must be unique per caller. Registering a window class twice fails,
/// and two devices read on two threads need two windows.
pub fn pump(
    class: &str,
    usage_page: u16,
    usage: u16,
    on_report: impl Fn(PHIDP_PREPARSED_DATA, &[u8]),
) {
    let class_name: Vec<u16> = class.encode_utf16().chain(std::iter::once(0)).collect();
    let mut descriptor: WNDCLASSW = unsafe { std::mem::zeroed() };
    descriptor.lpfnWndProc = Some(wndproc);
    descriptor.lpszClassName = class_name.as_ptr();
    unsafe { RegisterClassW(&descriptor) };

    // `HWND_MESSAGE` is a window that is never shown, never focused and never
    // in the taskbar. It exists only to be something Raw Input can deliver to.
    let window = unsafe {
        CreateWindowExW(
            0,
            class_name.as_ptr(),
            class_name.as_ptr(),
            0,
            0,
            0,
            0,
            0,
            HWND_MESSAGE,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    if window.is_null() {
        return;
    }

    let device = RAWINPUTDEVICE {
        usUsagePage: usage_page,
        usUsage: usage,
        // Delivered even unfocused, which this window always is: without this
        // flag a message-only window receives nothing at all.
        dwFlags: RIDEV_INPUTSINK,
        hwndTarget: window,
    };
    if unsafe { RegisterRawInputDevices(&device, 1, size_of::<RAWINPUTDEVICE>() as u32) } == 0 {
        return;
    }

    let mut message: MSG = unsafe { std::mem::zeroed() };
    // Returns -1 on error, 0 on WM_QUIT, positive otherwise -- so `> 0` is the
    // only correct test, and `!= 0` would spin forever on an error.
    while unsafe { GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) } > 0 {
        if message.message == WM_INPUT {
            deliver(message.lParam as isize as HRAWINPUT, &on_report);
        }
        unsafe { DispatchMessageW(&message) };
    }
}

/// Nothing to do: `WM_INPUT` is taken from the message loop rather than from
/// the procedure, so there is no state to keep between them.
unsafe extern "system" fn wndproc(window: HWND, message: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    unsafe { DefWindowProcW(window, message, w, l) }
}

/// Pull one `WM_INPUT` apart and hand each report inside it to the caller.
///
/// A single message can carry several reports -- `dwCount` above one is a
/// device that batched them -- and dropping the tail would quietly lose input
/// under exactly the fast movement that produces batching.
fn deliver(handle: HRAWINPUT, on_report: &impl Fn(PHIDP_PREPARSED_DATA, &[u8])) {
    let header = size_of::<RAWINPUTHEADER>() as u32;
    let mut size = 0u32;
    if unsafe { GetRawInputData(handle, RID_INPUT, std::ptr::null_mut(), &mut size, header) } != 0
        || size == 0
    {
        return;
    }
    let mut buffer = vec![0u8; size as usize];
    let read = unsafe {
        GetRawInputData(handle, RID_INPUT, buffer.as_mut_ptr().cast(), &mut size, header)
    };
    if read != size {
        return;
    }

    let raw = buffer.as_ptr() as *const RAWINPUT;
    let hid = unsafe { &(*raw).data.hid };
    let (count, stride) = (hid.dwCount as usize, hid.dwSizeHid as usize);
    if count == 0 || stride == 0 {
        return;
    }

    let Some(preparsed) = preparsed_data(unsafe { (*raw).header.hDevice }) else {
        return;
    };
    let data = preparsed.as_ptr() as PHIDP_PREPARSED_DATA;
    let reports = hid.bRawData.as_ptr();
    for index in 0..count {
        let report = unsafe { std::slice::from_raw_parts(reports.add(index * stride), stride) };
        on_report(data, report);
    }
}

/// The device's parsed report descriptor, which Windows keeps for us.
fn preparsed_data(device: *mut core::ffi::c_void) -> Option<Vec<u8>> {
    let mut size = 0u32;
    unsafe { GetRawInputDeviceInfoW(device, RIDI_PREPARSEDDATA, std::ptr::null_mut(), &mut size) };
    if size == 0 {
        return None;
    }
    let mut buffer = vec![0u8; size as usize];
    let read = unsafe {
        GetRawInputDeviceInfoW(device, RIDI_PREPARSEDDATA, buffer.as_mut_ptr().cast(), &mut size)
    };
    // The bytes written, or -1 as a `u32` when the buffer was short.
    if read == u32::MAX || read == 0 { None } else { Some(buffer) }
}

/// The raw value of a usage in this report, if the device carries it.
pub fn value(
    data: PHIDP_PREPARSED_DATA,
    report: &[u8],
    usage_page: u16,
    usage: u16,
) -> Option<i32> {
    let mut out = 0u32;
    let status = unsafe {
        HidP_GetUsageValue(
            HidP_Input,
            usage_page,
            0,
            usage,
            &mut out,
            data,
            report.as_ptr() as *mut u8,
            report.len() as u32,
        )
    };
    (status == HIDP_STATUS_SUCCESS).then_some(out as i32)
}

/// Whether a button usage is asserted in this report.
///
/// `HidP_GetUsages` fills the usages that are ON, so one the device has but is
/// not asserting comes back absent. `None` is "this call failed", not "up".
pub fn button(
    data: PHIDP_PREPARSED_DATA,
    report: &[u8],
    usage_page: u16,
    usage: u16,
) -> Option<bool> {
    Some(pressed(data, report, usage_page)?.contains(&usage))
}

/// Every button usage asserted in this report, for a caller that wants them
/// all rather than asking one at a time.
pub fn pressed(data: PHIDP_PREPARSED_DATA, report: &[u8], usage_page: u16) -> Option<Vec<u16>> {
    // Thirty-two is past any puck's button count and costs a third of a cache
    // line; a device with more simply has its extras ignored rather than
    // overflowing anything.
    let mut list = [0u16; 32];
    let mut length = list.len() as u32;
    let status = unsafe {
        HidP_GetUsages(
            HidP_Input,
            usage_page,
            0,
            list.as_mut_ptr(),
            &mut length,
            data,
            report.as_ptr() as *mut u8,
            report.len() as u32,
        )
    };
    (status == HIDP_STATUS_SUCCESS).then(|| list[..length as usize].to_vec())
}

/// The logical range the device declares for a usage.
///
/// Read rather than assumed: a tablet reporting 0..8191 and one reporting
/// 0..1023 are both ordinary, and assuming either would give the other eight
/// times the pressure or an eighth of it.
pub fn range(data: PHIDP_PREPARSED_DATA, usage_page: u16, usage: u16) -> Option<(i32, i32)> {
    let mut count = 0u16;
    // The count first, from a null buffer: the caps array is per device, and a
    // fixed guess would either truncate a rich device or waste a page on a
    // plain one.
    unsafe { HidP_GetValueCaps(HidP_Input, std::ptr::null_mut(), &mut count, data) };
    if count == 0 {
        return None;
    }
    let mut caps: Vec<HIDP_VALUE_CAPS> = vec![unsafe { std::mem::zeroed() }; count as usize];
    let status = unsafe { HidP_GetValueCaps(HidP_Input, caps.as_mut_ptr(), &mut count, data) };
    if status != HIDP_STATUS_SUCCESS {
        return None;
    }
    caps[..count as usize].iter().find_map(|cap| {
        let matches = cap.UsagePage == usage_page
            && !cap.IsRange
            && unsafe { cap.Anonymous.NotRange.Usage } == usage;
        matches.then_some((cap.LogicalMin, cap.LogicalMax))
    })
}
