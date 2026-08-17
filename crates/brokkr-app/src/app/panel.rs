// SPDX-License-Identifier: AGPL-3.0-or-later

//! The widget tree.
//!
//! Split out of `app.rs`, which was doing state, input handling and every
//! widget in one file. Nothing here holds state of its own: each function
//! reads `Brokkr` and returns an `Element`.
//!
//! This is a child module of `app` rather than a sibling so that `Brokkr`'s
//! fields can stay private. A sibling would have forced every one of them to
//! `pub(crate)` just to draw them.

use std::sync::Arc;

use brokkr_core::{BrushKind, FalloffCurve, Symmetry};
use iced::widget::{button, checkbox, column, container, pick_list, row, slider, stack, text};
use iced::{Alignment, Element, Length};

use super::{
    Brokkr, COARSEST_VOXEL_MM, FINEST_VOXEL_MM, PuckAction, SpaceMouseConfig, SpaceMouseSetting,
};
use crate::message::{ExportFormat, Message};
use crate::spacemouse::{self, ButtonAction};
use crate::tablet::Diagnosis;
use crate::theme;
use crate::viewport::Viewport;

impl Brokkr {
    pub fn view(&self) -> Element<'_, Message> {
        let viewport = iced::widget::shader(Viewport::new(Arc::clone(&self.shared)))
            .width(Length::Fill)
            .height(Length::Fill);

        let well = container(viewport)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(theme::viewport_well);

        let scene = stack![well, self.overlay()];

        column![
            self.header(),
            row![container(scene).width(Length::Fill).height(Length::Fill), self.tools()]
                .spacing(theme::S3)
        ]
        .spacing(theme::S3)
        .padding(theme::S3)
        .into()
    }

    fn header(&self) -> Element<'_, Message> {
        container(
            row![
                text("BROKKRSCULPT")
                    .size(theme::CAPTION_SIZE)
                    .font(theme::FONT)
                    .color(theme::ACCENT),
                text("M1 brush system").size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S4)
            .align_y(Alignment::Center),
        )
        .padding(theme::PANEL_PADDING)
        .width(Length::Fill)
        .style(theme::panel)
        .into()
    }

    /// The debug overlay: frame rate, frame time, triangles, bricks, resident
    /// memory and what history is holding.
    fn overlay(&self) -> Element<'_, Message> {
        let pool = self.shared.stats();
        let frame_ms = self.perf.average_frame_ms();
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };

        let volume_mb = self.volume_stats.resident_bytes as f64 / (1024.0 * 1024.0);
        let pool_mb =
            (pool.vertices as f64 * 24.0 + pool.triangles as f64 * 12.0) / (1024.0 * 1024.0);
        let history_mb = self.history_stats.bytes as f64 / (1024.0 * 1024.0);

        let mut lines = vec![
            format!(
                "{fps:6.1} fps    {frame_ms:5.2} ms avg   {:5.2} ms worst",
                self.perf.worst_frame_ms()
            ),
            format!(
                "edit {:5.3} ms   remesh {:5.3} ms   {} stamps   {} dirty   (load {:.0} ms)",
                self.perf.edit_ms,
                self.perf.remesh_ms,
                self.perf.stamps,
                self.perf.dirty_bricks,
                self.perf.load_ms
            ),
            format!(
                "{} triangles   {} drawn / {} culled bricks",
                pool.triangles, pool.drawn, pool.culled
            ),
            format!(
                "{} meshed bricks   pen {}",
                pool.bricks,
                match self.tablet.devices().first() {
                    Some(device) => format!("{:.2} ({})", self.perf.pressure, device.name),
                    None => self.tablet.diagnosis().explain().to_string(),
                }
            ),
            format!(
                "{} dense + {} uniform bricks   {volume_mb:.1} MB volume   {pool_mb:.1} MB mesh",
                self.volume_stats.dense_bricks, self.volume_stats.uniform_bricks
            ),
            format!(
                "history {} undo / {} redo   {history_mb:.1} MB of {} MB{}",
                self.history_stats.undo_entries,
                self.history_stats.redo_entries,
                self.history_stats.budget_bytes / (1024 * 1024),
                if self.history_stats.dropped > 0 {
                    format!("   {} dropped", self.history_stats.dropped)
                } else {
                    String::new()
                }
            ),
        ];
        if pool.overflowed > 0 {
            lines.push(format!("MESH POOL FULL: {} bricks missing from the view", pool.overflowed));
        }

        let readout = lines.into_iter().fold(column![].spacing(2), |stacked, line| {
            stacked.push(text(line).size(theme::TEXT_SIZE_SMALL).font(theme::MONO))
        });

        container(container(readout).padding(theme::S3).style(theme::overlay_card))
            .padding(theme::S4)
            .into()
    }

    fn tools(&self) -> Element<'_, Message> {
        let invert_hint = if self.brush.kind.is_directional() {
            "ctrl drag removes"
        } else {
            "no opposite: ctrl does nothing"
        };

        let radius = column![
            text(format!("Radius  {:.2} mm", self.brush.radius))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.25..=12.0, self.brush.radius, Message::BrushRadiusChanged).step(0.05),
        ]
        .spacing(theme::S2);

        let strength = column![
            text(format!("Strength  {:.2}", self.brush.strength))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.02..=0.80, self.brush.strength, Message::BrushStrengthChanged).step(0.01),
        ]
        .spacing(theme::S2);

        let falloff = column![
            text("Falloff").size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_DIM),
            pick_list(FalloffCurve::ALL, Some(self.brush.falloff), Message::FalloffChanged)
                .text_size(theme::TEXT_SIZE_SMALL)
                .width(Length::Fill),
        ]
        .spacing(theme::S2);

        let history = row![
            button(text("Undo").size(theme::TEXT_SIZE_SMALL))
                .on_press_maybe(self.history.can_undo().then_some(Message::Undo)),
            button(text("Redo").size(theme::TEXT_SIZE_SMALL))
                .on_press_maybe(self.history.can_redo().then_some(Message::Redo)),
        ]
        .spacing(theme::S2);

        container(
            column![
                text("BRUSH").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                pick_list(BrushKind::ALL, Some(self.brush.kind), Message::BrushKindChanged)
                    .text_size(theme::TEXT_SIZE_SMALL)
                    .width(Length::Fill),
                text(invert_hint).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                radius,
                strength,
                falloff,
                checkbox(self.symmetry == Symmetry::X)
                    .label("X symmetry")
                    .on_toggle(Message::SymmetryToggled)
                    .text_size(theme::TEXT_SIZE_SMALL),
                self.pen_panel(),
                self.spacemouse_panel(),
                text("HISTORY").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                history,
                self.detail_panel(),
                self.export_panel(),
                button(text("Reset sphere").size(theme::TEXT_SIZE_SMALL))
                    .on_press(Message::ResetSphere),
                text(
                    "drag: sculpt\nctrl drag: invert\nright drag: orbit\nshift right drag: pan\nwheel: zoom\nctrl z, ctrl shift z: undo, redo"
                )
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S4),
        )
        .padding(theme::PANEL_PADDING)
        .width(Length::Fixed(240.0))
        .height(Length::Fill)
        .style(theme::panel)
        .into()
    }

    /// SpaceMouse settings, plus the live readout.
    ///
    /// The readout is the point of this panel as much as the bindings are: a
    /// puck that is connected but silent, or one whose axes are not where the
    /// labels claim, is otherwise undiagnosable from inside the application.
    /// The pen panel exists for the same reason.
    fn spacemouse_panel(&self) -> Element<'_, Message> {
        let config = &self.spacemouse.config;
        let motion = self.spacemouse.motion();
        let full_scale = self.spacemouse.full_scale();

        let status = self.spacemouse.diagnosis();
        let header: Element<'_, Message> = match self.spacemouse.devices().first() {
            Some(device) => {
                text(device.name.clone()).size(theme::CAPTION_SIZE).color(theme::TEXT_DIM).into()
            }
            None => text(status.explain())
                .size(theme::CAPTION_SIZE)
                .color(if status == spacemouse::Diagnosis::NoDevice {
                    theme::TEXT_MUTE
                } else {
                    theme::WARN
                })
                .into(),
        };

        // One line per raw axis: name, a bar, and the number. Scaled against
        // the largest push seen, because a relative axis carries no range to
        // read from the device.
        let readout = spacemouse::Axis::ALL.into_iter().fold(
            column![].spacing(theme::S1),
            |assembled, axis| {
                let value = motion.axis(axis);
                let past_deadzone = value.abs() >= config.deadzone;
                assembled.push(
                    row![
                        text(axis.label())
                            .size(theme::CAPTION_SIZE)
                            .color(theme::TEXT_MUTE)
                            .width(Length::Fixed(84.0)),
                        text(format!("{value:>6.0}"))
                            .size(theme::CAPTION_SIZE)
                            .color(if past_deadzone { theme::ACCENT } else { theme::TEXT_MUTE }),
                        text(format!("{:>4.0}%", value / full_scale.max(1.0) * 100.0))
                            .size(theme::CAPTION_SIZE)
                            .color(theme::TEXT_MUTE),
                    ]
                    .spacing(theme::S2),
                )
            },
        );

        // One row per action: which axis drives it, and whether it is flipped.
        let bindings =
            PuckAction::ALL.into_iter().fold(column![].spacing(theme::S2), |assembled, action| {
                let binding = config.binding(action);
                assembled.push(
                    column![
                        text(action.label()).size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
                        row![
                            pick_list(spacemouse::Axis::ALL, Some(binding.source), move |axis| {
                                Message::SpaceMouse(SpaceMouseSetting::Binding(action, axis))
                            })
                            .text_size(theme::CAPTION_SIZE)
                            .width(Length::Fill),
                            checkbox(binding.invert)
                                .label("flip")
                                .on_toggle(move |invert| Message::SpaceMouse(
                                    SpaceMouseSetting::Invert(action, invert)
                                ))
                                .text_size(theme::CAPTION_SIZE),
                        ]
                        .spacing(theme::S2)
                        .align_y(Alignment::Center),
                    ]
                    .spacing(theme::S1),
                )
            });

        let buttons = (0..self.spacemouse.config.buttons.len()).fold(
            column![].spacing(theme::S2),
            |assembled, index| {
                assembled.push(
                    column![
                        text(format!("Button {}", index + 1))
                            .size(theme::CAPTION_SIZE)
                            .color(theme::TEXT_DIM),
                        pick_list(
                            ButtonAction::ALL,
                            Some(self.spacemouse.config.buttons[index]),
                            move |action| Message::SpaceMouse(SpaceMouseSetting::Button(
                                index, action
                            ))
                        )
                        .text_size(theme::CAPTION_SIZE)
                        .width(Length::Fill),
                    ]
                    .spacing(theme::S1),
                )
            },
        );

        // Sensitivities are shown as a multiple of the ported default rather
        // than as 6e-7, which is a number nobody can reason about.
        let sensitivity =
            |label: &'static str, value: f32, default: f32, make: fn(f32) -> SpaceMouseSetting| {
                let multiple = value / default;
                column![
                    text(format!("{label}  {multiple:.2}x"))
                        .size(theme::CAPTION_SIZE)
                        .color(theme::TEXT_DIM),
                    slider(0.1..=5.0, multiple, move |m| Message::SpaceMouse(make(m * default)))
                        .step(0.05)
                        .on_release(Message::SpaceMouse(SpaceMouseSetting::Save)),
                ]
                .spacing(theme::S1)
            };

        let defaults = SpaceMouseConfig::default();

        column![
            text("SPACEMOUSE").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            header,
            readout,
            row![
                text("Mode").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
                pick_list(
                    [spacemouse::Mode::Object, spacemouse::Mode::Camera],
                    Some(config.mode),
                    |mode| Message::SpaceMouse(SpaceMouseSetting::Mode(mode))
                )
                .text_size(theme::CAPTION_SIZE)
                .width(Length::Fill),
            ]
            .spacing(theme::S2)
            .align_y(Alignment::Center),
            column![
                text(format!("Deadzone  {:.0}", config.deadzone))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_DIM),
                slider(0.0..=120.0, config.deadzone, |value| Message::SpaceMouse(
                    SpaceMouseSetting::Deadzone(value)
                ))
                .step(1.0)
                .on_release(Message::SpaceMouse(SpaceMouseSetting::Save)),
            ]
            .spacing(theme::S1),
            sensitivity("Pan", config.pan_sens, defaults.pan_sens, SpaceMouseSetting::PanSens),
            sensitivity("Zoom", config.zoom_sens, defaults.zoom_sens, SpaceMouseSetting::ZoomSens),
            sensitivity(
                "Rotate",
                config.orbit_sens,
                defaults.orbit_sens,
                SpaceMouseSetting::OrbitSens
            ),
            bindings,
            buttons,
            button(text("Reset puck settings").size(theme::CAPTION_SIZE))
                .on_press(Message::SpaceMouse(SpaceMouseSetting::Reset)),
        ]
        .spacing(theme::S2)
        .into()
    }

    /// Resolution controls.
    ///
    /// Halving and doubling rather than a free slider: the voxel size sets
    /// memory by its inverse cube, so a dragged slider would walk a model
    /// straight past what the mesh pool holds. Two steps make the cost of each
    /// one obvious.
    fn detail_panel(&self) -> Element<'_, Message> {
        let finer = self.voxel_size / 2.0;
        let coarser = self.voxel_size * 2.0;

        column![
            text("DETAIL").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            text(format!("Voxel  {:.3} mm", self.voxel_size))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            row![
                button(text("finer").size(theme::TEXT_SIZE_SMALL))
                    .on_press_maybe((finer >= FINEST_VOXEL_MM).then_some(Message::Resample(finer))),
                button(text("coarser").size(theme::TEXT_SIZE_SMALL)).on_press_maybe(
                    (coarser <= COARSEST_VOXEL_MM).then_some(Message::Resample(coarser))
                ),
            ]
            .spacing(theme::S2),
        ]
        .spacing(theme::S2)
        .into()
    }

    /// Export controls, plus what the last attempt did.
    fn export_panel(&self) -> Element<'_, Message> {
        let buttons =
            ExportFormat::ALL.into_iter().fold(row![].spacing(theme::S2), |assembled, format| {
                assembled.push(
                    button(text(format.label()).size(theme::TEXT_SIZE_SMALL))
                        .on_press(Message::Export(format)),
                )
            });

        let status: Element<'_, Message> = if self.status.is_empty() {
            text(format!("to {}", Self::export_directory().display()))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE)
                .into()
        } else {
            text(self.status.clone())
                .size(theme::CAPTION_SIZE)
                .color(
                    if self.status.contains("not exported") || self.status.contains("could not") {
                        theme::ERROR
                    } else {
                        theme::OK
                    },
                )
                .into()
        };

        column![text("EXPORT").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE), buttons, status,]
            .spacing(theme::S2)
            .into()
    }

    /// Pen controls, plus enough of a live readout that a user can tell whether
    /// their tablet is being seen at all.
    ///
    /// Without this, a tablet that is connected but unreadable looks exactly
    /// like a mouse: strokes work, they just never vary. The device name, the
    /// device's own pressure range and a live peak turn that into something
    /// answerable in a few seconds.
    fn pen_panel(&self) -> Element<'_, Message> {
        let devices = self.tablet.devices();
        let status: Element<'_, Message> = match devices.first() {
            Some(device) => column![
                text(device.name.clone()).size(theme::CAPTION_SIZE).color(theme::OK),
                text(format!("{} levels", device.pressure_max))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE),
            ]
            .spacing(1)
            .into(),
            None => text(self.tablet.diagnosis().explain())
                .size(theme::CAPTION_SIZE)
                .color(match self.tablet.diagnosis() {
                    Diagnosis::PermissionDenied => theme::WARN,
                    _ => theme::TEXT_MUTE,
                })
                .into(),
        };

        let pen = self.tablet.state();
        let live = if pen.in_proximity {
            format!(
                "{} {:.2}  peak {:.2}\ntilt {:+.2} {:+.2}",
                if pen.eraser { "eraser" } else { "tip   " },
                pen.pressure,
                self.tablet.peak(),
                pen.tilt.x,
                pen.tilt.y
            )
        } else {
            format!("pen away   peak {:.2}", self.tablet.peak())
        };

        let capabilities = devices.first().map(|device| {
            let mut parts = Vec::new();
            if device.has_tilt {
                parts.push("tilt");
            }
            if device.has_eraser {
                parts.push("eraser");
            }
            if parts.is_empty() { "pressure only".to_string() } else { parts.join(", ") }
        });

        column![
            text("PEN").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            status,
            checkbox(self.pressure_enabled)
                .label("Pressure")
                .on_toggle(Message::PressureToggled)
                .text_size(theme::TEXT_SIZE_SMALL),
            text(format!("Curve  {:.2}", self.pressure_curve))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(0.30..=3.00, self.pressure_curve, Message::PressureCurveChanged).step(0.05),
            checkbox(self.tilt_enabled)
                .label("Tilt steers stroke")
                .on_toggle(Message::TiltToggled)
                .text_size(theme::TEXT_SIZE_SMALL),
            text(capabilities.unwrap_or_default())
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            row![
                text(live).size(theme::CAPTION_SIZE).font(theme::MONO).color(theme::TEXT_DIM),
                button(text("reset").size(theme::CAPTION_SIZE))
                    .on_press(Message::ResetPressurePeak),
            ]
            .spacing(theme::S2)
            .align_y(Alignment::Center),
        ]
        .spacing(theme::S2)
        .into()
    }
}
