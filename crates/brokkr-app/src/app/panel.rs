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

use brokkr_core::{BrushKind, FalloffCurve, MirrorAxis, PatternKind};
use iced::widget::{
    button, checkbox, column, container, pick_list, row, scrollable, slider, stack, text,
    text_input,
};
use iced::{Alignment, Element, Length, Padding};

use super::{
    Brokkr, COARSEST_VOXEL_MM, FINEST_VOXEL_MM, PuckAction, SpaceMouseConfig, SpaceMouseSetting,
};
use glam::Vec2;

use crate::app::SizingTarget;
use crate::message::{ExportFormat, Message, PanelSection, TopMenu};
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

        let scene = match self.menu {
            Some(at) => stack![well, self.overlay(), self.tool_menu(at)],
            None => stack![well, self.overlay()],
        };

        let body = column![
            self.header(),
            row![
                self.tool_strip(),
                container(scene).width(Length::Fill).height(Length::Fill),
                self.tools(),
            ]
            .spacing(theme::S3)
        ]
        .spacing(theme::S3)
        .padding(theme::S3);

        match self.top_menu {
            // Over everything, and offset down past the bar it drops from.
            Some(which) => stack![
                body,
                container(self.top_menu_panel(which))
                    .padding(Padding { top: 40.0, ..Padding::ZERO })
            ]
            .into(),
            None => body.into(),
        }
    }

    fn header(&self) -> Element<'_, Message> {
        let bar = TopMenu::ALL.into_iter().fold(
            row![text("BROKKRSCULPT").size(theme::CAPTION_SIZE).color(theme::ACCENT),]
                .spacing(theme::S4)
                .align_y(Alignment::Center),
            |assembled, menu| {
                assembled.push(
                    button(text(menu.label()).size(theme::TEXT_SIZE_SMALL))
                        .padding(Padding {
                            top: 2.0,
                            bottom: 2.0,
                            left: theme::S3,
                            right: theme::S3,
                        })
                        .style(if self.top_menu == Some(menu) {
                            theme::tool_button_active
                        } else {
                            theme::section_heading
                        })
                        .on_press(Message::TopMenuToggled(menu)),
                )
            },
        );

        // What the sculpt is called, and whether it has a home yet.
        let title = match &self.project_path {
            Some(path) => path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
            None => "untitled".to_string(),
        };

        container(
            row![
                bar,
                text(title).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE).width(Length::Fill),
                text(self.status.clone()).size(theme::CAPTION_SIZE).color(
                    if self.status.contains("not exported") || self.status.contains("could not") {
                        theme::ERROR
                    } else {
                        theme::TEXT_MUTE
                    }
                ),
            ]
            .spacing(theme::S4)
            .align_y(Alignment::Center),
        )
        .padding(Padding { top: theme::S2, bottom: theme::S2, left: theme::S4, right: theme::S4 })
        .width(Length::Fill)
        .style(theme::panel)
        .into()
    }

    /// The panel that drops from an open top bar menu.
    ///
    /// Positioned with the same padding trick the right click menu uses, since
    /// the widget tree is already a `stack!` over the viewport and iced 0.14 has
    /// no popup of its own. That is also why `iced_aw` is not here: it lags iced
    /// releases, and this codebase is pinned to iced 0.14 and wgpu 27 for
    /// reasons that have already cost a session.
    fn top_menu_panel(&self, which: TopMenu) -> Element<'_, Message> {
        let entry = |label: &'static str, message: Message| {
            button(text(label).size(theme::TEXT_SIZE_SMALL))
                .width(Length::Fill)
                .padding(Padding { top: 3.0, bottom: 3.0, left: theme::S3, right: theme::S3 })
                .style(theme::section_heading)
                .on_press(message)
        };
        let separator = || {
            container(text("").size(1))
                .width(Length::Fill)
                .height(Length::Fixed(1.0))
                .style(theme::panel)
        };

        let body = match which {
            TopMenu::File => {
                let exports = ExportFormat::ALL.into_iter().fold(
                    column![].spacing(1),
                    |assembled, format| {
                        assembled.push(entry(
                            match format {
                                ExportFormat::Stl => "Export STL…",
                                ExportFormat::Obj => "Export OBJ…",
                                ExportFormat::ThreeMf => "Export 3MF…",
                            },
                            Message::ExportRequested(format),
                        ))
                    },
                );
                column![
                    entry("New", Message::NewSculpt),
                    entry("Open…", Message::OpenRequested),
                    entry("Save", Message::SaveRequested),
                    entry("Save As…", Message::SaveAsRequested),
                    separator(),
                    exports,
                ]
                .spacing(1)
            }
            // No Import here on purpose. Importing a mesh is a voxeliser, not
            // file plumbing, so the item appears when that exists rather than
            // sitting greyed out promising something.
            //
            // No Settings either: the properties panel already carries every
            // setting there is, and a second surface for the same state would
            // drift from it.
            TopMenu::Help => column![
                text(format!("BrokkrSculpt {}", env!("CARGO_PKG_VERSION")))
                    .size(theme::TEXT_SIZE_SMALL)
                    .color(theme::TEXT),
                text(format!("build {}", super::build_commit()))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE),
                text("AGPL-3.0-or-later").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                separator(),
                entry("Copy diagnostics", Message::DiagnosticsCopied),
                entry("Report a bug…", Message::IssueOpened),
            ]
            .spacing(theme::S1),
        };

        // Under whichever button opened it, roughly. The bar is laid out by
        // iced so the exact x is not knowable here; left aligned under the
        // menus is close enough and cannot fall off the edge.
        let left = match which {
            TopMenu::File => 96.0,
            TopMenu::Help => 148.0,
        };

        container(
            container(body)
                .padding(theme::S2)
                .width(Length::Fixed(190.0))
                .style(theme::overlay_card),
        )
        .padding(Padding { left, top: 0.0, ..Padding::ZERO })
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
                "brush {}   mirror {}   radius {:.2} mm",
                self.effective_brush().kind,
                self.symmetry.label(),
                self.brush.radius
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

    /// The current tool's own controls, at the cursor.
    ///
    /// This is ZBrush's QuickMenu: the documented point of it there is reaching
    /// the brush numbers without moving the cursor to the edge of the screen,
    /// and that is the point of it here. Nomad reaches the same place from the
    /// other direction -- it has no radial menu at all, and its answer is a
    /// bound key plus a drag, which is the `s` and `u` gesture.
    ///
    /// Radius and strength get a typed field as well as a slider, because that
    /// is the thing a slider genuinely cannot do: name an exact 2.5 mm.
    fn tool_menu(&self, at: Vec2) -> Element<'_, Message> {
        const WIDTH: f32 = 210.0;
        // Roughly how tall it comes out. Only used to keep it on screen, so
        // being a little out costs nothing.
        const HEIGHT: f32 = 300.0;

        let numeric = |label: &'static str,
                       which: SizingTarget,
                       range: std::ops::RangeInclusive<f32>,
                       step: f32,
                       on_slide: fn(f32) -> Message| {
            let value = match which {
                SizingTarget::Radius => self.brush.radius,
                SizingTarget::Strength => self.brush.strength,
            };
            column![
                row![
                    text(label)
                        .size(theme::CAPTION_SIZE)
                        .color(theme::TEXT_DIM)
                        .width(Length::FillPortion(2)),
                    text_input("", &self.menu_field_text(which))
                        .on_input(move |text| Message::MenuFieldEdited(which, text))
                        .on_submit(Message::MenuFieldSubmitted)
                        .size(theme::CAPTION_SIZE)
                        .width(Length::FillPortion(1)),
                ]
                .spacing(theme::S2)
                .align_y(Alignment::Center),
                slider(range, value, on_slide).step(step),
            ]
            .spacing(theme::S1)
        };

        let falloff =
            FalloffCurve::ALL.into_iter().fold(row![].spacing(theme::S1), |assembled, curve| {
                assembled.push(
                    button(text(curve.label()).size(theme::CAPTION_SIZE))
                        .width(Length::Fill)
                        .style(if curve == self.brush.falloff {
                            theme::tool_button_active
                        } else {
                            theme::tool_button
                        })
                        .on_press(Message::FalloffChanged(curve)),
                )
            });

        let patterns =
            PatternKind::ALL.into_iter().fold(row![].spacing(theme::S1), |assembled, kind| {
                assembled.push(
                    button(text(kind.short_label()).size(theme::CAPTION_SIZE))
                        .width(Length::Fill)
                        .style(if kind == self.brush.pattern.kind {
                            theme::tool_button_active
                        } else {
                            theme::tool_button
                        })
                        .on_press(Message::PatternChanged(kind)),
                )
            });

        let mut body = column![
            text(self.effective_brush().kind.label().to_uppercase())
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            numeric(
                "Radius mm",
                SizingTarget::Radius,
                super::MIN_RADIUS_MM..=super::MAX_RADIUS_MM,
                0.05,
                Message::BrushRadiusChanged
            ),
            numeric(
                "Strength",
                SizingTarget::Strength,
                super::MIN_STRENGTH..=super::MAX_STRENGTH,
                0.01,
                Message::BrushStrengthChanged
            ),
            text("Falloff").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
            falloff,
            text("Pattern").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
            patterns,
        ]
        .spacing(theme::S3);

        // The pattern's own numbers only mean anything once one is chosen.
        if self.brush.pattern.kind != PatternKind::None {
            let floor = self.voxel_size * brokkr_core::MIN_SCALE_VOXELS;
            body = body.push(
                column![
                    text(format!("Feature  {:.2} mm", self.brush.pattern.scale_mm))
                        .size(theme::CAPTION_SIZE)
                        .color(theme::TEXT_DIM),
                    slider(
                        floor..=brokkr_core::MAX_SCALE_MM,
                        self.brush.pattern.scale_mm.clamp(floor, brokkr_core::MAX_SCALE_MM),
                        Message::PatternScaleChanged
                    )
                    .step(0.05),
                    text(format!("Depth  {:.2}", self.brush.pattern.depth))
                        .size(theme::CAPTION_SIZE)
                        .color(theme::TEXT_DIM),
                    slider(0.0..=1.0, self.brush.pattern.depth, Message::PatternDepthChanged)
                        .step(0.02),
                ]
                .spacing(theme::S1),
            );
        }

        // Kept on screen: opened near the right or bottom edge it would
        // otherwise hang off, and the controls furthest from the cursor are the
        // ones that would go.
        let left = at.x.min((self.viewport_size.x - WIDTH).max(0.0)).max(0.0);
        let top = at.y.min((self.viewport_size.y - HEIGHT).max(0.0)).max(0.0);

        container(
            container(body)
                .padding(theme::S4)
                .width(Length::Fixed(WIDTH))
                .style(theme::overlay_card),
        )
        .padding(Padding { left, top, ..Padding::ZERO })
        .into()
    }

    /// The always visible strip of brushes down the left.
    ///
    /// A strip rather than the dropdown this replaced: choosing a brush was two
    /// clicks and a read, for something that should be one glance and one key.
    /// The numbers match the 1..6 shortcuts, so the key and the button are
    /// visibly the same thing.
    ///
    /// Down the left rather than along the top, which is what SindriCAD's
    /// ribbon would imply. That is a deliberate departure: a strip costs no
    /// viewport height, and it keeps the tools on the side the tablet hand is
    /// already on. The colours are still SindriCAD's tokens, so the two
    /// applications stay a family.
    fn tool_strip(&self) -> Element<'_, Message> {
        let smoothing = self.shift;

        let brushes = BrushKind::ALL.into_iter().enumerate().fold(
            column![].spacing(theme::S2),
            |assembled, (index, kind)| {
                // While shift is held every stroke smooths, so the strip shows
                // Smooth as live. The selection underneath is untouched.
                let live =
                    if smoothing { kind == BrushKind::Smooth } else { kind == self.brush.kind };
                assembled.push(
                    button(
                        column![
                            text(kind.label()).size(theme::TEXT_SIZE_SMALL),
                            text(format!("{}", index + 1)).size(theme::CAPTION_SIZE),
                        ]
                        .spacing(0)
                        .align_x(Alignment::Center),
                    )
                    .width(Length::Fill)
                    .style(if live { theme::tool_button_active } else { theme::tool_button })
                    .on_press(Message::BrushKindChanged(kind)),
                )
            },
        );

        let mirrors =
            MirrorAxis::ALL.into_iter().fold(column![].spacing(theme::S2), |assembled, axis| {
                assembled.push(
                    button(text(axis.label()).size(theme::TEXT_SIZE_SMALL))
                        .width(Length::Fill)
                        .style(if self.symmetry.axis(axis) {
                            theme::tool_button_active
                        } else {
                            theme::tool_button
                        })
                        .on_press(Message::SymmetryAxisToggled(axis)),
                )
            });

        container(
            column![
                text("TOOL").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                brushes,
                text(if smoothing { "shift: smoothing" } else { "hold shift: smooth" })
                    .size(theme::CAPTION_SIZE)
                    .color(if smoothing { theme::ACCENT } else { theme::TEXT_MUTE }),
                text("MIRROR").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                mirrors,
            ]
            .spacing(theme::S3)
            .align_x(Alignment::Center),
        )
        .padding(theme::S3)
        .width(Length::Fixed(76.0))
        .height(Length::Fill)
        .style(theme::tool_strip)
        .into()
    }

    fn tools(&self) -> Element<'_, Message> {
        let invert_hint = if self.brush.kind.is_directional() {
            "ctrl or alt drag removes"
        } else {
            "no opposite: ctrl does nothing"
        };

        let radius = column![
            text(format!("Radius  {:.2} mm", self.brush.radius))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(
                super::MIN_RADIUS_MM..=super::MAX_RADIUS_MM,
                self.brush.radius,
                Message::BrushRadiusChanged
            )
            .step(0.05),
            checkbox(self.dynamic_radius)
                .label("scale with model")
                .on_toggle(Message::DynamicRadiusToggled)
                .text_size(theme::CAPTION_SIZE),
            text("hold s and drag to resize, u for strength")
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
        ]
        .spacing(theme::S2);

        let strength = column![
            text(format!("Strength  {:.2}", self.brush.strength))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            slider(
                super::MIN_STRENGTH..=super::MAX_STRENGTH,
                self.brush.strength,
                Message::BrushStrengthChanged
            )
            .step(0.01),
        ]
        .spacing(theme::S2);

        let falloff = column![
            text("Falloff").size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_DIM),
            FalloffCurve::ALL.into_iter().fold(row![].spacing(theme::S1), |assembled, curve| {
                assembled.push(
                    button(text(curve.label()).size(theme::CAPTION_SIZE))
                        .width(Length::Fill)
                        .style(if curve == self.brush.falloff {
                            theme::tool_button_active
                        } else {
                            theme::tool_button
                        })
                        .on_press(Message::FalloffChanged(curve)),
                )
            }),
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
            scrollable(column![
                text(self.brush.kind.label().to_uppercase())
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE),
                text(invert_hint).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                radius,
                strength,
                falloff,
                self.section(PanelSection::Pattern, || self.pattern_panel()),
                self.section(PanelSection::Pen, || self.pen_panel()),
                self.section(PanelSection::SpaceMouse, || self.spacemouse_panel()),
                text("HISTORY").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                history,
                self.section(PanelSection::Detail, || self.detail_panel()),
                self.section(PanelSection::Export, || self.export_panel()),
                button(text("Reset sphere").size(theme::TEXT_SIZE_SMALL))
                    .on_press(Message::ResetSphere),
                text(
                    "drag: sculpt\nctrl or alt drag: invert\nshift drag: smooth\nright drag: orbit\nshift right drag: pan\nwheel: zoom\n1-6: brush\nx y z: mirror\n[ ]: radius\nctrl z, ctrl shift z: undo, redo"
                )
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S4)
            // Room for the scrollbar, which iced draws over the content rather
            // than beside it. Without this the last button of a row is clipped.
            .padding(Padding { right: theme::S4, ..Padding::ZERO })),
        )
        .padding(theme::PANEL_PADDING)
        .width(Length::Fixed(258.0))
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
                            // Portioned rather than Fill, or the picker eats
                            // the row and clips the flip label off the edge.
                            .width(Length::FillPortion(2)),
                            checkbox(binding.invert)
                                .label("flip")
                                .on_toggle(move |invert| Message::SpaceMouse(
                                    SpaceMouseSetting::Invert(action, invert)
                                ))
                                .text_size(theme::CAPTION_SIZE)
                                .width(Length::FillPortion(1)),
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
            header,
            readout,
            // Where the tuned settings end up, so they can be read back and
            // pasted in as the built in defaults once they are right.
            text(
                spacemouse::Config::path()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "settings cannot be saved: no config directory".into())
            )
            .size(theme::CAPTION_SIZE)
            .color(theme::TEXT_MUTE),
            row![
                button(text("Invert all").size(theme::CAPTION_SIZE))
                    .width(Length::Fill)
                    .on_press(Message::SpaceMouse(SpaceMouseSetting::InvertAll)),
                button(text("Reset").size(theme::CAPTION_SIZE))
                    .width(Length::Fill)
                    .on_press(Message::SpaceMouse(SpaceMouseSetting::Reset)),
            ]
            .spacing(theme::S2),
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
        ]
        .spacing(theme::S2)
        .into()
    }

    /// A collapsible block: a clickable heading, and a body built only when it
    /// is open.
    ///
    /// The body is a closure rather than an `Element` so a closed section
    /// costs nothing to lay out — `view` runs every frame.
    fn section<'a>(
        &'a self,
        which: PanelSection,
        body: impl FnOnce() -> Element<'a, Message>,
    ) -> Element<'a, Message> {
        let open = self.expanded[which as usize];
        let heading = button(
            row![
                text(if open { "−" } else { "+" })
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE)
                    .width(Length::Fixed(10.0)),
                text(which.title()).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S2),
        )
        .width(Length::Fill)
        .padding(0)
        .style(theme::section_heading)
        .on_press(Message::SectionToggled(which));

        if open { column![heading, body()].spacing(theme::S2).into() } else { heading.into() }
    }

    /// The surface pattern, which multiplies into whichever brush is selected.
    ///
    /// One control rather than a brush per pattern: Clay plus Scales and
    /// Inflate plus Hair are both useful, and enumerating the product of two
    /// lists is exactly the complexity the brief asks to avoid.
    fn pattern_panel(&self) -> Element<'_, Message> {
        let pattern = self.brush.pattern;

        let kinds =
            PatternKind::ALL.into_iter().fold(column![].spacing(theme::S1), |assembled, kind| {
                assembled.push(
                    button(text(kind.label()).size(theme::CAPTION_SIZE))
                        .width(Length::Fill)
                        .style(if kind == pattern.kind {
                            theme::tool_button_active
                        } else {
                            theme::tool_button
                        })
                        .on_press(Message::PatternChanged(kind)),
                )
            });

        // The size and depth sliders only mean anything once a pattern is
        // chosen, so they are hidden rather than shown greyed: an empty row is
        // less to read than a disabled one.
        let settings: Element<'_, Message> = if pattern.kind == PatternKind::None {
            text("carve a pattern in with ctrl")
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE)
                .into()
        } else {
            column![
                text(format!("Feature size  {:.2} mm", pattern.scale_mm))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_DIM),
                // The lower bound follows the voxel size: a feature finer
                // than a few voxels cannot be represented, and offering it
                // would only produce a model the exporter then refuses.
                slider(
                    (self.voxel_size * brokkr_core::MIN_SCALE_VOXELS)..=brokkr_core::MAX_SCALE_MM,
                    pattern.scale_mm.clamp(
                        self.voxel_size * brokkr_core::MIN_SCALE_VOXELS,
                        brokkr_core::MAX_SCALE_MM
                    ),
                    Message::PatternScaleChanged
                )
                .step(0.05),
                text(format!("Depth  {:.2}", pattern.depth))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_DIM),
                slider(0.0..=1.0, pattern.depth, Message::PatternDepthChanged).step(0.02),
                text(if pattern.kind.follows_the_stroke() {
                    "combs along the drag"
                } else {
                    "fixed in world space, so strokes reinforce"
                })
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            ]
            .spacing(theme::S1)
            .into()
        };

        column![kinds, settings].spacing(theme::S2).into()
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

        column![buttons, status].spacing(theme::S2).into()
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
