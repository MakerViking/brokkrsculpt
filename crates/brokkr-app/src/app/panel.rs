// SPDX-License-Identifier: AGPL-3.0-only

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
    button, checkbox, column, container, mouse_area, pick_list, rich_text, row, scrollable, sensor,
    slider, space, span, stack, text, text_editor, text_input,
};
use iced::{Alignment, Element, Length, Padding};

use super::{
    Brokkr, COARSEST_VOXEL_MM, FINEST_VOXEL_MM, PuckAction, SpaceMouseConfig, SpaceMouseSetting,
};
use glam::Vec2;

use crate::app::SizingTarget;
use crate::message::{ConfirmChoice, ExportFormat, Message, PanelSection, TopMenu};
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

        let scene = match (self.menu, self.cube_menu) {
            // Mutually exclusive by construction -- opening either swallows the
            // press that would have opened the other -- but ordered anyway
            // rather than left to whichever `stack!` happened to draw last.
            (Some(at), _) => stack![well, self.overlay(), self.tool_menu(at)],
            (None, Some(cube)) => stack![well, self.overlay(), self.cube_menu_card(cube)],
            (None, None) => stack![well, self.overlay()],
        };

        let body = column![
            self.header(),
            row![
                self.tool_strip(),
                column![
                    self.timeline_strip(),
                    container(scene).width(Length::Fill).height(Length::Fill)
                ]
                .spacing(theme::S2)
                .width(Length::Fill),
                self.tools(),
            ]
            .spacing(theme::S3)
        ]
        .spacing(theme::S3)
        .padding(theme::S3);

        // The unsaved-work prompt outranks an open menu: it is modal, and a
        // menu drawn over it would be reachable while the prompt is not.
        if let Some(pending) = &self.confirm {
            return stack![body, self.confirm_card(pending), self.resize_frame()].into();
        }

        // Same reasoning, one rank down: the bug report is modal too, and it
        // yields to the unsaved prompt because losing work outranks filing a
        // report about it.
        if let Some(draft) = &self.bug_report {
            return stack![body, self.bug_report_card(draft), self.resize_frame()].into();
        }

        // And one rank below that. It only ever appears just after an import,
        // when neither of the two above can be up, but the ordering is stated
        // rather than assumed.
        if let Some(up) = self.orient_prompt {
            return stack![body, self.orient_prompt_card(up), self.resize_frame()].into();
        }

        match self.top_menu {
            // Over everything, and offset down past the bar it drops from.
            Some(which) => stack![
                body,
                container(self.top_menu_panel(which))
                    .padding(Padding { top: 40.0, ..Padding::ZERO }),
                self.resize_frame(),
            ]
            .into(),
            None => stack![body, self.resize_frame()].into(),
        }
    }

    /// The window's resize frame: eight invisible strips around the edges.
    ///
    /// An undecorated window gets **no resize border from the compositor** --
    /// Wayland has no concept of one for a client that has taken over its own
    /// decoration -- so without this the window cannot be resized at all. That
    /// was the state for exactly as long as it took to notice.
    ///
    /// Laid over the whole application as a `stack!` layer. That works because
    /// iced 0.14's stack layers do NOT block what is underneath: only the
    /// strips carry a `mouse_area`, and the large `space` in the middle
    /// handles no events, so every press that is not on an edge falls straight
    /// through to the application.
    fn resize_frame(&self) -> Element<'_, Message> {
        use iced::window::Direction;

        // Thin enough not to steal presses from a maximised panel's edge,
        // thick enough to hit on a 1.5x display, where this is nine physical
        // pixels.
        const EDGE: f32 = 6.0;
        const CORNER: f32 = 14.0;

        let grip = |width: Length, height: Length, direction: Direction| {
            mouse_area(space().width(width).height(height))
                .on_press(Message::ResizeStarted(direction))
                .interaction(match direction {
                    Direction::North | Direction::South => {
                        iced::mouse::Interaction::ResizingVertically
                    }
                    Direction::East | Direction::West => {
                        iced::mouse::Interaction::ResizingHorizontally
                    }
                    Direction::NorthWest | Direction::SouthEast => {
                        iced::mouse::Interaction::ResizingDiagonallyDown
                    }
                    Direction::NorthEast | Direction::SouthWest => {
                        iced::mouse::Interaction::ResizingDiagonallyUp
                    }
                })
        };
        let corner = |direction| grip(Length::Fixed(CORNER), Length::Fixed(CORNER), direction);

        column![
            row![
                corner(Direction::NorthWest),
                grip(Length::Fill, Length::Fixed(EDGE), Direction::North),
                corner(Direction::NorthEast),
            ],
            row![
                grip(Length::Fixed(EDGE), Length::Fill, Direction::West),
                space().width(Length::Fill).height(Length::Fill),
                grip(Length::Fixed(EDGE), Length::Fill, Direction::East),
            ]
            .height(Length::Fill),
            // Aligned to the END, because the row is as tall as the corners
            // and the strip is thinner: left at the default it sits at the TOP
            // of that row and leaves the window's last eight pixels -- the
            // ones anyone actually aims at -- uncovered. That is precisely how
            // this shipped the first time, with every edge resizing except
            // the bottom.
            row![
                corner(Direction::SouthWest),
                grip(Length::Fill, Length::Fixed(EDGE), Direction::South),
                corner(Direction::SouthEast),
            ]
            .align_y(Alignment::End),
        ]
        .into()
    }

    fn header(&self) -> Element<'_, Message> {
        // The wordmark, as ONE text run rather than two widgets side by side.
        // Two `text`s in a row cannot sit flush: a row's spacing applies
        // between them, and setting it to zero leaves the glyphs kerned as two
        // separate runs anyway. `rich_text` colours spans inside a single run,
        // which is what makes it read as BrokkrSCULPT and not Brokkr SCULPT.
        let wordmark = row![
            crate::logo::mark(24.0),
            rich_text::<(), Message, _, _>([
                span("Brokkr").color(theme::TEXT),
                span("SCULPT").color(theme::ACCENT),
            ])
            .size(theme::TEXT_SIZE),
        ]
        .spacing(theme::S2)
        .align_y(Alignment::Center);

        let menus = TopMenu::ALL.into_iter().fold(
            row![].spacing(theme::S2).align_y(Alignment::Center),
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

        // What the sculpt is called, and whether it has unwritten changes. The
        // star matches the window title, which shares `document_name`.
        let title = format!("{}{}", self.document_name(), if self.unsaved { "*" } else { "" });

        // The window controls. The window is undecorated, so if these are not
        // here there is no way to minimise, maximise or close it at all.
        let control = |glyph: &'static str, message: Message| {
            button(text(glyph).size(theme::TEXT_SIZE).align_x(Alignment::Center))
                .padding(Padding { top: 1.0, bottom: 1.0, left: theme::S3, right: theme::S3 })
                .style(theme::section_heading)
                .on_press(message)
        };

        let bar = row![
            wordmark,
            menus,
            text(title).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            // Takes the slack, so the status and the controls sit right and
            // everything else stays left.
            text(self.status.clone())
                .size(theme::CAPTION_SIZE)
                .width(Length::Fill)
                .align_x(Alignment::End)
                .color(
                    if self.status.contains("not exported") || self.status.contains("could not") {
                        theme::ERROR
                    } else {
                        theme::TEXT_MUTE
                    }
                ),
            control("\u{2013}", Message::WindowMinimise),
            control("\u{25a1}", Message::TitleBarDoubleClicked),
            control("\u{00d7}", Message::WindowClose),
        ]
        .spacing(theme::S4)
        .align_y(Alignment::Center);

        container(
            // Dragging the bar moves the window, and a double click maximises
            // it, because this IS the title bar now. `mouse_area` honours event
            // capture, so a press that lands on one of the buttons above is
            // taken by the button and never starts a drag.
            mouse_area(bar)
                .on_press(Message::TitleBarDragged)
                .on_double_click(Message::TitleBarDoubleClicked),
        )
        .padding(Padding { top: theme::S2, bottom: theme::S2, left: theme::S4, right: theme::S3 })
        .width(Length::Fill)
        .style(theme::panel)
        .into()
    }

    /// The modal prompt shown when an action would discard unsaved work.
    ///
    /// Two layers, and both are load bearing. The scrim is full size and
    /// swallows presses, because iced 0.14 has no modal and a bare `stack!`
    /// layer lets clicks through to the sliders and the Reset button
    /// underneath. The card sits on it, centred, using `menu_card` rather than
    /// `overlay_card`: at 0.82 alpha over the translucent debug overlay neither
    /// was readable.
    fn confirm_card(&self, pending: &super::PendingAction) -> Element<'_, Message> {
        // The three styles are all plain `fn`s of the same shape, so the
        // parameter is spelled out rather than left to inference -- otherwise
        // the first call fixes the closure to one specific function's type and
        // the other two stop compiling.
        type ButtonStyle = fn(&iced::Theme, button::Status) -> button::Style;
        let answer = |label: &'static str, choice: ConfirmChoice, style: ButtonStyle| {
            button(text(label).size(theme::TEXT_SIZE))
                .padding(Padding {
                    top: theme::S2,
                    bottom: theme::S2,
                    left: theme::S5,
                    right: theme::S5,
                })
                .style(style)
                .on_press(Message::ConfirmAnswered(choice))
        };

        let card = container(
            column![
                text("Unsaved changes").size(theme::TEXT_SIZE).color(theme::TEXT),
                // Names the document and the action, because "are you sure?" on
                // its own does not say what is about to be lost.
                text(format!(
                    "{} will discard the changes to {}.",
                    pending.describe(),
                    self.document_name()
                ))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
                row![
                    answer("Save", ConfirmChoice::Save, theme::tool_button_active),
                    answer("Discard", ConfirmChoice::Discard, theme::danger_button),
                    answer("Cancel", ConfirmChoice::Cancel, theme::tool_button),
                ]
                .spacing(theme::S3),
            ]
            .spacing(theme::S4),
        )
        .padding(theme::S5)
        .style(theme::menu_card);

        container(card)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(theme::scrim)
            .into()
    }

    /// The bug report dialog.
    ///
    /// It shows the payload rather than describing it. `diagnostics` was
    /// written on the principle that "the user can read exactly what they are
    /// sending before they send it", and uploading it would have quietly ended
    /// that -- so the whole body is on screen, scrollable, assembled by the
    /// same function that sends it. A summary of what is attached is a summary
    /// that goes out of date; this cannot.
    fn bug_report_card<'a>(&'a self, draft: &'a super::BugReport) -> Element<'a, Message> {
        let payload = self.assemble_report();
        let ready = payload.is_some() && !draft.sending;

        let preview: Element<'_, Message> = match &payload {
            Some(report) => scrollable(
                text(report.to_json())
                    .size(theme::CAPTION_SIZE)
                    .font(theme::MONO)
                    .color(theme::TEXT_MUTE),
            )
            .height(Length::Fixed(150.0))
            .into(),
            None => text("describe the problem and the exact payload appears here")
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE)
                .into(),
        };

        type ButtonStyle = fn(&iced::Theme, button::Status) -> button::Style;
        let action = |label: &'static str, message: Option<Message>, style: ButtonStyle| {
            button(text(label).size(theme::TEXT_SIZE))
                .padding(Padding {
                    top: theme::S2,
                    bottom: theme::S2,
                    left: theme::S5,
                    right: theme::S5,
                })
                .style(style)
                .on_press_maybe(message)
        };

        let card = container(
            column![
                text("Report a bug").size(theme::TEXT_SIZE).color(theme::TEXT),
                text("What happened? The first line becomes the title.")
                    .size(theme::TEXT_SIZE_SMALL)
                    .color(theme::TEXT_DIM),
                text_editor(&draft.description)
                    .height(Length::Fixed(110.0))
                    .on_action(Message::BugReportEdited),
                checkbox(draft.with_detail)
                    .label("attach the diagnostics and what this session did")
                    .on_toggle(Message::BugReportDetailToggled)
                    .text_size(theme::CAPTION_SIZE),
                text("Sent anonymously to tinkeratlas.com. Home directories are removed.")
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE),
                preview,
                row![
                    action(
                        if draft.sending { "Sending…" } else { "Send" },
                        ready.then_some(Message::BugReportSubmitted),
                        // Styled by whether it will actually do something. A
                        // button that looks pressable and is not is a worse
                        // failure than one that looks unavailable, because the
                        // user concludes the application ignored them.
                        if ready { theme::tool_button_active } else { theme::tool_button },
                    ),
                    // Always available, and not only as a fallback. A user
                    // without a network, or who would rather read it into an
                    // issue themselves, gets the same bytes.
                    action(
                        "Copy",
                        payload.is_some().then_some(Message::BugReportCopied),
                        theme::tool_button,
                    ),
                    action("Cancel", Some(Message::BugReportDismissed), theme::tool_button),
                ]
                .spacing(theme::S3),
            ]
            .spacing(theme::S3)
            .width(Length::Fixed(520.0)),
        )
        .padding(theme::S5)
        .style(theme::menu_card);

        container(card)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(theme::scrim)
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
                // Recent files, if there are any. Shown by file name, since the
                // full path is usually far wider than the menu -- but a name on
                // its own is ambiguous across directories, so the path goes in
                // the status line when one is chosen.
                let recent =
                    self.recent.paths().iter().fold(column![].spacing(1), |assembled, path| {
                        let label = path
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_else(|| path.display().to_string());
                        assembled.push(
                            button(text(label).size(theme::TEXT_SIZE_SMALL))
                                .width(Length::Fill)
                                .padding(Padding {
                                    top: 3.0,
                                    bottom: 3.0,
                                    left: theme::S3,
                                    right: theme::S3,
                                })
                                .style(theme::section_heading)
                                .on_press(Message::OpenRecent(path.clone())),
                        )
                    });

                let mut items = column![
                    entry("New", Message::NewSculpt),
                    entry("Open…", Message::OpenRequested),
                    entry("Import mesh…", Message::ImportRequested),
                ]
                .spacing(1);
                // Only when there is one to recover. A permanently greyed item
                // would say "this application crashes" every time the menu is
                // opened.
                if self.has_autosave() {
                    items = items.push(entry("Recover autosave", Message::RecoverAutosave));
                }
                if !self.recent.is_empty() {
                    items = items.push(separator()).push(
                        text("RECENT")
                            .size(theme::CAPTION_SIZE)
                            .color(theme::TEXT_MUTE)
                            .width(Length::Fill),
                    );
                    items = items.push(recent);
                    items = items.push(separator());
                }
                items
                    .push(entry("Save", Message::SaveRequested))
                    .push(entry("Save As…", Message::SaveAsRequested))
                    .push(separator())
                    .push(exports)
                    // Below the exports, because it IS an export -- of a file
                    // the user does not have to name or find afterwards.
                    .push(entry("Open in OrcaSlicer", Message::OpenInSlicer))
                    .push(entry("Check the printer", Message::PrinterChecked))
            }
            // No Settings menu: the properties panel already carries every
            // setting there is, and a second surface for the same state would
            // drift from it.
            TopMenu::Help => column![
                text(format!("BrokkrSculpt {}", env!("CARGO_PKG_VERSION")))
                    .size(theme::TEXT_SIZE_SMALL)
                    .color(theme::TEXT),
                text(format!("build {}", super::build_commit()))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE),
                text("AGPL-3.0-only").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                separator(),
                entry("Copy diagnostics", Message::DiagnosticsCopied),
                entry("Report a bug…", Message::BugReportOpened),
                entry("Open the issue tracker", Message::IssueOpened),
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
            container(body).padding(theme::S2).width(Length::Fixed(190.0)).style(theme::menu_card),
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

    /// What a face of the navigation cube should become, at the cursor.
    ///
    /// Fusion's ViewCube right-click menu. The difference worth stating is that
    /// this turns the MODEL, not the camera: a mesh that arrived lying down
    /// would still export lying down if all this did was re-aim the view, and
    /// landing upright on the plate is the whole reason it exists.
    fn cube_menu_card(&self, menu: super::CubeMenu) -> Element<'_, Message> {
        const WIDTH: f32 = 190.0;
        // Roughly how tall it comes out; only used to keep it on screen.
        const HEIGHT: f32 = 250.0;

        let choices = brokkr_core::Facing::ALL.into_iter().fold(
            column![].spacing(theme::S1),
            |assembled, facing| {
                let current = facing == menu.facing;
                assembled.push(
                    button(text(crate::navcube::facing_label(facing)).size(theme::CAPTION_SIZE))
                        .width(Length::Fill)
                        .style(if current { theme::tool_button_active } else { theme::tool_button })
                        // The face is already there, so there is nothing to do
                        // and a button that says so beats one that no-ops.
                        .on_press_maybe((!current).then_some(Message::OrientFace(facing))),
                )
            },
        );

        let body = column![
            text(format!("{} FACE", crate::navcube::facing_label(menu.facing).to_uppercase()))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
            text("move it to").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
            choices,
            text("exact, and reversible by turning back")
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_DIM),
        ]
        .spacing(theme::S3);

        // Kept on screen. The cube lives in the top right corner, so this one
        // opens against the right edge every time and would always hang off.
        let left = menu.at.x.min((self.viewport_size.x - WIDTH).max(0.0)).max(0.0);
        let top = menu.at.y.min((self.viewport_size.y - HEIGHT).max(0.0)).max(0.0);

        container(
            container(body).padding(theme::S4).width(Length::Fixed(WIDTH)).style(theme::menu_card),
        )
        .padding(Padding { left, top, ..Padding::ZERO })
        .into()
    }

    /// Offer to stand an imported model up.
    ///
    /// Modal, and on the same scrim as the unsaved-work prompt, for the reason
    /// documented there: a bare `stack!` layer in iced 0.14 lets clicks through
    /// to the sliders underneath.
    fn orient_prompt_card(&self, up: brokkr_core::Facing) -> Element<'_, Message> {
        type ButtonStyle = fn(&iced::Theme, button::Status) -> button::Style;
        let answer = |label: &'static str, accept: bool, style: ButtonStyle| {
            button(text(label).size(theme::TEXT_SIZE))
                .padding(Padding {
                    top: theme::S2,
                    bottom: theme::S2,
                    left: theme::S5,
                    right: theme::S5,
                })
                .style(style)
                .on_press(Message::OrientPromptAnswered(accept))
        };

        let card = container(
            column![
                text("This model came in lying down").size(theme::TEXT_SIZE).color(theme::TEXT),
                // Says what was observed, not just what it concluded. The
                // reader can then judge the guess instead of trusting it.
                text(format!(
                    "It stands on a flat base, and that base is now facing {}. \
                     Mesh files do not state which axis is up, and the model and \
                     printing worlds disagree, so this is a guess.",
                    crate::navcube::facing_label(up).to_lowercase()
                ))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
                text("Turning it is exact -- nothing is resampled -- and you can turn it back from the cube at any time.")
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_DIM),
                row![
                    answer("Stand it up", true, theme::tool_button_active),
                    answer("Leave it", false, theme::tool_button),
                ]
                .spacing(theme::S3),
            ]
            .spacing(theme::S4),
        )
        .padding(theme::S5)
        .width(Length::Fixed(420.0))
        .style(theme::menu_card);

        container(card)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(theme::scrim)
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
                super::MIN_RADIUS_MM..=self.max_radius(),
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
                    .step(0.05_f32),
                    text(format!("Depth  {:.2}", self.brush.pattern.depth))
                        .size(theme::CAPTION_SIZE)
                        .color(theme::TEXT_DIM),
                    slider(0.0..=1.0, self.brush.pattern.depth, Message::PatternDepthChanged)
                        .step(0.02_f32),
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
            container(body).padding(theme::S4).width(Length::Fixed(WIDTH)).style(theme::menu_card),
        )
        .padding(Padding { left, top, ..Padding::ZERO })
        .into()
    }

    /// The always visible strip of brushes down the left.
    ///
    /// A strip rather than the dropdown this replaced: choosing a brush was two
    /// clicks and a read, for something that should be one glance and one key.
    /// The numbers match the 1..7 shortcuts, so the key and the button are
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
                text("CUT").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                // Armed state is shown in the strip, not just in the status
                // line, because this is the one mode that changes what a left
                // drag does and a cut is not something to discover by accident.
                button(
                    text(if self.cut_armed { "armed" } else { "plane" })
                        .size(theme::TEXT_SIZE_SMALL)
                )
                .width(Length::Fill)
                .style(if self.cut_armed { theme::tool_button_active } else { theme::tool_button })
                .on_press(Message::CutToggled),
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

    /// The timeline strip: stored views, a playhead, and a play button.
    ///
    /// Above the viewport rather than over it. ZBrush floats its timeline on
    /// the canvas, which costs it a strip of the model; there is room here, and
    /// a widget in the layout is a widget iced can hit-test for us rather than
    /// one this file has to hit-test itself.
    ///
    /// Built from spacers rather than drawn, because `canvas` is behind a
    /// feature this build does not enable and turning it on would pull a
    /// tessellator in for four diamonds and a line. The strip's own pixel width
    /// comes back through `sensor`, which is what turns a pointer offset into a
    /// position along it.
    fn timeline_strip(&self) -> Element<'_, Message> {
        /// Height of the clickable strip.
        const STRIP_H: f32 = 18.0;
        /// Width of a key marker, and of the playhead.
        const KEY_W: f32 = 7.0;
        const HEAD_W: f32 = 2.0;

        let width = self.timeline.width();
        // Markers laid out as a row of spacers: each one is placed by the gap
        // in front of it, so the gaps have to be *differences* along the
        // strip. Running left to right and remembering where the last one
        // ended is what keeps them from piling up at the origin.
        let mut placed = 0.0_f32;
        let mut markers = row![];
        for (index, at) in self.timeline.positions() {
            let centre = at * width;
            let gap = (centre - KEY_W * 0.5 - placed).max(0.0);
            let lit =
                self.timeline.hovered == Some(index) || self.timeline.dragged_key() == Some(index);
            markers = markers.push(space().width(Length::Fixed(gap))).push(
                container(space())
                    .width(Length::Fixed(KEY_W))
                    .height(Length::Fixed(STRIP_H))
                    .style(if lit { theme::timeline_key_lit } else { theme::timeline_key }),
            );
            placed += gap + KEY_W;
        }

        let head_x = (self.timeline.playhead * width - HEAD_W * 0.5).max(0.0);
        let playhead = row![
            space().width(Length::Fixed(head_x)),
            container(space())
                .width(Length::Fixed(HEAD_W))
                .height(Length::Fixed(STRIP_H))
                .style(theme::timeline_playhead),
        ];

        let track = container(space())
            .width(Length::Fill)
            .height(Length::Fixed(STRIP_H))
            .style(theme::timeline_track);

        let strip = mouse_area(stack![track, markers, playhead])
            .on_move(|point| Message::TimelineHover(point.x))
            .on_press(Message::TimelinePressed)
            .on_release(Message::TimelineReleased)
            // A drag that leaves the strip has to let go of its key, because
            // `on_release` only fires while the pointer is still over it.
            .on_exit(Message::TimelineLeft)
            .on_right_press(Message::TimelineRemoveKey);

        let play = button(
            text(if self.timeline.playing { "\u{25a0}" } else { "\u{25b6}" })
                .size(theme::CAPTION_SIZE),
        )
        .padding(Padding { top: 1.0, right: 6.0, bottom: 1.0, left: 6.0 })
        .style(if self.timeline.playing { theme::tool_button_active } else { theme::tool_button })
        .on_press_maybe((self.timeline.keys.len() >= 2).then_some(Message::TimelinePlayToggled));

        let hint = match self.timeline.keys.len() {
            0 => "click to store a view".to_string(),
            1 => "1 key — drag to re-time, right click to remove".to_string(),
            many => format!("{many} keys — drag to re-time, right click to remove"),
        };

        row![
            play,
            sensor(strip)
                .on_show(|size| Message::TimelineResized(size.width))
                .on_resize(|size| Message::TimelineResized(size.width)),
            text(hint).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
        ]
        .spacing(theme::S2)
        .align_y(Alignment::Center)
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
                super::MIN_RADIUS_MM..=self.max_radius(),
                self.brush.radius,
                Message::BrushRadiusChanged
            )
            .step(0.05_f32),
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
            .step(0.01_f32),
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
                        .step(0.05_f32)
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
                .step(1.0_f32)
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
                .step(0.05_f32),
                text(format!("Depth  {:.2}", pattern.depth))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_DIM),
                slider(0.0..=1.0, pattern.depth, Message::PatternDepthChanged).step(0.02_f32),
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
            text("Print size").size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_DIM),
            row![
                text_input("longest mm", &self.working_size_field)
                    .on_input(Message::WorkingSizeTyped)
                    .on_submit(Message::WorkingSizeCommitted)
                    .size(theme::TEXT_SIZE_SMALL),
                button(text("set").size(theme::TEXT_SIZE_SMALL))
                    .on_press(Message::WorkingSizeCommitted),
            ]
            .spacing(theme::S2),
            // Scaling is free and changes no voxel, so the only honest thing to
            // show beside it is what the current resolution MEANS at that size.
            // Without this the field reads as a detail control, which it is not.
            text(&self.detail_advice).size(theme::TEXT_SIZE_SMALL).color(theme::TEXT_DIM),
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
            slider(0.30..=3.00, self.pressure_curve, Message::PressureCurveChanged).step(0.05_f32),
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
