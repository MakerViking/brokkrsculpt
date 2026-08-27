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

use brokkr_core::{BrushKind, FalloffCurve, MaskFilter, MirrorAxis, PatternKind};
use iced::widget::{
    button, checkbox, column, container, mouse_area, opaque, pick_list, rich_text, row, scrollable,
    sensor, slider, space, span, stack, text, text_editor, text_input,
};
use iced::{Alignment, Element, Length, Padding};

use super::{
    Brokkr, COARSEST_VOXEL_MM, FINEST_VOXEL_MM, PuckAction, SpaceMouseConfig, SpaceMouseSetting,
};
use glam::Vec2;

use crate::app::{MaskCard, SizingTarget, Tool};
use crate::icon;
use crate::message::{ConfirmChoice, ExportFormat, MaskGenerator, Message, PanelSection, TopMenu};
use crate::spacemouse::{self, ButtonAction};
use crate::tablet::Diagnosis;
use crate::theme;
use crate::viewport::{ThumbnailCell, Viewport};

/// A folder's body count, as a string that was never formatted.
///
/// **A table and not `format!`, because this is read in a panel row.** `view()`
/// runs at display rate, so formatting a number here is one allocation per
/// folder per frame -- the mistake `detail_advice` is cached to avoid,
/// multiplied by the number of folders. A folder can hold at most
/// [`brokkr_core::MAX_BODIES`] bodies, so the whole range fits a literal table
/// and the lookup is free.
fn count_label(count: usize) -> &'static str {
    const LABELS: [&str; brokkr_core::MAX_BODIES + 1] = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
        "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
        "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46",
        "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61",
        "62", "63", "64",
    ];
    LABELS.get(count).copied().unwrap_or("many")
}

/// Everything one body row draws that is not a field of its own `Node`.
///
/// **A struct, because every one of these is worked out ONCE for the whole
/// list.** Each is a subtree question -- how many bodies are under this row,
/// whether the active body is, whether solo is showing it -- and asked per row
/// each would be a walk per row: `O(n^2)` at display rate, which is the mistake
/// this panel's header exists to forbid. Bundled rather than passed as five
/// positional `bool`s, because five bare booleans at a call site is exactly the
/// shape that gets two of them swapped.
#[derive(Debug, Clone, Copy)]
struct RowFacts {
    /// Bodies in this row's subtree. Zero for a body, which counts nothing but
    /// itself.
    count: usize,
    /// Draw the accent bar: this row is active, or it is a COLLAPSED folder
    /// holding the active body, whose own row is therefore not on screen.
    marked: bool,
    /// Offer the solo circle: this row is active, or it is a folder holding the
    /// active body, collapsed or not.
    soloable: bool,
    /// This row is the one solo is showing, so its circle is the way out.
    soloed: bool,
    /// Solo is off, or this row is inside the soloed subtree. `false` is the
    /// fourth eye state: the row's own eye is on and it is still not drawn.
    in_scope: bool,
    /// A row drag is in flight, so every row tracks the pointer and the one
    /// under it decides where the block would go.
    ///
    /// List-wide rather than per row, as `refusing` below it is, and here
    /// anyway: the alternative is more positional arguments on `body_row`,
    /// which is the shape this struct exists to avoid.
    dragging: bool,
    /// ...and the gap the pointer is over refuses the block, so the cursor says
    /// so. There is no indicator in this state, which is exactly why the cursor
    /// has to carry it. A press that has not moved yet is NOT this: nothing has
    /// been refused, the drag simply has not started.
    refusing: bool,
    /// This row is the dragged block's root, folded away for the duration.
    ///
    /// Otherwise an expanded twelve-child folder makes twelve rows dead with no
    /// local feedback while the explanation appears in a status line at the
    /// other end of the window. **It is a drawing state and not a document
    /// edit**: the `collapsed` bit is in the file, and a drag that dirtied it
    /// would cost an undo press for a gesture that moved nothing.
    folded: bool,
    /// The drop would land INSIDE this row: it is a closed folder and the
    /// pointer is in its middle band.
    drop_into: bool,
}

/// One row of the body list, and how tall it is laid out.
///
/// **Fixed rather than left to the content**, and that is load-bearing rather
/// than tidy: which band of a row the pointer is in is a fraction of the row's
/// height, `mouse_area::on_move` reports a point and not a fraction, and the
/// only height the panel can divide by is one it chose. It is also the number
/// the scrollable is already sized in, so the six-rows-visible promise stops
/// being approximate.
const ROW_H: f32 = 32.0;
const ROW_H_BARE: f32 = 22.0;

/// How far one level of nesting indents a row, and the width of the accent bar
/// down its inside edge.
///
/// At module scope because the drop indicator is drawn at the same indent as
/// the row it would become a sibling of, and it is not a row.
const INDENT_PER_DEPTH: f32 = 12.0;
const MARKER: f32 = 2.0;

/// Where a row's first real column -- the chevron, or the blank standing in for
/// one -- begins, at a given depth.
///
/// **Four terms and not two, because [`Brokkr::body_row`] builds the offset out
/// of three widgets rather than one number**: its leading run is
/// `row![marker(MARKER), space(depth * INDENT_PER_DEPTH), ..].spacing(theme::S2)`,
/// so a spacing falls after the marker AND after the indent, and the second one
/// belongs to no child. Anything drawing at a row's depth without being a row of
/// that shape has to reproduce all four. Dropping the trailing `S2` lands half
/// an indent step short -- 6 px right of the depth above and 6 px left of the
/// depth named -- which is dead centre between the two depths the drop line
/// exists to tell apart.
fn row_content_x(depth: u8) -> f32 {
    MARKER + theme::S2 + f32::from(depth) * INDENT_PER_DEPTH + theme::S2
}

/// The two-pixel line that says where a dragged row would land, indented to the
/// depth it would take.
///
/// **The indent is the whole message.** A line between two rows says where in
/// the ORDER the block goes; only its indent says whose child it becomes, and
/// the two adjacent rows either side of a folder's last child name the same
/// gap at two different depths. A line with no indent would make the one
/// gesture that gets a row back out of a folder indistinguishable from the one
/// that keeps it in -- and a line at HALF a step is worse, because it reads as
/// ambiguous against both neighbours rather than as wrong. This row carries no
/// spacing of its own, so the whole offset is the one space.
fn drop_line<'a>(depth: u8) -> Element<'a, Message> {
    row![
        space().width(Length::Fixed(row_content_x(depth))),
        container(space()).width(Length::Fill).height(Length::Fixed(2.0)).style(theme::drop_line),
    ]
    .into()
}

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

        // Same reasoning, one rank down: a delete asks about one body where the
        // prompt above asks about the whole document, so losing the document
        // outranks it. It cannot in practice be up at the same time as either
        // of the two below, but the ordering is stated rather than assumed.
        if let Some(pending) = &self.pending_delete {
            return stack![body, self.delete_card(pending), self.resize_frame()].into();
        }

        // And the merge prompt, which asks the same question about the same
        // kind of size. It cannot be up at the same time as the delete one --
        // both are raised from a verb press and either returns before the
        // other could be -- but the ordering is stated rather than assumed.
        if let Some(pending) = &self.pending_merge {
            return stack![body, self.merge_card(pending), self.resize_frame()].into();
        }

        // And the split preview, which is the same family again: one row is
        // about to become several and the card is the only place the number is
        // ever said. Below the merge card by the same argument -- neither can
        // be up at once, and the order is stated rather than assumed.
        if let Some(pending) = &self.pending_split {
            return stack![body, self.split_card(pending), self.resize_frame()].into();
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
        //
        // Icons rather than the en dash, white square and multiplication sign
        // these used to be: a Unicode character is resolved through per-platform
        // font fallback, so the same three controls draw differently on another
        // machine. `window_control` carries the hover state in its background
        // because an icon cannot take the button's text colour.
        let control = |name: icon::IconName, style: theme::ButtonStyle, message: Message| {
            button(icon::icon(name, theme::ICON_CHROME, theme::TEXT_DIM))
                .padding(Padding { top: theme::S1, bottom: theme::S1, left: 7.0, right: 7.0 })
                .style(style)
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
            control(icon::IconName::Minimise, theme::window_control, Message::WindowMinimise),
            control(
                icon::IconName::Maximise,
                theme::window_control,
                Message::TitleBarDoubleClicked,
            ),
            control(icon::IconName::Close, theme::window_control_close, Message::WindowClose),
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
    /// Two layers, and both are load bearing. `modal_layer` is full size and
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

        modal_layer(card)
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

        modal_layer(card)
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
                    // "and replace" because that is what it does: the mesh
                    // becomes the whole document, and every body already in it
                    // goes. Named "Import mesh…" it read as an add, which is
                    // the one thing it is not -- and the day importing a mesh
                    // AS a new body ships, both entries can sit here and say
                    // which is which.
                    entry("Import and replace…", Message::ImportRequested),
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

    /// The stats readout over the viewport, and the button that opens it.
    ///
    /// Collapsed to a single icon by default. Seven lines of monospace sat
    /// permanently across the top-left of the model, which is the corner a
    /// sculpt is as likely to occupy as any other, and none of it is wanted
    /// while actually sculpting.
    ///
    /// **The mesh-pool warning is deliberately outside the collapse and shows
    /// in both states.** It does not report a statistic, it reports that part
    /// of the model is not on the screen — and this project has already shipped
    /// that failure twice as something the user had to go looking for. It is
    /// also not a flicker: `overflowed` clears only when the pool actually gets
    /// space back -- a whole-model rebuild, or a body being deleted -- so once
    /// it is up it stays up until the condition is really gone.
    fn overlay(&self) -> Element<'_, Message> {
        let pool = self.shared.stats();

        // `tool_toggle` rather than a style and a colour picked side by side:
        // the icon is a canvas drawn at a fixed colour and cannot follow the
        // button's foreground the way text does. See `icon.rs`'s header.
        let (style, ink) = theme::tool_toggle(self.stats_open);
        let toggle = button(icon::icon(icon::IconName::Info, theme::ICON_CHROME, ink))
            .padding(theme::S2)
            .style(style)
            .on_press(Message::StatsToggled);

        let mut stacked = column![toggle].spacing(theme::S2);

        if self.stats_open {
            stacked = stacked.push(
                container(self.stats_readout(pool)).padding(theme::S3).style(theme::overlay_card),
            );
        }

        if pool.overflowed > 0 {
            // What frees space is named, because the obvious remedy does not
            // work: hiding a body is a draw-time skip and it keeps every slice
            // it holds. A user reading "the pool is full" and reaching for the
            // eye would see the count stay exactly where it is.
            let warning = format!(
                "MESH POOL FULL: {} bricks missing from the view\ndelete a body or resample \
                 coarser -- hiding one frees nothing",
                pool.overflowed
            );
            stacked = stacked.push(
                container(
                    text(warning)
                        .size(theme::TEXT_SIZE_SMALL)
                        .font(theme::MONO)
                        .color(theme::ERROR),
                )
                .padding(theme::S3)
                .style(theme::overlay_card),
            );
        }

        if let Some(card) = &self.mask_card {
            stacked = stacked.push(self.mask_card(card));
        }

        container(stacked).padding(theme::S4).into()
    }

    /// The standing mask card, over the viewport, whenever anything is masked.
    ///
    /// See [`MaskCard`] for why it exists and why it is
    /// unconditional. This is the layout half; the two things worth reading
    /// here are the `opaque` and the absence of any formatting.
    ///
    /// # It MUST be `opaque`, and "a `stack!` child captures" is not true
    ///
    /// Capture only happens where a child actually captures. `Button` does, so
    /// the two verb buttons below are safe; everything else on this card is a
    /// `container` and a `text`, and neither captures anything.
    /// `Stack::update` levitates the cursor for the layers below only when the
    /// upper child's `mouse_interaction` is non-`None`
    /// (`iced_widget-0.14.2/src/stack.rs:266-273`), and both of those return
    /// `Interaction::None`. So without this wrapper a press on the card's
    /// padding falls straight through to the shader and starts a stroke on the
    /// model behind it — this project's own recorded gotcha, walked back into.
    /// `opaque` captures presses only, only inside its bounds, which is legal
    /// under the rule the viewport is written to; see [`modal_layer`].
    ///
    /// # The two reflex verbs, and why only when the card names this body
    ///
    /// Invert and Clear are here so that the two things a user reaches for
    /// mid-sculpt are one click away without a chord or a menu. They act on the
    /// ACTIVE body, so they are shown only when the percentage above them is
    /// about the active body: a card that is up purely because something is
    /// masked OFF screen would otherwise offer to clear a mask that is not
    /// there. See [`MaskCard::names_the_active_body`].
    ///
    /// # Nothing here computes or formats
    ///
    /// Both strings arrive built. `view()` runs at display rate, and a
    /// percentage worked out in here would be a pass over the mask every frame.
    fn mask_card<'a>(&self, card: &'a MaskCard) -> Element<'a, Message> {
        let mut body =
            column![text(card.headline.as_str()).size(theme::TEXT_SIZE_SMALL).color(theme::MASK)]
                .spacing(theme::S1);
        if !card.off_screen.is_empty() {
            body = body.push(
                text(card.off_screen.as_str()).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            );
        }
        if !card.elsewhere.is_empty() {
            body = body.push(
                text(card.elsewhere.as_str()).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            );
        }
        if card.names_the_active_body() {
            let verb = |label: &'static str, message: Message| {
                button(text(label).size(theme::CAPTION_SIZE))
                    .padding(theme::S1)
                    .style(theme::tool_button)
                    .on_press(message)
            };
            body = body.push(
                row![verb("invert", Message::MaskInverted), verb("clear", Message::MaskCleared),]
                    .spacing(theme::S2),
            );
        }
        opaque(container(body).padding(theme::S3).style(theme::overlay_card))
    }

    /// What the stats readout says: frame rate, frame time, triangles, bricks,
    /// resident memory and what history is holding.
    fn stats_readout(&self, pool: brokkr_gpu::PoolStats) -> Element<'_, Message> {
        let frame_ms = self.perf.average_frame_ms();
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };

        let volume_mb = self.doc_stats.resident_bytes as f64 / (1024.0 * 1024.0);
        let pool_mb =
            (pool.vertices as f64 * 24.0 + pool.triangles as f64 * 12.0) / (1024.0 * 1024.0);
        let history_mb = self.history_stats.bytes as f64 / (1024.0 * 1024.0);

        let lines = vec![
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
                "{} triangles   {} drawn / {} culled / {} hidden bricks",
                pool.triangles, pool.drawn, pool.culled, pool.hidden
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
                self.doc_stats.dense_bricks, self.doc_stats.uniform_bricks
            ),
            format!(
                "history {} undo / {} redo   {history_mb:.1} MB of {} MB{}{}",
                self.history_stats.undo_entries,
                self.history_stats.redo_entries,
                self.history_stats.budget_bytes / (1024 * 1024),
                // Only when there is one, because a deleted body is the
                // uncommon case and a permanent "+ 0 MB" would read as noise --
                // but while history IS holding one, the number it holds is not
                // in the figure to its left and has its own allowance.
                if self.history_stats.reclaim_bytes > 0 {
                    format!(
                        "   + {:.1} MB of {} MB deleted",
                        self.history_stats.reclaim_bytes as f64 / (1024.0 * 1024.0),
                        self.history_stats.reclaim_budget_bytes / (1024 * 1024)
                    )
                } else {
                    String::new()
                },
                if self.history_stats.dropped > 0 {
                    format!("   {} dropped", self.history_stats.dropped)
                } else {
                    String::new()
                }
            ),
        ];

        lines
            .into_iter()
            .fold(column![].spacing(2), |stacked, line| {
                stacked.push(text(line).size(theme::TEXT_SIZE_SMALL).font(theme::MONO))
            })
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
    /// Modal, and on the same `modal_layer` as the unsaved-work prompt, for
    /// the reason documented there: a bare `stack!` layer in iced 0.14 lets
    /// clicks through to the sliders underneath.
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

        modal_layer(card)
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
        // being a little out costs nothing. Sized for the taller of the two
        // blocks at the foot, which is the mask's: three verb buttons, two rows
        // of filters, an amount slider, a tint slider, a switch and a pattern
        // readout all come out well above the pattern block they replace.
        // **Sized once for all seven verbs**, which is why Grow and Shrink ship
        // with Blur and Sharpen rather than after them -- and once again for the
        // three generators, their two sliders and the masked split, which is
        // where the block's growth stops. At a 768-high window this card now
        // reaches most of the way down it; anything further has to take
        // something out rather than add to the number.
        const HEIGHT: f32 = 660.0;

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

        let patterns = || {
            PatternKind::ALL.into_iter().fold(row![].spacing(theme::S1), |assembled, kind| {
                let (style, ink) = theme::tool_toggle(kind == self.brush.pattern.kind);
                assembled.push(
                    button(icon::icon(icon::IconName::for_pattern(kind), theme::ICON_CHROME, ink))
                        .width(Length::Fill)
                        .style(style)
                        .on_press(Message::PatternChanged(kind)),
                )
            })
        };

        let mut body = column![
            text(self.live_tool_label()).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
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
                super::MIN_STRENGTH..=self.max_strength(),
                0.01,
                Message::BrushStrengthChanged
            ),
            text("Falloff").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
            falloff,
        ]
        .spacing(theme::S3);

        if self.tool == Tool::Mask {
            // The pattern's own row is NOT built here, and the mask block
            // carries a readout of it instead. See `mask_block`.
            body = body.push(self.mask_block());
        } else {
            body = body
                .push(text("Pattern").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM))
                .push(patterns());

            // The pattern's own numbers only mean anything once one is chosen.
            if self.brush.pattern.kind != PatternKind::None {
                let floor = self.doc.voxel_size() * brokkr_core::MIN_SCALE_VOXELS;
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

    /// The mask's own controls, in place of the PATTERN block.
    ///
    /// # The pattern is still live here, and that is why it is still named here
    ///
    /// Masking keeps the pattern deliberately -- it is ZBrush's alpha masking
    /// and it costs nothing, because a pattern is already a multiplier in
    /// 0..1 and so is a mask. But this block REPLACES the pattern's controls,
    /// so without the readout at the foot the pattern would go on multiplying
    /// into every mask stamp with nothing on screen saying which one. That is
    /// worse than switching it off, which is why the answer is a line of text
    /// and not a `PatternKind::None`. **Do not "fix" the missing pattern row by
    /// disabling the pattern.**
    ///
    /// # What the toggle does and does not reach
    ///
    /// The tint, and only the tint. The standing mask card is built in
    /// `refresh_mask_card` from the document alone and consults neither of
    /// these, so switching the tint off cannot produce a masked body with
    /// nothing on screen to say so. Both are session state and neither dirties
    /// the document.
    ///
    /// # Seven verbs in a two-column grid, and one slider
    ///
    /// Clear, Invert and Mask All happen on the press: all three are O(1) and
    /// there is nothing to drag. Blur, Sharpen, Grow and Shrink choose which
    /// filter the ONE absolute amount slider below them drives. Two columns
    /// rather than one row of four, and not only for the plan's sake: the four
    /// across the 178 px this block gets is 44 px a button, and "Sharpen" does
    /// not fit in it -- the falloff row above gets away with the same layout
    /// because its longest label is a character shorter.
    ///
    /// One shared slider rather than four labelled ones, because `const HEIGHT`
    /// is sized once for the whole block and four of them is 160 px of it.
    ///
    /// **The amount slider sits at zero between gestures and that is not a
    /// bug.** The filter is absolute, so the amount belongs to the drag and not
    /// to the document; leaving it at 1.0 would re-blur an already-blurred mask
    /// the instant the next grab touched it.
    fn mask_block(&self) -> Element<'_, Message> {
        let verb = |label: &'static str, message: Message| {
            button(text(label).size(theme::CAPTION_SIZE))
                .width(Length::Fill)
                .style(theme::tool_button)
                .on_press(message)
        };

        let filter = |which: MaskFilter| {
            button(text(which.label()).size(theme::CAPTION_SIZE))
                .width(Length::Fill)
                .style(if which == self.mask_filter {
                    theme::tool_button_active
                } else {
                    theme::tool_button
                })
                .on_press(Message::MaskFilterChosen(which))
        };

        // A generator is a press and never a drag: unlike the four filters it
        // has no amount, so there is nothing to hold open and nothing to commit
        // on release. What it reads instead are the two sliders below it, which
        // is why they sit under the row rather than one per button.
        let generator = |which: MaskGenerator| {
            button(text(which.label()).size(theme::CAPTION_SIZE))
                .width(Length::Fill)
                .style(theme::tool_button)
                .on_press(Message::MaskGenerated(which))
        };

        let mut block = column![
            text("Mask").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
            row![verb("Clear", Message::MaskCleared), verb("Invert", Message::MaskInverted)]
                .spacing(theme::S1),
            verb("Mask all", Message::MaskAllApplied),
            row![filter(MaskFilter::Blur), filter(MaskFilter::Sharpen)].spacing(theme::S1),
            row![filter(MaskFilter::Grow), filter(MaskFilter::Shrink)].spacing(theme::S1),
            text(format!("Amount  {:.2}", self.mask_amount))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_DIM),
            // `on_release` is what makes the whole drag ONE undo entry: every
            // step of it re-applies from the same snapshot, and the release is
            // where that snapshot goes into history.
            slider(0.0..=1.0, self.mask_amount, Message::MaskAmountChanged)
                .step(0.05_f32)
                .on_release(Message::MaskAmountReleased),
            // Not disabled when the tint is off: the slider is where a user who
            // turned it off finds out what they turned off, and moving it is
            // the obvious way to ask for it back.
            text(format!("Tint  {:.2}", self.mask_tint))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_DIM),
            slider(0.0..=1.0, self.mask_tint, Message::MaskTintChanged).step(0.05_f32),
            checkbox(self.show_mask)
                .label("Show mask")
                .on_toggle(|_| Message::ShowMaskToggled)
                .text_size(theme::CAPTION_SIZE),
            text("Make one from the shape").size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
            MaskGenerator::ALL.into_iter().fold(row![].spacing(theme::S1), |assembled, which| {
                assembled.push(generator(which))
            }),
            // In MILLIMETRES, which is the whole of what makes this better than
            // ZBrush's: its cavity masking is resolution-relative and this is
            // not, so the number means the same thing after a resample.
            text(format!("Feature  {:.2} mm", self.mask_feature_mm))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_DIM),
            slider(super::MASK_FEATURE_RANGE_MM, self.mask_feature_mm, Message::MaskFeatureChanged)
                .step(0.05_f32),
            // In VOXELS with the millimetres beside it, and the other way round
            // would be wrong: the ceiling is twice the narrow band, which is a
            // property of the field and not of the model, so a millimetre
            // slider's maximum would move under a resample.
            text(format!(
                "Thinner than  {} vx ({:.2} mm)",
                self.mask_thickness_voxels,
                self.mask_thickness_voxels as f32 * self.doc.voxel_size()
            ))
            .size(theme::CAPTION_SIZE)
            .color(theme::TEXT_DIM),
            slider(
                1.0..=brokkr_core::MAX_THICKNESS_VOXELS as f32,
                self.mask_thickness_voxels as f32,
                Message::MaskThicknessChanged
            )
            .step(1.0_f32),
            // Here rather than in the bodies panel's verb row, which is icons:
            // this is a mask operation, it reads with the verbs it belongs to,
            // and it costs no new icon and no new icon test.
            verb("Split off the mask", Message::BodySplitMasked),
        ]
        .spacing(theme::S1);

        if self.brush.pattern.kind != PatternKind::None {
            block = block.push(
                text(format!("Pattern: {}", self.brush.pattern.kind.label()))
                    .size(theme::CAPTION_SIZE)
                    .color(theme::TEXT_MUTE),
            );
        }

        block.into()
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
    ///
    /// # The vertical budget, and what the mask button cost to fit
    ///
    /// This column has to fit a 768-high window without scrolling — it is a
    /// plain container with no `scrollable` and no clip, so anything past the
    /// bottom is simply not there. See [`theme::ICON_TOOL`]. Adding the mask as
    /// an eleventh button put it over, and two things were subtracted to pay
    /// for it: the two-line "hold shift: smooth" hint, which the live Smooth
    /// highlight beside it and the cheat sheet at the foot of the properties
    /// panel already say twice over, and the separate CUT and MASK headings,
    /// now one MODE heading over the two buttons that share that meaning.
    ///
    /// Arithmetic over the constants, and it is **arithmetic rather than a
    /// measurement**: the mask button adds about 63 px (an 18 px icon, an 11 px
    /// label, 10 px of button padding and an 8 px gap), the hint returned 34 and
    /// the second heading 21, so the strip grew by roughly 8 px net over the
    /// version that fitted. One screenshot at 768 settles it properly and none
    /// has been taken.
    fn tool_strip(&self) -> Element<'_, Message> {
        let smoothing = self.shift;

        let brushes = BrushKind::ALL.into_iter().enumerate().fold(
            column![].spacing(theme::S2),
            |assembled, (index, kind)| {
                // While shift is held every stroke smooths, so the strip shows
                // Smooth as live. The selection underneath is untouched.
                let live =
                    if smoothing { kind == BrushKind::Smooth } else { kind == self.brush.kind };
                let (style, ink) = theme::tool_toggle(live);
                assembled.push(
                    button(
                        // The word stays under the icon, and that is the whole
                        // mitigation for seven tools that all push a surface
                        // around. SindriCAD's icons carry their labels in the
                        // ribbon for the same reason; an icon-only strip would
                        // bet the tool picker on telling clay from draw at
                        // eighteen pixels.
                        column![
                            icon::icon(icon::IconName::for_brush(kind), theme::ICON_TOOL, ink),
                            text(kind.label()).size(theme::TEXT_SIZE_SMALL),
                            text(format!("{}", index + 1)).size(theme::CAPTION_SIZE),
                        ]
                        // `align_x` alone is not centring. A column defaults to
                        // `Shrink`, so it hugs its widest child and then sits at
                        // the LEFT of the button -- and `align_x` only centres
                        // the children within that shrunken box. The result is
                        // three elements that agree with each other and with
                        // nothing else, and it reads as centred only on
                        // whichever button holds the longest word. `Fill` is
                        // what makes the box the button.
                        .width(Length::Fill)
                        .spacing(0)
                        .align_x(Alignment::Center),
                    )
                    .width(Length::Fill)
                    .style(style)
                    .on_press(Message::BrushKindChanged(kind)),
                )
            },
        );

        // The mirror toggles keep their letters and get no icon. X, Y and Z are
        // plain ASCII, so they carry none of the font-fallback risk that made
        // the rest of this worth doing -- and an axis is a thing a letter names
        // better than a picture can. A mirror-plane glyph turned per axis reads
        // for X and Y and then has to say "depth" in two dimensions for Z,
        // which is the point at which a set starts inventing puzzles.
        let mirrors =
            MirrorAxis::ALL.into_iter().fold(column![].spacing(theme::S2), |assembled, axis| {
                assembled.push(
                    button(
                        text(axis.label())
                            .size(theme::TEXT_SIZE_SMALL)
                            // Same trap as the brush buttons: a `text` is
                            // `Shrink`, so a single letter in a `Fill` button
                            // sits against its left edge.
                            .width(Length::Fill)
                            .align_x(Alignment::Center),
                    )
                    .width(Length::Fill)
                    .style(if self.symmetry.axis(axis) {
                        theme::tool_button_active
                    } else {
                        theme::tool_button
                    })
                    .on_press(Message::SymmetryAxisToggled(axis)),
                )
            });

        let (cut_style, cut_ink) = theme::tool_toggle(self.tool == Tool::Cut);
        let (mask_style, mask_ink) = theme::tool_toggle(self.tool == Tool::Mask);
        let (move_style, move_ink) = theme::tool_toggle(self.tool == Tool::Transform);

        container(
            column![
                text("TOOL").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                brushes,
                text("MIRROR").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                mirrors,
                text("MODE").size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                // The live state is shown in the strip, not just in the status
                // line, because **these are the two modes that change what a
                // left drag does, and only one of the two is destructive.**
                // (That sentence used to read "this is the one mode", here and
                // in `handoff.md`; masking made both halves false.) That
                // asymmetry is also why Escape clears the cut and not the mask:
                // a cut is a pending destructive thing, a mask is expensive
                // work with no undo entry until the stroke lands.
                //
                // The words stay for exactly that reason: "armed" versus
                // "plane" is a state, and an icon says which tool this is, not
                // whether it is about to go off.
                button(
                    column![
                        icon::icon(icon::IconName::CutPlane, theme::ICON_TOOL, cut_ink),
                        text(if self.tool == Tool::Cut { "armed" } else { "plane" })
                            .size(theme::TEXT_SIZE_SMALL),
                    ]
                    .width(Length::Fill)
                    .spacing(0)
                    .align_x(Alignment::Center)
                )
                .width(Length::Fill)
                .style(cut_style)
                .on_press(Message::ToolChanged(Tool::Cut)),
                button(
                    column![
                        icon::icon(icon::IconName::Mask, theme::ICON_TOOL, mask_ink),
                        // The same live substitution the brush buttons make for
                        // Smooth: while shift is held a mask drag blurs, so the
                        // button says what the next drag will actually do.
                        text(match (self.tool == Tool::Mask, smoothing) {
                            (true, true) => "blur",
                            (true, false) => "on",
                            (false, _) => "mask",
                        })
                        .size(theme::TEXT_SIZE_SMALL),
                    ]
                    .width(Length::Fill)
                    .spacing(0)
                    .align_x(Alignment::Center)
                )
                .width(Length::Fill)
                .style(mask_style)
                .on_press(Message::ToolChanged(Tool::Mask)),
                // The third mode that changes what a left drag does. The word
                // under it says whether the next release will cost the surface
                // anything, which is the one thing about this tool a user
                // cannot see by looking at the model.
                button(
                    column![
                        icon::icon(icon::IconName::Gizmo, theme::ICON_TOOL, move_ink),
                        text(match self.tool == Tool::Transform {
                            true if smoothing => "free",
                            true => "snap",
                            false => "move",
                        })
                        .size(theme::TEXT_SIZE_SMALL),
                    ]
                    .width(Length::Fill)
                    .spacing(0)
                    .align_x(Alignment::Center)
                )
                .width(Length::Fill)
                .style(move_style)
                .on_press(Message::ToolChanged(Tool::Transform)),
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

        let (play_style, play_ink) = theme::tool_toggle(self.timeline.playing);
        let play = button(icon::icon(
            if self.timeline.playing { icon::IconName::Stop } else { icon::IconName::Play },
            theme::ICON_CHROME,
            play_ink,
        ))
        .padding(Padding { top: 1.0, right: 6.0, bottom: 1.0, left: 6.0 })
        .style(play_style)
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
        // In mask mode the modifier means something else entirely, and it means
        // it for every brush -- the mask has no "no opposite" case, because
        // protection always has one. Reading `brush.kind.is_directional()` here
        // while masking would tell three of the seven brushes that ctrl does
        // nothing, when it is the unmask gesture.
        let invert_hint = match (self.tool, self.brush.kind.is_directional()) {
            (Tool::Mask, _) => "ctrl or alt drag unmasks",
            (_, true) => "ctrl or alt drag removes",
            (_, false) => "no opposite: ctrl does nothing",
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
                super::MIN_STRENGTH..=self.max_strength(),
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
                .style(theme::tool_button)
                .on_press_maybe(self.history.can_undo().then_some(Message::Undo)),
            button(text("Redo").size(theme::TEXT_SIZE_SMALL))
                .style(theme::tool_button)
                .on_press_maybe(self.history.can_redo().then_some(Message::Redo)),
        ]
        .spacing(theme::S2);

        container(
            scrollable(column![
                self.section(PanelSection::Bodies, || self.bodies_panel()),
                text(match self.tool {
                    Tool::Mask => "MASK".to_string(),
                    _ => self.brush.kind.label().to_uppercase(),
                })
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
                    .style(theme::tool_button)
                    .on_press(Message::ResetSphere),
                // `1-7`, not `1-6`: `BrushKind::ALL` has been seven since Move
                // landed and this line was wrong before masking touched it.
                //
                // The ctrl line is here because the half-space mask has no
                // button anywhere: it rides the cut's own drag, which is what
                // makes it free, and a gesture with no control on screen is a
                // gesture nobody finds.
                text(
                    "drag: sculpt\nctrl or alt drag: invert\nshift drag: smooth\nright drag: orbit\nshift right drag: pan\nwheel: zoom\n1-7: brush\nm: mask\nctrl + cut drag: mask that half\nhold h: show the mask\nx y z: mirror\n[ ]: radius\nctrl z, ctrl shift z: undo, redo"
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
                    .style(theme::tool_button)
                    .on_press(Message::SpaceMouse(SpaceMouseSetting::InvertAll)),
                button(text("Reset").size(theme::CAPTION_SIZE))
                    .width(Length::Fill)
                    .style(theme::tool_button)
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
    ///
    /// The solo badge rides here rather than inside `bodies_panel` for one
    /// reason: it has to survive the section being collapsed. Soloing a body,
    /// folding the Bodies section away and forgetting is the same predicament as
    /// soloing and scrolling away, which is what the indicator exists for.
    fn section<'a>(
        &'a self,
        which: PanelSection,
        body: impl FnOnce() -> Element<'a, Message>,
    ) -> Element<'a, Message> {
        let open = self.expanded[which as usize];
        let heading = button(
            row![
                // The caret keeps `TEXT_MUTE` through hover while the title
                // beside it lifts to `TEXT`, because `section_heading` signals
                // hover through `text_color` and an icon cannot follow that.
                // Tolerable here, unlike on the window controls, precisely
                // because there IS a word next to it still doing the job.
                icon::icon(
                    if open { icon::IconName::CaretDown } else { icon::IconName::CaretRight },
                    theme::ICON_INLINE,
                    theme::TEXT_MUTE,
                ),
                text(which.title()).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
            ]
            .align_y(Alignment::Center)
            .spacing(theme::S2),
        )
        .width(Length::Fill)
        .padding(0)
        .style(theme::section_heading)
        .on_press(Message::SectionToggled(which));

        let heading: Element<'a, Message> = match self.solo_badge(which) {
            // Beside the heading and not inside it. Nesting would work — a
            // `Button` forwards to its content and returns early on
            // `is_event_captured`, the same mechanism the row's eye rides — but
            // it would also make a press on the word "SOLO" fold the section
            // away, which is not what a status indicator should do.
            Some(badge) => {
                row![heading, badge].align_y(Alignment::Center).spacing(theme::S2).into()
            }
            None => heading.into(),
        };

        if open { column![heading, body()].spacing(theme::S2).into() } else { heading }
    }

    /// `SOLO: <name>  [exit]`, drawn beside the Bodies heading while the mode is
    /// on.
    ///
    /// **Persistent, and outside the scrollable.** Without it, soloing a body
    /// deep in a twenty-node tree and scrolling away leaves one body on screen,
    /// nineteen missing, and no indicator anywhere: every eye in view still
    /// reads "visible", because solo is a mask over those bits and never a write
    /// to them. The MESH POOL FULL banner is the existing pattern for a mode the
    /// viewport cannot explain by itself, and this doubles as the exit.
    ///
    /// Nothing here is computed or formatted: the name is borrowed straight out
    /// of the document, and the whole badge is skipped by an `Option` when solo
    /// is off.
    fn solo_badge(&self, which: PanelSection) -> Option<Element<'_, Message>> {
        if which != PanelSection::Bodies {
            return None;
        }
        let node = self.doc.node(self.solo?)?;
        Some(
            row![
                text("SOLO").size(theme::CAPTION_SIZE).color(theme::ACCENT),
                text(node.name.as_str()).size(theme::CAPTION_SIZE).color(theme::TEXT_DIM),
                button(icon::icon(icon::IconName::Close, theme::ICON_INLINE, theme::TEXT_MUTE))
                    .padding(Padding { top: 1.0, bottom: 1.0, left: theme::S1, right: theme::S1 })
                    .style(theme::section_heading)
                    .on_press(Message::SoloExited),
            ]
            .align_y(Alignment::Center)
            .spacing(theme::S1)
            .into(),
        )
    }

    // --- the body list -------------------------------------------------------

    /// Every body in the sculpt, the verbs that act on one, and the switch that
    /// turns the pictures off.
    ///
    /// # Nothing in a row may compute anything
    ///
    /// `view()` runs at display rate off `window::frames()`, so a row that
    /// formats a triangle count, measures a bound or diffs staleness repeats the
    /// mistake `detail_advice` is cached to avoid, multiplied by the number of
    /// bodies. Every value a row reads here is either already in a field
    /// (`shown`, `active`) or is a borrowed `&str`; nothing allocates a string
    /// and nothing walks a brick map.
    ///
    /// # A plain `column`, deliberately not a `keyed_column`
    ///
    /// `keyed::Column::draw` iterates every child with no viewport test
    /// (`iced_widget-0.14.2/src/keyed/column.rs:355-364`) where plain
    /// `Column::draw` filters on `layout.bounds().intersects(viewport)`
    /// (`column.rs:328`), and a `scrollable` passes its translated visible
    /// bounds down as that viewport. At 128 rows the keyed one pushes 128
    /// heap-boxed primitives every frame to show six. The widget state it exists
    /// to preserve does not survive the case it would be chosen for either: its
    /// diff splices on a length change and otherwise zips by index
    /// (`keyed/column.rs:228-244`), so state does not follow a row moved by a
    /// same-length reorder.
    ///
    /// # Walked tree-shaped, and every per-row number worked out ONCE
    ///
    /// The two things a folder row has to show that a body row does not -- how
    /// many bodies are inside it, and whether the active body is one of them --
    /// are both subtree questions, and asked per row they would each be a walk
    /// per row: `O(n^2)` at display rate. Both are answered here in one pass
    /// apiece, before a single row is built. Nothing in a row computes
    /// anything.
    ///
    /// The counts need one `Vec` a frame, which is the deliberate trade: one
    /// small allocation for the whole list beats one per row, and this function
    /// is already building an `Element` per visible row.
    fn bodies_panel(&self) -> Element<'_, Message> {
        /// Six rows with pictures, eight without — the same 190-odd pixels
        /// either way, so turning the pictures off buys rows rather than space.
        const VISIBLE_ROWS: f32 = 6.0;
        const VISIBLE_ROWS_BARE: f32 = 8.0;

        // Bodies per folder subtree, by node position. One backward pass with a
        // fixed accumulator; see `Document::subtree_body_counts`.
        let mut counts = Vec::new();
        self.doc.subtree_body_counts(&mut counts);
        // Which folders have the active body somewhere beneath them. A forward
        // pass over the ancestor chain, which is the same shape
        // `resolve_visibility` uses and for the same reason: preorder means the
        // chain is a fixed-size array and never a search.
        let holds_active = self.folders_holding_the_active_body();
        // The rows solo is showing, as ONE range. A subtree is a contiguous
        // preorder run, so the scope test per row is an integer comparison --
        // the same property that makes the resolver's own solo test one
        // comparison, and the reason no row has to ask the document anything.
        let scope = self.solo.and_then(|id| self.doc.subtree_of(id));
        // The row being dragged, if one is, resolved ONCE for the whole list --
        // one scan of at most `MAX_NODES` rows, the same shape as the two passes
        // above and for the same reason. Everything else about the drag was
        // worked out in `update`; see `RowDrag`.
        //
        // `folding` is only the row that is really being DRAGGED, which is not
        // the same as the row that was pressed: a press is not a drag until the
        // pointer moves, and folding on the press alone would snap a folder
        // shut on every plain click on its row.
        let folding = self
            .row_drag
            .filter(|drag| drag.under_way())
            .and_then(|drag| self.doc.index_of(drag.id));
        let (target, into, refusing) = match self.row_drag {
            Some(drag) => (drag.target, drag.into, drag.said.is_some()),
            None => (None, None, false),
        };

        let mut rows = column![].spacing(1);
        // The depth below which rows belong to a collapsed subtree and are not
        // drawn. A collapsed folder changes only what is DRAWN -- never what a
        // command does -- which is the ZBrush failure this design is written
        // against.
        let mut skip_below: Option<u8> = None;
        // Whether the drop line has been placed yet. It goes in front of the
        // first DRAWN row at or after the insertion index, which is not the
        // same as the row at that index: the gap may be behind a folded
        // subtree, and the line still has to appear somewhere the eye can see
        // it.
        let mut line_placed = false;
        for (index, node) in self.doc.nodes().iter().enumerate() {
            if let Some(depth) = skip_below {
                if node.depth() > depth {
                    continue;
                }
                skip_below = None;
            }
            let folded = folding == Some(index);
            if node.collapsed || folded {
                skip_below = Some(node.depth());
            }
            // The line, in front of the first drawn row at or after the gap.
            // One integer comparison per row, which is all a row may cost.
            if let Some(target) = target
                && into.is_none()
                && !line_placed
                && index >= target.at
            {
                rows = rows.push(drop_line(target.depth));
                line_placed = true;
            }
            let holds = holds_active.get(index).copied().unwrap_or(false);
            let facts = RowFacts {
                count: counts.get(index).copied().unwrap_or(0),
                marked: node.id == self.doc.active() || (node.collapsed && holds),
                // The active row, or any folder holding it -- collapsed or not,
                // which is where this parts company with `marked`. Solo is
                // enterable from nowhere else, and that is what keeps "the
                // active body is displayed-visible" true on the way IN with no
                // extra machinery: a click on another row selects it first.
                soloable: node.id == self.doc.active() || holds,
                soloed: self.solo == Some(node.id),
                in_scope: scope.as_ref().is_none_or(|range| range.contains(&index)),
                dragging: self.row_drag.is_some(),
                refusing,
                folded,
                drop_into: into == Some(index),
            };
            rows = rows.push(self.body_row(index, node, facts));
        }
        // A gap past the last drawn row: the block goes to the very bottom.
        if let Some(target) = target
            && into.is_none()
            && !line_placed
        {
            rows = rows.push(drop_line(target.depth));
        }

        let row_h = self.row_height();
        let visible_rows = if self.thumbnails { VISIBLE_ROWS } else { VISIBLE_ROWS_BARE };

        let verbs = row![
            button(icon::icon(icon::IconName::Plus, theme::ICON_INLINE, theme::TEXT_DIM))
                .padding(Padding { top: 2.0, bottom: 2.0, left: theme::S3, right: theme::S3 })
                .style(if self.adding { theme::tool_button_active } else { theme::tool_button })
                .on_press(Message::PrimitiveMenuToggled),
            button(icon::icon(icon::IconName::Trash, theme::ICON_INLINE, theme::TEXT_DIM))
                .padding(Padding { top: 2.0, bottom: 2.0, left: theme::S3, right: theme::S3 })
                .style(theme::tool_button)
                // Greyed rather than refusing on press: the last body cannot go,
                // and a button that looks pressable and is not teaches the user
                // that the application ignores them.
                .on_press_maybe((self.doc.body_count() > 1).then_some(Message::BodyDeleted)),
            // Always pressable, and REFUSING on press with the count in the
            // status line -- never greyed. Three reasons, in order of weight:
            //
            // * it is what the plan asks for by name, and what `+` beside it
            //   already does. All four of duplicate's refusals then read the
            //   same way, where greying could only ever cover two of them: the
            //   memory and mesh-pool ceilings CANNOT be greyed for, because
            //   reading them costs a walk of a brick map and a lock on the pool
            //   stats and this row is built every frame;
            // * greying here would be INVISIBLE. `theme::tool_button` returns
            //   `PANEL_2` for `Status::Disabled` exactly as it does for
            //   `Status::Active`, and the icon's colour is passed in at this
            //   call site independently of button status, so iced would draw a
            //   dead button identical to a live one. A button that looks
            //   pressable and does nothing AND says nothing is the worst of the
            //   three options, not the safe one;
            // * a greyed button says only "no". The refusal says which ceiling
            //   and how far over it, which is the only form in which any of
            //   this is useful to someone with sixty-four bodies open.
            button(icon::icon(icon::IconName::Copy, theme::ICON_INLINE, theme::TEXT_DIM))
                .padding(Padding { top: 2.0, bottom: 2.0, left: theme::S3, right: theme::S3 })
                .style(theme::tool_button)
                .on_press(Message::BodyDuplicated),
            // Always pressable and refusing on press, for the three reasons
            // above -- and here the first of them is sharper still. Whether a
            // merge is legal is "is the next row a body at this depth", which
            // is cheap; whether it needs asking first is a walk of the source's
            // brick map, which is not, and this row is built every frame. Two
            // of merge's outcomes could be greyed for and the third could not,
            // so all three refuse the same way and each says which it is.
            button(icon::icon(icon::IconName::MergeDown, theme::ICON_INLINE, theme::TEXT_DIM))
                .padding(Padding { top: 2.0, bottom: 2.0, left: theme::S3, right: theme::S3 })
                .style(theme::tool_button)
                .on_press(Message::BodyMergedDown),
            // Always pressable and refusing on press, for the same three
            // reasons. Whether a split would do anything cannot be known
            // without walking every voxel of the body, which is the one thing
            // a row built every frame may never do -- so there is nothing to
            // grey on, and the answer arrives as a card or as a status line.
            button(icon::icon(icon::IconName::Split, theme::ICON_INLINE, theme::TEXT_DIM))
                .padding(Padding { top: 2.0, bottom: 2.0, left: theme::S3, right: theme::S3 })
                .style(theme::tool_button)
                .on_press(Message::BodySplit),
        ]
        .spacing(theme::S2);

        let mut stacked = column![
            // The pictures are session state and must not dirty the document.
            // Photoshop's Panel Options offers None beside three sizes and the
            // standard advice for a heavy document is None, so the switch is
            // faithful to the reference rather than a retreat from it.
            checkbox(self.thumbnails)
                .label("Thumbnails")
                .on_toggle(|_| Message::ThumbnailsToggled)
                .text_size(theme::CAPTION_SIZE),
            scrollable(rows).height(Length::Fixed(row_h * visible_rows)),
            verbs,
        ]
        .spacing(theme::S2);

        if self.adding {
            // As TEXT and not icons. Three glyphs that are not drawn, and a cube
            // at 11 px is a smudge -- the words are shorter to read than the
            // pictures would be to learn.
            let choices = brokkr_core::PrimitiveKind::ALL.into_iter().fold(
                row![].spacing(theme::S1),
                |assembled, kind| {
                    assembled.push(
                        button(text(kind.label()).size(theme::CAPTION_SIZE))
                            .width(Length::Fill)
                            .style(theme::tool_button)
                            .on_press(Message::PrimitiveAdded(kind)),
                    )
                },
            );
            stacked = stacked.push(choices);
        }

        stacked.into()
    }

    /// One row of the body list, body or folder.
    ///
    /// A body: `[marker][indent][disclosure spacer][thumbnail][name, Fill][eye]`.
    /// A folder: `[marker][indent][chevron][name, Fill][count][trash][eye]`.
    ///
    /// `count` and `marked` are worked out ONCE for the whole list by
    /// [`Brokkr::bodies_panel`], never here: both are subtree questions, and a
    /// subtree walk per row is `O(n^2)` at display rate. So is every other field
    /// of [`RowFacts`], for the same reason.
    ///
    /// # Clicking the eye must never change the active body, and it costs no
    /// code at all
    ///
    /// `MouseArea::update` (`iced_widget-0.14.2/src/mouse_area.rs:219-244`)
    /// forwards to its content first and then returns early on
    /// `shell.is_event_captured()`, and `Button` captures `ButtonPressed` when
    /// it has an `on_press` and the cursor is over it. So the eye's button eats
    /// the press and the row's `mouse_area` never sees it. **Do not "tidy" this
    /// into a row of buttons or an `on_press` on the eye's parent** -- this is
    /// Photoshop's quiet load-bearing property, the one that lets you audit
    /// visibility across twelve bodies without losing the body you were
    /// sculpting. The chevron and the folder's trash ride on the same
    /// mechanism, which is why neither needs to know the row exists.
    ///
    /// The thumbnail is INERT for the same mechanism read the other way: it is a
    /// styled `container`, which captures nothing, so the row press passes
    /// through it. It must not become a `Button` -- it is the largest target in
    /// the row, and Photoshop puts a different action on a ctrl-clicked layer
    /// thumbnail that users hit constantly.
    ///
    /// # Every row's eye means the same thing, including the active row's
    ///
    /// ZBrush's does not: clicking the selected row's eye there hides
    /// everything, and a click a few pixels lower hides just that one. That is
    /// documented behaviour and exactly the presentation this brief exists to
    /// escape.
    ///
    /// # Four eye states, two signals each
    ///
    /// `ICON_INLINE` is 11 px, where an eye with a hairline through it is a
    /// dot, so the glyph never carries a state on its own:
    ///
    /// | state | glyph | name |
    /// |---|---|---|
    /// | shown | `Eye` in `TEXT` | `TEXT` |
    /// | hidden by itself | `EyeOff` in `TEXT_MUTE` | `TEXT_MUTE` |
    /// | hidden by an ancestor folder | `Eye` in `TEXT_MUTE` | `TEXT_MUTE` |
    /// | out of the solo scope | `Eye` in `TEXT_MUTE`, row background dimmed | `TEXT_MUTE` |
    ///
    /// The last two fall out of the first two rather than being written: the
    /// glyph reads this row's OWN eye and the colour reads the RESOLVED answer,
    /// so a row whose own eye is on inside a hidden folder — or outside the solo
    /// scope — shows a bright-shaped eye in a muted colour. That is Photoshop's
    /// grey eye, and it is what says "your bit is still on; something above you
    /// is not". The fourth adds the recessed background, because it is the one
    /// state where the eye is also REFUSED, and a control that will not do what
    /// it looks like needs more than a shade of grey to say so.
    fn body_row<'a>(
        &'a self,
        index: usize,
        node: &'a brokkr_core::Node,
        facts: RowFacts,
    ) -> Element<'a, Message> {
        const THUMBNAIL: f32 = 28.0;
        /// The folder chevron's column. A body row spends it on empty space so
        /// that the two line up.
        const DISCLOSURE: f32 = 11.0;

        let active = node.id == self.doc.active();
        // Already resolved, once, by `publish_visibility`. A row must never
        // resolve it for itself: that is the walk that makes twelve rows twelve
        // walks, every frame.
        let drawn = self.shown.get(index).copied().unwrap_or(true);
        let ink = if drawn { theme::TEXT } else { theme::TEXT_MUTE };

        let eye = if node.visible { icon::IconName::Eye } else { icon::IconName::EyeOff };
        let eye_ink = if drawn { theme::TEXT } else { theme::TEXT_MUTE };

        let mut line = row![
            // **The marker is `marked` and not `active`**, and the difference
            // is the whole point of it on a folder: a collapsed folder holding
            // the active body draws the bar, because the row that would
            // otherwise carry it is not on screen. Without that, selecting a
            // body inside a folder and closing it leaves nothing anywhere
            // saying where the brush will land.
            container(space())
                .width(Length::Fixed(MARKER))
                .height(Length::Fill)
                .style(if facts.marked { theme::body_row_marker } else { theme::body_row },),
            // Everything before this point, plus the spacing either side of it,
            // is what [`row_content_x`] reproduces for the drop line. Changing
            // the marker, this indent or the spacing below moves the line too,
            // and `the_drop_line_starts_where_the_row_it_names_starts` is what
            // says so.
            space().width(Length::Fixed(f32::from(node.depth()) * INDENT_PER_DEPTH)),
        ]
        .spacing(theme::S2)
        .align_y(Alignment::Center);

        if node.is_body() {
            line = line.push(space().width(Length::Fixed(DISCLOSURE)));
            if self.thumbnails {
                // One hash lookup, which is all a row may do -- the same
                // shape as `self.shown.get(index)` above. The cell was
                // allocated in `update`, by `publish_visibility`; `view` only
                // ever reads it, so nothing here can queue a render.
                //
                // The picture sits inside the styled well rather than
                // replacing it, with a pixel of padding, so the well's border
                // still frames the cell and a body with no cell to draw --
                // past the atlas's sixty-four layers -- degrades to the flat
                // placeholder this was before there were pictures.
                let inner: Element<'_, Message> = match self.thumbs.cell(node.id) {
                    Some(cell) => iced::widget::shader(ThumbnailCell::new(cell))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .into(),
                    None => space().into(),
                };
                line = line.push(
                    container(inner)
                        .width(Length::Fixed(THUMBNAIL))
                        .height(Length::Fixed(THUMBNAIL))
                        .padding(1)
                        .style(theme::body_thumbnail),
                );
            }
        } else {
            // Folded reads as collapsed, because for the duration of the drag
            // it IS: the rows under it are not drawn, and a chevron still
            // pointing down over nothing would be the panel lying about its own
            // state.
            let chevron = if node.collapsed || facts.folded {
                icon::IconName::CaretRight
            } else {
                icon::IconName::CaretDown
            };
            line = line.push(
                button(icon::icon(chevron, theme::ICON_INLINE, theme::TEXT_MUTE))
                    .padding(0)
                    .style(theme::section_heading)
                    .on_press(Message::FolderCollapseToggled(node.id)),
            );
        }

        // At most one row is being renamed, so this is one `NodeId` compare and
        // no allocation -- the field's text is already owned by the
        // application and is handed to the widget by reference.
        let typed = match &self.renaming {
            Some((id, typed)) if *id == node.id => Some(typed),
            _ => None,
        };
        let mut line = match typed {
            Some(typed) => line.push(
                text_input("name", typed)
                    .id(super::RENAME_FIELD)
                    .on_input(Message::BodyRenameEdited)
                    .on_submit(Message::BodyRenameSubmitted)
                    .padding(Padding { top: 1.0, bottom: 1.0, left: theme::S1, right: theme::S1 })
                    .size(theme::TEXT_SIZE_SMALL)
                    .width(Length::Fill),
            ),
            None => line.push(
                text(node.name.as_str())
                    .size(theme::TEXT_SIZE_SMALL)
                    .color(ink)
                    .width(Length::Fill),
            ),
        };

        if !node.is_body() {
            line = line
                .push(
                    text(count_label(facts.count))
                        .size(theme::CAPTION_SIZE)
                        .color(theme::TEXT_MUTE),
                )
                // A folder's own delete, and NOT a mode of the verb row's. The
                // verb row names the active body, which is always a body, so a
                // collapsed folder can never swallow a body delete -- that is
                // the ZBrush hour-losing failure made unrepresentable rather
                // than remembered.
                .push(
                    button(icon::icon(icon::IconName::Trash, theme::ICON_INLINE, theme::TEXT_MUTE))
                        .padding(Padding {
                            top: 2.0,
                            bottom: 2.0,
                            left: theme::S1,
                            right: theme::S1,
                        })
                        .style(theme::section_heading)
                        .on_press(Message::FolderDeleted(node.id)),
                );
        }

        // The solo circle, on the active row and on the folders holding it and
        // nowhere else. Sixty-four rows each offering to solo is sixty-four
        // targets for one mode; one target beside the row that already means
        // "this is what I am working on" is the same idea the verb row is built
        // on. It rides the eye's capture mechanism, so pressing it does not
        // also select or rename the row.
        if facts.soloable {
            let ring = if facts.soloed { theme::ACCENT } else { theme::TEXT_MUTE };
            line = line.push(
                button(icon::icon(icon::IconName::Solo, theme::ICON_INLINE, ring))
                    .padding(Padding { top: 2.0, bottom: 2.0, left: theme::S1, right: theme::S1 })
                    .style(theme::section_heading)
                    // A toggle, so the way in is also a way out and the button
                    // never lies about what it will do. Escape and the header
                    // indicator are the other two.
                    .on_press(if facts.soloed {
                        Message::SoloExited
                    } else {
                        Message::SoloEntered(node.id)
                    }),
            );
        }

        let line = line.push(
            button(icon::icon(eye, theme::ICON_INLINE, eye_ink))
                .padding(Padding { top: 2.0, bottom: 2.0, left: theme::S1, right: theme::S1 })
                .style(theme::section_heading)
                .on_press(Message::BodyVisibilityToggled(node.id)),
        );

        // Active outranks out-of-scope, and it has to: selecting a row solo is
        // not showing is allowed -- a view mode never vetoes a structural
        // operation -- and losing the selection marker in that state would leave
        // nothing anywhere saying where the brush will land.
        //
        // The drop target outranks BOTH, and only while the button is down: it
        // is the answer to "what happens if I let go now", and it is gone the
        // instant the pointer moves off. The row it lands on is very often the
        // active one, which is why it is an outline and not just the tint.
        let background = match (facts.drop_into, active, facts.in_scope) {
            (true, ..) => theme::body_row_drop_into,
            (false, true, _) => theme::body_row_active,
            (false, false, false) => theme::body_row_out_of_scope,
            (false, false, true) => theme::body_row,
        };

        let row_h = self.row_height();
        let area = mouse_area(
            container(line)
                .padding(Padding { top: 0.0, bottom: 0.0, left: 0.0, right: theme::S1 })
                .width(Length::Fill)
                // Fixed, so the fraction the drag reads is a fraction of a
                // height this panel chose rather than of whatever the content
                // came out at. See [`ROW_H`].
                .center_y(Length::Fixed(row_h))
                .style(background),
        )
        .on_press(Message::BodySelected(node.id))
        // A DOUBLE click renames, and it is on the ROW rather than on the name.
        // `MouseArea::update` publishes `on_press` first and then the double
        // click (`mouse_area.rs:376-392`), so this selects the row AND starts
        // the rename, which is what Photoshop does. A single click cannot be
        // the gesture: the name is `Length::Fill` and therefore the row's main
        // target, so renaming on a single click there is renaming on every
        // attempt to select.
        //
        // The two rows-within-the-row that capture the press keep their
        // meaning for free: the eye's `button` captures it, so a double click
        // on the eye toggles twice and does not rename, and once a rename is in
        // flight the `text_input` captures it, so clicking inside the field
        // does not re-select the row underneath.
        //
        // *Deliberate shortcut:* this makes a double click on the THUMBNAIL
        // rename too, where Photoshop opens layer styles. Revisit when the
        // thumbnail gains an action of its own -- increment 15 draws it, and
        // does not give it one.
        .on_double_click(Message::BodyRenameBegan(node.id));

        if !facts.dragging {
            // **No drag, no handler.** `on_move` takes a boxed closure, so
            // attaching one unconditionally is a heap allocation per visible
            // row per frame for a gesture that is not happening -- and iced
            // would publish a message on every idle pointer move over the list.
            return area.into();
        }
        // `on_enter` and `on_exit` are deliberately NOT used: the indicator is
        // decided by the FRACTION of the way down a row the pointer is, not by
        // crossing a row's edge, so the events that matter are the ones inside
        // a row rather than the ones between two.
        let area = area
            .on_move(move |point| Message::BodyRowDragged {
                over: index,
                fraction: (point.y / row_h).clamp(0.0, 1.0),
            })
            .interaction(if facts.refusing {
                // The third indicator, and the one every shipped tree drag gets
                // wrong: an illegal gap draws NOTHING, so without the cursor
                // saying so the user reads "no line yet" as "keep going".
                iced::mouse::Interaction::NotAllowed
            } else {
                iced::mouse::Interaction::Grabbing
            });
        area.into()
    }

    /// How tall one row of the body list is laid out.
    ///
    /// One answer for the panel and for the drag, because the drag divides a
    /// pointer offset by it: two copies of this number is an indicator that
    /// drifts from the pointer down a long list.
    fn row_height(&self) -> f32 {
        if self.thumbnails { ROW_H } else { ROW_H_BARE }
    }

    /// Which rows have the active body somewhere beneath them, by node
    /// position.
    ///
    /// One forward pass over the ancestor chain, which preorder makes a
    /// fixed-size array rather than a search -- the same shape
    /// `resolve_visibility` uses, and it allocates nothing: the array is
    /// [`brokkr_core::MAX_NODES`] bools on the stack.
    ///
    /// Only a COLLAPSED folder draws the marker off this, but the answer is
    /// computed for every row because knowing it per row is the expensive
    /// version: a subtree walk per row is `O(n^2)` at display rate.
    fn folders_holding_the_active_body(&self) -> [bool; brokkr_core::MAX_NODES] {
        let mut holds = [false; brokkr_core::MAX_NODES];
        let Some(active_at) = self.doc.index_of(self.doc.active()) else {
            return holds;
        };
        let mut chain = [0usize; brokkr_core::MAX_DEPTH as usize];
        for (index, node) in self.doc.nodes().iter().enumerate().take(active_at + 1) {
            let depth = usize::from(node.depth()).min(chain.len() - 1);
            chain[depth] = index;
            if index == active_at {
                for ancestor in chain.iter().take(depth) {
                    if let Some(slot) = holds.get_mut(*ancestor) {
                        *slot = true;
                    }
                }
            }
        }
        holds
    }

    /// "This body is large enough that undo may not be able to hold it."
    ///
    /// The size is in the question, because 512 MB is the reclaim allowance and
    /// a delete at or over it is exactly the delete that can be evicted before
    /// it can be undone. Modal on the same `modal_layer` as the unsaved-work
    /// prompt, for the reason documented there.
    fn delete_card<'a>(&'a self, pending: &'a super::PendingDelete) -> Element<'a, Message> {
        type ButtonStyle = fn(&iced::Theme, button::Status) -> button::Style;
        let answer = |label: &'static str, message: Message, style: ButtonStyle| {
            button(text(label).size(theme::TEXT_SIZE))
                .padding(Padding {
                    top: theme::S2,
                    bottom: theme::S2,
                    left: theme::S5,
                    right: theme::S5,
                })
                .style(style)
                .on_press(message)
        };

        let card = container(
            column![
                text(format!("Delete {}?", pending.name)).size(theme::TEXT_SIZE).color(theme::TEXT),
                text(format!(
                    "It holds {} and {:.0} MB. Undo keeps a deleted body only while it fits a \
                     {} MB allowance, so this may not be recoverable.",
                    if pending.bodies == 1 {
                        "one body".to_string()
                    } else {
                        format!("{} bodies", pending.bodies)
                    },
                    pending.bytes as f64 / (1024.0 * 1024.0),
                    brokkr_core::DEFAULT_RECLAIM_BUDGET / (1024 * 1024),
                ))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
                row![
                    answer("Delete", Message::BodyDeleteConfirmed, theme::danger_button),
                    answer("Cancel", Message::BodyDeleteCancelled, theme::tool_button),
                ]
                .spacing(theme::S3),
            ]
            .spacing(theme::S4)
            .width(Length::Fixed(420.0)),
        )
        .padding(theme::S5)
        .style(theme::menu_card);

        modal_layer(card)
    }

    /// "This merge records more than undo can be relied on to keep."
    ///
    /// Both halves of the size are in the question, because they are two
    /// different things a user can reason about: the bricks the merge changes
    /// are the overlap between the two bodies, and the consumed body is the
    /// whole of the one being merged down. A merge of two bodies that barely
    /// touch is nearly all the second number; one that fully overlaps is both,
    /// which is how it reaches six times the stroke budget.
    fn merge_card<'a>(&'a self, pending: &'a super::PendingMerge) -> Element<'a, Message> {
        type ButtonStyle = fn(&iced::Theme, button::Status) -> button::Style;
        let answer = |label: &'static str, message: Message, style: ButtonStyle| {
            button(text(label).size(theme::TEXT_SIZE))
                .padding(Padding {
                    top: theme::S2,
                    bottom: theme::S2,
                    left: theme::S5,
                    right: theme::S5,
                })
                .style(style)
                .on_press(message)
        };
        let mb = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);

        let card = container(
            column![
                text(format!("Merge {} into {}?", pending.source_name, pending.target_name))
                    .size(theme::TEXT_SIZE)
                    .color(theme::TEXT),
                text(format!(
                    "Undoing it means keeping {:.0} MB: {:.0} MB of the bricks it changes and \
                     {:.0} MB for {}, which the merge consumes. History keeps one gesture only \
                     while it fits a {} MB budget, so this may not be recoverable.",
                    mb(pending.bytes),
                    mb(pending.stroke_bytes),
                    mb(pending.reclaim_bytes),
                    pending.source_name,
                    brokkr_core::DEFAULT_RECLAIM_BUDGET / (1024 * 1024),
                ))
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
                row![
                    answer("Merge", Message::BodyMergeConfirmed, theme::danger_button),
                    answer("Cancel", Message::BodyMergeCancelled, theme::tool_button),
                ]
                .spacing(theme::S3),
            ]
            .spacing(theme::S4)
            .width(Length::Fixed(420.0)),
        )
        .padding(theme::S5)
        .style(theme::menu_card);

        modal_layer(card)
    }

    /// "This body is N loose parts. Here is what pressing Split would leave."
    ///
    /// **The count is the whole card**, and it is why this one is raised
    /// unconditionally where the delete and merge cards are raised on a size.
    /// Nothing in the panel can tell a user whether a body is one shell or four
    /// thousand, so a split that just ran would be an operation whose result is
    /// its own first piece of information -- on a document that now has sixty
    /// new rows in it.
    ///
    /// The second line is the sweep, and it names WHICH rule did it. "The rest
    /// are specks" and "your document has room for nine more rows" send a
    /// reader to completely different remedies, and only the second one can be
    /// answered by tidying up and pressing again.
    fn split_card<'a>(&'a self, pending: &'a super::PendingSplit) -> Element<'a, Message> {
        type ButtonStyle = fn(&iced::Theme, button::Status) -> button::Style;
        let answer = |label: &'static str, message: Message, style: ButtonStyle| {
            button(text(label).size(theme::TEXT_SIZE))
                .padding(Padding {
                    top: theme::S2,
                    bottom: theme::S2,
                    left: theme::S5,
                    right: theme::S5,
                })
                .style(style)
                .on_press(message)
        };

        let plan = &pending.plan;
        let detail = if plan.has_fragments() {
            let why = if plan.capped {
                format!(
                    "the document has room for only {} more bodies out of {}",
                    plan.kept,
                    brokkr_core::MAX_BODIES,
                )
            } else {
                format!("they are all under {:.0} mm³", brokkr_core::SIGNIFICANT_MM3)
            };
            format!(
                "The largest {} become bodies of their own, down to {:.1} mm³. The other {} go \
                 into one \u{201c}{} fragments\u{201d} body, because {why}.",
                plan.kept,
                plan.smallest_kept_mm3(),
                plan.swept(),
                pending.name,
            )
        } else {
            format!(
                "Each one becomes a body of its own, the largest {:.1} mm³ and the smallest \
                 {:.1} mm³.",
                plan.parts.first().map_or(0.0, |part| part.mm3),
                plan.smallest_kept_mm3(),
            )
        };

        let card = container(
            column![
                text(format!("{} is {} loose parts.", pending.name, plan.found()))
                    .size(theme::TEXT_SIZE)
                    .color(theme::TEXT),
                text(detail).size(theme::CAPTION_SIZE).color(theme::TEXT_MUTE),
                text(if plan.bodies() == 1 {
                    "One body replaces it, and one undo puts it back.".to_string()
                } else {
                    format!("{} bodies replace it, and one undo puts it back.", plan.bodies())
                })
                .size(theme::CAPTION_SIZE)
                .color(theme::TEXT_MUTE),
                row![
                    answer("Split", Message::BodySplitConfirmed, theme::danger_button),
                    answer("Cancel", Message::BodySplitCancelled, theme::tool_button),
                ]
                .spacing(theme::S3),
            ]
            .spacing(theme::S4)
            .width(Length::Fixed(420.0)),
        )
        .padding(theme::S5)
        .style(theme::menu_card);

        modal_layer(card)
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
                let (style, ink) = theme::tool_toggle(kind == pattern.kind);
                assembled.push(
                    button(
                        // A container, because a row lays its children out from
                        // its left edge and a button does not centre `Shrink`
                        // content -- so neither `row` nor `button` alone can put
                        // this pair in the middle of a full-width row.
                        container(
                            row![
                                icon::icon(
                                    icon::IconName::for_pattern(kind),
                                    theme::ICON_CHROME,
                                    ink
                                ),
                                text(kind.label()).size(theme::CAPTION_SIZE),
                            ]
                            .spacing(theme::S2)
                            .align_y(Alignment::Center),
                        )
                        .width(Length::Fill)
                        .align_x(Alignment::Center),
                    )
                    .width(Length::Fill)
                    .style(style)
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
                    (self.doc.voxel_size() * brokkr_core::MIN_SCALE_VOXELS)
                        ..=brokkr_core::MAX_SCALE_MM,
                    pattern.scale_mm.clamp(
                        self.doc.voxel_size() * brokkr_core::MIN_SCALE_VOXELS,
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
        let finer = self.doc.voxel_size() / 2.0;
        let coarser = self.doc.voxel_size() * 2.0;

        column![
            text(format!("Voxel  {:.3} mm", self.doc.voxel_size()))
                .size(theme::TEXT_SIZE_SMALL)
                .color(theme::TEXT_DIM),
            row![
                button(text("finer").size(theme::TEXT_SIZE_SMALL))
                    .style(theme::tool_button)
                    .on_press_maybe((finer >= FINEST_VOXEL_MM).then_some(Message::Resample(finer))),
                button(text("coarser").size(theme::TEXT_SIZE_SMALL))
                    .style(theme::tool_button)
                    .on_press_maybe(
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
                    .style(theme::tool_button)
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
                        .style(theme::tool_button)
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
                    .style(theme::tool_button)
                    .on_press(Message::ResetPressurePeak),
            ]
            .spacing(theme::S2)
            .align_y(Alignment::Center),
        ]
        .spacing(theme::S2)
        .into()
    }
}

/// Centre a modal card on a layer that dims the application and swallows the
/// presses that would otherwise reach it.
///
/// # The dimming was never the part that made it modal
///
/// Every one of the three cards used to build this shape itself, with a
/// comment saying the scrim "swallows presses". It did not. `theme::scrim` is
/// a `container` *style*, and `container::update`
/// (`iced_widget-0.14.2/src/container.rs:298`) forwards the event to its child
/// and returns — it never calls `shell.capture_event`, so a press over the
/// dimmed area travelled straight on to the shader widget underneath and
/// sculpted the model the card was asking about. `iced::widget::opaque` is the
/// piece that was missing: it captures mouse presses inside its bounds
/// (`helpers.rs:577`), and `Stack::update` walks its children topmost-first and
/// stops at the first capture (`stack.rs:249-264`), so the layers below never
/// see it.
///
/// This is a legal capture under the rule the viewport is written to: **only
/// bounds-checked events may capture**. `opaque` captures presses only, only
/// when the cursor is over it — never a move and never a release, so a drag
/// that started before the card appeared still ends properly.
///
/// The window's resize strips are stacked *above* the card in `view`, so they
/// keep working: reverse traversal reaches them first.
fn modal_layer<'a>(card: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    opaque(
        container(card)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Alignment::Center)
            .align_y(Alignment::Center)
            .style(theme::scrim),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The drop line lands ON the depth it names, not between two depths.
    ///
    /// **The one thing about this panel a headless test can still check and the
    /// eye cannot easily.** The line and the rows are laid out by different
    /// widgets: the rows get their offset from a marker, an indent and the
    /// `row`'s own spacing, and the line gets it from a single space. When they
    /// disagree the failure is 6 px of ambiguity in the only signal that
    /// distinguishes "stay in the folder" from "come out" -- exactly the
    /// complaint section 3 of the plan is written against, and small enough to
    /// survive a visual pass.
    ///
    /// The expectation is spelled out from `body_row`'s children rather than
    /// calling [`row_content_x`] twice: this fails if either side moves.
    #[test]
    fn the_drop_line_starts_where_the_row_it_names_starts() {
        for depth in 0..brokkr_core::MAX_DEPTH {
            // `row![marker, indent].spacing(S2)`, walked left to right: the
            // marker, the spacing after it, the indent, and the spacing before
            // the first real column.
            let mut row_content = MARKER;
            row_content += theme::S2;
            row_content += f32::from(depth) * INDENT_PER_DEPTH;
            row_content += theme::S2;

            assert_eq!(
                row_content_x(depth),
                row_content,
                "the drop line at depth {depth} does not start where a depth-{depth} row does"
            );
        }

        // And one step per level, so no two depths can draw in the same place.
        assert_eq!(row_content_x(1) - row_content_x(0), INDENT_PER_DEPTH);
    }
}
