// SPDX-License-Identifier: AGPL-3.0-only

//! Undo and redo.
//!
//! One stroke is one entry. While a stroke is in progress the volume snapshots
//! each brick's prior contents the first time that stroke touches it, so a
//! stroke that passes over the same brick fifty times still costs one copy of
//! it.
//!
//! History is capped by a memory budget rather than a number of entries,
//! because entries vary by three orders of magnitude: a dab on one brick costs
//! 128 KB and a long sweep across a model can cost hundreds of megabytes. A
//! count based cap would either throw away useful history or blow the memory
//! ceiling, depending on how the user happened to be working.

use rustc_hash::FxHashMap;

use crate::brick::{Brick, BrickCoord};

/// Default ceiling on undo history.
///
/// The whole point of the volume design is to stay well under the roughly 3 GB
/// where comparable tools fall over, so history gets a bounded slice of that
/// rather than an open ended one.
pub const DEFAULT_HISTORY_BUDGET: usize = 256 * 1024 * 1024;

/// The prior contents of every brick one stroke changed.
///
/// A `None` brick means it did not exist before the stroke, which undo has to
/// restore just as faithfully as any content: leaving an empty brick behind
/// would leave its triangles on screen.
#[derive(Debug, Default)]
pub struct StrokeEdit {
    bricks: Vec<(BrickCoord, Option<Brick>)>,
    bytes: usize,
}

impl StrokeEdit {
    pub(crate) fn from_recording(recording: FxHashMap<BrickCoord, Option<Brick>>) -> Option<Self> {
        if recording.is_empty() {
            return None;
        }
        Some(Self::from_bricks(recording.into_iter().collect()))
    }

    pub(crate) fn from_bricks(bricks: Vec<(BrickCoord, Option<Brick>)>) -> Self {
        let bytes = bricks
            .iter()
            .map(|(_, brick)| {
                size_of::<(BrickCoord, Option<Brick>)>()
                    + brick.as_ref().map_or(0, Brick::heap_bytes)
            })
            .sum();
        Self { bricks, bytes }
    }

    pub(crate) fn into_bricks(self) -> Vec<(BrickCoord, Option<Brick>)> {
        self.bricks
    }

    /// Bricks this entry restores.
    #[inline]
    pub fn len(&self) -> usize {
        self.bricks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    /// Memory this entry holds, which is what the budget counts.
    #[inline]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// What the history is holding, for the debug overlay.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HistoryStats {
    pub undo_entries: usize,
    pub redo_entries: usize,
    pub bytes: usize,
    pub budget_bytes: usize,
    /// Entries dropped off the far end because the budget was reached.
    pub dropped: usize,
}

/// A bounded stack of stroke entries.
#[derive(Debug)]
pub struct History {
    undo: std::collections::VecDeque<StrokeEdit>,
    redo: Vec<StrokeEdit>,
    budget: usize,
    bytes: usize,
    dropped: usize,
}

impl History {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            undo: std::collections::VecDeque::new(),
            redo: Vec::new(),
            budget: budget_bytes,
            bytes: 0,
            dropped: 0,
        }
    }

    /// Record a finished stroke.
    ///
    /// Doing anything new invalidates the redo stack, which is the universal
    /// convention and the only one that cannot produce a history that never
    /// happened.
    pub fn push(&mut self, edit: StrokeEdit) {
        if edit.is_empty() {
            return;
        }
        self.bytes += edit.bytes();
        self.undo.push_back(edit);
        self.clear_redo();
        self.trim();
    }

    /// A single entry larger than the whole budget is kept anyway: dropping it
    /// would mean the user's last action could not be undone at all, which is
    /// worse than briefly exceeding a soft ceiling.
    fn trim(&mut self) {
        while self.bytes > self.budget && self.undo.len() > 1 {
            if let Some(dropped) = self.undo.pop_front() {
                self.bytes -= dropped.bytes();
                self.dropped += 1;
            }
        }
    }

    fn clear_redo(&mut self) {
        for edit in self.redo.drain(..) {
            self.bytes -= edit.bytes();
        }
    }

    #[inline]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    #[inline]
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Undo the most recent stroke. Returns false when there is nothing to undo.
    ///
    /// Applying an entry hands back its inverse, so the redo stack is built
    /// from the state that was actually replaced rather than from a guess.
    pub fn undo(&mut self, volume: &mut crate::Volume) -> bool {
        let Some(edit) = self.undo.pop_back() else {
            return false;
        };
        self.bytes -= edit.bytes();
        let inverse = volume.apply_edit(edit);
        self.bytes += inverse.bytes();
        self.redo.push(inverse);
        true
    }

    /// Redo the most recently undone stroke.
    pub fn redo(&mut self, volume: &mut crate::Volume) -> bool {
        let Some(edit) = self.redo.pop() else {
            return false;
        };
        self.bytes -= edit.bytes();
        let inverse = volume.apply_edit(edit);
        self.bytes += inverse.bytes();
        self.undo.push_back(inverse);
        true
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.bytes = 0;
        self.dropped = 0;
    }

    pub fn stats(&self) -> HistoryStats {
        HistoryStats {
            undo_entries: self.undo.len(),
            redo_entries: self.redo.len(),
            bytes: self.bytes,
            budget_bytes: self.budget,
            dropped: self.dropped,
        }
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new(DEFAULT_HISTORY_BUDGET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp, Volume};
    use glam::Vec3;

    fn sculpted() -> (Volume, Brush, BrushScratch) {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 24.0);
        let brush = Brush { kind: BrushKind::Draw, radius: 8.0, strength: 0.4, ..Brush::default() };
        (volume, brush, BrushScratch::new())
    }

    fn stroke(
        volume: &mut Volume,
        brush: &Brush,
        scratch: &mut BrushScratch,
        at: Vec3,
    ) -> StrokeEdit {
        volume.begin_stroke();
        let normal = volume.gradient_world(at);
        brush.apply(volume, &Stamp::new(at, normal, BrushDirection::Add), scratch);
        volume.end_stroke().expect("the stroke changed something")
    }

    #[test]
    fn undo_restores_the_field_exactly() {
        let (mut volume, brush, mut scratch) = sculpted();
        let probe = Vec3::new(24.0, 0.0, 0.0);
        let before = volume.sample_world(probe);

        let mut history = History::default();
        let edit = stroke(&mut volume, &brush, &mut scratch, probe);
        history.push(edit);
        assert_ne!(volume.sample_world(probe), before, "the stroke should have changed the field");

        assert!(history.undo(&mut volume));
        assert_eq!(volume.sample_world(probe), before, "undo did not restore the value");
    }

    #[test]
    fn redo_puts_it_back() {
        let (mut volume, brush, mut scratch) = sculpted();
        let probe = Vec3::new(24.0, 0.0, 0.0);

        let mut history = History::default();
        history.push(stroke(&mut volume, &brush, &mut scratch, probe));
        let sculpted_value = volume.sample_world(probe);

        assert!(history.undo(&mut volume));
        assert!(history.redo(&mut volume));
        assert_eq!(volume.sample_world(probe), sculpted_value, "redo did not restore the stroke");
    }

    #[test]
    fn undo_marks_the_restored_bricks_dirty() {
        // Restoring the field without scheduling a remesh would leave the old
        // triangles on screen, which looks exactly like undo not working.
        let (mut volume, brush, mut scratch) = sculpted();
        let mut history = History::default();
        history.push(stroke(&mut volume, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0)));

        let mut dirty = Vec::new();
        volume.take_dirty(&mut dirty);
        assert!(history.undo(&mut volume));
        volume.take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "undo scheduled no remesh");
    }

    #[test]
    fn many_strokes_undo_in_reverse_order() {
        let (mut volume, brush, mut scratch) = sculpted();
        let mut history = History::default();

        let probes =
            [Vec3::new(24.0, 0.0, 0.0), Vec3::new(0.0, 24.0, 0.0), Vec3::new(0.0, 0.0, 24.0)];
        let mut checkpoints = vec![probes.map(|p| volume.sample_world(p))];
        for probe in probes {
            history.push(stroke(&mut volume, &brush, &mut scratch, probe));
            checkpoints.push(probes.map(|p| volume.sample_world(p)));
        }

        for step in (0..probes.len()).rev() {
            assert!(history.undo(&mut volume));
            let now = probes.map(|p| volume.sample_world(p));
            assert_eq!(now, checkpoints[step], "state after undoing to step {step} is wrong");
        }
        assert!(!history.can_undo());
    }

    #[test]
    fn a_new_stroke_discards_the_redo_stack() {
        let (mut volume, brush, mut scratch) = sculpted();
        let mut history = History::default();
        history.push(stroke(&mut volume, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0)));
        assert!(history.undo(&mut volume));
        assert!(history.can_redo());

        history.push(stroke(&mut volume, &brush, &mut scratch, Vec3::new(0.0, 24.0, 0.0)));
        assert!(!history.can_redo(), "a new stroke must invalidate redo");
        assert_eq!(history.stats().bytes, history.stats().bytes, "byte count stays consistent");
    }

    #[test]
    fn a_stroke_that_changes_nothing_records_no_entry() {
        // Clicking on empty space must not fill history with blanks.
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 24.0);
        volume.begin_stroke();
        assert!(volume.end_stroke().is_none());
    }

    #[test]
    fn touching_a_brick_repeatedly_snapshots_it_once() {
        let (mut volume, brush, mut scratch) = sculpted();
        let at = Vec3::new(24.0, 0.0, 0.0);
        let normal = volume.gradient_world(at);

        volume.begin_stroke();
        brush.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        let after_one = volume.end_stroke().expect("changed something").len();

        volume.begin_stroke();
        for _ in 0..20 {
            brush.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        }
        let after_twenty = volume.end_stroke().expect("changed something").len();

        assert_eq!(
            after_one, after_twenty,
            "twenty stamps on the same spot should snapshot the same bricks once each"
        );
    }

    #[test]
    fn the_budget_drops_the_oldest_entries() {
        let (mut volume, brush, mut scratch) = sculpted();
        // A budget far below one entry, so every push but the last is dropped.
        let mut history = History::new(1);
        for probe in
            [Vec3::new(24.0, 0.0, 0.0), Vec3::new(0.0, 24.0, 0.0), Vec3::new(0.0, 0.0, 24.0)]
        {
            history.push(stroke(&mut volume, &brush, &mut scratch, probe));
        }

        let stats = history.stats();
        assert_eq!(stats.undo_entries, 1, "only the newest entry should survive");
        assert_eq!(stats.dropped, 2);
        // The most recent action stays undoable even though it is over budget.
        assert!(history.can_undo());
    }

    #[test]
    fn the_byte_count_returns_to_zero_when_cleared() {
        let (mut volume, brush, mut scratch) = sculpted();
        let mut history = History::default();
        history.push(stroke(&mut volume, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0)));
        assert!(history.stats().bytes > 0);
        history.clear();
        assert_eq!(history.stats().bytes, 0);
    }
}
