// SPDX-License-Identifier: AGPL-3.0-only

//! Undo and redo.
//!
//! One gesture is one entry. While a stroke is in progress the volume snapshots
//! each brick's prior contents the first time that stroke touches it, so a
//! stroke that passes over the same brick fifty times still costs one copy of
//! it.
//!
//! History is capped by a memory budget rather than a number of entries,
//! because entries vary by three orders of magnitude: a dab on one brick costs
//! 128 KB and a long sweep across a model can cost hundreds of megabytes. A
//! count based cap would either throw away useful history or blow the memory
//! ceiling, depending on how the user happened to be working.
//!
//! # One gesture, N bodies, one entry
//!
//! An [`Entry`] is a list of [`Change`]s. That is what lets one gesture span
//! bodies: a plane cut drawn across everything on screen is N `Change::Bricks`
//! in ONE entry, and a split is a `NodeRemoved` and two `NodeAdded`s in one
//! entry. Undoing has to put all of it back or none of it, because half a split
//! is not a document state anything downstream is written against.
//!
//! **A `Change` is written from the point of view of undoing.** Applying one
//! moves the document to the EARLIER state and returns the change that would
//! move it back, so `Change::NodeAdded` means "a node was added; applying this
//! removes it again" and carries only the id, while `Change::NodeRemoved`
//! carries the whole node because putting it back is what applying it does.
//!
//! **An entry's changes are applied in REVERSE order, and each inverse is
//! collected in the order it was produced.** Worked, because getting it wrong
//! is silent: a split records `[NodeRemoved(X), NodeAdded(A), NodeAdded(B)]` in
//! the order the split performed them, so undoing runs B, A, X -- remove B,
//! remove A, put X back. Applying in the recorded order would put X back into a
//! list that still holds A and B and then remove the wrong indices. The
//! inverses come out as `[inv(B), inv(A), inv(X)]`, which is the same rule read
//! backwards: applying THAT in reverse replays the original gesture in its
//! original order. Two consequences worth having: the rule is one line of code
//! with no special cases, and the inverse of the inverse is the entry you
//! started with, index for index. `undoing_and_redoing_a_split_shaped_entry_...`
//! is what holds it.
//!
//! # The invariant that deletes sixty lines of machinery
//!
//! **A body leaves the document ONLY through a [`Change::NodeRemoved`], or
//! through a whole-document replacement that clears the history.** The stack is
//! chronological and [`History::trim`] evicts oldest-first, so a removal always
//! sits ABOVE every brick edit to the body it removed, and a `Change::Bricks`
//! is therefore always applicable. That is why there is no liveness check here,
//! no `forget_body`, and no pruning in the middle of the stack. It also
//! constrains the byte policy: **eviction must stay a prefix drop.**
//!
//! # Two counters, one deque
//!
//! A stroke and a deleted body are both bytes, and one number cannot answer
//! both of the questions they raise. [`Entry::stroke_bytes`] is *how much
//! history am I holding* -- brick snapshots, which accumulate one gesture at a
//! time and are what [`DEFAULT_HISTORY_BUDGET`] was measured against.
//! [`Entry::reclaim_bytes`] is *what would this operation cost to keep* -- whole
//! volumes moved out of the document, where the entry IS the only copy and its
//! size is known before the operation runs, which is what lets a delete predict
//! its own prompt from the same number the allowance uses. Charging a 765 MB
//! body to the stroke budget would evict every stroke behind it; charging it to
//! nothing would let forty modest bodies sit in history unbounded.
//!
//! **This raises the effective history ceiling to 768 MB**, deliberately, and
//! that is a change to one of this project's four allocation ceilings. History
//! is invisible to the volume guard, so 768 MB of history over a 5 GiB document
//! is over the 6 GiB ceiling with nothing predicting it.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use crate::body::{Document, Node, NodeId, NodeMeta};
use crate::brick::{Brick, BrickCoord};

/// Default ceiling on the brick snapshots undo history holds.
///
/// The whole point of the volume design is to stay well under the roughly 3 GB
/// where comparable tools fall over, so history gets a bounded slice of that
/// rather than an open ended one.
pub const DEFAULT_HISTORY_BUDGET: usize = 256 * 1024 * 1024;

/// Default ceiling on the volumes undo history is holding on BEHALF of deleted
/// bodies, separate from [`DEFAULT_HISTORY_BUDGET`].
///
/// **512 MB is also the threshold at which deleting a body prompts with its
/// size, and that is one number rather than two that happen to coincide**: a
/// delete which would be evicted before it could be undone is exactly the
/// delete that has to warn first. Forty modest 20 MB bodies is 800 MB against
/// this allowance with no single body anywhere near it, which is why the
/// eviction itself also has to say something -- see [`HistoryStats::dropped_bodies`].
pub const DEFAULT_RECLAIM_BUDGET: usize = 512 * 1024 * 1024;

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

/// One thing an entry puts back.
///
/// Read every variant as the event it undoes, because that is what applying it
/// does; see the module doc. The payloads follow from that and not the other
/// way round -- `NodeRemoved` is the only one carrying a whole node, because
/// restoring one is the only thing that needs it.
pub enum Change {
    /// The prior contents of the bricks one gesture changed, in one body.
    Bricks { body: NodeId, edit: StrokeEdit },
    /// A node was added at this position. Applying removes it again.
    NodeAdded { at: usize, id: NodeId },
    /// A node was removed from this position. Applying puts it back.
    ///
    /// The node is MOVED in and never cloned: [`crate::Volume`] does not
    /// implement `Clone` at all, so a delete costs no allocation and peak
    /// memory does not rise -- it merely does not fall.
    NodeRemoved { at: usize, node: Box<Node> },
    /// A row's name, eye, collapse or depth changed. Applying writes `before`.
    NodeMeta { id: NodeId, before: NodeMeta, after: NodeMeta },
}

impl Change {
    /// Put this change into the document and hand back the change that would
    /// put the document back as it was.
    fn apply(self, doc: &mut Document) -> Self {
        match self {
            Change::Bricks { body, edit } => {
                // Always applicable: see the invariant in the module doc. A
                // body cannot have left the document while an edit to it is
                // still on the stack, because the removal that took it out
                // would sit above this entry and eviction is a prefix drop.
                let volume = doc
                    .volume_mut(body)
                    .expect("a brick edit names a body that is still in the document");
                Change::Bricks { body, edit: volume.apply_edit(edit) }
            }
            Change::NodeAdded { at, id } => {
                let node = doc.remove(at);
                debug_assert_eq!(node.id, id, "the node at {at} is not the one that was added");
                Change::NodeRemoved { at, node: Box::new(node) }
            }
            Change::NodeRemoved { at, node } => {
                let id = node.id;
                doc.insert(at, *node);
                Change::NodeAdded { at, id }
            }
            Change::NodeMeta { id, before, after } => {
                doc.set_meta(&before);
                // The inverse is the pair swapped, which applies `after`.
                Change::NodeMeta { id, before: after, after: before }
            }
        }
    }

    /// The node this change is about.
    fn node(&self) -> NodeId {
        match self {
            Change::Bricks { body, .. } => *body,
            Change::NodeAdded { id, .. } => *id,
            Change::NodeRemoved { node, .. } => node.id,
            Change::NodeMeta { id, .. } => *id,
        }
    }

    /// What this change costs the stroke budget.
    fn stroke_bytes(&self) -> usize {
        match self {
            Change::Bricks { edit, .. } => edit.bytes(),
            Change::NodeRemoved { .. } => 0,
            Change::NodeAdded { .. } => size_of::<Self>(),
            Change::NodeMeta { before, after, .. } => before.bytes() + after.bytes(),
        }
    }

    /// What this change costs the reclaim allowance.
    ///
    /// [`crate::Volume::stats`] and not `Brick::heap_bytes`: a brick knows
    /// nothing about whole volumes, and summing it over an entry that holds one
    /// counts a gigabyte as roughly zero -- which is the failure that lets a
    /// deleted dragon sit in history reporting nothing.
    fn reclaim_bytes(&self) -> usize {
        match self {
            Change::NodeRemoved { node, .. } => {
                size_of::<Node>()
                    + node.name.capacity()
                    + node.volume().map_or(0, |volume| volume.stats().resident_bytes)
            }
            _ => 0,
        }
    }
}

/// One gesture: everything it changed, across every body it touched.
pub struct Entry {
    changes: Vec<Change>,
    stroke_bytes: usize,
    reclaim_bytes: usize,
}

impl Entry {
    /// Both byte counts are summed once, here, because a `NodeRemoved` costs a
    /// walk of a whole brick map to measure and the budget asks for it on every
    /// push, trim and eviction.
    pub fn new(changes: Vec<Change>) -> Self {
        let stroke_bytes = changes.iter().map(Change::stroke_bytes).sum();
        let reclaim_bytes = changes.iter().map(Change::reclaim_bytes).sum();
        Self { changes, stroke_bytes, reclaim_bytes }
    }

    /// The common case: one stroke, on one body.
    pub fn stroke(body: NodeId, edit: StrokeEdit) -> Self {
        Self::new(vec![Change::Bricks { body, edit }])
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Brick snapshots, against [`DEFAULT_HISTORY_BUDGET`].
    #[inline]
    pub fn stroke_bytes(&self) -> usize {
        self.stroke_bytes
    }

    /// Volumes held on behalf of deleted bodies, against
    /// [`DEFAULT_RECLAIM_BUDGET`].
    #[inline]
    pub fn reclaim_bytes(&self) -> usize {
        self.reclaim_bytes
    }

    /// Whether this entry is the only copy of a body that is not in the
    /// document.
    #[inline]
    fn holds_a_body(&self) -> bool {
        self.changes.iter().any(|change| matches!(change, Change::NodeRemoved { .. }))
    }

    /// The first node this entry names, which is what the caller selects or
    /// reports on.
    fn first_node(&self) -> Option<NodeId> {
        self.changes.first().map(Change::node)
    }

    /// The first body this entry would change that the user cannot see, if
    /// there is one.
    ///
    /// **Two of the four changes are deliberately NOT gated, and both reasons
    /// were arrived at by trying it the other way.**
    ///
    /// `NodeRemoved` restores a node that is not in the document, so there is
    /// no resolved visibility to read and its own eye bit is the only input.
    /// Gating on that bit deadlocks: the body cannot be un-hidden, because it
    /// is not in the tree to click, so the entry can never be applied and it
    /// blocks every older entry behind it for the rest of the session.
    ///
    /// `NodeMeta` is how the eye itself is undone. Gating it means that undoing
    /// a hide is refused *because of the hide*, and the user is told to reveal
    /// the body by hand -- which is the very thing they pressed ctrl+Z to do.
    /// It is also visible in the panel whatever the eye says, so nothing is
    /// happening off screen.
    fn blocked_by(&self, doc: &Document, visible: &[bool]) -> Option<NodeId> {
        self.changes.iter().find_map(|change| {
            let id = match change {
                Change::Bricks { body, .. } => *body,
                Change::NodeAdded { id, .. } => *id,
                Change::NodeRemoved { .. } | Change::NodeMeta { .. } => return None,
            };
            // A node that is not in the document, or a mask too short to cover
            // it, counts as visible: refusing on a missing answer would be the
            // deadlock above with an extra step.
            let hidden =
                doc.index_of(id).and_then(|index| visible.get(index)).is_some_and(|shown| !shown);
            hidden.then_some(id)
        })
    }

    /// Apply every change and hand back the entry that would undo this one.
    ///
    /// Reverse order, inverses collected as they are produced. The module doc
    /// works through why, and why the two halves of that sentence are not
    /// interchangeable.
    fn apply(self, doc: &mut Document) -> Self {
        let mut inverses = Vec::with_capacity(self.changes.len());
        for change in self.changes.into_iter().rev() {
            inverses.push(change.apply(doc));
        }
        Self::new(inverses)
    }
}

/// What an undo or a redo did, and to what.
///
/// `Refused` carries the first body in the way so that the caller has a name to
/// print. Undo deliberately does NOT reveal or select that body itself: the eye
/// bit is persisted, undoable document state, so writing it from inside undo
/// would destroy a deliberate hide and mark the file dirty for a change the
/// user never made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoOutcome {
    /// Applied whole. The node named is the first the entry touched.
    Applied(NodeId),
    /// Applied nothing, because this body is not on screen.
    Refused(NodeId),
    /// The stack was empty.
    Nothing,
}

/// What the history is holding, for the debug overlay.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HistoryStats {
    pub undo_entries: usize,
    pub redo_entries: usize,
    /// Brick snapshots, on both stacks.
    pub bytes: usize,
    pub budget_bytes: usize,
    /// Volumes held on behalf of deleted bodies, on both stacks.
    pub reclaim_bytes: usize,
    pub reclaim_budget_bytes: usize,
    /// Entries dropped off the far end because a budget was reached.
    pub dropped: usize,
    /// Of those, the ones that were holding a deleted body.
    ///
    /// Separate from [`HistoryStats::dropped`] because it is the only one that
    /// has to be said out loud: two 300 MB folder deletes each pass the 512 MB
    /// prompt individually and then evict each other, and no per-operation
    /// prompt can catch that pair. A dropped stroke costs the user a redo they
    /// did not ask for; a dropped body costs them the body.
    pub dropped_bodies: usize,
}

/// A bounded stack of gestures.
pub struct History {
    undo: VecDeque<Entry>,
    redo: Vec<Entry>,
    budget: usize,
    reclaim_budget: usize,
    bytes: usize,
    reclaim_bytes: usize,
    dropped: usize,
    dropped_bodies: usize,
}

impl History {
    /// The stroke budget is a parameter and the reclaim allowance is not,
    /// because the one number tests and callers vary is the first.
    pub fn new(budget_bytes: usize) -> Self {
        Self::with_budgets(budget_bytes, DEFAULT_RECLAIM_BUDGET)
    }

    pub fn with_budgets(budget_bytes: usize, reclaim_budget_bytes: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            budget: budget_bytes,
            reclaim_budget: reclaim_budget_bytes,
            bytes: 0,
            reclaim_bytes: 0,
            dropped: 0,
            dropped_bodies: 0,
        }
    }

    /// Record a finished gesture.
    ///
    /// Doing anything new invalidates the redo stack, which is the universal
    /// convention and the only one that cannot produce a history that never
    /// happened.
    pub fn push(&mut self, entry: Entry) {
        if entry.is_empty() {
            return;
        }
        self.bytes += entry.stroke_bytes;
        self.reclaim_bytes += entry.reclaim_bytes;
        self.undo.push_back(entry);
        self.clear_redo();
        self.trim();
    }

    /// Drop the oldest entries while EITHER allowance is over.
    ///
    /// A single entry larger than the whole budget is kept anyway: dropping it
    /// would mean the user's last action could not be undone at all, which is
    /// worse than briefly exceeding a soft ceiling.
    ///
    /// Oldest-first and nothing else, ever. The module doc's invariant -- that
    /// a `Bricks` change is always applicable -- holds only because eviction is
    /// a prefix drop; taking the largest entry out of the middle instead would
    /// leave brick edits above a removal that is no longer there.
    fn trim(&mut self) {
        while (self.bytes > self.budget || self.reclaim_bytes > self.reclaim_budget)
            && self.undo.len() > 1
        {
            let Some(dropped) = self.undo.pop_front() else {
                break;
            };
            self.bytes -= dropped.stroke_bytes;
            self.reclaim_bytes -= dropped.reclaim_bytes;
            self.dropped += 1;
            if dropped.holds_a_body() {
                self.dropped_bodies += 1;
            }
        }
    }

    fn clear_redo(&mut self) {
        for entry in self.redo.drain(..) {
            self.bytes -= entry.stroke_bytes;
            self.reclaim_bytes -= entry.reclaim_bytes;
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

    /// Undo the most recent gesture.
    ///
    /// `visible` is the resolved display visibility, indexed by NODE position,
    /// and an entry is refused WHOLE if any body it would change is not in it.
    /// A partial apply is the screen-contradicting-the-engine failure this
    /// design guards against everywhere else, and half a split is not a state
    /// the document's invariants survive.
    ///
    /// Applying an entry hands back its inverse, so the redo stack is built
    /// from the state that was actually replaced rather than from a guess.
    pub fn undo(&mut self, doc: &mut Document, visible: &[bool]) -> UndoOutcome {
        debug_assert_eq!(
            visible.len(),
            doc.node_count(),
            "the visibility mask is indexed by node position and must cover every node"
        );
        let Some(entry) = self.undo.back() else {
            return UndoOutcome::Nothing;
        };
        if let Some(hidden) = entry.blocked_by(doc, visible) {
            // Left on the stack, not popped and pushed back: a refusal must
            // cost nothing, so that pressing ctrl+Z against a hidden body is a
            // message rather than a rearrangement of history.
            return UndoOutcome::Refused(hidden);
        }
        let entry = self.undo.pop_back().expect("checked just above");
        self.apply(entry, doc, |history, inverse| history.redo.push(inverse))
    }

    /// Redo the most recently undone gesture, under the same refusal.
    pub fn redo(&mut self, doc: &mut Document, visible: &[bool]) -> UndoOutcome {
        debug_assert_eq!(
            visible.len(),
            doc.node_count(),
            "the visibility mask is indexed by node position and must cover every node"
        );
        let Some(entry) = self.redo.last() else {
            return UndoOutcome::Nothing;
        };
        if let Some(hidden) = entry.blocked_by(doc, visible) {
            return UndoOutcome::Refused(hidden);
        }
        let entry = self.redo.pop().expect("checked just above");
        self.apply(entry, doc, |history, inverse| history.undo.push_back(inverse))
    }

    /// The half undo and redo share: swap the entry into the document and put
    /// its inverse on the other stack, keeping both counters straight.
    ///
    /// Neither direction trims. The two stacks together hold the same bytes
    /// they held a moment ago -- an inverse is the same bricks -- so trimming
    /// here could only evict an entry the user is in the middle of walking
    /// past.
    fn apply(
        &mut self,
        entry: Entry,
        doc: &mut Document,
        keep: fn(&mut Self, Entry),
    ) -> UndoOutcome {
        self.bytes -= entry.stroke_bytes;
        self.reclaim_bytes -= entry.reclaim_bytes;
        // Read off the entry being applied and not off its inverse: the
        // inverse's changes are in the opposite order, so its first node is
        // this entry's LAST one, which is not what "the body undo touched"
        // means to anything asking.
        let touched = entry.first_node().expect("an entry always holds at least one change");
        let inverse = entry.apply(doc);
        self.bytes += inverse.stroke_bytes;
        self.reclaim_bytes += inverse.reclaim_bytes;
        keep(self, inverse);
        UndoOutcome::Applied(touched)
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.bytes = 0;
        self.reclaim_bytes = 0;
        self.dropped = 0;
        self.dropped_bodies = 0;
    }

    pub fn stats(&self) -> HistoryStats {
        HistoryStats {
            undo_entries: self.undo.len(),
            redo_entries: self.redo.len(),
            bytes: self.bytes,
            budget_bytes: self.budget,
            reclaim_bytes: self.reclaim_bytes,
            reclaim_budget_bytes: self.reclaim_budget,
            dropped: self.dropped,
            dropped_bodies: self.dropped_bodies,
        }
    }
}

/// By its counters rather than its contents: an entry holds volumes, which do
/// not implement `Debug` and which nobody wants printed anyway.
impl std::fmt::Debug for History {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stats().fmt(f)
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
    use crate::volume::Volume;
    use crate::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp};
    use glam::Vec3;

    fn sculpted() -> (Document, Brush, BrushScratch) {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        let brush = Brush { kind: BrushKind::Draw, radius: 8.0, strength: 0.4, ..Brush::default() };
        (doc, brush, BrushScratch::new())
    }

    /// One stroke on the active body, as one entry.
    fn stroke(doc: &mut Document, brush: &Brush, scratch: &mut BrushScratch, at: Vec3) -> Entry {
        let body = doc.active();
        stroke_on(doc, body, brush, scratch, at)
    }

    fn stroke_on(
        doc: &mut Document,
        body: NodeId,
        brush: &Brush,
        scratch: &mut BrushScratch,
        at: Vec3,
    ) -> Entry {
        let volume = doc.volume_mut(body).expect("a body to stroke");
        volume.begin_stroke();
        let normal = volume.gradient_world(at);
        brush.apply(volume, &Stamp::new(at, normal, BrushDirection::Add), scratch);
        Entry::stroke(body, volume.end_stroke().expect("the stroke changed something"))
    }

    /// Nothing is hidden, which is what every test here means unless it says
    /// otherwise. Rebuilt per call because entries add and remove nodes and the
    /// mask is indexed by position.
    fn all_shown(doc: &Document) -> Vec<bool> {
        vec![true; doc.node_count()]
    }

    fn undo(history: &mut History, doc: &mut Document) -> UndoOutcome {
        let shown = all_shown(doc);
        history.undo(doc, &shown)
    }

    fn redo(history: &mut History, doc: &mut Document) -> UndoOutcome {
        let shown = all_shown(doc);
        history.redo(doc, &shown)
    }

    fn applied(outcome: UndoOutcome) -> NodeId {
        match outcome {
            UndoOutcome::Applied(id) => id,
            other => panic!("expected the entry to apply, got {other:?}"),
        }
    }

    /// A body big enough that its resident bytes cannot be confused with the
    /// few dozen a node record costs.
    fn heavy(voxel_size: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::ZERO, 16.0);
        volume
    }

    /// Everything about a document that undo is supposed to restore, in a form
    /// a failing assertion can print.
    ///
    /// The field is compared by its brick census plus probes rather than voxel
    /// by voxel: a census that matches with different voxels is possible in
    /// principle, and the probes are what make it not worth arranging.
    fn fingerprint(doc: &Document) -> Vec<String> {
        const PROBES: [Vec3; 4] = [
            Vec3::ZERO,
            Vec3::new(16.0, 0.0, 0.0),
            Vec3::new(0.0, 16.0, 0.0),
            Vec3::new(0.0, 0.0, 16.0),
        ];
        doc.nodes()
            .iter()
            .map(|node| {
                let field = match node.volume() {
                    Some(volume) => {
                        let stats = volume.stats();
                        let probes: Vec<String> = PROBES
                            .iter()
                            .map(|probe| format!("{:.6}", volume.sample_world(*probe)))
                            .collect();
                        format!(
                            "{} bricks ({} dense, {} uniform) {}",
                            volume.brick_count(),
                            stats.dense_bricks,
                            stats.uniform_bricks,
                            probes.join(" ")
                        )
                    }
                    None => "folder".to_string(),
                };
                format!(
                    "{:?} {:?} depth {} visible {} :: {field}",
                    node.id,
                    node.name,
                    node.depth(),
                    node.visible
                )
            })
            .collect()
    }

    #[test]
    fn undo_restores_the_field_exactly() {
        let (mut doc, brush, mut scratch) = sculpted();
        let probe = Vec3::new(24.0, 0.0, 0.0);
        let before = doc.active_volume().sample_world(probe);

        let mut history = History::default();
        let entry = stroke(&mut doc, &brush, &mut scratch, probe);
        history.push(entry);
        assert_ne!(
            doc.active_volume().sample_world(probe),
            before,
            "the stroke should have changed the field"
        );

        applied(undo(&mut history, &mut doc));
        assert_eq!(
            doc.active_volume().sample_world(probe),
            before,
            "undo did not restore the value"
        );
    }

    #[test]
    fn redo_puts_it_back() {
        let (mut doc, brush, mut scratch) = sculpted();
        let probe = Vec3::new(24.0, 0.0, 0.0);

        let mut history = History::default();
        let entry = stroke(&mut doc, &brush, &mut scratch, probe);
        history.push(entry);
        let sculpted_value = doc.active_volume().sample_world(probe);

        applied(undo(&mut history, &mut doc));
        applied(redo(&mut history, &mut doc));
        assert_eq!(
            doc.active_volume().sample_world(probe),
            sculpted_value,
            "redo did not restore the stroke"
        );
    }

    #[test]
    fn undo_marks_the_restored_bricks_dirty() {
        // Restoring the field without scheduling a remesh would leave the old
        // triangles on screen, which looks exactly like undo not working.
        let (mut doc, brush, mut scratch) = sculpted();
        let mut history = History::default();
        let entry = stroke(&mut doc, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0));
        history.push(entry);

        let mut dirty = Vec::new();
        doc.take_dirty(&mut dirty);
        applied(undo(&mut history, &mut doc));
        doc.take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "undo scheduled no remesh");
    }

    #[test]
    fn many_strokes_undo_in_reverse_order() {
        let (mut doc, brush, mut scratch) = sculpted();
        let mut history = History::default();

        let probes =
            [Vec3::new(24.0, 0.0, 0.0), Vec3::new(0.0, 24.0, 0.0), Vec3::new(0.0, 0.0, 24.0)];
        let mut checkpoints = vec![probes.map(|p| doc.active_volume().sample_world(p))];
        for probe in probes {
            let entry = stroke(&mut doc, &brush, &mut scratch, probe);
            history.push(entry);
            checkpoints.push(probes.map(|p| doc.active_volume().sample_world(p)));
        }

        for step in (0..probes.len()).rev() {
            applied(undo(&mut history, &mut doc));
            let now = probes.map(|p| doc.active_volume().sample_world(p));
            assert_eq!(now, checkpoints[step], "state after undoing to step {step} is wrong");
        }
        assert!(!history.can_undo());
        assert_eq!(undo(&mut history, &mut doc), UndoOutcome::Nothing);
    }

    #[test]
    fn a_new_stroke_discards_the_redo_stack() {
        let (mut doc, brush, mut scratch) = sculpted();
        let mut history = History::default();
        let entry = stroke(&mut doc, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0));
        history.push(entry);
        applied(undo(&mut history, &mut doc));
        assert!(history.can_redo());

        let entry = stroke(&mut doc, &brush, &mut scratch, Vec3::new(0.0, 24.0, 0.0));
        history.push(entry);
        assert!(!history.can_redo(), "a new stroke must invalidate redo");
        assert_eq!(history.stats().undo_entries, 1, "the discarded entry left one behind");
    }

    #[test]
    fn a_stroke_that_changes_nothing_records_no_entry() {
        // Clicking on empty space must not fill history with blanks.
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        doc.active_volume_mut().begin_stroke();
        assert!(doc.active_volume_mut().end_stroke().is_none());
    }

    #[test]
    fn touching_a_brick_repeatedly_snapshots_it_once() {
        let (mut doc, brush, mut scratch) = sculpted();
        let at = Vec3::new(24.0, 0.0, 0.0);
        let volume = doc.active_volume_mut();
        let normal = volume.gradient_world(at);

        volume.begin_stroke();
        brush.apply(volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        let after_one = volume.end_stroke().expect("changed something").len();

        volume.begin_stroke();
        for _ in 0..20 {
            brush.apply(volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        }
        let after_twenty = volume.end_stroke().expect("changed something").len();

        assert_eq!(
            after_one, after_twenty,
            "twenty stamps on the same spot should snapshot the same bricks once each"
        );
    }

    #[test]
    fn the_budget_drops_the_oldest_entries() {
        let (mut doc, brush, mut scratch) = sculpted();
        // A budget far below one entry, so every push but the last is dropped.
        let mut history = History::new(1);
        for probe in
            [Vec3::new(24.0, 0.0, 0.0), Vec3::new(0.0, 24.0, 0.0), Vec3::new(0.0, 0.0, 24.0)]
        {
            let entry = stroke(&mut doc, &brush, &mut scratch, probe);
            history.push(entry);
        }

        let stats = history.stats();
        assert_eq!(stats.undo_entries, 1, "only the newest entry should survive");
        assert_eq!(stats.dropped, 2);
        assert_eq!(stats.dropped_bodies, 0, "no body was deleted, so none was lost");
        // The most recent action stays undoable even though it is over budget.
        assert!(history.can_undo());
    }

    #[test]
    fn the_byte_count_returns_to_zero_when_cleared() {
        let (mut doc, brush, mut scratch) = sculpted();
        let mut history = History::default();
        let entry = stroke(&mut doc, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0));
        history.push(entry);
        assert!(history.stats().bytes > 0);
        history.clear();
        assert_eq!(history.stats().bytes, 0);
        assert_eq!(history.stats().reclaim_bytes, 0);
    }

    // ------------------------------------------------- compound entries

    /// The ordering rule, in the shape that panics when it is wrong: undoing
    /// has to restore the bricks BEFORE it removes the body they belong to. In
    /// the recorded order it removes the body first and the brick change then
    /// names a body that is not there.
    #[test]
    fn an_add_then_edit_unwinds_in_the_opposite_order() {
        let (mut doc, brush, mut scratch) = sculpted();
        let id = doc.add_body("Body 2", heavy(1.0));
        let at = doc.index_of(id).expect("the body that was just added");
        let edit = match stroke_on(&mut doc, id, &brush, &mut scratch, Vec3::new(16.0, 0.0, 0.0))
            .changes
            .pop()
        {
            Some(Change::Bricks { edit, .. }) => edit,
            _ => panic!("a stroke is one brick change"),
        };

        let mut history = History::default();
        // Recorded in the order the gesture performed them.
        history.push(Entry::new(vec![
            Change::NodeAdded { at, id },
            Change::Bricks { body: id, edit },
        ]));

        applied(undo(&mut history, &mut doc));
        assert_eq!(doc.body_count(), 1, "the added body should be gone again");
        assert!(doc.volume(id).is_none());
    }

    /// The mirror: redo replays the gesture in its ORIGINAL order, so the body
    /// comes back before the edit lands on it, and the field afterwards is the
    /// one the stroke left.
    #[test]
    fn redoing_an_add_then_edit_replays_them_in_the_original_order() {
        let (mut doc, brush, mut scratch) = sculpted();
        let id = doc.add_body("Body 2", heavy(1.0));
        let at = doc.index_of(id).expect("the body that was just added");
        let probe = Vec3::new(16.0, 0.0, 0.0);
        let edit = match stroke_on(&mut doc, id, &brush, &mut scratch, probe).changes.pop() {
            Some(Change::Bricks { edit, .. }) => edit,
            _ => panic!("a stroke is one brick change"),
        };
        let sculpted_value = doc.volume(id).expect("the new body").sample_world(probe);

        let mut history = History::default();
        history.push(Entry::new(vec![
            Change::NodeAdded { at, id },
            Change::Bricks { body: id, edit },
        ]));

        applied(undo(&mut history, &mut doc));
        applied(redo(&mut history, &mut doc));

        let volume = doc.volume(id).expect("redo should have put the body back");
        assert_eq!(volume.sample_world(probe), sculpted_value, "redo lost the edit on the body");
        assert_eq!(doc.index_of(id), Some(at), "the body came back in a different place");
    }

    /// The split shape -- one body out, two in, one gesture -- which is the
    /// smallest entry that a two-change pair cannot stand in for: it is the
    /// only one where applying in the wrong order removes a node whose index
    /// another change in the same entry still depends on.
    #[test]
    fn undoing_and_redoing_a_split_shaped_entry_returns_the_identical_document() {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        let original = doc.add_body("Original", heavy(1.0));
        let before = fingerprint(&doc);
        let active = doc.active();

        // The split itself: the body leaves the document into the entry, and
        // its two halves are added in its place.
        let at = doc.index_of(original).expect("the body being split");
        let node = doc.remove(at);
        let first = doc.add_body("Part 1", heavy(1.0));
        let second = doc.add_body("Part 2", heavy(1.0));
        let entry = Entry::new(vec![
            Change::NodeRemoved { at, node: Box::new(node) },
            Change::NodeAdded { at: doc.index_of(first).expect("part 1"), id: first },
            Change::NodeAdded { at: doc.index_of(second).expect("part 2"), id: second },
        ]);
        let after_split = fingerprint(&doc);

        let mut history = History::default();
        history.push(entry);

        applied(undo(&mut history, &mut doc));
        assert_eq!(fingerprint(&doc), before, "undoing the split did not restore the document");
        assert_eq!(doc.active(), active, "undo moved the selection");

        applied(redo(&mut history, &mut doc));
        assert_eq!(fingerprint(&doc), after_split, "redoing the split did not replay it");

        applied(undo(&mut history, &mut doc));
        assert_eq!(fingerprint(&doc), before, "the second round trip drifted");
    }

    /// The invariant that makes a liveness check unnecessary: a removal always
    /// sits above every brick edit to the body it removed, so the delete stays
    /// applicable however much is pushed on top of it.
    #[test]
    fn a_body_delete_is_still_undoable_after_a_later_stroke() {
        let (mut doc, brush, mut scratch) = sculpted();
        let doomed = doc.add_body("Body 2", heavy(1.0));
        let probe = Vec3::new(24.0, 0.0, 0.0);
        let before = fingerprint(&doc);

        let at = doc.index_of(doomed).expect("the body being deleted");
        let node = doc.remove(at);
        let mut history = History::default();
        history.push(Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]));

        // A later stroke on a different body, which must not make the delete
        // unreachable.
        let entry = stroke(&mut doc, &brush, &mut scratch, probe);
        history.push(entry);

        applied(undo(&mut history, &mut doc));
        applied(undo(&mut history, &mut doc));
        assert_eq!(fingerprint(&doc), before, "the delete did not come back whole");
    }

    /// A body moved into an entry comes back with everything marked, because
    /// [`crate::Volume::drain_dirty`] drains: without the marking in
    /// [`Document::insert`] the document is right, every assertion here passes,
    /// and the viewport never draws the body again.
    ///
    /// **The drain before the delete is the whole fixture.** Written without
    /// it, the body still carried the dirty set its seeding left behind --
    /// nothing had ever taken it, because a body outside the document is not
    /// walked -- so it came back dirty whatever `insert` did, and the test went
    /// green with the marking deleted. Measured, not reasoned.
    #[test]
    fn undoing_a_delete_marks_every_brick_of_the_restored_body() {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        let doomed = doc.add_body("Body 2", heavy(1.0));
        let bricks = doc.volume(doomed).expect("the new body").brick_count();
        assert!(bricks > 0, "the fixture must have bricks or this asserts nothing");

        let mut dirty = Vec::new();
        doc.take_dirty(&mut dirty);
        assert!(
            doc.volume(doomed).expect("the new body").dirty_count() == 0,
            "the body has to go into the entry with nothing marked, or this proves nothing"
        );

        let at = doc.index_of(doomed).expect("the body being deleted");
        let node = doc.remove(at);
        let mut history = History::default();
        history.push(Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]));

        applied(undo(&mut history, &mut doc));
        doc.take_dirty(&mut dirty);

        let restored = dirty.iter().filter(|(body, _)| *body == doomed).count();
        assert!(
            restored >= bricks,
            "the restored body scheduled {restored} bricks for remesh, not its {bricks}"
        );
    }

    // --------------------------------------------------- the two counters

    /// `Brick::heap_bytes` knows nothing about whole volumes, so summing it
    /// over an entry that holds one counts a deleted dragon as roughly zero.
    #[test]
    fn a_removed_body_reports_its_resident_bytes_and_not_roughly_nothing() {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        let doomed = doc.add_body("Body 2", heavy(1.0));
        let resident = doc.volume(doomed).expect("the new body").stats().resident_bytes;
        assert!(resident > 64 * 1024, "the fixture must be big enough to notice");

        let at = doc.index_of(doomed).expect("the body being deleted");
        let node = doc.remove(at);
        let entry = Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]);

        assert!(
            entry.reclaim_bytes() >= resident,
            "the entry reports {} bytes for a {resident} byte body",
            entry.reclaim_bytes()
        );
        assert_eq!(entry.stroke_bytes(), 0, "a deleted body is not a stroke");
    }

    /// The reason there are two counters and not one: a number that answers
    /// "how much history am I holding" cannot also answer "what would this
    /// operation cost to keep", and a whole volume charged to the stroke budget
    /// evicts every stroke behind it on the spot.
    #[test]
    fn a_deleted_body_is_charged_to_the_reclaim_allowance_and_not_the_stroke_budget() {
        let (mut doc, brush, mut scratch) = sculpted();
        let doomed = doc.add_body("Body 2", heavy(1.0));
        let at = doc.index_of(doomed).expect("the body being deleted");
        let node = doc.remove(at);
        let resident = node.volume().expect("a body").stats().resident_bytes;

        let mut history = History::default();
        history.push(Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]));
        let after_delete = history.stats();
        assert_eq!(after_delete.bytes, 0, "a deleted body was charged to the stroke budget");
        assert!(
            after_delete.reclaim_bytes >= resident,
            "a {resident} byte body counted as {} against the allowance",
            after_delete.reclaim_bytes
        );

        // And the other direction: a stroke moves one counter and not both.
        let entry = stroke(&mut doc, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0));
        let stroke_bytes = entry.stroke_bytes();
        assert!(stroke_bytes > 0, "the fixture stroke must cost something");
        history.push(entry);

        let stats = history.stats();
        assert_eq!(stats.bytes, stroke_bytes, "the stroke budget counts strokes and only strokes");
        assert_eq!(
            stats.reclaim_bytes, after_delete.reclaim_bytes,
            "a stroke moved the reclaim allowance"
        );
        assert_eq!(stats.dropped_bodies, 0, "nothing should have been evicted yet");
    }

    /// Two 300 MB folder deletes each pass a 512 MB prompt on their own and
    /// then evict each other, so the eviction has to be countable on its own --
    /// a dropped stroke costs a redo, a dropped body costs the body.
    #[test]
    fn an_eviction_that_drops_a_deleted_body_is_counted_apart_from_a_dropped_stroke() {
        let (mut doc, brush, mut scratch) = sculpted();
        let doomed = doc.add_body("Body 2", heavy(1.0));
        let at = doc.index_of(doomed).expect("the body being deleted");
        let node = doc.remove(at);

        // A generous stroke budget and an allowance of one byte: only the
        // reclaim side is over, and it still has to evict.
        let mut history = History::with_budgets(DEFAULT_HISTORY_BUDGET, 1);
        history.push(Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]));
        assert_eq!(history.stats().dropped_bodies, 0, "the only entry is kept over budget");

        let entry = stroke(&mut doc, &brush, &mut scratch, Vec3::new(24.0, 0.0, 0.0));
        history.push(entry);

        let stats = history.stats();
        assert_eq!(stats.undo_entries, 1, "the delete should have been evicted");
        assert_eq!(stats.dropped, 1);
        assert_eq!(stats.dropped_bodies, 1, "nothing said the body was unrecoverable");
        assert_eq!(stats.reclaim_bytes, 0, "the evicted body is still being counted");
    }

    // ------------------------------------------------------- the refusal

    #[test]
    fn an_entry_naming_a_hidden_body_is_refused_whole_and_costs_nothing() {
        let (mut doc, brush, mut scratch) = sculpted();
        let hidden = doc.add_body("Body 2", heavy(1.0));
        let probe = Vec3::new(16.0, 0.0, 0.0);
        let mut history = History::default();
        let entry = stroke_on(&mut doc, hidden, &brush, &mut scratch, probe);
        history.push(entry);
        let sculpted_value = doc.volume(hidden).expect("the second body").sample_world(probe);
        let before = history.stats();

        let mut shown = all_shown(&doc);
        shown[doc.index_of(hidden).expect("the second body")] = false;
        assert_eq!(history.undo(&mut doc, &shown), UndoOutcome::Refused(hidden));
        assert_eq!(
            doc.volume(hidden).expect("the second body").sample_world(probe),
            sculpted_value,
            "a refused undo changed the field anyway"
        );
        assert_eq!(history.stats(), before, "a refusal must leave the stacks exactly as they were");

        // Revealing it is all it takes; the entry was never consumed.
        applied(undo(&mut history, &mut doc));
        assert_ne!(
            doc.volume(hidden).expect("the second body").sample_world(probe),
            sculpted_value
        );
    }

    /// The refusal must never cover a body that is NOT in the document: it
    /// could not be revealed, so the entry would be unreachable forever and it
    /// would block every older entry behind it.
    #[test]
    fn restoring_a_body_that_was_hidden_when_it_was_deleted_is_not_refused() {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        let doomed = doc.add_body("Body 2", heavy(1.0));
        let mut meta = doc.meta(doomed).expect("the second body");
        meta.visible = false;
        doc.set_meta(&meta);

        let at = doc.index_of(doomed).expect("the body being deleted");
        let node = doc.remove(at);
        let mut history = History::default();
        history.push(Entry::new(vec![Change::NodeRemoved { at, node: Box::new(node) }]));

        assert_eq!(applied(undo(&mut history, &mut doc)), doomed);
        assert!(doc.volume(doomed).is_some(), "the hidden body never came back");
        assert!(!doc.node(doomed).expect("the restored body").visible, "its eye was rewritten");
    }

    /// Undoing a hide must not be refused BY the hide, which is the other way
    /// the blanket rule deadlocks.
    #[test]
    fn undoing_a_hide_is_not_refused_by_the_hide_it_undoes() {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        let other = doc.add_body("Body 2", heavy(1.0));

        let before = doc.meta(other).expect("the second body");
        let after = NodeMeta { visible: false, ..before.clone() };
        doc.set_meta(&after);

        let mut history = History::default();
        history.push(Entry::new(vec![Change::NodeMeta { id: other, before, after }]));

        let mut shown = all_shown(&doc);
        shown[doc.index_of(other).expect("the second body")] = false;
        assert_eq!(history.undo(&mut doc, &shown), UndoOutcome::Applied(other));
        assert!(doc.node(other).expect("the second body").visible, "the eye was not put back");
    }
}
