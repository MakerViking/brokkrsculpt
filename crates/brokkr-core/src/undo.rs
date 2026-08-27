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
//! chronological and [`History::trim`] evicts from the two FAR ENDS of it --
//! the oldest undo, then the furthest-future redo -- so a removal always sits
//! ABOVE every brick edit to the body it removed, and a `Change::Bricks`
//! is therefore always applicable. That is why there is no liveness check here,
//! no `forget_body`, and no pruning in the middle of the stack. It also
//! constrains the byte policy: **eviction must stay a prefix or a suffix drop.**
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
use crate::mask::{MaskBrick, MaskField};
use crate::volume::Volume;

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
///
/// # Three lists, not one, and the mask's is usually the empty one
///
/// A gesture can change the field, the mask, or the mask's polarity, and one
/// entry has to put back whatever it changed. They are kept apart rather than
/// folded into a single "brick" list because they cost two orders of magnitude
/// differently: a field brick is 131,104 B against a mask brick's 32,800, and
/// **a sculpt stroke through a mask records ZERO mask bytes** -- the mask is
/// read-only for the whole of one, so there is nothing of it to put back. That
/// is the property the storage decision in [`crate::mask`] was made for, and
/// `a_sculpt_stroke_through_a_mask_records_no_mask_bytes` is what holds it.
///
/// `polarity` is the value the mask's Invert bit held BEFORE the gesture, and
/// `None` means the gesture did not touch it. One bool rather than a rewritten
/// map is the whole reason Mask All and Invert are O(1) in undo as well as in
/// time and memory.
#[derive(Debug, Default)]
pub struct StrokeEdit {
    bricks: PriorBricks,
    masks: PriorMasks,
    polarity: Option<bool>,
    bytes: usize,
}

/// The prior contents of a set of field bricks. `None` is a brick that did not
/// exist, which undo puts back by removing it again.
type PriorBricks = Vec<(BrickCoord, Option<Brick>)>;

/// The same for mask bricks. Named alongside [`PriorBricks`] rather than
/// written out: the two differ by one word deep inside a nested generic, and
/// swapping them is a mistake the type system would catch and a reader would
/// not.
type PriorMasks = Vec<(BrickCoord, Option<MaskBrick>)>;

impl StrokeEdit {
    pub(crate) fn from_recording(
        bricks: FxHashMap<BrickCoord, Option<Brick>>,
        masks: FxHashMap<BrickCoord, Option<MaskBrick>>,
        polarity: Option<bool>,
    ) -> Option<Self> {
        if bricks.is_empty() && masks.is_empty() && polarity.is_none() {
            return None;
        }
        Some(Self::from_parts(bricks.into_iter().collect(), masks.into_iter().collect(), polarity))
    }

    /// An entry that puts back field bricks and nothing else.
    ///
    /// `#[cfg(test)]` because nothing in the shipping paths builds one any more:
    /// `end_stroke` and `apply_edit` both go through [`StrokeEdit::from_parts`]
    /// with all three lists, and a constructor that quietly drops the mask half
    /// is exactly the shape a future caller should not reach for.
    #[cfg(test)]
    pub(crate) fn from_bricks(bricks: PriorBricks) -> Self {
        Self::from_parts(bricks, Vec::new(), None)
    }

    pub(crate) fn from_parts(
        bricks: PriorBricks,
        masks: PriorMasks,
        polarity: Option<bool>,
    ) -> Self {
        let bytes = bricks.iter().map(|(_, brick)| prior_bytes(brick.as_ref())).sum::<usize>()
            + masks.iter().map(|(_, brick)| mask_prior_bytes(brick.as_ref())).sum::<usize>();
        Self { bricks, masks, polarity, bytes }
    }

    pub(crate) fn into_parts(self) -> (PriorBricks, PriorMasks, Option<bool>) {
        (self.bricks, self.masks, self.polarity)
    }

    /// Bricks this entry restores.
    #[inline]
    pub fn len(&self) -> usize {
        self.bricks.len()
    }

    /// Mask bricks this entry restores.
    #[inline]
    pub fn mask_len(&self) -> usize {
        self.masks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty() && self.masks.is_empty() && self.polarity.is_none()
    }

    /// Memory this entry holds, which is what the budget counts.
    #[inline]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// What one brick's prior contents cost the stroke budget.
///
/// `None` is a brick that did not exist, which costs the map entry and nothing
/// else -- and that is the whole reason a disjoint merge is nearly free where an
/// overlapping one is measured in gigabytes.
///
/// A free function rather than arithmetic written out at each site, because
/// [`crate::body::Document::merge_plan`] predicts a stroke's size from bricks it
/// has not recorded yet and the prediction is only worth having while it is
/// exact. Two copies of this formula would drift and nothing would say so.
#[inline]
pub(crate) fn prior_bytes(prior: Option<&Brick>) -> usize {
    size_of::<(BrickCoord, Option<Brick>)>() + prior.map_or(0, Brick::heap_bytes)
}

/// What one mask brick's prior contents cost the stroke budget.
///
/// The mask twin of [`prior_bytes`], and the numbers are the whole point of
/// keeping the two lists apart: 32 + 32,768 = **32,800 B** for a dense mask
/// brick against a field brick's 32 + 131,072 = 131,104. A mask stroke is a
/// quarter the undo weight of a sculpt stroke over the same bricks, and
/// `a_mask_entry_is_a_quarter_the_weight_of_a_sculpt_entry` pins both figures
/// rather than the ratio -- a ratio would pass on a mask brick that collapsed
/// to a tile, which costs nothing at all.
#[inline]
pub(crate) fn mask_prior_bytes(prior: Option<&MaskBrick>) -> usize {
    size_of::<(BrickCoord, Option<MaskBrick>)>() + prior.map_or(0, MaskBrick::heap_bytes)
}

/// What removing one node costs the reclaim allowance.
///
/// [`crate::Volume::stats`] and not `Brick::heap_bytes`: a brick knows nothing
/// about whole volumes, and summing it over an entry that holds one counts a
/// gigabyte as roughly zero -- which is the failure that lets a deleted dragon
/// sit in history reporting nothing. Shared with `merge_plan` for the reason
/// [`prior_bytes`] gives.
pub(crate) fn removed_node_bytes(node: &Node) -> usize {
    size_of::<Node>()
        + node.name.capacity()
        + node.volume().map_or(0, |volume| volume.stats().resident_bytes)
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
    /// A body's WHOLE mask was replaced. Applying puts the previous one back.
    ///
    /// Clear, Mask All and every absolute filter, and one variant for all of
    /// them because the thing they have in common is the thing that matters:
    /// the mask that came off is the ONLY copy of itself. It is MOVED in and
    /// never cloned, exactly as [`Change::NodeRemoved`] moves a node, so Clear
    /// allocates nothing however large the mask was.
    ///
    /// **Its bytes are charged to the reclaim allowance and never to the
    /// stroke budget**, for the reason the module doc gives about deleted
    /// bodies: this entry IS the storage, its size is known before the
    /// operation runs, and putting a gigabyte of protection against the 256 MB
    /// stroke budget would evict every stroke behind it.
    ///
    /// Polarity rides inside the field rather than beside it, which is what
    /// makes Mask All one change and not two.
    WholeMask { body: NodeId, mask: Box<MaskField> },
    /// A body's WHOLE field was replaced. Applying puts the previous one back.
    ///
    /// The transform gizmo's entry, and the shape follows [`Change::WholeMask`]
    /// exactly rather than by analogy: the field that came off IS the only copy
    /// of itself, [`crate::Volume`] does not implement `Clone` at all, so it is
    /// MOVED in and a bake allocates nothing to record itself.
    ///
    /// **Its bytes are charged to the reclaim allowance and never to the stroke
    /// budget.** A 765 MB field against the 256 MB stroke budget would evict
    /// every stroke behind it -- and the operation would then have destroyed
    /// the history it was trying to join. The reclaim allowance is the one that
    /// exists for "the entry is the storage, and its size is known before the
    /// operation runs", which is exactly this.
    ///
    /// **This is also what stops a transform clearing the history.**
    /// [`Document::rotate`] and [`Document::resample`] both make every older
    /// entry's brick coordinates meaningless, so their caller drops the whole
    /// stack; with the original field on the stack instead, undoing the bake
    /// makes those coordinates valid again and older strokes stay applicable.
    /// The invariant in the module doc holds unchanged: eviction is still a
    /// prefix or a suffix drop, and a body still only leaves through a
    /// `NodeRemoved`.
    WholeVolume { body: NodeId, volume: Box<Volume> },
    /// A row's name, eye, collapse or depth changed. Applying writes `before`.
    ///
    /// **The whole outline, both sides, over a FIXED id set.** That is what
    /// lets one variant cover rename, the eye, collapse, reorder, group and
    /// ungroup: a reparent is a permutation *plus* a depth edit, and neither
    /// half is expressible without the other. Its inverse is the pair swapped,
    /// which is why there is no second variant for "just the eye".
    ///
    /// A hundred and twenty-eight rows at some eighty bytes each is about
    /// 10 KB against a 256 MB budget. A group or an ungroup records this
    /// ALONGSIDE the `NodeAdded` or `NodeRemoved` that mints or retires the
    /// folder, never instead of it, and the order between them is worked
    /// through in [`crate::body::Document::group`].
    Outline { before: Vec<NodeMeta>, after: Vec<NodeMeta> },
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
            Change::WholeMask { body, mask } => {
                // Always applicable, by the same invariant `Change::Bricks`
                // rests on: a removal sits above every edit to the body it
                // removed, and eviction is a prefix or a suffix drop.
                let volume = doc
                    .volume_mut(body)
                    .expect("a mask change names a body that is still in the document");
                let previous = volume.replace_mask(*mask);
                Change::WholeMask { body, mask: Box::new(previous) }
            }
            Change::WholeVolume { body, volume } => {
                // Always applicable, by the same invariant `Change::Bricks`
                // rests on: a removal sits above every edit to the body it
                // removed, and eviction is a prefix or a suffix drop.
                let previous = doc
                    .replace_volume(body, *volume)
                    .expect("a field change names a body that is still in the document");
                // **Both sides marked dirty, and the outgoing side is the half
                // that is easy to forget.** The same pairing
                // [`crate::Volume::replace_mask`] makes, for a sharper reason:
                // the field that arrives brings its own dirty set, and that set
                // is EMPTY, because whatever remeshed it last drained it. So a
                // swap that marks nothing tells the renderer nothing, and the
                // screen goes on drawing the body where the change that is
                // being undone put it -- geometry and interaction in different
                // places, silently. The bricks the outgoing field occupied are
                // marked in the INCOMING one so that each of them remeshes to
                // an empty slice, which is also what hands its pool slot back.
                //
                // Not folded into [`Document::replace_volume`], which would be
                // the tidier home for it: the gizmo takes a field OUT by
                // swapping an empty placeholder in, and a blanket rule there
                // would mark the real field's bricks in a placeholder that is
                // about to be dropped, then mark nothing at all in the field
                // that replaces it. The dirty pairing belongs to the callers
                // that know which two fields are really being exchanged.
                let landed = doc
                    .volume_mut(body)
                    .expect("the field just swapped in is still in the document");
                landed.mark_everything_dirty();
                for coord in previous.brick_coords() {
                    landed.mark_dirty(coord);
                }
                Change::WholeVolume { body, volume: Box::new(previous) }
            }
            Change::Outline { before, after } => {
                debug_assert!(
                    same_ids(&before, &after),
                    "an outline change is a permutation over a fixed id set"
                );
                doc.set_outline(&before);
                // The inverse is the pair swapped, which applies `after`.
                Change::Outline { before: after, after: before }
            }
        }
    }

    /// The node this change is about.
    ///
    /// For an outline change that is the first row it actually moved or
    /// edited, and not simply the first row in the document: what the caller
    /// does with this is name it in a status line, and "renamed Body 1" for a
    /// rename of Body 7 is worse than saying nothing.
    fn node(&self) -> NodeId {
        match self {
            Change::Bricks { body, .. } => *body,
            Change::NodeAdded { id, .. } => *id,
            Change::NodeRemoved { node, .. } => node.id,
            Change::WholeMask { body, .. } => *body,
            Change::WholeVolume { body, .. } => *body,
            Change::Outline { before, after } => {
                before.iter().zip(after).find(|(was, now)| was != now).map_or_else(
                    || before.first().map_or(NodeId(0), |meta| meta.id),
                    |(was, _)| was.id,
                )
            }
        }
    }

    /// What this change costs the stroke budget.
    fn stroke_bytes(&self) -> usize {
        match self {
            Change::Bricks { edit, .. } => edit.bytes(),
            // Both hold a whole thing that was moved out of the document
            // rather than copied out of it. See `reclaim_bytes`.
            Change::NodeRemoved { .. } | Change::WholeMask { .. } | Change::WholeVolume { .. } => 0,
            Change::NodeAdded { .. } => size_of::<Self>(),
            Change::Outline { before, after } => {
                before.iter().chain(after).map(NodeMeta::bytes).sum()
            }
        }
    }

    /// What this change costs the reclaim allowance.
    ///
    /// See [`removed_node_bytes`], which is shared with the walk that predicts
    /// a merge's size before the merge happens.
    fn reclaim_bytes(&self) -> usize {
        match self {
            Change::NodeRemoved { node, .. } => removed_node_bytes(node),
            Change::WholeMask { mask, .. } => size_of::<MaskField>() + mask.bytes(),
            // `resident_bytes`, the same measure `removed_node_bytes` takes of
            // a deleted body's field, so a bake and a delete are charged in one
            // currency against one allowance.
            Change::WholeVolume { volume, .. } => {
                size_of::<Volume>() + volume.stats().resident_bytes
            }
            _ => 0,
        }
    }
}

/// Whether two outline snapshots name the same rows, in any order.
///
/// A `debug_assert` and not a refusal: an outline change that added or dropped
/// an id would be a structural edit wearing a permutation's clothes, and the
/// two sides of it could then never be each other's inverse. Sorted rather than
/// hashed because [`crate::body::MAX_NODES`] is 128 and this runs at an undo
/// press.
fn same_ids(before: &[NodeMeta], after: &[NodeMeta]) -> bool {
    let ids = |metas: &[NodeMeta]| {
        let mut ids: Vec<NodeId> = metas.iter().map(|meta| meta.id).collect();
        ids.sort_unstable();
        ids
    };
    ids(before) == ids(after)
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

    /// How many changes one gesture turned out to be.
    ///
    /// The number a caller checks when the whole promise of an operation is
    /// that it is ONE entry however many rows it touched -- a split into forty
    /// parts is forty `NodeAdded` and one `NodeRemoved`, and an assertion on
    /// the body count alone would pass just as well for forty entries.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.changes.len()
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
    /// **Two of the five changes are deliberately NOT gated, and both reasons
    /// were arrived at by trying it the other way.**
    ///
    /// `NodeRemoved` restores a node that is not in the document, so there is
    /// no resolved visibility to read and its own eye bit is the only input.
    /// Gating on that bit deadlocks: the body cannot be un-hidden, because it
    /// is not in the tree to click, so the entry can never be applied and it
    /// blocks every older entry behind it for the rest of the session.
    ///
    /// `Change::Outline` is how the eye itself is undone. Gating it means that
    /// undoing a hide is refused *because of the hide*, and the user is told to
    /// reveal the body by hand -- which is the very thing they pressed ctrl+Z
    /// to do. It is also visible in the panel whatever the eye says, so nothing
    /// is happening off screen.
    fn blocked_by(&self, doc: &Document, visible: &[bool]) -> Option<NodeId> {
        self.changes.iter().find_map(|change| {
            let id = match change {
                Change::Bricks { body, .. } => *body,
                Change::WholeMask { body, .. } => *body,
                Change::WholeVolume { body, .. } => *body,
                Change::NodeAdded { id, .. } => *id,
                Change::NodeRemoved { .. } | Change::Outline { .. } => return None,
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
    /// The gestures that have been undone, oldest-undone FIRST.
    ///
    /// A `VecDeque` and not a `Vec` because both ends are used: the back is the
    /// next entry to redo, and the front is the one furthest into the future,
    /// which is the end [`History::trim`] evicts from.
    redo: VecDeque<Entry>,
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
            redo: VecDeque::new(),
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

    /// Drop the entries furthest from where the user is standing, while EITHER
    /// allowance is over.
    ///
    /// **The entry at each end of the cursor is kept whatever it costs**, and
    /// the two protections are the same rule rather than two: dropping
    /// `undo.back()` means the user's last action cannot be undone, dropping
    /// `redo.back()` means the undo they just pressed cannot be taken back.
    /// Both are a silent loss of the gesture they are standing on, and both are
    /// worse than briefly exceeding a soft ceiling. Hence the guard is "while
    /// EITHER stack holds more than one entry" rather than a count of the two
    /// together: an over-budget pair of one undo and one redo is a state this
    /// deliberately settles in.
    ///
    /// **The undo stack is evicted oldest-first and nothing else, ever.** The
    /// module doc's invariant -- that a `Bricks` change is always applicable --
    /// holds only because eviction is a prefix drop; taking the largest entry
    /// out of the middle instead would leave brick edits above a removal that
    /// is no longer there.
    ///
    /// # Why it also pops the redo stack, and from the FRONT
    ///
    /// Read the two stacks as one timeline with a cursor between them: the undo
    /// stack is the past in order, and the redo stack is the future with its
    /// BACK nearest the cursor. So `redo.front()` is the entry furthest into
    /// the future, and dropping it is the exact mirror of dropping the oldest
    /// undo entry -- a suffix drop, which keeps every entry that remains
    /// applicable in order for the same reason the prefix drop does.
    ///
    /// Nothing evicted the redo stack before this, and it is not a hypothetical
    /// leak. Any entry whose INVERSE is larger than itself grows the history
    /// permanently: a stroke into empty space records `None` per brick, 32
    /// bytes, and hands back the 128 KB bricks it created. Masking makes that
    /// the normal case rather than the odd one, and two hundred such strokes
    /// undone is a redo stack of roughly a gigabyte that nothing would ever
    /// reclaim.
    fn trim(&mut self) {
        while (self.bytes > self.budget || self.reclaim_bytes > self.reclaim_budget)
            && (self.undo.len() > 1 || self.redo.len() > 1)
        {
            // The oldest past first, and only then the furthest future: the
            // user is more likely to reach for one more undo than for the redo
            // at the far end of a run they have already walked away from.
            //
            // The loop guard is what makes the second arm safe: reaching it
            // means the undo stack is down to its protected entry, so the redo
            // stack is the one holding more than one and `pop_front` cannot be
            // the entry nearest the cursor.
            let dropped =
                if self.undo.len() > 1 { self.undo.pop_front() } else { self.redo.pop_front() };
            let Some(dropped) = dropped else {
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
        self.apply(entry, doc, |history, inverse| history.redo.push_back(inverse))
    }

    /// Redo the most recently undone gesture, under the same refusal.
    pub fn redo(&mut self, doc: &mut Document, visible: &[bool]) -> UndoOutcome {
        debug_assert_eq!(
            visible.len(),
            doc.node_count(),
            "the visibility mask is indexed by node position and must cover every node"
        );
        let Some(entry) = self.redo.back() else {
            return UndoOutcome::Nothing;
        };
        if let Some(hidden) = entry.blocked_by(doc, visible) {
            return UndoOutcome::Refused(hidden);
        }
        let entry = self.redo.pop_back().expect("checked just above");
        self.apply(entry, doc, |history, inverse| history.undo.push_back(inverse))
    }

    /// The half undo and redo share: swap the entry into the document and put
    /// its inverse on the other stack, keeping both counters straight.
    ///
    /// **Both directions trim, and the reason is that an inverse is NOT the
    /// same bytes.** This used to say it was, and that was wrong in the one
    /// direction that matters: a stroke that CREATES bricks records `None` for
    /// each of them -- 32 bytes -- and its inverse holds the 128 KB bricks the
    /// stroke made. Undoing such a stroke therefore grows the history rather
    /// than moving it, and nothing was there to notice. See [`History::trim`]
    /// for which end it evicts from and why that end is safe.
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
        self.trim();
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
    use std::collections::HashSet;

    fn sculpted() -> (Document, Brush, BrushScratch) {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 24.0);
        let brush = Brush { kind: BrushKind::Draw, radius: 8.0, strength: 0.4, ..Brush::default() };
        (doc, brush, BrushScratch::new())
    }

    /// One MASK stroke on the active body, as one entry.
    fn mask_stroke(
        doc: &mut Document,
        brush: &Brush,
        scratch: &mut BrushScratch,
        at: Vec3,
        op: crate::MaskOp,
    ) -> Entry {
        let body = doc.active();
        let volume = doc.volume_mut(body).expect("a body to mask");
        volume.begin_stroke();
        let normal = volume.gradient_world(at);
        brush.apply_mask(volume, &Stamp::new(at, normal, BrushDirection::Add), op, scratch);
        Entry::stroke(body, volume.end_stroke().expect("the stroke changed something"))
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

    /// **A long undo run must not grow history without bound.**
    ///
    /// A stroke that CREATES bricks records `None` for each of them -- 32 bytes
    /// -- and its inverse holds the bricks it made, at 128 KB apiece. So undoing
    /// one grows the history rather than moving it, and until `trim` learned to
    /// pop the redo stack nothing ever reclaimed that: two hundred such strokes
    /// undone is about a gigabyte no budget could see. A draw into empty space
    /// has always had this shape; a mask stroke over previously-unmasked bricks
    /// makes it the ordinary one.
    ///
    /// Constructed rather than sculpted, because the SHAPE of the entry is the
    /// whole point: a brush that happened to pass over an existing brick would
    /// record its 128 KB on the way in and prove nothing.
    #[test]
    fn undoing_a_run_of_brick_creating_strokes_stays_inside_the_budget() {
        const STROKES: usize = 40;
        const BUDGET: usize = 512 * 1024;

        let mut doc = Document::new(1.0);
        let body = doc.active();
        let mut history = History::new(BUDGET);
        for index in 0..STROKES {
            let coord = BrickCoord::new(index as i32, 0, 0);
            let volume = doc.volume_mut(body).expect("the starting body");
            volume.insert_brick(coord, Brick::dense_filled(0.5));
            history.push(Entry::stroke(body, StrokeEdit::from_bricks(vec![(coord, None)])));
        }
        assert_eq!(history.stats().undo_entries, STROKES, "a push evicted something");
        assert!(
            history.stats().bytes <= BUDGET,
            "the pushes alone are {} bytes, so the fixture proves nothing",
            history.stats().bytes
        );

        for _ in 0..STROKES {
            undo(&mut history, &mut doc);
        }

        let stats = history.stats();
        assert!(
            stats.bytes <= BUDGET || stats.undo_entries + stats.redo_entries == 1,
            "{} entries holding {} bytes against a {BUDGET} byte budget",
            stats.undo_entries + stats.redo_entries,
            stats.bytes
        );
        assert!(stats.dropped > 0, "nothing was evicted, so the budget was never reached");
        assert!(history.can_redo(), "the redo nearest the user was evicted");
    }

    /// The end of the redo stack that gets evicted is the one furthest from the
    /// user, so what survives a run of undos is the redo they would press next.
    ///
    /// Getting this backwards would be worse than not evicting at all: pressing
    /// redo would replay a gesture from the middle of the run and skip the one
    /// the user was standing on.
    #[test]
    fn eviction_takes_the_far_end_of_the_redo_stack_and_leaves_the_near_one() {
        // Three brick-creating entries, and a budget that holds their `None`
        // priors easily and two of their 128 KB inverses not at all.
        const BUDGET: usize = 200 * 1024;

        let mut doc = Document::new(1.0);
        let body = doc.active();
        let coords = [BrickCoord::new(0, 0, 0), BrickCoord::new(1, 0, 0), BrickCoord::new(2, 0, 0)];
        let mut history = History::new(BUDGET);
        for coord in coords {
            let volume = doc.volume_mut(body).expect("the starting body");
            volume.insert_brick(coord, Brick::dense_filled(0.5));
            history.push(Entry::stroke(body, StrokeEdit::from_bricks(vec![(coord, None)])));
        }
        assert_eq!(history.stats().undo_entries, 3, "a push evicted something");

        for _ in 0..coords.len() {
            applied(undo(&mut history, &mut doc));
        }
        let volume = doc.volume(body).expect("the starting body");
        for coord in coords {
            assert!(volume.brick(coord).is_none(), "undo left {coord:?} behind");
        }

        let stats = history.stats();
        assert_eq!(stats.redo_entries, 1, "the squeeze kept the wrong number of redos");
        assert_eq!(stats.dropped, 2);

        applied(redo(&mut history, &mut doc));
        let volume = doc.volume(body).expect("the starting body");
        assert!(
            volume.brick(coords[0]).is_some(),
            "the surviving redo was not the one nearest the user"
        );
        assert!(volume.brick(coords[1]).is_none(), "a later gesture was replayed out of order");
        assert!(volume.brick(coords[2]).is_none());
        assert!(!history.can_redo(), "only one redo should have survived");
    }

    /// **A single undo must never eat the redo it just created.** The gesture
    /// the user is standing on is undoable and redoable at both ends of the
    /// cursor, and no budget is worth taking either away silently.
    ///
    /// The shape that used to break it: history near the ceiling, holding one
    /// small older entry and one brick-creating stroke. Undoing the stroke
    /// hands back the 128 KB bricks it made, which puts the total over -- and
    /// with only one undo entry left to protect, `trim` reached for the redo
    /// stack and took the one entry on it. Ctrl+Y then did nothing, while the
    /// smaller, older undo entry it could have dropped instead was kept.
    ///
    /// The two tests above cannot see this: both leave a run of several redos,
    /// so the last-one rule never comes up.
    #[test]
    fn one_undo_of_an_over_budget_stroke_still_leaves_something_to_redo() {
        // Well under one 128 KB inverse, so the undo is guaranteed to go over.
        const BUDGET: usize = 100 * 1024;

        let mut doc = Document::new(1.0);
        let body = doc.active();
        let older = BrickCoord::new(0, 0, 0);
        let newest = BrickCoord::new(1, 0, 0);
        let mut history = History::new(BUDGET);
        for coord in [older, newest] {
            let volume = doc.volume_mut(body).expect("the starting body");
            volume.insert_brick(coord, Brick::dense_filled(0.5));
            history.push(Entry::stroke(body, StrokeEdit::from_bricks(vec![(coord, None)])));
        }
        assert_eq!(history.stats().undo_entries, 2, "a push evicted something");

        applied(undo(&mut history, &mut doc));

        assert!(
            history.can_redo(),
            "the only redo was evicted by the undo that made it: {:?}",
            history.stats()
        );
        // And it is the right one: redoing puts back exactly the brick the undo
        // took away, rather than replaying something out of order.
        applied(redo(&mut history, &mut doc));
        let volume = doc.volume(body).expect("the starting body");
        assert!(volume.brick(newest).is_some(), "the redo did not put the stroke back");
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
        let outline_before = doc.outline();
        doc.set_meta(&NodeMeta { visible: false, ..before.clone() });

        let mut history = History::default();
        history.push(Entry::new(vec![Change::Outline {
            before: outline_before,
            after: doc.outline(),
        }]));

        let mut shown = all_shown(&doc);
        shown[doc.index_of(other).expect("the second body")] = false;
        assert_eq!(history.undo(&mut doc, &shown), UndoOutcome::Applied(other));
        assert!(doc.node(other).expect("the second body").visible, "the eye was not put back");
    }

    // --- the mask half of an entry -------------------------------------------

    /// A mask entry weighs a quarter of a sculpt entry, **to the byte**.
    ///
    /// Both numbers rather than the ratio between them, and that is not
    /// pedantry: a ratio passes on a mask brick that collapsed to a tile, which
    /// costs no heap at all, and it passes on an entry that recorded no mask
    /// bricks. 32 + 32,768 against 32 + 131,072 is the arithmetic the whole
    /// "+25% storage" argument for eight bits rests on, and it is checked here
    /// because this is where the undo budget reads it.
    #[test]
    fn a_mask_entry_is_a_quarter_the_weight_of_a_sculpt_entry() {
        let dense = MaskBrick::dense_filled(200);
        assert_eq!(mask_prior_bytes(Some(&dense)), 32_800);
        assert_eq!(mask_prior_bytes(Some(&MaskBrick::Uniform(200))), 32);
        assert_eq!(mask_prior_bytes(None), 32);

        let field = Brick::Dense(Box::new([0.0; crate::BRICK_VOXELS]));
        assert_eq!(prior_bytes(Some(&field)), 131_104);
    }

    /// **A sculpt stroke records ZERO mask bytes**, and that is what the whole
    /// storage decision was made for.
    ///
    /// The mask is read-only for the length of a sculpt stroke, so there is
    /// nothing of it to put back. Had the mask lived inside `Brick`, every
    /// ordinary carve would have snapshotted 32 KB per brick of a mask it never
    /// wrote -- and `record_for_undo` clones the whole brick, so nothing would
    /// have said so.
    #[test]
    fn a_sculpt_stroke_through_a_mask_records_no_mask_bytes() {
        let (mut doc, brush, mut scratch) = sculpted();
        let at = Vec3::new(24.0, 0.0, 0.0);
        let masking = mask_stroke(&mut doc, &brush, &mut scratch, at, crate::MaskOp::Raise);
        let mask_bricks = match masking.changes.first() {
            Some(Change::Bricks { edit, .. }) => edit.mask_len(),
            _ => panic!("a mask stroke is one brick change"),
        };
        assert!(mask_bricks > 0, "the fixture recorded no mask at all");

        // Somewhere else on the sphere, so the carve is not refused by the very
        // mask that was just painted.
        let carve = stroke(&mut doc, &brush, &mut scratch, Vec3::new(-24.0, 0.0, 0.0));
        match carve.changes.first() {
            Some(Change::Bricks { edit, .. }) => {
                assert_ne!(edit.len(), 0, "the fixture carved nothing");
                assert_eq!(edit.mask_len(), 0, "a sculpt stroke recorded mask bricks");
            }
            _ => panic!("a sculpt stroke is one brick change"),
        }
    }

    /// Undoing a mask stroke puts the protection back bit for bit, and marks
    /// bricks dirty so the picture follows.
    ///
    /// The second half is the one that would go unnoticed: the mask is baked
    /// into a vertex attribute at mesh time, so an undo that restored the bytes
    /// and marked nothing would leave the old tint on screen over the restored
    /// mask -- a state where the model and the document disagree and nothing
    /// says which is right.
    #[test]
    fn undoing_a_mask_stroke_restores_it_and_marks_bricks_dirty() {
        let (mut doc, brush, mut scratch) = sculpted();
        let at = Vec3::new(24.0, 0.0, 0.0);
        let body = doc.active();
        doc.volume_mut(body).expect("a body").take_dirty(&mut Vec::new());

        let before = doc.volume(body).expect("a body").mask_fill();
        let entry = mask_stroke(&mut doc, &brush, &mut scratch, at, crate::MaskOp::Raise);
        let after = doc.volume(body).expect("a body").mask_fill();
        assert!(after > before, "the fixture painted nothing: {before} then {after}");

        let sampled: Vec<u8> = (0..40)
            .map(|step| {
                let cell = glam::IVec3::new(24 - step / 4, step % 4, 0);
                doc.volume(body).expect("a body").mask().at(cell)
            })
            .collect();

        doc.volume_mut(body).expect("a body").take_dirty(&mut Vec::new());
        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(entry);
        let shown = all_shown(&doc);
        history.undo(&mut doc, &shown);

        let mut dirty = Vec::new();
        doc.volume_mut(body).expect("a body").take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "undoing a mask stroke marked nothing for a remesh");
        assert_eq!(
            doc.volume(body).expect("a body").mask_fill(),
            before,
            "undo did not put the mask back"
        );

        history.redo(&mut doc, &shown);
        let again: Vec<u8> = (0..40)
            .map(|step| {
                let cell = glam::IVec3::new(24 - step / 4, step % 4, 0);
                doc.volume(body).expect("a body").mask().at(cell)
            })
            .collect();
        assert_eq!(again, sampled, "redo did not put the mask back bit for bit");
    }

    // --- the whole-mask verbs --------------------------------------------

    /// A body with a real painted mask on it, and the mask's own byte count.
    fn masked() -> (Document, usize) {
        let (mut doc, brush, mut scratch) = sculpted();
        let body = doc.active();
        let volume = doc.volume_mut(body).expect("a body to mask");
        for step in 0..6 {
            let at = Vec3::new(24.0, step as f32 * 3.0 - 7.5, 0.0);
            let normal = volume.gradient_world(at);
            brush.apply_mask(
                volume,
                &Stamp::new(at, normal, BrushDirection::Add),
                crate::MaskOp::Raise,
                &mut scratch,
            );
        }
        volume.mask_mut().collapse();
        let bytes = volume.mask().bytes();
        assert!(bytes > 0, "the fixture painted no mask");
        (doc, bytes)
    }

    /// **Clear costs the reclaim allowance and not one byte of the stroke
    /// budget**, because the map it holds was MOVED out of the document rather
    /// than copied out of it. Charging it to the stroke budget would evict
    /// every stroke behind it on a large mask.
    #[test]
    fn clearing_a_mask_charges_the_reclaim_allowance_and_not_the_stroke_budget() {
        let (mut doc, bytes) = masked();
        let body = doc.active();
        let volume = doc.volume_mut(body).expect("a body");
        let cleared = volume.mask().cleared(false);
        let old = volume.replace_mask(cleared);

        let entry = Entry::new(vec![Change::WholeMask { body, mask: Box::new(old) }]);
        assert_eq!(entry.stroke_bytes(), 0, "a moved mask was charged to the stroke budget");
        assert!(
            entry.reclaim_bytes() >= bytes,
            "the reclaim allowance is not counting the {bytes} bytes it is holding"
        );
    }

    /// Clear gives the memory back, and undoing it puts the mask back bit for
    /// bit -- the same field object, not a rebuilt one.
    #[test]
    fn clearing_a_mask_returns_resident_bytes_and_undoing_it_restores_it_bit_for_bit() {
        let (mut doc, _) = masked();
        let body = doc.active();
        let masked_bytes = doc.totals().resident_bytes;

        let sampled: Vec<u8> = (0..64)
            .map(|step| {
                let cell = glam::IVec3::new(24 - step / 8, step % 8 - 4, 0);
                doc.volume(body).expect("a body").mask().at(cell)
            })
            .collect();
        assert!(sampled.iter().any(|value| *value > 0), "the samples miss the mask entirely");

        // What the document weighed before anything was painted: the same
        // volume with an empty mask on it.
        let bare = {
            let volume = doc.volume_mut(body).expect("a body");
            let cleared = volume.mask().cleared(false);
            let old = volume.replace_mask(cleared);
            let bare = doc.totals().resident_bytes;
            let volume = doc.volume_mut(body).expect("a body");
            volume.replace_mask(old);
            bare
        };
        assert!(bare < masked_bytes, "the fixture's mask weighs nothing");

        let volume = doc.volume_mut(body).expect("a body");
        let cleared = volume.mask().cleared(false);
        let old = volume.replace_mask(cleared);
        assert_eq!(
            doc.totals().resident_bytes,
            bare,
            "clearing the mask did not give its bytes back"
        );

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::new(vec![Change::WholeMask { body, mask: Box::new(old) }]));
        undo(&mut history, &mut doc);

        let restored: Vec<u8> = (0..64)
            .map(|step| {
                let cell = glam::IVec3::new(24 - step / 8, step % 8 - 4, 0);
                doc.volume(body).expect("a body").mask().at(cell)
            })
            .collect();
        assert_eq!(restored, sampled, "undoing a clear did not put the mask back bit for bit");
        assert_eq!(doc.totals().resident_bytes, masked_bytes);
    }

    /// Mask All is clear-then-invert as ONE change, so undoing it puts the map
    /// and the polarity back together -- the only state either is meaningful in.
    #[test]
    fn undoing_mask_all_puts_the_map_and_the_polarity_back_together() {
        let (mut doc, _) = masked();
        let body = doc.active();
        let before = doc.volume(body).expect("a body").mask_fill();

        let volume = doc.volume_mut(body).expect("a body");
        let all = volume.mask().cleared(true);
        let old = volume.replace_mask(all);
        assert!(volume.mask().protects_everything());

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::new(vec![Change::WholeMask { body, mask: Box::new(old) }]));
        undo(&mut history, &mut doc);

        let volume = doc.volume(body).expect("a body");
        assert!(!volume.mask().inverted(), "undo left the polarity inverted");
        assert!((volume.mask_fill() - before).abs() < 1.0e-9, "undo did not put the map back");

        redo(&mut history, &mut doc);
        assert!(
            doc.volume(body).expect("a body").mask().protects_everything(),
            "redo did not put Mask All back"
        );
    }

    /// **Invert costs nothing at all, in either direction.** One bool in, one
    /// bool out, no bricks marked and no bytes against either budget -- which
    /// is what stops ctrl+I allocating 1.04 GiB on a lightly masked model.
    #[test]
    fn inverting_a_mask_costs_no_bytes_and_marks_no_bricks_in_either_direction() {
        let (mut doc, _) = masked();
        let body = doc.active();
        doc.volume_mut(body).expect("a body").take_dirty(&mut Vec::new());

        let edit = doc.volume_mut(body).expect("a body").flip_mask_polarity();
        assert_eq!(edit.bytes(), 0, "a polarity flip recorded brick bytes");
        assert!(doc.volume(body).expect("a body").mask().inverted());

        let mut dirty = Vec::new();
        doc.volume_mut(body).expect("a body").take_dirty(&mut dirty);
        assert!(dirty.is_empty(), "a polarity flip marked {} bricks for a remesh", dirty.len());

        let entry = Entry::stroke(body, edit);
        assert_eq!(entry.stroke_bytes(), 0);
        assert_eq!(entry.reclaim_bytes(), 0);

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(entry);
        undo(&mut history, &mut doc);
        assert!(!doc.volume(body).expect("a body").mask().inverted(), "undo did not flip it back");

        let mut dirty = Vec::new();
        doc.volume_mut(body).expect("a body").take_dirty(&mut dirty);
        assert!(
            dirty.is_empty(),
            "undoing a polarity flip marked {} bricks, which is a 475 ms remesh at the ceiling",
            dirty.len()
        );
    }

    /// Replacing a mask marks the bricks either side of the swap, or the tint
    /// that was there stays on screen over a mask that is gone.
    #[test]
    fn replacing_a_mask_marks_the_bricks_either_side_of_the_swap() {
        let (mut doc, _) = masked();
        let body = doc.active();
        doc.volume_mut(body).expect("a body").take_dirty(&mut Vec::new());

        let volume = doc.volume_mut(body).expect("a body");
        let cleared = volume.mask().cleared(false);
        volume.replace_mask(cleared);

        let mut dirty = Vec::new();
        doc.volume_mut(body).expect("a body").take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "clearing the mask marked nothing for a remesh");
    }

    /// A mask change on a hidden body is refused like any other edit to one.
    #[test]
    fn a_whole_mask_change_on_a_hidden_body_is_refused_rather_than_applied() {
        let (mut doc, _) = masked();
        let body = doc.active();
        let volume = doc.volume_mut(body).expect("a body");
        let cleared = volume.mask().cleared(false);
        let old = volume.replace_mask(cleared);

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::new(vec![Change::WholeMask { body, mask: Box::new(old) }]));
        let hidden = vec![false; doc.node_count()];
        assert_eq!(history.undo(&mut doc, &hidden), UndoOutcome::Refused(body));
        assert!(
            doc.volume(body).expect("a body").mask().is_free(),
            "the refused undo put the mask back anyway"
        );
    }

    // ------------------------------------------------- a whole field replaced

    /// The gizmo's entry follows [`Change::WholeMask`]'s byte policy exactly,
    /// and the reason is the same one written larger: a 765 MB field against
    /// the 256 MB stroke budget would evict every stroke behind it, so the bake
    /// would destroy the history it was trying to join.
    #[test]
    fn a_whole_volume_change_is_charged_to_reclaim_and_not_the_stroke_budget() {
        let (mut doc, _, _) = sculpted();
        let body = doc.active();
        let resident = doc.volume(body).expect("a body").stats().resident_bytes;
        assert!(resident > 0, "the fixture holds no field");

        let moved = doc.volume(body).expect("a body").shifted(glam::IVec3::new(3, 0, 0));
        let previous = doc.replace_volume(body, moved).expect("the body is in the document");

        let entry = Entry::new(vec![Change::WholeVolume { body, volume: Box::new(previous) }]);
        assert_eq!(entry.stroke_bytes(), 0, "a moved field was charged to the stroke budget");
        assert!(
            entry.reclaim_bytes() >= resident,
            "the reclaim allowance is not counting the {resident} bytes it is holding"
        );
    }

    /// Undoing a bake has to give the field back bit for bit, because a bake is
    /// the one operation whose forward direction is lossy: "turn it back" is
    /// not a recovery path for a free-angle transform the way it is for a
    /// quarter turn.
    #[test]
    fn undoing_a_bake_puts_the_field_back_bit_for_bit() {
        let (mut doc, _, _) = sculpted();
        let body = doc.active();

        let sampled: Vec<f32> = (0..64)
            .map(|step| {
                let at = Vec3::new(20.0 - step as f32 * 0.5, step as f32 * 0.25 - 8.0, 0.0);
                doc.volume(body).expect("a body").sample_world(at)
            })
            .collect();

        // A free-angle turn, so the forward direction really is destructive.
        let placement =
            crate::Similarity::about(Vec3::ZERO, glam::Quat::from_rotation_y(0.4), 1.0, Vec3::ZERO);
        let baked = doc.volume(body).expect("a body").warped(placement);
        let previous = doc.replace_volume(body, baked).expect("the body is in the document");

        let after: Vec<f32> = (0..64)
            .map(|step| {
                let at = Vec3::new(20.0 - step as f32 * 0.5, step as f32 * 0.25 - 8.0, 0.0);
                doc.volume(body).expect("a body").sample_world(at)
            })
            .collect();
        assert_ne!(after, sampled, "the fixture's transform changed nothing");

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::new(vec![Change::WholeVolume { body, volume: Box::new(previous) }]));
        undo(&mut history, &mut doc);

        let restored: Vec<f32> = (0..64)
            .map(|step| {
                let at = Vec3::new(20.0 - step as f32 * 0.5, step as f32 * 0.25 - 8.0, 0.0);
                doc.volume(body).expect("a body").sample_world(at)
            })
            .collect();
        assert_eq!(restored, sampled, "undoing a bake did not put the field back");

        redo(&mut history, &mut doc);
        let again: Vec<f32> = (0..64)
            .map(|step| {
                let at = Vec3::new(20.0 - step as f32 * 0.5, step as f32 * 0.25 - 8.0, 0.0);
                doc.volume(body).expect("a body").sample_world(at)
            })
            .collect();
        assert_eq!(again, after, "redo did not put the transform back");
    }

    /// **Putting the field back is only half of undoing a bake; saying so is
    /// the other half, and it is the half that was missing.**
    ///
    /// A field arrives on the stack with an EMPTY dirty set, because whatever
    /// remeshed it last drained it. So a swap that marks nothing leaves the
    /// document holding the original body and the screen holding the moved one
    /// -- and every subsequent stroke, pick and brush ring then acts on a
    /// position the user cannot see. Asserting on the field alone is exactly
    /// what let that through, so this asserts on the dirty set instead, and on
    /// BOTH sides of the swap: the bricks the field arriving occupies, and the
    /// bricks the field leaving vacates, which have to remesh to nothing.
    #[test]
    fn undoing_a_bake_marks_the_bricks_both_fields_touched() {
        let (mut doc, _, _) = sculpted();
        let body = doc.active();

        let before: HashSet<BrickCoord> =
            doc.volume(body).expect("a body").brick_coords().collect();
        // Far enough to clear the original footprint entirely, so "the vacated
        // bricks" is a set the assertion below can actually be about.
        let moved = doc.volume(body).expect("a body").shifted(glam::IVec3::splat(96));
        let after: HashSet<BrickCoord> = moved.brick_coords().collect();
        assert!(
            before.is_disjoint(&after),
            "the fixture did not move the body clear of where it was"
        );
        let previous = doc.replace_volume(body, moved).expect("the body is in the document");

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::new(vec![Change::WholeVolume { body, volume: Box::new(previous) }]));

        // Drained first, so what the assertion sees is what the undo marked.
        let mut dirty = Vec::new();
        doc.take_dirty(&mut dirty);
        undo(&mut history, &mut doc);
        doc.take_dirty(&mut dirty);
        let marked: HashSet<BrickCoord> =
            dirty.iter().filter(|(id, _)| *id == body).map(|(_, coord)| *coord).collect();

        assert!(
            before.is_subset(&marked),
            "the field going back in was not scheduled for a remesh, so it stays invisible"
        );
        assert!(
            after.is_subset(&marked),
            "the bricks the moved field vacated were not scheduled, so its triangles stay on \
             screen and its pool slices are never handed back"
        );
    }

    /// The same for redo, which walks the identical code with the two fields
    /// the other way round -- and would fail the identical way.
    #[test]
    fn redoing_a_bake_marks_the_bricks_both_fields_touched() {
        let (mut doc, _, _) = sculpted();
        let body = doc.active();

        let before: HashSet<BrickCoord> =
            doc.volume(body).expect("a body").brick_coords().collect();
        let moved = doc.volume(body).expect("a body").shifted(glam::IVec3::splat(96));
        let after: HashSet<BrickCoord> = moved.brick_coords().collect();
        let previous = doc.replace_volume(body, moved).expect("the body is in the document");

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::new(vec![Change::WholeVolume { body, volume: Box::new(previous) }]));
        undo(&mut history, &mut doc);

        let mut dirty = Vec::new();
        doc.take_dirty(&mut dirty);
        redo(&mut history, &mut doc);
        doc.take_dirty(&mut dirty);
        let marked: HashSet<BrickCoord> =
            dirty.iter().filter(|(id, _)| *id == body).map(|(_, coord)| *coord).collect();

        assert!(before.is_subset(&marked), "redo left the vacated bricks on screen");
        assert!(after.is_subset(&marked), "redo did not schedule the field it put back");
    }

    /// **A bake does not clear the undo history.** `Document::rotate` and
    /// `Document::resample` both leave every older entry naming bricks of a
    /// volume that no longer exists, so their caller drops the whole stack; the
    /// gizmo puts the original field ON the stack instead, so undoing the bake
    /// makes those older coordinates valid again and the stroke behind it still
    /// applies.
    #[test]
    fn a_stroke_behind_a_bake_is_still_undoable_afterwards() {
        let (mut doc, brush, mut scratch) = sculpted();
        let body = doc.active();

        let at = Vec3::new(0.0, 25.0, 0.0);
        let volume = doc.volume_mut(body).expect("a body");
        volume.begin_stroke();
        let normal = volume.gradient_world(at);
        for _ in 0..4 {
            brush.apply(volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        }
        let edit = volume.end_stroke().expect("the stroke changed something");
        let after_stroke = doc.volume(body).expect("a body").sample_world(at);

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::stroke(body, edit));

        let moved = doc.volume(body).expect("a body").shifted(glam::IVec3::new(4, 0, 0));
        let previous = doc.replace_volume(body, moved).expect("the body is in the document");
        history.push(Entry::new(vec![Change::WholeVolume { body, volume: Box::new(previous) }]));

        // Undo the bake, then the stroke behind it. The second one is the
        // claim: it names bricks of the field the bake replaced.
        assert!(matches!(undo(&mut history, &mut doc), UndoOutcome::Applied(_)));
        assert!(
            (doc.volume(body).expect("a body").sample_world(at) - after_stroke).abs() < 1.0e-6,
            "undoing the bake did not restore the sculpted field"
        );
        assert!(matches!(undo(&mut history, &mut doc), UndoOutcome::Applied(_)));
        assert!(
            doc.volume(body).expect("a body").sample_world(at) > after_stroke,
            "the stroke behind the bake did not come off"
        );
    }

    /// A bake on a hidden body is refused for the same reason a stroke on one
    /// is: the field would change with nothing on screen saying so.
    #[test]
    fn undoing_a_bake_on_a_hidden_body_is_refused() {
        let (mut doc, _, _) = sculpted();
        let body = doc.active();
        let moved = doc.volume(body).expect("a body").shifted(glam::IVec3::new(2, 0, 0));
        let previous = doc.replace_volume(body, moved).expect("the body is in the document");

        let mut history = History::new(DEFAULT_HISTORY_BUDGET);
        history.push(Entry::new(vec![Change::WholeVolume { body, volume: Box::new(previous) }]));
        assert_eq!(history.undo(&mut doc, &[false]), UndoOutcome::Refused(body));
    }
}
