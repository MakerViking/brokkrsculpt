// SPDX-License-Identifier: AGPL-3.0-only

//! Which body's picture lives in which atlas cell, which pictures are out of
//! date, and when it is quiet enough to redraw one.
//!
//! The pixels are `brokkr-gpu`'s ([`brokkr_gpu::thumbnail`]); this is the
//! bookkeeping that decides what it is asked to draw. It is a separate module
//! because the refresh rule is the load-bearing part of the whole feature and
//! it is worth being able to test it without a GPU, a window or a document.
//!
//! # The refresh rule
//!
//! A cell is STALE when a brick of that body was marked dirty, or when the body
//! is new to the atlas. **Visibility, solo, rename, reorder, collapse, folder
//! membership and selection never stale a cell** -- soloing sixty-four bodies
//! costs zero renders. Staleness is set from `update()`, at the sites that
//! already mark bricks dirty, and never from `view()`.
//!
//! A stale cell is RENDERED when no stroke is in flight, no drag is in
//! progress, nothing else is already queued, and [`SETTLE`] has passed since
//! the geometry last changed.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use brokkr_core::{MAX_BODIES, NodeId};

/// How long the geometry has to have been still before a picture is redrawn.
///
/// **This runs off its own timer and not off `Brokkr::last_activity`, and that
/// is a correction rather than a preference.** `last_activity` is stamped in
/// exactly one non-test place, the top of `Brokkr::on_pointer`, so it measures
/// POINTER activity and fails in both directions here. Too loose: a held ctrl+Z
/// repeats at about 25 a second and touches no pointer event at all, so 200 ms
/// after the last mouse move every single frame would pay a full-body render on
/// top of the undo's own remesh. Too strict: a pen hovering over the model
/// after a stroke fires `on_pointer` every frame, the gate never opens, and the
/// picture stays minutes out of date.
pub const SETTLE: Duration = Duration::from_millis(200);

/// Cells, staleness and the settle timer.
#[derive(Debug)]
pub struct Thumbnails {
    /// Which atlas layer each body's picture lives in.
    ///
    /// **A map keyed on [`NodeId`], and neither of the two obvious cheaper
    /// things.** Not `NodeId.0` as an index: ids come out of files, are never
    /// reused, and run far past the sixty-four layers there are. Not the node's
    /// position in the list: a reorder is a permutation of that list, so every
    /// row below a moved one would silently start showing its neighbour's
    /// picture -- with nothing marked stale, because a reorder changes no
    /// geometry.
    cells: HashMap<NodeId, u32>,
    /// Layers no body is using.
    ///
    /// A queue rather than a stack, so a freed layer is the LAST one handed out
    /// again. A recycled layer still holds the previous body's picture until
    /// its first render lands a frame or two later, and taking the oldest free
    /// layer is the one-line way to make that window as rare as it can be.
    free: VecDeque<u32>,
    /// Bodies whose picture no longer matches their geometry, oldest first.
    ///
    /// A queue and not a set so that the order is the order they went stale:
    /// after an import every body is stale at once, and taking them in a fixed
    /// order means the list fills in from the top rather than at random.
    stale: VecDeque<NodeId>,
    /// Membership of `stale`, so pushing an already-stale body is a hash lookup
    /// rather than a scan. Sixty-four would not matter; keeping the two in step
    /// in one place is what does.
    is_stale: HashSet<NodeId>,
    /// Every body the last [`Thumbnails::sync`] was handed, in the order the
    /// panel lists them.
    ///
    /// **Kept and refilled, never allocated, and that is a hard requirement
    /// rather than a saving.** `sync` is called from
    /// `Brokkr::publish_visibility`, which runs after EVERY message including
    /// the frame tick, and whose own doc comment promises in bold that it
    /// allocates nothing. `Vec::clear` and `HashSet::clear` both keep their
    /// capacity, so after the first call these two cost sixty-four pushes and
    /// sixty-four hash inserts and no heap traffic at all.
    ///
    /// The order is what makes the layers deterministic: handing cells out by
    /// walking a `HashSet` would give a different body a different layer on
    /// every run, and the stale queue -- whose whole point is that an import
    /// fills the panel in from the top rather than at random -- would fill in a
    /// different order too.
    order: Vec<NodeId>,
    /// Membership of `order`, so taking back a departed body's cell is a hash
    /// lookup per cell rather than a scan of sixty-four.
    present: HashSet<NodeId>,
    /// When the geometry last changed. See [`SETTLE`].
    last_change: Instant,
}

impl Default for Thumbnails {
    fn default() -> Self {
        Self::new()
    }
}

impl Thumbnails {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            free: (0..MAX_BODIES as u32).collect(),
            stale: VecDeque::new(),
            is_stale: HashSet::new(),
            order: Vec::with_capacity(MAX_BODIES),
            present: HashSet::with_capacity(MAX_BODIES),
            last_change: Instant::now(),
        }
    }

    /// Which cell a body's picture is in, or `None` when it has none.
    ///
    /// One hash lookup, and it is what a panel row reads. A body without a cell
    /// draws the flat placeholder the panel drew before this feature existed,
    /// which is what a document over the sixty-four-layer atlas would get --
    /// unreachable today, since `MAX_BODIES` is that same sixty-four, and left
    /// as a fallback rather than a panic because the two constants living in
    /// two crates is exactly the pair that drifts.
    pub fn cell(&self, body: NodeId) -> Option<u32> {
        self.cells.get(&body).copied()
    }

    /// Give every body in the document a cell, and take back the cells of
    /// bodies that have left.
    ///
    /// Called from the one place that already walks the node list after every
    /// message, for the reason that place gives: "the handful of places that
    /// change an eye" is a list that goes out of date silently, and so is the
    /// handful of places that add or remove a body.
    ///
    /// Idempotent. A call where the body set has not changed marks nothing
    /// stale and stamps no timer, which is what keeps hiding, soloing, renaming
    /// and reordering free.
    ///
    /// # Departures are settled before arrivals are served
    ///
    /// **The two passes are in this order because the atlas can be full**, and
    /// a full atlas is where the cheap version of this went wrong. Opening or
    /// importing a second sixty-four-body document swaps every body for a
    /// different one: serve the newcomers first and every one of them finds an
    /// empty free list and silently gets no cell, and the layers stay stranded
    /// on bodies that no longer exist. Taking the departed cells back first
    /// makes a whole-document swap re-home every body on this same call.
    ///
    /// It is also why the removal pass runs unconditionally rather than behind
    /// a "did anything change" test. The obvious such test -- comparing the
    /// number of bodies against the number of cells -- is only equivalent to
    /// comparing the SETS while the free list has something in it, and a
    /// same-size swap over a full atlas is exactly the case where it does not:
    /// the counts match, the pass is skipped, and the panel shows sixty-four
    /// empty wells for the rest of the session with nothing saying why. Sixty-
    /// four hash lookups a frame is the price of not having a state like that.
    pub fn sync(&mut self, bodies: impl Iterator<Item = NodeId>) {
        // Taken out and put back rather than borrowed, so the passes below can
        // touch the other fields. Neither allocates: see `order`.
        let mut order = std::mem::take(&mut self.order);
        let mut present = std::mem::take(&mut self.present);
        order.clear();
        present.clear();
        for body in bodies {
            order.push(body);
            present.insert(body);
        }

        let free = &mut self.free;
        self.cells.retain(|body, cell| {
            if present.contains(body) {
                return true;
            }
            free.push_back(*cell);
            false
        });
        self.stale.retain(|body| present.contains(body));
        self.is_stale.retain(|body| present.contains(body));

        for body in &order {
            if self.cells.contains_key(body) {
                continue;
            }
            let Some(cell) = self.free.pop_front() else {
                continue;
            };
            self.cells.insert(*body, cell);
            // New to the atlas, so its layer holds either the placeholder or
            // some previous body's picture. Either way it needs drawing.
            self.mark_stale(*body);
        }

        self.order = order;
        self.present = present;
    }

    /// This body's geometry changed: its picture is out of date and the settle
    /// timer starts again.
    pub fn geometry_changed(&mut self, body: NodeId) {
        self.mark_stale(body);
        self.last_change = Instant::now();
    }

    /// Every picture is out of date. For turning the pictures back on, where
    /// nothing changed but nothing was being kept up to date either.
    pub fn stale_everything(&mut self) {
        let bodies: Vec<NodeId> = self.cells.keys().copied().collect();
        for body in bodies {
            self.mark_stale(body);
        }
    }

    fn mark_stale(&mut self, body: NodeId) {
        if self.is_stale.insert(body) {
            self.stale.push_back(body);
        }
    }

    /// The next picture that wants drawing, without taking it.
    ///
    /// A peek and not a take, because the caller has one more thing to check
    /// that this type cannot see -- whether the body still has any geometry to
    /// draw -- and a body taken and then not drawn would be a picture that
    /// never comes back.
    pub fn next_stale(&self) -> Option<NodeId> {
        self.stale.front().copied()
    }

    /// This body's picture has been asked for; stop offering it.
    pub fn requested(&mut self, body: NodeId) {
        if self.is_stale.remove(&body) {
            self.stale.retain(|queued| *queued != body);
        }
    }

    /// Whether the geometry has been still long enough to be worth a picture.
    pub fn settled(&self) -> bool {
        self.last_change.elapsed() >= SETTLE
    }

    /// Pretend the last change was long ago, so a test does not have to sleep.
    #[cfg(test)]
    pub fn settle_now(&mut self) {
        self.last_change = Instant::now() - SETTLE - Duration::from_millis(1);
    }

    /// When the settle timer was last restarted.
    ///
    /// Compared between two moments rather than against the wall clock: a test
    /// that asserted "not settled yet" would be asserting that the machine did
    /// not deschedule the thread for 200 ms, which under a full suite it
    /// sometimes does.
    #[cfg(test)]
    pub fn last_change(&self) -> Instant {
        self.last_change
    }

    /// How many pictures are waiting, for the tests that count them.
    #[cfg(test)]
    pub fn stale_count(&self) -> usize {
        self.stale.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(raw: &[u32]) -> Vec<NodeId> {
        raw.iter().map(|id| NodeId(*id)).collect()
    }

    /// The first sixty-four bodies each get a layer of their own, and a document
    /// past that draws placeholders rather than fighting over one.
    #[test]
    fn every_body_up_to_the_atlas_size_gets_a_cell_of_its_own() {
        let mut thumbs = Thumbnails::new();
        let bodies = ids(&(1..=MAX_BODIES as u32 + 4).collect::<Vec<_>>());
        thumbs.sync(bodies.iter().copied());

        let mut seen = HashSet::new();
        let mut with_cells = 0;
        for body in &bodies {
            if let Some(cell) = thumbs.cell(*body) {
                with_cells += 1;
                assert!(cell < MAX_BODIES as u32, "cell {cell} is outside the atlas");
                assert!(seen.insert(cell), "cell {cell} was handed out twice");
            }
        }
        assert_eq!(with_cells, MAX_BODIES, "the atlas handed out the wrong number of layers");
    }

    /// **A reorder must not move a single picture.** This is the whole reason
    /// the cell is keyed on the id rather than on the row's position: nothing
    /// about a reorder changes any geometry, so nothing would be marked stale,
    /// and every row below the move would quietly show its neighbour's model.
    #[test]
    fn reordering_the_rows_moves_no_picture_and_stales_nothing() {
        let mut thumbs = Thumbnails::new();
        let bodies = ids(&[7, 3, 9]);
        thumbs.sync(bodies.iter().copied());
        let before: Vec<Option<u32>> = bodies.iter().map(|body| thumbs.cell(*body)).collect();

        for body in &bodies {
            thumbs.requested(*body);
        }
        assert_eq!(thumbs.stale_count(), 0);

        // The same three bodies, in a different order and with one moved into a
        // folder -- which is what a reorder looks like from here.
        thumbs.sync(ids(&[9, 7, 3]).into_iter());
        let after: Vec<Option<u32>> = bodies.iter().map(|body| thumbs.cell(*body)).collect();

        assert_eq!(before, after, "a reorder moved a picture to another cell");
        assert_eq!(thumbs.stale_count(), 0, "a reorder queued a render");
    }

    /// A deleted body's layer comes back, and the body that gets it is redrawn
    /// rather than left showing the deleted one.
    #[test]
    fn a_deleted_bodys_cell_is_reused_and_the_body_that_gets_it_is_stale() {
        let mut thumbs = Thumbnails::new();
        thumbs.sync(ids(&[1, 2]).into_iter());
        let going = thumbs.cell(NodeId(2)).expect("body 2 has a cell");
        for body in ids(&[1, 2]) {
            thumbs.requested(body);
        }

        thumbs.sync(ids(&[1]).into_iter());
        assert_eq!(thumbs.cell(NodeId(2)), None, "a deleted body kept its cell");
        assert_eq!(thumbs.stale_count(), 0, "a delete queued a render for a body that is gone");

        // Fill the atlas so the freed layer is the only one left, which is what
        // makes this test about reuse rather than about the next fresh layer.
        let filling: Vec<NodeId> =
            (1..=MAX_BODIES as u32).map(NodeId).filter(|id| *id != NodeId(2)).collect();
        thumbs.sync(filling.iter().copied());
        let newcomer = NodeId(MAX_BODIES as u32 + 1);
        thumbs.sync(filling.iter().copied().chain(std::iter::once(newcomer)));

        assert_eq!(thumbs.cell(newcomer), Some(going), "the freed layer was not reused");
        assert!(
            thumbs.next_stale().is_some(),
            "a body handed a recycled layer was not marked for redrawing, so it shows the \
             deleted body's picture until its own geometry happens to change"
        );
    }

    /// **A whole-document swap over a FULL atlas has to re-home every body**,
    /// and the version of `sync` that served arrivals before it settled
    /// departures could not: every newcomer found the free list empty, and a
    /// short circuit on the two COUNTS being equal then skipped the removal
    /// pass, so the atlas stayed stranded on the sixty-four bodies that had
    /// gone. Nothing recovered it -- the free list never refilled, so every
    /// later sync short-circuited too -- and the panel showed sixty-four empty
    /// wells for the rest of the session.
    ///
    /// Every other test in this module runs at three or four bodies, well under
    /// the atlas, which is why none of them sees it.
    #[test]
    fn a_whole_document_swap_over_a_full_atlas_rehomes_every_body() {
        let mut thumbs = Thumbnails::new();
        let first: Vec<NodeId> = (1..=MAX_BODIES as u32).map(NodeId).collect();
        thumbs.sync(first.iter().copied());
        for body in &first {
            thumbs.requested(*body);
        }
        assert!(first.iter().all(|body| thumbs.cell(*body).is_some()), "the fixture never filled");

        // Opening a second document: the same number of bodies, not one of them
        // the same id, because ids restart at 1 per file and these must not
        // collide for the test to be about the swap.
        let second: Vec<NodeId> = (1..=MAX_BODIES as u32).map(|id| NodeId(id + 1_000)).collect();
        thumbs.sync(second.iter().copied());

        let mut seen = HashSet::new();
        for body in &second {
            let cell = thumbs.cell(*body).unwrap_or_else(|| {
                panic!("body {body:?} of the new document got no cell, so its row stays empty")
            });
            assert!(seen.insert(cell), "cell {cell} was handed to two bodies");
        }
        assert!(
            first.iter().all(|body| thumbs.cell(*body).is_none()),
            "a body from the closed document is still holding a layer"
        );
        assert_eq!(
            thumbs.stale_count(),
            MAX_BODIES,
            "the new document's pictures were not queued for drawing"
        );
    }

    /// The settle gate, and the queue's order.
    #[test]
    fn nothing_is_offered_until_the_geometry_has_been_still() {
        let mut thumbs = Thumbnails::new();
        thumbs.sync(ids(&[1, 2]).into_iter());
        thumbs.requested(NodeId(1));
        thumbs.requested(NodeId(2));

        thumbs.geometry_changed(NodeId(2));
        assert!(!thumbs.settled(), "a change that just happened is already settled");
        assert_eq!(thumbs.next_stale(), Some(NodeId(2)));

        thumbs.settle_now();
        assert!(thumbs.settled());

        // Oldest first, and a body already queued does not jump to the back.
        thumbs.geometry_changed(NodeId(1));
        thumbs.geometry_changed(NodeId(2));
        assert_eq!(thumbs.stale_count(), 2, "a body was queued twice");
        assert_eq!(thumbs.next_stale(), Some(NodeId(2)));
        thumbs.requested(NodeId(2));
        assert_eq!(thumbs.next_stale(), Some(NodeId(1)));
        thumbs.requested(NodeId(1));
        assert_eq!(thumbs.next_stale(), None);
    }
}
