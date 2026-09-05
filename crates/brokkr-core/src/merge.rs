// SPDX-License-Identifier: AGPL-3.0-only

//! Merging one body down into the body below it.
//!
//! This is the increment that shows what the one-lattice decision bought. Two
//! bodies of a document share a voxel size and a lattice origin, so brick `c`
//! of one covers **exactly** the same world box as brick `c` of the other. The
//! union of two signed distance fields is their pointwise `min`, so a merge is
//! a brick-by-brick, voxel-by-voxel `min` between two arrays that already line
//! up: **no resampling, no interpolation, and not one value that is not either
//! the target's own or the source's own.** A mesh sculptor pays a boolean
//! remesh here and loses detail on exactly the scanned input this application
//! is for.
//!
//! # Why iterating the SOURCE's bricks alone is enough
//!
//! An absent brick reads as [`OUTSIDE`], every stored value is clamped to the
//! band, and `min(t, OUTSIDE) == t` for every in-band `t`. So a target brick
//! the source has no brick for is provably unchanged: there is nothing to
//! compare it against but the saturated positive value it already loses to.
//! That is what makes a merge proportional to the SOURCE rather than to the
//! document -- merging a thumb-sized part into the dragon costs the thumb.
//!
//! # The four verdicts, and the two that cost nothing
//!
//! [`Fold`] classifies each of the source's bricks from one map lookup a side.
//! Two of the four write nothing at all: a source tile at `OUTSIDE` cannot
//! lower anything, and a target tile at [`INSIDE`] cannot be lowered. The
//! second is the "merging into solid interior is free" case, and it is one
//! float compare per brick rather than a scan of 32,768 voxels.
//!
//! # Dirty is the trap, not the contents
//!
//! A target brick the merge never touched still changes its *mesh* when a
//! neighbour changes, because a brick's apron reads one voxel into each of the
//! 26 around it. So every brick that is written marks a voxel range rather than
//! a coordinate: [`Volume::mark_dirty_voxel_range`] grows the range by one
//! voxel on each side, which is precisely the 26 neighbours. Mark only what was
//! written and there is a crack along the join that no slicer will accept.
//!
//! # Whose filament slot survives
//!
//! **The incoming body's, wherever it is painted AND has material in band;
//! the target's everywhere else.** An earlier draft of this header planned the
//! argmin -- the slot of whichever distance won the `min` -- and that is not
//! what landed, for a reason the argmin cannot see: a slot is written only in
//! the band and can be left behind outside it, so "the incoming body is
//! painted here" has to be checked against the incoming FIELD, not the colour
//! map alone. `ColourField::union_from` takes that predicate from
//! [`Volume::union_colour_from`]. Where both bodies are painted in band the
//! incoming one wins, because its material is what the merge is adding.
//! Neither rule can unpaint anything. The bricks it overwrites go into the
//! open stroke's recorder, so undoing a merge puts the target's own paint
//! back -- which the mask half below still does not do.
//!
//! The mask is the other way round: **`max` of the two masks, never the
//! argmin's mask.** Protection is a veto and vetoes union; a rule that let the
//! incoming body win would let an unmasked source strip protection along the
//! exact seam the merge just created. That half is live -- see the end of
//! [`Volume::union_from`] -- and unlike the field and colour halves it is not
//! in the undo entry yet. `StrokeEdit` gained a mask list in increment 21, so
//! the shape now exists; what is missing is `union_max_from` handing back the
//! bricks it overwrote. Revisit when a merge's undo is next touched.

use crate::body::{Document, NodeId};
use crate::brick::{BRICK_VOXELS, Brick, INSIDE, OUTSIDE};
use crate::undo::{Change, Entry, predicted_prior_bytes, removed_node_bytes};
use crate::volume::Volume;

/// What the union does to one brick of the shared lattice.
///
/// Worked out from the two bricks alone, so that the walk which PREDICTS a
/// merge's size and the walk which performs it cannot disagree about which
/// bricks are touched. They call this same function.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Fold {
    /// The target already wins every voxel. Nothing is recorded and nothing is
    /// written, which is also why a tie belongs here: the target's value, and
    /// with it the target's filament slot, stays exactly as it was.
    Keeps,
    /// The source wins every voxel and holds one value everywhere, so the brick
    /// becomes that tile whatever the target held. The whole brick is the
    /// source's, slot included.
    Tile(f32),
    /// The target holds nothing here, so the union IS the source's brick and it
    /// is adopted whole. Ties can only happen at `OUTSIDE`, where there is no
    /// surface and so no slot to lose.
    Adopts,
    /// Both carry detail. The `min` has to be taken voxel by voxel, and the
    /// argmin with it.
    Compares,
}

/// Which brick wins where, from one map lookup a side.
///
/// `target` is what the merge writes into and `source` is what is being merged
/// down onto it. Both are `Option` because an absent brick is a real state that
/// reads as [`OUTSIDE`]; the source's is `Some` at every coordinate the walk
/// visits, and the `None` arm is here so the function is total rather than
/// because a caller can reach it.
fn fold(target: Option<&Brick>, source: Option<&Brick>) -> Fold {
    let Some(source) = source else {
        return Fold::Keeps;
    };
    // A source tile at `OUTSIDE` cannot lower anything: every stored value is
    // clamped to the band on the way in and refused by the reader if it is not,
    // so `min(t, OUTSIDE) == t` for every `t` this volume can hold. The engine
    // never writes such a tile, but a file may carry one and it must cost
    // nothing rather than promote the target to dense.
    if let Brick::Uniform(value) = source
        && *value >= OUTSIDE
    {
        return Fold::Keeps;
    }
    match target {
        None => Fold::Adopts,
        // Solid interior. `min(INSIDE, anything in band)` is `INSIDE`, so the
        // target wins or ties every voxel and keeps its own slot. This is the
        // "merging into solid interior is free" case: one compare per brick.
        Some(Brick::Uniform(held)) if *held <= INSIDE => Fold::Keeps,
        Some(Brick::Uniform(held)) => match source {
            // Two tiles: the answer is a tile, and no brick is ever made dense
            // to find that out. `>=` and not `>` is the tie rule.
            Brick::Uniform(value) if *value >= *held => Fold::Keeps,
            Brick::Uniform(value) => Fold::Tile(*value),
            Brick::Dense(_) => Fold::Compares,
        },
        Some(Brick::Dense(_)) => match source {
            // A saturated source tile wins every voxel of a dense target, and
            // collapses 128 KiB of detail into 32 bytes as it does.
            Brick::Uniform(value) if *value <= INSIDE => Fold::Tile(*value),
            _ => Fold::Compares,
        },
    }
}

/// The `min`, in place, answering whether the result came out a saturated tile.
///
/// **The collapse test rides on the min rather than following it**, and that is
/// the difference between one pass over 128 KiB and two. `Brick::is_collapsible`
/// is a second full scan of the array, and a fully overlapping merge of the
/// reference dragon runs this 6,120 times -- 765 MiB of reads that the loop
/// which just wrote every one of those values can answer for free.
///
/// Only a SATURATED result is worth collapsing, which is
/// `Brick::is_collapsible`'s own rule and not a simplification of it: a brick of
/// uniform mid-band values is surface adjacent and about to be written again, so
/// releasing its allocation only buys re-allocating it.
///
/// # The strict `<` is the filament-slot rule
///
/// The source has to BEAT the target to win, so a tie leaves the target's value
/// -- and, when colour lands, the target's slot -- exactly where it was. That
/// branch is the argmin, and it is the one place a slot write goes. Written this
/// way rather than as `f32::min`, which computes the same number and erases
/// which side produced it.
fn lower_in_place(
    data: &mut [f32; BRICK_VOXELS],
    incoming: impl Iterator<Item = f32>,
) -> Option<f32> {
    let mut uniform = None;
    for (index, (held, incoming)) in data.iter_mut().zip(incoming).enumerate() {
        if incoming < *held {
            *held = incoming;
        }
        if index == 0 {
            uniform = Some(*held);
        } else if uniform != Some(*held) {
            uniform = None;
        }
    }
    uniform.filter(|value| *value == OUTSIDE || *value == INSIDE)
}

impl Volume {
    /// Union another field on the same lattice into this one, and return how
    /// many bricks were written.
    ///
    /// **`pub(crate)` so that the caller has to be [`Document::merge_down`]**,
    /// which owns the [`Volume::begin_stroke`] / [`Volume::end_stroke`]
    /// bracketing. An unbracketed merge is not a merge that cannot be undone,
    /// it is **silent total data loss**: [`Volume::record_for_undo`] returns
    /// `false` and does nothing at all when no recorder is open, so the source
    /// body is consumed, the target is rewritten, and the entry that is pushed
    /// restores neither. That is the press-ordering bug increment 8 fixed for
    /// one brush stamp, one whole body wide.
    ///
    /// # The assert on the lattice is a real one
    ///
    /// Not a `debug_assert`. Every body of a [`Document`] is on the document's
    /// voxel size by construction, so a mismatch here is not a caller passing
    /// the wrong thing, it is the one-lattice invariant already broken -- and
    /// merging across two lattices would read brick `c` of one body at the
    /// world box of brick `c` of another and silently produce a field that is
    /// neither body.
    pub(crate) fn union_from(&mut self, other: &Volume) -> usize {
        assert_eq!(
            self.voxel_size(),
            other.voxel_size(),
            "a merge is a brick-by-brick min and only means anything on one lattice"
        );
        debug_assert!(
            self.is_recording(),
            "an unbracketed merge records nothing and is silent total data loss"
        );

        let mut written = 0usize;
        for coord in other.brick_coords() {
            let source = other.brick(coord);
            match fold(self.brick(coord), source) {
                Fold::Keeps => continue,
                Fold::Tile(value) => {
                    self.record_for_undo(coord);
                    self.insert_brick(coord, Brick::Uniform(value));
                }
                Fold::Adopts => {
                    // The one clone in the whole operation, and it is the
                    // cheapest arrangement available: the alternative is to
                    // take the source apart brick by brick, which would mean
                    // the source could not then go into the undo entry whole.
                    let Some(brick) = source.cloned() else {
                        continue;
                    };
                    self.record_for_undo(coord);
                    self.insert_brick(coord, brick);
                }
                Fold::Compares => {
                    // Recorded before the brick is taken, because the recorder
                    // reads the prior contents out of the map. Taking rather
                    // than cloning is what keeps this to the one 128 KiB copy
                    // undo needs instead of two; see `Volume::take_brick`.
                    self.record_for_undo(coord);
                    let Some(mut brick) = self.take_brick(coord) else {
                        continue;
                    };
                    let data = brick.make_dense();
                    let collapsed = match source {
                        Some(Brick::Uniform(value)) => {
                            lower_in_place(data, std::iter::repeat(*value))
                        }
                        Some(Brick::Dense(source)) => lower_in_place(data, source.iter().copied()),
                        // `Fold::Compares` is only reachable with a source
                        // brick in hand.
                        None => None,
                    };
                    // A merge can only lower values, so a brick that came out
                    // `OUTSIDE` everywhere was empty before and is empty now --
                    // but it may have been a dense array of `OUTSIDE` all
                    // along, and dropping it releases the 128 KiB either way.
                    // Already recorded and already out of the map, so these only
                    // decide what goes back in.
                    match collapsed {
                        Some(value) if value >= OUTSIDE => {}
                        Some(value) => self.insert_brick(coord, Brick::Uniform(value)),
                        None => self.insert_brick(coord, brick),
                    }
                }
            }
            written += 1;
            // Per changed brick, and a voxel RANGE rather than a coordinate:
            // the one-voxel grow inside this covers all 26 neighbours, whose
            // aprons read into the brick that just changed. See the module doc.
            self.mark_dirty_voxel_range(coord.origin(), coord.max_voxel());
        }

        // The masks union by `max`, and the two rules do not contradict each
        // other: the union of two solids is the LOWER distance and the union of
        // two protections is the HIGHER one, because a merge must not be a way
        // to unprotect what either body protected.
        //
        // Not in the undo entry yet. `StrokeEdit` can carry mask bricks as of
        // increment 21, but `union_max_from` does not hand back the ones it
        // overwrote, so undoing a merge restores the field and leaves the merged
        // mask in place. Named here because it is the one thing about this call
        // that a reader would otherwise assume was covered.
        let mask = other.mask();
        self.mask_mut().union_max_from(mask, other.brick_coords());
        // And the paint: the incoming body's material brings its own slot,
        // where it really has material. Recorded, unlike the mask above.
        self.union_colour_from(other);

        written
    }
}

/// What a merge down would consume the active body into, or why it cannot.
///
/// A merge joins two BODIES and nothing else, so the two refusals are named
/// rather than folded into one `None`: "there is nothing below it" and "the row
/// below is a folder" are different mistakes and the second one names a row the
/// user can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeTarget {
    /// The body directly below, at the same depth and in the same container.
    Body(NodeId),
    /// Nothing below it in its own container: it is the bottom of its list.
    Bottom,
    /// The row directly below is a folder. A merge never reaches into one --
    /// merging into a container is ZBrush's MergeVisible and its universal
    /// "the button did nothing" reaction.
    Folder(NodeId),
}

/// What one merge down would cost history, worked out before anything is
/// written.
///
/// The two halves are counted apart because they are charged to different
/// allowances: the bricks go to the stroke budget and the consumed body goes to
/// the reclaim allowance. See [`crate::DEFAULT_HISTORY_BUDGET`] and
/// [`crate::DEFAULT_RECLAIM_BUDGET`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergePlan {
    /// The body that will be consumed.
    pub source: NodeId,
    /// The body it will be merged into.
    pub target: NodeId,
    /// Bricks of the target the merge will write.
    pub bricks: usize,
    /// What the `Change::Bricks` half will hold: the prior contents of every
    /// brick the merge writes.
    ///
    /// **Exact, not an estimate.** The walk that produces it classifies each
    /// brick with the same [`fold`] the merge itself uses, so it names the same
    /// bricks; and the prior of each is the target's brick as it stands right
    /// now, which is a value this can measure rather than predict.
    pub stroke_bytes: usize,
    /// What the `Change::NodeRemoved` half will hold: the source body itself,
    /// moved into the entry rather than dropped.
    pub reclaim_bytes: usize,
}

impl MergePlan {
    /// Everything the one entry will hold.
    ///
    /// The number to show a user, because a merge is one gesture and one entry:
    /// the two halves land in history together or not at all.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.stroke_bytes + self.reclaim_bytes
    }
}

/// What one merge down did.
pub struct MergeOutcome {
    /// The body that survived and now holds both fields.
    pub target: NodeId,
    /// Bricks of the target that were written. Zero when the source was already
    /// inside the target's material everywhere, which is a real outcome and not
    /// a failure.
    pub bricks: usize,
    /// **ONE entry for the whole gesture.** `[Bricks{target}, NodeRemoved
    /// {source}]`, and the order is load-bearing: an entry is applied in
    /// reverse, so undo puts the source body back FIRST and only then restores
    /// the target's bricks. Never `None` -- a merge always consumes a body,
    /// even when it changed not one voxel.
    pub entry: Entry,
}

impl Document {
    /// The body a merge down would consume the given one into.
    ///
    /// **The next SIBLING body: the very next row, at the same depth.** A body
    /// has no children, so "the next row" and "the next sibling" are the same
    /// question, and the whole of the legality test is one depth comparison. A
    /// row below at a SHALLOWER depth is outside this row's folder, which makes
    /// this row the bottom of its own list -- a merge that reached past it
    /// would pull a body out of the folder the user put it in.
    pub fn merge_target(&self, source: NodeId) -> MergeTarget {
        let Some(at) = self.index_of(source) else {
            return MergeTarget::Bottom;
        };
        let depth = self.nodes()[at].depth();
        match self.nodes().get(at + 1) {
            Some(below) if below.depth() == depth && below.is_body() => MergeTarget::Body(below.id),
            Some(below) if below.depth() == depth => MergeTarget::Folder(below.id),
            // Deeper is unreachable -- a body is never a parent -- and
            // shallower means the row below is outside this row's container.
            _ => MergeTarget::Bottom,
        }
    }

    /// What merging one body down would cost, without merging it.
    ///
    /// `None` when there is nothing to merge into; ask [`Document::merge_target`]
    /// for which of the two reasons.
    ///
    /// # Microseconds, and no allocation
    ///
    /// A walk of the source's brick coordinates with two map lookups apiece.
    /// The two extremes are worth stating because they are three orders of
    /// magnitude apart and both are ordinary. A DISJOINT merge is nearly free:
    /// every brick is adopted, every prior is `None`, so a 45,567-brick source
    /// records about 1.46 MB. A FULLY OVERLAPPING one is brutal: 6,120 dense
    /// priors at 128 KiB is 765 MiB, plus the same again for the source body
    /// the entry now owns -- about 1.53 GiB in one entry, six times the stroke
    /// budget. Peak memory is then 2.29 GiB from a 1.53 GiB document, and the
    /// application's memory guard reads resident volume bytes and so cannot see
    /// history at all. **That is what this exists for: so the user is asked
    /// before the allocation, and never after it.**
    pub fn merge_plan(&self, source: NodeId) -> Option<MergePlan> {
        let MergeTarget::Body(target) = self.merge_target(source) else {
            return None;
        };
        let from = self.volume(source)?;
        let into = self.volume(target)?;

        let mut bricks = 0usize;
        let mut stroke_bytes = 0usize;
        for coord in from.brick_coords() {
            let prior = into.brick(coord);
            if fold(prior, from.brick(coord)) == Fold::Keeps {
                continue;
            }
            bricks += 1;
            stroke_bytes += predicted_prior_bytes(prior);
        }

        let reclaim_bytes = self.node(source).map_or(0, removed_node_bytes);
        Some(MergePlan { source, target, bricks, stroke_bytes, reclaim_bytes })
    }

    /// Merge one body down into the body below it, as one undoable gesture.
    ///
    /// `None` when [`Document::merge_target`] finds nothing to merge into. The
    /// target survives, keeps its own name, and **is left selected at the
    /// position the source held** -- which needs no move, because taking the
    /// source out slides the target up into its row. Photoshop's merge down
    /// keeps the lower layer and its name for the same reason: the result has
    /// to appear where the user was looking, and never as a third thing
    /// deposited somewhere else.
    ///
    /// # The source is removed FIRST, and that is not an ordering accident
    ///
    /// Two bodies of one document cannot be borrowed at once, one of them
    /// mutably. Taking the source out hands the whole [`Node`] over by value,
    /// so its field can be read while the target's is written, and that same
    /// [`Node`](crate::Node) then moves into `Change::NodeRemoved` rather than
    /// being cloned -- [`Volume`] has no `Clone` at all. A merge therefore allocates only the
    /// bricks it adopts.
    ///
    /// **The folder cannot be left empty by this**, which is why there is no
    /// dissolve pass here as there is in a delete: the target is a sibling in
    /// the same container, so that container still holds at least one row.
    pub fn merge_down(&mut self, source: NodeId) -> Option<MergeOutcome> {
        let MergeTarget::Body(target) = self.merge_target(source) else {
            return None;
        };
        let at = self.index_of(source)?;

        let node = self.remove(at);
        let mut changes = Vec::with_capacity(2);
        let mut bricks = 0;

        if let Some(from) = node.volume() {
            let into = self.volume_mut(target).expect("the target is a body and is still here");
            // The bracketing this whole operation is `pub(crate)` to protect.
            into.begin_stroke();
            bricks = into.union_from(from);
            if let Some(edit) = into.end_stroke().filter(|edit| !edit.is_empty()) {
                changes.push(Change::Bricks { body: target, edit });
            }
        }
        // After the bricks, so that applying the entry in reverse puts the body
        // back before it restores the field it was merged into.
        changes.push(Change::NodeRemoved { at, node: Box::new(node) });

        // The result is selected. `Document::remove` has already moved the
        // selection here when the source was the active row, which it always is
        // from the panel; saying it outright costs one comparison and makes the
        // guarantee this function's rather than the caller's.
        self.set_active(target);
        Some(MergeOutcome { target, bricks, entry: Entry::new(changes) })
    }
}

#[cfg(test)]
mod tests {
    use glam::{IVec3, Vec3};

    use super::*;
    use crate::brick::{BRICK_DIM, BrickCoord};
    use crate::undo::History;
    use crate::undo::prior_bytes;

    const VOXEL: f32 = 0.5;

    /// Centres at `±SEPARATE` put the two fixture balls in brick maps that do
    /// not meet: every fold is [`Fold::Adopts`], every undo prior is `None`,
    /// and [`lower_in_place`] never runs.
    ///
    /// **A test that only ever uses this one proves nothing about the
    /// arithmetic**, which is why every test below that cares about VALUES runs
    /// [`TOGETHER`] as well, and asserts that it really did overlap. A ball
    /// of radius 8 spans 16 units, a brick spans 16 too, and `seed_sphere` pads
    /// its bounds by twice the narrow band -- so the margin here is one brick,
    /// not a rounding error.
    const SEPARATE: f32 = 12.0;

    /// Centres at `±TOGETHER`: the two balls interpenetrate, so their shells
    /// share bricks, the fold comes out [`Fold::Compares`], and the recorded
    /// priors are real 128 KiB arrays. The same separation
    /// [`two_merged_balls_export_watertight_with_no_crack_at_the_join`] uses.
    const TOGETHER: f32 = 5.0;

    /// A ball of `radius` at `centre`, on the shared lattice.
    fn ball(centre: Vec3, radius: f32) -> Volume {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(centre, radius);
        volume
    }

    /// Every voxel of the inclusive box, so a comparison can be exhaustive
    /// rather than a handful of probes.
    fn voxels(low: IVec3, high: IVec3) -> impl Iterator<Item = IVec3> {
        (low.z..=high.z).flat_map(move |z| {
            (low.y..=high.y).flat_map(move |y| (low.x..=high.x).map(move |x| IVec3::new(x, y, z)))
        })
    }

    /// The headline claim: a merge is the `min` of the two fields and nothing
    /// else. Voxel for voxel over a box that contains both bodies, so a merge
    /// that leaked a value from anywhere else has nowhere to hide.
    ///
    /// **Run on an OVERLAPPING pair as well as a separate one, and that is the
    /// half that earns the test.** Separate balls take [`Fold::Adopts`] at every
    /// brick -- a whole-brick clone -- so on that fixture alone the entire
    /// arithmetic this increment exists for never executes: replace the body of
    /// [`lower_in_place`] with a panic and the separate half still passes. An
    /// off-by-one in its `zip`, a transposed index or a `min` taken against the
    /// wrong operand is only observable where both bodies carry detail in the
    /// same brick. The `compares` count below asserts the fixture really gets
    /// there, so it cannot quietly drift back to two clones.
    #[test]
    fn merging_two_balls_gives_the_min_of_the_two_fields_voxel_for_voxel() {
        for (offset, what) in [(SEPARATE, "separate"), (TOGETHER, "overlapping")] {
            let mut into = ball(Vec3::new(-offset, 0.0, 0.0), 8.0);
            let from = ball(Vec3::new(offset, 0.0, 0.0), 8.0);
            let before = into.duplicated(IVec3::ZERO);

            let compares = from
                .brick_coords()
                .filter(|coord| fold(into.brick(*coord), from.brick(*coord)) == Fold::Compares)
                .count();
            assert_eq!(
                compares > 0,
                offset == TOGETHER,
                "{what}: the fixture no longer exercises the fold it was chosen for ({compares} \
                 bricks compared)"
            );

            into.begin_stroke();
            let written = into.union_from(&from);
            let edit = into.end_stroke().expect("a merge of two overlapping balls writes bricks");
            assert!(written > 0, "{what}: the merge wrote nothing");
            assert_eq!(written, edit.len(), "{what}: every written brick is recorded once");

            let low = IVec3::new(-60, -30, -30);
            let high = IVec3::new(60, 30, 30);
            for voxel in voxels(low, high) {
                let wanted = before.sample_voxel(voxel).min(from.sample_voxel(voxel));
                assert_eq!(
                    into.sample_voxel(voxel),
                    wanted,
                    "{what}: the merged field is not the min at {voxel}"
                );
            }
        }
    }

    /// The join is the whole reason a union of distance fields is worth having:
    /// two balls that overlap come out as one closed solid, with no crack where
    /// the two brick maps met.
    #[test]
    fn two_merged_balls_export_watertight_with_no_crack_at_the_join() {
        let mut into = ball(Vec3::new(-TOGETHER, 0.0, 0.0), 8.0);
        let from = ball(Vec3::new(TOGETHER, 0.0, 0.0), 8.0);

        into.begin_stroke();
        into.union_from(&from);
        let _ = into.end_stroke();
        into.mark_everything_dirty();

        let (mesh, report) = into.export_mesh();
        assert!(
            report.is_printable(),
            "the join left the model unprintable: {} ({} triangles)",
            report.summary(),
            mesh.triangles.len()
        );
    }

    /// Merging a body that is already buried inside the target's material is
    /// one float compare per brick and writes nothing at all. It is the cheap
    /// case the tile representation exists for, and it must not promote a
    /// single brick to dense.
    #[test]
    fn merging_into_solid_interior_touches_no_brick_and_records_no_edit() {
        // A big ball, so the small one below sits deep inside its uniform
        // interior tiles rather than in its shell.
        let mut into = ball(Vec3::ZERO, 30.0);
        let from = ball(Vec3::ZERO, 4.0);
        let dense_before = into.stats().dense_bricks;

        into.begin_stroke();
        let written = into.union_from(&from);
        let edit = into.end_stroke();

        assert_eq!(written, 0, "a merge into solid interior wrote {written} bricks");
        assert!(edit.is_none(), "a merge that changed nothing still recorded an entry");
        assert_eq!(
            into.stats().dense_bricks,
            dense_before,
            "a merge into solid interior promoted a brick to dense"
        );
    }

    /// The tie rule, at the only place it is observable before colour exists: a
    /// source that merely equals the target does not beat it, so the brick is
    /// not recorded, not written, and not promoted. When a filament slot rides
    /// along, this is the case where the target keeps its own.
    #[test]
    fn a_source_that_ties_the_target_everywhere_changes_nothing() {
        let coord = BrickCoord::new(0, 0, 0);
        let mut into = Volume::new(VOXEL);
        into.insert_brick(coord, Brick::Uniform(0.5));
        let mut from = Volume::new(VOXEL);
        from.insert_brick(coord, Brick::Uniform(0.5));

        into.begin_stroke();
        let written = into.union_from(&from);
        assert_eq!(written, 0, "a tie was written as though the source had won");
        assert!(into.end_stroke().is_none(), "a tie was recorded for undo");
    }

    /// A merge takes the GREATER protection, so a source that wins the distance
    /// everywhere still cannot unprotect what the target protected.
    ///
    /// The two rules do not contradict each other: the union of two solids is
    /// the lower distance and the union of two protections is the higher one.
    #[test]
    fn merging_an_unmasked_source_that_wins_everywhere_leaves_the_target_fully_masked() {
        use crate::{PROTECTED, UNMASKED};

        let coord = BrickCoord::new(0, 0, 0);
        let mut into = Volume::new(VOXEL);
        into.insert_brick(coord, Brick::Uniform(0.5));
        for voxel in voxels(coord.origin(), coord.max_voxel()) {
            into.mask_mut().write(voxel, PROTECTED);
        }
        into.mask_mut().collapse();

        // A solid tile beats 0.5 at every voxel of the brick.
        let mut from = Volume::new(VOXEL);
        from.insert_brick(coord, Brick::Uniform(INSIDE));
        assert_eq!(from.mask().at(coord.origin()), UNMASKED, "the source must carry no mask");

        into.begin_stroke();
        let written = into.union_from(&from);
        into.end_stroke();

        assert_eq!(written, 1, "the source has to have won the field, or this proves nothing");
        assert_eq!(into.sample_voxel(coord.origin()), INSIDE);
        for voxel in voxels(coord.origin(), coord.max_voxel()) {
            assert_eq!(into.mask().at(voxel), PROTECTED, "the merge unprotected {voxel}");
        }
        assert_eq!(
            into.stats().mask_bytes,
            0,
            "a fully protected brick has to stay a tile through a merge"
        );
    }

    /// The same rule read the other way round, and this is the direction that
    /// depends on the call: the protection has to ARRIVE in a target that had
    /// none, so deleting the `union_max_from` at the end of
    /// [`Volume::union_from`] fails here.
    ///
    /// The test above cannot see that deletion. It masks the TARGET and asserts
    /// the target is still masked, which is true of a merge that never touches
    /// the mask at all -- it is the argmin guard, not the coverage.
    ///
    /// Both source polarities, because they reach the target by different
    /// routes. A normal mask is carried by the source's own mask BRICKS; Mask
    /// All is an empty brick map with the polarity bit set, so the only thing
    /// that tells the merge where to write is the source's FIELD bricks -- the
    /// `also` iterator. Get the second one wrong and Mask All silently does not
    /// survive a merge down.
    #[test]
    fn a_masked_source_protects_the_target_it_is_merged_into() {
        use crate::{PROTECTED, UNMASKED};

        let coord = BrickCoord::new(0, 0, 0);
        for inverted in [false, true] {
            let mut into = Volume::new(VOXEL);
            into.insert_brick(coord, Brick::Uniform(0.5));
            assert_eq!(into.mask().at(coord.origin()), UNMASKED, "the target must carry no mask");

            let mut from = Volume::new(VOXEL);
            from.insert_brick(coord, Brick::Uniform(INSIDE));
            if inverted {
                // Mask All: the polarity flips and the brick map stays empty,
                // so the bit is the entire mask.
                from.mask_mut().set_inverted(true);
            } else {
                for voxel in voxels(coord.origin(), coord.max_voxel()) {
                    from.mask_mut().write(voxel, PROTECTED);
                }
                from.mask_mut().collapse();
            }
            assert_eq!(from.mask().at(coord.origin()), PROTECTED, "the fixture is not masked");

            into.begin_stroke();
            let written = into.union_from(&from);
            into.end_stroke();

            assert_eq!(written, 1, "the source has to have won the field, or this proves nothing");
            for voxel in voxels(coord.origin(), coord.max_voxel()) {
                assert_eq!(
                    into.mask().at(voxel),
                    PROTECTED,
                    "an inverted={inverted} source did not protect {voxel}"
                );
            }
            assert!(!into.mask().inverted(), "the target's polarity is not the source's to change");
            assert_eq!(
                into.stats().mask_bytes,
                0,
                "a fully protected brick has to arrive as a tile rather than 32 KiB"
            );
        }
    }

    /// Two dense bricks whose `min` comes out saturated everywhere release
    /// their 128 KiB, and the test that they do is what keeps the collapse
    /// riding on the min loop rather than on a second scan of the array.
    ///
    /// Both saturated directions, because they end differently: `INSIDE` goes
    /// back as a tile and `OUTSIDE` is dropped, since an absent brick already
    /// reads that way.
    /// The incoming body's paint arrives where it has material, the target's
    /// own paint stands where it has none, and the stroke that brackets the
    /// merge puts the target's paint back on undo -- the entry is APPLIED
    /// here, because a first version that discarded it passed with the colour
    /// union unrecorded and undo destroying paint.
    #[test]
    fn a_merged_body_brings_its_paint_leaves_ours_and_undoes_cleanly() {
        use crate::colour::UNPAINTED;

        let coord = BrickCoord::new(0, 0, 0);
        let mut into = Volume::new(VOXEL);
        into.insert_brick(coord, Brick::Uniform(0.5));
        let ours = coord.origin() + IVec3::new(1, 1, 1);
        let shared = coord.origin() + IVec3::new(3, 3, 3);
        let theirs = coord.origin() + IVec3::new(5, 5, 5);
        into.colour_mut().write(ours, 1);
        into.colour_mut().write(shared, 1);

        let mut from = Volume::new(VOXEL);
        // In band throughout, so every incoming slot is admitted.
        from.insert_brick(coord, Brick::Uniform(-0.5));
        from.colour_mut().write(theirs, 3);
        from.colour_mut().write(shared, 3);

        into.begin_stroke();
        into.union_from(&from);
        let edit = into.end_stroke().expect("a merge that changed the field recorded nothing");

        assert_eq!(into.colour().at(theirs), 3, "the incoming body's paint did not arrive");
        assert_eq!(into.colour().at(shared), 3, "where both are painted the incoming one wins");
        assert_eq!(into.colour().at(ours), 1, "the incoming body's unpainted voxel erased ours");
        assert_eq!(into.colour().at(coord.origin()), UNPAINTED);
        assert!(edit.colour_len() > 0, "the merge recorded no colour brick");

        let redo = into.apply_edit(edit);
        assert_eq!(into.colour().at(shared), 1, "undo did not put the target's own paint back");
        assert_eq!(into.colour().at(theirs), UNPAINTED, "undo left the incoming paint behind");
        into.apply_edit(redo);
        assert_eq!(into.colour().at(shared), 3, "redo did not re-merge the paint");
    }

    /// A slot the incoming body left behind outside its band -- carved away
    /// under it, or dragged out from under it -- is not material and must not
    /// arrive.
    #[test]
    fn a_merge_ignores_incoming_paint_where_the_incoming_body_has_no_material() {
        let coord = BrickCoord::new(0, 0, 0);
        let mut into = Volume::new(VOXEL);
        into.insert_brick(coord, Brick::Uniform(0.5));
        let cell = coord.origin() + IVec3::new(2, 2, 2);
        into.colour_mut().write(cell, 1);

        let mut from = Volume::new(VOXEL);
        // Solid throughout: saturated, so nothing in this brick is in band.
        from.insert_brick(coord, Brick::Uniform(INSIDE));
        from.colour_mut().write(cell, 3);

        into.begin_stroke();
        into.union_from(&from);
        let edit = into.end_stroke().expect("the field changed");
        assert_eq!(into.colour().at(cell), 1, "a stale incoming slot overwrote fresh paint");
        assert_eq!(edit.colour_len(), 0, "nothing changed, so nothing should be recorded");
    }

    #[test]
    fn a_dense_merge_that_comes_out_saturated_releases_the_allocation() {
        let solid = BrickCoord::new(0, 0, 0);
        let empty = BrickCoord::new(1, 0, 0);
        let mut into = Volume::new(VOXEL);
        into.insert_brick(solid, Brick::dense_filled(0.5));
        into.insert_brick(empty, Brick::dense_filled(OUTSIDE));
        let mut from = Volume::new(VOXEL);
        from.insert_brick(solid, Brick::dense_filled(INSIDE));
        from.insert_brick(empty, Brick::dense_filled(OUTSIDE));

        into.begin_stroke();
        assert_eq!(into.union_from(&from), 2, "both bricks are dense on both sides");
        let _ = into.end_stroke();

        let stats = into.stats();
        assert_eq!(stats.dense_bricks, 0, "neither dense brick was released");
        assert_eq!(stats.uniform_bricks, 1, "the empty brick was kept as a tile");
        assert_eq!(into.sample_voxel(solid.origin()), INSIDE);
        assert_eq!(into.sample_voxel(empty.origin()), OUTSIDE, "an absent brick reads as OUTSIDE");
    }

    /// A tile holding a mid-band value is something only a file can produce --
    /// the engine writes `INSIDE` and `OUTSIDE` tiles and nothing else -- and
    /// the reader accepts any in-band float for one. It has to merge by value
    /// like anything else, and without being made dense to do it.
    #[test]
    fn a_mid_band_tile_out_of_a_file_merges_by_value_and_stays_a_tile() {
        let coord = BrickCoord::new(2, -1, 3);
        let mut into = Volume::new(VOXEL);
        into.insert_brick(coord, Brick::Uniform(1.5));
        let mut from = Volume::new(VOXEL);
        from.insert_brick(coord, Brick::Uniform(-0.75));

        into.begin_stroke();
        assert_eq!(into.union_from(&from), 1, "the lower tile did not win");
        assert!(into.end_stroke().is_some(), "the change was not recorded");

        assert_eq!(into.sample_voxel(coord.origin()), -0.75);
        assert_eq!(into.stats().dense_bricks, 0, "a tile-to-tile merge allocated a dense brick");
        assert_eq!(into.stats().uniform_bricks, 1);
    }

    /// The other direction of the same case: a saturated source tile over a
    /// dense target collapses 128 KiB into a tile rather than leaving the
    /// allocation behind.
    #[test]
    fn a_solid_source_tile_collapses_the_dense_target_it_covers() {
        let coord = BrickCoord::new(0, 0, 0);
        let mut into = Volume::new(VOXEL);
        into.insert_brick(coord, Brick::dense_filled(0.25));
        let mut from = Volume::new(VOXEL);
        from.insert_brick(coord, Brick::Uniform(INSIDE));

        into.begin_stroke();
        assert_eq!(into.union_from(&from), 1);
        let _ = into.end_stroke();

        assert_eq!(into.stats().dense_bricks, 0, "the dense brick was not released");
        assert_eq!(into.sample_voxel(coord.origin()), INSIDE);
    }

    /// A merge marks the bricks it wrote AND their neighbours, because a
    /// neighbour's apron reads one voxel in. Get this wrong and the model on
    /// screen has a crack along the join while the field underneath is perfect.
    #[test]
    fn a_merged_brick_marks_the_neighbours_whose_aprons_read_into_it() {
        let coord = BrickCoord::new(4, 4, 4);
        let mut into = Volume::new(VOXEL);
        let mut from = Volume::new(VOXEL);
        from.insert_brick(coord, Brick::Uniform(INSIDE));

        into.begin_stroke();
        assert_eq!(into.union_from(&from), 1);
        let _ = into.end_stroke();

        // The brick itself plus all 26 around it.
        assert_eq!(into.dirty_count(), 27, "the join was marked without its neighbours");
    }

    /// The lattice assert is a real one, so that a merge across two voxel sizes
    /// cannot silently produce a field that is neither body.
    #[test]
    #[should_panic(expected = "one lattice")]
    fn merging_across_two_lattices_is_refused_outright() {
        let mut into = Volume::new(VOXEL);
        let from = Volume::new(VOXEL * 2.0);
        into.begin_stroke();
        let _ = into.union_from(&from);
    }

    /// An unbracketed merge is not "a merge you cannot undo", it is silent
    /// total data loss: `record_for_undo` does nothing with no recorder open,
    /// so the source is consumed and nothing anywhere can put it back.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "unbracketed merge")]
    fn a_merge_with_no_recorder_open_trips_the_assertion() {
        let mut into = ball(Vec3::ZERO, 4.0);
        let from = ball(Vec3::new(20.0, 0.0, 0.0), 4.0);
        let _ = into.union_from(&from);
    }

    // --- the document operation ----------------------------------------------

    /// Two bodies side by side at the top level, the upper one active, with
    /// their centres `offset` either side of the origin.
    ///
    /// See [`SEPARATE`] and [`TOGETHER`] for why the separation is a parameter
    /// and not a literal: it decides which arm of [`fold`] the whole fixture
    /// takes, and with it whether undo has any real brick contents to restore.
    fn two_balls_at(offset: f32) -> (Document, NodeId, NodeId) {
        let mut doc = Document::from_volume(ball(Vec3::new(-offset, 0.0, 0.0), 8.0));
        let upper = doc.active();
        let lower = doc.add_body("Lower", ball(Vec3::new(offset, 0.0, 0.0), 8.0));
        (doc, upper, lower)
    }

    /// Two bodies side by side at the top level, the upper one active. For the
    /// tests that are about the ROWS rather than about the field.
    fn two_balls() -> (Document, NodeId, NodeId) {
        two_balls_at(SEPARATE)
    }

    #[test]
    fn a_merge_consumes_the_source_and_leaves_the_target_selected_where_it_stood() {
        let (mut doc, upper, lower) = two_balls();
        let outcome = doc.merge_down(upper).expect("there is a body below");

        assert_eq!(outcome.target, lower);
        assert_eq!(doc.body_count(), 1, "the source was not consumed");
        assert!(doc.node(upper).is_none(), "the source row is still in the document");
        assert_eq!(doc.index_of(lower), Some(0), "the result did not land where the source stood");
        assert_eq!(doc.active(), lower, "the result was not selected");
    }

    /// The bottom row of a list has nothing to merge into, and a folder below is
    /// not a body -- both are refusals the caller can name, never a silent
    /// no-op and never a merge into a container.
    #[test]
    fn merge_down_refuses_the_bottom_of_a_list_and_a_folder_below_it() {
        let (mut doc, upper, lower) = two_balls();
        assert_eq!(doc.merge_target(lower), MergeTarget::Bottom);
        assert!(doc.merge_plan(lower).is_none());

        // Wrap the lower body in a folder, which puts a folder row directly
        // below the upper one.
        let (folder, _) = doc.group(lower, "Group 1").expect("room for one more row");
        assert_eq!(doc.merge_target(upper), MergeTarget::Folder(folder));
        assert!(doc.merge_plan(upper).is_none());
        assert!(doc.merge_down(upper).is_none(), "a folder was merged into");
        assert_eq!(doc.body_count(), 2, "a refused merge changed the document");
    }

    /// A body is merged into its sibling and not into whatever happens to be on
    /// the next line: the row below at a shallower depth belongs to the folder's
    /// parent, and reaching it would pull a body out of the folder it is in.
    #[test]
    fn the_last_body_in_a_folder_does_not_merge_into_the_row_below_the_folder() {
        let (mut doc, upper, lower) = two_balls();
        let (_, _) = doc.group(upper, "Group 1").expect("room for one more row");
        // Rows are now: Group 1, upper (depth 1), lower (depth 0).
        assert_eq!(doc.merge_target(upper), MergeTarget::Bottom);
        assert_eq!(doc.body_count(), 2);
        assert!(doc.node(lower).is_some());
    }

    /// The predicted size is what the entry turns out to hold. Not "close to":
    /// the prediction classifies the same bricks with the same function and
    /// measures the same priors, so a difference of one byte is a bug in one of
    /// the two walks.
    #[test]
    fn the_predicted_size_is_exactly_what_the_entry_ends_up_holding() {
        for (offset, what) in
            [(Vec3::new(24.0, 0.0, 0.0), "disjoint"), (Vec3::new(6.0, 0.0, 0.0), "overlapping")]
        {
            let mut doc = Document::from_volume(ball(Vec3::ZERO, 8.0));
            let upper = doc.active();
            let lower = doc.add_body("Lower", ball(offset, 8.0));

            let plan = doc.merge_plan(upper).expect("there is a body below");
            assert_eq!(plan.target, lower);
            let outcome = doc.merge_down(upper).expect("there is a body below");

            assert_eq!(plan.bricks, outcome.bricks, "{what}: the brick count was mispredicted");
            assert_eq!(
                plan.stroke_bytes,
                outcome.entry.stroke_bytes(),
                "{what}: the recorded bricks were mispredicted"
            );
            assert_eq!(
                plan.reclaim_bytes,
                outcome.entry.reclaim_bytes(),
                "{what}: the consumed body was mispredicted"
            );
        }
    }

    /// A disjoint merge adopts every brick, so every prior is `None` and the
    /// bricks half of the entry is one map entry each rather than 128 KiB. The
    /// claim the plan makes about cost, pinned as an exact number.
    #[test]
    fn a_disjoint_merge_records_pointers_and_not_bricks() {
        let mut doc = Document::from_volume(ball(Vec3::ZERO, 8.0));
        let upper = doc.active();
        // Four bricks clear: at this voxel size a brick spans 16 mm, and
        // `seed_sphere` pads its bounds by twice the narrow band, so anything
        // closer would have the two brick maps meeting and some prior would be
        // a real brick.
        doc.add_body("Lower", ball(Vec3::new(64.0, 0.0, 0.0), 8.0));

        let plan = doc.merge_plan(upper).expect("there is a body below");
        assert!(plan.bricks > 0, "the fixture merges nothing");
        assert_eq!(
            plan.stroke_bytes,
            plan.bricks * prior_bytes(None),
            "a disjoint merge recorded brick CONTENTS over {} bricks",
            plan.bricks
        );
    }

    /// Every stored voxel of one body folded into one number, so that "bit for
    /// bit" is a claim the assertion really checks and a failure still prints
    /// something a person can read. Four probes would not catch a merge that
    /// restored the right census with the wrong values.
    fn checksum(volume: &Volume) -> u64 {
        const SEED: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
        coords.sort_unstable();
        let mut hash = SEED;
        for coord in coords {
            for part in [coord.0.x, coord.0.y, coord.0.z] {
                hash = (hash ^ u64::from(part as u32)).wrapping_mul(PRIME);
            }
            let brick = volume.brick(coord).expect("a coordinate the map just handed over");
            for z in 0..BRICK_DIM {
                for y in 0..BRICK_DIM {
                    for x in 0..BRICK_DIM {
                        hash = (hash ^ u64::from(brick.get(x, y, z).to_bits())).wrapping_mul(PRIME);
                    }
                }
            }
        }
        hash
    }

    /// Everything about every row that undo is supposed to restore.
    ///
    /// **`VolumeStats::resident_bytes` is deliberately left out.** It counts the
    /// brick map's CAPACITY, and a map that grew to adopt the source's bricks
    /// does not shrink when undo takes them out again -- so including it would
    /// fail a merge that restored every voxel perfectly, over an allocator
    /// reservation that nothing about the document depends on.
    fn fingerprint(doc: &Document) -> Vec<String> {
        doc.nodes()
            .iter()
            .map(|node| {
                let field = node.volume().map_or_else(
                    || "folder".to_string(),
                    |volume| {
                        let stats = volume.stats();
                        format!(
                            "{} bricks ({} dense, {} uniform) #{:016x}",
                            volume.brick_count(),
                            stats.dense_bricks,
                            stats.uniform_bricks,
                            checksum(volume)
                        )
                    },
                );
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

    /// One ctrl+Z after a merge gives back BOTH bodies, and the field of each is
    /// what it was. The entry is one gesture, so half of it coming back would be
    /// a document state nothing downstream is written against.
    ///
    /// **The overlapping half is the one that can fail.** On separate bodies
    /// every brick is adopted and every recorded prior is `None`, so undo
    /// restores the target by REMOVING bricks and never puts a stored array
    /// back -- move [`Volume::record_for_undo`] after the write in the
    /// [`Fold::Compares`] arm and the separate half is still perfectly green
    /// while every overlapping merge in the hand comes back holding the union.
    /// The `stroke_bytes` assertion below is what stops the fixture drifting
    /// back to that.
    #[test]
    fn a_merge_and_an_undo_restore_both_bodies_bit_for_bit() {
        for (offset, what) in [(SEPARATE, "separate"), (TOGETHER, "overlapping")] {
            let (mut doc, upper, _) = two_balls_at(offset);
            let before = fingerprint(&doc);
            let plan = doc.merge_plan(upper).expect("there is a body below");

            let outcome = doc.merge_down(upper).expect("there is a body below");
            assert_ne!(fingerprint(&doc), before, "{what}: the merge changed nothing to undo");

            // A recorded brick with no contents is one map entry; anything more
            // is a real prior array. So this says "the priors are dense" without
            // reaching into the entry.
            let pointers_only = plan.bricks * prior_bytes(None);
            assert_eq!(
                outcome.entry.stroke_bytes() > pointers_only,
                offset == TOGETHER,
                "{what}: the fixture no longer records the kind of prior it was chosen for"
            );

            let mut history = History::new(crate::DEFAULT_HISTORY_BUDGET);
            history.push(outcome.entry);
            let shown = vec![true; doc.nodes().len()];
            history.undo(&mut doc, &shown);

            assert_eq!(fingerprint(&doc), before, "{what}: the undo did not restore both bodies");
        }
    }

    /// And the redo, because an entry that cannot be replayed is half an undo.
    /// Both separations again: replaying a stored prior over a brick the merge
    /// rewrote is a different code path from replaying an adopted one.
    #[test]
    fn redoing_a_merge_puts_the_merged_document_back() {
        for (offset, what) in [(SEPARATE, "separate"), (TOGETHER, "overlapping")] {
            let (mut doc, upper, _) = two_balls_at(offset);
            let outcome = doc.merge_down(upper).expect("there is a body below");
            let merged = fingerprint(&doc);

            let mut history = History::new(crate::DEFAULT_HISTORY_BUDGET);
            history.push(outcome.entry);
            let shown = vec![true; doc.nodes().len()];
            history.undo(&mut doc, &shown);
            let shown = vec![true; doc.nodes().len()];
            history.redo(&mut doc, &shown);

            assert_eq!(fingerprint(&doc), merged, "{what}: the redo did not put the merge back");
        }
    }

    /// A merge that changes not one voxel still consumes a body, so it is still
    /// one entry and still undoable.
    #[test]
    fn a_merge_that_writes_nothing_is_still_an_undoable_gesture() {
        let mut doc = Document::from_volume(ball(Vec3::ZERO, 4.0));
        let upper = doc.active();
        doc.add_body("Lower", ball(Vec3::ZERO, 30.0));

        let outcome = doc.merge_down(upper).expect("there is a body below");
        assert_eq!(outcome.bricks, 0, "the small ball was not already inside the large one");
        assert_eq!(doc.body_count(), 1);

        let mut history = History::new(crate::DEFAULT_HISTORY_BUDGET);
        history.push(outcome.entry);
        let shown = vec![true; doc.nodes().len()];
        history.undo(&mut doc, &shown);
        assert_eq!(doc.body_count(), 2, "the consumed body did not come back");
    }
}
