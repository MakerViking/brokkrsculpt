// SPDX-License-Identifier: AGPL-3.0-only

//! Cutting the model with a plane.
//!
//! This is where the engine's shape pays off. On a signed distance field, a
//! half-space cut is `max(field, plane)` -- exact, watertight by construction,
//! and it leaves a flat closed face where it cut without any repair step. A mesh
//! sculptor runs boolean remeshing for the same result, and the result is
//! fragile on exactly the input this is for.
//!
//! And this is what BrokkrSculpt is *for*. Scans arrive with defects, and the
//! first thing anyone wants to do is cut the bad part off and keep the rest
//! printable. So the cut has to leave a solid, not a hole: deleting geometry
//! would leave an opening that no slicer will accept, whereas `max` against a
//! plane closes the cut face as a side effect of the arithmetic.
//!
//! # Why this is not `edit_voxels`
//!
//! A cut passes through the whole model, and `edit_voxels` promotes every brick
//! it touches to dense. Most of a scan is solid interior stored as one number
//! per brick, so running the cut through that path would turn a 40 MB model into
//! a gigabyte before it had removed anything.
//!
//! Instead each brick is classified against the plane first, and only the ones
//! the plane actually passes through are ever made dense. The other two cases
//! cost nothing: a brick wholly on the keep side is not touched at all, and one
//! wholly on the cut side is dropped whatever it contained. That is the same
//! shape `resample` uses, and for the same reason.

use glam::{IVec3, Vec3};

use crate::body::{Document, NodeId};
use crate::brick::{BRICK_DIM, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE, brick_index};
use crate::undo::{Change, Entry};
use crate::volume::Volume;

/// A half-space to cut away.
///
/// Everything on the side the `normal` points toward is removed; everything
/// behind it is kept. The plane is infinite and the cut passes through the
/// entire model, which is what ZBrush's clip brushes do and what lopping a
/// defect off a scan needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipPlane {
    /// Any point on the plane.
    pub point: Vec3,
    /// Unit normal. The side it points to is the side that goes.
    pub normal: Vec3,
}

impl ClipPlane {
    /// Build from a point and a direction, normalising the direction.
    ///
    /// Returns `None` for a direction with no length, because a plane with no
    /// normal has no sides and the caller has almost certainly derived it from
    /// a degenerate drag -- a click rather than a stroke.
    pub fn new(point: Vec3, normal: Vec3) -> Option<Self> {
        let normal = normal.try_normalize()?;
        point.is_finite().then_some(Self { point, normal })
    }

    /// Signed distance from a world point to the plane, in millimetres.
    /// Positive on the side that gets cut away.
    #[inline]
    pub fn distance(&self, at: Vec3) -> f32 {
        (at - self.point).dot(self.normal)
    }

    /// The smallest and largest signed distance over an axis aligned box.
    ///
    /// The distance is linear, so the extremes are at opposite corners and can
    /// be had from the centre and the half extent without visiting all eight.
    #[inline]
    fn range_over_box(&self, centre: Vec3, half: Vec3) -> (f32, f32) {
        let middle = self.distance(centre);
        let reach = half.x * self.normal.x.abs()
            + half.y * self.normal.y.abs()
            + half.z * self.normal.z.abs();
        (middle - reach, middle + reach)
    }
}

/// What a plane does to one brick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cut {
    /// Wholly behind the plane by at least the band: nothing changes.
    Keeps,
    /// Wholly past the plane by at least the band: the brick goes entirely.
    Removes,
    /// The plane passes through it, so every voxel has to be resolved.
    Crosses,
}

impl Volume {
    /// Cut away everything on the normal side of `plane`.
    ///
    /// Returns how many bricks changed, which is zero when the plane misses the
    /// model entirely -- worth checking, because a cut that did nothing should
    /// not become an undo entry.
    ///
    /// Bracket a call in [`Volume::begin_stroke`] and [`Volume::end_stroke`] to
    /// make it undoable, exactly as a brush stroke is. Only bricks that really
    /// change are recorded, so cutting a corner off a large model costs an undo
    /// entry proportional to the corner rather than to the model.
    pub fn clip(&mut self, plane: ClipPlane) -> usize {
        let voxel_size = self.voxel_size();
        let band_mm = NARROW_BAND * voxel_size;
        let brick_mm = BRICK_DIM as f32 * voxel_size;
        // A brick spans `BRICK_DIM` voxel positions, so the distance from its
        // first to its last sample is one voxel short of its nominal size.
        let half = Vec3::splat(0.5 * (brick_mm - voxel_size));

        let coords: Vec<BrickCoord> = self.brick_coords().collect();
        let mut changed = 0usize;
        let mut touched: Vec<BrickCoord> = Vec::new();

        for coord in coords {
            let origin = coord.origin();
            let centre = origin.as_vec3() * voxel_size + half;
            let (nearest, farthest) = plane.range_over_box(centre, half);

            let verdict = if nearest >= band_mm {
                Cut::Removes
            } else if farthest <= -band_mm {
                Cut::Keeps
            } else {
                Cut::Crosses
            };

            match verdict {
                // The plane's own contribution is already saturated positive
                // across the whole brick, so `max` would make every voxel
                // OUTSIDE. An absent brick reads as OUTSIDE, so dropping it is
                // both correct and free.
                Cut::Removes => {
                    self.record_for_undo(coord);
                    self.remove_brick(coord);
                    changed += 1;
                    touched.push(coord);
                }
                // `max(field, plane)` is `field` everywhere in here. Touching it
                // would cost a dense promotion and an undo entry for no change.
                Cut::Keeps => {}
                Cut::Crosses => {
                    if self.brick(coord).is_none() {
                        // Absent already reads as OUTSIDE, and `max` cannot
                        // lower it, so there is nothing a cut can do here.
                        continue;
                    }
                    // Recorded before the brick is taken, because the recorder
                    // reads the prior contents out of the map. That copy is the
                    // one undo needs; taking rather than cloning avoids making a
                    // second one of every brick along the cut.
                    self.record_for_undo(coord);
                    let Some(mut brick) = self.take_brick(coord) else {
                        continue;
                    };
                    let data = brick.make_dense();
                    for z in 0..BRICK_DIM {
                        for y in 0..BRICK_DIM {
                            for x in 0..BRICK_DIM {
                                let at = (origin + IVec3::new(x as i32, y as i32, z as i32))
                                    .as_vec3()
                                    * voxel_size;
                                // Both terms in voxels, which is what the field
                                // stores. Mixing millimetres in here would move
                                // the cut by a factor of the voxel size.
                                let slot = brick_index(x, y, z);
                                let cut = plane.distance(at) / voxel_size;
                                data[slot] = data[slot].max(cut).clamp(INSIDE, OUTSIDE);
                            }
                        }
                    }
                    // A brick the plane merely grazed can come out entirely
                    // empty, and one it barely entered can come out unchanged.
                    // Collapsing releases the 128 KB either way.
                    // Already recorded and already taken out of the map, so
                    // these only decide what goes back in.
                    match brick.is_collapsible() {
                        Some(value) if value >= OUTSIDE => {}
                        Some(value) => self.insert_brick(coord, Brick::Uniform(value)),
                        None => self.insert_brick(coord, brick),
                    }
                    changed += 1;
                    touched.push(coord);
                }
            }
        }

        // Every brick that changed, plus its neighbours: a brick's apron reads
        // one voxel into each of the 26 around it, so remeshing only what was
        // written would leave a seam along the cut face.
        for coord in touched {
            let origin = coord.origin();
            self.mark_dirty_voxel_range(origin, coord.max_voxel());
            let _ = origin;
        }
        changed
    }
}

/// What one plane cut did, across the whole document.
///
/// **Both counts, because the status line has to tell "the line missed
/// everything" apart from "it crossed three bodies and changed two".** Those
/// two produce the same brick count and mean completely different things: the
/// first is a gesture that went nowhere near the model, the second is a cut
/// that passed through empty space inside bodies it really did cross.
pub struct CutOutcome {
    /// Bricks changed, summed over every body.
    pub bricks: usize,
    /// Bodies at least one brick of which changed.
    pub bodies_cut: usize,
    /// Bodies the half-space reached at all, whether or not it found anything
    /// there to remove. Always at least `bodies_cut`.
    pub bodies_crossed: usize,
    /// **ONE entry for the whole gesture**, or `None` when nothing changed.
    ///
    /// It is built here rather than handed back as a list of changes for the
    /// caller to wrap, so that "one gesture is one undo entry" is a property of
    /// this type rather than of everybody who calls it remembering. Undoing a
    /// cut that crossed four bodies has to put all four back or none of them:
    /// half a cut is not a document state anything downstream is written
    /// against.
    pub entry: Option<Entry>,
}

impl Document {
    /// Cut every body that is DRAWN with one plane, as one gesture.
    ///
    /// `visible` is indexed by node position and comes from
    /// [`Document::display_visibility`], which is where solo is applied -- so
    /// solo narrows the cut, and a hidden body a line passes over comes out
    /// bit-identical. That is the decision: **a cut is a line the user draws
    /// across what they can see**, so it acts on what is drawn and nothing else.
    /// Cutting a body that is not on screen would set `unsaved`, push history,
    /// pay a remesh and an upload, and change not one pixel.
    ///
    /// # Why this is not a loop over `Volume::clip` at the call site
    ///
    /// Two things that a call-site loop gets wrong and this does not. The undo
    /// entry is ONE [`Entry`] of N `Change::Bricks` rather than N entries, so
    /// one ctrl+Z undoes the whole cut. And the box gate below skips a body the
    /// half-space cannot reach without walking its brick map at all, which is
    /// what keeps a cut across a two-body document from costing a full scan of
    /// the dragon sitting behind it.
    ///
    /// [`Volume::clip`] itself is unchanged and still returns one `usize`; this
    /// sums them.
    pub fn clip(&mut self, plane: ClipPlane, visible: &[bool]) -> CutOutcome {
        debug_assert_eq!(
            visible.len(),
            self.nodes().len(),
            "the visibility mask is indexed by node position"
        );

        let band_mm = NARROW_BAND * self.voxel_size();
        // Resolved up front, so the loop below can take each body mutably in
        // turn without holding a borrow of the node list.
        let crossed: Vec<NodeId> = self
            .nodes()
            .iter()
            .enumerate()
            .filter(|(index, _)| visible.get(*index).copied().unwrap_or(false))
            .filter_map(|(_, node)| {
                let (low, high) = node.bounds()?;
                let centre = (low + high) * 0.5;
                let (_, farthest) = plane.range_over_box(centre, (high - low) * 0.5);
                // Wholly behind the plane by at least the band: every brick in
                // it would classify as `Keeps`, so there is nothing to do and
                // the line did not reach this body at all.
                (farthest > -band_mm).then_some(node.id)
            })
            .collect();

        let mut outcome =
            CutOutcome { bricks: 0, bodies_cut: 0, bodies_crossed: crossed.len(), entry: None };
        let mut changes = Vec::new();

        for body in crossed {
            let Some(volume) = self.volume_mut(body) else {
                continue;
            };
            volume.begin_stroke();
            let bricks = volume.clip(plane);
            let edit = volume.end_stroke();
            // The recorder is the authority on whether anything changed: a
            // count with no edit behind it would push an entry that restores
            // nothing.
            if let Some(edit) = edit.filter(|edit| !edit.is_empty()) {
                outcome.bricks += bricks;
                outcome.bodies_cut += 1;
                changes.push(Change::Bricks { body, edit });
            }
        }

        if !changes.is_empty() {
            outcome.entry = Some(Entry::new(changes));
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOXEL: f32 = 0.5;
    const RADIUS: f32 = 20.0;

    fn ball() -> Volume {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, RADIUS);
        volume.mark_everything_dirty();
        volume
    }

    /// The headline behaviour: everything on the normal side goes, everything
    /// behind it stays.
    #[test]
    fn a_cut_removes_the_side_the_normal_points_at() {
        let mut volume = ball();
        let plane = ClipPlane::new(Vec3::ZERO, Vec3::X).expect("a unit normal");
        assert!(volume.clip(plane) > 0, "the cut did nothing");

        assert!(
            volume.sample_world(Vec3::new(10.0, 0.0, 0.0)) > 0.0,
            "material survived on the cut side"
        );
        assert!(
            volume.sample_world(Vec3::new(-10.0, 0.0, 0.0)) < 0.0,
            "the kept side was removed as well"
        );
    }

    /// The whole reason for `max` rather than deleting geometry: the cut face
    /// has to be closed, or nothing downstream will print it.
    #[test]
    fn a_cut_model_is_still_watertight_and_manifold() {
        let mut volume = ball();
        volume.clip(ClipPlane::new(Vec3::new(3.0, 0.0, 0.0), Vec3::X).unwrap());

        let (mesh, report) = volume.export_mesh();
        assert!(
            report.is_printable(),
            "a cut left the model unprintable: {} ({} triangles)",
            report.summary(),
            mesh.triangles.len()
        );
    }

    /// An off-axis plane is the normal case for lopping a defect off a scan,
    /// and it is where a sloppy brick classification shows up.
    #[test]
    fn an_oblique_cut_is_also_watertight() {
        let mut volume = ball();
        let plane = ClipPlane::new(Vec3::new(2.0, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5)).unwrap();
        assert!(volume.clip(plane) > 0);

        let (_, report) = volume.export_mesh();
        assert!(report.is_printable(), "an oblique cut is not printable: {}", report.summary());
    }

    /// The cut goes through the entire model, front and back, rather than
    /// stopping at the first surface it meets.
    #[test]
    fn a_cut_passes_through_the_whole_model() {
        let mut volume = Volume::new(VOXEL);
        // Two separate balls, one either side of the origin along X.
        volume.seed_sphere(Vec3::new(-25.0, 0.0, 0.0), 8.0);
        volume.seed_sphere(Vec3::new(25.0, 0.0, 0.0), 8.0);
        volume.mark_everything_dirty();

        // A plane at the origin facing +X should take the far ball entirely.
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap());

        assert!(volume.sample_world(Vec3::new(-25.0, 0.0, 0.0)) < 0.0, "the near ball was removed");
        assert!(
            volume.sample_world(Vec3::new(25.0, 0.0, 0.0)) > 0.0,
            "the far ball survived, so the cut did not pass through the model"
        );
    }

    /// A cut that misses must be free and must not become an undo entry.
    #[test]
    fn a_plane_that_misses_the_model_changes_nothing() {
        let mut volume = ball();
        let before: Vec<BrickCoord> = volume.brick_coords().collect();
        let plane = ClipPlane::new(Vec3::new(500.0, 0.0, 0.0), Vec3::X).unwrap();

        assert_eq!(volume.clip(plane), 0, "a plane far outside the model changed bricks");
        assert_eq!(volume.brick_coords().count(), before.len());
    }

    /// The memory point. A cut through a solid model must not promote its
    /// interior tiles to dense: that is what would turn a 40 MB scan into a
    /// gigabyte, and it is invisible in any test that only looks at geometry.
    #[test]
    fn a_cut_does_not_promote_untouched_interior_bricks_to_dense() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 40.0);
        volume.mark_everything_dirty();
        let before = volume.stats();
        assert!(before.uniform_bricks > 0, "the fixture has no interior tiles to protect");

        // Cut a thin sliver off one end, so almost every interior tile is
        // untouched.
        volume.clip(ClipPlane::new(Vec3::new(36.0, 0.0, 0.0), Vec3::X).unwrap());
        let after = volume.stats();

        assert!(
            after.uniform_bricks >= before.uniform_bricks - 4,
            "interior tiles were promoted to dense: {} before, {} after",
            before.uniform_bricks,
            after.uniform_bricks
        );
        assert!(
            after.resident_bytes < before.resident_bytes * 2,
            "a sliver cut doubled the resident memory: {} -> {}",
            before.resident_bytes,
            after.resident_bytes
        );
    }

    /// One cut is one undo entry, and undoing it puts the model back exactly.
    #[test]
    fn a_cut_is_undoable_and_restores_the_field_exactly() {
        let mut volume = ball();
        let probe = Vec3::new(10.0, 0.0, 0.0);
        let before = volume.sample_world(probe);
        let bricks_before = volume.brick_coords().count();

        volume.begin_stroke();
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap());
        let edit = volume.end_stroke().expect("a cut that changed bricks should record an entry");

        assert_ne!(volume.sample_world(probe), before, "the cut did not change the probe");

        volume.apply_edit(edit);
        assert_eq!(volume.sample_world(probe), before, "undo did not restore the field");
        assert_eq!(volume.brick_coords().count(), bricks_before, "undo lost or added bricks");
    }

    /// Cutting the whole model away is legal and must not panic or leave a
    /// half-state.
    #[test]
    fn cutting_everything_away_leaves_an_empty_volume() {
        let mut volume = ball();
        volume.clip(ClipPlane::new(Vec3::new(-RADIUS - 5.0, 0.0, 0.0), Vec3::X).unwrap());
        assert!(
            volume.sample_world(Vec3::ZERO) > 0.0,
            "material survived a plane placed beyond the whole model"
        );
    }

    /// One oblique plane passing is luck; a spread of them passing is the
    /// property.
    ///
    /// This test is the reason the cut is a plain `max` and not something
    /// cleverer. A smooth max was tried first, to round the rim where the cut
    /// face meets the old surface, because that rim is where the mesher leaves
    /// four-way edges. It does not converge: one voxel of rounding left 1 bad
    /// edge, two left 0 on ONE plane but 12 on the (1,1,1) diagonal, three left
    /// 1, four left 6, six left 2. Rounding just moves which cells straddle the
    /// rim. The real answer was that those edges are harmless -- OrcaSlicer
    /// reports `manifold = yes` on a model carrying them -- so the validator
    /// changed instead of the geometry.
    #[test]
    fn oblique_cuts_at_many_angles_are_all_printable() {
        let normals = [
            Vec3::new(1.0, 2.0, -0.5),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-3.0, 1.0, 2.0),
            Vec3::new(0.3, -1.0, 0.7),
            Vec3::new(5.0, -2.0, -1.0),
            Vec3::new(-1.0, -1.0, 4.0),
            Vec3::new(2.0, 0.1, 0.0),
        ];
        for (index, normal) in normals.into_iter().enumerate() {
            // Offset the plane a little differently each time, so the cut does
            // not always land in the same place relative to the lattice.
            let point = Vec3::splat(index as f32 * 0.7 - 2.0);
            let mut volume = ball();
            let plane = ClipPlane::new(point, normal).unwrap();
            assert!(volume.clip(plane) > 0, "plane {index} cut nothing");

            let (_, report) = volume.export_mesh();
            assert!(
                report.is_printable(),
                "cut {index} with normal {normal:?} is not printable: {}",
                report.summary()
            );
        }
    }

    #[test]
    fn a_degenerate_drag_produces_no_plane() {
        assert!(ClipPlane::new(Vec3::ZERO, Vec3::ZERO).is_none());
        assert!(ClipPlane::new(Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)).is_none());
    }

    /// Two cuts in a row compose, which is how a real defect gets trimmed back.
    #[test]
    fn cuts_compose() {
        let mut volume = ball();
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::X).unwrap());
        volume.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap());

        assert!(volume.sample_world(Vec3::new(-10.0, -10.0, 0.0)) < 0.0, "the kept quadrant went");
        assert!(volume.sample_world(Vec3::new(10.0, -10.0, 0.0)) > 0.0);
        assert!(volume.sample_world(Vec3::new(-10.0, 10.0, 0.0)) > 0.0);

        let (_, report) = volume.export_mesh();
        assert!(report.is_printable(), "two cuts left it unprintable: {}", report.summary());
    }
}

/// The cut across a whole document, which is where the decision that a cut
/// crosses every VISIBLE body actually lives.
#[cfg(test)]
mod across_the_document {
    use super::*;
    use crate::body::Document;
    use crate::undo::{History, UndoOutcome};

    const VOXEL: f32 = 0.5;

    /// Two balls side by side along X, both straddling the Y = 0 plane, so one
    /// plane facing +Y takes the top off both.
    fn two_bodies() -> Document {
        let mut first = Volume::new(VOXEL);
        first.seed_sphere(Vec3::new(-15.0, 0.0, 0.0), 8.0);
        first.mark_everything_dirty();
        let mut doc = Document::from_volume(first);

        let mut second = Volume::new(VOXEL);
        second.seed_sphere(Vec3::new(15.0, 0.0, 0.0), 8.0);
        second.mark_everything_dirty();
        doc.add_body("Body 2", second);
        doc
    }

    fn shown(doc: &Document) -> Vec<bool> {
        let mut out = Vec::new();
        doc.display_visibility(None, &mut out);
        out
    }

    /// The headline: a cut acts on everything on screen, not on the active body.
    #[test]
    fn a_cut_crosses_every_visible_body() {
        let mut doc = two_bodies();
        let above = [Vec3::new(-15.0, 5.0, 0.0), Vec3::new(15.0, 5.0, 0.0)];
        let ids: Vec<_> = doc.bodies().map(|(id, _)| id).collect();
        for (id, probe) in ids.iter().zip(above) {
            assert!(doc.volume(*id).unwrap().sample_world(probe) < 0.0, "the fixture has a hole");
        }

        let visible = shown(&doc);
        let outcome = doc.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap(), &visible);

        assert_eq!(outcome.bodies_cut, 2, "the cut reached {} bodies", outcome.bodies_cut);
        assert_eq!(outcome.bodies_crossed, 2);
        assert!(outcome.bricks > 0);
        for (id, probe) in ids.iter().zip(above) {
            assert!(
                doc.volume(*id).unwrap().sample_world(probe) > 0.0,
                "the cut left {id:?} whole"
            );
        }
    }

    /// One gesture is ONE undo entry, however many bodies it touched. Undoing it
    /// has to put all of them back, because half a cut is not a state anything
    /// downstream is written against.
    #[test]
    fn a_cut_across_two_bodies_is_one_undo_entry_that_restores_both() {
        let mut doc = two_bodies();
        let ids: Vec<_> = doc.bodies().map(|(id, _)| id).collect();
        let above = [Vec3::new(-15.0, 5.0, 0.0), Vec3::new(15.0, 5.0, 0.0)];
        let before: Vec<f32> = ids
            .iter()
            .zip(above)
            .map(|(id, at)| doc.volume(*id).unwrap().sample_world(at))
            .collect();

        let visible = shown(&doc);
        let outcome = doc.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap(), &visible);
        let mut history = History::new(64 * 1024 * 1024);
        history.push(outcome.entry.expect("a cut that changed bricks records an entry"));
        assert_eq!(history.stats().undo_entries, 1, "one gesture became more than one entry");

        let visible = shown(&doc);
        assert!(matches!(history.undo(&mut doc, &visible), UndoOutcome::Applied(_)));
        for ((id, at), was) in ids.iter().zip(above).zip(before) {
            assert_eq!(
                doc.volume(*id).unwrap().sample_world(at),
                was,
                "one undo did not restore {id:?}"
            );
        }
        assert!(!history.can_undo(), "the cut left a second entry behind");
    }

    /// A body the user cannot see must come back bit-identical: hiding is a
    /// draw-time skip, so cutting one would cost a remesh and an upload and
    /// change not one pixel.
    #[test]
    fn a_hidden_body_the_line_passes_over_is_left_untouched() {
        let mut doc = two_bodies();
        let ids: Vec<_> = doc.bodies().map(|(id, _)| id).collect();
        let hidden = ids[1];
        let mut meta = doc.meta(hidden).expect("the second body");
        meta.visible = false;
        doc.set_meta(&meta);

        let probe = Vec3::new(15.0, 5.0, 0.0);
        let before = doc.volume(hidden).unwrap().sample_world(probe);
        let bricks_before = doc.volume(hidden).unwrap().brick_count();

        let visible = shown(&doc);
        let outcome = doc.clip(ClipPlane::new(Vec3::ZERO, Vec3::Y).unwrap(), &visible);

        assert_eq!(outcome.bodies_cut, 1, "the cut reached the hidden body");
        assert_eq!(outcome.bodies_crossed, 1);
        assert_eq!(
            doc.volume(hidden).unwrap().sample_world(probe),
            before,
            "the hidden body was cut"
        );
        assert_eq!(doc.volume(hidden).unwrap().brick_count(), bricks_before);
    }

    /// A plane nowhere near the model produces no entry at all -- an undo entry
    /// for a no-op is worse than none, because it costs the user a real one.
    #[test]
    fn a_line_that_misses_everything_records_nothing() {
        let mut doc = two_bodies();
        let visible = shown(&doc);
        let outcome =
            doc.clip(ClipPlane::new(Vec3::new(0.0, 500.0, 0.0), Vec3::Y).unwrap(), &visible);

        assert_eq!(outcome.bricks, 0);
        assert_eq!(outcome.bodies_cut, 0);
        assert_eq!(outcome.bodies_crossed, 0, "a plane 500 mm away counted as crossing a body");
        assert!(outcome.entry.is_none(), "a cut that changed nothing recorded an entry");
    }

    /// The two counts have to be able to differ, or the status line cannot tell
    /// "the line missed" from "it crossed something and found nothing there".
    ///
    /// The gate is a body's world BOX, and a box has corners the body does not
    /// fill: two blobs on a diagonal leave the other two corners of their shared
    /// box empty. A plane through one of those corners reaches the body and
    /// removes nothing.
    #[test]
    fn a_body_the_plane_reaches_but_finds_nothing_in_counts_as_crossed_and_not_cut() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::new(-20.0, -20.0, 0.0), 8.0);
        volume.seed_sphere(Vec3::new(20.0, 20.0, 0.0), 8.0);
        volume.mark_everything_dirty();
        let mut doc = Document::from_volume(volume);

        // Through the empty +X, -Y corner of the shared box, facing out of it.
        let plane = ClipPlane::new(Vec3::new(24.0, -24.0, 0.0), Vec3::new(1.0, -1.0, 0.0)).unwrap();

        let visible = shown(&doc);
        let outcome = doc.clip(plane, &visible);
        assert_eq!(outcome.bodies_crossed, 1, "the plane did not reach the body's box");
        assert_eq!(outcome.bodies_cut, 0, "the plane found something to remove in an empty corner");
        assert_eq!(outcome.bricks, 0);
        assert!(outcome.entry.is_none());
    }
}

#[cfg(test)]
mod provenance {
    use super::*;

    /// The brick classification must produce EXACTLY what touching every voxel
    /// would, or the cut is quietly wrong in the bricks it decided to skip.
    ///
    /// This started as a question -- was the oblique cut's non-manifold rim my
    /// classification or the lattice? -- and the answer was the lattice: both
    /// paths gave an identical 8 non manifold edges. It is kept as a permanent
    /// check because the classification is the part that can silently skip a
    /// brick that needed work, and nothing else here would notice.
    #[test]
    fn the_classification_matches_touching_every_voxel() {
        let plane = ClipPlane::new(Vec3::new(2.0, -1.0, 3.0), Vec3::new(1.0, 2.0, -0.5)).unwrap();

        let mut fast = Volume::new(0.5);
        fast.seed_sphere(Vec3::ZERO, 20.0);
        fast.mark_everything_dirty();
        fast.clip(plane);
        let (_, fast_report) = fast.export_mesh();

        let mut naive = Volume::new(0.5);
        naive.seed_sphere(Vec3::ZERO, 20.0);
        naive.mark_everything_dirty();
        let voxel_size = naive.voxel_size();
        let (lo, hi) = naive.voxel_bounds(Vec3::splat(-30.0), Vec3::splat(30.0));
        naive.edit_voxels(lo, hi, |_, position, value| {
            value.max(plane.distance(position) / voxel_size).clamp(INSIDE, OUTSIDE)
        });
        let (_, naive_report) = naive.export_mesh();

        eprintln!("clip:  {}", fast_report.summary());
        eprintln!("naive: {}", naive_report.summary());
        assert_eq!(
            fast_report.non_manifold_edges, naive_report.non_manifold_edges,
            "the brick classification changed the result, so it skipped a brick that mattered"
        );
        assert_eq!(fast_report.boundary_edges, naive_report.boundary_edges);
        assert_eq!(fast_report.triangles, naive_report.triangles);
    }
}
