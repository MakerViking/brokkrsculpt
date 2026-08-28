// SPDX-License-Identifier: AGPL-3.0-only

//! The 3D viewport, embedded in the Iced widget tree through the `shader`
//! widget.
//!
//! Iced owns the window, the surface and the wgpu device. The `shader` widget
//! hands us that same device inside its `prepare` and `render` callbacks, which
//! is why `wgpu` has to be the exact version `iced_wgpu` depends on. There is
//! no second window and no transparent overlay.
//!
//! The application state cannot be borrowed by the primitive, because a
//! primitive has to be `'static` and `Send`. Instead the two sides share an
//! [`SharedFrame`]: the application pushes finished brick meshes and camera
//! uniforms into it, and `prepare` drains them onto the GPU. Mesh buffers are
//! handed back after upload and reused, so a stroke settles into steady state
//! without allocating.

use std::sync::{Arc, Mutex};

use brokkr_core::{BrickCoord, BrickMesh, BrushKind, MirrorAxis, NodeId};
use brokkr_gpu::{Frustum, OverlayBatch, PixelRect, PoolStats, SculptRenderer, SlotKey, Uniforms};
use iced::mouse;
use iced::widget::shader;
use iced::{Rectangle, Vector};

use crate::app::{SizingTarget, Tool};
use crate::camera::OrbitCamera;
use crate::message::{Message, PointerButton, PointerEvent};
use crate::navcube;

/// One brick's mesh on its way to the GPU.
#[derive(Debug)]
pub struct PendingUpload {
    /// Which brick of which body. The pool is keyed on both, because two bodies
    /// near the world origin share brick coordinates.
    pub key: SlotKey,
    pub mesh: BrickMesh,
}

/// How many finished mesh buffers are kept for reuse.
///
/// **There is a cap because there was not one, and it retained gigabytes.**
/// `BrickMesh::clear` keeps its allocations -- that is the whole point of the
/// recycling -- and nothing ever trimmed the list, so one `rebuild_everything`
/// over the 45,567-brick document this pool is built for handed back one buffer
/// per brick and held every one of them. At an average brick (about 1100
/// vertices, so 26 kB of vertices, 26 kB of indices, 13 kB of cells and 1 kB of
/// mask) that is roughly 67 kB each and about 3 GB retained, none of it counted
/// by the document's own byte budget and none of it visible in any readout.
///
/// 1024 is chosen against what actually recurs: a stroke dirties tens of
/// bricks, so steady-state sculpting never reaches the cap and never allocates,
/// which is the property the recycling exists for. The case the cap bites is
/// the whole-model rebuild, and that one is already proportional to the whole
/// model -- paying for its buffers again is the cheaper half of that trade. At
/// 67 kB a buffer this is about 69 MB held at rest.
const MAX_SPARE_MESHES: usize = 1024;

/// One body's picture, asked for by the application and drawn by the render
/// callback.
///
/// The render thread has no `Document`, so the request carries everything the
/// framing needs. `bounds` is the body's world box, and the framing is worked
/// out from it inside `brokkr-gpu`.
///
/// # Why the box travels with the request rather than being measured here
///
/// **The plan asked for `Volume::surface_bounds` and the code refuses it, in
/// so many words.** `refuse_mirrors_the_body_does_not_straddle` in `app.rs`
/// says "re-measuring per stroke would put a full-model scan on every button
/// release, which is the thing the cache in `brokkr-core` refuses to hold for
/// exactly this reason", and `BodyCache`'s own doc comment refuses to cache the
/// surface box on the same grounds. A thumbnail is requested 200 ms after the
/// geometry stops changing, which for a sculpting session IS every button
/// release, so `surface_bounds` here would be that scan under a different name.
///
/// `Volume::world_bounds` is a walk of the brick map's keys -- tens of
/// microseconds where the surface scan is tens of milliseconds -- and it
/// over-reports by up to one brick a side, so a body fills a little less of its
/// cell than it could. **Revisit when split by loose parts lands** (plan
/// increment 16): a `"<name> fragments"` body holds specks scattered across the
/// whole source model's extent, and brick framing renders those into a blank
/// square. The answer then is a cached surface box on the body, refreshed where
/// split writes it, not a scan per request.
#[derive(Debug, Clone, Copy)]
pub struct ThumbRequest {
    pub body: NodeId,
    pub cell: u32,
    pub bounds: (glam::Vec3, glam::Vec3),
}

/// What the sculpt shader is told about the mask this frame.
///
/// Both halves are VIEW state and neither is document state: the polarity is
/// read out of the body being drawn rather than owned here, and the tint is a
/// preference the `show mask` toggle and its slider set. Nothing here can
/// change what a stroke does -- see [`brokkr_gpu::Uniforms::mask_tint`], which
/// is where the 3D-Coat confusion this avoids is written down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskView {
    /// The active body's polarity, applied to the vertex attribute by the
    /// shader rather than baked into the mesh.
    ///
    /// **The active body's, and every OTHER body is reconciled against it by
    /// [`SharedFrame::set_opposite_polarity`]** -- a mask is per body and this
    /// is one word, so on its own it cannot describe a document where Invert
    /// has been used on one body and not another.
    pub inverted: bool,
    /// How strongly protection is tinted, 0..1. Zero is the toggle switched
    /// off, and draws every body exactly as an unmasked one.
    pub tint: f32,
}

impl Default for MaskView {
    /// Tinted at full strength, because that is the state the application
    /// starts in and because the default has to fail in the direction where a
    /// mask that exists is visible.
    fn default() -> Self {
        Self { inverted: false, tint: 1.0 }
    }
}

/// The hand off between the application and the render callbacks.
#[derive(Debug, Default)]
pub struct SharedFrame {
    /// The camera, not a finished matrix. The projection depends on the
    /// widget's aspect ratio, which only the render callback knows, so the
    /// matrices are built there and there is no frame of lag after a resize.
    camera: Mutex<OrbitCamera>,
    pending: Mutex<Vec<PendingUpload>>,
    /// Mesh buffers returned after upload, so the next remesh can refill them
    /// instead of allocating. This is what keeps a stroke out of the allocator.
    spare: Mutex<Vec<BrickMesh>>,
    stats: Mutex<PoolStats>,
    /// The navigation cube's geometry, in its own batch because it is drawn in
    /// its own pass with its own matrix.
    cube: Mutex<OverlayBatch>,
    /// The transform gizmo's geometry. In its own batch because it is drawn in
    /// its own pass -- the sculpt's matrix, but over the sculpt's depth so a
    /// gizmo on a body's centroid is reachable rather than buried inside it.
    gizmo: Mutex<OverlayBatch>,
    /// Which GPU and backend iced actually chose, recorded the first time the
    /// pipeline is built. The application cannot ask wgpu directly -- iced owns
    /// the adapter -- and it is the first thing a bug report needs.
    adapter: Mutex<Option<String>>,
    /// The brush ring and the mirror planes, rebuilt by the application
    /// whenever something they depend on changes.
    overlay: Mutex<OverlayBatch>,
    /// Set by the application when the whole model is being rebuilt, so the
    /// pool starts from empty instead of fragmenting. See
    /// [`SharedFrame::request_pool_reset`].
    reset_pool: std::sync::atomic::AtomicBool,
    /// Bodies that have left the document and whose pool slots must go with
    /// them. See [`SharedFrame::forget_body`].
    forget: Mutex<Vec<NodeId>>,
    /// Bodies whose slots must go but which are STILL IN THE DOCUMENT, because
    /// their geometry has moved and is about to be uploaded again. See
    /// [`SharedFrame::release_body`].
    release: Mutex<Vec<NodeId>>,
    /// Bodies the renderer must not draw. See [`SharedFrame::set_hidden`].
    hidden: Mutex<Vec<NodeId>>,
    /// Bodies drawn with the opposite of [`MaskView::inverted`]. See
    /// [`SharedFrame::set_opposite_polarity`].
    opposite_polarity: Mutex<Vec<NodeId>>,
    /// How the mask is to be drawn. See [`MaskView`] and
    /// [`SharedFrame::set_mask_view`].
    mask: Mutex<MaskView>,
    /// The one thumbnail waiting to be drawn.
    ///
    /// **Capacity one, so the per-frame cap IS the type.** "At most one body
    /// per frame" is the property the whole feature's cost argument rests on --
    /// the largest body this application is built for is planned against about
    /// 8 ms, which fits inside a 16 ms frame exactly once -- and a queue with a
    /// counter beside it is a rule someone can get wrong. An `Option` is a rule
    /// nobody can get wrong.
    ///
    /// The application only ever fills this when it is empty, so a request that
    /// waits several frames (see [`SharedFrame::apply`]) delays the next one
    /// rather than being overwritten by it.
    thumb: Mutex<Option<ThumbRequest>>,
}

impl SharedFrame {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Take a mesh buffer to fill, reusing a returned one when there is any.
    pub fn take_mesh(&self) -> BrickMesh {
        self.spare.lock().expect("shared frame poisoned").pop().unwrap_or_default()
    }

    /// Queue a filled mesh for upload on the next frame.
    ///
    /// **The body comes from the caller and is not a constant.** Every
    /// `Volume` sits on the same lattice, so two bodies near the world origin
    /// share brick coordinates -- that is the normal case, not a corner one --
    /// and a pool keyed on the coordinate alone would have body B's upload
    /// reuse the slice body A is drawing from. The application passes
    /// `doc.active()` today, because that is the only body a stroke can reach.
    pub fn publish(&self, body: NodeId, coord: BrickCoord, mesh: BrickMesh) {
        let key = SlotKey { body, coord };
        self.pending.lock().expect("shared frame poisoned").push(PendingUpload { key, mesh });
    }

    pub fn set_camera(&self, camera: OrbitCamera) {
        *self.camera.lock().expect("shared frame poisoned") = camera;
    }

    /// Hand over this frame's overlay geometry, taking the caller's buffer and
    /// giving back the previous one so neither side allocates.
    pub fn swap_overlay(&self, batch: &mut OverlayBatch) {
        std::mem::swap(&mut *self.overlay.lock().expect("shared frame poisoned"), batch);
    }

    /// The same hand off for the navigation cube.
    pub fn swap_cube(&self, batch: &mut OverlayBatch) {
        std::mem::swap(&mut *self.cube.lock().expect("shared frame poisoned"), batch);
    }

    /// The same hand off for the transform gizmo.
    pub fn swap_gizmo(&self, batch: &mut OverlayBatch) {
        std::mem::swap(&mut *self.gizmo.lock().expect("shared frame poisoned"), batch);
    }

    /// The overlay geometry currently waiting for the renderer.
    ///
    /// The application swaps its buffer in here rather than keeping it, so this
    /// is the only place the current frame's geometry can be read.
    #[cfg(test)]
    pub fn overlay_snapshot(&self) -> OverlayBatch {
        self.overlay.lock().expect("shared frame poisoned").clone()
    }

    /// The same, for the transform gizmo.
    #[cfg(test)]
    pub fn gizmo_snapshot(&self) -> OverlayBatch {
        self.gizmo.lock().expect("shared frame poisoned").clone()
    }

    /// The GPU iced chose, for diagnostics. "unknown" until the first frame.
    pub fn adapter_summary(&self) -> String {
        self.adapter
            .lock()
            .expect("shared frame poisoned")
            .clone()
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn set_adapter(&self, summary: String) {
        let mut held = self.adapter.lock().expect("shared frame poisoned");
        if held.is_none() {
            *held = Some(summary);
        }
    }

    /// Ask the renderer to empty the pool before the next batch of uploads.
    ///
    /// For the moments that replace every brick at once -- resample, import,
    /// open, re-orient. The pool's allocator never splits or merges blocks, so
    /// those are precisely the moments that fragment it beyond use; emptying it
    /// first is exact and costs nothing, because none of the old slots survive
    /// the rebuild anyway. **The caller must mark everything dirty**, or the
    /// model leaves the GPU and does not come back.
    pub fn request_pool_reset(&self) {
        self.reset_pool.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Take the pending pool reset, exactly as `apply` does, so a test can ask
    /// whether one was requested.
    ///
    /// A take rather than a peek because the two uses always come in that
    /// order: clear whatever is pending, do the thing, assert it is back.
    ///
    /// **This is what the four whole-document swap sites rest on.** Each of
    /// them -- reset, open, import, re-orient -- used to mark the outgoing
    /// model's brick coordinates dirty in the incoming volume so that they
    /// would mesh to nothing and release their slices. All four of those loops
    /// were dead, because all four call `Brokkr::rebuild_everything`, which
    /// asks for this reset, and the reset drops every slot in the pool
    /// regardless of what any key says. They were deleted in increment 6 on
    /// exactly that reasoning -- so if the reset ever stops being asked for,
    /// the reasoning goes with it and four functions quietly start leaving the
    /// previous model on screen underneath the new one.
    #[cfg(test)]
    pub fn take_pool_reset_for_tests(&self) -> bool {
        self.reset_pool.swap(false, std::sync::atomic::Ordering::AcqRel)
    }

    /// Ask the renderer to drop every brick one body owns, because that body
    /// has left the document.
    ///
    /// # Why this is a queue rather than a call
    ///
    /// **`forget_body` cannot be called from where a delete happens.** The
    /// application has no access to the [`brokkr_gpu::MeshPool`] at all: the
    /// pool lives inside the pipeline that Iced owns and hands back only inside
    /// `prepare`, and `publish` runs on the application thread. So a delete
    /// that reached into the pool directly would release the body's slots while
    /// that same body's meshes were still sitting in `pending` -- and `prepare`
    /// would then upload them again, on the same frame, into fresh slices. What
    /// the user would see is a sliver of a deleted body drawn forever, holding
    /// pool space, with no counter moving and nothing in the log.
    ///
    /// [`SharedFrame::apply`] is where the ordering that avoids that is
    /// written down, and it is the reason this is a list and not a method call.
    ///
    /// # What the caller owes afterwards
    ///
    /// **The delete gesture must pair this with `mark_everything_dirty()` +
    /// `remesh_dirty()`**, the same pairing `rebuild_everything` makes around a
    /// pool reset. `MeshPool::forget_body` clears the pool-full banner when it
    /// frees space, and a brick the pool refused while it was full was dropped
    /// on the floor -- its coordinate is long gone from the application's dirty
    /// set, so nothing re-offers it. Freeing space without a remesh therefore
    /// takes the warning down and leaves the missing geometry missing, which is
    /// the silent-geometry-loss shape this project has shipped twice already.
    ///
    /// Its caller is `Brokkr::remove_body`, which is the delete gesture, and
    /// which makes the pairing above. The channel landed here an increment
    /// earlier than that gesture because the ordering above is the part that is
    /// easy to get wrong and impossible to see, and it was worth having pinned
    /// by a test before anything depended on it.
    pub fn forget_body(&self, body: NodeId) {
        self.forget.lock().expect("shared frame poisoned").push(body);
    }

    /// Drop a body's slots because its geometry MOVED, not because the body
    /// went away.
    ///
    /// **Distinct from [`SharedFrame::forget_body`], and conflating the two
    /// makes a body vanish permanently.** A forget means "this body has left
    /// the document", so `apply` drops every queued upload and the pending
    /// thumbnail that name it -- otherwise a delete leaves a ghost sliver drawn
    /// for ever. A transform needs the first half and must not have the second:
    /// the old slots are stale because the bricks are at new coordinates now,
    /// but the new meshes are queued in the SAME frame and dropping them leaves
    /// nothing to draw at all.
    ///
    /// That is exactly what shipped: a gizmo move called `forget_body`, the
    /// re-mesh that followed was discarded by the forget filter, and the body
    /// disappeared from the viewport with a black thumbnail while its field sat
    /// intact in the document, the status line reporting an exact move. The
    /// data was never in danger and that is what made it hard to see.
    pub fn release_body(&self, body: NodeId) {
        self.release.lock().expect("shared frame poisoned").push(body);
    }

    /// Replace the set of bodies the renderer must not draw.
    ///
    /// **Wholesale on every change, never mutated one body at a time.** The set
    /// is a pure function of the document and the solo mode, worked out in one
    /// place -- `Brokkr::publish_visibility`, over
    /// [`brokkr_core::Document::display_visibility`] -- and pushed here whole.
    /// An incremental version has a second owner of the same answer, and two
    /// owners is how the eye and the viewport come to disagree after an undo:
    /// a body invisible on screen that still raycasts and still carves.
    ///
    /// Copies rather than swaps so that the caller keeps its own buffer, and
    /// allocates nothing after the first call.
    pub fn set_hidden(&self, hidden: &[NodeId]) {
        let mut held = self.hidden.lock().expect("shared frame poisoned");
        held.clear();
        held.extend_from_slice(hidden);
    }

    /// The bodies the renderer has been told not to draw.
    ///
    /// The only way to see what the viewport is actually acting on. The check
    /// it exists for is `hidden_snapshot()` against
    /// `doc.display_visibility(solo)` after every message: two computations of
    /// one rule that must never disagree, and the failure when they do is a
    /// body missing from the screen that the panel still shows as visible.
    #[cfg(test)]
    pub fn hidden_snapshot(&self) -> Vec<NodeId> {
        self.hidden.lock().expect("shared frame poisoned").clone()
    }

    /// Say how the mask is to be drawn on the next frame.
    ///
    /// **Published from the one visibility pass, after every message, rather
    /// than from the handful of places that change it** -- the same argument
    /// [`SharedFrame::set_hidden`] makes. The handful is bigger than it looks:
    /// the toggle, the tint slider, the held peek key, choosing another body,
    /// and any undo that puts a polarity back. A list of those is a list that
    /// goes silently out of date, and what it goes out of date INTO is a mask
    /// drawn with the wrong polarity, which is the picture at its most
    /// misleading -- protected material shown as free.
    ///
    /// Costs one lock and two word writes, so it is affordable on the frame
    /// tick, which is what lets it be unconditional.
    pub fn set_mask_view(&self, view: MaskView) {
        *self.mask.lock().expect("shared frame poisoned") = view;
    }

    /// Replace the set of bodies drawn with the opposite of
    /// [`MaskView::inverted`].
    ///
    /// **The other half of `set_mask_view`, and it is published from the same
    /// place in the same pass** -- they are one answer in two pieces, and a
    /// polarity word that arrived without the set it is relative to would draw
    /// protected material as free for one frame. Copies rather than swaps, and
    /// allocates nothing after the first call, exactly as
    /// [`SharedFrame::set_hidden`] does.
    pub fn set_opposite_polarity(&self, opposite: &[NodeId]) {
        let mut held = self.opposite_polarity.lock().expect("shared frame poisoned");
        held.clear();
        held.extend_from_slice(opposite);
    }

    /// The bodies the renderer has been told draw with the opposite polarity.
    ///
    /// The same justification as [`SharedFrame::hidden_snapshot`].
    #[cfg(test)]
    pub fn opposite_polarity_snapshot(&self) -> Vec<NodeId> {
        self.opposite_polarity.lock().expect("shared frame poisoned").clone()
    }

    /// What the renderer has been told about the mask.
    ///
    /// The same justification as [`SharedFrame::hidden_snapshot`]: a test that
    /// asserted on the application's own fields would be asserting on the thing
    /// that SENDS the answer rather than on the answer that arrived, and
    /// deleting the send would leave it green.
    #[cfg(test)]
    pub fn mask_view(&self) -> MaskView {
        *self.mask.lock().expect("shared frame poisoned")
    }

    /// The bodies queued to be dropped from the pool, taken.
    ///
    /// Taken rather than read for the same reason
    /// [`SharedFrame::take_pool_reset_for_tests`] swaps: what a delete owes is
    /// that the request is THERE afterwards, and a test that could see a
    /// previous gesture's request would pass on the wrong evidence.
    #[cfg(test)]
    pub fn take_forgotten_for_tests(&self) -> Vec<NodeId> {
        std::mem::take(&mut *self.forget.lock().expect("shared frame poisoned"))
    }

    /// The released list, for the same reason and with the same swap semantics
    /// as [`SharedFrame::take_forgotten_for_tests`].
    #[cfg(test)]
    pub fn take_released_for_tests(&self) -> Vec<NodeId> {
        std::mem::take(&mut *self.release.lock().expect("shared frame poisoned"))
    }

    /// Ask for one body's picture to be redrawn on the next frame the pool is
    /// quiet.
    ///
    /// The caller must check [`SharedFrame::thumbnail_pending`] first: this
    /// replaces whatever is here, and the application's stale set is what
    /// remembers the request, so a silent overwrite would lose a picture until
    /// that body's geometry changed again.
    pub fn request_thumbnail(&self, request: ThumbRequest) {
        *self.thumb.lock().expect("shared frame poisoned") = Some(request);
    }

    /// Whether a thumbnail is already waiting. The application's "at most one
    /// in flight" gate.
    pub fn thumbnail_pending(&self) -> bool {
        self.thumb.lock().expect("shared frame poisoned").is_some()
    }

    /// Take the queued thumbnail, exactly as `apply` does, so a headless test
    /// can stand in for the render callback.
    ///
    /// A take rather than a peek for the same reason
    /// [`SharedFrame::take_pool_reset_for_tests`] swaps: what the refresh rule
    /// owes is that a request is there afterwards, and a test that could see a
    /// previous gesture's request would pass on the wrong evidence. It is also
    /// the only way to reach the "exactly one" assertions at all -- there is no
    /// `prepare` in a headless test to empty the mailbox.
    #[cfg(test)]
    pub fn take_thumbnail_for_tests(&self) -> Option<ThumbRequest> {
        self.thumb.lock().expect("shared frame poisoned").take()
    }

    /// Make the renderer match everything the application has published, in the
    /// one order that is correct.
    ///
    /// **The order is the whole content of this function**, and none of it is
    /// arbitrary:
    ///
    /// 1. The pool reset comes first, because it drops every slot: a brick
    ///    uploaded before it would be thrown away and never redrawn.
    /// 2. The forget list comes next, and in the same step every queued upload
    ///    naming a forgotten body is dropped. Applying the forgets after the
    ///    uploads would re-upload a deleted body's meshes into fresh slices --
    ///    a ghost sliver drawn forever with no counter moving. Dropping the
    ///    queued uploads is the other half: forgetting the slots alone still
    ///    leaves the meshes in `pending`, which is the same ghost by a longer
    ///    route.
    /// 3. The uploads, whose buffers all come back to `spare` whether they were
    ///    uploaded or dropped -- a dropped upload's allocation is worth exactly
    ///    as much as an uploaded one's.
    /// 4. The hidden set last, so that a body published and hidden on the same
    ///    frame is not drawn once before the skip arrives.
    ///
    /// 5. The thumbnail LAST of all, and only on a frame that did none of the
    ///    above. Two reasons, and both of them are pictures the user would have
    ///    to work out for themselves. After a pool reset the slots are gone
    ///    until the re-upload lands, so a thumbnail drawn in that window is a
    ///    correctly cleared empty cell -- which the application would then mark
    ///    clean, leaving a blank square that never refreshes. And a frame
    ///    carrying uploads is a frame where the pool is already busy, which is
    ///    the one place an extra full-body draw would be felt.
    ///
    /// Extracted out of `prepare` so that this ordering is testable at all;
    /// nothing in the workspace tested this seam before increment 6.
    pub fn apply(&self, renderer: &mut SculptRenderer, device: &wgpu::Device, queue: &wgpu::Queue) {
        let reset_happened = self.reset_pool.swap(false, std::sync::atomic::Ordering::AcqRel);
        if reset_happened {
            renderer.reset_pool();
        }

        let forgotten: Vec<NodeId> = {
            let mut forget = self.forget.lock().expect("shared frame poisoned");
            std::mem::take(&mut *forget)
        };
        for body in &forgotten {
            renderer.forget_body(*body);
        }

        // Released bodies free their slots in the same position as a forget --
        // before the uploads, so a stale slot at an old brick coordinate cannot
        // survive and draw the body twice -- but they are deliberately NOT
        // added to `forgotten`, so the meshes queued behind them in this very
        // frame still land. See `SharedFrame::release_body`.
        let released: Vec<NodeId> = {
            let mut release = self.release.lock().expect("shared frame poisoned");
            std::mem::take(&mut *release)
        };
        for body in &released {
            renderer.forget_body(*body);
        }

        let drained: Vec<PendingUpload> = {
            let mut pending = self.pending.lock().expect("shared frame poisoned");
            std::mem::take(&mut *pending)
        };
        let pool_was_quiet = drained.is_empty();
        if !drained.is_empty() {
            let mut spare = self.spare.lock().expect("shared frame poisoned");
            for upload in drained {
                if !forgotten.contains(&upload.key.body) {
                    renderer.upload_brick(device, queue, upload.key, &upload.mesh);
                }
                // Capped rather than kept: see MAX_SPARE_MESHES. Over the cap
                // the buffer is dropped here and its allocation returned to the
                // system.
                if spare.len() < MAX_SPARE_MESHES {
                    spare.push(upload.mesh);
                }
            }
        }

        {
            let hidden = self.hidden.lock().expect("shared frame poisoned");
            renderer.set_hidden(&hidden);
        }

        {
            let opposite = self.opposite_polarity.lock().expect("shared frame poisoned");
            renderer.set_opposite_polarity(&opposite);
        }

        if reset_happened || !pool_was_quiet {
            return;
        }
        let request = { self.thumb.lock().expect("shared frame poisoned").take() };
        if let Some(request) = request {
            // A request for a body that left the document on this very frame is
            // dropped, exactly as that body's queued uploads are and for the
            // same reason: its cell has already gone back to the application's
            // free list and may belong to somebody else by now.
            if !forgotten.contains(&request.body) {
                renderer.render_thumbnail(
                    device,
                    queue,
                    request.cell,
                    request.body,
                    request.bounds,
                );
            }
        }
    }

    /// The pool counters as of the last frame, for the debug overlay.
    pub fn stats(&self) -> PoolStats {
        *self.stats.lock().expect("shared frame poisoned")
    }

    /// Tests have no GPU and so no renderer to fill the counters in; this is
    /// how they stand in for one. The resample guard reads these numbers.
    #[cfg(test)]
    pub fn set_stats_for_tests(&self, stats: PoolStats) {
        *self.stats.lock().expect("shared frame poisoned") = stats;
    }
}

/// The `shader` widget program. Cheap to construct: `view` rebuilds it every
/// time the UI is laid out, so it holds nothing but a handle.
#[derive(Debug)]
pub struct Viewport {
    shared: Arc<SharedFrame>,
}

impl Viewport {
    pub fn new(shared: Arc<SharedFrame>) -> Self {
        Self { shared }
    }
}

/// Translate a widget local pixel position and the widget size into a pointer
/// event the application can act on.
fn pointer_position(bounds: Rectangle, cursor: mouse::Cursor) -> Option<(Vector, Vector)> {
    // Deliberately not `position_in`: a drag that leaves the widget must keep
    // orbiting, so positions outside the bounds are wanted, not discarded.
    let position = cursor.position()?;
    Some((
        Vector::new(position.x - bounds.x, position.y - bounds.y),
        Vector::new(bounds.width, bounds.height),
    ))
}

fn button_of(button: mouse::Button) -> Option<PointerButton> {
    match button {
        mouse::Button::Left => Some(PointerButton::Left),
        mouse::Button::Right => Some(PointerButton::Right),
        mouse::Button::Middle => Some(PointerButton::Middle),
        _ => None,
    }
}

/// How much one press of the radius keys changes it.
///
/// Multiplicative, because the radius spans fifty to one: a fixed step would
/// crawl at the coarse end and jump in whole multiples at the fine end.
const RADIUS_STEP: f32 = 1.15;

/// The message a key press means, if it means anything.
///
/// Pulled out of `update` so it can be tested without building a widget tree
/// or a window, which on this machine is the only way to test input at all:
/// it is a Wayland session, and XTEST pointer and key synthesis silently does
/// nothing there.
///
/// # This is a pure decode, and deliberately not a guard
///
/// It answers "what does this key spell", never "may it fire now". Whether the
/// application is in a state to accept a shortcut at all — a modal card up
/// over the document, say — is `Brokkr::on_key`'s question, because it is the
/// only one of the two that can see the document. Keep it that way: a state
/// check in here cannot be tested without building the state, which is the
/// whole reason this function is separate.
///
/// # Why `alt` is a parameter with no shortcut of its own yet
///
/// Alt is threaded through so that an unclaimed chord means *nothing* rather
/// than meaning the bare key. Without it `alt+x` toggled X symmetry and
/// `altgr+2` selected the second brush, which on a layout where AltGr composes
/// characters is a keystroke the user meant for something else entirely. It
/// also keeps `ctrl+alt+…` free for the chords the tool strip and the body
/// panel are about to claim.
pub(crate) fn shortcut(character: &str, command: bool, shift: bool, alt: bool) -> Option<Message> {
    if command || alt {
        // The only chorded shortcuts. Anything else with control or alt held
        // belongs to the toolkit, to the window manager, or to a shortcut this
        // application has not defined yet -- and must not fall through to the
        // bare-key table below.
        if command && character == "," {
            // Photoshop's own pair, one level up: ctrl+comma hides the layer
            // you are on, and the alt form is the "show me everything again"
            // escape hatch for having hidden things and lost track of what.
            return Some(if alt {
                Message::EveryBodyShown
            } else {
                Message::ActiveBodyVisibilityToggled
            });
        }
        if command && !alt && character.eq_ignore_ascii_case("g") {
            // Photoshop's pair, unchanged. Shift dissolves the folder the
            // active row is in, so the two are each other's inverse: press
            // ctrl+G then ctrl+shift+G and the document is as it was.
            //
            // `eq_ignore_ascii_case` and not `== "g"`, because a shifted key
            // arrives as "G" -- matching the lowercase form alone would make
            // ctrl+shift+G spell nothing at all, which is the bug this reads
            // as a typo.
            return Some(if shift { Message::BodyUngrouped } else { Message::BodyGrouped });
        }
        let undo_chord = command && !alt && character.eq_ignore_ascii_case("z");
        return undo_chord.then_some(if shift { Message::Redo } else { Message::Undo });
    }

    // Brushes are numbered in the order the tool strip shows them, so the key
    // and the button are always the same thing.
    if let Ok(digit) = character.parse::<usize>()
        && let Some(index) = digit.checked_sub(1)
        && let Some(kind) = BrushKind::ALL.get(index)
    {
        return Some(Message::BrushKindChanged(*kind));
    }

    match character.to_ascii_lowercase().as_str() {
        // ZBrush's own keys for these two sliders, taken rather than invented so
        // muscle memory carries over: S is Draw Size, U is Z Intensity.
        "s" => Some(Message::SizingStarted(SizingTarget::Radius)),
        "u" => Some(Message::SizingStarted(SizingTarget::Strength)),
        // Mask, and `m` rather than `8`. The digits index `BrushKind::ALL`, and
        // making the mask a `BrushKind` would force an arm in five exhaustive
        // matches inside `brokkr-core` for a variant that must never write the
        // field. Pressed while masking it goes back to sculpting, which
        // `Message::ToolChanged` does for every tool.
        "m" => Some(Message::ToolChanged(Tool::Mask)),
        // The transform gizmo. `w` is ZBrush's, Blender's and Maya's key for
        // Move, taken rather than invented so muscle memory carries over --
        // and one key rather than three, because this is one gizmo with move,
        // turn and scale handles on it rather than three modes. Pressed while
        // it is up it goes back to sculpting, which `Message::ToolChanged`
        // does for every tool.
        "w" => Some(Message::ToolChanged(Tool::Transform)),
        // Hold to see the mask, whatever the `show mask` toggle says. ZBrush's
        // own mask-visibility key is H, taken rather than invented so muscle
        // memory carries over, and held rather than latched because that is the
        // gesture a latched toggle cannot do: look, and be back where you were
        // without a second press. The release is decoded in `key_event`
        // alongside the sizing keys -- this function only ever sees presses.
        "h" => Some(Message::MaskPeekStarted),
        // Blender's and Nomad's own key for this, taken rather than invented.
        // It is the one keystroke that recovers ANY camera, however lost, and
        // it is the other half of letting the camera move freely at all.
        "f" => Some(Message::CameraFramedOnActive),
        "x" => Some(Message::SymmetryAxisToggled(MirrorAxis::X)),
        "y" => Some(Message::SymmetryAxisToggled(MirrorAxis::Y)),
        "z" => Some(Message::SymmetryAxisToggled(MirrorAxis::Z)),
        "[" => Some(Message::BrushRadiusScaled(1.0 / RADIUS_STEP)),
        "]" => Some(Message::BrushRadiusScaled(RADIUS_STEP)),
        _ => None,
    }
}

/// Translate an event into the pointer event the application should see, and
/// whether the viewport may CAPTURE it — which stops every widget after it in
/// traversal order from seeing the event at all.
///
/// # Capture is the whole bug surface here, so the rule is stated once
///
/// **Only events that are bounds-checked may capture.** The viewport
/// deliberately handles cursor moves and button releases wherever the cursor
/// is, because a sculpting drag that leaves the viewport must keep sculpting
/// and its release must still end the stroke. From the day the viewport
/// existed (M0, 2026-08-17) this function captured those too — and `iced`'s
/// `button` is exactly the widget that
/// honours capture (`if shell.is_event_captured() { return; }`), while slider,
/// checkbox and text_input ignore it. The shader traverses before the right
/// panel, so every release anywhere in the window was swallowed before a panel
/// button could see it: presses armed the buttons (that arm is bounds-gated),
/// releases never arrived, and **every button in the properties panel was
/// unclickable** while every slider beside them worked. Verified live on
/// 2026-08-21 with a raw-event probe: ~20 presses over panel buttons, zero
/// messages dispatched.
///
/// Publishing without capturing is safe from misfires by iced's own design: a
/// button only fires when the press STARTED on it, so a sculpt release passing
/// over the panel cannot click anything.
///
/// Keyboard shortcuts used to be captured here too, which was its own bug —
/// the capture stole `1`–`7`, `s`, `u`, `x`, `y`, `z` from every text field in
/// the application, because the shader traverses first. They now live in a
/// subscription over events the widget tree IGNORED (`app.rs`), which is what
/// makes them focus-aware: a focused text input consumes its keystrokes, and
/// the shortcut only fires when nothing wanted the key. That subscription
/// forwards the raw key to `Brokkr::on_key`, which is where a shortcut is
/// allowed or refused: it cannot decide that itself, because `listen_with`
/// takes a bare `fn` and can never see the application.
fn route_pointer(
    event: &iced::Event,
    bounds: Rectangle,
    cursor: mouse::Cursor,
) -> Option<(PointerEvent, bool)> {
    let routed = match event {
        iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
            let (position, size) = pointer_position(bounds, cursor)?;
            (PointerEvent::Moved { position, size }, false)
        }
        iced::Event::Mouse(mouse::Event::ButtonPressed(button)) => {
            // Only start a drag that began inside the viewport, so a click
            // on a panel does not sculpt.
            cursor.position_in(bounds)?;
            let (position, size) = pointer_position(bounds, cursor)?;
            (PointerEvent::Pressed { button: button_of(*button)?, position, size }, true)
        }
        iced::Event::Mouse(mouse::Event::ButtonReleased(button)) => {
            // Handled wherever the cursor is: releasing outside the widget
            // still has to end the drag. NOT captured, for the reason above.
            (PointerEvent::Released { button: button_of(*button)? }, false)
        }
        iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
            cursor.position_in(bounds)?;
            let (position, size) = pointer_position(bounds, cursor)?;
            let amount = match delta {
                mouse::ScrollDelta::Lines { y, .. } => *y,
                // Pixel deltas are far larger per notch than line deltas.
                mouse::ScrollDelta::Pixels { y, .. } => *y / 40.0,
            };
            (PointerEvent::Scrolled { amount, position, size }, true)
        }
        iced::Event::Keyboard(iced::keyboard::Event::ModifiersChanged(modifiers)) => (
            PointerEvent::Modifiers {
                shift: modifiers.shift(),
                control: modifiers.control(),
                alt: modifiers.alt(),
            },
            // Broadcast state, not an interaction: capturing it would starve
            // every other widget of the same update.
            false,
        ),
        _ => return None,
    };
    Some(routed)
}

impl shader::Program<Message> for Viewport {
    // Drag state lives in the application, not the widget, because sculpting
    // mutates the volume and that is the application's to own.
    type State = ();
    type Primitive = SculptPrimitive;

    fn draw(&self, _state: &(), _cursor: mouse::Cursor, _bounds: Rectangle) -> SculptPrimitive {
        SculptPrimitive::Viewport { shared: Arc::clone(&self.shared) }
    }

    fn update(
        &self,
        _state: &mut (),
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Message>> {
        let (pointer, captures) = route_pointer(event, bounds, cursor)?;
        let action = shader::Action::publish(Message::Pointer(pointer));
        Some(if captures { action.and_capture() } else { action })
    }

    fn mouse_interaction(
        &self,
        _state: &(),
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if cursor.is_over(bounds) {
            mouse::Interaction::Crosshair
        } else {
            mouse::Interaction::default()
        }
    }
}

/// One row of the body panel, showing one cell of the thumbnail atlas.
///
/// **Inert, and that is the mechanism doing the work rather than an option
/// somewhere.** `Program::update` is left at its default, which returns `None`
/// (`iced_widget-0.14.2/src/shader/program.rs:26-34`), so this widget captures
/// nothing and the press falls through to the row's `mouse_area` behind it. It
/// must not become a `Button`: the thumbnail is the largest target in the row,
/// and Photoshop puts a completely different action on a ctrl-clicked layer
/// thumbnail, which users hit constantly.
#[derive(Debug)]
pub struct ThumbnailCell {
    cell: u32,
}

impl ThumbnailCell {
    pub fn new(cell: u32) -> Self {
        Self { cell }
    }
}

impl shader::Program<Message> for ThumbnailCell {
    type State = ();
    type Primitive = SculptPrimitive;

    fn draw(&self, _state: &(), _cursor: mouse::Cursor, _bounds: Rectangle) -> SculptPrimitive {
        SculptPrimitive::Thumbnail { cell: self.cell }
    }
}

/// What a `shader` widget draws this frame.
///
/// # One type with two arms, and the alternative is a second half-gigabyte pool
///
/// A separate `ThumbnailPrimitive` looks obviously tidier and is the refactor a
/// future reader attempts. It does not work. `iced_wgpu` stores pipelines in a
/// map keyed on `TypeId::of::<T>()` where **T is the PRIMITIVE type**
/// (`iced_wgpu-0.14.0/src/primitive.rs:133`, `:218`), so a second primitive
/// type -- even one declaring `type Pipeline = SculptPipeline` -- gets its own
/// [`SculptPipeline`]: a second [`SculptRenderer`], a second
/// [`brokkr_gpu::MeshPool`], and `MeshPool::new` eagerly creates buffer pair 0.
/// That is half a gigabyte of VRAM reserved for a pool nothing ever uploads to,
/// rendering blank thumbnails, with no error anywhere.
///
/// The [`SculptPrimitive::Thumbnail`] arm holds no `Arc` and touches no mutex:
/// a row's primitive is four bytes and its `prepare` does nothing. That matters
/// because `prepare` is NOT culled per instance
/// (`iced_wgpu-0.14.0/src/lib.rs:366-375`); only `draw` is (`:544-552`).
#[derive(Debug)]
pub enum SculptPrimitive {
    Viewport { shared: Arc<SharedFrame> },
    Thumbnail { cell: u32 },
}

impl shader::Primitive for SculptPrimitive {
    type Pipeline = SculptPipeline;

    fn prepare(
        &self,
        pipeline: &mut SculptPipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &shader::Viewport,
    ) {
        let Self::Viewport { shared } = self else {
            // A row does nothing here. See the type's doc comment: `prepare` runs
            // for every instance whether or not it is on screen, so this arm has
            // to stay free.
            return;
        };

        // The depth attachment has to match the colour attachment, which is the
        // whole window rather than just this widget.
        let size = viewport.physical_size();
        pipeline.renderer.resize(device, size.width, size.height);

        let camera = *shared.camera.lock().expect("shared frame poisoned");
        let mask = *shared.mask.lock().expect("shared frame poisoned");
        let aspect = bounds.width / bounds.height.max(1.0);
        let view = camera.view();
        let view_projection = camera.projection(aspect) * view;
        let uniforms = Uniforms {
            view_projection: view_projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            srgb_target: u32::from(pipeline.renderer.target_is_srgb()),
            mask_inverted: u32::from(mask.inverted),
            mask_tint: mask.tint,
            padding: [0; 1],
        };
        pipeline.renderer.write_uniforms(queue, &uniforms);
        {
            let overlay = shared.overlay.lock().expect("shared frame poisoned");
            pipeline.renderer.write_overlay(device, queue, &overlay, view_projection);
        }
        {
            let cube = shared.cube.lock().expect("shared frame poisoned");
            pipeline.renderer.write_cube(device, queue, &cube, navcube::view_projection(&camera));
        }
        {
            // The SCULPT's matrix, not one of the gizmo's own: it is world
            // space geometry sitting on the body it moves, and only its size is
            // held constant on screen.
            let gizmo = shared.gizmo.lock().expect("shared frame poisoned");
            pipeline.renderer.write_gizmo(device, queue, &gizmo, view_projection);
        }
        // The cube's corner box is defined in logical pixels so it stays the
        // same physical size at any scale factor, but a render pass wants
        // physical ones. Worked out here because `render` sees neither the
        // widget's logical bounds nor the scale factor.
        let scale = viewport.scale_factor();
        let (corner, size) = navcube::corner_rect(glam::Vec2::new(bounds.width, bounds.height));
        pipeline.cube_offset = PixelRect {
            x: (corner.x * scale).max(0.0) as u32,
            y: (corner.y * scale).max(0.0) as u32,
            width: (size.x * scale).max(1.0) as u32,
            height: (size.y * scale).max(1.0) as u32,
        };
        // Culling has to use the same matrix the vertex shader will, or bricks
        // vanish a frame before or after they should.
        pipeline.frustum = Frustum::from_view_projection(view_projection);

        // Everything the application published since the last frame, in the one
        // order that is correct: the reset, then the forgets, then the uploads,
        // then the hidden set, then at most one thumbnail. See
        // `SharedFrame::apply`.
        //
        // The thumbnail is drained HERE, in the viewport's arm, and never in a
        // row's. A row's `prepare` runs whether or not the row is on screen and
        // there may be sixty-four of them; the viewport's runs exactly once a
        // frame, which is what makes "at most one body per frame" true of the
        // schedule as well as of the mailbox.
        shared.apply(&mut pipeline.renderer, device, queue);

        *shared.stats.lock().expect("shared frame poisoned") = pipeline.renderer.stats();
        shared.set_adapter(pipeline.renderer.adapter_summary());
    }

    /// A row is a flat blit inside iced's own pass; the viewport is not.
    ///
    /// iced's render pass has `depth_stencil_attachment: None`
    /// (`iced_wgpu-0.14.0/src/lib.rs:452`), so depth-tested 3D can never ride
    /// it -- which is why the viewport returns `false` here and takes a pass of
    /// its own in [`Self::render`], and why the thumbnail's 3D happened
    /// offscreen during `prepare` and all that is left for this call is a
    /// textured quad. iced has already set the pass's viewport and scissor to
    /// this row's bounds (`lib.rs:551`, `:560`), so the blit needs no rectangle
    /// of its own.
    fn draw(&self, pipeline: &SculptPipeline, pass: &mut wgpu::RenderPass<'_>) -> bool {
        match self {
            Self::Viewport { .. } => false,
            Self::Thumbnail { cell } => pipeline.renderer.blit_thumbnail(pass, *cell),
        }
    }

    fn render(
        &self,
        pipeline: &SculptPipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        if !matches!(self, Self::Viewport { .. }) {
            return;
        }

        pipeline.renderer.render(
            encoder,
            target,
            PixelRect {
                x: clip_bounds.x,
                y: clip_bounds.y,
                width: clip_bounds.width,
                height: clip_bounds.height,
            },
            &pipeline.frustum,
        );

        // The cube's box, offset from the widget into the window. Clamped to the
        // clip rect so a viewport narrower than the cube cannot scissor outside
        // its own bounds and paint over the panels.
        let cube = &pipeline.cube_offset;
        let x = clip_bounds.x + cube.x;
        let y = clip_bounds.y + cube.y;
        let width = cube.width.min(clip_bounds.width.saturating_sub(cube.x));
        let height = cube.height.min(clip_bounds.height.saturating_sub(cube.y));
        // **The gizmo BEFORE the cube, and the order is decided by the press
        // dispatch rather than by taste.** Both clear depth, so whichever draws
        // last wins the pixels they share -- and `Brokkr::on_pointer` tests the
        // navigation cube before it lets the gizmo claim anything. Drawing them
        // the other way round would show a gizmo arrow over the cube that,
        // clicked, flew the camera: the two would disagree about what is on top,
        // and the one the user can see would be the one that is wrong.
        pipeline.renderer.render_gizmo(
            encoder,
            target,
            PixelRect {
                x: clip_bounds.x,
                y: clip_bounds.y,
                width: clip_bounds.width,
                height: clip_bounds.height,
            },
        );
        pipeline.renderer.render_cube(encoder, target, PixelRect { x, y, width, height });
    }
}

/// The GPU state shared by every frame of the viewport. Iced builds this once,
/// the first time it sees a [`SculptPrimitive`].
#[derive(Debug)]
pub struct SculptPipeline {
    renderer: SculptRenderer,
    /// The cube's corner box in physical pixels, relative to the widget's own
    /// origin. Worked out in `prepare`, which is the only place the scale factor
    /// and the logical bounds are both available.
    cube_offset: PixelRect,
    /// Rebuilt in `prepare` from the same matrix the shader gets, and used in
    /// `render`, which is a separate call and has no access to the camera.
    frustum: Frustum,
}

impl shader::Pipeline for SculptPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self {
            renderer: SculptRenderer::new(device, queue, format),
            cube_offset: PixelRect { x: 0, y: 0, width: 0, height: 0 },
            frustum: Frustum::from_view_projection(glam::Mat4::IDENTITY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::widget::shader::Program;
    use iced::{Point, Size};

    /// The capture rule, pinned: **only bounds-checked events may capture.**
    ///
    /// The failure this guards against was live for a year and looked like a
    /// broken panel, not a broken viewport: the shader traverses before the
    /// properties panel, `button` honours capture, and a captured release
    /// anywhere in the window meant no panel button could ever fire. If one of
    /// these assertions starts failing, read `route_pointer`'s comment before
    /// "fixing" the test.
    #[test]
    fn only_bounds_checked_events_capture() {
        let inside = mouse::Cursor::Available(Point::new(500.0, 300.0));
        let outside = mouse::Cursor::Available(Point::new(950.0, 690.0));

        let moved = iced::Event::Mouse(mouse::Event::CursorMoved { position: Point::ORIGIN });
        let pressed = iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left));
        let released = iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left));

        // A move is wanted wherever the cursor is, and must never capture:
        // hover in every widget after the shader depends on seeing it.
        let (_, captures) = route_pointer(&moved, bounds(), outside).expect("moves are routed");
        assert!(!captures, "a cursor move outside the viewport was captured");
        let (_, captures) = route_pointer(&moved, bounds(), inside).expect("moves are routed");
        assert!(!captures, "even inside the viewport, capturing moves starves the panel");

        // A release must arrive wherever the cursor is -- it ends a drag that
        // left the widget -- and must never capture: it is the event a button
        // fires on.
        let (_, captures) =
            route_pointer(&released, bounds(), outside).expect("releases are routed");
        assert!(!captures, "a release outside the viewport was captured");

        // A press inside the viewport is genuinely ours and may capture...
        let (_, captures) = route_pointer(&pressed, bounds(), inside).expect("press inside");
        assert!(captures, "a press inside the viewport should be claimed");
        // ...and a press outside is none of our business at all.
        assert!(
            route_pointer(&pressed, bounds(), outside).is_none(),
            "a press outside the viewport must not reach the sculpt at all"
        );
    }

    /// Keyboard events are none of `route_pointer`'s business: shortcuts fire
    /// from a subscription over IGNORED events, so a focused text field eats
    /// its own keystrokes. This is what keeps `1`-`7`, `s`, `u`, `x`, `y`, `z`
    /// typeable in the print-size field.
    #[test]
    fn keyboard_events_are_not_routed_by_the_viewport() {
        let key = iced::Event::Keyboard(iced::keyboard::Event::KeyPressed {
            key: iced::keyboard::Key::Character("s".into()),
            modified_key: iced::keyboard::Key::Character("s".into()),
            physical_key: iced::keyboard::key::Physical::Code(iced::keyboard::key::Code::KeyS),
            location: iced::keyboard::Location::Standard,
            modifiers: iced::keyboard::Modifiers::default(),
            text: None,
            repeat: false,
        });
        let cursor = mouse::Cursor::Available(Point::new(500.0, 300.0));
        assert!(route_pointer(&key, bounds(), cursor).is_none());
    }

    fn bounds() -> Rectangle {
        Rectangle::new(Point::new(100.0, 50.0), Size::new(800.0, 600.0))
    }

    fn press(at: Point) -> (Viewport, iced::Event, mouse::Cursor) {
        (
            Viewport::new(SharedFrame::new()),
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            mouse::Cursor::Available(at),
        )
    }

    /// The routing from an Iced event to an application message has no other
    /// coverage: the offscreen render test drives the brush directly, and a
    /// broken translation here leaves an application that draws perfectly and
    /// does nothing when clicked.
    #[test]
    fn a_press_inside_the_viewport_reports_a_widget_local_position() {
        let (viewport, event, cursor) = press(Point::new(300.0, 200.0));
        let action = viewport
            .update(&mut (), &event, bounds(), cursor)
            .expect("a press inside the viewport must produce a message");

        let (message, _, status) = action.into_inner();
        assert_eq!(status, iced::event::Status::Captured);
        match message {
            Some(Message::Pointer(PointerEvent::Pressed { button, position, size })) => {
                assert_eq!(button, PointerButton::Left);
                // Widget local, so the widget's own origin is subtracted.
                assert_eq!((position.x, position.y), (200.0, 150.0));
                assert_eq!((size.x, size.y), (800.0, 600.0));
            }
            other => panic!("expected a pressed pointer message, got {other:?}"),
        }
    }

    #[test]
    fn a_press_outside_the_viewport_is_ignored() {
        // Otherwise clicking a panel would also carve the model.
        let (viewport, event, cursor) = press(Point::new(10.0, 10.0));
        assert!(viewport.update(&mut (), &event, bounds(), cursor).is_none());
    }

    #[test]
    fn a_release_outside_the_viewport_still_ends_the_drag() {
        // A drag that runs off the edge of the widget has to be finishable.
        let viewport = Viewport::new(SharedFrame::new());
        let event = iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left));
        let action = viewport
            .update(&mut (), &event, bounds(), mouse::Cursor::Available(Point::new(5.0, 5.0)))
            .expect("a release anywhere must end the drag");

        assert!(matches!(
            action.into_inner().0,
            Some(Message::Pointer(PointerEvent::Released { button: PointerButton::Left }))
        ));
    }

    #[test]
    fn a_mesh_buffer_handed_back_after_upload_is_reused() {
        // This is what keeps a stroke out of the allocator, so it is worth
        // pinning rather than assuming.
        let shared = SharedFrame::new();
        let mut mesh = shared.take_mesh();
        mesh.vertices.reserve(4096);
        let capacity = mesh.vertices.capacity();
        assert!(capacity >= 4096);

        // Any body; this is about the buffer coming back, not about the key.
        shared.publish(NodeId(1), BrickCoord::new(0, 0, 0), mesh);
        // Stand in for what prepare does once the upload has happened.
        let drained: Vec<PendingUpload> =
            std::mem::take(&mut *shared.pending.lock().expect("shared frame poisoned"));
        for upload in drained {
            shared.spare.lock().expect("shared frame poisoned").push(upload.mesh);
        }

        let recycled = shared.take_mesh();
        assert_eq!(recycled.vertices.capacity(), capacity, "the buffer was not reused");
    }
}

/// The ordering inside [`SharedFrame::apply`], against a real pool.
///
/// These need a `wgpu::Device`, because a pool slot is a slice of a real
/// buffer and there is no way to reserve one without creating them. They are
/// the first tests in this crate that do -- everything else here is headless --
/// and that is deliberate: the failure they guard against is a deleted body's
/// queued meshes being uploaded again after its slots were released, which
/// leaves a sliver drawn forever with no counter moving, and no stand-in for
/// the pool can tell you whether the real one kept the slot.
#[cfg(test)]
mod apply_tests {
    use super::*;
    use brokkr_core::Vertex;

    /// A device, or `None` on a machine with no adapter -- but a FAILURE on CI,
    /// where an adapter is always meant to be there.
    ///
    /// The same guard `brokkr-gpu`'s pool tests make, and for the same reason:
    /// `cargo test --workspace` captures output, so a test that printed
    /// "skipping" and returned would report `ok` having asserted nothing, and
    /// someone could put the uploads back above the forgets and leave CI green.
    /// `CI` is set to `true` by GitHub Actions for every step, and by every
    /// other runner worth naming.
    fn device_or_skip(what: &str) -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let opened =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()
                .and_then(|adapter| {
                    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                        .ok()
                });
        if opened.is_some() {
            return opened;
        }
        eprintln!("no usable wgpu adapter, skipping the {what} test");
        assert!(
            std::env::var_os("CI").is_none(),
            "no usable wgpu adapter on CI, so the {what} test asserted nothing. The runner image \
             is meant to provide one (Mesa's lavapipe); if it no longer does, fix the image \
             rather than letting this pass."
        );
        None
    }

    fn renderer(device: &wgpu::Device, queue: &wgpu::Queue) -> SculptRenderer {
        SculptRenderer::new(device, queue, wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    /// A mesh with real geometry in it. The shape is meaningless: these tests
    /// are about which slots exist, not about what is in them.
    fn mesh() -> BrickMesh {
        BrickMesh {
            vertices: vec![Vertex { position: [0.0; 3], normal: [0.0, 1.0, 0.0] }; 64],
            indices: (0..64).collect(),
            cells: Vec::new(),
            mask: vec![0; 64],
        }
    }

    fn publish(shared: &SharedFrame, body: NodeId, bricks: i32) {
        for brick in 0..bricks {
            shared.publish(body, BrickCoord::new(brick, 0, 0), mesh());
        }
    }

    /// Publish over a chosen span of coordinates, so a body can be re-published
    /// somewhere it was not before -- which is what a MOVE does and what
    /// [`publish`] cannot express.
    fn publish_over(shared: &SharedFrame, body: NodeId, span: std::ops::Range<i32>) {
        for brick in span {
            shared.publish(body, BrickCoord::new(brick, 0, 0), mesh());
        }
    }

    /// A mesh that actually covers pixels, for the one test whose claim is
    /// about what is in a cell rather than about which slots exist.
    ///
    /// [`mesh`] cannot serve: its sixty-four vertices are all at the origin, so
    /// every triangle it makes is degenerate and rasterizes to nothing. This is
    /// one quad across the middle of the box those thumbnail tests frame. The
    /// sculpt pipeline is built with `cull_mode: None`, so the winding does not
    /// matter; the normal faces the camera so the matcap lookup lands on the
    /// lit part of the clay rather than on the dark rim.
    fn visible_mesh() -> BrickMesh {
        let corner = |x: f32, y: f32| Vertex { position: [x, y, 0.0], normal: [0.0, 0.0, 1.0] };
        BrickMesh {
            vertices: vec![
                corner(-8.0, -8.0),
                corner(8.0, -8.0),
                corner(8.0, 8.0),
                corner(-8.0, 8.0),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            cells: Vec::new(),
            mask: vec![0; 4],
        }
    }

    /// One cell of the thumbnail atlas, tightly packed, in the texture's own
    /// channel order.
    ///
    /// A near copy of the same helper in `brokkr-gpu`'s `offscreen.rs`, which
    /// is an integration test and so cannot be reached from here. There is no
    /// other way to see a thumbnail, and "the picture in this cell survived"
    /// is the only assertion that can tell the forget guard's two designs
    /// apart.
    fn cell_bytes(
        renderer: &SculptRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell: u32,
    ) -> Vec<u8> {
        let size = brokkr_gpu::THUMBNAIL_SIZE;
        let padded = (size * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thumbnail readback"),
            size: u64::from(padded * size),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thumbnail") });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: renderer.thumbnails().texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: cell },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("readback map failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll failed");

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((size * size * 4) as usize);
        for row in 0..size {
            let start = (row * padded) as usize;
            pixels.extend_from_slice(&mapped[start..start + (size * 4) as usize]);
        }
        drop(mapped);
        readback.unmap();
        pixels
    }

    /// A deleted body's queued meshes must never reach the pool, in either
    /// order, and the pool must come back to the size it was BEFORE that body
    /// was ever published.
    ///
    /// "Unchanged" is the wrong assertion and it is wrong in the direction the
    /// leak hides in: a pool that uploaded the ghost and kept it would also
    /// report a count that had not changed since the upload. So the count is
    /// taken before body B exists at all.
    #[test]
    fn a_forgotten_bodys_queued_uploads_never_reach_the_pool() {
        let Some((device, queue)) = device_or_skip("shared frame apply") else {
            return;
        };

        const KEPT: NodeId = NodeId(1);
        const GOING: NodeId = NodeId(2);

        // Both orders, because the queue does not record when anything was
        // pushed and the two are indistinguishable to `apply` by design.
        for forget_first in [false, true] {
            let mut renderer = renderer(&device, &queue);
            let shared = SharedFrame::new();

            publish(&shared, KEPT, 3);
            shared.apply(&mut renderer, &device, &queue);
            let before = renderer.stats();
            assert_eq!(before.bricks, 3, "the fixture published nothing");

            if forget_first {
                shared.forget_body(GOING);
                publish(&shared, GOING, 4);
            } else {
                publish(&shared, GOING, 4);
                shared.forget_body(GOING);
            }
            shared.apply(&mut renderer, &device, &queue);

            let after = renderer.stats();
            assert_eq!(
                renderer.body_bricks(GOING),
                0,
                "a body forgotten {} its meshes were published still has {} slots",
                if forget_first { "before" } else { "after" },
                renderer.body_bricks(GOING)
            );
            assert_eq!(after.bricks, before.bricks, "the pool grew for a body that was deleted");
            assert_eq!(after.triangles, before.triangles);
            assert_eq!(renderer.body_bricks(KEPT), 3, "the surviving body lost bricks");
        }
    }

    /// **A RELEASED body's new meshes must reach the pool in the same frame.**
    ///
    /// This is the second meaning of "the body's old slots are finished with",
    /// and the distinction between it and a forget is a bug that shipped: a
    /// gizmo move called `forget_body`, `SharedFrame::apply` dropped every
    /// queued upload naming it, and the cube vanished with a black thumbnail
    /// while its field sat intact in the document and the status line said
    /// "moved 50.63 mm -- exact".
    ///
    /// **The test that missed it asserted `world_bounds()` had changed** --
    /// the DATA, which was never in danger -- and nothing asserted what the
    /// RENDERER had been told. Nothing in the GPU module exercised
    /// `release_body` at all: it had exactly two call sites and no test.
    ///
    /// The body is re-published at coordinates it did NOT occupy before,
    /// because that is what a move produces and it is the case that does not
    /// heal itself. `rebake_gizmo` marks the outgoing coordinates dirty in the
    /// incoming field, so those slots are handed back by `upload`'s
    /// empty-indices arm; what is not covered is the halo, since the outgoing
    /// list is stored bricks only while the remesh covers stored bricks plus
    /// their 26 neighbours. Those neighbour slots are in the pool, in nobody's
    /// dirty set, and they are what draws at the old position.
    #[test]
    fn a_released_bodys_new_meshes_reach_the_pool_in_the_same_frame() {
        let Some((device, queue)) = device_or_skip("shared frame release") else {
            return;
        };

        const MOVED: NodeId = NodeId(1);
        const STILL: NodeId = NodeId(2);

        // Both orders, for the same reason the forget test takes both: `apply`
        // does not record when anything was pushed and cannot tell them apart.
        for release_first in [false, true] {
            let mut renderer = renderer(&device, &queue);
            let shared = SharedFrame::new();

            publish_over(&shared, MOVED, 0..5);
            publish_over(&shared, STILL, 100..103);
            shared.apply(&mut renderer, &device, &queue);
            assert_eq!(renderer.body_bricks(MOVED), 5, "the fixture published nothing");
            assert_eq!(renderer.body_bricks(STILL), 3, "the fixture published no second body");

            // The move: released and re-published somewhere else, one frame.
            if release_first {
                shared.release_body(MOVED);
                publish_over(&shared, MOVED, 20..23);
            } else {
                publish_over(&shared, MOVED, 20..23);
                shared.release_body(MOVED);
            }
            shared.apply(&mut renderer, &device, &queue);

            // The assertion that pins a deleted release loop: a release must
            // not swallow the uploads queued alongside it.
            assert!(
                renderer.body_bricks(MOVED) > 0,
                "a released body has no slots at all, so the release ate the meshes queued \
                 with it -- which is a forget wearing a release's name (released {})",
                if release_first { "before publishing" } else { "after publishing" },
            );
            assert_eq!(
                renderer.body_bricks(MOVED),
                3,
                "a released body holds {} slots against the 3 it was re-published with, so the \
                 old position is still in the pool and still drawing",
                renderer.body_bricks(MOVED),
            );
            assert_eq!(renderer.body_bricks(STILL), 3, "an untouched body lost slots");
        }
    }

    /// The forget list is drained, not remembered: a body forgotten on one
    /// frame must not swallow a body that is given the same id later.
    ///
    /// Ids are never reused inside one document, but a document is replaced
    /// wholesale by every open, import and reset, and the new one starts
    /// numbering from 1 again.
    #[test]
    fn the_forget_list_only_applies_to_the_frame_it_was_asked_on() {
        let Some((device, queue)) = device_or_skip("shared frame forget drain") else {
            return;
        };

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();

        shared.forget_body(NodeId(1));
        publish(&shared, NodeId(1), 2);
        shared.apply(&mut renderer, &device, &queue);
        assert_eq!(renderer.stats().bricks, 0);

        publish(&shared, NodeId(1), 2);
        shared.apply(&mut renderer, &device, &queue);
        assert_eq!(
            renderer.stats().bricks,
            2,
            "the forget from the previous frame was still being applied"
        );
    }

    /// A dropped upload's buffer is worth exactly as much as an uploaded one's,
    /// and it has to come back for reuse rather than be freed.
    #[test]
    fn the_meshes_of_a_forgotten_body_come_back_for_reuse() {
        let Some((device, queue)) = device_or_skip("shared frame recycling") else {
            return;
        };

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();
        shared.forget_body(NodeId(9));
        publish(&shared, NodeId(9), 5);
        shared.apply(&mut renderer, &device, &queue);

        assert_eq!(renderer.stats().bricks, 0, "the forgotten body was uploaded");
        assert_eq!(
            shared.spare.lock().expect("shared frame poisoned").len(),
            5,
            "the dropped uploads' buffers were thrown away instead of recycled"
        );
    }

    /// The recycled list is capped, because it was not and that retained about
    /// three gigabytes after one whole-model rebuild.
    ///
    /// `BrickMesh::clear` keeps its allocations -- which is the point of the
    /// recycling -- and nothing ever trimmed the list, so a rebuild over a
    /// 45,567-brick document handed back one buffer per brick and held every
    /// one of them, uncounted by any budget and invisible in every readout.
    #[test]
    fn the_recycled_mesh_list_is_capped() {
        let Some((device, queue)) = device_or_skip("shared frame spare cap") else {
            return;
        };

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();

        let over = MAX_SPARE_MESHES as i32 + 64;
        publish(&shared, NodeId(1), over);
        shared.apply(&mut renderer, &device, &queue);

        assert_eq!(
            shared.spare.lock().expect("shared frame poisoned").len(),
            MAX_SPARE_MESHES,
            "the recycled list grew past its cap"
        );
        // And every brick still got uploaded: the cap throws buffers away, not
        // geometry.
        assert_eq!(renderer.stats().bricks, over as usize);
    }

    /// **A thumbnail must never write the viewport's camera**, and nothing
    /// else in the workspace can see it if it does.
    ///
    /// `SculptRenderer` has exactly one 144-byte viewport uniform buffer, and
    /// iced runs `prepare` for EVERY primitive before `render` for any of them
    /// (`iced_wgpu-0.14.0/src/lib.rs:146-147`). So a thumbnail that wrote its
    /// matrix into that buffer would leave the main viewport drawing the whole
    /// model at an 84 pixel thumbnail's camera for that frame -- no panic, no
    /// validation error, no wrong pixel in any headless harness, just a model
    /// that jumps every time a picture refreshes. That is why the atlas carries
    /// a second buffer and a second bind group over the same layout rather than
    /// two slots in one, and this is the only assertion that can tell the two
    /// designs apart.
    #[test]
    fn a_thumbnail_leaves_the_viewports_own_matrix_alone() {
        let Some((device, queue)) = device_or_skip("thumbnail uniform isolation") else {
            return;
        };

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();

        // A matrix nothing could produce by accident, so a value that survives
        // really is this one and not a default or a thumbnail's.
        let mut wanted = Uniforms::default();
        wanted.view_projection[0][0] = 1234.5;
        wanted.view[3][2] = -67.5;
        renderer.write_uniforms(&queue, &wanted);

        publish(&shared, NodeId(1), 2);
        shared.apply(&mut renderer, &device, &queue);

        shared.request_thumbnail(ThumbRequest {
            body: NodeId(1),
            cell: 0,
            bounds: (glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0)),
        });
        shared.apply(&mut renderer, &device, &queue);
        assert!(!shared.thumbnail_pending(), "the fixture never drained, so it asserts nothing");

        let held = renderer.read_viewport_uniforms(&device, &queue);
        assert_eq!(
            held.view_projection, wanted.view_projection,
            "a thumbnail overwrote the viewport's projection: the model would be drawn at the \
             thumbnail's camera for a frame"
        );
        assert_eq!(held.view, wanted.view, "a thumbnail overwrote the viewport's view matrix");
    }

    /// The thumbnail is drained LAST, at most one a frame, and never on a frame
    /// where the pool is catching up.
    ///
    /// After a reset the slots are gone until the re-upload lands, so a
    /// thumbnail drawn in that window is a correctly cleared EMPTY cell -- and
    /// the application would then mark it clean, leaving a blank square that
    /// never refreshes. A frame carrying uploads is a frame where the pool is
    /// already busy, which is the one place an extra full-body draw is felt.
    #[test]
    fn a_thumbnail_waits_for_a_frame_where_the_pool_has_caught_up() {
        let Some((device, queue)) = device_or_skip("thumbnail scheduling") else {
            return;
        };

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();
        let request = ThumbRequest {
            body: NodeId(1),
            cell: 2,
            bounds: (glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0)),
        };

        // A frame carrying uploads leaves it where it is.
        shared.request_thumbnail(request);
        publish(&shared, NodeId(1), 3);
        shared.apply(&mut renderer, &device, &queue);
        assert!(shared.thumbnail_pending(), "a thumbnail was drawn on a frame carrying uploads");

        // So does a frame that resets the pool, even a quiet one: the slots are
        // gone until the re-upload lands.
        shared.request_pool_reset();
        shared.apply(&mut renderer, &device, &queue);
        assert!(shared.thumbnail_pending(), "a thumbnail was drawn on the frame of a pool reset");

        // And the next quiet frame takes it, exactly once.
        shared.apply(&mut renderer, &device, &queue);
        assert!(!shared.thumbnail_pending(), "the quiet frame did not take the request");
        shared.apply(&mut renderer, &device, &queue);
        assert!(!shared.thumbnail_pending(), "a drained request came back");
    }

    /// A request for a body that left the document on the same frame is
    /// dropped, exactly as that body's queued uploads are.
    ///
    /// **The damage is a BLANKED cell, not a wrong picture**, and the first
    /// version of this comment named the wrong one. By the time the drain runs
    /// the forgotten body's slots are already gone from the pool, so
    /// `render_thumbnail` would draw nothing into the cell -- and the pass
    /// CLEARS before it draws, so what lands is the placeholder. The cell has
    /// meanwhile gone back to the application's free list and may already
    /// belong to another body, so the row that goes blank is that body's, and
    /// it stays blank until its own geometry next changes.
    ///
    /// **The assertion has to see the cell, and the first version of this test
    /// did not.** It asserted `!shared.thumbnail_pending()` and
    /// `body_bricks(GONE) == 0`, and both of those hold whether or not the
    /// guard exists: the mailbox is drained either way and a forgotten body has
    /// no slots either way. Replacing the `if !forgotten.contains(...)` guard
    /// in `apply` with `if true` left every test in this file green. So the
    /// heir's picture is rendered into the cell FIRST and the cell is read back
    /// and compared byte for byte -- which is also why the fixture asserts that
    /// its own render put something other than the background there.
    #[test]
    fn a_thumbnail_for_a_body_forgotten_on_the_same_frame_is_dropped() {
        let Some((device, queue)) = device_or_skip("thumbnail forget") else {
            return;
        };

        /// The body that is deleted on the same frame its picture was asked for.
        const GONE: NodeId = NodeId(1);
        /// The body that inherits its cell.
        const HEIR: NodeId = NodeId(2);
        /// The layer they both point at, which is the whole hazard.
        const SHARED_CELL: u32 = 4;
        const BOUNDS: (glam::Vec3, glam::Vec3) =
            (glam::Vec3::new(-10.0, -10.0, -10.0), glam::Vec3::new(10.0, 10.0, 10.0));

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();

        publish(&shared, GONE, 2);
        shared.publish(HEIR, BrickCoord::new(0, 0, 0), visible_mesh());
        shared.apply(&mut renderer, &device, &queue);

        // The heir's picture goes in first, so there is a real one to lose.
        shared.request_thumbnail(ThumbRequest { body: HEIR, cell: SHARED_CELL, bounds: BOUNDS });
        shared.apply(&mut renderer, &device, &queue);
        let painted = cell_bytes(&renderer, &device, &queue, SHARED_CELL);
        let background = brokkr_gpu::background_texel(wgpu::TextureFormat::Rgba8UnormSrgb);
        assert!(
            painted.as_chunks::<4>().0.iter().any(|texel| *texel != background),
            "the fixture drew nothing into the cell, so the comparison below would pass for a \
             cleared cell too"
        );

        shared.forget_body(GONE);
        shared.request_thumbnail(ThumbRequest { body: GONE, cell: SHARED_CELL, bounds: BOUNDS });
        shared.apply(&mut renderer, &device, &queue);

        // Taken from the mailbox either way -- a request nobody drains would
        // block every later one -- and the body's slots really are gone.
        assert!(!shared.thumbnail_pending(), "the request for a deleted body was left queued");
        assert_eq!(renderer.body_bricks(GONE), 0);
        // ...and the heir's row still has its picture.
        assert!(
            cell_bytes(&renderer, &device, &queue, SHARED_CELL) == painted,
            "the dropped request was drawn anyway: it cleared the cell, so the body that \
             inherited that layer shows an empty well until its own geometry next changes"
        );
    }

    /// The hidden set has to arrive at the renderer, and the frame it arrives
    /// on has to be one where the body it names is not drawn.
    ///
    /// `apply` sets it AFTER the uploads for that reason: a body published and
    /// hidden on the same frame would otherwise be drawn once first.
    ///
    /// **Every assertion here is on the RENDERER, and the first version of
    /// this test got that wrong.** It asserted `renderer.stats().bricks == 4`,
    /// `renderer.body_bricks(NodeId(2)) == 2` and
    /// `shared.hidden_snapshot() == vec![NodeId(2)]`. The first two hold
    /// whether or not the set was delivered -- hiding is a draw-time skip, so
    /// slot counts are invariant across it by design -- and the third reads the
    /// `SharedFrame`'s own mutex, which is the SENDER. Replacing the
    /// `renderer.set_hidden(&hidden)` line in `apply` with `let _ = &hidden;`
    /// left the whole workspace suite green, with the eye muting the panel row
    /// and the body still fully drawn on screen. `hidden_bodies()` exists so
    /// this test can ask the receiver what it was told.
    #[test]
    fn the_hidden_set_reaches_the_renderer_on_the_frame_it_is_published() {
        let Some((device, queue)) = device_or_skip("shared frame hidden set") else {
            return;
        };

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();

        publish(&shared, NodeId(1), 2);
        publish(&shared, NodeId(2), 2);
        shared.set_hidden(&[NodeId(2)]);
        shared.apply(&mut renderer, &device, &queue);

        // Hidden is a draw-time skip, so the slots are all there...
        assert_eq!(renderer.stats().bricks, 4);
        assert_eq!(renderer.body_bricks(NodeId(2)), 2);
        // ...and the renderer itself, not the channel that fed it, is what says
        // the set arrived.
        assert_eq!(renderer.hidden_bodies(), &[NodeId(2)]);

        // Showing it again is the empty set, not the absence of a call.
        shared.set_hidden(&[]);
        shared.apply(&mut renderer, &device, &queue);
        assert!(
            renderer.hidden_bodies().is_empty(),
            "showing a body again never reached the renderer"
        );
    }

    /// The opposite-polarity set has to arrive at the renderer too, and for
    /// the same reason, asserted the same way.
    ///
    /// Polarity is a bind-group choice at draw time, so like hiding it moves no
    /// slot count a test could look at, and like hiding it is delivered through
    /// `apply`. Every assertion is on the RECEIVER: `opposite_polarity_snapshot`
    /// would read the sender's own mutex and stay green if the delivering line
    /// were deleted.
    #[test]
    fn the_opposite_polarity_set_reaches_the_renderer_on_the_frame_it_is_published() {
        let Some((device, queue)) = device_or_skip("shared frame polarity set") else {
            return;
        };

        let mut renderer = renderer(&device, &queue);
        let shared = SharedFrame::new();

        publish(&shared, NodeId(1), 2);
        publish(&shared, NodeId(2), 2);
        shared.set_opposite_polarity(&[NodeId(2)]);
        shared.apply(&mut renderer, &device, &queue);

        assert_eq!(renderer.stats().bricks, 4, "polarity is not a slot count");
        assert_eq!(renderer.opposite_polarity_bodies(), &[NodeId(2)]);

        // Agreeing again is the empty set, not the absence of a call.
        shared.set_opposite_polarity(&[]);
        shared.apply(&mut renderer, &device, &queue);
        assert!(
            renderer.opposite_polarity_bodies().is_empty(),
            "the body agreeing again never reached the renderer"
        );
    }
}

#[cfg(test)]
mod shortcut_tests {
    use super::*;

    /// Note the test target: `shortcut`, not a synthesised key event. This is
    /// a Wayland session, where XTEST key and pointer synthesis silently does
    /// nothing, so driving the real widget from a test is not an option.
    fn press(character: &str) -> Option<Message> {
        shortcut(character, false, false, false)
    }

    /// The same key with control held, and nothing else.
    fn control(character: &str) -> Option<Message> {
        shortcut(character, true, false, false)
    }

    #[test]
    fn the_digits_select_brushes_in_the_order_the_tool_strip_shows_them() {
        for (index, kind) in BrushKind::ALL.into_iter().enumerate() {
            let key = (index + 1).to_string();
            assert!(
                matches!(press(&key), Some(Message::BrushKindChanged(selected)) if selected == kind),
                "key {key} should select {kind}"
            );
        }
        // One past the last brush, so the mapping cannot silently wrap round.
        let past_the_end = (BrushKind::ALL.len() + 1).to_string();
        assert!(press(&past_the_end).is_none());
        assert!(press("0").is_none(), "there is no brush zero");
    }

    #[test]
    fn xyz_toggle_their_own_mirror_plane() {
        for (key, axis) in [("x", MirrorAxis::X), ("y", MirrorAxis::Y), ("z", MirrorAxis::Z)] {
            assert!(
                matches!(press(key), Some(Message::SymmetryAxisToggled(a)) if a == axis),
                "{key} should toggle {}",
                axis.label()
            );
            // Shift alone must not change what a letter means.
            assert!(matches!(
                shortcut(key, false, true, false),
                Some(Message::SymmetryAxisToggled(_))
            ));
        }
        // Capitals reach the same place.
        assert!(matches!(press("X"), Some(Message::SymmetryAxisToggled(MirrorAxis::X))));
    }

    /// `z` is the collision worth pinning: bare it mirrors, with control it
    /// undoes, and the two must never be confused.
    #[test]
    fn control_z_still_undoes_rather_than_mirroring() {
        assert!(matches!(control("z"), Some(Message::Undo)));
        assert!(matches!(shortcut("z", true, true, false), Some(Message::Redo)));
        assert!(matches!(control("Z"), Some(Message::Undo)));
        assert!(matches!(press("z"), Some(Message::SymmetryAxisToggled(MirrorAxis::Z))));
    }

    #[test]
    fn the_bracket_keys_scale_the_radius_in_opposite_directions() {
        let Some(Message::BrushRadiusScaled(down)) = press("[") else {
            panic!("[ did not change the radius");
        };
        let Some(Message::BrushRadiusScaled(up)) = press("]") else {
            panic!("] did not change the radius");
        };
        assert!(down < 1.0, "[ should shrink the brush");
        assert!(up > 1.0, "] should grow it");
        // One in each direction returns to where it started.
        assert!((down * up - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn keys_that_mean_nothing_are_left_for_the_toolkit() {
        assert!(press("q").is_none());
        assert!(press("").is_none());
        // Every other control chord belongs to the toolkit or the window
        // manager, so claiming it here would steal it.
        assert!(control("s").is_none());
        assert!(control("x").is_none(), "ctrl x is not a mirror toggle");
        assert!(control("1").is_none());
    }

    /// Alt is the modifier that was silently ignored, and this is what the
    /// fourth parameter buys. On a layout where AltGr composes characters,
    /// `altgr+2` is the user typing something, not asking for the second
    /// brush; and `ctrl+alt+z` has to stay free rather than being a second
    /// spelling of undo.
    #[test]
    fn a_chord_this_application_has_not_claimed_means_nothing_at_all() {
        let alt = |character: &str| shortcut(character, false, false, true);
        for key in ["x", "y", "z", "1", "2", "s", "u", "[", "]"] {
            assert!(alt(key).is_none(), "alt {key} fell through to the bare key");
        }
        assert!(shortcut("z", true, false, true).is_none(), "ctrl alt z is not undo");
        assert!(shortcut("z", true, true, true).is_none(), "ctrl alt shift z is not redo");

        // The control, without which every assertion above passes on a
        // function that always returns None.
        assert!(press("x").is_some());
        assert!(control("z").is_some());
    }

    /// `ctrl+G` and `ctrl+shift+G`, and the capital is the whole point.
    ///
    /// A shifted key arrives as `"G"`, so matching the lowercase form alone
    /// would make ctrl+shift+G spell nothing at all -- the ungroup half of the
    /// pair would simply never fire, with no error anywhere.
    #[test]
    fn ctrl_g_groups_and_ctrl_shift_g_ungroups_whatever_case_arrives() {
        assert!(matches!(control("g"), Some(Message::BodyGrouped)));
        assert!(matches!(control("G"), Some(Message::BodyGrouped)));
        assert!(matches!(shortcut("G", true, true, false), Some(Message::BodyUngrouped)));
        assert!(matches!(shortcut("g", true, true, false), Some(Message::BodyUngrouped)));

        // Bare `g` is unclaimed and must stay so, and ctrl+alt+G is not a
        // second spelling of either.
        assert!(press("g").is_none(), "bare g was claimed");
        assert!(shortcut("g", true, false, true).is_none(), "ctrl alt g is not group");
    }
}
