// SPDX-License-Identifier: AGPL-3.0-only

//! The document: an ordered list of nodes sharing one voxel lattice.
//!
//! # One lattice, N volumes
//!
//! A body is a [`Volume`] plus a name, an eye and a stable id. A [`Document`]
//! owns the list and **one** `voxel_size`, and that single number is the whole
//! reason this type exists: every body sits on the same lattice, so voxel
//! (0,0,0) is world (0,0,0) in all of them and brick `c` covers the same world
//! box in all of them. A boolean between two bodies is then a brick-by-brick
//! `min`/`max` with no resampling and no interpolation loss, and a body's
//! position IS its brick occupancy -- there is no per-body transform anywhere
//! in this file, and adding one would give away the property the design is
//! built on. See `bricks_of_two_volumes_at_one_voxel_size_cover_the_same_world_box`
//! at the bottom of this file, which is the measurement everything else rests
//! on.
//!
//! # Why a tree type when there is one body
//!
//! [`Node`] carries `depth`, `collapsed` and an optional body from its first
//! commit, and the application pins all of it: one node, `depth` 0, `body`
//! always `Some`. That is deliberate rather than speculative. Folder rows are a
//! decided requirement, and the alternative is converting a `Vec<Body>` into a
//! `Vec<Node>` across six later changes plus every test that names a body by
//! index. The invariants that hold *today* are asserted in
//! [`Document::assert_invariants`], in one place, so relaxing them later is an
//! edit to that function rather than an archaeology exercise.
//!
//! Position in the tree is `(preorder index, depth)` and nothing else. There is
//! no parent pointer, which is what makes a cycle unrepresentable rather than
//! merely detectable.
//!
//! `nodes` is a `Vec` and never a hash map. `project::write` has to produce
//! identical bytes twice -- `writing_the_same_volume_twice_gives_identical_bytes`
//! pins it -- and a hash order breaks that nondeterministically, which is the
//! shape of failure that passes locally and fails on CI. List order is also
//! user-visible state that has to round-trip.

use glam::Vec3;
use rayon::prelude::*;

use crate::brick::{BRICK_VOXELS, Brick, BrickCoord, NARROW_BAND};
use crate::mesh::{BrickMesh, MeshScratch};
use crate::project::MAX_VOLUME_BYTES;
use crate::raycast::{Hit, raycast};
use crate::volume::{PARALLEL_MESH_THRESHOLD, Volume, VolumeStats};

/// Which node, for as long as the document lives.
///
/// It is `NodeId` and not `BodyId` because the tree the panel shows holds
/// folder rows as well as bodies, and an id that could name either is one name
/// rather than two. Zero is reserved for "no node", so a live id is always
/// nonzero.
///
/// This type appeared first in `brokkr-gpu`'s mesh pool, which was the first
/// thing in the application that had to tell two bodies apart. It lives here
/// now, where ids are actually handed out, and `brokkr-gpu` re-exports it --
/// never the other way round, because `brokkr-core` may not depend on a GPU
/// crate and CI fails the build if it ever does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// At most 64 bodies.
///
/// A count read out of a file decides an allocation, so it needs a bound for
/// the same reason `MAX_KEYS` has one. 64 rather than 256 because the BYTES
/// refuse first: 64 of the default seeded sphere (about 408 bricks, some 40
/// MiB) is 2.5 GiB against the 6 GiB ceiling, and nine bodies the size of the
/// measured dragon (765 MiB) is 6.7 GiB. Eight is 5.98 GiB and does fit, which
/// is why the real limit is stated in the high single digits rather than here.
pub const MAX_BODIES: usize = 64;

/// At most 128 nodes -- bodies plus folder rows.
///
/// Separate from [`MAX_BODIES`] because a folder row costs a few dozen bytes
/// and no brick stream, so the bytes argument does not bound it. A count read
/// from a file still decides an allocation, so it needs its own bound.
pub const MAX_NODES: usize = 128;

/// Eight levels of nesting: legal depths are `0 ..= MAX_DEPTH - 1`.
///
/// Always written as `depth < MAX_DEPTH` and never `<=`, because the two
/// readings of this constant differ by one.
///
/// **This is a panel width bound and nothing else.** Say that plainly or
/// somebody will harden it into a false constraint: nothing in the reader
/// recurses, and [`MAX_NODES`] is the real backstop, since a preorder list of
/// N nodes can be at most N-1 deep whatever this says.
pub const MAX_DEPTH: u8 = 8;

/// What a body row holds that a folder row does not.
///
/// Boxed inside [`Node`] so that a folder row does not carry a `Volume`-sized
/// hole. Derives nothing, because [`Volume`] derives nothing -- not even
/// `Debug`, and deliberately not `Clone`. A body delete MOVES its volume into
/// the undo entry rather than cloning it; a clone is simply not available, and
/// that pushes the right way. Duplicating a body will get an explicitly named
/// `Volume::duplicated`, because `.clone()` is one keystroke and a name is
/// something a reviewer stops on.
struct BodyData {
    volume: Volume,
    cache: BodyCache,
}

/// The per-body numbers a caller cannot afford to work out for itself.
///
/// **One field, and the reason there is one rather than four is the cost of
/// keeping each honest.** The design this came from listed four: the stats, the
/// brick-extent box, the tight surface box and a radius. Three of those are not
/// here:
///
/// - the stats are summed by [`Document::totals`], which walks the brick maps
///   exactly as the single-volume code it replaced did, and a cache with one
///   writer and one reader would buy nothing but a way for the two to disagree;
/// - the tight surface box comes from [`Volume::surface_bounds`], which scans
///   every dense brick and whose own documentation forbids calling it per
///   frame. Held here it would have to be refreshed on every remesh, which is
///   to say on every pointer event of every stroke: a full-model scan per
///   event. Its callers -- the mirror-straddle refusal, the resize readout --
///   are user actions and pay for it there;
/// - the radius is [`Volume::content_radius`] over the box below.
///
/// So the cache holds the one number a per-pointer-event caller genuinely needs
/// and cannot recompute: the pick gate's box.
#[derive(Debug, Clone, Copy, Default)]
struct BodyCache {
    /// A world box that CONTAINS this body's bricks, or `None` when it holds
    /// none.
    ///
    /// **A superset, never a subset, and that asymmetry is the whole safety
    /// argument.** It gates a raycast: a box that is too big costs a march that
    /// finds nothing, and a box that is too small drops a hit the user can see
    /// on screen -- the cursor would simply stop working over part of the
    /// model. So the incremental refresh in [`Document::take_dirty`] only ever
    /// GROWS it, by taking in each dirty brick, and never shrinks it. A body
    /// carved back down to a third of its size keeps the box it once needed
    /// until something recomputes it in full, which costs a few marches that
    /// miss and cannot cost a hit.
    bounds: Option<(Vec3, Vec3)>,
}

impl BodyCache {
    /// Take one brick into the box.
    ///
    /// Grows only. See [`BodyCache::bounds`] for why that direction is the safe
    /// one and what it costs.
    fn take_in(&mut self, coord: BrickCoord, voxel_size: f32) {
        let low = coord.origin().as_vec3() * voxel_size;
        let high = coord.max_voxel().as_vec3() * voxel_size;
        self.bounds = Some(match self.bounds {
            Some((was_low, was_high)) => (was_low.min(low), was_high.max(high)),
            None => (low, high),
        });
    }
}

impl BodyData {
    fn new(volume: Volume) -> Self {
        let cache = BodyCache { bounds: volume.world_bounds() };
        Self { volume, cache }
    }

    /// Recompute the box from scratch, for the operations that rewrite the
    /// whole field or move the lattice under it.
    ///
    /// A walk of the brick map's keys. Rescale, resample and a quarter turn all
    /// need it: each of them changes where an unchanged brick sits in the
    /// world, so growing the old box would be growing the wrong box.
    fn recompute_bounds(&mut self) {
        self.cache.bounds = self.volume.world_bounds();
    }
}

/// Everything about a node that is NOT its volume: the whole of what rename,
/// the eye, collapse, reorder, group and ungroup can change.
///
/// `depth` is in here and that is load-bearing -- without it the whole-outline
/// change that folders need cannot express a reparent, which is a permutation
/// *plus* a depth edit. It also lives here rather than beside
/// [`crate::undo::Change`], which is its first consumer, because [`Node::depth`]
/// is a private field: a snapshot type in another module could not read it
/// without opening the field to the whole crate, and the point of the private
/// field is that only [`Document::assert_invariants`] decides what a legal
/// depth is.
///
/// By value rather than by reference because it is also the first element of
/// the export path's per-body tuple, which outlives any borrow of the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeMeta {
    pub id: NodeId,
    pub depth: u8,
    pub name: String,
    pub visible: bool,
    pub collapsed: bool,
}

impl NodeMeta {
    /// What one snapshot costs the undo budget.
    ///
    /// `capacity` and not `len`, because the budget is counting what is really
    /// held rather than what would be written to a file.
    #[inline]
    pub fn bytes(&self) -> usize {
        size_of::<Self>() + self.name.capacity()
    }
}

/// One row of the document: a body, or (later) a folder.
pub struct Node {
    pub id: NodeId,
    /// `0 ..= MAX_DEPTH - 1`. Position in the tree IS (preorder index, depth).
    ///
    /// Private because every node is at depth 0 in this build and the
    /// invariant check is what says so; the setter arrives with folders.
    depth: u8,
    /// At most 32 bytes of UTF-8, which is the file format's fixed field.
    pub name: String,
    /// This node's OWN eye. Never written by an ancestor's eye, never by solo,
    /// and never by undo -- all three are masks applied on top of it, and the
    /// resolver that combines them arrives with the second body there is to
    /// hide.
    pub visible: bool,
    /// Folders only; repaired to false on a body.
    ///
    /// A collapsed folder must never change what a command does. In ZBrush,
    /// deleting a subtool inside a closed folder deletes the whole folder, and
    /// a user reported losing an unrecoverable hour to it. Collapse changes
    /// only what is drawn.
    pub collapsed: bool,
    /// `None` IS the folder.
    ///
    /// The kind is derived rather than stored, so there is no enum tag that can
    /// disagree with the payload.
    body: Option<Box<BodyData>>,
}

impl Node {
    /// A body row at depth 0.
    fn body(id: NodeId, name: String, volume: Volume) -> Self {
        Self {
            id,
            depth: 0,
            name,
            visible: true,
            collapsed: false,
            body: Some(Box::new(BodyData::new(volume))),
        }
    }

    /// A body row rebuilt from a snapshot, which is what a node table read out
    /// of a file amounts to.
    ///
    /// **The depth is clamped here rather than trusted, and this is the
    /// clamping constructor [`MAX_DEPTH`] refers to.** `resolve_visibility`
    /// walks a fixed `[bool; MAX_DEPTH]` ancestor chain, and a depth past the
    /// end of it is an index out of bounds -- a panic in the one function every
    /// frame calls. The reader refuses a non-zero depth outright at container
    /// version 3, so nothing reaches this clamp today; it is here because the
    /// reader is not the only thing that builds a document, and the split, the
    /// group and every test helper come through this constructor instead.
    pub(crate) fn from_meta(meta: NodeMeta, volume: Volume) -> Self {
        Self {
            id: meta.id,
            depth: meta.depth.min(MAX_DEPTH - 1),
            name: meta.name,
            visible: meta.visible,
            collapsed: meta.collapsed,
            body: Some(Box::new(BodyData::new(volume))),
        }
    }

    #[inline]
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// Whether this row holds a field, as opposed to being a folder.
    #[inline]
    pub fn is_body(&self) -> bool {
        self.body.is_some()
    }

    /// This node's field, or `None` when it is a folder.
    #[inline]
    pub fn volume(&self) -> Option<&Volume> {
        self.body.as_ref().map(|data| &data.volume)
    }

    #[inline]
    pub fn volume_mut(&mut self) -> Option<&mut Volume> {
        self.body.as_mut().map(|data| &mut data.volume)
    }

    /// A world box that contains this body's bricks, or `None` for a folder row
    /// and for a body holding nothing.
    ///
    /// Cached rather than measured, and a superset rather than a fit: see
    /// [`BodyCache::bounds`]. This is what [`Document::pick`] tests a ray
    /// against before it will march a body, which is what keeps a hover over a
    /// document of bodies from costing one sphere trace each.
    #[inline]
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        self.body.as_ref().and_then(|data| data.cache.bounds)
    }

    /// Everything about this row except its field.
    pub fn meta(&self) -> NodeMeta {
        NodeMeta {
            id: self.id,
            depth: self.depth,
            name: self.name.clone(),
            visible: self.visible,
            collapsed: self.collapsed,
        }
    }

    /// Put a snapshot back, field for field.
    ///
    /// Exhaustive on purpose: every field of [`NodeMeta`] is written, so a
    /// field added to one and not the other is a compile error rather than a
    /// value that silently stops being undoable. The id is not written -- it is
    /// what says the snapshot belongs to this row -- so it is checked instead.
    pub(crate) fn set_meta(&mut self, meta: &NodeMeta) {
        debug_assert_eq!(self.id, meta.id, "a node's snapshot belongs to another node");
        let NodeMeta { id: _, depth, ref name, visible, collapsed } = *meta;
        self.depth = depth;
        self.name.clone_from(name);
        self.visible = visible;
        self.collapsed = collapsed;
    }
}

/// Every body in the sculpt, in display order, on one lattice.
///
/// There is always at least one body. Deleting the last one is refused, which
/// removes `Option<NodeId>` from every signature downstream for the price of
/// one guard in one place.
pub struct Document {
    /// The lattice. One number for the whole document, because bodies share
    /// the lattice; a per-body voxel size is not representable and that is the
    /// point.
    voxel_size: f32,
    /// Preorder, which is display order, which is file order.
    nodes: Vec<Node>,
    /// Always names a node that HOLDS a volume.
    active: NodeId,
    next_id: u32,
}

impl Document {
    /// What the first body is called before anything renames it.
    ///
    /// A name rather than an empty string because it is what the panel shows
    /// and what a version 1 or 2 file -- which carries no node table at all --
    /// is read back as.
    pub const FIRST_BODY_NAME: &'static str = "Body 1";

    /// A document holding one empty body at the given world space voxel size.
    pub fn new(voxel_size: f32) -> Self {
        Self::from_volume(Volume::new(voxel_size))
    }

    /// A document holding one body, which is `volume`.
    ///
    /// The lattice comes from the volume rather than from a parameter, because
    /// there is no arrangement in which those two could differ and the document
    /// still be honest. Every whole-model replacement -- open, import, reset --
    /// goes through here.
    pub fn from_volume(volume: Volume) -> Self {
        let id = NodeId(1);
        let doc = Self {
            voxel_size: volume.voxel_size(),
            nodes: vec![Node::body(id, Self::FIRST_BODY_NAME.to_string(), volume)],
            active: id,
            next_id: 2,
        };
        doc.assert_invariants();
        doc
    }

    /// A document rebuilt from a file's node table, in the file's own order.
    ///
    /// `active` is an INDEX rather than a [`NodeId`], because that is what the
    /// file carries and what the reader has already repaired to name a body
    /// row. It is clamped here as well rather than trusted, so that no
    /// arrangement of bytes can leave `active` naming a row that is not there.
    ///
    /// **`next_id` is `max(id) + 1` and nothing else will do.** Nothing in the
    /// file carries it, and ids never reuse, so a saved document routinely has
    /// a sparse set: delete bodies 2, 3 and 4 from a five-body document and the
    /// file holds [1, 5]. Deriving the next id from `rows.len()` instead would
    /// mint 3, then 4, then 5 -- and 5 is already here. The document would then
    /// hold two nodes under one id, so one body's mesh slots would overwrite
    /// the other's and an undo entry would route to the wrong body; and the
    /// very next save would write a file the reader refuses for duplicate ids.
    /// **The build would have written a file it will not read.**
    ///
    /// Every row is a body, because container version 3 refuses a `kind` byte
    /// that is not zero. Folders widen this to `Vec<(NodeMeta, Option<Volume>)>`
    /// in the increment that makes a folder row representable at all.
    pub(crate) fn from_table(
        voxel_size: f32,
        rows: Vec<(NodeMeta, Volume)>,
        active: usize,
    ) -> Self {
        let nodes: Vec<Node> = rows
            .into_iter()
            .map(|(meta, volume)| {
                debug_assert_eq!(
                    volume.voxel_size(),
                    voxel_size,
                    "every body shares the document's lattice"
                );
                Node::from_meta(meta, volume)
            })
            .collect();
        let highest = nodes.iter().map(|node| node.id.0).max().expect(
            "the reader refuses a table with no rows before it gets here, and every other \
             caller builds its own",
        );
        let active = nodes[active.min(nodes.len() - 1)].id;
        // `highest + 1` cannot overflow: the reader refuses an id of
        // `u32::MAX` for exactly this reason, so the highest id that can reach
        // here is `u32::MAX - 1`.
        let doc = Self { voxel_size, nodes, active, next_id: highest + 1 };
        doc.assert_invariants();
        doc
    }

    /// The one voxel size in the document, in millimetres.
    ///
    /// **This replaced a copy of the same number held by the application.** A
    /// duplicate was harmless while there was one `Volume` to disagree with;
    /// with a document lattice there is exactly one correct value and it lives
    /// here.
    #[inline]
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// The rows, in display order.
    #[inline]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// How many rows there are, bodies and folders together.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// How many rows hold a field.
    #[inline]
    pub fn body_count(&self) -> usize {
        self.nodes.iter().filter(|node| node.is_body()).count()
    }

    /// Which body edits land on.
    #[inline]
    pub fn active(&self) -> NodeId {
        self.active
    }

    /// Choose the body edits land on.
    ///
    /// **Selecting a body changes no geometry**, which is the whole reason the
    /// application must not run its Dynamic-radius rescale from here: see
    /// `Brokkr::rescale_radius`.
    pub fn set_active(&mut self, id: NodeId) {
        debug_assert!(
            self.volume(id).is_some(),
            "the active node must hold a volume, and {id:?} does not"
        );
        self.active = id;
        self.assert_invariants();
    }

    /// One body's field, or `None` when no such body exists.
    ///
    /// A scan of at most [`MAX_NODES`] `u32` comparisons rather than a map
    /// lookup, so that nothing on the meshing path has to allocate an index to
    /// use it.
    pub fn volume(&self, id: NodeId) -> Option<&Volume> {
        self.nodes.iter().find(|node| node.id == id).and_then(Node::volume)
    }

    pub fn volume_mut(&mut self, id: NodeId) -> Option<&mut Volume> {
        self.nodes.iter_mut().find(|node| node.id == id).and_then(Node::volume_mut)
    }

    /// The active body's field.
    ///
    /// Named `active_volume` rather than `volume` deliberately: after this type
    /// existed, "the volume" stopped being a well-defined thing, and a short
    /// name would let a call site go on meaning the old one by accident.
    pub fn active_volume(&self) -> &Volume {
        self.volume(self.active).expect("the active node always holds a volume")
    }

    pub fn active_volume_mut(&mut self) -> &mut Volume {
        let active = self.active;
        self.volume_mut(active).expect("the active node always holds a volume")
    }

    /// Swap the active body's field for another one on the SAME lattice.
    ///
    /// For an operation that rewrites a whole field in place and keeps the row
    /// it belongs to. The lattice check is a `debug_assert!` rather than a
    /// refusal because every caller derives the replacement from the volume it
    /// is replacing.
    ///
    /// **Re-orienting used to be its one caller and is not any more**, because
    /// a quarter turn turns every body rather than the active one; see
    /// [`Document::rotate`]. What is left is the application's own test
    /// fixtures, which is honest about how much of this is load-bearing today —
    /// it is the shape merge and split will want, and it is not on any path a
    /// user can reach right now.
    pub fn replace_active_volume(&mut self, volume: Volume) {
        debug_assert_eq!(
            volume.voxel_size(),
            self.voxel_size,
            "a body may not be swapped for one on a different lattice"
        );
        let active = self.active;
        let data = self
            .nodes
            .iter_mut()
            .find(|node| node.id == active)
            .and_then(|node| node.body.as_mut())
            .expect("the active node always holds a volume");
        data.volume = volume;
        // In full rather than grown: the incoming field is a different one, and
        // a box grown from the outgoing field's would be a box around geometry
        // that is no longer there.
        data.recompute_bounds();
    }

    /// Add a body at the end of the list and return its id.
    ///
    /// Nothing in the interface calls this yet -- the application holds exactly
    /// one body until primitives land. It exists because every test from here
    /// on that has anything to say about two bodies needs to build two, and
    /// because the id policy (monotonic, never reused, never zero) is worth
    /// having one implementation of.
    pub fn add_body(&mut self, name: impl Into<String>, volume: Volume) -> NodeId {
        debug_assert_eq!(
            volume.voxel_size(),
            self.voxel_size,
            "every body shares the document's lattice"
        );
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.push(Node::body(id, name.into(), volume));
        self.assert_invariants();
        id
    }

    /// Where a node sits in display order, or `None` when it is not here.
    ///
    /// The one translation between a [`NodeId`] and a position, which is what
    /// every vector indexed by node position -- today the visibility mask
    /// [`crate::undo::History::undo`] is handed -- has to go through.
    pub fn index_of(&self, id: NodeId) -> Option<usize> {
        self.nodes.iter().position(|node| node.id == id)
    }

    /// One row by id.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Everything about one row except its field, for recording in an undo
    /// entry.
    pub fn meta(&self, id: NodeId) -> Option<NodeMeta> {
        self.node(id).map(Node::meta)
    }

    /// Put a row's snapshot back.
    ///
    /// The whole of rename, the eye and collapse in one call, so that undoing
    /// them is one change rather than three that can be applied by halves.
    pub fn set_meta(&mut self, meta: &NodeMeta) {
        let node = self
            .nodes
            .iter_mut()
            .find(|node| node.id == meta.id)
            .expect("a snapshot names a node that is in the document");
        node.set_meta(meta);
        self.assert_invariants();
    }

    /// Put a node back at a position, and mark every brick it brought with it.
    ///
    /// **The dirty marking is the whole reason this is not `nodes.insert`.**
    /// [`Volume::drain_dirty`] drains, so a body that has been sitting in an
    /// undo entry comes back with an empty dirty set: every headless assertion
    /// passes, the document is exactly right, and the viewport shows nothing at
    /// all, permanently, because nothing ever asks for those bricks to be
    /// meshed again. This project has shipped that class of bug twice, and both
    /// times it was invisible to the test suite.
    ///
    /// `at` may be the end of the list, as [`Vec::insert`] allows, because the
    /// node that was removed from the end has to go back there.
    pub fn insert(&mut self, at: usize, mut node: Node) {
        debug_assert!(at <= self.nodes.len(), "position {at} is past the end of the document");
        debug_assert!(
            node.id.0 != 0 && node.id.0 < self.next_id,
            "{:?} was never handed out by this document",
            node.id
        );
        if let Some(volume) = node.volume_mut() {
            debug_assert_eq!(
                volume.voxel_size(),
                self.voxel_size,
                "every body shares the document's lattice"
            );
            volume.mark_everything_dirty();
        }
        self.nodes.insert(at, node);
        self.assert_invariants();
    }

    /// Take a node out of the document and hand it to the caller.
    ///
    /// Moves rather than clones: the volume goes straight into the undo entry,
    /// so a delete allocates nothing and peak memory does not rise -- it merely
    /// does not fall.
    ///
    /// **The bricks that were on screen are the caller's problem.** They are
    /// not in any dirty set once the node is gone, so whoever removes a body
    /// has to have collected its brick coordinates first and published them
    /// against this id, which is what releases their slots in the renderer's
    /// pool. Meshing a pair that names a body which is no longer here produces
    /// an empty mesh for exactly that reason; see [`Document::mesh_dirty`].
    ///
    /// The active selection moves to the nearest surviving body when the row
    /// removed is the active one, because the invariant that `active` always
    /// holds a volume is what keeps `Option<NodeId>` out of every signature
    /// downstream. That means undoing an operation which added the body you
    /// have selected leaves the selection somewhere else; restoring it is the
    /// business of whatever built the entry, not of the document.
    pub fn remove(&mut self, at: usize) -> Node {
        debug_assert!(at < self.nodes.len(), "there is no node at {at} to remove");
        debug_assert!(
            !self.nodes[at].is_body() || self.body_count() > 1,
            "the last body cannot be removed"
        );
        let node = self.nodes.remove(at);
        if self.active == node.id {
            self.active = self
                .nearest_body(at)
                .expect("a document always holds at least one body to fall back to");
        }
        self.assert_invariants();
        node
    }

    /// The body nearest to a position, looking forwards first and then back.
    ///
    /// Forwards first because a row taken out of the list is replaced on screen
    /// by the one below it, and the selection following the eye is less
    /// surprising than the selection jumping upwards.
    fn nearest_body(&self, from: usize) -> Option<NodeId> {
        let below = self.nodes[from.min(self.nodes.len())..].iter();
        let above = self.nodes[..from.min(self.nodes.len())].iter().rev();
        below.chain(above).find(|node| node.is_body()).map(|node| node.id)
    }

    /// Every body, in display order, with its id.
    pub fn bodies(&self) -> impl Iterator<Item = (NodeId, &Volume)> {
        self.nodes.iter().filter_map(|node| node.volume().map(|volume| (node.id, volume)))
    }

    /// Every body, in display order, with its id, for writing.
    pub fn bodies_mut(&mut self) -> impl Iterator<Item = (NodeId, &mut Volume)> {
        self.nodes.iter_mut().filter_map(|node| {
            let id = node.id;
            node.volume_mut().map(|volume| (id, volume))
        })
    }

    /// What the whole document costs, summed over every body.
    ///
    /// Recomputed rather than cached. It walks each body's brick map, which is
    /// exactly what the single-volume code it replaced did once per remesh, and
    /// a cache with one writer and one reader would buy nothing but a way for
    /// the two to disagree.
    pub fn totals(&self) -> VolumeStats {
        let mut totals = VolumeStats::default();
        for (_, volume) in self.bodies() {
            let stats = volume.stats();
            totals.dense_bricks += stats.dense_bricks;
            totals.uniform_bricks += stats.uniform_bricks;
            totals.resident_bytes += stats.resident_bytes;
        }
        totals
    }

    /// A world box containing every body's bricks, or `None` when the document
    /// holds no geometry at all.
    ///
    /// **Over every body, visible or not**, and the two callers both need it
    /// that way: a primitive placed clear of only the bodies that are drawn is a
    /// primitive inside a hidden one, which is the same invisible-on-the-first-
    /// press failure that placing off-origin exists to prevent, one reveal
    /// later.
    ///
    /// Read from the per-body cache rather than measured, so it costs one min
    /// and one max per row and may safely be asked at a user action. That cache
    /// is a superset that only ever grows (see `BodyCache::bounds`), so this box
    /// is a superset too -- which for placing something clear of it is the safe
    /// direction to be wrong in.
    pub fn world_bounds(&self) -> Option<(Vec3, Vec3)> {
        self.nodes.iter().filter_map(Node::bounds).reduce(|(low, high), (other_low, other_high)| {
            (low.min(other_low), high.max(other_high))
        })
    }

    /// How big the BIGGEST SINGLE body is: the largest half-diagonal of any one
    /// row's box, or `None` when the document holds no geometry.
    ///
    /// **Deliberately not half the diagonal of [`Document::world_bounds`], and
    /// the difference is a bug this shipped with.** The union's diagonal spans
    /// the *gaps between* bodies as well as the bodies, so it grows every time
    /// anything is added away from the origin. Sizing a new primitive off it
    /// made each one larger than the last: measured through
    /// [`crate::primitive::placement`] on a 30 mm ball at a 0.25 mm voxel, eight
    /// presses of `+` gave cubes 37, 48, 66, 93, 128, 181, 249 and 342 mm
    /// across -- a factor of about 1.38 per press, unbounded, with the sixth one
    /// alone allocating some 460 MB of bricks. Measured per body it is a fixed
    /// point instead: a primitive at a third of the biggest body's radius is
    /// smaller than that body, so the maximum does not move.
    ///
    /// Read from the per-body cache, like [`Document::world_bounds`], so it
    /// costs one length per row and may be asked at a user action. A superset,
    /// for the same reason and in the same safe direction.
    pub fn largest_body_radius(&self) -> Option<f32> {
        self.nodes
            .iter()
            .filter_map(Node::bounds)
            .map(|(low, high)| (high - low).length() * 0.5)
            .reduce(f32::max)
    }

    /// What this document will and will not make room for, right now.
    ///
    /// The one place an add path asks "does another body fit"; see
    /// [`GrowthGuard`] for what it then answers with, and why nothing may work
    /// this out for itself.
    ///
    /// **`pool_headroom` is `vertex_capacity - vertices_watermark` and never
    /// `- vertices_reserved`.** The two differ by however much the allocator's
    /// free lists hold in granule classes nothing is asking for, and what runs
    /// the pool out of room is the bump pointer rather than the live count.
    /// The resample guard in the application may use `reserved`, and its own
    /// comment says that is honest *only because* a resample empties the pool
    /// first. **Adding a body empties nothing**, so the same substitution here
    /// would say a body fits and then watch the pool overflow -- which is the
    /// failure `PoolStats` is documented against and the one this project has
    /// shipped twice.
    ///
    /// Taken as a parameter rather than read, because the number lives in the
    /// renderer and `brokkr-core` may not depend on a GPU crate. A `u64`
    /// because that is what `PoolStats` carries.
    pub fn growth_guard(&self, pool_headroom: u64) -> GrowthGuard {
        GrowthGuard {
            resident_bytes: self.totals().resident_bytes as f64,
            pool_headroom: pool_headroom as f64,
        }
    }

    /// Where two bodies claim the same voxels, and how many.
    ///
    /// One entry per pair that overlaps at all, in display order, with the
    /// count of voxels **both** bodies read as solid. A pair that does not
    /// overlap is absent rather than reported as zero, so an empty result means
    /// "nothing interpenetrates" and the length is the number of collisions.
    ///
    /// **Nothing else in this codebase can see an interpenetration.**
    /// [`crate::export::ExportMesh::validate`] counts edge incidence over
    /// shared vertex *indices*, and two bodies welded separately share no
    /// indices at all -- so two spheres passing through each other are two
    /// closed surfaces, and both report watertight, and the slicer resolves the
    /// union without complaint or reports it as a self-intersection depending
    /// on which slicer it is. This is the only measurement that says so.
    ///
    /// Gated on the bodies' world AABBs, which is what makes the common case
    /// -- bodies laid out side by side -- a handful of float comparisons. The
    /// per-body cache the panel will keep is not here yet, so the boxes are
    /// computed once up front rather than per pair: [`Volume::world_bounds`]
    /// walks the brick keys, which is cheap, but doing it `n^2` times is not.
    ///
    /// Costs a walk of every shared brick's voxels for the pairs that do
    /// overlap, so it is a user-action operation and must not run per frame.
    pub fn overlaps(&self) -> Vec<(NodeId, NodeId, usize)> {
        let bodies: Vec<(NodeId, &Volume, (glam::Vec3, glam::Vec3))> = self
            .bodies()
            .filter_map(|(id, volume)| volume.world_bounds().map(|box_| (id, volume, box_)))
            .collect();

        let mut found = Vec::new();
        for (index, (one, here, one_box)) in bodies.iter().enumerate() {
            for (other, there, other_box) in &bodies[index + 1..] {
                if !boxes_meet(*one_box, *other_box) {
                    continue;
                }
                let shared = here
                    .brick_coords()
                    .filter(|coord| there.brick(*coord).is_some())
                    .collect::<Vec<_>>();
                let voxels: usize =
                    shared.par_iter().map(|coord| solid_in_both(here, there, *coord)).sum();
                if voxels > 0 {
                    found.push((*one, *other, voxels));
                }
            }
        }
        found
    }

    /// The nearest surface a ray meets across every body that is DRAWN, and
    /// which body that is.
    ///
    /// `visible` is indexed by node position and comes from
    /// [`Document::display_visibility`]. A body the user cannot see cannot be
    /// picked: hiding is a draw-time skip, so the depth buffer where that body
    /// sits is empty, and a press that carved it would set `unsaved`, push a
    /// history entry, pay a remesh and an upload, and change not one pixel.
    ///
    /// # The box gate is not an optimisation that can be left out
    ///
    /// A raycast that MISSES costs the whole march -- measured at 0.025 ms --
    /// so 64 unguarded bodies is 1.6 ms of a 16 ms frame on every pointer move,
    /// for a gesture that is mostly misses. A ray-slab test against a cached
    /// box is about 20 ns. That ratio is why [`BodyCache`] exists at all.
    ///
    /// # The gate also buys a reach the march does not have
    ///
    /// [`crate::raycast`] advances by at most [`NARROW_BAND`] voxels a step,
    /// because that is where the field saturates, so its total travel is
    /// bounded by `MAX_STEPS * NARROW_BAND * voxel_size` -- 46 mm at a 0.03 mm
    /// voxel, against a camera that frames a model from about three times its
    /// radius. At the finest lattice the ray ran out of steps in the empty
    /// space in front of the model and the cursor quietly stopped working.
    /// Starting the march where the ray ENTERS the body's box spends none of
    /// those steps crossing that emptiness.
    pub fn pick(
        &self,
        origin: Vec3,
        direction: Vec3,
        far: f32,
        visible: &[bool],
    ) -> Option<(NodeId, Hit)> {
        debug_assert_eq!(
            visible.len(),
            self.nodes.len(),
            "the visibility mask is indexed by node position"
        );
        let mut best: Option<(NodeId, Hit)> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            if !visible.get(index).copied().unwrap_or(false) {
                continue;
            }
            // Never further than the nearest hit so far. A body behind one
            // already found cannot win, and the march is the expensive part.
            let reach = best.map_or(far, |(_, hit)| hit.distance);
            if let Some(hit) = self.march(node, origin, direction, reach) {
                best = Some((node.id, hit));
            }
        }
        best
    }

    /// The surface one named body's ray meets, gated and advanced exactly as
    /// [`Document::pick`] gates and advances.
    ///
    /// This is what a gesture already committed to a body wants -- a live
    /// stroke keeps carving the body it started on, whatever the cursor passes
    /// over on the way. It shares [`Document::march`] with the picker so that
    /// the two cannot disagree about where the surface is, which they would the
    /// moment one of them had the box advance and the other did not.
    pub fn pick_body(&self, id: NodeId, origin: Vec3, direction: Vec3, far: f32) -> Option<Hit> {
        self.march(self.node(id)?, origin, direction, far)
    }

    /// One body: the box test, the advance, and the march.
    fn march(&self, node: &Node, origin: Vec3, direction: Vec3, far: f32) -> Option<Hit> {
        let volume = node.volume()?;
        let entry = ray_meets_box(origin, direction, node.bounds()?, far)?;
        // Backed off a narrow band, because that is the width of the only part
        // of the field that carries a distance: a surface sitting against the
        // box face has to be bracketed from outside it to be found at all.
        let start = (entry - NARROW_BAND * volume.voxel_size()).max(0.0);
        let hit = raycast(volume, origin + direction * start, direction, far - start)?;
        Some(Hit { distance: hit.distance + start, ..hit })
    }

    /// What is DRAWN: the pick gate, the panel's muted names, and every
    /// direct-manipulation gesture (today: the plane cut). Indexed by NODE
    /// position.    ///
    /// Solo belongs here and nowhere else in the two named call sites, because
    /// direct manipulation acts on what the user can see. See
    /// [`resolve_visibility`].
    pub fn display_visibility(&self, solo: Option<NodeId>, out: &mut Vec<bool>) {
        resolve_visibility(&self.nodes, solo, out);
    }

    /// What is KEPT: the file, and the export. Indexed by NODE position.
    ///
    /// It is [`resolve_visibility`] with no solo, and the `None` is the whole
    /// of the difference. **Export must never see solo** -- a view mode
    /// silently dropping a part from a print is exactly the class of failure
    /// the eye is being careful about, and it is why these two are named
    /// functions rather than one function with a parameter everybody has to
    /// remember the right value for.
    pub fn saved_visibility(&self, out: &mut Vec<bool>) {
        resolve_visibility(&self.nodes, None, out);
    }

    /// Move every body's dirty set into `out`, tagged with the body it came
    /// from, keeping both allocations.
    ///
    /// The tag is what lets one remesh cover the whole document; see
    /// [`Document::mesh_dirty`].
    ///
    /// **This is also where the pick gate's box is kept honest**, and it is
    /// here rather than in a dozen call sites because a write that does not
    /// mark a brick dirty is already a bug -- it leaves the screen stale -- so
    /// the dirty set is the one signal that cannot miss a change to the brick
    /// map. Each drained coordinate is taken into the body's box, which costs
    /// one min and one max per brick already being meshed.
    pub fn take_dirty(&mut self, out: &mut Vec<(NodeId, BrickCoord)>) {
        out.clear();
        for node in &mut self.nodes {
            let id = node.id;
            let Some(data) = node.body.as_mut() else {
                continue;
            };
            let BodyData { volume, cache } = &mut **data;
            let voxel_size = volume.voxel_size();
            volume.drain_dirty(|coord| {
                out.push((id, coord));
                cache.take_in(coord, voxel_size);
            });
        }
    }

    /// Mark every brick of every body as needing a remesh.
    ///
    /// A load-time operation, proportional to the whole document. Never per
    /// frame.
    pub fn mark_everything_dirty(&mut self) {
        for (_, volume) in self.bodies_mut() {
            volume.mark_everything_dirty();
        }
    }

    /// Mesh one batch of dirty bricks drawn from anywhere in the document.
    ///
    /// **The batching is the point, and it is why this is not a loop over
    /// bodies at the call site.** [`Volume::mesh_bricks`] decides between the
    /// serial and the parallel path on the number of coordinates *in one call*
    /// ([`PARALLEL_MESH_THRESHOLD`]). Eight bodies with three dirty bricks each,
    /// meshed body by body, is eight calls that each fall under the threshold:
    /// twenty-four bricks all on one core on a twenty-four thread machine, with
    /// nothing anywhere reporting it. Taking `(body, coord)` pairs makes that
    /// decision once, over the real total.
    ///
    /// **Every pair is dispatched through that body's [`Volume::mesh_brick`],
    /// which gathers its own apron.** This is the first new entry point into
    /// the meshing path since the apron rule was written and therefore the
    /// first place the rule is at risk: a brick meshed against an apron
    /// gathered from the wrong body -- or against no apron at all -- produces a
    /// seam that looks like a meshing bug and is not one. `ApronBuffer`'s
    /// contents stay `pub(crate)` so that there is no way to write this
    /// function wrongly from outside the crate.
    ///
    /// A pair naming a body that is no longer here meshes to nothing, which is
    /// what releases its slot in the renderer's pool rather than leaving the
    /// triangles on screen.
    pub fn mesh_dirty(&self, work: &[(NodeId, BrickCoord)], out: &mut [BrickMesh]) {
        assert_eq!(work.len(), out.len(), "one output mesh per dirty brick");

        if work.len() < PARALLEL_MESH_THRESHOLD {
            let mut scratch = MeshScratch::new();
            for ((body, coord), mesh) in work.iter().zip(out.iter_mut()) {
                self.mesh_one(*body, *coord, &mut scratch, mesh);
            }
            return;
        }

        out.par_iter_mut().zip(work.par_iter()).for_each_init(
            MeshScratch::new,
            |scratch, (mesh, (body, coord))| {
                self.mesh_one(*body, *coord, scratch, mesh);
            },
        );
    }

    /// One brick of one body, through the only path from voxels to triangles.
    fn mesh_one(
        &self,
        body: NodeId,
        coord: BrickCoord,
        scratch: &mut MeshScratch,
        out: &mut BrickMesh,
    ) {
        match self.volume(body) {
            Some(volume) => volume.mesh_brick(coord, scratch, out),
            None => out.clear(),
        }
    }

    /// Scale the whole document in world space, without touching a voxel.
    ///
    /// Free and lossless, exactly as [`Volume::rescale`] is, and applied to
    /// **every** body because the lattice is shared: scaling one body alone
    /// would hand it a lattice its siblings do not have.
    pub fn rescale(&mut self, factor: f32) {
        for data in self.bodies_data_mut() {
            data.volume.rescale(factor);
            // Every brick sits somewhere else in the world now, at the same
            // coordinate, so the box has to be remeasured rather than grown.
            data.recompute_bounds();
        }
        self.voxel_size *= factor;
        debug_assert!(self.lattice_agrees(), "rescale left the bodies on different lattices");
    }

    /// Rebuild every body at a different voxel size.
    ///
    /// The outgoing bricks are marked dirty in the *incoming* volume, because
    /// after a resample their coordinates mean something else: without that the
    /// renderer keeps drawing slices nothing will ever overwrite.
    pub fn resample(&mut self, voxel_size: f32) {
        for data in self.bodies_data_mut() {
            let mut resampled = data.volume.resampled(voxel_size);
            for coord in data.volume.brick_coords() {
                resampled.mark_dirty(coord);
            }
            data.volume = resampled;
            data.recompute_bounds();
        }
        self.voxel_size = voxel_size;
        debug_assert!(self.lattice_agrees(), "resample left the bodies on different lattices");
    }

    /// Turn **every** body by a multiple of a quarter turn, about the lattice
    /// origin.
    ///
    /// Exact, and the same operation [`Volume::rotated`] performs, applied
    /// across the document for the same reason [`Document::rescale`] is: the
    /// lattice is shared.
    ///
    /// **Turning only the active body would destroy the relative placement of
    /// the others**, which under a one-lattice design IS the bodies' only
    /// positional state -- there is no per-body transform to compensate with.
    /// It would also make the status line's promise that turning it back the
    /// same way undoes it a lie, because the un-turned bodies would come back
    /// somewhere new.
    ///
    /// Costs a permuted copy of every dense brick, so it is a user action.
    /// Undo history is not this function's business and does not survive it;
    /// see [`crate::rotate`] for why a quarter turn is its own undo.
    pub fn rotate(&mut self, rotation: crate::orientation::AxisRotation) {
        for data in self.bodies_data_mut() {
            data.volume = data.volume.rotated(rotation);
            data.recompute_bounds();
        }
    }

    /// Every body's payload, for the operations that rewrite a whole field.
    ///
    /// Private, and it hands out the cache as well as the volume, which is what
    /// makes "rewrote the field and left a box around where it used to be"
    /// impossible to do by halves from inside this module.
    fn bodies_data_mut(&mut self) -> impl Iterator<Item = &mut BodyData> {
        self.nodes.iter_mut().filter_map(|node| node.body.as_deref_mut())
    }

    /// Whether every body really is on the document's lattice.
    fn lattice_agrees(&self) -> bool {
        self.bodies().all(|(_, volume)| volume.voxel_size() == self.voxel_size)
    }

    /// Everything that is true of a document in this build, in one place.
    ///
    /// Three of these are permanent -- ids unique and nonzero, at least one
    /// body, the active node holds a volume. Two are temporary and say what the
    /// application is pinned to rather than what the type can express: every
    /// node is at depth 0, and every node holds a body. **The document type
    /// holds N nodes from its first commit; it is the application that holds
    /// one**, which is why relaxing those two is an edit here and nowhere else.
    ///
    /// Debug only, because it is O(nodes squared) in the uniqueness check and
    /// it runs after every mutation.
    fn assert_invariants(&self) {
        debug_assert!(!self.nodes.is_empty(), "a document always holds at least one node");
        debug_assert!(
            self.nodes.len() <= MAX_NODES,
            "a document holds at most {MAX_NODES} nodes, not {}",
            self.nodes.len()
        );
        debug_assert!(
            self.body_count() >= 1,
            "a document always holds at least one BODY, not just folder rows"
        );
        debug_assert!(
            self.body_count() <= MAX_BODIES,
            "a document holds at most {MAX_BODIES} bodies, not {}",
            self.body_count()
        );
        debug_assert!(self.volume(self.active).is_some(), "the active node must hold a volume");
        for (index, node) in self.nodes.iter().enumerate() {
            debug_assert!(node.id.0 != 0, "id zero is reserved for \"no node\"");
            debug_assert!(
                node.id.0 < self.next_id,
                "{:?} was never handed out by this document",
                node.id
            );
            debug_assert!(
                self.nodes.iter().skip(index + 1).all(|other| other.id != node.id),
                "two nodes share {:?}",
                node.id
            );
            debug_assert!(node.depth < MAX_DEPTH, "depth {} is past the panel's cap", node.depth);
            // Pinned rather than permanent: folders relax both of these.
            debug_assert_eq!(node.depth, 0, "every node is at depth 0 in this build");
            debug_assert!(node.is_body(), "every node holds a body in this build");
        }
    }
}

/// Where a ray enters a world space box, or `None` when it never does inside
/// `far`.
///
/// Zero for a ray that starts inside, which is what a caller wants: the march
/// begins where it already is.
///
/// **Written out per axis rather than as the branchless reciprocal form, and
/// the reason is a failure this had before it had a test.** The usual trick
/// multiplies by `1 / direction` and leans on `f32::min` discarding the NaN
/// that `0 * infinity` produces. That works only while the other operand is
/// finite: a ray running exactly along a box face gives NaN against `+infinity`
/// on the same axis, `min` keeps the infinity, and the box is reported as
/// missed by a ray that grazes it. Three explicit branches cost nothing
/// measurable beside the sphere trace they are protecting and cannot be wrong
/// that way.
fn ray_meets_box(
    origin: Vec3,
    direction: Vec3,
    (low, high): (Vec3, Vec3),
    far: f32,
) -> Option<f32> {
    let mut enters = 0.0_f32;
    let mut leaves = far;
    for axis in 0..3 {
        if direction[axis] == 0.0 {
            // Parallel to this pair of faces: the ray is either between them
            // for its whole length or it never meets the box at all.
            if origin[axis] < low[axis] || origin[axis] > high[axis] {
                return None;
            }
            continue;
        }
        let to_low = (low[axis] - origin[axis]) / direction[axis];
        let to_high = (high[axis] - origin[axis]) / direction[axis];
        enters = enters.max(to_low.min(to_high));
        leaves = leaves.min(to_low.max(to_high));
    }
    (enters <= leaves).then_some(enters)
}

/// Whether two world space boxes share any volume at all.
///
/// Touching counts as meeting: the boxes come from brick extents, and two
/// bodies whose bricks abut share the lattice plane between them, where a voxel
/// of one is a voxel of the other.
fn boxes_meet(
    (one_low, one_high): (glam::Vec3, glam::Vec3),
    (low, high): (glam::Vec3, glam::Vec3),
) -> bool {
    one_low.x <= high.x
        && low.x <= one_high.x
        && one_low.y <= high.y
        && low.y <= one_high.y
        && one_low.z <= high.z
        && low.z <= one_high.z
}

/// Voxels of one brick coordinate that BOTH bodies read as solid.
///
/// Solid is `< 0.0` and not `<= 0.0`, which is the mesher's own rule: an exact
/// zero is biased to the inside as `-0.0` by the voxeliser, and `-0.0 < 0.0` is
/// false in Rust, so `fast-surface-nets` reads such a voxel as OUTSIDE.
/// Counting it as an overlap here would report an interpenetration along every
/// surface two bodies merely touch along.
fn solid_in_both(here: &Volume, there: &Volume, coord: BrickCoord) -> usize {
    match (here.brick(coord), there.brick(coord)) {
        // A uniform tile is one value for all 32,768 of its voxels, so the two
        // cheap cases are worth having: neither needs to be read at all.
        (Some(Brick::Uniform(one)), Some(Brick::Uniform(other))) => {
            usize::from(*one < 0.0 && *other < 0.0) * BRICK_VOXELS
        }
        (Some(Brick::Uniform(one)), Some(Brick::Dense(data)))
        | (Some(Brick::Dense(data)), Some(Brick::Uniform(one)))
            if *one < 0.0 =>
        {
            data.iter().filter(|value| **value < 0.0).count()
        }
        (Some(Brick::Dense(one)), Some(Brick::Dense(other))) => one
            .iter()
            .zip(other.iter())
            .filter(|(here, there)| **here < 0.0 && **there < 0.0)
            .count(),
        // An absent brick reads as OUTSIDE, so it can overlap nothing.
        _ => 0,
    }
}

/// The ONE place the three inputs to visibility are combined: a node's own eye,
/// every ancestor folder's eye, and solo. `out` is indexed by NODE position.
///
/// **All three are masks over a node's own bit and none of them writes it.**
/// That is taken verbatim from two independent references that converge on the
/// same wording -- Maxon on SubTool folders, "toggling the visibility state of
/// the folder will not change that of the individual SubTools", and Photoshop,
/// which draws the suppressed child with a grey eye. Both compose answers fall
/// out of it: a hidden child in a shown folder stays hidden, and a shown child
/// in a hidden folder is restored exactly on re-show, because nothing was
/// mutated. Nomad ships the opposite and users filed it as unintuitive.
///
/// **Solo is a PARAMETER and never a field** -- not of [`Document`], not of
/// `ProjectState`, and above all not of `View`. `project::write(out, doc,
/// state)` takes exactly two data parameters and solo is a field of neither, so
/// there is no expression that can pass solo to the writer. That is not
/// discipline; the call does not typecheck. `View` is written to the file *and*
/// is the payload of every timeline keyframe, so solo there would make jumping
/// to a key change which bodies exist on screen.
///
/// Solo NARROWS and never widens, for the same reason the other two masks do:
/// soloing a body whose own eye is off leaves it hidden here. Making that
/// gesture show the body is the business of whatever handles the click -- it
/// sets the eye -- and not of this function, which has no business rewriting a
/// bit the user set.
///
/// One forward pass. Preorder guarantees every ancestor is already resolved,
/// and the ancestor chain fits a fixed-size array, so this allocates nothing
/// beyond `out`'s own growth and recurses nowhere -- a hostile file cannot make
/// it recurse AT ALL. The depth index is clamped for the same reason
/// [`Node::from_meta`] clamps: an index past [`MAX_DEPTH`] would be a panic in
/// a function every frame calls.
///
/// **The signature is final from here, before folders and before solo exist**,
/// with every caller passing `None` until they do. Three inputs are a decided
/// requirement and the alternative is revisiting every call site later. With
/// one node at depth 0 both the ancestor walk and the solo scope are no-ops,
/// which is exactly why this is worth fixing once rather than in six places.
pub fn resolve_visibility(nodes: &[Node], solo: Option<NodeId>, out: &mut Vec<bool>) {
    out.clear();
    out.reserve(nodes.len());

    // Whether the node at each depth resolved as shown, so its children can
    // read their own ancestor's answer out of the slot above them.
    const DEPTHS: usize = MAX_DEPTH as usize;
    let mut ancestors = [true; DEPTHS];
    // The depth of the soloed node, for as long as we are inside its subtree.
    // A subtree is a preorder run of everything deeper than its root, which is
    // what makes the scope test one integer comparison rather than a search.
    let mut soloed_at: Option<u8> = None;

    for node in nodes {
        let depth = usize::from(node.depth).min(DEPTHS - 1);
        let ancestors_shown = depth == 0 || ancestors[depth - 1];
        ancestors[depth] = ancestors_shown && node.visible;

        if let Some(wanted) = solo {
            soloed_at = match soloed_at {
                Some(root) if node.depth > root => Some(root),
                _ if node.id == wanted => Some(node.depth),
                _ => None,
            };
        }
        let in_scope = solo.is_none() || soloed_at.is_some();

        out.push(ancestors_shown && node.visible && in_scope);
    }
}

/// The ceilings a growing document is measured against, and the one place the
/// arithmetic that predicts a refusal lives.
///
/// **This exists because there was no RAM or pool ceiling on the add-a-body
/// path at all.** The application's resample guard consulted both, and it was
/// the only thing that did: it returned early unless the request was *finer*
/// than the current lattice, so nothing that GREW the document ever reached
/// either ceiling. A second body was simply admitted, and the first thing to
/// notice would have been the pool logging `MESH POOL FULL` to stderr while the
/// model on screen quietly lost parts of itself. This project has shipped
/// silent geometry loss twice and the whole point of this type is that it does
/// not happen a third time.
///
/// Build one with [`Document::growth_guard`], which is what makes the byte
/// figure a sum over every body rather than the active one's.
#[derive(Debug, Clone, Copy)]
pub struct GrowthGuard {
    /// Voxel bytes the whole document already holds.
    resident_bytes: f64,
    /// Vertices the mesh pool can still hand out: capacity minus WATERMARK.
    pool_headroom: f64,
}

impl GrowthGuard {
    /// How much under an exact fit a suggested size lands.
    ///
    /// Three percent, matching the resample guard's own margin and for the same
    /// reason: the estimate is an estimate, and a prediction that lands at
    /// exactly 100% of a ceiling helps nobody.
    const MARGIN: f64 = 0.97;

    /// Why a body costing `bytes` of voxel data and `vertices` of mesh will not
    /// fit, and the fraction of its linear size that would.
    ///
    /// `None` is the answer that means "go ahead". The `f32` in the refusal is
    /// a **linear** scale factor, because both costs are a shell over a
    /// surface: a body at half the size has a quarter of the surface, a quarter
    /// of the bricks and a quarter of the vertices. That is the same square law
    /// the resample guard runs against voxel size, transposed onto the only
    /// lever an add path has. A caller may offer it, shrink to it, or simply
    /// print the message -- but it must not have to work the number out for
    /// itself, because the established pattern in this codebase is that a
    /// refusal names the size that WOULD work.
    ///
    /// **Vertices and not indices.** The index buffer is provisioned at six
    /// times the vertex buffer and a closed surface produces about six -- the
    /// dragon reserves 51.9 million indices against 8.65 million vertices -- so
    /// the vertex count is the binding one. [`crate::voxelise`]'s import
    /// preflight makes the same call for the same reason, and a second
    /// parameter every caller has to estimate would buy a ceiling nothing
    /// reaches first.
    pub fn no_room_for(&self, bytes: f64, vertices: f64) -> Option<(String, f32)> {
        let byte_headroom = (MAX_VOLUME_BYTES - self.resident_bytes).max(0.0);
        let byte_fit = if bytes > 0.0 { byte_headroom / bytes } else { f64::INFINITY };
        let vertex_fit = if vertices > 0.0 { self.pool_headroom / vertices } else { f64::INFINITY };
        let fit = byte_fit.min(vertex_fit);
        if fit >= 1.0 {
            return None;
        }

        // Which ceiling is the tighter one decides what the message says,
        // because "it needs 3.4 GB" and "it needs 40M vertices" send a reader
        // to completely different remedies.
        let why = if byte_fit <= vertex_fit {
            format!(
                "it needs about {:.1} GB of memory and the document has {:.1} GB left of a \
                 {:.0} GB ceiling",
                bytes / (1024.0 * 1024.0 * 1024.0),
                byte_headroom / (1024.0 * 1024.0 * 1024.0),
                MAX_VOLUME_BYTES / (1024.0 * 1024.0 * 1024.0),
            )
        } else {
            format!(
                "it needs about {:.1}M vertices and the mesh pool has {:.1}M left",
                vertices / 1.0e6,
                self.pool_headroom / 1.0e6,
            )
        };
        let workable = (fit.sqrt() * Self::MARGIN) as f32;
        Some((format!("{why} -- {:.0}% of that size would fit", workable * 100.0), workable))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// **The measurement the whole design rests on**, and nothing tested it
    /// before this type existed, because nothing had ever held two `Volume`s at
    /// once.
    ///
    /// If brick `c` does not cover exactly the same world box in two bodies at
    /// the same voxel size, then a boolean between them is a resample rather
    /// than a `min`, the plane cut means a different plane per body, and a
    /// brick-keyed mesh pool draws one body's geometry at another's position.
    /// Bit-identical rather than approximately equal: the coordinates are
    /// integers scaled by one shared `f32`, so there is no rounding to be
    /// tolerant of, and a tolerance here would hide exactly the drift it exists
    /// to catch.
    ///
    /// Both sides go through [`Volume::voxel_position`] rather than doing the
    /// `coord * voxel_size` arithmetic inline. Inline was the first version and
    /// it was worthless: with the multiply written out here, the only thing
    /// either body contributed was `voxel_size()`, both fixtures were built at
    /// 0.25, and the assertion reduced to `X * 0.25 == X * 0.25`. Deleting the
    /// second volume from it entirely left all ten tests in this module green,
    /// which was measured rather than argued. The whole point is to fail the
    /// day a body gets its own anchor or transform, and a per-body world
    /// mapping has to land in `voxel_position` -- that is the one function
    /// that turns this volume's voxel into a world point -- so calling it is
    /// what puts the property under test.
    #[test]
    fn bricks_of_two_volumes_at_one_voxel_size_cover_the_same_world_box() {
        let mut here = Volume::new(0.25);
        let mut there = Volume::new(0.25);
        // Different centres, so the two brick maps overlap without matching:
        // the shared coordinates are the ones this is about.
        here.seed_sphere(Vec3::ZERO, 8.0);
        there.seed_sphere(Vec3::new(6.0, -3.0, 2.0), 8.0);

        let shared: Vec<BrickCoord> =
            here.brick_coords().filter(|coord| there.brick(*coord).is_some()).collect();
        assert!(!shared.is_empty(), "the fixture must actually overlap");

        for coord in shared {
            let low = here.voxel_position(coord.origin());
            let high = here.voxel_position(coord.max_voxel());
            let other_low = there.voxel_position(coord.origin());
            let other_high = there.voxel_position(coord.max_voxel());
            assert_eq!(low, other_low, "brick {coord:?} starts somewhere else in the other body");
            assert_eq!(high, other_high, "brick {coord:?} ends somewhere else in the other body");
        }
    }

    #[test]
    fn a_fresh_document_holds_one_body_that_is_active() {
        let doc = Document::new(0.25);
        assert_eq!(doc.node_count(), 1);
        assert_eq!(doc.body_count(), 1);
        assert_eq!(doc.nodes()[0].name, Document::FIRST_BODY_NAME);
        assert_eq!(doc.active(), doc.nodes()[0].id);
        assert_eq!(doc.voxel_size(), 0.25);
    }

    #[test]
    fn added_bodies_get_fresh_ids_and_leave_the_active_one_alone() {
        let mut doc = Document::new(0.5);
        let first = doc.active();
        let second = doc.add_body("Body 2", Volume::new(0.5));
        let third = doc.add_body("Body 3", Volume::new(0.5));
        assert_ne!(second, first);
        assert_ne!(third, second);
        assert_eq!(doc.active(), first, "adding a body must not steal the selection");
        assert_eq!(doc.body_count(), 3);

        doc.set_active(third);
        assert_eq!(doc.active(), third);
    }

    /// The batching claim in [`Document::mesh_dirty`] is only worth anything if
    /// each brick is meshed against its OWN body. Two bodies whose bricks
    /// collide at the origin is the case that tells a correct implementation
    /// from one that looks up the active body every time.
    #[test]
    fn meshing_a_batch_uses_each_bricks_own_body() {
        let mut doc = Document::new(0.5);
        let a = doc.active();
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 10.0);
        let mut second = Volume::new(0.5);
        second.seed_sphere(Vec3::ZERO, 4.0);
        let b = doc.add_body("Body 2", second);

        let mut work = Vec::new();
        doc.take_dirty(&mut work);
        assert!(work.iter().any(|(id, _)| *id == a), "body A had dirty bricks");
        assert!(work.iter().any(|(id, _)| *id == b), "body B had dirty bricks");

        let mut batched = vec![BrickMesh::default(); work.len()];
        doc.mesh_dirty(&work, &mut batched);

        // The same bricks meshed one body at a time, which is what the batch
        // has to agree with.
        let mut scratch = MeshScratch::new();
        for ((body, coord), batch) in work.iter().zip(batched.iter()) {
            let mut alone = BrickMesh::default();
            doc.volume(*body).expect("a live body").mesh_brick(*coord, &mut scratch, &mut alone);
            assert_eq!(alone.vertices, batch.vertices, "{body:?} {coord:?} meshed differently");
            assert_eq!(alone.indices, batch.indices, "{body:?} {coord:?} meshed differently");
        }
        assert!(
            batched.iter().any(|mesh| !mesh.is_empty()),
            "the fixture must produce some triangles or it asserts nothing"
        );
    }

    /// The serial and the parallel path have to agree, and the threshold is the
    /// only thing that chooses between them.
    #[test]
    fn the_serial_and_parallel_meshing_paths_agree() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 10.0);
        let mut work = Vec::new();
        doc.take_dirty(&mut work);
        assert!(work.len() > PARALLEL_MESH_THRESHOLD, "the fixture must reach the parallel path");

        let mut parallel = vec![BrickMesh::default(); work.len()];
        doc.mesh_dirty(&work, &mut parallel);

        let mut serial = vec![BrickMesh::default(); work.len()];
        for index in 0..work.len() {
            // One pair per call, which is under the threshold by construction.
            doc.mesh_dirty(&work[index..index + 1], &mut serial[index..index + 1]);
        }

        for (index, (one, other)) in serial.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(one.vertices, other.vertices, "brick {index} differs between the paths");
            assert_eq!(one.indices, other.indices, "brick {index} differs between the paths");
        }
    }

    /// A pair naming a body that has gone must mesh to nothing rather than
    /// panic or, worse, mesh the active body's brick under a dead body's key --
    /// which would leave the departed geometry drawn for the rest of the
    /// session.
    #[test]
    fn a_brick_of_a_body_that_is_gone_meshes_to_nothing() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 10.0);
        let mut work = Vec::new();
        doc.take_dirty(&mut work);
        let ghost: Vec<(NodeId, BrickCoord)> =
            work.iter().map(|(_, coord)| (NodeId(9999), *coord)).collect();

        let mut out = vec![BrickMesh::default(); ghost.len()];
        doc.mesh_dirty(&ghost, &mut out);
        assert!(out.iter().all(BrickMesh::is_empty), "a dead body's bricks produced triangles");
    }

    #[test]
    fn taking_the_dirty_set_empties_every_body() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 6.0);
        let mut second = Volume::new(0.5);
        second.seed_sphere(Vec3::new(30.0, 0.0, 0.0), 6.0);
        doc.add_body("Body 2", second);

        let mut work = Vec::new();
        doc.take_dirty(&mut work);
        assert!(!work.is_empty());

        let mut again = Vec::new();
        doc.take_dirty(&mut again);
        assert!(again.is_empty(), "the dirty set was not drained");
    }

    #[test]
    fn rescaling_moves_every_body_onto_the_new_lattice() {
        let mut doc = Document::new(0.5);
        doc.add_body("Body 2", Volume::new(0.5));
        doc.rescale(2.0);
        assert_eq!(doc.voxel_size(), 1.0);
        for (_, volume) in doc.bodies() {
            assert_eq!(volume.voxel_size(), 1.0, "a body was left on the old lattice");
        }
    }

    #[test]
    fn resampling_moves_every_body_onto_the_new_lattice() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 6.0);
        doc.add_body("Body 2", Volume::new(0.5));
        doc.resample(0.25);
        assert_eq!(doc.voxel_size(), 0.25);
        for (_, volume) in doc.bodies() {
            assert_eq!(volume.voxel_size(), 0.25, "a body was left on the old lattice");
        }
    }

    #[test]
    fn the_totals_are_the_sum_over_every_body() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 8.0);
        let mut second = Volume::new(0.5);
        second.seed_sphere(Vec3::new(40.0, 0.0, 0.0), 8.0);
        let one = doc.active_volume().stats();
        let other = second.stats();
        doc.add_body("Body 2", second);

        let totals = doc.totals();
        assert_eq!(totals.dense_bricks, one.dense_bricks + other.dense_bricks);
        assert_eq!(totals.uniform_bricks, one.uniform_bricks + other.uniform_bricks);
        assert_eq!(totals.resident_bytes, one.resident_bytes + other.resident_bytes);
    }

    /// Removing the SELECTED row has to leave the selection on a live body.
    ///
    /// Every other test in the workspace that removes a node removes one it
    /// had not selected, so until this existed the reassignment in
    /// [`Document::remove`] was never reached: changing its guard to `if false
    /// && ...` left the whole suite green, which was measured rather than
    /// assumed. The gesture that reaches it -- deleting the body you have
    /// selected, and undoing the primitive you just added -- is the first
    /// thing either does.
    ///
    /// The cost of getting it wrong is not a wrong selection. `active` holding
    /// a volume is the invariant that keeps `Option<NodeId>` out of every
    /// signature downstream, so a debug build trips
    /// [`Document::assert_invariants`] and a release build, where that assert
    /// is compiled out, reaches [`Document::active_volume`]'s `expect` on the
    /// next frame instead. A hard crash in the shipped binary.
    ///
    /// Forwards first, which is what `remove`'s doc comment promises: the row
    /// that took the deleted one's place on screen is the one the selection
    /// follows.
    #[test]
    fn removing_the_selected_body_moves_the_selection_to_the_row_below_it() {
        let mut doc = Document::new(0.5);
        let second = doc.add_body("Body 2", Volume::new(0.5));
        let third = doc.add_body("Body 3", Volume::new(0.5));

        doc.set_active(second);
        let at = doc.index_of(second).expect("the body just selected is in the document");
        doc.remove(at);

        assert_eq!(doc.active(), third, "the selection should follow the row that took its place");
        assert_eq!(doc.nodes()[at].id, third, "and that row is the one now at {at}");
        assert!(doc.volume(doc.active()).is_some(), "the active row must still hold a field");
        assert_eq!(doc.body_count(), 2);
    }

    /// The same, for the row at the END of the list -- the only case where
    /// looking forwards finds nothing.
    ///
    /// Split out from the test above because it is the branch where
    /// `nearest_body`'s `from.min(self.nodes.len())` clamp and its `.rev()`
    /// fallback are the only code doing any work: after the `Vec::remove` the
    /// position handed in is one past the end, so an unclamped slice would
    /// panic and a forwards-only search would find nothing to fall back to.
    #[test]
    fn removing_the_selected_body_from_the_end_moves_the_selection_upwards() {
        let mut doc = Document::new(0.5);
        let second = doc.add_body("Body 2", Volume::new(0.5));
        let third = doc.add_body("Body 3", Volume::new(0.5));

        doc.set_active(third);
        let at = doc.index_of(third).expect("the body just selected is in the document");
        assert_eq!(at, doc.node_count() - 1, "the fixture must remove the last row");
        doc.remove(at);

        assert_eq!(doc.active(), second, "there is nothing below, so the selection moves up");
        assert!(doc.volume(doc.active()).is_some(), "the active row must still hold a field");
        assert_eq!(doc.body_count(), 2);
    }

    // --- the visibility resolver ------------------------------------------

    /// A row at a chosen depth with a chosen eye, built straight rather than
    /// through a [`Document`].
    ///
    /// The document pins every node to depth 0 in this build and asserts it
    /// after every mutation, so a tree deep enough to exercise the ancestor
    /// walk cannot be built through one. [`resolve_visibility`] is a free
    /// function over `&[Node]` precisely so it can be tested ahead of the
    /// feature that produces such a list.
    fn row(id: u32, depth: u8, visible: bool) -> Node {
        Node::from_meta(
            NodeMeta {
                id: NodeId(id),
                depth,
                name: format!("row {id}"),
                visible,
                collapsed: false,
            },
            Volume::new(0.5),
        )
    }

    #[test]
    fn a_hidden_folder_hides_its_children_without_writing_their_eyes() {
        // folder(hidden) > child(shown) > grandchild(shown), then a sibling of
        // the folder that must be untouched by any of it.
        let nodes = vec![row(1, 0, false), row(2, 1, true), row(3, 2, true), row(4, 0, true)];

        let mut shown = Vec::new();
        resolve_visibility(&nodes, None, &mut shown);
        assert_eq!(shown, vec![false, false, false, true]);

        // The eyes themselves are untouched, which is the whole composition
        // rule: re-showing the folder restores the descendants exactly, because
        // nothing was ever written to them.
        assert!(nodes[1].visible, "the ancestor's eye was written into the child's");
        assert!(nodes[2].visible);
    }

    /// The other half of the same rule: a hidden child inside a SHOWN folder
    /// stays hidden. An ancestor's eye is an AND-mask and never an override.
    #[test]
    fn a_shown_folder_does_not_reveal_a_child_that_is_hidden() {
        let nodes = vec![row(1, 0, true), row(2, 1, false), row(3, 2, true)];
        let mut shown = Vec::new();
        resolve_visibility(&nodes, None, &mut shown);
        assert_eq!(
            shown,
            vec![true, false, false],
            "a shown folder revealed a hidden child, and its grandchild with it"
        );
    }

    /// Solo scopes to the soloed node's SUBTREE, which is the preorder run of
    /// everything deeper than it, and stops at the first row that is not.
    #[test]
    fn solo_shows_the_soloed_node_and_its_subtree_and_nothing_else() {
        let nodes = vec![
            row(1, 0, true), // before
            row(2, 0, true), // soloed folder
            row(3, 1, true), // inside
            row(4, 2, true), // inside, deeper
            row(5, 0, true), // after, back at the soloed depth
        ];

        let mut shown = Vec::new();
        resolve_visibility(&nodes, Some(NodeId(2)), &mut shown);
        assert_eq!(shown, vec![false, true, true, true, false]);

        // And with no solo the same list is all shown, so the test above is
        // measuring solo rather than something else.
        resolve_visibility(&nodes, None, &mut shown);
        assert_eq!(shown, vec![true; 5]);
    }

    /// **Solo narrows and never widens.** Soloing a body whose own eye is off
    /// leaves it hidden here, because solo is a mask over that bit like the
    /// other two and not a rewrite of it.
    ///
    /// The gesture "solo a hidden body and see it" is a decided requirement,
    /// and it is met by the click handler turning the eye on -- which is
    /// undoable and visible in the panel -- rather than by this function
    /// quietly disagreeing with the eye the user is looking at.
    #[test]
    fn soloing_a_hidden_node_does_not_reveal_it() {
        let nodes = vec![row(1, 0, true), row(2, 0, false)];
        let mut shown = Vec::new();
        resolve_visibility(&nodes, Some(NodeId(2)), &mut shown);
        assert_eq!(shown, vec![false, false], "solo overrode an eye instead of masking it");
    }

    /// Soloing something that is not in the list hides everything rather than
    /// showing everything. A stale id is a bug either way, and the failure that
    /// is visible immediately is better than the one that looks like solo did
    /// not fire.
    #[test]
    fn soloing_an_id_that_is_not_here_shows_nothing() {
        let nodes = vec![row(1, 0, true), row(2, 0, true)];
        let mut shown = Vec::new();
        resolve_visibility(&nodes, Some(NodeId(99)), &mut shown);
        assert_eq!(shown, vec![false, false]);
    }

    /// Every combination of the three inputs, at the full depth the panel
    /// allows, against the rule stated independently of the implementation.
    ///
    /// The tests above are the cases worth reading; this is the one that says
    /// there is no *other* case. The tree is a straight chain of
    /// [`MAX_DEPTH`] rows, which is the shape that exercises the ancestor walk
    /// to its last slot. Every assignment of eyes down that chain is tried (256
    /// of them), against no solo and against solo on each row in turn -- 2,304
    /// resolutions, which is what "every combination" costs at this depth.
    ///
    /// The expectation is recomputed from the rule -- "shown if every eye from
    /// the root down to and including its own is on, AND it is inside the solo
    /// scope" -- rather than read from a table, so this cannot be made to agree
    /// with a wrong implementation by pasting its output into a fixture.
    #[test]
    fn the_resolver_matches_the_rule_at_every_depth_for_every_eye_and_every_solo() {
        const DEPTH: usize = MAX_DEPTH as usize;
        assert_eq!(DEPTH, 8, "the ancestor array is fixed size, so its length is part of this");

        let mut shown = Vec::new();
        for eyes in 0_u32..(1 << DEPTH) {
            // Row n is at depth n, so row n's ancestors are rows 0..n.
            let nodes: Vec<Node> =
                (0..DEPTH).map(|d| row(d as u32 + 1, d as u8, eyes >> d & 1 == 1)).collect();

            let solos = std::iter::once(None).chain((0..DEPTH).map(|d| Some(NodeId(d as u32 + 1))));
            for solo in solos {
                resolve_visibility(&nodes, solo, &mut shown);
                assert_eq!(shown.len(), DEPTH);

                for (index, &resolved) in shown.iter().enumerate() {
                    let eyes_on = (0..=index).all(|above| eyes >> above & 1 == 1);
                    // In a chain, one row's subtree is that row and everything
                    // after it, because everything after it is deeper.
                    let in_scope = solo.is_none_or(|wanted| index + 1 >= wanted.0 as usize);
                    assert_eq!(
                        resolved,
                        eyes_on && in_scope,
                        "row {index} of the chain with eyes {eyes:08b} and solo {solo:?}"
                    );
                }
            }
        }
    }

    /// The buffer is reused, so it has to be emptied rather than appended to.
    #[test]
    fn resolving_twice_into_one_buffer_does_not_accumulate() {
        let doc = Document::new(0.5);
        let mut shown = vec![true; 17];
        doc.saved_visibility(&mut shown);
        assert_eq!(shown.len(), doc.node_count());
    }

    /// The two named call sites differ in exactly one thing, and that one thing
    /// is what keeps a view mode out of a print.
    #[test]
    fn the_saved_visibility_ignores_solo_and_the_displayed_one_does_not() {
        let mut doc = Document::new(0.5);
        let second = doc.add_body("Body 2", Volume::new(0.5));

        let mut displayed = Vec::new();
        doc.display_visibility(Some(second), &mut displayed);
        assert_eq!(displayed, vec![false, true], "solo did not narrow what is drawn");

        let mut saved = Vec::new();
        doc.saved_visibility(&mut saved);
        assert_eq!(saved, vec![true, true], "solo reached the file and the export");
    }

    // --- interpenetration --------------------------------------------------

    /// Two bodies that pass through each other, and two that do not.
    ///
    /// This is the only measurement in the codebase that can tell them apart:
    /// each body welds separately, so `MeshReport::validate` counts edges over
    /// disjoint index spaces and both report watertight however deeply they
    /// interpenetrate.
    #[test]
    fn overlapping_bodies_are_counted_and_disjoint_ones_are_not() {
        let mut doc = Document::new(0.5);
        let first = doc.active();
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 8.0);

        let mut through = Volume::new(0.5);
        through.seed_sphere(Vec3::new(4.0, 0.0, 0.0), 8.0);
        let second = doc.add_body("Body 2", through);

        let found = doc.overlaps();
        assert_eq!(found.len(), 1, "expected exactly one colliding pair: {found:?}");
        assert_eq!((found[0].0, found[0].1), (first, second));
        assert!(found[0].2 > 0, "the pair was reported with no overlapping voxels");

        // Both still export as closed surfaces, which is what makes the count
        // above the only thing that would ever say so.
        for (_, volume) in doc.bodies() {
            let (_, report) = volume.export_mesh();
            assert!(report.is_printable(), "the fixture should be two clean solids");
        }
    }

    #[test]
    fn bodies_that_do_not_touch_report_no_overlap_at_all() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 8.0);
        let mut apart = Volume::new(0.5);
        apart.seed_sphere(Vec3::new(80.0, 0.0, 0.0), 8.0);
        doc.add_body("Body 2", apart);

        assert!(doc.overlaps().is_empty(), "two bodies 80 mm apart were reported as colliding");
    }

    /// Bodies whose bricks meet but whose material does not are NOT an overlap.
    ///
    /// The AABB gate would pass them -- brick extents round out to whole 32
    /// voxel bricks, so two spheres several millimetres apart share bricks --
    /// and reporting them would make the count useless on any laid-out
    /// document. The voxel comparison is what earns it.
    #[test]
    fn bodies_whose_bricks_meet_but_whose_material_does_not_are_not_an_overlap() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 8.0);
        let mut beside = Volume::new(0.5);
        // Just clear of the first sphere, and well inside the brick rounding.
        beside.seed_sphere(Vec3::new(18.0, 0.0, 0.0), 8.0);
        doc.add_body("Body 2", beside);

        let one = doc.active_volume().world_bounds().expect("the first body has bricks");
        let other = doc
            .bodies()
            .nth(1)
            .and_then(|(_, volume)| volume.world_bounds())
            .expect("the second body has bricks");
        assert!(
            boxes_meet(one, other),
            "the fixture must share bricks or it is not testing the voxel comparison"
        );
        assert!(doc.overlaps().is_empty(), "touching bricks were reported as touching material");
    }

    /// **The size of the biggest body is not the size of the document**, and
    /// the difference is what [`Document::largest_body_radius`] exists for.
    ///
    /// Two small bodies laid out a long way apart have a union box far larger
    /// than either of them. Sizing anything off that box makes it grow every
    /// time something is added beside the model, which is exactly what
    /// [`crate::primitive::placement`] did until this method replaced the
    /// document-wide radius it was using.
    #[test]
    fn the_largest_body_radius_measures_a_body_and_not_the_gap_between_two() {
        let mut near = Volume::new(0.5);
        near.seed_sphere(Vec3::ZERO, 8.0);
        // Through `from_volume` and `add_body` rather than `active_volume_mut`,
        // because the per-body box is cached when the body arrives and seeding
        // into a body already in the document does not refresh it.
        let mut doc = Document::from_volume(near);
        let mut far = Volume::new(0.5);
        far.seed_sphere(Vec3::new(400.0, 0.0, 0.0), 8.0);
        doc.add_body("Body 2", far);

        let biggest = doc.largest_body_radius().expect("both bodies have bricks");
        let (low, high) = doc.world_bounds().expect("both bodies have bricks");
        let union = (high - low).length() * 0.5;

        // A 16 mm ball rounds out to whole bricks, so the figure is tens of
        // millimetres rather than eight -- and the point is that it does not
        // move when the second body lands 400 mm away.
        assert!(biggest < 40.0, "one 16 mm ball measured {biggest} mm of radius");
        assert!(union > 200.0, "the fixture must be spread out or it tests nothing");
    }

    // --- the growth guard --------------------------------------------------
    /// A body the document has room for is admitted, and the guard says so by
    /// answering nothing at all.
    #[test]
    fn a_body_that_fits_is_not_refused() {
        let doc = Document::new(0.5);
        let guard = doc.growth_guard(80_000_000);
        assert!(guard.no_room_for(100.0 * 1024.0 * 1024.0, 1_000_000.0).is_none());
    }

    /// The RAM ceiling, which is the one that fires first on a real document.
    ///
    /// The message has to name the ceiling as well as the shortfall, because
    /// "it needs 4 GB" without "there is 1.5 GB left" is not something a user
    /// can act on -- and the fraction has to be one that actually fits, or the
    /// refusal has sent them somewhere that will refuse them again.
    #[test]
    fn a_body_too_big_for_the_memory_ceiling_is_refused_with_a_size_that_fits() {
        let mut doc = Document::new(0.5);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 6.0);
        let held = doc.totals().resident_bytes as f64;
        // Everything but a gigabyte of the ceiling is already spoken for.
        let guard = GrowthGuard {
            resident_bytes: MAX_VOLUME_BYTES - 1024.0 * 1024.0 * 1024.0,
            pool_headroom: 80_000_000.0,
        };
        assert!(held > 0.0, "the fixture body must cost something");

        let wanted = 4.0 * 1024.0 * 1024.0 * 1024.0;
        let (why, workable) =
            guard.no_room_for(wanted, 1_000_000.0).expect("4 GB does not fit in 1 GB");
        assert!(why.contains("GB of memory"), "the memory ceiling should be named: {why}");
        assert!(why.contains("6 GB ceiling"), "the ceiling itself is not in the message: {why}");
        assert!(why.contains('%'), "the message has to name a size that would work: {why}");

        // A shell scales with the square of its linear size, so the suggested
        // fraction has to be applied that way -- and once applied it must
        // itself be admitted, which is the fixpoint the whole refusal rests on.
        let shrunk = wanted * (workable * workable) as f64;
        assert!(
            guard.no_room_for(shrunk, 1_000_000.0).is_none(),
            "the size the guard named does not itself fit: {workable}"
        );
    }

    /// The pool ceiling, and the one place the WATERMARK is the number that
    /// matters.
    ///
    /// `Document::growth_guard` takes headroom rather than capacity precisely
    /// so this cannot be computed from `vertices_reserved` by mistake: adding a
    /// body resets nothing, so the space behind a fragmented bump pointer is
    /// not space at all.
    #[test]
    fn a_body_the_pool_cannot_hold_is_refused_in_vertices() {
        let doc = Document::new(0.5);
        // Plenty of RAM, almost no pool.
        let guard = doc.growth_guard(2_000_000);
        let (why, workable) =
            guard.no_room_for(1024.0 * 1024.0, 8_000_000.0).expect("8M vertices do not fit in 2M");
        assert!(why.contains("vertices"), "the pool ceiling should be named: {why}");
        assert!(workable > 0.0 && workable < 1.0, "a fraction was expected, got {workable}");

        let shrunk = 8_000_000.0 * (workable * workable) as f64;
        assert!(
            guard.no_room_for(1024.0 * 1024.0, shrunk).is_none(),
            "the size the guard named does not itself fit: {workable}"
        );
    }

    /// A document already past the ceiling has no room for anything, and must
    /// still answer with a finite fraction rather than a negative or a NaN.
    #[test]
    fn a_document_already_over_the_ceiling_refuses_without_arithmetic_nonsense() {
        let guard = GrowthGuard { resident_bytes: MAX_VOLUME_BYTES * 2.0, pool_headroom: 0.0 };
        let (_, workable) = guard.no_room_for(1024.0, 1024.0).expect("there is no room at all");
        assert!(workable.is_finite(), "the refusal named {workable} as a size");
        assert_eq!(workable, 0.0, "nothing fits, so no fraction of it fits either");
    }

    // ------------------------------------------------------------- the pick

    /// Two bodies side by side, both visible, at a voxel size a test can
    /// afford.
    fn two_balls() -> (Document, NodeId, NodeId) {
        let mut near = Volume::new(0.5);
        near.seed_sphere(Vec3::new(0.0, 0.0, 40.0), 8.0);
        let mut doc = Document::from_volume(near);
        let first = doc.active();

        let mut far = Volume::new(0.5);
        far.seed_sphere(Vec3::ZERO, 8.0);
        let second = doc.add_body("Body 2", far);
        (doc, first, second)
    }

    fn all_shown(doc: &Document) -> Vec<bool> {
        let mut shown = Vec::new();
        doc.saved_visibility(&mut shown);
        shown
    }

    /// The whole point of picking rather than raycasting one body: with two
    /// bodies on the ray, the one in front is the one the press means.
    #[test]
    fn the_pick_returns_the_nearest_body_the_ray_meets() {
        let (doc, near, far) = two_balls();
        let shown = all_shown(&doc);

        // Down minus Z, so the ball at z = 40 is met before the one at the
        // origin.
        let (body, hit) = doc
            .pick(Vec3::new(0.0, 0.0, 200.0), Vec3::NEG_Z, 1000.0, &shown)
            .expect("the ray runs through both balls");
        assert_eq!(body, near, "the pick chose the body behind the other one");
        assert!(hit.position.z > 30.0, "the hit was on the far ball at {}", hit.position.z);

        // And from the other end, the answer is the other body -- so this is
        // measuring the ray and not the list order.
        let (body, _) = doc
            .pick(Vec3::new(0.0, 0.0, -200.0), Vec3::Z, 1000.0, &shown)
            .expect("the ray runs through both balls");
        assert_eq!(body, far, "approaching from behind picked the wrong body");
    }

    /// Hiding is a draw-time skip, so a hidden body is not on screen -- and a
    /// press that carved one would set `unsaved`, push a history entry, pay a
    /// remesh and change not one pixel.
    #[test]
    fn a_hidden_body_is_never_picked() {
        let (mut doc, near, far) = two_balls();
        let mut meta = doc.meta(near).expect("the near body");
        meta.visible = false;
        doc.set_meta(&meta);
        let shown = all_shown(&doc);

        let (body, _) = doc
            .pick(Vec3::new(0.0, 0.0, 200.0), Vec3::NEG_Z, 1000.0, &shown)
            .expect("the ball behind the hidden one is still there");
        assert_eq!(body, far, "the pick went through a body that is not drawn");

        // With both hidden there is nothing to pick at all, which is what makes
        // a press over a hidden body do nothing rather than fall through to
        // something else.
        let mut meta = doc.meta(far).expect("the far body");
        meta.visible = false;
        doc.set_meta(&meta);
        let shown = all_shown(&doc);
        assert!(doc.pick(Vec3::new(0.0, 0.0, 200.0), Vec3::NEG_Z, 1000.0, &shown).is_none());
    }

    /// A ray that passes the document by finds nothing, and the box gate is
    /// what makes that cheap rather than 64 full marches.
    #[test]
    fn a_ray_that_misses_everything_picks_nothing() {
        let (doc, _, _) = two_balls();
        let shown = all_shown(&doc);
        assert!(
            doc.pick(Vec3::new(500.0, 0.0, 200.0), Vec3::NEG_Z, 1000.0, &shown).is_none(),
            "a ray five hundred millimetres to one side found a surface"
        );
    }

    /// **The reach the march does not have on its own.**
    ///
    /// `raycast` advances by at most `NARROW_BAND` voxels a step, so its total
    /// travel is `MAX_STEPS * NARROW_BAND * voxel_size`: 46 mm at a 0.03 mm
    /// voxel. A camera framing the default model sits about 90 mm out, so at
    /// the finest lattice the ray ran out of steps in the empty space in front
    /// of the model and the cursor stopped working, with nothing anywhere
    /// saying why.
    ///
    /// The fixture is a small ball at a far viewpoint rather than a large one,
    /// because a 30 mm ball at 0.03 mm is some 12,000 dense bricks and over a
    /// gigabyte. The distance is what the failure is about, and it is the same
    /// distance either way.
    ///
    /// Both halves are asserted. Without the raycast line this would pass with
    /// the gate deleted, on a march that simply had far enough to go.
    #[test]
    fn the_box_gate_reaches_a_model_the_march_alone_runs_out_of_steps_before() {
        let voxel_size = 0.03;
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::ZERO, 3.0);
        let doc = Document::from_volume(volume);
        let body = doc.active();

        // 512 steps of 3 voxels is 46.08 mm, and the eye is 90 mm out.
        let eye = Vec3::new(0.0, 0.0, 90.0);
        assert!(
            crate::raycast::raycast(doc.active_volume(), eye, Vec3::NEG_Z, 1000.0).is_none(),
            "the march reached the model unaided, so this fixture no longer measures anything"
        );

        let hit = doc
            .pick_body(body, eye, Vec3::NEG_Z, 1000.0)
            .expect("the gate should start the march at the body's box");
        assert!(
            (hit.position.z - 3.0).abs() < 0.5,
            "the surface is at z = 3, and the pick reported {}",
            hit.position.z
        );
        assert!(
            (hit.distance - 87.0).abs() < 0.5,
            "the distance must be measured from the EYE, not from the box; got {}",
            hit.distance
        );

        // And through the picker, which is the path a hover actually takes.
        let shown = all_shown(&doc);
        let (picked, _) = doc
            .pick(eye, Vec3::NEG_Z, 1000.0, &shown)
            .expect("the same gate, reached the same way the pointer reaches it");
        assert_eq!(picked, body);
    }

    /// A ray running exactly along a box face is where the branchless slab test
    /// used to answer "misses" for a ray that hits, and it is why
    /// [`ray_meets_box`] is written out per axis.
    #[test]
    fn the_box_test_survives_a_ray_lying_in_the_plane_of_a_face() {
        let box_ = (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        // Along +X at exactly y = -1 and z = -1, which is an edge of the box.
        let along = ray_meets_box(Vec3::new(-10.0, -1.0, -1.0), Vec3::X, box_, 100.0);
        assert_eq!(along, Some(9.0), "a ray along an edge was refused");

        assert_eq!(ray_meets_box(Vec3::ZERO, Vec3::X, box_, 100.0), Some(0.0), "inside is zero");
        assert_eq!(
            ray_meets_box(Vec3::new(-10.0, 5.0, 0.0), Vec3::X, box_, 100.0),
            None,
            "a ray passing above the box was admitted"
        );
        assert_eq!(
            ray_meets_box(Vec3::new(-10.0, 0.0, 0.0), Vec3::X, box_, 5.0),
            None,
            "a box past `far` was admitted"
        );
        assert_eq!(
            ray_meets_box(Vec3::new(10.0, 0.0, 0.0), Vec3::X, box_, 100.0),
            None,
            "a box behind the ray was admitted"
        );
    }

    /// The cache is what the gate reads, so a body that has grown since it was
    /// measured must not be gated out of its own new material. Growing is the
    /// only direction this promises; see [`BodyCache::bounds`].
    #[test]
    fn the_cached_box_takes_in_material_a_stroke_adds() {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 8.0);
        let mut doc = Document::from_volume(volume);
        let body = doc.active();
        let mut dirty = Vec::new();
        doc.take_dirty(&mut dirty);

        let before = doc.node(body).expect("the body").bounds().expect("a seeded ball has bricks");

        // A second ball well outside the first, written straight into the same
        // body, which is what a Draw stroke walking outwards amounts to.
        doc.active_volume_mut().seed_sphere(Vec3::new(60.0, 0.0, 0.0), 4.0);
        doc.take_dirty(&mut dirty);

        let after = doc.node(body).expect("the body").bounds().expect("still has bricks");
        assert!(after.1.x > before.1.x + 40.0, "the box did not follow the material: {after:?}");

        let truth = doc.active_volume().world_bounds().expect("bricks");
        assert!(
            after.0.cmple(truth.0).all() && after.1.cmpge(truth.1).all(),
            "the cached box {after:?} does not contain the bricks {truth:?}"
        );

        // And the gate now admits a ray that only meets the new material.
        let shown = all_shown(&doc);
        assert!(
            doc.pick(Vec3::new(60.0, 0.0, 200.0), Vec3::NEG_Z, 1000.0, &shown).is_some(),
            "the new material is inside the box and still was not picked"
        );
    }

    // ----------------------------------------------------------- the turning

    /// **Every body turns, and the arrangement survives it.**
    ///
    /// Bodies share one lattice and have no transform, so the arrangement of
    /// their bricks is the only positional state this document has. Turning the
    /// active body alone would scatter it with nothing to put it back -- and it
    /// would make the interface's promise that turning it back the same way
    /// undoes the turn a lie for every body that did not move.
    ///
    /// Bit-identical rather than approximately equal: a quarter turn maps
    /// voxels onto voxels, so there is nothing to be tolerant of, and a
    /// tolerance would hide exactly the resampling this claims not to do.
    #[test]
    fn two_bodies_survive_four_quarter_turns_with_their_placement_and_their_bits() {
        use crate::brick::BRICK_DIM;
        use crate::orientation::{AxisRotation, Facing};

        let mut here = Volume::new(0.5);
        here.seed_sphere(Vec3::new(0.0, 0.0, 20.0), 6.0);
        let mut doc = Document::from_volume(here);
        let first = doc.active();
        let mut there = Volume::new(0.5);
        there.seed_sphere(Vec3::new(30.0, 0.0, 0.0), 4.0);
        let second = doc.add_body("Body 2", there);

        let snapshot = |doc: &Document, id: NodeId| -> Vec<(BrickCoord, Vec<f32>)> {
            let volume = doc.volume(id).expect("a live body");
            let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
            coords.sort_unstable();
            coords
                .into_iter()
                .map(|coord| {
                    let brick = volume.brick(coord).expect("came from the map");
                    let mut values = Vec::with_capacity(BRICK_VOXELS);
                    for z in 0..BRICK_DIM {
                        for y in 0..BRICK_DIM {
                            for x in 0..BRICK_DIM {
                                values.push(brick.get(x, y, z));
                            }
                        }
                    }
                    (coord, values)
                })
                .collect()
        };
        let before_first = snapshot(&doc, first);
        let before_second = snapshot(&doc, second);
        let apart = |doc: &Document| {
            let one = doc.volume(first).expect("a live body").world_bounds().expect("bricks");
            let other = doc.volume(second).expect("a live body").world_bounds().expect("bricks");
            (one.0 + one.1) * 0.5 - (other.0 + other.1) * 0.5
        };
        let before_apart = apart(&doc);

        // One turn first: the second body must have MOVED, or the four-turn
        // assertion below would pass on a rotation that did nothing to it.
        let quarter = AxisRotation::taking(Facing::Up, Facing::Front);
        doc.rotate(quarter);
        assert_ne!(apart(&doc), before_apart, "the second body did not turn with the first");
        assert!(
            (apart(&doc).length() - before_apart.length()).abs() < 1.0e-3,
            "the bodies changed their distance apart, so the turn was not rigid"
        );

        for _ in 0..3 {
            doc.rotate(quarter);
        }
        assert_eq!(snapshot(&doc, first), before_first, "the first body came back changed");
        assert_eq!(snapshot(&doc, second), before_second, "the second body came back changed");
        assert_eq!(apart(&doc), before_apart, "the arrangement did not come back");
    }
}
