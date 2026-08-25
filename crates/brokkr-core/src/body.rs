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

use rayon::prelude::*;

use crate::brick::BrickCoord;
use crate::mesh::{BrickMesh, MeshScratch};
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
/// hole. It is a struct with one field rather than a bare `Box<Volume>`
/// because the per-body cache -- the four numbers the panel, the guards and the
/// pick gate read -- joins it later. That cache is deliberately **not** here
/// yet: two of its four numbers come from `Volume::surface_bounds`, which scans
/// every dense brick and whose own documentation forbids calling it per frame,
/// so a cache filled on every remesh would be a per-stroke full-model scan. It
/// arrives with the first caller that needs a bound it cannot afford to
/// recompute.
///
/// Derives nothing, because [`Volume`] derives nothing -- not even `Debug`, and
/// deliberately not `Clone`. A body delete MOVES its volume into the undo entry
/// rather than cloning it; a clone is simply not available, and that pushes the
/// right way. Duplicating a body will get an explicitly named
/// `Volume::duplicated`, because `.clone()` is one keystroke and a name is
/// something a reviewer stops on.
struct BodyData {
    volume: Volume,
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
            body: Some(Box::new(BodyData { volume })),
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
    /// For the operations that rewrite a whole field in place and keep the row
    /// it belongs to -- re-orienting is the one today. The lattice check is a
    /// `debug_assert!` rather than a refusal because every caller derives the
    /// replacement from the volume it is replacing.
    pub fn replace_active_volume(&mut self, volume: Volume) {
        debug_assert_eq!(
            volume.voxel_size(),
            self.voxel_size,
            "a body may not be swapped for one on a different lattice"
        );
        *self.active_volume_mut() = volume;
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

    /// Move every body's dirty set into `out`, tagged with the body it came
    /// from, keeping both allocations.
    ///
    /// The tag is what lets one remesh cover the whole document; see
    /// [`Document::mesh_dirty`].
    pub fn take_dirty(&mut self, out: &mut Vec<(NodeId, BrickCoord)>) {
        out.clear();
        for node in &mut self.nodes {
            let id = node.id;
            let Some(volume) = node.volume_mut() else {
                continue;
            };
            volume.drain_dirty(|coord| out.push((id, coord)));
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
        for (_, volume) in self.bodies_mut() {
            volume.rescale(factor);
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
        for (_, volume) in self.bodies_mut() {
            let mut resampled = volume.resampled(voxel_size);
            for coord in volume.brick_coords() {
                resampled.mark_dirty(coord);
            }
            *volume = resampled;
        }
        self.voxel_size = voxel_size;
        debug_assert!(self.lattice_agrees(), "resample left the bodies on different lattices");
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
}
