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
//! # The tree: a flat preorder array with a depth column
//!
//! Position in the tree is `(preorder index, depth)` and nothing else. There is
//! no parent pointer, which is what makes a cycle **unrepresentable** rather
//! than merely detectable: there is nothing in this encoding to point at an
//! ancestor with, so no arrangement of bytes -- and no sequence of edits --
//! can produce one. Validating the whole tree is therefore a fold over one
//! integer, [`Document::assert_invariants`], and neither the reader nor the
//! visibility resolver recurses at all.
//!
//! Do not "improve" this into parent indices with a cycle check. That admits
//! the bad state and then hunts for it, it needs an invented canonical sibling
//! order before the file can be byte-identical twice, and it reintroduces the
//! recursive parser a hostile file could overflow.
//!
//! The one tree primitive anybody needs is [`subtree`]: group, ungroup,
//! move-to-folder and delete-folder are all range moves over it, and every
//! legality question is a range comparison. "You cannot drop a folder into
//! itself" is `!range.contains(&target)`, not a graph search.
//!
//! Two rules taken from the references because both REMOVE state:
//!
//! - **A folder can never be empty.** Removing its last child dissolves it, in
//!   the same undo entry. It is not asserted as an invariant, and that is
//!   deliberate -- undo restores a deleted subtree one row at a time, and the
//!   folder necessarily stands alone for the instant between its own row going
//!   back and its children's. See [`Document::assert_invariants`].
//! - **A collapsed folder NEVER changes what a command does.** In ZBrush,
//!   deleting a subtool inside a closed folder deletes the whole folder; a user
//!   reported losing an unrecoverable hour to it. Collapse changes only what is
//!   drawn.
//!
//! `nodes` is a `Vec` and never a hash map. `project::write` has to produce
//! identical bytes twice -- `writing_the_same_volume_twice_gives_identical_bytes`
//! pins it -- and a hash order breaks that nondeterministically, which is the
//! shape of failure that passes locally and fails on CI. List order is also
//! user-visible state that has to round-trip.

use std::ops::Range;

use glam::Vec3;
use rayon::prelude::*;

use crate::brick::{BRICK_VOXELS, Brick, BrickCoord, NARROW_BAND};
use crate::mesh::{BrickMesh, MeshScratch};
use crate::project::MAX_VOLUME_BYTES;
use crate::raycast::{Hit, raycast};
use crate::undo::Change;
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
    /// Private because [`Document::assert_invariants`] is the one place that
    /// decides what a legal depth is, and because every write to it has to go
    /// through a clamp: [`resolve_visibility`] indexes a fixed
    /// `[bool; MAX_DEPTH]` ancestor chain, so an over-deep row is a panic in a
    /// function every frame calls.
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
    /// A body row at a chosen depth.
    ///
    /// **The depth is a parameter and not a zero, and the increment that made
    /// folders reachable is the increment that had to stop it being a zero.** A
    /// row's position in the tree IS (preorder index, depth), so a constructor
    /// that hard-codes one of the two can only ever build a top-level row --
    /// and the one caller that inserts mid-list, duplicate, then drops a
    /// depth-0 row into the middle of a folder's preorder run, which ends that
    /// folder early and silently evicts everything after the copy.
    ///
    /// Clamped for the same reason [`Node::from_meta`] and [`Node::set_meta`]
    /// clamp: every write to `depth` goes through one, because
    /// [`resolve_visibility`] indexes a fixed `[bool; MAX_DEPTH]` array with
    /// it. Whether the depth is legal *at a position* is a question about the
    /// document rather than the row, and [`Document::insert_body`] answers it.
    fn body(id: NodeId, depth: u8, name: String, volume: Volume) -> Self {
        Self {
            id,
            depth: depth.min(MAX_DEPTH - 1),
            name,
            visible: true,
            collapsed: false,
            body: Some(Box::new(BodyData::new(volume))),
        }
    }

    /// A body row rebuilt from a snapshot, which is what a node table read out
    /// of a file amounts to. `None` for the volume IS the folder.
    ///
    /// **The depth is clamped here rather than trusted, and this is the
    /// clamping constructor [`MAX_DEPTH`] refers to.** `resolve_visibility`
    /// walks a fixed `[bool; MAX_DEPTH]` ancestor chain, and a depth past the
    /// end of it is an index out of bounds -- a panic in the one function every
    /// frame calls. The reader has its own clamp, which is the one that keeps
    /// the *forest* valid; this one only keeps the array in bounds, and it is
    /// here because the reader is not the only thing that builds a document --
    /// the split, the group and every test helper come through this
    /// constructor instead.
    pub(crate) fn from_meta(meta: NodeMeta, volume: Option<Volume>) -> Self {
        Self {
            id: meta.id,
            depth: meta.depth.min(MAX_DEPTH - 1),
            name: meta.name,
            visible: meta.visible,
            // A body row has nothing to collapse, and the reader repairs the
            // bit away rather than refusing it. Doing it here as well is what
            // stops a test helper from building a document whose write ->
            // read -> write differs by one bit.
            collapsed: meta.collapsed && volume.is_none(),
            body: volume.map(|volume| Box::new(BodyData::new(volume))),
        }
    }

    /// A folder row: a name, a depth, and no field at all.
    fn folder(id: NodeId, name: String, depth: u8) -> Self {
        Self {
            id,
            depth: depth.min(MAX_DEPTH - 1),
            name,
            visible: true,
            collapsed: false,
            body: None,
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
        // Clamped for the same reason `from_meta` clamps: this is the other
        // write to `depth`, and a snapshot is only as good as whatever built
        // it. A legal depth passes through untouched, so undo stays exact.
        self.depth = depth.min(MAX_DEPTH - 1);
        self.name.clone_from(name);
        self.visible = visible;
        self.collapsed = collapsed && !self.is_body();
    }

    /// Move this row one level in or out, clamped to the panel's cap.
    ///
    /// `by` is signed because a group deepens and an ungroup shallows, and
    /// writing the two out separately is two places for the clamp to be
    /// forgotten in.
    fn shift_depth(&mut self, by: i16) {
        self.depth = i16::from(self.depth)
            .saturating_add(by)
            .clamp(0, i16::from(MAX_DEPTH) - 1)
            .try_into()
            .expect("clamped into 0 ..= MAX_DEPTH - 1");
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
            nodes: vec![Node::body(id, 0, Self::FIRST_BODY_NAME.to_string(), volume)],
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
    /// Every row that carries a `Some` volume is a body; a `None` IS a folder.
    /// The reader has already clamped the depth column into a valid forest, so
    /// this does not re-derive it -- it only asserts it, through
    /// [`Document::assert_invariants`].
    pub(crate) fn from_table(
        voxel_size: f32,
        rows: Vec<(NodeMeta, Option<Volume>)>,
        active: usize,
    ) -> Self {
        let nodes: Vec<Node> = rows
            .into_iter()
            .map(|(meta, volume)| {
                debug_assert!(
                    volume.as_ref().is_none_or(|volume| volume.voxel_size() == voxel_size),
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
        // Belt and braces over the reader's own repair: a folder cannot be the
        // active row, and the invariant that says so is a `debug_assert`, which
        // a release build would sail straight past into the first `expect` that
        // asks for the active body's field.
        let active = match nodes.iter().find(|node| node.id == active) {
            Some(node) if node.is_body() => active,
            _ => {
                nodes
                    .iter()
                    .find(|node| node.is_body())
                    .expect("the reader refuses a table with no body rows before it gets here")
                    .id
            }
        };
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
    /// The add-a-primitive path and every test that needs a second body. It is
    /// [`Document::insert_body`] at the end of the list, and it stays as its own
    /// name because "put it last" is what almost every caller means and
    /// `nodes().len()` at each of them is a position that can be computed
    /// wrongly.
    pub fn add_body(&mut self, name: impl Into<String>, volume: Volume) -> NodeId {
        self.insert_body(self.nodes.len(), 0, name, volume)
    }

    /// Add a body at a POSITION and a DEPTH in display order and return its id.
    ///
    /// Duplicate is what needs it: the copy goes directly below the row it came
    /// from, where the user is looking, rather than at the bottom of a list of
    /// sixty-four.
    ///
    /// **The depth is a parameter rather than a zero, and a caller that means
    /// "beside this row" passes that row's depth.** A position alone does not
    /// say where in the TREE a row lands -- a row's place is (preorder index,
    /// depth) and nothing less -- so a mid-list insert that assumed depth 0
    /// dropped the copy of a body inside a folder at the top level, ended the
    /// folder's preorder run at the copy, and silently evicted every sibling
    /// after it. That was invisible to the panel's own smoke test because the
    /// active row there happened to be at depth 0.
    ///
    /// The depth is clamped to what is legal AT `at` rather than trusted, which
    /// is the same bargain the reader strikes with a file: no argument can
    /// leave the document a non-forest. A row may be at most one level below
    /// the row above it and only a folder may be a parent at all, which sets
    /// the ceiling; and the row that will now FOLLOW this body may not be
    /// deeper than it, which sets the floor. The floor never exceeds the
    /// ceiling, because the following row already satisfied both against the
    /// row this one is displacing.
    ///
    /// **This mints a new id and [`Document::insert`] does not, and the two must
    /// not be confused.** `insert` puts a node the document has already handed
    /// an id to back where it was, which is what undo does; this one is the
    /// document growing by a row that never existed. The id policy -- monotonic,
    /// never reused, never zero -- has exactly one implementation and it is
    /// here.
    ///
    /// **The volume's dirty set is the caller's business**, as it is for
    /// `add_body`: this does not mark anything, because the two things that
    /// build a field for it -- [`crate::primitive::build`] and
    /// [`Volume::duplicated`] -- both hand back a volume already marked, and a
    /// second full marking here would walk the brick map again for nothing.
    /// `Document::insert` marks precisely because the node it takes has been
    /// sitting in an undo entry with a drained dirty set.
    pub fn insert_body(
        &mut self,
        at: usize,
        depth: u8,
        name: impl Into<String>,
        volume: Volume,
    ) -> NodeId {
        debug_assert!(at <= self.nodes.len(), "position {at} is past the end of the document");
        debug_assert_eq!(
            volume.voxel_size(),
            self.voxel_size,
            "every body shares the document's lattice"
        );
        let id = NodeId(self.next_id);
        self.next_id += 1;
        let depth = self.legal_body_depth_at(at, depth);
        self.nodes.insert(at, Node::body(id, depth, name.into(), volume));
        self.assert_invariants();
        id
    }

    /// The depth nearest to `wanted` that a BODY inserted at `at` may hold.
    ///
    /// The whole of what [`Document::insert_body`]'s doc comment describes as
    /// the ceiling and the floor, written once because getting it wrong is a
    /// document that is no longer a forest -- and a forest is what every fold
    /// over the depth column, the visibility resolver first among them, is
    /// entitled to assume.
    fn legal_body_depth_at(&self, at: usize, wanted: u8) -> u8 {
        let ceiling = match at.checked_sub(1).map(|above| &self.nodes[above]) {
            None => 0,
            Some(above) if above.is_body() => above.depth(),
            Some(above) => above.depth() + 1,
        };
        let floor = self.nodes.get(at).map_or(0, Node::depth);
        wanted.min(ceiling).min(MAX_DEPTH - 1).max(floor)
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

    // --- the tree ----------------------------------------------------------

    /// Everything about every row, in display order.
    ///
    /// The payload of [`Change::Outline`], and the reason that variant can
    /// cover rename, the eye, collapse, reorder, group and ungroup at once: a
    /// whole-outline snapshot is a permutation *plus* field edits over a fixed
    /// id set, and its inverse is the pair swapped.
    ///
    /// A hundred and twenty-eight rows at some eighty bytes each is about
    /// 10 KB, against a 256 MB history budget. That is the whole cost argument,
    /// and it is why there is no narrower variant for "just the eye".
    pub fn outline(&self) -> Vec<NodeMeta> {
        self.nodes.iter().map(Node::meta).collect()
    }

    /// Put a whole outline back: the order, and every row's fields.
    ///
    /// **The id multiset must be exactly the document's own**, which is what
    /// makes this a permutation rather than a structural edit. A group records
    /// its new folder as a separate [`Change::NodeAdded`] precisely so that this
    /// stays true; see the module doc's account of the ordering rule.
    ///
    /// `O(n^2)` in the lookup, with `n` at most [`MAX_NODES`]. This runs at a
    /// user action and at an undo press, never per frame.
    pub(crate) fn set_outline(&mut self, outline: &[NodeMeta]) {
        debug_assert_eq!(
            outline.len(),
            self.nodes.len(),
            "an outline snapshot covers every row or none of them"
        );
        let mut taken: Vec<Option<Node>> = self.nodes.drain(..).map(Some).collect();
        for meta in outline {
            let found =
                taken.iter().position(|node| node.as_ref().is_some_and(|node| node.id == meta.id));
            let Some(index) = found else {
                debug_assert!(false, "an outline snapshot names {:?}, which is not here", meta.id);
                continue;
            };
            let mut node = taken[index].take().expect("found just above");
            node.set_meta(meta);
            self.nodes.push(node);
        }
        debug_assert!(
            taken.iter().all(Option::is_none),
            "an outline snapshot left rows out of the document"
        );
        self.assert_invariants();
    }

    /// The rows one node's subtree occupies, or `None` when it is not here.
    ///
    /// See [`subtree`]. A body's subtree is always exactly itself.
    pub fn subtree_of(&self, id: NodeId) -> Option<Range<usize>> {
        self.index_of(id).map(|at| subtree(&self.nodes, at))
    }

    /// How many rows in one node's subtree hold a field.
    ///
    /// What a delete has to know before it runs: whether it would take the last
    /// body, and how many slots the renderer has to be told to forget.
    pub fn subtree_body_count(&self, id: NodeId) -> usize {
        self.subtree_of(id)
            .map_or(0, |range| self.nodes[range].iter().filter(|node| node.is_body()).count())
    }

    /// The folder one row sits directly inside, or `None` at the top level.
    ///
    /// A backward scan for the first row shallower than this one, which by the
    /// preorder invariant IS the parent. No pointer, no search.
    pub fn parent_of(&self, id: NodeId) -> Option<NodeId> {
        let at = self.index_of(id)?;
        let depth = self.nodes[at].depth();
        if depth == 0 {
            return None;
        }
        self.nodes[..at].iter().rev().find(|node| node.depth() < depth).map(|node| node.id)
    }

    /// Every folder row, in display order.
    pub fn folders(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter().filter(|node| !node.is_body())
    }

    /// How many bodies each row's subtree holds, written into `out` by node
    /// position. Zero for a body row, which counts nothing but itself.
    ///
    /// **One backward pass and a fixed-size accumulator, because the panel asks
    /// for this and the panel is rebuilt at display rate.** Done per row it
    /// would be a subtree walk per row, which is the `O(n^2)` per frame the
    /// panel's own header forbids. Allocates nothing beyond `out`'s growth.
    pub fn subtree_body_counts(&self, out: &mut Vec<usize>) {
        out.clear();
        out.resize(self.nodes.len(), 0);
        // `pending[d]` is the number of bodies in the subtrees at depth `d`
        // seen since the last row shallower than `d`. Preorder read backwards
        // means a row's children are all counted before the row itself.
        let mut pending = [0usize; MAX_DEPTH as usize + 1];
        for (index, node) in self.nodes.iter().enumerate().rev() {
            let depth = usize::from(node.depth()).min(MAX_DEPTH as usize - 1);
            let bodies = if node.is_body() { 1 } else { pending[depth + 1] };
            out[index] = if node.is_body() { 0 } else { bodies };
            // Consumed by this row; nothing deeper can be outstanding, because
            // the fold forbids a jump of more than one level.
            pending[depth + 1] = 0;
            pending[depth] += bodies;
        }
    }

    /// Wrap one row's subtree in a new folder, in place.
    ///
    /// Returns the folder's id and the changes one undo press has to reverse,
    /// or `None` when the document has no room for another row or the subtree
    /// is already as deep as the panel goes.
    ///
    /// # The order of the two changes is load-bearing
    ///
    /// `[NodeAdded, Outline]`, and never the other way round. An entry is
    /// applied in reverse, so undo runs the `Outline` first -- putting the
    /// depths back while the folder is still in the list, which is what keeps
    /// the outline's id multiset equal to the document's -- and only then
    /// removes the folder. Recording the `Outline` first would hand
    /// [`Document::set_outline`] a snapshot one row short of the document.
    pub fn group(&mut self, id: NodeId, name: impl Into<String>) -> Option<(NodeId, Vec<Change>)> {
        let at = self.index_of(id)?;
        if self.nodes.len() >= MAX_NODES {
            return None;
        }
        let range = subtree(&self.nodes, at);
        let deepest = self.nodes[range.clone()].iter().map(Node::depth).max()?;
        if deepest + 1 >= MAX_DEPTH {
            return None;
        }

        let depth = self.nodes[at].depth();
        let folder = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(at, Node::folder(folder, name.into(), depth));

        // Snapshotted with the folder already in the list and the subtree still
        // beside it rather than inside it. That intermediate arrangement is a
        // valid forest -- an empty folder followed by its future children --
        // which is exactly why an empty folder is not an invariant.
        let before = self.outline();
        for node in &mut self.nodes[at + 1..=range.end] {
            node.shift_depth(1);
        }
        let after = self.outline();

        self.assert_invariants();
        Some((
            folder,
            vec![Change::NodeAdded { at, id: folder }, Change::Outline { before, after }],
        ))
    }

    /// Dissolve a folder: its children rise out of it and keep their order.
    ///
    /// `None` when the id names a body or is not here. The mirror of
    /// [`Document::group`], including the ordering rule: `[Outline,
    /// NodeRemoved]`, so undo puts the folder back first and only then
    /// re-deepens the children into it.
    pub fn ungroup(&mut self, folder: NodeId) -> Option<Vec<Change>> {
        let at = self.index_of(folder)?;
        if self.nodes[at].is_body() {
            return None;
        }
        let range = subtree(&self.nodes, at);

        let before = self.outline();
        for node in &mut self.nodes[at + 1..range.end] {
            node.shift_depth(-1);
        }
        let after = self.outline();

        let node = self.nodes.remove(at);
        debug_assert_ne!(self.active, node.id, "a folder was never the active row");
        self.assert_invariants();
        Some(vec![
            Change::Outline { before, after },
            Change::NodeRemoved { at, node: Box::new(node) },
        ])
    }

    /// Move one row's subtree into a folder, or out to the top level.
    ///
    /// A destination named by *what it is* rather than by where the pointer is,
    /// which is the form every caller outside the panel wants: the tests below,
    /// and anything that has a folder id in its hand rather than a gap. It is a
    /// two-line translation into a [`DropTarget`] and then
    /// [`Document::reparent`], and it is written that way ON PURPOSE -- a
    /// second copy of the splice is a second copy of the vacated-slot
    /// arithmetic, which is exactly the line the property test caught a bug in.
    ///
    /// `None` for everything `reparent` returns `None` for, plus a target that
    /// is a body row: a body is never a parent.
    pub fn move_to_folder(&mut self, id: NodeId, into: Option<NodeId>) -> Option<Vec<Change>> {
        let target = match into {
            Some(folder) => {
                let folder_at = self.index_of(folder)?;
                if self.nodes[folder_at].is_body() {
                    return None;
                }
                DropTarget {
                    at: subtree(&self.nodes, folder_at).end,
                    depth: self.nodes[folder_at].depth() + 1,
                }
            }
            // Out to the end of the top level, which is where a row with no
            // parent belongs and the only unambiguous place to put it.
            None => DropTarget { at: self.nodes.len(), depth: 0 },
        };
        self.reparent(id, target)
    }

    /// Splice one row's subtree into a gap in the list, at a chosen depth.
    ///
    /// **The subtree moves as a block**, which is what flat preorder buys: the
    /// rows under a folder are a contiguous run, so re-parenting twelve of them
    /// is one `drain` and one `splice` rather than a walk. Nothing here can
    /// build a cycle, because there is nothing in this encoding to point at an
    /// ancestor with -- the refusal below is about a block landing *inside its
    /// own run*, which would lose rows rather than loop.
    ///
    /// `None` when [`drop_refusal`] refuses the pair, and `None` when the move
    /// would change nothing -- so a drag that puts a row back where it was
    /// costs no undo press and no unsaved flag. Those two are deliberately not
    /// distinguished in the return: the panel has already asked
    /// [`drop_refusal`] itself, to draw the indicator, and a caller that has
    /// not is not entitled to a reason.
    ///
    /// A folder the departure leaves empty is dissolved in the same list of
    /// changes, after the reorder, so one ctrl+Z puts both back.
    pub fn reparent(&mut self, id: NodeId, target: DropTarget) -> Option<Vec<Change>> {
        let at = self.index_of(id)?;
        let range = subtree(&self.nodes, at);
        if drop_refusal(&self.nodes, range.clone(), target).is_some() {
            return None;
        }

        let DropTarget { at: insert_at, depth } = target;
        let by = i16::from(depth) - i16::from(self.nodes[at].depth());
        // Already exactly where it is being sent: the rows would come out in
        // the same order at the same depths.
        if by == 0 && (insert_at == range.start || insert_at == range.end) {
            return None;
        }

        let before = self.outline();
        let moved: Vec<Node> = self.nodes.drain(range.clone()).collect();
        // The drain shifted everything after the subtree down by its length.
        let landed = if insert_at > range.start { insert_at - moved.len() } else { insert_at };
        self.nodes.splice(landed..landed, moved);
        for node in &mut self.nodes[landed..landed + range.len()] {
            node.shift_depth(by);
        }
        let after = self.outline();

        // **Where the subtree LEFT FROM, in the list as it now stands**, which
        // is not `range.start` whenever the block landed in front of it: the
        // splice shifted the vacated slot up by the block's own length. Getting
        // this wrong dissolves nothing and leaves an empty folder behind --
        // caught by the property test rather than by reasoning, on a move of a
        // folder's only child into a folder ABOVE it.
        let vacated = if landed < range.start { range.start + range.len() } else { range.start };
        let mut changes = vec![Change::Outline { before, after }];
        changes.extend(self.dissolve_empty_folders_above(vacated));
        self.assert_invariants();
        Some(changes)
    }

    /// Show or hide a folder's children in the panel.
    ///
    /// Recorded like any other field of the outline, because `collapsed` is
    /// written to the file: a change nobody could undo would still be a change
    /// the next save keeps. `None` for a body row, which has nothing to fold
    /// away, and for a folder already in that state.
    pub fn set_collapsed(&mut self, folder: NodeId, collapsed: bool) -> Option<Vec<Change>> {
        let at = self.index_of(folder)?;
        if self.nodes[at].is_body() || self.nodes[at].collapsed == collapsed {
            return None;
        }
        let before = self.outline();
        self.nodes[at].collapsed = collapsed;
        let after = self.outline();
        self.assert_invariants();
        Some(vec![Change::Outline { before, after }])
    }

    /// Take a row and everything under it out of the document.
    ///
    /// `None` when it would take the last body, which a document may not be
    /// without. Otherwise a `Change::NodeRemoved` per row, **in removal order:
    /// deepest and last first, the subtree's own root last.** An entry is
    /// applied in reverse, so that is precisely what makes undo put the FOLDER
    /// back before the bodies that live in it -- every insertion then lands at
    /// an index that is already correct, and no intermediate state has a body
    /// standing where a folder should be.
    ///
    /// The volumes MOVE into the changes; [`Volume`] has no `Clone` at all, so
    /// a folder delete of three bodies allocates nothing and peak memory does
    /// not rise, it merely does not fall.
    ///
    /// A folder the delete leaves empty is dissolved into the same list.
    pub fn delete_subtree(&mut self, id: NodeId) -> Option<Vec<Change>> {
        let at = self.index_of(id)?;
        let range = subtree(&self.nodes, at);
        let going = self.nodes[range.clone()].iter().filter(|node| node.is_body()).count();
        if going >= self.body_count() {
            return None;
        }

        let mut changes = Vec::with_capacity(range.len());
        for index in range.clone().rev() {
            let node = self.nodes.remove(index);
            changes.push(Change::NodeRemoved { at: index, node: Box::new(node) });
        }
        changes.extend(self.dissolve_empty_folders_above(range.start));

        if !self.nodes.iter().any(|node| node.id == self.active) {
            self.active = self
                .nearest_body(range.start)
                .expect("a document always holds at least one body to fall back to");
        }
        self.assert_invariants();
        Some(changes)
    }

    /// Remove every folder immediately above `at` that has just been left with
    /// no children, innermost first.
    ///
    /// **A folder can never be empty**, which is ZBrush's rule and which
    /// removes the empty-folder state from the panel and the resolver rather
    /// than adding a case to each. Recorded innermost-first so that undo, which
    /// runs backwards, puts the outermost one back first and every insertion
    /// index is already right.
    fn dissolve_empty_folders_above(&mut self, at: usize) -> Vec<Change> {
        let mut changes = Vec::new();
        let mut at = at.min(self.nodes.len());
        while at > 0 {
            let index = at - 1;
            let node = &self.nodes[index];
            if node.is_body() {
                break;
            }
            let has_a_child =
                self.nodes.get(index + 1).is_some_and(|next| next.depth() > node.depth());
            if has_a_child {
                break;
            }
            let removed = self.nodes.remove(index);
            changes.push(Change::NodeRemoved { at: index, node: Box::new(removed) });
            at = index;
        }
        changes
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
            totals.mask_bricks += stats.mask_bricks;
            totals.mask_dense_bricks += stats.mask_dense_bricks;
            totals.mask_bytes += stats.mask_bytes;
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

    /// Everything that is true of a document, in one place.
    ///
    /// Ids unique and nonzero, at least one body, the active node holds a
    /// volume -- and **the tree, which is a fold over one integer**: the first
    /// row is at depth 0, no row is more than one level deeper than the row
    /// above it, and a body is never a parent. That is the whole acyclicity
    /// check, and it is a fold rather than a traversal because the encoding
    /// cannot express a cycle in the first place.
    ///
    /// **An empty folder is deliberately NOT here.** Every operation dissolves
    /// one, in the same undo entry, but undo restores a deleted subtree row by
    /// row and the folder necessarily stands alone for the instant between its
    /// own row going back and its first child's. Asserting it would fail on a
    /// correct undo; the rule lives in the operations, where it can be true
    /// between gestures rather than between statements.
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
            debug_assert!(!node.is_body() || !node.collapsed, "a body row has nothing to collapse");
            match index.checked_sub(1).map(|above| &self.nodes[above]) {
                None => debug_assert_eq!(node.depth, 0, "the first row is at the top level"),
                Some(above) if above.is_body() => debug_assert!(
                    node.depth <= above.depth,
                    "a body is not a parent, so depth {} cannot follow a body at {}",
                    node.depth,
                    above.depth
                ),
                Some(above) => debug_assert!(
                    node.depth <= above.depth + 1,
                    "depth {} skips a level below {}",
                    node.depth,
                    above.depth
                ),
            }
        }
    }
}

/// The half-open preorder range this node's subtree occupies: `[at, j)` where
/// `j` is the first index after `at` whose depth is at most `depth[at]`.
///
/// **The only tree primitive there is.** Group, ungroup, move-to-folder and
/// delete-folder are range moves over this, and every legality question is a
/// range comparison rather than a graph search. A body's subtree is always
/// exactly itself, because a body is never a parent.
///
/// Panics on an index past the end, exactly as slicing would: a caller that
/// does not know whether the row is there asks [`Document::subtree_of`].
pub fn subtree(nodes: &[Node], at: usize) -> Range<usize> {
    let depth = nodes[at].depth();
    let end = nodes[at + 1..]
        .iter()
        .position(|node| node.depth() <= depth)
        .map_or(nodes.len(), |offset| at + 1 + offset);
    at..end
}

/// Where a drop would land: a gap in the preorder list, and the depth the
/// dragged subtree's root takes in it.
///
/// **The same value drives the indicator and the commit**, which is what stops
/// the two disagreeing -- the panel draws whatever [`drop_target`] returned and
/// then hands that exact value to [`Document::reparent`], rather than each of
/// them working the answer out from the pointer. Every shipped tree drag that
/// puts a line in one place and the row in another has two copies of this
/// arithmetic.
///
/// `at` is an insertion index into the list AS IT STANDS, so `at == range.end`
/// of the dragged block means "immediately after myself" and not "at the end of
/// whatever is left once I have gone". [`Document::reparent`] does the shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DropTarget {
    pub at: usize,
    pub depth: u8,
}

/// Why a drop cannot happen, in the words the status line uses.
///
/// Three reasons and not a bare `None`, because "nothing happened" is the
/// failure this panel is written against: the drop indicator vanishes at the
/// exact moment the user most needs to be told why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropRefusal {
    /// The block would land inside its own run, which would lose the rows it
    /// was dragged by. **This is the "a folder into its own child" case**, and
    /// it is a range comparison rather than a graph search -- the depth
    /// encoding has nothing to point at an ancestor with, so a cycle is not
    /// detected here, it is unrepresentable.
    IntoItself,
    /// The deepest row of the block would sit past [`MAX_DEPTH`].
    ///
    /// **Refused with the reason and never clamped.** A clamp would flatten a
    /// three-level subtree into a two-level one on a gesture whose whole
    /// meaning was "keep this shape, move it there".
    TooDeep,
    /// There is no such gap, or nothing in front of it could be the block's
    /// parent. Not reachable from the panel; it is what a stale message or a
    /// hand-built [`DropTarget`] gets.
    Nowhere,
}

impl DropRefusal {
    /// The clause the status line puts a row's name in front of.
    #[inline]
    pub fn reason(self) -> &'static str {
        match self {
            Self::IntoItself => "cannot go inside itself",
            Self::TooDeep => "would sit past the eighth level, which is as deep as the panel goes",
            Self::Nowhere => "cannot go there",
        }
    }
}

/// The band of a row, as a fraction of its height, that means "before it" --
/// and, mirrored, "after it".
///
/// A third rather than a half wherever there are three answers, so the middle
/// band is as big as the two edges and dropping INTO a folder is not a
/// pixel-hunt. Rows are 22 or 32 px tall, so a third is 7 px at worst.
const DROP_EDGE: f32 = 1.0 / 3.0;

/// The drag state machine: where the pointer is, reduced to where the block
/// would go.
///
/// A pure function of the list and three numbers -- the row that was pressed,
/// the row the pointer is over, and how far down that row it is -- so the whole
/// gesture is testable without a window, a pointer or a frame.
///
/// # Which gap a row means
///
/// | the row under the pointer | top band | middle | bottom band |
/// |---|---|---|---|
/// | a body | before it | -- | after it |
/// | an OPEN folder | before it | -- | its first child |
/// | a CLOSED folder | before it | inside it, last | after all of it |
///
/// An open folder has no "inside" band, and it needs none: the row directly
/// below it already IS inside it, so the gap under its own row is its first
/// child. That is also what makes a closed folder the only row that draws the
/// filled indicator, which is what section 3 of the plan asks for.
///
/// # Getting back OUT of a folder
///
/// The last child of a folder and the row after the folder are adjacent on
/// screen, and the two of them name the same gap at two different depths --
/// the bottom of the child keeps the block in the folder, the top of the row
/// below takes it out. That is the whole mechanism, and it is why the depth is
/// read off the row the pointer is over rather than off the gap.
///
/// When the folder is the last thing in the document there is no row below it,
/// and the way out is then the top band of the folder's own outermost
/// ancestor -- which is always on screen, because a row is only visible when
/// every folder above it is open.
pub fn drop_target(
    nodes: &[Node],
    dragged: usize,
    over: usize,
    fraction: f32,
) -> Result<DropTarget, DropRefusal> {
    let Some(target) = drop_gap(nodes, dragged, over, fraction) else {
        return Err(DropRefusal::Nowhere);
    };
    match drop_refusal(nodes, subtree(nodes, dragged), target) {
        Some(refusal) => Err(refusal),
        None => Ok(target),
    }
}

/// The gap the pointer names, before any question of whether the block may go
/// there. `None` only when one of the indices is not a row.
///
/// Split out of [`drop_target`] so that **the refused target is still a value**.
/// `drop_target` throws it away on the way to an `Err`, which is fine for the
/// panel -- it draws nothing for a refusal -- but it left the property test
/// unable to hand [`Document::reparent`] the very target the indicator had just
/// turned down, and therefore unable to measure the one thing it claims:
/// that the two ends of the gesture cannot disagree. It stays private because
/// a gap nobody has checked is not something a caller should be able to reach
/// for by accident; the band table lives on `drop_target`.
fn drop_gap(nodes: &[Node], dragged: usize, over: usize, fraction: f32) -> Option<DropTarget> {
    if dragged >= nodes.len() || over >= nodes.len() {
        return None;
    }
    let depth = nodes[over].depth();
    let end = subtree(nodes, over).end;

    // **The dragged block is folded away for the duration of the drag**, so its
    // own row answers to the closed-folder rules whatever its `collapsed` bit
    // says. Without that line, dropping a folder into itself is a state nobody
    // can produce with the pointer and the refusal is untested theatre.
    let closed = !nodes[over].is_body() && (nodes[over].collapsed || over == dragged);
    Some(if closed {
        if fraction < DROP_EDGE {
            DropTarget { at: over, depth }
        } else if fraction > 1.0 - DROP_EDGE {
            DropTarget { at: end, depth }
        } else {
            DropTarget { at: end, depth: depth + 1 }
        }
    } else if nodes[over].is_body() {
        if fraction < 0.5 { DropTarget { at: over, depth } } else { DropTarget { at: end, depth } }
    } else if fraction < 0.5 {
        DropTarget { at: over, depth }
    } else {
        // A folder is never empty, so `over + 1` is always its first child and
        // the depth below always exists.
        DropTarget { at: over + 1, depth: depth + 1 }
    })
}

/// Whether a block may land in a gap, and why not.
///
/// **The one legality predicate**, asked by [`drop_target`] to draw the
/// indicator and again by [`Document::reparent`] to perform the move, so the
/// two cannot admit different things. `range` is the dragged block's own
/// preorder run.
///
/// The three questions, in the order a reader wants the answer in:
///
/// 1. would the block land inside itself;
/// 2. would its deepest row go past the panel's cap;
/// 3. is the gap a gap at all -- something in front of it has to be able to be
///    the block's parent, and nothing after it may end up swallowed by it.
pub fn drop_refusal(
    nodes: &[Node],
    range: Range<usize>,
    target: DropTarget,
) -> Option<DropRefusal> {
    let DropTarget { at, depth } = target;
    if at > nodes.len() || range.start >= nodes.len() {
        return Some(DropRefusal::Nowhere);
    }
    // Strictly inside the block's own run: the splice would cut the block in
    // half with itself.
    if range.start < at && at < range.end {
        return Some(DropRefusal::IntoItself);
    }

    // The row that would become the block's parent: the nearest one before the
    // gap that is shallower than the depth being asked for. Read off the list
    // AS IT STANDS, which is the whole of why `at == range.end` at a deeper
    // depth is self-nesting -- the row in front of that gap is the block's own
    // last row.
    let anchor =
        (depth > 0).then(|| (0..at).rev().find(|index| nodes[*index].depth() < depth)).flatten();
    if anchor.is_some_and(|index| range.contains(&index)) {
        return Some(DropRefusal::IntoItself);
    }

    let Some(deepest) =
        nodes.get(range.clone()).and_then(|block| block.iter().map(Node::depth).max())
    else {
        // An empty or out-of-bounds run, which is not a block at all.
        return Some(DropRefusal::Nowhere);
    };
    let by = i16::from(depth) - i16::from(nodes[range.start].depth());
    if i16::from(deepest) + by >= i16::from(MAX_DEPTH) {
        return Some(DropRefusal::TooDeep);
    }

    // A depth with no parent to hang off, or a row after the gap that the block
    // would swallow as a descendant. Neither is reachable from the panel; both
    // are what stops a hand-built target from producing a list that is not a
    // forest.
    if depth > 0 && anchor.is_none_or(|index| nodes[index].depth() + 1 != depth) {
        return Some(DropRefusal::Nowhere);
    }
    if nodes.get(at).is_some_and(|node| node.depth() > depth) && at != range.start {
        return Some(DropRefusal::Nowhere);
    }
    None
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
    /// A guard over two numbers stated outright.
    ///
    /// **Test-only, and it exists because the ceilings cannot be reached by
    /// building a document.** Six gigabytes of resident bricks is six gigabytes
    /// of real allocation, so the refusals in this file construct the struct
    /// directly -- and a refusal that lives in another module, such as
    /// [`crate::split::SplitPlan`]'s, has no way to do that without this.
    #[cfg(test)]
    pub(crate) fn of(resident_bytes: f64, pool_headroom: f64) -> Self {
        Self { resident_bytes, pool_headroom }
    }

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
        let (why, fit) = self.shortfall(bytes, vertices)?;
        let workable = (fit.sqrt() * Self::MARGIN) as f32;
        Some((format!("{why} -- {:.0}% of that size would fit", workable * 100.0), workable))
    }

    /// [`GrowthGuard::no_room_for`] for an add that has **no size lever**.
    ///
    /// Duplicate is the case, and merge will be the second: a copy is the size
    /// of what it copies, and there is no control anywhere in the interface
    /// that makes one smaller. Telling that user "45% of that size would fit"
    /// names a size they cannot ask for, which is worse than saying nothing --
    /// the established pattern is that a refusal names the size that WOULD
    /// work, and the honest reading of it here is that no size would.
    ///
    /// Both ceilings, both numbers and the same arithmetic; only the suggestion
    /// is dropped. Sharing [`GrowthGuard::shortfall`] rather than being written
    /// out again is what keeps the two refusals from drifting into quoting
    /// different headroom for the same document.
    pub fn no_room_for_a_copy(&self, bytes: f64, vertices: f64) -> Option<String> {
        self.shortfall(bytes, vertices).map(|(why, _)| why)
    }

    /// Why a body of this cost does not fit, and the fraction of it that would.
    ///
    /// The fraction is of the COST, not of the linear size; squaring it back
    /// into a size belongs to the caller that has a size to offer.
    fn shortfall(&self, bytes: f64, vertices: f64) -> Option<(String, f64)> {
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
        Some((why, fit))
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

    // --- the tree ----------------------------------------------------------

    /// The depth column of a document, which is the whole of its shape.
    fn depths(doc: &Document) -> Vec<u8> {
        doc.nodes().iter().map(Node::depth).collect()
    }

    /// Three bodies at the top level, named for what they are.
    fn three() -> (Document, NodeId, NodeId, NodeId) {
        let mut doc = Document::new(0.5);
        let first = doc.active();
        let second = doc.add_body("Body 2", Volume::new(0.5));
        let third = doc.add_body("Body 3", Volume::new(0.5));
        (doc, first, second, third)
    }

    /// The one tree primitive, on the case that decides every range move: a
    /// folder's subtree ends where the depth column comes back up, and a
    /// body's is exactly itself.
    #[test]
    fn a_subtree_is_the_preorder_run_of_everything_deeper_than_its_root() {
        let (mut doc, first, second, third) = three();
        // folder > [folder > first] , second, third
        let (inner, _) = doc.group(first, "Inner").expect("the inner group");
        let (outer, _) = doc.group(inner, "Outer").expect("the outer group");

        assert_eq!(depths(&doc), vec![0, 1, 2, 0, 0]);
        assert_eq!(doc.subtree_of(outer), Some(0..3), "the outer folder lost its descendants");
        assert_eq!(doc.subtree_of(inner), Some(1..3));
        assert_eq!(doc.subtree_of(first), Some(2..3), "a body's subtree is exactly itself");
        assert_eq!(doc.subtree_of(second), Some(3..4));
        assert_eq!(doc.subtree_of(third), Some(4..5), "the last row's subtree ran off the end");
        assert_eq!(doc.subtree_body_count(outer), 1);
        assert_eq!(doc.parent_of(first), Some(inner));
        assert_eq!(doc.parent_of(outer), None);
    }

    /// ctrl+G then ctrl+shift+G is the identity, outline for outline, which is
    /// what makes the pair safe to press without thinking.
    #[test]
    fn grouping_and_ungrouping_leaves_the_outline_exactly_as_it_was() {
        let (mut doc, _, second, _) = three();
        let was = doc.outline();

        let (folder, _) = doc.group(second, "Group 1").expect("the group");
        assert_eq!(depths(&doc), vec![0, 0, 1, 0], "the grouped row did not move inside");
        assert_eq!(doc.node_count(), 4);

        doc.ungroup(folder).expect("the ungroup");
        assert_eq!(doc.outline(), was, "the pair is not each other's inverse");
    }

    /// The whole gesture, undone and redone, with the folder and the depths
    /// moving together.
    ///
    /// It is two changes in one entry and the ORDER between them is the thing
    /// under test: undo runs them backwards, so the depths go back while the
    /// folder is still in the list and only then does the folder go. Recorded
    /// the other way round, `set_outline` would be handed a snapshot one row
    /// short of the document.
    #[test]
    fn one_undo_takes_a_whole_group_apart_and_one_redo_rebuilds_it() {
        let (mut doc, _, second, _) = three();
        let was = doc.outline();

        let (_, changes) = doc.group(second, "Group 1").expect("the group");
        let grouped = doc.outline();
        let mut history = crate::undo::History::new(1 << 20);
        history.push(crate::undo::Entry::new(changes));

        let shown = all_shown(&doc);
        history.undo(&mut doc, &shown);
        assert_eq!(doc.outline(), was, "one undo did not take the group apart");

        let shown = all_shown(&doc);
        history.redo(&mut doc, &shown);
        assert_eq!(doc.outline(), grouped, "one redo did not put the group back");
    }

    /// Eight levels is the cap, so the deepest legal row is at depth 7 and the
    /// press that would make an eighth is refused rather than clamped.
    ///
    /// Clamping would be worse than refusing: the row would appear to move and
    /// then not have, with a folder minted around it either way.
    #[test]
    fn grouping_stops_at_the_eighth_level_rather_than_clamping_into_it() {
        let mut doc = Document::new(0.5);
        let body = doc.active();
        // Seven folders wrap the body, which puts it at depth 7 -- the deepest
        // legal row, since MAX_DEPTH is 8 and legal depths are 0..=7.
        for level in 0..MAX_DEPTH - 1 {
            doc.group(body, format!("Group {level}")).expect("a legal group");
        }
        assert_eq!(doc.node(body).expect("the body").depth(), MAX_DEPTH - 1);
        assert_eq!(doc.node_count(), usize::from(MAX_DEPTH));

        let rows = doc.node_count();
        assert!(doc.group(body, "One too many").is_none(), "an eighth level was allowed");
        assert_eq!(doc.node_count(), rows, "the refused group left a folder behind");
    }

    /// **A cycle is unrepresentable, and this is the check that stands in for
    /// the acyclicity validator the encoding deletes.**
    ///
    /// There is no way to express "this folder's parent is its own child" in a
    /// preorder array plus a depth column, so the only thing a move CAN get
    /// wrong is being asked to send a subtree inside itself -- and that is one
    /// range comparison, not a graph search.
    #[test]
    fn a_folder_cannot_be_moved_into_its_own_subtree() {
        let (mut doc, first, _, _) = three();
        let (inner, _) = doc.group(first, "Inner").expect("the inner group");
        let (outer, _) = doc.group(inner, "Outer").expect("the outer group");
        let was = doc.outline();

        assert!(doc.move_to_folder(outer, Some(inner)).is_none(), "a folder went inside itself");
        assert!(doc.move_to_folder(outer, Some(outer)).is_none(), "a folder went inside itself");
        assert_eq!(doc.outline(), was, "a refused move changed the document");
    }

    /// A row moves into a folder, and back out to the top level, carrying its
    /// depth with it.
    #[test]
    fn a_body_moves_into_a_folder_and_back_out_again() {
        let (mut doc, first, second, third) = three();
        let (folder, _) = doc.group(first, "Group 1").expect("the group");

        doc.move_to_folder(second, Some(folder)).expect("the move in");
        assert_eq!(
            doc.nodes().iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![folder, first, second, third],
            "the row did not land inside the folder"
        );
        assert_eq!(depths(&doc), vec![0, 1, 1, 0]);

        doc.move_to_folder(second, None).expect("the move out");
        assert_eq!(
            doc.nodes().iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![folder, first, third, second],
            "the row did not come back out to the end of the top level"
        );
        assert_eq!(depths(&doc), vec![0, 1, 0, 0]);
    }

    /// **A folder can never be empty**, so taking its last child out takes the
    /// folder with it -- and in the SAME list of changes, so one ctrl+Z puts
    /// both back.
    #[test]
    fn moving_a_folders_last_child_out_dissolves_the_folder_in_one_entry() {
        let (mut doc, first, _, _) = three();
        let (folder, _) = doc.group(first, "Group 1").expect("the group");
        let was = doc.outline();

        let changes = doc.move_to_folder(first, None).expect("the move out");
        assert!(
            doc.node(folder).is_none(),
            "the folder its last child left is still in the document"
        );
        assert_eq!(depths(&doc), vec![0, 0, 0]);

        let mut history = crate::undo::History::new(1 << 20);
        history.push(crate::undo::Entry::new(changes));
        assert_eq!(history.stats().undo_entries, 1, "the move and the dissolve are two gestures");
        let shown = all_shown(&doc);
        history.undo(&mut doc, &shown);
        assert_eq!(doc.outline(), was, "one undo did not restore the folder and the row together");
    }

    /// A folder delete is N removals in ONE entry, and undo puts the FOLDER
    /// back before the bodies that live in it.
    ///
    /// The order is the thing under test and getting it wrong is not silent --
    /// a body restored into a list with no folder above it is a body at a depth
    /// the fold refuses -- but it would only fail inside a `debug_assert`, so
    /// the row order is checked here where the failure has a name.
    #[test]
    fn a_folder_delete_is_one_entry_that_restores_the_folder_before_its_bodies() {
        let (mut doc, first, second, third) = three();
        let keeper = doc.add_body("Keeper", Volume::new(0.5));
        let (folder, _) = doc.group(first, "Group 1").expect("the group");
        doc.move_to_folder(second, Some(folder)).expect("the second in");
        doc.move_to_folder(third, Some(folder)).expect("the third in");
        doc.set_active(keeper);
        let was = doc.outline();
        assert_eq!(doc.subtree_body_count(folder), 3);

        let changes = doc.delete_subtree(folder).expect("the folder delete");
        assert_eq!(changes.len(), 4, "a folder of three bodies is four removals");
        assert_eq!(doc.node_count(), 1, "the delete left rows behind");

        let entry = crate::undo::Entry::new(changes);
        let mut history = crate::undo::History::new(1 << 20);
        history.push(entry);
        assert_eq!(history.stats().undo_entries, 1, "four removals became four gestures");

        let shown = all_shown(&doc);
        history.undo(&mut doc, &shown);
        assert_eq!(doc.outline(), was, "one undo did not put the whole folder back");
    }

    /// The reclaim allowance is what predicts the 512 MB prompt, so a folder
    /// delete has to charge it the SUM of what it is holding rather than one
    /// body's worth.
    #[test]
    fn a_folder_delete_charges_the_reclaim_allowance_the_sum_of_its_bodies() {
        let mut doc = Document::new(1.0);
        let keeper = doc.active();
        let mut resident = 0usize;
        let mut inside = Vec::new();
        for n in 0..3 {
            let mut volume = Volume::new(1.0);
            volume.seed_sphere(Vec3::new(n as f32 * 80.0, 0.0, 0.0), 16.0);
            resident += volume.stats().resident_bytes;
            inside.push(doc.add_body(format!("Body {n}"), volume));
        }
        let (folder, _) = doc.group(inside[0], "Group 1").expect("the group");
        doc.move_to_folder(inside[1], Some(folder)).expect("the second in");
        doc.move_to_folder(inside[2], Some(folder)).expect("the third in");
        doc.set_active(keeper);

        let entry = crate::undo::Entry::new(doc.delete_subtree(folder).expect("the folder delete"));
        assert!(
            entry.reclaim_bytes() >= resident,
            "three bodies of {resident} bytes were charged only {}",
            entry.reclaim_bytes()
        );
    }

    /// **Deleting a body inside a collapsed folder deletes the body, never the
    /// folder.** Collapse changes only what is drawn.
    ///
    /// In ZBrush it does the other thing, a user reported losing an
    /// unrecoverable hour to it, the bundled Delete macro had the same hole,
    /// and a third-party plugin exists solely to intercept it.
    #[test]
    fn deleting_a_body_inside_a_collapsed_folder_never_takes_the_folder() {
        let (mut doc, first, second, _) = three();
        let (folder, _) = doc.group(first, "Group 1").expect("the group");
        doc.move_to_folder(second, Some(folder)).expect("the second in");
        doc.set_collapsed(folder, true).expect("the collapse");

        doc.delete_subtree(first).expect("the body delete");
        assert!(doc.node(folder).is_some(), "a collapsed folder swallowed a body delete");
        assert!(doc.node(second).is_some(), "the folder's other body went with it");
        assert!(doc.node(first).is_none(), "the body the user asked about is still here");
        assert!(doc.node(folder).expect("the folder").collapsed, "the collapse was thrown away");
    }

    /// Deleting the LAST body in a folder does take the folder, because a
    /// folder can never be empty -- and that is the other half of the rule
    /// above rather than a contradiction of it.
    #[test]
    fn deleting_a_folders_last_body_dissolves_the_folder_with_it() {
        let (mut doc, first, _, _) = three();
        let (folder, _) = doc.group(first, "Group 1").expect("the group");

        let changes = doc.delete_subtree(first).expect("the body delete");
        assert_eq!(changes.len(), 2, "the body and its emptied folder are one entry of two");
        assert!(doc.node(folder).is_none(), "an empty folder was left behind");
    }

    /// The last body cannot go, however it is asked for.
    #[test]
    fn a_delete_that_would_take_every_body_is_refused() {
        let mut doc = Document::new(0.5);
        let only = doc.active();
        let (folder, _) = doc.group(only, "Group 1").expect("the group");

        assert!(doc.delete_subtree(only).is_none(), "the last body was deleted");
        assert!(doc.delete_subtree(folder).is_none(), "the folder holding the last body went");
        assert_eq!(doc.body_count(), 1);
    }

    /// **The property test the flat encoding exists to make possible.**
    ///
    /// A thousand random operations over a document that starts with six
    /// bodies: group, ungroup, move, collapse, delete and duplicate, each aimed
    /// at a row chosen by the same seeded noise. Every one of them runs
    /// `Document::assert_invariants`, so the tree fold, the id uniqueness and
    /// the active-holds-a-volume rule are checked after every single step; this
    /// asserts the things that fold cannot see, and that the sequence never
    /// panics.
    ///
    /// A recursive validator is exactly what is NOT needed here, and that is
    /// the point: no sequence of these operations can produce a cycle, because
    /// there is nothing in the encoding to express one with.
    ///
    /// **The counted control matters as much as the property.** A version of
    /// this that refused every operation would pass it perfectly, so the run
    /// asserts that each of the seven actually landed and that the tree really
    /// got deep. Deletes are the arm that needs the last one: with nothing
    /// adding bodies back, a six-body document runs out of them in five presses
    /// and the delete arm goes quiet.
    #[test]
    fn a_thousand_random_tree_operations_keep_the_tree_a_tree() {
        let mut noise = crate::testing::Noise::seeded(0x5eed_1234);
        let mut doc = Document::new(0.5);
        for n in 1..6 {
            doc.add_body(format!("Body {n}"), Volume::new(0.5));
        }

        let mut landed = [0usize; 7];
        let mut deepest = 0u8;
        for step in 0..1000 {
            let ids: Vec<NodeId> = doc.nodes().iter().map(|node| node.id).collect();
            let chosen = ids[noise.below(ids.len())];
            let operation = noise.below(7);
            let done = match operation {
                0 => {
                    doc.node_count() < MAX_NODES
                        && doc.group(chosen, format!("Group {step}")).is_some()
                }
                1 => doc.ungroup(chosen).is_some(),
                2 => {
                    let into = (noise.below(2) == 0).then(|| ids[noise.below(ids.len())]);
                    doc.move_to_folder(chosen, into).is_some()
                }
                3 => doc.set_collapsed(chosen, noise.below(2) == 0).is_some(),
                4 => doc.delete_subtree(chosen).is_some(),
                // The duplicate button: a body copied BESIDE its source, which
                // is the one insert in the application that lands in the middle
                // of the list rather than at the end. It is here because the
                // first five operations never exercised a mid-list insert, and
                // the one that shipped assumed depth 0 -- so a copy made inside
                // a folder ended that folder's run and evicted its siblings,
                // with the whole suite green.
                5 => {
                    let at = doc.index_of(chosen).expect("a chosen id is in the document");
                    let room = doc.body_count() < MAX_BODIES && doc.node_count() < MAX_NODES;
                    let body = doc.nodes()[at].is_body();
                    if room && body {
                        let depth = doc.nodes()[at].depth();
                        doc.insert_body(at + 1, depth, format!("Copy {step}"), Volume::new(0.5));
                    }
                    room && body
                }
                // Bodies come back, or the run runs out of them in five
                // presses and the delete arm stops proving anything.
                _ => {
                    let room = doc.body_count() < MAX_BODIES && doc.node_count() < MAX_NODES;
                    if room {
                        doc.add_body(format!("Body {step}"), Volume::new(0.5));
                    }
                    room
                }
            };
            landed[operation] += usize::from(done);
            deepest = deepest.max(depths(&doc).into_iter().max().unwrap_or(0));

            // What the fold cannot see, checked here rather than left to a
            // release build's silence.
            assert!(doc.body_count() >= 1, "step {step} deleted every body");
            assert!(
                doc.volume(doc.active()).is_some(),
                "step {step} left the active row without a field"
            );
            // **A folder can never be empty**, which is the one rule that is
            // NOT a document invariant -- undo restores a subtree row by row,
            // so it cannot hold between statements. It has to hold between
            // gestures, and this is where that is checked.
            for (index, node) in doc.nodes().iter().enumerate() {
                assert!(
                    node.is_body()
                        || doc
                            .nodes()
                            .get(index + 1)
                            .is_some_and(|next| next.depth() > node.depth()),
                    "step {step} left {} empty at row {index}: {:?}",
                    node.name,
                    depths(&doc)
                );
            }
            let mut resolved = Vec::new();
            doc.display_visibility(None, &mut resolved);
            assert_eq!(resolved.len(), doc.node_count(), "step {step}: the resolver lost a row");
        }

        assert!(
            landed.iter().all(|count| *count > 10),
            "some operation never landed, so the run proves less than it looks: {landed:?}"
        );
        assert_eq!(deepest, MAX_DEPTH - 1, "the run never reached the deepest legal level");
    }

    /// The bug the property test above found, pinned as its own case so that a
    /// regression fails with a name rather than at step 355 of a random run.
    ///
    /// Moving a folder's only child into a folder that sits ABOVE it inserts
    /// the block in FRONT of the slot it left, which shifts that slot up by the
    /// block's own length. Dissolving at the old index then looks at the wrong
    /// row, finds a body, stops -- and the emptied folder stays in the list
    /// with no children, which is a state nothing else in the design expects.
    #[test]
    fn moving_a_child_into_a_folder_above_it_still_dissolves_the_one_it_left() {
        let (mut doc, first, second, _) = three();
        let (target, _) = doc.group(first, "Target").expect("the folder above");
        let (source, _) = doc.group(second, "Source").expect("the folder below");
        assert_eq!(depths(&doc), vec![0, 1, 0, 1, 0], "the fixture is not the shape under test");
        assert!(
            doc.index_of(target) < doc.index_of(source),
            "the target folder has to sit above the source one"
        );

        doc.move_to_folder(second, Some(target)).expect("the move up");
        assert!(doc.node(source).is_none(), "the folder its only child left is still here");
        assert!(doc.node(target).is_some());
        assert_eq!(depths(&doc), vec![0, 1, 1, 0]);
    }

    // --- the drag ------------------------------------------------------------

    /// A depth-3 tree with one of everything a drop can land on: an open
    /// folder, a CLOSED folder with a row hidden inside it, a nested body, two
    /// top-level rows and a folder holding one child.
    ///
    /// ```text
    /// 0 Outer            depth 0, open
    /// 1   Inner          depth 1, open
    /// 2     Deep         depth 2, CLOSED
    /// 3       Body 1     depth 3
    /// 4     Nested       depth 2
    /// 5 Loose            depth 0
    /// 6 Shelf            depth 0, open
    /// 7   Shelved        depth 1
    /// ```
    fn depth_three() -> Document {
        let mut doc = Document::new(0.5);
        let buried = doc.active();
        let nested = doc.add_body("Nested", Volume::new(0.5));
        doc.add_body("Loose", Volume::new(0.5));
        let shelved = doc.add_body("Shelved", Volume::new(0.5));

        let (deep, _) = doc.group(buried, "Deep").expect("Deep");
        let (inner, _) = doc.group(deep, "Inner").expect("Inner");
        doc.move_to_folder(nested, Some(inner)).expect("Nested into Inner");
        doc.group(inner, "Outer").expect("Outer");
        doc.group(shelved, "Shelf").expect("Shelf");
        doc.set_collapsed(deep, true).expect("Deep closed");

        assert_eq!(depths(&doc), vec![0, 1, 2, 3, 2, 0, 0, 1], "the fixture is the wrong shape");
        assert_eq!(
            doc.nodes().iter().map(|node| node.name.as_str()).collect::<Vec<_>>(),
            vec!["Outer", "Inner", "Deep", "Body 1", "Nested", "Loose", "Shelf", "Shelved"],
            "the fixture's rows are in the wrong order"
        );
        doc
    }

    /// Every band of every row kind, written out as a table, because the whole
    /// of the gesture is this mapping and a reader should be able to check it
    /// against the doc comment's table without running anything.
    ///
    /// Read the fixture in [`depth_three`] alongside it.
    #[test]
    fn where_a_drop_lands_is_a_table_over_the_row_under_the_pointer() {
        let doc = depth_three();
        let nodes = doc.nodes();
        // Dragging Loose, which is a top-level body and therefore the source
        // that makes every destination legal that can be legal.
        const LOOSE: usize = 5;

        type Case = (&'static str, usize, f32, Result<DropTarget, DropRefusal>);
        let cases: [Case; 14] = [
            // A body: two bands, because a body has no inside.
            ("above Body 1", 3, 0.1, Ok(DropTarget { at: 3, depth: 3 })),
            ("below Body 1", 3, 0.9, Ok(DropTarget { at: 4, depth: 3 })),
            ("just above Body 1's middle", 3, 0.49, Ok(DropTarget { at: 3, depth: 3 })),
            ("just below it", 3, 0.51, Ok(DropTarget { at: 4, depth: 3 })),
            // An OPEN folder: the gap under its row is its first child.
            ("above Outer", 0, 0.1, Ok(DropTarget { at: 0, depth: 0 })),
            ("below Outer", 0, 0.9, Ok(DropTarget { at: 1, depth: 1 })),
            ("below Shelf", 6, 0.9, Ok(DropTarget { at: 7, depth: 1 })),
            // A CLOSED folder: three bands, and the middle one is inside it.
            ("above Deep", 2, 0.1, Ok(DropTarget { at: 2, depth: 2 })),
            ("into Deep", 2, 0.5, Ok(DropTarget { at: 4, depth: 3 })),
            ("below Deep, past its hidden child", 2, 0.9, Ok(DropTarget { at: 4, depth: 2 })),
            // The two ways to name one gap, which is how a row gets OUT.
            ("below Shelved, which keeps it in Shelf", 7, 0.9, Ok(DropTarget { at: 8, depth: 1 })),
            ("above Loose, which is the top level", 5, 0.1, Ok(DropTarget { at: 5, depth: 0 })),
            // Nested sits at depth 2 and Loose is one row.
            ("into Deep from Loose is fine", 2, 0.5, Ok(DropTarget { at: 4, depth: 3 })),
            ("a row that is not here", 99, 0.5, Err(DropRefusal::Nowhere)),
        ];

        for (what, over, fraction, want) in cases {
            assert_eq!(drop_target(nodes, LOOSE, over, fraction), want, "{what}");
        }
    }

    /// The three illegal cases, each from the gesture that produces it, because
    /// a refusal nobody can reach with the pointer is untested theatre.
    #[test]
    fn a_folder_dropped_into_itself_a_block_too_deep_and_a_gap_that_is_not_one_are_refused() {
        let doc = depth_three();
        let nodes = doc.nodes();

        // Outer is folded away for the duration of its own drag, so its row
        // answers to the closed-folder rules and its middle band is "inside
        // me". That is the cycle, and it is the one the depth encoding makes
        // unrepresentable rather than merely detectable.
        assert_eq!(drop_target(nodes, 0, 0, 0.5), Err(DropRefusal::IntoItself), "Outer into Outer");
        // And the same folder over a row of its own, which is only reachable
        // through a stale message -- the rows are not drawn during the drag.
        assert_eq!(drop_target(nodes, 0, 2, 0.5), Err(DropRefusal::IntoItself), "Outer into Deep");

        // The cap. It takes a tall block AND a deep destination to reach, which
        // a depth-3 tree on its own cannot do: eight levels is a lot of room.
        let deep = a_tower_and_a_tall_block();
        assert_eq!(
            depths(&deep),
            vec![0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "the tower is the wrong shape"
        );
        assert_eq!(
            drop_target(deep.nodes(), 7, 6, 0.9),
            Err(DropRefusal::TooDeep),
            "a three-level block under a depth-6 row would put its deepest row at 8"
        );
        // One level shallower and the same block fits, so the refusal above is
        // the cap talking and not the gesture failing.
        assert!(drop_target(deep.nodes(), 7, 4, 0.9).is_ok(), "the block does not fit anywhere");

        // A gap with nothing in front of it that could be a parent. Not
        // reachable from the panel; this is what a hand-built target gets.
        assert_eq!(
            drop_refusal(nodes, 5..6, DropTarget { at: 0, depth: 3 }),
            Some(DropRefusal::Nowhere),
            "a depth-3 row at the very top of the list has no parent"
        );
        assert!(!DropRefusal::IntoItself.reason().is_empty());
        assert!(!DropRefusal::TooDeep.reason().is_empty());
        assert!(!DropRefusal::Nowhere.reason().is_empty());
    }

    /// A chain six folders deep with a body at the bottom, and beside it a
    /// three-level block. The only shape in which the eighth-level cap can
    /// actually bite.
    ///
    /// ```text
    /// 0 L0 .. 5 L5      folders, depths 0 to 5
    /// 6   Pit           a body at depth 6
    /// 7 Block           depth 0
    /// 8   B1            depth 1
    /// 9     Leaf        depth 2
    /// ```
    fn a_tower_and_a_tall_block() -> Document {
        let mut doc = Document::new(0.5);
        let pit = doc.active();
        let leaf = doc.add_body("Leaf", Volume::new(0.5));

        let mut outermost = pit;
        for level in (0..6).rev() {
            let (folder, _) = doc.group(outermost, format!("L{level}")).expect("a level");
            outermost = folder;
        }
        let (b1, _) = doc.group(leaf, "B1").expect("B1");
        doc.group(b1, "Block").expect("Block");
        doc
    }

    /// **Every (source, destination) pair over a depth-3 tree**, at every band
    /// of every row: each one either produces a legal forest or is refused with
    /// a reason, and the predicate that drew the indicator is the predicate the
    /// commit obeyed.
    ///
    /// Run over the tower as well, because a depth-3 tree has too much headroom
    /// under the eighth level for the cap to ever bite -- and a refusal no case
    /// reaches is a refusal nothing tests.
    ///
    /// The counted control at the bottom matters as much as the property: a
    /// version of this that refused everything would pass it perfectly.
    #[test]
    fn every_drop_over_a_depth_three_tree_is_a_legal_forest_or_a_named_refusal() {
        type Fixture = (&'static str, fn() -> Document);
        let fixtures: [Fixture; 2] =
            [("the depth-3 tree", depth_three), ("the tower", a_tower_and_a_tall_block)];
        let mut landed = 0usize;
        let mut refused = [0usize; 3];
        let mut moved_something = 0usize;

        for (fixture, build) in fixtures {
            let rows = build().node_count();
            let bodies = build().body_count();
            for source in 0..rows {
                for over in 0..rows {
                    for band in [0.1_f32, 0.5, 0.9] {
                        // Rebuilt per case: `Volume` has no `Clone` at all, so
                        // there is no snapshot to restore and no shared state
                        // to leak between cases.
                        let mut doc = build();
                        let id = doc.nodes()[source].id;
                        let block: Vec<NodeId> = doc.nodes()[subtree(doc.nodes(), source)]
                            .iter()
                            .map(|node| node.id)
                            .collect();
                        let was = depths(&doc);

                        let outcome = drop_target(doc.nodes(), source, over, band);
                        let what = format!("{fixture}: {source} over {over} at {band}");
                        let Ok(target) = outcome else {
                            match outcome.expect_err("checked above") {
                                DropRefusal::IntoItself => refused[0] += 1,
                                DropRefusal::TooDeep => refused[1] += 1,
                                DropRefusal::Nowhere => refused[2] += 1,
                            }
                            // **The indicator and the commit cannot disagree**:
                            // the gap the panel refused to draw a line in is a
                            // gap the document also refuses to splice into.
                            // THE SAME TARGET, which is why this reaches past
                            // `drop_target` for it -- an assertion over a
                            // target built here by hand would be a constant,
                            // and one was: `DropTarget { at: 0, depth:
                            // MAX_DEPTH }` is `TooDeep` for every block in
                            // every document, so it passed whatever refusal was
                            // under test.
                            let refused_target = drop_gap(doc.nodes(), source, over, band)
                                .expect("both indices are rows");
                            assert!(
                                doc.reparent(id, refused_target).is_none(),
                                "{what}: the commit performed a move the indicator refused"
                            );
                            assert_eq!(
                                depths(&doc),
                                was,
                                "{what}: a refused move still moved rows"
                            );
                            continue;
                        };
                        landed += 1;

                        if doc.reparent(id, target).is_none() {
                            assert_eq!(
                                depths(&doc),
                                was,
                                "{what}: a refused move still moved rows"
                            );
                            continue;
                        }
                        moved_something += 1;

                        // A legal forest: the fold over one integer that IS the
                        // whole tree check.
                        let after = depths(&doc);
                        assert_eq!(after[0], 0, "{what}: the first row is not at the top level");
                        for (index, pair) in after.windows(2).enumerate() {
                            assert!(
                                pair[1] <= pair[0] + 1,
                                "{what}: row {} skipped a level in {after:?}",
                                index + 1
                            );
                        }
                        assert!(after.iter().all(|depth| *depth < MAX_DEPTH), "{what}: {after:?}");
                        // Nothing was lost. Rows CAN go -- a folder its last
                        // child left is dissolved into the same entry -- but a
                        // body never can, and neither can any row of the block
                        // that was dragged.
                        assert_eq!(doc.body_count(), bodies, "{what}: a body went missing");
                        for id in &block {
                            assert!(doc.node(*id).is_some(), "{what}: a dragged row went missing");
                        }
                        // The block stayed a block, in its own order: the rows
                        // are still contiguous and still the same shape.
                        let landed_at =
                            doc.index_of(block[0]).expect("the block's root is still here");
                        let arrived: Vec<NodeId> = doc.nodes()[landed_at..landed_at + block.len()]
                            .iter()
                            .map(|node| node.id)
                            .collect();
                        assert_eq!(arrived, block, "{what}: the block did not move as a block");

                        // **A folder is never empty**, which is a rule between
                        // gestures rather than a document invariant.
                        for (index, node) in doc.nodes().iter().enumerate() {
                            assert!(
                                node.is_body()
                                    || doc
                                        .nodes()
                                        .get(index + 1)
                                        .is_some_and(|next| next.depth() > node.depth()),
                                "{what}: {} was left empty in {after:?}",
                                node.name
                            );
                        }
                        assert!(
                            doc.volume(doc.active()).is_some(),
                            "{what}: the active row lost its field"
                        );
                    }
                }
            }
        }

        assert!(landed > 100, "too few drops were legal for this to prove anything: {landed}");
        assert!(moved_something > 50, "almost every legal drop was a no-op: {moved_something}");
        assert!(refused[0] > 0, "no drop was ever refused for landing inside itself");
        assert!(refused[1] > 0, "no drop was ever refused for being too deep");
        // **Zero, and asserted as zero.** `Nowhere` is the refusal for a gap
        // that is not one, and `drop_target` cannot produce one: every band of
        // the table lands on an index the list has, at a depth the row in front
        // of it can parent. It is reachable only by a hand-built target or a
        // stale message, which is what
        // `a_folder_dropped_into_itself_a_block_too_deep_and_a_gap_that_is_not_one_are_refused`
        // covers. If this ever counts one, the band table has grown a gap the
        // panel could point at and nothing else here would have said so.
        assert_eq!(refused[2], 0, "the pointer produced a gap that is not a gap");
    }

    /// The gap under a folder's last child and the gap above the row after the
    /// folder are the SAME index at two depths, and that is the whole of how a
    /// row gets back out of a folder.
    #[test]
    fn one_gap_at_two_depths_is_what_takes_a_row_out_of_a_folder() {
        let mut doc = depth_three();
        let shelved = doc.nodes()[7].id;
        // Nothing below Shelf, so the way out is the top band of a top-level
        // row -- here, Loose, which sits above it.
        let out = drop_target(doc.nodes(), 7, 5, 0.1).expect("above Loose");
        assert_eq!(out, DropTarget { at: 5, depth: 0 });

        doc.reparent(shelved, out).expect("the move out");
        assert_eq!(depths(&doc), vec![0, 1, 2, 3, 2, 0, 0], "Shelved did not come out");
        assert_eq!(doc.nodes()[5].id, shelved, "it did not land above Loose");
        assert_eq!(
            doc.nodes().iter().filter(|node| node.name == "Shelf").count(),
            0,
            "the folder its only child left is still here"
        );
    }

    /// A drag that puts a row back where it already was is not a change, and a
    /// change is what an undo entry costs. Both bands of both neighbours.
    #[test]
    fn a_drop_that_changes_nothing_is_not_a_move() {
        let mut doc = depth_three();
        let nested = doc.nodes()[4].id;
        let was = doc.outline();

        for (what, over, band) in [
            ("its own top band", 4, 0.1),
            ("its own bottom band", 4, 0.9),
            ("above itself", 4, 0.4),
        ] {
            let target = drop_target(doc.nodes(), 4, over, band).expect(what);
            assert!(doc.reparent(nested, target).is_none(), "{what} counted as a move");
            assert_eq!(doc.outline(), was, "{what} changed the document");
        }
    }

    /// The outline is a permutation plus field edits, so putting one back has
    /// to restore the ORDER as well as the fields.
    #[test]
    fn an_outline_snapshot_restores_the_order_and_not_only_the_fields() {
        let (mut doc, first, second, third) = three();
        let was = doc.outline();

        let (folder, _) = doc.group(third, "Group 1").expect("the group");
        doc.move_to_folder(first, Some(folder)).expect("the move");
        assert_ne!(
            doc.nodes().iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![first, second, third],
            "nothing moved, so this proves nothing"
        );

        // The folder has to go before the snapshot can: a snapshot names the
        // rows it covers, and `was` predates the folder entirely.
        doc.move_to_folder(third, None).expect("the folder's other child out");
        doc.move_to_folder(first, None).expect("the last child out, which dissolves the folder");
        assert!(doc.node(folder).is_none());
        doc.set_outline(&was);
        assert_eq!(doc.outline(), was, "the outline did not come back");
    }

    // --- the visibility resolver ------------------------------------------

    /// A body row at a chosen depth with a chosen eye, built straight rather
    /// than through a [`Document`].
    ///
    /// [`resolve_visibility`] is a free function over `&[Node]` precisely so
    /// that a tree can be handed to it row by row, without the group operation
    /// that would otherwise have to exist to build one.
    fn row(id: u32, depth: u8, visible: bool) -> Node {
        Node::from_meta(
            NodeMeta {
                id: NodeId(id),
                depth,
                name: format!("row {id}"),
                visible,
                collapsed: false,
            },
            Some(Volume::new(0.5)),
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

    /// **The duplicate refusal, which is the 4 GB case.** A second copy of a
    /// 4 GB body against a 6 GB ceiling does not fit, and the message has to
    /// carry BOTH numbers -- what it needs and what is left -- because a user
    /// who is told only one of them cannot tell whether to delete a body,
    /// resample coarser, or give up.
    ///
    /// It must NOT name a fraction. Duplicate has no size lever anywhere in the
    /// interface: a copy is the size of what it copies, so "62% of that size
    /// would fit" points at a control that does not exist.
    #[test]
    fn a_copy_that_does_not_fit_is_refused_with_both_numbers_and_no_size_to_try() {
        let gigabyte = 1024.0 * 1024.0 * 1024.0;
        let guard = GrowthGuard { resident_bytes: 4.0 * gigabyte, pool_headroom: 80_000_000.0 };

        let why = guard
            .no_room_for_a_copy(4.0 * gigabyte, 1_000_000.0)
            .expect("a second 4 GB body does not fit under a 6 GB ceiling");
        assert!(why.contains("4.0 GB of memory"), "what it needs is not in the message: {why}");
        assert!(why.contains("2.0 GB left"), "what is left is not in the message: {why}");
        assert!(why.contains("6 GB ceiling"), "the ceiling itself is not in the message: {why}");
        assert!(!why.contains('%'), "a copy has no size lever, so no size may be offered: {why}");
    }

    /// The same arithmetic as [`GrowthGuard::no_room_for`] and only the
    /// suggestion dropped, checked at the boundary in both directions so that
    /// the two cannot drift into disagreeing about whether a body fits.
    #[test]
    fn the_two_refusals_admit_and_refuse_exactly_the_same_bodies() {
        let guard = GrowthGuard { resident_bytes: MAX_VOLUME_BYTES / 2.0, pool_headroom: 4_000.0 };
        for bytes in [0.0, 1.0, MAX_VOLUME_BYTES / 2.0 - 1.0, MAX_VOLUME_BYTES] {
            for vertices in [0.0, 1.0, 4_000.0, 4_001.0] {
                assert_eq!(
                    guard.no_room_for(bytes, vertices).is_none(),
                    guard.no_room_for_a_copy(bytes, vertices).is_none(),
                    "the two disagreed about {bytes} bytes and {vertices} vertices"
                );
            }
        }
    }

    /// A copy goes directly below the row it came from, not at the bottom of a
    /// list of sixty-four, and it gets an id of its own.
    #[test]
    fn a_body_inserted_at_a_position_lands_there_with_a_fresh_id() {
        let mut doc = Document::new(0.5);
        let first = doc.active();
        let last = doc.add_body("Last", Volume::new(0.5));

        let between = doc.insert_body(1, 0, "Between", Volume::new(0.5));
        assert_eq!(
            doc.nodes().iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![first, between, last],
            "the row did not land between the two it was asked for"
        );
        assert_ne!(between, first);
        assert_ne!(between, last);
        assert_eq!(doc.index_of(between), Some(1));
        // Which body edits land on is the caller's business: inserting a row
        // must not move the selection out from under a gesture.
        assert_eq!(doc.active(), first, "inserting a row changed the active body");
    }

    /// Duplicating a folder's ONLY child, which is the case that shows nothing
    /// and reports success: no assertion can fire, because a body at depth 0
    /// after a body at depth 1 is a perfectly good forest -- it is simply a
    /// different one, with the copy at the top level and the folder it was
    /// copied inside left holding one row.
    #[test]
    fn a_copy_of_a_folders_only_child_stays_inside_that_folder() {
        let (mut doc, first, _, _) = three();
        let (folder, _) = doc.group(first, "Group 1").expect("the folder");
        let at = doc.index_of(first).expect("the child");

        let copy = doc.insert_body(
            at + 1,
            doc.node(first).expect("the child").depth(),
            "Copy",
            Volume::new(0.5),
        );

        assert_eq!(doc.parent_of(copy), Some(folder), "the copy left the folder it was made in");
        let range = doc.subtree_of(folder).expect("the folder is still here");
        assert_eq!(range.len(), 3, "the folder no longer holds both the body and its copy");
    }

    /// Duplicating a child that has a SIBLING below it, which is the case that
    /// does fire: a depth-0 copy in the middle of the run leaves the sibling at
    /// depth 1 following a body at depth 0, so the fold every other operation
    /// rests on no longer holds -- and in a release build, with the assertion
    /// gone, the sibling silently leaves the folder instead.
    #[test]
    fn a_copy_made_beside_a_sibling_leaves_that_sibling_where_it_was() {
        let (mut doc, first, second, _) = three();
        let (folder, _) = doc.group(first, "Group 1").expect("the folder");
        doc.move_to_folder(second, Some(folder)).expect("the sibling in");
        assert_eq!(depths(&doc), vec![0, 1, 1, 0], "the fixture is not the shape under test");
        let at = doc.index_of(first).expect("the child");

        let copy = doc.insert_body(
            at + 1,
            doc.node(first).expect("the child").depth(),
            "Copy",
            Volume::new(0.5),
        );

        assert_eq!(depths(&doc), vec![0, 1, 1, 1, 0]);
        assert_eq!(doc.parent_of(copy), Some(folder), "the copy left the folder");
        assert_eq!(doc.parent_of(second), Some(folder), "the sibling fell out of the folder");
        doc.assert_invariants();
    }

    /// A depth no position could hold is clamped rather than trusted, exactly
    /// as the reader clamps a file's depth column: a caller asking for depth 3
    /// directly under a top-level BODY gets depth 0, because a body is not a
    /// parent, and one asking for depth 0 directly above a row at depth 1 gets
    /// depth 1, because that row would otherwise follow a body two levels up.
    #[test]
    fn an_insert_depth_no_position_could_hold_is_clamped_to_one_that_can() {
        let (mut doc, first, second, _) = three();
        let deep = doc.insert_body(1, 3, "Too deep", Volume::new(0.5));
        assert_eq!(doc.node(deep).expect("the row").depth(), 0, "a body is not a parent");

        let (folder, _) = doc.group(second, "Group 1").expect("the folder");
        let at = doc.index_of(folder).expect("the folder") + 1;
        let shallow = doc.insert_body(at, 0, "Too shallow", Volume::new(0.5));
        assert_eq!(
            doc.node(shallow).expect("the row").depth(),
            1,
            "the row it was inserted above is at depth 1, so it cannot be shallower"
        );
        assert_eq!(doc.parent_of(shallow), Some(folder));
        assert!(doc.node(first).is_some());
        doc.assert_invariants();
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
