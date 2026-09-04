// SPDX-License-Identifier: AGPL-3.0-only

//! A shared vertex and index buffer that bricks own slices of.
//!
//! Each brick's mesh lives in a suballocation of one large buffer, so a remesh
//! writes into that slice instead of creating a buffer. Nothing here allocates
//! GPU memory after startup: the two big buffers are made once and the
//! suballocator hands out and reclaims ranges inside them.

use std::sync::atomic::{AtomicUsize, Ordering};

pub use brokkr_core::NodeId;
use brokkr_core::{BrickCoord, BrickMesh, Vertex};
use glam::Vec3;
use rustc_hash::FxHashMap;

use crate::frustum::Frustum;

/// Vertices ONE buffer of the pool holds. At 24 bytes each this is 264 MB.
///
/// **The binding constraint is `wgpu`'s default `max_buffer_size`, which is
/// 256 MiB (268,435,456 bytes)** -- not VRAM, and not a measurement. Iced
/// creates the device (`iced_wgpu::window::compositor` asks for
/// `wgpu::Limits::default()` with no hook to change it), so this crate cannot
/// request a larger limit however much the adapter would allow.
///
/// The answer is more buffers rather than a bigger one: see [`MAX_BUFFERS`].
/// 11,000,000 x 24 = 264,000,000 bytes, as close to the ceiling as a round
/// number gets.
pub const VERTEX_CAPACITY: u64 = 11_000_000;

/// Indices one buffer holds. At 4 bytes each this is 264 MB, under the same
/// ceiling.
///
/// Six times the vertex capacity, which is what a closed surface produces: the
/// dragon reserves 51.9 million indices against 8.65 million vertices.
pub const INDEX_CAPACITY: u64 = 66_000_000;

/// Bytes of per-vertex attributes, in the pool's THIRD buffer.
///
/// **Four, and it is one byte of mask plus three that are not spent yet.**
/// `Unorm8x4` is the narrowest format that fits: a vertex buffer's
/// `array_stride` must be a multiple of 4 and wgpu rejects anything else at
/// pipeline creation, so a one-byte attribute would cost the same four bytes
/// with nothing to show for it. Byte 0 is the mask, byte 1 is reserved for the
/// painted filament slot and byte 2 for the imported one, byte 3 spare -- which
/// is why colour, when it lands, costs the pool no capacity at all.
///
/// A buffer of its own rather than three more bytes on [`Vertex`], so `Vertex`
/// stays 24 bytes and [`VERTEX_CAPACITY`] does not move. 11,000,000 x 4 =
/// 44,000,000 B per pair, 16.4% of [`MAX_BUFFER_BYTES`].
const ATTRIBUTE_BYTES: u64 = 4;

/// How many buffer pairs the pool may grow to.
///
/// Buffers are created **on demand**, so a small model pays for one pair
/// (572 MB) and only a model that needs more allocates more. Eight pairs is
/// 4.58 GB of VRAM at full stretch, which is the point at which refusing is
/// kinder than trying on any card this application is likely to meet.
///
/// A "pair" is three buffers since the mask landed -- 264 MB of vertices,
/// 264 MB of indices and 44 MB of attributes -- and the name is kept because
/// what it means is "the set of buffers one brick's mesh must land in
/// together".
///
/// This is what lifts the ceiling that one buffer imposes. The dragon at
/// 200 mm and 0.113 mm fills 79% of a single pair; halving the voxel from
/// there needs four.
pub const MAX_BUFFERS: usize = 8;

/// Total vertices the pool can reach, across every buffer it may create.
pub const TOTAL_VERTEX_CAPACITY: u64 = VERTEX_CAPACITY * MAX_BUFFERS as u64;
/// Total indices the pool can reach.
pub const TOTAL_INDEX_CAPACITY: u64 = INDEX_CAPACITY * MAX_BUFFERS as u64;

/// `wgpu`'s default `max_buffer_size`, which neither pool buffer may exceed.
///
/// Not a number this crate chooses -- it is the guaranteed floor of the limit
/// every adapter reports, and iced creates the device, so it is what we get.
const MAX_BUFFER_BYTES: u64 = 268_435_456;

// Exceeding it is a validation error inside `Device::create_buffer` -- which is
// to say the application dies on startup, on every machine, in a way no unit
// test that avoids the GPU would catch. Raising a capacity without doing the
// arithmetic is an easy mistake and this is what makes it a compile error
// instead.
const _: () = assert!(
    VERTEX_CAPACITY * size_of::<Vertex>() as u64 <= MAX_BUFFER_BYTES,
    "the vertex buffer would exceed wgpu's default max_buffer_size"
);
const _: () = assert!(
    INDEX_CAPACITY * size_of::<u32>() as u64 <= MAX_BUFFER_BYTES,
    "the index buffer would exceed wgpu's default max_buffer_size"
);
const _: () = assert!(
    VERTEX_CAPACITY * ATTRIBUTE_BYTES <= MAX_BUFFER_BYTES,
    "the attribute buffer would exceed wgpu's default max_buffer_size"
);

// And the mask stays in that buffer rather than on the vertex. Four more bytes
// on `Vertex` would be 28, which is 308 MB at this capacity -- over the ceiling
// -- so `VERTEX_CAPACITY` would have to come DOWN to about 9.5 million, and
// every model between there and 11 million would start needing a second pair.
// A third buffer costs 44 MB and moves nothing.
const _: () = assert!(
    size_of::<Vertex>() == 24,
    "Vertex must stay 24 bytes: per-vertex data belongs in the attribute buffer"
);

/// Allocation granularity, in elements.
///
/// This replaced power of two size classes, which wasted most of the pool. An
/// average brick meshes to around 1100 vertices, and a power of two scheme
/// rounds that up to 2048: at M2 scale the reserved space came to nearly twice
/// what was in use and the pool overflowed on a model it had room for. Rounding
/// to a multiple of this costs about a tenth instead.
///
/// Small enough to keep the waste down, large enough that the free lists stay
/// short and a brick meshing to three vertices does not get its own bucket.
const GRANULARITY: u64 = 256;

/// The id `brokkr_core::Document::from_volume` gives the body it builds.
///
/// The application no longer names this: it passes `doc.active()`, which is a
/// real id out of a real node table. What is left are the callers on this side
/// of the line -- the render tests and the render bench -- which draw bricks
/// without a `Document` to ask, and want a name rather than a bare `NodeId(1)`
/// repeated down the file.
///
/// Nonzero because the node table reserves id zero for "no node".
pub const THE_ONLY_BODY: NodeId = NodeId(1);

/// Which brick of which body a pool slot holds.
///
/// **The body half is not decoration.** Every `Volume` sits on the same
/// lattice -- voxel (0,0,0) is world (0,0,0) in all of them, there is no origin
/// held anywhere -- so two bodies near the world origin share brick
/// coordinates. That is the normal case, not a corner one. Keyed on the
/// coordinate alone, body B's upload reuses the slice body A was drawing from,
/// or, if B's brick meshes to nothing, hands A's block back to the free list
/// outright. No log, no counter, `overflowed` stays zero: the geometry simply
/// leaves the screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotKey {
    pub body: NodeId,
    pub coord: BrickCoord,
}

/// A range inside one of the big buffers, measured in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    offset: u64,
    capacity: u64,
    /// Which buffer pair of the pool this lives in.
    buffer: u16,
}

/// Fixed granularity suballocator over a fixed range.
///
/// Blocks are never split or merged, so a freed block can only be reused by a
/// request that rounds to the same size. Brick meshes cluster tightly in size
/// AT ONE VOXEL SIZE, which is what makes that acceptable -- and it stops being
/// true the moment the voxel size changes.
///
/// # Why [`BlockAllocator::reset`] exists
///
/// Resampling changes every brick's mesh size at once, so the free lists fill
/// with granule classes nothing will ask for again while `bump` climbs toward
/// the ceiling. Going up and down the detail buttons a few times therefore
/// exhausts the pool with most of it free: observed 2026-08-22 on the dragon,
/// `MESH POOL FULL: 2755 bricks missing` while `live` was around 7.4M of 11M.
///
/// It is not worth a splitting-and-coalescing allocator, because the moment
/// fragmentation appears is also the moment nothing in the pool is worth
/// keeping: a resample rebuilds every brick. Reset is exact and free.
#[derive(Debug)]
struct BlockAllocator {
    capacity: u64,
    bump: u64,
    /// Free block offsets, indexed by how many granules they hold.
    free: Vec<Vec<u64>>,
    live: u64,
    /// Which buffer pair this allocator hands out space in.
    buffer: u16,
}

impl BlockAllocator {
    fn new(capacity: u64, buffer: u16) -> Self {
        Self { capacity, bump: 0, free: Vec::new(), live: 0, buffer }
    }

    /// Granules needed for a request, never zero.
    fn granules(count: u64) -> usize {
        count.max(1).div_ceil(GRANULARITY) as usize
    }

    fn allocate(&mut self, count: u64) -> Option<Block> {
        let granules = Self::granules(count);
        let capacity = granules as u64 * GRANULARITY;

        if self.free.len() <= granules {
            self.free.resize(granules + 1, Vec::new());
        }
        if let Some(offset) = self.free[granules].pop() {
            self.live += capacity;
            return Some(Block { offset, capacity, buffer: self.buffer });
        }

        if self.bump + capacity > self.capacity {
            return None;
        }
        let offset = self.bump;
        self.bump += capacity;
        self.live += capacity;
        Some(Block { offset, capacity, buffer: self.buffer })
    }

    fn release(&mut self, block: Block) {
        let granules = (block.capacity / GRANULARITY) as usize;
        if self.free.len() <= granules {
            self.free.resize(granules + 1, Vec::new());
        }
        self.free[granules].push(block.offset);
        self.live -= block.capacity;
    }

    /// Elements handed out, including the padding inside each block.
    fn live(&self) -> u64 {
        self.live
    }

    /// The high-water mark: how far the bump pointer has run. **This, not
    /// [`Self::live`], is what runs the pool out of room** -- an allocation
    /// fails when `bump + capacity` passes the ceiling, however much has been
    /// freed behind it.
    fn watermark(&self) -> u64 {
        self.bump
    }

    /// Forget every block and start again from zero.
    ///
    /// Only sound when the caller is discarding every slot in the same breath;
    /// [`MeshPool::reset`] is the one caller and it does exactly that.
    fn reset(&mut self) {
        self.bump = 0;
        self.live = 0;
        for class in &mut self.free {
            class.clear();
        }
    }
}

/// One vertex buffer and its index buffer, with the allocators over them.
///
/// Three buffers rather than two since the mask landed. The attributes are
/// suballocated at the SAME block offsets as the vertices -- they are one byte
/// per vertex by construction -- so they need no allocator of their own and
/// cannot drift out of step with one.
#[derive(Debug)]
struct BufferPair {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// One `Unorm8x4` per vertex: the mask in byte 0, and three bytes reserved.
    /// See [`ATTRIBUTE_BYTES`].
    attributes: wgpu::Buffer,
    vertex_allocator: BlockAllocator,
    index_allocator: BlockAllocator,
}

impl BufferPair {
    /// The capacities are passed in rather than read from [`VERTEX_CAPACITY`]
    /// and [`INDEX_CAPACITY`] so that a test can build a pool small enough to
    /// FILL. The real one is 4.58 GB at full stretch, which is not a thing a
    /// test can exhaust, and what happens at the ceiling is exactly the
    /// behaviour worth pinning.
    fn new(device: &wgpu::Device, index: u16, vertices: u64, indices: u64) -> Self {
        Self {
            vertices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brokkr brick vertices"),
                size: vertices * size_of::<Vertex>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            indices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brokkr brick indices"),
                size: indices * size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            attributes: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brokkr brick attributes"),
                size: vertices * ATTRIBUTE_BYTES,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            vertex_allocator: BlockAllocator::new(vertices, index),
            index_allocator: BlockAllocator::new(indices, index),
        }
    }
}

/// Where one brick's mesh lives in the pool.
#[derive(Debug, Clone, Copy)]
struct Slot {
    vertices: Block,
    indices: Block,
    vertex_count: u32,
    index_count: u32,
    /// World space bounds of this brick's geometry, for culling. Taken from the
    /// mesh rather than from the brick's extent, because a brick is usually only
    /// clipped by the surface in a thin band and the tighter box culls better.
    minimum: Vec3,
    maximum: Vec3,
}

/// Bounds of a set of vertices, or `None` if there are none.
fn bounds(vertices: &[Vertex]) -> Option<(Vec3, Vec3)> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for vertex in vertices {
        let position = Vec3::from_array(vertex.position);
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }
    (vertices.is_empty()).then_some(()).map_or(Some((minimum, maximum)), |_| None)
}

/// What the pool is currently holding, for the debug overlay.
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolStats {
    pub bricks: usize,
    pub triangles: usize,
    /// Vertices and indices actually in use.
    pub vertices: usize,
    pub indices: usize,
    /// Space handed out for them, which includes the padding inside each block.
    /// The gap between this and the counts above is the allocator's waste, and
    /// it is what runs the pool out of room, so both are worth showing.
    pub vertices_reserved: u64,
    pub indices_reserved: u64,
    /// How far the bump pointer has run. **The number that decides whether the
    /// next allocation fits**, and it can be far above `*_reserved` once the
    /// free lists hold granule classes nothing asks for -- which is what a
    /// resample leaves behind. A prediction made against `*_reserved` will say
    /// a model fits and then watch it overflow; predict against these.
    pub vertices_watermark: u64,
    pub indices_watermark: u64,
    pub vertex_capacity: u64,
    pub index_capacity: u64,
    /// Bricks skipped because the pool was full. Any value above zero means
    /// the model on screen is incomplete.
    pub overflowed: usize,
    /// Bricks drawn and bricks culled on the last frame.
    pub drawn: usize,
    pub culled: usize,
    /// Bricks skipped on the last frame because the body holding them is not
    /// visible.
    ///
    /// A third category rather than more `culled`, because the two are
    /// different answers to different questions: culled means "off screen this
    /// frame", hidden means "the user turned the eye off". Reported separately
    /// so that `drawn + culled + hidden` accounts for every brick in the pool
    /// -- a readout where they did not add up would look like the leak this
    /// pool has already shipped twice.
    pub hidden: usize,
}

/// The two bind groups [`MeshPool::draw`] chooses between, one body at a time.
///
/// **They differ in exactly one word -- [`crate::Uniforms::mask_inverted`] --
/// and in nothing else.** The mask is per BODY and a uniform is per draw, so a
/// document where one body has been inverted and another has not cannot be
/// drawn correctly from a single group. Increment 24 shipped Invert and Mask
/// All and made that reachable; before this, every body was drawn through the
/// ACTIVE body's polarity, which drew an unmasked body fully tinted and, in the
/// direction that matters, drew a fully PROTECTED body with no tint at all.
///
/// A pair of borrows in a named struct rather than two arguments, because two
/// arguments of the same type sitting next to each other can be swapped
/// silently and the result is the bug this exists to fix, inverted.
#[derive(Debug, Clone, Copy)]
pub struct MaskPolarity<'a> {
    /// Bound for every body except those named by
    /// [`MeshPool::set_opposite_polarity`]. Carries the uniforms exactly as the
    /// caller wrote them.
    pub as_published: &'a wgpu::BindGroup,
    /// The same uniforms with `mask_inverted` complemented.
    pub opposite: &'a wgpu::BindGroup,
}

/// The shared mesh buffers and the map from brick to slice.
#[derive(Debug)]
pub struct MeshPool {
    /// One pair per buffer of the pool, created on demand. Never shrinks: the
    /// buffers are reused by [`MeshPool::reset`] rather than dropped, because
    /// a rebuild almost always needs them again immediately.
    buffers: Vec<BufferPair>,
    /// Slots bucketed by the buffer pair they live in AND the body they belong
    /// to, rather than held in one flat map.
    ///
    /// `draw` has to bind a buffer pair before it can draw out of it, so with a
    /// flat map it walked EVERY slot once per pair and skipped the ones that
    /// did not belong -- O(pairs x slots). At the scale this pool is built for
    /// that is real money: a [`Slot`] is 80 bytes and a map entry about 103
    /// once the key and hashbrown's load factor are counted, so 45,567 slots is
    /// around 4.7 MB and eight passes over it is around 38 MB of memory traffic
    /// per frame before a single triangle is drawn. Bucketing visits each slot
    /// exactly once.
    ///
    /// The cost of the change is one extra pair of buffer binds per body per
    /// pair, since a pair holding two bodies is now bound twice. That is two
    /// commands against a whole extra walk of the slot map, and with one body
    /// -- which is every model until increment 2 -- it is exactly what it was.
    ///
    /// The body half of the key is also what makes [`MeshPool::draw_body`] a
    /// lookup rather than a rescan of every slot in the pool.
    buckets: FxHashMap<(u16, NodeId), FxHashMap<BrickCoord, Slot>>,
    /// Vertices and indices each buffer pair of this pool holds. Fields rather
    /// than the constants, so a test can build a pool it can fill. See
    /// [`MeshPool::with_capacities`].
    vertex_capacity: u64,
    index_capacity: u64,
    triangles: usize,
    vertices: usize,
    overflowed: usize,
    warned_about_overflow: bool,
    /// The bodies that must not be drawn, replaced wholesale by
    /// [`MeshPool::set_hidden`].
    ///
    /// **Bodies, not a bitmask, and the difference is not cosmetic.** The
    /// design this was built from called for a `hidden: u64` on the frame
    /// channel, indexed by a body's position in the document, with that
    /// position stored on each [`Slot`]. That does not survive a delete: the
    /// bodies after the deleted one all shift down by one, so every slot they
    /// own is left naming the wrong bit -- and the pool cannot repair it,
    /// because a body that had no bricks in the pool leaves no record of the
    /// position it used to occupy, and the shift is invisible from here. A
    /// pool keyed on [`NodeId`] (which is what the buckets already are, since
    /// increment 1) needs no position at all: an id is stable for the life of
    /// the body, so nothing here can drift out of step with the document.
    ///
    /// Held as the HIDDEN set rather than the shown one because it is almost
    /// always empty, and an empty `Vec` makes the per-bucket test free.
    hidden: Vec<NodeId>,
    /// The bodies drawn with the OPPOSITE of [`crate::Uniforms::mask_inverted`],
    /// replaced wholesale by [`MeshPool::set_opposite_polarity`].
    ///
    /// **A mask is per body and a uniform is per draw, and this is how the two
    /// are reconciled.** Increment 24 shipped Invert and Mask All, which made a
    /// document whose bodies disagree about polarity reachable; the application
    /// publishes the ACTIVE body's polarity in the uniform, so without this
    /// every other body was drawn through it -- a Mask All on the head drew an
    /// unmasked torso fully tinted, and, worse, a Mask All on a body that was
    /// not active drew that body's stored zeros as free. A fully protected body
    /// with no tint on it is the failure the whole masking design is arranged
    /// around.
    ///
    /// Held as the set that DIFFERS rather than the set that is inverted, for
    /// the reason `hidden` is held as the hidden set: in a document where every
    /// body agrees -- which is every document that has not used the verbs -- it
    /// is empty, and an empty `Vec` makes the per-bucket test free. It is
    /// bodies rather than a bitmask for the reason `hidden` gives in full.
    opposite: Vec<NodeId>,
    /// Counts from the last draw. Atomic because drawing takes a shared borrow
    /// and Iced requires the pipeline it owns to be `Sync`.
    drawn: AtomicUsize,
    culled: AtomicUsize,
    skipped_as_hidden: AtomicUsize,
    /// One brick's attributes, widened from one byte per vertex to four, on
    /// their way to the GPU.
    ///
    /// **Kept here so that `upload` allocates nothing**, which it must not: it
    /// runs inside `prepare`, on the frame path. `BrickMesh::mask` is one byte
    /// per vertex -- `brokkr-core` has no business knowing what the other three
    /// are for -- and the buffer is four, so the widening has to happen
    /// somewhere; it happens once into a vector that keeps its capacity, which
    /// after the first few bricks is every brick.
    staging: Vec<[u8; ATTRIBUTE_BYTES as usize]>,
}

impl MeshPool {
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_capacities(device, VERTEX_CAPACITY, INDEX_CAPACITY)
    }

    /// A pool whose buffer pairs hold the given number of elements each.
    ///
    /// Private, and there are exactly two callers: [`MeshPool::new`] with the
    /// real capacities, and the tests with capacities small enough that the
    /// pool can actually be run out of. Overflow behaviour is the part of this
    /// file that has put holes in a real model, and a ceiling of
    /// [`TOTAL_VERTEX_CAPACITY`] vertices across 4.58 GB of buffers is not one
    /// a test can reach.
    fn with_capacities(device: &wgpu::Device, vertices: u64, indices: u64) -> Self {
        Self {
            buffers: vec![BufferPair::new(device, 0, vertices, indices)],
            buckets: FxHashMap::default(),
            vertex_capacity: vertices,
            index_capacity: indices,
            triangles: 0,
            vertices: 0,
            overflowed: 0,
            warned_about_overflow: false,
            // Never allocated again: a document holds at most `MAX_BODIES`
            // bodies, so neither of these sets can outgrow this and both
            // `set_hidden` and `set_opposite_polarity` run on the frame path.
            hidden: Vec::with_capacity(brokkr_core::MAX_BODIES),
            opposite: Vec::with_capacity(brokkr_core::MAX_BODIES),
            drawn: AtomicUsize::new(0),
            culled: AtomicUsize::new(0),
            skipped_as_hidden: AtomicUsize::new(0),
            staging: Vec::new(),
        }
    }

    /// Reserve space for one brick, growing the pool by a buffer if the ones
    /// that exist have no room.
    ///
    /// Vertices and indices must land in the SAME pair: `draw` binds a pair at
    /// a time, so a brick split across two could not be drawn. When a pair has
    /// room for one and not the other the whole request moves on to the next.
    fn reserve(
        &mut self,
        device: &wgpu::Device,
        need_vertices: u64,
        need_indices: u64,
    ) -> Option<(Block, Block)> {
        for pair in &mut self.buffers {
            // Ask for the vertices first, and hand them straight back if the
            // indices do not also fit -- otherwise a near-full pair leaks a
            // block every time it is passed over.
            let Some(vertices) = pair.vertex_allocator.allocate(need_vertices) else {
                continue;
            };
            match pair.index_allocator.allocate(need_indices) {
                Some(indices) => return Some((vertices, indices)),
                None => pair.vertex_allocator.release(vertices),
            }
        }

        if self.buffers.len() >= MAX_BUFFERS {
            return None;
        }
        let index = self.buffers.len() as u16;
        log::info!(
            "mesh pool growing to {} buffer pairs ({} MB reserved)",
            index + 1,
            (index as u64 + 1)
                * (self.vertex_capacity * size_of::<Vertex>() as u64
                    + self.index_capacity * size_of::<u32>() as u64
                    + self.vertex_capacity * ATTRIBUTE_BYTES)
                / (1024 * 1024)
        );
        self.buffers.push(BufferPair::new(
            device,
            index,
            self.vertex_capacity,
            self.index_capacity,
        ));
        let pair = self.buffers.last_mut()?;
        Some((
            pair.vertex_allocator.allocate(need_vertices)?,
            pair.index_allocator.allocate(need_indices)?,
        ))
    }

    /// The slot a key currently occupies, and which buffer pair it is in.
    ///
    /// A brick's pair is not known before the lookup -- `reserve` puts it in
    /// whichever pair had room for both halves -- so this asks each pair in
    /// turn. With one pair, which is every model that fits in 264 MB, that is
    /// the single hash lookup the flat map used to do. Keeping a separate
    /// key-to-pair index to save the other seven would be a second map to hold
    /// in step with this one, and this runs per dirty brick rather than per
    /// frame.
    fn find(&self, key: SlotKey) -> Option<(u16, Slot)> {
        (0..self.buffers.len() as u16).find_map(|pair| {
            self.buckets
                .get(&(pair, key.body))
                .and_then(|bucket| bucket.get(&key.coord))
                .map(|slot| (pair, *slot))
        })
    }

    /// Drop a slot entirely: out of its bucket, its space back to the pair it
    /// came from, and its mesh off the counters.
    ///
    /// All three always happen together, which is why they are one function.
    /// They used to be spread across [`MeshPool::upload`], and the counters
    /// ran ahead of the slot -- see the note there.
    fn forget(&mut self, pair: u16, key: SlotKey, slot: Slot) {
        if let Some(bucket) = self.buckets.get_mut(&(pair, key.body)) {
            bucket.remove(&key.coord);
            // An empty bucket is dropped rather than kept, so `draw` iterates
            // over exactly the buckets that have something in them and the
            // brick count in `stats` needs no separate tally.
            if bucket.is_empty() {
                self.buckets.remove(&(pair, key.body));
            }
        }
        self.release(slot);
        self.uncount(slot);
    }

    /// Take a slot's mesh off the triangle and vertex counters.
    fn uncount(&mut self, slot: Slot) {
        self.triangles -= slot.index_count as usize / 3;
        self.vertices -= slot.vertex_count as usize;
    }

    /// Put one brick's mesh in the pool, replacing whatever that key held.
    ///
    /// **Space for the new mesh is reserved BEFORE the old slice is handed
    /// back.** The other order reads as harmless and is not: it released the
    /// block and removed the slot, and only THEN returned early if `reserve`
    /// failed. So near a full pool a brick that GREW lost the space it already
    /// had and got nothing in return -- and the brick that grows is the one
    /// under the brush, so what the user saw was a permanent hole opening in
    /// the part of the model they were actively sculpting, with nothing
    /// putting it back until the whole model was rebuilt. Reserving first
    /// costs one block of headroom for the length of this call and turns that
    /// into "the previous mesh keeps drawing", which is one remesh out of date
    /// and otherwise correct.
    ///
    /// **The triangle and vertex counters move with the slot for the same
    /// reason.** They used to be decremented at the top, before anything could
    /// fail, so a refused upload left them counting a slot that was still live
    /// and still on screen.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: SlotKey,
        mesh: &BrickMesh,
    ) {
        let existing = self.find(key);

        // Nothing to draw any more: the surface has left this brick.
        if mesh.indices.is_empty() {
            if let Some((pair, slot)) = existing {
                self.forget(pair, key, slot);
            }
            return;
        }

        let need_vertices = mesh.vertices.len() as u64;
        let need_indices = mesh.indices.len() as u64;

        // Keep the existing slices when the new mesh still fits them, which is
        // the common case during a stroke and avoids touching the free lists.
        let fits = existing.filter(|(_, slot)| {
            slot.vertices.capacity >= need_vertices && slot.indices.capacity >= need_indices
        });

        let slot = match fits {
            Some((_, slot)) => {
                // The slice survives, but the mesh in it does not, so its
                // counts come off here and the new ones go on below.
                self.uncount(slot);
                slot
            }
            None => {
                let Some((vertices, indices)) = self.reserve(device, need_vertices, need_indices)
                else {
                    self.overflowed += 1;
                    if !self.warned_about_overflow {
                        self.warned_about_overflow = true;
                        // The remedy is named because the obvious one does not
                        // work: hiding a body is a draw-time skip and it keeps
                        // every slice it holds, so the eye frees nothing at
                        // all. Only deleting a body or resampling the document
                        // coarser gives the pool anything back.
                        log::error!(
                            "mesh pool is full at {MAX_BUFFERS} buffers ({} vertices, {} \
                             indices), so parts of the model are missing from the screen. \
                             Delete a body or resample coarser to free pool space; hiding a \
                             body does not, because a hidden body keeps its slices.",
                            self.vertex_capacity * MAX_BUFFERS as u64,
                            self.index_capacity * MAX_BUFFERS as u64,
                        );
                    }
                    // Whatever was here is deliberately left alone: a mesh one
                    // remesh out of date beats a hole in the model.
                    return;
                };
                if let Some((pair, previous)) = existing {
                    self.forget(pair, key, previous);
                }
                Slot {
                    vertices,
                    indices,
                    vertex_count: 0,
                    index_count: 0,
                    minimum: Vec3::ZERO,
                    maximum: Vec3::ZERO,
                }
            }
        };

        debug_assert_eq!(
            mesh.mask.len(),
            mesh.vertices.len(),
            "one mask byte per vertex, or the tail of this brick reads the previous tenant's \
             attributes"
        );
        debug_assert!(
            mesh.colour.is_empty() || mesh.colour.len() == mesh.vertices.len(),
            "colour must be one byte per vertex or absent entirely"
        );
        // Widened into the kept buffer before anything else is borrowed, so
        // that the three writes below can share `&self`. See `staging`.
        self.staging.clear();
        // Byte 0 the mask, byte 1 the painted slot -- the layout `ATTRIBUTE_BYTES`
        // reserved. Zipped rather than a second pass: `upload` runs inside
        // `prepare` on the frame path, and `staging` is a kept buffer precisely
        // so this allocates nothing.
        self.staging.extend(
            mesh.mask
                .iter()
                .zip(mesh.colour.iter().chain(std::iter::repeat(&0)))
                .map(|(mask, slot)| [*mask, *slot, 0, 0]),
        );

        let pair = &self.buffers[slot.vertices.buffer as usize];
        queue.write_buffer(
            &pair.vertices,
            slot.vertices.offset * size_of::<Vertex>() as u64,
            bytemuck::cast_slice(&mesh.vertices),
        );
        queue.write_buffer(
            &pair.indices,
            slot.indices.offset * size_of::<u32>() as u64,
            bytemuck::cast_slice(&mesh.indices),
        );
        // The same block offset as the vertices, in the same order: the
        // attribute buffer is suballocated by the vertex allocator and has none
        // of its own.
        queue.write_buffer(
            &pair.attributes,
            slot.vertices.offset * ATTRIBUTE_BYTES,
            bytemuck::cast_slice(&self.staging),
        );

        self.triangles += mesh.indices.len() / 3;
        self.vertices += mesh.vertices.len();
        let (minimum, maximum) = bounds(&mesh.vertices).unwrap_or((Vec3::ZERO, Vec3::ZERO));
        self.buckets.entry((slot.vertices.buffer, key.body)).or_default().insert(
            key.coord,
            Slot {
                vertex_count: mesh.vertices.len() as u32,
                index_count: mesh.indices.len() as u32,
                minimum,
                maximum,
                ..slot
            },
        );
    }

    /// Hand a slot's space back to the pair it came from.
    fn release(&mut self, slot: Slot) {
        let pair = &mut self.buffers[slot.vertices.buffer as usize];
        pair.vertex_allocator.release(slot.vertices);
        pair.index_allocator.release(slot.indices);
    }

    /// Record one indexed draw per visible brick.
    ///
    /// Indices are brick local, so the slice's vertex offset goes in as the base
    /// vertex. Bricks outside the frustum are skipped: at M2 scale a model is
    /// several thousand bricks and most of them are off screen at any moment, so
    /// not drawing those is the cheapest saving available.
    ///
    /// **Visibility is a draw-time skip and is never anything else.** A hidden
    /// body keeps every voxel, every slot and every byte of pool space it had;
    /// the only thing that changes is that this loop steps over its buckets.
    /// The alternative -- dropping its bricks, or marking them dirty and
    /// meshing them to nothing -- would make showing it again a whole-body
    /// remesh, and would make the eye a thing that can lose geometry. It also
    /// means hiding buys no pool headroom, which is why the overflow message
    /// above says so.
    ///
    /// # Polarity is per bucket, and that is what makes the mask per body
    ///
    /// A bucket is exactly one body, so the bind group is chosen here rather
    /// than once for the whole pass; see [`MaskPolarity`] and the `opposite`
    /// field. The group is bound only when it CHANGES, so a document where
    /// every body agrees -- which is every document that has not touched Invert
    /// or Mask All -- pays one `set_bind_group` for the whole draw, the same as
    /// before.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        frustum: &Frustum,
        polarity: MaskPolarity<'_>,
    ) {
        self.drawn.store(0, Ordering::Relaxed);
        self.culled.store(0, Ordering::Relaxed);
        self.skipped_as_hidden.store(0, Ordering::Relaxed);
        if self.buckets.is_empty() {
            return;
        }

        // One pass over each bucket, and a bucket is one buffer pair's slots
        // for one body -- so every slot is visited exactly once and the pair is
        // bound at most once per bucket. This used to walk all the slots once
        // per pair and skip the ones that did not belong; see the field's
        // documentation for what that cost.
        let mut drawn = 0;
        let mut culled = 0;
        let mut hidden = 0;
        // `None` until the first bucket that actually draws, so this loop binds
        // the polarity it wants rather than trusting whatever the caller left
        // bound. After that it rebinds only on a change: a document where every
        // body agrees -- which is every document that has not touched Invert or
        // Mask All -- binds exactly once, and a mixed one binds at worst once
        // per bucket, since the map's order is arbitrary and the two polarities
        // are not grouped. That ceiling is `pairs x bodies`, which is the same
        // order as the hidden test above.
        let mut bound_polarity: Option<bool> = None;
        for (&(index, body), bucket) in &self.buckets {
            // Per bucket rather than per slot: a bucket is exactly one body, so
            // the test runs at most `pairs x bodies` times a frame instead of
            // once per brick.
            if self.hidden.contains(&body) {
                hidden += bucket.len();
                continue;
            }
            let pair = &self.buffers[index as usize];
            let opposite = self.opposite.contains(&body);
            let mut bound = false;
            for slot in bucket.values() {
                if slot.index_count == 0 {
                    continue;
                }
                if !frustum.intersects(slot.minimum, slot.maximum) {
                    culled += 1;
                    continue;
                }
                if !bound {
                    // Inside the frustum test, so a body entirely off screen
                    // binds nothing at all -- including its polarity.
                    if bound_polarity != Some(opposite) {
                        let group =
                            if opposite { polarity.opposite } else { polarity.as_published };
                        pass.set_bind_group(0, group, &[]);
                        bound_polarity = Some(opposite);
                    }
                    pass.set_vertex_buffer(0, pair.vertices.slice(..));
                    pass.set_vertex_buffer(1, pair.attributes.slice(..));
                    pass.set_index_buffer(pair.indices.slice(..), wgpu::IndexFormat::Uint32);
                    bound = true;
                }
                drawn += 1;
                let start = slot.indices.offset as u32;
                pass.draw_indexed(
                    start..start + slot.index_count,
                    slot.vertices.offset as i32,
                    0..1,
                );
            }
        }
        self.drawn.store(drawn, Ordering::Relaxed);
        self.culled.store(culled, Ordering::Relaxed);
        self.skipped_as_hidden.store(hidden, Ordering::Relaxed);
    }

    /// Replace the set of bodies that must not be drawn.
    ///
    /// **Wholesale, every time, and never mutated one body at a time.** The
    /// set is a pure function of the document and the solo mode, and the one
    /// place it is worked out is the application's visibility pass; recomputing
    /// it here from an add or a remove would make two owners of one answer, and
    /// two owners is how the eye and the viewport come to disagree after an
    /// undo -- a body invisible on screen that still raycasts and still carves.
    ///
    /// Copies rather than takes the caller's buffer so that the caller can keep
    /// its own allocation, and never allocates: the vector is built with room
    /// for `MAX_BODIES` and a document cannot hold more than that.
    pub fn set_hidden(&mut self, hidden: &[NodeId]) {
        debug_assert!(
            hidden.len() <= brokkr_core::MAX_BODIES,
            "{} hidden bodies, which is more than a document can hold",
            hidden.len()
        );
        self.hidden.clear();
        self.hidden.extend_from_slice(hidden);
    }

    /// Replace the set of bodies drawn with the opposite of
    /// [`crate::Uniforms::mask_inverted`].
    ///
    /// **Wholesale, every time, for the reason [`MeshPool::set_hidden`] gives
    /// in full**: it is a pure function of the document and of which body is
    /// active, worked out in one place -- the application's visibility pass --
    /// and a second owner of it here is how the picture comes to disagree with
    /// the protection under it. Never allocates, for the same reason.
    ///
    /// It is relative to the uniform rather than absolute so that
    /// `Uniforms::mask_inverted` stays exactly what it says it is, and so that
    /// a document where every body agrees publishes an empty set.
    pub fn set_opposite_polarity(&mut self, opposite: &[NodeId]) {
        debug_assert!(
            opposite.len() <= brokkr_core::MAX_BODIES,
            "{} bodies of opposite polarity, which is more than a document can hold",
            opposite.len()
        );
        self.opposite.clear();
        self.opposite.extend_from_slice(opposite);
    }

    /// Drop every slot one body owns, and give its space back.
    ///
    /// Returns how many slots went, which is what a caller has to look at to
    /// tell "the body had no bricks in the pool" from "the body was not there".
    ///
    /// **The caller must mark the surviving bodies dirty and remesh after
    /// this**, exactly as [`MeshPool::reset`] requires, and for a sharper
    /// reason than that one. A brick the pool refused while it was full is
    /// simply dropped -- `upload`'s overflow arm returns after counting it, and
    /// the application has already drained that coordinate out of its dirty
    /// set, so nothing will offer the brick again until something else dirties
    /// it. Freeing space here does not bring those bricks back; only a remesh
    /// does. Skip the remesh and the user gets the worst outcome this file can
    /// produce: `overflowed` is cleared below, so the red MESH POOL FULL banner
    /// disappears while the geometry it was warning about is still missing from
    /// the screen, permanently and silently. That is strictly worse than the
    /// stale banner the clear exists to avoid.
    ///
    /// **Two pieces of bookkeeping here are not obvious and both have a
    /// symptom.** Any buffer pair this empties has its allocators reset, so the
    /// pair's bump pointer goes back to zero rather than staying wherever the
    /// deleted body left it -- without that, deleting a body frees its blocks
    /// into granule classes nothing may ask for again and the pair keeps its
    /// high-water mark, which is the number that actually decides whether the
    /// next allocation fits. And `overflowed` is cleared, because it and
    /// `warned_about_overflow` were previously cleared only by
    /// [`MeshPool::reset`]: the red banner would otherwise stay up, with a
    /// stale count, over a document that now fits comfortably. The clear is
    /// only sound because of the remesh above -- the remesh re-offers every
    /// brick, so any that still do not fit put the banner straight back with an
    /// honest count, which is precisely how `reset` gets away with it inside
    /// `rebuild_everything`.
    pub fn forget_body(&mut self, body: NodeId) -> usize {
        let mut forgotten = 0;
        for pair in 0..self.buffers.len() as u16 {
            let Some(bucket) = self.buckets.remove(&(pair, body)) else {
                continue;
            };
            for slot in bucket.into_values() {
                self.release(slot);
                self.uncount(slot);
                forgotten += 1;
            }
        }
        if forgotten == 0 {
            return 0;
        }

        // A pair with no buckets left has no live block in it, which is exactly
        // the precondition `BlockAllocator::reset` asks for.
        for (index, pair) in self.buffers.iter_mut().enumerate() {
            if self.buckets.keys().any(|&(held, _)| held == index as u16) {
                continue;
            }
            pair.vertex_allocator.reset();
            pair.index_allocator.reset();
        }

        self.overflowed = 0;
        self.warned_about_overflow = false;
        forgotten
    }

    /// How many of one body's bricks the pool is holding.
    ///
    /// The question "is this body really gone" has no other answer from
    /// outside: `PoolStats` counts the whole pool, so a body whose slots
    /// survived a delete would be invisible in it the moment a second body
    /// exists.
    pub fn body_bricks(&self, body: NodeId) -> usize {
        (0..self.buffers.len() as u16)
            .filter_map(|pair| self.buckets.get(&(pair, body)))
            .map(FxHashMap::len)
            .sum()
    }

    /// What the pool was last told not to draw.
    ///
    /// The same justification as [`MeshPool::body_bricks`]: the question has no
    /// other answer from outside. Hiding is a draw-time skip, so it moves no
    /// counter a caller can see -- `bricks`, `body_bricks` and every reserved
    /// and watermark number are invariant across it by design, and `hidden` in
    /// [`PoolStats`] is written by `draw`, so it stays zero until a frame is
    /// actually drawn. Without this, a test asserting that the hidden set
    /// reached the pool can only assert on the thing that SENT it, which is
    /// not the same claim: deleting the delivering call entirely left the
    /// whole workspace suite green when it was tried.
    pub fn hidden_bodies(&self) -> &[NodeId] {
        &self.hidden
    }

    /// What the pool was last told draws with the opposite polarity.
    ///
    /// The same justification as [`MeshPool::hidden_bodies`], word for word:
    /// polarity is a bind-group choice at draw time, so it moves no counter,
    /// and a test that asserted on the application's own field would be
    /// asserting on the thing that SENDS the answer rather than on the answer
    /// that arrived.
    pub fn opposite_polarity_bodies(&self) -> &[NodeId] {
        &self.opposite
    }

    /// Draw every brick of ONE body, with no culling at all.
    ///
    /// **There is deliberately no frustum parameter, and the absence is the
    /// point.** The caller this exists for is the per-body thumbnail pass,
    /// which frames the whole body by construction: every brick of it is
    /// inside the view volume, so a cull test can only ever be wrong, and
    /// skipping the AABB tests saves what they cost -- around 0.68 ms at the
    /// 45,567 bricks this pool is built for.
    ///
    /// It is a distinct method rather than an `Option<&Frustum>` on
    /// [`MeshPool::draw`] so that there is no parameter for anyone to pass a
    /// frustum into. The only frustum in scope where a thumbnail would be
    /// drained is the viewport's `pipeline.frustum` -- the USER'S camera,
    /// rebuilt every frame eight lines above the drain -- and it would
    /// typecheck, cull a thumbnail against a view it has nothing to do with,
    /// and hand back a picture with most of the body missing. A signature with
    /// no such parameter cannot be misused that way.
    ///
    /// It also must **not** write the `drawn` and `culled` counters. Those are
    /// published as the viewport's per-frame readout, and a thumbnail drawn
    /// between two sculpt frames would overwrite them with numbers about a
    /// picture nobody is looking at.
    ///
    /// No caller yet: increment 15 is the thumbnail pass. What exists today is
    /// the pool half of it, which is the half that had to land with the
    /// bucketing rather than after it.
    ///
    /// It also ignores [`MeshPool::set_hidden`], deliberately: the panel draws
    /// a muted row for a hidden body and still shows its picture, so a
    /// thumbnail pass that skipped hidden bodies would leave a blank square
    /// where the muting is the cue.
    pub fn draw_body(&self, pass: &mut wgpu::RenderPass<'_>, body: NodeId) {
        for index in 0..self.buffers.len() as u16 {
            let Some(bucket) = self.buckets.get(&(index, body)) else {
                continue;
            };
            let pair = &self.buffers[index as usize];
            let mut bound = false;
            for slot in bucket.values() {
                if slot.index_count == 0 {
                    continue;
                }
                if !bound {
                    pass.set_vertex_buffer(0, pair.vertices.slice(..));
                    pass.set_vertex_buffer(1, pair.attributes.slice(..));
                    pass.set_index_buffer(pair.indices.slice(..), wgpu::IndexFormat::Uint32);
                    bound = true;
                }
                let start = slot.indices.offset as u32;
                pass.draw_indexed(
                    start..start + slot.index_count,
                    slot.vertices.offset as i32,
                    0..1,
                );
            }
        }
    }

    /// Discard every brick and start the allocator over.
    ///
    /// For the moments that rebuild the WHOLE model -- a resample, an import,
    /// opening a file, re-orienting -- where every slot is about to be replaced
    /// anyway. Without it those are exactly the moments that fragment the pool
    /// beyond use: see [`BlockAllocator`].
    ///
    /// **The caller must remesh everything after this**, or the model is not on
    /// the GPU any more. `Volume::mark_everything_dirty` is the partner.
    pub fn reset(&mut self) {
        self.buckets.clear();
        for pair in &mut self.buffers {
            pair.vertex_allocator.reset();
            pair.index_allocator.reset();
        }
        self.triangles = 0;
        self.vertices = 0;
        self.overflowed = 0;
        self.warned_about_overflow = false;
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            bricks: self.buckets.values().map(FxHashMap::len).sum(),
            triangles: self.triangles,
            vertices: self.vertices,
            indices: self.triangles * 3,
            vertices_reserved: self.buffers.iter().map(|p| p.vertex_allocator.live()).sum(),
            indices_reserved: self.buffers.iter().map(|p| p.index_allocator.live()).sum(),
            vertices_watermark: self.buffers.iter().map(|p| p.vertex_allocator.watermark()).sum(),
            indices_watermark: self.buffers.iter().map(|p| p.index_allocator.watermark()).sum(),
            vertex_capacity: self.vertex_capacity * MAX_BUFFERS as u64,
            index_capacity: self.index_capacity * MAX_BUFFERS as u64,
            overflowed: self.overflowed,
            drawn: self.drawn.load(Ordering::Relaxed),
            culled: self.culled.load(Ordering::Relaxed),
            hidden: self.skipped_as_hidden.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_round_up_to_one_granule() {
        let mut allocator = BlockAllocator::new(1 << 20, 0);
        let block = allocator.allocate(3).expect("space is available");
        assert_eq!(block.capacity, GRANULARITY);
        assert_eq!(block.offset, 0);
    }

    #[test]
    fn the_padding_on_a_typical_brick_stays_small() {
        // The regression test for a real overflow. Power of two size classes
        // rounded an average brick's 1100 vertices up to 2048, so the pool
        // reserved nearly twice what was in use and ran out of room on a model
        // it had space for.
        let mut allocator = BlockAllocator::new(1 << 24, 0);
        for request in [900_u64, 1100, 1500, 2100, 3300] {
            let block = allocator.allocate(request).expect("space is available");
            let waste = block.capacity - request;
            assert!(
                waste < GRANULARITY,
                "a request for {request} reserved {} and wasted {waste}",
                block.capacity
            );
        }
    }

    #[test]
    fn freed_blocks_are_reused_rather_than_bumping() {
        let mut allocator = BlockAllocator::new(1 << 20, 0);
        let first = allocator.allocate(1000).expect("space is available");
        let second = allocator.allocate(1000).expect("space is available");
        assert_ne!(first.offset, second.offset);

        allocator.release(first);
        let third = allocator.allocate(1000).expect("the freed block is reusable");
        assert_eq!(third.offset, first.offset, "should have come off the free list");
    }

    /// A block always carries the buffer it came from, and vertices and
    /// indices for one brick always land in the SAME pair -- `draw` binds a
    /// pair at a time, so a brick split across two could not be drawn at all.
    #[test]
    fn a_brick_never_straddles_two_buffers() {
        // Two pairs, the first with room for the vertices but not the indices.
        let mut vertex_pools =
            [BlockAllocator::new(GRANULARITY * 8, 0), BlockAllocator::new(GRANULARITY * 8, 1)];
        let mut index_pools =
            [BlockAllocator::new(GRANULARITY, 0), BlockAllocator::new(GRANULARITY * 8, 1)];

        // Reserve mirrors MeshPool::reserve: take the vertices, and hand them
        // straight back if the indices do not also fit in the same pair.
        let mut chosen = None;
        for pair in 0..2 {
            let Some(vertices) = vertex_pools[pair].allocate(GRANULARITY * 2) else { continue };
            match index_pools[pair].allocate(GRANULARITY * 4) {
                Some(indices) => {
                    chosen = Some((vertices, indices));
                    break;
                }
                None => vertex_pools[pair].release(vertices),
            }
        }

        let (vertices, indices) = chosen.expect("the second pair has room for both");
        assert_eq!(vertices.buffer, indices.buffer, "a brick must live in one pair");
        assert_eq!(vertices.buffer, 1, "the first pair could not hold the indices");
        assert_eq!(
            vertex_pools[0].live(),
            0,
            "passing over a pair must not leak the vertices it had already taken"
        );
    }

    /// The failure that put holes in the dragon: going up and down the detail
    /// buttons exhausts the pool while most of it is free.
    ///
    /// Blocks are never split or merged, so a freed block only serves a
    /// request of the same granule count. A resample changes every brick's
    /// size at once, so the free lists fill with classes nothing asks for
    /// again and the bump pointer climbs regardless of how much was returned.
    #[test]
    fn alternating_block_sizes_exhaust_the_pool_while_most_of_it_is_free() {
        let mut allocator = BlockAllocator::new(GRANULARITY * 64, 0);

        // Two rounds of "coarse" then "fine", each freed before the next.
        let mut blocks: Vec<Block> = Vec::new();
        for round in 0..2 {
            let size = if round % 2 == 0 { GRANULARITY } else { GRANULARITY * 3 };
            for _ in 0..8 {
                blocks.push(allocator.allocate(size).expect("early rounds fit"));
            }
            for block in blocks.drain(..) {
                allocator.release(block);
            }
        }

        assert_eq!(allocator.live(), 0, "everything was returned");
        assert!(
            allocator.watermark() > 0,
            "yet the bump pointer has run on -- this is the leak the reset exists for"
        );

        // A size neither round used cannot reuse any of it.
        let fresh = GRANULARITY * 7;
        let before = allocator.watermark();
        allocator.allocate(fresh).expect("still room here");
        assert!(
            allocator.watermark() > before,
            "a new size class had to bump past everything freed"
        );

        // Reset makes the whole pool available again, which is the fix.
        allocator.reset();
        assert_eq!(allocator.watermark(), 0);
        assert_eq!(allocator.live(), 0);
        assert!(
            allocator.allocate(GRANULARITY * 64).is_some(),
            "after a reset the entire pool is one contiguous run again"
        );
    }

    #[test]
    fn allocation_fails_rather_than_overrunning_the_buffer() {
        let mut allocator = BlockAllocator::new(1 << 10, 0);
        assert!(allocator.allocate(1 << 10).is_some());
        assert!(allocator.allocate(1).is_none(), "must refuse to hand out space it does not have");
    }

    #[test]
    fn live_count_returns_to_zero_after_releasing_everything() {
        let mut allocator = BlockAllocator::new(1 << 20, 0);
        let blocks: Vec<_> = (0..16).map(|n| allocator.allocate(n * 37 + 1).unwrap()).collect();
        assert!(allocator.live() > 0);
        for block in blocks {
            allocator.release(block);
        }
        assert_eq!(allocator.live(), 0);
    }

    /// The import preflight refuses a model whose estimated vertices exceed
    /// what the pool can hold, and it keeps its own copy of that number because
    /// `brokkr-core` may not depend on this crate -- CI fails the build if it
    /// ever does.
    ///
    /// A copy with nothing enforcing it is a constant waiting to drift. When it
    /// drifts high the pool overflows and the model silently loses bricks off
    /// the screen; when it drifts low, imports are refused that would have been
    /// fine. Both have happened here in other guises, which is why this exists
    /// rather than a comment asking the next person to remember.
    #[test]
    fn the_import_ceiling_matches_the_pool_it_is_protecting() {
        assert_eq!(
            brokkr_core::voxelise::VERTEX_CAPACITY,
            TOTAL_VERTEX_CAPACITY as f64,
            "brokkr-core's VERTEX_CAPACITY has drifted from the pool's real total"
        );
    }

    // ------------------------------------------------------- against a device
    //
    // The tests below need a real `wgpu::Device`, because a slot is a slice of
    // a real buffer and there is no way to reserve one without creating them.
    // They build the pool through `with_capacities` with room for two bricks
    // per pair, which is the only way to reach the ceiling: the real pool is
    // 4.58 GB at full stretch.

    /// A device, or `None` when this machine has no adapter.
    fn open_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// A device, or `None` on a developer machine that has no adapter -- but a
    /// FAILURE on CI, where an adapter is always meant to be there.
    ///
    /// The rest of this repo handles the adapter-less case by printing
    /// "skipping the ... test" and returning, and CI catches a runner that
    /// really has no adapter by running `tests/offscreen.rs` with `--nocapture`
    /// and grepping that phrase out of the output. That guard covers exactly
    /// one test binary. These two tests live in the LIB binary, which CI runs
    /// as part of `cargo test --workspace` -- without `--nocapture`, so the
    /// message is captured and thrown away, and nothing greps it either way.
    /// Printing and returning here would therefore report `ok` having asserted
    /// nothing, on the half of the increment (reserve-before-release) that
    /// `tests/offscreen.rs` does not exercise at all: someone could put the
    /// release back above the reserve, or collapse the bucket key back to the
    /// bare coordinate, and the job would stay green.
    ///
    /// So the check is made by the test rather than by a grep in the workflow,
    /// which also means it holds however CI decides to invoke `cargo test`.
    /// `CI` is set to `true` by GitHub Actions for every step, and by every
    /// other runner worth naming. The message is still printed, so widening the
    /// workflow's grep to the lib binary would work too and would be belt and
    /// braces rather than a duplicate.
    fn device_or_skip(what: &str) -> Option<(wgpu::Device, wgpu::Queue)> {
        if let Some(device) = open_device() {
            return Some(device);
        }
        eprintln!("no usable wgpu adapter, skipping the {what} test");
        assert!(
            std::env::var_os("CI").is_none(),
            "no usable wgpu adapter on CI, so the {what} test asserted nothing. \
             The runner image is meant to provide one (Mesa's lavapipe); if it \
             no longer does, fix the image rather than letting this pass."
        );
        None
    }

    /// A mesh of the requested size. The geometry is meaningless -- these tests
    /// are about which slice a brick lands in, not about what is in it.
    ///
    /// The mask is a run of zeros of the right LENGTH, which is the part that
    /// matters here: `upload` writes one attribute per vertex, so a mesh with a
    /// short mask would leave the tail of the slice reading whatever the
    /// previous tenant wrote.
    fn mesh_of(vertices: usize, indices: usize) -> BrickMesh {
        BrickMesh {
            vertices: vec![Vertex { position: [0.0; 3], normal: [0.0, 1.0, 0.0] }; vertices],
            indices: (0..indices as u32).map(|index| index % vertices.max(1) as u32).collect(),
            cells: Vec::new(),
            mask: vec![0; vertices],
            colour: vec![0; vertices],
        }
    }

    fn key(body: u32, coord: i32) -> SlotKey {
        SlotKey { body: NodeId(body), coord: BrickCoord::new(coord, 0, 0) }
    }

    /// The failure the widened key exists to stop.
    ///
    /// Two bodies near the world origin share brick coordinates -- there is no
    /// origin held anywhere, so voxel (0,0,0) is world (0,0,0) in every volume
    /// -- and keyed on the coordinate alone, the second body's upload took over
    /// the first body's slice. Nothing logged it and `overflowed` stayed zero.
    #[test]
    fn two_bodies_sharing_a_brick_coordinate_keep_separate_slots() {
        let Some((device, queue)) = device_or_skip("mesh pool slot key") else {
            return;
        };

        let mut pool = MeshPool::with_capacities(&device, GRANULARITY * 8, GRANULARITY * 8);
        let mesh = mesh_of(GRANULARITY as usize, GRANULARITY as usize - 1);
        pool.upload(&device, &queue, key(1, 0), &mesh);
        pool.upload(&device, &queue, key(2, 0), &mesh);

        let (_, first) = pool.find(key(1, 0)).expect("body 1 kept its slot");
        let (_, second) = pool.find(key(2, 0)).expect("body 2 got a slot of its own");
        assert_ne!(
            first.vertices.offset, second.vertices.offset,
            "the two bodies were handed the same vertex slice"
        );
        assert_ne!(
            first.indices.offset, second.indices.offset,
            "the two bodies were handed the same index slice"
        );

        let stats = pool.stats();
        assert_eq!(stats.bricks, 2, "one brick coordinate, two bodies, two slots");
        assert_eq!(stats.overflowed, 0);

        // And a body whose brick meshes to nothing must free only its OWN
        // slice. This is the other half of the bug: an empty mesh at the shared
        // coordinate handed the surviving body's block back to the free list.
        pool.upload(&device, &queue, key(2, 0), &BrickMesh::default());
        let (_, still_there) = pool.find(key(1, 0)).expect("body 1 still has its slot");
        assert_eq!(still_there.vertices.offset, first.vertices.offset);
        assert_eq!(still_there.vertex_count, mesh.vertices.len() as u32);
        assert_eq!(pool.stats().bricks, 1);
    }

    /// A brick that grows when the pool is full keeps what it already had.
    ///
    /// `upload` used to release the old block and remove the slot BEFORE it
    /// found out whether `reserve` could give it a new one, so near a full pool
    /// a growing brick lost its space and got nothing back. The brick that
    /// grows is the one under the brush, so that was a permanent hole opening
    /// in the part of the model being sculpted -- and nothing put it back until
    /// the whole model was rebuilt.
    ///
    /// "Still draws" here means the slot still exists, still points at the same
    /// slice, and still declares the vertex and index counts of the mesh that
    /// was written into it -- which is exactly what `draw` reads. The pixels
    /// are checked in `tests/offscreen.rs`; this pins the bookkeeping, which is
    /// where the bug was.
    #[test]
    fn a_brick_that_grows_when_the_pool_is_full_keeps_the_space_it_had() {
        let Some((device, queue)) = device_or_skip("mesh pool overflow") else {
            return;
        };

        // Two granules of each per pair, so a pair holds exactly two of the
        // small meshes below and eight pairs hold sixteen.
        let mut pool = MeshPool::with_capacities(&device, GRANULARITY * 2, GRANULARITY * 2);
        let small = mesh_of(GRANULARITY as usize, GRANULARITY as usize - 1);
        let filled = 2 * MAX_BUFFERS;
        for brick in 0..filled {
            pool.upload(&device, &queue, key(1, brick as i32), &small);
        }

        let full = pool.stats();
        assert_eq!(full.bricks, filled, "the pool should have taken every brick");
        assert_eq!(full.overflowed, 0, "filling it exactly must not overflow");

        let (_, before) = pool.find(key(1, 0)).expect("brick zero has a slot");

        // Now grow brick zero past its block. Nothing is free and no pair can
        // bump, so the reservation must fail.
        let grown = mesh_of(GRANULARITY as usize + 1, GRANULARITY as usize + 1);
        pool.upload(&device, &queue, key(1, 0), &grown);

        let after = pool.stats();
        assert_eq!(after.overflowed, 1, "the refused upload must be counted");
        assert_eq!(after.bricks, filled, "the slot was dropped instead of being kept");
        assert_eq!(
            after.triangles, full.triangles,
            "the counters ran ahead of the slot: they must only move when it does"
        );
        assert_eq!(after.vertices, full.vertices);

        let (_, kept) = pool.find(key(1, 0)).expect("brick zero must still have its slot");
        assert_eq!(kept.vertices.offset, before.vertices.offset, "it lost its vertex slice");
        assert_eq!(kept.indices.offset, before.indices.offset, "it lost its index slice");
        assert_eq!(
            kept.index_count,
            small.indices.len() as u32,
            "it must still draw the mesh that is actually in the buffer"
        );

        // The same brick shrinking back to a mesh that fits is fine, which is
        // what makes the pool recoverable rather than merely not-worse.
        pool.upload(&device, &queue, key(1, 0), &small);
        assert_eq!(pool.stats().bricks, filled);
        assert_eq!(pool.stats().overflowed, 1, "the count of refusals is cumulative");
    }

    /// Deleting a body must put the pool back exactly where it was before that
    /// body arrived.
    ///
    /// "Unchanged" is the wrong assertion here and it is wrong in exactly the
    /// direction a leak hides in: a pool that kept the deleted body's slots
    /// would also report an unchanged count. So the count is taken BEFORE the
    /// second body is published and compared against the count after it is
    /// forgotten, and the surviving body is checked slice by slice.
    #[test]
    fn forgetting_a_body_returns_the_pool_to_what_it_held_before_that_body() {
        let Some((device, queue)) = device_or_skip("mesh pool forget body") else {
            return;
        };

        let mut pool = MeshPool::with_capacities(&device, GRANULARITY * 16, GRANULARITY * 16);
        let mesh = mesh_of(GRANULARITY as usize, GRANULARITY as usize - 1);
        for brick in 0..3 {
            pool.upload(&device, &queue, key(1, brick), &mesh);
        }
        let before = pool.stats();
        let (_, kept) = pool.find(key(1, 0)).expect("body 1 has brick zero");

        for brick in 0..4 {
            pool.upload(&device, &queue, key(2, brick), &mesh);
        }
        assert_eq!(pool.stats().bricks, before.bricks + 4, "the fixture published nothing");

        assert_eq!(pool.forget_body(NodeId(2)), 4, "every one of body 2's slots should have gone");
        let after = pool.stats();
        assert_eq!(after.bricks, before.bricks, "body 2 kept slots after it was forgotten");
        assert_eq!(after.triangles, before.triangles);
        assert_eq!(after.vertices, before.vertices);
        assert_eq!(after.vertices_reserved, before.vertices_reserved, "its space was not returned");
        assert_eq!(after.indices_reserved, before.indices_reserved);
        assert_eq!(pool.body_bricks(NodeId(2)), 0);

        // And the body that stayed is untouched, down to the slice it draws
        // from -- forgetting one body must not disturb another's bookkeeping.
        assert_eq!(pool.body_bricks(NodeId(1)), 3);
        let (_, still_there) = pool.find(key(1, 0)).expect("body 1 still has brick zero");
        assert_eq!(still_there.vertices.offset, kept.vertices.offset);
        assert_eq!(still_there.index_count, kept.index_count);

        // Forgetting a body that has nothing here is not an error and reports
        // honestly, which is how a caller tells it apart from a real delete.
        assert_eq!(pool.forget_body(NodeId(2)), 0);
        assert_eq!(pool.forget_body(NodeId(77)), 0);
    }

    /// A delete that empties the pool has to take the high-water mark down
    /// with it, and take the MESH POOL FULL banner down too.
    ///
    /// `overflowed` and `warned_about_overflow` used to clear only in
    /// [`MeshPool::reset`], so the red banner stayed up with a stale count over
    /// a document that now fits. And the watermark -- not `live` -- is the
    /// number that decides whether the next allocation fits, so a pair left at
    /// its old bump pointer after the body that pushed it there has gone is a
    /// pool that refuses work it has room for.
    #[test]
    fn forgetting_the_body_that_filled_the_pool_clears_the_banner_and_the_watermark() {
        let Some((device, queue)) = device_or_skip("mesh pool forget clears overflow") else {
            return;
        };

        let mut pool = MeshPool::with_capacities(&device, GRANULARITY * 2, GRANULARITY * 2);
        let small = mesh_of(GRANULARITY as usize, GRANULARITY as usize - 1);
        for brick in 0..2 * MAX_BUFFERS {
            pool.upload(&device, &queue, key(1, brick as i32), &small);
        }
        // One more than it can hold, so the overflow counter is really set.
        pool.upload(&device, &queue, key(1, 900), &small);
        assert!(pool.stats().overflowed > 0, "the fixture never filled the pool");
        assert!(pool.stats().vertices_watermark > 0);

        pool.forget_body(NodeId(1));
        let after = pool.stats();
        assert_eq!(after.bricks, 0);
        assert_eq!(after.overflowed, 0, "the banner would have stayed up over an empty pool");
        assert_eq!(
            after.vertices_watermark, 0,
            "every pair is empty, so every bump pointer should have gone back to zero"
        );
        assert_eq!(after.indices_watermark, 0);

        // And the emptied pool is usable again to its full depth, which is the
        // point of resetting the pairs rather than merely freeing the blocks.
        for brick in 0..2 * MAX_BUFFERS {
            pool.upload(&device, &queue, key(3, brick as i32), &small);
        }
        assert_eq!(pool.stats().bricks, 2 * MAX_BUFFERS);
        assert_eq!(pool.stats().overflowed, 0, "the pool did not really recover its space");
    }

    /// **Freeing space does not un-refuse a brick, and the banner going down
    /// does not mean the model is whole.** This is the precondition on
    /// [`MeshPool::forget_body`] written as an assertion, because the doc
    /// comment on its own is not something a future change can trip over.
    ///
    /// A brick refused while the pool was full is dropped -- `upload`'s
    /// overflow arm returns after counting it -- and the application drained
    /// that coordinate out of its dirty set on the way in, so nothing offers it
    /// again. Deleting some other body clears `overflowed`, which takes the red
    /// MESH POOL FULL banner down over a document that is still missing
    /// geometry. The only thing that brings the brick back is a remesh, which
    /// is why the delete gesture owes one.
    #[test]
    fn a_brick_refused_while_the_pool_was_full_stays_missing_after_a_delete() {
        let Some((device, queue)) = device_or_skip("mesh pool refused brick after delete") else {
            return;
        };

        let mut pool = MeshPool::with_capacities(&device, GRANULARITY * 2, GRANULARITY * 2);
        let small = mesh_of(GRANULARITY as usize, GRANULARITY as usize - 1);
        for brick in 0..2 * MAX_BUFFERS {
            pool.upload(&device, &queue, key(1, brick as i32), &small);
        }
        // Body 2's one brick arrives at a pool with nothing left, and is
        // refused. Body 2 is NOT deleted below -- it is the survivor, and this
        // brick is a piece of the model the user can still see the rest of.
        pool.upload(&device, &queue, key(2, 0), &small);
        assert_eq!(pool.stats().overflowed, 1, "the fixture never filled the pool");
        assert_eq!(pool.body_bricks(NodeId(2)), 0);

        pool.forget_body(NodeId(1));

        assert_eq!(pool.stats().overflowed, 0, "the banner is down...");
        assert_eq!(
            pool.body_bricks(NodeId(2)),
            0,
            "...and the brick it was warning about is still missing, which is the whole \
             hazard: without the remesh the caller owes, this state is permanent and silent"
        );

        // And the remesh is what fixes it. One re-offer of the same brick, into
        // the space the delete freed, and the model is whole again.
        pool.upload(&device, &queue, key(2, 0), &small);
        assert_eq!(pool.body_bricks(NodeId(2)), 1);
        assert_eq!(pool.stats().overflowed, 0);
    }

    /// A mask-only remesh reuses the slice it already had and touches no free
    /// list.
    ///
    /// Painting a mask does not move a vertex -- it is a read-only multiplier on
    /// what a brush may do -- so the brick comes back with exactly the same
    /// counts and only its attributes changed. `upload`'s "still fits" arm is
    /// what makes that free; without it, every mask stamp would hand its blocks
    /// back and take new ones, which is how the allocator fragments (see
    /// [`BlockAllocator`]) and is a cost the mask has no reason to pay.
    #[test]
    fn a_mask_only_remesh_keeps_the_slice_it_had_and_returns_nothing_to_the_free_list() {
        let Some((device, queue)) = device_or_skip("mesh pool mask remesh") else {
            return;
        };

        let mut pool = MeshPool::with_capacities(&device, GRANULARITY * 16, GRANULARITY * 16);
        let mesh = mesh_of(GRANULARITY as usize, GRANULARITY as usize - 1);
        pool.upload(&device, &queue, key(1, 0), &mesh);
        let (_, before) = pool.find(key(1, 0)).expect("the brick has a slot");
        let stats = pool.stats();

        // The same geometry, a different mask: what a mask stamp produces.
        let masked = BrickMesh { mask: vec![200; mesh.vertices.len()], ..mesh.clone() };
        pool.upload(&device, &queue, key(1, 0), &masked);

        let (_, after) = pool.find(key(1, 0)).expect("the brick still has a slot");
        assert_eq!(after.vertices.offset, before.vertices.offset, "it was given a new slice");
        assert_eq!(after.indices.offset, before.indices.offset);
        assert_eq!(after.vertex_count, before.vertex_count);
        let now = pool.stats();
        assert_eq!(now.vertices_reserved, stats.vertices_reserved, "space was handed back");
        assert_eq!(now.vertices_watermark, stats.vertices_watermark, "the bump pointer moved");
        assert_eq!(now.triangles, stats.triangles);
        assert_eq!(now.bricks, stats.bricks);
    }

    /// **Hiding is a draw-time skip and nothing else**, and this is the
    /// assertion that says so: not one byte of pool space moves, no slot is
    /// dropped, and no mesh is touched.
    ///
    /// The alternative -- dropping a hidden body's bricks, or marking them
    /// dirty and meshing them to nothing -- would make showing it again a
    /// whole-body remesh and would make the eye a thing that loses geometry.
    /// It is also the half of the rule that cannot be seen from the
    /// application: `vertices_reserved` and `vertices_watermark` only exist
    /// here.
    #[test]
    fn hiding_a_body_moves_no_pool_space_at_all() {
        let Some((device, queue)) = device_or_skip("mesh pool hidden set") else {
            return;
        };

        let mut pool = MeshPool::with_capacities(&device, GRANULARITY * 16, GRANULARITY * 16);
        let mesh = mesh_of(GRANULARITY as usize, GRANULARITY as usize - 1);
        for body in 1..=2 {
            for brick in 0..3 {
                pool.upload(&device, &queue, key(body, brick), &mesh);
            }
        }
        let before = pool.stats();

        pool.set_hidden(&[NodeId(2)]);
        let after = pool.stats();
        assert_eq!(after.bricks, before.bricks, "hiding a body dropped its slots");
        assert_eq!(after.vertices_reserved, before.vertices_reserved, "hiding freed pool space");
        assert_eq!(after.indices_reserved, before.indices_reserved);
        assert_eq!(after.vertices_watermark, before.vertices_watermark);
        assert_eq!(after.indices_watermark, before.indices_watermark);
        assert_eq!(after.triangles, before.triangles);
        assert_eq!(pool.body_bricks(NodeId(2)), 3, "the hidden body must keep every brick");

        // Showing it again is a set with the body left out, and costs nothing.
        pool.set_hidden(&[]);
        assert_eq!(pool.stats().bricks, before.bricks);
        assert_eq!(pool.stats().vertices_watermark, before.vertices_watermark);
    }
}
