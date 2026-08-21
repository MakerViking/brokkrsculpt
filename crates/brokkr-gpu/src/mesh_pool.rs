// SPDX-License-Identifier: AGPL-3.0-only

//! A shared vertex and index buffer that bricks own slices of.
//!
//! Each brick's mesh lives in a suballocation of one large buffer, so a remesh
//! writes into that slice instead of creating a buffer. Nothing here allocates
//! GPU memory after startup: the two big buffers are made once and the
//! suballocator hands out and reclaims ranges inside them.

use std::sync::atomic::{AtomicUsize, Ordering};

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

/// How many buffer pairs the pool may grow to.
///
/// Buffers are created **on demand**, so a small model pays for one pair
/// (528 MB) and only a model that needs more allocates more. Eight pairs is
/// 4.2 GB of VRAM at full stretch, which is the point at which refusing is
/// kinder than trying on any card this application is likely to meet.
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
#[derive(Debug)]
struct BufferPair {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    vertex_allocator: BlockAllocator,
    index_allocator: BlockAllocator,
}

impl BufferPair {
    fn new(device: &wgpu::Device, index: u16) -> Self {
        Self {
            vertices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brokkr brick vertices"),
                size: VERTEX_CAPACITY * size_of::<Vertex>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            indices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brokkr brick indices"),
                size: INDEX_CAPACITY * size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            vertex_allocator: BlockAllocator::new(VERTEX_CAPACITY, index),
            index_allocator: BlockAllocator::new(INDEX_CAPACITY, index),
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
}

/// The shared mesh buffers and the map from brick to slice.
#[derive(Debug)]
pub struct MeshPool {
    /// One pair per buffer of the pool, created on demand. Never shrinks: the
    /// buffers are reused by [`MeshPool::reset`] rather than dropped, because
    /// a rebuild almost always needs them again immediately.
    buffers: Vec<BufferPair>,
    slots: FxHashMap<BrickCoord, Slot>,
    triangles: usize,
    vertices: usize,
    overflowed: usize,
    warned_about_overflow: bool,
    /// Counts from the last draw. Atomic because drawing takes a shared borrow
    /// and Iced requires the pipeline it owns to be `Sync`.
    drawn: AtomicUsize,
    culled: AtomicUsize,
}

impl MeshPool {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buffers: vec![BufferPair::new(device, 0)],
            slots: FxHashMap::default(),
            triangles: 0,
            vertices: 0,
            overflowed: 0,
            warned_about_overflow: false,
            drawn: AtomicUsize::new(0),
            culled: AtomicUsize::new(0),
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
                * (VERTEX_CAPACITY * size_of::<Vertex>() as u64
                    + INDEX_CAPACITY * size_of::<u32>() as u64)
                / (1024 * 1024)
        );
        self.buffers.push(BufferPair::new(device, index));
        let pair = self.buffers.last_mut()?;
        Some((
            pair.vertex_allocator.allocate(need_vertices)?,
            pair.index_allocator.allocate(need_indices)?,
        ))
    }

    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        coord: BrickCoord,
        mesh: &BrickMesh,
    ) {
        if let Some(previous) = self.slots.get(&coord) {
            self.triangles -= previous.index_count as usize / 3;
            self.vertices -= previous.vertex_count as usize;
        }

        if mesh.indices.is_empty() {
            if let Some(slot) = self.slots.remove(&coord) {
                self.release(slot);
            }
            return;
        }

        let need_vertices = mesh.vertices.len() as u64;
        let need_indices = mesh.indices.len() as u64;

        // Keep the existing slices when the new mesh still fits them, which is
        // the common case during a stroke and avoids touching the free lists.
        let slot = match self.slots.get(&coord).copied() {
            Some(slot)
                if slot.vertices.capacity >= need_vertices
                    && slot.indices.capacity >= need_indices =>
            {
                slot
            }
            other => {
                if let Some(slot) = other {
                    self.release(slot);
                    self.slots.remove(&coord);
                }
                let Some((vertices, indices)) = self.reserve(device, need_vertices, need_indices)
                else {
                    self.overflowed += 1;
                    if !self.warned_about_overflow {
                        self.warned_about_overflow = true;
                        log::error!(
                            "mesh pool is full at {MAX_BUFFERS} buffers ({TOTAL_VERTEX_CAPACITY} \
                             vertices, {TOTAL_INDEX_CAPACITY} indices), so parts of the model are \
                             missing from the screen"
                        );
                    }
                    return;
                };
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

        self.triangles += mesh.indices.len() / 3;
        self.vertices += mesh.vertices.len();
        let (minimum, maximum) = bounds(&mesh.vertices).unwrap_or((Vec3::ZERO, Vec3::ZERO));
        self.slots.insert(
            coord,
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
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, frustum: &Frustum) {
        self.drawn.store(0, Ordering::Relaxed);
        self.culled.store(0, Ordering::Relaxed);
        if self.slots.is_empty() {
            return;
        }

        // Grouped by buffer pair, because a bind is per pass and a brick's
        // vertices and indices always live in the same pair. One pass over the
        // slots per pair keeps the binds to one each rather than one per brick,
        // and with a single pair -- the common case -- this is exactly what it
        // was before.
        let mut drawn = 0;
        let mut culled = 0;
        for (index, pair) in self.buffers.iter().enumerate() {
            let index = index as u16;
            let mut bound = false;
            for slot in self.slots.values() {
                if slot.vertices.buffer != index || slot.index_count == 0 {
                    continue;
                }
                if !frustum.intersects(slot.minimum, slot.maximum) {
                    // Counted once, on the pass that owns the brick.
                    culled += 1;
                    continue;
                }
                if !bound {
                    pass.set_vertex_buffer(0, pair.vertices.slice(..));
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
        self.slots.clear();
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
            bricks: self.slots.len(),
            triangles: self.triangles,
            vertices: self.vertices,
            indices: self.triangles * 3,
            vertices_reserved: self.buffers.iter().map(|p| p.vertex_allocator.live()).sum(),
            indices_reserved: self.buffers.iter().map(|p| p.index_allocator.live()).sum(),
            vertices_watermark: self.buffers.iter().map(|p| p.vertex_allocator.watermark()).sum(),
            indices_watermark: self.buffers.iter().map(|p| p.index_allocator.watermark()).sum(),
            vertex_capacity: TOTAL_VERTEX_CAPACITY,
            index_capacity: TOTAL_INDEX_CAPACITY,
            overflowed: self.overflowed,
            drawn: self.drawn.load(Ordering::Relaxed),
            culled: self.culled.load(Ordering::Relaxed),
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
}
