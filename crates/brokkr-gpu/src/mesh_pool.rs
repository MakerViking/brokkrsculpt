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

/// Vertices the pool can hold. At 24 bytes each this is 192 MB.
///
/// Sized from measurement rather than guessed: a 30 mm sphere at a 0.055 mm
/// voxel comes to 11.2 million triangles, 6.2 million vertices and 33.6 million
/// indices, which is the scale M2 targets. This leaves about a quarter again on
/// top of that.
///
/// Reserved up front. Growing a wgpu buffer means making a new one and copying,
/// and the pool reports an overflow loudly rather than doing that mid stroke.
pub const VERTEX_CAPACITY: u64 = 8_000_000;

/// Indices the pool can hold. At 4 bytes each this is 176 MB.
pub const INDEX_CAPACITY: u64 = 44_000_000;

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
}

/// Fixed granularity suballocator over a fixed range.
///
/// Blocks are never split or merged, so a freed block can only be reused by a
/// request that rounds to the same size. Brick meshes cluster tightly in size,
/// which is what makes that acceptable.
#[derive(Debug)]
struct BlockAllocator {
    capacity: u64,
    bump: u64,
    /// Free block offsets, indexed by how many granules they hold.
    free: Vec<Vec<u64>>,
    live: u64,
}

impl BlockAllocator {
    fn new(capacity: u64) -> Self {
        Self { capacity, bump: 0, free: Vec::new(), live: 0 }
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
            return Some(Block { offset, capacity });
        }

        if self.bump + capacity > self.capacity {
            return None;
        }
        let offset = self.bump;
        self.bump += capacity;
        self.live += capacity;
        Some(Block { offset, capacity })
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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    slots: FxHashMap<BrickCoord, Slot>,
    vertex_allocator: BlockAllocator,
    index_allocator: BlockAllocator,
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
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brokkr brick vertices"),
            size: VERTEX_CAPACITY * size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brokkr brick indices"),
            size: INDEX_CAPACITY * size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            slots: FxHashMap::default(),
            vertex_allocator: BlockAllocator::new(VERTEX_CAPACITY),
            index_allocator: BlockAllocator::new(INDEX_CAPACITY),
            triangles: 0,
            vertices: 0,
            overflowed: 0,
            warned_about_overflow: false,
            drawn: AtomicUsize::new(0),
            culled: AtomicUsize::new(0),
        }
    }

    /// Replace one brick's mesh.
    ///
    /// An empty mesh releases the brick's slices, which is what happens when a
    /// stroke carves a brick away entirely.
    pub fn upload(&mut self, queue: &wgpu::Queue, coord: BrickCoord, mesh: &BrickMesh) {
        if let Some(previous) = self.slots.get(&coord) {
            self.triangles -= previous.index_count as usize / 3;
            self.vertices -= previous.vertex_count as usize;
        }

        if mesh.indices.is_empty() {
            if let Some(slot) = self.slots.remove(&coord) {
                self.vertex_allocator.release(slot.vertices);
                self.index_allocator.release(slot.indices);
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
                    self.vertex_allocator.release(slot.vertices);
                    self.index_allocator.release(slot.indices);
                    self.slots.remove(&coord);
                }
                let (Some(vertices), Some(indices)) = (
                    self.vertex_allocator.allocate(need_vertices),
                    self.index_allocator.allocate(need_indices),
                ) else {
                    self.overflowed += 1;
                    if !self.warned_about_overflow {
                        self.warned_about_overflow = true;
                        log::error!(
                            "mesh pool is full at {VERTEX_CAPACITY} vertices and {INDEX_CAPACITY} \
                             indices, so parts of the model are missing from the screen"
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

        queue.write_buffer(
            &self.vertex_buffer,
            slot.vertices.offset * size_of::<Vertex>() as u64,
            bytemuck::cast_slice(&mesh.vertices),
        );
        queue.write_buffer(
            &self.index_buffer,
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

        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        let mut drawn = 0;
        let mut culled = 0;
        for slot in self.slots.values() {
            if slot.index_count == 0 {
                continue;
            }
            if !frustum.intersects(slot.minimum, slot.maximum) {
                culled += 1;
                continue;
            }
            drawn += 1;
            let start = slot.indices.offset as u32;
            pass.draw_indexed(start..start + slot.index_count, slot.vertices.offset as i32, 0..1);
        }
        self.drawn.store(drawn, Ordering::Relaxed);
        self.culled.store(culled, Ordering::Relaxed);
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            bricks: self.slots.len(),
            triangles: self.triangles,
            vertices: self.vertices,
            indices: self.triangles * 3,
            vertices_reserved: self.vertex_allocator.live(),
            indices_reserved: self.index_allocator.live(),
            vertex_capacity: VERTEX_CAPACITY,
            index_capacity: INDEX_CAPACITY,
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
        let mut allocator = BlockAllocator::new(1 << 20);
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
        let mut allocator = BlockAllocator::new(1 << 24);
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
        let mut allocator = BlockAllocator::new(1 << 20);
        let first = allocator.allocate(1000).expect("space is available");
        let second = allocator.allocate(1000).expect("space is available");
        assert_ne!(first.offset, second.offset);

        allocator.release(first);
        let third = allocator.allocate(1000).expect("the freed block is reusable");
        assert_eq!(third.offset, first.offset, "should have come off the free list");
    }

    #[test]
    fn allocation_fails_rather_than_overrunning_the_buffer() {
        let mut allocator = BlockAllocator::new(1 << 10);
        assert!(allocator.allocate(1 << 10).is_some());
        assert!(allocator.allocate(1).is_none(), "must refuse to hand out space it does not have");
    }

    #[test]
    fn live_count_returns_to_zero_after_releasing_everything() {
        let mut allocator = BlockAllocator::new(1 << 20);
        let blocks: Vec<_> = (0..16).map(|n| allocator.allocate(n * 37 + 1).unwrap()).collect();
        assert!(allocator.live() > 0);
        for block in blocks {
            allocator.release(block);
        }
        assert_eq!(allocator.live(), 0);
    }
}
