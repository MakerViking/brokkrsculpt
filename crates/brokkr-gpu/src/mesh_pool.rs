// SPDX-License-Identifier: AGPL-3.0-or-later

//! A shared vertex and index buffer that bricks own slices of.
//!
//! Each brick's mesh lives in a suballocation of one large buffer, so a remesh
//! writes into that slice instead of creating a buffer. Nothing here allocates
//! GPU memory after startup: the two big buffers are made once and the
//! suballocator hands out and reclaims ranges inside them.

use brokkr_core::{BrickCoord, BrickMesh, Vertex};
use rustc_hash::FxHashMap;

/// Vertices the pool can hold. At 24 bytes each this is 48 MB.
///
/// A 256 cubed sphere meshes to roughly 300k vertices before per brick seam
/// duplication, so this leaves several times the headroom M0 needs. Growing the
/// pool at run time is deliberately not implemented: the GPU rewrite in M2
/// replaces this with a brick pool and an atomic free list.
pub const VERTEX_CAPACITY: u64 = 2_000_000;

/// Indices the pool can hold. At 4 bytes each this is 32 MB.
pub const INDEX_CAPACITY: u64 = 8_000_000;

/// Smallest suballocation, as a power of two count of elements.
///
/// Rounding every request up to at least this keeps the free lists short and
/// stops a brick that meshes to three vertices from fragmenting the pool.
const MIN_BLOCK_SHIFT: u32 = 8;

/// A range inside one of the big buffers, measured in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    offset: u64,
    capacity: u64,
}

/// Power of two size class suballocator over a fixed range.
///
/// Blocks are never split or merged, so a freed block can only be reused by a
/// request in the same size class. Brick meshes cluster tightly in size, which
/// is what makes that acceptable.
#[derive(Debug)]
struct BlockAllocator {
    capacity: u64,
    bump: u64,
    free: Vec<Vec<u64>>,
    live: u64,
}

impl BlockAllocator {
    fn new(capacity: u64) -> Self {
        Self { capacity, bump: 0, free: Vec::new(), live: 0 }
    }

    fn class_shift(count: u64) -> u32 {
        count.max(1).next_power_of_two().trailing_zeros().max(MIN_BLOCK_SHIFT)
    }

    fn allocate(&mut self, count: u64) -> Option<Block> {
        let shift = Self::class_shift(count);
        let capacity = 1u64 << shift;
        let class = (shift - MIN_BLOCK_SHIFT) as usize;

        if self.free.len() <= class {
            self.free.resize(class + 1, Vec::new());
        }
        if let Some(offset) = self.free[class].pop() {
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
        let class = (block.capacity.trailing_zeros() - MIN_BLOCK_SHIFT) as usize;
        if self.free.len() <= class {
            self.free.resize(class + 1, Vec::new());
        }
        self.free[class].push(block.offset);
        self.live -= block.capacity;
    }

    /// Elements handed out, including the padding inside each size class.
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
}

/// What the pool is currently holding, for the debug overlay.
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolStats {
    pub bricks: usize,
    pub triangles: usize,
    /// Vertices actually in use, as opposed to the space reserved for them.
    pub vertices: usize,
    pub vertices_used: u64,
    pub indices_used: u64,
    pub vertex_capacity: u64,
    pub index_capacity: u64,
    /// Bricks skipped because the pool was full. Any value above zero means
    /// the model on screen is incomplete.
    pub overflowed: usize,
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
                Slot { vertices, indices, vertex_count: 0, index_count: 0 }
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
        self.slots.insert(
            coord,
            Slot {
                vertex_count: mesh.vertices.len() as u32,
                index_count: mesh.indices.len() as u32,
                ..slot
            },
        );
    }

    /// Record one indexed draw per brick.
    ///
    /// Per brick draws are fine at M0 scale and get replaced by batching and
    /// indirect draws in M2. Indices are brick local, so the slice's vertex
    /// offset goes in as the base vertex.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.slots.is_empty() {
            return;
        }
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for slot in self.slots.values() {
            if slot.index_count == 0 {
                continue;
            }
            let start = slot.indices.offset as u32;
            pass.draw_indexed(start..start + slot.index_count, slot.vertices.offset as i32, 0..1);
        }
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            bricks: self.slots.len(),
            triangles: self.triangles,
            vertices: self.vertices,
            vertices_used: self.vertex_allocator.live(),
            indices_used: self.index_allocator.live(),
            vertex_capacity: VERTEX_CAPACITY,
            index_capacity: INDEX_CAPACITY,
            overflowed: self.overflowed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_round_up_to_the_minimum_class() {
        let mut allocator = BlockAllocator::new(1 << 20);
        let block = allocator.allocate(3).expect("space is available");
        assert_eq!(block.capacity, 1 << MIN_BLOCK_SHIFT);
        assert_eq!(block.offset, 0);
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
