// SPDX-License-Identifier: AGPL-3.0-only

//! The apron buffer: a brick's voxels plus the one voxel halo of neighbour
//! data that meshing requires.
//!
//! # The apron rule
//!
//! Surface nets estimates a vertex per lattice cell, and a cell needs the
//! eight corner samples around it. A brick that meshed only its own 32 cubed
//! voxels would have no data for the cells straddling its outer faces, so the
//! surface would stop one voxel short on every side and every brick boundary
//! would show a crack.
//!
//! The fix is to mesh a 34 cubed sample array covering world voxels
//! `origin - 1` through `origin + BRICK_DIM` inclusive, gathered from the
//! brick and its 26 neighbours.
//!
//! This type exists so that rule cannot be forgotten. There is no public way
//! to turn a [`crate::Brick`] into a mesh. The only meshing entry point is
//! [`crate::Volume::mesh_brick`], which gathers the apron itself, so no call
//! site can mesh a brick without the halo.

use crate::brick::{APRON_DIM, APRON_VOXELS, BrickCoord, apron_index};

/// A brick's sample array including its one voxel halo.
///
/// Allocate once and reuse across bricks. Filling one is
/// [`crate::Volume::gather_apron`]'s job.
#[derive(Clone)]
pub struct ApronBuffer {
    /// Which brick this was last gathered for, and `None` if it has never been
    /// gathered. Meshing a buffer that was never gathered would silently
    /// produce an empty mesh, so the meshing path checks this.
    coord: Option<BrickCoord>,
    samples: Box<[f32; APRON_VOXELS]>,
}

impl ApronBuffer {
    /// An ungathered buffer with its storage allocated.
    pub fn new() -> Self {
        let samples: Box<[f32]> = vec![0.0; APRON_VOXELS].into_boxed_slice();
        let samples: Box<[f32; APRON_VOXELS]> = samples.try_into().expect("length is APRON_VOXELS");
        Self { coord: None, samples }
    }

    /// The brick this buffer currently holds, or `None` if never gathered.
    #[inline]
    pub fn coord(&self) -> Option<BrickCoord> {
        self.coord
    }

    /// Sample at a coordinate local to the apron, where `(1, 1, 1)` is the
    /// brick's own minimum corner and `(0, 0, 0)` is the halo.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.samples[apron_index(x, y, z)]
    }

    /// The raw sample array, laid out with X fastest.
    #[inline]
    pub fn samples(&self) -> &[f32; APRON_VOXELS] {
        &self.samples
    }

    /// Mutable access for the gather. Crate private so no outside code can
    /// hand meshing a hand rolled buffer with no halo in it.
    #[inline]
    pub(crate) fn samples_mut(&mut self) -> &mut [f32; APRON_VOXELS] {
        &mut self.samples
    }

    /// Record which brick the gather just filled this buffer for.
    #[inline]
    pub(crate) fn set_coord(&mut self, coord: BrickCoord) {
        self.coord = Some(coord);
    }
}

impl Default for ApronBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ApronBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApronBuffer")
            .field("coord", &self.coord)
            .field("dim", &APRON_DIM)
            .finish_non_exhaustive()
    }
}
