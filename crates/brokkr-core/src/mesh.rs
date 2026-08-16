// SPDX-License-Identifier: AGPL-3.0-or-later

//! Surface nets meshing of a single brick from its apron buffer.

use bytemuck::{Pod, Zeroable};
use fast_surface_nets::ndshape::ConstShape3u32;
use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};
use glam::Vec3;

use crate::apron::ApronBuffer;
use crate::brick::{APRON_DIM, BrickCoord};

/// Shape of the apron sample array as surface nets sees it.
///
/// The const generic arguments must stay equal to [`APRON_DIM`]. The assertion
/// below fails the build if they ever drift apart.
type ApronShape = ConstShape3u32<34, 34, 34>;

const _: () = assert!(
    APRON_DIM == 34,
    "ApronShape is hard coded to 34 because const generics cannot take APRON_DIM directly. \
     If BRICK_DIM changes, change ApronShape to match."
);

/// One mesh vertex. Normals are unit length.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// The triangles produced for one brick, in world space.
#[derive(Debug, Default, Clone)]
pub struct BrickMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl BrickMesh {
    /// Drop the contents but keep the allocations for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Reusable working memory for meshing.
///
/// Holding one of these across many bricks is what keeps meshing out of the
/// allocator. Create it once per worker and pass it in every time.
pub struct MeshScratch {
    pub(crate) apron: ApronBuffer,
    pub(crate) surface_nets: SurfaceNetsBuffer,
}

impl MeshScratch {
    pub fn new() -> Self {
        Self { apron: ApronBuffer::new(), surface_nets: SurfaceNetsBuffer::default() }
    }

    /// The apron gathered by the last meshing call, for inspection in tests.
    pub fn apron(&self) -> &ApronBuffer {
        &self.apron
    }
}

impl Default for MeshScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Run surface nets over an already gathered apron and write world space
/// triangles into `out`.
///
/// Crate private on purpose. Public meshing goes through
/// [`crate::Volume::mesh_brick`], which is the only thing that can produce a
/// gathered apron, so the halo can never be skipped.
///
/// The apron covers world voxels `origin - 1` through `origin + BRICK_DIM`,
/// so a sample at apron local coordinate `l` sits at world voxel
/// `l + origin - 1`. Surface nets is asked for the full `[0, 33]` range: it
/// emits vertices for every cell in that range but skips faces on the positive
/// boundary, which is exactly what makes adjacent bricks tile without gaps or
/// double covered quads.
pub(crate) fn mesh_apron(
    apron: &ApronBuffer,
    coord: BrickCoord,
    voxel_size: f32,
    scratch: &mut SurfaceNetsBuffer,
    out: &mut BrickMesh,
) {
    out.clear();

    surface_nets(
        apron.samples().as_slice(),
        &ApronShape {},
        [0; 3],
        [(APRON_DIM - 1) as u32; 3],
        scratch,
    );

    if scratch.indices.is_empty() {
        return;
    }

    // Apron local coordinate to world voxel, then to world space.
    let origin = coord.origin().as_vec3() - Vec3::ONE;

    out.vertices.reserve(scratch.positions.len());
    for (position, normal) in scratch.positions.iter().zip(scratch.normals.iter()) {
        let local = Vec3::from_array(*position);
        let world = (local + origin) * voxel_size;
        // Surface nets returns unnormalised gradients. Normalise once here
        // rather than per fragment. Degenerate gradients fall back to up so a
        // zero vector never reaches the shader.
        let n = Vec3::from_array(*normal);
        let n = n.try_normalize().unwrap_or(Vec3::Y);
        out.vertices.push(Vertex { position: world.to_array(), normal: n.to_array() });
    }

    out.indices.extend_from_slice(&scratch.indices);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::apron_index;
    use fast_surface_nets::ndshape::ConstShape;

    /// The apron gather writes with X fastest. If ndshape ever disagreed, every
    /// mesh would come out transposed and the seams would be subtly wrong, so
    /// pin the two layouts together.
    #[test]
    fn apron_layout_matches_ndshape() {
        for (x, y, z) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (33, 33, 33), (7, 11, 29)] {
            assert_eq!(
                apron_index(x, y, z) as u32,
                ApronShape::linearize([x as u32, y as u32, z as u32]),
                "layout mismatch at ({x}, {y}, {z})"
            );
        }
    }

    #[test]
    fn apron_shape_size_matches_constants() {
        assert_eq!(ApronShape::SIZE as usize, crate::brick::APRON_VOXELS);
    }
}
