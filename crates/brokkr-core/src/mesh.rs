// SPDX-License-Identifier: AGPL-3.0-only

//! Surface nets meshing of a single brick from its apron buffer.

use bytemuck::{Pod, Zeroable};
use fast_surface_nets::ndshape::ConstShape3u32;
use fast_surface_nets::{SurfaceNetsBuffer, surface_nets};
use glam::{IVec3, Vec3};
use rustc_hash::FxHashMap;

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
    /// The lattice cell each vertex came from, in world voxel coordinates.
    ///
    /// Surface nets puts at most one vertex in each cell, and two bricks sharing
    /// a seam derive that cell from the same world coordinate. So this is an
    /// exact identity for a vertex, which is what export welds on.
    ///
    /// Welding on the positions instead does not work. Two bricks compute the
    /// same seam vertex from the same corner values but at different
    /// intermediate magnitudes, so the results differ in the last bits, and any
    /// scheme that rounds a position onto a grid will split the pair whenever
    /// they straddle a boundary. That happened to about one vertex in a hundred
    /// and left hundreds of holes in an otherwise closed model.
    pub cells: Vec<IVec3>,
    /// The OTHER cell a vertex sits between, for the vertices
    /// [`split_paint_boundaries`] adds at the midpoint of an edge whose ends
    /// are painted differently. Equal to `cells[i]` for every vertex surface
    /// nets made, so `(cells[i], partners[i])` is an exact identity for both
    /// kinds and is what the export welds on: two bricks either side of a seam
    /// split the same edge at the same two cells and get the same key.
    ///
    /// **Always the same length as `vertices`.**
    pub partners: Vec<IVec3>,
    /// The mask's STORED byte at each vertex's own lattice cell, one per vertex.
    ///
    /// **Stored, with the polarity NOT applied**, because the polarity is
    /// resolved in the shader from a uniform: that is what makes Invert and
    /// Mask All one `u32` write instead of a remesh of the whole body. See
    /// [`crate::MaskField::byte_at`], whose own documentation names this as one
    /// of its two legitimate callers.
    ///
    /// Sampled by the vertex's CELL and not by its position, which is what
    /// makes the tint continuous across a brick seam: two bricks that both emit
    /// a vertex at a shared seam derive that cell from the same world
    /// coordinate (see [`BrickMesh::cells`]), so they look up the same voxel
    /// and get the same byte, bit for bit.
    ///
    /// **Always the same length as `vertices`**, including for an unmasked
    /// body, where it is a run of zeros. The pool writes it into a vertex
    /// buffer of its own, so a short one would leave the tail of a brick
    /// reading whatever the previous tenant of that slice left behind.
    pub mask: Vec<u8>,
    /// The painted filament slot at each vertex's own lattice cell, one per
    /// vertex.
    ///
    /// Sampled by CELL for the same reason the mask is: two bricks that both
    /// emit a vertex at a shared seam derive that cell from the same world
    /// coordinate, so they read the same voxel and agree bit for bit. A slot is
    /// an INDEX, so a seam that disagreed would not shade slightly differently
    /// -- it would print in another filament.
    ///
    /// **Always the same length as `vertices`**, including for an unpainted
    /// body, where it is a run of zeros, for the reason `mask` gives: the pool
    /// writes it into a shared buffer and a short run leaves the tail of a
    /// brick reading whatever the previous tenant left there.
    pub colour: Vec<u8>,
}

impl BrickMesh {
    /// Drop the contents but keep the allocations for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.cells.clear();
        self.partners.clear();
        self.mask.clear();
        self.colour.clear();
    }

    /// The weld key of vertex `at`: the two cells it sits between, in a fixed
    /// order, so a midpoint reached from either end reads the same.
    #[inline]
    pub fn weld_key(&self, at: usize) -> (IVec3, IVec3) {
        let (a, b) = (self.cells[at], self.partners[at]);
        if (a.x, a.y, a.z) <= (b.x, b.y, b.z) { (a, b) } else { (b, a) }
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
    out.cells.reserve(scratch.positions.len());
    // Apron local coordinate of the cell each vertex came from, offset to world.
    let cell_origin = coord.origin() - IVec3::ONE;
    for (index, (position, normal)) in
        scratch.positions.iter().zip(scratch.normals.iter()).enumerate()
    {
        let cell = scratch.surface_points[index];
        let cell = cell_origin + IVec3::new(cell[0] as i32, cell[1] as i32, cell[2] as i32);
        out.cells.push(cell);
        out.partners.push(cell);

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

/// Cut every triangle whose corners are painted differently along the
/// midpoints of its mixed edges, so the paint boundary runs at half-triangle
/// precision instead of stepping a whole triangle at a time.
///
/// # Why this is the fix, and what it costs
///
/// A slot is read per vertex and drawn per triangle -- the first vertex's, in
/// the shader and in the 3MF alike -- so a triangle straddling the boundary
/// takes one side whole, and a curve painted across a sphere comes out as a
/// sawtooth of triangle-sized teeth. Slicers solve the same problem by
/// subdividing painted triangles; this does it once, at mesh time, so the
/// viewport and the export show the same edge.
///
/// **Every piece's FIRST vertex is an original vertex of the side the piece
/// belongs to.** That is the whole trick: the provoking-vertex rule and the
/// export's per-triangle rule both read the first vertex, so the midpoints
/// themselves need no meaningful slot. (The centre piece of a three-way corner
/// has only midpoints; it takes the lower slot of its first midpoint's ends,
/// which both bricks compute alike.) Winding is preserved by construction --
/// each piece is a cyclic rotation of a sub-triangle cut from the original.
///
/// A mixed edge is shared by two triangles and is cut once: midpoints are
/// deduplicated within the brick on the pair of cells they sit between, and
/// across bricks by the export weld on the same key. Both bricks either side
/// of a seam see the same two endpoint slots, so both cut, so there is no
/// T-junction and the mesh stays closed; `a_painted_body_exports_watertight`
/// pins that with the boundary both on and off a brick seam.
///
/// Cost is proportional to the length of the paint boundary: a stroke's edge
/// is a curve of triangles, and each costs two midpoint vertices and two extra
/// triangles. A stencilled pattern is the worst case, near every triangle
/// mixed, and the parity-painted bench row measures that ceiling. An unpainted
/// brick pays one scan of its colour bytes and nothing else.
pub(crate) fn split_paint_boundaries(out: &mut BrickMesh) {
    if out.colour.iter().all(|slot| *slot == 0) {
        return;
    }
    debug_assert_eq!(out.colour.len(), out.vertices.len());
    debug_assert_eq!(out.partners.len(), out.vertices.len());

    let mut midpoints: FxHashMap<(IVec3, IVec3), u32> = FxHashMap::default();
    let original = std::mem::take(&mut out.indices);
    out.indices.reserve(original.len());

    for triangle in original.as_chunks::<3>().0 {
        let [a, b, c] = *triangle;
        let slots = [out.colour[a as usize], out.colour[b as usize], out.colour[c as usize]];
        if slots[0] == slots[1] && slots[1] == slots[2] {
            out.indices.extend_from_slice(triangle);
            continue;
        }
        let mut mid = |p: u32, q: u32, out: &mut BrickMesh| midpoint(out, &mut midpoints, p, q);

        if slots[0] != slots[1] && slots[1] != slots[2] && slots[0] != slots[2] {
            // Three ways: a corner piece each, and a centre with no owner.
            let ab = mid(a, b, out);
            let bc = mid(b, c, out);
            let ca = mid(c, a, out);
            out.indices.extend_from_slice(&[a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
            continue;
        }
        // One corner differs from the other two. Rotate so it is `c`.
        let (a, b, c) = if slots[0] != slots[1] && slots[0] != slots[2] {
            (b, c, a)
        } else if slots[1] != slots[0] && slots[1] != slots[2] {
            (c, a, b)
        } else {
            (a, b, c)
        };
        let bc = mid(b, c, out);
        let ca = mid(c, a, out);
        // (a, b, bc) and (a, bc, ca) on the pair's side, (c, ca, bc) on the
        // corner's; all three cyclic rotations of pieces of (a, b, c), so the
        // winding stands, and each leads with an original vertex of its side.
        out.indices.extend_from_slice(&[a, b, bc, a, bc, ca, c, ca, bc]);
    }
}

/// The vertex at the middle of edge `p`-`q`, made once per brick.
fn midpoint(
    out: &mut BrickMesh,
    midpoints: &mut FxHashMap<(IVec3, IVec3), u32>,
    p: u32,
    q: u32,
) -> u32 {
    let (p, q) = (p as usize, q as usize);
    let key = {
        let (a, b) = (out.cells[p], out.cells[q]);
        if (a.x, a.y, a.z) <= (b.x, b.y, b.z) { (a, b) } else { (b, a) }
    };
    *midpoints.entry(key).or_insert_with(|| {
        let (vp, vq) = (out.vertices[p], out.vertices[q]);
        let position = (Vec3::from_array(vp.position) + Vec3::from_array(vq.position)) * 0.5;
        let normal = (Vec3::from_array(vp.normal) + Vec3::from_array(vq.normal))
            .try_normalize()
            .unwrap_or(Vec3::from_array(vp.normal));
        out.vertices.push(Vertex { position: position.to_array(), normal: normal.to_array() });
        out.cells.push(key.0);
        out.partners.push(key.1);
        // Protection is continuous, so the midpoint takes the mean; a slot is
        // not, and this one is read only as a provoking vertex of a three-way
        // centre piece, where the lower of the two is the symmetric choice.
        out.mask.push(((u16::from(out.mask[p]) + u16::from(out.mask[q])) / 2) as u8);
        out.colour.push(out.colour[p].min(out.colour[q]));
        (out.vertices.len() - 1) as u32
    })
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

    /// A flat pair of triangles: (0,1,2) and (2,1,3), sharing the edge 1-2.
    fn quad(slots: [u8; 4]) -> BrickMesh {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let mut mesh = BrickMesh::default();
        for (index, position) in positions.into_iter().enumerate() {
            mesh.vertices.push(Vertex { position, normal: [0.0, 0.0, 1.0] });
            let cell = IVec3::new(index as i32, 0, 0);
            mesh.cells.push(cell);
            mesh.partners.push(cell);
            mesh.mask.push((index * 60) as u8);
            mesh.colour.push(slots[index]);
        }
        mesh.indices.extend_from_slice(&[0, 1, 2, 2, 1, 3]);
        mesh
    }

    fn winding_z(mesh: &BrickMesh, triangle: &[u32]) -> f32 {
        let at = |i: u32| Vec3::from_array(mesh.vertices[i as usize].position);
        let (a, b, c) = (at(triangle[0]), at(triangle[1]), at(triangle[2]));
        (b - a).cross(c - a).z
    }

    #[test]
    fn an_unpainted_or_uniformly_painted_mesh_is_left_bit_for_bit() {
        for slots in [[0; 4], [3; 4]] {
            let mut mesh = quad(slots);
            let before = mesh.clone();
            split_paint_boundaries(&mut mesh);
            assert_eq!(mesh.indices, before.indices);
            assert_eq!(mesh.vertices.len(), before.vertices.len());
        }
    }

    /// One corner differs: three pieces, every piece led by an original
    /// vertex of its own side, winding kept, the shared edge cut once.
    #[test]
    fn a_mixed_triangle_is_cut_at_its_midpoints_and_led_by_its_own_side() {
        // Vertex 2 is the odd corner of the first triangle and vertex 3 of the
        // second, so edge 1-2 is mixed in both and 2-3 is not mixed at all.
        let mut mesh = quad([1, 1, 2, 1]);
        split_paint_boundaries(&mut mesh);
        assert_eq!(mesh.triangle_count(), 6, "two mixed triangles make three pieces each");
        assert_eq!(mesh.vertices.len(), 4 + 3, "edges 0-2, 1-2 and 1-3 are cut, 1-2 only once");
        for triangle in mesh.indices.as_chunks::<3>().0 {
            let first = triangle[0] as usize;
            assert!(first < 4, "a piece is led by a midpoint: {triangle:?}");
            assert!(winding_z(&mesh, triangle) > 0.0, "a piece flipped: {triangle:?}");
            // The piece's slot is its first vertex's: the other two originals
            // in it, if any, agree.
            for other in &triangle[1..] {
                if (*other as usize) < 4 {
                    assert_eq!(mesh.colour[*other as usize], mesh.colour[first], "{triangle:?}");
                }
            }
        }
        // Area is conserved, so the pieces tile the originals exactly.
        let area: f32 = mesh.indices.as_chunks::<3>().0.iter().map(|t| winding_z(&mesh, t)).sum();
        assert!((area - 2.0).abs() < 1e-5, "the pieces do not tile the quad: {area}");
        // The midpoint carries the mean mask and a pair key.
        let mid = mesh.vertices.len() - 1;
        assert_ne!(mesh.cells[mid], mesh.partners[mid]);
        assert_eq!(mesh.mask.len(), mesh.vertices.len());
        assert_eq!(mesh.colour.len(), mesh.vertices.len());
    }

    #[test]
    fn a_three_way_corner_makes_four_pieces_that_tile_it() {
        let mut mesh = quad([1, 2, 3, 3]);
        split_paint_boundaries(&mut mesh);
        // Triangle (0,1,2) is three-way: 4 pieces. Triangle (2,1,3) has one
        // odd corner (1): 3 pieces.
        assert_eq!(mesh.triangle_count(), 7);
        let area: f32 = mesh.indices.as_chunks::<3>().0.iter().map(|t| winding_z(&mesh, t)).sum();
        assert!((area - 2.0).abs() < 1e-5, "the pieces do not tile the quad: {area}");
        for triangle in mesh.indices.as_chunks::<3>().0 {
            assert!(winding_z(&mesh, triangle) > 0.0, "a piece flipped: {triangle:?}");
        }
    }

    /// The same edge cut from two bricks yields the same key, whichever end
    /// each brick lists first.
    #[test]
    fn a_midpoints_weld_key_does_not_depend_on_which_end_came_first() {
        let mut one = quad([1, 1, 2, 1]);
        split_paint_boundaries(&mut one);
        let mut two = quad([1, 1, 2, 1]);
        two.indices = vec![1, 2, 0, 1, 3, 2]; // the same triangles, rotated
        split_paint_boundaries(&mut two);
        let keys = |mesh: &BrickMesh| {
            let mut keys: Vec<_> = (4..mesh.vertices.len()).map(|at| mesh.weld_key(at)).collect();
            keys.sort_unstable_by_key(|(a, b)| (a.x, a.y, a.z, b.x, b.y, b.z));
            keys
        };
        assert_eq!(keys(&one), keys(&two));
    }
}
