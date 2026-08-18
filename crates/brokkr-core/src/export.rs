// SPDX-License-Identifier: AGPL-3.0-or-later

//! Turning the sculpt into a mesh fit to print.
//!
//! # Why this is not just a concatenation
//!
//! Bricks are meshed independently, which means the vertices along every brick
//! seam exist twice, once in each neighbour. That is exactly what makes the
//! renderer's job easy and a printer's job impossible: a slicer sees two
//! coincident vertices as two separate surfaces with a crack between them, and
//! refuses the model or silently fills it wrong.
//!
//! So export welds coincident vertices into one, drops the triangles that
//! collapse to nothing in the process, and then checks the result rather than
//! assuming it. Watertight and manifold output is a hard requirement here, not
//! a nice to have.
//!
//! # What is checked
//!
//! A closed surface has every edge shared by exactly two triangles. An edge used
//! once is a hole. An edge used three or more times is a place where more than
//! two surfaces meet, which a slicer cannot resolve into inside and outside.
//! [`ExportMesh::validate`] counts both, and [`Volume::export_mesh`] reports them
//! so a caller can refuse to write a file that would not print.
//!
//! # Units
//!
//! World units are millimetres throughout, so no conversion happens here. STL
//! and OBJ carry no unit information and every slicer assumes millimetres for
//! them. 3MF states it explicitly.
//!
//! # Axes
//!
//! [`ExportMesh`] is in sculpt space, which is Y-up. The three writers rotate to
//! the Z-up the printing formats are read as, each at the moment it writes a
//! vector out, so nothing upstream of a file has to think about it. See
//! [`crate::orientation`] for why that rotation is not the axis swap it looks
//! like.

pub mod obj;
pub mod stl;
pub mod threemf;

use glam::Vec3;
use rustc_hash::FxHashMap;

use crate::mesh::BrickMesh;
use crate::volume::Volume;

/// A single welded mesh, ready to write out.
#[derive(Debug, Default, Clone)]
pub struct ExportMesh {
    pub positions: Vec<Vec3>,
    /// Averaged unit normal per position, for formats that carry them.
    pub normals: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

/// What the mesh turned out to be.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MeshReport {
    pub vertices: usize,
    pub triangles: usize,
    /// Triangles dropped because welding collapsed two of their corners
    /// together, leaving no surface.
    pub collapsed_triangles: usize,
    /// Edges used by exactly one triangle. Every one of these is a hole.
    pub boundary_edges: usize,
    /// Edges used by three or more triangles. A slicer cannot tell inside from
    /// outside across one of these.
    pub non_manifold_edges: usize,
    /// Triangles with three distinct corners but no area. Harmless to most
    /// slicers and not removed, because removing one would open a hole, but
    /// worth reporting as a quality signal.
    pub zero_area_triangles: usize,
}

impl MeshReport {
    /// Whether this mesh is fit to print: closed, and with no edge shared by
    /// more than two triangles.
    pub fn is_printable(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0 && self.triangles > 0
    }

    /// A one line summary for an interface or a log.
    pub fn summary(&self) -> String {
        if self.is_printable() {
            format!("{} triangles, {} vertices, watertight", self.triangles, self.vertices)
        } else {
            format!(
                "{} triangles, {} vertices, NOT watertight: {} holes and {} non manifold edges",
                self.triangles, self.vertices, self.boundary_edges, self.non_manifold_edges
            )
        }
    }
}

impl ExportMesh {
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Count what would stop this mesh printing.
    ///
    /// Edges are compared as unordered pairs, so a triangle wound the other way
    /// still shares an edge with its neighbour. Winding is checked separately by
    /// the tests, since surface nets is consistent about it and a wrong result
    /// there would be a bug rather than something to repair here.
    pub fn validate(&self) -> MeshReport {
        let mut uses: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        uses.reserve(self.triangles.len() * 3);

        let mut zero_area = 0;
        for triangle in &self.triangles {
            let [a, b, c] = *triangle;
            let area = (self.positions[b as usize] - self.positions[a as usize])
                .cross(self.positions[c as usize] - self.positions[a as usize])
                .length();
            if area <= f32::EPSILON {
                zero_area += 1;
            }
            for (from, to) in [(a, b), (b, c), (c, a)] {
                let key = if from <= to { (from, to) } else { (to, from) };
                *uses.entry(key).or_insert(0) += 1;
            }
        }

        MeshReport {
            vertices: self.positions.len(),
            triangles: self.triangles.len(),
            collapsed_triangles: 0,
            boundary_edges: uses.values().filter(|count| **count == 1).count(),
            non_manifold_edges: uses.values().filter(|count| **count > 2).count(),
            zero_area_triangles: zero_area,
        }
    }
}

impl Volume {
    /// Weld every brick's mesh into one, ready to write out.
    ///
    /// Returns the mesh and what it turned out to be. Callers should refuse to
    /// write a file when [`MeshReport::is_printable`] is false rather than
    /// handing a slicer something it cannot use.
    pub fn export_mesh(&self) -> (ExportMesh, MeshReport) {
        let mut coords: Vec<_> = self.brick_coords().collect();
        // The bricks with data, plus a one brick margin: a brick with no voxels
        // of its own still owns the quads on its low faces, and leaving those
        // out would open a seam all the way round the model.
        expand_by_one(&mut coords);

        let mut meshes = vec![BrickMesh::default(); coords.len()];
        self.mesh_bricks(&coords, &mut meshes);

        // Welded on the lattice cell each vertex came from, which is an exact
        // integer identity. See BrickMesh::cells for why welding on positions
        // instead leaves holes.
        let mut welded: FxHashMap<glam::IVec3, u32> = FxHashMap::default();
        let mut mesh = ExportMesh::default();
        let mut normal_sums: Vec<Vec3> = Vec::new();
        let mut collapsed = 0;

        // Rough sizing so the common case does not spend its time growing.
        let total_vertices: usize = meshes.iter().map(|brick| brick.vertices.len()).sum();
        welded.reserve(total_vertices);
        mesh.positions.reserve(total_vertices);
        normal_sums.reserve(total_vertices);
        mesh.triangles.reserve(meshes.iter().map(BrickMesh::triangle_count).sum());

        let mut remap: Vec<u32> = Vec::new();
        for brick in &meshes {
            remap.clear();
            remap.reserve(brick.vertices.len());
            for (vertex, cell) in brick.vertices.iter().zip(brick.cells.iter()) {
                let position = Vec3::from_array(vertex.position);
                let normal = Vec3::from_array(vertex.normal);
                let index = *welded.entry(*cell).or_insert_with(|| {
                    mesh.positions.push(position);
                    normal_sums.push(Vec3::ZERO);
                    (mesh.positions.len() - 1) as u32
                });
                normal_sums[index as usize] += normal;
                remap.push(index);
            }

            for triangle in brick.indices.chunks_exact(3) {
                let a = remap[triangle[0] as usize];
                let b = remap[triangle[1] as usize];
                let c = remap[triangle[2] as usize];
                // Welding pulls some triangles down to a line or a point. Those
                // carry no surface, and leaving them in is what turns a closed
                // mesh into a non manifold one: their edges get counted twice.
                if a == b || b == c || a == c {
                    collapsed += 1;
                    continue;
                }
                mesh.triangles.push([a, b, c]);
            }
        }

        mesh.normals =
            normal_sums.into_iter().map(|sum| sum.try_normalize().unwrap_or(Vec3::Y)).collect();

        let mut report = mesh.validate();
        report.collapsed_triangles = collapsed;
        (mesh, report)
    }
}

/// Grow a brick list by one in every direction.
fn expand_by_one(coords: &mut Vec<crate::brick::BrickCoord>) {
    use crate::brick::BrickCoord;
    use std::collections::BTreeSet;

    let mut set: BTreeSet<BrickCoord> = coords.iter().copied().collect();
    for coord in coords.iter() {
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    set.insert(BrickCoord::new(coord.0.x + dx, coord.0.y + dy, coord.0.z + dz));
                }
            }
        }
    }
    coords.clear();
    coords.extend(set);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp};

    fn sphere(voxel_size: f32, radius: f32) -> Volume {
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::new(radius * 0.1, -radius * 0.2, radius * 0.05), radius);
        volume
    }

    #[test]
    fn a_seeded_sphere_exports_watertight() {
        let volume = sphere(1.0, 24.0);
        let (mesh, report) = volume.export_mesh();

        assert!(report.triangles > 5_000, "expected a substantial mesh: {report:?}");
        assert_eq!(report.boundary_edges, 0, "the mesh has holes: {}", report.summary());
        assert_eq!(report.non_manifold_edges, 0, "the mesh is not manifold: {}", report.summary());
        assert!(report.is_printable());
        assert_eq!(mesh.positions.len(), report.vertices);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
    }

    #[test]
    fn welding_actually_removes_the_seam_duplicates() {
        // Without welding, every brick boundary vertex exists twice. If this
        // stopped working the mesh would still look right on screen and fail in
        // a slicer, so pin the count.
        let volume = sphere(1.0, 24.0);
        let (welded, _) = volume.export_mesh();

        let mut coords: Vec<_> = volume.brick_coords().collect();
        expand_by_one(&mut coords);
        let mut meshes = vec![BrickMesh::default(); coords.len()];
        volume.mesh_bricks(&coords, &mut meshes);
        let unwelded: usize = meshes.iter().map(|mesh| mesh.vertices.len()).sum();

        assert!(
            welded.positions.len() < unwelded,
            "welding removed nothing: {} against {unwelded}",
            welded.positions.len()
        );
    }

    #[test]
    fn a_patterned_model_exports_watertight() {
        // Patterns are the one thing that varies inside a single stamp, so
        // they are also the one thing that can pinch the surface into a
        // non-manifold edge. `is_printable` checks manifoldness as well as
        // closure, which is exactly what a pinch would fail.
        use crate::pattern::{Pattern, PatternKind};

        for kind in PatternKind::ALL {
            let mut volume = sphere(1.0, 24.0);
            let mut scratch = BrushScratch::new();
            let brush = Brush {
                kind: BrushKind::Draw,
                radius: 7.0,
                strength: 0.6,
                // Finer than the field can carry, so the engine's clamp is
                // what keeps this printable.
                pattern: Pattern { kind, scale_mm: 0.01, depth: 1.0 },
                ..Brush::default()
            };

            for at in
                [Vec3::new(24.0, 0.0, 0.0), Vec3::new(0.0, 32.0, 0.0), Vec3::new(-32.0, 0.0, 32.0)]
            {
                let normal = volume.gradient_world(at);
                brush.apply(
                    &mut volume,
                    &Stamp::new(at, normal, BrushDirection::Add).with_tangent(Vec3::X),
                    &mut scratch,
                );
            }

            let (_, report) = volume.export_mesh();
            assert!(
                report.is_printable(),
                "a model patterned with {kind} must still print: {}",
                report.summary()
            );
        }
    }

    #[test]
    fn a_sculpted_model_exports_watertight() {
        // The case that matters: an edited model, including strokes placed on
        // brick corners where the tiling is hardest.
        let mut volume = sphere(1.0, 24.0);
        let mut scratch = BrushScratch::new();

        for kind in BrushKind::ALL {
            let brush = Brush { kind, radius: 7.0, strength: 0.6, ..Brush::default() };
            for at in
                [Vec3::new(24.0, 0.0, 0.0), Vec3::new(0.0, 32.0, 0.0), Vec3::new(-32.0, 0.0, 32.0)]
            {
                let normal = volume.gradient_world(at);
                brush.apply(
                    &mut volume,
                    // Move needs a drag to follow, and a tangent is harmless to
                    // every other brush that is not running a comb pattern.
                    &Stamp::new(at, normal, BrushDirection::Add).with_tangent(Vec3::Y),
                    &mut scratch,
                );
            }
        }

        let (_, report) = volume.export_mesh();
        assert!(report.is_printable(), "a sculpted model must still print: {}", report.summary());
    }

    #[test]
    fn carving_a_model_in_two_still_exports_watertight() {
        // Two separate solids is a perfectly valid thing to print, and a common
        // way to produce a hole in the surface if the tiling is wrong.
        let mut volume = sphere(1.0, 24.0);
        let mut scratch = BrushScratch::new();
        let brush =
            Brush { kind: BrushKind::Inflate, radius: 12.0, strength: 0.9, ..Brush::default() };

        // Cut a trench right through the middle.
        for step in -4..=4 {
            let at = Vec3::new(step as f32 * 6.0, 0.0, 0.0);
            brush.apply(
                &mut volume,
                &Stamp::new(at, Vec3::Y, BrushDirection::Subtract),
                &mut scratch,
            );
        }

        let (_, report) = volume.export_mesh();
        assert!(report.is_printable(), "a carved model must still print: {}", report.summary());
    }

    #[test]
    fn an_empty_volume_exports_nothing_and_says_so() {
        let volume = Volume::new(0.5);
        let (mesh, report) = volume.export_mesh();
        assert!(mesh.is_empty());
        assert_eq!(report.triangles, 0);
        // Nothing to print is not printable, so a caller cannot write an empty
        // file believing it succeeded.
        assert!(!report.is_printable());
    }

    #[test]
    fn the_validator_notices_a_hole() {
        // The check has to be able to fail, or the tests above prove nothing.
        let volume = sphere(1.0, 24.0);
        let (mut mesh, report) = volume.export_mesh();
        assert!(report.is_printable());

        mesh.triangles.pop();
        let holed = mesh.validate();
        assert_eq!(holed.boundary_edges, 3, "removing one triangle should open three edges");
        assert!(!holed.is_printable());
    }

    #[test]
    fn the_validator_notices_a_surface_meeting_itself() {
        let volume = sphere(1.0, 24.0);
        let (mut mesh, _) = volume.export_mesh();

        // Duplicate a triangle, so its three edges are each used twice over.
        let extra = mesh.triangles[0];
        mesh.triangles.push(extra);
        let doubled = mesh.validate();
        assert_eq!(doubled.non_manifold_edges, 3);
        assert!(!doubled.is_printable());
    }

    #[test]
    fn export_scales_with_the_voxel_size_and_stays_watertight() {
        // Finer voxels are the point of the resample operation, and the export
        // has to survive it.
        for voxel_size in [2.0_f32, 1.0, 0.5] {
            let volume = sphere(voxel_size, 20.0);
            let (_, report) = volume.export_mesh();
            assert!(
                report.is_printable(),
                "voxel {voxel_size} did not export cleanly: {}",
                report.summary()
            );
        }
    }

    #[test]
    fn normals_point_outward() {
        // A slicer does not need these, but anything else reading the OBJ does,
        // and an inverted normal is the sort of thing nobody notices until a
        // render looks inside out.
        let volume = sphere(1.0, 24.0);
        let centre = Vec3::new(2.4, -4.8, 1.2);
        let (mesh, _) = volume.export_mesh();

        let mut outward = 0;
        for (position, normal) in mesh.positions.iter().zip(mesh.normals.iter()) {
            if (*position - centre).normalize_or_zero().dot(*normal) > 0.0 {
                outward += 1;
            }
        }
        let fraction = outward as f32 / mesh.positions.len() as f32;
        assert!(fraction > 0.99, "only {:.1}% of normals faced outward", fraction * 100.0);
    }
}
