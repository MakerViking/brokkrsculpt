// SPDX-License-Identifier: AGPL-3.0-or-later

//! The seam test.
//!
//! Meshing a brick without the one voxel halo from its neighbours leaves the
//! surface a voxel short on every face, so the model shows a crack at every
//! brick boundary. That is the failure the apron rule exists to prevent and it
//! is the thing most likely to go wrong, so it gets a direct test rather than a
//! visual check.
//!
//! The property asserted is that the union of the per brick meshes is closed:
//! every triangle edge is shared by exactly two triangles. A crack shows up as
//! an edge used only once.

use std::collections::HashMap;

use brokkr_core::{BrickCoord, BrickMesh, MeshScratch, Vertex, Volume};
use glam::{IVec3, Vec3};

/// Weld tolerance. Two bricks compute the shared vertex on their common face
/// from identical corner values, so the results agree to within float rounding
/// and this is generous by several orders of magnitude.
const WELD: f64 = 1.0e-3;

type WeldedVertex = (i64, i64, i64);

fn weld(vertex: &Vertex) -> WeldedVertex {
    let q = |v: f32| ((v as f64) / WELD).round() as i64;
    (q(vertex.position[0]), q(vertex.position[1]), q(vertex.position[2]))
}

/// Every brick coordinate that could carry geometry: the bounding box of the
/// stored bricks, grown by one so the absent bricks that own boundary quads are
/// included too.
fn brick_range(volume: &Volume) -> Vec<BrickCoord> {
    let mut min = IVec3::splat(i32::MAX);
    let mut max = IVec3::splat(i32::MIN);
    for coord in volume.brick_coords() {
        min = min.min(coord.0);
        max = max.max(coord.0);
    }
    assert!(min.x <= max.x, "volume has no bricks");

    let mut coords = Vec::new();
    for z in min.z - 1..=max.z + 1 {
        for y in min.y - 1..=max.y + 1 {
            for x in min.x - 1..=max.x + 1 {
                coords.push(BrickCoord::new(x, y, z));
            }
        }
    }
    coords
}

/// Mesh the given bricks and count how many triangles use each undirected edge.
fn edge_use_counts(
    volume: &Volume,
    coords: &[BrickCoord],
) -> (HashMap<(WeldedVertex, WeldedVertex), u32>, usize) {
    let mut scratch = MeshScratch::new();
    let mut mesh = BrickMesh::default();
    let mut edges: HashMap<(WeldedVertex, WeldedVertex), u32> = HashMap::new();
    let mut triangles = 0;

    for &coord in coords {
        volume.mesh_brick(coord, &mut scratch, &mut mesh);
        for triangle in mesh.indices.chunks_exact(3) {
            let v: Vec<WeldedVertex> =
                triangle.iter().map(|i| weld(&mesh.vertices[*i as usize])).collect();
            // A degenerate triangle contributes no surface and no real edges.
            if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
                continue;
            }
            triangles += 1;
            for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                let key = if a <= b { (a, b) } else { (b, a) };
                *edges.entry(key).or_insert(0) += 1;
            }
        }
    }
    (edges, triangles)
}

fn sphere_spanning_several_bricks() -> Volume {
    // Voxel size 1, radius 40, centred so the sphere straddles brick boundaries
    // on all three axes rather than sitting neatly inside one brick.
    let mut volume = Volume::new(1.0);
    volume.seed_sphere(Vec3::new(48.0, 48.0, 48.0), 40.0);
    volume
}

#[test]
fn per_brick_meshing_produces_a_closed_surface() {
    let volume = sphere_spanning_several_bricks();
    let coords = brick_range(&volume);
    let (edges, triangles) = edge_use_counts(&volume, &coords);

    assert!(triangles > 10_000, "expected a substantial sphere, got {triangles} triangles");

    let open: Vec<_> = edges.iter().filter(|(_, count)| **count != 2).collect();
    assert!(
        open.is_empty(),
        "{} of {} edges are not shared by exactly two triangles, so the surface has cracks. \
         First few: {:?}",
        open.len(),
        edges.len(),
        &open.iter().take(5).collect::<Vec<_>>()
    );
}

#[test]
fn the_seam_test_can_actually_detect_a_gap() {
    // Control: meshing a single brick out of a sphere that spans many must
    // leave that brick's patch open at its rim. If this passed, the test above
    // would prove nothing.
    let volume = sphere_spanning_several_bricks();

    // Find a brick the shell actually crosses. Most bricks here are either
    // fully interior or fully exterior and carry no triangles at all.
    let mut scratch = MeshScratch::new();
    let mut mesh = BrickMesh::default();
    let shell = brick_range(&volume)
        .into_iter()
        .find(|coord| {
            volume.mesh_brick(*coord, &mut scratch, &mut mesh);
            mesh.triangle_count() > 100
        })
        .expect("the sphere shell must cross some brick");

    let (edges, triangles) = edge_use_counts(&volume, &[shell]);

    assert!(triangles > 0, "brick {shell:?} should carry part of the sphere");
    let open = edges.values().filter(|count| **count != 2).count();
    assert!(open > 0, "a lone brick patch must have an open rim, so the closure check is real");
}

#[test]
fn sculpting_keeps_the_surface_closed() {
    // Cracks that only appear after an edit are the more likely failure: the
    // edit dirties bricks and the remesh has to keep the tiling consistent.
    use brokkr_core::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp};

    let mut volume = sphere_spanning_several_bricks();
    let mut scratch = BrushScratch::new();

    // Every brush, and deliberately stamped right on brick corners, which is
    // the worst case for tiling.
    for (index, kind) in BrushKind::ALL.into_iter().enumerate() {
        let brush = Brush { kind, radius: 9.0, strength: 0.5, ..Brush::default() };
        for (centre, direction) in [
            (Vec3::new(64.0, 64.0, 64.0), BrushDirection::Add),
            (Vec3::new(32.0, 32.0, 48.0), BrushDirection::Subtract),
            (Vec3::new(48.0, 8.0, 48.0 + index as f32), BrushDirection::Add),
        ] {
            let normal = volume.gradient_world(centre);
            brush.apply(
                &mut volume,
                // Move is steered by the drag rather than the surface, and
                // without one it would sit this test out entirely.
                &Stamp::new(centre, normal, direction).with_tangent(Vec3::Y),
                &mut scratch,
            );
        }
    }

    let coords = brick_range(&volume);
    let (edges, triangles) = edge_use_counts(&volume, &coords);
    assert!(triangles > 10_000);

    let open: Vec<_> = edges.iter().filter(|(_, count)| **count != 2).collect();
    assert!(
        open.is_empty(),
        "sculpting opened {} of {} edges. First few: {:?}",
        open.len(),
        edges.len(),
        &open.iter().take(5).collect::<Vec<_>>()
    );
}

#[test]
fn remeshing_only_dirty_bricks_matches_remeshing_everything() {
    // The performance property is that a stroke remeshes only what it touched.
    // That is only safe if the dirty set is complete: anything it misses stays
    // stale on screen. Compare an incremental remesh against a full one.
    use brokkr_core::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp};

    let mut volume = sphere_spanning_several_bricks();
    let all = brick_range(&volume);

    let mut scratch = MeshScratch::new();
    let mut mesh = BrickMesh::default();

    // Full mesh of the seeded sphere, keyed by brick.
    let mut reference: HashMap<BrickCoord, Vec<WeldedVertex>> = HashMap::new();
    for &coord in &all {
        volume.mesh_brick(coord, &mut scratch, &mut mesh);
        reference.insert(coord, mesh.vertices.iter().map(weld).collect());
    }

    // Sculpt, then remesh only what the volume reported as dirty.
    let mut dirty = Vec::new();
    volume.take_dirty(&mut dirty);
    let brush = Brush { kind: BrushKind::Draw, radius: 7.0, strength: 0.35, ..Brush::default() };
    let at = Vec3::new(48.0, 88.0, 48.0);
    let normal = volume.gradient_world(at);
    brush.apply(
        &mut volume,
        &Stamp::new(at, normal, BrushDirection::Add),
        &mut BrushScratch::new(),
    );
    volume.take_dirty(&mut dirty);

    assert!(!dirty.is_empty(), "a stroke must dirty something");
    assert!(
        dirty.len() < all.len(),
        "a small stroke dirtied {} of {} bricks, which is not proportional to the brush",
        dirty.len(),
        all.len()
    );

    for &coord in &dirty {
        volume.mesh_brick(coord, &mut scratch, &mut mesh);
        reference.insert(coord, mesh.vertices.iter().map(weld).collect());
    }

    // Now the incremental result must equal a from scratch mesh of everything.
    for &coord in &all {
        volume.mesh_brick(coord, &mut scratch, &mut mesh);
        let expected: Vec<WeldedVertex> = mesh.vertices.iter().map(weld).collect();
        let actual = reference.get(&coord).expect("every brick was meshed");
        assert_eq!(
            actual, &expected,
            "brick {coord:?} is stale after an incremental remesh, so the dirty set missed it"
        );
    }
}

#[test]
fn patterned_sculpting_keeps_the_surface_closed() {
    // Deliberately asks for a feature finer than the field can carry. The
    // engine clamps it up to MIN_SCALE_VOXELS, and this test is what proves
    // the clamp is doing its job: without it, one edge comes back shared by
    // four triangles and the export validator would refuse the model.
    const REQUESTED_SCALE: f32 = 0.01;
    // A pattern is the one thing that varies *within* a stamp rather than
    // between stamps, so if anything could put a discontinuity across a brick
    // boundary it is this. Full depth, and a feature size of a couple of
    // voxels, which is the harshest setting the interface offers.
    use brokkr_core::{
        Brush, BrushDirection, BrushKind, BrushScratch, Pattern, PatternKind, Stamp,
    };

    let mut volume = sphere_spanning_several_bricks();
    let mut scratch = BrushScratch::new();

    for (index, kind) in PatternKind::ALL.into_iter().enumerate() {
        let brush = Brush {
            kind: BrushKind::Draw,
            radius: 9.0,
            strength: 0.6,
            pattern: Pattern { kind, scale_mm: REQUESTED_SCALE, depth: 1.0 },
            ..Brush::default()
        };
        // Right on brick corners, which is the worst case for tiling.
        for (centre, direction) in [
            (Vec3::new(64.0, 64.0, 64.0), BrushDirection::Add),
            (Vec3::new(32.0, 32.0, 48.0), BrushDirection::Subtract),
            (Vec3::new(48.0, 8.0, 48.0 + index as f32), BrushDirection::Add),
        ] {
            let normal = volume.gradient_world(centre);
            brush.apply(
                &mut volume,
                &Stamp::new(centre, normal, direction).with_tangent(Vec3::X),
                &mut scratch,
            );
        }
    }

    let coords = brick_range(&volume);
    let (edges, triangles) = edge_use_counts(&volume, &coords);
    assert!(triangles > 10_000);

    let open: Vec<_> = edges.iter().filter(|(_, count)| **count != 2).collect();
    assert!(
        open.is_empty(),
        "patterned sculpting opened {} of {} edges. First few: {:?}",
        open.len(),
        edges.len(),
        &open.iter().take(5).collect::<Vec<_>>()
    );
}

/// An imported model has to tile as cleanly as a sculpted one.
///
/// This is the specific guard for the voxeliser's binning invariant. A brick is
/// binned only when a triangle reaches its band-expanded box; if that test were
/// ever narrowed, the bricks just outside the shell would stop being filled and
/// the band would be truncated at a brick face. What that produces is a flat
/// facet aligned to the 32 voxel grid -- which renders perfectly, exports with a
/// plausible triangle count, and is invisible to everything except an edge
/// count. The mesh is deliberately built off the lattice and spanning several
/// brick boundaries so the seams have somewhere to go wrong.
#[test]
fn a_voxelised_import_produces_a_closed_surface() {
    use brokkr_core::ExportMesh;
    use brokkr_core::voxelise::{VoxeliseOptions, voxelise};

    // A cube deliberately NOT aligned to the lattice, big enough to span
    // several bricks on every axis.
    let low = Vec3::new(-17.3, -21.7, -14.1);
    let high = Vec3::new(19.9, 15.3, 22.7);
    let c = [
        Vec3::new(low.x, low.y, low.z),
        Vec3::new(high.x, low.y, low.z),
        Vec3::new(high.x, high.y, low.z),
        Vec3::new(low.x, high.y, low.z),
        Vec3::new(low.x, low.y, high.z),
        Vec3::new(high.x, low.y, high.z),
        Vec3::new(high.x, high.y, high.z),
        Vec3::new(low.x, high.y, high.z),
    ];
    let faces: [[u32; 3]; 12] = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];
    let mesh = ExportMesh {
        positions: c.to_vec(),
        normals: Vec::new(),
        triangles: faces.to_vec(),
        slots: Vec::new(),
    };

    let (volume, report) =
        voxelise(&mesh, &VoxeliseOptions::at(0.25)).expect("an off-lattice cube should voxelise");
    assert!(report.uniform_bricks > 0, "the import came back hollow");

    let coords = brick_range(&volume);
    let (edges, triangles) = edge_use_counts(&volume, &coords);
    assert!(triangles > 10_000, "only {triangles} triangles, so this proves little");

    let open: Vec<_> = edges.iter().filter(|(_, count)| **count != 2).collect();
    assert!(
        open.is_empty(),
        "a voxelised import left {} of {} edges open. First few: {:?}",
        open.len(),
        edges.len(),
        &open.iter().take(5).collect::<Vec<_>>()
    );
}
