// SPDX-License-Identifier: AGPL-3.0-only

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

use crate::body::{Document, NodeMeta};
use crate::mesh::BrickMesh;
use crate::volume::Volume;

/// A single welded mesh, ready to write out.
#[derive(Debug, Default, Clone)]
pub struct ExportMesh {
    pub positions: Vec<Vec3>,
    /// Averaged unit normal per position, for formats that carry them.
    pub normals: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// Filament slot per position, 1-based, or 0 for "not assigned".
    ///
    /// **Empty means the whole mesh is unassigned**, which is what every caller
    /// produces today. A writer that carries colour must treat an empty vector
    /// and a vector of zeros the same way rather than indexing blindly.
    ///
    /// A slot, not a colour. What a 3MF carries per triangle is which filament
    /// prints it; the RGB lives in the slicer's own filament setup. See
    /// [`super::export::threemf`] for why that is the contract.
    pub slots: Vec<u8>,
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
    /// Edges used by three or more triangles.
    ///
    /// Reported, but deliberately NOT a reason to refuse an export -- see
    /// [`MeshReport::is_printable`].
    pub non_manifold_edges: usize,
    /// Edges shared by exactly two triangles that both traverse them the SAME
    /// way round, meaning the two disagree about which side is outside.
    ///
    /// **The third of the three defects a slicer calls "non-manifold", and the
    /// one that is easiest to forget.** A mesh can score zero on holes and zero
    /// on over-used edges and still be rejected outright for this: OrcaSlicer
    /// reported exactly 956 on a mesh with none of the other two, counting such
    /// an edge once per adjacent triangle.
    ///
    /// This code has no business producing one -- surface nets winds
    /// consistently -- and measured on a real generated model it produces zero.
    /// It is counted anyway, because "it cannot happen" was the previous
    /// position and an unmeasured assumption is how the scan-line repair came
    /// to be dead for months.
    pub inconsistent_edges: usize,
    /// Triangles with three distinct corners but no area. Harmless to most
    /// slicers and not removed, because removing one would open a hole, but
    /// worth reporting as a quality signal.
    pub zero_area_triangles: usize,
}

impl MeshReport {
    /// Whether this mesh is fit to print: closed, with something in it.
    ///
    /// **Holes are fatal and non manifold edges are not**, which is a
    /// deliberate change from treating them alike, and the evidence is
    /// measured rather than argued. A plane cut through a model puts a handful
    /// of four-way edges exactly along the rim where the flat cut face meets
    /// the old curved surface -- six of them on 49,224 triangles for an oblique
    /// cut through a ball, every one within a quarter of a voxel of the cut
    /// plane. They are a property of meshing a dihedral edge on a uniform
    /// lattice, not a defect in the model: the surface is still closed, still
    /// one part, and still encloses the right volume.
    ///
    /// OrcaSlicer 2.4 reports `manifold = yes` on exactly the export this used
    /// to refuse. Refusing it meant a tool built to make scans printable would
    /// decline to export the result of its own cut tool, which is the worst
    /// possible place for a validator to be stricter than the slicer.
    ///
    /// A hole is different in kind. There is no fill rule that recovers inside
    /// from outside across a boundary edge, so that stays fatal.
    ///
    /// **An inconsistently wound edge is fatal too**, and for the same reason a
    /// hole is: a slicer cannot tell which side of the surface is solid, so it
    /// rejects the file rather than guessing. That this code has never produced
    /// one is not a reason to let it out unchecked.
    ///
    /// The counts are still gathered and still shown by [`MeshReport::summary`],
    /// so a model that is genuinely coming apart is visible rather than silent.
    pub fn is_printable(&self) -> bool {
        self.boundary_edges == 0 && self.inconsistent_edges == 0 && self.triangles > 0
    }

    /// A one line summary for an interface or a log.
    pub fn summary(&self) -> String {
        if self.boundary_edges > 0 {
            return format!(
                "{} triangles, {} vertices, NOT watertight: {} holes",
                self.triangles, self.vertices, self.boundary_edges
            );
        }
        // Closed, but the two sides disagree about which is out. Named
        // separately because "NOT watertight: 0 holes" is the sort of message
        // that sends the next person looking for a hole that is not there.
        if self.inconsistent_edges > 0 {
            return format!(
                "{} triangles, {} vertices, closed but wound inconsistently: \
                 {} edges where neighbouring triangles disagree which side is out",
                self.triangles, self.vertices, self.inconsistent_edges
            );
        }
        if self.triangles == 0 {
            return "nothing to export".to_string();
        }
        if self.non_manifold_edges > 0 {
            // Worth saying out loud even though it does not stop an export, so
            // a model that really is coming apart is not silent.
            return format!(
                "{} triangles, {} vertices, watertight ({} non manifold edges)",
                self.triangles, self.vertices, self.non_manifold_edges
            );
        }
        format!("{} triangles, {} vertices, watertight", self.triangles, self.vertices)
    }

    /// Every body's report added together, for a line that says what the whole
    /// document came to.
    ///
    /// **The verdict is NOT taken from this**, and that is the entire reason
    /// this is a separate function from [`document_verdict`] rather than a
    /// convenience it could be built on. Every field here is additive, so a
    /// document of two bodies -- one with 40,000 triangles and one with none at
    /// all -- sums to 40,000 triangles, zero holes, and
    /// [`MeshReport::is_printable`] returns true. Half the print is missing and
    /// the union says it is fine. `is_printable` asks a mesh whether it is
    /// closed; asking it about a pile of meshes is asking a different question
    /// and getting the wrong answer confidently.
    ///
    /// So: this is for the status line, and [`document_verdict`] decides
    /// whether a file is written.
    pub fn summed(reports: impl IntoIterator<Item = MeshReport>) -> MeshReport {
        let mut total = MeshReport::default();
        for report in reports {
            total.vertices += report.vertices;
            total.triangles += report.triangles;
            total.collapsed_triangles += report.collapsed_triangles;
            total.boundary_edges += report.boundary_edges;
            total.non_manifold_edges += report.non_manifold_edges;
            total.inconsistent_edges += report.inconsistent_edges;
            total.zero_area_triangles += report.zero_area_triangles;
        }
        total
    }
}

/// One exported body: what it is called, what it welded to, and what that
/// turned out to be.
///
/// A tuple rather than a struct because it is exactly the three things every
/// consumer wants together and no fourth is coming: the writers take the name
/// and the mesh, and the verdict takes the name and the report.
pub type ExportedBody = (NodeMeta, ExportMesh, MeshReport);

/// The whole document's verdict, which is every body's and never the union's.
///
/// `Ok(())` only when there is at least one body to write and **every one of
/// them** is fit to print. The error names the body, because "not watertight"
/// over a twelve-body document is not an answer anybody can act on.
///
/// A visible body with nothing in it refuses the whole export rather than being
/// skipped. That is the safe direction and it is deliberate: the alternative is
/// a file that quietly holds fewer parts than the panel shows, which is the
/// failure the omitted count exists to make impossible, arriving through
/// another door.
pub fn document_verdict(bodies: &[ExportedBody]) -> Result<(), String> {
    let Some((meta, _, report)) = bodies.iter().find(|(_, _, report)| !report.is_printable())
    else {
        if bodies.is_empty() {
            return Err("nothing to export -- every body is hidden".to_string());
        }
        return Ok(());
    };
    Err(format!("{}: {}", meta.name, report.summary()))
}

impl Document {
    /// Weld every VISIBLE body into its own mesh, ready to write out.
    ///
    /// `visible` is indexed by NODE position and comes from
    /// [`Document::saved_visibility`] -- never from `display_visibility`, and
    /// the two are named apart precisely so this call site cannot pick up solo
    /// by accident. A view mode silently dropping a part from a print is the
    /// class of failure the eye is being careful about.
    ///
    /// **Welded per body and never through one shared weld map.** The weld key
    /// is a lattice cell, and every body in a document shares the lattice
    /// (that is the whole design), so one map across all of them would fuse two
    /// unrelated bodies into a single vertex wherever their cells coincide.
    /// Where their cells coincide is exactly where they touch or interpenetrate
    /// -- so the failure appears only on the documents where it matters, joins
    /// two parts with triangles neither one has, and produces a mesh that
    /// validates as watertight while being wrong.
    ///
    /// The caller is expected to name how many bodies were left out:
    /// `body_count() - result.len()`. That count is unconditional in the
    /// status line, because the eye is one unprotected bit in a 40-byte record
    /// and it is the bit that decides whether a part reaches the printer.
    pub fn export_bodies(&self, visible: &[bool]) -> Vec<ExportedBody> {
        assert_eq!(
            visible.len(),
            self.node_count(),
            "the visibility mask is indexed by node position and must cover every node"
        );
        self.nodes()
            .iter()
            .zip(visible)
            .filter(|(_, shown)| **shown)
            .filter_map(|(node, _)| {
                let volume = node.volume()?;
                let (mesh, report) = volume.export_mesh();
                Some((node.meta(), mesh, report))
            })
            .collect()
    }
}

impl ExportMesh {
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Count what would stop this mesh printing.
    ///
    /// Edges are compared as unordered pairs, so a triangle wound the other way
    /// still shares an edge with its neighbour -- and the direction each
    /// triangle traverses it in is counted alongside, which is what catches the
    /// pair that agree on the edge and disagree on which side is out.
    ///
    /// All three of the defects a slicer flattens into "non-manifold" are
    /// counted separately, because fixing one leaves the others reporting and
    /// they have different causes: a hole, an over-used edge, and a winding
    /// disagreement are not the same problem.
    pub fn validate(&self) -> MeshReport {
        // Uses of each edge, and how many of those traversed it low-to-high.
        // Two triangles sharing an edge should traverse it in OPPOSITE
        // directions; both going the same way means one of them is flipped.
        let mut uses: FxHashMap<(u32, u32), (u32, u32)> = FxHashMap::default();
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
                let forward = from <= to;
                let key = if forward { (from, to) } else { (to, from) };
                let entry = uses.entry(key).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += u32::from(forward);
            }
        }

        MeshReport {
            vertices: self.positions.len(),
            triangles: self.triangles.len(),
            collapsed_triangles: 0,
            boundary_edges: uses.values().filter(|(count, _)| *count == 1).count(),
            non_manifold_edges: uses.values().filter(|(count, _)| *count > 2).count(),
            inconsistent_edges: uses
                .values()
                .filter(|(count, forward)| *count == 2 && (*forward == 2 || *forward == 0))
                .count(),
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

/// The committed export goldens, and the one mesh they are written from.
///
/// # Why a file on disk rather than a comparison in memory
///
/// Each of the three writers has a `write(mesh)` that is now a one-line wrapper
/// over `write_all(&[(name, mesh)])`. Asserting that those two agree therefore
/// asserts nothing: both sides are the same code, they move together, and a
/// change to the shared body is invisible to the comparison. That was
/// **measured, not argued** -- mutating the OBJ header comment and the STL 80
/// byte banner left the whole workspace suite green.
///
/// So the bytes are pinned against something that cannot move with the code: a
/// file in `tests/fixtures/`, beside the container fixtures and for the same
/// reason. An STL out of a build of this exporter has been sliced and printed;
/// "the bytes did not move" is the cheapest regression check this module has,
/// and it only means anything if one side of the comparison is frozen.
///
/// # Why a hand-written cube rather than a meshed sphere
///
/// The golden pins the *writers*, not the mesher. A mesh taken from
/// [`Volume::export_mesh`] would make every surface-nets change a golden
/// failure, which trains the reader to regenerate rather than to look. [`cube`]
/// is eight literal positions and twelve literal triangles, so the only thing
/// that can move the golden bytes is a writer.
///
/// # Why the coordinates look the way they do
///
/// Every number in [`cube`] is exactly representable as an `f32` and stays
/// exact through [`to_print_space`](crate::orientation::to_print_space), which
/// is an axis swap with a sign flip rather than an arithmetic rotation. So no
/// golden byte depends on rounding, and the face normal the STL writer computes
/// from a cross product comes out as an exact `±1.0` on an axis. The three
/// extents are deliberately 4, 3 and 5 millimetres and the box is off centre,
/// so a writer that swapped or mirrored an axis produces visibly different
/// bytes instead of a symmetric file that still matches.
#[cfg(test)]
pub(crate) mod golden {
    use super::ExportMesh;
    use glam::Vec3;

    /// One component of a unit vector down a cube's body diagonal, near enough
    /// to `1/sqrt(3)`. A literal rather than a computed value: this module is
    /// pinning text that a float turns into, so the float has to be fixed too.
    const OCTANT: f32 = 0.577_350_26;

    /// The mesh every golden is written from: a closed, correctly wound box.
    ///
    /// It is watertight and manifold --- see
    /// [`the_golden_cube_is_a_mesh_the_writers_would_be_allowed_to_write`] ---
    /// because a golden built from a mesh the exporter would have refused would
    /// pin bytes no user can ever receive.
    ///
    /// The slots are mixed on purpose: the 3MF writer takes a triangle's
    /// filament from its first corner, and the golden has to exercise both
    /// branches of that. Slot 0 leaves a `<triangle>` bare, and slots 1, 3 and 4
    /// produce the paint codes `4`, `0C` and `1C` --- one single character and
    /// two double, so a writer that assumed a fixed code width shows up here.
    pub(crate) fn cube() -> ExportMesh {
        let (x0, x1) = (-3.0, 1.0);
        let (y0, y1) = (-0.5, 2.5);
        let (z0, z1) = (0.25, 5.25);

        let positions = vec![
            Vec3::new(x0, y0, z0),
            Vec3::new(x1, y0, z0),
            Vec3::new(x1, y1, z0),
            Vec3::new(x0, y1, z0),
            Vec3::new(x0, y0, z1),
            Vec3::new(x1, y0, z1),
            Vec3::new(x1, y1, z1),
            Vec3::new(x0, y1, z1),
        ];

        // A corner's normal is the average of the three faces meeting there,
        // which for a box is the body diagonal pointing out of that octant.
        let normals = positions
            .iter()
            .map(|position| {
                let sign = |value: f32, low: f32| if value == low { -OCTANT } else { OCTANT };
                Vec3::new(sign(position.x, x0), sign(position.y, y0), sign(position.z, z0))
            })
            .collect();

        // Wound counter-clockwise seen from outside, so the STL writer's face
        // normals agree with the winding rather than fighting it.
        let triangles = vec![
            [4, 5, 6],
            [4, 6, 7], // +Z
            [1, 0, 3],
            [1, 3, 2], // -Z
            [0, 4, 7],
            [0, 7, 3], // -X
            [5, 1, 2],
            [5, 2, 6], // +X
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 1, 5],
            [0, 5, 4], // -Y
        ];

        let slots = vec![0, 1, 2, 3, 4, 0, 2, 3];

        ExportMesh { positions, normals, triangles, slots }
    }

    /// Where the goldens live: beside the container fixtures, because they are
    /// the same kind of thing --- a committed file that does not change when
    /// this code does.
    pub(crate) fn path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    /// Assert that `produced` is byte for byte the committed golden `name`.
    ///
    /// **If this fails, do not reach for the regenerator first.** A golden moves
    /// only when the file a user receives moves, and that is a decision about
    /// what slicers get, not a test to be made green. The message names the
    /// first differing byte so the answer to "what moved" is in the failure
    /// rather than in a diff the reader has to go and produce.
    pub(crate) fn assert_bytes(name: &str, produced: &[u8]) {
        let path = path(name);
        let committed = std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "could not read the committed golden at {}: {error}. If it is missing, write it \
                 once with BROKKR_REGENERATE_EXPORT_GOLDENS=1 -- see \
                 regenerate_the_committed_export_goldens.",
                path.display()
            )
        });

        if let Some(at) = committed.iter().zip(produced).position(|(theirs, ours)| theirs != ours) {
            panic!(
                "{name} differs from what this build writes at byte {at}: the committed golden \
                 holds {:#04x} and this build writes {:#04x}",
                committed[at], produced[at]
            );
        }
        assert_eq!(
            committed.len(),
            produced.len(),
            "{name} is {} bytes and this build writes {} -- the shorter one is a prefix of the \
             other, so a section was added or dropped rather than changed",
            committed.len(),
            produced.len()
        );
    }

    /// Every golden, and the writer call that has to reproduce it.
    ///
    /// The name passed in is `BrokkrSculpt` --- what `write` has always put in
    /// an OBJ `o` line and a 3MF sidecar --- so the goldens pin the file the
    /// previous build produced rather than a file that only the N-body path can
    /// make. A real one-body export names the body instead (`Body 1`), which is
    /// the single byte range that moves; see the byte-identical tests in each
    /// writer for why that narrowing was taken deliberately.
    fn each_golden() -> Vec<(&'static str, Vec<u8>)> {
        let mesh = cube();
        let bodies = [("BrokkrSculpt", &mesh)];

        let mut stl = Vec::new();
        super::stl::write_all(&bodies, &mut stl).expect("writing to a vector cannot fail");
        let mut obj = Vec::new();
        super::obj::write_all(&bodies, &mut obj).expect("writing to a vector cannot fail");
        let mut threemf = Vec::new();
        super::threemf::write_all(&bodies, &mut threemf).expect("writing to a vector cannot fail");

        vec![("export-cube.stl", stl), ("export-cube.obj", obj), ("export-cube.3mf", threemf)]
    }

    /// Writes the three goldens. A one-off act rather than a check, and like
    /// `regenerate_the_committed_fixtures` in `project.rs` it takes *two*
    /// deliberate steps:
    ///
    /// ```text
    /// BROKKR_REGENERATE_EXPORT_GOLDENS=1 cargo test -p brokkr-core --lib -- --ignored regenerate_the_committed_export_goldens
    /// ```
    ///
    /// The variable is deliberately **not** the `BROKKR_REGENERATE_FIXTURES`
    /// that the container fixtures use, even though both live in the same
    /// directory. `--ignored` is a sweep rather than a per-test opt-in, so one
    /// shared variable would mean that anyone legitimately regenerating a
    /// container fixture also silently rewrote all three export goldens from
    /// the build in front of them --- which is precisely the tautology this
    /// module exists to remove. Two variables, two decisions.
    #[test]
    #[ignore = "regenerates the committed export goldens; running it is a decision, not a check"]
    fn regenerate_the_committed_export_goldens() {
        if std::env::var_os("BROKKR_REGENERATE_EXPORT_GOLDENS").is_none() {
            eprintln!(
                "refusing to rewrite the committed export goldens as part of an --ignored sweep: \
                 set BROKKR_REGENERATE_EXPORT_GOLDENS=1 if you really mean to replace them with \
                 what this build writes. They record the bytes a slicer has already been given."
            );
            return;
        }

        let directory = path("");
        std::fs::create_dir_all(&directory).expect("could not make the fixtures directory");
        for (name, bytes) in each_golden() {
            let path = path(name);
            std::fs::write(&path, &bytes).expect("could not write the golden");
            eprintln!("wrote {} bytes to {}", bytes.len(), path.display());
        }
    }

    /// The fixture is a mesh this exporter would actually agree to write.
    ///
    /// Without this the golden could quietly be pinning the bytes of a model
    /// that [`crate::Brokkr::export`] refuses, and the regression check would be
    /// guarding a file no user can ever receive.
    #[test]
    fn the_golden_cube_is_a_mesh_the_writers_would_be_allowed_to_write() {
        let mesh = cube();
        assert_eq!(mesh.normals.len(), mesh.positions.len(), "OBJ needs a normal per vertex");
        assert_eq!(mesh.slots.len(), mesh.positions.len(), "3MF indexes slots by vertex");

        let report = mesh.validate();
        assert!(
            report.is_printable(),
            "the golden cube would be refused by the exporter: {}",
            report.summary()
        );
        assert_eq!(report.boundary_edges, 0, "the golden cube has holes: {}", report.summary());
    }
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

    /// The control for the third defect, and the reason it is now counted.
    ///
    /// A slicer flattens holes, over-used edges and winding disagreements into
    /// one phrase, and a mesh can be clean of the first two and still be
    /// rejected for this. Nothing in this codebase should produce one -- surface
    /// nets winds consistently, and a real generated model measured zero -- but
    /// "cannot happen" was also the standing position on the scan-line repair,
    /// which turned out to be unreachable from the path that mattered.
    #[test]
    fn the_validator_notices_a_triangle_wound_the_wrong_way() {
        let volume = sphere(1.0, 24.0);
        let (mut mesh, report) = volume.export_mesh();
        assert!(report.is_printable());
        assert_eq!(report.inconsistent_edges, 0, "our own meshing should wind consistently");

        // Turn one triangle over. It still shares all three of its edges with
        // the same neighbours, so this opens no hole and over-uses no edge:
        // the ONLY signal is the direction each edge is traversed in.
        let flipped = mesh.triangles[0];
        mesh.triangles[0] = [flipped[0], flipped[2], flipped[1]];
        let wrong = mesh.validate();

        assert_eq!(wrong.boundary_edges, 0, "flipping a triangle should not open a hole");
        assert_eq!(wrong.non_manifold_edges, 0, "flipping a triangle should not over-use an edge");
        assert_eq!(
            wrong.inconsistent_edges, 3,
            "a flipped triangle disagrees with each of its three neighbours"
        );
        assert!(
            !wrong.is_printable(),
            "a mesh that disagrees about which side is out is not fit to print"
        );
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

    /// A surface meeting itself must still be *noticed*, even though it is no
    /// longer a reason to refuse an export.
    ///
    /// The count is the quality signal and it has to keep working. What
    /// changed is only what is done about it: OrcaSlicer 2.4 reports
    /// `manifold = yes` on this exact mesh, one part, correct volume, so
    /// refusing to write it would have been our validator overruling the
    /// slicer the model is going to.
    #[test]
    fn the_validator_notices_a_surface_meeting_itself() {
        let volume = sphere(1.0, 24.0);
        let (mut mesh, _) = volume.export_mesh();

        // Duplicate a triangle, so its three edges are each used twice over.
        let extra = mesh.triangles[0];
        mesh.triangles.push(extra);
        let doubled = mesh.validate();
        assert_eq!(doubled.non_manifold_edges, 3, "the doubled face was not detected");
        assert!(
            doubled.summary().contains("non manifold"),
            "it is detected but not reported, so nobody would ever see it: {}",
            doubled.summary()
        );
        assert!(
            doubled.is_printable(),
            "a non manifold edge blocked an export again -- see is_printable's doc comment \
             before changing this back"
        );
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

    // --- the document path -------------------------------------------------

    use crate::body::Document;

    /// A document of `count` bodies, each a sphere, laid out along X so they do
    /// not touch. Returns it with the ids in display order.
    fn document_of(count: usize) -> Document {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 12.0);
        for index in 1..count {
            let mut volume = Volume::new(1.0);
            volume.seed_sphere(Vec3::new(index as f32 * 64.0, 0.0, 0.0), 12.0);
            doc.add_body(format!("Body {}", index + 1), volume);
        }
        doc
    }

    #[test]
    fn the_export_omits_hidden_bodies_and_keeps_the_rest_in_order() {
        let doc = document_of(4);
        let hidden = doc.nodes()[2].id;
        let mut visible = vec![true; doc.node_count()];
        visible[2] = false;

        let bodies = doc.export_bodies(&visible);
        assert_eq!(bodies.len(), 3, "one of four bodies was hidden");
        assert!(
            bodies.iter().all(|(meta, _, _)| meta.id != hidden),
            "a hidden body reached the export"
        );
        let names: Vec<&str> = bodies.iter().map(|(meta, _, _)| meta.name.as_str()).collect();
        assert_eq!(names, vec!["Body 1", "Body 2", "Body 4"], "display order was not kept");
        for (_, mesh, report) in &bodies {
            assert!(report.is_printable(), "{}", report.summary());
            assert!(!mesh.is_empty());
        }
        // The caller's arithmetic for the omitted count, which is the line the
        // whole defence rests on.
        assert_eq!(doc.body_count() - bodies.len(), 1);
    }

    /// **Each body is welded on its own, and the proof is that it comes out
    /// exactly as it would alone.**
    ///
    /// The weld key is a lattice cell and every body shares the lattice, so one
    /// map across the document would fuse two bodies into a single vertex
    /// wherever their cells coincide. The fixture is two spheres that
    /// interpenetrate, because that is where their cells DO coincide -- two
    /// bodies laid out apart would pass this test with a shared map and prove
    /// nothing at all.
    #[test]
    fn two_bodies_sharing_lattice_cells_are_welded_separately() {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 12.0);
        let mut through = Volume::new(1.0);
        through.seed_sphere(Vec3::new(6.0, 0.0, 0.0), 12.0);
        doc.add_body("Body 2", through);
        assert!(!doc.overlaps().is_empty(), "the fixture must actually interpenetrate");

        let bodies = doc.export_bodies(&vec![true; doc.node_count()]);
        assert_eq!(bodies.len(), 2);
        for ((meta, mesh, report), (_, volume)) in bodies.iter().zip(doc.bodies()) {
            let (alone, alone_report) = volume.export_mesh();
            assert_eq!(
                mesh.positions, alone.positions,
                "{} welded differently as part of a document",
                meta.name
            );
            assert_eq!(mesh.triangles, alone.triangles, "{}", meta.name);
            assert_eq!(*report, alone_report, "{}", meta.name);
        }
    }

    /// **The verdict is per body and the summary is over the union, and this is
    /// the case that tells them apart.**
    ///
    /// A document whose second body is empty sums to a report with triangles,
    /// no holes and no winding disagreements -- `is_printable` on the sum says
    /// yes -- while half of what the panel shows would be missing from the
    /// print. Asking a pile of meshes whether it is closed is a different
    /// question from asking each one, and it is the one with the comfortable
    /// wrong answer.
    #[test]
    fn a_document_with_an_empty_body_is_refused_even_though_the_union_reads_printable() {
        let mut doc = Document::new(1.0);
        doc.active_volume_mut().seed_sphere(Vec3::ZERO, 12.0);
        doc.add_body("Body 2", Volume::new(1.0));

        let bodies = doc.export_bodies(&vec![true; doc.node_count()]);
        let union = MeshReport::summed(bodies.iter().map(|(_, _, report)| *report));
        assert!(
            union.is_printable(),
            "the union has to look fine, or this test is not about what it says it is"
        );

        let refusal = document_verdict(&bodies).expect_err("an empty body must refuse the export");
        assert!(refusal.starts_with("Body 2"), "the refusal must name the body: {refusal}");
        assert!(refusal.contains("nothing to export"), "{refusal}");
    }

    #[test]
    fn a_document_whose_bodies_all_print_is_admitted() {
        let doc = document_of(3);
        let bodies = doc.export_bodies(&vec![true; doc.node_count()]);
        assert_eq!(document_verdict(&bodies), Ok(()));
    }

    /// Hiding everything is refused rather than silently writing an empty file.
    #[test]
    fn a_document_with_every_body_hidden_is_refused_and_says_why() {
        let doc = document_of(2);
        let bodies = doc.export_bodies(&vec![false; doc.node_count()]);
        assert!(bodies.is_empty());
        let refusal = document_verdict(&bodies).expect_err("there is nothing to write");
        assert!(refusal.contains("every body is hidden"), "{refusal}");
    }

    /// A folder row is not a body, so it is neither exported nor counted as
    /// omitted -- and a folder whose eye is off takes its children out of the
    /// print, which is the one place the composition rule has real consequences.
    ///
    /// **The omitted count is the line the whole defence rests on**, and a
    /// folder is the way to get it wrong: `node_count() - result.len()` would
    /// report a phantom omission for every folder in the document, and a user
    /// told "1 body omitted" who hid nothing has been taught to ignore the
    /// line.
    #[test]
    fn a_hidden_folder_takes_its_children_out_of_the_export_and_counts_only_them() {
        let mut doc = document_of(3);
        let second = doc.nodes()[1].id;
        let (folder, _) = doc.group(second, "Group 1").expect("the group");

        // Everything shown: the folder itself contributes no mesh and no
        // omission.
        let mut visible = Vec::new();
        doc.saved_visibility(&mut visible);
        let bodies = doc.export_bodies(&visible);
        assert_eq!(bodies.len(), 3, "a folder row was counted as a body");
        assert_eq!(doc.body_count() - bodies.len(), 0, "a folder was reported as omitted");

        // The folder's eye off, with the child's own eye untouched.
        let meta = doc.meta(folder).expect("the folder");
        doc.set_meta(&crate::body::NodeMeta { visible: false, ..meta });
        assert!(doc.node(second).expect("the child").visible, "the child's own bit was written");

        doc.saved_visibility(&mut visible);
        let bodies = doc.export_bodies(&visible);
        assert_eq!(bodies.len(), 2, "the folder's eye did not reach its child");
        assert!(bodies.iter().all(|(meta, _, _)| meta.id != second));
        assert_eq!(doc.body_count() - bodies.len(), 1, "the omitted count is not the child alone");
    }

    #[test]
    fn the_summed_report_adds_every_field() {
        let one = MeshReport {
            vertices: 3,
            triangles: 5,
            collapsed_triangles: 7,
            boundary_edges: 11,
            non_manifold_edges: 13,
            inconsistent_edges: 17,
            zero_area_triangles: 19,
        };
        let total = MeshReport::summed([one, one, one]);
        assert_eq!(
            total,
            MeshReport {
                vertices: 9,
                triangles: 15,
                collapsed_triangles: 21,
                boundary_edges: 33,
                non_manifold_edges: 39,
                inconsistent_edges: 51,
                zero_area_triangles: 57,
            },
            "a field added to MeshReport and not to `summed` would show up here"
        );
    }
}
