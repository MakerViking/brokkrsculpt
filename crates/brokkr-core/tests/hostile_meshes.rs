// SPDX-License-Identifier: AGPL-3.0-only

//! The mesh readers against files that are wrong on purpose.
//!
//! Import is the one place in this application where bytes nobody here wrote
//! reach real parsing code, and the models it exists to open are *defective
//! scans* -- so a malformed file is the expected case, not the exotic one. A
//! panic here takes the whole window down and loses the sculpt, because the
//! read runs inside an iced `Task` on the process that owns the surface.
//!
//! The property under test is therefore not "bad input is rejected" but the
//! weaker and more important one: **every input produces an answer**. Either an
//! [`ImportError`], or a mesh whose indices are in bounds and whose positions
//! are finite -- because an out of range index panics the voxeliser and a NaN
//! coordinate silently fills the field with NaN, which meshes into nothing and
//! reads as an empty model rather than as a bad file.
//!
//! Deterministic rather than random: the same seeds run on every machine and in
//! CI, so a failure here can be reproduced from the seed printed with it. That
//! is the trade against a real fuzzer, which explores further but cannot be a
//! gate. This is the gate; `cargo fuzz` remains worth running by hand.

use brokkr_core::export::ExportMesh;
use brokkr_core::import::{obj, stl, threemf};
use brokkr_core::voxelise::{VoxeliseOptions, voxelise};
use glam::{IVec3, Vec3};

/// Mutants generated per seed file per strategy.
///
/// Sized so the whole file stays around a second in a debug build, which is
/// what `cargo test` runs. Raising it explores more and is the first thing to
/// try when hardening this further.
const MUTANTS: usize = 400;

/// A seeded xorshift, so a failure is reproducible from the seed alone.
///
/// Hand rolled rather than pulled in: `rand` is not a dependency of this
/// workspace and adding one to shuffle bytes in a test would be a poor trade.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, limit: usize) -> usize {
        if limit == 0 { 0 } else { (self.next() % limit as u64) as usize }
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
}

/// A small closed mesh to corrupt: a tetrahedron.
fn seed_mesh() -> ExportMesh {
    ExportMesh {
        positions: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
        ],
        normals: Vec::new(),
        triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        slots: Vec::new(),
    }
}

/// A reader, as the free function the crate exposes.
type Reader = fn(&[u8]) -> Result<ExportMesh, brokkr_core::ImportError>;

/// Every reader, with a valid file of its own format to start from.
fn readers() -> Vec<(&'static str, Vec<u8>, Reader)> {
    let mesh = seed_mesh();

    let mut binary_stl = Vec::new();
    brokkr_core::export::stl::write(&mesh, &mut binary_stl).expect("writing to a Vec cannot fail");

    let mut wavefront = Vec::new();
    brokkr_core::export::obj::write(&mesh, &mut wavefront).expect("writing to a Vec cannot fail");

    let mut package = Vec::new();
    brokkr_core::export::threemf::write(&mesh, &mut package).expect("writing to a Vec cannot fail");

    let ascii_stl = b"solid thing
 facet normal 0 0 1
  outer loop
   vertex 0 0 0
   vertex 1 0 0
   vertex 0 1 0
  endloop
 endfacet
endsolid thing
"
    .to_vec();

    vec![
        ("binary stl", binary_stl, stl::read as Reader),
        ("ascii stl", ascii_stl, stl::read as Reader),
        ("obj", wavefront, obj::read as Reader),
        ("3mf", package, threemf::read as Reader),
    ]
}

/// What a reader that answered `Ok` has to have produced.
///
/// Checked here rather than trusted, because both failures are silent
/// downstream: an index past the end of `positions` panics inside the
/// voxeliser's triangle loop, and a NaN corner poisons every distance computed
/// from it, which meshes into no triangles at all and looks like an empty file.
fn check(mesh: &ExportMesh, what: &str) {
    for (index, triangle) in mesh.triangles.iter().enumerate() {
        for corner in triangle {
            assert!(
                (*corner as usize) < mesh.positions.len(),
                "{what}: triangle {index} refers to vertex {corner} of {}",
                mesh.positions.len()
            );
        }
    }
    for (index, position) in mesh.positions.iter().enumerate() {
        assert!(position.is_finite(), "{what}: vertex {index} is {position:?}");
    }
}

/// Run one candidate file through a reader and hold it to the property.
///
/// Returns whether it was accepted, which is what the coverage control below
/// counts: a fuzz whose every mutant is rejected at the first length check has
/// not tested the parser, only the guard in front of it.
fn feed(read: Reader, bytes: &[u8], what: &str) -> bool {
    match read(bytes) {
        Ok(mesh) => {
            check(&mesh, what);
            voxelise_it(&mesh, what);
            true
        }
        Err(_) => false,
    }
}

/// The rest of the import, which is where a mesh that merely *parsed* has to
/// survive being turned into a field.
///
/// Reading is only half the path a hostile file travels: `File > Import mesh`
/// reads and then voxelises, on the same task, and a panic in the second half
/// loses the sculpt exactly as surely as one in the first. The reader's own
/// checks do not cover what the voxeliser cares about -- a coordinate of 1e20
/// is perfectly finite, and a sliver at that distance has an area the
/// preflight is happy with while sitting where the voxel lattice cannot
/// address it.
fn voxelise_it(mesh: &ExportMesh, what: &str) {
    // Coarse deliberately: the point is to reach the grid arithmetic, not to
    // spend the test budget building fields.
    let options = VoxeliseOptions::at(1.0);
    let Ok((volume, report)) = voxelise(mesh, &options) else {
        return;
    };
    for coord in volume.brick_coords() {
        // What the lattice has to be able to say about any brick it stores.
        // This is the arithmetic that overflowed in `project::read`.
        let origin = coord.origin();
        assert!(
            origin.max(coord.max_voxel()).cmplt(IVec3::splat(i32::MAX)).all(),
            "{what}: a brick landed at {coord:?}, whose own voxels do not fit the lattice"
        );
    }
    assert!(
        report.longest_mm.is_finite() && report.longest_mm < RESCALED_CEILING_MM,
        "{what}: it built a model {} mm across",
        report.longest_mm
    );
}

/// A ceiling on what this file will accept as a built model, in millimetres.
///
/// Not the engine's own `IMPLAUSIBLY_LARGE_MM`, which is 500 and is where
/// `refit_if_implausible` starts rescaling. This is deliberately looser: the
/// property being pinned is that an implausible model comes back *rescaled to
/// something the lattice can hold*, not that it lands on any particular size,
/// and pinning the engine's exact threshold from out here would break this
/// file every time that number is retuned.
const RESCALED_CEILING_MM: f32 = 10_000.0;

#[test]
fn a_corrupted_file_is_answered_rather_than_survived() {
    for (format, valid, read) in readers() {
        // The seed itself has to read, or the mutations below are all
        // measuring the same rejection.
        let mesh = read(&valid).unwrap_or_else(|error| panic!("the {format} seed is bad: {error}"));
        check(&mesh, format);

        let mut accepted = 0usize;
        let mut tried = 0usize;
        for strategy in 0..5 {
            for run in 0..MUTANTS {
                let seed = (strategy as u64) << 32 | run as u64 | 1;
                let mut noise = Noise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut bytes = valid.clone();
                let what = &format!("{format}, strategy {strategy}, seed {seed}");

                match strategy {
                    // A handful of flipped bits, which is what a bad download
                    // or a failing disk actually produces.
                    0 => {
                        for _ in 0..1 + noise.below(8) {
                            let at = noise.below(bytes.len());
                            bytes[at] ^= 1 << noise.below(8);
                        }
                    }
                    // Cut short. The one every length field and every fixed
                    // size record has to survive.
                    1 => {
                        let keep = noise.below(bytes.len());
                        bytes.truncate(keep);
                    }
                    // Whole runs replaced, which is how a header's counts and
                    // sizes end up describing something that is not there.
                    2 => {
                        let at = noise.below(bytes.len());
                        let run = noise.below(64).min(bytes.len() - at);
                        for slot in &mut bytes[at..at + run] {
                            *slot = noise.byte();
                        }
                    }
                    // Bytes inserted, which shifts every offset after them.
                    3 => {
                        let at = noise.below(bytes.len());
                        let extra: Vec<u8> =
                            (0..1 + noise.below(32)).map(|_| noise.byte()).collect();
                        bytes.splice(at..at, extra);
                    }
                    // Not the format at all.
                    _ => {
                        bytes = (0..noise.below(4096)).map(|_| noise.byte()).collect();
                    }
                }

                accepted += usize::from(feed(read, &bytes, what));
                tried += 1;
            }
        }

        // The control, in the spirit of the seam test's: a reader that
        // rejected every mutant at its first length check would pass
        // everything above without the parser having run, and this file would
        // be measuring the guard rather than the reader. One in fifty is a
        // floor and not a target -- the rates when this was written were 37%
        // for binary STL, 10% ASCII, 5% OBJ and 6% 3MF, the last being low
        // because a deflate stream under a CRC catches most byte changes at
        // the container. The XML underneath it is fuzzed separately, where it
        // can be reached, in `import::threemf`'s own tests.
        let floor = tried / 50;
        assert!(
            accepted >= floor,
            "{format}: only {accepted} of {tried} mutants got as far as the parser, \
             so this test is measuring the length check and not the reader"
        );
    }
}

#[test]
fn the_degenerate_inputs_every_parser_trips_over() {
    // The cases a random mutation almost never lands on, each of which has
    // been a real crash in some reader somewhere.
    let inputs: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("one byte", vec![0]),
        ("all zeroes", vec![0; 4096]),
        ("all 0xff", vec![0xff; 4096]),
        // A binary STL header claiming four billion triangles it does not
        // have. Believing it is a two hundred gigabyte allocation.
        ("a huge declared count", {
            let mut bytes = vec![0u8; 80];
            bytes.extend_from_slice(&u32::MAX.to_le_bytes());
            bytes
        }),
        // The same, sized so the length rule accepts it and the body is one
        // record short.
        ("one record short", {
            let mut bytes = vec![0u8; 80];
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&[0u8; 50]);
            bytes
        }),
        ("a lone BOM", vec![0xef, 0xbb, 0xbf]),
        ("invalid utf8", vec![0x80, 0x81, 0x82, 0x83]),
        ("a very long line", {
            let mut text = b"vertex ".to_vec();
            text.extend(std::iter::repeat_n(b'9', 100_000));
            text
        }),
    ];

    for (what, bytes) in &inputs {
        feed(stl::read, bytes, &format!("stl: {what}"));
        feed(obj::read, bytes, &format!("obj: {what}"));
        feed(threemf::read, bytes, &format!("3mf: {what}"));
    }
}

#[test]
fn a_mesh_the_lattice_cannot_hold_is_refused_rather_than_built() {
    // The cases the reader has no opinion about, because every coordinate in
    // them is a perfectly good finite number. It is the *voxeliser* that has
    // to object, and what it objects with is the preflight -- which bounds an
    // import by its surface AREA. A sliver is the case that tests whether area
    // is enough on its own: it is tiny, so the preflight is happy, and it sits
    // where the voxel lattice cannot address it.
    let far = 1.0e20_f32;
    let cases: Vec<(&str, ExportMesh)> = vec![
        (
            "a sliver a hundred quintillion millimetres out",
            ExportMesh {
                positions: vec![
                    Vec3::new(far, 0.0, 0.0),
                    Vec3::new(far + 1.0, 0.0, 0.0),
                    Vec3::new(far, 1.0, 0.0),
                    Vec3::new(far, 0.0, 1.0),
                ],
                normals: Vec::new(),
                triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                slots: Vec::new(),
            },
        ),
        (
            "a tetrahedron spanning the float range",
            ExportMesh {
                positions: vec![
                    Vec3::new(-far, -far, -far),
                    Vec3::new(far, -far, -far),
                    Vec3::new(-far, far, -far),
                    Vec3::new(-far, -far, far),
                ],
                normals: Vec::new(),
                triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                slots: Vec::new(),
            },
        ),
        (
            "every triangle degenerate",
            ExportMesh {
                positions: vec![Vec3::ZERO, Vec3::ZERO, Vec3::ZERO],
                normals: Vec::new(),
                triangles: vec![[0, 1, 2], [0, 1, 2]],
                slots: Vec::new(),
            },
        ),
        (
            "one triangle, no volume",
            ExportMesh {
                positions: vec![Vec3::ZERO, Vec3::X * 10.0, Vec3::Y * 10.0],
                normals: Vec::new(),
                triangles: vec![[0, 1, 2]],
                slots: Vec::new(),
            },
        ),
    ];

    for (what, mesh) in &cases {
        // An answer either way. What must not happen is a panic, an overflowed
        // lattice coordinate, or an allocation the machine cannot serve.
        //
        // The spanning tetrahedron is the one that gets *built*: it is
        // implausibly large rather than impossible, so `refit_if_implausible`
        // rescales it to something the lattice can hold and reports the factor.
        // The other three are refused, each for a reason of its own.
        voxelise_it(mesh, what);
    }
}
