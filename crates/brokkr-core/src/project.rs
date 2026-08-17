// SPDX-License-Identifier: AGPL-3.0-or-later

//! The `.brokkr` project format: saving and loading a sculpt.
//!
//! Until this existed nothing made in BrokkrSculpt survived quitting, which is
//! a strange thing to say about a tool meant for work that takes hours.
//!
//! # Why it lives in `brokkr-core`
//!
//! [`Volume::brick`] and [`Volume::insert_brick`] are `pub(crate)`. Widening
//! them so the shell could write this would leak the storage representation out
//! of the engine, which is precisely what keeps the Iced choice reversible. So
//! the format lives beside the thing it serialises.
//!
//! # Why it is a flat file and not a ZIP
//!
//! SindriCAD's `.sindri` is a ZIP because it carries many separately addressed
//! blobs plus JSON across a language boundary. A `.brokkr` file carries a header
//! and one stream of bricks. Copying the ZIP shape would mean either a new
//! dependency or a hand rolled ZIP reader, both of which cost more than this
//! format is worth. What *is* worth copying from `container.rs` is its
//! discipline, and that is all below: two independent version numbers, a size
//! cap checked before anything is read, and never trusting a number from inside
//! the file without bounding it first.
//!
//! # The lattice constants are the whole risk
//!
//! [`BRICK_DIM`] and [`NARROW_BAND`] are compile time constants, and `brick.rs`
//! calls the first of them "a tuning knob, not a magic number". Changing either
//! reshapes every brick in memory **without reshaping the file**. A file written
//! at one setting and read at another is not a load error — it is a plausible
//! looking sculpt made of misinterpreted numbers. So both are written into the
//! header and both are checked on load, and a mismatch is refused.
//!
//! # What is deliberately not saved
//!
//! Undo history. It is bounded by a memory budget rather than by meaning, it can
//! be larger than the model, and a reopened file starting with an empty history
//! is what a user expects. The dirty set is not saved either: it is rebuilt by
//! `mark_everything_dirty` on load, which its own documentation already
//! describes as a load time operation.

use std::io::{Read, Write};

use glam::{IVec3, Vec3};

use crate::brick::{BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, NARROW_BAND};
use crate::volume::Volume;

/// Magic bytes. Long enough that a truncated or mistyped file is rejected
/// immediately rather than part way through a brick.
const MAGIC: &[u8; 8] = b"BROKKR\x00\x01";

/// Version of the file's *layout*: the header fields and the order of sections.
const CONTAINER_VERSION: u16 = 1;

/// Version of how a brick's field values are *encoded*. Kept separate from the
/// container version, because the two change for different reasons — a new
/// header field is not a new encoding, and a new encoding is not a new layout.
/// SindriCAD's container learned this distinction the hard way.
const FIELD_VERSION: u16 = 1;

/// Largest file this will read, as a guard against a corrupt or hostile header
/// asking for an allocation the machine cannot serve.
///
/// Eight gigabytes is far above any real sculpt — the densest measured model,
/// eleven million triangles at a 0.055 mm voxel, is about 1 GB resident — and
/// far below anything that would take the process down.
///
/// Note this is an *absolute* cap and not a compression ratio test. SindriCAD
/// tried a ratio test and removed it because it fires on legitimately sparse
/// data, and a narrow band field clamped to ±3 is about as sparse as data gets.
const MAX_BRICKS: u64 = 8 * 1024 * 1024 * 1024 / BRICK_BYTES as u64;

/// Bytes a dense brick occupies in the file.
const BRICK_BYTES: usize = BRICK_VOXELS * 4;

/// Tag for a brick stored as a single value.
const TAG_UNIFORM: u8 = 0;
/// Tag for a brick stored as a full array.
const TAG_DENSE: u8 = 1;

/// Everything about a session that is worth reopening into, beyond the field
/// itself.
///
/// Deliberately small. These are conveniences — where the camera was, what the
/// brush was set to — and a file whose header is readable but whose settings are
/// nonsense should still load its geometry, so every one of these is clamped on
/// read rather than trusted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectState {
    pub camera_target: Vec3,
    pub camera_distance: f32,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_roll: f32,
    pub brush_radius: f32,
    pub brush_strength: f32,
    /// Mirror planes, as three flags in `MirrorAxis::ALL` order.
    pub mirror: [bool; 3],
}

impl Default for ProjectState {
    fn default() -> Self {
        Self {
            camera_target: Vec3::ZERO,
            camera_distance: 100.0,
            camera_yaw: 0.6,
            camera_pitch: 0.35,
            camera_roll: 0.0,
            brush_radius: 3.0,
            brush_strength: 0.15,
            mirror: [false; 3],
        }
    }
}

/// Why a `.brokkr` file could not be read.
///
/// Every variant says what was expected as well as what was found, because the
/// realistic causes — a file from a different build, a partial download, a
/// truncated write — are indistinguishable from each other without it.
#[derive(Debug)]
pub enum ProjectError {
    Io(std::io::Error),
    /// Not a `.brokkr` file at all.
    NotABrokkrFile,
    /// Written by a layout this build does not understand.
    ContainerVersion {
        found: u16,
        supported: u16,
    },
    /// Written with a field encoding this build does not understand.
    FieldVersion {
        found: u16,
        supported: u16,
    },
    /// Written by a build whose voxel lattice was shaped differently. Loading it
    /// would produce a plausible looking sculpt made of misread numbers.
    LatticeMismatch {
        field: &'static str,
        found: f64,
        expected: f64,
    },
    /// The header asks for more than this will read.
    TooLarge {
        bricks: u64,
        limit: u64,
    },
    /// A brick tag that is neither uniform nor dense.
    UnknownBrickTag(u8),
    /// A distance value that is not a finite number.
    NonFiniteValue,
    /// The file ended in the middle of something.
    Truncated,
}

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectError::Io(error) => write!(f, "{error}"),
            ProjectError::NotABrokkrFile => write!(f, "this is not a BrokkrSculpt file"),
            ProjectError::ContainerVersion { found, supported } => write!(
                f,
                "the file's layout is version {found}, and this build understands {supported}"
            ),
            ProjectError::FieldVersion { found, supported } => write!(
                f,
                "the file's field encoding is version {found}, and this build understands \
                 {supported}"
            ),
            ProjectError::LatticeMismatch { field, found, expected } => write!(
                f,
                "the file was written with {field} of {found}, and this build uses {expected}. \
                 Loading it would misread every value"
            ),
            ProjectError::TooLarge { bricks, limit } => {
                write!(f, "the file claims {bricks} bricks, and the limit is {limit}")
            }
            ProjectError::UnknownBrickTag(tag) => write!(f, "unknown brick kind {tag}"),
            ProjectError::NonFiniteValue => {
                write!(f, "the file holds a distance that is not a finite number")
            }
            ProjectError::Truncated => write!(f, "the file ends part way through"),
        }
    }
}

impl std::error::Error for ProjectError {}

impl From<std::io::Error> for ProjectError {
    fn from(error: std::io::Error) -> Self {
        // An unexpected end of file is a truncated project, which is worth
        // saying plainly rather than passing through as an io error.
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            ProjectError::Truncated
        } else {
            ProjectError::Io(error)
        }
    }
}

type Result<T> = std::result::Result<T, ProjectError>;

fn write_u16(out: &mut impl Write, value: u16) -> Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u32(out: &mut impl Write, value: u32) -> Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64(out: &mut impl Write, value: u64) -> Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f32(out: &mut impl Write, value: f32) -> Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_vec3(out: &mut impl Write, value: Vec3) -> Result<()> {
    for component in value.to_array() {
        write_f32(out, component)?;
    }
    Ok(())
}

fn read_exact<const N: usize>(input: &mut impl Read) -> Result<[u8; N]> {
    let mut bytes = [0u8; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u16(input: &mut impl Read) -> Result<u16> {
    Ok(u16::from_le_bytes(read_exact::<2>(input)?))
}

fn read_u32(input: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact::<4>(input)?))
}

fn read_u64(input: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact::<8>(input)?))
}

fn read_f32(input: &mut impl Read) -> Result<f32> {
    Ok(f32::from_le_bytes(read_exact::<4>(input)?))
}

fn read_vec3(input: &mut impl Read) -> Result<Vec3> {
    Ok(Vec3::new(read_f32(input)?, read_f32(input)?, read_f32(input)?))
}

/// Write a sculpt.
///
/// Bricks go out in a deterministic order, so saving the same volume twice
/// produces byte identical files. That is worth having for its own sake and it
/// makes a round trip test able to compare bytes rather than semantics.
pub fn write(out: &mut impl Write, volume: &Volume, state: &ProjectState) -> Result<()> {
    out.write_all(MAGIC)?;
    write_u16(out, CONTAINER_VERSION)?;
    write_u16(out, FIELD_VERSION)?;

    // The lattice. Written as the values themselves rather than as a hash, so a
    // mismatch can say what it found.
    write_u32(out, BRICK_DIM as u32)?;
    write_f32(out, NARROW_BAND)?;

    write_f32(out, volume.voxel_size())?;

    write_vec3(out, state.camera_target)?;
    write_f32(out, state.camera_distance)?;
    write_f32(out, state.camera_yaw)?;
    write_f32(out, state.camera_pitch)?;
    write_f32(out, state.camera_roll)?;
    write_f32(out, state.brush_radius)?;
    write_f32(out, state.brush_strength)?;
    out.write_all(&[
        u8::from(state.mirror[0]),
        u8::from(state.mirror[1]),
        u8::from(state.mirror[2]),
    ])?;

    let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
    coords.sort_unstable();
    write_u64(out, coords.len() as u64)?;

    for coord in coords {
        let Some(brick) = volume.brick(coord) else {
            continue;
        };
        for component in coord.0.to_array() {
            out.write_all(&component.to_le_bytes())?;
        }
        match brick {
            Brick::Uniform(value) => {
                out.write_all(&[TAG_UNIFORM])?;
                write_f32(out, *value)?;
            }
            Brick::Dense(values) => {
                out.write_all(&[TAG_DENSE])?;
                // One value at a time rather than a cast of the whole array:
                // little endian has to be explicit, or a file written on one
                // architecture is silently wrong on another.
                for value in values.iter() {
                    write_f32(out, *value)?;
                }
            }
        }
    }

    out.flush()?;
    Ok(())
}

/// Read a sculpt.
pub fn read(input: &mut impl Read) -> Result<(Volume, ProjectState)> {
    let magic: [u8; 8] = read_exact(input)?;
    if &magic != MAGIC {
        return Err(ProjectError::NotABrokkrFile);
    }

    let container = read_u16(input)?;
    if container != CONTAINER_VERSION {
        return Err(ProjectError::ContainerVersion {
            found: container,
            supported: CONTAINER_VERSION,
        });
    }
    let field = read_u16(input)?;
    if field != FIELD_VERSION {
        return Err(ProjectError::FieldVersion { found: field, supported: FIELD_VERSION });
    }

    // The check this format exists to make. See the module documentation: a
    // mismatch here is not a load error, it is a sculpt made of misread numbers.
    let brick_dim = read_u32(input)?;
    if brick_dim as usize != BRICK_DIM {
        return Err(ProjectError::LatticeMismatch {
            field: "brick size",
            found: brick_dim as f64,
            expected: BRICK_DIM as f64,
        });
    }
    let narrow_band = read_f32(input)?;
    if narrow_band != NARROW_BAND {
        return Err(ProjectError::LatticeMismatch {
            field: "narrow band",
            found: narrow_band as f64,
            expected: NARROW_BAND as f64,
        });
    }

    let voxel_size = read_f32(input)?;
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(ProjectError::NonFiniteValue);
    }

    let mut state = ProjectState {
        camera_target: read_vec3(input)?,
        camera_distance: read_f32(input)?,
        camera_yaw: read_f32(input)?,
        camera_pitch: read_f32(input)?,
        camera_roll: read_f32(input)?,
        brush_radius: read_f32(input)?,
        brush_strength: read_f32(input)?,
        mirror: [false; 3],
    };
    let mirror: [u8; 3] = read_exact(input)?;
    state.mirror = [mirror[0] != 0, mirror[1] != 0, mirror[2] != 0];

    // Settings are conveniences, so a file with nonsense in them still loads its
    // geometry rather than refusing outright. The field itself gets no such
    // latitude, below.
    if !state.camera_target.is_finite() {
        state.camera_target = Vec3::ZERO;
    }
    for value in [
        &mut state.camera_distance,
        &mut state.camera_yaw,
        &mut state.camera_pitch,
        &mut state.camera_roll,
        &mut state.brush_radius,
        &mut state.brush_strength,
    ] {
        if !value.is_finite() {
            *value = 0.0;
        }
    }

    let count = read_u64(input)?;
    if count > MAX_BRICKS {
        return Err(ProjectError::TooLarge { bricks: count, limit: MAX_BRICKS });
    }

    let mut volume = Volume::new(voxel_size);
    for _ in 0..count {
        let coord = BrickCoord(IVec3::new(
            i32::from_le_bytes(read_exact::<4>(input)?),
            i32::from_le_bytes(read_exact::<4>(input)?),
            i32::from_le_bytes(read_exact::<4>(input)?),
        ));

        let tag: [u8; 1] = read_exact(input)?;
        let brick = match tag[0] {
            TAG_UNIFORM => {
                let value = read_f32(input)?;
                if !value.is_finite() {
                    return Err(ProjectError::NonFiniteValue);
                }
                Brick::Uniform(value)
            }
            TAG_DENSE => {
                let mut values = vec![0.0f32; BRICK_VOXELS];
                let mut bytes = vec![0u8; BRICK_BYTES];
                input.read_exact(&mut bytes)?;
                for (slot, chunk) in values.iter_mut().zip(bytes.chunks_exact(4)) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !value.is_finite() {
                        return Err(ProjectError::NonFiniteValue);
                    }
                    *slot = value;
                }
                let boxed: Box<[f32; BRICK_VOXELS]> = values
                    .into_boxed_slice()
                    .try_into()
                    .expect("the vector was built at exactly BRICK_VOXELS");
                Brick::Dense(boxed)
            }
            other => return Err(ProjectError::UnknownBrickTag(other)),
        };
        volume.insert_brick(coord, brick);
    }

    // Nothing has been meshed yet, and the renderer holds whatever the previous
    // model left. Marking everything dirty is what makes the load visible.
    volume.mark_everything_dirty();

    Ok((volume, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::{Brush, BrushDirection, BrushKind, BrushScratch, Stamp};

    fn sculpted() -> Volume {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 20.0);
        let mut scratch = BrushScratch::new();
        let brush = Brush { kind: BrushKind::Draw, radius: 6.0, strength: 0.7, ..Brush::default() };
        for at in [Vec3::new(20.0, 0.0, 0.0), Vec3::new(0.0, 20.0, 0.0), Vec3::new(0.0, 0.0, -20.0)]
        {
            let normal = volume.gradient_world(at);
            brush.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        }
        volume
    }

    fn round_trip(volume: &Volume, state: &ProjectState) -> (Volume, ProjectState) {
        let mut bytes = Vec::new();
        write(&mut bytes, volume, state).expect("write failed");
        read(&mut bytes.as_slice()).expect("read failed")
    }

    /// The property that matters: what comes back is the same sculpt, not
    /// something that merely looks like it.
    #[test]
    fn a_sculpted_model_survives_a_round_trip_value_for_value() {
        let volume = sculpted();
        let (loaded, _) = round_trip(&volume, &ProjectState::default());

        assert_eq!(loaded.voxel_size(), volume.voxel_size());

        let mut original: Vec<BrickCoord> = volume.brick_coords().collect();
        let mut returned: Vec<BrickCoord> = loaded.brick_coords().collect();
        original.sort_unstable();
        returned.sort_unstable();
        assert_eq!(original, returned, "the set of bricks changed");
        assert!(!original.is_empty());

        for coord in original {
            let before = volume.brick(coord).expect("listed");
            let after = loaded.brick(coord).expect("listed");
            match (before, after) {
                (Brick::Uniform(a), Brick::Uniform(b)) => assert_eq!(a, b, "at {coord:?}"),
                (Brick::Dense(a), Brick::Dense(b)) => {
                    assert_eq!(a.as_slice(), b.as_slice(), "at {coord:?}")
                }
                _ => panic!("brick at {coord:?} changed kind across the round trip"),
            }
        }
    }

    /// Writing twice must give the same bytes, or a round trip test can only
    /// compare semantics and a file diff means nothing.
    #[test]
    fn writing_the_same_volume_twice_gives_identical_bytes() {
        let volume = sculpted();
        let state = ProjectState::default();
        let mut first = Vec::new();
        let mut second = Vec::new();
        write(&mut first, &volume, &state).unwrap();
        write(&mut second, &volume, &state).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn the_session_settings_come_back_too() {
        let state = ProjectState {
            camera_target: Vec3::new(1.0, -2.0, 3.5),
            camera_distance: 42.0,
            camera_yaw: 0.75,
            camera_pitch: -0.25,
            camera_roll: 0.1,
            brush_radius: 4.25,
            brush_strength: 0.33,
            mirror: [true, false, true],
        };
        let (_, loaded) = round_trip(&sculpted(), &state);
        assert_eq!(loaded, state);
    }

    /// A reloaded model has to be ready to mesh, or it loads into an empty
    /// screen and looks like a failure.
    #[test]
    fn everything_is_marked_dirty_so_the_load_is_visible() {
        let (mut loaded, _) = round_trip(&sculpted(), &ProjectState::default());
        let mut dirty = Vec::new();
        loaded.take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "a freshly loaded model had nothing to mesh");
    }

    /// The check this whole format exists to make. A file written at a
    /// different lattice is not a load error, it is a plausible looking sculpt
    /// made of misread numbers, so it must be refused rather than accepted.
    #[test]
    fn a_file_written_at_a_different_lattice_is_refused() {
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted(), &ProjectState::default()).unwrap();

        // The brick size sits after the magic and the two versions.
        let offset = MAGIC.len() + 2 + 2;
        let mut tampered = bytes.clone();
        tampered[offset..offset + 4].copy_from_slice(&(BRICK_DIM as u32 + 1).to_le_bytes());
        assert!(
            matches!(
                read(&mut tampered.as_slice()),
                Err(ProjectError::LatticeMismatch { field: "brick size", .. })
            ),
            "a different brick size was accepted"
        );

        let mut tampered = bytes.clone();
        let band = offset + 4;
        tampered[band..band + 4].copy_from_slice(&(NARROW_BAND + 1.0).to_le_bytes());
        assert!(
            matches!(
                read(&mut tampered.as_slice()),
                Err(ProjectError::LatticeMismatch { field: "narrow band", .. })
            ),
            "a different narrow band was accepted"
        );
    }

    #[test]
    fn something_that_is_not_a_project_is_refused_immediately() {
        assert!(matches!(
            read(&mut b"not a sculpt at all".as_slice()),
            Err(ProjectError::NotABrokkrFile)
        ));
        // An STL, which is the file most likely to be opened here by mistake.
        let mut stl = vec![0u8; 84];
        stl[0..5].copy_from_slice(b"solid");
        assert!(matches!(read(&mut stl.as_slice()), Err(ProjectError::NotABrokkrFile)));
    }

    #[test]
    fn a_newer_layout_or_encoding_is_refused_with_both_versions_named() {
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted(), &ProjectState::default()).unwrap();

        let mut newer = bytes.clone();
        newer[MAGIC.len()..MAGIC.len() + 2].copy_from_slice(&(CONTAINER_VERSION + 1).to_le_bytes());
        match read(&mut newer.as_slice()) {
            Err(ProjectError::ContainerVersion { found, supported }) => {
                assert_eq!((found, supported), (CONTAINER_VERSION + 1, CONTAINER_VERSION));
            }
            Ok(_) => panic!("a newer container version was accepted"),
            Err(other) => panic!("expected a container version error, got {other}"),
        }

        let mut newer = bytes.clone();
        let at = MAGIC.len() + 2;
        newer[at..at + 2].copy_from_slice(&(FIELD_VERSION + 1).to_le_bytes());
        assert!(matches!(read(&mut newer.as_slice()), Err(ProjectError::FieldVersion { .. })));
    }

    /// A partial write or a partial download must not load as a smaller sculpt.
    #[test]
    fn a_truncated_file_is_refused_rather_than_loaded_short() {
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted(), &ProjectState::default()).unwrap();

        for fraction in [0.2, 0.5, 0.9, 0.999] {
            let cut = (bytes.len() as f64 * fraction) as usize;
            let result = read(&mut bytes[..cut].as_ref());
            assert!(result.is_err(), "a file cut to {fraction} of its length loaded anyway");
        }
    }

    /// The header is the only thing standing between a corrupt count and an
    /// allocation the machine cannot serve.
    #[test]
    fn an_absurd_brick_count_is_refused_before_anything_is_read() {
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted(), &ProjectState::default()).unwrap();

        // The count sits after the fixed size header; find it by writing an
        // empty volume and measuring.
        let mut empty = Vec::new();
        write(&mut empty, &Volume::new(0.5), &ProjectState::default()).unwrap();
        let count_at = empty.len() - 8;

        let mut absurd = bytes.clone();
        absurd[count_at..count_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        match read(&mut absurd.as_slice()) {
            Err(ProjectError::TooLarge { bricks, limit }) => {
                assert_eq!(bricks, u64::MAX);
                assert_eq!(limit, MAX_BRICKS);
            }
            Ok(_) => panic!("an absurd brick count was accepted"),
            Err(other) => panic!("expected a size refusal, got {other}"),
        }
    }

    #[test]
    fn a_non_finite_distance_is_refused() {
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 10.0);
        let mut bytes = Vec::new();
        write(&mut bytes, &volume, &ProjectState::default()).unwrap();

        // Poison the last four bytes, which are inside the final brick.
        let end = bytes.len();
        bytes[end - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(read(&mut bytes.as_slice()), Err(ProjectError::NonFiniteValue)));
    }

    #[test]
    fn an_unknown_brick_kind_is_refused() {
        let mut bytes = Vec::new();
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 10.0);
        write(&mut bytes, &volume, &ProjectState::default()).unwrap();

        let mut empty = Vec::new();
        write(&mut empty, &Volume::new(0.5), &ProjectState::default()).unwrap();
        // First brick's tag: past the header, past the count, past the coord.
        let tag_at = empty.len() + 12;
        bytes[tag_at] = 99;
        assert!(matches!(read(&mut bytes.as_slice()), Err(ProjectError::UnknownBrickTag(99))));
    }

    /// Empty and solid space cost one value per brick in memory, and the file
    /// must not undo that by writing them out in full.
    ///
    /// Asserted as the exact size the encoding implies rather than as a ratio.
    /// A ratio depends on how much of the model happens to straddle the surface
    /// -- the first version of this test used one and failed on a sphere whose
    /// bricks were nearly all boundary bricks, which said nothing about the
    /// encoding at all.
    #[test]
    fn a_uniform_brick_costs_seventeen_bytes_and_not_a_hundred_and_thirty_kilobytes() {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 60.0);
        let stats = volume.stats();
        assert!(stats.uniform_bricks > 0, "this model has no uniform bricks to test with");
        assert!(stats.dense_bricks > 0, "this model has no dense bricks to test with");

        let mut bytes = Vec::new();
        write(&mut bytes, &volume, &ProjectState::default()).unwrap();

        // An empty volume is exactly the header plus the brick count, which is
        // how the fixed part is measured rather than counted by hand.
        let mut empty = Vec::new();
        write(&mut empty, &Volume::new(1.0), &ProjectState::default()).unwrap();

        // Per brick: twelve bytes of coordinate, one tag byte, then either one
        // value or the whole array.
        const PER_BRICK: usize = 12 + 1;
        let expected = empty.len()
            + stats.uniform_bricks * (PER_BRICK + 4)
            + stats.dense_bricks * (PER_BRICK + BRICK_BYTES);
        assert_eq!(
            bytes.len(),
            expected,
            "{} uniform and {} dense bricks should come to exactly {expected} bytes",
            stats.uniform_bricks,
            stats.dense_bricks
        );

        // And the saving is real: written out in full they would cost this much.
        let all_dense =
            empty.len() + (stats.uniform_bricks + stats.dense_bricks) * (PER_BRICK + BRICK_BYTES);
        assert!(bytes.len() < all_dense, "uniform bricks cost as much as dense ones");
    }

    #[test]
    fn an_empty_volume_round_trips_as_an_empty_volume() {
        let volume = Volume::new(0.25);
        let (loaded, _) = round_trip(&volume, &ProjectState::default());
        assert_eq!(loaded.voxel_size(), 0.25);
        assert_eq!(loaded.brick_coords().count(), 0);
    }
}
