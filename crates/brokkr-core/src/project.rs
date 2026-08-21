// SPDX-License-Identifier: AGPL-3.0-only

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

use crate::brick::{BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE};
use crate::volume::Volume;

/// Magic bytes. Long enough that a truncated or mistyped file is rejected
/// immediately rather than part way through a brick.
const MAGIC: &[u8; 8] = b"BROKKR\x00\x01";

/// Version of the file's *layout*: the header fields and the order of sections.
///
/// 2 added the timeline key trailer after the brick stream. Bumping it did not
/// have to orphan every file already written, and did not: see
/// [`OLDEST_CONTAINER_VERSION`].
const CONTAINER_VERSION: u16 = 2;

/// The oldest layout this build still reads.
///
/// Version 1 is version 2 without the key trailer, so reading one is a matter
/// of not looking for the trailer -- there is no conversion and no second code
/// path through the geometry. This exists because the alternative was refusing
/// every `.brokkr` file in existence to add a feature none of them use, which
/// is not a trade a save format gets to make. **Only add a version here when
/// the old layout is genuinely still readable**; the point of refusing an
/// unknown one is that a plausible-looking sculpt made of misread numbers is
/// worse than an error.
const OLDEST_CONTAINER_VERSION: u16 = 1;

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

/// How far from the origin a brick coordinate may sit.
///
/// Derived rather than chosen: [`BrickCoord::origin`] multiplies the coordinate
/// by [`BRICK_DIM`] and [`BrickCoord::max_voxel`] adds the last voxel, so this
/// is the largest coordinate whose own voxel range still fits in an `i32`.
/// Beyond it the multiply overflows, which is a panic in a debug build and a
/// wrapped coordinate in a release one -- a brick that silently lands somewhere
/// else in the model.
///
/// Nothing this build writes comes close: it is 67 million bricks, which at the
/// shipped 0.25 mm voxel is over five hundred kilometres. It is a bound on what
/// the lattice can represent, not a judgement about what a sculpt should
/// contain, which is why it is derived from the arithmetic rather than picked.
/// A corrupt file is what reaches it.
const MAX_BRICK_COORD: i32 = (i32::MAX - (BRICK_DIM as i32 - 1)) / BRICK_DIM as i32;

/// Most timeline keys a file may carry.
///
/// A ceiling rather than a limit anyone should meet: the strip is a few hundred
/// pixels wide, so a thousand keys is already more than one per pixel. It is
/// here for the same reason [`MAX_BRICKS`] is -- a count read out of a file
/// decides an allocation, and a corrupt one should not decide a large one.
const MAX_KEYS: u32 = 1024;

/// Read one distance and hold it to what every writer here guarantees.
fn read_distance(input: &mut impl Read) -> Result<f32> {
    checked_distance(read_f32(input)?)
}

/// Refuse a distance that no writer in this build could have produced.
///
/// Every value reaching a brick has been clamped to the narrow band --
/// `write_voxels` does it on every edit, and the seeding, voxelising, clipping
/// and resampling paths all go through it -- so a value outside the band did
/// not come from here. Accepting one is not harmless: the band is what the
/// brushes' skipping reasons about and what the mesher's classification
/// assumes, so an out of band value is a sculpt that behaves differently from
/// every other sculpt in ways nothing reports.
///
/// Refused rather than clamped, deliberately. Clamping would load a file that
/// is provably corrupt while showing something plausible, which is the failure
/// mode the lattice constants in the header exist to prevent.
fn checked_distance(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(ProjectError::NonFiniteValue);
    }
    if !(INSIDE..=OUTSIDE).contains(&value) {
        return Err(ProjectError::OutsideTheBand { found: value });
    }
    Ok(value)
}

/// Bytes a dense brick occupies in the file.
const BRICK_BYTES: usize = BRICK_VOXELS * 4;

/// Tag for a brick stored as a single value.
const TAG_UNIFORM: u8 = 0;
/// Tag for a brick stored as a full array.
const TAG_DENSE: u8 = 1;

/// Where the camera was and what the brush was set to.
///
/// Deliberately small. These are conveniences, and a file whose header is
/// readable but whose settings are nonsense should still load its geometry, so
/// every one of these is clamped on read rather than trusted.
///
/// Its own type because two things restore it: reopening a file, and jumping to
/// a timeline key. Keeping them one type is what stops a key from quietly
/// restoring less than a reopen does when a field is added here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
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

impl Default for View {
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

/// One stored view on the timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyframe {
    /// Where along the strip it sits, from 0 at the left to 1 at the right.
    ///
    /// A position rather than a duration, because the strip is a fixed length
    /// a user drags keys around on. What that length means in seconds is the
    /// application's business, not the file's.
    pub at: f32,
    pub view: View,
}

/// Everything about a session that is worth reopening into, beyond the field
/// itself.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectState {
    /// Where things were when the file was saved.
    pub view: View,
    /// Timeline keys, in ascending `at` order.
    ///
    /// Held sorted rather than sorted on use, because everything that reads
    /// them wants neighbours -- the pair a playhead sits between, the key
    /// nearest a click -- and a list that is sorted only sometimes makes every
    /// one of those a bug waiting for the one caller that forgot.
    pub keys: Vec<Keyframe>,
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
    /// A distance value outside the narrow band, which every writer in this
    /// build clamps into it.
    OutsideTheBand {
        found: f32,
    },
    /// A brick coordinate so far out that its own voxel origin does not fit in
    /// the lattice.
    BrickOutOfRange {
        found: [i32; 3],
        limit: i32,
    },
    /// More timeline keys than [`MAX_KEYS`].
    TooManyKeys {
        keys: u32,
        limit: u32,
    },
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
            ProjectError::TooManyKeys { keys, limit } => {
                write!(f, "the file claims {keys} timeline keys, and the limit is {limit}")
            }
            ProjectError::NonFiniteValue => {
                write!(f, "the file holds a distance that is not a finite number")
            }
            ProjectError::OutsideTheBand { found } => write!(
                f,
                "the file holds a distance of {found}, and the narrow band is \
                 {INSIDE} to {OUTSIDE}"
            ),
            ProjectError::BrickOutOfRange { found, limit } => {
                write!(f, "the file places a brick at {found:?}, and the lattice reaches {limit}")
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

/// Write the camera and brush settings.
///
/// One function for both the session's own view and every timeline key, so the
/// two can never store different things. Adding a field here adds it to both.
fn write_view(out: &mut impl Write, view: &View) -> Result<()> {
    write_vec3(out, view.camera_target)?;
    write_f32(out, view.camera_distance)?;
    write_f32(out, view.camera_yaw)?;
    write_f32(out, view.camera_pitch)?;
    write_f32(out, view.camera_roll)?;
    write_f32(out, view.brush_radius)?;
    write_f32(out, view.brush_strength)?;
    out.write_all(&[u8::from(view.mirror[0]), u8::from(view.mirror[1]), u8::from(view.mirror[2])])?;
    Ok(())
}

/// Read the camera and brush settings, repaired rather than refused.
///
/// The field itself gets no such latitude -- see `checked_distance` -- but
/// these are conveniences. A file whose geometry is intact and whose camera is
/// a NaN should open, show the sculpt, and put the camera somewhere sensible;
/// refusing it would lose the model to protect a number nobody would miss.
fn read_view(input: &mut impl Read) -> Result<View> {
    let mut view = View {
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
    view.mirror = [mirror[0] != 0, mirror[1] != 0, mirror[2] != 0];

    if !view.camera_target.is_finite() {
        view.camera_target = Vec3::ZERO;
    }
    for value in [
        &mut view.camera_distance,
        &mut view.camera_yaw,
        &mut view.camera_pitch,
        &mut view.camera_roll,
        &mut view.brush_radius,
        &mut view.brush_strength,
    ] {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    Ok(view)
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

    write_view(out, &state.view)?;

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

    // The key trailer, after the bricks rather than in the header. A version 1
    // file is exactly this file without it, which is what lets one still be
    // read: the brick count says where the geometry ends, and there is simply
    // nothing after it.
    write_u32(out, state.keys.len() as u32)?;
    for key in &state.keys {
        write_f32(out, key.at)?;
        write_view(out, &key.view)?;
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
    if !(OLDEST_CONTAINER_VERSION..=CONTAINER_VERSION).contains(&container) {
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

    let mut state = ProjectState { view: read_view(input)?, keys: Vec::new() };

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
        // Compared against both ends rather than through `abs`, because
        // `i32::MIN.abs()` overflows -- the check would then panic on exactly
        // the value it exists to refuse.
        let limit = IVec3::splat(MAX_BRICK_COORD);
        if coord.0.cmpgt(limit).any() || coord.0.cmplt(-limit).any() {
            return Err(ProjectError::BrickOutOfRange {
                found: coord.0.to_array(),
                limit: MAX_BRICK_COORD,
            });
        }

        let tag: [u8; 1] = read_exact(input)?;
        let brick = match tag[0] {
            TAG_UNIFORM => Brick::Uniform(read_distance(input)?),
            TAG_DENSE => {
                let mut values = vec![0.0f32; BRICK_VOXELS];
                let mut bytes = vec![0u8; BRICK_BYTES];
                input.read_exact(&mut bytes)?;
                for (slot, chunk) in values.iter_mut().zip(bytes.chunks_exact(4)) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    *slot = checked_distance(value)?;
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

    // The key trailer, which only version 2 and later carry. A version 1 file
    // ends with its last brick, and reading one is a matter of not looking.
    if container >= 2 {
        let count = read_u32(input)?;
        if count > MAX_KEYS {
            return Err(ProjectError::TooManyKeys { keys: count, limit: MAX_KEYS });
        }
        state.keys.reserve(count as usize);
        for _ in 0..count {
            let at = read_f32(input)?;
            // Repaired rather than refused, like the view itself: a key at a
            // nonsense position is a key in the wrong place, not a reason to
            // lose the sculpt it was saved beside.
            let at = if at.is_finite() { at.clamp(0.0, 1.0) } else { 0.0 };
            state.keys.push(Keyframe { at, view: read_view(input)? });
        }
        // Sorted on the way in, because everything downstream asks for
        // neighbours and a file is free to have been written by anything.
        state.keys.sort_by(|a, b| a.at.partial_cmp(&b.at).expect("clamped, so no NaN"));
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
    use crate::testing::Noise;

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

    /// Bytes the key trailer occupies when there are no keys: just its count.
    ///
    /// Several tests below reach for the *last brick value*, which used to mean
    /// the last four bytes of the file. Since version 2 the file ends with the
    /// key trailer instead, so they have to step back over it. Named rather
    /// than written as a bare 4, because when the trailer grows every one of
    /// them has to move with it.
    const EMPTY_TRAILER_BYTES: usize = 4;

    /// Where the final brick's last distance value starts.
    fn last_distance_at(bytes: &[u8]) -> usize {
        bytes.len() - EMPTY_TRAILER_BYTES - 4
    }

    /// Where the first brick's coordinate starts, found by measuring the
    /// header with an empty volume rather than by counting field widths.
    fn first_brick_at() -> usize {
        let mut empty = Vec::new();
        write(&mut empty, &Volume::new(0.5), &ProjectState::default()).expect("write failed");
        empty.len() - EMPTY_TRAILER_BYTES
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
            view: View {
                camera_target: Vec3::new(1.0, -2.0, 3.5),
                camera_distance: 42.0,
                camera_yaw: 0.75,
                camera_pitch: -0.25,
                camera_roll: 0.1,
                brush_radius: 4.25,
                brush_strength: 0.33,
                mirror: [true, false, true],
            },
            keys: Vec::new(),
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

        let count_at = first_brick_at() - 8;

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

        // Poison the final brick's last value, which sits just before the
        // key trailer rather than at the end of the file.
        let at = last_distance_at(&bytes);
        bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(read(&mut bytes.as_slice()), Err(ProjectError::NonFiniteValue)));
    }

    #[test]
    fn an_unknown_brick_kind_is_refused() {
        let mut bytes = Vec::new();
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 10.0);
        write(&mut bytes, &volume, &ProjectState::default()).unwrap();

        // First brick's tag: past the header, past the count, past the coord.
        let tag_at = first_brick_at() + 12;
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

    #[test]
    fn a_corrupted_project_is_answered_rather_than_survived() {
        // The targeted corruptions above each aim at one field. This aims at
        // nothing in particular, which is what an interrupted autosave, a
        // half synced file or a failing disk actually produce.
        //
        // The property is that every input gets an answer. A panic here is
        // worse than in the mesh readers: this runs on File > Open and on
        // File > Recover, so the file most likely to be corrupt is the crash
        // net a user is reaching for precisely because something already went
        // wrong. Whatever loads has to be a volume the rest of the
        // application can use -- finite everywhere, and inside the narrow
        // band, because a value outside it means a brick that meshes into
        // nothing or into a surface where there is none.
        let mut valid = Vec::new();
        write(&mut valid, &sculpted(), &ProjectState::default()).expect("write failed");

        let mut loaded = 0usize;
        let mut reached_the_bricks = 0usize;
        let mut tried = 0usize;

        for strategy in 0..3 {
            for run in 0..600u64 {
                let seed = (strategy as u64) << 32 | run | 1;
                let mut noise = Noise::seeded(seed);
                let mut bytes = valid.clone();

                match strategy {
                    // Flipped bits anywhere, header included.
                    0 => {
                        for _ in 0..1 + noise.below(8) {
                            let at = noise.below(bytes.len());
                            bytes[at] ^= 1 << noise.below(8);
                        }
                    }
                    // Cut short at an arbitrary point rather than at one of
                    // the four fractions the targeted test uses. This is what
                    // a save interrupted by a crash leaves behind.
                    1 => {
                        let keep = noise.below(bytes.len());
                        bytes.truncate(keep);
                    }
                    // A run of the body replaced wholesale, which is what a
                    // partial sync or a bad sector looks like.
                    _ => {
                        let at = noise.below(bytes.len());
                        let run = noise.below(256).min(bytes.len() - at);
                        for slot in &mut bytes[at..at + run] {
                            *slot = noise.byte();
                        }
                    }
                }

                tried += 1;
                let volume = match read(&mut bytes.as_slice()) {
                    Ok((volume, _)) => {
                        reached_the_bricks += 1;
                        volume
                    }
                    // Refusals that can only be raised from inside the brick
                    // loop, so each one is proof the body ran on this mutant.
                    Err(
                        ProjectError::NonFiniteValue
                        | ProjectError::OutsideTheBand { .. }
                        | ProjectError::UnknownBrickTag(_)
                        | ProjectError::BrickOutOfRange { .. },
                    ) => {
                        reached_the_bricks += 1;
                        continue;
                    }
                    Err(_) => continue,
                };
                loaded += 1;

                assert!(volume.voxel_size() > 0.0, "seed {seed}: a non positive voxel size loaded");
                for coord in volume.brick_coords().collect::<Vec<_>>() {
                    let origin = coord.origin();
                    for offset in [IVec3::ZERO, IVec3::splat(BRICK_DIM as i32 - 1)] {
                        let value = volume.sample_voxel(origin + offset);
                        assert!(
                            value.is_finite() && (INSIDE..=OUTSIDE).contains(&value),
                            "seed {seed}: brick {coord:?} loaded {value}, which is outside the band"
                        );
                    }
                }
            }
        }

        // The control: without it a reader that refused everything at the
        // magic bytes would pass this file having parsed nothing.
        //
        // It counts mutants that reached the brick loop, not mutants that
        // loaded, and the difference is the point. The band check refuses
        // almost every corrupted brick -- one flipped exponent bit is enough
        // -- so counting only what loads measures how strict the validation
        // is rather than how far the reader got. Measured at 1200 of 1800
        // reaching the loop, of which 20 loaded whole.
        assert!(
            reached_the_bricks > tried / 4,
            "only {reached_the_bricks} of {tried} corrupted projects got as far as the brick \
             loop, so this is measuring the header check and not the reader"
        );
        assert!(loaded > 0, "not one corrupted project loaded, so nothing was checked on load");
    }

    /// The overflow the fuzz above found, pinned as its own case so that a
    /// future change to the bound fails here with a clear name rather than
    /// somewhere inside two thousand random mutants.
    #[test]
    fn a_brick_placed_past_the_lattice_is_refused_rather_than_overflowing() {
        // `BrickCoord::origin` multiplies by BRICK_DIM, so a coordinate near
        // i32::MAX overflows: a panic in a debug build, and in a release one a
        // wrapped coordinate that puts the brick somewhere else in the model
        // with nothing reported. Nothing here writes such a file; a corrupt
        // one is what carries it.
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted(), &ProjectState::default()).expect("write failed");

        let coord_at = first_brick_at();

        for offending in [i32::MAX, i32::MIN, MAX_BRICK_COORD + 1, -(MAX_BRICK_COORD + 1)] {
            let mut broken = bytes.clone();
            broken[coord_at..coord_at + 4].copy_from_slice(&offending.to_le_bytes());
            match read(&mut broken.as_slice()) {
                Err(ProjectError::BrickOutOfRange { found, limit }) => {
                    assert_eq!(found[0], offending);
                    assert_eq!(limit, MAX_BRICK_COORD);
                }
                Ok(_) => panic!("a brick at {offending} was accepted"),
                Err(other) => panic!("expected a range refusal for {offending}, got {other}"),
            }
        }

        // And the bound is not so tight that it refuses the edge it allows.
        let mut fine = bytes.clone();
        fine[coord_at..coord_at + 4].copy_from_slice(&MAX_BRICK_COORD.to_le_bytes());
        let (volume, _) = read(&mut fine.as_slice()).expect("the largest legal brick was refused");
        let coord = volume
            .brick_coords()
            .find(|coord| coord.0.x == MAX_BRICK_COORD)
            .expect("the brick did not survive the load");
        // The thing the bound is for: this is what overflowed.
        let _ = coord.max_voxel();
    }

    #[test]
    fn a_distance_outside_the_narrow_band_is_refused() {
        // Every writer here clamps into the band, so a value outside it did
        // not come from this build. It is refused rather than clamped: a file
        // that is provably corrupt should say so, not show something
        // plausible.
        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 10.0);
        let mut bytes = Vec::new();
        write(&mut bytes, &volume, &ProjectState::default()).expect("write failed");

        for offending in [OUTSIDE * 2.0, INSIDE * 2.0, 1e30, -1e30] {
            let mut broken = bytes.clone();
            let at = last_distance_at(&broken);
            broken[at..at + 4].copy_from_slice(&offending.to_le_bytes());
            match read(&mut broken.as_slice()) {
                Err(ProjectError::OutsideTheBand { found }) => assert_eq!(found, offending),
                Ok(_) => panic!("a distance of {offending} was accepted"),
                Err(other) => panic!("expected a band refusal for {offending}, got {other}"),
            }
        }
    }

    fn a_key(at: f32, distance: f32) -> Keyframe {
        Keyframe {
            at,
            view: View {
                camera_distance: distance,
                camera_yaw: at * 2.0,
                mirror: [at > 0.5, false, true],
                ..View::default()
            },
        }
    }

    #[test]
    fn timeline_keys_survive_a_round_trip_in_order() {
        let state = ProjectState {
            view: View { camera_distance: 77.0, ..View::default() },
            keys: vec![a_key(0.0, 10.0), a_key(0.25, 20.0), a_key(0.9, 30.0)],
        };
        let (_, loaded) = round_trip(&sculpted(), &state);
        assert_eq!(loaded, state, "the keys did not come back as they went in");
    }

    #[test]
    fn keys_written_out_of_order_come_back_sorted() {
        // The file is a stream of bytes and anything may have written it, so
        // the invariant `ProjectState::keys` documents -- ascending `at` -- has
        // to be established on read rather than assumed of the writer.
        let state = ProjectState {
            view: View::default(),
            keys: vec![a_key(0.8, 1.0), a_key(0.1, 2.0), a_key(0.5, 3.0)],
        };
        let (_, loaded) = round_trip(&sculpted(), &state);
        let order: Vec<f32> = loaded.keys.iter().map(|key| key.at).collect();
        assert_eq!(order, vec![0.1, 0.5, 0.8]);
        // And each key kept its own view rather than merely its position.
        assert_eq!(loaded.keys[0].view.camera_distance, 2.0);
        assert_eq!(loaded.keys[2].view.camera_distance, 1.0);
    }

    /// The one that matters most about the version bump: every `.brokkr` file
    /// already written is a version 1 file, and there were real ones on disk
    /// when the timeline was added.
    ///
    /// A version 1 file is a version 2 file without the key trailer, so one is
    /// built here by writing a version 2 file and taking the trailer back off.
    /// That is exactly what the old writer produced, byte for byte, which is
    /// what makes this a test of compatibility rather than of a fixture.
    #[test]
    fn a_file_from_before_the_timeline_still_opens() {
        let volume = sculpted();
        let state = ProjectState {
            view: View { camera_distance: 55.0, brush_radius: 7.0, ..View::default() },
            keys: Vec::new(),
        };
        let mut bytes = Vec::new();
        write(&mut bytes, &volume, &state).expect("write failed");

        // Back to version 1: stamp the old number in, and drop the trailer.
        bytes[8..10].copy_from_slice(&1u16.to_le_bytes());
        bytes.truncate(bytes.len() - EMPTY_TRAILER_BYTES);

        let (loaded, loaded_state) =
            read(&mut bytes.as_slice()).expect("a version 1 file was refused");
        assert_eq!(loaded_state.view, state.view, "the old file's settings came back wrong");
        assert!(loaded_state.keys.is_empty(), "a file with no trailer gained keys");
        assert_eq!(
            loaded.brick_coords().count(),
            volume.brick_coords().count(),
            "the geometry did not survive"
        );
    }

    #[test]
    fn a_layout_newer_than_this_build_is_still_refused() {
        // Widening the accepted range must not have widened it upward. A file
        // from a future build may put anything after the header, and reading
        // it as though it were this one is the plausible-looking-sculpt failure
        // the version numbers exist to prevent.
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted(), &ProjectState::default()).expect("write failed");
        bytes[8..10].copy_from_slice(&(CONTAINER_VERSION + 1).to_le_bytes());
        match read(&mut bytes.as_slice()) {
            Err(ProjectError::ContainerVersion { found, supported }) => {
                assert_eq!(found, CONTAINER_VERSION + 1);
                assert_eq!(supported, CONTAINER_VERSION);
            }
            Ok(_) => panic!("a newer layout was accepted"),
            Err(other) => panic!("expected a version refusal, got {other}"),
        }
        // And so is one older than anything that ever existed.
        bytes[8..10].copy_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            read(&mut bytes.as_slice()),
            Err(ProjectError::ContainerVersion { found: 0, .. })
        ));
    }

    #[test]
    fn an_absurd_key_count_is_refused_before_anything_is_read() {
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted(), &ProjectState::default()).expect("write failed");
        let at = bytes.len() - EMPTY_TRAILER_BYTES;
        bytes[at..].copy_from_slice(&u32::MAX.to_le_bytes());
        match read(&mut bytes.as_slice()) {
            Err(ProjectError::TooManyKeys { keys, limit }) => {
                assert_eq!(keys, u32::MAX);
                assert_eq!(limit, MAX_KEYS);
            }
            Ok(_) => panic!("a count of four billion keys was accepted"),
            Err(other) => panic!("expected a key count refusal, got {other}"),
        }
    }

    #[test]
    fn a_key_at_a_nonsense_position_is_repaired_rather_than_refused() {
        // Keys are conveniences like the camera is. A key in the wrong place
        // is not a reason to lose the sculpt saved beside it -- but it does
        // have to land somewhere on the strip, or it is a key nothing can
        // reach.
        for offending in [f32::NAN, f32::INFINITY, -5.0, 900.0] {
            let state = ProjectState {
                view: View::default(),
                keys: vec![Keyframe { at: offending, view: View::default() }],
            };
            let (_, loaded) = round_trip(&sculpted(), &state);
            let at = loaded.keys[0].at;
            assert!(
                (0.0..=1.0).contains(&at),
                "a key written at {offending} came back at {at}, which is off the strip"
            );
        }
    }

    #[test]
    fn a_key_with_a_nonsense_view_is_repaired_the_way_the_session_view_is() {
        // The repair lives in `read_view`, which both go through, so this is
        // really asserting that they still do.
        let state = ProjectState {
            view: View::default(),
            keys: vec![Keyframe {
                at: 0.5,
                view: View {
                    camera_target: Vec3::splat(f32::NAN),
                    camera_distance: f32::INFINITY,
                    ..View::default()
                },
            }],
        };
        let (_, loaded) = round_trip(&sculpted(), &state);
        assert!(loaded.keys[0].view.camera_target.is_finite());
        assert!(loaded.keys[0].view.camera_distance.is_finite());
    }
}
