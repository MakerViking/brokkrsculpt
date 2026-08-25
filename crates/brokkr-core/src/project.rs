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
//! and one stream of bricks per body. Copying the ZIP shape would mean either a
//! new dependency or a hand rolled ZIP reader, both of which cost more than this
//! format is worth. What *is* worth copying from `container.rs` is its
//! discipline, and that is all below: two independent version numbers, a size
//! cap checked before anything is read, and never trusting a number from inside
//! the file without bounding it first.
//!
//! # The exact layout, at container version 3
//!
//! ```text
//!  0..8    MAGIC  b"BROKKR\x00\x01"
//!  8..10   container u16 LE = 3
//! 10..12   field     u16 LE            a RANGE, see `OLDEST_FIELD_VERSION`
//! 12..16   BRICK_DIM u32 LE            refused on mismatch
//! 16..20   NARROW_BAND f32 LE          refused on mismatch
//! 20..24   voxel_size f32 LE           DOCUMENT-wide: bodies share the lattice
//! 24..63   View, 39 bytes
//! --- version 3 begins here; a version 1 or 2 file has the u64 brick count at 63
//! 63..67   node_count   u32 LE   1 ..= MAX_NODES
//! 67..71   active_index u32 LE   an index into the table; must name a body
//! 71..     node_count records of EXACTLY NODE_RECORD_BYTES, in PREORDER
//! --- then, IN TABLE ORDER, one brick stream per body record
//!          brick_count u64 LE, then that many brick records
//! --- then the key trailer, LAST
//!          key_count u32 LE, then 43 bytes per key
//! ```
//!
//! **Bytes 0..63 are byte for byte what version 2 wrote**, which is what lets
//! every header offset the tests hardcode go on aiming at what it names. New
//! header fields go after the view, never before the lattice constants.
//!
//! **The per-body brick streams are CONSECUTIVE, and the reader is
//! `&mut impl Read` that never seeks.** There is no length prefix in front of a
//! body's stream and there cannot be one: `read(&mut bytes.as_slice())`
//! throughout the tests and `BufReader<File>` in the shell both depend on the
//! reader being a forward stream. So appending a second stream per body -- a
//! mask, a colour, anything per voxel -- is a [`FIELD_VERSION`] change and
//! **never** a container change: the container says where the sections are, and
//! the field version says what is in them.
//!
//! # Why the node table precedes the geometry
//!
//! Version 2 appended a strict *suffix*, so reading a version 1 file was a
//! matter of not looking for the trailer -- no conversion, no second code path.
//! A node table cannot be a suffix, because the reader never seeks and the
//! table is what says how many brick streams follow. So the table sits before
//! the geometry, and a version 1 or 2 file gets **one synthesized default
//! body**. That is still not a converter and still one geometry loop, but it is
//! a different shape, and this is where it is written down.
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

use crate::body::{Document, MAX_BODIES, MAX_NODES, Node, NodeId, NodeMeta};
use crate::brick::{BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND, OUTSIDE};
use crate::volume::Volume;

/// Magic bytes. Long enough that a truncated or mistyped file is rejected
/// immediately rather than part way through a brick.
const MAGIC: &[u8; 8] = b"BROKKR\x00\x01";

/// Version of the file's *layout*: the header fields and the order of sections.
///
/// 2 added the timeline key trailer after the brick stream. 3 added the node
/// table between the view and the geometry, and made the brick stream one
/// stream per body. Neither bump had to orphan the files already written, and
/// neither did: see [`OLDEST_CONTAINER_VERSION`].
///
/// **Folders do not bump this.** They widen the legal range of two bytes that
/// version 3 already reserves in every node record, which is the whole reason
/// those two bytes are reserved rather than absent.
const CONTAINER_VERSION: u16 = 3;

/// The oldest layout this build still reads.
///
/// Version 1 is version 2 without the key trailer, so reading one is a matter
/// of not looking for the trailer. Version 2 is version 3 without the node
/// table, so reading one is a matter of synthesizing a single default body --
/// which is a different shape but still no conversion and still one geometry
/// loop. This exists because the alternative was refusing every `.brokkr` file
/// in existence to add a feature none of them use, which is not a trade a save
/// format gets to make. **Only add a version here when the old layout is
/// genuinely still readable**; the point of refusing an unknown one is that a
/// plausible-looking sculpt made of misread numbers is worse than an error.
const OLDEST_CONTAINER_VERSION: u16 = 1;

/// Version of how a brick's field values are *encoded*. Kept separate from the
/// container version, because the two change for different reasons — a new
/// header field is not a new encoding, and a new encoding is not a new layout.
/// SindriCAD's container learned this distinction the hard way.
///
/// This is the NEWEST encoding this build understands, and what it *writes* is
/// [`lowest_field_version`] rather than this. See [`OLDEST_FIELD_VERSION`] for
/// why both halves of that are needed.
const FIELD_VERSION: u16 = 1;

/// The oldest field encoding this build still reads.
///
/// A range rather than the exact equality this check used to be, and it is the
/// load-bearing half of the version discipline on this axis. Per-voxel payload
/// -- a colour, a mask -- rides the field version, so the day one of those
/// ships this constant is what stops every file ever written from being refused
/// by the build that adds it, with an error that reads backwards.
///
/// It buys one direction only: a NEW build reading an OLD file. The other
/// direction is bought by [`lowest_field_version`], and both are needed. A
/// writer that stamped the newest version on every save would make each of
/// today's documents unreadable by every build that predates the next
/// encoding, whether or not the document uses any of it.
const OLDEST_FIELD_VERSION: u16 = 1;

/// The lowest field encoding that can express everything in `doc`.
///
/// **Write the lowest version the document NEEDS, never the newest the build
/// knows.** There is nothing per-voxel beyond the field itself yet, so this is
/// [`OLDEST_FIELD_VERSION`] for every document -- and it is here, with its
/// caller and its test, from the commit that adds the range check, because the
/// rule is what makes the next bump survivable in both directions and it is
/// unenforceable once a second encoding exists and this function does not.
fn lowest_field_version(doc: &Document) -> u16 {
    // Nothing per-voxel beyond the field itself exists yet, so nothing in the
    // document can ask for a newer encoding. The parameter is here because the
    // CALLER has to be written to ask the document rather than to reach for
    // the constant, and that is the whole of the rule.
    let _ = doc;
    OLDEST_FIELD_VERSION
}

/// Largest sculpt this will read, as a guard against a corrupt or hostile
/// header asking for an allocation the machine cannot serve.
///
/// Eight gigabytes is far above any real sculpt — the densest measured model,
/// eleven million triangles at a 0.055 mm voxel, is about 1 GB resident — and
/// far below anything that would take the process down.
///
/// **It is a running DOCUMENT total and not a per-body one.** Checked per body
/// it would let a file claiming 64 bodies ask for 64 times this, which is the
/// whole cap gone for the price of one extra node record. The ceiling is
/// closer than it looks: the 6 GiB the renderer's own guard allows is 49,152
/// dense bricks, so this count cap sits only 33% above the RAM cap.
///
/// Note this is an *absolute* cap and not a compression ratio test. SindriCAD
/// tried a ratio test and removed it because it fires on legitimately sparse
/// data, and a narrow band field clamped to ±3 is about as sparse as data gets.
const MAX_BRICKS: u64 = 8 * 1024 * 1024 * 1024 / BRICK_BYTES as u64;

/// How much system memory the whole DOCUMENT may occupy in voxel data.
///
/// The mesh pool grows itself buffers now, so it is no longer the first thing
/// a fine resample runs out of: the dragon at 0.0565 mm sits at 48% of the
/// pool and 4.15 GB of RAM. Without this the detail buttons would walk a
/// machine into swap.
///
/// Matches [`crate::voxelise`]'s own import ceiling, and carries the same
/// caveat: it is a guess at what a machine can spare, made without asking the
/// machine. Reading the available memory and taking a fraction of it is the
/// honest version and is not done.
///
/// **It lives here, beside [`MAX_BRICKS`], rather than in `brokkr-app` where
/// it was written.** It was read by exactly one function -- the resample guard
/// -- so `brokkr-core` could not reach it, and `brokkr-core` is where the two
/// things that have to agree with it now are: [`Document::growth_guard`],
/// which refuses a body the document has no room for, and the mask field,
/// whose bytes ride beside the field they mask. CI fails the build if the
/// dependency ever runs the other way, so a constant the engine needs cannot
/// live in the shell.
///
/// An `f64` rather than a `u64` because every consumer multiplies it by a
/// predicted growth factor; see [`crate::voxelise`]'s `MAX_IMPORT_BYTES`,
/// which is the same number in the same shape for the same reason.
pub const MAX_VOLUME_BYTES: f64 = 6.0 * 1024.0 * 1024.0 * 1024.0;

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

/// Bytes in one node record, and it is FIXED at forty.
///
/// Load-bearing rather than tidy. The tests find the start of the brick stream
/// by *measuring* the length of an empty write, so a fixed record grows the
/// empty header by a constant and every offset self-adjusts. A record whose
/// length depended on the name would be right for the default document and
/// wrong for every renamed one, which is the shape of failure that passes the
/// suite and corrupts a user's file.
const NODE_RECORD_BYTES: usize = 40;

/// Bytes a node's name occupies in its record: fixed, UTF-8, NUL padded.
///
/// **The name is the bytes up to the first NUL, or all thirty-two of them if
/// there is none.** Stated that way because a full thirty-two byte name has no
/// terminator and must still round-trip -- the rename field in the interface
/// enforces exactly this length, so the application actively encourages
/// producing the one name a "must be NUL terminated" rule would destroy.
const NAME_BYTES: usize = 32;

/// `kind` for a row that holds a field.
const KIND_BODY: u8 = 0;

/// This row's own eye.
const FLAG_VISIBLE: u16 = 1 << 0;

/// Folders only, and repaired to false on a body row.
const FLAG_COLLAPSED: u16 = 1 << 1;

/// The fourteen flag bits that must be zero.
///
/// Refused rather than ignored, which reserves them at no cost on disk: a build
/// that later *sets* one of these makes its files unreadable by every shipped
/// version 3 build, so the refusal is what stops a future increment from
/// spending a bit here by accident. Mask presence, when it arrives, is implied
/// by a non-empty mask stream and spends none of them.
const RESERVED_FLAGS: u16 = !(FLAG_VISIBLE | FLAG_COLLAPSED);

/// The record's own arithmetic, checked at build time rather than trusted: four
/// bytes of id, two of flags, one of kind, one of depth, then the name field.
/// Widening the name without widening the record would move every brick stream
/// in the file while leaving the tests that *measure* the header perfectly
/// happy, because they measure the same wrong thing.
const _: () = assert!(NODE_RECORD_BYTES == 4 + 2 + 1 + 1 + NAME_BYTES);

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

/// How far a read got before it returned.
///
/// The reader answers with one `Result`, and the error variant is not enough to
/// say *where* it gave up. [`ProjectError::NonFiniteValue`] is raised for a
/// `voxel_size` that is not a positive finite number — read at byte 20, in the
/// header — and again for a distance inside a brick, forty-three bytes and a
/// whole section later. Anything that wants to distinguish "this file's header
/// is wrong" from "this file's geometry is corrupt" has to be told, not left to
/// infer it from the variant.
///
/// That inference is not hypothetical. The fuzz control below inferred it from
/// a closed four-variant match for two versions of this file and counted every
/// header refusal as having reached the geometry; against a 4 MB corpus that
/// was four bytes' worth of noise, but the corpus is going to get much smaller.
/// So the reader reports its own reach and the control counts it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Progress {
    /// Nodes read from the node table.
    ///
    /// Zero for a version 1 or 2 file, and that is a reading rather than a
    /// miss: those layouts carry no table at all, so the one implicit body they
    /// hold was synthesized rather than read. A version 3 file that refuses
    /// part way through its table leaves this holding the rows that parsed,
    /// which is the reach this type exists to report.
    pub nodes: u32,
    /// Bricks the reader has *started*, which is not the same as bricks it
    /// finished.
    ///
    /// Counted at the top of each iteration, so after a successful read it is
    /// the count the header claimed, and after a failure inside the loop it is
    /// the one-based index of the brick that failed. Both readings are wanted:
    /// the first is a progress figure and the second is the reach this type
    /// exists to report. Counting *completed* bricks instead would report a
    /// file whose very first brick is corrupt as never having reached the
    /// geometry at all, which is the miscount in the other direction.
    pub bricks: u64,
    /// Values the reader repaired rather than refused.
    ///
    /// The node table's repairs, and only those: a name that was not UTF-8 or
    /// was empty, a `collapsed` bit set on a body row, an `active_index` that
    /// named nothing or named a folder. The camera and key repairs in
    /// [`read_view`] are deliberately *not* counted here — they happen on every
    /// ordinary file that was saved with a NaN camera, so folding them in would
    /// give the field a second meaning that the first caller to use it would
    /// have to subtract back out.
    pub repairs: u32,
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
    /// A node table with no rows, or with more than [`MAX_NODES`].
    NodeCount {
        found: u32,
        limit: u32,
    },
    /// More body rows than [`MAX_BODIES`].
    TooManyBodies {
        found: usize,
        limit: usize,
    },
    /// A node id of 0, which is reserved for "no node", or of `u32::MAX`, which
    /// is refused because the next id is reconstituted as `max(id) + 1` and
    /// would overflow.
    ReservedNodeId {
        found: u32,
    },
    /// Two rows sharing one id, which aliases both the renderer's mesh pool key
    /// and the routing of an undo entry.
    DuplicateNodeId {
        found: u32,
    },
    /// A node record with one of the fourteen reserved flag bits set.
    ReservedFlags {
        found: u16,
    },
    /// A node `kind` outside the legal set. It decides whether a brick stream
    /// follows, so it sits on the geometry side of the refuse/repair line for
    /// the same reason [`ProjectError::UnknownBrickTag`] does: repairing it
    /// would misalign every stream after it.
    UnknownNodeKind(u8),
    /// A node at a non-zero `depth`, which container version 3 reserves. The
    /// increment that makes folders representable replaces this refusal with a
    /// clamp, because a clamped depth is still a valid forest.
    ReservedDepth {
        found: u8,
    },
    /// A node table holding no body rows at all. Nothing would then answer
    /// "which body is active", and the first thing to touch the active body
    /// would panic on a file that had otherwise loaded cleanly.
    NoBodies,
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
                write!(f, "{bricks} bricks is past this build's limit of {limit}")
            }
            ProjectError::NodeCount { found, limit } => {
                write!(f, "the file claims {found} rows, and a document holds 1 to {limit}")
            }
            ProjectError::TooManyBodies { found, limit } => {
                write!(f, "the file claims {found} bodies, and the limit is {limit}")
            }
            ProjectError::ReservedNodeId { found } => {
                write!(f, "the file gives a body the reserved id {found}")
            }
            ProjectError::DuplicateNodeId { found } => {
                write!(f, "the file gives two rows the same id, {found}")
            }
            ProjectError::ReservedFlags { found } => {
                write!(f, "the file sets a reserved flag bit: {found:#018b}")
            }
            ProjectError::UnknownNodeKind(kind) => write!(f, "unknown row kind {kind}"),
            ProjectError::ReservedDepth { found } => {
                write!(f, "the file nests a row {found} deep, and this build has no folders")
            }
            ProjectError::NoBodies => write!(f, "the file holds no bodies at all"),
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

/// Write a document.
///
/// Bricks go out in a deterministic order and the body list is a `Vec` rather
/// than a map, so saving the same document twice produces byte identical files.
/// That is worth having for its own sake and it makes a round trip test able to
/// compare bytes rather than semantics.
///
/// **Takes the whole [`Document`] and never one [`Volume`].** During the move
/// to N bodies a signature taking the active volume went on compiling at every
/// call site while silently saving one body of many, which is a whole document
/// lost with "saved" in the status line.
pub fn write(out: &mut impl Write, doc: &Document, state: &ProjectState) -> Result<()> {
    refuse_what_could_not_be_read_back(doc)?;

    out.write_all(MAGIC)?;
    write_u16(out, CONTAINER_VERSION)?;
    // The LOWEST encoding this document needs, not the newest this build knows.
    write_u16(out, lowest_field_version(doc))?;

    // The lattice. Written as the values themselves rather than as a hash, so a
    // mismatch can say what it found.
    write_u32(out, BRICK_DIM as u32)?;
    write_f32(out, NARROW_BAND)?;

    // One voxel size for the whole document, because bodies share the lattice.
    write_f32(out, doc.voxel_size())?;

    write_view(out, &state.view)?;

    write_u32(out, doc.node_count() as u32)?;
    write_u32(out, active_index(doc) as u32)?;
    for node in doc.nodes() {
        write_node_record(out, node)?;
    }

    // One brick stream per body, in TABLE ORDER, with nothing between them.
    // `Document::bodies` yields the rows in display order, which is table
    // order, which is the order the reader consumes them in.
    for (_, volume) in doc.bodies() {
        write_bricks(out, volume)?;
    }

    // The key trailer, after the bricks rather than in the header. A version 1
    // file is exactly this file without it, which is what lets one still be
    // read: the brick counts say where the geometry ends, and there is simply
    // nothing after it.
    write_u32(out, state.keys.len() as u32)?;
    for key in &state.keys {
        write_f32(out, key.at)?;
        write_view(out, &key.view)?;
    }

    out.flush()?;
    Ok(())
}

/// Refuse to write a file this build's own reader would refuse.
///
/// **Every one of these was a read-side check only, and that asymmetry is the
/// bug.** Add bodies over an afternoon, cross the brick cap, press ctrl+S: the
/// write succeeds, the status says "saved", the asterisk clears and the crash
/// net is deleted -- and reopening the file refuses it. The work is then in a
/// file nothing will open, with every safety net cleared on the writer's word.
/// So the caps are checked on both sides of the format and a symmetry test
/// pins them together.
///
/// Checked before a single byte goes out, so a refusal leaves the output
/// untouched rather than half written.
fn refuse_what_could_not_be_read_back(doc: &Document) -> Result<()> {
    let nodes = doc.node_count();
    if nodes == 0 || nodes > MAX_NODES {
        return Err(ProjectError::NodeCount { found: nodes as u32, limit: MAX_NODES as u32 });
    }
    let bodies = doc.body_count();
    if bodies == 0 {
        return Err(ProjectError::NoBodies);
    }
    if bodies > MAX_BODIES {
        return Err(ProjectError::TooManyBodies { found: bodies, limit: MAX_BODIES });
    }
    // The document total, exactly as the reader accumulates it.
    let bricks: u64 = doc.bodies().map(|(_, volume)| volume.brick_count() as u64).sum();
    if bricks > MAX_BRICKS {
        return Err(ProjectError::TooLarge { bricks, limit: MAX_BRICKS });
    }
    Ok(())
}

/// Where the active body sits in the node table.
///
/// The search can fail, which is why this is not an `expect`: deleting a folder
/// that transitively contains the active body removes it, and a stale `active`
/// would then kill the session inside a save, on the UI thread, with the user's
/// file already open for writing. A release build writes a repairable file --
/// the reader moves `active_index` to the first body row -- and a debug build
/// fails loudly in whatever test produced the stale id.
fn active_index(doc: &Document) -> usize {
    let found = doc.index_of(doc.active());
    debug_assert!(found.is_some(), "the document's active id is not in its own node list");
    found.unwrap_or(0)
}

/// One forty-byte node record.
///
/// **The field order at offset +4 is load-bearing and not cosmetic.** It is
/// `flags u16`, then `kind u8`, then `depth u8`. A version 3 file can only
/// carry `01 00 00 00` or `00 00 00 00` there, which under this layout reads as
/// `flags = 1/0, kind = 0, depth = 0` -- correct, bit for bit. Reverse `kind`
/// and `flags` and a visible body's `flags = 1` reads as `kind = 1`, a folder:
/// every body becomes an empty folder with no brick stream, and the reader then
/// takes the first body's brick count as the key trailer's key count. That is a
/// plausible-looking sculpt made of misread numbers under an unchanged version
/// number, which is the exact failure the lattice check exists to prevent.
/// Someone will want to tidy this into `kind, depth, flags`; this comment is
/// what stops them.
fn write_node_record(out: &mut impl Write, node: &Node) -> Result<()> {
    write_u32(out, node.id.0)?;

    let mut flags = 0u16;
    if node.visible {
        flags |= FLAG_VISIBLE;
    }
    // `collapsed` belongs to a folder row, and the reader repairs it away on a
    // body. Writing it on a body would make write -> read -> write differ by
    // one bit, which is the property the save path's own verification rests on.
    if node.collapsed && !node.is_body() {
        flags |= FLAG_COLLAPSED;
    }
    write_u16(out, flags)?;

    // `kind` and `depth`, RESERVED AT ZERO until folders exist. They are
    // written as literal zeroes rather than derived from the node, because this
    // build's reader refuses a non-zero value in either -- deriving them would
    // let a document that should never exist produce a file this build cannot
    // read. The increment that makes folders representable changes both ends
    // together.
    debug_assert!(node.is_body(), "there are no folder rows to write yet");
    debug_assert_eq!(node.depth(), 0, "there is no nesting to write yet");
    out.write_all(&[KIND_BODY, 0])?;

    out.write_all(&name_bytes(&node.name))?;
    Ok(())
}

/// A name in its fixed field: at most [`NAME_BYTES`] of UTF-8, NUL padded.
///
/// **Truncated on a CHAR BOUNDARY.** Slicing at byte 32 through the middle of a
/// multi-byte sequence writes bytes that are not UTF-8; the reader then
/// correctly repairs the name to `Body {n}`, and the user silently loses a name
/// this build wrote itself. `is_char_boundary(0)` is true, so the walk back
/// always terminates.
fn name_bytes(name: &str) -> [u8; NAME_BYTES] {
    let mut end = name.len().min(NAME_BYTES);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = [0u8; NAME_BYTES];
    out[..end].copy_from_slice(&name.as_bytes()[..end]);
    out
}

/// One body's brick stream: a count, then that many brick records.
///
/// **The count is taken from the pairs actually resolved, not from the
/// coordinate list.** The two can differ, and at version 2 the damage was
/// bounded because the trailer was last and short. Now the streams are
/// consecutive: a body that writes fewer records than it declared makes the
/// reader consume the NEXT body's coordinates as this body's bricks, and every
/// remaining body in the file is misparsed into plausible-looking rubbish.
fn write_bricks(out: &mut impl Write, volume: &Volume) -> Result<()> {
    let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
    coords.sort_unstable();
    let pairs: Vec<(BrickCoord, &Brick)> = coords
        .into_iter()
        .filter_map(|coord| volume.brick(coord).map(|brick| (coord, brick)))
        .collect();

    write_u64(out, pairs.len() as u64)?;
    for (coord, brick) in pairs {
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
    Ok(())
}

/// Read a sculpt.
///
/// A thin wrapper over [`read_reporting`] for the callers that only want the
/// sculpt. Keeping the reach reporting out of the public signature is
/// deliberate: the shell opens a file and either gets a model or gets an error
/// it can show, and threading an out-parameter through `File > Open` to be
/// dropped at the other end would be ceremony.
pub fn read(input: &mut impl Read) -> Result<(Document, ProjectState)> {
    read_reporting(input, &mut Progress::default())
}

/// What a file says it holds, from its header and node table alone.
///
/// A few hundred bytes rather than gigabytes: the whole point is that this can
/// be read back straight after a save, on the thread that draws, to check that
/// what landed on disk is the document that was in memory. See
/// `Brokkr::save_project`, which will not delete the crash net until it agrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outline {
    pub voxel_size: f32,
    /// Rows in the node table, bodies and folders together.
    pub nodes: usize,
    /// Rows that carry a field.
    pub bodies: usize,
}

/// Read a file's header and node table, and stop before the geometry.
///
/// Every refusal the full reader makes about the header and the table is made
/// here too, because it is the same code. What it does NOT do is validate one
/// brick, so a file that passes this can still be refused by [`read`].
pub fn read_outline(input: &mut impl Read) -> Result<Outline> {
    let header = read_header(input)?;
    let mut progress = Progress::default();
    let (rows, _) = read_table_or_synthesize(input, header.container, &mut progress)?;
    Ok(Outline {
        voxel_size: header.voxel_size,
        nodes: rows.len(),
        bodies: rows.iter().filter(|row| row.is_body).count(),
    })
}

/// One row of a file's node table: everything about it except the field it
/// may own.
struct Row {
    meta: NodeMeta,
    is_body: bool,
}

/// Everything the fixed header carries, once it has been checked.
struct Header {
    container: u16,
    voxel_size: f32,
    view: View,
}

/// The first sixty-three bytes, which are the same in every version.
fn read_header(input: &mut impl Read) -> Result<Header> {
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
    // A RANGE on this axis too, and it is the whole of what the next per-voxel
    // encoding asks of this commit. Exact equality here would refuse every file
    // ever written on the day a mask or a colour bumps the field version --
    // including every multi-body file saved between now and then -- with an
    // error that reads backwards. Still only downward: a file written by a
    // NEWER encoding is refused flatly, because reading one as though it were
    // this one is the plausible-looking-sculpt failure again.
    let field = read_u16(input)?;
    if !(OLDEST_FIELD_VERSION..=FIELD_VERSION).contains(&field) {
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

    Ok(Header { container, voxel_size, view: read_view(input)? })
}

/// The node table of a version 3 file, or the one implicit body an older one
/// holds.
///
/// **This is the whole of the compatibility branch**, and it has the shape of
/// the one already above it for the key trailer. A version 1 or 2 file carries
/// no table, so what it holds is one default body -- and the brick loop below
/// then runs exactly once, reading the u64 at byte 63 exactly as it always did.
fn read_table_or_synthesize(
    input: &mut impl Read,
    container: u16,
    progress: &mut Progress,
) -> Result<(Vec<Row>, usize)> {
    if container >= 3 {
        return read_node_table(input, progress);
    }
    let meta = NodeMeta {
        id: NodeId(1),
        depth: 0,
        name: Document::FIRST_BODY_NAME.to_string(),
        visible: true,
        collapsed: false,
    };
    // `progress.nodes` stays 0: nothing was read from a table that is not there.
    Ok((vec![Row { meta, is_body: true }], 0))
}

/// Read the flat preorder node table, and the index of the active row.
///
/// **Cycles are unrepresentable here rather than merely detected**, and that is
/// the whole reason for this encoding. A row's position in the tree is
/// `(preorder index, depth)` and nothing else: there is no parent pointer to
/// point at a descendant, no traversal, no `visited` set and no recursion, so a
/// crafted file has no lever that could make this loop run twice over one row.
/// Whatever the depth column holds, the result is a forest. That is what keeps
/// the mutation fuzz below meaningful without a recursive parser for it to
/// overflow.
///
/// Validating the tree is therefore a fold over one `u8` -- and at version 3 it
/// is the degenerate case of that fold, since `depth` is reserved at zero and
/// refused otherwise. The increment that makes folders representable replaces
/// the refusal with the clamp
/// `depth[i] = min(depth[i], depth[i - 1] + 1, MAX_DEPTH - 1)`, which is closed
/// over the invariant and so needs no error variant of its own.
///
/// A crafted file's only levers are `node_count`, bounded before the allocation
/// it decides; `kind`, a set membership test; and `depth`, above.
fn read_node_table(input: &mut impl Read, progress: &mut Progress) -> Result<(Vec<Row>, usize)> {
    let node_count = read_u32(input)?;
    if node_count == 0 || node_count as usize > MAX_NODES {
        return Err(ProjectError::NodeCount { found: node_count, limit: MAX_NODES as u32 });
    }
    let active_index = read_u32(input)?;

    // Bounded above, so this allocation is at most MAX_NODES records however
    // hostile the file: the table is read and validated whole before one brick
    // is allocated.
    let mut rows: Vec<Row> = Vec::with_capacity(node_count as usize);
    let mut bodies = 0usize;
    for index in 0..node_count as usize {
        let id = read_u32(input)?;
        // Zero is "no node" and `u32::MAX` would overflow the `max(id) + 1`
        // that reconstitutes the next id; a duplicate aliases both the mesh
        // pool's key and the routing of an undo entry.
        if id == 0 || id == u32::MAX {
            return Err(ProjectError::ReservedNodeId { found: id });
        }
        if rows.iter().any(|row| row.meta.id.0 == id) {
            return Err(ProjectError::DuplicateNodeId { found: id });
        }

        let flags = read_u16(input)?;
        if flags & RESERVED_FLAGS != 0 {
            return Err(ProjectError::ReservedFlags { found: flags });
        }

        let kind = read_exact::<1>(input)?[0];
        if kind != KIND_BODY {
            return Err(ProjectError::UnknownNodeKind(kind));
        }
        let depth = read_exact::<1>(input)?[0];
        if depth != 0 {
            return Err(ProjectError::ReservedDepth { found: depth });
        }

        let name = read_name(input, index, progress)?;

        // Every row is a body while `kind` must be zero. The count is kept
        // rather than derived so that the two checks that need it -- the body
        // cap and the "no bodies at all" refusal -- read the same way from the
        // increment that makes a folder row representable.
        let is_body = kind == KIND_BODY;
        if is_body {
            bodies += 1;
            if bodies > MAX_BODIES {
                return Err(ProjectError::TooManyBodies { found: bodies, limit: MAX_BODIES });
            }
        }

        let mut collapsed = flags & FLAG_COLLAPSED != 0;
        if collapsed && is_body {
            // Repaired, not refused: it decides only whether a triangle is
            // drawn beside a row.
            collapsed = false;
            progress.repairs += 1;
        }

        rows.push(Row {
            meta: NodeMeta {
                id: NodeId(id),
                depth,
                name,
                visible: flags & FLAG_VISIBLE != 0,
                collapsed,
            },
            is_body,
        });
        progress.nodes += 1;
    }

    // A table of folders and nothing else parses perfectly, consumes exactly
    // the right number of bytes, and leaves a document with no body for
    // `active` to name -- so the first thing that touches the active body
    // panics on a file that loaded without complaint. The mutation fuzz cannot
    // catch it either: its band assertion loops over zero bodies and passes.
    // Unreachable while `kind` must be zero, and one line ahead of the
    // increment that makes it reachable.
    if bodies == 0 {
        return Err(ProjectError::NoBodies);
    }

    // Moved to the first BODY row rather than to row 0, because row 0 may be a
    // folder once folders exist.
    let active = match rows.get(active_index as usize) {
        Some(row) if row.is_body => active_index as usize,
        _ => {
            progress.repairs += 1;
            rows.iter().position(|row| row.is_body).expect("a table with no bodies was refused")
        }
    };
    Ok((rows, active))
}

/// One name field, repaired rather than refused.
///
/// A name decides only what a row is called, so a file whose geometry is intact
/// and whose name field is rubbish should open and show the sculpt under a
/// default name -- exactly as a NaN camera does.
fn read_name(input: &mut impl Read, index: usize, progress: &mut Progress) -> Result<String> {
    let field: [u8; NAME_BYTES] = read_exact(input)?;
    // Up to the first NUL, or the whole field when there is none. A full
    // thirty-two byte name has no terminator and has to round-trip.
    let end = field.iter().position(|byte| *byte == 0).unwrap_or(NAME_BYTES);
    match std::str::from_utf8(&field[..end]) {
        Ok(name) if !name.is_empty() => Ok(name.to_string()),
        _ => {
            progress.repairs += 1;
            Ok(format!("Body {}", index + 1))
        }
    }
}

/// Read a sculpt, recording how far the read got in `progress`.
///
/// `progress` is written as the read proceeds and is left holding whatever was
/// reached when this returns, error or not — that is the whole point of it, so
/// it must not be reset or rolled back on the way out.
pub(crate) fn read_reporting(
    input: &mut impl Read,
    progress: &mut Progress,
) -> Result<(Document, ProjectState)> {
    let header = read_header(input)?;
    let mut state = ProjectState { view: header.view, keys: Vec::new() };

    let (rows, active) = read_table_or_synthesize(input, header.container, progress)?;

    // The brick streams, one per body, consecutive and in table order.
    let mut total = 0u64;
    let mut loaded: Vec<(NodeMeta, Volume)> = Vec::with_capacity(rows.len());
    for row in rows {
        debug_assert!(row.is_body, "a version 3 file's rows are all bodies");
        let count = read_u64(input)?;
        // The running DOCUMENT total, checked before this body's bricks are
        // touched. A per-body check would let sixty-four bodies ask for
        // sixty-four times the cap.
        //
        // It cannot be a *sum of every count up front*: the counts are
        // interleaved with the streams they describe and the reader never
        // seeks, so a file whose first body alone is legal necessarily gets
        // that body read before the second count is even visible. What this
        // does buy is that no body is allocated once the declared total is
        // past the cap, which is what bounds the damage.
        total = total.saturating_add(count);
        if total > MAX_BRICKS {
            return Err(ProjectError::TooLarge { bricks: total, limit: MAX_BRICKS });
        }
        loaded.push((row.meta, read_bricks(input, count, header.voxel_size, progress)?));
    }

    // The key trailer, which only version 2 and later carry. A version 1 file
    // ends with its last brick, and reading one is a matter of not looking.
    if header.container >= 2 {
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

    let mut doc = Document::from_table(header.voxel_size, loaded, active);
    // Nothing has been meshed yet, and the renderer holds whatever the previous
    // model left. Marking everything dirty is what makes the load visible.
    doc.mark_everything_dirty();
    Ok((doc, state))
}

/// One body's brick stream, into a volume of its own.
fn read_bricks(
    input: &mut impl Read,
    count: u64,
    voxel_size: f32,
    progress: &mut Progress,
) -> Result<Volume> {
    let mut volume = Volume::new(voxel_size);
    for _ in 0..count {
        // Counted here rather than after the insert, so that a file whose very
        // first brick is corrupt still reports the geometry as reached. See
        // `Progress::bricks`.
        progress.bricks += 1;
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
    Ok(volume)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::NodeId;
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

    /// The sculpt every test below saves, as the one-body document the format
    /// actually takes.
    fn sculpted_doc() -> Document {
        Document::from_volume(sculpted())
    }

    /// A ball as a one-body document, for the tests that only need some bricks
    /// to corrupt.
    fn ball(voxel_size: f32, radius: f32) -> Document {
        let mut volume = Volume::new(voxel_size);
        volume.seed_sphere(Vec3::ZERO, radius);
        Document::from_volume(volume)
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
    ///
    /// **Single body and folder free**, like [`first_brick_at`]: it assumes the
    /// file ends with one body's last brick and then the empty trailer. Add a
    /// second body to a fixture that uses this and it aims into that body
    /// rather than the first, silently.
    fn last_distance_at(bytes: &[u8]) -> usize {
        bytes.len() - EMPTY_TRAILER_BYTES - 4
    }

    /// Where the first brick's coordinate starts, found by measuring the header
    /// of an empty document rather than by counting field widths.
    ///
    /// **This is why the node record is a fixed forty bytes.** The measurement
    /// self-adjusts for anything that grows the header *before* the brick
    /// stream -- the node table did, and not one offset here had to move -- but
    /// it cannot adjust for a record whose length depends on a name.
    ///
    /// **Single body and folder free.** It measures a one-body document, so the
    /// answer is where the FIRST body's stream begins. A future test that adds
    /// a folder or a second body to the shared fixture gets a silently wrong
    /// offset and a passing assertion, not a failure. It also does not
    /// self-adjust for anything appended AFTER the bricks -- see
    /// `EMPTY_TRAILER_BYTES`.
    fn first_brick_at() -> usize {
        let mut empty = Vec::new();
        write(&mut empty, &Document::new(0.5), &ProjectState::default()).expect("write failed");
        empty.len() - EMPTY_TRAILER_BYTES
    }

    /// Where the node table begins: the whole of the version 1 and 2 header.
    ///
    /// Confirmed against the real files with `od` rather than counted from the
    /// writer, and asserted against the writer in
    /// [`the_version_3_header_is_the_version_2_header_plus_a_node_table`].
    const NODE_TABLE_AT: usize = 63;

    /// Where one node record begins.
    fn record_at(index: usize) -> usize {
        NODE_TABLE_AT + 4 + 4 + index * NODE_RECORD_BYTES
    }

    /// Returns a whole [`Document`], because that is what the reader answers
    /// with.
    fn round_trip(doc: &Document, state: &ProjectState) -> (Document, ProjectState) {
        let mut bytes = Vec::new();
        write(&mut bytes, doc, state).expect("write failed");
        read(&mut bytes.as_slice()).expect("read failed")
    }

    /// The property that matters: what comes back is the same sculpt, not
    /// something that merely looks like it.
    #[test]
    fn a_sculpted_model_survives_a_round_trip_value_for_value() {
        let doc = sculpted_doc();
        let volume = doc.active_volume();
        let (loaded, _) = round_trip(&doc, &ProjectState::default());

        assert_eq!(loaded.voxel_size(), volume.voxel_size());

        let loaded = loaded.active_volume();
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
        let doc = sculpted_doc();
        let state = ProjectState::default();
        let mut first = Vec::new();
        let mut second = Vec::new();
        write(&mut first, &doc, &state).unwrap();
        write(&mut second, &doc, &state).unwrap();
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
        let (_, loaded) = round_trip(&sculpted_doc(), &state);
        assert_eq!(loaded, state);
    }

    /// A reloaded model has to be ready to mesh, or it loads into an empty
    /// screen and looks like a failure.
    #[test]
    fn everything_is_marked_dirty_so_the_load_is_visible() {
        let (mut loaded, _) = round_trip(&sculpted_doc(), &ProjectState::default());
        let mut dirty: Vec<(NodeId, BrickCoord)> = Vec::new();
        loaded.take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "a freshly loaded model had nothing to mesh");
    }

    /// The check this whole format exists to make. A file written at a
    /// different lattice is not a load error, it is a plausible looking sculpt
    /// made of misread numbers, so it must be refused rather than accepted.
    #[test]
    fn a_file_written_at_a_different_lattice_is_refused() {
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted_doc(), &ProjectState::default()).unwrap();

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
        write(&mut bytes, &sculpted_doc(), &ProjectState::default()).unwrap();

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
        write(&mut bytes, &sculpted_doc(), &ProjectState::default()).unwrap();

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
        write(&mut bytes, &sculpted_doc(), &ProjectState::default()).unwrap();

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
        let mut bytes = Vec::new();
        write(&mut bytes, &ball(0.5, 10.0), &ProjectState::default()).unwrap();

        // Poison the final brick's last value, which sits just before the
        // key trailer rather than at the end of the file.
        let at = last_distance_at(&bytes);
        bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(read(&mut bytes.as_slice()), Err(ProjectError::NonFiniteValue)));
    }

    #[test]
    fn an_unknown_brick_kind_is_refused() {
        let mut bytes = Vec::new();
        write(&mut bytes, &ball(0.5, 10.0), &ProjectState::default()).unwrap();

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
        write(&mut bytes, &Document::from_volume(volume), &ProjectState::default()).unwrap();

        // An empty volume is exactly the header plus the brick count, which is
        // how the fixed part is measured rather than counted by hand.
        let mut empty = Vec::new();
        write(&mut empty, &Document::new(1.0), &ProjectState::default()).unwrap();

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
        let (loaded, _) = round_trip(&Document::new(0.25), &ProjectState::default());
        assert_eq!(loaded.voxel_size(), 0.25);
        assert_eq!(loaded.active_volume().brick_coords().count(), 0);
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
        write(&mut valid, &sculpted_doc(), &ProjectState::default()).expect("write failed");

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
                let mut progress = Progress::default();
                let outcome = read_reporting(&mut bytes.as_slice(), &mut progress);
                // Counted rather than inferred from which error came back.
                // This used to match four variants said to be raisable "only
                // from inside the brick loop", and one of them was not:
                // `NonFiniteValue` is also how a corrupt `voxel_size` at byte
                // 20 is refused, forty-three bytes before the brick count. So
                // mutants stopped at the header were counted as having reached
                // the geometry, and the control was measuring less than it
                // claimed.
                if progress.bricks > 0 {
                    reached_the_bricks += 1;
                }
                let Ok((doc, _)) = outcome else {
                    continue;
                };
                loaded += 1;
                let volume = doc.active_volume();

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
        // is rather than how far the reader got.
        //
        // Measured, from the counter rather than from the error variants:
        // **1800 of 1800** mutants reach the brick loop, of which **22** load
        // whole. All 1800 is not a surprise once counted -- the header is 119
        // of 4,194,843 bytes, so a handful of flipped bits, a cut at a random
        // offset or a 256-byte overwrite lands in the geometry essentially
        // every time.
        //
        // **Both numbers are unchanged by the node table**, re-measured when it
        // landed: the table added 48 bytes to a four megabyte corpus, which is
        // one mutant in eighty-seven thousand.
        //
        // The previously recorded "1200 of 1800" was wrong in both directions
        // and must not be treated as a floor: it over-counted header refusals
        // (`NonFiniteValue` from byte 20) and under-counted every mutant that
        // died inside the loop with a variant the match did not list, chiefly
        // `Truncated`, which is the whole of strategy 1.
        eprintln!(
            "{reached_the_bricks} of {tried} corrupted projects reached the brick loop, \
             {loaded} loaded whole, from a {} byte corpus",
            valid.len()
        );
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
        write(&mut bytes, &sculpted_doc(), &ProjectState::default()).expect("write failed");

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
        let (doc, _) = read(&mut fine.as_slice()).expect("the largest legal brick was refused");
        let coord = doc
            .active_volume()
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
        let mut bytes = Vec::new();
        write(&mut bytes, &ball(0.5, 10.0), &ProjectState::default()).expect("write failed");

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
        let (_, loaded) = round_trip(&sculpted_doc(), &state);
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
        let (_, loaded) = round_trip(&sculpted_doc(), &state);
        let order: Vec<f32> = loaded.keys.iter().map(|key| key.at).collect();
        assert_eq!(order, vec![0.1, 0.5, 0.8]);
        // And each key kept its own view rather than merely its position.
        assert_eq!(loaded.keys[0].view.camera_distance, 2.0);
        assert_eq!(loaded.keys[2].view.camera_distance, 1.0);
    }

    /// The one that matters most about the version bumps: every `.brokkr` file
    /// already written is a version 1 or 2 file, and there are real ones on
    /// disk.
    ///
    /// An old file is this build's own output with the node table taken back
    /// out and, for version 1, the key trailer as well -- which is exactly what
    /// the old writers produced, byte for byte, and is what makes this a test
    /// of compatibility rather than of a fixture. See [`as_older_container`].
    #[test]
    fn a_file_from_before_the_timeline_still_opens() {
        let doc = sculpted_doc();
        let bricks = doc.active_volume().brick_coords().count();
        let state = ProjectState {
            view: View { camera_distance: 55.0, brush_radius: 7.0, ..View::default() },
            keys: Vec::new(),
        };
        let mut bytes = Vec::new();
        write(&mut bytes, &doc, &state).expect("write failed");
        let bytes = as_older_container(1, bytes);

        let (loaded, loaded_state) =
            read(&mut bytes.as_slice()).expect("a version 1 file was refused");
        assert_eq!(loaded_state.view, state.view, "the old file's settings came back wrong");
        assert!(loaded_state.keys.is_empty(), "a file with no trailer gained keys");
        assert_eq!(
            loaded.active_volume().brick_coords().count(),
            bricks,
            "the geometry did not survive"
        );
        assert_eq!(loaded.body_count(), 1, "a file with no node table is one implicit body");
        assert_eq!(loaded.nodes()[0].name, Document::FIRST_BODY_NAME);
    }

    #[test]
    fn a_layout_newer_than_this_build_is_still_refused() {
        // Widening the accepted range must not have widened it upward. A file
        // from a future build may put anything after the header, and reading
        // it as though it were this one is the plausible-looking-sculpt failure
        // the version numbers exist to prevent.
        let mut bytes = Vec::new();
        write(&mut bytes, &sculpted_doc(), &ProjectState::default()).expect("write failed");
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
        write(&mut bytes, &sculpted_doc(), &ProjectState::default()).expect("write failed");
        let at = bytes.len() - EMPTY_TRAILER_BYTES;
        // Bounded rather than written as `bytes[at..]`, which is the same four
        // bytes today and a slice-length *panic* the moment the trailer grows
        // -- and a panic here would read as this test having found a bug rather
        // than as the test needing to move with the trailer.
        bytes[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
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
            let (_, loaded) = round_trip(&sculpted_doc(), &state);
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
        let (_, loaded) = round_trip(&sculpted_doc(), &state);
        assert!(loaded.keys[0].view.camera_target.is_finite());
        assert!(loaded.keys[0].view.camera_distance.is_finite());
    }

    /// The reach the error variant cannot express, pinned as its own case.
    ///
    /// Both halves of this return `NonFiniteValue`. One is refused at byte 20
    /// of the header, before a single brick has been looked at; the other is
    /// refused in the last brick of the file. Anything that tries to tell them
    /// apart by matching on the variant gets the same answer for both, which
    /// is exactly the bug this increment removes from the fuzz control.
    #[test]
    fn the_reader_says_how_far_it_got_because_the_error_variant_cannot() {
        let doc = sculpted_doc();
        let bricks = doc.active_volume().brick_coords().count() as u64;
        let mut valid = Vec::new();
        write(&mut valid, &doc, &ProjectState::default()).expect("write failed");

        // `voxel_size` sits after the magic, the two versions and the lattice
        // pair -- byte 20, forty-three bytes before the node table at 63.
        let voxel_size_at = MAGIC.len() + 2 + 2 + 4 + 4;
        assert_eq!(voxel_size_at, 20);
        let mut header_broken = valid.clone();
        header_broken[voxel_size_at..voxel_size_at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut progress = Progress::default();
        assert!(matches!(
            read_reporting(&mut header_broken.as_slice(), &mut progress),
            Err(ProjectError::NonFiniteValue)
        ));
        assert_eq!(progress.bricks, 0, "a header refusal reported reaching the geometry");
        assert_eq!(progress.nodes, 0, "a header refusal reported reading a node table");

        let mut brick_broken = valid.clone();
        let at = last_distance_at(&brick_broken);
        brick_broken[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut progress = Progress::default();
        assert!(matches!(
            read_reporting(&mut brick_broken.as_slice(), &mut progress),
            Err(ProjectError::NonFiniteValue)
        ));
        assert_eq!(
            progress.bricks, bricks,
            "a refusal in the last brick did not report the whole geometry as reached"
        );
        assert_eq!(
            progress.nodes, 1,
            "the table was read whole before the geometry, so its rows are reached even when the \
             geometry is not"
        );

        // And a whole file reports every brick it was given.
        let mut progress = Progress::default();
        read_reporting(&mut valid.as_slice(), &mut progress).expect("the valid file was refused");
        assert_eq!(progress.bricks, bricks);
        // The other two counters, now that the format carries a node table for
        // them to count. One row, and nothing repaired: this build's writer
        // must not produce anything this build's reader has to fix.
        assert_eq!((progress.nodes, progress.repairs), (1, 0));
    }

    // ---------------------------------------------------------------------
    // The node table.
    // ---------------------------------------------------------------------

    /// One node record's forty bytes, built field by field so that a test can
    /// put anything at all in any of them. The writer above cannot produce most
    /// of what these tests need -- that is the point of them.
    struct Record {
        id: u32,
        flags: u16,
        kind: u8,
        depth: u8,
        name: &'static str,
    }

    impl Record {
        /// A visible body, which is every legal record this build can write.
        fn body(id: u32, name: &'static str) -> Self {
            Self { id, flags: FLAG_VISIBLE, kind: KIND_BODY, depth: 0, name }
        }

        fn bytes(&self) -> Vec<u8> {
            let mut out = Vec::with_capacity(NODE_RECORD_BYTES);
            out.extend_from_slice(&self.id.to_le_bytes());
            out.extend_from_slice(&self.flags.to_le_bytes());
            out.push(self.kind);
            out.push(self.depth);
            out.extend_from_slice(&name_bytes(self.name));
            assert_eq!(out.len(), NODE_RECORD_BYTES);
            out
        }
    }

    /// A version 3 file with a node table of the caller's choosing and one
    /// empty brick stream per body row.
    ///
    /// The header comes from this build's own writer, so a test built on this
    /// is aiming at the table and at nothing else.
    fn crafted(active: u32, records: &[Record]) -> Vec<u8> {
        let mut bytes = Vec::new();
        write(&mut bytes, &Document::new(0.5), &ProjectState::default()).expect("write failed");
        bytes.truncate(NODE_TABLE_AT);
        bytes.extend_from_slice(&(records.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&active.to_le_bytes());
        for record in records {
            bytes.extend_from_slice(&record.bytes());
        }
        for _ in records.iter().filter(|record| record.kind == KIND_BODY) {
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes
    }

    /// A volume holding exactly one uniform brick, for the tests that care
    /// about where a stream starts and ends rather than what is in it.
    fn one_brick(at: BrickCoord, value: f32) -> Volume {
        let mut volume = Volume::new(0.5);
        volume.insert_brick(at, Brick::Uniform(value));
        volume
    }

    /// Three bodies, each with its own bricks, none of them empty.
    fn three_bodies() -> Document {
        let mut doc = Document::from_volume(one_brick(BrickCoord::new(0, 0, 0), INSIDE));
        doc.add_body("Left arm", one_brick(BrickCoord::new(-4, 1, 2), -0.5));
        let third = doc.add_body("Right arm", one_brick(BrickCoord::new(7, -1, 3), 1.25));
        doc.set_active(third);
        doc
    }

    /// The layout claim, measured against this build's own writer rather than
    /// counted by hand.
    ///
    /// Bytes 0..63 must stay byte for byte what version 2 wrote, because every
    /// header offset the tests above hardcode aims into that range -- and
    /// because it is what lets an older file be manufactured by stripping.
    #[test]
    fn the_version_3_header_is_the_version_2_header_plus_a_node_table() {
        let mut empty = Vec::new();
        write(&mut empty, &Document::new(0.5), &ProjectState::default()).expect("write failed");

        assert_eq!(
            u16::from_le_bytes([empty[8], empty[9]]),
            3,
            "this build no longer writes container version 3"
        );
        assert_eq!(
            NODE_TABLE_AT, 63,
            "the node table moved, so every offset keyed off it is now wrong"
        );
        // Four of `node_count`, four of `active_index`, one record, then the
        // u64 brick count.
        assert_eq!(first_brick_at(), NODE_TABLE_AT + 4 + 4 + NODE_RECORD_BYTES + 8);
        // The whole of an empty single-body file, to the byte.
        assert_eq!(empty.len(), 123);

        assert_eq!(u32::from_le_bytes(empty[63..67].try_into().unwrap()), 1, "one row");
        assert_eq!(u32::from_le_bytes(empty[67..71].try_into().unwrap()), 0, "the first row");
        assert_eq!(u32::from_le_bytes(empty[71..75].try_into().unwrap()), 1, "id 1");
        assert_eq!(u16::from_le_bytes([empty[75], empty[76]]), FLAG_VISIBLE);
        assert_eq!((empty[77], empty[78]), (KIND_BODY, 0), "kind and depth are reserved at zero");
        assert_eq!(&empty[79..79 + Document::FIRST_BODY_NAME.len()], b"Body 1");
    }

    /// N bodies come back as N bodies, in order, each with its own field.
    #[test]
    fn a_three_body_document_comes_back_with_its_names_order_and_geometry() {
        let doc = three_bodies();
        let (loaded, _) = round_trip(&doc, &ProjectState::default());

        assert_eq!(loaded.body_count(), 3);
        let names: Vec<&str> = loaded.nodes().iter().map(|node| node.name.as_str()).collect();
        assert_eq!(names, vec!["Body 1", "Left arm", "Right arm"]);
        let ids: Vec<NodeId> = loaded.nodes().iter().map(|node| node.id).collect();
        assert_eq!(ids, doc.nodes().iter().map(|node| node.id).collect::<Vec<_>>());

        // Each body's own bricks, which is what the consecutive streams have to
        // keep apart. Meshing one body's brick against another body's field is
        // the failure this catches.
        for ((_, before), (_, after)) in doc.bodies().zip(loaded.bodies()) {
            assert_same_bricks(before, after);
        }

        assert_eq!(
            loaded.active(),
            doc.active(),
            "the body the user had selected did not come back selected"
        );
    }

    /// **Write, read, write, and compare the two files byte for byte.**
    ///
    /// The test that did not exist and that the repair rules make necessary.
    /// `writing_the_same_volume_twice_gives_identical_bytes` writes the same
    /// in-memory value twice and never runs the reader; the round trip above
    /// compares semantics rather than bytes. So "this build's reader repairs
    /// nothing this build's writer produced" was asserted by nothing at all --
    /// and a reader that quietly renamed a body, dropped a flag or reordered
    /// the list would pass both of them.
    #[test]
    fn a_three_body_document_survives_write_read_write_byte_for_byte() {
        let doc = three_bodies();
        let state = ProjectState {
            view: View { camera_distance: 12.0, ..View::default() },
            keys: vec![a_key(0.4, 9.0)],
        };

        let mut first = Vec::new();
        write(&mut first, &doc, &state).expect("write failed");

        // And twice from the same document, because a hash order in the body
        // list would break this nondeterministically -- passing locally and
        // failing on CI. `Document::nodes` is a `Vec` for exactly this reason.
        let mut again = Vec::new();
        write(&mut again, &doc, &state).expect("write failed");
        assert_eq!(first, again, "two writes of one document differ");

        let mut progress = Progress::default();
        let (reread, reread_state) = read_reporting(&mut first.as_slice(), &mut progress)
            .expect("this build refused its own file");
        assert_eq!(progress.nodes, 3);
        assert_eq!(progress.repairs, 0, "the reader repaired something this build wrote");

        let mut second = Vec::new();
        write(&mut second, &reread, &reread_state).expect("write failed");
        assert_eq!(first.len(), second.len(), "the rewrite is a different length");
        let differs = first.iter().zip(&second).position(|(one, other)| one != other);
        assert_eq!(differs, None, "the rewrite differs at byte {differs:?}");
    }

    /// A body with no bricks at all still frames its stream, or every body
    /// after it is misparsed.
    #[test]
    fn an_empty_body_writes_an_empty_stream_and_the_bodies_after_it_survive() {
        let mut doc = Document::from_volume(one_brick(BrickCoord::new(0, 0, 0), INSIDE));
        doc.add_body("Nothing yet", Volume::new(0.5));
        doc.add_body("Something", one_brick(BrickCoord::new(3, 3, 3), -1.0));

        let (loaded, _) = round_trip(&doc, &ProjectState::default());
        assert_eq!(loaded.body_count(), 3);
        assert_eq!(loaded.nodes()[1].volume().expect("a body").brick_count(), 0);
        assert_eq!(loaded.nodes()[2].volume().expect("a body").brick_count(), 1);
        assert_same_bricks(
            doc.nodes()[2].volume().expect("a body"),
            loaded.nodes()[2].volume().expect("a body"),
        );
    }

    /// The id policy, which nothing in the file carries and everything
    /// downstream depends on.
    ///
    /// Ids never reuse, so a saved file routinely holds a sparse set. Deriving
    /// the next one from the row count would mint an id that is already in the
    /// document: two rows under one id, one body's mesh slots overwriting the
    /// other's, undo entries routed to the wrong body -- and the next save
    /// writing a file this build refuses for duplicate ids.
    #[test]
    fn a_body_added_to_a_sparse_id_file_gets_the_highest_id_plus_one() {
        let records =
            [Record::body(1, "Body 1"), Record::body(5, "Body 5"), Record::body(9, "Body 9")];
        let bytes = crafted(0, &records);
        let (mut doc, _) = read(&mut bytes.as_slice()).expect("a sparse id file was refused");
        assert_eq!(
            doc.nodes().iter().map(|node| node.id.0).collect::<Vec<_>>(),
            vec![1, 5, 9],
            "the ids in the file were not the ids in the document"
        );

        let fresh = doc.add_body("Body 10", Volume::new(0.5));
        assert_eq!(fresh, NodeId(10), "the next id must clear the highest one in the file");
    }

    /// Every refusal that decides how the rest of the file is parsed, in one
    /// place. Each of these aliases a key, misaligns a stream or leaves the
    /// document without an active body.
    #[test]
    fn a_node_table_that_could_not_be_this_builds_own_is_refused() {
        let good = Record::body(1, "Body 1");
        assert!(read(&mut crafted(0, &[good]).as_slice()).is_ok(), "the control file was refused");

        // Zero is "no node"; `u32::MAX` would overflow `max(id) + 1`.
        for offending in [0, u32::MAX] {
            let bytes = crafted(0, &[Record::body(offending, "Body 1")]);
            match read(&mut bytes.as_slice()) {
                Err(ProjectError::ReservedNodeId { found }) => assert_eq!(found, offending),
                Ok(_) => panic!("the reserved id {offending} was accepted"),
                Err(other) => panic!("expected an id refusal for {offending}, got {other}"),
            }
        }

        let bytes = crafted(0, &[Record::body(3, "One"), Record::body(3, "Two")]);
        assert!(
            matches!(read(&mut bytes.as_slice()), Err(ProjectError::DuplicateNodeId { found: 3 })),
            "two rows shared an id, which aliases the mesh pool key and the undo routing"
        );

        // A reserved flag bit. Bit 5 is one of the fourteen held at zero, and
        // holding them there is what stops a future build from spending one and
        // making its files unreadable by every shipped version 3 build.
        let bytes =
            crafted(0, &[Record { flags: FLAG_VISIBLE | 1 << 5, ..Record::body(1, "Body 1") }]);
        assert!(matches!(read(&mut bytes.as_slice()), Err(ProjectError::ReservedFlags { .. })));

        // `kind` decides whether a brick stream follows, so it is refused
        // rather than repaired: repairing it would misalign every stream after
        // it, exactly as an unknown brick tag would.
        let bytes = crafted(0, &[Record { kind: 1, ..Record::body(1, "Folder") }]);
        assert!(matches!(read(&mut bytes.as_slice()), Err(ProjectError::UnknownNodeKind(1))));

        // `depth` is reserved at zero until folders exist. The increment that
        // makes them representable replaces this refusal with a clamp.
        let bytes = crafted(0, &[Record { depth: 1, ..Record::body(1, "Body 1") }]);
        assert!(matches!(
            read(&mut bytes.as_slice()),
            Err(ProjectError::ReservedDepth { found: 1 })
        ));
    }

    /// A count read out of a file decides an allocation, so it is bounded
    /// before the allocation happens.
    #[test]
    fn a_row_count_of_zero_or_past_the_cap_is_refused() {
        for offending in [0u32, MAX_NODES as u32 + 1, u32::MAX] {
            let mut bytes = crafted(0, &[Record::body(1, "Body 1")]);
            bytes[NODE_TABLE_AT..NODE_TABLE_AT + 4].copy_from_slice(&offending.to_le_bytes());
            match read(&mut bytes.as_slice()) {
                Err(ProjectError::NodeCount { found, limit }) => {
                    assert_eq!(found, offending);
                    assert_eq!(limit, MAX_NODES as u32);
                }
                Ok(_) => panic!("a row count of {offending} was accepted"),
                Err(other) => panic!("expected a row count refusal for {offending}, got {other}"),
            }
        }
    }

    /// One more body than the cap allows, refused while reading the table and
    /// before one brick stream is touched.
    #[test]
    fn a_sixty_fifth_body_is_refused() {
        let records: Vec<Record> =
            (1..=MAX_BODIES as u32 + 1).map(|id| Record::body(id, "Body")).collect();
        let bytes = crafted(0, &records);
        let mut progress = Progress::default();
        match read_reporting(&mut bytes.as_slice(), &mut progress) {
            Err(ProjectError::TooManyBodies { found, limit }) => {
                assert_eq!((found, limit), (MAX_BODIES + 1, MAX_BODIES));
            }
            Ok(_) => panic!("a 65 body file was accepted"),
            Err(other) => panic!("expected a body count refusal, got {other}"),
        }
        assert_eq!(progress.bricks, 0, "the table was refused after a brick had been read");

        // And the cap itself is not so tight that it refuses what it allows.
        let records: Vec<Record> =
            (1..=MAX_BODIES as u32).map(|id| Record::body(id, "Body")).collect();
        let (doc, _) = read(&mut crafted(0, &records).as_slice()).expect("64 bodies were refused");
        assert_eq!(doc.body_count(), MAX_BODIES);
    }

    /// The hole every source design left open, and where it is actually
    /// plugged today.
    ///
    /// A table whose rows are all folders parses perfectly, declares no brick
    /// stream and consumes exactly the right number of bytes -- and leaves a
    /// document with no body for `active` to name, so the first thing to touch
    /// the active body panics on a file that loaded without complaint. The
    /// mutation fuzz cannot catch it either: its band assertion loops over zero
    /// bodies and passes. At container version 3 that file cannot be built,
    /// because `kind` must be zero and the row is refused one check earlier.
    /// [`ProjectError::NoBodies`] is the same hole guarded one layer later; it
    /// becomes reachable in the increment that makes a folder row legal, and
    /// this test is what records that the two guards are for one thing.
    #[test]
    fn a_table_whose_only_row_is_a_folder_is_refused_at_the_kind() {
        let bytes = crafted(0, &[Record { kind: 1, ..Record::body(1, "Folder") }]);
        // 63 of header, 8 of counts, 40 of record, 4 of empty trailer: no brick
        // stream at all, and every length in the file agrees with every other.
        assert_eq!(bytes.len(), NODE_TABLE_AT + 8 + NODE_RECORD_BYTES + 4);
        assert!(matches!(read(&mut bytes.as_slice()), Err(ProjectError::UnknownNodeKind(1))));
    }

    /// A name decides only what a row is called, so it is repaired rather than
    /// refused -- exactly as a NaN camera is.
    #[test]
    fn a_name_that_is_empty_or_not_utf8_is_repaired_to_a_default() {
        let mut bytes = crafted(0, &[Record::body(1, "One"), Record::body(2, "Two")]);

        // The second row's name field, filled with a byte no UTF-8 sequence
        // may start with.
        let at = record_at(1) + 8;
        bytes[at..at + NAME_BYTES].fill(0xff);
        // And the first row's, emptied.
        let at = record_at(0) + 8;
        bytes[at..at + NAME_BYTES].fill(0);

        let mut progress = Progress::default();
        let (doc, _) = read_reporting(&mut bytes.as_slice(), &mut progress)
            .expect("a rubbish name lost the sculpt");
        assert_eq!(doc.nodes()[0].name, "Body 1", "an empty name was not repaired");
        assert_eq!(doc.nodes()[1].name, "Body 2", "a name that is not UTF-8 was not repaired");
        assert_eq!(progress.repairs, 2, "the repairs were not counted");
    }

    /// A full thirty-two byte name has no terminator, and the rename field in
    /// the interface enforces exactly that length -- so the application
    /// actively encourages producing the one name a "must be NUL terminated"
    /// rule would destroy.
    #[test]
    fn a_full_thirty_two_byte_name_round_trips_unchanged() {
        let name = "0123456789abcdef0123456789abcdef";
        assert_eq!(name.len(), NAME_BYTES);
        let doc = named(name);

        let mut bytes = Vec::new();
        write(&mut bytes, &doc, &ProjectState::default()).expect("write failed");
        let mut progress = Progress::default();
        let (loaded, _) =
            read_reporting(&mut bytes.as_slice(), &mut progress).expect("read failed");
        assert_eq!(loaded.nodes()[0].name, name);
        assert_eq!(progress.repairs, 0, "a full length name was repaired away");
    }

    /// A name too long for the field is cut on a CHAR BOUNDARY.
    ///
    /// Slicing at byte 32 through the middle of a multi-byte sequence writes
    /// bytes that are not UTF-8; the reader then correctly repairs the name to
    /// a default, and the user silently loses a name this build wrote itself.
    #[test]
    fn a_name_too_long_for_the_field_is_cut_on_a_char_boundary() {
        // Eleven three-byte characters is thirty-three bytes, so the eleventh
        // straddles byte thirty-two: a naive slice would leave one byte of it
        // behind and make the whole field invalid UTF-8.
        let name = "。。。。。。。。。。。";
        assert_eq!(name.len(), 33);
        let doc = named(name);

        let mut bytes = Vec::new();
        write(&mut bytes, &doc, &ProjectState::default()).expect("write failed");
        let mut progress = Progress::default();
        let (loaded, _) =
            read_reporting(&mut bytes.as_slice(), &mut progress).expect("read failed");

        assert_eq!(
            loaded.nodes()[0].name,
            "。。。。。。。。。。",
            "the name was cut mid character"
        );
        assert_eq!(progress.repairs, 0, "a truncated name came back as rubbish and was repaired");
    }

    /// A one-body document whose body carries the given name.
    fn named(name: &str) -> Document {
        let mut doc = Document::new(0.5);
        let meta = doc.meta(doc.active()).expect("the active row is in the document");
        doc.set_meta(&NodeMeta { name: name.to_string(), ..meta });
        doc
    }

    /// An `active_index` that names nothing is moved to the first BODY row,
    /// which is what keeps `Option<NodeId>` out of every signature downstream.
    #[test]
    fn an_active_index_past_the_end_is_moved_to_a_body_rather_than_refused() {
        for offending in [2u32, 99, u32::MAX] {
            let bytes = crafted(offending, &[Record::body(4, "One"), Record::body(6, "Two")]);
            let mut progress = Progress::default();
            let (doc, _) = read_reporting(&mut bytes.as_slice(), &mut progress)
                .expect("an active index off the end lost the sculpt");
            assert_eq!(doc.active(), NodeId(4), "the selection did not move to the first body");
            assert_eq!(progress.repairs, 1, "the repair was not counted");
        }

        // And an index that names a real body is left exactly where it is.
        let bytes = crafted(1, &[Record::body(4, "One"), Record::body(6, "Two")]);
        let mut progress = Progress::default();
        let (doc, _) = read_reporting(&mut bytes.as_slice(), &mut progress).expect("read failed");
        assert_eq!(doc.active(), NodeId(6));
        assert_eq!(progress.repairs, 0);
    }

    /// The brick cap is a DOCUMENT total. Checked per body it would let a file
    /// claiming sixty-four bodies ask for sixty-four times the cap.
    #[test]
    fn the_brick_cap_is_a_document_total_and_not_a_per_body_one() {
        let mut doc = Document::from_volume(one_brick(BrickCoord::new(0, 0, 0), INSIDE));
        doc.add_body("Body 2", one_brick(BrickCoord::new(1, 1, 1), OUTSIDE));
        let mut bytes = Vec::new();
        write(&mut bytes, &doc, &ProjectState::default()).expect("write failed");

        // The second body's count sits after the first body's stream: its own
        // u64, then twelve of coordinate, one tag byte and one value.
        let second_count_at = record_at(2) + 8 + 12 + 1 + 4;
        assert_eq!(
            u64::from_le_bytes(bytes[second_count_at..second_count_at + 8].try_into().unwrap()),
            1,
            "the second body's brick count is not where this test thinks it is"
        );
        // A count that is legal on its own -- it is exactly the cap -- and puts
        // the document one brick past it.
        bytes[second_count_at..second_count_at + 8].copy_from_slice(&MAX_BRICKS.to_le_bytes());

        let mut progress = Progress::default();
        match read_reporting(&mut bytes.as_slice(), &mut progress) {
            Err(ProjectError::TooLarge { bricks, limit }) => {
                assert_eq!(bricks, MAX_BRICKS + 1);
                assert_eq!(limit, MAX_BRICKS);
            }
            Ok(_) => panic!("a document past the cap was accepted"),
            Err(other) => panic!("expected a size refusal, got {other}"),
        }
        assert_eq!(
            progress.bricks, 1,
            "the refusal came after the second body's bricks had been read, not before"
        );
    }

    /// A hostile file's claim is bounded before it decides an allocation.
    ///
    /// Sixty-four bodies of sixty-five thousand bricks is half a terabyte of
    /// claim in a hundred and seventy bytes of header. Note what the reader can
    /// and cannot do about it: the counts are interleaved with the streams they
    /// describe and it never seeks, so it cannot sum them up front. What it can
    /// do -- and does -- is refuse the moment the running total passes the cap,
    /// before the body that pushed it over is allocated at all.
    #[test]
    fn a_hostile_brick_claim_is_refused_before_any_allocation() {
        let records: Vec<Record> =
            (1..=MAX_BODIES as u32).map(|id| Record::body(id, "Body")).collect();
        let mut bytes = crafted(0, &records);
        let claim = MAX_BODIES as u64 * MAX_BRICKS;
        let at = record_at(records.len());
        bytes[at..at + 8].copy_from_slice(&claim.to_le_bytes());

        let mut progress = Progress::default();
        match read_reporting(&mut bytes.as_slice(), &mut progress) {
            Err(ProjectError::TooLarge { bricks, limit }) => {
                assert_eq!(bricks, claim);
                assert_eq!(limit, MAX_BRICKS);
            }
            Ok(_) => panic!("a claim of {claim} bricks was accepted"),
            Err(other) => panic!("expected a size refusal, got {other}"),
        }
        assert_eq!(progress.bricks, 0, "a brick was read before the claim was refused");
        assert_eq!(progress.nodes, MAX_BODIES as u32, "the table itself parsed");
    }

    /// **The cap is enforced on the WRITE side too, and it was not before.**
    ///
    /// Add bodies over an afternoon, cross the cap, press ctrl+S: the write
    /// succeeded, the status said "saved", the asterisk cleared and
    /// `clear_autosave` deleted the crash net -- and reopening the file refused
    /// it. Never let this build write a file this build will not read.
    #[test]
    fn a_document_past_the_brick_cap_is_refused_by_the_writer_as_well() {
        // Split across two bodies, so this also pins the write-side check as a
        // document total: neither body is anywhere near the cap alone.
        let each = MAX_BRICKS / 2 + 1;
        let mut first = Volume::new(0.5);
        let mut second = Volume::new(0.5);
        for index in 0..each as i32 {
            first.insert_brick(BrickCoord::new(index, 0, 0), Brick::Uniform(OUTSIDE));
            second.insert_brick(BrickCoord::new(index, 1, 0), Brick::Uniform(OUTSIDE));
        }
        let mut doc = Document::from_volume(first);
        doc.add_body("Body 2", second);

        let mut bytes = Vec::new();
        match write(&mut bytes, &doc, &ProjectState::default()) {
            Err(ProjectError::TooLarge { bricks, limit }) => {
                assert_eq!(bricks, each * 2);
                assert_eq!(limit, MAX_BRICKS);
            }
            Ok(()) => panic!("an oversized document was written, and could not be read back"),
            Err(other) => panic!("expected a size refusal, got {other}"),
        }
        assert!(bytes.is_empty(), "a refused write left a partial file behind");
    }

    /// The field version is a RANGE, and the writer stamps the lowest version
    /// the document needs rather than the newest this build knows.
    ///
    /// The range protects a new build reading an old file. It does nothing for
    /// an old build reading a new one -- that is what writing the lowest
    /// version buys, and both halves are needed. Without the second, the day a
    /// mask bumps the field version every save stamps the new number whether or
    /// not the document uses any of it, and every one of those files is refused
    /// by every build that predates the mask.
    #[test]
    fn the_field_version_is_a_range_and_the_writer_stamps_the_lowest_needed() {
        const _: () = assert!(OLDEST_FIELD_VERSION <= FIELD_VERSION);

        let mut bytes = Vec::new();
        write(&mut bytes, &three_bodies(), &ProjectState::default()).expect("write failed");
        let stamped = u16::from_le_bytes([bytes[10], bytes[11]]);
        assert_eq!(
            stamped, OLDEST_FIELD_VERSION,
            "a document that needs nothing new was stamped with a newer encoding, which would \
             refuse it on every build that predates that encoding"
        );

        // Still only downward. A file written by a newer encoding may put
        // anything in a brick, and reading it as though it were this one is the
        // plausible-looking-sculpt failure the versions exist to prevent.
        let mut newer = bytes.clone();
        newer[10..12].copy_from_slice(&(FIELD_VERSION + 1).to_le_bytes());
        assert!(matches!(
            read(&mut newer.as_slice()),
            Err(ProjectError::FieldVersion { found, .. }) if found == FIELD_VERSION + 1
        ));

        // And an encoding older than anything that ever existed is refused too.
        let mut older = bytes.clone();
        older[10..12].copy_from_slice(&(OLDEST_FIELD_VERSION - 1).to_le_bytes());
        assert!(matches!(read(&mut older.as_slice()), Err(ProjectError::FieldVersion { .. })));
    }

    /// The outline is what a save reads back to check that what landed on disk
    /// is the document that was meant to go into it.
    #[test]
    fn the_outline_reports_the_rows_without_reading_the_geometry() {
        let doc = three_bodies();
        let mut bytes = Vec::new();
        write(&mut bytes, &doc, &ProjectState::default()).expect("write failed");

        let outline = read_outline(&mut bytes.as_slice()).expect("the outline was refused");
        assert_eq!(outline.nodes, 3);
        assert_eq!(outline.bodies, 3);
        assert_eq!(outline.voxel_size, doc.voxel_size());

        // The geometry is deliberately not read, so a file that stops right
        // after its table still has a readable outline.
        let table_ends = record_at(3);
        let outline = read_outline(&mut bytes[..table_ends].as_ref())
            .expect("the outline needed bytes past the table");
        assert_eq!(outline.bodies, 3);

        // An older file has no table, and the one implicit body it holds is
        // what the outline reports.
        let mut single = Vec::new();
        write(&mut single, &sculpted_doc(), &ProjectState::default()).expect("write failed");
        let older = as_older_container(1, single);
        let outline = read_outline(&mut older.as_slice()).expect("a version 1 outline was refused");
        assert_eq!((outline.nodes, outline.bodies), (1, 1));
    }

    // ---------------------------------------------------------------------
    // The committed version 1 and version 2 fixtures.
    //
    // These exist because the compatibility test above builds its old file
    // out of this build's own writer, which proves the reader agrees with the
    // writer it shipped beside -- and stops proving anything the moment the
    // writer changes, which is the moment it is needed. A committed file
    // cannot drift with the code.
    //
    // They are MANUFACTURED rather than cut down from the real projects on
    // disk. A real file cannot be truncated and stay valid: the u64 count at
    // byte 63 says how many bricks follow, so shortening one means rewriting
    // that count, which needs a tool that already reads the format. And they
    // are dense -- even a hundred bricks is about 13 MB, which does not
    // belong in git. A handful of uniform bricks is a few hundred bytes and
    // is byte for byte what the old writer produced for the same volume.
    // ---------------------------------------------------------------------

    /// Bytes the header occupied while versions 1 and 2 were current.
    ///
    /// Not counted from the writer but confirmed against the real files with
    /// `od`: eight of magic, two versions, the lattice pair, `voxel_size` and
    /// thirty-nine of view, and then the u64 brick count at 63.
    const LEGACY_HEADER_BYTES: usize = 63;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    /// The volume both fixtures carry.
    ///
    /// Uniform bricks only, placed by `insert_brick` -- which is `pub(crate)`,
    /// and is the reason a fixture of exactly chosen bricks can only be built
    /// from inside this crate. The values are spread across the band and over
    /// both signs so that a reader which misreads one produces a visibly
    /// different answer rather than a plausible one, and the first coordinate
    /// is `(-2, -3, -4)` because that is what sits at byte 71 of the three
    /// real version 1 files.
    fn fixture_volume() -> Volume {
        let mut volume = Volume::new(0.5);
        // Sorted by (z, y, x) on the way out, so the negative z comes first.
        for (coord, value) in [
            (BrickCoord::new(-2, -3, -4), INSIDE),
            (BrickCoord::new(0, 0, -1), -0.75),
            (BrickCoord::new(1, 0, 0), 0.0),
            (BrickCoord::new(0, 5, 1), 1.5),
            (BrickCoord::new(7, -1, 2), OUTSIDE),
        ] {
            volume.insert_brick(coord, Brick::Uniform(value));
        }
        volume
    }

    /// The session settings both fixtures carry. Deliberately not the
    /// defaults: a reader that skipped the view entirely would still pass
    /// against default values.
    fn fixture_state(keys: Vec<Keyframe>) -> ProjectState {
        ProjectState {
            view: View {
                camera_target: Vec3::new(-1.5, 2.0, 0.25),
                camera_distance: 88.0,
                camera_yaw: 1.25,
                camera_pitch: -0.5,
                camera_roll: 0.125,
                brush_radius: 5.5,
                brush_strength: 0.4,
                mirror: [true, false, true],
            },
            keys,
        }
    }

    /// Build a fixture's bytes: what a build of the given container version
    /// would have written for [`fixture_volume`].
    fn manufactured_fixture(container: u16, keys: Vec<Keyframe>) -> Vec<u8> {
        assert!(
            container >= 2 || keys.is_empty(),
            "a version 1 file has no trailer to put keys in"
        );

        let mut bytes = Vec::new();
        write(&mut bytes, &Document::from_volume(fixture_volume()), &fixture_state(keys))
            .expect("write failed");
        as_older_container(container, bytes)
    }

    /// What a build of an older container version would have written for the
    /// same one-body document, byte for byte.
    ///
    /// The whole difference between the layouts is a section this build knows
    /// how to take back out, which is what lets a committed fixture stay
    /// meaningful: the brick encoding has not changed since version 1, so
    /// everything after the header is already identical.
    ///
    /// **The strip is MEASURED rather than written as a literal range**, and
    /// that measurement was in place before the node table existed, when the
    /// drain was empty. It is bytes 63..111 today -- four of `node_count`, four
    /// of `active_index` and one forty-byte record -- and it is emphatically
    /// *not* 63..119, which would also swallow the u64 brick count and leave a
    /// brick coordinate where the count belongs. Measuring is what makes the
    /// right answer automatic when the header grows again.
    ///
    /// Single body only, like everything else keyed off [`first_brick_at`]: a
    /// two-body document has a second brick stream that no older layout can
    /// express at all.
    fn as_older_container(container: u16, mut bytes: Vec<u8>) -> Vec<u8> {
        let count_at = first_brick_at() - 8;
        assert!(
            count_at >= LEGACY_HEADER_BYTES,
            "the header is shorter than it was at version 2, so an older file cannot be \
             manufactured by stripping and has to be rebuilt by hand"
        );
        bytes.drain(LEGACY_HEADER_BYTES..count_at);

        bytes[MAGIC.len()..MAGIC.len() + 2].copy_from_slice(&container.to_le_bytes());
        if container < 2 {
            // A version 1 file ends with its last brick. There is no trailer,
            // not even an empty one.
            bytes.truncate(bytes.len() - EMPTY_TRAILER_BYTES);
        }
        bytes
    }

    /// Two keys, so the version 2 fixture exercises the trailer rather than
    /// merely carrying an empty one. Written out of order on purpose: the
    /// reader sorts, and the committed bytes record that it has to.
    fn fixture_keys() -> Vec<Keyframe> {
        vec![a_key(0.75, 12.0), a_key(0.2, 34.0)]
    }

    /// Writes the two fixtures. A one-off act rather than a check, and it
    /// takes *two* deliberate steps to make it happen:
    ///
    /// ```text
    /// BROKKR_REGENERATE_FIXTURES=1 cargo test -p brokkr-core --lib -- --ignored regenerate_the_committed_fixtures
    /// ```
    ///
    /// `#[ignore]` alone was not enough, and the near-miss is worth recording
    /// because it is exactly the failure this increment exists to prevent.
    /// `--ignored` is not a per-test opt-in: it is a *sweep*, and the other
    /// ignored test in this file --- `the_real_projects_on_this_machine_still_open`
    /// --- is one the maintainer is told to run deliberately. So the natural
    /// invocation `cargo test -p brokkr-core -- --ignored` used to rewrite the
    /// committed fixtures in the same breath, from the current build. That
    /// turns [`the_committed_fixtures_are_byte_for_byte_what_the_old_writers_produced`]
    /// into a tautology: it would then be comparing the file against
    /// `manufactured_fixture`, the very function that had just written it, and
    /// would report "ok" having checked nothing. Demonstrated, not imagined ---
    /// a single flipped byte in `container-v1.brokkr` failed three tests
    /// loudly, and one `--ignored` run put it back and turned them all green.
    ///
    /// So the write is gated on an environment variable that nothing in CI and
    /// no sweep sets. Without it the test passes without touching a byte,
    /// which keeps the sweep quiet without letting it destroy anything.
    #[test]
    #[ignore = "regenerates the committed fixtures; running it is a decision, not a check"]
    fn regenerate_the_committed_fixtures() {
        if std::env::var_os("BROKKR_REGENERATE_FIXTURES").is_none() {
            eprintln!(
                "refusing to rewrite the committed fixtures as part of an --ignored sweep: set \
                 BROKKR_REGENERATE_FIXTURES=1 if you really mean to replace them with what this \
                 build writes. They stand in for files on users' disks, which do not change when \
                 this code does."
            );
            return;
        }

        let directory = fixture_path("");
        std::fs::create_dir_all(&directory).expect("could not make the fixtures directory");
        for (name, bytes) in [
            ("container-v1.brokkr", manufactured_fixture(1, Vec::new())),
            ("container-v2.brokkr", manufactured_fixture(2, fixture_keys())),
        ] {
            let path = fixture_path(name);
            std::fs::write(&path, &bytes).expect("could not write the fixture");
            eprintln!("wrote {} bytes to {}", bytes.len(), path.display());
        }
    }

    fn committed_fixture(name: &str) -> Vec<u8> {
        let path = fixture_path(name);
        std::fs::read(&path).unwrap_or_else(|error| {
            panic!("could not read the committed fixture at {}: {error}", path.display())
        })
    }

    /// The fixtures are still byte for byte what this build would produce for
    /// the old layouts.
    ///
    /// **If this fails, do not regenerate them.** They stand in for the files
    /// on users' disks, which do not change when this code does. A failure
    /// here means the writer moved something inside a version 1 or 2 file, and
    /// the question to answer is what the reader now does with the real ones.
    /// The only legitimate reason to re-run `regenerate_the_committed_fixtures`
    /// is that the header grew a section this build knows how to strip, which
    /// `manufactured_fixture` handles without changing a byte of the output.
    /// That regenerator will not run without `BROKKR_REGENERATE_FIXTURES` set,
    /// precisely so that reaching for it has to be a decision.
    #[test]
    fn the_committed_fixtures_are_byte_for_byte_what_the_old_writers_produced() {
        for (name, manufactured) in [
            ("container-v1.brokkr", manufactured_fixture(1, Vec::new())),
            ("container-v2.brokkr", manufactured_fixture(2, fixture_keys())),
        ] {
            let committed = committed_fixture(name);
            assert_eq!(
                committed.len(),
                manufactured.len(),
                "{name} is {} bytes and this build would write {}",
                committed.len(),
                manufactured.len()
            );
            let differs =
                committed.iter().zip(&manufactured).position(|(theirs, ours)| theirs != ours);
            if let Some(at) = differs {
                panic!(
                    "{name} differs from this build's output at byte {at}: the committed file \
                     holds {:#04x} and this build writes {:#04x}",
                    committed[at], manufactured[at]
                );
            }
        }
    }

    /// A version 1 file -- the layout of every `.brokkr` written before the
    /// timeline existed, and of three of the real projects on disk -- still
    /// opens, from a committed file rather than from this build's own writer.
    #[test]
    fn the_committed_version_1_fixture_opens_with_its_geometry_and_settings() {
        let bytes = committed_fixture("container-v1.brokkr");
        assert_eq!(
            u16::from_le_bytes([bytes[8], bytes[9]]),
            1,
            "the version 1 fixture is not stamped version 1"
        );

        let mut progress = Progress::default();
        let (doc, state) = read_reporting(&mut bytes.as_slice(), &mut progress)
            .expect("the committed version 1 fixture was refused");

        assert_eq!(state.view, fixture_state(Vec::new()).view);
        assert!(state.keys.is_empty(), "a file with no trailer gained keys");
        assert_eq!(progress.bricks, 5);
        assert_eq!(doc.body_count(), 1, "a file with no node table is one implicit body");
        assert_eq!(doc.nodes()[0].name, Document::FIRST_BODY_NAME);
        assert_eq!(progress.nodes, 0, "there was no node table to read nodes from");
        assert_same_bricks(&fixture_volume(), doc.active_volume());
    }

    /// And a version 2 file, whose trailer has to come back sorted.
    #[test]
    fn the_committed_version_2_fixture_opens_with_its_timeline_keys() {
        let bytes = committed_fixture("container-v2.brokkr");
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 2);

        let mut progress = Progress::default();
        let (doc, state) = read_reporting(&mut bytes.as_slice(), &mut progress)
            .expect("the committed version 2 fixture was refused");

        assert_eq!(state.view, fixture_state(Vec::new()).view);
        let mut expected = fixture_keys();
        expected.sort_by(|a, b| a.at.partial_cmp(&b.at).expect("no NaN in the fixture"));
        assert_eq!(state.keys, expected, "the trailer did not come back in order");
        assert_eq!(progress.bricks, 5);
        assert_eq!(doc.body_count(), 1, "a file with no node table is one implicit body");
        assert_same_bricks(&fixture_volume(), doc.active_volume());
    }

    /// The two fixtures differ only in the version stamp and the trailer, so
    /// the same geometry has to come out of both.
    #[test]
    fn the_two_fixtures_are_the_same_sculpt_and_differ_only_in_the_trailer() {
        let one = committed_fixture("container-v1.brokkr");
        let two = committed_fixture("container-v2.brokkr");
        assert_eq!(
            one.len() + EMPTY_TRAILER_BYTES + fixture_keys().len() * (4 + 39),
            two.len(),
            "the two fixtures differ by more than the key trailer"
        );
        // Everything from the field encoding to the last brick is identical.
        assert_eq!(one[10..], two[10..one.len()], "the fixtures carry different sculpts");

        let (from_one, _) = read(&mut one.as_slice()).expect("version 1 refused");
        let (from_two, _) = read(&mut two.as_slice()).expect("version 2 refused");
        assert_same_bricks(from_one.active_volume(), from_two.active_volume());
    }

    /// Compare two volumes brick for brick, so a fixture failure names the
    /// brick rather than saying only that something changed.
    fn assert_same_bricks(expected: &Volume, found: &Volume) {
        assert_eq!(found.voxel_size(), expected.voxel_size(), "the voxel size changed");
        let mut wanted: Vec<BrickCoord> = expected.brick_coords().collect();
        let mut got: Vec<BrickCoord> = found.brick_coords().collect();
        wanted.sort_unstable();
        got.sort_unstable();
        assert_eq!(wanted, got, "the set of bricks changed");
        for coord in wanted {
            match (expected.brick(coord).expect("listed"), found.brick(coord).expect("listed")) {
                (Brick::Uniform(a), Brick::Uniform(b)) => assert_eq!(a, b, "at {coord:?}"),
                (Brick::Dense(a), Brick::Dense(b)) => {
                    assert_eq!(a.as_slice(), b.as_slice(), "at {coord:?}")
                }
                _ => panic!("the brick at {coord:?} changed kind"),
            }
        }
    }

    /// Compare two documents body for body, so a failure names the body and
    /// the brick rather than saying only that something changed.
    fn assert_same_document(expected: &Document, found: &Document) {
        assert_eq!(found.voxel_size(), expected.voxel_size(), "the document's lattice changed");
        assert_eq!(found.body_count(), expected.body_count(), "the number of bodies changed");
        assert_eq!(found.node_count(), expected.node_count(), "the number of rows changed");
        assert_eq!(found.active(), expected.active(), "the selection moved");
        for (before, after) in expected.nodes().iter().zip(found.nodes()) {
            assert_eq!(after.id, before.id, "the ids came back in a different order");
            assert_eq!(after.name, before.name, "{:?} came back renamed", before.id);
            assert_eq!(after.visible, before.visible, "{:?} came back with another eye", before.id);
            match (before.volume(), after.volume()) {
                (Some(before), Some(after)) => assert_same_bricks(before, after),
                (None, None) => {}
                _ => panic!("a row changed between a body and a folder"),
            }
        }
    }

    /// The real projects on the maintainer's machine, opened for real -- and,
    /// at container version 3, written back out and read again.
    ///
    /// Ignored, and it has to stay ignored: these are tens of megabytes each
    /// and one of them is a 2.3 GB document, so this is a deliberate act on one
    /// machine and never part of a CI run. The manufactured fixtures above are
    /// what guards the format continuously; this is what confirms the guard is
    /// guarding the right thing.
    ///
    /// **The round trip is the half that is new, and it is here because the
    /// first thing to exercise a new container version in the wild is the
    /// autosave -- unwatched, every two minutes, over the user's own crash
    /// net.** It costs scratch space and memory equal to the file: the document
    /// is held, written to a temporary, and read back into a second document
    /// beside the first. `BROKKR_SCRATCH` moves the temporary off the default
    /// temp directory, which is worth doing here because on this machine that
    /// directory is a RAM disk.
    ///
    /// Every path that is absent is skipped and said so, because the file set
    /// moves -- the autosave is rewritten every session and the downloads come
    /// and go. **A run that reports "0 of 5" has checked nothing**, which the
    /// summary line says out loud rather than passing quietly.
    #[test]
    #[ignore = "opens the real multi-megabyte projects on the maintainer's machine"]
    fn the_real_projects_on_this_machine_still_open() {
        let home = std::env::var("HOME").unwrap_or_default();
        let paths = [
            "/storage/offload/Downloads/sculpt.brokkr".to_string(),
            "/storage/offload/Downloads/sculpt1.brokkr".to_string(),
            "/storage/offload/Downloads/sculpt2.brokkr".to_string(),
            "/storage/offload/Downloads/Meshy_AI_Dragonstone_Carver_0823214323_generate_obj/\
             Brokkr.brokkr"
                .to_string(),
            format!("{home}/.local/state/brokkrsculpt/autosave.brokkr"),
        ];

        let mut opened = 0usize;
        for path in &paths {
            let path = std::path::Path::new(path);
            if !path.exists() {
                eprintln!("skipping {}, which is not on this machine", path.display());
                continue;
            }
            let size = std::fs::metadata(path).expect("stat failed").len();
            let file = std::fs::File::open(path).expect("could not open a file that exists");
            // Buffered, and generously. The reader takes four bytes at a time
            // by design -- little endian has to be explicit -- so an unbuffered
            // read of a gigabyte file is hundreds of millions of syscalls.
            let mut reader = std::io::BufReader::with_capacity(1 << 20, file);

            let mut progress = Progress::default();
            let (doc, state) = match read_reporting(&mut reader, &mut progress) {
                Ok(loaded) => loaded,
                Err(error) => panic!("{} ({size} bytes) was refused: {error}", path.display()),
            };
            let volume = doc.active_volume();
            assert_eq!(
                progress.bricks,
                volume.brick_coords().count() as u64,
                "{} reported a different number of bricks than it loaded",
                path.display()
            );
            assert!(volume.voxel_size() > 0.0);
            assert_eq!(doc.body_count(), 1, "these files all predate the node table");
            eprintln!(
                "{}: {size} bytes, {} bricks, voxel {} mm, {} keys",
                path.display(),
                progress.bricks,
                volume.voxel_size(),
                state.keys.len()
            );

            round_trip_through_version_3(path, &doc, &state);
            opened += 1;
        }
        eprintln!("opened {opened} of {} real projects", paths.len());
    }

    /// Save a real document at the current container version and read it back,
    /// comparing brick for brick.
    ///
    /// Through a file rather than a `Vec`, because that is what the application
    /// does and because a `BufWriter` over a file is where a partial write
    /// would actually happen.
    fn round_trip_through_version_3(
        source: &std::path::Path,
        doc: &Document,
        state: &ProjectState,
    ) {
        let scratch = std::env::var_os("BROKKR_SCRATCH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let name = source.file_name().expect("a file has a name");
        let target = scratch.join(format!("brokkr-v3-check-{}", name.to_string_lossy()));

        let started = std::time::Instant::now();
        {
            let file = std::fs::File::create(&target).expect("could not create the scratch file");
            let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
            write(&mut writer, doc, state).expect("this build could not write a real document");
        }
        let written = std::fs::metadata(&target).expect("stat failed").len();

        let file = std::fs::File::open(&target).expect("could not reopen the scratch file");
        let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
        let reread = read(&mut reader);
        // Removed before the assertion, so a failure does not leave gigabytes
        // behind in a temp directory that may be a RAM disk.
        std::fs::remove_file(&target).ok();

        let (reread, reread_state) =
            reread.expect("this build refused a file it had just written itself");
        assert_same_document(doc, &reread);
        assert_eq!(reread_state.view, state.view, "the view did not survive the round trip");
        assert_eq!(reread_state.keys, state.keys, "the timeline did not survive the round trip");
        eprintln!(
            "  round tripped through container {CONTAINER_VERSION}: {written} bytes, {:.0} ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}
