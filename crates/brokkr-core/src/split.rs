// SPDX-License-Identifier: AGPL-3.0-only

//! Splitting one body into its loose parts.
//!
//! A defective scan is exactly the input that arrives as thousands of
//! disconnected shells, and the point of this operation is to get the parts the
//! user actually wants onto rows of their own, so the rest can be hidden,
//! deleted or repaired one at a time.
//!
//! # The walk is per VOXEL, and that is measured rather than argued
//!
//! A brick-level walk -- "two bricks are joined if both hold material and they
//! touch" -- is a hundred times cheaper and gives the wrong answer on every
//! model this feature exists for. On the reference dragon it returns exactly
//! **one** component at 0.25, 0.19, 0.125 and 0.0625 mm, while the voxel walk
//! returns 29, 47, 85 and 182: it fuses 100% of the loose parts and would
//! report "nothing to split" on the model the button was pressed for. On a real
//! four-part model it returns 2 against 4, fusing parts of 4.6% and 4.0% of the
//! model through 4 shared bricks of 781 -- so the error is not proportional to
//! the shared fraction and cannot be waved away as a rounding difference.
//!
//! # `d < 0.0`, never `d <= 0.0`
//!
//! [`crate::voxelise`] biases an exact zero to the inside as `-0.0`, and
//! `-0.0 < 0.0` is false in Rust, so `fast-surface-nets` -- which classifies
//! with `d < 0.0` -- reads such a voxel as OUTSIDE. Anything here that counted
//! it as solid would join two parts through a film of voxels the mesher does
//! not draw, which is invisible in the viewport and wrong in the file.
//! [`crate::cavity`] documents the same trap from the other side, where the
//! same mistake cost a whole sheet of surface.
//!
//! # Four passes
//!
//! **A, parallel.** Six-connected components of `d < 0.0` *within* each brick.
//! Each brick keeps only what the join needs: which of its 6,144 face voxels
//! are solid, as a 768-byte bitmask, **promoted to a 12 KB `u16` plane only for
//! the bricks that hold more than one local component** -- measured at 711 of
//! 22,119 on the dragon, so about 21 MB of scratch instead of 382.
//!
//! **B, serial.** A disjoint set over the `+X`, `+Y` and `+Z` faces only, so
//! each pair of touching bricks is visited once, from the lower side.
//!
//! **C.** Resolve the roots, sum the voxels, sort descending. **Iterating a
//! SORTED coordinate vector and never hash order**, or component ids come out
//! different on every run and nothing downstream -- the names, the fragments
//! sweep, a test -- can be relied on.
//!
//! **D, parallel.** Emit. Each output body gets the source's brick with every
//! voxel belonging to another part raised to [`OUTSIDE`]; a brick that
//! collapses to `OUTSIDE` is dropped, which is what makes the debris bricks
//! vanish rather than being copied into every part.
//!
//! # Why an output also gets bricks it has no material in
//!
//! [`Volume::mesh_brick`] gathers a one voxel apron out of the 26 bricks around
//! the one it is meshing, and an absent brick reads as [`OUTSIDE`]. So if a
//! part's surface runs up to a brick face and the brick on the other side is
//! dropped, the crossing between the last solid voxel and the apron is computed
//! against a saturated `+3` rather than against the small positive value that
//! was really there, and the surface moves by a fraction of a voxel along the
//! whole of that face. That is a seam, and a seam is the class of defect this
//! application exists to remove.
//!
//! So a brick is given to an output when that output has material in it **or on
//! the part of a neighbouring brick that touches it**. The touching test is a
//! `u64` of outputs per brick face, ANDed over the faces a diagonal neighbour
//! shares, which is why a speck in the middle of a brick pulls in nothing at
//! all -- and that matters, because the fragments body is thousands of specks
//! scattered over the whole model and a neighbourhood taken whole would hand it
//! most of the model back.
//!
//! **Measured on the real four-part model**, which is the only honest way to
//! quote what the ring costs: 781 bricks in and 783 out over four bodies at
//! 0.25 mm, 180 in and 186 out at 0.5 mm. The plan predicted the total would
//! come out *smaller* than the input (21,178 of 22,119 on the dragon); it comes
//! out a fraction larger, and the difference is this ring. Paying 0.3% to know
//! the parts mesh exactly as the source did is the right side of that trade.
//!
//! # The ceiling rule runs BEFORE the split, not as a failure after it
//!
//! [`MAX_BODIES`] is 64 and the dragon exceeds it from about 0.125 mm downward,
//! so "split and see" would be an operation that fails on its own headline
//! input. The rule is a significance threshold in **mm³ -- never a rank and
//! never a voxel count**: every part over [`SIGNIFICANT_MM3`] becomes a body,
//! the rest are swept into one `"<name> fragments"` body, and a rank cap is
//! applied on top of that. When even that cap is zero -- 63 bodies or more, or
//! 127 rows -- the split is refused by name rather than squeezed in, because
//! the source's own row is not free until the parts are already there.
//!
//! A rank alone fails because at 0.0625 mm the dragon's largest eight include
//! six specks of 0.023 to 0.064 mm³; a voxel count alone fails because 64
//! voxels is 1.0 mm³ at 0.25 mm and 0.0156 mm³ at 0.0625, so the same model
//! would split differently at each detail level. Meshmixer and Blender both
//! converged on filtering before separating, and Blender's operator hung for
//! over three hours on 110,512 isolated vertices.
//!
//! # The uncomfortable thing, said plainly
//!
//! **Split is the only operation whose peak is twice the thing it operates
//! on**, because history holds the original while the outputs are fresh
//! allocations: measured 2,038 MB held plus 1,920 MB live on the 22k-brick
//! dragon. So the 0.0565 mm, 4.15 GiB dragon needs about 8.3 GiB against the
//! 6 GiB [`crate::MAX_VOLUME_BYTES`] ceiling and is refused before the walk --
//! **the case that motivates split is the case the guard rejects.** The refusal
//! names the coarser voxel size that would fit. This first version handles
//! documents where 2x fits.
//!
//! The way out is to MOVE bricks instead of copying them: 99.3% of bricks
//! belong to exactly one output, which would take the peak to about 1.01x. It
//! is deferred, and the trigger is certain to fire -- the first person who
//! splits a scan at a resin voxel size. That is an unfinished increment
//! honestly labelled and not a real deferral.

use std::time::{Duration, Instant};

use glam::IVec3;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::body::{Document, GrowthGuard, MAX_BODIES, MAX_NODES, NodeId};
use crate::brick::{BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, OUTSIDE};
use crate::cavity::{DIRECTIONS, face_tables, neighbours_in_brick};
use crate::mask::MaskField;
use crate::project::name_that_fits;
use crate::undo::{Change, Entry};
use crate::volume::Volume;

/// The volume below which a loose part is debris rather than a part.
///
/// **Cubic millimetres, so the same model splits the same way at every voxel
/// size.** One cubic millimetre is a speck a hair over a millimetre across; at
/// the resolutions this application prints at, nothing smaller is a thing
/// somebody meant to keep, and a scan produces them in the thousands.
pub const SIGNIFICANT_MM3: f64 = 1.0;

/// Past this, a split is slow enough to be worth saying so out loud.
pub const SLOW_SPLIT: Duration = Duration::from_millis(100);

/// The protection at which a voxel goes to the masked half.
///
/// **Half, and a split is the one operation that has to pick a side.** The mask
/// is eight bits everywhere else precisely so that nothing has to; here there
/// is no half-a-voxel to give each body, so the soft value is thresholded once,
/// in one place, with the consequence written down beside it -- see
/// [`Document::split_masked`].
pub const MASKED_ENOUGH_TO_SPLIT: u8 = 128;

/// Voxels on one face of a brick.
const FACE_VOXELS: usize = BRICK_DIM * BRICK_DIM;

/// Face voxels in a brick, over all six faces.
const FACE_SLOTS: usize = 6 * FACE_VOXELS;

/// Words in the 768-byte face bitmask.
const FACE_WORDS: usize = FACE_SLOTS / 64;

/// The face label of a voxel that is not solid.
///
/// A brick holds at most `BRICK_VOXELS / 2` components -- a three dimensional
/// checkerboard, 16,384 of them -- so no real label can reach this value.
const NO_LABEL: u16 = u16::MAX;

/// Which local component each of a brick's face voxels belongs to.
///
/// The whole reason this is an enum: a brick with one component needs one bit
/// per face voxel and a brick with several needs sixteen, and paying sixteen
/// everywhere is 382 MB of scratch on the dragon against 21.
enum Faces {
    /// Every solid face voxel belongs to local component 0. The bit is set when
    /// the voxel is solid.
    One(Box<[u64; FACE_WORDS]>),
    /// The local component of each face voxel, or [`NO_LABEL`].
    Many(Box<[u16; FACE_SLOTS]>),
}

impl Faces {
    /// The local component at `slot` of `face`, or [`NO_LABEL`].
    #[inline]
    fn label(&self, face: usize, slot: usize) -> u16 {
        let index = face * FACE_VOXELS + slot;
        match self {
            Faces::One(bits) => {
                if bits[index / 64] & (1u64 << (index % 64)) != 0 {
                    0
                } else {
                    NO_LABEL
                }
            }
            Faces::Many(labels) => labels[index],
        }
    }
}

/// A 768-byte face bitmask with nothing set.
fn empty_bits() -> Box<[u64; FACE_WORDS]> {
    // On the heap directly: 768 bytes is nothing, but one per brick of a large
    // model is not, and `Brick::dense_filled` sets the same discipline.
    vec![0u64; FACE_WORDS].into_boxed_slice().try_into().expect("length is FACE_WORDS")
}

/// What pass A learned about one brick.
struct Local {
    /// Solid voxels in each local component, in local label order.
    sizes: Vec<u32>,
    faces: Faces,
}

/// Buffers a worker thread reuses across the bricks it is handed.
///
/// A brick's label array is 64 KB; allocating one per brick would be 22,000
/// allocations of it on the dragon, which is most of the pass.
#[derive(Default)]
struct Scratch {
    labels: Vec<u16>,
    stack: Vec<usize>,
    neighbours: Vec<usize>,
}

/// Six-connected components of `d < 0.0` inside one dense brick.
///
/// Writes the label of every voxel into `scratch.labels` and returns the voxel
/// count of each component, in label order. **A pure function of the brick's
/// contents**, which is what lets pass D re-derive the labelling rather than
/// pass A carrying 64 KB per brick across to it: the seed order is the flat
/// voxel order, so the same array always produces the same labels.
///
/// The seed test is written `>= 0.0` where the flood test is written `< 0.0`.
/// Those are the same test on a field that holds no NaN -- every value is
/// clamped to the narrow band on the way in and a reader refuses one that is
/// not -- and `>= 0.0` is the form [`crate::cavity`] settled on for the reason
/// its own `OUTSIDE_OR_ON_IT` gives: it is the one that reads `-0.0` as
/// outside, exactly as the mesher does.
fn label_dense(data: &[f32; BRICK_VOXELS], scratch: &mut Scratch) -> Vec<u32> {
    let Scratch { labels, stack, neighbours } = scratch;
    labels.clear();
    labels.resize(BRICK_VOXELS, NO_LABEL);

    let mut sizes: Vec<u32> = Vec::new();
    for seed in 0..BRICK_VOXELS {
        if data[seed] >= 0.0 || labels[seed] != NO_LABEL {
            continue;
        }
        let label = u16::try_from(sizes.len()).expect("a brick holds at most 16,384 components");
        let mut count = 0u32;
        labels[seed] = label;
        stack.clear();
        stack.push(seed);
        while let Some(voxel) = stack.pop() {
            count += 1;
            neighbours_in_brick(voxel, neighbours);
            for other in neighbours.iter() {
                if data[*other] < 0.0 && labels[*other] == NO_LABEL {
                    labels[*other] = label;
                    stack.push(*other);
                }
            }
        }
        sizes.push(count);
    }
    sizes
}

/// Pass A for one brick.
fn walk_brick(brick: &Brick, tables: &[Vec<(usize, usize)>; 6], scratch: &mut Scratch) -> Local {
    match brick {
        // A solid tile is one component covering every voxel, and answering
        // that from the tile value is what keeps a filled interior from being
        // flooded 32,768 voxels at a time.
        Brick::Uniform(value) if *value < 0.0 => {
            let mut bits = empty_bits();
            bits.fill(u64::MAX);
            Local { sizes: vec![BRICK_VOXELS as u32], faces: Faces::One(bits) }
        }
        Brick::Uniform(_) => Local { sizes: Vec::new(), faces: Faces::One(empty_bits()) },
        Brick::Dense(data) => {
            let sizes = label_dense(data, scratch);
            let labels = &scratch.labels;
            let faces = if sizes.len() <= 1 {
                let mut bits = empty_bits();
                for (face, table) in tables.iter().enumerate() {
                    for (slot, (here, _)) in table.iter().enumerate() {
                        if labels[*here] != NO_LABEL {
                            let index = face * FACE_VOXELS + slot;
                            bits[index / 64] |= 1u64 << (index % 64);
                        }
                    }
                }
                Faces::One(bits)
            } else {
                let mut plane = vec![NO_LABEL; FACE_SLOTS];
                for (face, table) in tables.iter().enumerate() {
                    for (slot, (here, _)) in table.iter().enumerate() {
                        plane[face * FACE_VOXELS + slot] = labels[*here];
                    }
                }
                Faces::Many(plane.into_boxed_slice().try_into().expect("length is FACE_SLOTS"))
            };
            Local { sizes, faces }
        }
    }
}

/// Union-find over the global label space, with path halving.
///
/// **A root is always the LOWEST id in its set.** Union by rank would be a
/// little faster and would make the tree's shape depend on the order the unions
/// arrive in; keeping the lowest means the root of a set is a property of the
/// set alone, which is one fewer thing standing between this and the
/// determinism pass C promises.
struct Sets {
    parent: Vec<u32>,
}

impl Sets {
    fn new(count: usize) -> Self {
        Self { parent: (0..count as u32).collect() }
    }

    fn find(&mut self, mut label: u32) -> u32 {
        while self.parent[label as usize] != label {
            let grandparent = self.parent[self.parent[label as usize] as usize];
            self.parent[label as usize] = grandparent;
            label = grandparent;
        }
        label
    }

    fn union(&mut self, one: u32, other: u32) {
        let (one, other) = (self.find(one), self.find(other));
        if one == other {
            return;
        }
        let (low, high) = if one < other { (one, other) } else { (other, one) };
        self.parent[high as usize] = low;
    }
}

/// Everything pass D needs, so the walk that answered the user's question is
/// the walk that emits the answer.
struct Labelling {
    /// The source's brick coordinates, SORTED. Every index below is into this.
    coords: Vec<BrickCoord>,
    /// The first global label of each brick.
    offsets: Vec<u32>,
    /// The component of each global label.
    component_of: Vec<u32>,
    /// Which outputs each brick has to be given, as a bit per output.
    ///
    /// One `u64` and not a set, because there are at most [`MAX_BODIES`]
    /// outputs and that is exactly the width of one.
    needed: Vec<u64>,
}

/// One loose part, as the preview card describes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Part {
    /// Solid voxels in it.
    pub voxels: u64,
    /// What that is in cubic millimetres, which is the unit the threshold and
    /// the card are both in.
    pub mm3: f64,
}

/// What splitting one body would produce, worked out before anything is built.
pub struct SplitPlan {
    /// The body that will be consumed.
    pub source: NodeId,
    /// Every loose part the walk found, **largest first**.
    pub parts: Vec<Part>,
    /// How many of them become a body of their own.
    pub kept: usize,
    /// Whether the rank cap, rather than the significance threshold, is what
    /// sent parts to the fragments body.
    ///
    /// The two need different words on the card: "the rest are specks" and
    /// "your document has room for nine more rows" are different facts about
    /// the same press, and only one of them can be answered by pressing again
    /// after tidying up.
    pub capped: bool,
    /// How long the walk took.
    pub elapsed: Duration,
    labelling: Labelling,
    /// The output each component goes to: `kept` bodies, then the fragments
    /// body when there is one.
    output_of: Vec<u32>,
    /// How many output bodies there will be, fragments included.
    outputs: usize,
}

impl SplitPlan {
    /// Loose parts found, which is the number the user sees first.
    #[must_use]
    pub fn found(&self) -> usize {
        self.parts.len()
    }

    /// Parts swept into the one fragments body. Zero when there is none.
    #[must_use]
    pub fn swept(&self) -> usize {
        self.parts.len() - self.kept
    }

    /// Whether a fragments body will be made at all.
    #[must_use]
    pub fn has_fragments(&self) -> bool {
        self.swept() > 0
    }

    /// Bodies the split will leave behind, fragments included.
    #[must_use]
    pub fn bodies(&self) -> usize {
        self.outputs
    }

    /// Whether there is anything to do: one part is one body, and that body is
    /// the one already on the row.
    #[must_use]
    pub fn is_one_piece(&self) -> bool {
        self.parts.len() <= 1
    }

    /// The smallest part that becomes a body of its own, in mm³.
    ///
    /// What the card quotes as the cut-off, because the threshold constant is
    /// not the whole story once the rank cap has bitten.
    #[must_use]
    pub fn smallest_kept_mm3(&self) -> f64 {
        self.parts.get(self.kept.saturating_sub(1)).map_or(0.0, |part| part.mm3)
    }
}

/// What one split did.
pub struct SplitOutcome {
    /// The new bodies, in display order. The first is the largest part and is
    /// left selected; the last is the fragments body when there is one.
    pub bodies: Vec<NodeId>,
    /// The fragments body, when one was made.
    pub fragments: Option<NodeId>,
    /// **ONE entry for the whole gesture**: `[NodeAdded x M, NodeRemoved
    /// {source}]`.
    ///
    /// # The order is the opposite of the one the plan asked for, and it has to
    /// be
    ///
    /// The plan for this increment says `[NodeRemoved{source}, NodeAdded x M]`.
    /// That cannot be built and cannot be undone. It cannot be built because
    /// [`Document::remove`] refuses to take the last body and the body being
    /// split is usually the only one there is; and it cannot be undone because
    /// an entry is applied in reverse, so undo would take the outputs away one
    /// by one and hit that same refusal on the last of them, with the source
    /// not yet restored.
    ///
    /// Adding first and removing last has neither problem: the document never
    /// holds fewer than one body in either direction, and every recorded
    /// position is the position that row really occupied when the change was
    /// made.
    pub entry: Entry,
    /// How long the emit took, on top of [`SplitPlan::elapsed`].
    pub elapsed: Duration,
}

impl SplitOutcome {
    /// The whole gesture, walk excluded. See [`SLOW_SPLIT`].
    #[must_use]
    pub fn is_slow(&self, plan_elapsed: Duration) -> bool {
        self.elapsed + plan_elapsed > SLOW_SPLIT
    }
}

impl Document {
    /// Why this body cannot be split right now, worked out from cached stats
    /// and **without walking a single voxel**.
    ///
    /// `None` means go ahead. There are two refusals and both are free.
    ///
    /// The first is the document's row budget. A split holds the source while
    /// it builds the parts, so making even two bodies needs two free rows, and
    /// [`Document::rank_cap`] returning zero says there are not two. That is
    /// said here rather than survived later: the alternative is a document one
    /// row past a ceiling every fold downstream is written against, and in a
    /// debug build a panic out of [`Document::insert_body`] on a real gesture.
    ///
    /// The second is memory, and it is the real one: see the module doc for why
    /// a split's peak is twice the body it splits. The message names the coarser
    /// voxel size the document would have to be at, because the established
    /// pattern in this codebase is that a refusal names the size that WOULD
    /// work, and a resample is the only lever there is.
    ///
    /// **The mesh pool is deliberately not part of this.** A split hands the
    /// same surface back under different body ids, and the application releases
    /// the source's slots in the same frame it marks the outputs dirty, so the
    /// pool's demand afterwards is what it already holds. The memory ceiling is
    /// the one that binds, and it is the one that was measured.
    pub fn split_guard(&self, source: NodeId) -> Option<String> {
        let bytes = self.volume(source)?.stats().resident_bytes as f64;
        if self.rank_cap() == 0 {
            return Some(no_room_for_a_split(self.body_count(), self.node_count()));
        }
        // Zero vertices asks `GrowthGuard` for the memory ceiling alone, so the
        // pool headroom it is handed is never read.
        split_refusal(&self.growth_guard(0), bytes, self.voxel_size())
    }

    /// Find the loose parts of one body without changing anything.
    ///
    /// `None` when the id names no body. Everything else is a real answer,
    /// "it is one piece" included -- ask [`SplitPlan::is_one_piece`].
    ///
    /// Proportional to the body, and it is the expensive half: passes A, B and
    /// C all happen here, so that the card the user answers is describing a
    /// walk that has already run.
    pub fn split_plan(&self, source: NodeId) -> Option<SplitPlan> {
        let started = Instant::now();
        let volume = self.volume(source)?;

        // Sorted, and everything downstream indexes into this vector rather
        // than into a map. Hash order would renumber the components on every
        // run and the fragments sweep would take a different set each time.
        let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
        coords.sort_unstable();

        let tables = face_tables();
        let locals: Vec<Local> = coords
            .par_iter()
            .map_init(Scratch::default, |scratch, coord| {
                let brick = volume.brick(*coord).expect("the coordinate came from the map");
                walk_brick(brick, &tables, scratch)
            })
            .collect();

        let mut offsets = Vec::with_capacity(coords.len());
        let mut total = 0u32;
        for local in &locals {
            offsets.push(total);
            total += local.sizes.len() as u32;
        }
        let index_of: FxHashMap<BrickCoord, usize> =
            coords.iter().enumerate().map(|(index, coord)| (*coord, index)).collect();

        // Pass B: the join, over the three positive faces only, so each pair of
        // touching bricks is visited exactly once.
        let mut sets = Sets::new(total as usize);
        for (index, coord) in coords.iter().enumerate() {
            for face in [0usize, 2, 4] {
                let Some(&other) = index_of.get(&BrickCoord(coord.0 + DIRECTIONS[face])) else {
                    continue;
                };
                for slot in 0..FACE_VOXELS {
                    let here = locals[index].faces.label(face, slot);
                    if here == NO_LABEL {
                        continue;
                    }
                    // `face + 1` is the opposite direction and the slot order
                    // inside a face is shared: see `cavity::face_tables`.
                    let there = locals[other].faces.label(face + 1, slot);
                    if there != NO_LABEL {
                        sets.union(
                            offsets[index] + u32::from(here),
                            offsets[other] + u32::from(there),
                        );
                    }
                }
            }
        }

        // Pass C: number the components in ascending global label order, which
        // is brick order, which is sorted coordinate order.
        let mut sizes: Vec<u32> = Vec::with_capacity(total as usize);
        for local in &locals {
            sizes.extend_from_slice(&local.sizes);
        }
        let mut component_of = vec![u32::MAX; total as usize];
        let mut voxels: Vec<u64> = Vec::new();
        for label in 0..total {
            let root = sets.find(label);
            let id = if component_of[root as usize] == u32::MAX {
                let id = voxels.len() as u32;
                component_of[root as usize] = id;
                voxels.push(0);
                id
            } else {
                component_of[root as usize]
            };
            component_of[label as usize] = id;
            voxels[id as usize] += u64::from(sizes[label as usize]);
        }

        let mm3_per_voxel = f64::from(self.voxel_size()).powi(3);
        // Largest first, ties broken by the component id so that two runs over
        // the same document produce the same order.
        let mut ranked: Vec<u32> = (0..voxels.len() as u32).collect();
        ranked.sort_by(|a, b| voxels[*b as usize].cmp(&voxels[*a as usize]).then(a.cmp(b)));

        let cap = self.rank_cap();
        let mut kept = 0usize;
        let mut capped = false;
        for (rank, component) in ranked.iter().enumerate() {
            if rank >= cap {
                capped = true;
                break;
            }
            let big_enough = voxels[*component as usize] as f64 * mm3_per_voxel >= SIGNIFICANT_MM3;
            // The largest part is always a body of its own, whatever its size.
            // "Everything here is debris" is not an answer a user can act on,
            // and a document whose only row is a fragments body is a worse
            // place to be than the one they pressed the button from.
            if rank > 0 && !big_enough {
                break;
            }
            kept += 1;
        }

        let outputs = kept + usize::from(kept < ranked.len());
        debug_assert!(outputs <= MAX_BODIES, "the rank cap has to keep the outputs inside a u64");
        let mut output_of = vec![kept as u32; voxels.len()];
        for (rank, component) in ranked.iter().take(kept).enumerate() {
            output_of[*component as usize] = rank as u32;
        }

        let needed = brick_demand(&coords, &offsets, &component_of, &output_of, &locals, &index_of);

        let parts = ranked
            .iter()
            .map(|component| {
                let voxels = voxels[*component as usize];
                Part { voxels, mm3: voxels as f64 * mm3_per_voxel }
            })
            .collect();

        Some(SplitPlan {
            source,
            parts,
            kept,
            capped,
            elapsed: started.elapsed(),
            labelling: Labelling { coords, offsets, component_of, needed },
            output_of,
            outputs,
        })
    }

    /// How many bodies of its own a split may make, against both ceilings.
    ///
    /// **Neither ceiling counts the row the source frees**, and that is not an
    /// oversight: the outputs are inserted while the source is still in the
    /// document, so the transient count really is `existing + outputs` and the
    /// document's own invariants are checked against it on every insert. One
    /// row of the budget is then reserved for the fragments body, which is what
    /// the `- 1` is.
    ///
    /// **Zero is a real answer and must not be floored to one.** This used to
    /// end in `.max(1)`, on the reasoning that a document at the ceiling can
    /// still afford one part and a fragments body because the source is about
    /// to go. The source goes LAST -- see [`SplitOutcome::entry`] for why it has
    /// to -- so at 63 or 64 bodies that pair is inserted against a document with
    /// room for at most one, and [`Document::insert_body`] trips the document's
    /// own invariant on the second of them: measured, "a document holds at most
    /// 64 bodies, not 65", which is a `debug_assert` and so a crash in the
    /// profile `cargo run` builds. Zero means the split is refused by
    /// [`Document::split_guard`] before the walk, which is where every other
    /// reason a split cannot run is already said.
    fn rank_cap(&self) -> usize {
        let by_bodies = MAX_BODIES.saturating_sub(self.body_count()).saturating_sub(1);
        let by_nodes = MAX_NODES.saturating_sub(self.node_count()).saturating_sub(1);
        by_bodies.min(by_nodes)
    }

    /// Carry out a split that has already been planned.
    ///
    /// `None` when the source has left the document since the plan was made,
    /// when the plan found nothing to split, or when the document has no room
    /// for the parts.
    ///
    /// The largest part takes the source's row position and is left selected.
    /// The others follow it in size order and the fragments body is last, which
    /// is where a user looks for the thing they are least likely to want.
    pub fn split(&mut self, plan: SplitPlan) -> Option<SplitOutcome> {
        // `kept == 0` is the row budget having run out, and it is refused
        // before the walk by `Document::split_guard`. Reaching here means a
        // caller planned around that guard, and the answer is still no: one
        // body called "<name> fragments" holding the whole model is not a split
        // worth pushing the document past its own ceiling for.
        if plan.is_one_piece() || plan.kept == 0 {
            return None;
        }
        let started = Instant::now();
        let at = self.index_of(plan.source)?;
        let depth = self.nodes()[at].depth();
        let name = self.node(plan.source)?.name.clone();

        let volumes = self.emit(&plan)?;
        let mut changes = Vec::with_capacity(volumes.len() + 1);
        let mut bodies = Vec::with_capacity(volumes.len());
        let fragments_at = plan.has_fragments().then(|| volumes.len() - 1);

        for (offset, volume) in volumes.into_iter().enumerate() {
            let label = if fragments_at == Some(offset) {
                format!("{name} fragments")
            } else {
                format!("{name} {}", offset + 1)
            };
            // Directly below the source, which is still in the document: see
            // `SplitOutcome::entry` for why the removal comes last.
            let position = at + 1 + offset;
            let id = self.insert_body(position, depth, name_that_fits(&label), volume);
            changes.push(Change::NodeAdded { at: position, id });
            bodies.push(id);
        }

        let node = self.remove(at);
        changes.push(Change::NodeRemoved { at, node: Box::new(node) });

        let first = *bodies.first().expect("a split that is not one piece makes bodies");
        self.set_active(first);
        Some(SplitOutcome {
            fragments: fragments_at.map(|offset| bodies[offset]),
            bodies,
            entry: Entry::new(changes),
            elapsed: started.elapsed(),
        })
    }

    /// Split the source in two along its mask, as one gesture.
    ///
    /// **The main 3D-printing payoff of masking, and it needs no connectivity
    /// walk at all**: which side a voxel goes to is a question its own
    /// protection answers, so the whole operation is one parallel pass over the
    /// bricks. Everything after that is [`Document::split`]'s shape --
    /// `[NodeAdded x2, NodeRemoved{source}]` in that order, for the reason
    /// [`SplitOutcome::entry`] gives at length.
    ///
    /// `None` when the id names no body, or when the mask does not actually
    /// divide it: an all-or-nothing mask would produce one body holding
    /// everything and one holding nothing, which is a rename with extra steps.
    /// Ask [`Document::split_masked_guard`] FIRST -- this allocates both halves
    /// while the source is still resident.
    ///
    /// # Both halves arrive unmasked
    ///
    /// Which is what ZBrush does, and here it is also what keeps resident mask
    /// bytes from doubling at the moment memory is tightest. The mask that
    /// decided the split is on the source, and the source is in the history
    /// entry, so undo brings it back with the protection intact.
    ///
    /// # The boundary is hard, and the doc comment says so rather than the
    /// release notes
    ///
    /// The mask is soft, and a split is not: a voxel goes wholly to one body or
    /// wholly to the other, at [`MASKED_ENOUGH_TO_SPLIT`]. So the cut face is at
    /// voxel resolution and its distances are a step rather than a distance,
    /// exactly as the loose-parts split's are -- feathering cannot help,
    /// because there is no half-a-voxel to give each side.
    pub fn split_masked(&mut self, source: NodeId) -> Option<MaskedSplitOutcome> {
        let started = Instant::now();
        let at = self.index_of(source)?;
        let depth = self.nodes()[at].depth();
        let name = self.node(source)?.name.clone();

        let (masked, free) = self.emit_masked(source)?;
        if masked.brick_count() == 0 || free.brick_count() == 0 {
            return None;
        }

        let mut changes = Vec::with_capacity(3);
        let mut bodies = Vec::with_capacity(2);
        // The masked half first, because it is the one the user selected and
        // the one they will reach for; `Document::split` puts the largest part
        // first for the same reason.
        for (offset, (label, volume)) in
            [(format!("{name} masked"), masked), (format!("{name} rest"), free)]
                .into_iter()
                .enumerate()
        {
            let position = at + 1 + offset;
            let id = self.insert_body(position, depth, name_that_fits(&label), volume);
            changes.push(Change::NodeAdded { at: position, id });
            bodies.push(id);
        }

        let node = self.remove(at);
        changes.push(Change::NodeRemoved { at, node: Box::new(node) });

        let masked = bodies[0];
        self.set_active(masked);
        Some(MaskedSplitOutcome {
            masked,
            rest: bodies[1],
            entry: Entry::new(changes),
            elapsed: started.elapsed(),
        })
    }

    /// Why this body cannot be split along its mask right now, from cached
    /// stats and **without walking a voxel**.
    ///
    /// `None` means go ahead. Three refusals, all free.
    ///
    /// The row budget is [`Document::split_guard`]'s, unchanged: a split holds
    /// the source while it builds the halves, so it needs two free rows.
    ///
    /// The mask itself is the second, and it is the cheap half of "does the
    /// mask divide this body": an empty mask and a Mask All are both one bool
    /// away from being obvious, so neither has to reach the walk.
    ///
    /// # Memory, and where this parts company with the plan's 2.5 R
    ///
    /// The plan predicts `2 x (R + 0.25 R)` on the reading that both copies
    /// carry a mask. They do not -- both halves arrive unmasked, which the same
    /// paragraph also says -- so the number that binds is a different one: the
    /// two halves together hold each brick once, PLUS a second copy of every
    /// brick the boundary passes through, and a generated mask's boundary
    /// passes through essentially every band brick. So the allowance here is
    /// **twice the FIELD**, on top of a source that is still resident with its
    /// mask. That is stricter than 2.5 R at the boundary-heavy end and it is
    /// the side to err on, because the thing being predicted is an
    /// out-of-memory kill and not a slow frame.
    pub fn split_masked_guard(&self, source: NodeId) -> Option<String> {
        self.split_masked_guard_against(source, &self.growth_guard(0))
    }

    /// [`Document::split_masked_guard`] against a stated budget.
    ///
    /// **The budget is a parameter for the same reason `resolve_visibility`
    /// takes solo as one: it is the only way anything can check the
    /// arithmetic.** [`Document::growth_guard`] reads real resident bytes and
    /// the ceiling is six gigabytes, so a test that goes through the public
    /// entry point can only ever observe the fits-easily answer -- and a
    /// refusal proved by calling [`split_refusal`] with hand-written numbers
    /// proves nothing about THIS function, which is where the `2.0 *` and the
    /// `resident - mask` subtraction live. Both of those are one edit away from
    /// reaching a user as an out-of-memory kill on a document with unsaved
    /// work, so both are pinned at the boundary by the tests below.
    fn split_masked_guard_against(&self, source: NodeId, guard: &GrowthGuard) -> Option<String> {
        let volume = self.volume(source)?;
        if volume.mask().is_free() {
            return Some(format!("{} carries no mask to split along", self.name_of(source)));
        }
        if volume.mask().protects_everything() {
            return Some(format!(
                "the mask covers all of {}, so there is nothing on the other side of it",
                self.name_of(source)
            ));
        }
        if self.rank_cap() == 0 {
            return Some(no_room_for_a_split(self.body_count(), self.node_count()));
        }
        let stats = volume.stats();
        let field = stats.resident_bytes.saturating_sub(stats.mask_bytes) as f64;
        split_refusal(guard, 2.0 * field, self.voxel_size())
    }

    /// One volume for the protected side and one for the rest.
    fn emit_masked(&self, source: NodeId) -> Option<(Volume, Volume)> {
        let volume = self.volume(source)?;
        let mask = volume.mask();
        let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
        coords.sort_unstable();

        let split: Vec<(BrickCoord, Option<Brick>, Option<Brick>)> = coords
            .par_iter()
            .map(|coord| {
                let brick = volume.brick(*coord).expect("the coordinate came from the map");
                let (a, b) = split_brick_by_mask(brick, mask, *coord);
                (*coord, a, b)
            })
            .collect();

        let mut masked = Volume::new(self.voxel_size());
        let mut rest = Volume::new(self.voxel_size());
        for (coord, a, b) in split {
            if let Some(brick) = a {
                masked.insert_brick(coord, brick);
            }
            if let Some(brick) = b {
                rest.insert_brick(coord, brick);
            }
        }
        // Both halves are brand new to the renderer, and a volume built by
        // `insert_brick` has marked nothing.
        masked.mark_everything_dirty();
        rest.mark_everything_dirty();
        Some((masked, rest))
    }

    /// One body's name, for a refusal that has to say which body it is about.
    fn name_of(&self, body: NodeId) -> String {
        self.node(body).map_or_else(|| "this body".to_string(), |node| node.name.clone())
    }

    /// Pass D: one volume per output, built in parallel over the source's
    /// bricks.
    fn emit(&self, plan: &SplitPlan) -> Option<Vec<Volume>> {
        let volume = self.volume(plan.source)?;
        let Labelling { coords, offsets, component_of, needed } = &plan.labelling;

        let written: Vec<Vec<(u32, BrickCoord, Brick)>> = coords
            .par_iter()
            .enumerate()
            .map_init(Scratch::default, |scratch, (index, coord)| {
                // A coordinate the source no longer holds emits nothing rather
                // than panicking. **The plan is only valid while the document
                // stands still**, which the preview card guarantees by being
                // modal; this is what keeps a stale plan a wrong answer instead
                // of a crash.
                let Some(brick) = volume.brick(*coord) else {
                    return Vec::new();
                };
                emit_brick(
                    brick,
                    *coord,
                    offsets[index] as usize,
                    component_of,
                    &plan.output_of,
                    needed[index],
                    scratch,
                )
            })
            .collect();

        let mut volumes: Vec<Volume> =
            (0..plan.outputs).map(|_| Volume::new(self.voxel_size())).collect();
        for batch in written {
            for (output, coord, brick) in batch {
                volumes[output as usize].insert_brick(coord, brick);
            }
        }
        for volume in &mut volumes {
            // Every output is brand new to the renderer, and a volume built by
            // `insert_brick` has marked nothing. Without this the bodies are
            // right in every headless assertion and invisible on screen, which
            // is a failure this project has shipped twice.
            volume.mark_everything_dirty();
        }
        Some(volumes)
    }
}

/// One source brick as the two halves of a masked split see it.
///
/// `(the protected half, the rest)`, and either may be `None` when nothing of
/// that half is in this brick -- which is what keeps a body split down the
/// middle from costing two whole copies of itself.
///
/// The three arms are the same three [`emit_brick`] has and for the same
/// reason: a brick the mask does not divide is answered structurally, without a
/// voxel loop and without a 128 KB promotion, and on a hand-painted mask that
/// is nearly all of them.
fn split_brick_by_mask(
    brick: &Brick,
    mask: &MaskField,
    coord: BrickCoord,
) -> (Option<Brick>, Option<Brick>) {
    // Resolved once for the whole brick, exactly as the cut and the brush
    // resolve it, so an unmasked brick and a collapsed tile both pay nothing
    // per voxel.
    if let Some(protection) = mask.protection_fill(coord) {
        let whole = Some(brick.clone());
        return if protection >= MASKED_ENOUGH_TO_SPLIT { (whole, None) } else { (None, whole) };
    }

    let slab = mask.slab(coord);
    let mut protected: Option<Vec<f32>> = None;
    let mut rest: Option<Vec<f32>> = None;
    for voxel in 0..BRICK_VOXELS {
        let value = match brick {
            Brick::Uniform(value) => *value,
            Brick::Dense(data) => data[voxel],
        };
        // A voxel that holds nothing goes to neither half, so an empty brick
        // with a mask boundary through it stays empty in both.
        let (mine, theirs) = if mask.resolve(slab.byte_at(voxel)) >= MASKED_ENOUGH_TO_SPLIT {
            (&mut protected, &mut rest)
        } else {
            (&mut rest, &mut protected)
        };
        mine.get_or_insert_with(|| vec![OUTSIDE; BRICK_VOXELS])[voxel] = value;
        // Written explicitly rather than left at the fill, so that adding an
        // arm to this match cannot silently leave a voxel in both halves.
        theirs.get_or_insert_with(|| vec![OUTSIDE; BRICK_VOXELS])[voxel] = OUTSIDE;
    }

    let finish = |values: Option<Vec<f32>>| {
        let values = values?;
        let boxed: Box<[f32]> = values.into_boxed_slice();
        let mut brick = Brick::Dense(boxed.try_into().expect("length is BRICK_VOXELS"));
        if let Some(value) = brick.is_collapsible() {
            if value >= OUTSIDE {
                return None;
            }
            brick = Brick::Uniform(value);
        }
        Some(brick)
    };
    (finish(protected), finish(rest))
}

/// What one masked split did.
pub struct MaskedSplitOutcome {
    /// The half the mask protected. Left selected, because it is the half the
    /// user chose.
    pub masked: NodeId,
    /// Everything else.
    pub rest: NodeId,
    /// **ONE entry for the whole gesture**: `[NodeAdded x2,
    /// NodeRemoved{source}]`, in that order -- see [`SplitOutcome::entry`] for
    /// why the other order cannot be built and cannot be undone.
    pub entry: Entry,
    /// How long the pass took.
    pub elapsed: Duration,
}

/// Which outputs each brick has to be handed, as a bit per output.
///
/// An output needs a brick when it has material in it, and also when it has
/// material in a neighbour **on the part of that neighbour which touches this
/// brick** -- see the module doc on the apron. The touching test for a diagonal
/// neighbour is the AND of the neighbour's face masks over the faces the two
/// share, which is a superset of the exact shared edge or corner and is one
/// `u64` operation instead of a second indexing scheme.
fn brick_demand(
    coords: &[BrickCoord],
    offsets: &[u32],
    component_of: &[u32],
    output_of: &[u32],
    locals: &[Local],
    index_of: &FxHashMap<BrickCoord, usize>,
) -> Vec<u64> {
    // The outputs on each face of each brick, and anywhere in it.
    let masks: Vec<([u64; 6], u64)> = locals
        .par_iter()
        .enumerate()
        .map(|(index, local)| {
            let base = offsets[index] as usize;
            let bit = |label: u16| 1u64 << output_of[component_of[base + label as usize] as usize];
            let mut own = 0u64;
            for label in 0..local.sizes.len() {
                own |= bit(label as u16);
            }
            let mut faces = [0u64; 6];
            if own != 0 {
                for (face, mask) in faces.iter_mut().enumerate() {
                    for slot in 0..FACE_VOXELS {
                        let label = local.faces.label(face, slot);
                        if label != NO_LABEL {
                            *mask |= bit(label);
                        }
                    }
                }
            }
            (faces, own)
        })
        .collect();

    coords
        .par_iter()
        .enumerate()
        .map(|(index, coord)| {
            let mut wanted = masks[index].1;
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if (dx, dy, dz) == (0, 0, 0) {
                            continue;
                        }
                        let Some(other) =
                            index_of.get(&BrickCoord(coord.0 + IVec3::new(dx, dy, dz)))
                        else {
                            continue;
                        };
                        // The neighbour's faces that point back at this brick.
                        // `DIRECTIONS` pairs opposites, so the face for `+1` on
                        // an axis is the one for `-1` and the other way round.
                        let mut touching = u64::MAX;
                        for (axis, step) in [dx, dy, dz].into_iter().enumerate() {
                            if step != 0 {
                                let face = 2 * axis + usize::from(step > 0);
                                touching &= masks[*other].0[face];
                            }
                        }
                        wanted |= touching;
                    }
                }
            }
            wanted
        })
        .collect()
}

/// One brick of the source, as each output that needs it sees it.
fn emit_brick(
    brick: &Brick,
    coord: BrickCoord,
    base: usize,
    component_of: &[u32],
    output_of: &[u32],
    needed: u64,
    scratch: &mut Scratch,
) -> Vec<(u32, BrickCoord, Brick)> {
    let outputs = || (0..u64::BITS).filter(move |bit| needed & (1u64 << bit) != 0);
    let mut out = Vec::new();
    match brick {
        Brick::Uniform(value) if *value < 0.0 => {
            // A solid tile is one component, so exactly one output keeps it and
            // every other reads it as material that has been taken away, which
            // is a tile of `OUTSIDE` and is dropped.
            let owner = output_of[component_of[base] as usize];
            out.push((owner, coord, Brick::Uniform(*value)));
        }
        Brick::Uniform(value) => {
            // No solid voxels, so nothing is taken away from anyone. A
            // saturated empty tile is dropped; a mid-band one -- which only a
            // file can produce -- is band data around somebody's surface and
            // goes to every output that reaches into this brick.
            if *value < OUTSIDE {
                out.extend(outputs().map(|output| (output, coord, Brick::Uniform(*value))));
            }
        }
        Brick::Dense(data) => {
            label_dense(data, scratch);
            for output in outputs() {
                let mut copy = data.to_vec();
                for (voxel, value) in copy.iter_mut().enumerate() {
                    let label = scratch.labels[voxel];
                    if label != NO_LABEL
                        && output_of[component_of[base + label as usize] as usize] != output
                    {
                        *value = OUTSIDE;
                    }
                }
                let boxed: Box<[f32]> = copy.into_boxed_slice();
                let mut brick = Brick::Dense(boxed.try_into().expect("length is BRICK_VOXELS"));
                if let Some(value) = brick.is_collapsible() {
                    // Everything this output had here belonged to another part,
                    // so the brick is empty for it and does not exist. That is
                    // what makes the debris bricks vanish rather than being
                    // copied into every one of forty bodies.
                    if value >= OUTSIDE {
                        continue;
                    }
                    brick = Brick::Uniform(value);
                }
                out.push((output, coord, brick));
            }
        }
    }
    out
}

/// Why a document with no row budget left cannot be split, and what to do.
///
/// Names whichever of the two ceilings is the one that has run out, because
/// "you are out of bodies" and "you are out of rows" send a reader to different
/// remedies: the second can be answered by deleting a folder.
fn no_room_for_a_split(bodies: usize, nodes: usize) -> String {
    let (held, ceiling, what) =
        if MAX_BODIES.saturating_sub(bodies) <= MAX_NODES.saturating_sub(nodes) {
            (bodies, MAX_BODIES, "bodies")
        } else {
            (nodes, MAX_NODES, "rows")
        };
    format!(
        "this document holds {held} of its {ceiling} {what}, and a split holds the source row \
         while it builds the parts -- so it needs two free rows before it can make even two \
         bodies. Delete or merge a row first"
    )
}

/// Why a body of `bytes` cannot be split against this guard, and what to do.
///
/// Separated from [`Document::split_guard`] because the ceiling it judges
/// against cannot be reached by building a document -- six gigabytes of
/// resident bricks is six gigabytes of real allocation -- so the only way to
/// test the refusal is to state the numbers.
fn split_refusal(guard: &GrowthGuard, bytes: f64, voxel_size: f32) -> Option<String> {
    let why = guard.no_room_for_a_copy(bytes, 0.0)?;
    let (_, workable) = guard.no_room_for(bytes, 0.0)?;
    // `workable` is a LINEAR fraction and a body's cost is a shell over a
    // surface, so a body at that fraction of the size costs what this one costs
    // at `voxel_size / workable`. The same square law the resample guard runs,
    // read the other way round.
    let coarser = voxel_size / workable.max(f32::MIN_POSITIVE);
    Some(format!(
        "{why} -- a split holds the original and the parts at the same time, so resampling to \
         about {coarser:.3} mm first is what would fit"
    ))
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use rustc_hash::FxHashSet;

    use super::*;
    use crate::export::ExportMesh;
    use crate::project::MAX_VOLUME_BYTES;
    use crate::undo::History;
    use crate::voxelise::{VoxeliseOptions, voxelise};

    const VOXEL: f32 = 0.5;

    /// The brick-level walk this module refuses to use, so that the tests can
    /// show what it would have answered instead of taking the plan's word.
    ///
    /// Two bricks are joined when both hold material and they touch. That is
    /// the cheap approximation, and every fixture below that is about
    /// connectivity states both numbers.
    fn brick_walk(volume: &Volume) -> usize {
        let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
        coords.sort_unstable();
        let index_of: FxHashMap<BrickCoord, usize> =
            coords.iter().enumerate().map(|(index, coord)| (*coord, index)).collect();
        let solid: Vec<bool> = coords
            .iter()
            .map(|coord| match volume.brick(*coord) {
                Some(Brick::Uniform(value)) => *value < 0.0,
                Some(Brick::Dense(data)) => data.iter().any(|value| *value < 0.0),
                None => false,
            })
            .collect();

        let mut sets = Sets::new(coords.len());
        for (index, coord) in coords.iter().enumerate() {
            if !solid[index] {
                continue;
            }
            for face in [0usize, 2, 4] {
                if let Some(&other) = index_of.get(&BrickCoord(coord.0 + DIRECTIONS[face]))
                    && solid[other]
                {
                    sets.union(index as u32, other as u32);
                }
            }
        }
        let mut roots: Vec<u32> = (0..coords.len() as u32)
            .filter(|index| solid[*index as usize])
            .map(|index| sets.find(index))
            .collect();
        roots.sort_unstable();
        roots.dedup();
        roots.len()
    }

    /// Every voxel of a volume the mesher would call solid.
    fn solid_voxels(volume: &Volume) -> u64 {
        volume
            .brick_coords()
            .map(|coord| match volume.brick(coord) {
                Some(Brick::Uniform(value)) if *value < 0.0 => BRICK_VOXELS as u64,
                Some(Brick::Dense(data)) => {
                    data.iter().filter(|value| **value < 0.0).count() as u64
                }
                _ => 0,
            })
            .sum()
    }

    /// Bricks a volume holds that carry any solid voxel of their own.
    ///
    /// **The apron ring's whole effect is the gap between this and
    /// [`Volume::brick_count`]**, so a fixture where the two are equal is a
    /// fixture that cannot see the ring.
    fn bricks_holding_material(volume: &Volume) -> usize {
        volume
            .brick_coords()
            .filter(|coord| match volume.brick(*coord) {
                Some(Brick::Uniform(value)) => *value < 0.0,
                Some(Brick::Dense(data)) => data.iter().any(|value| *value < 0.0),
                None => false,
            })
            .count()
    }

    /// A vertex position as three exact bit patterns, so "the surface did not
    /// move" is an equality and not a tolerance.
    fn bits(position: &Vec3) -> [u32; 3] {
        [position.x.to_bits(), position.y.to_bits(), position.z.to_bits()]
    }

    /// Every stored voxel folded into one number, so that "bit for bit" is a
    /// claim an assertion really checks. Taken from `merge.rs`, which needed the
    /// same thing for the same reason: four probes would not catch a restore
    /// that put the right census back with the wrong values.
    fn checksum(volume: &Volume) -> u64 {
        let mut coords: Vec<BrickCoord> = volume.brick_coords().collect();
        coords.sort_unstable();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for coord in coords {
            for part in [coord.0.x, coord.0.y, coord.0.z] {
                hash = (hash ^ part as u64).wrapping_mul(0x1000_0000_01b3);
            }
            match volume.brick(coord) {
                None => {}
                Some(Brick::Uniform(value)) => {
                    hash = (hash ^ u64::from(value.to_bits())).wrapping_mul(0x1000_0000_01b3);
                }
                Some(Brick::Dense(data)) => {
                    for value in data.iter() {
                        hash = (hash ^ u64::from(value.to_bits())).wrapping_mul(0x1000_0000_01b3);
                    }
                }
            }
        }
        hash
    }

    /// A field holding every ball, on one lattice.
    ///
    /// **Not `seed_sphere` called several times on one volume**: it *replaces*
    /// everything inside the brick box it touches, so seeding a second ball
    /// whose bricks overlap the first would delete the first. Each ball gets its
    /// own volume and they are unioned, which is the same `min` a merge is.
    fn balls(centres_and_radii: &[(Vec3, f32)]) -> Volume {
        let mut volume = Volume::new(VOXEL);
        for (centre, radius) in centres_and_radii {
            let mut one = Volume::new(VOXEL);
            one.seed_sphere(*centre, *radius);
            volume.begin_stroke();
            volume.union_from(&one);
            let _ = volume.end_stroke();
        }
        volume.mark_everything_dirty();
        volume
    }

    fn document_of(volume: Volume) -> (Document, NodeId) {
        let doc = Document::from_volume(volume);
        let id = doc.active();
        (doc, id)
    }

    /// Two balls a millimetre apart, close enough to share every brick.
    ///
    /// This is the whole argument for the voxel walk in eight bricks: the brick
    /// map cannot tell them apart and the voxels can.
    fn two_spheres_sharing_bricks() -> Volume {
        balls(&[(Vec3::new(-5.0, 0.0, 0.0), 3.0), (Vec3::new(5.0, 0.0, 0.0), 3.0)])
    }

    /// Four loose parts with the shape of `basic150.stl`'s: one large, two
    /// middling, one small, three of them near enough to share bricks and one
    /// well clear.
    ///
    /// **A stand-in for a reduction of `basic150.stl` that could not be made**,
    /// and the reason is measured rather than assumed. That model is 12.8 MB of
    /// binary STL and is not in the repository, so the plan asked for a reduced
    /// copy to be committed. There is no reduction: re-meshing it through the
    /// voxeliser at 0.5 mm produces 357,832 triangles and a 17.9 MB file --
    /// *larger* than the original, because a marching mesh spends two triangles
    /// a surface voxel -- and every coarser resampling destroys the property
    /// under test, since the model's thin parts perforate. Measured on the real
    /// file: 4 parts at 0.5 mm and at 0.25 mm, but **12 at 1.0 mm and 13 at
    /// 2.0 mm**, which is a shattered model rather than a smaller one.
    ///
    /// So the real file is exercised by
    /// [`the_four_parts_of_the_reference_model_are_found_where_a_brick_walk_finds_two`],
    /// which is `#[ignore]`d because the model is on one machine, and this
    /// fixture carries the same signature -- voxel walk 4, brick walk 2 -- into
    /// every checkout.
    fn four_loose_parts() -> Volume {
        balls(&[
            (Vec3::ZERO, 8.0),
            (Vec3::new(11.0, 0.0, 0.0), 2.0),
            (Vec3::new(0.0, 11.0, 0.0), 2.0),
            (Vec3::new(0.0, 0.0, 60.0), 5.0),
        ])
    }

    /// Two balls with a brick face between them, and the reason the placement
    /// is so fussy.
    ///
    /// The apron ring only does anything when a part's SOLID voxels reach the
    /// last layer of a brick and its surface falls between that layer and the
    /// first layer of the brick beyond -- that is exactly when the crossing cell
    /// straddles the face and the brick on the other side is one the part has no
    /// material in. A brick is 32 voxels, so 16 mm at this fixture's 0.5 mm: a
    /// ball of radius 7.75 centred on (8, 8, 8) has its surface at 15.75 on all
    /// three axes, half a voxel short of the face at 16, and the second ball
    /// puts another part's material in the brick beyond it.
    ///
    /// **Measured both ways.** With the ring, the larger part comes out holding
    /// 6 bricks of which 1 carries its own material, and every one of its 4,496
    /// vertices is a vertex the source had. With `brick_demand` short-circuited
    /// to its own-material mask it comes out holding 1 brick, the same 4,496
    /// vertices, and 180 of them in places the source's surface never was.
    fn across_a_brick_face() -> Volume {
        balls(&[(Vec3::splat(8.0), 7.75), (Vec3::new(24.0, 8.0, 8.0), 7.0)])
    }

    /// Thirty-six equal balls in a grid, seeded in one order or the other.
    ///
    /// Every part is the same size, so the ranking falls through to the
    /// component id on every single comparison and the id is decided by the
    /// order the bricks were walked in. Parts of different sizes hide the whole
    /// question behind the size ranking, which is why [`four_loose_parts`]
    /// cannot be the fixture for it.
    fn a_grid_of_equal_parts(reversed: bool) -> Volume {
        let mut placed = Vec::new();
        for row in 0..6 {
            for column in 0..6 {
                placed.push((Vec3::new(row as f32 * 20.0, column as f32 * 20.0, 0.0), 3.0));
            }
        }
        if reversed {
            placed.reverse();
        }
        balls(&placed)
    }

    fn box_mesh(min: Vec3, max: Vec3, inward: bool) -> ExportMesh {
        let corners = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        let faces: [[usize; 3]; 12] = [
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
        let triangles = faces
            .into_iter()
            .map(|face| {
                // An inner shell faces the other way, or the winding number
                // counts it as more solid rather than as a void.
                if inward {
                    [face[0] as u32, face[2] as u32, face[1] as u32]
                } else {
                    [face[0] as u32, face[1] as u32, face[2] as u32]
                }
            })
            .collect();
        ExportMesh {
            positions: corners.to_vec(),
            normals: Vec::new(),
            triangles,
            slots: Vec::new(),
        }
    }

    /// A box with a sealed box of nothing inside it, left UNFILLED.
    ///
    /// The wall is 6 mm at a 0.5 mm voxel, so it is twelve voxels thick and the
    /// two surfaces are nowhere near each other's band.
    fn sealed_cavity() -> Volume {
        let mut mesh = box_mesh(Vec3::splat(-15.0), Vec3::splat(15.0), false);
        let inner = box_mesh(Vec3::splat(-9.0), Vec3::splat(9.0), true);
        let offset = mesh.positions.len() as u32;
        mesh.positions.extend(inner.positions);
        mesh.triangles
            .extend(inner.triangles.into_iter().map(|triangle| triangle.map(|i| i + offset)));

        let options = VoxeliseOptions {
            voxel_size: VOXEL,
            centre: false,
            refit_if_implausible: false,
            // OFF, deliberately: the point is that a void the fill would have
            // closed does not read as a second part while it is still open.
            fill_sealed_cavities: false,
            repair_broken_scan_lines: true,
            coarsen_to_fit: false,
            refine_to_resolve: false,
            already_reserved: 0.0,
        };
        let (volume, _) = voxelise(&mesh, &options).expect("the fixture should voxelise");
        volume
    }

    // --- what the walk finds -------------------------------------------------

    #[test]
    fn two_spheres_that_share_every_brick_are_still_two_parts() {
        let (doc, id) = document_of(two_spheres_sharing_bricks());
        assert_eq!(doc.active_volume().brick_count(), 8, "the fixture is not the eight brick one");
        assert_eq!(brick_walk(doc.active_volume()), 1, "the fixture's bricks have to be fused");

        let plan = doc.split_plan(id).expect("the body is there");
        assert_eq!(plan.found(), 2, "the voxel walk did not separate two balls in one brick map");
        assert!(!plan.is_one_piece());
        assert_eq!(plan.kept, 2, "two balls of 113 mm^3 are both well over the threshold");
        assert!(!plan.has_fragments(), "nothing here is debris");
    }

    #[test]
    fn four_loose_parts_are_found_where_a_brick_walk_finds_two() {
        let (doc, id) = document_of(four_loose_parts());
        assert_eq!(
            brick_walk(doc.active_volume()),
            2,
            "the fixture does not carry the reference model's signature"
        );
        let plan = doc.split_plan(id).expect("the body is there");
        assert_eq!(plan.found(), 4, "the voxel walk lost a part");
        assert_eq!(plan.kept, 4, "every part here is over a cubic millimetre");
        assert_eq!(plan.bodies(), 4, "four parts and no debris is four bodies");
    }

    /// The real model, on the one machine that has it.
    ///
    /// `#[ignore]`d because `basic150.stl` is not in the repository and cannot
    /// be reduced into it -- see [`four_loose_parts`] for the measurement that
    /// settled that. Run it with
    /// `cargo test -p brokkr-core --lib -- --ignored reference_model --nocapture`,
    /// optionally with `BROKKR_BASIC150` naming the file.
    ///
    /// Measured 2026-08-25, and these are the numbers the plan quotes: at
    /// 0.25 mm the model is 781 bricks, the voxel walk finds 4 parts and the
    /// brick walk finds 2; at 0.5 mm it is 180 bricks with the same 4 and 2.
    #[test]
    #[ignore = "reads a model that is not in the repository"]
    fn the_four_parts_of_the_reference_model_are_found_where_a_brick_walk_finds_two() {
        let path = std::env::var("BROKKR_BASIC150").map_or_else(
            |_| {
                std::path::PathBuf::from(std::env::var("HOME").expect("a home directory"))
                    .join("Models/basic150.stl")
            },
            std::path::PathBuf::from,
        );
        let mesh = crate::import::read_path(&path).expect("reading the reference model");
        for (voxel, bricks) in [(0.5f32, 180usize), (0.25, 781)] {
            let options = VoxeliseOptions {
                refine_to_resolve: false,
                coarsen_to_fit: false,
                ..VoxeliseOptions::at(voxel)
            };
            let (volume, _) = voxelise(&mesh, &options).expect("voxelising");
            let (doc, id) = document_of(volume);
            let plan = doc.split_plan(id).expect("the body is there");
            eprintln!(
                "{voxel} mm: {} bricks, voxel walk {}, brick walk {}",
                doc.active_volume().brick_count(),
                plan.found(),
                brick_walk(doc.active_volume())
            );
            assert_eq!(doc.active_volume().brick_count(), bricks);
            assert_eq!(plan.found(), 4, "the voxel walk at {voxel} mm");
            assert_eq!(brick_walk(doc.active_volume()), 2, "the brick walk at {voxel} mm");

            // What the apron ring really costs, on a real model rather than on
            // four balls: the number the module doc quotes.
            let mut doc = doc;
            let plan = doc.split_plan(id).expect("the body is there");
            let outcome = doc.split(plan).expect("four parts is a split");
            let out: usize = outcome
                .bodies
                .iter()
                .map(|body| doc.volume(*body).expect("a new body").brick_count())
                .sum();
            eprintln!("  {bricks} bricks in, {out} out over {} bodies", outcome.bodies.len());
        }
    }

    #[test]
    fn a_model_with_a_sealed_void_inside_it_is_one_body() {
        let (doc, id) = document_of(sealed_cavity());
        assert!(
            doc.active_volume().sample_world(Vec3::ZERO) > 0.0,
            "the fixture is not hollow, so this proves nothing"
        );
        let plan = doc.split_plan(id).expect("the body is there");
        assert_eq!(plan.found(), 1, "the inner surface was read as a second part");
        assert!(plan.is_one_piece());
    }

    /// Planning is a pure read, and two plans over one document agree.
    ///
    /// **This does NOT hold the sort in `split_plan`, whatever its old name
    /// said.** `FxHashMap`'s hasher is fixed and unseeded, so one map always
    /// iterates the same way and this test passes with `coords.sort_unstable()`
    /// deleted -- confirmed by deleting it. What it does hold is that nothing
    /// inside the walk is order-dependent between calls, the parallel pass A
    /// included. The sort has a test of its own below.
    #[test]
    fn planning_the_same_document_twice_gives_the_same_plan() {
        let (doc, id) = document_of(four_loose_parts());
        let one = doc.split_plan(id).expect("the body is there");
        let two = doc.split_plan(id).expect("the body is there");
        assert_eq!(one.parts, two.parts, "the parts came out in a different order or size");
        assert_eq!(one.output_of, two.output_of, "the same component went to a different body");
        assert_eq!(one.labelling.component_of, two.labelling.component_of);
    }

    /// The property `coords.sort_unstable()` really provides.
    ///
    /// It is not "two calls agree" -- see above for why that is free. It is that
    /// the answer survives two different build HISTORIES of the same geometry:
    /// an import, a reload of the file it was saved to, and a map that has been
    /// edited since all hold the same bricks in different slots. Here the same
    /// balls are seeded in opposite orders.
    ///
    /// **Measured**: the two maps hold the same 128 keys, walk them in different
    /// orders, and produce identical parts. With the sort removed they produce
    /// the same thirty-six parts on differently ordered rows, so "Scan 1" is a
    /// different lump of geometry depending on how the document was built.
    #[test]
    fn the_parts_come_out_in_the_same_order_whichever_way_the_brick_map_was_built() {
        let forward = a_grid_of_equal_parts(false);
        let backward = a_grid_of_equal_parts(true);
        assert_eq!(
            checksum(&forward),
            checksum(&backward),
            "the two seed orders built different fields, so this compares nothing"
        );
        assert_ne!(
            forward.brick_coords().collect::<Vec<_>>(),
            backward.brick_coords().collect::<Vec<_>>(),
            "both routes now walk the map in the same order, so this fixture can no longer see \
             the sort at all"
        );

        let parts_of = |volume: Volume| {
            let (mut doc, id) = document_of(volume);
            let plan = doc.split_plan(id).expect("the body is there");
            assert_eq!(plan.found(), 36, "the fixture is not thirty-six parts");
            assert_eq!(plan.kept, 36, "a ball of 113 mm^3 is well over the threshold");
            let outcome = doc.split(plan).expect("thirty-six parts is a split");
            outcome
                .bodies
                .iter()
                .map(|body| checksum(doc.volume(*body).expect("a new body")))
                .collect::<Vec<u64>>()
        };
        assert_eq!(
            parts_of(forward),
            parts_of(backward),
            "the same geometry built two ways put different parts on the same rows, so the \
             names a user reads are not stable across a reload"
        );
    }

    // --- what the split produces ---------------------------------------------

    #[test]
    fn the_outputs_hold_exactly_the_solid_voxels_the_source_held() {
        let (mut doc, id) = document_of(four_loose_parts());
        let before = solid_voxels(doc.active_volume());
        let plan = doc.split_plan(id).expect("the body is there");
        let outcome = doc.split(plan).expect("four parts is a split");

        let after: u64 = outcome
            .bodies
            .iter()
            .map(|body| solid_voxels(doc.volume(*body).expect("a new body")))
            .sum();
        assert_eq!(after, before, "material was lost or duplicated by the emit");
    }

    #[test]
    fn each_part_becomes_a_body_that_exports_watertight() {
        let (mut doc, id) = document_of(two_spheres_sharing_bricks());
        let plan = doc.split_plan(id).expect("the body is there");
        let outcome = doc.split(plan).expect("two balls is a split");
        assert_eq!(outcome.bodies.len(), 2);

        for body in &outcome.bodies {
            let volume = doc.volume(*body).expect("a new body");
            let (mesh, report) = volume.export_mesh();
            assert!(!mesh.triangles.is_empty(), "a part came out with no surface");
            assert!(
                report.is_printable(),
                "a part that shared its bricks with another did not come out closed: {}",
                report.summary()
            );
        }
    }

    /// The apron ring, pinned by GEOMETRY rather than by a census.
    ///
    /// A vertex count cannot see this and neither can watertightness: with the
    /// ring removed the part still exports 4,496 vertices and still closes, and
    /// 180 of those vertices have moved a fraction of a voxel. That is the seam
    /// [`across_a_brick_face`] exists to catch, and the two assertions here are
    /// the two halves of it -- the first that the fixture still exercises the
    /// ring at all, the second that the surface came through it unmoved.
    #[test]
    fn a_part_whose_surface_runs_up_to_a_brick_face_comes_out_where_the_source_had_it() {
        let (mut doc, id) = document_of(across_a_brick_face());
        let source: FxHashSet<[u32; 3]> =
            doc.active_volume().export_mesh().0.positions.iter().map(bits).collect();

        let plan = doc.split_plan(id).expect("the body is there");
        assert_eq!(plan.found(), 2, "the fixture is not two parts");
        let outcome = doc.split(plan).expect("two parts is a split");
        let part = doc.volume(outcome.bodies[0]).expect("the larger part");

        assert!(
            part.brick_count() > bricks_holding_material(part),
            "every one of the part's {} bricks carries its own material, so this fixture no \
             longer exercises the apron ring",
            part.brick_count()
        );

        let mesh = part.export_mesh().0;
        let strays = mesh.positions.iter().filter(|p| !source.contains(&bits(p))).count();
        assert_eq!(
            strays,
            0,
            "{strays} of the part's {} vertices are in places the source's surface never was, \
             so its apron read OUTSIDE where the source had band values -- that is a seam",
            mesh.positions.len()
        );
    }

    #[test]
    fn the_largest_part_takes_the_source_row_and_the_selection() {
        let (mut doc, id) = document_of(four_loose_parts());
        doc.add_body("Below", Volume::new(VOXEL));
        let at = doc.index_of(id).expect("the source is there");
        let plan = doc.split_plan(id).expect("the body is there");
        let biggest = plan.parts[0].voxels;
        let outcome = doc.split(plan).expect("four parts is a split");

        assert_eq!(doc.index_of(outcome.bodies[0]), Some(at), "the largest part moved row");
        assert_eq!(doc.active(), outcome.bodies[0], "the largest part is not selected");
        assert_eq!(
            solid_voxels(doc.volume(outcome.bodies[0]).expect("a new body")),
            biggest,
            "the row the source held is not the largest part"
        );
        assert!(doc.index_of(id).is_none(), "the source survived its own split");
    }

    #[test]
    fn the_parts_are_named_after_the_body_they_came_from() {
        let (mut doc, id) = document_of(four_loose_parts());
        doc.set_meta(&crate::NodeMeta { name: "Scan".to_string(), ..doc.meta(id).expect("meta") });
        let plan = doc.split_plan(id).expect("the body is there");
        let outcome = doc.split(plan).expect("four parts is a split");
        let names: Vec<&str> = outcome
            .bodies
            .iter()
            .map(|id| doc.node(*id).expect("a new body").name.as_str())
            .collect();
        assert_eq!(names, vec!["Scan 1", "Scan 2", "Scan 3", "Scan 4"]);
    }

    // --- the sweep and the ceilings ------------------------------------------

    #[test]
    fn parts_under_a_cubic_millimetre_are_swept_into_one_fragments_body() {
        // One real ball and three specks. A ball of radius 0.4 mm is 0.27 mm^3,
        // comfortably under the threshold and comfortably over one voxel.
        let volume = balls(&[
            (Vec3::ZERO, 6.0),
            (Vec3::new(9.0, 0.0, 0.0), 0.4),
            (Vec3::new(0.0, 9.0, 0.0), 0.4),
            (Vec3::new(0.0, 0.0, 9.0), 0.4),
        ]);
        let (mut doc, id) = document_of(volume);
        doc.set_meta(&crate::NodeMeta { name: "Scan".to_string(), ..doc.meta(id).expect("meta") });
        let plan = doc.split_plan(id).expect("the body is there");
        assert_eq!(plan.found(), 4, "the fixture is not four parts");
        assert_eq!(plan.kept, 1, "a speck was kept as a body of its own");
        assert_eq!(plan.swept(), 3);
        assert!(!plan.capped, "the ceiling did not do this, the threshold did");
        assert_eq!(plan.bodies(), 2, "one part and one fragments body");

        let outcome = doc.split(plan).expect("four parts is a split");
        assert_eq!(outcome.bodies.len(), 2);
        let fragments = outcome.fragments.expect("three specks make a fragments body");
        assert_eq!(doc.node(fragments).expect("the fragments body").name, "Scan fragments");
        assert_eq!(
            *outcome.bodies.last().expect("two bodies"),
            fragments,
            "the fragments body is not last"
        );
        // Every speck is in it, and nothing else is.
        let specks = solid_voxels(doc.volume(fragments).expect("the fragments body"));
        let kept = solid_voxels(doc.volume(outcome.bodies[0]).expect("the part"));
        assert!(specks > 0, "the fragments body is empty");
        assert!(kept > specks * 10, "the sweep took the ball rather than the specks");
    }

    #[test]
    fn the_rank_cap_leaves_room_for_the_fragments_body_and_never_breaches_the_ceiling() {
        // A document already carrying sixty bodies has room for three more, of
        // which one is reserved for fragments.
        let mut doc = Document::new(VOXEL);
        for _ in 1..60 {
            doc.add_body("Filler", Volume::new(VOXEL));
        }
        assert_eq!(doc.body_count(), 60);
        assert_eq!(doc.rank_cap(), 3, "64 bodies less 60 held less one for fragments");

        let id = doc.add_body("Scan", four_loose_parts());
        assert_eq!(doc.rank_cap(), 2, "the source itself is one of the sixty-one");
        let plan = doc.split_plan(id).expect("the body is there");
        assert_eq!(plan.found(), 4);
        assert_eq!(plan.kept, 2, "the cap did not bite");
        assert!(plan.capped, "the cap bit but did not say so");
        assert_eq!(plan.bodies(), 3);

        doc.split(plan).expect("four parts is a split");
        // Sixty-one in, one out, three in: the transient peak was sixty-four.
        assert_eq!(doc.body_count(), 63);
        assert!(doc.body_count() <= MAX_BODIES);
    }

    /// The three documents nearest the ceiling, one row apart.
    ///
    /// The test above stops at sixty-one bodies, which is the last count that
    /// fits, and a `.max(1)` floor on [`Document::rank_cap`] made the next two
    /// crash: the cap came out zero, was raised to one, and the split inserted a
    /// part and a fragments body against a document with room for at most one of
    /// them. Measured before the floor was dropped -- "a document holds at most
    /// 64 bodies, not 65", out of `Document::insert_body`, on a real gesture. So
    /// the cases worth having are the ones one row past the last one that works.
    #[test]
    fn a_document_with_no_room_for_the_parts_is_refused_rather_than_pushed_past_the_ceiling() {
        for (held, cap) in [(62usize, 1usize), (63, 0), (64, 0)] {
            let mut doc = Document::new(VOXEL);
            for _ in 2..held {
                doc.add_body("Filler", Volume::new(VOXEL));
            }
            let id = doc.add_body("Scan", four_loose_parts());
            assert_eq!(doc.body_count(), held, "the fixture is not {held} bodies");
            assert_eq!(doc.rank_cap(), cap, "the cap at {held} bodies");

            let plan = doc.split_plan(id).expect("the body is there");
            if cap == 0 {
                let why = doc.split_guard(id).expect("a document at the ceiling has to refuse");
                assert!(why.contains("two free rows"), "the refusal does not say why: {why}");
                assert!(doc.split(plan).is_none(), "a document at the ceiling split anyway");
                assert_eq!(doc.body_count(), held, "a refused split changed the document");
            } else {
                assert!(doc.split_guard(id).is_none(), "{held} bodies has room and was refused");
                assert_eq!(plan.kept, cap, "the cap did not bite at {held} bodies");
                assert_eq!(plan.bodies(), cap + 1, "one part and one fragments body");
                doc.split(plan).expect("four parts is a split");
                assert_eq!(doc.body_count(), held + cap, "{held} in, one out, {} in", cap + 1);
            }
            assert!(doc.body_count() <= MAX_BODIES, "the ceiling was breached at {held}");
        }
    }

    #[test]
    fn a_body_too_big_to_hold_twice_is_refused_with_the_size_that_would_fit() {
        // Everything but a gigabyte of the ceiling is already spoken for, and
        // the body wants four.
        let guard = GrowthGuard::of(MAX_VOLUME_BYTES - 1024.0 * 1024.0 * 1024.0, 0.0);
        let wanted = 4.0 * 1024.0 * 1024.0 * 1024.0;
        let why = split_refusal(&guard, wanted, 0.0565).expect("4 GB does not fit in 1 GB");
        assert!(why.contains("GB of memory"), "the memory ceiling is not named: {why}");
        assert!(why.contains("6 GB ceiling"), "the ceiling itself is not in the message: {why}");
        assert!(why.contains("mm first"), "the refusal does not name a voxel size: {why}");
        // A quarter of the cost is half the linear size, so the size named has
        // to be about twice as coarse -- and it must itself be admitted.
        assert!(why.contains("0.11"), "0.0565 mm should double to about 0.113 mm: {why}");
    }

    #[test]
    fn an_ordinary_body_is_not_refused_and_the_walk_runs() {
        let (doc, id) = document_of(four_loose_parts());
        assert!(doc.split_guard(id).is_none(), "a four ball fixture was refused");
        assert_eq!(doc.split_plan(id).expect("the body is there").found(), 4);
    }

    // --- splitting along the mask ---------------------------------------------

    /// A sphere with the whole `x > 0` half protected, feathered across the
    /// middle so the mask is a real soft one rather than a bitmask wearing
    /// eight bits.
    fn half_masked_sphere() -> Volume {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 8.0);
        let lo = IVec3::splat(-40);
        let hi = IVec3::splat(40);
        volume.edit_mask(lo, hi, |_, at, _| {
            let across = (at.x / 2.0).clamp(-1.0, 1.0) * 0.5 + 0.5;
            (across * 255.0).round() as u8
        });
        volume.mark_everything_dirty();
        volume
    }

    #[test]
    fn splitting_along_a_mask_makes_two_bodies_and_consumes_the_source() {
        let (mut doc, id) = document_of(half_masked_sphere());
        assert!(doc.split_masked_guard(id).is_none(), "a half masked sphere was refused");
        let outcome = doc.split_masked(id).expect("a half masked sphere divides");

        assert_eq!(doc.body_count(), 2, "a masked split makes exactly two bodies");
        assert!(doc.index_of(id).is_none(), "the source has to be consumed");
        assert_eq!(doc.active(), outcome.masked, "the half the user chose is left selected");
        assert_eq!(doc.node_count(), 2, "the source row went and two took its place");
    }

    /// The property the whole operation exists for: every voxel of the source
    /// ends up in exactly one of the two halves, and neither half invents any.
    #[test]
    fn every_solid_voxel_of_the_source_lands_in_exactly_one_half() {
        let (mut doc, id) = document_of(half_masked_sphere());
        let solid = |volume: &Volume| {
            let mut count = 0u64;
            for coord in volume.brick_coords() {
                let origin = coord.origin();
                for z in 0..BRICK_DIM as i32 {
                    for y in 0..BRICK_DIM as i32 {
                        for x in 0..BRICK_DIM as i32 {
                            if volume.sample_voxel(origin + IVec3::new(x, y, z)) < 0.0 {
                                count += 1;
                            }
                        }
                    }
                }
            }
            count
        };
        let before = solid(doc.active_volume());
        let outcome = doc.split_masked(id).expect("a half masked sphere divides");
        let masked = solid(doc.volume(outcome.masked).expect("the masked half is there"));
        let rest = solid(doc.volume(outcome.rest).expect("the other half is there"));
        assert!(masked > 0 && rest > 0, "one half came out empty: {masked} and {rest}");
        assert_eq!(masked + rest, before, "voxels were lost or duplicated across the split");
    }

    /// ZBrush's behaviour, and here it is also what keeps resident mask bytes
    /// from doubling at the moment memory is tightest.
    #[test]
    fn both_halves_of_a_masked_split_arrive_unmasked() {
        let (mut doc, id) = document_of(half_masked_sphere());
        let outcome = doc.split_masked(id).expect("a half masked sphere divides");
        for (half, what) in [(outcome.masked, "masked"), (outcome.rest, "rest")] {
            let volume = doc.volume(half).expect("the half is there");
            assert!(volume.mask().is_free(), "the {what} half arrived carrying a mask");
        }
    }

    #[test]
    fn a_masked_split_and_an_undo_restore_the_source_bit_for_bit() {
        let (mut doc, id) = document_of(half_masked_sphere());
        let before = checksum(doc.active_volume());
        let nodes = doc.node_count();

        let outcome = doc.split_masked(id).expect("a half masked sphere divides");
        let mut history = History::new(crate::DEFAULT_HISTORY_BUDGET);
        history.push(outcome.entry);
        let shown = vec![true; doc.node_count()];
        history.undo(&mut doc, &shown);

        assert_eq!(doc.node_count(), nodes, "undo left the outputs behind");
        assert_eq!(checksum(doc.active_volume()), before, "undo did not restore the source");
        assert!(
            !doc.active_volume().mask().is_free(),
            "undo restored the geometry but not the mask that chose the split"
        );
    }

    /// An all-or-nothing mask is a rename with extra steps, and the two O(1)
    /// cases are refused before the walk rather than after it.
    #[test]
    fn a_mask_that_does_not_divide_the_body_is_refused_before_the_walk() {
        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 8.0);
        let (doc, id) = document_of(volume);
        let why = doc.split_masked_guard(id).expect("an unmasked body cannot be split by mask");
        assert!(why.contains("no mask"), "the refusal does not say what is wrong: {why}");

        let mut volume = Volume::new(VOXEL);
        volume.seed_sphere(Vec3::ZERO, 8.0);
        let all = volume.mask().cleared(true);
        volume.replace_mask(all);
        let (doc, id) = document_of(volume);
        let why = doc.split_masked_guard(id).expect("a fully masked body cannot be split by mask");
        assert!(why.contains("covers all"), "the refusal does not say what is wrong: {why}");
    }

    /// What a refusal at a realistic scale SAYS, at numbers a fixture cannot
    /// reach: four gigabytes against one.
    ///
    /// This one calls [`split_refusal`] directly and so pins the wording and
    /// nothing else. The arithmetic that decides WHEN it is reached lives in
    /// `Document::split_masked_guard` and is pinned by the test below, which
    /// goes through the guard.
    #[test]
    fn a_masked_split_too_big_to_hold_the_field_twice_is_refused_with_a_coarser_voxel() {
        let guard = GrowthGuard::of(MAX_VOLUME_BYTES - 1024.0 * 1024.0 * 1024.0, 0.0);
        let field = 2.0 * 1024.0 * 1024.0 * 1024.0;
        let why = split_refusal(&guard, 2.0 * field, 0.0565).expect("4 GB does not fit in 1 GB");
        assert!(why.contains("GB of memory"), "the memory ceiling is not named: {why}");
        assert!(why.contains("mm first"), "the refusal does not name a voxel size: {why}");
    }

    /// The prediction is over the FIELD twice and not over the source twice,
    /// for the reason `split_masked_guard` gives: both halves arrive unmasked.
    ///
    /// **Through the guard, at the boundary, from a real body's own stats**,
    /// which is the only arrangement that can see either half of that sentence.
    /// The budget is straddled a quarter of a field either side of an exact fit,
    /// so the two directions catch the two edits that would hurt: dropping the
    /// `2.0 *` lets the tight case through, and dropping the `- mask_bytes`
    /// refuses the roomy one. A refusal proved by handing `split_refusal` its
    /// own answer proves neither.
    #[test]
    fn a_masked_split_is_refused_exactly_when_twice_the_field_stops_fitting() {
        let (doc, id) = document_of(half_masked_sphere());
        let stats = doc.volume(id).expect("the body is there").stats();
        assert!(
            stats.mask_bytes > 0,
            "the fixture carries no mask, so subtracting one is not being tested"
        );
        let field = stats.resident_bytes.saturating_sub(stats.mask_bytes) as f64;
        let margin = field * 0.25;

        let tight = GrowthGuard::of(MAX_VOLUME_BYTES - 2.0 * field + margin, 0.0);
        let why = doc
            .split_masked_guard_against(id, &tight)
            .expect("twice the field does not fit in 1.75 times it");
        assert!(why.contains("GB of memory"), "the memory ceiling is not named: {why}");
        assert!(why.contains("mm first"), "the refusal does not name a voxel size: {why}");

        let roomy = GrowthGuard::of(MAX_VOLUME_BYTES - 2.0 * field - margin, 0.0);
        assert_eq!(
            doc.split_masked_guard_against(id, &roomy),
            None,
            "twice the field was refused with 2.25 times it free, so the guard is asking for \
             more than the field -- the mask is being counted twice"
        );
    }

    // --- history --------------------------------------------------------------

    #[test]
    fn a_split_and_an_undo_restore_the_source_bit_for_bit() {
        let (mut doc, id) = document_of(four_loose_parts());
        let before = checksum(doc.active_volume());
        let nodes = doc.node_count();

        let plan = doc.split_plan(id).expect("the body is there");
        let outcome = doc.split(plan).expect("four parts is a split");
        assert_eq!(outcome.entry.len(), 5, "four adds and one removal is one entry of five");

        let mut history = History::new(crate::DEFAULT_HISTORY_BUDGET);
        history.push(outcome.entry);
        let shown = vec![true; doc.node_count()];
        history.undo(&mut doc, &shown);

        assert_eq!(doc.node_count(), nodes, "undo did not put the document back");
        assert_eq!(doc.index_of(id), Some(0), "the source did not come back where it was");
        assert_eq!(
            checksum(doc.volume(id).expect("the source is back")),
            before,
            "the source came back changed"
        );

        let shown = vec![true; doc.node_count()];
        history.redo(&mut doc, &shown);
        assert_eq!(doc.body_count(), 4, "redo did not split it again");
        assert!(doc.index_of(id).is_none(), "redo left the source in the document");
    }

    /// The case that breaks the entry order the plan asked for.
    ///
    /// A document holding one body is the ordinary state of an import, and it
    /// is where `[NodeRemoved, NodeAdded x M]` cannot be built or undone: see
    /// [`SplitOutcome::entry`].
    #[test]
    fn splitting_the_only_body_in_a_document_undoes_without_ever_emptying_it() {
        let (mut doc, id) = document_of(two_spheres_sharing_bricks());
        assert_eq!(doc.body_count(), 1, "the fixture is not the single body case");
        let plan = doc.split_plan(id).expect("the body is there");
        let outcome = doc.split(plan).expect("two balls is a split");
        assert_eq!(doc.body_count(), 2);

        let mut history = History::new(crate::DEFAULT_HISTORY_BUDGET);
        history.push(outcome.entry);
        let shown = vec![true; doc.node_count()];
        history.undo(&mut doc, &shown);
        assert_eq!(doc.body_count(), 1);
        assert_eq!(doc.active(), id, "the restored source is not selected");
    }

    #[test]
    fn a_split_into_forty_parts_is_forty_additions_in_one_entry() {
        let mut placed = Vec::new();
        for index in 0..40 {
            placed.push((Vec3::new(index as f32 * 6.0, 0.0, 0.0), 1.0));
        }
        let (mut doc, id) = document_of(balls(&placed));
        let plan = doc.split_plan(id).expect("the body is there");
        assert_eq!(plan.found(), 40, "the fixture is not forty parts");
        assert_eq!(plan.kept, 40, "a ball of 4.2 mm^3 is over the threshold");
        assert!(!plan.has_fragments());

        let outcome = doc.split(plan).expect("forty parts is a split");
        assert_eq!(outcome.bodies.len(), 40);
        assert_eq!(outcome.entry.len(), 41, "forty additions and one removal, in one entry");
        assert_eq!(doc.body_count(), 40);

        // Charged to the reclaim allowance and not to the stroke budget: the
        // only thing this entry holds is the source body, and forty additions
        // are forty pointers.
        assert!(
            outcome.entry.reclaim_bytes() > 0,
            "the consumed body is not charged to the reclaim allowance"
        );
        assert!(
            outcome.entry.stroke_bytes() < 8 * 1024,
            "a split charged {} bytes to the stroke budget",
            outcome.entry.stroke_bytes()
        );

        let mut history = History::new(crate::DEFAULT_HISTORY_BUDGET);
        let entries = history.stats().undo_entries;
        history.push(outcome.entry);
        assert_eq!(history.stats().undo_entries, entries + 1, "a split is not one gesture");
        let shown = vec![true; doc.node_count()];
        history.undo(&mut doc, &shown);
        assert_eq!(doc.body_count(), 1, "one undo did not take all forty apart");
    }

    #[test]
    fn a_body_that_is_one_piece_is_never_split() {
        let (mut doc, id) = document_of(balls(&[(Vec3::ZERO, 6.0)]));
        let plan = doc.split_plan(id).expect("the body is there");
        assert!(plan.is_one_piece());
        assert!(doc.split(plan).is_none(), "a single part was split into itself");
        assert_eq!(doc.body_count(), 1);
    }

    #[test]
    fn a_part_that_sits_inside_a_folder_keeps_its_depth() {
        let (mut doc, id) = document_of(four_loose_parts());
        let (_, _) = doc.group(id, "Group").expect("grouping one body");
        let depth = doc.node(id).expect("the source").depth();
        assert_eq!(depth, 1, "the fixture is not inside a folder");

        let plan = doc.split_plan(id).expect("the body is there");
        let outcome = doc.split(plan).expect("four parts is a split");
        for body in &outcome.bodies {
            assert_eq!(
                doc.node(*body).expect("a new body").depth(),
                1,
                "a part escaped the folder its source was in"
            );
        }
    }
}
