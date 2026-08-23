// SPDX-License-Identifier: AGPL-3.0-only

//! Filling the sealed voids inside an imported model.
//!
//! A mesh from a generator is often a *skin*: a closed surface wrapped around a
//! hollow interior. `Meshy_AI_Neon_Velocity...obj` is a car body about 0.97 mm
//! thick at the size it imports, enclosing 84,800 mm^3 of nothing behind
//! 174,477 mm^2 of surface. A distance field cannot hold a wall that thin --
//! the narrow band is +/- 3 voxels, so the wall's two faces sit inside each
//! other's band, and wherever it dips below a voxel the surface perforates.
//! Measured on that file: 30.6% of the surface lost at a 0.5 mm voxel, 2.5% at
//! 0.25 mm, and 0.125 mm is refused because it would need 3.3 GB.
//!
//! Resolution is not the way out. Making the model solid is: fill the void and
//! only the outer surface -- smooth, and comfortably resolved -- has to be
//! represented at all. It also roughly halves the mesh, because the inner
//! surface stops existing.
//!
//! # Why the obvious flood fill does not work
//!
//! [`crate::voxelise`]'s own module documentation rejects flood fill as a way
//! to decide *sign*, and the reason applies here too: "one hole inverts a whole
//! scan line, or leaks and empties the model". The model that needs filling is
//! precisely the model with holes in it. A flood started outside runs in
//! through a puncture, finds the cavity from the wrong side, and fills nothing.
//!
//! So the flood is **sealed by the narrow band**. It only travels through
//! voxels at least [`SEAL_VOXELS`] outside the surface. A puncture narrower
//! than about two voxels contains no such voxel and is closed to it, while
//! genuinely open space -- the outside of the model, the inside of an open
//! vase -- is full of them. Nothing extra is stored; the band is already paid
//! for.
//!
//! That seal keeps the flood a voxel clear of every surface, including the
//! outer one, so a naive "fill whatever was not reached" would swell the model
//! by a voxel all over. [`DILATE_ROUNDS`] rounds of growth back down to the
//! surface fix that, and the asymmetry is the whole trick: the *outer* skin is
//! within reach of the flood and survives, the *cavity* skin is not and is
//! filled.
//!
//! # What it will not do
//!
//! Best effort, and never worse than not running. A model whose openings are
//! wide enough to admit the sealed flood simply does not get filled, and the
//! import is exactly what it would have been. The report says which happened
//! rather than leaving it to be guessed from the look of the model.
//!
//! **It does not rescue `Meshy_AI_Neon_Velocity...obj`, and that is not a bug
//! in the seal.** That car looked like the case this module was written for --
//! a 0.97 mm skin, 2.4% of its surface lost to perforation -- but its interior
//! turns out to be *open*, through the windows and the underside. Measured by
//! raising the seal as far as the band allows: at 1 voxel it filled 155,321
//! voxels, at 2 it filled 415,845, at 3 -- saturated voxels only, the strongest
//! seal that exists here -- 835,507, and at every setting **not one whole brick
//! was enclosed**. A sealed cavity would have been found at any threshold; a
//! ladder like that is the signature of a void that is genuinely connected to
//! the outside. Those numbers are real fills of real little pockets, and the
//! car is simply a thin open shell, which is a different problem: see the
//! handoff.

use glam::IVec3;

use crate::brick::{BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, OUTSIDE, brick_index};
use crate::volume::Volume;

/// How far outside the surface the flood has to stay, in voxels.
///
/// This is the seal. Below it the flood follows the surface into every
/// sub-voxel puncture; far above it the flood cannot enter a genuinely open but
/// narrow space, and a vase with a slim neck would come out solid. One voxel
/// closes a puncture up to about two voxels across, which is the size the
/// perforation actually takes.
const SEAL_VOXELS: f32 = 1.0;

/// Rounds of growth back toward the surface once the sealed flood has settled.
///
/// Enough to cross the gap the seal left, and no more: each round also reaches
/// this far into a cavity through any puncture, which is a dimple rather than a
/// hole but is still not free.
const DILATE_ROUNDS: usize = 2;

/// The value at which a voxel stops counting as inside the model.
///
/// **Zero, not "greater than zero", and the difference is a whole sheet of
/// surface.** `voxelise` biases an exact zero to the inside as `-0.0` so the
/// raycast and the mesher agree about it, and `-0.0 < 0.0` is false in Rust --
/// so `fast-surface-nets`, which classifies with `d < 0.0`, reads such a voxel
/// as OUTSIDE. Filling only `d > 0.0` therefore leaves every on-surface voxel
/// behind, and a cavity filled that way keeps a one-voxel film of "outside"
/// exactly where its old wall was, which meshes into the inner surface the fill
/// was supposed to delete. Observed: the sealed void filled correctly, the
/// centre read solid, and the triangle count went *up* by 1324.
///
/// Comparing with `>=` against zero is what makes this agree with the mesher,
/// because `-0.0 >= 0.0` is true.
const OUTSIDE_OR_ON_IT: f32 = 0.0;

/// Words in a per-brick voxel bitmap.
const MASK_WORDS: usize = BRICK_VOXELS / 64;

/// Bricks past which the box is refused rather than allocated for.
///
/// Well above anything [`crate::voxelise::preflight`] admits; this is a
/// backstop against arithmetic, not a policy.
const MAX_BRICKS: i64 = 8_000_000;

type Mask = Box<[u64; MASK_WORDS]>;

fn empty_mask() -> Mask {
    // Heap directly rather than via a stack temporary: 32768 bits is only 4 KB,
    // but the same discipline as `Brick::dense_filled` is worth keeping.
    vec![0u64; MASK_WORDS].into_boxed_slice().try_into().expect("length is MASK_WORDS")
}

#[inline]
fn get(mask: &Mask, voxel: usize) -> bool {
    mask[voxel / 64] & (1u64 << (voxel % 64)) != 0
}

#[inline]
fn set(mask: &mut Mask, voxel: usize) {
    mask[voxel / 64] |= 1u64 << (voxel % 64);
}

/// What the flood can do with a brick, decided once from its contents.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Every voxel is passable. An absent brick reads as `OUTSIDE` and is this.
    Open,
    /// No voxel is passable: a solid interior tile.
    Blocked,
    /// Decided voxel by voxel.
    Mixed,
}

const DIRECTIONS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// For each direction, every (voxel on this brick's face, the voxel it touches
/// in the neighbour) pair.
///
/// Built once. Rebuilding it per brick visit was the first version and it
/// dominated the run time.
fn face_tables() -> [Vec<(usize, usize)>; 6] {
    let last = BRICK_DIM - 1;
    std::array::from_fn(|d| {
        let direction = DIRECTIONS[d];
        let mut pairs = Vec::with_capacity(BRICK_DIM * BRICK_DIM);
        for a in 0..BRICK_DIM {
            for b in 0..BRICK_DIM {
                pairs.push(match (direction.x, direction.y, direction.z) {
                    (1, 0, 0) => (brick_index(last, a, b), brick_index(0, a, b)),
                    (-1, 0, 0) => (brick_index(0, a, b), brick_index(last, a, b)),
                    (0, 1, 0) => (brick_index(a, last, b), brick_index(a, 0, b)),
                    (0, -1, 0) => (brick_index(a, 0, b), brick_index(a, last, b)),
                    (0, 0, 1) => (brick_index(a, b, last), brick_index(a, b, 0)),
                    _ => (brick_index(a, b, 0), brick_index(a, b, last)),
                });
            }
        }
        pairs
    })
}

/// The six in-brick neighbours of a voxel, clipped to the brick.
fn neighbours_in_brick(voxel: usize, out: &mut Vec<usize>) {
    let last = BRICK_DIM - 1;
    let x = voxel % BRICK_DIM;
    let y = (voxel / BRICK_DIM) % BRICK_DIM;
    let z = voxel / (BRICK_DIM * BRICK_DIM);
    out.clear();
    if x > 0 {
        out.push(brick_index(x - 1, y, z));
    }
    if x < last {
        out.push(brick_index(x + 1, y, z));
    }
    if y > 0 {
        out.push(brick_index(x, y - 1, z));
    }
    if y < last {
        out.push(brick_index(x, y + 1, z));
    }
    if z > 0 {
        out.push(brick_index(x, y, z - 1));
    }
    if z < last {
        out.push(brick_index(x, y, z + 1));
    }
}

/// The brick box, and what the flood knows about it.
struct Grid {
    lo: IVec3,
    hi: IVec3,
    dims: IVec3,
    kind: Vec<Kind>,
    /// The dense values of every `Mixed` brick, copied out so the volume is not
    /// borrowed while it is being written to.
    values: Vec<Option<Box<[f32; BRICK_VOXELS]>>>,
    reached: Vec<Option<Mask>>,
}

impl Grid {
    #[inline]
    fn at(&self, brick: IVec3) -> usize {
        let d = brick - self.lo;
        (d.z as usize * self.dims.y as usize + d.y as usize) * self.dims.x as usize + d.x as usize
    }

    #[inline]
    fn coord_of(&self, index: usize) -> IVec3 {
        let x = index % self.dims.x as usize;
        let y = (index / self.dims.x as usize) % self.dims.y as usize;
        let z = index / (self.dims.x as usize * self.dims.y as usize);
        self.lo + IVec3::new(x as i32, y as i32, z as i32)
    }

    #[inline]
    fn inside_box(&self, brick: IVec3) -> bool {
        brick.cmpge(self.lo).all() && brick.cmple(self.hi).all()
    }

    /// Whether the flood may occupy a voxel, at a given threshold.
    #[inline]
    fn passable(&self, index: usize, voxel: usize, threshold: f32) -> bool {
        match self.kind[index] {
            Kind::Open => true,
            Kind::Blocked => false,
            Kind::Mixed => self.values[index].as_ref().is_some_and(|data| data[voxel] >= threshold),
        }
    }

    #[inline]
    fn is_reached(&self, index: usize, voxel: usize) -> bool {
        self.reached[index].as_ref().is_some_and(|mask| get(mask, voxel))
    }
}

/// Fill every void the outside cannot reach.
///
/// `b_min` and `b_max` are the brick box the volume was built over. Returns how
/// many voxels were filled: zero when the model is solid already, and zero when
/// the flood leaked, which is the same outcome as not running.
pub(crate) fn fill_sealed_cavities(
    volume: &mut Volume,
    b_min: BrickCoord,
    b_max: BrickCoord,
) -> usize {
    // One brick of margin outside the box the model was built in. That ring is
    // the only trustworthy seed: an absent brick anywhere else is
    // indistinguishable from a cavity, since both read as OUTSIDE.
    let lo = b_min.0 - IVec3::ONE;
    let hi = b_max.0 + IVec3::ONE;
    let dims = hi - lo + IVec3::ONE;
    if dims.cmple(IVec3::ZERO).any() {
        return 0;
    }
    let count = dims.x as i64 * dims.y as i64 * dims.z as i64;
    if count > MAX_BRICKS {
        return 0;
    }
    let count = count as usize;

    let mut grid = Grid {
        lo,
        hi,
        dims,
        kind: vec![Kind::Open; count],
        values: (0..count).map(|_| None).collect(),
        reached: (0..count).map(|_| None).collect(),
    };

    for index in 0..count {
        let coord = BrickCoord(grid.coord_of(index));
        match volume.brick(coord) {
            None => {}
            Some(Brick::Uniform(value)) => {
                grid.kind[index] = if *value >= SEAL_VOXELS { Kind::Open } else { Kind::Blocked };
            }
            Some(Brick::Dense(data)) => {
                grid.kind[index] = Kind::Mixed;
                let copy: Box<[f32]> = data.to_vec().into_boxed_slice();
                grid.values[index] = Some(copy.try_into().expect("length is BRICK_VOXELS"));
            }
        }
    }

    let faces = face_tables();
    let mut queue: Vec<usize> = Vec::new();
    let mut queued = vec![false; count];

    // Seed from the border ring.
    for (index, queued) in queued.iter_mut().enumerate() {
        let c = grid.coord_of(index);
        let border = c.cmpeq(lo).any() || c.cmpeq(hi).any();
        if border && grid.kind[index] == Kind::Open {
            let mut mask = empty_mask();
            for voxel in 0..BRICK_VOXELS {
                set(&mut mask, voxel);
            }
            grid.reached[index] = Some(mask);
            queue.push(index);
            *queued = true;
        }
    }

    flood(&mut grid, &faces, &mut queue, &mut queued, SEAL_VOXELS);

    // Grow back down to the surface, so the outer skin the seal held the flood
    // away from is not mistaken for a cavity.
    for _ in 0..DILATE_ROUNDS {
        dilate(&mut grid, &faces);
    }

    // Anything outside the surface the flood never got to is enclosed.
    let mut filled = 0usize;
    let mut writes: Vec<(BrickCoord, Brick)> = Vec::new();
    for index in 0..count {
        let coord = BrickCoord(grid.coord_of(index));
        match grid.kind[index] {
            Kind::Blocked => {}
            Kind::Open => match &grid.reached[index] {
                // A whole brick of void. It stays a tile, which is what keeps a
                // filled interior from costing 128 KB a brick.
                None => {
                    writes.push((coord, Brick::Uniform(INSIDE)));
                    filled += BRICK_VOXELS;
                }
                Some(mask) => {
                    // Partly reached, which happens only where the dilation ran
                    // out. Filling this all-or-nothing would leave a whole
                    // brick of cavity behind for the sake of two voxels.
                    let unreached = (0..BRICK_VOXELS).filter(|v| !get(mask, *v)).count();
                    if unreached == BRICK_VOXELS {
                        writes.push((coord, Brick::Uniform(INSIDE)));
                        filled += BRICK_VOXELS;
                    } else if unreached > 0 {
                        let mut data = vec![OUTSIDE; BRICK_VOXELS];
                        for (voxel, value) in data.iter_mut().enumerate() {
                            if !get(mask, voxel) {
                                *value = INSIDE;
                            }
                        }
                        let boxed: Box<[f32]> = data.into_boxed_slice();
                        writes.push((
                            coord,
                            Brick::Dense(boxed.try_into().expect("length is BRICK_VOXELS")),
                        ));
                        filled += unreached;
                    }
                }
            },
            Kind::Mixed => {
                let Some(data) = grid.values[index].as_ref() else { continue };
                let mut turned: Option<Vec<f32>> = None;
                for voxel in 0..BRICK_VOXELS {
                    if data[voxel] >= OUTSIDE_OR_ON_IT && !grid.is_reached(index, voxel) {
                        turned.get_or_insert_with(|| data.to_vec())[voxel] = INSIDE;
                        filled += 1;
                    }
                }
                if let Some(turned) = turned {
                    let boxed: Box<[f32]> = turned.into_boxed_slice();
                    let mut brick = Brick::Dense(boxed.try_into().expect("length is BRICK_VOXELS"));
                    if let Some(value) = brick.is_collapsible() {
                        brick = Brick::Uniform(value);
                    }
                    writes.push((coord, brick));
                }
            }
        }
    }

    for (coord, brick) in writes {
        volume.insert_brick(coord, brick);
    }
    filled
}

/// Run the flood to a fixed point.
fn flood(
    grid: &mut Grid,
    faces: &[Vec<(usize, usize)>; 6],
    queue: &mut Vec<usize>,
    queued: &mut [bool],
    threshold: f32,
) {
    let mut seeds: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut neighbours: Vec<usize> = Vec::with_capacity(6);

    while let Some(index) = queue.pop() {
        queued[index] = false;
        let here = grid.coord_of(index);
        for (d, direction) in DIRECTIONS.into_iter().enumerate() {
            let neighbour = here + direction;
            if !grid.inside_box(neighbour) {
                continue;
            }
            let target = grid.at(neighbour);
            if grid.kind[target] == Kind::Blocked {
                continue;
            }

            // A brick with nothing in its way floods whole the moment it is
            // touched at all, so it needs no search: every voxel is passable
            // and therefore every voxel is connected to every other. Sending
            // the outside of a model through a 32768 voxel breadth first search
            // per brick instead is most of what made this pass expensive.
            if grid.kind[target] == Kind::Open {
                if grid.reached[target].is_some() {
                    continue;
                }
                let reachable =
                    faces[d].iter().any(|(here_voxel, _)| grid.is_reached(index, *here_voxel));
                if !reachable {
                    continue;
                }
                let mut mask = empty_mask();
                mask.fill(u64::MAX);
                grid.reached[target] = Some(mask);
                if !queued[target] {
                    queued[target] = true;
                    queue.push(target);
                }
                continue;
            }

            // Mixed, so the values decide. Fetched once rather than through a
            // method per voxel: the inner loop below runs six times per voxel
            // of every dense brick, and that is the hot path of the whole pass.
            let Some(data) = grid.values[target].as_deref() else { continue };

            seeds.clear();
            for (here_voxel, there_voxel) in &faces[d] {
                if grid.is_reached(index, *here_voxel)
                    && data[*there_voxel] >= threshold
                    && !grid.is_reached(target, *there_voxel)
                {
                    seeds.push(*there_voxel);
                }
            }
            if seeds.is_empty() {
                continue;
            }

            let mut mask = grid.reached[target].take().unwrap_or_else(empty_mask);
            stack.clear();
            for seed in &seeds {
                set(&mut mask, *seed);
                stack.push(*seed);
            }
            while let Some(voxel) = stack.pop() {
                neighbours_in_brick(voxel, &mut neighbours);
                for other in &neighbours {
                    if data[*other] >= threshold && !get(&mask, *other) {
                        set(&mut mask, *other);
                        stack.push(*other);
                    }
                }
            }
            grid.reached[target] = Some(mask);

            if !queued[target] {
                queued[target] = true;
                queue.push(target);
            }
        }
    }
}

/// One round of growing the reached set by a single voxel, through anything
/// outside the surface rather than only through the sealed interior.
///
/// Reads a snapshot so the round grows by exactly one voxel everywhere rather
/// than racing across a brick in whichever order the loop happened to run.
fn dilate(grid: &mut Grid, faces: &[Vec<(usize, usize)>; 6]) {
    let previous: Vec<Option<Mask>> = grid.reached.clone();
    let mut neighbours: Vec<usize> = Vec::with_capacity(6);

    for index in 0..grid.kind.len() {
        if grid.kind[index] == Kind::Blocked {
            continue;
        }
        let here = grid.coord_of(index);

        // Most bricks of a real model have nothing to do here, and finding that
        // out by testing all 32768 bits is what made importing a 542k triangle
        // sphere at a 0.125 mm voxel take 1.25 s instead of 0.21 s. Two cheap
        // rejections cover almost all of them.
        let full = |mask: &Option<Mask>| {
            mask.as_ref().is_some_and(|bits| bits.iter().all(|word| *word == u64::MAX))
        };
        if full(&previous[index]) {
            // Nothing left to grow into.
            continue;
        }
        let touches_reached = previous[index].is_some()
            || DIRECTIONS.into_iter().any(|direction| {
                let neighbour = here + direction;
                grid.inside_box(neighbour) && previous[grid.at(neighbour)].is_some()
            });
        if !touches_reached {
            // Nothing to grow from, here or next door.
            continue;
        }

        let was_reached =
            |i: usize, voxel: usize| previous[i].as_ref().is_some_and(|mask| get(mask, voxel));

        let mut mask = grid.reached[index].take().unwrap_or_else(empty_mask);
        let mut changed = false;

        for voxel in 0..BRICK_VOXELS {
            if was_reached(index, voxel) || !grid.passable(index, voxel, OUTSIDE_OR_ON_IT) {
                continue;
            }
            neighbours_in_brick(voxel, &mut neighbours);
            let touching = neighbours.iter().any(|other| was_reached(index, *other));
            if touching {
                set(&mut mask, voxel);
                changed = true;
            }
        }

        for (d, direction) in DIRECTIONS.into_iter().enumerate() {
            let neighbour = here + direction;
            if !grid.inside_box(neighbour) {
                continue;
            }
            let source = grid.at(neighbour);
            for (here_voxel, there_voxel) in &faces[d] {
                if !was_reached(index, *here_voxel)
                    && grid.passable(index, *here_voxel, OUTSIDE_OR_ON_IT)
                    && was_reached(source, *there_voxel)
                {
                    set(&mut mask, *here_voxel);
                    changed = true;
                }
            }
        }

        if changed || previous[index].is_some() {
            grid.reached[index] = Some(mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::ExportMesh;
    use crate::voxelise::{VoxeliseOptions, voxelise};
    use glam::Vec3;

    /// An axis aligned box as a closed triangle soup, wound outward.
    fn box_mesh(min: Vec3, max: Vec3, inward: bool) -> ExportMesh {
        let c = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        // Each face as two triangles, wound counter clockwise seen from outside.
        let faces: [[usize; 3]; 12] = [
            [0, 2, 1],
            [0, 3, 2], // -z
            [4, 5, 6],
            [4, 6, 7], // +z
            [0, 1, 5],
            [0, 5, 4], // -y
            [3, 7, 6],
            [3, 6, 2], // +y
            [0, 4, 7],
            [0, 7, 3], // -x
            [1, 2, 6],
            [1, 6, 5], // +x
        ];
        let triangles = faces
            .into_iter()
            .map(|f| {
                // An inner shell has to face the other way, or the winding
                // number would count it as more solid rather than as a void.
                if inward {
                    [f[0] as u32, f[2] as u32, f[1] as u32]
                } else {
                    [f[0] as u32, f[1] as u32, f[2] as u32]
                }
            })
            .collect();
        ExportMesh { positions: c.to_vec(), normals: Vec::new(), triangles, slots: Vec::new() }
    }

    fn join(mut a: ExportMesh, b: ExportMesh) -> ExportMesh {
        let offset = a.positions.len() as u32;
        a.positions.extend(b.positions);
        a.triangles.extend(b.triangles.into_iter().map(|t| t.map(|i| i + offset)));
        a
    }

    /// A box with a smaller box inside it, wound so the inner one is a void.
    ///
    /// Big enough that the void spans whole bricks: at a 0.5 mm voxel a 60 mm
    /// void is 120 voxels across, so several 32-voxel bricks fall entirely
    /// inside it and the cheap tile path is exercised rather than only the
    /// per-voxel one.
    fn hollow() -> ExportMesh {
        join(
            box_mesh(Vec3::splat(-45.0), Vec3::splat(45.0), false),
            box_mesh(Vec3::splat(-30.0), Vec3::splat(30.0), true),
        )
    }

    fn options(fill: bool) -> VoxeliseOptions {
        VoxeliseOptions {
            voxel_size: 0.5,
            centre: false,
            refit_if_implausible: false,
            fill_sealed_cavities: fill,
            repair_broken_scan_lines: true,
            coarsen_to_fit: false,
            refine_to_resolve: false,
        }
    }

    #[test]
    fn a_sealed_void_is_filled_and_becomes_solid_tiles() {
        let mesh = hollow();
        let (hollow_volume, hollow_report) =
            voxelise(&mesh, &options(false)).expect("it should voxelise");
        let (filled_volume, filled_report) =
            voxelise(&mesh, &options(true)).expect("it should voxelise");

        assert_eq!(hollow_report.filled_voxels, 0, "nothing should be filled with the pass off");
        assert!(
            filled_report.filled_voxels > 1_000_000,
            "a 60 mm cube of void at a 0.5 mm voxel is over a million voxels, filled {}",
            filled_report.filled_voxels
        );

        // The void is gone: a point at the centre now reads as solid.
        assert!(
            hollow_volume.sample_world(Vec3::ZERO) > 0.0,
            "the fixture is not hollow, so the test proves nothing"
        );
        assert!(
            filled_volume.sample_world(Vec3::ZERO) < 0.0,
            "the centre of the void did not become solid"
        );

        // And it is held as tiles rather than as 128 KB a brick.
        assert!(
            filled_volume.stats().uniform_bricks > hollow_volume.stats().uniform_bricks,
            "the filled interior did not collapse to tiles: {} -> {}",
            hollow_volume.stats().uniform_bricks,
            filled_volume.stats().uniform_bricks
        );

        // The inner surface stops existing, which is most of the point.
        let (before, _) = hollow_volume.export_mesh();
        let (after, _) = filled_volume.export_mesh();
        assert!(
            after.triangles.len() < before.triangles.len(),
            "filling did not remove the inner surface: {} -> {}",
            before.triangles.len(),
            after.triangles.len()
        );
    }

    #[test]
    fn open_space_between_two_parts_is_not_mistaken_for_a_cavity() {
        // The control, and the property that matters on every ordinary model:
        // the gap between two legs, two wings, or two parts on a plate is open
        // to the world and must stay open. Fill it and every model with a gap
        // in it becomes a solid lump.
        let mesh = join(
            box_mesh(Vec3::new(-30.0, -20.0, -20.0), Vec3::new(-10.0, 20.0, 20.0), false),
            box_mesh(Vec3::new(10.0, -20.0, -20.0), Vec3::new(30.0, 20.0, 20.0), false),
        );
        let (volume, report) = voxelise(&mesh, &options(true)).expect("it should voxelise");

        assert_eq!(report.filled_voxels, 0, "the gap between two parts was filled in");
        assert!(
            volume.sample_world(Vec3::ZERO) > 0.0,
            "the space between the two parts became solid"
        );
        // Both parts are still there.
        assert!(volume.sample_world(Vec3::new(-20.0, 0.0, 0.0)) < 0.0);
        assert!(volume.sample_world(Vec3::new(20.0, 0.0, 0.0)) < 0.0);
    }

    #[test]
    fn a_deep_pocket_open_at_one_end_stays_open() {
        // A cavity that is reachable, but only down a channel. This is the
        // shape of an open container, and the seal is the thing most likely to
        // get it wrong -- too strong a seal and the flood cannot get down the
        // channel, so a cup silently imports as a solid block.
        //
        // Built as five overlapping slabs rather than as a box inside a box: an
        // inner shell poking out through the outer one leaves the region beyond
        // the outer wall with a winding number of -1, which the nonzero rule
        // reads as SOLID, so the "opening" seals itself and the fixture proves
        // the opposite of what it claims. Overlapping same-wound boxes union
        // cleanly under the same rule.
        let walls = [
            // back wall, and the four sides, leaving +x open.
            (Vec3::new(-45.0, -45.0, -45.0), Vec3::new(-35.0, 45.0, 45.0)),
            (Vec3::new(-45.0, 35.0, -45.0), Vec3::new(45.0, 45.0, 45.0)),
            (Vec3::new(-45.0, -45.0, -45.0), Vec3::new(45.0, -35.0, 45.0)),
            (Vec3::new(-45.0, -45.0, 35.0), Vec3::new(45.0, 45.0, 45.0)),
            (Vec3::new(-45.0, -45.0, -45.0), Vec3::new(45.0, 45.0, -35.0)),
        ];
        let mesh = walls
            .into_iter()
            .map(|(min, max)| box_mesh(min, max, false))
            .reduce(join)
            .expect("five walls");
        let (volume, report) = voxelise(&mesh, &options(true)).expect("it should voxelise");

        assert!(
            volume.sample_world(Vec3::new(-30.0, 0.0, 0.0)) > 0.0,
            "the bottom of the pocket became solid, so a cup would import as a block"
        );
        // Not exactly zero, and the reason is worth naming rather than hiding
        // behind a loose bound: in the crease where two walls meet, the distance
        // never reaches `SEAL_VOXELS`, so the sealed flood cannot get in, and
        // the growth rounds only reach `DILATE_ROUNDS` voxels. A voxel or two
        // right in the corner is therefore left unreached and filled. That is a
        // filled crease, not a filled pocket, and it is invisible at any voxel
        // size -- but if this number starts climbing, the seal has begun closing
        // openings it should be walking through.
        assert!(
            report.filled_voxels < 100,
            "an open pocket was filled in: {} voxels",
            report.filled_voxels
        );
        // And the walls are still there, so the fixture is a pocket and not
        // just empty space.
        assert!(volume.sample_world(Vec3::new(-40.0, 0.0, 0.0)) < 0.0, "the back wall is missing");
        assert!(volume.sample_world(Vec3::new(0.0, 40.0, 0.0)) < 0.0, "a side wall is missing");
    }

    #[test]
    fn a_solid_model_is_untouched() {
        // The control that matters most for every ordinary import: a model with
        // no void must come through the pass bit for bit.
        let mesh = box_mesh(Vec3::splat(-20.0), Vec3::splat(20.0), false);
        let (plain, plain_report) = voxelise(&mesh, &options(false)).expect("it should voxelise");
        let (passed, passed_report) = voxelise(&mesh, &options(true)).expect("it should voxelise");

        assert_eq!(passed_report.filled_voxels, 0, "a solid model had voxels filled");
        assert_eq!(plain_report.dense_bricks, passed_report.dense_bricks);
        assert_eq!(plain_report.uniform_bricks, passed_report.uniform_bricks);

        let mut plain_coords: Vec<_> = plain.brick_coords().collect();
        let mut passed_coords: Vec<_> = passed.brick_coords().collect();
        plain_coords.sort_unstable();
        passed_coords.sort_unstable();
        assert_eq!(plain_coords, passed_coords, "the pass changed which bricks exist");
        for coord in plain_coords {
            let a = plain.brick(coord).expect("came from the map");
            let b = passed.brick(coord).expect("same coordinate");
            for voxel in 0..BRICK_VOXELS {
                let (x, y, z) = (
                    voxel % BRICK_DIM,
                    (voxel / BRICK_DIM) % BRICK_DIM,
                    voxel / (BRICK_DIM * BRICK_DIM),
                );
                assert_eq!(
                    a.get(x, y, z),
                    b.get(x, y, z),
                    "{coord:?} voxel {voxel} changed under a pass that should be a no-op"
                );
            }
        }
    }

    #[test]
    fn filling_does_not_move_the_outer_surface() {
        // The seal holds the flood a voxel clear of every surface, including the
        // outer one, so without the growth rounds this pass would swell the
        // model all over. Measure the outside, not the inside.
        let mesh = hollow();
        let (volume, report) = voxelise(&mesh, &options(true)).expect("it should voxelise");
        assert!(report.filled_voxels > 0, "nothing was filled, so this proves nothing");

        // Probe either side of the known face at x = 45.
        assert!(
            volume.sample_world(Vec3::new(44.0, 0.0, 0.0)) < 0.0,
            "just inside the outer wall is not solid"
        );
        assert!(
            volume.sample_world(Vec3::new(46.0, 0.0, 0.0)) > 0.0,
            "the outer surface grew outward when the cavity was filled"
        );
    }
}
