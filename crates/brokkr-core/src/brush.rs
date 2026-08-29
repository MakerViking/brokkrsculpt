// SPDX-License-Identifier: AGPL-3.0-only

//! Brushes: falloff weighted modification of the distance field.
//!
//! Every brush reads from a snapshot of the affected box taken before any
//! writing starts, so each voxel sees the same starting field and the result
//! does not depend on which voxel happened to be visited first. See
//! [`crate::region`].
//!
//! # Working in an implicit field
//!
//! A mesh sculptor moves vertices. There are no vertices here, so each brush is
//! whatever operation on the distance field produces the same visible effect.
//! They fall into two groups.
//!
//! Three of them blend the value toward a target, which is inherently stable
//! because the target is itself a legal distance:
//!
//! - Smooth blends toward the average of the neighbouring values.
//! - Flatten blends toward the tangent plane under the cursor.
//! - Clay blends toward a plane held slightly outside the surface, kept to the
//!   direction that adds material, which is how clay is actually built up.
//!
//! Four of them move material, and those have to be handled with more care.
//! Inflate offsets the whole level set, which moves every point of the surface
//! along its own normal, and is the natural operation on a distance field.
//! Draw, pinch and move instead resample the field from a shifted position:
//! draw reads from behind along the stroke normal, which slides the patch
//! outward, pinch reads from slightly nearer the brush axis, which squeezes a
//! ridge into a crease, and move reads from behind along the drag, which pulls
//! the patch after the pointer.
//!
//! Move is the odd one out in a second way: it is the only brush whose unit of
//! work is a whole gesture rather than a stamp. See [`MoveStroke`].
//!
//! Draw and pinch were both first written the obvious way, as a value the
//! brush adds or amplifies, and both had to be rewritten. Anything that
//! multiplies a displacement by the local gradient, or that amplifies the
//! difference from a local average, has gain above one somewhere and turns its
//! own rounding error into visible crust over the course of a stroke. Warping
//! where the field is read from cannot introduce detail that was not there.
//! Move was written that way from the start for the same reason.
//!
//! # Reading outside the box being written
//!
//! The brushes that warp the domain read the field from somewhere other than
//! the voxel they are writing, so the box that is snapshotted is not the box
//! that is edited. [`Brush::read_reach`] is how far past the radius the reads
//! can go, and it is what separates the two boxes.
//!
//! Getting that wrong fails silently. [`FieldRegion::get`] clamps a read
//! outside the stored box to its edge rather than panicking, so a read that
//! overshoots smears the rim value across the brush instead of crashing, and
//! every value it produces is still legally inside the narrow band. Nothing
//! but looking at the model catches it.
//!
//! None of these preserve the eikonal property: after many overlapping stamps
//! the gradient magnitude drifts from 1 and the surface moves slightly less per
//! stamp than the nominal displacement. Clamping to the narrow band bounds the
//! drift, and [`MAX_STAMP_VOXELS`] keeps any single stamp small enough that the
//! field stays well formed. A renormalisation pass belongs with the GPU
//! rewrite, not here.

use glam::{IVec3, Vec3};

use crate::brick::{BRICK_DIM, BrickCoord, INSIDE, OUTSIDE};
use crate::mask::{PROTECTED, UNMASKED};
use crate::pattern::{Pattern, Prepared};
use crate::region::FieldRegion;
use crate::volume::{BrickPreview, BrickVerdict, Volume};

/// How far outside the surface the clay plane sits, as a fraction of radius.
///
/// Zero would make clay identical to flatten. Too large and it stops following
/// the form and just deposits a slab.
const CLAY_OFFSET: f32 = 0.35;

/// Most a single stamp may move the surface, in voxels.
///
/// This one is load bearing. The field only carries a real distance inside the
/// narrow band, so a stamp that displaces further than about a voxel saturates
/// everything it touches, and the next stamp has no usable field left to read.
/// The result is a crust of aliased spikes rather than a smooth push, and every
/// value in it is still legally inside the band, so nothing but looking at it
/// catches the problem.
///
/// Strokes get their reach from laying down many small stamps, not from one
/// big one, which is also what makes stroke interpolation worth having.
const MAX_STAMP_VOXELS: f32 = 1.0;

/// Displacement per stamp as a fraction of the brush radius, before the cap.
///
/// Only small brushes ever fall below the cap, which is the point: a brush one
/// voxel across should not shove the surface a whole voxel per stamp.
const STAMP_FRACTION_OF_RADIUS: f32 = 0.25;

/// How far pinch drags the field toward the brush axis per stamp, in voxels.
///
/// Kept under one voxel so the warped read stays inside the snapshot's padding.
const PINCH_PULL_VOXELS: f32 = 0.85;

/// Fraction of the fold threshold a whole Move gesture is allowed to use.
///
/// A domain warp folds the field back through itself once the displacement
/// changes faster than the distance it is spread over, which happens at
/// `radius / falloff.max_slope()` exactly. Backing off to three quarters of
/// that keeps the warp comfortably invertible for every falloff curve, and
/// buys a property the reads depend on: the source of every voxel a Move
/// touches stays inside the brush's own box, so the box that is read and the
/// box that is written are the same one. See [`MoveStroke`].
const MOVE_DRAG_MARGIN: f32 = 0.75;

/// How far the drag has to change before a Move gesture redoes its warp, in
/// voxels.
///
/// The warp is recomputed from the locked field every pointer event, so an
/// event that has not moved the pointer by a visible amount would repeat a
/// whole pass over the brush box for an identical result. Below a quarter of a
/// voxel the answer is the same to within the interpolation, so it is skipped.
///
/// This is also what makes dragging on past the cap free: once the
/// displacement has clamped, further motion the same way leaves it unchanged.
const MOVE_SETTLE_VOXELS: f32 = 0.25;

/// Rotate a surface normal by a lean, giving the direction a tilted stylus is
/// pushing in.
///
/// `lean` is a world space vector whose length is the tilt angle in radians and
/// whose direction is the way the pen is leaning. Only the part of it lying in
/// the surface's tangent plane can steer anything: a lean straight into or out
/// of the surface would just be asking the brush to push harder, which is what
/// pressure is for.
///
/// Every brush reads the stamp normal, so tilting it steers all of them at
/// once. Draw pushes clay sideways, the clay and flatten planes tip over so a
/// surface can be flattened at an angle, and pinch's axis leans with the pen.
///
/// An upright pen returns the normal unchanged, so nothing about the existing
/// behaviour depends on a tablet being present.
pub fn lean_normal(normal: Vec3, lean: Vec3) -> Vec3 {
    let angle = lean.length();
    if angle < 1.0e-5 {
        return normal;
    }
    let Some(direction) = lean.try_normalize() else {
        return normal;
    };
    let Some(tangential) = (direction - normal * direction.dot(normal)).try_normalize() else {
        // Leaning exactly along the normal steers nothing.
        return normal;
    };
    (normal * angle.cos() + tangential * angle.sin()).normalize_or(normal)
}

/// Which way a brush moves the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushDirection {
    /// Add clay: move the surface out along its normal.
    Add,
    /// Remove clay: move the surface in.
    Subtract,
}

impl BrushDirection {
    /// Sign to apply to a distance value to grow or shrink the solid.
    ///
    /// Distance is negative inside, so making a value more negative grows it.
    #[inline]
    pub fn field_sign(self) -> f32 {
        match self {
            BrushDirection::Add => -1.0,
            BrushDirection::Subtract => 1.0,
        }
    }

    /// Sign along the surface normal: outward when adding.
    #[inline]
    pub fn outward_sign(self) -> f32 {
        match self {
            BrushDirection::Add => 1.0,
            BrushDirection::Subtract => -1.0,
        }
    }

    #[inline]
    pub fn inverted(self) -> Self {
        match self {
            BrushDirection::Add => BrushDirection::Subtract,
            BrushDirection::Subtract => BrushDirection::Add,
        }
    }
}

/// Shape of the brush's radial weighting.
///
/// Every curve is 1 at the centre, 0 at the rim and monotonic in between, so
/// swapping between them changes the shape of a stroke without changing how far
/// it reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FalloffCurve {
    /// Smoothstep. Zero slope at both ends, so repeated strokes leave no ring
    /// at the brush edge. The default for good reason.
    #[default]
    Smooth,
    /// Straight line. Predictable, with a faint edge where the slope stops.
    Linear,
    /// Concentrated at the centre, for detail work and small features.
    Sharp,
    /// Broad and flat with a quick roll off at the rim, for filling areas.
    Wide,
}

impl FalloffCurve {
    /// Weight at a distance from the centre, given as a fraction of the radius.
    #[inline]
    pub fn weight(self, normalised_distance: f32) -> f32 {
        let t = (1.0 - normalised_distance).clamp(0.0, 1.0);
        match self {
            FalloffCurve::Smooth => t * t * (3.0 - 2.0 * t),
            FalloffCurve::Linear => t,
            FalloffCurve::Sharp => t * t * t,
            FalloffCurve::Wide => {
                let inverse = 1.0 - t;
                1.0 - inverse * inverse * inverse
            }
        }
    }

    pub const ALL: [FalloffCurve; 4] =
        [FalloffCurve::Smooth, FalloffCurve::Linear, FalloffCurve::Sharp, FalloffCurve::Wide];

    /// Steepest the weight ever changes, per unit of normalised distance.
    ///
    /// Read off the derivative of each curve rather than measured: smoothstep
    /// peaks at three halves in the middle, a line is 1 the whole way, and both
    /// cubics reach 3 at the end where they are steepest.
    ///
    /// Only [`MoveStroke`] needs it, and it needs it because a displacement
    /// spread over a falloff steeper than this folds the field through itself.
    #[inline]
    pub fn max_slope(self) -> f32 {
        match self {
            FalloffCurve::Smooth => 1.5,
            FalloffCurve::Linear => 1.0,
            FalloffCurve::Sharp | FalloffCurve::Wide => 3.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            FalloffCurve::Smooth => "Smooth",
            FalloffCurve::Linear => "Linear",
            FalloffCurve::Sharp => "Sharp",
            FalloffCurve::Wide => "Wide",
        }
    }
}

impl std::fmt::Display for FalloffCurve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Which operation a brush performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrushKind {
    /// Push the surface out along the stroke direction.
    #[default]
    Draw,
    /// Build material up toward a plane held just outside the surface.
    Clay,
    /// Blend toward the average of the surrounding field.
    Smooth,
    /// Offset the whole level set, moving every point along its own normal.
    Inflate,
    /// Squeeze the surface toward the brush axis, sharpening ridges into
    /// creases.
    Pinch,
    /// Blend toward the tangent plane under the cursor.
    Flatten,
    /// Drag the surface along with the pointer.
    ///
    /// The falloff region follows the drag: the field is resampled from behind
    /// along the direction of travel, so whatever was under the cursor arrives
    /// where the cursor now is, tapering to nothing at the rim.
    ///
    /// Unlike every other brush, one Move is a whole gesture rather than a
    /// stamp. It locks the field at the moment the button goes down and warps
    /// that locked copy by the TOTAL drag on every pointer event, which is what
    /// Nomad and ZBrush do with a locked vertex selection. See [`MoveStroke`]
    /// for why, and for what that costs.
    ///
    /// Dragging out and back therefore returns the form, near enough exactly:
    /// the second half of the gesture is not undoing the first, it is warping
    /// the same locked field by a drag that has shrunk back to nothing.
    ///
    /// How far one gesture can carry the surface is bounded -- see
    /// [`Brush::max_drag`]. Past that the surface stops following rather than
    /// tearing, and the way to move something further is a second gesture.
    ///
    /// It drags along the surface rather than through it, because the pointer
    /// direction comes from where the cursor meets the model. Pulling a form
    /// out toward the camera is draw's job, not this one's.
    Move,
}

impl BrushKind {
    pub const ALL: [BrushKind; 7] = [
        BrushKind::Draw,
        BrushKind::Clay,
        BrushKind::Smooth,
        BrushKind::Inflate,
        BrushKind::Pinch,
        BrushKind::Flatten,
        BrushKind::Move,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BrushKind::Draw => "Draw",
            BrushKind::Clay => "Clay",
            BrushKind::Smooth => "Smooth",
            BrushKind::Inflate => "Inflate",
            BrushKind::Pinch => "Pinch",
            BrushKind::Flatten => "Flatten",
            BrushKind::Move => "Move",
        }
    }

    /// What strength this brush wants when it is first selected.
    ///
    /// One number for every brush was wrong in one specific way: for Move,
    /// strength is the FRACTION of the drag the surface follows, so the 0.15
    /// that suits Draw means the form crawls at a seventh of the pointer and
    /// reads as the tool barely working. A grab should follow the hand.
    pub fn default_strength(self) -> f32 {
        match self {
            // Follows the cursor essentially one to one.
            BrushKind::Move => 1.0,
            // Smoothing at full strength erases detail in one pass.
            BrushKind::Smooth => 0.4,
            _ => 0.15,
        }
    }

    /// Whether inverting the stroke means anything for this brush.
    ///
    /// Smooth and flatten are their own opposite: there is no such thing as
    /// unsmoothing toward a plane. Move already takes its direction from the
    /// pointer, so inverting it would drag the surface the opposite way from
    /// the hand doing the dragging, which is not an operation anybody wants.
    /// Holding the invert key with any of the three selected does nothing,
    /// which is worth saying in the interface rather than leaving the user to
    /// wonder.
    pub fn is_directional(self) -> bool {
        !matches!(self, BrushKind::Smooth | BrushKind::Flatten | BrushKind::Move)
    }
}

impl std::fmt::Display for BrushKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// One of the three world planes a stamp can be mirrored across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorAxis {
    /// Mirror across the plane x = 0.
    X,
    /// Mirror across the plane y = 0.
    Y,
    /// Mirror across the plane z = 0.
    Z,
}

impl MirrorAxis {
    pub const ALL: [MirrorAxis; 3] = [MirrorAxis::X, MirrorAxis::Y, MirrorAxis::Z];

    pub fn label(self) -> &'static str {
        match self {
            MirrorAxis::X => "X",
            MirrorAxis::Y => "Y",
            MirrorAxis::Z => "Z",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Which planes every stamp is mirrored across.
///
/// A set rather than a single choice, because the combinations are what make
/// it useful: x and y together give the four way symmetry a face or a wheel
/// wants, and all three give eight way.
///
/// # The centre is a parameter and not a field, and that is deliberate
///
/// [`Symmetry::mirrors`] and [`Symmetry::flips`] both take a `centre: Vec3`
/// rather than reading one from here, so the mirror plane is a property of the
/// call and not of the switch. **The value passed today is always the lattice
/// origin**, which is what this type meant before the parameter existed and
/// what the interface has always drawn.
///
/// The reason it is not a field is that the axis and the centre must have the
/// same scope. This set is global -- one switch for the whole document -- and a
/// global switch whose plane came from the selection would make "X on" a
/// different physical plane depending on which row is highlighted, with the
/// only evidence on screen a translucent patch. The day the axis becomes
/// per-body, the centre moves with it, and the parameter is what makes that a
/// change to the callers rather than to every mirroring formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Symmetry {
    enabled: [bool; 3],
}

/// One mirrored copy: where a point lands, and which way a direction turns.
///
/// A reflection across a plane through `centre` is `centre + (at - centre) *
/// sign`, which is `at * sign + offset` with `offset = centre * (1 - sign)`.
/// Splitting it that way is the whole reason this is a type rather than a bare
/// sign: a **position** needs the offset and a **direction** -- a surface
/// normal, a stroke tangent, a Move drag vector -- must not have it. Reflecting
/// a direction through [`Flip::point`] would be silently correct while the
/// centre is the lattice origin and would send every mirrored stroke across the
/// model the day it is not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Flip {
    /// Componentwise sign, `-1` on each mirrored axis. This alone reflects a
    /// direction.
    pub sign: Vec3,
    /// What a reflected position picks up from the centre. `ZERO` while the
    /// centre is the lattice origin, which it is at every call site today.
    pub offset: Vec3,
}

impl Flip {
    /// The copy that is not a copy: used to fill the caller's array before
    /// [`Symmetry::flips`] says how much of it is real.
    pub const IDENTITY: Flip = Flip { sign: Vec3::ONE, offset: Vec3::ZERO };

    /// Where a world position lands on the other side of the plane.
    #[inline]
    pub fn point(self, at: Vec3) -> Vec3 {
        at * self.sign + self.offset
    }
}

impl Symmetry {
    /// No mirroring at all.
    pub const OFF: Symmetry = Symmetry { enabled: [false; 3] };

    /// Mirroring across x = 0 only, which is the common case and the one the
    /// application starts every other feature's tests from.
    pub const X: Symmetry = Symmetry { enabled: [true, false, false] };

    /// Most twins a single stamp can have: one per non-empty combination of
    /// three planes, so `2^3 - 1`.
    pub const MAX_MIRRORS: usize = 7;

    pub fn is_off(self) -> bool {
        self.enabled.iter().all(|on| !on)
    }

    pub fn axis(self, axis: MirrorAxis) -> bool {
        self.enabled[axis.index()]
    }

    pub fn with_axis(mut self, axis: MirrorAxis, on: bool) -> Self {
        self.enabled[axis.index()] = on;
        self
    }

    pub fn toggled(self, axis: MirrorAxis) -> Self {
        self.with_axis(axis, !self.axis(axis))
    }

    /// Write every mirrored twin as a [`Flip`] about `centre`, returning how
    /// many there are. Never includes the identity.
    ///
    /// A mirror is a reflection, so it acts on a position, a normal and a
    /// direction of travel alike -- but only the sign is shared between those
    /// three, which is why this hands back a [`Flip`] and not a bare vector.
    pub(crate) fn flips(self, centre: Vec3, out: &mut [Flip; Self::MAX_MIRRORS]) -> usize {
        let mut count = 0;
        // Each combination is a bit per axis; 0 is the original, which is the
        // caller's to apply and not a twin.
        for combination in 1..=Self::MAX_MIRRORS {
            let flips = |index: usize| combination & (1 << index) != 0;
            if (0..3).any(|index| flips(index) && !self.enabled[index]) {
                continue;
            }
            let mut sign = Vec3::ONE;
            for index in 0..3 {
                if flips(index) {
                    sign[index] = -1.0;
                }
            }
            out[count] = Flip { sign, offset: centre * (Vec3::ONE - sign) };
            count += 1;
        }
        count
    }

    /// Write every mirrored twin of a stamp into `out`, returning how many
    /// there are. Never includes the stamp itself.
    ///
    /// `centre` is the point every enabled plane passes through; see the type's
    /// documentation for why it is a parameter and why every caller passes the
    /// lattice origin today.
    ///
    /// Fills a caller owned array rather than returning a `Vec`, because this
    /// runs once per stamp and a stroke lays down thousands: allocating here
    /// would put the sculpt loop back in the allocator.
    ///
    /// A stamp landing on a mirror plane is applied twice at nearly the same
    /// place. That is deliberate: the two falloffs overlap smoothly, whereas
    /// suppressing the twin near the plane would put a visible step in the
    /// stroke strength exactly where the user is trying to work.
    pub fn mirrors(
        self,
        stamp: &Stamp,
        centre: Vec3,
        out: &mut [Stamp; Self::MAX_MIRRORS],
    ) -> usize {
        let mut flips = [Flip::IDENTITY; Self::MAX_MIRRORS];
        let count = self.flips(centre, &mut flips);
        for (twin, flip) in out.iter_mut().zip(&flips[..count]) {
            *twin = *stamp;
            // Reflecting across a plane moves the position to the other side of
            // it and negates the component of the surface normal and of the
            // direction the stroke is travelling -- a twin that kept the
            // original tangent would comb its pattern, and drag its material,
            // the wrong way round. The normal and the tangent take the sign
            // alone, because a direction has no position to reflect about.
            twin.centre = flip.point(stamp.centre);
            twin.normal *= flip.sign;
            twin.tangent *= flip.sign;
        }
        count
    }

    /// The enabled planes as "Off", "X", or "XZ".
    pub fn label(self) -> String {
        if self.is_off() {
            return "Off".to_string();
        }
        MirrorAxis::ALL
            .into_iter()
            .filter(|axis| self.axis(*axis))
            .map(|axis| axis.label())
            .collect()
    }
}

/// What a mask stroke does to the protection under the brush.
///
/// Three operations and no fourth, and the absence of one is worth naming: a
/// destructive whole-mask operation is **never** the degenerate no-movement
/// case of a local one. ZBrush's blur is the zero-drag case of its mask paint,
/// which has been its top masking complaint since 2007 and which its vendor
/// answers by telling people to set the blur strength to zero and leave the
/// gesture firing. A mask stroke of zero length here is a zero-length stroke,
/// full stop; Clear, Invert and the absolute Blur are buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskOp {
    /// Paint protection in. The plain left drag.
    Raise,
    /// Take it away. Control, alt, or the eraser end of the stylus -- the
    /// direction is worked out by the caller and NOT by
    /// [`BrushKind::is_directional`], which answers false for three of the seven
    /// brushes and would silently invert this for them.
    Lower,
    /// Soften what is already there, locally, under the brush. Shift.
    Blur,
}

/// One application of a brush at a point.
#[derive(Debug, Clone, Copy)]
pub struct Stamp {
    /// Where the brush is centred, in world space.
    pub centre: Vec3,
    /// Outward surface normal at that point, used by the directional brushes
    /// and by the flatten and clay planes.
    pub normal: Vec3,
    /// Stylus pressure from 0 to 1, scaling strength. Defaults to 1, which is
    /// what a mouse always reports.
    pub pressure: f32,
    /// Which way the stroke is travelling, in world space. The patterns that
    /// comb read it, and move is steered entirely by it. A zero vector means
    /// "not known": the patterns cope by picking any direction across the
    /// surface, and move declines to do anything at all, because a stroke that
    /// has not travelled has not dragged anything.
    ///
    /// Its LENGTH matters to move and to nothing else. The patterns normalise
    /// it, so a unit vector is the right thing to pass for them; move reads it
    /// as the whole drag, because [`Brush::apply`] with
    /// [`BrushKind::Move`] is one entire gesture rather than one stamp of one.
    pub tangent: Vec3,
    pub direction: BrushDirection,
}

impl Stamp {
    pub fn new(centre: Vec3, normal: Vec3, direction: BrushDirection) -> Self {
        Self { centre, normal, pressure: 1.0, tangent: Vec3::ZERO, direction }
    }

    pub fn with_pressure(mut self, pressure: f32) -> Self {
        self.pressure = pressure.clamp(0.0, 1.0);
        self
    }

    /// Set the stroke's direction of travel, which is what combs a hair
    /// pattern along the drag and what the move brush drags along.
    ///
    /// Move reads the whole vector, not just its direction. See
    /// [`Stamp::tangent`].
    pub fn with_tangent(mut self, tangent: Vec3) -> Self {
        self.tangent = tangent;
        self
    }
}

/// Whether a stamp may leave out the bricks it can prove it would not change.
///
/// Two spellings of one operation, which is the shape that goes quietly wrong,
/// so they are pinned against each other by a test rather than trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Skipping {
    On,
    Off,
}

/// Reusable working memory for stamping.
///
/// Holding one across a stroke is what keeps sculpting out of the allocator.
#[derive(Debug, Default)]
pub struct BrushScratch {
    region: FieldRegion,
    /// Move's locked field, for the one shot [`Brush::apply`] path. The
    /// interactive path keeps its own [`MoveStroke`] alive across the whole
    /// gesture instead, which is the entire point of the brush.
    locked: MoveStroke,
}

impl BrushScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A configured brush.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brush {
    pub kind: BrushKind,
    /// Radius of influence in world units.
    pub radius: f32,
    /// How hard one stamp bites. Keep well below 1: a stroke lays down many
    /// overlapping stamps, so this is per stamp and not per stroke.
    pub strength: f32,
    pub falloff: FalloffCurve,
    /// A surface pattern multiplied into the weight. See [`crate::pattern`]:
    /// it modifies whichever brush is selected rather than being a brush of
    /// its own.
    pub pattern: Pattern,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            kind: BrushKind::Draw,
            radius: 3.0,
            strength: 0.15,
            falloff: FalloffCurve::Smooth,
            pattern: Pattern::default(),
        }
    }
}

impl Brush {
    /// Spacing between stamps along a stroke, in world units.
    ///
    /// A quarter of the radius keeps consecutive stamps heavily overlapped, so
    /// a fast drag leaves a continuous cut instead of a dotted trail. Never
    /// smaller than a voxel, or a slow drag would stamp the same voxels over
    /// and over for no visible gain.
    ///
    /// Move does not use this: a gesture is one locked warp recomputed per
    /// pointer event, not a trail of stamps. See [`MoveStroke`].
    pub fn spacing(&self, voxel_size: f32) -> f32 {
        (self.radius * 0.25).max(voxel_size)
    }

    /// Furthest one Move gesture may carry the surface, in world units.
    ///
    /// The displacement is the drag times a falloff, so it changes by up to
    /// `drag * falloff.max_slope() / radius` per unit of distance across the
    /// brush. Once that reaches 1 the warp stops being invertible and the field
    /// folds back through itself, which is a crease that no amount of narrow
    /// band clamping catches. [`MOVE_DRAG_MARGIN`] backs off from that
    /// threshold, giving half the radius for the default smooth falloff, three
    /// quarters for the linear one and a quarter for the two steep cubics.
    ///
    /// It is not much of a restriction in practice, because this formulation
    /// saturates near there anyway: the source of a voxel is `p - drag * w(p)`,
    /// and solving for where material actually lands shows it stops advancing
    /// somewhere around two thirds of the radius however far the pointer goes.
    /// Past the cap the surface simply stops following, which is a shape the
    /// user can see and work with. It does not tear, and it does not smear a
    /// rim value across the brush, because the cap is exactly what keeps every
    /// read inside the locked box.
    pub fn max_drag(&self) -> f32 {
        self.radius * MOVE_DRAG_MARGIN / self.falloff.max_slope()
    }

    /// Paint the mask under one stamp, plus its mirrors when symmetry is on.
    ///
    /// Mirroring is not in the plan's list for this increment and is here on
    /// purpose: the three mirror toggles stay lit in the tool strip while the
    /// mask tool is live, and the mirror planes stay drawn in the viewport, so a
    /// mask that ignored them would be the interface saying one thing and the
    /// tool doing another. It costs the same twin loop the field already runs
    /// and no new concept.
    pub fn apply_mask_symmetric(
        &self,
        volume: &mut Volume,
        stamp: &Stamp,
        op: MaskOp,
        symmetry: Symmetry,
        centre: Vec3,
        scratch: &mut BrushScratch,
    ) {
        self.apply_mask(volume, stamp, op, scratch);
        if symmetry.is_off() {
            return;
        }
        let mut twins = [*stamp; Symmetry::MAX_MIRRORS];
        let count = symmetry.mirrors(stamp, centre, &mut twins);
        for twin in &twins[..count] {
            self.apply_mask(volume, twin, op, scratch);
        }
    }

    /// Paint the mask under one stamp.
    ///
    /// # Why this is not a `BrushKind`
    ///
    /// Masking is a TOOL and never a brush: the digits index
    /// [`BrushKind::ALL`], and a `BrushKind` that must never write the field
    /// would need an arm in five exhaustive matches inside this crate for a
    /// variant every one of them would have to refuse. It borrows the brush's
    /// radius, falloff, pattern and pressure -- which is what makes `s`, `[`,
    /// `]` and the falloff curves work on it with no new code, and which is
    /// what ZBrush does -- and it has its own strength, held by the caller.
    ///
    /// # Blend toward the target, never add to the value
    ///
    /// `new = old + (target - old) * weight`, with `weight` in `0..=1` by
    /// construction, exactly as Smooth, Flatten and Clay blend toward theirs.
    /// The two properties that buys are the reason it is written this way:
    /// repeated stamps converge on the target instead of overshooting it, and
    /// the edge of every stroke is FEATHERED because the falloff is a factor of
    /// the blend rather than an addition to it. The second is a rule and not a
    /// preference -- [`crate::mask`] gives its three independent
    /// justifications, of which the sharpest is that a step in the mask is a
    /// fold in the geometry under Move.
    ///
    /// The pattern multiplies in here as it does on the field, and that is
    /// deliberate: it is already pinned to `0..=1`, so masking through Scales or
    /// Cracks cannot escape the range, and it is ZBrush's alpha masking arriving
    /// for free. Turning it off would be code added to remove capability.
    pub fn apply_mask(
        &self,
        volume: &mut Volume,
        stamp: &Stamp,
        op: MaskOp,
        scratch: &mut BrushScratch,
    ) {
        if self.radius <= 0.0 || self.strength <= 0.0 || stamp.pressure <= 0.0 {
            return;
        }

        let voxel_size = volume.voxel_size();
        let inverse_radius = 1.0 / self.radius;
        let gain = (self.strength * stamp.pressure).clamp(0.0, 1.0);
        let extent = Vec3::splat(self.radius);
        let (lo, hi) = volume.voxel_bounds(stamp.centre - extent, stamp.centre + extent);

        // Blur is the one mask operation that is not a per voxel function, so
        // it runs the same two-phase shape every field brush that reads its
        // neighbours runs: copy the box out, then write back using only the
        // copy. Without it a voxel would average a mixture of old and new
        // values and the result would depend on visiting order.
        if op == MaskOp::Blur {
            volume.snapshot_mask(lo, hi, &mut scratch.region);
        }
        let region = &scratch.region;

        let pattern = if self.pattern.is_off() {
            Prepared::OFF
        } else {
            self.pattern.prepare(voxel_size, stamp.normal, stamp.tangent)
        };

        let falloff = self.falloff;
        let centre = stamp.centre;
        let radius_squared = self.radius * self.radius;

        volume.edit_mask(lo, hi, |voxel, position, protection| {
            let offset = position - centre;
            if offset.length_squared() >= radius_squared {
                return protection;
            }
            let distance = offset.length() * inverse_radius;
            let weight = falloff.weight(distance) * gain * pattern.weight(position);
            if weight <= 0.0 {
                return protection;
            }
            let target = match op {
                MaskOp::Raise => PROTECTED as f32,
                MaskOp::Lower => UNMASKED as f32,
                // The kernel Smooth already uses on the field, over a snapshot
                // of the mask instead of a snapshot of the distances.
                MaskOp::Blur => region.neighbour_average(voxel),
            };
            let held = protection as f32;
            let next = held + (target - held) * weight;
            // **Quantised toward the target only when the target is within
            // reach**, and the asymmetry between the two directions is the
            // whole of this.
            //
            // Eight bits cannot hold a proportional blend exactly, so the
            // rounding has to go somewhere. To-nearest alone sends it the wrong
            // way at the end: at a weight of 0.4 the blend reaches 254 and then
            // stops -- `254 + (255 - 254) * 0.4` rounds back to 254 -- so "keep
            // painting until it is protected" never arrives. That is not a
            // cosmetic one level: the planner's skip and the plane cut's
            // spared-brick test both compare against `PROTECTED` exactly, so a
            // mask a user had painted solid would still let a cut through.
            //
            // Rounding TOWARD the target unconditionally fixes that and buys a
            // worse bug, because a step of `(255 - held) * w` ceils to 1 however
            // small `w` is: every voxel the brush touches at all then gains a
            // level per stamp, the whole footprint ratchets to `PROTECTED` in a
            // few hundred stamps, and the feathered rim `crate::mask` requires
            // -- the rim `mask_drag_scale`'s half-margin rests on -- collapses
            // into the single voxel at the radius. Measured at radius 8,
            // strength 0.4: 300 stamps gave `[255 x 8, 0]` where one stamp gave
            // `[102, 98, 87, 70, 51, 33, 16, 5, 0]`.
            //
            // So the snap is confined to the last level, where it is the only
            // thing that can move the value at all, and everything before it
            // rounds to nearest and settles at a plateau that varies smoothly
            // with the weight. Exact arrival and a permanently graded rim are
            // mutually exclusive for a memoryless 8-bit blend -- either the
            // fixed point is the target for every non-zero weight, which is the
            // ratchet, or it is weight-dependent, which cannot be the target
            // everywhere -- so each direction takes the one that costs less.
            //
            // **Lower keeps the ratchet deliberately, because `UNMASKED` is not
            // a value but the absent state.** [`crate::MaskField::collapse`]
            // drops a brick only at exactly 0, so a residue of 2 left by a
            // rounded erase is permanent: `is_free` stays false, the standing
            // card goes on naming a body at 1%, Move's cap goes on being halved,
            // and no amount of further erasing clears it. Falling short of 255
            // has no equivalent -- it costs sparing at the rim of a cut and
            // nothing else.
            //
            // Blur snaps at neither end, because it has no exact target:
            // rounding its step away from the value would make a settled region
            // dither one level either side of its own neighbourhood average
            // forever.
            let next = match op {
                MaskOp::Raise if target - next < 1.0 => next.ceil(),
                MaskOp::Lower => next.floor(),
                _ => next.round(),
            };
            next.clamp(UNMASKED as f32, PROTECTED as f32) as u8
        });
    }

    /// Apply one stamp, plus its mirrors when symmetry is on.
    ///
    /// `centre` is the point the enabled mirror planes pass through. See
    /// [`Symmetry`] for why it is threaded through as a parameter and why every
    /// caller passes the lattice origin today.
    pub fn apply_symmetric(
        &self,
        volume: &mut Volume,
        stamp: &Stamp,
        symmetry: Symmetry,
        centre: Vec3,
        scratch: &mut BrushScratch,
    ) {
        // Move mirrors itself, rather than being applied once per twin. Two
        // twins whose boxes overlap near a mirror plane each warp from their
        // own locked copy, so applying them in turn would have the second
        // rewrite the first's work with the field as it stood before it.
        if self.kind == BrushKind::Move {
            self.drag_once(volume, stamp, symmetry, centre, scratch, Skipping::On);
            return;
        }

        self.apply(volume, stamp, scratch);
        if symmetry.is_off() {
            return;
        }
        let mut twins = [*stamp; Symmetry::MAX_MIRRORS];
        let count = symmetry.mirrors(stamp, centre, &mut twins);
        for twin in &twins[..count] {
            self.apply(volume, twin, scratch);
        }
    }

    /// One whole Move gesture in a single call, dragging by `stamp.tangent`.
    ///
    /// The interactive path does not come through here -- it holds a
    /// [`MoveStroke`] open for the length of the gesture, which is what makes
    /// dragging out and back return the form. This is that same warp with the
    /// field locked and released around one pointer event, which is what a
    /// caller holding a `Brush` and a `Stamp` can express.
    fn drag_once(
        &self,
        volume: &mut Volume,
        stamp: &Stamp,
        symmetry: Symmetry,
        centre: Vec3,
        scratch: &mut BrushScratch,
        skipping: Skipping,
    ) {
        if stamp.tangent == Vec3::ZERO {
            return;
        }
        scratch.locked.begin(volume, self, stamp.centre, symmetry, centre);
        scratch.locked.drag_to_where(
            volume,
            stamp.centre + stamp.tangent,
            stamp.pressure,
            skipping,
        );
        scratch.locked.end();
    }

    /// How far past its own radius one stamp reads the field, in world units.
    ///
    /// The brushes that warp the domain resample from `position - shift`
    /// rather than reading the voxel they are writing, and nothing about the
    /// radius covers that shift. This is the bound on it: the shift is the
    /// per-stamp displacement scaled by a weight that never exceeds `gain`, so
    /// `gain * displacement` bounds it whatever the falloff curve does in
    /// between.
    ///
    /// Deliberately a crude bound rather than a tight one. The tight bound
    /// depends on the maximum slope of the falloff curve, on
    /// [`STAMP_FRACTION_OF_RADIUS`] and on the snapshot's padding all at once,
    /// and a bound that three unrelated constants have to agree on is a bound
    /// that breaks the first time one of them moves. This one costs at most a
    /// voxel of extra snapshot and cannot break that way.
    ///
    /// Move is absent because it never reaches this code: [`MoveStroke`] works
    /// out its own boxes, and the cap on its drag is what lets the box it reads
    /// and the box it writes be the same one.
    ///
    /// The value brushes read only the voxel they write, or its immediate
    /// neighbours, which the snapshot's padding already covers.
    /// Whether this brush reads the field anywhere other than the voxel it is
    /// writing, and therefore needs a snapshot taken before the edit.
    ///
    /// Three of the seven do: draw and pinch resample through
    /// [`FieldRegion::sample`], and smooth averages its neighbours. The rest
    /// compute a new value from the old one and a plane, and never look at the
    /// copy at all.
    ///
    /// Taking the snapshot regardless was costing them a great deal. It is a
    /// straight copy of the read box, which at an 80 voxel radius is 15.9 MB,
    /// measured at **0.95 ms of Clay's 2.21 ms total** -- forty five percent of
    /// the brush's cost spent copying memory nothing would read.
    ///
    /// `reads_the_field_elsewhere_is_honest` pins this against drift: it runs
    /// every brush that answers `false` against a deliberately poisoned region
    /// and requires an identical field, so adding a `region` read to one of
    /// them without updating this fails loudly rather than silently sampling
    /// stale data.
    fn reads_the_field(&self) -> bool {
        match self.kind {
            BrushKind::Draw | BrushKind::Pinch | BrushKind::Smooth => true,
            // Move never reaches this path at all; it has its own locked copy.
            BrushKind::Clay | BrushKind::Flatten | BrushKind::Inflate | BrushKind::Move => false,
        }
    }

    fn read_reach(&self, voxel_size: f32, gain: f32, displacement: f32) -> f32 {
        let voxels = match self.kind {
            BrushKind::Draw => gain * displacement,
            BrushKind::Pinch => gain * PINCH_PULL_VOXELS,
            BrushKind::Move
            | BrushKind::Clay
            | BrushKind::Smooth
            | BrushKind::Inflate
            | BrushKind::Flatten => 0.0,
        };
        voxels * voxel_size
    }

    /// Whether one stamp provably leaves a region alone, given that the region
    /// and everything the stamp can read around it already hold the single
    /// value `value` at every voxel.
    ///
    /// This is what lets a large brush ignore the deep interior and the far
    /// exterior, which between them are most of its box: a 20 mm brush at a
    /// quarter millimetre voxel spans four million voxels and only a shell a
    /// few voxels thick around the surface carries anything it can act on.
    ///
    /// It is a claim about the arithmetic in [`Brush::apply`] and it has to
    /// stay true of it. `every_brush_skips_only_what_it_would_not_have_changed`
    /// runs every brush both ways and compares the field bit for bit, so a
    /// wrong answer here fails a test rather than showing up as a quiet dent in
    /// somebody's model.
    fn leaves_constant_alone(&self, value: f32, direction: BrushDirection) -> bool {
        // A floor rather than a necessity. Every brush below that answers yes
        // would in fact leave any constant alone, saturated or not, and a
        // uniform brick part way through the band can exist -- a tile that an
        // edit clipped but never reached is put back exactly as it was. But a
        // constant mid band value means a flat patch of real surface, which is
        // where a brush is supposed to be doing something, and a claim that
        // holds only for the two ends of the band is a much easier one for the
        // next brush to keep. Removing this line does not break anything
        // today; keeping it means it cannot.
        if value != INSIDE && value != OUTSIDE {
            return false;
        }
        match self.kind {
            // Resampling a constant field gives the constant back, whatever the
            // weight and wherever it reads from, provided everything it can
            // read is covered by the halo -- which is what `apply` declares.
            BrushKind::Draw | BrushKind::Pinch | BrushKind::Move => true,
            // The mean of six equal values is that value, so the blend has
            // nothing to blend toward.
            BrushKind::Smooth => true,
            // Adding a constant offset lands back on the clamp it started at,
            // but only in the direction the clamp is already holding: inflating
            // outward cannot make solid interior any more solid, and carving
            // cannot make empty space any emptier.
            BrushKind::Inflate => match direction {
                BrushDirection::Add => value == INSIDE,
                BrushDirection::Subtract => value == OUTSIDE,
            },
            // Both blend toward a plane, and the plane's own value varies
            // across the region. Far enough behind it the blend does clamp
            // straight back, but proving that needs the region's box and the
            // plane together, and neither of them is the expensive case: these
            // two are the cheapest brushes there are. They get the radius
            // culling and nothing more.
            BrushKind::Clay | BrushKind::Flatten => false,
        }
    }

    /// Apply one stamp.
    ///
    /// Work is proportional to the brush volume, never to the size of the
    /// model.
    pub fn apply(&self, volume: &mut Volume, stamp: &Stamp, scratch: &mut BrushScratch) {
        self.stamp(volume, stamp, scratch, Skipping::On);
    }

    /// One stamp, with the option of visiting every brick of the box whether it
    /// can change or not.
    ///
    /// The two are meant to produce identical fields and identical undo
    /// entries, and `skipping_leaves_the_same_field_and_the_same_undo_entry`
    /// is what holds them to it. Nothing outside that test asks for
    /// [`Skipping::Off`]: it exists so the optimisation has something honest to
    /// be compared against.
    fn stamp(
        &self,
        volume: &mut Volume,
        stamp: &Stamp,
        scratch: &mut BrushScratch,
        skipping: Skipping,
    ) {
        if self.radius <= 0.0 || self.strength <= 0.0 || stamp.pressure <= 0.0 {
            return;
        }

        // Move is a gesture rather than a stamp, and is steered by the drag
        // rather than by the surface, so a stroke that has not travelled yet
        // has nothing to tell it. Answered once per stamp rather than per
        // voxel, and before any of the work below.
        if self.kind == BrushKind::Move {
            // Symmetry is OFF here, so there is nothing for a centre to be the
            // centre OF; `ZERO` is the value that cannot be wrong.
            self.drag_once(volume, stamp, Symmetry::OFF, Vec3::ZERO, scratch, skipping);
            return;
        }

        let voxel_size = volume.voxel_size();
        let inverse_radius = 1.0 / self.radius;
        let gain = self.strength * stamp.pressure;
        // Displacement at full weight, in voxels, capped so one stamp can never
        // saturate the narrow band. See MAX_STAMP_VOXELS.
        let displacement =
            (self.radius / voxel_size * STAMP_FRACTION_OF_RADIUS).min(MAX_STAMP_VOXELS);
        let field_sign = stamp.direction.field_sign();

        // Two boxes, not one. The box being written is the brush's own reach.
        // The box being read is that grown by however far the warp resamples
        // from, which for the brushes that only read the voxel they write is
        // not at all. Reading and writing the same box is correct for those and
        // silently wrong for the rest -- see the module docs.
        let extent = Vec3::splat(self.radius);
        let (lo, hi) = volume.voxel_bounds(stamp.centre - extent, stamp.centre + extent);
        let read_extent = extent + Vec3::splat(self.read_reach(voxel_size, gain, displacement));
        let (read_lo, read_hi) =
            volume.voxel_bounds(stamp.centre - read_extent, stamp.centre + read_extent);

        // Only the resampling brushes need the copy. See `reads_the_field`.
        if self.reads_the_field() {
            volume.snapshot(read_lo, read_hi, &mut scratch.region);
        }
        let region = &scratch.region;

        // Reference plane for clay and flatten, in world space. Clay holds it
        // just outside the surface so material builds up to it.
        let plane_point = match self.kind {
            BrushKind::Clay => {
                stamp.centre
                    + stamp.normal * (self.radius * CLAY_OFFSET * stamp.direction.outward_sign())
            }
            _ => stamp.centre,
        };

        // Resolved once per stamp rather than per voxel: the projection axes
        // and the reciprocal scale do not vary inside one stamp.
        let pattern = if self.pattern.is_off() {
            Prepared::OFF
        } else {
            self.pattern.prepare(voxel_size, stamp.normal, stamp.tangent)
        };

        let kind = self.kind;
        let falloff = self.falloff;
        let centre = stamp.centre;
        let stroke_normal = stamp.normal;
        let direction = stamp.direction;

        // Which of that box can actually change. Without this a large brush
        // pays for its whole bounding cube: the corners the ball never reaches,
        // and the deep interior and far exterior that are already saturated at
        // one value. Both are far bigger than the shell of surface the stamp is
        // really working on.
        //
        // How far past the voxel it is writing the stamp reads, which is what
        // the volume resolves the constant regions against. It comes from the
        // same `read_reach` the snapshot is sized with, so a brush that starts
        // resampling from further away widens both together.
        let read_voxels = (self.read_reach(voxel_size, gain, displacement) / voxel_size).ceil()
            as i32
            // The trilinear tap lands one voxel past wherever the warp reads,
            // and the value brushes that warp nothing still read their six
            // axis neighbours.
            + 1;
        let radius = self.radius;
        let decide = |preview: &BrickPreview| {
            if skipping == Skipping::Off {
                return BrickVerdict::Whole;
            }
            // Nothing here may be written at all, and unlike every other skip
            // in this function that needs no reasoning about neighbours: the
            // mask kills the write rather than the read. Resolved protection,
            // so this fires on a Mask All rather than in spite of one.
            if preview.mask == Some(PROTECTED) {
                return BrickVerdict::Skip;
            }
            // A voxel further from the centre than the radius gets its own
            // value back, so a brick the ball misses cannot change. Grown by a
            // voxel so this is a bound on the per voxel test rather than a race
            // with the last bit of its rounding.
            let slack = Vec3::splat(voxel_size);
            let box_min = preview.lo.as_vec3() * voxel_size - slack;
            let box_max = preview.hi.as_vec3() * voxel_size + slack;
            if centre.clamp(box_min, box_max).distance(centre) >= radius {
                return BrickVerdict::Skip;
            }
            match preview.uniform {
                Some(value) if self.leaves_constant_alone(value, direction) => {
                    BrickVerdict::OnlyNearDifferentNeighbours
                }
                // The two that blend toward a plane, which the claim above
                // cannot make because the target varies across the brick.
                // Here the brick's own box is in hand, so the target's range
                // over it can be worked out and the claim made properly. See
                // `blend_toward_plane_clamps_back`.
                Some(value)
                    if matches!(kind, BrushKind::Clay | BrushKind::Flatten)
                        && blend_toward_plane_clamps_back(
                            value,
                            box_min,
                            box_max,
                            plane_point,
                            stroke_normal,
                            voxel_size,
                        ) =>
                {
                    // Not `OnlyNearDifferentNeighbours`: these two read the
                    // voxel they write and nothing else, so there is no
                    // neighbour that could reach in, and the whole box is
                    // provably left alone rather than all but a rim of it.
                    BrickVerdict::Skip
                }
                _ => BrickVerdict::Whole,
            }
        };

        // Squared, so the voxels outside the brush never pay for a square root.
        // The box handed to `edit_voxels_where` is a CUBE and the brush is a
        // SPHERE, so a little under half of everything visited is outside the
        // radius and exists only to be rejected -- and `Vec3::distance` is a
        // `sqrt` each time. Comparing squares rejects them with a dot product
        // and a compare, and the root is then taken only for the voxels that
        // are actually going to be written.
        let radius_squared = self.radius * self.radius;

        volume.edit_voxels_where(
            lo,
            hi,
            read_voxels,
            true,
            decide,
            |voxel, position, value, free| {
                let offset = position - centre;
                if offset.length_squared() >= radius_squared {
                    return value;
                }
                let distance = offset.length() * inverse_radius;
                let shaped = falloff.weight(distance) * gain;
                if shaped <= 0.0 {
                    return value;
                }
                // The pattern is one extra multiply, and the mask a second, and
                // both are evaluated only for voxels the falloff has not already
                // zeroed. Both stay in 0..=1, so the product does too and the
                // blending brushes below keep a legal lerp factor -- which is what
                // makes smooth, flatten and clay converge on their target instead
                // of extrapolating away from it.
                let weight = shaped * pattern.weight(position) * free;
                if weight <= 0.0 {
                    return value;
                }

                match kind {
                    BrushKind::Inflate => value + field_sign * weight * displacement,

                    BrushKind::Draw => {
                        // Translate the field along the stroke normal, which slides
                        // this patch of surface out from under the cursor.
                        //
                        // The tempting version, weighting an offset by how much the
                        // local gradient faces the stroke, does not survive a
                        // stroke: wherever the field is flat or saturated that
                        // gradient is noise, and multiplying a displacement by it
                        // turns the noise into geometry. A resample cannot
                        // introduce detail that was not already there.
                        let shift = stroke_normal
                            * (weight * displacement * voxel_size * direction.outward_sign());
                        region.sample((position - shift) / voxel_size)
                    }

                    BrushKind::Smooth => {
                        let average = region.neighbour_average(voxel);
                        value + (average - value) * weight
                    }

                    BrushKind::Pinch => {
                        // Read the field from slightly closer to the brush axis,
                        // which drags the surface sideways and squeezes whatever
                        // ridge runs through the brush into a crease.
                        //
                        // The obvious alternative, an unsharp mask on the values,
                        // looks the same for one stamp and then compounds: it is a
                        // high pass filter with gain above 1, so across a stroke it
                        // amplifies its own rounding error into a crust. Warping
                        // the domain only ever resamples what is already there.
                        let to_centre = centre - position;
                        // Along the surface only. Pulling along the normal as well
                        // would make pinch quietly double as inflate.
                        let lateral = to_centre - stroke_normal * to_centre.dot(stroke_normal);
                        let Some(direction_to_axis) = lateral.try_normalize() else {
                            return value;
                        };
                        let spread = match direction {
                            BrushDirection::Add => 1.0,
                            BrushDirection::Subtract => -1.0,
                        };
                        let pull =
                            direction_to_axis * (weight * PINCH_PULL_VOXELS * voxel_size * spread);
                        region.sample((position + pull) / voxel_size)
                    }

                    BrushKind::Flatten => {
                        let plane = (position - plane_point).dot(stroke_normal) / voxel_size;
                        value + (plane - value) * weight
                    }

                    BrushKind::Move => {
                        // Unreachable: `apply` sends Move to `drag_once` before any
                        // of this, because a gesture is not a stamp.
                        value
                    }

                    BrushKind::Clay => {
                        let plane = (position - plane_point).dot(stroke_normal) / voxel_size;
                        let blended = value + (plane - value) * weight;
                        // Keep only the half of the operation that moves material
                        // the way the user asked for. Without this, clay carves
                        // away any bump standing above its plane.
                        match direction {
                            BrushDirection::Add => blended.min(value),
                            BrushDirection::Subtract => blended.max(value),
                        }
                    }
                }
            },
        );
    }
}

/// Whether clay's and flatten's blend toward a plane provably clamps straight
/// back to `value` everywhere in a world space box already holding `value`.
///
/// Both compute `value + (plane - value) * weight` and clamp the result to the
/// narrow band, where `plane` is the signed distance to the reference plane in
/// voxels and `weight` is somewhere in `0..=1`. So whenever `plane` sits at or
/// beyond a saturated `value` on the same side, the blend can only push further
/// out and the clamp puts it straight back. Clay's extra `min`/`max` against
/// `value` moves it the same way or not at all, so it does not weaken this.
///
/// That is a claim about a whole box rather than about one value, which is why
/// it cannot live in [`Brush::leaves_constant_alone`] alongside the others: the
/// plane's distance varies across a brick, so the brick's extent has to be in
/// hand. `plane` is linear in position, so its range over the box is its value
/// at the centre give or take the box's half extent projected onto the normal
/// -- there is no need to visit the eight corners.
///
/// Conservative in the only direction that matters: a box grown by slack widens
/// that range and so makes the claim harder to satisfy, never easier.
fn blend_toward_plane_clamps_back(
    value: f32,
    box_min: Vec3,
    box_max: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
    voxel_size: f32,
) -> bool {
    // In voxels, because that is the unit `apply` puts the plane distance in
    // and the unit the band's own bounds are expressed in.
    let middle = ((box_min + box_max) * 0.5 - plane_point).dot(plane_normal) / voxel_size;
    let spread = ((box_max - box_min) * 0.5).dot(plane_normal.abs()) / voxel_size;
    if value == INSIDE {
        middle + spread <= INSIDE
    } else if value == OUTSIDE {
        middle - spread >= OUTSIDE
    } else {
        // Same floor as `leaves_constant_alone`: a constant part way through
        // the band is a flat patch of real surface, which is where these two
        // are meant to be doing something.
        false
    }
}

/// One locked copy of the field: the gesture itself, or one of its mirrors.
///
/// The box is fixed at the moment the gesture starts and never moves, which is
/// what makes every pointer event write over exactly the voxels the last one
/// wrote and leave no residue behind.
#[derive(Debug, Clone, Copy, Default)]
struct MoveAnchor {
    /// Where the gesture was locked, in world space.
    origin: Vec3,
    /// Componentwise sign this anchor's mirror applies to the drag. All ones
    /// for the gesture itself.
    flip: Vec3,
    /// Inclusive voxel box that gets written, which is the brush's own reach.
    lo: IVec3,
    hi: IVec3,
}

/// Which bricks of one locked copy hold a single value everywhere.
///
/// The warp resamples from the copy, so what a voxel ends up holding is decided
/// by the copy and not by the volume being written. Answering "was everything
/// this brick could pull from already the same value" needs the copy classified
/// the way [`Volume::edit_voxels_where`] classifies the volume, and it is
/// classified once per lock rather than once per pointer event.
#[derive(Debug, Default)]
struct LockedFills {
    /// Inclusive voxel box the copy stores, taken from it rather than
    /// reconstructed, because a read outside it clamps to its rim.
    lo: IVec3,
    hi: IVec3,
    /// Minimum brick of the grid, and its extent in bricks.
    min: IVec3,
    size: IVec3,
    /// `Some(v)` when every voxel of that brick held `v` at the lock, laid out
    /// with X fastest from `min`.
    fills: Vec<Option<f32>>,
}

impl LockedFills {
    /// Classify every brick the copy covers. Reuses the last lock's storage.
    fn build(&mut self, volume: &Volume, lo: IVec3, hi: IVec3) {
        self.lo = lo;
        self.hi = hi;
        self.min = BrickCoord::containing(lo).0;
        self.size = BrickCoord::containing(hi).0 - self.min + IVec3::ONE;

        self.fills.clear();
        self.fills.reserve(self.size.as_i64vec3().element_product().max(0) as usize);
        for z in 0..self.size.z {
            for y in 0..self.size.y {
                for x in 0..self.size.x {
                    let coord = BrickCoord(self.min + IVec3::new(x, y, z));
                    self.fills.push(volume.brick_fill(coord));
                }
            }
        }
    }

    /// Which part of an inclusive voxel range could read something other than
    /// `value` out of the copy, given that the warp displaces a voxel by
    /// somewhere in the voxel box `back_min..=back_max`.
    ///
    /// `None` when everything the range can reach held `value`, so it is
    /// untouchable. Otherwise a bounding box of the part that is not, which is
    /// exact whenever the differing bricks all lie on one side -- the usual
    /// case, because what makes one differ is the surface passing through it.
    ///
    /// [`Volume::edit_voxels_where`] answers the same question for the brushes
    /// that read the volume they write. This is the version for a warp, and it
    /// differs in the two ways that matter: the grid is the locked copy rather
    /// than the volume, and the reach is a box offset along the drag rather
    /// than a couple of voxels in every direction.
    fn reachable_from_elsewhere(
        &self,
        lo: IVec3,
        hi: IVec3,
        back_min: IVec3,
        back_max: IVec3,
        value: f32,
    ) -> Option<(IVec3, IVec3)> {
        let read_lo = lo - back_max;
        let read_hi = hi - back_min;
        // A read past the stored box is answered by its rim rather than by the
        // brick it nominally lands in, so rather than reason about that, give
        // up and let the whole range be resolved. It only arises at the outer
        // rim of the box, where the falloff has gone to nothing and the warp is
        // barely displacing anything anyway.
        if read_lo.cmplt(self.lo).any() || read_hi.cmpgt(self.hi).any() {
            return Some((lo, hi));
        }

        let dim = BRICK_DIM as i32;
        let b_min = BrickCoord::containing(read_lo).0;
        let b_max = BrickCoord::containing(read_hi).0;
        let mut bad_lo = IVec3::MAX;
        let mut bad_hi = IVec3::MIN;
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let brick = IVec3::new(bx, by, bz) - self.min;
                    let index =
                        brick.x + brick.y * self.size.x + brick.z * self.size.x * self.size.y;
                    if self.fills[index as usize] == Some(value) {
                        continue;
                    }
                    // The part of that brick the range actually reads.
                    let origin = IVec3::new(bx, by, bz) * dim;
                    bad_lo = bad_lo.min(origin.max(read_lo));
                    bad_hi = bad_hi.max((origin + IVec3::splat(dim - 1)).min(read_hi));
                }
            }
        }
        if bad_lo.cmpgt(bad_hi).any() {
            return None;
        }

        // A voxel reads from itself minus the displacement, so the ones that
        // could have read the differing material are that material shifted
        // forward by the same box.
        let touched_lo = (bad_lo + back_min).max(lo);
        let touched_hi = (bad_hi + back_max).min(hi);
        touched_lo.cmple(touched_hi).all().then_some((touched_lo, touched_hi))
    }
}

/// One anchor's locked copy of the field, and what is known about its bricks.
#[derive(Debug, Default)]
struct Locked {
    field: FieldRegion,
    fills: LockedFills,
}

/// A Move gesture, holding the field as it stood when the button went down.
///
/// # Why the field is locked rather than warped a little at a time
///
/// The obvious formulation resamples the field a little further along on every
/// stamp. It does not work, and the failure is not a tuning problem: a stamp
/// that displaces the field further than about a voxel saturates the narrow
/// band, so each stamp's warp is tiny, while the brush centre advances a whole
/// spacing per stamp. Only material dead centre at full falloff weight keeps
/// pace with the pointer and everything else slides out from under the brush.
/// Measured, before this replaced it: a full viewport drag moved the surface
/// **0.02 mm** at the default 3 mm radius. Finer spacing was tried and does not
/// fix it -- it buys a millimetre at large radii for six times the cost, and
/// nothing at all at the default radius.
///
/// So the field is copied once, at the start, and every pointer event warps
/// that same copy by the TOTAL drag from where the gesture began. Nothing
/// accumulates, so nothing has to be small. That is what Nomad and ZBrush do
/// with a locked vertex selection, and it buys the same two properties:
///
/// - **Out and back returns the form.** The way back is not undoing the way
///   out; it is the same warp with a drag that has shrunk to nothing.
/// - **The surface follows the pointer**, at the full drag rather than a
///   fiftieth of it.
///
/// One pointer event is cheaper than it was: the old version took a snapshot
/// and ran a pass over the brush box for each of the N stamps it interpolated,
/// and this runs one pass over one box and takes no snapshot at all after the
/// first. Measured through the application, the worst event of a radius 20 mm
/// drag went from 20 ms to 14 ms, and the default radius from 0.6 ms to 0.5 ms.
///
/// A whole drag costs MORE, and it is worth being straight about why: the old
/// version only did anything once the pointer had travelled a full stamp
/// spacing, which at that radius is 5 mm, so it sat out most events. This
/// re-warps whenever the pointer has moved a quarter of a voxel. Per millimetre
/// of surface actually moved it is still the cheaper of the two, by a third.
/// What is left is the cost of one pass over the brush box, which at 20 mm and
/// a quarter millimetre voxel is four million voxels and belongs to
/// [`Volume::edit_voxels`], not here.
///
/// # The warp, and why the read box is the write box
///
/// A voxel at `p` reads from `p - drag * falloff(|p - origin| / radius)`. That
/// is the standard approximation to the inverse of a falloff warp, which has no
/// closed form; it is stable, and it is a domain warp rather than a value edit,
/// so it cannot invent detail that was not already in the field.
///
/// Because the drag is capped at [`Brush::max_drag`], the distance from the
/// origin to a read, `u * radius + drag * w(u)`, rises monotonically in `u` and
/// so peaks at the rim, where the falloff is zero and it is exactly the radius.
/// Every read therefore lands inside the box that is written, and the one voxel
/// of padding [`Volume::snapshot`] adds covers the interpolation. That matters
/// more than it sounds: [`FieldRegion::get`] clamps a read outside its box to
/// the edge rather than panicking, so getting this wrong would smear the rim
/// value across the brush while every value stayed legally inside the narrow
/// band and every test still passed.
///
/// # Symmetry
///
/// Each mirror is an anchor of its own with its own locked copy, and every
/// anchor contributes its own displacement to every voxel, summed. Applying
/// the twins one after another instead would have each rewrite the last one's
/// work from the field as it stood before it, wherever two boxes overlap near
/// a mirror plane. The sum is clamped to the same cap, which is what keeps the
/// bound above true when a voxel is inside two brushes at once -- and the
/// copies are only grown to cover that when two anchors are close enough for
/// it to arise, so symmetry used away from a mirror plane costs nothing extra.
///
/// # What a pointer event is allowed to leave out
///
/// The box is the brush's whole bounding cube and the ball fills only half of
/// it, and inside the ball most of what a large brush covers is deep interior
/// or empty space. Neither can change, and the argument is the one every other
/// brush uses with one extra term for the warp.
///
/// - Outside every falloff the warp hands a voxel its own value straight back,
///   so a brick the ball never reaches is untouchable. That is half the cube.
/// - A brick holding one value everywhere gets that value back **if everything
///   the warp could pull into it held that value too**. Resampling a constant
///   region gives the constant, whatever the weight and wherever inside it the
///   read lands. The region it can pull from is the brick grown by the largest
///   displacement the warp applies anywhere in it, which is the drag scaled by
///   the falloff weight at the brick's nearest point -- so the bricks furthest
///   from the origin, where the weight has fallen away, barely reach past
///   themselves at all, and those are exactly the ones deepest in the material.
///
/// The second is why Move could not simply be handed the volume's own constant
/// test. That test asks about a brick and its 26 neighbours; the warp reads
/// from the LOCKED COPY, tens of voxels away, and it is the copy that has to be
/// saturated. So the copy is classified when it is taken -- once per lock,
/// which is a few hundred map lookups against a pass over four million voxels
/// -- and [`LockedFills`] is what answers it.
#[derive(Debug, Default)]
pub struct MoveStroke {
    /// Empty when no gesture is in progress. `locked` is kept allocated across
    /// gestures and is only ever read at indices below this length.
    anchors: Vec<MoveAnchor>,
    locked: Vec<Locked>,
    /// Locked with the anchors, so dragging the radius slider mid-gesture
    /// cannot invalidate the boxes that were snapshotted for it.
    brush: Brush,
    max_drag: f32,
    /// The displacement already standing in the volume, so an event that has
    /// not moved the pointer is free.
    applied: Vec3,
}

impl MoveStroke {
    pub fn new() -> Self {
        Self::default()
    }

    /// True while a gesture is locked and being dragged.
    #[inline]
    pub fn is_active(&self) -> bool {
        !self.anchors.is_empty()
    }

    /// Lock the field around `at`, plus one copy per mirror.
    ///
    /// `centre` is the point the enabled mirror planes pass through; see
    /// [`Symmetry`].
    ///
    /// Reuses whatever storage the last gesture left behind. The snapshots are
    /// the whole cost of a Move gesture that is not a pointer event, and they
    /// are taken exactly once.
    pub fn begin(
        &mut self,
        volume: &Volume,
        brush: &Brush,
        at: Vec3,
        symmetry: Symmetry,
        centre: Vec3,
    ) {
        self.anchors.clear();
        self.applied = Vec3::ZERO;
        self.brush = *brush;
        self.max_drag = brush.max_drag() * mask_drag_scale(volume);
        if brush.radius <= 0.0 || brush.strength <= 0.0 {
            return;
        }

        let mut flips = [Flip::IDENTITY; Symmetry::MAX_MIRRORS];
        let mirrors = if symmetry.is_off() { 0 } else { symmetry.flips(centre, &mut flips) };

        for flip in std::iter::once(Flip::IDENTITY).chain(flips[..mirrors].iter().copied()) {
            let origin = flip.point(at);
            let (lo, hi) = volume.voxel_bounds(
                origin - Vec3::splat(brush.radius),
                origin + Vec3::splat(brush.radius),
            );
            // The SIGN and not the whole flip: this is applied to the drag
            // vector, which is a direction and takes no offset.
            self.anchors.push(MoveAnchor { origin, flip: flip.sign, lo, hi });
        }

        // A voxel can only be inside two brushes at once if two anchors are
        // within a diameter of each other, which happens when the gesture is
        // working near a mirror plane. Only then is a voxel's displacement a
        // sum rather than a single term, and only then does the argument that
        // every read lands inside the brush's own box stop covering it -- so
        // only then does the copy have to grow by the cap. Symmetry used away
        // from the plane, which is most of the time, costs nothing extra.
        let diameter = brush.radius * 2.0;
        let crowded = self.anchors.iter().enumerate().any(|(index, one)| {
            self.anchors[index + 1..]
                .iter()
                .any(|other| one.origin.distance(other.origin) < diameter)
        });
        let read = Vec3::splat(brush.radius + if crowded { self.max_drag } else { 0.0 });

        for (index, anchor) in self.anchors.iter().enumerate() {
            if self.locked.len() <= index {
                self.locked.push(Locked::default());
            }
            let (read_lo, read_hi) =
                volume.voxel_bounds(anchor.origin - read, anchor.origin + read);
            let locked = &mut self.locked[index];
            volume.snapshot(read_lo, read_hi, &mut locked.field);
            let (stored_lo, stored_hi) = locked.field.bounds();
            locked.fills.build(volume, stored_lo, stored_hi);
        }
    }

    /// Warp the locked field so the material under the origin follows the
    /// pointer to `to`.
    ///
    /// One pass over each anchor's box, however far the gesture has travelled
    /// and however many events it has taken to get there. Nothing accumulates:
    /// this writes the same answer whether it is the first event of the gesture
    /// or the thousandth.
    pub fn drag_to(&mut self, volume: &mut Volume, to: Vec3, pressure: f32) {
        self.drag_to_where(volume, to, pressure, Skipping::On);
    }

    /// One pointer event, with the option of visiting every brick of the box
    /// whether it can change or not.
    ///
    /// The two are meant to produce identical fields and identical undo
    /// entries, and `skipping_leaves_the_same_field_and_the_same_undo_entry`
    /// holds Move to that alongside every other brush. Nothing outside the
    /// tests asks for [`Skipping::Off`].
    fn drag_to_where(&mut self, volume: &mut Volume, to: Vec3, pressure: f32, skipping: Skipping) {
        let Some(anchor) = self.anchors.first() else {
            return;
        };
        let gain = (self.brush.strength * pressure).max(0.0);
        let drag = ((to - anchor.origin) * gain).clamp_length_max(self.max_drag);
        if !drag.is_finite() {
            return;
        }

        let voxel_size = volume.voxel_size();
        if drag.distance(self.applied) < voxel_size * MOVE_SETTLE_VOXELS {
            return;
        }
        self.applied = drag;

        let inverse_radius = 1.0 / self.brush.radius;
        let falloff = self.brush.falloff;
        let cap = self.max_drag;
        let anchors = self.anchors.as_slice();

        for (anchor, locked) in anchors.iter().zip(&self.locked) {
            let field = &locked.field;
            let fills = &locked.fills;
            // Which bricks of the box this event could change at all. See the
            // type docs: the ball misses half the cube, and of what is left the
            // bricks that already hold one value can only change if the region
            // the warp pulls them from held something else.
            let decide = |preview: &BrickPreview| {
                if skipping == Skipping::Off {
                    return BrickVerdict::Whole;
                }
                // Nothing here may be written at all. See the same three lines
                // in `Brush::stamp`: resolved protection, so it fires on a Mask
                // All rather than in spite of one.
                if preview.mask == Some(PROTECTED) {
                    return BrickVerdict::Skip;
                }
                // Grown by a voxel so this bounds the per voxel test rather
                // than racing the last bit of its rounding.
                let slack = Vec3::splat(voxel_size);
                let box_min = preview.lo.as_vec3() * voxel_size - slack;
                let box_max = preview.hi.as_vec3() * voxel_size + slack;

                // How far the warp displaces anything in this brick, as a box
                // rather than a radius. Every anchor's contribution is the drag
                // scaled by a weight between zero and its largest here, so the
                // total lies inside the box spanned by those segments -- and
                // the final clamp to the cap only shrinks it. Direction is the
                // whole point: the source region is this brick swept BACK along
                // the drag, and at a 20 mm radius that is a couple of bricks
                // against the hundred an isotropic grow by the same distance
                // would have to look at.
                //
                // The falloff never rises with distance, so evaluating it at
                // the brick's nearest point to each anchor bounds it over the
                // whole brick.
                //
                // The cull is the WEIGHT and not the displacement, for the same
                // reason the per voxel test is: a drag that has come back to
                // where it started displaces nothing anywhere, and those voxels
                // are precisely the ones that have to be rewritten from the
                // locked copy to put the form back.
                let mut back_min = Vec3::ZERO;
                let mut back_max = Vec3::ZERO;
                let mut reached = false;
                for one in anchors {
                    let near = one.origin.clamp(box_min, box_max).distance(one.origin);
                    let weight = falloff.weight(near * inverse_radius);
                    if weight <= 0.0 {
                        continue;
                    }
                    reached = true;
                    let far = drag * one.flip * weight;
                    back_min += far.min(Vec3::ZERO);
                    back_max += far.max(Vec3::ZERO);
                }
                if !reached {
                    return BrickVerdict::Skip;
                }

                let Some(value) = preview.uniform else {
                    return BrickVerdict::Whole;
                };
                // A voxel at `p` reads from `p - displacement`, rounded outward
                // and then by the one further voxel the trilinear cell
                // straddles.
                let back_min = (back_min / voxel_size).floor().as_ivec3() - IVec3::ONE;
                let back_max = (back_max / voxel_size).ceil().as_ivec3() + IVec3::ONE;
                match fills
                    .reachable_from_elsewhere(preview.lo, preview.hi, back_min, back_max, value)
                {
                    None => BrickVerdict::Skip,
                    Some((near_lo, near_hi)) => BrickVerdict::OnlyWithin(near_lo, near_hi),
                }
            };

            // Zero reach: this never answers `OnlyNearDifferentNeighbours`,
            // because what it has to prove is about the locked copy and not
            // about the brick's neighbours in the volume.
            volume.edit_voxels_where(
                anchor.lo,
                anchor.hi,
                0,
                true,
                decide,
                |_, position, value, free| {
                    // Explicitly, and not by way of a zero displacement. A zero
                    // displacement makes the warp resample the LOCKED COPY at
                    // the voxel, which is the value at lock time rather than the
                    // value now; within a gesture those coincide by induction,
                    // but this is a different expression and it is exactly the
                    // class of difference the On against Off equivalence test
                    // exists to catch.
                    if free <= 0.0 {
                        return value;
                    }
                    let Some(displacement) =
                        move_displacement(anchors, drag, position, inverse_radius, falloff, cap)
                    else {
                        // Outside every falloff, so no drag this gesture could
                        // have would move it and it still holds its locked
                        // value. Resampling would be eight reads and seven
                        // interpolations to arrive back at itself, over the near
                        // half of the box that a ball does not fill.
                        //
                        // The test is the falloff and NOT the displacement: a
                        // displacement of zero is what a drag that has come back
                        // to where it started produces, and those voxels are
                        // exactly the ones that have to be written to put the
                        // form back.
                        return value;
                    };
                    // Inside, this reads from behind along the drag, which is
                    // also what puts the field back where a previous, longer
                    // drag had moved it from: nothing here accumulates.
                    //
                    // The mask scales the DISPLACEMENT rather than blending the
                    // warped value against the old one. A partly applied domain
                    // warp is still a domain warp and still cannot invent
                    // detail; a blend between two resamples is a comb filter,
                    // and across the thirty overlapping stamps of a real stroke
                    // that is a visible ghost.
                    field.sample((position - displacement * free) / voxel_size)
                },
            );
        }
    }

    /// Release the lock, keeping the buffers for the next gesture.
    pub fn end(&mut self) {
        self.anchors.clear();
        self.applied = Vec3::ZERO;
    }

    /// How far the surface has actually been carried so far, after the gain
    /// and the fold-safe clamp.
    ///
    /// The caller needs this to know when a gesture has used up its allowance
    /// and should re-anchor. It is not the same as the distance the pointer has
    /// travelled -- strength scales it down and [`Brush::max_drag`] caps it.
    pub fn applied(&self) -> Vec3 {
        self.applied
    }

    /// Whether the gesture has reached the furthest it can safely warp in one
    /// lock, so the caller should snapshot again and carry on from there.
    pub fn is_at_the_limit(&self) -> bool {
        self.max_drag > 0.0 && self.applied.length() >= self.max_drag * 0.98
    }
}

/// What [`Brush::max_drag`] is multiplied by while a body carries a mask.
///
/// One half, and it buys back a fold bound rather than being cautious for the
/// sake of it. [`Brush::max_drag`] is derived from the condition that the warp
/// stops being invertible once `drag * max_slope / radius` reaches 1; with a
/// mask the gradient the warp actually applies is that of `w * free`, and
/// `|grad(w * free)| <= |grad w| + |grad free|`. Halving the cap covers the
/// second term exactly when the mask's own gradient is no steeper than the
/// brush's falloff, which holds for any mask painted at a comparable radius
/// and for any blurred one. It does not hold for a mask painted with a much
/// smaller brush, which is why every path that writes the mask writes a
/// FEATHERED edge and never a step -- see [`crate::mask`].
///
/// One bool probe per gesture, not per event: an unmasked body pays nothing and
/// keeps the full cap.
fn mask_drag_scale(volume: &Volume) -> f32 {
    if volume.mask().is_free() { 1.0 } else { 0.5 }
}

/// How far the field under `position` is displaced, summed over every mirror,
/// or `None` when the position is outside every falloff and so is not this
/// gesture's to write at all.
///
/// The distinction matters: a voxel inside the falloff with the drag back at
/// zero has a displacement of zero too, and it is precisely those that have to
/// be rewritten from the locked copy to put the form back.
#[inline]
fn move_displacement(
    anchors: &[MoveAnchor],
    drag: Vec3,
    position: Vec3,
    inverse_radius: f32,
    falloff: FalloffCurve,
    cap: f32,
) -> Option<Vec3> {
    // The overwhelmingly common case, and the one the bound on the read box was
    // proved for. Kept separate so symmetry costs nothing when it is off.
    if let [only] = anchors {
        let weight = falloff.weight(position.distance(only.origin) * inverse_radius);
        return (weight > 0.0).then(|| drag * weight);
    }

    let mut total = Vec3::ZERO;
    let mut reached = false;
    for anchor in anchors {
        let weight = falloff.weight(position.distance(anchor.origin) * inverse_radius);
        if weight > 0.0 {
            total += drag * anchor.flip * weight;
            reached = true;
        }
    }
    // Continuous, so two overlapping brushes crease rather than tear, and it is
    // what keeps the summed reach inside the copies that were taken for it.
    reached.then(|| total.clamp_length_max(cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::{INSIDE, OUTSIDE};

    fn sphere() -> Volume {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 24.0);
        volume
    }

    /// A point on the surface of the seeded sphere, and its outward normal.
    fn surface(volume: &Volume) -> (Vec3, Vec3) {
        let point = Vec3::new(24.0, 0.0, 0.0);
        (point, volume.gradient_world(point))
    }

    fn brush(kind: BrushKind) -> Brush {
        Brush { kind, radius: 8.0, strength: 0.4, ..Brush::default() }
    }

    #[test]
    fn every_falloff_curve_runs_from_one_to_zero_and_never_rises() {
        for curve in FalloffCurve::ALL {
            assert!((curve.weight(0.0) - 1.0).abs() < 1.0e-6, "{curve} is not 1 at the centre");
            assert_eq!(curve.weight(1.0), 0.0, "{curve} is not 0 at the rim");
            assert_eq!(curve.weight(2.5), 0.0, "{curve} leaks past the rim");

            let mut previous = f32::INFINITY;
            for step in 0..=100 {
                let value = curve.weight(step as f32 / 100.0);
                assert!(value <= previous + 1.0e-6, "{curve} rose at step {step}");
                assert!((0.0..=1.0).contains(&value), "{curve} left the unit range");
                previous = value;
            }
        }
    }

    #[test]
    fn sharp_is_tighter_than_smooth_and_wide_is_broader() {
        let half = 0.5;
        assert!(FalloffCurve::Sharp.weight(half) < FalloffCurve::Smooth.weight(half));
        assert!(FalloffCurve::Wide.weight(half) > FalloffCurve::Smooth.weight(half));
    }

    #[test]
    fn adding_and_removing_move_the_field_in_opposite_directions() {
        for kind in [BrushKind::Draw, BrushKind::Clay, BrushKind::Inflate] {
            let mut volume = sphere();
            let (point, normal) = surface(&volume);
            let before = volume.sample_world(point);
            let mut scratch = BrushScratch::new();

            brush(kind).apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Add),
                &mut scratch,
            );
            let added = volume.sample_world(point);

            let mut volume = sphere();
            brush(kind).apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Subtract),
                &mut scratch,
            );
            let removed = volume.sample_world(point);

            assert!(added < before, "{kind} did not add material: {before} then {added}");
            assert!(removed > before, "{kind} did not remove material: {before} then {removed}");
        }
    }

    #[test]
    fn every_brush_keeps_values_inside_the_narrow_band() {
        for kind in BrushKind::ALL {
            let mut volume = sphere();
            let (point, normal) = surface(&volume);
            let mut scratch = BrushScratch::new();
            let brush = Brush { kind, radius: 8.0, strength: 0.9, ..Brush::default() };

            for _ in 0..40 {
                brush.apply(
                    &mut volume,
                    // Across the surface, so move has a drag to follow rather
                    // than declining to do anything and passing for free.
                    &Stamp::new(point, normal, BrushDirection::Add).with_tangent(Vec3::Y),
                    &mut scratch,
                );
            }

            for step in -12..=12 {
                let probe = point + normal * step as f32;
                let value = volume.sample_world(probe);
                assert!(
                    (INSIDE..=OUTSIDE).contains(&value),
                    "{kind} left the band at {probe:?}: {value}"
                );
                assert!(value.is_finite(), "{kind} produced a non finite value");
            }
        }
    }

    #[test]
    fn smooth_reduces_roughness() {
        // Build a bumpy patch, then check smoothing flattens the variation.
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let poke = Brush {
            kind: BrushKind::Draw,
            radius: 3.0,
            strength: 0.8,
            falloff: FalloffCurve::Sharp,
            ..Brush::default()
        };
        for offset in [-6.0_f32, -2.0, 2.0, 6.0] {
            let at = point + Vec3::new(0.0, offset, 0.0);
            poke.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
        }

        let roughness = |volume: &Volume| {
            let mut total = 0.0;
            for step in -8..=8 {
                let a = volume.sample_world(point + Vec3::new(0.0, step as f32, 0.0));
                let b = volume.sample_world(point + Vec3::new(0.0, step as f32 + 1.0, 0.0));
                total += (b - a).abs();
            }
            total
        };

        let before = roughness(&volume);
        let smooth = Brush {
            kind: BrushKind::Smooth,
            radius: 10.0,
            strength: 0.9,
            falloff: FalloffCurve::Wide,
            ..Brush::default()
        };
        for _ in 0..12 {
            smooth.apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Add),
                &mut scratch,
            );
        }
        let after = roughness(&volume);

        assert!(after < before, "smoothing did not reduce roughness: {before} then {after}");
    }

    #[test]
    fn pinch_is_the_opposite_of_smooth() {
        // Smooth pulls the surface toward its local average and pinch squeezes
        // it toward the brush axis. They should not move a probe beside the
        // stroke the same way, which is what a sign error would look like.
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let probe = point + Vec3::new(0.0, 2.0, 0.0);
        let base = volume.sample_world(probe);

        let mut smoothed = sphere();
        brush(BrushKind::Smooth).apply(
            &mut smoothed,
            &Stamp::new(point, normal, BrushDirection::Add),
            &mut scratch,
        );
        let after_smooth = smoothed.sample_world(probe);

        brush(BrushKind::Pinch).apply(
            &mut volume,
            &Stamp::new(point, normal, BrushDirection::Add),
            &mut scratch,
        );
        let after_pinch = volume.sample_world(probe);

        // Smooth pulls toward the local average, pinch pushes away from it.
        assert!(
            (after_smooth - base).signum() != (after_pinch - base).signum()
                || (after_smooth - base).abs() < 1.0e-7,
            "smooth moved to {after_smooth} and pinch to {after_pinch} from {base}, same direction"
        );
    }

    #[test]
    fn flatten_pulls_a_bump_down_toward_the_plane() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let bump = Brush {
            kind: BrushKind::Draw,
            radius: 4.0,
            strength: 0.9,
            falloff: FalloffCurve::Sharp,
            ..Brush::default()
        };
        for _ in 0..4 {
            bump.apply(&mut volume, &Stamp::new(point, normal, BrushDirection::Add), &mut scratch);
        }

        // Height of the surface along the normal, found by walking outward.
        let height = |volume: &Volume| {
            let mut last = 0.0;
            for step in 0..80 {
                let t = step as f32 * 0.25;
                if volume.sample_world(point + normal * t) >= 0.0 {
                    return last;
                }
                last = t;
            }
            last
        };

        let raised = height(&volume);
        let flatten = Brush {
            kind: BrushKind::Flatten,
            radius: 8.0,
            strength: 0.8,
            falloff: FalloffCurve::Wide,
            ..Brush::default()
        };
        for _ in 0..10 {
            flatten.apply(
                &mut volume,
                &Stamp::new(point, normal, BrushDirection::Add),
                &mut scratch,
            );
        }
        assert!(height(&volume) < raised, "flatten did not bring the bump down");
    }

    #[test]
    fn clay_only_adds_when_adding() {
        // Clay's plane sits outside the surface, so without the one sided clamp
        // it would shave off anything already standing proud of that plane.
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();

        let spike = Brush {
            kind: BrushKind::Draw,
            radius: 2.5,
            strength: 1.0,
            falloff: FalloffCurve::Sharp,
            ..Brush::default()
        };
        for _ in 0..6 {
            spike.apply(&mut volume, &Stamp::new(point, normal, BrushDirection::Add), &mut scratch);
        }

        let probes: Vec<Vec3> =
            (-6..=6).map(|step| point + Vec3::new(0.0, step as f32, 0.0) + normal * 0.5).collect();
        let before: Vec<f32> = probes.iter().map(|p| volume.sample_world(*p)).collect();

        brush(BrushKind::Clay).apply(
            &mut volume,
            &Stamp::new(point, normal, BrushDirection::Add),
            &mut scratch,
        );

        for (probe, was) in probes.iter().zip(before) {
            let now = volume.sample_world(*probe);
            assert!(now <= was + 1.0e-6, "clay removed material at {probe:?}: {was} then {now}");
        }
    }

    #[test]
    fn draw_follows_the_stroke_direction_and_inflate_does_not() {
        // This is the whole difference between the two. Draw translates the
        // patch along the direction the stroke is pushing, so a stroke coming
        // in at an angle to the surface bites by the cosine of that angle.
        // Inflate offsets the level set, so every point moves along its own
        // normal by the same amount whichever way the stroke points.
        let mut scratch = BrushScratch::new();
        let point = Vec3::new(24.0, 0.0, 0.0);
        // The surface normal at `point` is along X, so this comes in at 45
        // degrees to it.
        let tilted = Vec3::new(1.0, 1.0, 0.0).normalize();

        let mut bite = |kind: BrushKind, stroke_normal: Vec3| {
            let mut volume = sphere();
            let before = volume.sample_world(point);
            let brush = Brush { kind, radius: 8.0, strength: 1.0, ..Brush::default() };
            brush.apply(
                &mut volume,
                &Stamp::new(point, stroke_normal, BrushDirection::Add),
                &mut scratch,
            );
            before - volume.sample_world(point)
        };

        let draw_square_on = bite(BrushKind::Draw, Vec3::X);
        let draw_at_an_angle = bite(BrushKind::Draw, tilted);
        assert!(draw_square_on > 0.0, "draw did nothing at all");
        assert!(
            draw_at_an_angle < draw_square_on * 0.9,
            "draw ignored the stroke direction: {draw_at_an_angle} against {draw_square_on}"
        );
        // Should land near cos 45 degrees, which is what makes it a projection
        // rather than an arbitrary reduction.
        let ratio = draw_at_an_angle / draw_square_on;
        assert!(
            (ratio - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.15,
            "the bite fell off by {ratio}, which is not the cosine of the angle"
        );

        let inflate_square_on = bite(BrushKind::Inflate, Vec3::X);
        let inflate_at_an_angle = bite(BrushKind::Inflate, tilted);
        assert!(
            (inflate_at_an_angle - inflate_square_on).abs() < 1.0e-5,
            "inflate should not care about the stroke direction: \
             {inflate_at_an_angle} against {inflate_square_on}"
        );
    }

    #[test]
    fn no_brush_carves_a_pit_where_it_should_raise_a_bump() {
        // The failure mode that only shows up on screen: a stamp big enough to
        // saturate the narrow band destroys the gradient it just wrote, and the
        // next stamp reads noise from it. The surface comes out as a crust of
        // aliased spikes while every value stays legally inside the band.
        //
        // A stroke of overlapping stamps must leave a surface that is still
        // smooth, measured here as the height varying gently across the bump.
        for kind in [
            BrushKind::Draw,
            BrushKind::Inflate,
            BrushKind::Clay,
            BrushKind::Pinch,
            BrushKind::Move,
        ] {
            let mut volume = sphere();
            let mut scratch = BrushScratch::new();
            let (point, normal) = surface(&volume);
            let brush = Brush { kind, radius: 8.0, strength: 0.9, ..Brush::default() };

            for step in -4..=4 {
                let at = point + Vec3::new(0.0, step as f32 * 2.0, 0.0);
                for _ in 0..4 {
                    brush.apply(
                        &mut volume,
                        &Stamp::new(at, normal, BrushDirection::Add).with_tangent(Vec3::Y),
                        &mut scratch,
                    );
                }
            }

            // Walk across the worked area and measure the surface height at
            // each step. Neighbouring samples should differ by a fraction of a
            // voxel, not leap about.
            let heights: Vec<f32> = (-8..=8)
                .map(|step| {
                    let probe = point + Vec3::new(0.0, step as f32, 0.0);
                    let mut last = 0.0;
                    for walk in 0..200 {
                        let t = walk as f32 * 0.1;
                        if volume.sample_world(probe + normal * t) >= 0.0 {
                            return last;
                        }
                        last = t;
                    }
                    last
                })
                .collect();

            let worst_jump =
                heights.windows(2).map(|pair| (pair[1] - pair[0]).abs()).fold(0.0_f32, f32::max);
            assert!(
                worst_jump < 2.0,
                "{kind} left a surface that jumps {worst_jump} units between neighbouring \
                 samples, which is the crust a saturating stamp produces: {heights:?}"
            );
        }
    }

    #[test]
    fn pressure_scales_the_bite() {
        let (point, normal) = surface(&sphere());
        let mut scratch = BrushScratch::new();
        let brush = brush(BrushKind::Draw);

        let mut bite = |pressure: f32| {
            let mut volume = sphere();
            let before = volume.sample_world(point);
            let stamp = Stamp::new(point, normal, BrushDirection::Add).with_pressure(pressure);
            brush.apply(&mut volume, &stamp, &mut scratch);
            before - volume.sample_world(point)
        };

        let light = bite(0.25);
        let full = bite(1.0);
        assert!(light > 0.0, "a light touch should still cut");
        assert!(light < full, "pressure did not scale the stroke: {light} against {full}");
        assert_eq!(bite(0.0), 0.0, "zero pressure should do nothing");
    }

    #[test]
    fn x_symmetry_mirrors_the_stroke_across_the_origin() {
        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let at = Vec3::new(12.0, 20.0, 0.0);
        let normal = volume.gradient_world(at);
        let mirrored_probe = Vec3::new(-at.x, at.y, at.z);
        let before = volume.sample_world(mirrored_probe);

        brush(BrushKind::Draw).apply_symmetric(
            &mut volume,
            &Stamp::new(at, normal, BrushDirection::Add),
            Symmetry::X,
            Vec3::ZERO,
            &mut scratch,
        );

        assert!(
            volume.sample_world(mirrored_probe) < before,
            "the mirrored half of the stroke never landed"
        );
    }

    /// The reason symmetry became a set: the combinations are the useful part.
    #[test]
    fn each_axis_mirrors_across_its_own_plane_and_combinations_reach_every_octant() {
        let stamp =
            Stamp::new(Vec3::new(3.0, 5.0, 7.0), Vec3::new(1.0, 0.0, 0.0), BrushDirection::Add);
        let mut twins = [stamp; Symmetry::MAX_MIRRORS];

        // One plane gives one twin, with exactly that component negated.
        for (axis, expected) in [
            (MirrorAxis::X, Vec3::new(-3.0, 5.0, 7.0)),
            (MirrorAxis::Y, Vec3::new(3.0, -5.0, 7.0)),
            (MirrorAxis::Z, Vec3::new(3.0, 5.0, -7.0)),
        ] {
            let count = Symmetry::OFF.with_axis(axis, true).mirrors(&stamp, Vec3::ZERO, &mut twins);
            assert_eq!(count, 1, "{} alone should give one twin", axis.label());
            assert_eq!(twins[0].centre, expected);
        }

        // Two planes give three twins, three planes give seven: every octant
        // except the one the original is already in.
        let two = Symmetry::OFF.with_axis(MirrorAxis::X, true).with_axis(MirrorAxis::Y, true);
        assert_eq!(two.mirrors(&stamp, Vec3::ZERO, &mut twins), 3);

        let all = Symmetry { enabled: [true; 3] };
        let count = all.mirrors(&stamp, Vec3::ZERO, &mut twins);
        assert_eq!(count, Symmetry::MAX_MIRRORS);

        let mut octants: Vec<[bool; 3]> = twins[..count]
            .iter()
            .map(|twin| [twin.centre.x < 0.0, twin.centre.y < 0.0, twin.centre.z < 0.0])
            .collect();
        octants.sort();
        octants.dedup();
        assert_eq!(octants.len(), 7, "the twins overlapped instead of covering seven octants");
        assert!(
            !octants.contains(&[false, false, false]),
            "a twin landed on top of the original stamp"
        );
    }

    #[test]
    fn a_mirrored_normal_is_reflected_along_with_the_position() {
        let stamp =
            Stamp::new(Vec3::new(4.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), BrushDirection::Add);
        let mut twins = [stamp; Symmetry::MAX_MIRRORS];
        Symmetry::X.mirrors(&stamp, Vec3::ZERO, &mut twins);

        // Still pointing out of the sphere, not back into it. Reflecting the
        // position without the normal would make the mirrored stamp carve.
        assert_eq!(twins[0].normal, Vec3::new(-1.0, 0.0, 0.0));
        assert!(twins[0].normal.dot(twins[0].centre) > 0.0);
    }

    /// The centre parameter, measured somewhere it can actually be wrong.
    ///
    /// Every call site passes the lattice origin today, so with the centre at
    /// zero a mirror that ignored the parameter entirely would pass every other
    /// test in this file. This is the one that fails the day the offset is
    /// dropped -- and the one that fails if it is applied to the normal or the
    /// tangent as well, which would send the twin's material the wrong way.
    #[test]
    fn a_mirror_about_an_offset_centre_moves_the_position_and_not_the_direction() {
        let stamp = Stamp::new(Vec3::new(12.0, 3.0, 0.0), Vec3::X, BrushDirection::Add)
            .with_tangent(Vec3::new(0.0, 0.0, 1.0));
        let mut twins = [stamp; Symmetry::MAX_MIRRORS];

        let centre = Vec3::new(10.0, 0.0, 0.0);
        assert_eq!(Symmetry::X.mirrors(&stamp, centre, &mut twins), 1);
        // Two millimetres past the plane at x = 10 goes to two before it, and
        // the axes the plane does not cross are untouched.
        assert_eq!(twins[0].centre, Vec3::new(8.0, 3.0, 0.0));
        // A direction has no position to be reflected about, so it takes the
        // sign alone: an offset here would push every twin along +x by twice
        // the centre.
        assert_eq!(twins[0].normal, Vec3::NEG_X);
        assert_eq!(twins[0].tangent, Vec3::new(0.0, 0.0, 1.0));

        // And a centre of zero is the behaviour every caller has today.
        assert_eq!(Symmetry::X.mirrors(&stamp, Vec3::ZERO, &mut twins), 1);
        assert_eq!(twins[0].centre, Vec3::new(-12.0, 3.0, 0.0));
    }

    #[test]
    fn symmetry_off_produces_no_twins_at_all() {
        let stamp = Stamp::new(Vec3::X, Vec3::X, BrushDirection::Add);
        let mut twins = [stamp; Symmetry::MAX_MIRRORS];
        assert!(Symmetry::OFF.is_off());
        assert_eq!(Symmetry::OFF.mirrors(&stamp, Vec3::ZERO, &mut twins), 0);
        assert_eq!(Symmetry::default(), Symmetry::OFF);
    }

    #[test]
    fn y_symmetry_mirrors_a_real_stroke_to_the_other_side() {
        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let at = Vec3::new(0.0, 20.0, 12.0);
        let normal = volume.gradient_world(at);
        let mirrored_probe = Vec3::new(at.x, -at.y, at.z);
        let before = volume.sample_world(mirrored_probe);

        brush(BrushKind::Draw).apply_symmetric(
            &mut volume,
            &Stamp::new(at, normal, BrushDirection::Add),
            Symmetry::OFF.with_axis(MirrorAxis::Y, true),
            Vec3::ZERO,
            &mut scratch,
        );

        assert!(
            volume.sample_world(mirrored_probe) < before,
            "the y mirrored half of the stroke never landed"
        );
    }

    #[test]
    fn the_label_names_every_enabled_plane() {
        assert_eq!(Symmetry::OFF.label(), "Off");
        assert_eq!(Symmetry::X.label(), "X");
        assert_eq!(
            Symmetry::OFF.with_axis(MirrorAxis::X, true).with_axis(MirrorAxis::Z, true).label(),
            "XZ"
        );
        assert_eq!(Symmetry { enabled: [true; 3] }.label(), "XYZ");
    }

    #[test]
    fn symmetry_off_leaves_the_other_side_untouched() {
        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let at = Vec3::new(12.0, 20.0, 0.0);
        let normal = volume.gradient_world(at);
        let mirrored_probe = Vec3::new(-at.x, at.y, at.z);
        let before = volume.sample_world(mirrored_probe);

        brush(BrushKind::Draw).apply_symmetric(
            &mut volume,
            &Stamp::new(at, normal, BrushDirection::Add),
            Symmetry::OFF,
            Vec3::ZERO,
            &mut scratch,
        );
        assert_eq!(volume.sample_world(mirrored_probe), before);
    }

    #[test]
    fn a_stamp_far_from_the_model_does_not_disturb_it() {
        let mut volume = sphere();
        let mut scratch = BrushScratch::new();
        let probe = Vec3::new(24.0, 0.0, 0.0);
        let before = volume.sample_world(probe);

        brush(BrushKind::Draw).apply(
            &mut volume,
            &Stamp::new(Vec3::splat(2000.0), Vec3::Y, BrushDirection::Add),
            &mut scratch,
        );
        assert_eq!(volume.sample_world(probe), before);
    }

    #[test]
    fn smooth_and_flatten_declare_themselves_non_directional() {
        assert!(!BrushKind::Smooth.is_directional());
        assert!(!BrushKind::Flatten.is_directional());
        assert!(BrushKind::Draw.is_directional());
    }

    #[test]
    fn stamp_spacing_never_falls_below_a_voxel() {
        let tiny = Brush { radius: 0.01, ..Brush::default() };
        assert_eq!(tiny.spacing(0.25), 0.25);
        let large = Brush { radius: 8.0, ..Brush::default() };
        assert_eq!(large.spacing(0.25), 2.0);
    }
}
#[cfg(test)]
mod snapshot_tests {
    use super::*;

    /// A brush that reports it does not read the field must genuinely not read
    /// it, proved by handing it a poisoned copy.
    ///
    /// Skipping the snapshot is only sound while that holds, and "sound because
    /// nobody currently reads it" is the kind of invariant that rots the moment
    /// somebody adds a `region.sample` to one of these arms. Poisoning the copy
    /// makes that a failing test rather than a silent read of stale data.
    #[test]
    fn reads_the_field_elsewhere_is_honest() {
        for kind in BrushKind::ALL {
            let brush = Brush {
                kind,
                radius: 6.0,
                strength: 0.7,
                falloff: FalloffCurve::Smooth,
                ..Brush::default()
            };
            if brush.reads_the_field() || kind == BrushKind::Move {
                continue;
            }

            let build = |poison: bool| {
                let mut volume = Volume::new(0.5);
                volume.seed_sphere(Vec3::ZERO, 12.0);
                volume.mark_everything_dirty();
                let mut dirty = Vec::new();
                volume.take_dirty(&mut dirty);

                let mut scratch = BrushScratch::new();
                if poison {
                    // Fill the region with a value nothing in a real field
                    // would hold, over a box that overlaps the brush. If the
                    // brush reads it, the result cannot help but differ.
                    let lo = IVec3::splat(-64);
                    let hi = IVec3::splat(64);
                    let values = scratch.region.resize(lo, hi);
                    values.fill(-999.0);
                }
                let at = Vec3::new(0.0, 0.0, 12.0);
                let normal = volume.gradient_world(at);
                brush.apply_symmetric(
                    &mut volume,
                    &Stamp::new(at, normal, BrushDirection::Add),
                    Symmetry::OFF,
                    Vec3::ZERO,
                    &mut scratch,
                );
                volume
            };

            let clean = build(false);
            let poisoned = build(true);
            for coord in clean.brick_coords() {
                let origin = coord.origin();
                for step in 0..BRICK_DIM {
                    let voxel =
                        origin + IVec3::new(step as i32, (step % 11) as i32, (step % 7) as i32);
                    assert_eq!(
                        clean.sample_voxel(voxel),
                        poisoned.sample_voxel(voxel),
                        "{kind} read the snapshot despite reporting it does not, at {voxel:?}"
                    );
                }
            }
        }
    }

    /// The counterpart: a brush that DOES report reading the field had better
    /// be affected by poisoning it, or the test above proves nothing.
    #[test]
    fn a_resampling_brush_really_does_read_the_copy() {
        let brush = Brush {
            kind: BrushKind::Draw,
            radius: 6.0,
            strength: 0.7,
            falloff: FalloffCurve::Smooth,
            ..Brush::default()
        };
        assert!(brush.reads_the_field());

        let mut volume = Volume::new(0.5);
        volume.seed_sphere(Vec3::ZERO, 12.0);
        volume.mark_everything_dirty();
        let mut dirty = Vec::new();
        volume.take_dirty(&mut dirty);
        let before = volume.sample_world(Vec3::new(0.0, 0.0, 12.0));

        let mut scratch = BrushScratch::new();
        let at = Vec3::new(0.0, 0.0, 12.0);
        let normal = volume.gradient_world(at);
        brush.apply_symmetric(
            &mut volume,
            &Stamp::new(at, normal, BrushDirection::Add),
            Symmetry::OFF,
            Vec3::ZERO,
            &mut scratch,
        );
        assert_ne!(
            volume.sample_world(Vec3::new(0.0, 0.0, 12.0)),
            before,
            "draw changed nothing, so the poisoning test above has no control"
        );
    }
}

/// The optimisation that lets a large brush cost less than its bounding box:
/// bricks it can prove it would not change are never read, never made dense
/// and never written.
///
/// Everything here is about that proof. An optimisation that changes the
/// sculpt is a bug, and this one would fail quietly if it were wrong -- a dent
/// that does not appear, on one brush, in one direction, only when the surface
/// happens to sit a couple of bricks away.
#[cfg(test)]
mod skipping_tests {
    use super::*;
    use crate::brick::{BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, OUTSIDE, brick_index};
    use crate::testing::assert_same_field;

    /// Every brush, both ways round the stroke, which is the grid the claims in
    /// [`Brush::leaves_constant_alone`] are made over.
    fn every_brush() -> impl Iterator<Item = (BrushKind, BrushDirection)> {
        BrushKind::ALL.into_iter().flat_map(|kind| {
            [BrushDirection::Add, BrushDirection::Subtract].map(move |d| (kind, d))
        })
    }

    /// A stamp with everything the directional brushes need filled in, so no
    /// brush passes a test by declining to do anything.
    fn stamp_at(centre: Vec3, normal: Vec3, direction: BrushDirection) -> Stamp {
        Stamp::new(centre, normal, direction)
            .with_tangent(normal.cross(Vec3::Z).normalize_or(Vec3::Y))
    }

    /// A solid half space, `x <= surface`, over a block of bricks wide enough
    /// that whole bricks of interior and whole bricks of empty space both sit
    /// two bricks clear of the one the surface passes through.
    ///
    /// A sphere small enough to test with quickly cannot arrange that. A brick
    /// is 32 voxels and the halo disqualifies the brick next to the surface, so
    /// a stamp has to reach two of them before it finds a constant it is
    /// allowed to skip. The slab puts that within a radius the test can afford
    /// to run, and every dense brick in it is the same brick, so building one
    /// and cloning it costs a memcpy instead of a million evaluations.
    fn slab(surface: f32) -> Volume {
        let mut volume = Volume::new(1.0);
        let span = -1..=7;
        let mut transition: Option<Brick> = None;

        for z in span.clone() {
            for y in span.clone() {
                for x in span.clone() {
                    let coord = BrickCoord::new(x, y, z);
                    let origin = coord.origin();
                    let near = origin.x as f32 - surface;
                    let far = (origin.x + BRICK_DIM as i32 - 1) as f32 - surface;
                    if far <= INSIDE {
                        volume.insert_brick(coord, Brick::Uniform(INSIDE));
                    } else if near >= OUTSIDE {
                        // Left absent, which already reads as OUTSIDE.
                    } else {
                        let brick = transition.get_or_insert_with(|| {
                            let mut data = vec![0.0_f32; BRICK_VOXELS];
                            for vz in 0..BRICK_DIM {
                                for vy in 0..BRICK_DIM {
                                    for vx in 0..BRICK_DIM {
                                        data[brick_index(vx, vy, vz)] =
                                            ((origin.x + vx as i32) as f32 - surface)
                                                .clamp(INSIDE, OUTSIDE);
                                    }
                                }
                            }
                            let data: Box<[f32; BRICK_VOXELS]> =
                                data.into_boxed_slice().try_into().expect("one brick of values");
                            Brick::Dense(data)
                        });
                        volume.insert_brick(coord, brick.clone());
                    }
                }
            }
        }
        volume.take_dirty(&mut Vec::new());
        volume
    }

    /// Where the surface sits in [`slab`], and the three places a stamp is put
    /// against it: on the surface, two bricks into the solid, and two bricks
    /// out into the empty side.
    const SURFACE: f32 = 3.0 * BRICK_DIM as f32;
    const STAND_OFF: f32 = 44.0;

    /// How many bricks one stamp visits, with and without the skipping.
    fn visits(brush: Brush, at: Vec3, direction: BrushDirection) -> (usize, usize) {
        let mut with = slab(SURFACE);
        brush.stamp(
            &mut with,
            &stamp_at(at, Vec3::X, direction),
            &mut BrushScratch::new(),
            Skipping::On,
        );
        let mut without = slab(SURFACE);
        brush.stamp(
            &mut without,
            &stamp_at(at, Vec3::X, direction),
            &mut BrushScratch::new(),
            Skipping::Off,
        );
        (with.last_visited_bricks(), without.last_visited_bricks())
    }

    #[test]
    fn a_saturated_field_is_left_alone_by_every_brush_that_says_so() {
        // The claim under test, stated the other way round: wherever
        // `leaves_constant_alone` answers yes, running the stamp anyway must
        // change nothing at all. Checked with the skipping switched OFF, so it
        // is the brush arithmetic being measured and not the shortcut agreeing
        // with itself.
        let centre = Vec3::splat(BRICK_DIM as f32 * 2.5);
        let mut ever_changed = false;

        for value in [INSIDE, OUTSIDE] {
            for (kind, direction) in every_brush() {
                // A block of bricks all holding one value. OUTSIDE is what an
                // absent brick already reads as, so leaving them out is the
                // same field and covers the absent case as well.
                let build = || {
                    let mut volume = Volume::new(1.0);
                    if value != OUTSIDE {
                        for z in 0..5 {
                            for y in 0..5 {
                                for x in 0..5 {
                                    volume.insert_brick(
                                        BrickCoord::new(x, y, z),
                                        Brick::Uniform(value),
                                    );
                                }
                            }
                        }
                    }
                    volume
                };

                let mut volume = build();
                let brush = Brush { kind, radius: 12.0, strength: 0.9, ..Brush::default() };
                brush.stamp(
                    &mut volume,
                    &stamp_at(centre, Vec3::X, direction),
                    &mut BrushScratch::new(),
                    Skipping::Off,
                );

                let untouched = build();
                let claim = brush.leaves_constant_alone(value, direction);
                if claim {
                    assert_same_field(
                        &volume,
                        &untouched,
                        &format!("{kind} {direction:?} claims it leaves a field of {value} alone"),
                    );
                } else {
                    ever_changed |= volume.brick_count() != untouched.brick_count();
                }
            }
        }

        assert!(
            ever_changed,
            "no brush changed a saturated field at all, so this test cannot tell a mistake \
             from a pass"
        );
    }

    #[test]
    fn skipping_leaves_the_same_field_and_the_same_undo_entry() {
        // Bit for bit, every brush, both directions.
        //
        // Two stamps against the slab. The first sits on the surface, where the
        // ball reaches two bricks into the solid and the deep ones are a
        // constant the resampling brushes can skip. The second stands off in
        // the empty side, where the far ones are a constant of the other sign.
        // Between them every branch of `leaves_constant_alone` is reached with
        // a real stamp behind it.
        let stamps = [
            Vec3::new(SURFACE, 100.0, 100.0),
            Vec3::new(SURFACE - STAND_OFF, 100.0, 100.0),
            Vec3::new(SURFACE + STAND_OFF, 100.0, 100.0),
        ];

        for (kind, direction) in every_brush() {
            let brush = Brush { kind, radius: 48.0, strength: 0.5, ..Brush::default() };

            let run = |skipping| {
                let mut volume = slab(SURFACE);
                volume.begin_stroke();
                let mut scratch = BrushScratch::new();
                let mut visited = 0;
                for at in stamps {
                    brush.stamp(
                        &mut volume,
                        &stamp_at(at, Vec3::X, direction),
                        &mut scratch,
                        skipping,
                    );
                    visited += volume.last_visited_bricks();
                }
                (volume, visited)
            };

            let (mut skipped, visited_when_skipping) = run(Skipping::On);
            let (mut whole, visited_when_not) = run(Skipping::Off);

            // Correctness first. Skipping that changes the sculpt is a bug;
            // skipping that saves nothing is only a missed optimisation, and
            // checking the cheap-but-less-important property first would hide
            // the expensive-but-vital one.
            assert_same_field(&skipped, &whole, &format!("{kind} {direction:?}"));
            assert!(
                visited_when_skipping < visited_when_not,
                "{kind} {direction:?} skipped nothing: {visited_when_skipping} bricks either way"
            );

            match (skipped.end_stroke(), whole.end_stroke()) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.len(), b.len(), "{kind} {direction:?} recorded a different undo");
                    assert_eq!(a.bytes(), b.bytes(), "{kind} {direction:?} undo entry differs");
                }
                (None, None) => {}
                (a, b) => panic!(
                    "{kind} {direction:?} recorded an undo entry one way and not the other: \
                     {} against {}",
                    a.is_some(),
                    b.is_some()
                ),
            }
        }
    }

    #[test]
    fn a_stroke_of_overlapping_stamps_survives_the_skipping() {
        // The single stamp cases above cannot catch a brick that is safely
        // skipped once and then has to be picked up again when a later stamp
        // moves the surface into it. Cheap enough at a small radius to run as
        // a real stroke.
        for (kind, direction) in every_brush() {
            let brush = Brush { kind, radius: 9.0, strength: 0.8, ..Brush::default() };
            let run = |skipping| {
                let mut volume = Volume::new(1.0);
                volume.seed_sphere(Vec3::splat(48.0), 30.0);
                volume.take_dirty(&mut Vec::new());
                volume.begin_stroke();
                let mut scratch = BrushScratch::new();
                for step in 0..24 {
                    let angle = step as f32 / 24.0 * std::f32::consts::TAU;
                    let at = Vec3::splat(48.0) + Vec3::new(angle.cos(), angle.sin(), 0.35) * 30.0;
                    let normal = volume.gradient_world(at);
                    brush.stamp(
                        &mut volume,
                        &stamp_at(at, normal, direction),
                        &mut scratch,
                        skipping,
                    );
                }
                volume
            };

            let mut skipped = run(Skipping::On);
            let mut whole = run(Skipping::Off);
            assert_same_field(&skipped, &whole, &format!("{kind} {direction:?} over a stroke"));
            assert_eq!(
                skipped.end_stroke().map(|edit| (edit.len(), edit.bytes())),
                whole.end_stroke().map(|edit| (edit.len(), edit.bytes())),
                "{kind} {direction:?} recorded a different undo entry over a stroke"
            );
        }
    }

    #[test]
    fn the_constant_skip_reaches_past_what_a_radius_cull_alone_would() {
        // Two things at once. That the constant skip is doing anything at all
        // -- a brush that uses it has to visit strictly fewer bricks than one
        // that cannot, at the same radius in the same place. And that
        // inflate's answer really does turn on the direction: it can leave
        // solid interior alone only while it is adding and empty space alone
        // only while it is carving, and getting that backwards would erode a
        // model from the inside where nobody looks.
        //
        // The control is inflate pushed the way it cannot skip, which is the
        // one brush and direction left with no constant test of any kind, so
        // what it visits is what the radius cull alone leaves behind. Flatten
        // used to serve as that control and no longer can: it proves its own
        // constants against the plane it blends toward. A control has to be
        // something that provably lacks the feature, not something that
        // happens not to have it yet.
        let brush = |kind| Brush { kind, radius: 48.0, strength: 0.5, ..Brush::default() };
        let deep = Vec3::new(SURFACE - STAND_OFF, 100.0, 100.0);
        let clear = Vec3::new(SURFACE + STAND_OFF, 100.0, 100.0);

        for (where_, at, saturating) in [
            ("into the solid", deep, BrushDirection::Add),
            ("out in the open", clear, BrushDirection::Subtract),
        ] {
            let (control, whole) = visits(brush(BrushKind::Inflate), at, saturating.inverted());
            assert!(control < whole, "the radius cull did nothing, so there is no control here");

            // Move is in here too. It proves its constants against the locked
            // copy rather than against the brick's neighbours -- see
            // `MoveStroke` -- but the property being asserted is the same one:
            // a brush that can leave a saturated region alone has to visit
            // strictly fewer bricks than the radius cull alone leaves behind.
            for kind in [
                BrushKind::Draw,
                BrushKind::Smooth,
                BrushKind::Pinch,
                BrushKind::Move,
                BrushKind::Clay,
                BrushKind::Flatten,
            ] {
                let (skipped, _) = visits(brush(kind), at, saturating);
                assert!(
                    skipped < control,
                    "{kind} {where_}: skipped {skipped} of {control}, so the constant test \
                     bought nothing over the radius cull"
                );
            }

            let (with_grain, _) = visits(brush(BrushKind::Inflate), at, saturating);
            assert!(
                with_grain < control,
                "inflate {where_} should leave a constant it can only push against the clamp \
                 alone, and visit less than the same brush pushed the other way"
            );
        }
    }

    // ------------------------------------------------------- masked skipping

    /// A sphere with enough curvature under the stamp that every one of the
    /// seven brushes has something to do there.
    ///
    /// [`slab`] cannot serve: its field is linear, so smooth's neighbour
    /// average, flatten's plane and a tangential Move warp all hand a voxel its
    /// own value straight back, and a control that asks "did the unmasked stamp
    /// change anything" would fail on three brushes for reasons that have
    /// nothing to do with masking.
    fn ball() -> Volume {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(48.0), 30.0);
        volume.take_dirty(&mut Vec::new());
        volume
    }

    /// A point on [`ball`]'s surface, chosen off the axes so the stamp box
    /// straddles brick boundaries in two dimensions.
    const ON_THE_BALL: Vec3 = Vec3::new(66.0, 72.0, 48.0);

    /// Every voxel of every brick the inclusive box touches, set to whatever
    /// `protection` says for that voxel.
    ///
    /// WHOLE bricks, and that is the point rather than convenience: a brick
    /// protected only in part carries detail, so `MaskField::protection_fill`
    /// answers `None` and the planner cannot see it at all. Painting whole
    /// bricks is what lets one fixture put a mask the planner acts on and a
    /// mask only the per voxel multiply can act on side by side.
    fn paint_bricks(volume: &mut Volume, lo: IVec3, hi: IVec3, protection: impl Fn(IVec3) -> u8) {
        let b_lo = BrickCoord::containing(lo).0;
        let b_hi = BrickCoord::containing(hi).0;
        for bz in b_lo.z..=b_hi.z {
            for by in b_lo.y..=b_hi.y {
                for bx in b_lo.x..=b_hi.x {
                    let origin = BrickCoord::new(bx, by, bz).origin();
                    for z in 0..BRICK_DIM as i32 {
                        for y in 0..BRICK_DIM as i32 {
                            for x in 0..BRICK_DIM as i32 {
                                let cell = origin + IVec3::new(x, y, z);
                                volume.mask_mut().write(cell, protection(cell));
                            }
                        }
                    }
                }
            }
        }
    }

    /// One protection value over whole bricks.
    fn protect_bricks(volume: &mut Volume, lo: IVec3, hi: IVec3, value: u8) {
        paint_bricks(volume, lo, hi, |_| value);
    }

    /// A smooth ramp of protection, never 0 and never 255.
    ///
    /// No brick collapses to a tile and none of them qualifies for the
    /// planner's skip, which is what makes this the arm where the per voxel
    /// multiply is the only thing doing any work. Smooth in WORLD space rather
    /// than restarting per brick, because a ramp that restarts at a brick
    /// boundary is a step, and a step is what the mask must never carry.
    fn feathered(cell: IVec3) -> u8 {
        let across = cell.as_vec3().dot(Vec3::ONE) * 0.05;
        ((0.5 + 0.4 * across.sin()) * u8::MAX as f32) as u8
    }

    /// The largest difference between two fields over an inclusive voxel box.
    ///
    /// The counterpart to [`assert_same_field`], for the controls: a test that
    /// only ever asserts "nothing changed" passes just as well when the stamp
    /// was never applied at all.
    fn worst_difference(a: &Volume, b: &Volume, lo: IVec3, hi: IVec3) -> f32 {
        let mut worst = 0.0_f32;
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let voxel = IVec3::new(x, y, z);
                    worst = worst.max((a.sample_voxel(voxel) - b.sample_voxel(voxel)).abs());
                }
            }
        }
        worst
    }

    /// Skipping leaves the same field and the same undo entry THROUGH A MASK
    /// too, and visits fewer bricks than the same stroke unmasked.
    ///
    /// This is the answer to the question the increment asks, put as a test
    /// rather than as an argument: does a mask leave the uniform-brick skip
    /// correct, merely conservative, or wrong?
    ///
    /// Correct, and the reasoning is that every proof in `decide` is quantified
    /// over ALL weights in `0..=1` rather than over a particular one -- the
    /// radius cull is position-only, [`Brush::leaves_constant_alone`] argues
    /// from the constant value "whatever the weight and wherever inside it the
    /// read lands", `blend_toward_plane_clamps_back` states its premise as
    /// "`weight` is somewhere in `0..=1`", and Move's verdict spans the
    /// displacement box from zero outwards. A mask factor in `0..=1` turns a
    /// legal weight into another legal weight, so all four survive verbatim.
    ///
    /// It becomes conservative only in the sense that a fully protected brick
    /// was already unchangeable and nothing told the planner, which is what the
    /// new skip fixes. The third assertion is what pins that down: the masked
    /// stroke visits STRICTLY FEWER bricks than the unmasked one. A stroke over
    /// a masked region gets cheaper, not dearer.
    ///
    /// Three arms INTERLEAVED brick by brick, and every part of that sentence
    /// was arrived at by watching a weaker fixture stay green under a
    /// deliberately broken skip.
    ///
    /// The arms are fully protected tiles, which the planner skips outright;
    /// HALF protected tiles, which it can see and must not skip; and a
    /// feathered ramp, which it cannot see at all. Drop the middle one and a
    /// skip written as "any uniform mask brick" or "any protection over half"
    /// passes everything here, because a two-arm fixture holds no uniform brick
    /// between the extremes for it to bite on. Interleaving them by brick
    /// rather than laying them out in blocks is what puts a different arm in
    /// every neighbouring brick, so a slab fetched for the wrong coordinate
    /// lands on a different protection instead of the same one.
    ///
    /// The radius is large rather than the 9 the tests above use, and that is
    /// load-bearing too: at radius 9 only two bricks of the box are inside the
    /// ball at all, so the radius cull reaches the rest before the mask does
    /// and two of the three arms are never exercised.
    #[test]
    fn skipping_through_a_mask_leaves_the_same_field_and_visits_fewer_bricks() {
        let at = ON_THE_BALL;
        let radius = 24.0;
        let reach = Vec3::splat(radius + 2.0);
        let (lo, hi) = ball().voxel_bounds(at - reach, at + reach);
        let arm_of = |cell: IVec3| {
            let brick = BrickCoord::containing(cell).0;
            match (brick.x + brick.y + brick.z).rem_euclid(3) {
                0 => PROTECTED,
                1 => PROTECTED / 2,
                _ => feathered(cell),
            }
        };

        for (kind, direction) in every_brush() {
            let brush = Brush { kind, radius, strength: 0.8, ..Brush::default() };
            let normal = ball().gradient_world(at);
            let stamp = stamp_at(at, normal, direction);

            let run = |masked: bool, skipping: Skipping| {
                let mut volume = ball();
                if masked {
                    paint_bricks(&mut volume, lo, hi, arm_of);
                    // Only the two uniform arms collapse to tiles; the ramp
                    // carries detail and stays dense, which is the point.
                    volume.mask_mut().collapse();
                }
                volume.begin_stroke();
                brush.stamp(&mut volume, &stamp, &mut BrushScratch::new(), skipping);
                let visited = volume.last_visited_bricks();
                (volume, visited)
            };

            let (mut skipped, visited_when_skipping) = run(true, Skipping::On);
            let (mut whole, visited_when_not) = run(true, Skipping::Off);
            let (_, visited_unmasked) = run(false, Skipping::On);

            assert_same_field(&skipped, &whole, &format!("{kind} {direction:?} through a mask"));
            match (skipped.end_stroke(), whole.end_stroke()) {
                (Some(a), Some(b)) => {
                    assert_eq!(
                        (a.len(), a.bytes()),
                        (b.len(), b.bytes()),
                        "{kind} {direction:?} recorded a different undo entry through a mask"
                    );
                }
                (None, None) => {}
                (a, b) => panic!(
                    "{kind} {direction:?} recorded an undo entry one way and not the other \
                     through a mask: {} against {}",
                    a.is_some(),
                    b.is_some()
                ),
            }

            assert!(
                visited_when_skipping < visited_when_not,
                "{kind} {direction:?} skipped nothing through a mask: \
                 {visited_when_skipping} bricks either way"
            );
            assert!(
                visited_when_skipping < visited_unmasked,
                "{kind} {direction:?} did not get CHEAPER for being masked: \
                 {visited_when_skipping} bricks masked against {visited_unmasked} unmasked"
            );
        }
    }

    /// What shape the protection under the stamp is in.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Protection {
        /// Mask All: an empty map under inversion. O(1), the most-used masking
        /// state there is, and the one an "absent means free" reading of the
        /// mask sculpts straight through.
        Everything,
        /// Painted fully protected and then collapsed, the way end of stroke
        /// leaves it. Uniform tiles, so the planner can see it.
        Tiles,
        /// Painted fully protected and left dense. The planner cannot see it,
        /// so the per voxel multiply is the only thing standing between the
        /// brush and the material.
        Dense,
    }

    /// A fully masked block feels no brush at all, and a half masked one feels
    /// half of it.
    ///
    /// **The four skipping tests above pass under every mask bug, which is why
    /// this exists.** `skipping_leaves_the_same_field_and_the_same_undo_entry`
    /// compares On against Off through the SAME per voxel closure over fixtures
    /// carrying no mask, so an inverted sense, a slab fetched for the wrong
    /// brick, an off-by-one index or the mask ignored entirely all give
    /// identical results on both sides of it; and
    /// `the_constant_skip_reaches_past_what_a_radius_cull_alone_would` asserts
    /// only that fewer bricks were visited, never that the field is right.
    ///
    /// Three claims per brush and direction, and the third is the one that
    /// keeps the first two honest.
    ///
    /// - Fully protected: the field is bit identical to not stamping at all,
    ///   nothing was recorded for undo, and no brick was promoted to dense --
    ///   which is what [`assert_same_field`] compares the representation for.
    /// - Fully protected as tiles: nothing was even visited, so the new skip
    ///   really is firing rather than the multiply quietly doing the work.
    /// - Half protected: the field differs from BOTH the untouched fixture and
    ///   the unmasked stamp. Without it, a mask that protected everything
    ///   always would pass every assertion above.
    #[test]
    fn a_fully_masked_block_feels_no_brush_at_all() {
        let at = ON_THE_BALL;
        let radius = 9.0;
        // Everything the stamp can write, plus slack, in whole bricks.
        let reach = Vec3::splat(radius + 2.0);
        let (lo, hi) = ball().voxel_bounds(at - reach, at + reach);

        for (kind, direction) in every_brush() {
            let brush = Brush { kind, radius, strength: 0.8, ..Brush::default() };
            let untouched = ball();
            let normal = untouched.gradient_world(at);
            let stamp = stamp_at(at, normal, direction);

            // The control the other three rest on: this stamp does something.
            let mut plain = ball();
            brush.apply(&mut plain, &stamp, &mut BrushScratch::new());
            let moved = worst_difference(&plain, &untouched, lo, hi);
            assert!(
                moved > 1.0e-3,
                "{kind} {direction:?} changed nothing without a mask, so this test cannot tell \
                 a mistake from a pass"
            );

            for protection in [Protection::Everything, Protection::Tiles, Protection::Dense] {
                let mut volume = ball();
                match protection {
                    Protection::Everything => volume.mask_mut().set_inverted(true),
                    Protection::Tiles => {
                        protect_bricks(&mut volume, lo, hi, PROTECTED);
                        volume.mask_mut().collapse();
                    }
                    Protection::Dense => protect_bricks(&mut volume, lo, hi, PROTECTED),
                }

                volume.begin_stroke();
                brush.apply(&mut volume, &stamp, &mut BrushScratch::new());
                let visited = volume.last_visited_bricks();

                assert_same_field(
                    &volume,
                    &untouched,
                    &format!("{kind} {direction:?} through a {protection:?} mask"),
                );
                assert!(
                    volume.end_stroke().is_none(),
                    "{kind} {direction:?} recorded an undo entry for a stroke it was not \
                     allowed to make, through a {protection:?} mask"
                );

                match protection {
                    // The mask kills the write, so there is nothing to visit.
                    Protection::Everything | Protection::Tiles => assert_eq!(
                        visited, 0,
                        "{kind} {direction:?} visited {visited} bricks through a {protection:?} \
                         mask, so the skip is not firing"
                    ),
                    // The planner cannot see a dense mask, so the multiply is
                    // the only thing protecting the material -- and this is the
                    // arm that would catch it being dropped.
                    Protection::Dense => assert!(
                        visited > 0,
                        "{kind} {direction:?} skipped a dense mask, so the multiply was never \
                         exercised"
                    ),
                }
            }

            // Half protected differs from both, which is what stops a mask that
            // protects everything from passing this whole test.
            let mut half = ball();
            protect_bricks(&mut half, lo, hi, PROTECTED / 2);
            half.mask_mut().collapse();
            brush.apply(&mut half, &stamp, &mut BrushScratch::new());
            let from_untouched = worst_difference(&half, &untouched, lo, hi);
            let from_plain = worst_difference(&half, &plain, lo, hi);
            assert!(
                from_untouched > 1.0e-3,
                "{kind} {direction:?} did nothing at all through a half mask"
            );
            assert!(
                from_plain > 1.0e-3,
                "{kind} {direction:?} ignored a half mask: it did exactly what it does unmasked"
            );
        }
    }

    /// A ball small enough that one brush can empty whole bricks of it, sitting
    /// on the brick lattice so that emptying it saturates bricks rather than
    /// leaving every one of them straddling the surface.
    fn small_ball() -> Volume {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::splat(16.0), 20.0);
        volume.take_dirty(&mut Vec::new());
        volume
    }

    /// A solid half space `x <= 0`, as a real distance field.
    ///
    /// [`slab`] cannot serve here either, and for a second reason: it clones one
    /// transition brick into every brick column that straddles the surface, so
    /// its zero crossing is not where `surface` says it is. That is harmless to
    /// the On-against-Off comparisons it was written for, which only ever ask
    /// the two paths to agree, and fatal to anything that measures where the
    /// surface actually ended up.
    fn half_space() -> Volume {
        let mut volume = Volume::new(1.0);
        volume.edit_voxels(IVec3::new(-24, -40, -40), IVec3::new(32, 40, 40), |_, position, _| {
            position.x
        });
        volume.take_dirty(&mut Vec::new());
        volume
    }

    /// A masked carving stroke still saturates the bricks it is allowed to
    /// carve, so they still qualify for the collapse that releases their
    /// 128 KB.
    ///
    /// The failure this rules out is a slow one rather than a wrong picture:
    /// masking that leaves bricks dense and part way through the band where an
    /// unmasked carve would have emptied them, so the allocations pile up over
    /// a session and nothing ever releases them.
    #[test]
    fn a_masked_carving_stroke_still_leaves_bricks_is_collapsible_accepts() {
        let at = Vec3::splat(16.0);
        // Wide enough that whole bricks sit well inside the falloff, which is
        // what it takes to saturate one: a brick is 32 voxels across, so its
        // corners are 27.7 from its centre and a brush that only just covers it
        // leaves them at a weight that never fills the band.
        let brush =
            Brush { kind: BrushKind::Inflate, radius: 48.0, strength: 0.9, ..Brush::default() };

        let carve = |masked: bool| {
            let mut volume = small_ball();
            if masked {
                // The near half in X, so the stroke has both something it may
                // carve and something it may not.
                protect_bricks(
                    &mut volume,
                    IVec3::new(-8, -8, -8),
                    IVec3::new(15, 40, 40),
                    PROTECTED,
                );
                volume.mask_mut().collapse();
            }
            let mut scratch = BrushScratch::new();
            for _ in 0..20 {
                brush.apply(
                    &mut volume,
                    &stamp_at(at, Vec3::X, BrushDirection::Subtract),
                    &mut scratch,
                );
            }
            let collapsible = volume
                .brick_coords()
                .filter(|coord| match volume.brick(*coord) {
                    Some(brick @ Brick::Dense(_)) => brick.is_collapsible().is_some(),
                    _ => false,
                })
                .count();
            (collapsible, volume.stats().dense_bricks)
        };

        let (plain_collapsible, plain_dense) = carve(false);
        let (masked_collapsible, masked_dense) = carve(true);

        assert!(
            plain_collapsible > 0,
            "the unmasked carve saturated nothing, so this test cannot tell a mistake from a pass"
        );
        assert!(
            masked_collapsible > 0,
            "a masked carve left no dense brick the collapse would accept, against \
             {plain_collapsible} unmasked"
        );
        assert!(
            masked_dense <= plain_dense,
            "masking made MORE dense bricks than not masking at all: {masked_dense} against \
             {plain_dense}"
        );
    }

    /// Thirty overlapping stamps through a half mask move the surface about
    /// half as far.
    ///
    /// A QUANTITATIVE claim, which is the whole of its value: "the masked
    /// result lies between the old surface and the new one" is satisfied by a
    /// mask read at the wrong strength, a mask applied on the first stamp and
    /// then forgotten, or a mask resolved from a neighbouring brick over a
    /// stroke that walks across brick boundaries. All three move the ratio well
    /// outside the window below, and the last of them is the reason this is
    /// thirty stamps rather than one.
    ///
    /// **It does NOT catch the comb-filter spelling of the multiply, and the
    /// plan expected it to.** Measured this session: replacing the scaled
    /// displacement with `value + (drawn - value) * free` leaves this ratio
    /// unchanged to three figures, because a domain warp over a locally linear
    /// field differs from a blend toward that same warp only to second order in
    /// the shift, and Draw's shift is a fraction of a voxel. The blend IS
    /// caught, by
    /// `a_masked_move_gesture_ends_where_the_drag_ended_however_many_events_it_took`,
    /// where a whole gesture's displacement compounds across events instead of
    /// being recomputed from a locked copy -- also measured, at 0.33 against a
    /// tolerance of 0.001.
    #[test]
    fn thirty_stamps_through_a_half_mask_move_the_surface_about_half_as_far() {
        let at = Vec3::ZERO;
        let radius = 12.0;
        let reach = Vec3::splat(radius + 2.0);
        let (lo, hi) = half_space().voxel_bounds(at - reach, at + reach);
        let brush = Brush { kind: BrushKind::Draw, radius, strength: 0.08, ..Brush::default() };

        // Where the surface sits along X at a point across the stamp, found by
        // marching rather than assumed, because the whole question is where the
        // surface ended up.
        let surface_x = |volume: &Volume, y: f32| {
            let mut last = -16.0_f32;
            for step in 0..4000 {
                let x = -16.0 + step as f32 * 0.01;
                if volume.sample_world(Vec3::new(x, y, 0.0)) >= 0.0 {
                    return last;
                }
                last = x;
            }
            last
        };

        // Peak to trough of the profile across the stamp: the crest under the
        // brush against the untouched surface out past its rim.
        let peak_to_trough = |volume: &Volume| {
            let mut lowest = f32::MAX;
            let mut highest = f32::MIN;
            for step in -16..=16 {
                let height = surface_x(volume, step as f32);
                lowest = lowest.min(height);
                highest = highest.max(height);
            }
            highest - lowest
        };

        let stroke = |protection: Option<u8>| {
            let mut volume = half_space();
            if let Some(value) = protection {
                protect_bricks(&mut volume, lo, hi, value);
                volume.mask_mut().collapse();
            }
            let mut scratch = BrushScratch::new();
            for _ in 0..30 {
                brush.apply(&mut volume, &stamp_at(at, Vec3::X, BrushDirection::Add), &mut scratch);
            }
            peak_to_trough(&volume)
        };

        // 128 and not 127.5: protection is an integer, so the closest a half
        // mask gets to exactly half is one part in 255 away from it.
        const HALF: u8 = 128;
        let free = (u8::MAX - HALF) as f32 / u8::MAX as f32;

        let whole = stroke(None);
        let halved = stroke(Some(HALF));
        assert!(whole > 0.5, "the unmasked stroke barely moved the surface: {whole} voxels");

        // A few percent and not a rounding: measured at 0.520 against the 0.498
        // the mask asks for. The residual is the warp saturating, not the mask
        // being wrong -- the full strength stroke carries the surface far
        // enough that the falloff where it now sits has fallen away, so it
        // advances slightly less than linearly while the half strength one
        // stays in the linear regime. The strength is deliberately low for that
        // reason: at 0.3 the same measurement gives 0.61, which says more about
        // saturation than about the mask.
        let ratio = halved / whole;
        assert!(
            (ratio - free).abs() < 0.05,
            "thirty stamps through a mask leaving {free} of the brush moved the surface \
             {halved} voxels against {whole} unmasked, a ratio of {ratio}"
        );
    }
}

#[cfg(test)]
mod move_tests {
    use glam::IVec3;

    use super::*;
    use crate::brick::{INSIDE, OUTSIDE};

    /// Where the sphere's surface is along a ray from the origin through
    /// `(24, y, 0)`, which is how a bump is measured without assuming it has
    /// stayed put.
    fn surface_radius(volume: &Volume, y: f32) -> f32 {
        let direction = Vec3::new(24.0, y, 0.0).normalize();
        let mut last = 0.0;
        for step in 0..400 {
            let t = 20.0 + step as f32 * 0.05;
            if volume.sample_world(direction * t) >= 0.0 {
                return last;
            }
            last = t;
        }
        last
    }

    /// Where along Y the raised material sits, weighted by how much of it there
    /// is. A drag should carry this along with it.
    fn bump_centre(volume: &Volume) -> f32 {
        bump(volume).0
    }

    /// Where the raised material sits along Y, and how much of it there is.
    fn bump(volume: &Volume) -> (f32, f32) {
        let mut weighted = 0.0;
        let mut total = 0.0;
        for step in -12..=12 {
            let y = step as f32;
            let raised = (surface_radius(volume, y) - 24.0).max(0.0);
            weighted += y * raised;
            total += raised;
        }
        assert!(total > 0.0, "there is no bump to measure");
        (weighted / total, total)
    }

    /// A sphere with a lump on it at `(24, 0, 0)`, which is something a drag
    /// can visibly carry. Dragging an unblemished sphere is a no op by
    /// definition, because a sphere slid sideways is the same sphere.
    fn sphere_with_a_bump() -> Volume {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 24.0);
        let mut scratch = BrushScratch::new();
        let poke = Brush {
            kind: BrushKind::Draw,
            radius: 5.0,
            strength: 0.9,
            falloff: FalloffCurve::Sharp,
            ..Brush::default()
        };
        let at = Vec3::new(24.0, 0.0, 0.0);
        for _ in 0..6 {
            poke.apply(&mut volume, &Stamp::new(at, Vec3::X, BrushDirection::Add), &mut scratch);
        }
        volume
    }

    /// A field rising steadily along X at `slope` per voxel, over a box wide
    /// enough to hold a stamp and everything it can read.
    ///
    /// Not a distance field, deliberately. A warp of a linear field is exactly
    /// a subtraction, so the answer a resample should give is known in closed
    /// form and any deviation is the resample reading the wrong place. The
    /// slope is kept small enough that nothing clips against the narrow band.
    fn ramp_along_x(slope: f32) -> Volume {
        let mut volume = Volume::new(1.0);
        let half = 24;
        volume.edit_voxels(IVec3::splat(-half), IVec3::splat(half), |_, position, _| {
            position.x * slope
        });
        volume
    }

    /// One gesture: press at `at`, then drag through each waypoint in turn.
    ///
    /// This is what the application does. Every event re-warps the copy locked
    /// by the press, so the waypoints in between change what the user sees
    /// while the drag is happening and leave no trace in where it ends up.
    fn gesture(volume: &mut Volume, brush: &Brush, at: Vec3, through: &[Vec3]) {
        let mut stroke = MoveStroke::new();
        stroke.begin(volume, brush, at, Symmetry::OFF, Vec3::ZERO);
        for point in through {
            stroke.drag_to(volume, *point, 1.0);
        }
        stroke.end();
    }

    #[test]
    fn dragging_carries_the_material_along_the_drag() {
        // The whole point of the brush.
        let mut volume = sphere_with_a_bump();
        let before = bump_centre(&volume);

        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 0.8, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);
        gesture(&mut volume, &brush, at, &[at + Vec3::Y * 4.0]);

        let after = bump_centre(&volume);
        assert!(
            after > before + 0.5,
            "the bump did not travel with the drag: {before} then {after}"
        );

        // And the other way puts it back on the other side of where it began,
        // which rules out a drift that happens to point at plus Y.
        let mut volume = sphere_with_a_bump();
        gesture(&mut volume, &brush, at, &[at - Vec3::Y * 4.0]);
        assert!(
            bump_centre(&volume) < before - 0.5,
            "the drag ignored its own direction: {before} then {}",
            bump_centre(&volume)
        );
    }

    #[test]
    fn a_drag_out_and_back_returns_the_form() {
        // The property locking the field at the start of a gesture exists for,
        // and the one the old incremental version could not have: the way back
        // is not undoing the way out, it is the same warp of the same locked
        // copy by a drag that has shrunk to nothing. Nothing accumulates, so
        // nothing is lost on the way.
        //
        // Not bit exact, and it cannot be -- every event resamples the locked
        // field with trilinear interpolation, and the final one resamples it at
        // a coordinate that is a float division away from the voxel it started
        // on. A thousandth of a voxel is the size of that, and it is three
        // orders of magnitude below what the old version left behind.
        let original = sphere_with_a_bump();
        let mut volume = sphere_with_a_bump();
        let started_at = bump_centre(&original);
        let (_, started_with) = bump(&original);

        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 0.8, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);

        // Out in several events, so this is a gesture and not a single warp,
        // then all the way back to where the button went down.
        let mut waypoints: Vec<Vec3> =
            (1..=8).map(|step| at + Vec3::Y * step as f32 * 0.5).collect();
        waypoints.extend((0..=8).rev().map(|step| at + Vec3::Y * step as f32 * 0.5));
        gesture(&mut volume, &brush, at, &waypoints);

        let mut worst = 0.0_f32;
        for step in -10..=10 {
            for out in -3..=3 {
                let probe = Vec3::new(24.0 + out as f32, step as f32, 0.0);
                worst =
                    worst.max((volume.sample_world(probe) - original.sample_world(probe)).abs());
            }
        }
        assert!(
            worst < 1.0e-3,
            "a drag out and back did not return the form: worst difference {worst}"
        );

        let (ended_at, ended_with) = bump(&volume);
        assert!(
            (ended_at - started_at).abs() < 0.05,
            "the bump did not come back to where it started: {started_at} then {ended_at}"
        );
        assert!(
            (ended_with - started_with).abs() < started_with * 0.01,
            "the round trip lost material: {started_with} then {ended_with}"
        );
    }

    #[test]
    fn the_surface_follows_the_pointer_by_the_whole_drag() {
        // Not "something changed" -- how far. On a ramp of known slope a domain
        // warp is exactly a subtraction, so the change in value at a voxel,
        // divided by the slope, is how far the field moved under it.
        //
        // At the brush centre the falloff is 1, so the answer is the whole
        // drag. Anything less and the surface is lagging behind the pointer,
        // which is exactly what the incremental version did.
        let slope = 0.05;
        for radius in [3.0_f32, 8.0, 20.0] {
            let brush = Brush { kind: BrushKind::Move, radius, strength: 1.0, ..Brush::default() };
            // Comfortably inside the cap, so this is measuring the follow and
            // not the clamp.
            let drag = brush.max_drag() * 0.5;

            let before = ramp_along_x(slope);
            let mut volume = ramp_along_x(slope);
            gesture(&mut volume, &brush, Vec3::ZERO, &[Vec3::X * drag]);

            let moved = (before.sample_world(Vec3::ZERO) - volume.sample_world(Vec3::ZERO)) / slope;
            assert!(
                (moved - drag).abs() < 0.05,
                "a radius {radius} brush dragged {drag} moved the field {moved}"
            );
        }
    }

    #[test]
    fn strength_scales_how_much_of_the_drag_the_surface_follows() {
        let slope = 0.05;
        let follow = |strength: f32| {
            let brush = Brush { kind: BrushKind::Move, radius: 12.0, strength, ..Brush::default() };
            let before = ramp_along_x(slope);
            let mut volume = ramp_along_x(slope);
            gesture(&mut volume, &brush, Vec3::ZERO, &[Vec3::X * 4.0]);
            (before.sample_world(Vec3::ZERO) - volume.sample_world(Vec3::ZERO)) / slope
        };

        // Both are under the 6 mm cap a radius 12 smooth brush has, so this is
        // strength doing the scaling rather than the clamp.
        assert!((follow(1.0) - 4.0).abs() < 0.05, "full strength did not follow: {}", follow(1.0));
        assert!((follow(0.5) - 2.0).abs() < 0.05, "half strength did not halve: {}", follow(0.5));
    }

    #[test]
    fn dragging_past_the_cap_stops_the_surface_rather_than_tearing_it() {
        // What the user sees when they ask for more than one gesture can give.
        // It has to stop, and it has to stop in a shape that is still a shape:
        // the failure to design against is a rim value smeared across the
        // brush, which stays legally inside the narrow band and looks like
        // geometry.
        let slope = 0.05;
        let brush = Brush { kind: BrushKind::Move, radius: 8.0, strength: 1.0, ..Brush::default() };
        let cap = brush.max_drag();

        let before = ramp_along_x(slope);
        let at_the_cap = {
            let mut volume = ramp_along_x(slope);
            gesture(&mut volume, &brush, Vec3::ZERO, &[Vec3::X * cap]);
            volume
        };
        let far_past_it = {
            let mut volume = ramp_along_x(slope);
            gesture(&mut volume, &brush, Vec3::ZERO, &[Vec3::X * cap * 20.0]);
            volume
        };

        for z in -4..=4 {
            for y in -10..=10 {
                for x in -14..=14 {
                    let probe = Vec3::new(x as f32, y as f32, z as f32);
                    let stopped = at_the_cap.sample_world(probe);
                    let dragged = far_past_it.sample_world(probe);
                    assert!(
                        (stopped - dragged).abs() < 1.0e-4,
                        "dragging twenty times past the cap kept moving at {probe:?}: \
                         {stopped} against {dragged}"
                    );
                }
            }
        }

        // And what it stopped at is the cap, reached from the field rather than
        // from the rim: a clamped read would give the value at the edge of the
        // box, which for a radius 8 brush is about -0.45.
        let moved =
            (before.sample_world(Vec3::ZERO) - far_past_it.sample_world(Vec3::ZERO)) / slope;
        assert!(
            (moved - cap).abs() < 0.05,
            "the field stopped {moved} from where it started rather than at the {cap} cap"
        );
    }

    #[test]
    fn a_warp_never_reads_outside_the_copy_it_locked() {
        // The silent failure the cap is what protects against. FieldRegion::get
        // clamps a read outside the snapshot to its edge instead of panicking,
        // so a read that overshoots does not crash and does not produce
        // garbage: it smears the rim value across the brush, and every value it
        // writes is still legally inside the narrow band.
        //
        // The cap is set so that `u * radius + drag * w(u)` rises with `u` and
        // therefore peaks at the rim, where the falloff is zero and it is
        // exactly the radius. This measures that: on a ramp the value at a
        // voxel says precisely where its source was, and a clamped read would
        // report the rim instead.
        let slope = 0.1;
        for falloff in FalloffCurve::ALL {
            let radius = 6.0;
            let brush =
                Brush { kind: BrushKind::Move, radius, strength: 1.0, falloff, ..Brush::default() };
            let drag = brush.max_drag();

            let mut volume = ramp_along_x(slope);
            gesture(&mut volume, &brush, Vec3::ZERO, &[Vec3::X * drag]);

            for step in -6..=6 {
                let probe = Vec3::new(step as f32, 0.0, 0.0);
                let weight = falloff.weight(probe.length() / radius);
                let expected = (probe.x - drag * weight) * slope;
                let measured = volume.sample_world(probe);
                assert!(
                    (measured - expected).abs() < 0.02,
                    "{falloff} read from the wrong place at {probe:?}: {measured} against \
                     the {expected} a read from {} back gives",
                    drag * weight
                );
            }
        }
    }

    #[test]
    fn one_gesture_never_moves_the_field_further_than_the_cap() {
        // Measured rather than trusted, across every falloff curve, because the
        // cap is per curve: the two cubics are three times as steep as a line
        // and fold three times as easily.
        let slope = 0.05;

        for falloff in FalloffCurve::ALL {
            for radius in [2.5_f32, 6.0, 15.0] {
                let brush = Brush {
                    kind: BrushKind::Move,
                    radius,
                    strength: 1.0,
                    falloff,
                    ..Brush::default()
                };
                let cap = brush.max_drag();
                let before = ramp_along_x(slope);
                let mut volume = ramp_along_x(slope);
                // Asking for far more than the cap allows, which is the case
                // that has to be bounded.
                gesture(&mut volume, &brush, Vec3::ZERO, &[Vec3::X * radius * 10.0]);

                let mut furthest = 0.0_f32;
                for z in -8..=8 {
                    for y in -8..=8 {
                        for x in -20..=20 {
                            let probe = Vec3::new(x as f32, y as f32, z as f32);
                            let moved = (before.sample_world(probe) - volume.sample_world(probe))
                                .abs()
                                / slope;
                            furthest = furthest.max(moved);
                        }
                    }
                }

                assert!(
                    furthest <= cap + 0.05,
                    "a radius {radius} {falloff} gesture moved the field {furthest}, past its \
                     {cap} cap"
                );
            }
        }
    }

    #[test]
    fn the_cap_keeps_every_falloff_curve_short_of_folding() {
        // The arithmetic the cap is derived from, pinned so a new curve cannot
        // be added with a slope that quietly lets its warp fold.
        for falloff in FalloffCurve::ALL {
            let brush = Brush { kind: BrushKind::Move, radius: 10.0, falloff, ..Brush::default() };
            let steepness = brush.max_drag() * falloff.max_slope() / brush.radius;
            assert!(
                steepness < 1.0,
                "{falloff} allows a warp {steepness} times as steep as the distance it is spread \
                 over, which folds the field through itself"
            );
            assert!(
                (steepness - MOVE_DRAG_MARGIN).abs() < 1.0e-5,
                "{falloff} is not using the margin the constant says it does: {steepness}"
            );
        }

        // And the slopes themselves are the derivatives they claim to be.
        for falloff in FalloffCurve::ALL {
            let mut worst = 0.0_f32;
            for step in 0..=1000 {
                let u = step as f32 / 1000.0;
                let h = 1.0e-3;
                let slope = (falloff.weight(u + h) - falloff.weight(u - h)).abs() / (2.0 * h);
                worst = worst.max(slope);
            }
            assert!(
                worst <= falloff.max_slope() + 0.02,
                "{falloff} really reaches a slope of {worst}, past the {} it declares",
                falloff.max_slope()
            );
        }
    }

    #[test]
    fn a_stroke_that_has_not_travelled_yet_drags_nothing() {
        // The first stamp of every stroke arrives with no direction of travel.
        // Picking one would drag the surface somewhere the user never pointed.
        let mut volume = sphere_with_a_bump();
        let before: Vec<f32> =
            (-8..=8).map(|step| volume.sample_world(Vec3::new(24.0, step as f32, 0.0))).collect();

        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 0.8, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);
        let mut scratch = BrushScratch::new();
        brush.apply(&mut volume, &Stamp::new(at, Vec3::X, BrushDirection::Add), &mut scratch);
        // And a locked gesture that has not been dragged anywhere either.
        gesture(&mut volume, &brush, at, &[at]);

        for (step, was) in (-8..=8).zip(before) {
            let now = volume.sample_world(Vec3::new(24.0, step as f32, 0.0));
            assert_eq!(now, was, "a directionless stamp moved the field at y {step}");
        }
    }

    #[test]
    fn a_dragged_model_still_exports_watertight() {
        let mut volume = sphere_with_a_bump();
        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 0.8, ..Brush::default() };

        // Several gestures that turn a corner, so the drag direction changes
        // under the same material rather than only ever pushing one way.
        for (at, to) in [
            (Vec3::new(24.0, 0.0, 0.0), Vec3::new(24.0, 4.0, 0.0)),
            (Vec3::new(24.0, 4.0, 0.0), Vec3::new(24.0, 4.0, 4.0)),
            (Vec3::new(23.0, 4.0, 4.0), Vec3::new(23.0, 0.0, 4.0)),
        ] {
            gesture(&mut volume, &brush, at, &[to]);
        }

        for step in -12..=12 {
            let value = volume.sample_world(Vec3::new(24.0, step as f32, 0.0));
            assert!((INSIDE..=OUTSIDE).contains(&value), "the drag left the band: {value}");
        }

        let (_, report) = volume.export_mesh();
        assert!(report.is_printable(), "a dragged model must still print: {}", report.summary());
    }

    #[test]
    fn a_mirrored_gesture_drags_the_twin_the_other_way() {
        // The mirror reflects the drag as well as the position. A twin that
        // kept the original drag would pull both halves the same way, which is
        // not symmetry.
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 24.0);
        let mut scratch = BrushScratch::new();
        let poke = Brush {
            kind: BrushKind::Draw,
            radius: 5.0,
            strength: 0.9,
            falloff: FalloffCurve::Sharp,
            ..Brush::default()
        };
        // A bump on each side, so there is something on the mirrored half for
        // the twin to carry.
        for at in [Vec3::new(24.0, 0.0, 0.0), Vec3::new(-24.0, 0.0, 0.0)] {
            let normal = at.normalize();
            for _ in 0..6 {
                poke.apply(&mut volume, &Stamp::new(at, normal, BrushDirection::Add), &mut scratch);
            }
        }

        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 1.0, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);
        let mut stroke = MoveStroke::new();
        stroke.begin(&volume, &brush, at, Symmetry::X, Vec3::ZERO);
        stroke.drag_to(&mut volume, at + Vec3::Y * 3.0, 1.0);
        stroke.end();

        // Both bumps rose along plus Y, because the mirror is across x and Y is
        // untouched by it.
        let rise = |x: f32| {
            let mut weighted = 0.0;
            let mut total = 0.0;
            for step in -12..=12 {
                let y = step as f32;
                let direction = Vec3::new(x, y, 0.0).normalize();
                let mut last = 0.0;
                for walk in 0..400 {
                    let t = 20.0 + walk as f32 * 0.05;
                    if volume.sample_world(direction * t) >= 0.0 {
                        break;
                    }
                    last = t;
                }
                let raised = (last - 24.0).max(0.0);
                weighted += y * raised;
                total += raised;
            }
            assert!(total > 0.0, "there is no bump at x {x} to measure");
            weighted / total
        };

        assert!(rise(24.0) > 0.5, "the gesture itself did not carry its bump: {}", rise(24.0));
        assert!(rise(-24.0) > 0.5, "the mirrored twin never landed: {}", rise(-24.0));
        assert!(
            (rise(24.0) - rise(-24.0)).abs() < 0.3,
            "the twin drifted a different distance from the original: {} against {}",
            rise(24.0),
            rise(-24.0)
        );
    }

    /// A masked body gets half the drag cap, and that is bought rather than
    /// free.
    ///
    /// [`Brush::max_drag`] is derived from the condition that a domain warp
    /// stops being invertible once `drag * max_slope / radius` reaches one. Under
    /// a mask the gradient the warp applies is that of `weight * free`, and the
    /// mask contributes a second term to it, so the cap has to come down or a
    /// mask edge becomes a fold in the geometry. Halved on one bool probe, so
    /// an unmasked body keeps the whole of its reach.
    ///
    /// The mask here is fully protected and nowhere near the brush, so the
    /// per voxel multiply is 1.0 everywhere the gesture writes: what is being
    /// measured is the cap and nothing else.
    #[test]
    fn a_masked_body_gives_a_move_gesture_half_the_drag_cap() {
        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 0.8, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);
        // Far past the cap, so both gestures clamp and it is the clamp being
        // compared rather than the pointer.
        let far = at + Vec3::Y * 200.0;

        let reach = |mask: bool| {
            let mut volume = sphere_with_a_bump();
            if mask {
                volume.mask_mut().write(IVec3::new(400, 400, 400), PROTECTED);
            }
            let mut stroke = MoveStroke::new();
            stroke.begin(&volume, &brush, at, Symmetry::OFF, Vec3::ZERO);
            stroke.drag_to(&mut volume, far, 1.0);
            let applied = stroke.applied().length();
            stroke.end();
            applied
        };

        let unmasked = reach(false);
        let masked = reach(true);
        assert!(
            (unmasked - brush.max_drag()).abs() < 1.0e-4,
            "the unmasked gesture did not reach its own cap: {unmasked} against {}",
            brush.max_drag()
        );
        assert!(
            (masked - unmasked * 0.5).abs() < 1.0e-4,
            "a masked body did not halve the drag cap: {masked} against {unmasked} unmasked"
        );
    }

    /// **A mask must travel with the material, not sit in the air waiting.**
    ///
    /// Reported from the running app: with the rest of a hand masked, dragging
    /// a finger towards the ring made the finger "inherit mask while moving --
    /// like there is a sticky mask in the air there". It is exactly that. The
    /// warp resamples the FIELD from `position - displacement * free` and never
    /// touches the mask, so a voxel's protection is whatever happened to be at
    /// the place the material arrived at. Drag unmasked material into masked
    /// space and it comes out masked, and no further stroke will touch it.
    ///
    /// `Volume::warped` -- the whole-body version of this same domain warp --
    /// carries the mask through it, and so do `shifted`, `rotated` and
    /// `resampled`, each with a test saying so. The brush was the one warp that
    /// did not.
    ///
    /// A PARTIAL mask ahead of the drag rather than a full one, because `free`
    /// scales the displacement: fully protected space admits no material at
    /// all, so the bug cannot be shown there. Half protected is where material
    /// arrives AND protection is waiting for it.
    ///
    /// **Reproduction, not yet a fix.** Ignored so the suite stays green while
    /// it stands as the executable description of the bug. Making it pass means
    /// locking a snapshot of the MASK beside the field and warping both by the
    /// one displacement -- `edit_voxels_where` hands the warp
    /// `(position, value, free)` and takes back a single `f32`, so there is no
    /// seam to write a mask through today. It also has to settle which mask
    /// scales the displacement: `free` is read live at the DESTINATION, which
    /// is why half-masked air brakes part of a finger and tears it, and the
    /// source's own protection is the defensible answer. The file's own warning
    /// -- the mask is read fresh per event while the copy is locked once -- is
    /// exactly the trap that change has to avoid.
    #[test]
    #[ignore = "reproduces the sticky-mask-in-the-air bug; the warp does not carry the mask yet"]
    fn material_dragged_into_masked_space_does_not_pick_up_its_mask() {
        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 0.8, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);
        let mut volume = sphere_with_a_bump();

        // Empty space just off the bump, half protected, in the direction the
        // drag will carry the material.
        let ahead = IVec3::new(34, 0, 0);
        assert_eq!(volume.mask().at(ahead), UNMASKED, "the fixture starts unmasked");
        for x in 30..40 {
            for y in -6..6 {
                for z in -6..6 {
                    volume.mask_mut().write(IVec3::new(x, y, z), PROTECTED / 2);
                }
            }
        }

        let mut stroke = MoveStroke::new();
        stroke.begin(&volume, &brush, at, Symmetry::OFF, Vec3::ZERO);
        stroke.drag_to(&mut volume, at + Vec3::X * 8.0, 1.0);
        stroke.end();

        // The material that arrived brought its own protection with it, which
        // was none. Not the half-mask that was hanging in the air.
        assert!(
            volume.mask().at(ahead) < PROTECTED / 4,
            "unmasked material dragged into masked space came out masked: {} at {ahead:?}",
            volume.mask().at(ahead)
        );
    }

    /// A masked gesture ends where the drag ended, however many pointer events
    /// it took to get there.
    ///
    /// **The suite has no analogue of this and cannot grow one by accident.**
    /// `Skipping::Off` reaches Move only through `drag_once`, which locks,
    /// drags and releases around a SINGLE event, so every equivalence test in
    /// the file exercises the one-event path. The mask is read fresh on every
    /// event while the copy is locked once, and that is precisely where a
    /// per-event accumulation would hide.
    ///
    /// The failure it rules out is the tempting spelling of the multiply:
    /// blending the warped value against the old one,
    /// `value + (warped - value) * free`, instead of scaling the displacement.
    /// Nothing about one event distinguishes the two -- both leave the surface
    /// half way -- but the blend compounds over events, so thirty of them land
    /// somewhere thirty times further on than three do, and the gesture drifts
    /// on while the pointer stands still.
    ///
    /// Both runs end on the same waypoint, and the one before it is the
    /// opposite end of the drag, so [`MOVE_SETTLE_VOXELS`] cannot swallow the
    /// last event of either. Not bit exact for the same reason
    /// `a_drag_out_and_back_returns_the_form` is not: a brick that one run
    /// classified as uniform and the other did not is resampled rather than
    /// left alone, and the two answers differ by the interpolation.
    #[test]
    fn a_masked_move_gesture_ends_where_the_drag_ended_however_many_events_it_took() {
        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 0.8, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);
        let out = |along: f32| at + Vec3::Y * along;

        // A protection ramp across the brush, feathered rather than stepped --
        // which is a rule and not a preference, because a step in the mask is a
        // fold in the geometry once the combined gradient reaches one.
        let build = || {
            let mut volume = sphere_with_a_bump();
            for z in -14..=14 {
                for y in -14..=14 {
                    for x in 10..=38 {
                        let across = ((x - 24) as f32 / 28.0 + 0.5).clamp(0.0, 1.0);
                        let protection = (across * u8::MAX as f32).round() as u8;
                        volume.mask_mut().write(IVec3::new(x, y, z), protection);
                    }
                }
            }
            volume.mask_mut().collapse();
            volume
        };

        let run = |waypoints: &[Vec3]| {
            let mut volume = build();
            gesture(&mut volume, &brush, at, waypoints);
            volume
        };

        let mut many: Vec<Vec3> =
            (0..28).map(|step| out((step as f32 * 0.7).sin() * 3.0)).collect();
        many.extend([out(-3.0), out(3.0)]);
        let few = run(&[out(3.0), out(-3.0), out(3.0)]);
        let many = run(&many);

        let mut worst = 0.0_f32;
        for z in -12..=12 {
            for y in -12..=12 {
                for x in 12..=36 {
                    let probe = IVec3::new(x, y, z);
                    worst = worst.max((few.sample_voxel(probe) - many.sample_voxel(probe)).abs());
                }
            }
        }
        assert!(
            worst < 1.0e-3,
            "a masked gesture to the same waypoint landed somewhere else depending on how many \
             events it took: worst difference {worst}"
        );

        // And the control, or the two could agree by both doing nothing: the
        // gesture moved the surface, and moved it LESS than the same gesture
        // without a mask.
        let untouched = sphere_with_a_bump();
        let mut unmasked = sphere_with_a_bump();
        gesture(&mut unmasked, &brush, at, &[out(3.0), out(-3.0), out(3.0)]);

        let mut masked_travel = 0.0_f32;
        let mut plain_travel = 0.0_f32;
        for z in -12..=12 {
            for y in -12..=12 {
                for x in 12..=36 {
                    let probe = IVec3::new(x, y, z);
                    let was = untouched.sample_voxel(probe);
                    masked_travel = masked_travel.max((few.sample_voxel(probe) - was).abs());
                    plain_travel = plain_travel.max((unmasked.sample_voxel(probe) - was).abs());
                }
            }
        }
        assert!(masked_travel > 1.0e-3, "the masked gesture moved nothing at all");
        assert!(
            masked_travel < plain_travel,
            "the mask did not hold the gesture back at all: {masked_travel} against \
             {plain_travel} unmasked"
        );
    }

    #[test]
    fn a_gesture_straddling_a_mirror_plane_still_reads_the_field() {
        // Two anchors close enough to overlap is the one case where a voxel's
        // displacement is a sum rather than a single term, and where the
        // argument that keeps every read inside the brush's own box no longer
        // covers it -- so the copies are grown by the cap for exactly this.
        //
        // Getting it wrong would not crash. FieldRegion::get clamps a read
        // outside its box to the edge, so an overshoot smears the rim value and
        // every value it writes is still legally inside the narrow band.
        //
        // The geometry is picked so the overshoot actually happens, which most
        // arrangements do not produce. The brush centre sits half a radius from
        // the plane, which puts it exactly on the FACE of its own twin's box;
        // the twin is applied last, so the twin's copy is what has to reach a
        // full cap beyond that face. On a ramp the right answer is arithmetic.
        let slope = 0.1;
        let radius = 6.0;
        let brush = Brush { kind: BrushKind::Move, radius, strength: 1.0, ..Brush::default() };
        let cap = brush.max_drag();
        let at = Vec3::new(radius * 0.5, 0.0, 0.0);

        let mut volume = ramp_along_x(slope);
        let mut stroke = MoveStroke::new();
        stroke.begin(&volume, &brush, at, Symmetry::X, Vec3::ZERO);
        // Away from the plane, so the read at `at` goes further from the twin
        // rather than back toward it.
        stroke.drag_to(&mut volume, at - Vec3::X * cap, 1.0);
        stroke.end();

        // At the brush centre the falloff is 1 and the twin's is 0, so the
        // field is read from a whole cap along plus X.
        let expected = (at.x + cap) * slope;
        let measured = volume.sample_world(at);
        assert!(
            (measured - expected).abs() < 0.02,
            "the twin's copy did not reach the field it had to read: {measured} against the \
             {expected} a read from {} gives. A copy grown only to the brush box would clamp at \
             the TWIN's rim and give about {}.",
            at.x + cap,
            (-at.x + radius + 1.0) * slope
        );
    }

    #[test]
    fn a_whole_gesture_is_one_undo_entry_however_many_events_it_takes() {
        // Re-warping the locked copy on every event writes over the same bricks
        // again and again. That is only safe because `record_for_undo` captures
        // a brick on FIRST touch, so what history holds is the field as it
        // stood before the gesture, not as the previous event left it.
        let original = sphere_with_a_bump();
        let mut volume = sphere_with_a_bump();
        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 1.0, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);

        volume.begin_stroke();
        let mut stroke = MoveStroke::new();
        stroke.begin(&volume, &brush, at, Symmetry::OFF, Vec3::ZERO);
        for step in 1..=10 {
            stroke.drag_to(&mut volume, at + Vec3::Y * step as f32 * 0.4, 1.0);
        }
        stroke.end();
        let edit = volume.end_stroke().expect("a gesture that moved the surface recorded nothing");

        assert!(bump_centre(&volume) > bump_centre(&original) + 0.5, "the gesture did nothing");

        volume.apply_edit(edit);
        for step in -10..=10 {
            for out in -3..=3 {
                let probe = Vec3::new(24.0 + out as f32, step as f32, 0.0);
                assert_eq!(
                    volume.sample_world(probe),
                    original.sample_world(probe),
                    "undoing the gesture did not restore {probe:?}"
                );
            }
        }
    }

    #[test]
    fn an_event_that_has_not_moved_the_pointer_costs_nothing() {
        // The warp is recomputed from scratch every event, so repeating one has
        // to be recognised rather than redone. Observed through the dirty set,
        // which is what a redundant pass would refill.
        let mut volume = sphere_with_a_bump();
        let brush = Brush { kind: BrushKind::Move, radius: 9.0, strength: 1.0, ..Brush::default() };
        let at = Vec3::new(24.0, 0.0, 0.0);
        let mut dirty = Vec::new();

        let mut stroke = MoveStroke::new();
        stroke.begin(&volume, &brush, at, Symmetry::OFF, Vec3::ZERO);
        stroke.drag_to(&mut volume, at + Vec3::Y * 3.0, 1.0);
        volume.take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "the first event of a gesture must do the work");

        stroke.drag_to(&mut volume, at + Vec3::Y * 3.0, 1.0);
        volume.take_dirty(&mut dirty);
        assert!(dirty.is_empty(), "the same drag was applied twice");

        // And so does dragging on past the cap, because the displacement has
        // already clamped and is no longer changing.
        stroke.drag_to(&mut volume, at + Vec3::Y * 500.0, 1.0);
        volume.take_dirty(&mut dirty);
        assert!(!dirty.is_empty(), "reaching the cap is still a change");
        stroke.drag_to(&mut volume, at + Vec3::Y * 900.0, 1.0);
        volume.take_dirty(&mut dirty);
        assert!(dirty.is_empty(), "dragging further past the cap redid the warp for nothing");
        stroke.end();
    }
}

#[cfg(test)]
mod tilt_tests {
    use super::*;

    #[test]
    fn an_upright_pen_leaves_the_normal_alone() {
        // The whole feature has to be invisible without a tablet.
        let normal = Vec3::new(0.3, 0.9, -0.2).normalize();
        assert_eq!(lean_normal(normal, Vec3::ZERO), normal);
    }

    #[test]
    fn leaning_rotates_the_normal_by_exactly_that_angle() {
        let normal = Vec3::Y;
        let angle = 0.4_f32;
        let tilted = lean_normal(normal, Vec3::X * angle);

        assert!((tilted.length() - 1.0).abs() < 1.0e-5, "the result must stay a unit vector");
        let measured = normal.dot(tilted).clamp(-1.0, 1.0).acos();
        assert!((measured - angle).abs() < 1.0e-4, "rotated by {measured} instead of {angle}");
        // And it leaned the way the pen did.
        assert!(tilted.x > 0.0, "leaned along minus X: {tilted:?}");
    }

    #[test]
    fn only_the_tangential_part_of_a_lean_steers() {
        let normal = Vec3::Y;
        // Leaning straight along the normal cannot mean anything directional.
        assert_eq!(lean_normal(normal, Vec3::Y * 0.5), normal);

        // A lean partly into the surface still steers by its tangential part,
        // rotating within the plane the pen is leaning in.
        let tilted = lean_normal(normal, Vec3::new(0.3, 0.3, 0.0));
        assert!(tilted.x > 0.0);
        assert!((tilted.length() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn opposite_leans_give_opposite_results() {
        let normal = Vec3::Z;
        let left = lean_normal(normal, Vec3::X * 0.3);
        let right = lean_normal(normal, -Vec3::X * 0.3);
        assert!((left.x + right.x).abs() < 1.0e-5, "{left:?} against {right:?}");
        assert!((left.z - right.z).abs() < 1.0e-5);
    }

    #[test]
    fn a_leaning_pen_steers_where_the_clay_goes() {
        // The point of the whole thing: the same stroke on the same spot moves
        // material to a different place when the pen is leaned.
        let mut scratch = BrushScratch::new();
        let point = Vec3::new(24.0, 0.0, 0.0);
        let brush = Brush { kind: BrushKind::Draw, radius: 8.0, strength: 1.0, ..Brush::default() };

        let mut sculpt = |lean: Vec3| {
            let mut volume = Volume::new(1.0);
            volume.seed_sphere(Vec3::ZERO, 24.0);
            let normal = lean_normal(volume.gradient_world(point), lean);
            for _ in 0..6 {
                brush.apply(
                    &mut volume,
                    &Stamp::new(point, normal, BrushDirection::Add),
                    &mut scratch,
                );
            }
            // Where the material ended up, measured to one side of the stroke.
            volume.sample_world(point + Vec3::new(0.0, 6.0, 0.0))
        };

        let upright = sculpt(Vec3::ZERO);
        let leaned_towards = sculpt(Vec3::Y * 0.7);
        let leaned_away = sculpt(-Vec3::Y * 0.7);

        assert!(
            leaned_towards < upright,
            "leaning toward the probe should have pushed clay that way: {upright} then {leaned_towards}"
        );
        assert!(
            leaned_away > leaned_towards,
            "leaning the other way should not pile material in the same place"
        );
    }
}
/// The mask brush: the tool that writes protection instead of distance.
///
/// Its own module, matching the shape of `move_tests` and `skipping_tests`
/// above -- these all run against a masked fixture, and the assertions are
/// about a different map from the one every test in `tests` is about.
#[cfg(test)]
mod mask_tests {
    use super::*;
    use crate::brick::{BRICK_DIM, BrickCoord};
    use glam::IVec3;

    fn sphere() -> Volume {
        let mut volume = Volume::new(1.0);
        volume.seed_sphere(Vec3::ZERO, 24.0);
        volume
    }

    /// A point on the surface of the seeded sphere, and its outward normal.
    fn surface(volume: &Volume) -> (Vec3, Vec3) {
        let point = Vec3::new(24.0, 0.0, 0.0);
        (point, volume.gradient_world(point))
    }

    fn brush(kind: BrushKind) -> Brush {
        Brush { kind, radius: 8.0, strength: 0.4, ..Brush::default() }
    }

    /// The one property the whole mask design rests on at this layer: painting
    /// protection changes protection and NOTHING about the field.
    ///
    /// A field that moved here would mean the mask tool had become a very quiet
    /// sculpt brush, which is the failure a user would find days later in an
    /// exported print rather than on screen.
    #[test]
    fn a_mask_stamp_changes_the_mask_and_not_one_field_value() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let before: Vec<(BrickCoord, Vec<f32>)> = volume
            .brick_coords()
            .map(|coord| {
                let origin = coord.origin();
                let values = (0..BRICK_DIM as i32)
                    .flat_map(move |z| {
                        (0..BRICK_DIM as i32).flat_map(move |y| {
                            (0..BRICK_DIM as i32).map(move |x| origin + IVec3::new(x, y, z))
                        })
                    })
                    .collect::<Vec<_>>();
                (coord, values)
            })
            .map(|(coord, cells)| {
                (coord, cells.into_iter().map(|cell| volume.sample_voxel(cell)).collect())
            })
            .collect();

        let mut scratch = BrushScratch::new();
        brush(BrushKind::Draw).apply_mask(
            &mut volume,
            &Stamp::new(point, normal, BrushDirection::Add),
            MaskOp::Raise,
            &mut scratch,
        );

        assert!(volume.mask_fill() > 0.0, "the mask stamp painted nothing at all");
        for (coord, values) in before {
            let origin = coord.origin();
            for (index, expected) in values.into_iter().enumerate() {
                let x = index % BRICK_DIM;
                let y = (index / BRICK_DIM) % BRICK_DIM;
                let z = index / (BRICK_DIM * BRICK_DIM);
                let cell = origin + IVec3::new(x as i32, y as i32, z as i32);
                assert_eq!(
                    volume.sample_voxel(cell),
                    expected,
                    "the mask stamp moved the field at {cell:?}"
                );
            }
        }
    }

    /// Raise and Lower are each other's inverse in direction, whichever brush
    /// happens to be selected.
    ///
    /// **Every brush, and that is the point.** The direction of a mask stroke is
    /// worked out by the caller and must never consult
    /// `BrushKind::is_directional`, which answers false for Smooth, Flatten and
    /// Move -- so a mask painter that reused the field's own direction rule
    /// would invert for three of the seven and there would be nothing on screen
    /// to say which three. This pins the half that lives in this crate: the op
    /// decides, and the brush kind does not enter into it.
    #[test]
    fn lowering_the_mask_undoes_raising_it_with_every_brush_selected() {
        for kind in BrushKind::ALL {
            let mut volume = sphere();
            let (point, normal) = surface(&volume);
            let mut scratch = BrushScratch::new();
            let stamp = Stamp::new(point, normal, BrushDirection::Add);

            brush(kind).apply_mask(&mut volume, &stamp, MaskOp::Raise, &mut scratch);
            let raised = volume.mask_fill();
            assert!(raised > 0.0, "{kind}: raising painted nothing");

            for _ in 0..12 {
                brush(kind).apply_mask(&mut volume, &stamp, MaskOp::Lower, &mut scratch);
            }
            assert!(
                volume.mask_fill() < raised * 0.1,
                "{kind}: lowering left {} of {raised}",
                volume.mask_fill()
            );
        }
    }

    /// Erasing a mask reaches the FREE state, not merely a small number.
    ///
    /// This is the half of the quantiser's asymmetry that a fill percentage
    /// cannot see. [`crate::MaskField::collapse`] drops a brick only when it is
    /// uniformly `UNMASKED`, so a residue of two levels left behind by a rounded
    /// erase is permanent in every way the user meets it: the body reads as
    /// masked forever, the standing card goes on naming it, Move's cap goes on
    /// being halved, and erasing again cannot help because the residue is
    /// exactly where the blend has stopped moving. That is why `Lower` floors
    /// unconditionally where `Raise` snaps only within a level of its target --
    /// see the note in `apply_mask`.
    #[test]
    fn erasing_a_mask_clears_it_to_the_free_state_and_not_to_a_residue() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();
        let stamp = Stamp::new(point, normal, BrushDirection::Add);

        brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Raise, &mut scratch);
        assert!(!volume.mask().is_free(), "raising left the mask free");

        // The same brush, at the same place, held until it has nothing left to
        // take: a weak stamp removes a level at a time by design.
        for _ in 0..64 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Lower, &mut scratch);
        }
        volume.mask_mut().collapse();
        assert!(volume.mask().is_free(), "erasing left {} behind", volume.mask_fill());
    }

    /// Repeated stamps converge on the target rather than overshooting it.
    ///
    /// This is what "blend toward a legal target with a weight in 0..=1" buys,
    /// and it is the same argument Smooth, Flatten and Clay rest on. A mask
    /// painter written as `value += weight * 255` would pass a one-stamp test
    /// and then saturate a whole brush footprint to a hard edge over a stroke,
    /// which is the step the storage design forbids.
    #[test]
    fn repeated_mask_stamps_converge_on_full_protection_and_never_pass_it() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();
        let stamp = Stamp::new(point, normal, BrushDirection::Add);
        let cell = (point / volume.voxel_size()).round().as_ivec3();

        let mut previous = 0u8;
        for pass in 0..30 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Raise, &mut scratch);
            let now = volume.mask().at(cell);
            assert!(now >= previous, "pass {pass} went backwards: {previous} then {now}");
            previous = now;
        }
        assert_eq!(previous, PROTECTED, "thirty stamps did not reach the target");
    }

    /// The rim of a mask stroke is feathered, not a step.
    ///
    /// A rule rather than a preference, with three independent justifications in
    /// `crate::mask`. The sharpest of them is that a step in the mask is a fold
    /// in the geometry under Move, which no amount of narrow-band clamping
    /// catches.
    #[test]
    fn a_mask_stroke_leaves_a_feathered_rim_rather_than_a_step() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();
        brush(BrushKind::Draw).apply_mask(
            &mut volume,
            &Stamp::new(point, normal, BrushDirection::Add),
            MaskOp::Raise,
            &mut scratch,
        );

        // Sampled along the surface away from the centre, out past the radius.
        let mut seen = Vec::new();
        for step in 0..=8 {
            let along = point + Vec3::new(0.0, step as f32, 0.0);
            let cell = (along / volume.voxel_size()).round().as_ivec3();
            seen.push(volume.mask().at(cell));
        }
        assert!(seen[0] > 0, "the centre of the stroke was not masked");
        assert_eq!(*seen.last().expect("eight samples"), UNMASKED, "the mask leaked past the rim");
        let between = seen.iter().filter(|value| (1..PROTECTED).contains(value)).count();
        assert!(between >= 3, "only {between} intermediate values: this is a step, not a feather");
    }

    /// And it still has a rim after a scrub, which the one-stamp test above
    /// cannot see.
    ///
    /// The one-stamp profile is feathered whatever the arithmetic does, because
    /// the falloff supplies the shape. What eats a feather is repetition: a
    /// quantiser that moves every touched voxel by a level per stamp drives the
    /// whole footprint to `PROTECTED` however small its weight was, and a few
    /// seconds of scrubbing is a few hundred stamps. Measured on this fixture
    /// before the fix, 300 stamps gave `[255 x 8, 0]` -- not one intermediate
    /// value left, and a full 255 step against the untouched voxel outside the
    /// radius, which is exactly the mask gradient [`mask_drag_scale`]'s
    /// half-margin assumes cannot happen.
    ///
    /// **What is asserted is what the arithmetic can actually promise**: the
    /// centre still arrives at `PROTECTED`, and the grade in between survives.
    /// The last voxel inside the radius against the first one outside is NOT
    /// bounded here and cannot be -- that discontinuity belongs to the falloff's
    /// support, not to the quantiser, and closing it needs a stroke-wide
    /// accumulation buffer rather than a rounding rule.
    #[test]
    fn three_hundred_stamps_still_leave_a_grade_between_the_centre_and_the_rim() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();
        let stamp = Stamp::new(point, normal, BrushDirection::Add);
        for _ in 0..300 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Raise, &mut scratch);
        }

        let mut seen = Vec::new();
        for step in 0..=8 {
            let along = point + Vec3::new(0.0, step as f32, 0.0);
            let cell = (along / volume.voxel_size()).round().as_ivec3();
            seen.push(volume.mask().at(cell));
        }
        assert_eq!(seen[0], PROTECTED, "scrubbing no longer reaches full protection: {seen:?}");
        let between = seen.iter().filter(|value| (1..PROTECTED).contains(value)).count();
        assert!(between >= 3, "{seen:?} has {between} intermediate values: the rim ratcheted flat");
    }

    /// Blur softens what is there and reads a SNAPSHOT rather than the values it
    /// is writing.
    ///
    /// The two-phase shape is the whole of the correctness here. Reading the
    /// live mask would make each voxel average a mixture of old and new values
    /// and the answer would depend on which voxel was visited first -- the same
    /// hazard `crate::region` exists for on the field side, which is why blur
    /// borrows that machinery rather than inventing a second copy of it.
    #[test]
    fn blurring_pulls_a_hard_mask_edge_toward_its_neighbours() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();
        let stamp = Stamp::new(point, normal, BrushDirection::Add);

        // A deliberately hard edge: a block of full protection with nothing
        // feathering it, written straight into the mask.
        for z in -3..=3 {
            for y in -3..=3 {
                for x in -3..=3 {
                    let cell =
                        (point / volume.voxel_size()).round().as_ivec3() + IVec3::new(x, y, z);
                    volume.mask_mut().write(cell, PROTECTED);
                }
            }
        }
        let edge = (point / volume.voxel_size()).round().as_ivec3() + IVec3::new(0, 3, 0);
        assert_eq!(volume.mask().at(edge), PROTECTED, "the fixture did not build an edge");

        for _ in 0..4 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Blur, &mut scratch);
        }
        let softened = volume.mask().at(edge);
        assert!(softened < PROTECTED, "the edge stayed hard at {softened}");
        assert!(softened > UNMASKED, "the blur erased the mask instead of softening it");
    }

    /// A mask stroke on a masked body is still just protection: the field the
    /// mask itself protects is never consulted, and a fully protected voxel is
    /// not immune to being UNMASKED.
    ///
    /// Worth pinning because the obvious mistake -- running the mask painter
    /// through the same `use_mask` multiply the field edits use -- makes the
    /// mask self-protecting, and a fully masked body can then never be unmasked
    /// by the tool that masked it.
    #[test]
    fn the_mask_does_not_protect_itself_from_being_edited() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();
        let stamp = Stamp::new(point, normal, BrushDirection::Add);
        let cell = (point / volume.voxel_size()).round().as_ivec3();

        for _ in 0..30 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Raise, &mut scratch);
        }
        assert_eq!(volume.mask().at(cell), PROTECTED, "the fixture did not fully protect it");

        for _ in 0..30 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Lower, &mut scratch);
        }
        assert_eq!(volume.mask().at(cell), UNMASKED, "a full mask could not be taken off again");
    }

    /// Painting protection under an inverted mask paints PROTECTION.
    ///
    /// `MaskField::write` applies polarity on both sides, and so does
    /// `edit_brick`; the trap is a painter that writes the stored byte and
    /// leaves the user painting holes in their own Mask All.
    #[test]
    fn painting_under_an_inverted_mask_still_paints_protection() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        volume.mask_mut().set_inverted(true);
        let cell = (point / volume.voxel_size()).round().as_ivec3();
        assert_eq!(volume.mask().at(cell), PROTECTED, "inversion did not protect everything");

        let mut scratch = BrushScratch::new();
        let stamp = Stamp::new(point, normal, BrushDirection::Add);
        for _ in 0..30 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Lower, &mut scratch);
        }
        assert_eq!(volume.mask().at(cell), UNMASKED, "unmasking under inversion did not free it");
        for _ in 0..30 {
            brush(BrushKind::Draw).apply_mask(&mut volume, &stamp, MaskOp::Raise, &mut scratch);
        }
        assert_eq!(volume.mask().at(cell), PROTECTED, "masking under inversion did not protect it");
    }

    /// A masked stroke over a mask is a stroke that does nothing to the field.
    ///
    /// The end-to-end claim, on the two halves together: paint protection, then
    /// sculpt through it, and the field comes back bit-identical to not having
    /// sculpted at all.
    #[test]
    fn sculpting_through_a_mask_this_brush_painted_leaves_the_field_alone() {
        let mut volume = sphere();
        let (point, normal) = surface(&volume);
        let mut scratch = BrushScratch::new();
        let stamp = Stamp::new(point, normal, BrushDirection::Add);

        // A wide mask, so the whole of the narrower sculpt brush lands inside
        // it: the rim of a feathered mask is deliberately not full protection.
        let painter = Brush { radius: 20.0, strength: 1.0, ..brush(BrushKind::Draw) };
        for _ in 0..40 {
            painter.apply_mask(&mut volume, &stamp, MaskOp::Raise, &mut scratch);
        }

        let before: Vec<f32> = (-6..=6)
            .map(|step| volume.sample_world(point + Vec3::new(0.0, step as f32, 0.0)))
            .collect();
        let carver = Brush { radius: 5.0, strength: 0.6, ..brush(BrushKind::Draw) };
        for _ in 0..5 {
            carver.apply(&mut volume, &stamp, &mut scratch);
        }
        for (index, expected) in before.into_iter().enumerate() {
            let step = index as i32 - 6;
            assert_eq!(
                volume.sample_world(point + Vec3::new(0.0, step as f32, 0.0)),
                expected,
                "a fully masked stroke moved the field at step {step}"
            );
        }
    }
}
