// SPDX-License-Identifier: AGPL-3.0-only

//! BrokkrSculpt engine: sparse brick volume, brushes and meshing.
//!
//! This crate has no UI, windowing or GPU dependencies and must keep it that
//! way. Everything about how the application looks or reads input lives in
//! `brokkr-app`, which is what keeps the shell choice reversible.
//!
//! # Shape of the sculpt loop
//!
//! 1. [`raycast`] sphere traces the field to find the point under the cursor.
//! 2. [`Stroke::advance`] walks from the last stamp to that point, so a fast
//!    drag lays a continuous line rather than a dotted one.
//! 3. [`Brush::apply_symmetric`] resolves the bricks each stamp's box touches,
//!    allocates the missing ones and edits their voxels.
//! 4. The edit marks those bricks and their apron neighbours dirty, and
//!    snapshots their prior contents for undo on first touch.
//! 5. The caller meshes only the dirty bricks with [`Volume::mesh_brick`].
//!
//! Every step is proportional to what the brush touched and not to the size of
//! the model. That is the property the whole design exists to protect.

pub mod apron;
pub mod body;
pub mod brick;
pub mod brush;
pub mod cavity;
pub mod clip;
pub mod export;
pub mod generate;
pub mod import;
pub mod mask;
pub mod merge;
pub mod mesh;
pub mod orientation;
pub mod pattern;
pub mod primitive;
pub mod project;
pub mod raycast;
pub mod redistance;
pub mod region;
pub mod resample;
pub mod rotate;
pub mod similarity;
pub mod split;
pub mod stroke;
#[cfg(test)]
mod testing;
pub mod transform;
pub mod undo;
pub mod volume;
pub mod voxelise;

pub use apron::ApronBuffer;
pub use body::{
    Document, DropRefusal, DropTarget, GrowthGuard, MAX_BODIES, MAX_DEPTH, MAX_NODES, Node, NodeId,
    NodeMeta, drop_refusal, drop_target, resolve_visibility, subtree,
};
pub use brick::{
    APRON_DIM, APRON_VOXELS, BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND,
    OUTSIDE,
};
pub use brush::{
    Brush, BrushDirection, BrushKind, BrushScratch, FalloffCurve, MaskOp, MirrorAxis, MoveStroke,
    Stamp, Symmetry, lean_normal,
};
pub use clip::{ClipCounts, ClipPlane, CutOutcome};
pub use export::{ExportMesh, ExportedBody, MeshReport, document_verdict};
pub use generate::{MAX_THICKNESS_VOXELS, MaskRecipe};
pub use import::{ImportError, MESH_EXTENSIONS};
pub use mask::{MaskBrick, MaskField, MaskFilter, MaskSlab, PROTECTED, UNMASKED};
pub use merge::{MergeOutcome, MergePlan, MergeTarget};
pub use mesh::{BrickMesh, MeshScratch, Vertex};
pub use orientation::{AxisRotation, Facing, from_print_space, resting_up, to_print_space};
pub use pattern::{MAX_SCALE_MM, MIN_SCALE_VOXELS, Pattern, PatternKind};
pub use primitive::PrimitiveKind;
pub use project::{
    Keyframe, MAX_NAME_BYTES, MAX_VOLUME_BYTES, Outline, ProjectError, ProjectState, View,
    name_that_fits,
};
pub use raycast::{Hit, raycast};
pub use region::FieldRegion;
pub use similarity::{Bake, Similarity};
pub use split::{
    MASKED_ENOUGH_TO_SPLIT, MaskedSplitOutcome, Part, SIGNIFICANT_MM3, SLOW_SPLIT, SplitOutcome,
    SplitPlan,
};
pub use stroke::{MAX_STAMPS_PER_EVENT, Stroke};
pub use transform::warps_made_on_this_thread;
pub use undo::{
    Change, DEFAULT_HISTORY_BUDGET, DEFAULT_RECLAIM_BUDGET, Entry, History, HistoryStats,
    StrokeEdit, UndoOutcome,
};
pub use volume::{BrickPreview, BrickVerdict, PlanStats, Volume, VolumeStats};
