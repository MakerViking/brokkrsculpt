// SPDX-License-Identifier: AGPL-3.0-or-later

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
pub mod brick;
pub mod brush;
pub mod export;
pub mod mesh;
pub mod raycast;
pub mod region;
pub mod resample;
pub mod stroke;
pub mod undo;
pub mod volume;

pub use apron::ApronBuffer;
pub use brick::{
    APRON_DIM, APRON_VOXELS, BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND,
    OUTSIDE,
};
pub use brush::{
    Brush, BrushDirection, BrushKind, BrushScratch, FalloffCurve, Stamp, Symmetry, lean_normal,
};
pub use export::{ExportMesh, MeshReport};
pub use mesh::{BrickMesh, MeshScratch, Vertex};
pub use raycast::{Hit, raycast};
pub use region::FieldRegion;
pub use stroke::{MAX_STAMPS_PER_EVENT, Stroke};
pub use undo::{DEFAULT_HISTORY_BUDGET, History, HistoryStats, StrokeEdit};
pub use volume::{Volume, VolumeStats};
