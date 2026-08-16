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
//! 2. [`brush::DrawBrush::apply`] resolves the bricks its box touches,
//!    allocates the missing ones and edits their voxels.
//! 3. The edit marks those bricks and their apron neighbours dirty.
//! 4. The caller meshes only the dirty bricks with [`Volume::mesh_brick`].
//!
//! Every step is proportional to what the brush touched and not to the size of
//! the model. That is the property the whole design exists to protect.

pub mod apron;
pub mod brick;
pub mod brush;
pub mod mesh;
pub mod raycast;
pub mod volume;

pub use apron::ApronBuffer;
pub use brick::{
    APRON_DIM, APRON_VOXELS, BRICK_DIM, BRICK_VOXELS, Brick, BrickCoord, INSIDE, NARROW_BAND,
    OUTSIDE,
};
pub use brush::{BrushDirection, DrawBrush};
pub use mesh::{BrickMesh, MeshScratch, Vertex};
pub use raycast::{Hit, raycast};
pub use volume::{Volume, VolumeStats};
