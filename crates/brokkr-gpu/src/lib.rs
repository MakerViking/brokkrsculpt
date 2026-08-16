// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU resources for BrokkrSculpt: the mesh buffer pool and the sculpt
//! renderer.
//!
//! Nothing here depends on a UI toolkit. The renderer takes a device, a queue
//! and a target format, and draws into whatever texture view it is handed, so
//! the Iced specific glue stays in `brokkr-app`.

pub mod matcap;
pub mod mesh_pool;
pub mod renderer;

pub use mesh_pool::{INDEX_CAPACITY, MeshPool, PoolStats, VERTEX_CAPACITY};
pub use renderer::{PixelRect, SculptRenderer, Uniforms};
