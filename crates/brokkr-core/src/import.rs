// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reading meshes in, as the front half of importing one.
//!
//! This module only *parses*. Turning a triangle soup into a signed distance
//! field is [`crate::voxelise`]'s job, and keeping the two apart is what lets a
//! reader be tested against bytes and the voxeliser against geometry.
//!
//! # Why there is no `.sindri` reader
//!
//! A `.sindri` file holds SindriCAD's parametric feature tree and its
//! OpenCASCADE B-rep, not a mesh: its face references are `nearest`-point
//! fingerprints that only resolve against an evaluated solid, and its geometry
//! is BinTools binary that needs the kernel to read. The container has a `mesh/`
//! section, but nothing writes to it -- SindriCAD's own save path passes an
//! empty mesh key list. Importing one would therefore mean shipping a CAD
//! kernel inside a voxel sculptor to recover geometry that SindriCAD already
//! exports as STL or 3MF. So the extension is recognised and refused with a
//! message that says what to do instead, which is cheaper and far clearer than
//! failing to parse a ZIP.

pub mod obj;
pub mod stl;
pub mod threemf;

use std::path::Path;

use crate::export::ExportMesh;

/// An absolute ceiling on how much file will be read.
///
/// Checked before allocating, because the alternative is that a corrupt header
/// claiming four billion triangles is a request to allocate two hundred
/// gigabytes. Two hundred megabytes of STL is about four million triangles,
/// which is already far more than the lattice can carry detail for.
pub const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;

/// What went wrong reading a mesh.
#[derive(Debug)]
pub enum ImportError {
    Io(std::io::Error),
    /// The bytes are not the format they claim to be.
    Malformed(String),
    /// A file type we recognise and deliberately do not read.
    Unsupported(String),
    /// Parsed, but there is no surface in it.
    Empty,
    /// Bigger than [`MAX_INPUT_BYTES`].
    TooLarge {
        bytes: u64,
    },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Io(error) => write!(out, "{error}"),
            ImportError::Malformed(why) => write!(out, "{why}"),
            ImportError::Unsupported(why) => write!(out, "{why}"),
            ImportError::Empty => write!(out, "there are no triangles in it"),
            ImportError::TooLarge { bytes } => write!(
                out,
                "it is {:.1} GB, over the {:.0} MB import limit",
                *bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                MAX_INPUT_BYTES as f64 / (1024.0 * 1024.0),
            ),
        }
    }
}

impl std::error::Error for ImportError {}

impl From<std::io::Error> for ImportError {
    fn from(error: std::io::Error) -> Self {
        ImportError::Io(error)
    }
}

/// Every extension the open dialog should offer.
pub const MESH_EXTENSIONS: [&str; 3] = ["stl", "obj", "3mf"];

/// Read a mesh, choosing the parser by extension.
///
/// Extension rather than content sniffing, because two of the three formats we
/// read have no reliable magic between them -- an ASCII STL and an OBJ are both
/// plain text -- and because a user who renamed a file is better served by a
/// clear "that is not an STL" than by a parser guessing.
pub fn read_path(path: &Path) -> Result<ExportMesh, ImportError> {
    let extension =
        path.extension().map(|raw| raw.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();

    if extension == "sindri" {
        return Err(ImportError::Unsupported(
            "a .sindri file holds a CAD feature tree, not a mesh -- export STL from SindriCAD \
             and import that"
                .to_string(),
        ));
    }

    let size = std::fs::metadata(path)?.len();
    if size > MAX_INPUT_BYTES {
        return Err(ImportError::TooLarge { bytes: size });
    }
    let bytes = std::fs::read(path)?;

    match extension.as_str() {
        "stl" => stl::read(&bytes),
        "obj" => obj::read(&bytes),
        "3mf" => threemf::read(&bytes),
        other => Err(ImportError::Unsupported(format!("there is no reader for a .{other} file"))),
    }
}

/// Collapse a triangle soup into an indexed mesh, welding on exact bits.
///
/// Exact bits, deliberately, and this is the opposite of the rule export
/// follows. Export welds on *lattice cell identity* because two bricks compute
/// the same seam vertex at different intermediate magnitudes and their results
/// differ in the last bits -- there, rounding to a grid splits pairs that
/// belong together. Here there is no lattice and no shared derivation: a file's
/// bytes are the whole truth, two corners are the same corner only if the
/// author wrote the same number, and any tolerance would fuse genuinely
/// distinct detail below it. Over-reporting seams is the safe direction,
/// because it can only make the quality report pessimistic.
pub(crate) struct SoupWelder {
    mesh: ExportMesh,
    seen: rustc_hash::FxHashMap<[u32; 3], u32>,
}

impl SoupWelder {
    pub(crate) fn new() -> Self {
        Self { mesh: ExportMesh::default(), seen: rustc_hash::FxHashMap::default() }
    }

    /// Index one corner, reusing an existing one if those exact bits are known.
    pub(crate) fn vertex(&mut self, position: glam::Vec3) -> u32 {
        let key = [position.x.to_bits(), position.y.to_bits(), position.z.to_bits()];
        if let Some(&index) = self.seen.get(&key) {
            return index;
        }
        let index = self.mesh.positions.len() as u32;
        self.mesh.positions.push(position);
        self.seen.insert(key, index);
        index
    }

    /// Add a triangle, dropping it if two corners are the same point.
    ///
    /// A degenerate triangle has no surface and no usable winding, and the
    /// voxeliser's sign sweep would take its zero projected area as a
    /// contribution of nothing anyway -- but it would still be binned and
    /// distance-tested against every voxel near it, so it is cheaper to refuse
    /// it here.
    pub(crate) fn triangle(&mut self, corners: [glam::Vec3; 3]) {
        let [a, b, c] = corners.map(|corner| self.vertex(corner));
        if a == b || b == c || a == c {
            return;
        }
        self.mesh.triangles.push([a, b, c]);
    }

    pub(crate) fn finish(self) -> Result<ExportMesh, ImportError> {
        if self.mesh.triangles.is_empty() {
            return Err(ImportError::Empty);
        }
        Ok(self.mesh)
    }
}

/// Reject a coordinate that cannot be put on a lattice.
///
/// A NaN would propagate silently through every distance computation and come
/// out as a field full of NaN, which meshes to nothing and looks like an empty
/// import rather than a bad file.
pub(crate) fn finite(position: glam::Vec3, what: &str) -> Result<glam::Vec3, ImportError> {
    if position.is_finite() {
        Ok(position)
    } else {
        Err(ImportError::Malformed(format!("{what} has a coordinate that is not a number")))
    }
}
