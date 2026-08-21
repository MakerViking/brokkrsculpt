// SPDX-License-Identifier: AGPL-3.0-only

//! Just enough ZIP to open an OPC package.
//!
//! This is the reading half of what [`crate::export::threemf`] writes by hand,
//! and it is much longer than the writer despite doing less. A writer only has
//! to produce one shape of container and can pick the easiest one; a reader has
//! to accept every shape the tools in the wild already produce. Two storage
//! methods are read -- stored, and deflate through `yazi` -- because real 3MF is
//! deflated and our own degenerate stored-only output is not representative of
//! anything but itself.
//!
//! # Where the truth about an entry lives
//!
//! In the central directory, never in the local file header. General purpose
//! flag bit 3 lets a writer that could not seek backwards leave the local
//! header's sizes and CRC at zero and append a data descriptor after the data
//! instead, and streaming writers do exactly that. A reader that trusts the
//! local header therefore sees a zero length entry and produces an empty model
//! rather than an error. Only the name and extra field lengths are taken from
//! the local header, because those two describe that header's own layout and
//! are allowed to differ from the central copy.
//!
//! # Why nothing here can write to the filesystem
//!
//! Entry names are only ever compared against part names that are already
//! known, and no name reaches [`std::fs`] or [`std::path`]. Zip slip -- an
//! entry called `../../.ssh/authorized_keys` escaping the extraction directory
//! -- is structurally impossible rather than defended against, because there is
//! no extraction directory and nothing is ever written out.

use crate::export::threemf::crc32;
use crate::import::ImportError;

/// The most a package may declare it will inflate to, across every entry.
///
/// Checked against the central directory before a single byte is inflated,
/// because the whole trick of a zip bomb is that the container is small: a few
/// hundred kilobytes of input can declare terabytes of output, and a reader
/// that allocates first and notices second is the bug being exploited. Two
/// gigabytes is far past any real 3MF, so this only ever fires on a file built
/// to fire it.
const MAX_UNCOMPRESSED_BYTES: u64 = 2 * 1024 * 1024 * 1024;

const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const ZIP64_EOCD_SIGNATURE: u32 = 0x0606_4b50;
const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;

/// The end of central directory record without its trailing comment.
const EOCD_FIXED_BYTES: usize = 22;
/// A ZIP comment is length prefixed by a `u16`, so the record can be no further
/// from the end of the file than this, which is what bounds the backwards scan.
const MAX_COMMENT_BYTES: usize = u16::MAX as usize;

/// The sentinel a 32 bit size or offset field carries when the real value is in
/// a ZIP64 extra field instead.
const ZIP64_ESCAPE_32: u32 = 0xFFFF_FFFF;
/// The same, for the 16 bit entry count in the end of central directory record.
const ZIP64_ESCAPE_16: u16 = 0xFFFF;

const METHOD_STORED: u16 = 0;
const METHOD_DEFLATE: u16 = 8;

/// A ZIP opened for reading, with its central directory already parsed.
pub(super) struct Archive<'a> {
    bytes: &'a [u8],
    entries: Vec<Entry>,
}

struct Entry {
    /// The name exactly as stored, minus any leading slash. Compared against,
    /// never resolved as a path.
    name: String,
    method: u16,
    crc: u32,
    compressed_size: u64,
    uncompressed_size: u64,
    header_offset: u64,
}

impl<'a> Archive<'a> {
    pub(super) fn open(bytes: &'a [u8]) -> Result<Self, ImportError> {
        let eocd = find_eocd(bytes)?;

        let mut count = u64::from(u16_at(bytes, eocd + 10).ok_or_else(truncated)?);
        let mut directory_offset = u64::from(u32_at(bytes, eocd + 16).ok_or_else(truncated)?);
        let mut directory_size = u64::from(u32_at(bytes, eocd + 12).ok_or_else(truncated)?);

        // Any one of the three escaping is enough to mean the real values are
        // in the ZIP64 record, and they do not all escape together: a file with
        // ten entries past the four gigabyte mark escapes the offset alone.
        let escaped = count == u64::from(ZIP64_ESCAPE_16)
            || directory_offset == u64::from(ZIP64_ESCAPE_32)
            || directory_size == u64::from(ZIP64_ESCAPE_32);
        if escaped {
            let zip64 = find_zip64_eocd(bytes, eocd)?;
            count = u64_at(bytes, zip64 + 32).ok_or_else(truncated)?;
            directory_size = u64_at(bytes, zip64 + 40).ok_or_else(truncated)?;
            directory_offset = u64_at(bytes, zip64 + 48).ok_or_else(truncated)?;
        }

        let start = usize::try_from(directory_offset).map_err(|_| truncated())?;
        let end = start
            .checked_add(usize::try_from(directory_size).map_err(|_| truncated())?)
            .ok_or_else(truncated)?;
        if end > bytes.len() {
            return Err(truncated());
        }

        let mut entries = Vec::new();
        let mut at = start;
        for _ in 0..count {
            let (entry, next) = read_central_entry(bytes, at, end)?;
            entries.push(entry);
            at = next;
        }

        let declared: u64 = entries.iter().map(|entry| entry.uncompressed_size).sum();
        if declared > MAX_UNCOMPRESSED_BYTES {
            return Err(ImportError::Malformed(format!(
                "it declares that it unpacks to {:.1} GB, over the {:.0} GB limit -- either it is \
                 corrupt or it is built to exhaust memory",
                declared as f64 / (1024.0 * 1024.0 * 1024.0),
                MAX_UNCOMPRESSED_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
            )));
        }

        Ok(Self { bytes, entries })
    }

    /// Read one part by name, or `None` if the package has no such part.
    ///
    /// Names are matched without regard to ASCII case, because OPC part names
    /// compare case insensitively and exporters disagree about whether the
    /// model lives in `3D/` or `3d/`.
    pub(super) fn read(&self, name: &str) -> Result<Option<Vec<u8>>, ImportError> {
        let wanted = name.trim_start_matches('/');
        let Some(entry) = self.entries.iter().find(|entry| entry.name.eq_ignore_ascii_case(wanted))
        else {
            return Ok(None);
        };
        self.read_entry(entry).map(Some)
    }

    fn read_entry(&self, entry: &Entry) -> Result<Vec<u8>, ImportError> {
        let header = usize::try_from(entry.header_offset).map_err(|_| truncated())?;
        if u32_at(self.bytes, header) != Some(LOCAL_HEADER_SIGNATURE) {
            return Err(ImportError::Malformed(format!(
                "the entry for `{}` does not point at a file header",
                entry.name
            )));
        }
        // Only these two come from the local header: they describe the length
        // of the header itself, and the specification allows them to differ
        // from the central directory's copy.
        let name_bytes = usize::from(u16_at(self.bytes, header + 26).ok_or_else(truncated)?);
        let extra_bytes = usize::from(u16_at(self.bytes, header + 28).ok_or_else(truncated)?);

        let start = header
            .checked_add(30)
            .and_then(|at| at.checked_add(name_bytes))
            .and_then(|at| at.checked_add(extra_bytes))
            .ok_or_else(truncated)?;
        let length = usize::try_from(entry.compressed_size).map_err(|_| truncated())?;
        let end = start.checked_add(length).ok_or_else(truncated)?;
        let data = self.bytes.get(start..end).ok_or_else(truncated)?;

        let declared = usize::try_from(entry.uncompressed_size).map_err(|_| truncated())?;
        let unpacked = match entry.method {
            METHOD_STORED => {
                if data.len() != declared {
                    return Err(ImportError::Malformed(format!(
                        "`{}` is stored but its two sizes disagree",
                        entry.name
                    )));
                }
                data.to_vec()
            }
            METHOD_DEFLATE => inflate(data, declared, &entry.name)?,
            other => {
                return Err(ImportError::Unsupported(format!(
                    "`{}` inside it is compressed with method {other}, and only stored and \
                     deflate are read",
                    entry.name
                )));
            }
        };

        // The CRC is the only thing that catches a file that unpacked to
        // exactly the right length out of the wrong bytes, which is what a
        // partial download looks like.
        if crc32(&unpacked) != entry.crc {
            return Err(ImportError::Malformed(format!(
                "`{}` inside it is corrupt: its checksum does not match",
                entry.name
            )));
        }
        Ok(unpacked)
    }
}

/// Inflate into a buffer of exactly the declared size.
///
/// A fixed buffer rather than a growing vector, deliberately: it is the cap
/// that makes the bomb check above binding. Without it a bomb could declare a
/// harmless total in its central directory and then inflate to whatever it
/// liked, because the declared size would never be checked against what the
/// bitstream actually produces.
fn inflate(data: &[u8], declared: usize, name: &str) -> Result<Vec<u8>, ImportError> {
    let mut out = vec![0u8; declared];
    // Boxed because the decoder carries its Huffman trees inline and is far too
    // large to want on the stack.
    let mut decoder = yazi::Decoder::boxed();
    decoder.set_format(yazi::Format::Raw);

    let mut stream = decoder.stream_into_buf(&mut out);
    // Both calls happen either way: `finish` is what releases the borrow on the
    // buffer, and a write error is the more specific of the two to report.
    let write = stream.write(data);
    let finish = stream.finish();
    let (written, _) = match write {
        Ok(()) => finish,
        Err(error) => Err(error),
    }
    .map_err(|error| match error {
        yazi::Error::Overflow => ImportError::Malformed(format!(
            "`{name}` inside it inflates to more than it declares, which is how a zip bomb hides"
        )),
        _ => ImportError::Malformed(format!("`{name}` inside it is not valid deflate data")),
    })?;

    if written != declared as u64 {
        return Err(ImportError::Malformed(format!(
            "`{name}` inside it is truncated: it declares {declared} bytes and inflates to \
             {written}"
        )));
    }
    Ok(out)
}

/// Find the end of central directory record by scanning backwards.
///
/// Backwards because the record is at the end of the file by definition, and
/// bounded by the maximum comment length because that is the only thing allowed
/// to follow it. Candidates are checked against their own comment length rather
/// than taken on the strength of the signature alone: those four bytes turn up
/// by chance inside compressed data often enough to matter, and the consistency
/// check is what tells a real record from a coincidence.
fn find_eocd(bytes: &[u8]) -> Result<usize, ImportError> {
    if bytes.len() < EOCD_FIXED_BYTES {
        return Err(ImportError::Malformed(
            "it is too short to be a ZIP container, so it is not a 3MF".to_string(),
        ));
    }
    let last = bytes.len() - EOCD_FIXED_BYTES;
    let first = last.saturating_sub(MAX_COMMENT_BYTES);

    for at in (first..=last).rev() {
        if u32_at(bytes, at) != Some(EOCD_SIGNATURE) {
            continue;
        }
        let comment = usize::from(u16_at(bytes, at + 20).ok_or_else(truncated)?);
        if at + EOCD_FIXED_BYTES + comment == bytes.len() {
            return Ok(at);
        }
    }
    Err(ImportError::Malformed(
        "it has no ZIP directory at the end, so it is not a 3MF package".to_string(),
    ))
}

/// Follow the ZIP64 locator that sits immediately before the ordinary record.
fn find_zip64_eocd(bytes: &[u8], eocd: usize) -> Result<usize, ImportError> {
    let locator = eocd.checked_sub(20).ok_or_else(|| {
        ImportError::Malformed("it claims to be ZIP64 but has no ZIP64 locator".to_string())
    })?;
    if u32_at(bytes, locator) != Some(ZIP64_LOCATOR_SIGNATURE) {
        return Err(ImportError::Malformed(
            "it claims to be ZIP64 but has no ZIP64 locator".to_string(),
        ));
    }
    let offset = u64_at(bytes, locator + 8).ok_or_else(truncated)?;
    let at = usize::try_from(offset).map_err(|_| truncated())?;
    if u32_at(bytes, at) != Some(ZIP64_EOCD_SIGNATURE) {
        return Err(ImportError::Malformed(
            "its ZIP64 locator does not point at a ZIP64 directory".to_string(),
        ));
    }
    Ok(at)
}

/// Read one central directory entry, returning it and where the next one starts.
fn read_central_entry(bytes: &[u8], at: usize, end: usize) -> Result<(Entry, usize), ImportError> {
    if u32_at(bytes, at) != Some(CENTRAL_SIGNATURE) {
        return Err(ImportError::Malformed(
            "its ZIP directory has fewer entries than it says".to_string(),
        ));
    }
    let method = u16_at(bytes, at + 10).ok_or_else(truncated)?;
    let crc = u32_at(bytes, at + 16).ok_or_else(truncated)?;
    let compressed = u32_at(bytes, at + 20).ok_or_else(truncated)?;
    let uncompressed = u32_at(bytes, at + 24).ok_or_else(truncated)?;
    let name_bytes = usize::from(u16_at(bytes, at + 28).ok_or_else(truncated)?);
    let extra_bytes = usize::from(u16_at(bytes, at + 30).ok_or_else(truncated)?);
    let comment_bytes = usize::from(u16_at(bytes, at + 32).ok_or_else(truncated)?);
    let header_offset = u32_at(bytes, at + 42).ok_or_else(truncated)?;

    let name_at = at.checked_add(46).ok_or_else(truncated)?;
    let extra_at = name_at.checked_add(name_bytes).ok_or_else(truncated)?;
    let next = extra_at
        .checked_add(extra_bytes)
        .and_then(|to| to.checked_add(comment_bytes))
        .ok_or_else(truncated)?;
    if next > end {
        return Err(truncated());
    }

    // Names are bytes, and are UTF-8 only when flag bit 11 says so. Everything
    // this reader does with a name is compare it against an ASCII part name, so
    // a name in some other encoding can be mangled without consequence -- it
    // simply will not match, which is the right answer for a part we are not
    // looking for anyway.
    let name =
        String::from_utf8_lossy(bytes.get(name_at..extra_at).ok_or_else(truncated)?).into_owned();
    let extra = bytes.get(extra_at..extra_at + extra_bytes).ok_or_else(truncated)?;

    let mut entry = Entry {
        name: name.trim_start_matches('/').to_string(),
        method,
        crc,
        compressed_size: u64::from(compressed),
        uncompressed_size: u64::from(uncompressed),
        header_offset: u64::from(header_offset),
    };
    read_zip64_extra(extra, &mut entry, uncompressed, compressed, header_offset)?;
    Ok((entry, next))
}

/// Replace whatever escaped to `0xFFFFFFFF` with the real 64 bit value.
///
/// The ZIP64 extra field is positional and lists only the fields that escaped,
/// in a fixed order, with no tags -- so which eight bytes mean what depends
/// entirely on which of the 32 bit fields were sentinels. Reading it as though
/// all three were always present is the classic way to come out with a garbage
/// offset.
fn read_zip64_extra(
    extra: &[u8],
    entry: &mut Entry,
    uncompressed: u32,
    compressed: u32,
    header_offset: u32,
) -> Result<(), ImportError> {
    let mut at = 0usize;
    while at + 4 <= extra.len() {
        let id = u16_at(extra, at).ok_or_else(truncated)?;
        let size = usize::from(u16_at(extra, at + 2).ok_or_else(truncated)?);
        let body_at = at + 4;
        let body_end = body_at.checked_add(size).ok_or_else(truncated)?;
        if body_end > extra.len() {
            return Err(truncated());
        }
        if id == 0x0001 {
            let body = &extra[body_at..body_end];
            let mut cursor = 0usize;
            let mut take = |slot: &mut u64| -> Result<(), ImportError> {
                *slot = u64_at(body, cursor).ok_or_else(truncated)?;
                cursor += 8;
                Ok(())
            };
            if uncompressed == ZIP64_ESCAPE_32 {
                take(&mut entry.uncompressed_size)?;
            }
            if compressed == ZIP64_ESCAPE_32 {
                take(&mut entry.compressed_size)?;
            }
            if header_offset == ZIP64_ESCAPE_32 {
                take(&mut entry.header_offset)?;
            }
            return Ok(());
        }
        at = body_end;
    }
    Ok(())
}

/// Every length and offset read here comes out of the file itself, so a
/// truncated download has to reach this rather than a panic on a slice.
fn truncated() -> ImportError {
    ImportError::Malformed("the ZIP container inside it is truncated".to_string())
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    let end = at.checked_add(2)?;
    Some(u16::from_le_bytes(bytes.get(at..end)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    Some(u32::from_le_bytes(bytes.get(at..end)?.try_into().ok()?))
}

fn u64_at(bytes: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    Some(u64::from_le_bytes(bytes.get(at..end)?.try_into().ok()?))
}
