#![forbid(unsafe_code)]
pub(crate) mod guard;
mod snapshot;
use crate::error::{BookError, BookErrorKind};
use indexmap::IndexMap;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberKind {
    File,
    Directory,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveMember {
    pub index: usize,
    pub kind: MemberKind,
    pub compressed_size: u64,
    pub expanded_size: u64,
    pub compression_method: u16,
    pub crc32: u32,
    pub sha256: String,
}
#[derive(Debug)]
pub struct ArchiveInventory {
    pub source_bytes_sha256: String,
    pub members: IndexMap<String, ArchiveMember>,
}
#[derive(Clone, Copy)]
struct Limits {
    members: usize,
    member: u64,
    total: u64,
}
const LIMITS: Limits = Limits {
    members: 10000,
    member: 512 * 1024 * 1024,
    total: 4 * 1024 * 1024 * 1024,
};

pub struct Archive {
    zip: ZipArchive<File>,
    members: IndexMap<String, ArchiveMember>,
    snapshot: snapshot::Snapshot,
}
impl Archive {
    pub fn open(input: &Path) -> Result<Self, BookError> {
        Self::open_with_limits(input, LIMITS)
    }
    fn open_with_limits(input: &Path, limits: Limits) -> Result<Self, BookError> {
        let mut file = File::open(input).map_err(io_error)?;
        let snapshot = snapshot::Snapshot::capture(&mut file, input)?;
        let directory = guard::inspect(&mut file, snapshot.length(), limits)?;
        let config = zip::read::Config {
            archive_offset: zip::read::ArchiveOffset::Known(directory.offset),
        };
        let mut zip = ZipArchive::with_config(config, file)
            .map_err(|error| archive_error("invalid_zip", error.to_string()))?;
        let mut members = IndexMap::new();
        let mut total = 0;
        let mut declared = 0_u64;
        let mut seen = HashSet::new();
        if zip.len() != directory.count {
            return Err(archive_error(
                "duplicate_entry",
                "directory identities collapsed",
            ));
        }
        for index in 0..zip.len() {
            let entry = zip
                .by_index_data(index)
                .map_err(|error| archive_error("invalid_zip", error.to_string()))?;
            let name = entry
                .name()
                .map_err(|error| archive_error("invalid_zip", error.to_string()))?
                .into_owned();
            if !guard::safe_name(&name) {
                return Err(
                    archive_error("unsafe_entry", "unsafe ZIP member name").with_resource(name)
                );
            }
            if !seen.insert(name.strip_suffix('/').unwrap_or(&name).to_owned()) {
                return Err(
                    archive_error("duplicate_entry", "duplicate ZIP identity").with_resource(name)
                );
            }
            if entry.encrypted() {
                return Err(
                    archive_error("encrypted_entry", "encrypted ZIP member").with_resource(name)
                );
            }
            if entry
                .unix_mode()
                .is_some_and(|mode| !matches!(mode & 0o170000, 0 | 0o100000 | 0o040000))
            {
                return Err(
                    archive_error("special_entry", "special ZIP member").with_resource(name)
                );
            }
            if entry.size() > limits.member {
                return Err(
                    archive_error("member_too_large", "declared member exceeds limit")
                        .with_resource(name),
                );
            }
            declared = declared
                .checked_add(entry.size())
                .filter(|sum| *sum <= limits.total)
                .ok_or_else(|| {
                    archive_error("archive_too_large", "declared expansion exceeds limit")
                })?;
            let compression_method = match entry.compression() {
                zip::CompressionMethod::Stored => 0,
                zip::CompressionMethod::Deflated => 8,
                zip::CompressionMethod::Bzip2 => 12,
                zip::CompressionMethod::Lzma => 14,
                _ => {
                    return Err(archive_error(
                        "unsupported_compression",
                        "unsupported ZIP compression",
                    )
                    .with_resource(name));
                }
            };
            drop(entry);
            let mut stream = zip.by_index(index).map_err(|error| {
                archive_error("member_read", error.to_string()).with_resource(&name)
            })?;
            let mut member = ArchiveMember {
                index,
                kind: if name.ends_with('/') {
                    MemberKind::Directory
                } else {
                    MemberKind::File
                },
                compressed_size: stream.compressed_size(),
                expanded_size: stream.size(),
                compression_method,
                crc32: stream.crc32(),
                sha256: String::new(),
            };
            member.sha256 =
                stream_member(&mut stream, member.expanded_size, limits, &mut total, None)
                    .map_err(|error| error.with_resource(&name))?;
            members.insert(name, member);
        }
        Ok(Self {
            zip,
            members,
            snapshot,
        })
    }
    pub fn members(&self) -> &IndexMap<String, ArchiveMember> {
        &self.members
    }
    pub fn source_bytes_sha256(&self) -> &str {
        &self.snapshot.sha256
    }
    pub fn read_resource(&mut self, name: &str) -> Result<Vec<u8>, BookError> {
        let member = self.members.get(name).ok_or_else(|| {
            archive_error("missing_resource", "resource is absent").with_resource(name)
        })?;
        let mut stream = self
            .zip
            .by_index(member.index)
            .map_err(|error| archive_error("member_read", error.to_string()).with_resource(name))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(
                usize::try_from(member.expanded_size).map_err(|_| {
                    archive_error("member_too_large", "member exceeds address space")
                })?,
            )
            .map_err(|_| archive_error("member_too_large", "cannot allocate admitted resource"))?;
        let sha256 = stream_member(
            &mut stream,
            member.expanded_size,
            LIMITS,
            &mut 0,
            Some(&mut bytes),
        )
        .map_err(|error| error.with_resource(name))?;
        if sha256 != member.sha256 {
            return Err(snapshot::changed().with_resource(name));
        }
        Ok(bytes)
    }
    pub fn finish(self) -> Result<ArchiveInventory, BookError> {
        let mut reader = self.zip.into_inner();
        self.snapshot.validate(&mut reader)?;
        Ok(ArchiveInventory {
            source_bytes_sha256: self.snapshot.sha256,
            members: self.members,
        })
    }
}

fn stream_member(
    stream: &mut impl Read,
    declared: u64,
    limits: Limits,
    total: &mut u64,
    mut output: Option<&mut Vec<u8>>,
) -> Result<String, BookError> {
    let mut buffer = [0; 64 * 1024];
    let mut count = 0_u64;
    let mut hash = Sha256::new();
    loop {
        let remaining = limits
            .member
            .saturating_sub(count)
            .min(limits.total.saturating_sub(*total));
        let request = remaining.saturating_add(1).min(buffer.len() as u64) as usize;
        let read = stream
            .read(&mut buffer[..request])
            .map_err(|error| archive_error("member_read", error.to_string()))?;
        if read == 0 {
            break;
        }
        count = count
            .checked_add(read as u64)
            .ok_or_else(|| archive_error("member_too_large", "member count overflow"))?;
        *total = total
            .checked_add(read as u64)
            .ok_or_else(|| archive_error("archive_too_large", "aggregate count overflow"))?;
        if count > limits.member {
            return Err(archive_error(
                "member_too_large",
                "actual member exceeds limit",
            ));
        }
        if *total > limits.total {
            return Err(archive_error(
                "archive_too_large",
                "actual aggregate exceeds limit",
            ));
        }
        hash.update(&buffer[..read]);
        if let Some(bytes) = output.as_mut() {
            bytes.extend_from_slice(&buffer[..read]);
        }
    }
    if count != declared {
        return Err(archive_error(
            "member_size_mismatch",
            "actual member size differs",
        ));
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn archive_error(code: &'static str, detail: impl Into<String>) -> BookError {
    BookError::new(BookErrorKind::Archive, code, detail)
}
pub(super) fn io_error(error: std::io::Error) -> BookError {
    BookError::new(BookErrorKind::Io, "source_io", error.to_string())
}

#[cfg(test)]
mod tests;
