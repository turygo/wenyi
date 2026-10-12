#![forbid(unsafe_code)]
use super::{Limits, archive_error, io_error};
use crate::error::BookError;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const METADATA_LIMIT: u64 = 64 * 1024 * 1024;
pub(super) struct Directory {
    pub count: usize,
    pub offset: u64,
}

pub(crate) fn safe_name(name: &str) -> bool {
    if name.is_empty() || name.contains(['\0', '\\']) || name.starts_with('/') {
        return false;
    }
    let mut parts = name.split('/').peekable();
    while let Some(part) = parts.next() {
        if part == "." || part == ".." || (part.is_empty() && parts.peek().is_some()) {
            return false;
        }
    }
    true
}
fn u16_at(bytes: &[u8], n: usize) -> u16 {
    u16::from_le_bytes(bytes[n..n + 2].try_into().unwrap())
}
fn u32_at(bytes: &[u8], n: usize) -> u32 {
    u32::from_le_bytes(bytes[n..n + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], n: usize) -> u64 {
    u64::from_le_bytes(bytes[n..n + 8].try_into().unwrap())
}
fn read_at(file: &mut File, offset: u64, bytes: &mut [u8]) -> Result<(), BookError> {
    file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
    file.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            archive_error("invalid_zip", "truncated ZIP metadata")
        } else {
            io_error(error)
        }
    })
}
fn charge(used: &mut u64, bytes: u64) -> Result<(), BookError> {
    *used = used
        .checked_add(bytes)
        .filter(|total| *total <= METADATA_LIMIT)
        .ok_or_else(|| archive_error("archive_metadata_limit", "ZIP metadata exceeds limit"))?;
    Ok(())
}
fn count_bound(count: u64, limits: Limits) -> Result<(), BookError> {
    if count > limits.members as u64 {
        return Err(archive_error(
            "archive_member_limit",
            "too many ZIP members",
        ));
    }
    Ok(())
}
struct Candidate {
    position: u64,
    header: [u8; 56],
}
struct Footer {
    count: u64,
    size: u64,
    relative: u64,
    end: u64,
}

// 一次固定窗口扫描收集尾记录固定头；候选存储本身也计入元数据预算。
fn candidates(
    file: &mut File,
    length: u64,
    used: &mut u64,
) -> Result<(Vec<Candidate>, Vec<Candidate>), BookError> {
    let mut ends = Vec::new();
    let mut large_ends = Vec::new();
    let mut buffer = [0; 64 * 1024];
    let mut cursor = 0;
    while length.saturating_sub(cursor) >= 4 {
        let size = (length - cursor).min(buffer.len() as u64) as usize;
        read_at(file, cursor, &mut buffer[..size])?;
        for n in 0..=size - 4 {
            let magic = &buffer[n..n + 4];
            let header_size = if magic == b"PK\x05\x06" {
                22
            } else if magic == b"PK\x06\x06" {
                56
            } else {
                continue;
            };
            let position = cursor + n as u64;
            if length - position < header_size {
                continue;
            }
            charge(used, 2 * std::mem::size_of::<Candidate>() as u64)?;
            let mut header = [0; 56];
            read_at(file, position, &mut header[..header_size as usize])?;
            let entries = if header_size == 22 {
                &mut ends
            } else {
                &mut large_ends
            };
            entries.try_reserve(1).map_err(|_| {
                archive_error(
                    "archive_metadata_limit",
                    "cannot allocate metadata candidates",
                )
            })?;
            entries.push(Candidate { position, header });
        }
        cursor += (size - 3) as u64;
    }
    Ok((ends, large_ends))
}

fn footer(
    file: &mut File,
    candidate: &Candidate,
    records: &[Candidate],
    length: u64,
    limits: Limits,
    used: &mut u64,
) -> Result<Option<Footer>, BookError> {
    let position = candidate.position;
    let end = &candidate.header;
    let overhead = 22 + u16_at(end, 20) as u64;
    if position
        .checked_add(overhead)
        .is_none_or(|end| end > length)
    {
        return Ok(None);
    }
    charge(used, overhead)?;
    let count = u16_at(end, 10) as u64;
    let size = u32_at(end, 12) as u64;
    let relative = u32_at(end, 16) as u64;
    let may_be_zip64 = count == u16::MAX as u64
        || u16_at(end, 8) == u16::MAX
        || size == u32::MAX as u64
        || relative == u32::MAX as u64;
    if may_be_zip64 && position >= 20 {
        let locator_position = position - 20;
        let mut locator = [0; 20];
        read_at(file, locator_position, &mut locator)?;
        if locator[..4] == *b"PK\x06\x07" {
            charge(used, 20)?;
            let first = records.partition_point(|record| record.position < u64_at(&locator, 8));
            let last = records.partition_point(|record| record.position < locator_position);
            let mut selected = None;
            for record in records.get(first..last).unwrap_or(&[]) {
                charge(used, 56)?;
                let record_size = u64_at(&record.header, 4);
                let Some(record_end) = record
                    .position
                    .checked_add(12)
                    .and_then(|p| p.checked_add(record_size))
                else {
                    continue;
                };
                if record_size < 40 || record_end > locator_position {
                    continue;
                }
                if record_size < 44 {
                    return Err(archive_error("invalid_zip", "short ZIP64 record"));
                }
                let count = u64_at(&record.header, 32);
                count_bound(count, limits)?;
                charge(used, record_size - 44)?;
                charge(used, u64_at(&record.header, 40))?;
                if record_end == locator_position && selected.is_none() {
                    selected = Some(Footer {
                        count,
                        size: u64_at(&record.header, 40),
                        relative: u64_at(&record.header, 48),
                        end: record.position,
                    });
                }
            }
            return selected
                .map(Some)
                .ok_or_else(|| archive_error("invalid_zip", "missing bounded ZIP64 record"));
        }
    }
    count_bound(count, limits)?;
    charge(used, size)?;
    Ok(Some(Footer {
        count,
        size,
        relative,
        end: position,
    }))
}

// 只检查固定中央头和变长字段跨度；库独占名字、额外字段及格式语义。
fn entry_extents(
    file: &mut File,
    start: u64,
    footer: &Footer,
    limits: Limits,
) -> Result<usize, BookError> {
    let mut cursor = start;
    let mut count = 0_usize;
    while cursor < footer.end {
        if count >= limits.members {
            return Err(archive_error(
                "archive_member_limit",
                "too many central entries",
            ));
        }
        if footer.end - cursor < 46 {
            return Err(archive_error("invalid_zip", "truncated central header"));
        }
        let mut header = [0; 46];
        read_at(file, cursor, &mut header)?;
        if header[..4] != *b"PK\x01\x02" {
            return Err(archive_error("invalid_zip", "invalid central header"));
        }
        cursor = cursor
            .checked_add(
                46 + u16_at(&header, 28) as u64
                    + u16_at(&header, 30) as u64
                    + u16_at(&header, 32) as u64,
            )
            .filter(|next| *next <= footer.end)
            .ok_or_else(|| archive_error("invalid_zip", "central entry exceeds directory"))?;
        count += 1;
    }
    if count as u64 != footer.count {
        return Err(archive_error("invalid_zip", "central entry count mismatch"));
    }
    Ok(count)
}

pub(super) fn inspect(
    file: &mut File,
    length: u64,
    limits: Limits,
) -> Result<Directory, BookError> {
    let mut used = 0;
    let (ends, records) = candidates(file, length, &mut used)?;
    let mut selected: Option<Directory> = None;
    for candidate in &ends {
        let Some(footer) = footer(file, candidate, &records, length, limits, &mut used)? else {
            continue;
        };
        let start = footer
            .end
            .checked_sub(footer.size)
            .ok_or_else(|| archive_error("invalid_zip", "invalid directory size"))?;
        let offset = start
            .checked_sub(footer.relative)
            .ok_or_else(|| archive_error("invalid_zip", "invalid directory offset"))?;
        let count = entry_extents(file, start, &footer, limits)?;
        if selected
            .as_ref()
            .is_some_and(|directory| directory.offset != offset)
        {
            return Err(archive_error(
                "invalid_zip",
                "inconsistent candidate directory offsets",
            ));
        }
        selected = Some(Directory { count, offset });
    }
    selected.ok_or_else(|| archive_error("invalid_zip", "missing bounded central directory"))
}
