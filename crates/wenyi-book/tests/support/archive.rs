#![forbid(unsafe_code)]
#![allow(dead_code)]
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

pub struct Temp {
    pub root: PathBuf,
    pub path: PathBuf,
}
impl Temp {
    pub fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "wenyi-archive-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("fallback.epub");
        std::fs::write(&path, bytes).unwrap();
        Self { root, path }
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}
pub fn zip(entries: &[(&str, &[u8])], compression: CompressionMethod) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    for &(name, bytes) in entries {
        writer
            .start_file(
                name,
                SimpleFileOptions::default().compression_method(compression),
            )
            .unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}
pub fn positions(bytes: &[u8], magic: &[u8; 4]) -> Vec<usize> {
    bytes
        .windows(4)
        .enumerate()
        .filter_map(|(n, value)| (value == magic).then_some(n))
        .collect()
}
pub fn set16(bytes: &mut [u8], n: usize, value: u16) {
    bytes[n..n + 2].copy_from_slice(&value.to_le_bytes());
}
pub fn set32(bytes: &mut [u8], n: usize, value: u32) {
    bytes[n..n + 4].copy_from_slice(&value.to_le_bytes());
}
pub fn crc(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & 0_u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}
pub fn raw_member(name: &[u8], flags: u16, method: u16, payload: &[u8], decoded: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; 30];
    bytes[..4].copy_from_slice(b"PK\x03\x04");
    set16(&mut bytes, 4, 20);
    set16(&mut bytes, 6, flags);
    set16(&mut bytes, 8, method);
    set32(&mut bytes, 14, crc(decoded));
    set32(&mut bytes, 18, payload.len() as u32);
    set32(&mut bytes, 22, decoded.len() as u32);
    set16(&mut bytes, 26, name.len() as u16);
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(payload);
    let start = bytes.len();
    let mut header = vec![0; 46];
    header[..4].copy_from_slice(b"PK\x01\x02");
    set16(&mut header, 4, (3 << 8) | 20);
    set16(&mut header, 6, 20);
    set16(&mut header, 8, flags);
    set16(&mut header, 10, method);
    set32(&mut header, 16, crc(decoded));
    set32(&mut header, 20, payload.len() as u32);
    set32(&mut header, 24, decoded.len() as u32);
    set16(&mut header, 28, name.len() as u16);
    bytes.extend(header);
    bytes.extend_from_slice(name);
    let size = bytes.len() - start;
    let mut end = vec![0; 22];
    end[..4].copy_from_slice(b"PK\x05\x06");
    set16(&mut end, 8, 1);
    set16(&mut end, 10, 1);
    set32(&mut end, 12, size as u32);
    set32(&mut end, 16, start as u32);
    bytes.extend(end);
    bytes
}
pub fn zip64(mut bytes: Vec<u8>, count: u64, extension: &[u8]) -> Vec<u8> {
    let eocd = positions(&bytes, b"PK\x05\x06")[0];
    let old = bytes.split_off(eocd);
    let directory_size = u32::from_le_bytes(old[12..16].try_into().unwrap()) as u64;
    let directory_start = u32::from_le_bytes(old[16..20].try_into().unwrap()) as u64;
    bytes.extend_from_slice(b"PK\x06\x06");
    bytes.extend_from_slice(&(44_u64 + extension.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&[45, 0, 45, 0]);
    bytes.extend_from_slice(&[0; 8]);
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&directory_size.to_le_bytes());
    bytes.extend_from_slice(&directory_start.to_le_bytes());
    bytes.extend_from_slice(extension);
    bytes.extend_from_slice(b"PK\x06\x07");
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&(eocd as u64).to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    let mut old = old;
    set16(&mut old, 8, u16::MAX);
    set16(&mut old, 10, u16::MAX);
    set32(&mut old, 12, u32::MAX);
    set32(&mut old, 16, u32::MAX);
    bytes.extend(old);
    bytes
}
