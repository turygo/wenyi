#![forbid(unsafe_code)]
use super::io_error;
use crate::error::{BookError, BookErrorKind};
use same_file::Handle;
use sha2::{Digest, Sha256};
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub(super) struct Snapshot {
    pub sha256: String,
    path: PathBuf,
    identity: Handle,
    length: u64,
    modified: SystemTime,
}
impl Snapshot {
    pub fn capture(file: &mut File, path: &Path) -> Result<Self, BookError> {
        let metadata = file.metadata().map_err(io_error)?;
        let identity = Handle::from_file(file.try_clone().map_err(io_error)?).map_err(io_error)?;
        let mut snapshot = Self {
            sha256: String::new(),
            path: path.to_owned(),
            identity,
            length: metadata.len(),
            modified: metadata.modified().map_err(io_error)?,
        };
        snapshot.sha256 = hash(file)?;
        snapshot.validate_metadata(file)?;
        Ok(snapshot)
    }
    pub fn length(&self) -> u64 {
        self.length
    }
    fn matches(&self, metadata: &Metadata) -> bool {
        metadata.len() == self.length && metadata.modified().ok() == Some(self.modified)
    }
    fn validate_metadata(&self, file: &File) -> Result<(), BookError> {
        let metadata = file.metadata().map_err(|_| changed())?;
        let current = Handle::from_path(&self.path).map_err(|_| changed())?;
        let selected = std::fs::metadata(&self.path).map_err(|_| changed())?;
        if current != self.identity || !self.matches(&metadata) || !self.matches(&selected) {
            return Err(changed());
        }
        Ok(())
    }
    pub fn validate(&self, file: &mut File) -> Result<(), BookError> {
        self.validate_metadata(file)?;
        if hash(file)? != self.sha256 {
            return Err(changed());
        }
        self.validate_metadata(file)
    }
}
fn hash(file: &mut File) -> Result<String, BookError> {
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    let mut buffer = [0; 64 * 1024];
    let mut hash = Sha256::new();
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub(super) fn changed() -> BookError {
    BookError::new(
        BookErrorKind::Source,
        "source_changed",
        "selected source changed during admission",
    )
}
