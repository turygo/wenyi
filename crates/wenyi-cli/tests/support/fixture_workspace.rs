use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::paths::{check_hash, check_size, existing_file, io_error};
use super::{AdmittedFixture, FileRecord, FixtureError, FixtureWorkspace};

impl FixtureWorkspace {
    pub(crate) fn new(fixture: &AdmittedFixture) -> Result<Self, FixtureError> {
        let temporary =
            tempfile::tempdir().map_err(|error| io_error("temporary workspace", error))?;
        let workspace = Self { temporary };
        let mut canonical_paths = HashSet::with_capacity(fixture.files.len());
        for file in &fixture.files {
            let record = &fixture.metadata.files[file.record_index];
            let (source, metadata) = existing_file(&fixture.root, &file.source_path)?;
            if canonical_paths.contains(&source) {
                return Err(FixtureError::UnsafePath(record.path.clone()));
            }
            check_size(&record.path, record.size_bytes, metadata.len())?;
            check_hash(&source, record)?;
            canonical_paths.insert(source);
        }
        for file in &fixture.files {
            let record = &fixture.metadata.files[file.record_index];
            let Some(destination) = &file.workspace_path else {
                continue;
            };
            let (source, _) = existing_file(&fixture.root, &file.source_path)?;
            let destination = workspace.temporary.path().join(destination.to_path_buf());
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
            }
            copy_checked(&source, &destination, record)?;
        }
        Ok(workspace)
    }

    pub(crate) fn root(&self) -> &Path {
        self.temporary.path()
    }
}
fn copy_checked(
    source: &Path,
    destination: &Path,
    record: &FileRecord,
) -> Result<(), FixtureError> {
    let mut input = File::open(source).map_err(|error| io_error(source, error))?;
    let metadata = input.metadata().map_err(|error| io_error(source, error))?;
    if !metadata.is_file() {
        return Err(FixtureError::UnsafePath(record.path.clone()));
    }
    check_size(&record.path, record.size_bytes, metadata.len())?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|error| io_error(destination, error))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut size = 0_u64;
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| io_error(source, error))?;
        if count == 0 {
            break;
        }
        size = size.saturating_add(count as u64);
        if size > record.size_bytes {
            return Err(FixtureError::ArtifactSizeMismatch {
                path: record.path.clone(),
                expected: record.size_bytes,
                actual: size,
            });
        }
        hasher.update(&buffer[..count]);
        output
            .write_all(&buffer[..count])
            .map_err(|error| io_error(destination, error))?;
    }
    check_size(&record.path, record.size_bytes, size)?;
    if format!("{:x}", hasher.finalize()) != record.sha256 {
        return Err(FixtureError::HashMismatch {
            path: record.path.clone(),
        });
    }
    Ok(())
}
