use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{
    AdmittedFile, FileRecord, FixtureError, MAX_PATH_BYTES, MAX_PATH_COMPONENTS, RelativePath,
};

impl RelativePath {
    pub(super) fn parse(raw: &str) -> Result<Self, FixtureError> {
        if raw.is_empty()
            || raw.len() > MAX_PATH_BYTES
            || raw.starts_with('/')
            || raw.ends_with('/')
            || raw.contains('\\')
            || raw.contains(':')
            || raw.contains('\0')
        {
            return Err(FixtureError::UnsafePath(raw.to_owned()));
        }
        let components: Vec<_> = raw.split('/').map(str::to_owned).collect();
        if components.len() > MAX_PATH_COMPONENTS
            || components
                .iter()
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(FixtureError::UnsafePath(raw.to_owned()));
        }
        Ok(Self {
            raw: raw.to_owned(),
            components,
        })
    }

    pub(super) fn to_path_buf(&self) -> PathBuf {
        let mut path = PathBuf::new();
        for component in &self.components {
            path.push(component);
        }
        path
    }
}
pub(super) fn validate_workspace_layout(files: &[AdmittedFile]) -> Result<(), FixtureError> {
    let mut destinations: Vec<&RelativePath> = files
        .iter()
        .filter_map(|file| file.workspace_path.as_ref())
        .collect();
    destinations.sort_unstable_by(|left, right| left.components.cmp(&right.components));
    for pair in destinations.windows(2) {
        if pair[1].components.starts_with(&pair[0].components) {
            return Err(FixtureError::InvalidMetadata(format!(
                "workspace destinations overlap: {} and {}",
                pair[0].raw, pair[1].raw
            )));
        }
    }

    let scratch = tempfile::tempdir().map_err(|error| io_error("workspace alias check", error))?;
    for destination in destinations {
        let path = scratch.path().join(destination.to_path_buf());
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                if error.kind() == io::ErrorKind::AlreadyExists {
                    FixtureError::InvalidMetadata(format!(
                        "workspace destination aliases another path: {}",
                        destination.raw
                    ))
                } else {
                    io_error(&path, error)
                }
            })?;
    }
    Ok(())
}
pub(super) fn existing_file(
    root: &Path,
    relative: &RelativePath,
) -> Result<(PathBuf, Metadata), FixtureError> {
    let mut current = root.to_path_buf();
    let relative_path = relative.to_path_buf();
    let mut components = relative_path.components().peekable();
    while let Some(component) = components.next() {
        let Component::Normal(name) = component else {
            return Err(FixtureError::UnsafePath(relative.raw.clone()));
        };
        current.push(name);
        let metadata = fs::symlink_metadata(&current).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                FixtureError::MissingArtifact(relative.raw.clone())
            } else {
                io_error(&current, error)
            }
        })?;
        if metadata.file_type().is_symlink() {
            return Err(FixtureError::UnsafePath(relative.raw.clone()));
        }
        if components.peek().is_some() && !metadata.is_dir() {
            return Err(FixtureError::UnsafePath(relative.raw.clone()));
        }
    }
    let canonical = fs::canonicalize(&current).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            FixtureError::MissingArtifact(relative.raw.clone())
        } else {
            io_error(&current, error)
        }
    })?;
    if !canonical.starts_with(root) {
        return Err(FixtureError::UnsafePath(relative.raw.clone()));
    }
    let metadata = fs::metadata(&canonical).map_err(|error| io_error(&canonical, error))?;
    if !metadata.is_file() {
        return Err(FixtureError::UnsafePath(relative.raw.clone()));
    }
    Ok((canonical, metadata))
}
pub(super) fn check_size(path: &str, expected: u64, actual: u64) -> Result<(), FixtureError> {
    if expected != actual {
        return Err(FixtureError::ArtifactSizeMismatch {
            path: path.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}
pub(super) fn check_hash(path: &Path, record: &FileRecord) -> Result<(), FixtureError> {
    let (hash, size) = hash_file(path).map_err(|error| io_error(path, error))?;
    check_size(&record.path, record.size_bytes, size)?;
    if hash != record.sha256 {
        return Err(FixtureError::HashMismatch {
            path: record.path.clone(),
        });
    }
    Ok(())
}
fn hash_file(path: &Path) -> io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut size = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        size = size.saturating_add(count as u64);
        hasher.update(&buffer[..count]);
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}
pub(super) fn io_error(path: impl AsRef<Path>, error: io::Error) -> FixtureError {
    let path = path.as_ref();
    if error.kind() == io::ErrorKind::NotFound {
        FixtureError::MissingArtifact(path.display().to_string())
    } else {
        FixtureError::Io {
            path: path.display().to_string(),
            message: error.to_string(),
        }
    }
}
