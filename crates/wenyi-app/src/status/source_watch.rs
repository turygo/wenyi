#![forbid(unsafe_code)]

use super::StatusError;
use same_file::Handle;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, Metadata},
    io::{Read, Seek},
    path::{Path, PathBuf},
};

pub(super) struct SourceWatch {
    path: PathBuf,
    file: File,
    identity: Handle,
    before: Metadata,
}

impl SourceWatch {
    pub(super) fn capture(path: &Path) -> Result<Self, StatusError> {
        let capture = || -> std::io::Result<Self> {
            let file = File::open(path)?;
            let before = file.metadata()?;
            if !before.is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "source is not a regular file",
                ));
            }
            let identity = Handle::from_file(file.try_clone()?)?;
            Ok(Self {
                path: path.to_path_buf(),
                file,
                identity,
                before,
            })
        };
        capture().map_err(|error| StatusError::Io {
            path: path.to_path_buf(),
            detail: error.to_string(),
        })
    }

    pub(super) fn check(&mut self, expected_hash: &str) -> Result<(), String> {
        self.check_inner(expected_hash)
            .map_err(|error| error.to_string())
    }

    fn check_inner(&mut self, expected_hash: &str) -> std::io::Result<()> {
        self.file.rewind()?;
        let mut hash = Sha256::new();
        let mut bytes = [0_u8; 64 * 1024];
        let mut count = 0_u64;
        loop {
            let read = self.file.read(&mut bytes)?;
            if read == 0 {
                break;
            }
            count += read as u64;
            if count > self.before.len() {
                return changed();
            }
            hash.update(&bytes[..read]);
        }
        let after = self.file.metadata()?;
        let path_identity = Handle::from_path(&self.path)?;
        let path_metadata = std::fs::metadata(&self.path)?;
        if count != self.before.len()
            || format!("{:x}", hash.finalize()) != expected_hash
            || self.identity != path_identity
            || !same_metadata(&self.before, &after)?
            || !same_metadata(&after, &path_metadata)?
        {
            return changed();
        }
        Ok(())
    }
}

fn same_metadata(left: &Metadata, right: &Metadata) -> std::io::Result<bool> {
    Ok(right.is_file() && left.len() == right.len() && left.modified()? == right.modified()?)
}
fn changed<T>() -> std::io::Result<T> {
    Err(std::io::Error::other(
        "source identity or content changed during status observation",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_watch_detects_same_length_changes_replacement_and_removal() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("source.epub");
        let expected = format!("{:x}", Sha256::digest(b"original"));
        fs_write(&path, b"original");
        let mut watch = SourceWatch::capture(&path).unwrap();
        watch.check(&expected).unwrap();
        fs_write(&path, b"modified");
        assert!(watch.check(&expected).is_err());
        fs_write(&path, b"original");
        let mut watch = SourceWatch::capture(&path).unwrap();
        let replacement = directory.path().join("replacement.epub");
        fs_write(&replacement, b"original");
        std::fs::rename(&replacement, &path).unwrap();
        assert!(watch.check(&expected).is_err());
        let mut watch = SourceWatch::capture(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(watch.check(&expected).is_err());
    }

    fn fs_write(path: &Path, contents: &[u8]) {
        std::fs::write(path, contents).unwrap();
    }
}
