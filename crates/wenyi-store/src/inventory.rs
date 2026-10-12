use crate::{StoreError, StoreErrorKind};
use same_file::Handle;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::SystemTime,
};

const FILES: &[&str] = &[
    "manifest.json",
    ".run.lock",
    "journal.json",
    "note_migration.json",
    "usage.wal.json",
    "usage.json",
    "output_selection.json",
    "glossary.db",
    "glossary.db-wal",
    "glossary.db-shm",
    "glossary.db-journal",
];
const DIRECTORIES: &[&str] = &["chapters", "chapters_v2", "note_migration_backup"];

#[derive(Debug, PartialEq, Eq)]
struct Fingerprint {
    identity: Handle,
    size: u64,
    modified: SystemTime,
    digest: [u8; 32],
}

#[derive(Debug)]
pub(crate) struct Inventory {
    directories: BTreeMap<PathBuf, (Handle, Vec<OsString>)>,
    files: BTreeMap<PathBuf, Fingerprint>,
    pub bytes: BTreeMap<PathBuf, Vec<u8>>,
}

impl Inventory {
    pub(crate) fn capture(run: &Path) -> Result<Self, StoreError> {
        Self::capture_with_bytes(run, true)
    }

    fn capture_with_bytes(run: &Path, retain: bool) -> Result<Self, StoreError> {
        let mut inventory = Self {
            directories: BTreeMap::new(),
            files: BTreeMap::new(),
            bytes: BTreeMap::new(),
        };
        inventory.directory(run)?;
        for name in FILES {
            inventory.file(&run.join(name), retain)?;
        }
        for name in DIRECTORIES {
            let directory = run.join(name);
            if exists(&directory)? {
                let names = inventory.directory(&directory)?;
                for name in names {
                    inventory.file(&directory.join(name), retain)?;
                }
            }
        }
        Ok(inventory)
    }

    fn directory(&mut self, path: &Path) -> Result<Vec<OsString>, StoreError> {
        let identity = Handle::from_path(path).map_err(|error| StoreError::io(path, error))?;
        let mut names = fs::read_dir(path)
            .map_err(|error| StoreError::io(path, error))?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| StoreError::io(path, error))?;
        names.sort();
        self.directories
            .insert(path.to_owned(), (identity, names.clone()));
        Ok(names)
    }

    fn file(&mut self, path: &Path, retain: bool) -> Result<(), StoreError> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(StoreError::io(path, error)),
        };
        let before = file
            .metadata()
            .map_err(|error| StoreError::io(path, error))?;
        if !before.is_file() {
            return Err(StoreError::new(
                StoreErrorKind::Unavailable,
                "nonregular_artifact",
                path,
                "observation requires regular artifacts",
            ));
        }
        let identity = Handle::from_file(
            file.try_clone()
                .map_err(|error| StoreError::io(path, error))?,
        )
        .map_err(|error| StoreError::io(path, error))?;
        let mut bytes = Vec::new();
        let retain_json = retain
            && path
                .extension()
                .is_some_and(|extension| extension == "json");
        let retain_journal = retain
            && path
                .file_name()
                .is_some_and(|name| name == "glossary.db-journal");
        let mut buffer = [0_u8; 8192];
        let mut digest = Sha256::new();
        let mut size = 0_u64;
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| StoreError::io(path, error))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
            if retain_json {
                bytes.extend_from_slice(&buffer[..count]);
            } else if retain_journal && bytes.len() < 8 {
                bytes.extend_from_slice(&buffer[..count.min(8 - bytes.len())]);
            }
            size += count as u64;
        }
        let after = file
            .metadata()
            .map_err(|error| StoreError::io(path, error))?;
        let modified = after
            .modified()
            .map_err(|error| StoreError::io(path, error))?;
        if before.len() != after.len()
            || before
                .modified()
                .map_err(|error| StoreError::io(path, error))?
                != modified
            || size != after.len()
        {
            return Err(StoreError::new(
                StoreErrorKind::Unavailable,
                "changing_files",
                path,
                "artifact changed while being read",
            ));
        }
        self.files.insert(
            path.to_owned(),
            Fingerprint {
                identity,
                size: after.len(),
                modified,
                digest: digest.finalize().into(),
            },
        );
        self.bytes.insert(path.to_owned(), bytes);
        Ok(())
    }

    pub(crate) fn unchanged(&self, run: &Path) -> Result<bool, StoreError> {
        let after = Self::capture_with_bytes(run, false)?;
        Ok(self.directories == after.directories && self.files == after.files)
    }
    pub(crate) fn get(&self, path: &Path) -> Option<&[u8]> {
        self.bytes.get(path).map(Vec::as_slice)
    }
    pub(crate) fn size(&self, path: &Path) -> Option<u64> {
        self.files.get(path).map(|fingerprint| fingerprint.size)
    }
    pub(crate) fn matches_file(&self, path: &Path, file: &File) -> Result<bool, StoreError> {
        let handle = Handle::from_file(
            file.try_clone()
                .map_err(|error| StoreError::io(path, error))?,
        )
        .map_err(|error| StoreError::io(path, error))?;
        Ok(self
            .files
            .get(path)
            .is_some_and(|fingerprint| fingerprint.identity == handle))
    }
}

pub(crate) fn exists(path: &Path) -> Result<bool, StoreError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(StoreError::io(path, error)),
    }
}

pub(crate) enum ObservationLock {
    Absent,
    Held(File),
    Contended,
}

pub(crate) fn lock(run: &Path) -> Result<ObservationLock, StoreError> {
    let path = run.join(".run.lock");
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ObservationLock::Absent);
        }
        Err(error) => return Err(StoreError::io(&path, error)),
    };
    match file.try_lock_shared() {
        Ok(()) => Ok(ObservationLock::Held(file)),
        Err(std::fs::TryLockError::WouldBlock) => Ok(ObservationLock::Contended),
        Err(std::fs::TryLockError::Error(error)) => Err(StoreError::io(&path, error)),
    }
}
