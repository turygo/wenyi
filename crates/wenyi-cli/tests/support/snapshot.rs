use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, Metadata},
    io::Read,
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct SnapshotEntry {
    kind: (bool, bool, bool),
    size: u64,
    modified: SystemTime,
    permissions: u32,
    symlink_target: Option<PathBuf>,
    sha256: String,
}

pub(crate) fn snapshot_tree(root: &Path) -> BTreeMap<PathBuf, SnapshotEntry> {
    let mut result = BTreeMap::new();
    visit(root, root, &mut result);
    result
}

fn visit(root: &Path, path: &Path, result: &mut BTreeMap<PathBuf, SnapshotEntry>) {
    let metadata = fs::symlink_metadata(path).expect("read snapshot metadata");
    let kind = metadata.file_type();
    result.insert(
        path.strip_prefix(root).unwrap().to_path_buf(),
        SnapshotEntry {
            kind: (kind.is_file(), kind.is_dir(), kind.is_symlink()),
            size: metadata.len(),
            modified: metadata.modified().unwrap(),
            permissions: permissions(&metadata),
            symlink_target: kind.is_symlink().then(|| fs::read_link(path).unwrap()),
            sha256: if kind.is_file() {
                file_hash(path)
            } else {
                String::new()
            },
        },
    );
    if kind.is_dir() {
        for entry in fs::read_dir(path).unwrap() {
            visit(root, &entry.unwrap().path(), result);
        }
    }
}

fn file_hash(path: &Path) -> String {
    let mut file = File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut bytes = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut bytes).unwrap();
        if count == 0 {
            break;
        }
        hasher.update(&bytes[..count]);
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(unix)]
fn permissions(metadata: &Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode()
}
#[cfg(not(unix))]
fn permissions(metadata: &Metadata) -> u32 {
    u32::from(metadata.permissions().readonly())
}
