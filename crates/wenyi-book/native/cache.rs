#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Component as PathComponent, Path, PathBuf},
};

pub type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
pub(crate) const MAX_ARCHIVE: u64 = 128 * 1024 * 1024;
pub(crate) const MAX_EXTRACTED: u64 = 1024 * 1024 * 1024;
pub(crate) const MAX_FILES: usize = 100_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceLock {
    pub schema_version: u32,
    pub components: Vec<Component>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub name: String,
    pub version: String,
    pub archive: Archive,
    pub license: String,
    pub notices: Vec<String>,
    pub patches: Vec<serde_json::Value>,
    pub features: Vec<String>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Archive {
    pub filename: String,
    pub url: String,
    pub sha256: String,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub name: String,
    pub version: String,
    pub archive_sha256: String,
    pub files: Vec<FileDigest>,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FileDigest {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
}

pub fn lock(bytes: &[u8]) -> Result<SourceLock> {
    let lock: SourceLock = serde_json::from_slice(bytes)?;
    if lock.schema_version != 1 || lock.components.len() != 3 {
        return Err("unsupported native source lock".into());
    }
    let mut previous = "";
    for item in &lock.components {
        if item.name.as_str() <= previous || !item.patches.is_empty() {
            return Err("native lock order/patch policy mismatch".into());
        }
        previous = &item.name;
        safe_relative(&item.archive.filename)?;
        safe_relative(&item.key())?;
        if item.license.is_empty() || item.features.is_empty() || item.notices.is_empty() {
            return Err("native license/features/notices declarations are required".into());
        }
        for notice in &item.notices {
            safe_relative(notice)?;
        }
        if item.archive.filename.contains('/')
            || !item.archive.url.starts_with("https://")
            || item.archive.sha256.len() != 64
            || !item
                .archive
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("invalid native archive declaration".into());
        }
    }
    Ok(lock)
}
impl Component {
    pub fn key(&self) -> String {
        format!("{}-{}", self.name, self.version)
    }
    pub fn source(&self, cache: &Path) -> PathBuf {
        cache.join("sources").join(self.key())
    }
    pub(crate) fn manifest(&self, cache: &Path) -> PathBuf {
        cache.join("manifests").join(format!("{}.json", self.key()))
    }
}

pub fn safe_relative(value: &str) -> Result<PathBuf> {
    if value.is_empty() || value.contains(['\\', ':', '\0']) {
        return Err(format!("unsafe native path: {value:?}").into());
    }
    let path = Path::new(value);
    if path
        .components()
        .any(|part| !matches!(part, PathComponent::Normal(_)))
    {
        return Err(format!("unsafe native path: {value:?}").into());
    }
    Ok(path.to_owned())
}

pub fn hash_file(path: &Path, maximum: u64) -> Result<(u64, String)> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.len() > maximum {
        return Err(format!(
            "native input is not a bounded regular file: {}",
            path.display()
        )
        .into());
    }
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        total = total
            .checked_add(size as u64)
            .ok_or("native input length overflow")?;
        if total > maximum {
            return Err("native input grew beyond limit".into());
        }
        hash.update(&buffer[..size]);
    }
    if total != meta.len() {
        return Err("native input changed while hashing".into());
    }
    Ok((total, format!("{:x}", hash.finalize())))
}

fn inventory(root: &Path) -> Result<Vec<FileDigest>> {
    if !fs::symlink_metadata(root)?.is_dir() {
        return Err("native source root is not a directory".into());
    }
    let mut pending = vec![root.to_owned()];
    let mut files = Vec::new();
    let mut total = 0_u64;
    let mut entries = 0_usize;
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            entries += 1;
            if entries > MAX_FILES {
                return Err("native source inventory limit exceeded".into());
            }
            let path = entry.path();
            let meta = fs::symlink_metadata(&path)?;
            if meta.is_dir() {
                pending.push(path);
                continue;
            }
            if !meta.is_file() {
                return Err(format!("special native input: {}", path.display()).into());
            }
            let relative = path
                .strip_prefix(root)?
                .to_str()
                .ok_or("non-UTF8 native input path")?
                .replace('\\', "/");
            safe_relative(&relative)?;
            let (size_bytes, sha256) = hash_file(&path, MAX_EXTRACTED)?;
            total = total
                .checked_add(size_bytes)
                .ok_or("native inventory overflow")?;
            if total > MAX_EXTRACTED {
                return Err("native inventory size limit exceeded".into());
            }
            files.push(FileDigest {
                path: relative,
                size_bytes,
                sha256,
            });
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.is_empty() {
        return Err("empty native source inventory".into());
    }
    Ok(files)
}

pub(crate) fn make_manifest(item: &Component, root: &Path) -> Result<Manifest> {
    Ok(Manifest {
        schema_version: 1,
        name: item.name.clone(),
        version: item.version.clone(),
        archive_sha256: item.archive.sha256.clone(),
        files: inventory(root)?,
    })
}

pub fn verify(cache: &Path, item: &Component) -> Result<PathBuf> {
    let archive = cache.join("archives").join(&item.archive.filename);
    let (_, actual) = hash_file(&archive, MAX_ARCHIVE).map_err(|error| {
        format!(
            "native archive {}: {error}; run cargo xtask native provision --cache DIR",
            archive.display()
        )
    })?;
    if actual != item.archive.sha256 {
        return Err(format!("native archive SHA256 mismatch: {}", archive.display()).into());
    }
    let manifest_path = item.manifest(cache);
    let manifest: Manifest = read_manifest(&manifest_path).map_err(|error| {
        format!(
            "native completeness manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    let source = item.source(cache);
    let actual = make_manifest(item, &source)?;
    if manifest.schema_version != actual.schema_version
        || manifest.name != actual.name
        || manifest.version != actual.version
        || manifest.archive_sha256 != actual.archive_sha256
    {
        return Err(format!(
            "native manifest identity mismatch: {}",
            manifest_path.display()
        )
        .into());
    }
    if manifest.files != actual.files {
        let first = manifest
            .files
            .iter()
            .zip(&actual.files)
            .find(|(expected, observed)| expected != observed)
            .map(|(expected, _)| expected.path.as_str())
            .or_else(|| {
                let index = manifest.files.len().min(actual.files.len());
                manifest
                    .files
                    .get(index)
                    .or_else(|| actual.files.get(index))
                    .map(|file| file.path.as_str())
            })
            .unwrap_or("<inventory>");
        return Err(format!(
            "native source inventory mismatch: {}/{}",
            source.display(),
            first
        )
        .into());
    }
    Ok(source)
}

fn read_manifest(path: &Path) -> Result<Manifest> {
    const MAX_MANIFEST: u64 = 32 * 1024 * 1024;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.len() > MAX_MANIFEST {
        return Err("invalid native manifest size/type".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_MANIFEST + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("native manifest grew beyond limit".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_and_corrupt_cache_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let lock = lock(include_bytes!("../../../native/sources.lock.json")).unwrap();
        let item = &lock.components[0];
        assert!(
            verify(directory.path(), item)
                .unwrap_err()
                .to_string()
                .contains("native archive")
        );
        fs::create_dir(directory.path().join("archives")).unwrap();
        fs::write(
            directory
                .path()
                .join("archives")
                .join(&item.archive.filename),
            b"corrupt",
        )
        .unwrap();
        assert!(
            verify(directory.path(), item)
                .unwrap_err()
                .to_string()
                .contains("SHA256 mismatch")
        );
    }
    #[test]
    fn extracted_bytes_inventory_and_completeness_are_rechecked() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        for name in ["archives", "sources", "manifests"] {
            fs::create_dir(root.join(name)).unwrap();
        }
        let mut lock = lock(include_bytes!("../../../native/sources.lock.json")).unwrap();
        let item = &mut lock.components[0];
        let archive = root.join("archives").join(&item.archive.filename);
        fs::write(&archive, b"isolated native integrity test").unwrap();
        item.archive.sha256 = hash_file(&archive, MAX_ARCHIVE).unwrap().1;
        let source = item.source(root);
        fs::create_dir(&source).unwrap();
        fs::write(source.join("input.c"), b"original").unwrap();
        assert!(
            verify(root, item)
                .unwrap_err()
                .to_string()
                .contains("completeness")
        );
        let manifest = make_manifest(item, &source).unwrap();
        fs::write(item.manifest(root), serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert_eq!(verify(root, item).unwrap(), source);
        fs::write(source.join("input.c"), b"modified").unwrap();
        assert!(
            verify(root, item)
                .unwrap_err()
                .to_string()
                .contains("inventory mismatch")
        );
        fs::write(source.join("input.c"), b"original").unwrap();
        fs::write(source.join("undeclared.c"), b"extra").unwrap();
        assert!(
            verify(root, item)
                .unwrap_err()
                .to_string()
                .contains("inventory mismatch")
        );
    }
}
