#![forbid(unsafe_code)]
#[cfg(test)]
use super::cache;
use super::cache::{
    MAX_ARCHIVE, MAX_EXTRACTED, MAX_FILES, Result, SourceLock, hash_file, make_manifest,
    safe_relative, verify,
};
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
};
fn unpack(archive: &Path, destination: &Path, filename: &str) -> Result<()> {
    let file = fs::File::open(archive)?;
    let reader: Box<dyn Read> = if filename.ends_with(".xz") {
        Box::new(xz2::read::XzDecoder::new(file))
    } else {
        Box::new(flate2::read::GzDecoder::new(file))
    };
    let mut tar = tar::Archive::new(reader);
    let mut seen = BTreeSet::new();
    let mut root_name: Option<String> = None;
    let mut total = 0_u64;
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let text = path
            .to_str()
            .ok_or("non-UTF8 native archive path")?
            .trim_end_matches('/');
        let safe = safe_relative(text)?;
        if !seen.insert(text.to_owned()) || seen.len() > MAX_FILES {
            return Err(format!("duplicate/overlimit native archive entry: {text}").into());
        }
        let mut parts = safe.components();
        let root = parts
            .next()
            .ok_or("empty archive path")?
            .as_os_str()
            .to_str()
            .ok_or("invalid archive root")?;
        if root_name.as_deref().is_some_and(|name| name != root) {
            return Err("multiple native archive roots".into());
        }
        root_name.get_or_insert_with(|| root.to_owned());
        let relative: PathBuf = parts.collect();
        let kind = entry.header().entry_type();
        if !kind.is_dir() && !kind.is_file() {
            return Err(format!("unsupported native archive link/special entry: {text}; requires reviewed pin-specific decision").into());
        }
        if relative.as_os_str().is_empty() {
            if !kind.is_dir() {
                return Err("native archive root is not directory".into());
            }
            continue;
        }
        let output = destination.join(relative);
        if kind.is_dir() {
            fs::create_dir_all(output)?;
            continue;
        }
        total = total
            .checked_add(entry.size())
            .ok_or("native extraction overflow")?;
        if total > MAX_EXTRACTED {
            return Err("native extraction limit exceeded".into());
        }
        fs::create_dir_all(output.parent().ok_or("invalid native destination")?)?;
        let mut target = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)?;
        if io::copy(&mut entry, &mut target)? != entry.size() {
            return Err("native archive entry truncated".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                &output,
                fs::Permissions::from_mode(entry.header().mode()? & 0o755),
            )?;
        }
    }
    Ok(())
}

pub fn provision(cache: &Path, lock: &SourceLock) -> Result<()> {
    fs::create_dir_all(cache)?;
    for directory in ["archives", "sources", "manifests"] {
        fs::create_dir_all(cache.join(directory))?;
    }
    for item in &lock.components {
        let archive = cache.join("archives").join(&item.archive.filename);
        let source = item.source(cache);
        let manifest_path = item.manifest(cache);
        if [&archive, &source, &manifest_path]
            .iter()
            .any(|path| path.symlink_metadata().is_ok())
        {
            verify(cache, item)?;
            continue;
        }
        let stage = tempfile::Builder::new()
            .prefix("native-stage-")
            .tempdir_in(cache)?;
        let staged_archive = stage.path().join(&item.archive.filename);
        let mut response = ureq::get(&item.archive.url).call()?;
        let mut target = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staged_archive)?;
        let written = io::copy(
            &mut response.body_mut().as_reader().take(MAX_ARCHIVE + 1),
            &mut target,
        )?;
        if written > MAX_ARCHIVE {
            return Err("native download limit exceeded".into());
        }
        target.sync_all()?;
        if hash_file(&staged_archive, MAX_ARCHIVE)?.1 != item.archive.sha256 {
            return Err(format!(
                "downloaded native archive SHA256 mismatch: {}",
                item.archive.filename
            )
            .into());
        }
        let staged_source = stage.path().join("source");
        fs::create_dir(&staged_source)?;
        unpack(&staged_archive, &staged_source, &item.archive.filename)?;
        let manifest = make_manifest(item, &staged_source)?;
        let staged_manifest = stage.path().join("manifest.json");
        fs::write(&staged_manifest, serde_json::to_vec_pretty(&manifest)?)?;
        let publication_lock = cache.join(format!("{}.provision-lock", item.key()));
        let guard = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&publication_lock)?;
        let publication = (|| -> Result<()> {
            if [&archive, &source, &manifest_path]
                .iter()
                .any(|path| path.symlink_metadata().is_ok())
            {
                return Err(
                    "native cache entry appeared during provisioning; refusing overwrite".into(),
                );
            }
            fs::rename(&staged_archive, &archive)?;
            fs::rename(&staged_source, &source)?;
            fs::rename(&staged_manifest, &manifest_path)?;
            Ok(())
        })();
        drop(guard);
        fs::remove_file(publication_lock)?;
        publication?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupt_existing_archive_is_not_downloaded_or_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let lock = cache::lock(include_bytes!("../../../native/sources.lock.json")).unwrap();
        let item = &lock.components[0];
        fs::create_dir(directory.path().join("archives")).unwrap();
        let path = directory
            .path()
            .join("archives")
            .join(&item.archive.filename);
        fs::write(&path, b"corrupt").unwrap();
        assert!(
            provision(directory.path(), &lock)
                .unwrap_err()
                .to_string()
                .contains("SHA256 mismatch")
        );
        assert_eq!(fs::read(path).unwrap(), b"corrupt");
    }
    #[test]
    fn unpack_rejects_links_and_escape() {
        for value in ["../escape", "/root", "a/../b", "a\\b", "C:/file"] {
            assert!(safe_relative(value).is_err());
        }
        let directory = tempfile::tempdir().unwrap();
        let archive = directory.path().join("input.tgz");
        let gz = flate2::write::GzEncoder::new(
            fs::File::create(&archive).unwrap(),
            flate2::Compression::default(),
        );
        let mut builder = tar::Builder::new(gz);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_size(0);
        header.set_mode(0o777);
        builder
            .append_link(&mut header, "root/link", "../../escape")
            .unwrap();
        builder.into_inner().unwrap().finish().unwrap();
        assert!(
            unpack(&archive, directory.path(), "input.tgz")
                .unwrap_err()
                .to_string()
                .contains("link/special")
        );
    }
}
