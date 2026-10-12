use super::{invalid, usage::hash};
use crate::{StoreError, inventory::Inventory, wire};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};
use std::{collections::HashSet, path::Path};
use wenyi_core::{SourceIdentity, digest::stable_digest};

pub(crate) fn validate(
    value: &Value,
    inventory: &Inventory,
    run: &Path,
    source: &SourceIdentity,
    path: &Path,
) -> Result<(), StoreError> {
    let record = wire::object(value, path)?;
    if record.get("version").and_then(Value::as_u64) != Some(1) {
        return Err(invalid(path, "unsupported note migration version"));
    }
    let source_hash: String = wire::required(record, "source_bytes_sha256", path)?;
    if source_hash != source.source_bytes_sha256 {
        return Err(invalid(path, "note migration source binding differs"));
    }
    let before_meta = payload(record, "before_meta", path)?;
    let after_meta = payload(record, "after_meta", path)?;
    let before_manifest = payload(record, "before_manifest", path)?;
    let after_manifest = payload(record, "after_manifest", path)?;
    if before_manifest.get("meta") != Some(before_meta)
        || after_manifest.get("meta") != Some(after_meta)
    {
        return Err(invalid(path, "note migration manifest metadata mismatch"));
    }
    let before = wire::manifest(before_manifest, source, path)?;
    let after = wire::manifest(after_manifest, source, path)?;
    let before_indices: Vec<_> = before
        .chapters
        .iter()
        .map(|chapter| chapter.index)
        .collect();
    let after_indices: Vec<_> = after.chapters.iter().map(|chapter| chapter.index).collect();
    if before_indices != after_indices {
        return Err(invalid(path, "note migration chapter index set differs"));
    }
    let current_path = run.join("manifest.json");
    let current = read(inventory, &current_path)?;
    before_or_after(&current, before_manifest, after_manifest, path)?;
    backup(
        inventory,
        &run.join("note_migration_backup/manifest_meta.json"),
        &source_hash,
        before_meta,
        path,
    )?;
    let chapters = record
        .get("chapters")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(path, "note migration lacks chapter list"))?;
    let mut seen = HashSet::new();
    for item in chapters {
        validate_chapter(
            item,
            (&before, &after),
            &mut seen,
            inventory,
            run,
            &source_hash,
            path,
        )?;
    }
    Ok(())
}

fn validate_chapter(
    item: &Value,
    manifests: (&wire::Manifest, &wire::Manifest),
    seen: &mut HashSet<usize>,
    inventory: &Inventory,
    run: &Path,
    source: &str,
    path: &Path,
) -> Result<(), StoreError> {
    let item = wire::object(item, path)?;
    let index: usize = wire::required(item, "index", path)?;
    if !manifests
        .0
        .chapters
        .iter()
        .any(|chapter| chapter.index == index)
        || !seen.insert(index)
    {
        return Err(invalid(
            path,
            "note migration chapter is duplicate or unlisted",
        ));
    }
    let before = payload(item, "before", path)?;
    let after = payload(item, "after", path)?;
    if before.get("index").and_then(Value::as_u64) != Some(index as u64)
        || after.get("index").and_then(Value::as_u64) != Some(index as u64)
    {
        return Err(invalid(path, "note migration chapter index mismatch"));
    }
    for (data, manifest) in [(before, manifests.0), (after, manifests.1)] {
        let chapter_index = manifest
            .chapters
            .iter()
            .find(|chapter| chapter.index == index)
            .ok_or_else(|| invalid(path, "note migration chapter is unlisted"))?;
        wire::chapter(data, chapter_index, crate::ChapterProgress::default(), path)?;
    }
    let encoded: String = wire::required(item, "before_bytes_base64", path)?;
    let bytes = STANDARD
        .decode(&encoded)
        .map_err(|error| invalid(path, error.to_string()))?;
    let before_hash: String = wire::required(item, "before_file_sha256", path)?;
    wire::sha256(&before_hash, path)?;
    if hash(&bytes) != before_hash
        || digest(&wire::json(&bytes, path)?, path)? != digest(before, path)?
    {
        return Err(invalid(
            path,
            "note migration chapter backup digest/payload mismatch",
        ));
    }
    let current = read(inventory, &run.join(format!("chapters_v2/ch{index}.json")))?;
    before_or_after(&current, before, after, path)?;
    let backup_data = json!({"chapter_bytes_base64": encoded, "chapter_sha256": before_hash});
    backup(
        inventory,
        &run.join(format!("note_migration_backup/ch{index}.json")),
        source,
        &backup_data,
        path,
    )?;
    Ok(())
}

fn payload<'a>(
    record: &'a Map<String, Value>,
    name: &str,
    path: &Path,
) -> Result<&'a Value, StoreError> {
    let wrapped = wire::object(
        record
            .get(name)
            .ok_or_else(|| invalid(path, format!("missing {name}")))?,
        path,
    )?;
    let data = wrapped
        .get("data")
        .ok_or_else(|| invalid(path, format!("missing {name} payload")))?;
    wire::object(data, path)?;
    let expected: String = wire::required(wrapped, "sha256", path)?;
    wire::sha256(&expected, path)?;
    if digest(data, path)? != expected {
        return Err(invalid(path, format!("invalid {name} digest")));
    }
    Ok(data)
}
fn digest(value: &Value, path: &Path) -> Result<String, StoreError> {
    stable_digest(value).map_err(|error| invalid(path, error.to_string()))
}
fn read(inventory: &Inventory, path: &Path) -> Result<Value, StoreError> {
    let bytes = inventory
        .get(path)
        .ok_or_else(|| invalid(path, "missing note migration artifact"))?;
    wire::json(bytes, path)
}
fn before_or_after(
    current: &Value,
    before: &Value,
    after: &Value,
    path: &Path,
) -> Result<(), StoreError> {
    let current = digest(current, path)?;
    if current != digest(before, path)? && current != digest(after, path)? {
        return Err(invalid(
            path,
            "artifact changed independently during note migration",
        ));
    }
    Ok(())
}
fn backup(
    inventory: &Inventory,
    backup_path: &Path,
    source: &str,
    data: &Value,
    path: &Path,
) -> Result<(), StoreError> {
    if let Some(bytes) = inventory.get(backup_path) {
        let actual = wire::json(bytes, backup_path)?;
        let expected = json!({"version": 1, "source_bytes_sha256": source, "data": data});
        if actual != expected {
            return Err(invalid(path, "conflicting note migration backup"));
        }
    }
    Ok(())
}
