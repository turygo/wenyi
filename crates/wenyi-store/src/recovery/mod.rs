mod note;
mod usage;
mod usage_json;

use crate::{StoreError, StoreErrorKind, observation::ObservedChapter, wire};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

pub(crate) use note::validate as note;
pub(crate) use usage::{snapshot as usage_snapshot, wal as usage_wal};

fn invalid(path: &Path, detail: impl Into<String>) -> StoreError {
    StoreError::new(
        StoreErrorKind::Recovery,
        "invalid_recovery_record",
        path,
        detail,
    )
}

#[derive(Deserialize)]
struct Journal {
    kind: String,
    chapter: usize,
    start: usize,
    count: usize,
    #[serde(default)]
    targets: Option<Vec<String>>,
}

pub(crate) fn checkpoint(
    value: &Value,
    chapters: &[ObservedChapter],
    path: &Path,
) -> Result<(), StoreError> {
    let record = journal(value, path)?;
    let chapter = chapters
        .iter()
        .find(|chapter| chapter.index == record.chapter)
        .ok_or_else(|| invalid(path, "checkpoint references missing chapter"))?;
    let length = chapter
        .segments
        .iter()
        .filter(|segment| !segment.source.trim().is_empty())
        .count();
    if record
        .start
        .checked_add(record.count)
        .is_none_or(|end| end > length)
    {
        return Err(invalid(
            path,
            "checkpoint batch is outside chapter text segments",
        ));
    }
    Ok(())
}

pub(crate) fn checkpoint_shape(value: &Value, path: &Path) -> Result<(), StoreError> {
    journal(value, path).map(|_| ())
}

fn journal(value: &Value, path: &Path) -> Result<Journal, StoreError> {
    let record: Journal = wire::decode(value, path)?;
    if !matches!(record.kind.as_str(), "translate" | "polish")
        || record.count == 0
        || record.start.checked_add(record.count).is_none()
    {
        return Err(invalid(path, "unknown checkpoint kind or invalid batch"));
    }
    if record.kind == "polish"
        && record
            .targets
            .as_ref()
            .is_none_or(|targets| targets.len() != record.count)
    {
        return Err(invalid(path, "polish checkpoint lacks exact target list"));
    }
    Ok(record)
}
