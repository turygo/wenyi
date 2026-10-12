mod rich;
mod segment;

use crate::{StoreError, StoreErrorKind, observation::*};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value};
use std::{collections::BTreeMap, path::Path};
use wenyi_core::SourceIdentity;

pub(crate) fn decode<T: DeserializeOwned>(value: &Value, path: &Path) -> Result<T, StoreError> {
    T::deserialize(value).map_err(|error| StoreError::wire(path, error.to_string()))
}
pub(crate) fn json(bytes: &[u8], path: &Path) -> Result<Value, StoreError> {
    serde_json::from_slice(bytes).map_err(|error| StoreError::wire(path, error.to_string()))
}
pub(crate) fn object<'a>(
    value: &'a Value,
    path: &Path,
) -> Result<&'a Map<String, Value>, StoreError> {
    value
        .as_object()
        .ok_or_else(|| StoreError::wire(path, "expected JSON object"))
}
pub(crate) fn required<T: DeserializeOwned>(
    map: &Map<String, Value>,
    key: &str,
    path: &Path,
) -> Result<T, StoreError> {
    decode(
        map.get(key)
            .ok_or_else(|| StoreError::wire(path, format!("missing {key}")))?,
        path,
    )
}
pub(crate) fn strict(
    map: &Map<String, Value>,
    fields: &[&str],
    path: &Path,
) -> Result<(), StoreError> {
    if let Some(key) = map.keys().find(|key| !fields.contains(&key.as_str())) {
        return Err(StoreError::wire(path, format!("unknown field {key}")));
    }
    Ok(())
}
pub(crate) fn sha256(value: &str, path: &Path) -> Result<(), StoreError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(StoreError::wire(path, "expected lowercase SHA256"));
    }
    Ok(())
}
pub(crate) fn version_error(path: &Path, code: &'static str, detail: &str) -> StoreError {
    StoreError::new(StoreErrorKind::Version, code, path, detail)
}
pub(crate) fn mismatch(path: &Path, detail: &str) -> StoreError {
    StoreError::new(
        StoreErrorKind::SourceMismatch,
        "source_mismatch",
        path,
        detail,
    )
}

#[derive(Deserialize)]
pub(crate) struct Manifest {
    #[serde(default = "current_version")]
    pub run_state_schema: u32,
    identity: Identity,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub title_translated: Option<String>,
    #[serde(default)]
    fmt: String,
    #[serde(default, rename = "source_path")]
    _source_path: String,
    #[serde(default)]
    pub source_lang: String,
    #[serde(default)]
    pub target_lang: String,
    #[serde(default = "segment::empty_object")]
    pub meta: Value,
    #[serde(default, rename = "initialized")]
    _initialized: bool,
    #[serde(default)]
    pub chapters: Vec<ChapterIndex>,
    #[serde(default)]
    pub progress: BTreeMap<String, Progress>,
    #[serde(default)]
    pub nodes: BTreeMap<String, Node>,
    #[serde(default, rename = "analysis_flags")]
    _analysis_flags: AnalysisFlags,
}
#[derive(Deserialize, Default)]
struct AnalysisFlags {
    #[serde(default, rename = "term_mining_done")]
    _term_mining_done: bool,
}
#[derive(Deserialize)]
struct Identity {
    source_bytes_sha256: String,
    #[serde(default = "input_version")]
    run_input_schema_version: u32,
    #[serde(default)]
    translation_policy_version: u32,
    #[serde(default, rename = "source_lang")]
    _source_lang: String,
    #[serde(default, rename = "target_lang")]
    _target_lang: String,
}
fn input_version() -> u32 {
    2
}

#[derive(Deserialize)]
pub(crate) struct ChapterIndex {
    pub index: usize,
    #[serde(default, rename = "title")]
    _title: String,
    #[serde(default, rename = "href")]
    _href: Option<String>,
    #[serde(default)]
    pub toc_entry_id: Option<String>,
    #[serde(default)]
    pub title_translated: Option<String>,
    #[serde(default)]
    pub processing: Option<ChapterProcessing>,
    #[serde(default)]
    pub status: Option<Value>,
}
fn current_version() -> u32 {
    4
}
#[derive(Deserialize, Default)]
pub(crate) struct Progress {
    #[serde(default)]
    status: ChapterStatus,
    #[serde(default)]
    pending_polish: Vec<PolishBatch>,
    #[serde(default)]
    lint_issues: Vec<Value>,
    #[serde(default)]
    repair_ledger: BTreeMap<String, RepairIssue>,
    #[serde(default)]
    back_matter_mode: Option<String>,
}
#[derive(Deserialize)]
pub(crate) struct Node {
    pub node_id: String,
    #[serde(default)]
    pub status: NodeStatus,
    #[serde(default)]
    pub attempts: u64,
    #[serde(default)]
    pub input_fingerprint: String,
    #[serde(default)]
    pub failure: Option<NodeFailure>,
    #[serde(default, rename = "started_at")]
    _started_at: Option<String>,
    #[serde(default, rename = "finished_at")]
    _finished_at: Option<String>,
    #[serde(default, rename = "output")]
    _output: Map<String, Value>,
}

pub(crate) fn manifest(
    value: &Value,
    source: &SourceIdentity,
    path: &Path,
) -> Result<Manifest, StoreError> {
    let manifest: Manifest = decode(value, path)?;
    if !matches!(manifest.run_state_schema, 1..=4) {
        return Err(version_error(
            path,
            "unsupported_run_schema",
            "unsupported bookkeeping version",
        ));
    }
    sha256(&manifest.identity.source_bytes_sha256, path)?;
    if manifest.identity.source_bytes_sha256 != source.source_bytes_sha256 {
        return Err(mismatch(path, "manifest source bytes do not match input"));
    }
    if !manifest.title.is_empty() && manifest.title != source.title {
        return Err(mismatch(path, "persisted source title differs"));
    }
    if manifest.identity.run_input_schema_version != 2
        || manifest.identity.translation_policy_version != 5
        || object(&manifest.meta, path)?
            .get("epub_schema")
            .and_then(Value::as_u64)
            != Some(4)
    {
        return Err(version_error(
            path,
            "unsupported_source_contract",
            "input/source slots/translation policy must be 2/4/5",
        ));
    }
    if !manifest.fmt.is_empty() && manifest.fmt != "epub" {
        return Err(StoreError::wire(path, "state format must be epub"));
    }
    let mut previous = None;
    for chapter in &manifest.chapters {
        if previous.is_some_and(|index| index >= chapter.index) {
            return Err(StoreError::wire(
                path,
                "chapter indices must be unique and ordered",
            ));
        }
        previous = Some(chapter.index);
        if let Some(processing) = &chapter.processing {
            validate_processing(processing, path)?;
        }
    }
    for key in manifest.progress.keys() {
        let index = key
            .parse::<usize>()
            .map_err(|_| StoreError::wire(path, "invalid progress chapter key"))?;
        if key != &index.to_string()
            || !manifest
                .chapters
                .iter()
                .any(|chapter| chapter.index == index)
        {
            return Err(StoreError::wire(
                path,
                "progress references unlisted chapter",
            ));
        }
    }
    Ok(manifest)
}

pub(crate) fn processing_progress(
    raw: Progress,
    chapter: usize,
    path: &Path,
) -> Result<ChapterProgress, StoreError> {
    if raw
        .pending_polish
        .iter()
        .any(|batch| batch.count == 0 || batch.start.checked_add(batch.count).is_none())
    {
        return Err(StoreError::wire(path, "invalid polish batch"));
    }
    for (key, issue) in &raw.repair_ledger {
        if key != &issue.key || issue.chapter != chapter {
            return Err(StoreError::wire(path, "repair ledger key/chapter mismatch"));
        }
    }
    if raw.lint_issues.iter().any(|issue| !issue.is_object()) {
        return Err(StoreError::wire(path, "lint issues must be objects"));
    }
    Ok(ChapterProgress {
        status: raw.status,
        pending_polish: raw.pending_polish,
        repair_ledger: raw.repair_ledger.into_values().collect(),
        lint_issues: raw.lint_issues,
        back_matter_mode: raw.back_matter_mode,
    })
}

#[derive(Deserialize)]
struct Chapter {
    index: usize,
    #[serde(default)]
    title: String,
    #[serde(default)]
    segments: Vec<Value>,
    #[serde(default)]
    href: Option<String>,
    #[serde(default = "segment::empty_object")]
    meta: Value,
    #[serde(default)]
    processing: Option<ChapterProcessing>,
}

pub(crate) fn chapter(
    value: &Value,
    index: &ChapterIndex,
    progress: ChapterProgress,
    path: &Path,
) -> Result<ObservedChapter, StoreError> {
    let raw: Chapter = decode(value, path)?;
    object(&raw.meta, path)?;
    if raw.index != index.index {
        return Err(StoreError::wire(path, "chapter file/index mismatch"));
    }
    if raw.processing != index.processing {
        return Err(StoreError::wire(
            path,
            "chapter processing differs from manifest",
        ));
    }
    if let Some(processing) = &raw.processing {
        validate_processing(processing, path)?;
    }
    let segments = raw
        .segments
        .iter()
        .map(|value| segment::admit(value, path))
        .collect::<Result<Vec<_>, _>>()?;
    let mut previous = None;
    for segment in &segments {
        if previous.is_some_and(|index| index >= segment.index) {
            return Err(StoreError::wire(
                path,
                "segment indices must be unique and ordered",
            ));
        }
        previous = Some(segment.index);
    }
    let text_count = segments
        .iter()
        .filter(|segment| !segment.source.trim().is_empty())
        .count();
    if progress.pending_polish.iter().any(|batch| {
        batch
            .start
            .checked_add(batch.count)
            .is_none_or(|end| end > text_count)
    }) || progress
        .repair_ledger
        .iter()
        .any(|issue| !segments.iter().any(|segment| segment.index == issue.index))
    {
        return Err(StoreError::wire(
            path,
            "progress range references missing segments",
        ));
    }
    Ok(ObservedChapter {
        index: raw.index,
        title: raw.title,
        title_translated: index.title_translated.clone(),
        href: raw.href,
        toc_entry_id: index.toc_entry_id.clone(),
        processing: raw.processing,
        progress,
        segments,
        meta: raw.meta,
    })
}

fn validate_processing(value: &ChapterProcessing, path: &Path) -> Result<(), StoreError> {
    if value.reason.trim().is_empty()
        || value.source_sha256.trim().is_empty()
        || value.strategy_version != "chapter_semantics_v2"
    {
        return Err(StoreError::wire(
            path,
            "invalid chapter processing decision",
        ));
    }
    Ok(())
}

pub(crate) use segment::selection;
