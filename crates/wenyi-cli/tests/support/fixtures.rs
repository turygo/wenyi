use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use tempfile::TempDir;

#[path = "fixture_paths.rs"]
mod paths;
#[path = "fixture_workspace.rs"]
mod workspace;

use paths::{check_hash, check_size, existing_file, io_error, validate_workspace_layout};

const FIXTURE_ID: &str = "structured-epub-balanced-v1";
const BASELINE_COMMIT: &str = "61835587606b5cb81cdcd81eec8427fd4f638df4";
const REFERENCE_PATH: &str = "state/structured-epub-balanced-v1/reference.json";
const MAX_REFERENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_FILES: usize = 10_000;
const MAX_PATH_BYTES: usize = 4096;
const MAX_PATH_COMPONENTS: usize = 128;

#[derive(Debug)]
pub(crate) enum FixtureError {
    InvalidMetadata(String),
    BaselineMismatch {
        expected: &'static str,
        actual: String,
    },
    UnsafePath(String),
    MissingArtifact(String),
    ArtifactSizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    HashMismatch {
        path: String,
    },
    Io {
        path: String,
        message: String,
    },
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMetadata(message) => {
                write!(formatter, "invalid fixture metadata: {message}")
            }
            Self::BaselineMismatch { expected, actual } => {
                write!(
                    formatter,
                    "fixture baseline mismatch: expected {expected}, got {actual}"
                )
            }
            Self::UnsafePath(path) => write!(formatter, "unsafe fixture path: {path}"),
            Self::MissingArtifact(path) => write!(formatter, "missing fixture artifact: {path}"),
            Self::ArtifactSizeMismatch {
                path,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "fixture size mismatch at {path}: expected {expected}, got {actual}"
                )
            }
            Self::HashMismatch { path } => write!(formatter, "fixture hash mismatch at {path}"),
            Self::Io { path, message } => {
                write!(formatter, "fixture I/O error at {path}: {message}")
            }
        }
    }
}

impl std::error::Error for FixtureError {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureReference {
    schema_version: u32,
    fixture_id: String,
    baseline_commit: String,
    reference_inputs: Value,
    environment: Value,
    wire_versions: Value,
    files: Vec<FileRecord>,
    outputs: Vec<OutputRecord>,
    semantic_projection: ProjectionRecord,
    generation: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileRecord {
    path: String,
    size_bytes: u64,
    sha256: String,
    #[serde(default)]
    workspace_path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputRecord {
    role: OutputRole,
    file: String,
    proof: String,
    receipt: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OutputRole {
    Mono,
    BilingualTargetFirst,
    BilingualSourceFirst,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectionRecord {
    path: String,
    sha256: String,
    format_version: u32,
}

#[derive(Clone, Debug)]
struct RelativePath {
    raw: String,
    components: Vec<String>,
}

struct AdmittedFile {
    record_index: usize,
    source_path: RelativePath,
    workspace_path: Option<RelativePath>,
}

pub(crate) struct AdmittedFixture {
    root: PathBuf,
    metadata: FixtureReference,
    files: Vec<AdmittedFile>,
}

impl AdmittedFixture {
    pub(crate) fn fixture_id(&self) -> &str {
        &self.metadata.fixture_id
    }

    pub(crate) fn workspace_mappings(&self) -> impl Iterator<Item = (&str, &str)> {
        self.files.iter().filter_map(|file| {
            let destination = file.workspace_path.as_ref()?;
            let record = &self.metadata.files[file.record_index];
            Some((record.path.as_str(), destination.raw.as_str()))
        })
    }
}

pub(crate) fn admit_fixture(root: &Path) -> Result<AdmittedFixture, FixtureError> {
    let canonical_root = fs::canonicalize(root).map_err(|error| io_error(root, error))?;
    let root_metadata =
        fs::metadata(&canonical_root).map_err(|error| io_error(&canonical_root, error))?;
    if !root_metadata.is_dir() {
        return Err(FixtureError::UnsafePath(root.display().to_string()));
    }
    let reference_relative = RelativePath::parse(REFERENCE_PATH)?;
    let (reference_file, reference_metadata) = existing_file(&canonical_root, &reference_relative)?;
    if reference_metadata.len() > MAX_REFERENCE_BYTES {
        return Err(FixtureError::InvalidMetadata(
            "reference.json exceeds the metadata size limit".to_owned(),
        ));
    }
    let reference_bytes =
        fs::read(&reference_file).map_err(|error| io_error(&reference_file, error))?;
    let metadata: FixtureReference = serde_json::from_slice(&reference_bytes)
        .map_err(|error| FixtureError::InvalidMetadata(error.to_string()))?;

    validate_envelope(&metadata)?;
    let mut admitted_files = Vec::with_capacity(metadata.files.len());
    let mut canonical_paths = HashSet::with_capacity(metadata.files.len() + 1);
    canonical_paths.insert(reference_file);
    let mut workspace_paths = HashSet::with_capacity(metadata.files.len());
    let mut total_bytes = 0_u64;
    if metadata
        .files
        .windows(2)
        .any(|pair| pair[0].path.as_str() >= pair[1].path.as_str())
    {
        return Err(FixtureError::InvalidMetadata(
            "files must be sorted by unique path".to_owned(),
        ));
    }

    for (record_index, record) in metadata.files.iter().enumerate() {
        let source_path = RelativePath::parse(&record.path)?;
        validate_sha256(&record.sha256, &record.path)?;
        if record.path == REFERENCE_PATH {
            return Err(FixtureError::InvalidMetadata(
                "reference.json must not list itself".to_owned(),
            ));
        }
        if record.size_bytes > MAX_ARTIFACT_BYTES {
            return Err(FixtureError::InvalidMetadata(format!(
                "artifact exceeds the size limit: {}",
                record.path
            )));
        }
        total_bytes = total_bytes.checked_add(record.size_bytes).ok_or_else(|| {
            FixtureError::InvalidMetadata("total artifact size overflows".to_owned())
        })?;
        if total_bytes > MAX_TOTAL_BYTES {
            return Err(FixtureError::InvalidMetadata(
                "total artifact size exceeds the limit".to_owned(),
            ));
        }
        let workspace_raw = record.workspace_path.as_deref();
        let workspace_path = workspace_raw.map(RelativePath::parse).transpose()?;
        if let Some(path) = workspace_raw
            && !workspace_paths.insert(path)
        {
            return Err(FixtureError::InvalidMetadata(format!(
                "duplicate workspace destination: {path}"
            )));
        }

        let (canonical_path, actual_metadata) = existing_file(&canonical_root, &source_path)?;
        if canonical_paths.contains(&canonical_path) {
            return Err(FixtureError::UnsafePath(record.path.clone()));
        }
        check_size(&record.path, record.size_bytes, actual_metadata.len())?;
        check_hash(&canonical_path, record)?;
        canonical_paths.insert(canonical_path);
        admitted_files.push(AdmittedFile {
            record_index,
            source_path,
            workspace_path,
        });
    }
    drop(workspace_paths);

    validate_workspace_layout(&admitted_files)?;
    validate_references(&metadata)?;
    Ok(AdmittedFixture {
        root: canonical_root,
        metadata,
        files: admitted_files,
    })
}

fn validate_envelope(metadata: &FixtureReference) -> Result<(), FixtureError> {
    if metadata.schema_version != 1 {
        return Err(FixtureError::InvalidMetadata(format!(
            "unsupported schema_version: {}",
            metadata.schema_version
        )));
    }
    if metadata.fixture_id != FIXTURE_ID {
        return Err(FixtureError::InvalidMetadata(format!(
            "unexpected fixture_id: {}",
            metadata.fixture_id
        )));
    }
    if metadata.baseline_commit != BASELINE_COMMIT {
        return Err(FixtureError::BaselineMismatch {
            expected: BASELINE_COMMIT,
            actual: metadata.baseline_commit.clone(),
        });
    }
    for (name, value) in [
        ("reference_inputs", &metadata.reference_inputs),
        ("environment", &metadata.environment),
        ("wire_versions", &metadata.wire_versions),
        ("generation", &metadata.generation),
    ] {
        if !value.is_object() {
            return Err(FixtureError::InvalidMetadata(format!(
                "{name} must be a JSON object"
            )));
        }
    }
    for field in ["origin_bound_fields", "omissions", "executed_checks"] {
        if !metadata.generation.get(field).is_some_and(Value::is_array) {
            return Err(FixtureError::InvalidMetadata(format!(
                "generation.{field} must be an array"
            )));
        }
    }
    if metadata.files.is_empty() || metadata.files.len() > MAX_FILES {
        return Err(FixtureError::InvalidMetadata(
            "files must contain a bounded, nonempty artifact list".to_owned(),
        ));
    }
    if metadata.outputs.len() != 3 {
        return Err(FixtureError::InvalidMetadata(
            "outputs must contain exactly three role records".to_owned(),
        ));
    }
    Ok(())
}

fn validate_references(metadata: &FixtureReference) -> Result<(), FixtureError> {
    let known_files: HashSet<&str> = metadata
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    let mut roles = HashSet::with_capacity(3);
    for output in &metadata.outputs {
        let role = match output.role {
            OutputRole::Mono => "mono",
            OutputRole::BilingualTargetFirst => "bilingual_target_first",
            OutputRole::BilingualSourceFirst => "bilingual_source_first",
        };
        if !roles.insert(role) {
            return Err(FixtureError::InvalidMetadata(format!(
                "duplicate output role: {role}"
            )));
        }
        for path in [&output.file, &output.proof, &output.receipt] {
            RelativePath::parse(path)?;
            if !known_files.contains(path.as_str()) {
                return Err(FixtureError::InvalidMetadata(format!(
                    "output references an unlisted artifact: {path}"
                )));
            }
        }
    }
    validate_origins(metadata, &known_files)?;
    let projection = &metadata.semantic_projection;
    RelativePath::parse(&projection.path)?;
    validate_sha256(&projection.sha256, &projection.path)?;
    if projection.format_version != 1 {
        return Err(FixtureError::InvalidMetadata(format!(
            "unsupported semantic projection version: {}",
            projection.format_version
        )));
    }
    let Some(file) = metadata
        .files
        .iter()
        .find(|file| file.path == projection.path)
    else {
        return Err(FixtureError::InvalidMetadata(
            "semantic projection path is not a listed artifact".to_owned(),
        ));
    };
    if file.sha256 != projection.sha256 {
        return Err(FixtureError::InvalidMetadata(
            "semantic projection hash does not match its file record".to_owned(),
        ));
    }
    Ok(())
}

fn validate_origins(
    metadata: &FixtureReference,
    known_files: &HashSet<&str>,
) -> Result<(), FixtureError> {
    let origins = metadata.generation["origin_bound_fields"]
        .as_array()
        .expect("generation arrays were validated");
    let mut seen = HashSet::with_capacity(origins.len());
    for origin in origins {
        let invalid = || FixtureError::InvalidMetadata("invalid origin-bound field".to_owned());
        let record = origin.as_object().ok_or_else(invalid)?;
        if record.len() != 3 {
            return Err(invalid());
        }
        let artifact = record
            .get("artifact")
            .and_then(Value::as_str)
            .ok_or_else(invalid)?;
        let pointer = record
            .get("pointer")
            .and_then(Value::as_str)
            .ok_or_else(invalid)?;
        let value = record
            .get("value")
            .and_then(Value::as_str)
            .ok_or_else(invalid)?;
        RelativePath::parse(artifact)?;
        if !known_files.contains(artifact)
            || !pointer.starts_with('/')
            || value.is_empty()
            || !seen.insert((artifact, pointer))
        {
            return Err(invalid());
        }
    }
    Ok(())
}

fn validate_sha256(value: &str, path: &str) -> Result<(), FixtureError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(FixtureError::InvalidMetadata(format!(
            "sha256 must be 64 lowercase hexadecimal characters for {path}"
        )));
    }
    Ok(())
}

pub(crate) struct FixtureWorkspace {
    temporary: TempDir,
}
