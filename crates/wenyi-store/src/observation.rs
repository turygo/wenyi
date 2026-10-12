use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};
use wenyi_core::{RichSource, RichTarget, SourceSlot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum ObservationOutcome {
    Current,
    MissingRun,
    RecoveryNeeded,
    MigrationNeeded,
    ObservationUnavailable,
    InvalidState,
}

#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub outcome: ObservationOutcome,
    pub code: String,
    pub artifact: Option<PathBuf>,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunObservation {
    pub run_dir: PathBuf,
    pub title: Option<String>,
    pub title_translated: Option<String>,
    pub source_lang: Option<String>,
    pub target_lang: Option<String>,
    pub chapters: Vec<ObservedChapter>,
    pub nodes: Vec<NodeObservation>,
    pub glossary: Option<GlossaryCounts>,
    pub output_selection: Option<SavedOutputSelection>,
    pub diagnostics: Vec<Diagnostic>,
    pub outcome: ObservationOutcome,
}

impl RunObservation {
    pub(crate) fn empty(run_dir: PathBuf) -> Self {
        Self {
            run_dir,
            title: None,
            title_translated: None,
            source_lang: None,
            target_lang: None,
            chapters: Vec::new(),
            nodes: Vec::new(),
            glossary: None,
            output_selection: None,
            diagnostics: Vec::new(),
            outcome: ObservationOutcome::Current,
        }
    }

    pub(crate) fn diagnose(
        &mut self,
        outcome: ObservationOutcome,
        code: impl Into<String>,
        artifact: PathBuf,
        detail: impl Into<String>,
    ) {
        self.outcome = self.outcome.max(outcome);
        self.diagnostics.push(Diagnostic {
            outcome,
            code: code.into(),
            artifact: Some(artifact),
            detail: detail.into(),
        });
    }

    pub(crate) fn discard_data(&mut self) {
        self.title = None;
        self.title_translated = None;
        self.source_lang = None;
        self.target_lang = None;
        self.chapters.clear();
        self.nodes.clear();
        self.glossary = None;
        self.output_selection = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GlossaryCounts {
    pub terms: u64,
    pub open_conflicts: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SavedOutputSelection {
    pub selection: Value,
    pub origins: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeObservation {
    pub key: String,
    pub node_id: String,
    pub status: NodeStatus,
    pub attempts: u64,
    pub input_fingerprint: String,
    pub failure: Option<NodeFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    #[default]
    Pending,
    Running,
    Succeeded,
    Skipped,
    FailedRetryable,
    FailedPermanent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NodeFailure {
    pub kind: NodeFailureKind,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeFailureKind {
    ProviderRetryable,
    ProviderPermanent,
    Protocol,
    Business,
    Interrupted,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObservedChapter {
    pub index: usize,
    pub title: String,
    pub title_translated: Option<String>,
    pub href: Option<String>,
    pub toc_entry_id: Option<String>,
    pub processing: Option<ChapterProcessing>,
    pub progress: ChapterProgress,
    pub segments: Vec<ObservedSegment>,
    pub meta: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChapterStatus {
    #[default]
    Pending,
    Done,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ChapterProgress {
    #[serde(default)]
    pub status: ChapterStatus,
    #[serde(default)]
    pub pending_polish: Vec<PolishBatch>,
    #[serde(default)]
    pub repair_ledger: Vec<RepairIssue>,
    #[serde(default)]
    pub lint_issues: Vec<Value>,
    #[serde(default)]
    pub back_matter_mode: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PolishBatch {
    pub start: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RepairIssue {
    pub key: String,
    pub chapter: usize,
    pub index: usize,
    #[serde(rename = "type")]
    pub issue_type: String,
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub attempts: u64,
    #[serde(default)]
    pub status: RepairStatus,
    #[serde(default)]
    pub last_detail: String,
    #[serde(default)]
    pub committed_target_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RepairStatus {
    #[default]
    Pending,
    Repairing,
    Resolved,
    AcceptedAfterExhaustion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProcessingAction {
    Translate,
    Preserve,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChapterProcessing {
    pub action: ProcessingAction,
    pub review_required: bool,
    pub reason: String,
    pub source_sha256: String,
    pub strategy_version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObservedSegment {
    pub index: usize,
    pub source: String,
    pub kind: String,
    pub target: Option<String>,
    pub anchor: Option<String>,
    pub resource_href: Option<String>,
    pub epub_state: Option<AdmittedSourceState>,
    pub rich_target: Option<RichTarget>,
    pub meta: Value,
    pub cont: bool,
    pub preserve_source: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdmittedSourceState {
    pub resource_href: String,
    pub resource_sha256: String,
    pub block_path: Vec<i32>,
    pub block_fingerprint: String,
    pub parse_mode: String,
    pub slots: Vec<SourceSlot>,
    pub slot_contract_sha256: String,
    pub rich_source: Option<RichSource>,
}
