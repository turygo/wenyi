#![forbid(unsafe_code)]

pub mod coordinates;
pub mod digest;
pub mod identity;
pub mod model;
pub mod rich;

pub use coordinates::{ScalarOffset, ScalarRange};
pub use digest::{rich_source_digest, slot_contract_digest, stable_digest};
pub use identity::{
    RUN_INPUT_SCHEMA_VERSION, RUN_STATE_SCHEMA_VERSION, SOURCE_SLOT_SCHEMA_VERSION, SlotField,
    SourceFormat, SourceIdentity, SourceSlot, TRANSLATION_POLICY_VERSION,
};
pub use model::{ModelReference, ThinkingLevel, parse_model_reference};
pub use rich::{
    AtomKind, InlineAtom, InlineMark, InlineRun, MarkKind, NoteKind, NoteReference, OmissionKind,
    OrderedAttributes, RichSource, RichTarget, SemanticRole, note_label_key, semantic_roles,
    tag_semantics, validate_rich_target,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreErrorKind {
    Coordinate,
    Identity,
    RichContract,
    Digest,
    ModelCapability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreError {
    pub kind: CoreErrorKind,
    pub code: &'static str,
    pub detail: String,
}

impl CoreError {
    pub fn new(kind: CoreErrorKind, code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for CoreError {}
