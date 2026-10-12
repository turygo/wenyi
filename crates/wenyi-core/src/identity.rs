use crate::{CoreError, CoreErrorKind};
use serde::Serialize;

pub const RUN_INPUT_SCHEMA_VERSION: u32 = 2;
pub const RUN_STATE_SCHEMA_VERSION: u32 = 4;
pub const SOURCE_SLOT_SCHEMA_VERSION: u32 = 4;
pub const TRANSLATION_POLICY_VERSION: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceFormat {
    Epub,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceIdentity {
    pub source_bytes_sha256: String,
    pub title: String,
    pub run_slug: String,
    pub format: SourceFormat,
    pub input_schema_version: u32,
    pub source_slot_schema_version: u32,
}

impl SourceIdentity {
    pub fn new(
        source_bytes_sha256: String,
        title: String,
        run_slug: String,
    ) -> Result<Self, CoreError> {
        if source_bytes_sha256.len() != 64
            || !source_bytes_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(CoreError::new(
                CoreErrorKind::Identity,
                "invalid_source_sha256",
                "source digest must be lowercase SHA256",
            ));
        }
        if title.is_empty() || run_slug.is_empty() {
            return Err(CoreError::new(
                CoreErrorKind::Identity,
                "empty_source_identity",
                "title and run slug must not be empty",
            ));
        }
        Ok(Self {
            source_bytes_sha256,
            title,
            run_slug,
            format: SourceFormat::Epub,
            input_schema_version: RUN_INPUT_SCHEMA_VERSION,
            source_slot_schema_version: SOURCE_SLOT_SCHEMA_VERSION,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SlotField {
    Text,
    Tail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceSlot {
    pub id: String,
    pub element_path: Vec<i32>,
    pub field: SlotField,
    pub source_value: String,
    pub target_value: Option<String>,
}
