use super::{decode, object, rich, strict};
use crate::{
    StoreError,
    observation::{AdmittedSourceState, ObservedSegment},
};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashSet, path::Path};
use wenyi_core::{SlotField, SourceSlot, digest::slot_contract_digest, rich::validate_rich_target};

#[derive(Deserialize)]
struct Segment {
    index: usize,
    source: String,
    #[serde(default = "text")]
    kind: String,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    anchor: Option<String>,
    #[serde(default)]
    resource_href: Option<String>,
    #[serde(default)]
    cont: bool,
    #[serde(default)]
    preserve_source: bool,
    #[serde(default = "empty_object")]
    meta: Value,
    #[serde(default)]
    epub_state: Option<Value>,
    #[serde(default)]
    rich_target: Option<Value>,
}
fn text() -> String {
    "text".to_owned()
}
pub(super) fn empty_object() -> Value {
    Value::Object(Default::default())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Slot {
    id: String,
    #[serde(default)]
    element_path: Vec<i32>,
    field: String,
    source_value: String,
    #[serde(default)]
    target_value: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    resource_href: String,
    resource_sha256: String,
    block_path: Vec<i32>,
    block_fingerprint: String,
    parse_mode: String,
    slots: Vec<Slot>,
    slot_contract_sha256: String,
    #[serde(default)]
    rich_source: Option<Value>,
}

pub(super) fn admit(value: &Value, path: &Path) -> Result<ObservedSegment, StoreError> {
    let raw: Segment = decode(value, path)?;
    object(&raw.meta, path)?;
    let state = raw
        .epub_state
        .as_ref()
        .ok_or_else(|| StoreError::wire(path, "EPUB segment lacks source geometry"))?;
    let state = admit_state(state, path)?;
    if raw
        .resource_href
        .as_deref()
        .is_some_and(|href| href != state.resource_href)
    {
        return Err(StoreError::wire(
            path,
            "segment resource does not match source geometry",
        ));
    }
    let target = raw
        .rich_target
        .as_ref()
        .map(|value| rich::target(value, path))
        .transpose()?;
    if let Some(target) = &target {
        let source = state
            .rich_source
            .as_ref()
            .ok_or_else(|| StoreError::wire(path, "rich target lacks source inventory"))?;
        validate_rich_target(source, target, raw.target.as_deref())
            .map_err(|error| StoreError::wire(path, error.to_string()))?;
    }
    Ok(ObservedSegment {
        index: raw.index,
        source: raw.source,
        kind: raw.kind,
        target: raw.target,
        anchor: raw.anchor,
        resource_href: raw.resource_href,
        epub_state: Some(state),
        rich_target: target,
        meta: raw.meta,
        cont: raw.cont,
        preserve_source: raw.preserve_source,
    })
}

fn admit_state(value: &Value, path: &Path) -> Result<AdmittedSourceState, StoreError> {
    let raw: State = decode(value, path)?;
    rich::coordinates(&raw.block_path, path)?;
    for hash in [
        &raw.resource_sha256,
        &raw.block_fingerprint,
        &raw.slot_contract_sha256,
    ] {
        super::sha256(hash, path)?;
    }
    if raw.resource_href.is_empty() || !matches!(raw.parse_mode.as_str(), "xml" | "recovered") {
        return Err(StoreError::wire(
            path,
            "invalid source resource or parse mode",
        ));
    }
    let mut ids = HashSet::new();
    let mut locations = HashSet::new();
    for slot in &raw.slots {
        rich::coordinates(&slot.element_path, path)?;
        if !ids.insert(slot.id.as_str())
            || !locations.insert((&slot.element_path, slot.field.as_str()))
        {
            return Err(StoreError::wire(
                path,
                "duplicate source slot identity or location",
            ));
        }
        if slot.source_value.trim().is_empty()
            && slot
                .target_value
                .as_deref()
                .is_some_and(|target| !target.is_empty() && target != slot.source_value)
        {
            return Err(StoreError::wire(
                path,
                "whitespace-only slot target must be empty or exact source",
            ));
        }
    }
    drop(ids);
    drop(locations);
    let mut slots = Vec::with_capacity(raw.slots.len());
    for slot in raw.slots {
        let field = match slot.field.as_str() {
            "text" => SlotField::Text,
            "tail" => SlotField::Tail,
            _ => return Err(StoreError::wire(path, "unknown slot field")),
        };
        slots.push(SourceSlot {
            id: slot.id,
            element_path: slot.element_path,
            field,
            source_value: slot.source_value,
            target_value: slot.target_value,
        });
    }
    let digest =
        slot_contract_digest(&slots).map_err(|error| StoreError::wire(path, error.to_string()))?;
    if digest != raw.slot_contract_sha256 {
        return Err(StoreError::wire(
            path,
            "source slot contract digest mismatch",
        ));
    }
    let source = raw
        .rich_source
        .as_ref()
        .map(|value| rich::source(value, path))
        .transpose()?;
    if let Some(source) = &source {
        validate_source_slots(source, &slots, path)?;
    }
    Ok(AdmittedSourceState {
        resource_href: raw.resource_href,
        resource_sha256: raw.resource_sha256,
        block_path: raw.block_path,
        block_fingerprint: raw.block_fingerprint,
        parse_mode: raw.parse_mode,
        slots,
        slot_contract_sha256: raw.slot_contract_sha256,
        rich_source: source,
    })
}

fn validate_source_slots(
    source: &wenyi_core::RichSource,
    slots: &[SourceSlot],
    path: &Path,
) -> Result<(), StoreError> {
    let mut seen = HashSet::new();
    for run in &source.runs {
        if let Some(id) = &run.slot_id {
            let slot = slots
                .iter()
                .find(|slot| slot.id == *id)
                .ok_or_else(|| StoreError::wire(path, "rich source references unknown slot"))?;
            if !seen.insert(id) || slot.source_value != run.text {
                return Err(StoreError::wire(
                    path,
                    "rich source slot ownership/text mismatch",
                ));
            }
        }
    }
    if seen.len() != slots.len() {
        return Err(StoreError::wire(
            path,
            "rich source slot inventory is incomplete",
        ));
    }
    Ok(())
}

pub(crate) fn selection(
    value: &Value,
    source_hash: &str,
    path: &Path,
) -> Result<Option<crate::SavedOutputSelection>, StoreError> {
    let map = object(value, path)?;
    strict(
        map,
        &[
            "schema_version",
            "source_bytes_sha256",
            "selection",
            "origins",
        ],
        path,
    )?;
    let version = map
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| StoreError::wire(path, "selection schema must be an integer"))?;
    if !matches!(version, 1 | 2) {
        return Err(super::version_error(
            path,
            "unknown_output_selection_schema",
            "unsupported output selection version",
        ));
    }
    let hash: String = super::required(map, "source_bytes_sha256", path)?;
    super::sha256(&hash, path)?;
    if hash != source_hash {
        return Err(super::mismatch(path, "selection source binding differs"));
    }
    let selection: Value = super::required(map, "selection", path)?;
    object(&selection, path)?;
    let origins = super::required(map, "origins", path)?;
    if version == 1 {
        return Ok(None);
    }
    Ok(Some(crate::SavedOutputSelection { selection, origins }))
}
