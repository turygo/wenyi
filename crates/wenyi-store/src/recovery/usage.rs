use super::{invalid, usage_json};
use crate::{StoreError, wire};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(crate) fn snapshot(value: &Value, path: &Path) -> Result<(), StoreError> {
    let map = wire::object(value, path)?;
    if !map.is_empty() && map.get("schema_version").and_then(Value::as_u64) != Some(2) {
        return Err(invalid(path, "unsupported nonempty usage snapshot schema"));
    }
    for key in [
        "by_agent",
        "by_operation",
        "by_provider",
        "by_model",
        "by_stage",
    ] {
        if map
            .get(key)
            .is_some_and(|value| truthy(value) && !value.is_object())
        {
            return Err(invalid(
                path,
                format!("usage dimension {key} must be an object when nonempty"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn wal(value: &Value, current: Option<&[u8]>, path: &Path) -> Result<(), StoreError> {
    let map = wire::object(value, path)?;
    if map.get("wal_schema_version").and_then(Value::as_u64) != Some(1) {
        return Err(invalid(path, "unsupported usage WAL schema"));
    }
    let before: Option<String> = wire::required(map, "before_sha256", path)?;
    if let Some(hash) = &before {
        wire::sha256(hash, path)?;
    }
    let after_hash: String = wire::required(map, "after_sha256", path)?;
    wire::sha256(&after_hash, path)?;
    let after = required_value(map, "after", path)?;
    let tracker = required_value(map, "tracker_after", path)?;
    snapshot(after, path)?;
    snapshot(tracker, path)?;
    if hash(&usage_json::serialize(after)) != after_hash {
        return Err(invalid(path, "usage WAL after serialization hash differs"));
    }
    let current_hash = current.map(hash);
    if current_hash != before && current_hash.as_deref() != Some(after_hash.as_str()) {
        return Err(invalid(
            path,
            "usage WAL does not match current usage bytes",
        ));
    }
    Ok(())
}

fn required_value<'a>(
    map: &'a Map<String, Value>,
    name: &str,
    path: &Path,
) -> Result<&'a Value, StoreError> {
    map.get(name)
        .ok_or_else(|| invalid(path, format!("missing {name}")))
}

pub(super) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}
