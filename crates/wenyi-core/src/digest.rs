use serde::{
    Serialize, Serializer,
    ser::{SerializeMap, SerializeSeq},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{CoreError, CoreErrorKind, SourceSlot, rich::RichSource};

pub fn stable_digest(value: &Value) -> Result<String, CoreError> {
    compact_digest(&canonical_json(value))
}

pub fn canonical_json(value: &Value) -> impl Serialize + '_ {
    Canonical(value)
}

struct Canonical<'a>(&'a Value);
impl Serialize for Canonical<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Value::Object(values) => {
                let mut keys: Vec<_> = values.keys().collect();
                keys.sort_unstable();
                let mut map = serializer.serialize_map(Some(values.len()))?;
                for key in keys {
                    map.serialize_entry(key, &Canonical(&values[key]))?;
                }
                map.end()
            }
            Value::Array(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(&Canonical(value))?;
                }
                sequence.end()
            }
            value => value.serialize(serializer),
        }
    }
}

pub fn slot_contract_digest(slots: &[SourceSlot]) -> Result<String, CoreError> {
    compact_digest(&SlotProjection(slots))
}

struct SlotProjection<'a>(&'a [SourceSlot]);
impl Serialize for SlotProjection<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Slot<'a> {
            id: &'a str,
            element_path: &'a [i32],
            field: crate::identity::SlotField,
            source_value: &'a str,
        }
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for slot in self.0 {
            sequence.serialize_element(&Slot {
                id: &slot.id,
                element_path: &slot.element_path,
                field: slot.field,
                source_value: &slot.source_value,
            })?;
        }
        sequence.end()
    }
}

pub fn rich_source_digest(source: &RichSource) -> Result<String, CoreError> {
    source.validate()?;
    compact_digest(source)
}

fn compact_digest(value: &impl Serialize) -> Result<String, CoreError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CoreError::new(
            CoreErrorKind::Digest,
            "digest_serialization",
            error.to_string(),
        )
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
