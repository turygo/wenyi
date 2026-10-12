use serde_json::Value;

pub(super) fn serialize(value: &Value) -> Vec<u8> {
    let mut output = serde_json::to_vec_pretty(&wenyi_core::digest::canonical_json(value))
        .expect("JSON Value serialization into Vec cannot fail");
    output.push(b'\n');
    output
}
