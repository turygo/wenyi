use std::path::Path;

use serde_json::{Map, Value};

use super::{ConfigError, ConfigErrorKind, LoadedConfig, Quality, invalid, models, output};

pub(super) fn admit(text: &str, origin: &Path) -> Result<LoadedConfig, ConfigError> {
    let value = serde_saphyr::from_str::<Option<Value>>(text).map_err(|error| {
        ConfigError::new(
            ConfigErrorKind::Yaml,
            "yaml_syntax",
            "yaml",
            error.to_string(),
        )
    })?;
    let Some(root) = value else {
        return Ok(LoadedConfig {
            origin: Some(origin.into()),
            ..LoadedConfig::default()
        });
    };
    let entries = fields(&root, "config", &["llm", "quality", "output"])?;
    let mut llm = None;
    let mut quality = Quality::Balanced;
    let mut output = None;
    for (key, value) in entries {
        match key.as_str() {
            "llm" => llm = Some(models::admit(value)?),
            "quality" => {
                quality = match string(value, "quality")? {
                    "economy" => Quality::Economy,
                    "balanced" => Quality::Balanced,
                    "quality" => Quality::Quality,
                    _ => return Err(invalid("quality", "expected economy, balanced, or quality")),
                }
            }
            "output" => output = Some(output::admit(value, origin)?),
            _ => unreachable!(),
        }
    }
    Ok(LoadedConfig {
        llm: llm.unwrap_or_default(),
        quality,
        policy: quality.policy(),
        output: output.unwrap_or_default(),
        origin: Some(origin.into()),
    })
}

pub(super) fn fields<'a>(
    value: &'a Value,
    field: &str,
    allowed: &[&str],
) -> Result<&'a Map<String, Value>, ConfigError> {
    let entries = value
        .as_object()
        .ok_or_else(|| invalid(field, "expected a mapping"))?;
    if let Some(key) = entries.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(
            field,
            format!("unknown or deprecated field: {key}"),
        ));
    }
    Ok(entries)
}

pub(super) fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str, ConfigError> {
    value
        .as_str()
        .ok_or_else(|| invalid(field, "expected a string"))
}

pub(super) fn boolean(value: &Value, field: &str) -> Result<bool, ConfigError> {
    value
        .as_bool()
        .ok_or_else(|| invalid(field, "expected a boolean"))
}

pub(super) fn sequence<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], ConfigError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(field, "expected an ordered list"))
}
