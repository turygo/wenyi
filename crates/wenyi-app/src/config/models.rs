use std::collections::HashSet;

use serde_json::Value;
use wenyi_core::{ModelReference, parse_model_reference};

use super::{
    ConfigError, ConfigErrorKind, invalid,
    schema::{boolean, fields, sequence, string},
};

const TRANSLATOR: &str = "openrouter/google/gemini-3.8-flash:low";
const GENERAL: &str = "opencode-go/muse-spark-1.3-contributor:low";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCandidate {
    pub selection: String,
    pub reference: ModelReference,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoles {
    pub translator: Vec<ModelCandidate>,
    pub analyst: Vec<ModelCandidate>,
    pub editor: Vec<ModelCandidate>,
    pub fast: Vec<ModelCandidate>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderRouting {
    pub only: Option<Vec<String>>,
    pub order: Option<Vec<String>>,
    pub allow_fallbacks: Option<bool>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlmConfig {
    pub models: ModelRoles,
    pub provider_routing: Vec<(String, ProviderRouting)>,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
}

impl Default for ModelRoles {
    fn default() -> Self {
        Self {
            translator: baseline(TRANSLATOR),
            analyst: baseline(GENERAL),
            editor: baseline(GENERAL),
            fast: baseline(GENERAL),
        }
    }
}
fn baseline(value: &str) -> Vec<ModelCandidate> {
    vec![ModelCandidate {
        selection: value.into(),
        reference: parse_model_reference(value).expect("fixed model selection"),
    }]
}

pub(super) fn admit(value: &Value) -> Result<LlmConfig, ConfigError> {
    let entries = fields(
        value,
        "llm",
        &["models", "provider_routing", "base_url", "api_key_env"],
    )?;
    let mut models = None;
    let mut routing = Vec::new();
    let mut base_url = None;
    let mut api_key_env = None;
    for (name, value) in entries {
        match name.as_str() {
            "models" => models = Some(roles(value)?),
            "provider_routing" => routing = routing_table(value)?,
            "base_url" => base_url = optional_string(value, "llm.base_url")?,
            "api_key_env" => api_key_env = optional_string(value, "llm.api_key_env")?,
            _ => unreachable!(),
        }
    }
    let models = models.unwrap_or_default();
    let has_custom = [
        &models.translator,
        &models.analyst,
        &models.editor,
        &models.fast,
    ]
    .into_iter()
    .flatten()
    .any(|candidate| candidate.reference.provider == "openai-compatible");
    if has_custom {
        if base_url
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(model_error(
                "llm.base_url",
                "openai-compatible requires a nonblank service address",
            ));
        }
    } else if base_url.is_some() || api_key_env.is_some() {
        return Err(model_error(
            "llm",
            "base_url and api_key_env are only valid for openai-compatible",
        ));
    }
    Ok(LlmConfig {
        models,
        provider_routing: routing,
        base_url,
        api_key_env,
    })
}

fn roles(value: &Value) -> Result<ModelRoles, ConfigError> {
    let entries = fields(
        value,
        "llm.models",
        &["translator", "analyst", "editor", "fast"],
    )?;
    let get = |name: &str, field, default| match entries.get(name) {
        Some(value) => candidates(value, field),
        None => Ok(baseline(default)),
    };
    Ok(ModelRoles {
        translator: get("translator", "llm.models.translator", TRANSLATOR)?,
        analyst: get("analyst", "llm.models.analyst", GENERAL)?,
        editor: get("editor", "llm.models.editor", GENERAL)?,
        fast: get("fast", "llm.models.fast", GENERAL)?,
    })
}

fn candidates(value: &Value, field: &str) -> Result<Vec<ModelCandidate>, ConfigError> {
    let values = sequence(value, field)?;
    if values.is_empty() {
        return Err(invalid(field, "model candidate list must not be empty"));
    }
    let mut result = Vec::with_capacity(values.len());
    let mut seen = HashSet::new();
    for value in values {
        let selection = string(value, field)?.trim();
        if selection.is_empty() || !seen.insert(selection) {
            return Err(invalid(
                field,
                "model candidates must be nonblank and unique",
            ));
        }
        let reference = parse_model_reference(selection)
            .map_err(|error| model_error(field, error.to_string()))?;
        result.push(ModelCandidate {
            selection: selection.into(),
            reference,
        });
    }
    Ok(result)
}

fn routing_table(value: &Value) -> Result<Vec<(String, ProviderRouting)>, ConfigError> {
    let entries = value
        .as_object()
        .ok_or_else(|| invalid("llm.provider_routing", "expected a mapping"))?;
    let mut result = Vec::with_capacity(entries.len());
    for (model, value) in entries {
        let Some((provider, name)) = model.split_once('/') else {
            return Err(model_error(
                "llm.provider_routing",
                "routing keys require provider/model syntax",
            ));
        };
        if provider.trim() != "openrouter" || name.trim().is_empty() {
            return Err(model_error(
                "llm.provider_routing",
                "routing keys must identify an OpenRouter model",
            ));
        }
        result.push((model.clone(), routing(value)?));
    }
    Ok(result)
}

fn routing(value: &Value) -> Result<ProviderRouting, ConfigError> {
    let entries = fields(
        value,
        "llm.provider_routing",
        &["only", "order", "allow_fallbacks"],
    )?;
    let mut result = ProviderRouting::default();
    for (key, value) in entries {
        if value.is_null() {
            continue;
        }
        match key.as_str() {
            "only" => result.only = Some(strings(value, "llm.provider_routing.only")?),
            "order" => result.order = Some(strings(value, "llm.provider_routing.order")?),
            "allow_fallbacks" => {
                result.allow_fallbacks =
                    Some(boolean(value, "llm.provider_routing.allow_fallbacks")?)
            }
            _ => unreachable!(),
        }
    }
    Ok(result)
}
fn strings(value: &Value, field: &str) -> Result<Vec<String>, ConfigError> {
    let values = sequence(value, field)?;
    if values.is_empty() {
        return Err(invalid(field, "routing list must not be empty"));
    }
    values
        .iter()
        .map(|value| string(value, field).map(str::to_owned))
        .collect()
}
fn optional_string(value: &Value, field: &str) -> Result<Option<String>, ConfigError> {
    if value.is_null() {
        Ok(None)
    } else {
        string(value, field).map(|value| Some(value.into()))
    }
}
fn model_error(field: &str, detail: impl Into<String>) -> ConfigError {
    ConfigError::new(ConfigErrorKind::Model, "model_selection", field, detail)
}
