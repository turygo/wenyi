use crate::{CoreError, CoreErrorKind};
use serde::Serialize;

pub const BUILTIN_PROVIDERS: [&str; 9] = [
    "deepseek",
    "opencode-go",
    "bailian",
    "openai",
    "openrouter",
    "openai-compatible",
    "ollama",
    "vllm",
    "fake",
];
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Low,
    Medium,
    High,
    Max,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelReference {
    pub provider: String,
    pub model: String,
    pub thinking: Option<ThinkingLevel>,
}

const DEFAULT_EFFORTS: &[ThinkingLevel] = &[
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
];
const ALL_EFFORTS: &[ThinkingLevel] = &[
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::Max,
];

fn catalogue(provider: &str, model: &str) -> Option<&'static [ThinkingLevel]> {
    match (provider, model) {
        ("opencode-go", "deepseek-v4-flash") => Some(&[ThinkingLevel::High, ThinkingLevel::Max]),
        ("opencode-go", "muse-spark-1.3-contributor") => Some(&[ThinkingLevel::Low]),
        ("opencode-go", "mimo-v2.5" | "glm-5.3-flash") => Some(&[]),
        ("openrouter", "google/gemini-3.8-flash") => Some(DEFAULT_EFFORTS),
        (
            "openrouter",
            "tencent/hy-mt2-30b-a3b" | "tencent/hy-mt2-30b-a3b-20260521" | "tencent/hy3",
        ) => Some(&[]),
        ("bailian", "deepseek-v4-flash" | "deepseek-v4-flash-0731") => Some(ALL_EFFORTS),
        (
            "bailian",
            "deepseek-v4-pro"
            | "deepseek-v4-pro-us"
            | "deepseek-v4-pro-0813"
            | "qwen3.7-flash"
            | "qwen3.7-flash-2026-07-15"
            | "qwen3.7-plus"
            | "qwen3.7-plus-us"
            | "qwen3.7-plus-2026-05-26"
            | "qwen3.8-max",
        ) => Some(&[]),
        _ => None,
    }
}

pub fn parse_model_reference(value: &str) -> Result<ModelReference, CoreError> {
    let (provider, value) = value.split_once('/').ok_or_else(|| {
        invalid(
            "model_syntax",
            "model selection must use provider/model syntax",
        )
    })?;
    let provider = provider.trim();
    let value = value.trim();
    if !BUILTIN_PROVIDERS.contains(&provider) {
        return Err(invalid(
            "unknown_provider",
            format!("unknown provider: {provider:?}"),
        ));
    }
    if value.is_empty() {
        return Err(invalid("empty_model", "model ID must not be empty"));
    }
    let mut model = value;
    let mut thinking = None;
    if let Some((base, suffix)) = value.rsplit_once(':') {
        let level = match suffix {
            "off" => Some(ThinkingLevel::Off),
            "low" => Some(ThinkingLevel::Low),
            "medium" => Some(ThinkingLevel::Medium),
            "high" => Some(ThinkingLevel::High),
            "max" => Some(ThinkingLevel::Max),
            _ => None,
        };
        if !base.is_empty() && level.is_some() {
            model = base;
            thinking = level;
        } else if catalogue(provider, base).is_some() {
            return Err(invalid(
                "unknown_thinking_level",
                format!("unknown thinking level: {suffix:?}; allowed: off, low, medium, high, max"),
            ));
        }
    }
    if let Some(level) = thinking {
        let efforts = catalogue(provider, model).unwrap_or({
            if matches!(provider, "deepseek" | "openai" | "openrouter") {
                DEFAULT_EFFORTS
            } else {
                &[]
            }
        });
        if level != ThinkingLevel::Off && !efforts.contains(&level) {
            return Err(invalid(
                "unsupported_thinking_level",
                format!(
                    "{provider}:{model} does not support thinking level {level:?}; supported: Off, {efforts:?}"
                ),
            ));
        }
    }
    Ok(ModelReference {
        provider: provider.to_owned(),
        model: model.to_owned(),
        thinking,
    })
}

fn invalid(code: &'static str, detail: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorKind::ModelCapability, code, detail)
}
