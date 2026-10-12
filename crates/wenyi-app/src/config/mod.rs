#![forbid(unsafe_code)]

mod models;
mod output;
mod saved_output;
mod schema;

use std::fs;
use std::path::{Component, Path, PathBuf};

pub use models::{LlmConfig, ModelCandidate, ModelRoles, ProviderRouting};
pub use output::{
    AssetReference, BilingualOrder, OutputFields, Presence, ResolvedOutput, resolve_output,
};
pub(crate) use saved_output::admit_saved_output;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigErrorKind {
    Yaml,
    Schema,
    Model,
    Output,
    Io,
}

#[derive(Debug)]
pub struct ConfigError {
    pub kind: ConfigErrorKind,
    pub code: &'static str,
    pub field: String,
    pub detail: String,
    pub origin: Option<PathBuf>,
}

impl ConfigError {
    pub(crate) fn new(
        kind: ConfigErrorKind,
        code: &'static str,
        field: &str,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            code,
            field: field.into(),
            detail: detail.into(),
            origin: None,
        }
    }
    pub(crate) fn at(mut self, path: &Path) -> Self {
        self.origin = Some(path.to_path_buf());
        self
    }
}
impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(path) = &self.origin {
            write!(f, "{}: ", path.display())?;
        }
        write!(f, "{} ({}): {}", self.code, self.field, self.detail)
    }
}
impl std::error::Error for ConfigError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Economy,
    Balanced,
    Quality,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityPolicy {
    pub polish: bool,
    pub single_segment_translation: bool,
    pub protocol_retry_limit: u32,
    pub rolling_context_segments: usize,
    pub prescan_concurrency: usize,
    pub max_chars_per_batch: usize,
    pub max_chars_per_segment: usize,
    pub glossary_scope: &'static str,
    pub inflight_glossary: bool,
}
impl Quality {
    pub fn policy(self) -> QualityPolicy {
        QualityPolicy {
            polish: self == Self::Quality,
            single_segment_translation: false,
            protocol_retry_limit: 2,
            rolling_context_segments: 6,
            prescan_concurrency: 4,
            max_chars_per_batch: 1800,
            max_chars_per_segment: 1200,
            glossary_scope: "chapter",
            inflight_glossary: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedConfig {
    pub llm: LlmConfig,
    pub quality: Quality,
    pub policy: QualityPolicy,
    pub output: OutputFields,
    pub origin: Option<PathBuf>,
}
impl Default for LoadedConfig {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            quality: Quality::Balanced,
            policy: Quality::Balanced.policy(),
            output: OutputFields::default(),
            origin: None,
        }
    }
}

pub fn load_config(path: &Path) -> Result<LoadedConfig, ConfigError> {
    let target = expand_home(path)?;
    match fs::metadata(&target) {
        Ok(metadata) if !metadata.is_file() => return Ok(LoadedConfig::default()),
        Ok(_) => {}
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(LoadedConfig::default());
        }
        Err(error) => {
            return Err(ConfigError::new(
                ConfigErrorKind::Io,
                "config_read",
                "config",
                error.to_string(),
            )
            .at(&target));
        }
    }
    let text = fs::read_to_string(&target).map_err(|error| {
        ConfigError::new(
            ConfigErrorKind::Io,
            "config_read",
            "config",
            error.to_string(),
        )
        .at(&target)
    })?;
    parse_config(&text, &target)
}

pub fn parse_config(text: &str, origin: &Path) -> Result<LoadedConfig, ConfigError> {
    schema::admit(text, origin).map_err(|error| error.at(origin))
}

pub(crate) fn expand_home(path: &Path) -> Result<PathBuf, ConfigError> {
    let mut components = path.components();
    let Some(Component::Normal(first)) = components.next() else {
        return Ok(path.to_path_buf());
    };
    let Some(first) = first.to_str() else {
        return Ok(path.to_path_buf());
    };
    let Some(user) = first.strip_prefix('~') else {
        return Ok(path.to_path_buf());
    };
    let home = if user.is_empty() {
        dirs::home_dir()
    } else {
        named_home(user)
    }
    .ok_or_else(|| {
        ConfigError::new(
            ConfigErrorKind::Io,
            "home_unavailable",
            "path",
            "cannot resolve the selected home directory",
        )
        .at(path)
    })?;
    Ok(home.join(components.as_path()))
}

#[cfg(unix)]
fn named_home(name: &str) -> Option<PathBuf> {
    use uzers::os::unix::UserExt;
    uzers::get_user_by_name(name).map(|user| user.home_dir().to_path_buf())
}
#[cfg(windows)]
fn named_home(name: &str) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let current = std::env::var_os("USERNAME")?;
    (home.file_name() == Some(current.as_os_str()))
        .then(|| home.parent().map(|parent| parent.join(name)))
        .flatten()
}
#[cfg(not(any(unix, windows)))]
fn named_home(_: &str) -> Option<PathBuf> {
    None
}

pub(crate) fn invalid(field: &str, detail: impl Into<String>) -> ConfigError {
    ConfigError::new(ConfigErrorKind::Schema, "config_schema", field, detail)
}
