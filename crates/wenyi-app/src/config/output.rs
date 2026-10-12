use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use serde_json::Value;

use super::{
    ConfigError, expand_home, invalid,
    schema::{boolean, fields, string},
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Presence<T> {
    #[default]
    Unset,
    Set(T),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetReference {
    Builtin(&'static str),
    File(PathBuf),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BilingualOrder {
    TargetFirst,
    SourceFirst,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputFields {
    pub mono: Presence<bool>,
    pub bilingual: Presence<bool>,
    pub bilingual_order: Presence<BilingualOrder>,
    pub override_theme: Presence<Option<AssetReference>>,
    pub bilingual_styles: Presence<AssetReference>,
    pub origins: BTreeMap<&'static str, PathBuf>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOutput {
    pub mono: bool,
    pub bilingual: bool,
    pub bilingual_order: BilingualOrder,
    pub override_theme: Option<AssetReference>,
    pub bilingual_styles: AssetReference,
    pub origins: BTreeMap<&'static str, PathBuf>,
    pub normalized: bool,
}

pub fn resolve_output(
    cli: &OutputFields,
    config: &OutputFields,
    saved: Option<&OutputFields>,
) -> ResolvedOutput {
    let empty = OutputFields::default();
    let saved = saved.unwrap_or(&empty);
    let mono = choose(&cli.mono, &config.mono, &saved.mono)
        .copied()
        .unwrap_or(true);
    let bilingual = choose(&cli.bilingual, &config.bilingual, &saved.bilingual)
        .copied()
        .unwrap_or(true);
    let override_theme = choose(
        &cli.override_theme,
        &config.override_theme,
        &saved.override_theme,
    )
    .cloned()
    .flatten();
    let bilingual_styles = choose(
        &cli.bilingual_styles,
        &config.bilingual_styles,
        &saved.bilingual_styles,
    )
    .cloned()
    .unwrap_or(AssetReference::Builtin("builtin:bilingual"));
    let mut origins = BTreeMap::new();
    for (field, cli_set, config_set, custom) in [
        (
            "override_theme.styles",
            matches!(cli.override_theme, Presence::Set(_)),
            matches!(config.override_theme, Presence::Set(_)),
            matches!(override_theme, Some(AssetReference::File(_))),
        ),
        (
            "bilingual_styles",
            matches!(cli.bilingual_styles, Presence::Set(_)),
            matches!(config.bilingual_styles, Presence::Set(_)),
            matches!(bilingual_styles, AssetReference::File(_)),
        ),
    ] {
        let selected = if cli_set {
            cli
        } else if config_set {
            config
        } else {
            saved
        };
        if custom && let Some(origin) = selected.origins.get(field) {
            origins.insert(field, origin.clone());
        }
    }
    ResolvedOutput {
        mono: mono || !bilingual,
        bilingual,
        bilingual_order: choose(
            &cli.bilingual_order,
            &config.bilingual_order,
            &saved.bilingual_order,
        )
        .copied()
        .unwrap_or(BilingualOrder::TargetFirst),
        override_theme,
        bilingual_styles,
        origins,
        normalized: !mono && !bilingual,
    }
}
fn choose<'a, T>(
    cli: &'a Presence<T>,
    config: &'a Presence<T>,
    saved: &'a Presence<T>,
) -> Option<&'a T> {
    [cli, config, saved]
        .into_iter()
        .find_map(|value| match value {
            Presence::Unset => None,
            Presence::Set(value) => Some(value),
        })
}

pub(super) fn admit(value: &Value, origin: &Path) -> Result<OutputFields, ConfigError> {
    let entries = fields(
        value,
        "output",
        &["mono", "bilingual", "override_theme", "bilingual_styles"],
    )?;
    let mut output = OutputFields::default();
    for (key, value) in entries {
        match key.as_str() {
            "mono" => output.mono = Presence::Set(boolean(value, "output.mono")?),
            "bilingual" => bilingual(value, &mut output)?,
            "override_theme" => {
                let asset = if value.is_null() {
                    None
                } else {
                    let theme = fields(value, "output.override_theme", &["styles"])?;
                    let value = theme.get("styles").ok_or_else(|| {
                        invalid("output.override_theme.styles", "required field is missing")
                    })?;
                    Some(asset(
                        string(value, "output.override_theme.styles")?,
                        "builtin:chinese-reading",
                        origin,
                    )?)
                };
                if matches!(asset, Some(AssetReference::File(_))) {
                    output
                        .origins
                        .insert("override_theme.styles", origin.to_path_buf());
                }
                output.override_theme = Presence::Set(asset);
            }
            "bilingual_styles" => {
                let asset = asset(
                    string(value, "output.bilingual_styles")?,
                    "builtin:bilingual",
                    origin,
                )?;
                if matches!(asset, AssetReference::File(_)) {
                    output
                        .origins
                        .insert("bilingual_styles", origin.to_path_buf());
                }
                output.bilingual_styles = Presence::Set(asset);
            }
            _ => unreachable!(),
        }
    }
    Ok(output)
}

fn bilingual(value: &Value, output: &mut OutputFields) -> Result<(), ConfigError> {
    for (key, value) in fields(value, "output.bilingual", &["enabled", "order"])? {
        match key.as_str() {
            "enabled" => {
                output.bilingual = Presence::Set(boolean(value, "output.bilingual.enabled")?)
            }
            "order" => {
                output.bilingual_order =
                    Presence::Set(order(string(value, "output.bilingual.order")?)?)
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}
fn order(value: &str) -> Result<BilingualOrder, ConfigError> {
    match value {
        "target_first" => Ok(BilingualOrder::TargetFirst),
        "source_first" => Ok(BilingualOrder::SourceFirst),
        _ => Err(invalid(
            "output.bilingual.order",
            "expected target_first or source_first",
        )),
    }
}
fn asset(value: &str, builtin: &'static str, origin: &Path) -> Result<AssetReference, ConfigError> {
    if value.trim().is_empty() || value.contains('\0') {
        return Err(invalid(
            "output.styles",
            "asset reference must be nonblank and contain no NUL",
        ));
    }
    if value.starts_with("builtin:") {
        return if value == builtin {
            Ok(AssetReference::Builtin(builtin))
        } else {
            Err(invalid(
                "output.styles",
                format!("only {builtin} is supported"),
            ))
        };
    }
    let path = expand_home(Path::new(value))?;
    let path = if path.is_absolute() {
        path
    } else {
        origin.parent().unwrap_or(Path::new(".")).join(path)
    };
    std::path::absolute(path)
        .map(AssetReference::File)
        .map_err(|error| invalid("output.styles", error.to_string()))
}
