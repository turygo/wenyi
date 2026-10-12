use super::{AssetReference, ConfigError, OutputFields, Presence, invalid, output};
use std::path::Path;
use wenyi_store::SavedOutputSelection;

pub(crate) fn admit_saved_output(
    saved: &SavedOutputSelection,
) -> Result<OutputFields, ConfigError> {
    for (slot, value) in [
        (
            "override_theme.styles",
            saved
                .selection
                .get("override_theme")
                .and_then(|theme| theme.get("styles")),
        ),
        ("bilingual_styles", saved.selection.get("bilingual_styles")),
    ] {
        if let Some(value) = value.and_then(serde_json::Value::as_str)
            && !value.starts_with("builtin:")
            && !Path::new(value).is_absolute()
        {
            return Err(invalid(
                slot,
                "saved asset references must be absolute or builtin",
            ));
        }
    }
    let mut fields = output::admit(&saved.selection, Path::new(""))?;
    fields.origins.clear();
    for (slot, custom) in [
        (
            "override_theme.styles",
            matches!(
                fields.override_theme,
                Presence::Set(Some(AssetReference::File(_)))
            ),
        ),
        (
            "bilingual_styles",
            matches!(
                fields.bilingual_styles,
                Presence::Set(AssetReference::File(_))
            ),
        ),
    ] {
        if custom && let Some(origin) = saved.origins.get(slot) {
            fields.origins.insert(slot, Path::new(origin).to_path_buf());
        }
    }
    Ok(fields)
}
