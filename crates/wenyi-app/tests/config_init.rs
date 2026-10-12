#![forbid(unsafe_code)]

use std::{fs, path::Path};
use wenyi_app::{
    Application, InitResult,
    config::{self, AssetReference, BilingualOrder, OutputFields, Presence, Quality},
};

fn parse(text: &str) -> Result<config::LoadedConfig, config::ConfigError> {
    config::parse_config(text, Path::new("config.yaml"))
}

#[test]
fn role_order_and_custom_endpoint_constraints_are_preserved() {
    let value = parse("llm:\n  models:\n    translator: [' fake/literal:part ', openai-compatible/custom]\n  base_url: local-service\n  api_key_env: null\nquality: quality\n").unwrap();
    assert_eq!(
        value
            .llm
            .models
            .translator
            .iter()
            .map(|model| model.selection.as_str())
            .collect::<Vec<_>>(),
        ["fake/literal:part", "openai-compatible/custom"]
    );
    assert_eq!(
        value.llm.models.translator[0].reference.model,
        "literal:part"
    );
    assert_eq!(value.llm.base_url.as_deref(), Some("local-service"));
    assert!(value.policy.polish);
    assert_eq!(value.quality, Quality::Quality);
    for text in [
        "llm: {models: {fast: []}}",
        "llm: {models: {fast: [fake/a, ' fake/a ']}}",
        "llm: {models: {fast: [' ']}}",
        "llm: {models: {fast: [1]}}",
        "llm: {models: {fast: fake/a}}",
        "llm: {models: {fast: [openai-compatible/a]}}",
        "llm: {models: {fast: [openai-compatible/a]}, base_url: ' '}",
        "llm: {base_url: unused}",
        "llm: {api_key_env: UNUSED_KEY}",
    ] {
        assert!(parse(text).is_err(), "{text}");
    }
}

#[test]
fn malformed_yaml_and_invalid_public_fields_return_typed_errors() {
    let origin = Path::new("nested/config.yaml");
    let error = config::parse_config("[malformed", origin).unwrap_err();
    assert_eq!(error.kind, config::ConfigErrorKind::Yaml);
    assert_eq!(error.field, "yaml");
    assert_eq!(error.origin.as_deref(), Some(origin));
    for (text, field) in [
        ("output: {mono: 'true'}", "output.mono"),
        ("output: {mono: 1}", "output.mono"),
        ("quality: true", "quality"),
        ("quality: unknown", "quality"),
        ("output: {bilingual: false}", "output.bilingual"),
        ("llm: null", "llm"),
        ("output: {theme: x}", "output"),
        ("pipeline: {}", "config"),
        ("[quality, balanced]", "config"),
        ("llm: {models: {unknown: [fake/a]}}", "llm.models"),
        (
            "output: {override_theme: {}}",
            "output.override_theme.styles",
        ),
        (
            "output: {bilingual: {order: unknown}}",
            "output.bilingual.order",
        ),
    ] {
        let error = config::parse_config(text, origin).unwrap_err();
        assert_eq!(error.kind, config::ConfigErrorKind::Schema, "{text}");
        assert_eq!(error.field, field, "{text}");
        assert_eq!(error.origin.as_deref(), Some(origin));
    }
}

#[test]
fn routing_lists_and_nullable_fields_keep_their_business_values() {
    let value = parse("llm:\n  provider_routing:\n    openrouter/unlisted:literal:\n      only: [first, second]\n      order: [second, first]\n      allow_fallbacks: false\n    openrouter/another:\n      only: null\n      order: null\n      allow_fallbacks: null\n").unwrap();
    let routing = &value
        .llm
        .provider_routing
        .iter()
        .find(|(key, _)| key == "openrouter/unlisted:literal")
        .unwrap()
        .1;
    assert_eq!(
        routing.only.as_deref(),
        Some(["first".into(), "second".into()].as_slice())
    );
    assert_eq!(
        routing.order.as_deref(),
        Some(["second".into(), "first".into()].as_slice())
    );
    assert_eq!(routing.allow_fallbacks, Some(false));
    let nullable = &value
        .llm
        .provider_routing
        .iter()
        .find(|(key, _)| key == "openrouter/another")
        .unwrap()
        .1;
    assert_eq!(nullable.only, None);
    assert_eq!(nullable.order, None);
    assert_eq!(nullable.allow_fallbacks, None);
    for text in [
        "llm: {provider_routing: {'fake/a': {only: [a]}}}",
        "llm: {provider_routing: {'openrouter/a': {only: []}}}",
        "llm: {provider_routing: {'openrouter/a': {order: []}}}",
        "llm: {provider_routing: {'openrouter/a': {only: [1]}}}",
        "llm: {provider_routing: {'openrouter/a': {allow_fallbacks: 'true'}}}",
        "llm: {provider_routing: {'openrouter/a': {allow_fallbacks: 1}}}",
        "llm: {provider_routing: {'openrouter/a': {unknown: true}}}",
    ] {
        assert!(parse(text).is_err(), "{text}");
    }
}

#[test]
fn checked_in_plain_config_is_admitted() {
    let origin = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/migration/state/structured-epub-balanced-v1/config.yaml");
    let value = config::load_config(&origin).unwrap();
    assert_eq!(value.origin.as_deref(), Some(origin.as_path()));
    assert_eq!(value.quality, Quality::Balanced);
    assert_eq!(
        value.llm.models.translator[0].selection,
        "deepseek/deepseek-chat"
    );
    assert_eq!(value.output.mono, Presence::Set(true));
    assert_eq!(value.output.bilingual, Presence::Set(true));
    assert_eq!(
        value.output.bilingual_styles,
        Presence::Set(AssetReference::Builtin("builtin:bilingual"))
    );
}

#[test]
fn field_presence_null_clear_and_normalization_are_independent() {
    let config =
        parse("output: {mono: false, bilingual: {enabled: false}, override_theme: null}").unwrap();
    assert_eq!(config.output.mono, Presence::Set(false));
    assert_eq!(config.output.bilingual_order, Presence::Unset);
    let saved = OutputFields {
        mono: Presence::Set(true),
        bilingual: Presence::Set(true),
        bilingual_order: Presence::Set(BilingualOrder::SourceFirst),
        override_theme: Presence::Set(Some(AssetReference::Builtin("builtin:chinese-reading"))),
        ..OutputFields::default()
    };
    let resolved = config::resolve_output(&OutputFields::default(), &config.output, Some(&saved));
    assert!(resolved.mono);
    assert!(!resolved.bilingual);
    assert!(resolved.normalized);
    assert_eq!(resolved.bilingual_order, BilingualOrder::SourceFirst);
    assert_eq!(resolved.override_theme, None);
    let cli = OutputFields {
        bilingual: Presence::Set(true),
        ..OutputFields::default()
    };
    let resolved = config::resolve_output(&cli, &config.output, Some(&saved));
    assert!(!resolved.mono);
    assert!(resolved.bilingual);
    assert!(!resolved.normalized);
}

#[test]
fn unused_asset_paths_keep_config_origin_without_opening_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nested/config.yaml");
    let value = config::parse_config(
        "output: {override_theme: {styles: missing.css}, bilingual_styles: other.css}",
        &path,
    )
    .unwrap();
    assert_eq!(
        value.output.override_theme,
        Presence::Set(Some(AssetReference::File(
            directory.path().join("nested/missing.css")
        )))
    );
    assert_eq!(value.output.origins["override_theme.styles"], path);
    assert!(!directory.path().join("nested").exists());
    assert!(parse("output: {bilingual_styles: builtin:unknown}").is_err());
}

#[test]
fn missing_nonfile_empty_and_null_config_do_not_create_files() {
    let directory = tempfile::tempdir().unwrap();
    let absent = directory.path().join("absent/config.yaml");
    assert_eq!(
        config::load_config(&absent).unwrap().quality,
        Quality::Balanced
    );
    assert!(!absent.parent().unwrap().exists());
    assert_eq!(
        config::load_config(directory.path()).unwrap().quality,
        Quality::Balanced
    );
    for text in ["", "null", "~"] {
        assert_eq!(parse(text).unwrap().quality, Quality::Balanced);
    }
}

#[test]
fn init_exclusively_creates_refuses_then_overwrites_without_loading_old_yaml() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nested/config.yaml");
    assert_eq!(
        Application::init(&path, false).unwrap(),
        InitResult::Created { path: path.clone() }
    );
    let value = config::load_config(&path).unwrap();
    assert_eq!(
        value.llm.models.translator[0].selection,
        "openrouter/google/gemini-3.8-flash:low"
    );
    assert_eq!(
        value.llm.models.fast[0].selection,
        "opencode-go/muse-spark-1.3-contributor:low"
    );
    fs::write(&path, "[malformed").unwrap();
    assert_eq!(
        Application::init(&path, false).unwrap(),
        InitResult::Existing { path: path.clone() }
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "[malformed");
    assert_eq!(
        Application::init(&path, true).unwrap(),
        InitResult::Created { path: path.clone() }
    );
    assert_eq!(
        config::load_config(&path).unwrap().quality,
        Quality::Balanced
    );
    assert!(!directory.path().join("state").exists());
    let blocked = directory.path().join("file");
    fs::write(&blocked, "unchanged").unwrap();
    assert!(Application::init(&blocked.join("config.yaml"), false).is_err());
    assert_eq!(fs::read_to_string(blocked).unwrap(), "unchanged");
}
