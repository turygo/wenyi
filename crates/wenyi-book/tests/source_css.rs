#![forbid(unsafe_code)]
#[path = "support/archive.rs"]
mod support;
use support::Temp;
use wenyi_book::archive::Archive;
use wenyi_book::css::{ElementStyle, SourceCss};
use wenyi_book::error::BookErrorKind;
use wenyi_book::source::{NodeId, SourceTree, parse_source_markup};
use wenyi_core::SemanticRole;
use zip::CompressionMethod;

fn fixture(body: &str, css: &str, extra: &[(&str, &[u8])]) -> (Temp, Archive, SourceTree) {
    let html = format!(
        "<html xmlns='http://www.w3.org/1999/xhtml'><head><link rel='stylesheet' href='../styles/main.css'/></head><body>{body}</body></html>"
    );
    let mut members = vec![
        ("mimetype", b"application/epub+zip".as_slice()),
        ("text/chapter.xhtml", html.as_bytes()),
        ("styles/main.css", css.as_bytes()),
    ];
    members.extend_from_slice(extra);
    let temp = Temp::new(&support::zip(&members, CompressionMethod::Stored));
    let mut archive = Archive::open(&temp.path).unwrap();
    let document =
        parse_source_markup(&archive.read_resource("text/chapter.xhtml").unwrap()).unwrap();
    (temp, archive, document.tree)
}
fn id(tree: &SourceTree, value: &str) -> NodeId {
    tree.nodes
        .iter()
        .enumerate()
        .find_map(|(index, _)| {
            let node = NodeId::new(index).unwrap();
            (tree.attribute(node, "id") == Some(value)).then_some(node)
        })
        .unwrap()
}
fn styles(body: &str, sheet: &str, ids: &[&str]) -> Vec<ElementStyle> {
    let (_temp, mut archive, tree) = fixture(body, sheet, &[]);
    let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
    let nodes: Vec<_> = ids.iter().map(|value| id(&tree, value)).collect();
    let result = css.evidence(&tree, &nodes, "text/chapter.xhtml").unwrap();
    archive.finish().unwrap();
    result
}
#[test]
fn explicit_reset_cancels_native_strong_and_inherited_styles_are_not_duplicated() {
    let result = styles(
        "<strong id='reset'>plain</strong><p id='parent'><span id='child'>inherited</span></p>",
        "#reset {font-weight:normal} #parent {font-weight:700;font-style:italic}",
        &["reset", "parent", "child"],
    );
    assert!(result[0].roles.is_empty());
    assert!(!result[0].required);
    assert_eq!(
        result[1].roles,
        vec![SemanticRole::Bold, SemanticRole::Italic]
    );
    assert!(result[2].roles.is_empty());
}
#[test]
fn importance_inline_and_standard_font_shorthand_preserve_cascade() {
    let result = styles(
        "<span id='a' style='font-weight:normal'>A</span><span id='b' style='font:italic 650 16px serif'>B</span><span id='c' style='font-weight:normal!important'>C</span>",
        "span {font-weight:750!important} #b {font-weight:300} #c {font-weight:900!important}",
        &["a", "b", "c"],
    );
    assert_eq!(result[0].roles, vec![SemanticRole::Bold]);
    assert_eq!(
        result[1].roles,
        vec![SemanticRole::Bold, SemanticRole::Italic]
    );
    assert!(result[2].roles.is_empty());
}
#[test]
fn numeric_weight_is_own_increase_not_inherited_duplication() {
    let result = styles(
        "<p id='p'><span id='same'>same</span><span id='up'>up</span></p>",
        "#p {font-weight:550} #up {font-weight:650}",
        &["p", "same", "up"],
    );
    assert_eq!(result[0].roles, vec![SemanticRole::Bold]);
    assert!(result[1].roles.is_empty());
    assert_eq!(result[2].roles, vec![SemanticRole::Bold]);
}
#[test]
fn standard_nested_has_nth_of_and_namespace_selectors_match_native_tree() {
    let result = styles(
        "<section><p class='chosen'><span id='a'>A</span></p><p class='chosen'><span id='b'>B</span></p></section>",
        "@namespace h url('http://www.w3.org/1999/xhtml'); h|section { & > p:has(span):nth-child(2 of .chosen) > :is(span):not(.absent) {font-style:italic} }",
        &["a", "b"],
    );
    assert!(result[0].roles.is_empty());
    assert_eq!(result[1].roles, vec![SemanticRole::Italic]);
}
#[test]
fn repeated_local_imports_keep_order_and_original_asset_bytes() {
    let bytes = b"/* preserved bytes */\nspan {font-weight:700}\n";
    let (_temp, mut archive, tree) = fixture(
        "<span id='a'>A</span>",
        "@import 'other.css'; @import 'other.css'; span {font-weight:normal}",
        &[("styles/other.css", bytes)],
    );
    let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
    assert_eq!(css.assets().len(), 2);
    assert_eq!(
        css.assets()
            .iter()
            .find(|asset| asset.href == "styles/other.css")
            .unwrap()
            .bytes,
        bytes
    );
    assert!(
        css.evidence(&tree, &[id(&tree, "a")], "text/chapter.xhtml")
            .unwrap()[0]
            .roles
            .is_empty()
    );
    archive.finish().unwrap();
}
#[test]
fn active_import_cycle_is_an_explicit_css_error() {
    let (_temp, mut archive, tree) = fixture(
        "<p id='a'>A</p>",
        "@import 'other.css';",
        &[("styles/other.css", b"@import 'main.css';")],
    );
    let error = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml")
        .err()
        .unwrap();
    assert_eq!(error.kind, BookErrorKind::Css);
    assert_eq!(error.code, "css-import-cycle");
}
#[test]
fn unknown_media_and_supports_conflicts_are_rejected() {
    for sheet in [
        "@media (min-width:600px) {span {font-weight:bold}}",
        "@supports (display:grid) {span {font-style:italic}}",
    ] {
        let (_temp, mut archive, tree) = fixture("<span id='a'>A</span>", sheet, &[]);
        let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
        let error = css
            .evidence(&tree, &[id(&tree, "a")], "text/chapter.xhtml")
            .unwrap_err();
        assert_eq!(error.code, "css-semantic-ambiguity");
    }
}
#[test]
fn conditional_alternatives_with_equal_semantics_are_admitted() {
    let result = styles(
        "<span id='a'>A</span>",
        "span {font-weight:700} @media print {span {font-weight:800}} @supports (display:grid) {span {font-weight:900}}",
        &["a"],
    );
    assert_eq!(result[0].roles, vec![SemanticRole::Bold]);
}
#[test]
fn irrelevant_font_family_and_nonmatching_animation_do_not_change_evidence() {
    let result = styles(
        "<span id='a'>A</span>",
        "body {font-family:serif} .absent {font-weight:var(--unknown);animation:pulse 1s} span {font-style:italic}",
        &["a"],
    );
    assert_eq!(result[0].roles, vec![SemanticRole::Italic]);
    assert_eq!(result[0].evidence.len(), 1);
}
#[test]
fn matched_inline_and_potential_animation_are_rejected() {
    for (body, sheet) in [
        ("<span id='a'>A</span>", "span {animation:pulse 1s}"),
        ("<span id='a' style='animation:none'>A</span>", ""),
        ("<span id='a'>A</span>", "span:hover {animation:pulse 1s}"),
    ] {
        let (_temp, mut archive, tree) = fixture(body, sheet, &[]);
        let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
        let error = css
            .evidence(&tree, &[id(&tree, "a")], "text/chapter.xhtml")
            .unwrap_err();
        assert!(matches!(error.code, "css-animation" | "css-selector-state"));
    }
}
#[test]
fn pseudo_elements_do_not_become_ordinary_source_marks() {
    let result = styles(
        "<p id='a'>A sentence</p>",
        "p::first-letter {font-weight:bold;font-size:3em;float:left} p::before {font-style:italic}",
        &["a"],
    );
    assert!(result[0].roles.is_empty());
    assert!(!result[0].decoration);
}
#[test]
fn decoration_requires_own_single_character_and_admitted_geometry() {
    let result = styles(
        "<span id='a'>A</span><span id='b'>AB</span>",
        "span {font-size:3em;float:left;font-weight:bold}",
        &["a", "b"],
    );
    assert!(result[0].decoration);
    assert!(!result[0].required);
    assert!(result[0].roles.is_empty());
    assert!(!result[1].decoration);
    assert_eq!(result[1].roles, vec![SemanticRole::Bold]);
}
#[test]
fn encoding_bom_and_declared_icu_conversion_preserve_asset_bytes() {
    let sheet = b"@charset \"ISO-8859-1\"; /* caf\xe9 */ span {font-style:italic}";
    let html = b"<html><head><link rel='stylesheet' href='styles.css'/></head><body><span id='a'>A</span></body></html>";
    let temp = Temp::new(&support::zip(
        &[("chapter.xhtml", html), ("styles.css", sheet)],
        CompressionMethod::Stored,
    ));
    let mut archive = Archive::open(&temp.path).unwrap();
    let tree = parse_source_markup(html).unwrap().tree;
    let css = SourceCss::load(&mut archive, &tree, "chapter.xhtml").unwrap();
    assert_eq!(css.assets()[0].bytes, sheet);
    assert_eq!(
        css.evidence(&tree, &[id(&tree, "a")], "chapter.xhtml")
            .unwrap()[0]
            .roles,
        vec![SemanticRole::Italic]
    );
}
#[test]
fn finite_conditional_budget_rejects_overflow() {
    let sheet: String = (1..=7)
        .map(|index| format!("@media (min-width:{index}px) {{span {{font-weight:bold}}}}"))
        .collect();
    let (_temp, mut archive, tree) = fixture("<span id='a'>A</span>", &sheet, &[]);
    let error = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml")
        .err()
        .unwrap();
    assert_eq!(error.code, "css-alternatives");
}
#[test]
fn wide_resets_and_explicit_inheritance_use_standard_declaration_values() {
    let result = styles(
        "<p id='p'><strong id='reset'>reset</strong><em id='normal'>normal</em><span id='inherit'>inherit</span><sup id='sup'>sup</sup></p>",
        "#p {font-weight:700;font-style:italic} #reset {font:normal 400 16px serif!important} #normal {font-style:initial} #inherit {font-weight:inherit;font-style:inherit} #sup {vertical-align:baseline}",
        &["reset", "normal", "inherit", "sup"],
    );
    for style in result {
        assert!(style.roles.is_empty());
    }
}
#[test]
fn winning_dynamic_semantic_values_fail_but_overridden_values_do_not() {
    let result = styles(
        "<span id='a'>A</span>",
        "span {font-weight:var(--weight);font-weight:normal}",
        &["a"],
    );
    assert!(result[0].roles.is_empty());
    let (_temp, mut archive, tree) = fixture(
        "<span id='a'>A</span>",
        "span {font-style:var(--style)}",
        &[],
    );
    let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
    assert_eq!(
        css.evidence(&tree, &[id(&tree, "a")], "text/chapter.xhtml")
            .unwrap_err()
            .code,
        "css-value"
    );
}
#[test]
fn selected_html_base_and_embedded_sheets_use_archive_local_resolution() {
    let html = b"<html><head><base href='../styles/base.css'/><link rel='stylesheet' href='main.css'/><style>span {font-style:italic}</style></head><body><span id='a'>A</span></body></html>";
    let temp = Temp::new(&support::zip(
        &[
            ("text/chapter.xhtml", html),
            ("styles/main.css", b"span {font-weight:700}"),
        ],
        CompressionMethod::Stored,
    ));
    let mut archive = Archive::open(&temp.path).unwrap();
    let tree = parse_source_markup(html).unwrap().tree;
    let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
    assert_eq!(css.assets()[0].href, "styles/main.css");
    assert_eq!(
        css.evidence(&tree, &[id(&tree, "a")], "text/chapter.xhtml")
            .unwrap()[0]
            .roles,
        vec![SemanticRole::Bold, SemanticRole::Italic]
    );
}
#[test]
fn unsupported_cascade_scope_only_rejects_matched_semantic_rules() {
    let result = styles(
        "<span id='a'>A</span>",
        "@layer book {.absent {font-weight:700} body {font-family:serif}}",
        &["a"],
    );
    assert!(result[0].roles.is_empty());
    let (_temp, mut archive, tree) = fixture(
        "<span id='a'>A</span>",
        "@layer book {span {font-weight:700}}",
        &[],
    );
    let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
    assert_eq!(
        css.evidence(&tree, &[id(&tree, "a")], "text/chapter.xhtml")
            .unwrap_err()
            .code,
        "css-rule"
    );
}
#[test]
fn utf16_bom_uses_native_conversion_without_changing_asset_bytes() {
    let mut bytes = vec![0xff, 0xfe];
    for unit in "span {font-weight:700}".encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    let html = b"<html><head><link rel='stylesheet' href='styles.css'/></head><body><span id='a'>A</span></body></html>";
    let temp = Temp::new(&support::zip(
        &[("chapter.xhtml", html), ("styles.css", &bytes)],
        CompressionMethod::Stored,
    ));
    let mut archive = Archive::open(&temp.path).unwrap();
    let tree = parse_source_markup(html).unwrap().tree;
    let css = SourceCss::load(&mut archive, &tree, "chapter.xhtml").unwrap();
    assert_eq!(css.assets()[0].bytes, bytes);
    assert_eq!(
        css.evidence(&tree, &[id(&tree, "a")], "chapter.xhtml")
            .unwrap()[0]
            .roles,
        vec![SemanticRole::Bold]
    );
}
#[test]
fn active_bold_scope_survives_partial_weight_and_cancels_only_normal_child() {
    let result = styles(
        "<strong id='outer'>outer<span id='partial'>partial</span><span id='reset'>reset</span><span id='sibling'>sibling</span></strong>",
        "#partial {font-weight:600} #reset {font-weight:normal}",
        &["outer", "partial", "reset", "sibling"],
    );
    let outer = result[0].node;
    assert_eq!(result[0].active_emphasis, vec![(outer, SemanticRole::Bold)]);
    assert_eq!(result[1].active_emphasis, vec![(outer, SemanticRole::Bold)]);
    assert!(result[2].active_emphasis.is_empty());
    assert_eq!(result[3].active_emphasis, vec![(outer, SemanticRole::Bold)]);
}
#[test]
fn active_italic_reset_is_local_and_sibling_restores_parent_scope() {
    let result = styles(
        "<em id='outer'>outer<span id='reset'>reset</span><span id='sibling'>sibling</span></em>",
        "#reset {font-style:normal}",
        &["outer", "reset", "sibling"],
    );
    let outer = result[0].node;
    assert!(result[1].active_emphasis.is_empty());
    assert_eq!(
        result[2].active_emphasis,
        vec![(outer, SemanticRole::Italic)]
    );
}
#[test]
fn nested_bold_scopes_keep_outer_origin_when_inner_contrast_is_cancelled() {
    let result = styles(
        "<strong id='outer'>outer<strong id='inner'>inner<span id='reset'>reset</span></strong></strong>",
        "#reset {font-weight:700}",
        &["outer", "inner", "reset"],
    );
    let outer = result[0].node;
    let inner = result[1].node;
    assert_eq!(
        result[1].active_emphasis,
        vec![(outer, SemanticRole::Bold), (inner, SemanticRole::Bold)]
    );
    assert_eq!(result[2].active_emphasis, vec![(outer, SemanticRole::Bold)]);
    assert!(result[2].roles.is_empty());
}
#[test]
fn conditions_compare_active_ancestor_footprints_not_only_own_roles() {
    let (_temp, mut archive, tree) = fixture(
        "<span id='outer'><span id='child'>child</span></span>",
        "body {font-weight:500} #outer {font-weight:700} #child {font-weight:600} @media print {body {font-weight:400}}",
        &[],
    );
    let css = SourceCss::load(&mut archive, &tree, "text/chapter.xhtml").unwrap();
    assert_eq!(
        css.evidence(&tree, &[id(&tree, "child")], "text/chapter.xhtml")
            .unwrap_err()
            .code,
        "css-semantic-ambiguity"
    );
    let result = styles(
        "<strong id='outer'><span id='child'>child</span></strong>",
        "#child {font-weight:600} @media print {#child {font-weight:650}}",
        &["outer", "child"],
    );
    assert_eq!(
        result[1].active_emphasis,
        vec![(result[0].node, SemanticRole::Bold)]
    );
}
#[test]
fn decoration_removes_visual_scopes_but_preserves_core_structural_roles() {
    let result = styles(
        "<strong id='outer'><a id='link' href='chapter.xhtml'>A</a><code id='literal'>B</code><pre id='pre'>C</pre><span id='sibling'>D</span></strong>",
        "a,code,pre {font-size:3em;float:left;font-style:italic}",
        &["outer", "link", "literal", "pre", "sibling"],
    );
    assert!(result[1].decoration);
    assert_eq!(result[1].roles, vec![SemanticRole::Link]);
    assert!(result[1].required);
    assert_eq!(result[2].roles, vec![SemanticRole::Literal]);
    assert_eq!(result[3].roles, vec![SemanticRole::Literal]);
    for style in &result[1..4] {
        assert!(style.active_emphasis.is_empty());
    }
    assert_eq!(
        result[4].active_emphasis,
        vec![(result[0].node, SemanticRole::Bold)]
    );
}
