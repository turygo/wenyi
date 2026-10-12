#![forbid(unsafe_code)]
use super::*;
use crate::BookErrorKind;

#[test]
fn loaded_native_identities_are_pinned() {
    let versions = native_versions();
    assert_eq!(versions.libxml2, "21504");
    assert_eq!(versions.expat, "expat_2.9.0");
    assert_eq!(versions.icu, "78.3");
}

#[test]
fn expat_package_namespace_and_ordered_attributes_are_owned() {
    let document = parse_xml(br#"<?xml version="1.0"?><package xmlns="urn:opf" xmlns:epub="urn:epub" z="2" a="1"><metadata><title>Harbor &amp; Notes</title></metadata></package>"#, XmlPurpose::Package).unwrap();
    let tree = document.tree;
    let root = tree.node(tree.root).unwrap();
    let NodeKind::Element(element) = &root.kind else {
        panic!("root must be element");
    };
    assert_eq!(element.name.expanded(), "{urn:opf}package");
    assert_eq!(
        element
            .attributes
            .iter()
            .map(|attr| attr.name.local.as_str())
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert_eq!(element.namespaces.len(), 2);
    assert_eq!(tree.itertext(tree.root).unwrap(), "Harbor & Notes");
    assert_eq!(
        parse_xml(b"<package><broken></package>", XmlPurpose::Package)
            .unwrap_err()
            .kind,
        BookErrorKind::XmlSyntax
    );
}

#[test]
fn strict_tree_preserves_opaque_nodes_cdata_siblings_and_entities() {
    let bytes = br#"<?xml version="1.0"?><!--before--><?before x?><!DOCTYPE html [<!ENTITY word "safe">]><html xmlns="http://www.w3.org/1999/xhtml" xmlns:e="urn:e"><body><p z="2" e:a="1">A<![CDATA[<B>]]><!--note-->tail<?inside x?>after<em>C</em>end&word;</p></body></html><!--after-->"#;
    let document = parse_source_markup(bytes).unwrap();
    assert_eq!(document.envelope.mode, ParseMode::Xml);
    assert!(document.envelope.declaration);
    assert_eq!(document.tree.before_root.len(), 2);
    assert_eq!(document.tree.after_root.len(), 1);
    let body = document
        .tree
        .resolve_element_path(document.tree.root, &[0])
        .unwrap();
    let block = document.tree.resolve_element_path(body, &[0]).unwrap();
    let comment = document.tree.resolve_element_path(block, &[-1]).unwrap();
    assert!(matches!(
        document.tree.node(comment).unwrap().kind,
        NodeKind::Comment(_)
    ));
    assert_eq!(
        document.tree.node(comment).unwrap().tail.as_deref(),
        Some("tail")
    );
    assert_eq!(
        document
            .tree
            .local_name(document.tree.resolve_element_path(block, &[0]).unwrap()),
        Some("em")
    );
    assert!(document.tree.resolve_element_path(block, &[-3]).is_err());
    let serialized = serialize_source(&document).unwrap();
    let text = String::from_utf8(serialized.clone()).unwrap();
    assert!(text.contains("<![CDATA[<B>]]>"));
    assert!(text.contains("&word;"));
    assert!(text.contains("<!ENTITY word"));
    assert!(!text.contains("<meta"));
    let reparsed = parse_source_markup(&serialized).unwrap();
    assert_eq!(document.tree, reparsed.tree);
    assert_eq!(serialize_block(&document.tree, block).unwrap(), br#"<p xmlns="http://www.w3.org/1999/xhtml" xmlns:e="urn:e" z="2" e:a="1">A<![CDATA[<B>]]><!--note-->tail<?inside x?>after<em>C</em>end&word;</p>"#);
}

#[test]
fn html_recovery_is_not_package_recovery() {
    let document = parse_source_markup(b"<html><body><p>one<br>two</body></html>").unwrap();
    assert_eq!(document.envelope.mode, ParseMode::RecoveredHtml);
    assert_eq!(document.envelope.encoding, "UTF-8");
    let serialized = serialize_source(&document).unwrap();
    assert!(!String::from_utf8_lossy(&serialized).contains("<meta"));
    assert!(parse_source_markup(&serialized).is_ok());
    assert!(
        parse_xml(
            b"<html><body><p>one<br>two</body></html>",
            XmlPurpose::Container
        )
        .is_err()
    );
    assert_eq!(
        parse_xml(b"<html><body><p>one<br>two</body></html>", XmlPurpose::Nav)
            .unwrap()
            .envelope
            .mode,
        ParseMode::RecoveredHtml
    );
}

#[test]
fn external_resources_cannot_be_read_or_recovered() {
    for bytes in [
        br#"<!DOCTYPE html [<!ENTITY bad SYSTEM "file:///wenyi-forbidden-input">]><html><body>&bad;</body></html>"#.as_slice(),
        br#"<!DOCTYPE html [<!ENTITY bad SYSTEM "https://invalid.example/wenyi-forbidden-input">]><html><body>&bad;</body></html>"#.as_slice(),
    ] {
        assert_eq!(parse_source_markup(bytes).unwrap_err().kind, BookErrorKind::XmlSafety);
        assert_eq!(parse_xml(bytes, XmlPurpose::Package).unwrap_err().kind, BookErrorKind::XmlSafety);
    }
    let document = parse_source_markup(br#"<!DOCTYPE html SYSTEM "https://invalid.example/never.dtd"><html><body>safe</body></html>"#).unwrap();
    assert!(document.envelope.doctype.unwrap().contains("never.dtd"));
    let document = parse_source_markup(br#"<html xmlns:xi="http://www.w3.org/2001/XInclude"><body><xi:include href="file:///wenyi-forbidden-input"/></body></html>"#).unwrap();
    assert_eq!(document.tree.itertext(document.tree.root).unwrap(), "");
    assert!(document.tree.nodes.iter().any(
        |node| matches!(&node.kind, NodeKind::Element(element) if element.name.local == "include")
    ));
}

#[test]
fn safety_and_encoding_failures_do_not_become_html() {
    let nested = format!(
        "{}text{}",
        "<a>".repeat(MAX_DEPTH + 2),
        "</a>".repeat(MAX_DEPTH + 2)
    );
    assert_eq!(
        parse_source_markup(nested.as_bytes()).unwrap_err().kind,
        BookErrorKind::XmlSafety
    );
    let recovered = parse_source_markup(b"<html><body>\xff</body></html>").unwrap();
    assert_eq!(recovered.envelope.mode, ParseMode::RecoveredHtml);
    assert_eq!(
        recovered.tree.itertext(recovered.tree.root).unwrap(),
        "\u{ff}"
    );
    let too_large = vec![b'x'; MAX_BYTES + 1];
    assert_eq!(
        parse_source_markup(&too_large).unwrap_err().kind,
        BookErrorKind::XmlSafety
    );
}

#[test]
fn bom_utf16_and_cjk_encodings_roundtrip() {
    let source = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><html><body><p>中文日本語𠮷e\u{301}</p></body></html>";
    let mut utf8 = vec![0xef, 0xbb, 0xbf];
    utf8.extend_from_slice(source.as_bytes());
    let document = parse_source_markup(&utf8).unwrap();
    assert_eq!(document.envelope.bom, Some(BomKind::Utf8));
    let output = serialize_source(&document).unwrap();
    assert!(output.starts_with(&[0xef, 0xbb, 0xbf]));
    assert_eq!(document.tree, parse_source_markup(&output).unwrap().tree);
    for (encoding, bom, kind) in [
        ("UTF-16LE", [0xff, 0xfe], BomKind::Utf16Le),
        ("UTF-16BE", [0xfe, 0xff], BomKind::Utf16Be),
    ] {
        let text = source.replace("UTF-8", encoding);
        let encoded =
            convert_encoding(text.as_bytes(), "UTF-8", encoding, ConversionPolicy::Strict).unwrap();
        let mut bytes = bom.to_vec();
        bytes.extend_from_slice(&encoded);
        let parsed = parse_source_markup(&bytes).unwrap();
        assert_eq!(parsed.envelope.bom, Some(kind));
        let output = serialize_source(&parsed).unwrap();
        assert!(output.starts_with(&bom));
        assert_eq!(parsed.tree, parse_source_markup(&output).unwrap().tree);
        assert_eq!(
            parsed.tree,
            parse_xml(&bytes, XmlPurpose::Package).unwrap().tree
        );
    }
    for (encoding, text) in [("GB18030", "中文𠮷"), ("Shift_JIS", "日本語")] {
        let source = format!(
            "<?xml version=\"1.0\" encoding=\"{encoding}\"?><html><body>{text}</body></html>"
        );
        let encoded = convert_encoding(
            source.as_bytes(),
            "UTF-8",
            encoding,
            ConversionPolicy::Strict,
        )
        .unwrap();
        let parsed = parse_source_markup(&encoded).unwrap();
        assert_eq!(parsed.tree.itertext(parsed.tree.root).unwrap(), text);
        let output = serialize_source(&parsed).unwrap();
        assert_eq!(parsed.tree, parse_source_markup(&output).unwrap().tree);
    }
}

#[test]
fn strict_converter_rejects_and_replace_is_explicit() {
    assert_eq!(
        convert_encoding(&[0xff], "UTF-8", "UTF-8", ConversionPolicy::Strict)
            .unwrap_err()
            .kind,
        BookErrorKind::Encoding
    );
    assert_eq!(
        convert_encoding(&[0xff], "UTF-8", "UTF-8", ConversionPolicy::Replace).unwrap(),
        "\u{fffd}".as_bytes()
    );
    assert!(
        convert_encoding(
            b"x",
            "unknown-wenyi-encoding",
            "UTF-8",
            ConversionPolicy::Strict
        )
        .is_err()
    );
    assert!(convert_encoding(b"x", "UTF-8\0suffix", "UTF-8", ConversionPolicy::Strict).is_err());
}

#[test]
fn block_excludes_own_tail_but_retains_child_tails() {
    let document =
        parse_source_markup(b"<html><body><p>A<b>B</b>C<!--x-->D<?p q?>E</p>outside</body></html>")
            .unwrap();
    let block = document
        .tree
        .resolve_element_path(document.tree.root, &[0, 0])
        .unwrap();
    assert_eq!(
        serialize_block(&document.tree, block).unwrap(),
        b"<p>A<b>B</b>C<!--x-->D<?p q?>E</p>"
    );
    assert_eq!(document.tree.itertext(block).unwrap(), "ABCDE");
    assert!(
        document
            .tree
            .resolve_element_path(document.tree.root, &[20])
            .is_err()
    );
    assert!(
        document
            .tree
            .resolve_element_path(document.tree.root, &[i32::MIN])
            .is_err()
    );
}

#[test]
fn native_entity_expansion_is_bounded() {
    let mut declarations = String::from("<!ENTITY a '0123456789'>");
    for (name, previous) in [
        ('b', 'a'),
        ('c', 'b'),
        ('d', 'c'),
        ('e', 'd'),
        ('f', 'e'),
        ('g', 'f'),
    ] {
        declarations.push_str(&format!(
            "<!ENTITY {name} '{}'>",
            format!("&{previous};").repeat(10)
        ));
    }
    let source =
        format!("<!DOCTYPE html [{declarations}]><html value='&g;'><body>safe</body></html>");
    assert_eq!(
        parse_source_markup(source.as_bytes()).unwrap_err().kind,
        BookErrorKind::XmlSafety
    );
    assert_eq!(
        parse_xml(source.as_bytes(), XmlPurpose::Package)
            .unwrap_err()
            .kind,
        BookErrorKind::XmlSafety
    );
}

#[test]
fn nav_html_uses_native_meta_encoding_detection() {
    let text = "<html><head><meta charset='Shift_JIS'></head><body>日本語</body></html>";
    let encoded = convert_encoding(
        text.as_bytes(),
        "UTF-8",
        "Shift_JIS",
        ConversionPolicy::Strict,
    )
    .unwrap();
    let parsed = parse_xml(&encoded, XmlPurpose::Nav).unwrap();
    assert_eq!(parsed.tree.itertext(parsed.tree.root).unwrap(), "日本語");
    assert_eq!(parsed.envelope.encoding, "UTF-8");
}

#[test]
fn recovered_serialization_retains_declaration_presence() {
    let parsed = parse_source_markup(
        b"<?xml version='1.0' encoding='UTF-8'?><html><body><p>one<br>two</body></html>",
    )
    .unwrap();
    assert_eq!(parsed.envelope.mode, ParseMode::RecoveredHtml);
    assert!(
        serialize_source(&parsed)
            .unwrap()
            .starts_with(b"<?xml version='1.0' encoding='UTF-8'?>")
    );
}

#[test]
fn admitted_resource_budget_is_not_reduced_to_sixteen_mebibytes() {
    let text = "x".repeat(1024 * 1024);
    let mut bytes = String::from("<html><body>");
    for _ in 0..17 {
        bytes.push_str("<p>");
        bytes.push_str(&text);
        bytes.push_str("</p>");
    }
    bytes.push_str("</body></html>");
    let document = parse_source_markup(bytes.as_bytes()).unwrap();
    let body = document
        .tree
        .resolve_element_path(document.tree.root, &[0])
        .unwrap();
    let blocks = document.tree.element_children(body);
    let mut count = 0;
    for block in blocks {
        assert_eq!(
            document.tree.node(block).unwrap().text.as_deref(),
            Some(text.as_str())
        );
        count += 1;
    }
    assert_eq!(count, 17);
    assert_eq!(serialize_source(&document).unwrap(), bytes.as_bytes());
}

#[test]
fn ordinary_paragraphs_are_not_limited_by_an_invented_event_count() {
    let mut bytes = String::from("<html><body>");
    for index in 0..120_000 {
        bytes.push_str(&format!("<p>{index}</p>"));
    }
    bytes.push_str("</body></html>");
    let document = parse_source_markup(bytes.as_bytes()).unwrap();
    let body = document
        .tree
        .resolve_element_path(document.tree.root, &[0])
        .unwrap();
    let blocks = document.tree.element_children(body);
    let mut count = 0;
    for (index, block) in blocks.enumerate() {
        assert_eq!(
            document.tree.node(block).unwrap().text.as_deref(),
            Some(index.to_string().as_str())
        );
        count += 1;
    }
    assert_eq!(count, 120_000);
}

#[test]
fn source_html_recovery_detects_legacy_meta_charset() {
    let source = "<html><head><meta charset='Shift_JIS'></head><body>日本語</body></html>";
    let bytes = convert_encoding(
        source.as_bytes(),
        "UTF-8",
        "Shift_JIS",
        ConversionPolicy::Strict,
    )
    .unwrap();
    let document = parse_source_markup(&bytes).unwrap();
    assert_eq!(document.envelope.mode, ParseMode::RecoveredHtml);
    let body = document
        .tree
        .resolve_element_path(document.tree.root, &[1])
        .unwrap();
    assert_eq!(
        document.tree.node(body).unwrap().text.as_deref(),
        Some("日本語")
    );
}

#[test]
fn expat_depth_uses_package_semantics_not_source_libxml_depth() {
    let source = format!("{}text{}", "<a>".repeat(1024), "</a>".repeat(1024));
    let parsed = parse_xml(source.as_bytes(), XmlPurpose::Package).unwrap();
    assert_eq!(parsed.tree.itertext(parsed.tree.root).unwrap(), "text");
    assert_eq!(
        parse_xml(&serialize_source(&parsed).unwrap(), XmlPurpose::Package)
            .unwrap()
            .tree,
        parsed.tree
    );
}

#[test]
fn native_text_node_limit_is_safety_and_never_html_recovery() {
    let source = format!("<html><body>{}</body></html>", "x".repeat(10_000_001));
    assert_eq!(
        parse_source_markup(source.as_bytes()).unwrap_err().kind,
        BookErrorKind::XmlSafety
    );
}

#[test]
fn native_serialization_escapes_owned_values_and_preserves_utf8_block_bytes() {
    let document = parse_source_markup("<html><body><p a='&amp;&quot;&#9;&#10;&#13;'>中文🙂 &amp; &lt; &gt; &#13;</p></body></html>".as_bytes()).unwrap();
    let block = document
        .tree
        .resolve_element_path(document.tree.root, &[0, 0])
        .unwrap();
    assert_eq!(
        serialize_block(&document.tree, block).unwrap(),
        "<p a=\"&amp;&quot;&#9;&#10;&#13;\">中文🙂 &amp; &lt; &gt; &#13;</p>".as_bytes()
    );
}

#[test]
fn serialization_rejects_owned_cycles_and_inconsistent_parents() {
    let mut document = parse_source_markup(b"<html><body><p>x</p></body></html>").unwrap();
    let root = document.tree.root;
    document.tree.nodes[root.index()].children.push(root);
    assert!(serialize_source(&document).is_err());
}

#[test]
fn intermediate_utf16_buffers_do_not_inherit_half_the_raw_budget() {
    let bytes = vec![b'x'; 9 * 1024 * 1024];
    let utf16 = convert_encoding(&bytes, "UTF-8", "UTF-16LE", ConversionPolicy::Strict).unwrap();
    assert_eq!(utf16.len(), bytes.len() * 2);
    assert_eq!(
        convert_encoding(&utf16, "UTF-16LE", "UTF-8", ConversionPolicy::Strict).unwrap(),
        bytes
    );
}

#[test]
fn all_frozen_r0_blocks_match_native_serialization_fingerprints() {
    use sha2::{Digest, Sha256};
    use std::{
        collections::HashMap,
        io::{Cursor, Read},
    };
    let archive = include_bytes!("../../../../fixtures/migration/epub/structured-v1.epub");
    let chapters = [
        include_str!(
            "../../../../fixtures/migration/state/structured-epub-balanced-v1/run/state/Harbor_Notes/chapters_v2/ch0.json"
        ),
        include_str!(
            "../../../../fixtures/migration/state/structured-epub-balanced-v1/run/state/Harbor_Notes/chapters_v2/ch1.json"
        ),
    ];
    let mut archive = zip::ZipArchive::new(Cursor::new(archive)).unwrap();
    let mut documents = HashMap::new();
    let mut qualified = 0;
    for chapter in chapters {
        let chapter: serde_json::Value = serde_json::from_str(chapter).unwrap();
        for segment in chapter["segments"].as_array().unwrap() {
            let state = &segment["epub_state"];
            let href = state["resource_href"].as_str().unwrap();
            if !documents.contains_key(href) {
                let mut bytes = Vec::new();
                archive
                    .by_name(href)
                    .unwrap()
                    .read_to_end(&mut bytes)
                    .unwrap();
                assert_eq!(
                    format!("{:x}", Sha256::digest(&bytes)),
                    state["resource_sha256"].as_str().unwrap()
                );
                documents.insert(href.to_owned(), parse_source_markup(&bytes).unwrap());
            }
            let document = &documents[href];
            assert_eq!(document.envelope.mode, ParseMode::Xml);
            let path = state["block_path"]
                .as_array()
                .unwrap()
                .iter()
                .map(|step| i32::try_from(step.as_i64().unwrap()).unwrap())
                .collect::<Vec<_>>();
            let node = document
                .tree
                .resolve_element_path(document.tree.root, &path)
                .unwrap();
            let bytes = serialize_block(&document.tree, node).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(bytes)),
                state["block_fingerprint"].as_str().unwrap(),
                "{href} {path:?}"
            );
            qualified += 1;
        }
    }
    assert_eq!(qualified, 17);
}

#[test]
fn recovered_html_and_nav_preserve_valueless_attributes() {
    let input = b"<html><body><p class>text</p></body></html>";
    for document in [
        parse_source_markup(input).unwrap(),
        parse_xml(input, XmlPurpose::Nav).unwrap(),
    ] {
        let block = document
            .tree
            .resolve_element_path(document.tree.root, &[0, 0])
            .unwrap();
        assert_eq!(
            serialize_block(&document.tree, block).unwrap(),
            b"<p class=\"\">text</p>"
        );
    }
}

#[test]
fn recovered_html_preserves_literal_colon_attributes_in_native_serialization() {
    let document = parse_source_markup(
        b"<html xmlns:epub=\"urn:epub\"><body><p epub:type=\"chapter\">text<br></p></body></html>",
    )
    .unwrap();
    assert_eq!(document.envelope.mode, ParseMode::RecoveredHtml);
    let block = document
        .tree
        .resolve_element_path(document.tree.root, &[0, 0])
        .unwrap();
    assert_eq!(
        serialize_block(&document.tree, block).unwrap(),
        b"<p epub:type=\"chapter\">text<br/></p>"
    );
    let serialized = serialize_source(&document).unwrap();
    let serialized = std::str::from_utf8(&serialized).unwrap();
    assert!(serialized.contains("xmlns:epub=\"urn:epub\""));
    assert!(serialized.contains("epub:type=\"chapter\""));
}
