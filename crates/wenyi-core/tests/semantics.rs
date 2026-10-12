#![forbid(unsafe_code)]

use serde_json::{Value, json};
use wenyi_core::{
    AtomKind, CoreErrorKind, InlineAtom, InlineMark, InlineRun, MarkKind, NoteKind, NoteReference,
    OmissionKind, OrderedAttributes, RichSource, RichTarget, ScalarOffset, ScalarRange,
    SemanticRole, SlotField, SourceIdentity, SourceSlot, ThinkingLevel, note_label_key,
    parse_model_reference, rich_source_digest, slot_contract_digest, stable_digest,
    validate_rich_target,
};

fn mark(id: &str, tag: &str, path: &[i32]) -> InlineMark {
    InlineMark {
        id: id.into(),
        path: path.to_vec(),
        tag: tag.into(),
        attributes: OrderedAttributes::default(),
        source_text: "source".into(),
        kind: match tag {
            "b" | "strong" => MarkKind::Bold,
            "i" | "em" => MarkKind::Italic,
            "a" => MarkKind::Link,
            _ => MarkKind::Style,
        },
        style_evidence: vec![],
        semantics: wenyi_core::tag_semantics(tag).to_vec(),
        required: !wenyi_core::tag_semantics(tag).is_empty(),
    }
}
fn run(text: &str, marks: &[&str]) -> InlineRun {
    InlineRun {
        text: text.into(),
        marks: marks.iter().map(|value| (*value).into()).collect(),
        ..InlineRun::default()
    }
}
fn source(marks: Vec<InlineMark>) -> RichSource {
    RichSource {
        version: 2,
        marks,
        atoms: vec![],
        runs: vec![],
        note_references: vec![],
    }
}
fn target(runs: Vec<InlineRun>) -> RichTarget {
    RichTarget { version: 2, runs }
}

#[test]
fn scalar_coordinates_are_not_bytes_or_graphemes() {
    let text = "a𠀀e\u{301}";
    assert_eq!(ScalarOffset::from_byte(text, 5).unwrap().value(), 2);
    assert!(ScalarOffset::from_byte(text, 2).is_err());
    assert!(ScalarOffset::from_byte(text, text.len() + 1).is_err());
    assert_eq!(
        ScalarRange::new(1, 3, text)
            .unwrap()
            .byte_range(text)
            .unwrap(),
        1..6
    );
    assert_eq!(
        ScalarRange::new(3, 4, text)
            .unwrap()
            .byte_range(text)
            .unwrap(),
        6..8
    );
    assert_eq!(
        ScalarRange::new(4, 4, text)
            .unwrap()
            .byte_range(text)
            .unwrap(),
        8..8
    );
    assert!(ScalarRange::new(3, 2, text).is_err());
    assert!(ScalarRange::new(0, 5, text).is_err());
    assert!(
        ScalarRange::new(0, 4, text)
            .unwrap()
            .byte_range("x")
            .is_err()
    );
}

#[test]
fn source_identity_and_note_labels_validate_business_values() {
    let identity = SourceIdentity::new("a".repeat(64), "Title".into(), "Title".into()).unwrap();
    assert_eq!(identity.input_schema_version, 2);
    assert_eq!(identity.source_slot_schema_version, 4);
    assert_eq!(
        SourceIdentity::new("A".repeat(64), "T".into(), "T".into())
            .unwrap_err()
            .kind,
        CoreErrorKind::Identity
    );
    assert!(SourceIdentity::new("a".repeat(64), "".into(), "T".into()).is_err());
    assert_eq!(note_label_key(" ［１２］. ").unwrap(), "12");
    assert_eq!(note_label_key("†††").unwrap(), "†††");
    for label in ["††††", "12345", "[1)", "chapter", "1 2", "*."] {
        assert!(note_label_key(label).is_err());
    }
}

#[test]
fn frozen_model_catalogue_and_literal_colons() {
    for provider in wenyi_core::model::BUILTIN_PROVIDERS {
        assert_eq!(
            parse_model_reference(&format!(" {provider} / model:32b "))
                .unwrap()
                .model,
            "model:32b"
        );
        assert_eq!(
            parse_model_reference(&format!("{provider}/unknown:off"))
                .unwrap()
                .thinking,
            Some(ThinkingLevel::Off)
        );
    }
    assert_eq!(
        parse_model_reference("openrouter/google/gemini-3.8-flash:high")
            .unwrap()
            .model,
        "google/gemini-3.8-flash"
    );
    assert!(parse_model_reference("opencode-go/deepseek-v4-flash:max").is_ok());
    assert!(parse_model_reference("opencode-go/deepseek-v4-flash:low").is_err());
    assert!(parse_model_reference("bailian/deepseek-v4-flash-0731:max").is_ok());
    assert!(parse_model_reference("bailian/deepseek-v4-pro:high").is_err());
    assert!(parse_model_reference("opencode-go/muse-spark-1.3-contributor:low").is_ok());
    assert!(parse_model_reference("opencode-go/muse-spark-1.3-contributor:medium").is_err());
    assert!(parse_model_reference("openrouter/tencent/hy3:unknown").is_err());
    assert!(parse_model_reference("openai/unknown:high").is_ok());
    assert!(parse_model_reference("ollama/qwen:32b:high").is_err());
    assert!(parse_model_reference("missing/model").is_err());
    assert!(parse_model_reference("openai/ ").is_err());
    assert!(parse_model_reference("model").is_err());
}

#[test]
fn rich_inventory_run_shapes_and_exact_text() {
    let mut src = source(vec![mark("b", "strong", &[0])]);
    let accepted = target(vec![run("中𠀀e\u{301}", &["b"])]);
    validate_rich_target(&src, &accepted, Some("中𠀀e\u{301}")).unwrap();
    assert!(validate_rich_target(&src, &accepted, Some("中𠀀é")).is_err());
    assert!(validate_rich_target(&src, &target(vec![run("text", &[])]), None).is_err());
    assert!(validate_rich_target(&src, &target(vec![run("text", &["unknown"])]), None).is_err());
    assert!(validate_rich_target(&src, &target(vec![run("text", &["b", "b"])]), None).is_err());
    let mut slot = run("text", &["b"]);
    slot.slot_id = Some("s".into());
    assert!(validate_rich_target(&src, &target(vec![slot]), None).is_err());
    src.marks.push(src.marks[0].clone());
    assert!(src.validate().is_err());
    let mut src = source(vec![]);
    src.atoms.push(InlineAtom {
        id: "object".into(),
        path: vec![0],
        kind: AtomKind::Media,
        label: "".into(),
    });
    let atom = InlineRun {
        atom: Some("object".into()),
        ..InlineRun::default()
    };
    src.runs.push(atom.clone());
    src.validate().unwrap();
    validate_rich_target(&src, &target(vec![atom.clone()]), Some("")).unwrap();
    assert!(validate_rich_target(&src, &target(vec![]), None).is_err());
    assert!(validate_rich_target(&src, &target(vec![atom.clone(), atom]), None).is_err());
    src.runs[0].text = "invalid".into();
    assert!(src.validate().is_err());
}

#[test]
fn link_scope_and_anchor_identity() {
    let src = source(vec![mark("link", "a", &[0]), mark("bold", "strong", &[1])]);
    validate_rich_target(
        &src,
        &target(vec![run("a", &["link"]), run("b", &["link", "bold"])]),
        None,
    )
    .unwrap();
    assert!(
        validate_rich_target(
            &src,
            &target(vec![
                run("a", &["link"]),
                run("b", &["bold"]),
                run("c", &["link"])
            ]),
            None
        )
        .is_err()
    );
    let nested = source(vec![mark("a", "a", &[0]), mark("b", "a", &[0, 0])]);
    assert!(validate_rich_target(&nested, &target(vec![run("x", &["a", "b"])]), None).is_err());
    let mut anchor = mark("anchor", "span", &[0]);
    anchor.source_text.clear();
    anchor.attributes.0.push(("id".into(), "identity".into()));
    let src = source(vec![anchor]);
    validate_rich_target(&src, &target(vec![run("", &["anchor"])]), None).unwrap();
    assert!(validate_rich_target(&src, &target(vec![]), None).is_err());
    let empty = source(vec![mark("span", "span", &[0])]);
    assert!(validate_rich_target(&empty, &target(vec![run("", &["span"])]), None).is_err());
}

#[test]
fn omissions_require_eligible_semantics_and_unique_empty_ranges() {
    let src = source(vec![mark("b", "strong", &[0])]);
    let omission = InlineRun {
        marks: vec!["b".into()],
        omission: Some(OmissionKind::Implicit),
        explanation: "implicit in translated phrase".into(),
        ..InlineRun::default()
    };
    let mut tgt = RichTarget {
        version: 3,
        runs: vec![run("译文", &[]), omission.clone()],
    };
    validate_rich_target(&src, &tgt, Some("译文")).unwrap();
    tgt.runs.push(omission.clone());
    assert!(validate_rich_target(&src, &tgt, None).is_err());
    tgt.runs = vec![run("译文", &["b"]), omission.clone()];
    assert!(validate_rich_target(&src, &tgt, None).is_err());
    tgt.runs = vec![omission.clone()];
    tgt.version = 2;
    assert!(tgt.validate().is_err());
    let mut src = source(vec![mark("b", "a", &[0])]);
    tgt.version = 3;
    assert!(validate_rich_target(&src, &tgt, None).is_err());
    src.marks[0] = mark("b", "span", &[0]);
    src.marks[0].required = true;
    validate_rich_target(&src, &tgt, None).unwrap();
    src.marks[0].kind = MarkKind::Decoration;
    assert!(validate_rich_target(&src, &tgt, None).is_err());
}

fn note_source() -> RichSource {
    let mut link = mark("note", "a", &[0]);
    link.source_text = "[1]".into();
    let mut child = mark("sup", "sup", &[0, 0]);
    child.source_text = "[1]".into();
    let mut src = source(vec![link, child]);
    src.note_references.push(NoteReference {
        mark_id: "note".into(),
        label: "[1]".into(),
        kind: NoteKind::Noteref,
        target_resource: "notes.xhtml".into(),
        target_path: vec![1],
        content_marks: vec!["note".into(), "sup".into()],
    });
    src
}

#[test]
fn note_evidence_chain_restoration_and_existing_labels() {
    let mut src = note_source();
    src.validate().unwrap();
    validate_rich_target(&src, &target(vec![run("１", &["note", "sup"])]), None).unwrap();
    assert!(validate_rich_target(&src, &target(vec![run("2", &["note", "sup"])]), None).is_err());
    let note = InlineRun {
        marks: vec!["note".into(), "sup".into()],
        note: Some("note".into()),
        ..InlineRun::default()
    };
    let mut tgt = RichTarget {
        version: 3,
        runs: vec![note.clone()],
    };
    validate_rich_target(&src, &tgt, Some("")).unwrap();
    tgt.runs.push(note.clone());
    assert!(validate_rich_target(&src, &tgt, None).is_err());
    tgt.runs = vec![note.clone(), run("1", &["note", "sup"])];
    assert!(validate_rich_target(&src, &tgt, None).is_err());
    tgt.runs = vec![note];
    tgt.runs[0].marks.pop();
    assert!(validate_rich_target(&src, &tgt, None).is_err());
    src.note_references[0].content_marks.pop();
    assert!(src.validate().is_err());
    let mut src = note_source();
    src.marks[1].path = vec![1];
    assert!(src.validate().is_err());
    let mut src = note_source();
    src.note_references.push(src.note_references[0].clone());
    assert!(src.validate().is_err());
}

#[test]
fn payload_hashes_ignore_key_order_and_source_slot_targets() {
    let left = json!({"z": "é", "a": [1, 2]});
    let right = json!({"a": [1, 2], "z": "é"});
    assert_eq!(
        stable_digest(&left).unwrap(),
        stable_digest(&right).unwrap()
    );
    let mut slot = SourceSlot {
        id: "s".into(),
        element_path: vec![0],
        field: SlotField::Text,
        source_value: "é".into(),
        target_value: None,
    };
    let digest = slot_contract_digest(&[slot.clone()]).unwrap();
    slot.target_value = Some("译文".into());
    assert_eq!(digest, slot_contract_digest(&[slot]).unwrap());
}

#[test]
fn frozen_fixture_preserves_rich_projection_and_slot_digest() {
    let chapter: Value = serde_json::from_str(include_str!("../../../fixtures/migration/state/structured-epub-balanced-v1/run/state/Harbor_Notes/chapters_v2/ch0.json")).unwrap();
    let evidence = &chapter["segments"][1]["epub_state"];
    let wire = &evidence["rich_source"];
    let item = &wire["marks"][0];
    let mut bold = mark(
        item["id"].as_str().unwrap(),
        item["tag"].as_str().unwrap(),
        &[0],
    );
    bold.source_text = item["source_text"].as_str().unwrap().into();
    bold.kind = MarkKind::Bold;
    bold.required = true;
    bold.semantics = vec![SemanticRole::Bold];
    bold.style_evidence = item["style_evidence"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().into())
        .collect();
    let mut src = source(vec![bold]);
    let slots: Vec<_> = evidence["slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|slot| SourceSlot {
            id: slot["id"].as_str().unwrap().into(),
            element_path: slot["element_path"]
                .as_array()
                .unwrap()
                .iter()
                .map(|part| part.as_i64().unwrap() as i32)
                .collect(),
            field: if slot["field"] == "text" {
                SlotField::Text
            } else {
                SlotField::Tail
            },
            source_value: slot["source_value"].as_str().unwrap().into(),
            target_value: None,
        })
        .collect();
    src.runs = wire["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| InlineRun {
            text: item["text"].as_str().unwrap().into(),
            marks: item["marks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|key| key.as_str().unwrap().into())
                .collect(),
            slot_id: item["slot_id"].as_str().map(str::to_owned),
            ..InlineRun::default()
        })
        .collect();
    assert_eq!(
        serde_json::to_string(&src).unwrap(),
        serde_json::to_string(wire).unwrap()
    );
    use sha2::{Digest, Sha256};
    assert_eq!(
        rich_source_digest(&src).unwrap(),
        format!("{:x}", Sha256::digest(serde_json::to_vec(wire).unwrap()))
    );
    assert_eq!(
        slot_contract_digest(&slots).unwrap(),
        evidence["slot_contract_sha256"].as_str().unwrap()
    );
}

#[test]
fn note_atoms_cannot_be_wrapped_in_links() {
    let mut src = source(vec![mark("link", "a", &[0])]);
    src.atoms.push(InlineAtom {
        id: "atom".into(),
        path: vec![1],
        kind: AtomKind::Note,
        label: "1".into(),
    });
    src.runs.push(InlineRun {
        atom: Some("atom".into()),
        ..InlineRun::default()
    });
    let atom = InlineRun {
        atom: Some("atom".into()),
        marks: vec!["link".into()],
        ..InlineRun::default()
    };
    assert_eq!(
        validate_rich_target(&src, &target(vec![run("text", &["link"]), atom]), None)
            .unwrap_err()
            .code,
        "nested_note_links"
    );
}
