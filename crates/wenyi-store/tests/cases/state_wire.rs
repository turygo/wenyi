use crate::support::Fixture;
use serde_json::{Value, json};
use std::fs;
use wenyi_store::{NodeStatus, ObservationOutcome as Outcome, RepairStatus};

#[test]
fn bookkeeping_omissions_are_in_memory_defaults_but_unknown_fields_do_not_write() {
    let fixture = Fixture::new();
    fixture.modify("manifest.json", |value| {
        let object = value.as_object_mut().unwrap();
        object.remove("run_state_schema");
        object.remove("title");
        object.remove("fmt");
        object.remove("progress");
        object.insert("future_bookkeeping".into(), json!({"unchanged": true}));
    });
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(
        observation.outcome,
        Outcome::Current,
        "{:?}",
        observation.diagnostics
    );
    assert_eq!(observation.title.as_deref(), Some(""));
    assert!(
        observation
            .chapters
            .iter()
            .all(|chapter| chapter.progress.repair_ledger.is_empty())
    );
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn missing_and_unsupported_source_binding_is_invalid() {
    let mutations: &[fn(&mut Value)] = &[
        |value| {
            value.as_object_mut().unwrap().remove("identity");
        },
        |value| value["identity"]["source_bytes_sha256"] = "".into(),
        |value| value["identity"]["source_bytes_sha256"] = "f".repeat(64).into(),
        |value| value["identity"]["run_input_schema_version"] = 1.into(),
        |value| value["identity"]["translation_policy_version"] = 4.into(),
        |value| value["meta"]["epub_schema"] = 3.into(),
        |value| value["title"] = "Different book".into(),
        |value| value["fmt"] = "txt".into(),
        |value| value["run_state_schema"] = 91.into(),
        |value| value["run_state_schema"] = "4".into(),
        |value| value["progress"] = Value::Null,
        |value| value["nodes"]["prepare"]["status"] = "unknown".into(),
        |value| value["nodes"]["prepare"]["attempts"] = (-1).into(),
        |value| value["initialized"] = "false".into(),
    ];
    for mutate in mutations {
        let fixture = Fixture::new();
        fixture.modify("manifest.json", mutate);
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
        assert_eq!(before, fixture.snapshot());
    }
}

#[test]
fn recognized_historical_versions_diagnose_migration_and_invalid_precedes_it() {
    for version in [2, 3] {
        let fixture = Fixture::new();
        fixture.modify("manifest.json", |value| {
            value["run_state_schema"] = version.into()
        });
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::MigrationNeeded);
        assert_eq!(before, fixture.snapshot());
        fixture.modify("manifest.json", |value| {
            value["identity"]["translation_policy_version"] = 0.into()
        });
        let observation = fixture.inspect();
        assert_eq!(observation.outcome, Outcome::InvalidState);
        assert!(
            observation
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.outcome == Outcome::MigrationNeeded)
        );
    }
}

#[test]
fn v1_layout_is_only_inferred_from_explicit_old_evidence() {
    let fixture = Fixture::new();
    fs::rename(
        fixture.run.join("chapters_v2"),
        fixture.run.join("chapters"),
    )
    .unwrap();
    fixture.modify("manifest.json", |value| {
        let object = value.as_object_mut().unwrap();
        object.remove("run_state_schema");
        object.remove("nodes");
        object.remove("progress");
        for chapter in object.get_mut("chapters").unwrap().as_array_mut().unwrap() {
            chapter["status"] = "done".into();
        }
    });
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(
        observation.outcome,
        Outcome::MigrationNeeded,
        "{:?}",
        observation.diagnostics
    );
    assert_eq!(observation.chapters.len(), 2);
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn v1_backup_beside_current_layout_does_not_change_omission_default() {
    let fixture = Fixture::new();
    fs::create_dir(fixture.run.join("chapters")).unwrap();
    for index in [0, 1] {
        fs::copy(
            fixture.run.join(format!("chapters_v2/ch{index}.json")),
            fixture.run.join(format!("chapters/ch{index}.json")),
        )
        .unwrap();
    }
    fixture.modify("manifest.json", |value| {
        value.as_object_mut().unwrap().remove("run_state_schema");
    });
    assert_eq!(fixture.inspect().outcome, Outcome::Current);
}

#[test]
fn incomplete_nodes_stay_current_and_stranded_work_needs_recovery() {
    for status in ["pending", "skipped", "failed_retryable", "failed_permanent"] {
        let fixture = Fixture::new();
        fixture.modify("manifest.json", |value| {
            value["nodes"]["prepare"]["status"] = status.into()
        });
        assert_eq!(fixture.inspect().outcome, Outcome::Current);
    }
    let fixture = Fixture::new();
    fixture.modify("manifest.json", |value| {
        value["nodes"]["prepare"]["status"] = "running".into();
        value["progress"]["0"]["repair_ledger"] = json!({"issue": {
            "key": "issue", "chapter": 0, "index": 1, "type": "lint", "attempts": 7,
            "status": "repairing", "committed_target_fingerprint": "paid-content"
        }});
    });
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(observation.outcome, Outcome::RecoveryNeeded);
    assert_eq!(
        observation
            .nodes
            .iter()
            .find(|node| node.key == "prepare")
            .unwrap()
            .status,
        NodeStatus::Running
    );
    assert_eq!(
        observation.chapters[0].progress.repair_ledger[0].status,
        RepairStatus::Repairing
    );
    assert_eq!(
        observation.chapters[0].progress.repair_ledger[0].attempts,
        7
    );
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn chapter_indices_progress_and_processing_are_admitted_strictly() {
    let mutations: &[fn(&mut Value)] = &[
        |value| value["chapters"][1]["index"] = 0.into(),
        |value| value["chapters"][0]["index"] = (-1).into(),
        |value| value["progress"]["99"] = json!({}),
        |value| value["chapters"][0]["processing"]["unexpected"] = true.into(),
        |value| value["chapters"][0]["processing"]["review_required"] = 1.into(),
        |value| value["chapters"][0]["processing"]["strategy_version"] = "old".into(),
        |value| value["progress"]["0"]["pending_polish"] = json!([{"start": 999, "count": 1}]),
    ];
    for mutate in mutations {
        let fixture = Fixture::new();
        fixture.modify("manifest.json", mutate);
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    }
    let fixture = Fixture::new();
    fixture.modify("chapters_v2/ch0.json", |value| value["index"] = 1.into());
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
}

#[test]
fn rich_and_slot_defects_are_rejected_without_text_sanitization() {
    let mutations: &[fn(&mut Value)] = &[
        |value| value["segments"][1]["epub_state"] = Value::Null,
        |value| value["segments"][1]["epub_state"]["block_path"] = json!([-1]),
        |value| {
            value["segments"][1]["epub_state"]
                .as_object_mut()
                .unwrap()
                .remove("block_path");
        },
        |value| value["segments"][1]["epub_state"]["slots"][1]["id"] = "tn1_1:s1".into(),
        |value| value["segments"][1]["epub_state"]["slots"][1]["element_path"] = json!([]),
        |value| value["segments"][1]["epub_state"]["slots"][0]["source_value"] = "altered".into(),
        |value| value["segments"][1]["epub_state"]["unknown"] = true.into(),
        |value| value["segments"][1]["rich_target"]["unknown"] = true.into(),
        |value| value["segments"][1]["rich_target"]["runs"][0]["slot_id"] = "tn1_1:s1".into(),
        |value| value["segments"][1]["rich_target"]["runs"][1]["marks"] = json!(["missing"]),
        |value| value["segments"][1]["target"] = "mismatched\u{0001}".into(),
    ];
    for mutate in mutations {
        let fixture = Fixture::new();
        fixture.modify("chapters_v2/ch0.json", mutate);
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
        assert_eq!(before, fixture.snapshot());
    }
}

#[test]
fn selection_v1_is_discarded_only_in_memory_and_v2_is_typed() {
    let fixture = Fixture::new();
    fixture.modify("output_selection.json", |value| {
        value["schema_version"] = 1.into()
    });
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(observation.outcome, Outcome::Current);
    assert!(observation.output_selection.is_none());
    assert_eq!(before, fixture.snapshot());
    for (field, value) in [
        ("schema_version", json!(3)),
        ("origins", json!({"label": 3})),
        ("selection", json!([])),
        ("source_bytes_sha256", json!("f".repeat(64))),
        ("extra", json!(true)),
    ] {
        let fixture = Fixture::new();
        fixture.modify("output_selection.json", |selection| {
            selection[field] = value
        });
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    }
    let fixture = Fixture::new();
    fs::remove_file(fixture.run.join("output_selection.json")).unwrap();
    assert!(fixture.inspect().output_selection.is_none());
    assert_eq!(fixture.inspect().outcome, Outcome::Current);
}

#[test]
fn opaque_source_atoms_keep_their_non_element_coordinate_scheme() {
    let fixture = Fixture::new();
    fixture.modify("chapters_v2/ch0.json", |value| {
        let segment = &mut value["segments"][1];
        segment["epub_state"]["rich_source"]["atoms"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":"a:-1","path":[-1],"kind":"opaque","label":"source comment"}));
        segment["epub_state"]["rich_source"]["runs"]
            .as_array_mut()
            .unwrap()
            .push(json!({"atom":"a:-1"}));
        segment["rich_target"]["runs"]
            .as_array_mut()
            .unwrap()
            .push(json!({"atom":"a:-1"}));
    });
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(
        observation.outcome,
        Outcome::Current,
        "{:?}",
        observation.diagnostics
    );
    assert_eq!(
        observation.chapters[0].segments[1]
            .epub_state
            .as_ref()
            .unwrap()
            .rich_source
            .as_ref()
            .unwrap()
            .atoms[0]
            .path,
        [-1]
    );
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn accepted_target_characters_are_not_sanitized_or_normalized() {
    let fixture = Fixture::new();
    let target = "exact\u{0001}target e\u{0301}";
    fixture.modify("chapters_v2/ch0.json", |value| {
        value["segments"][0]["target"] = target.into();
        value["segments"][0]["rich_target"]["runs"][0]["text"] = target.into();
    });
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(
        observation.outcome,
        Outcome::Current,
        "{:?}",
        observation.diagnostics
    );
    assert_eq!(
        observation.chapters[0].segments[0].target.as_deref(),
        Some(target)
    );
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn whitespace_slot_target_assignment_uses_exact_source_contract() {
    for target in [
        Value::Null,
        json!(""),
        json!(" \t"),
        json!("not whitespace"),
    ] {
        let fixture = Fixture::new();
        fixture.modify("chapters_v2/ch0.json", |value| {
            let state = &mut value["segments"][1]["epub_state"];
            state["slots"].as_array_mut().unwrap().push(json!({
                "id":"whitespace", "element_path":[99], "field":"tail",
                "source_value":" \t", "target_value":target
            }));
            state["rich_source"]["runs"]
                .as_array_mut()
                .unwrap()
                .push(json!({"text":" \t","slot_id":"whitespace"}));
            let slots: Vec<_> = state["slots"]
                .as_array()
                .unwrap()
                .iter()
                .map(|slot| wenyi_core::SourceSlot {
                    id: slot["id"].as_str().unwrap().to_owned(),
                    element_path: serde_json::from_value(slot["element_path"].clone()).unwrap(),
                    field: if slot["field"] == "text" {
                        wenyi_core::SlotField::Text
                    } else {
                        wenyi_core::SlotField::Tail
                    },
                    source_value: slot["source_value"].as_str().unwrap().to_owned(),
                    target_value: slot["target_value"].as_str().map(str::to_owned),
                })
                .collect();
            state["slot_contract_sha256"] = wenyi_core::digest::slot_contract_digest(&slots)
                .unwrap()
                .into();
        });
        let expected = if target == json!("not whitespace") {
            Outcome::InvalidState
        } else {
            Outcome::Current
        };
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, expected);
        assert_eq!(before, fixture.snapshot());
    }
}

#[test]
fn diagnoses_collect_invalid_unavailable_migration_and_recovery_with_precedence() {
    let fixture = Fixture::new();
    fixture.modify("manifest.json", |value| {
        value["run_state_schema"] = 3.into();
        value["nodes"]["prepare"]["status"] = "running".into();
    });
    fs::write(fixture.run.join("glossary.db-wal"), b"uncheckpointed").unwrap();
    fixture.modify("output_selection.json", |value| {
        value["schema_version"] = 99.into()
    });
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(observation.outcome, Outcome::InvalidState);
    for expected in [
        Outcome::InvalidState,
        Outcome::ObservationUnavailable,
        Outcome::MigrationNeeded,
        Outcome::RecoveryNeeded,
    ] {
        assert!(
            observation
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.outcome == expected)
        );
    }
    assert_eq!(before, fixture.snapshot());
}
