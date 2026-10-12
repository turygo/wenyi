use crate::support::Fixture;
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use wenyi_core::digest::stable_digest;
use wenyi_store::ObservationOutcome as Outcome;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn wrapped(value: Value) -> Value {
    json!({"sha256": stable_digest(&value).unwrap(), "data": value})
}

fn note_record(fixture: &Fixture) -> Value {
    let before_manifest = fixture.read("manifest.json");
    let mut after_manifest = before_manifest.clone();
    after_manifest["meta"]["note_recovery_test"] = true.into();
    let before_bytes = fs::read(fixture.run.join("chapters_v2/ch0.json")).unwrap();
    let before: Value = serde_json::from_slice(&before_bytes).unwrap();
    let mut after = before.clone();
    after["meta"]["note_recovery_test"] = true.into();
    json!({
        "version": 1,
        "source_bytes_sha256": fixture.source.source_bytes_sha256,
        "before_meta": wrapped(before_manifest["meta"].clone()),
        "after_meta": wrapped(after_manifest["meta"].clone()),
        "before_manifest": wrapped(before_manifest),
        "after_manifest": wrapped(after_manifest),
        "chapters": [{"index": 0, "before": wrapped(before), "after": wrapped(after),
            "before_bytes_base64": STANDARD.encode(&before_bytes), "before_file_sha256": hash(&before_bytes)}]
    })
}

#[test]
fn valid_translate_and_polish_journals_are_observed_without_commit_or_clear() {
    for record in [
        json!({"kind":"translate","chapter":0,"start":0,"count":2}),
        json!({"kind":"polish","chapter":0,"start":0,"count":2,"targets":["first", "second"]}),
    ] {
        let fixture = Fixture::new();
        fixture.write("journal.json", &record);
        let before = fixture.snapshot();
        let observation = fixture.inspect();
        assert_eq!(
            observation.outcome,
            Outcome::RecoveryNeeded,
            "{:?}",
            observation.diagnostics
        );
        assert_eq!(before, fixture.snapshot());
        assert_eq!(fixture.read("journal.json"), record);
    }
}

#[test]
fn corrupt_checkpoint_records_are_invalid_instead_of_cleared() {
    for record in [
        json!({"kind":"unknown","chapter":0,"start":0,"count":1}),
        json!({"kind":"translate","chapter":99,"start":0,"count":1}),
        json!({"kind":"translate","chapter":0,"start":0,"count":0}),
        json!({"kind":"translate","chapter":0,"start":-1,"count":1}),
        json!({"kind":"translate","chapter":0,"start":0,"count":999}),
        json!({"kind":"polish","chapter":0,"start":0,"count":2,"targets":["only-one"]}),
        json!({"kind":"polish","chapter":0,"start":0,"count":1,"targets":[3]}),
    ] {
        let fixture = Fixture::new();
        fixture.write("journal.json", &record);
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
        assert_eq!(before, fixture.snapshot());
    }
    let fixture = Fixture::new();
    fs::write(fixture.run.join("journal.json"), b"{").unwrap();
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn note_migration_accepts_stable_before_after_and_existing_valid_backups() {
    let fixture = Fixture::new();
    let record = note_record(&fixture);
    fixture.write("note_migration.json", &record);
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::RecoveryNeeded);
    assert_eq!(before, fixture.snapshot());
    assert!(!fixture.run.join("note_migration_backup").exists());
    fixture.write("manifest.json", &record["after_manifest"]["data"]);
    fixture.write(
        "chapters_v2/ch0.json",
        &record["chapters"][0]["after"]["data"],
    );
    fs::create_dir(fixture.run.join("note_migration_backup")).unwrap();
    fixture.write("note_migration_backup/manifest_meta.json", &json!({"version":1,"source_bytes_sha256":fixture.source.source_bytes_sha256,"data":record["before_meta"]["data"]}));
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(
        observation.outcome,
        Outcome::RecoveryNeeded,
        "{:?}",
        observation.diagnostics
    );
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn note_migration_corruption_is_not_hidden_by_recovery_diagnosis() {
    let mutations: &[fn(&mut Value)] = &[
        |record| record["version"] = 2.into(),
        |record| record["source_bytes_sha256"] = "f".repeat(64).into(),
        |record| record["before_meta"]["sha256"] = "f".repeat(64).into(),
        |record| record["chapters"][0]["index"] = 99.into(),
        |record| record["chapters"][0]["before_bytes_base64"] = "not base64!".into(),
        |record| record["chapters"][0]["before_file_sha256"] = "f".repeat(64).into(),
        |record| {
            let chapter = record["chapters"][0].clone();
            record["chapters"].as_array_mut().unwrap().push(chapter);
        },
        |record| {
            record["before_meta"]["data"]["different"] = true.into();
            record["before_meta"]["sha256"] = stable_digest(&record["before_meta"]["data"])
                .unwrap()
                .into();
        },
    ];
    for mutate in mutations {
        let fixture = Fixture::new();
        let mut record = note_record(&fixture);
        mutate(&mut record);
        fixture.write("note_migration.json", &record);
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
        assert_eq!(before, fixture.snapshot());
    }
}

#[test]
fn note_migration_detects_unrelated_current_changes_and_conflicting_backups() {
    let fixture = Fixture::new();
    fixture.write("note_migration.json", &note_record(&fixture));
    fixture.modify("manifest.json", |value| {
        value["meta"]["independent"] = true.into()
    });
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    let fixture = Fixture::new();
    fixture.write("note_migration.json", &note_record(&fixture));
    fs::create_dir(fixture.run.join("note_migration_backup")).unwrap();
    fixture.write(
        "note_migration_backup/manifest_meta.json",
        &json!({"version":1,"data":{}}),
    );
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    assert_eq!(before, fixture.snapshot());
}

fn usage_record(fixture: &Fixture) -> (Value, Vec<u8>) {
    let after = json!({"schema_version":2,"totals":{"calls":1},"unknown":"preserved"});
    let serialized = b"{\n  \"schema_version\": 2,\n  \"totals\": {\n    \"calls\": 1\n  },\n  \"unknown\": \"preserved\"\n}\n".to_vec();
    let before = fs::read(fixture.run.join("usage.json")).unwrap();
    (
        json!({"wal_schema_version":1,"before_sha256":hash(&before),"after_sha256":hash(&serialized),"after":after,"tracker_after":{"schema_version":2}}),
        serialized,
    )
}

#[test]
fn usage_wal_allows_absent_before_but_never_creates_current_snapshot() {
    let fixture = Fixture::new();
    let (mut record, _) = usage_record(&fixture);
    record["before_sha256"] = Value::Null;
    fs::remove_file(fixture.run.join("usage.json")).unwrap();
    fixture.write("usage.wal.json", &record);
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::RecoveryNeeded);
    assert_eq!(before, fixture.snapshot());
    assert!(!fixture.run.join("usage.json").exists());
}

#[test]
fn corrupt_usage_wal_and_old_nonempty_snapshots_are_invalid_without_backfill() {
    let mutations: &[fn(&mut Value)] = &[
        |record| record["wal_schema_version"] = 2.into(),
        |record| record["before_sha256"] = "bad".into(),
        |record| record["after_sha256"] = "f".repeat(64).into(),
        |record| record["after"]["schema_version"] = 1.into(),
        |record| record["tracker_after"]["schema_version"] = 1.into(),
        |record| {
            record.as_object_mut().unwrap().remove("before_sha256");
        },
    ];
    for mutate in mutations {
        let fixture = Fixture::new();
        let (mut record, _) = usage_record(&fixture);
        mutate(&mut record);
        fixture.write("usage.wal.json", &record);
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
        assert_eq!(before, fixture.snapshot());
    }
    for usage in [
        json!({"totals":{"calls":7}}),
        json!({"schema_version":1,"totals":{"calls":7}}),
    ] {
        let fixture = Fixture::new();
        fixture.write("usage.json", &usage);
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
        assert_eq!(before, fixture.snapshot());
    }
}

#[test]
fn current_usage_preserves_unknown_accounting_and_rejects_malformed_dimensions() {
    let fixture = Fixture::new();
    fixture.write(
        "usage.json",
        &json!({
            "schema_version":2,
            "totals":{"calls":7, "not_a_known_counter":{"preserved":true}},
            "by_agent":{"agent":{"unknown_measurement":"retained"}},
            "unknown_accounting":{"exact":[1,2,3]}
        }),
    );
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::Current);
    assert_eq!(before, fixture.snapshot());
    fixture.write(
        "usage.json",
        &json!({"schema_version":2,"by_agent":["not a map"]}),
    );
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn current_usage_bytes_must_match_the_wal_before_or_after_hash() {
    let fixture = Fixture::new();
    let (record, _) = usage_record(&fixture);
    fixture.write("usage.wal.json", &record);
    fixture.write(
        "usage.json",
        &json!({"schema_version":2,"totals":{"calls":999}}),
    );
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    assert_eq!(before, fixture.snapshot());
}
