use crate::support::{Fixture, snapshot};
use rusqlite::{Connection, types::Value};
use std::fs::{self, File};
use wenyi_store::{ObservationOutcome as Outcome, inspect_run};

#[test]
fn missing_run_never_creates_state_or_lock() {
    let fixture = Fixture::new();
    let missing = fixture.temporary.path().join("absent/state");
    let before = fixture.snapshot();
    let observation = inspect_run(&missing, &fixture.source).unwrap();
    assert_eq!(observation.outcome, Outcome::MissingRun);
    assert!(observation.chapters.is_empty());
    assert!(observation.glossary.is_none());
    assert_eq!(before, fixture.snapshot());
    assert!(!missing.exists());
}

#[test]
fn completed_r0_reopens_without_dereferencing_historical_source_path() {
    let fixture = Fixture::new();
    let manifest = fixture.read("manifest.json");
    assert_eq!(
        manifest["identity"]["source_bytes_sha256"],
        fixture.source.source_bytes_sha256
    );
    assert!(
        manifest["source_path"]
            .as_str()
            .unwrap()
            .contains("wenyi-r0-reference-")
    );
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(
        observation.outcome,
        Outcome::Current,
        "{:?}",
        observation.diagnostics
    );
    assert!(observation.diagnostics.is_empty());
    assert_eq!(observation.title.as_deref(), Some("Harbor Notes"));
    assert_eq!(observation.title_translated.as_deref(), Some("港湾札记"));
    assert_eq!(observation.source_lang.as_deref(), Some("en"));
    assert_eq!(observation.target_lang.as_deref(), Some("zh"));
    assert_eq!(observation.chapters.len(), 2);
    assert_eq!(observation.chapters[0].segments.len(), 15);
    assert_eq!(
        observation.chapters[0].segments[1].source,
        "A quiet passage holds a story."
    );
    assert_eq!(
        observation.chapters[0].segments[1].target.as_deref(),
        Some("一段安静的文字讲述着故事。")
    );
    assert_eq!(
        observation.chapters[0].segments[1]
            .rich_target
            .as_ref()
            .unwrap()
            .runs[1]
            .marks,
        ["m:0"]
    );
    assert_eq!(
        observation.chapters[0].segments[1]
            .epub_state
            .as_ref()
            .unwrap()
            .slots
            .len(),
        3
    );
    assert_eq!(
        observation.nodes.len(),
        manifest["nodes"].as_object().unwrap().len()
    );
    assert_eq!(observation.glossary.unwrap().terms, 0);
    assert_eq!(observation.glossary.unwrap().open_conflicts, 0);
    assert!(observation.output_selection.is_some());
    assert_eq!(before, fixture.snapshot());
    assert!(!fixture.run.join(".run.lock").exists());
}

#[test]
fn shared_observation_lock_does_not_write_empty_or_nonempty_lock_file() {
    for contents in [&b""[..], &b"\0"[..]] {
        let fixture = Fixture::new();
        fs::write(fixture.run.join(".run.lock"), contents).unwrap();
        let before = fixture.snapshot();
        assert_eq!(fixture.inspect().outcome, Outcome::Current);
        assert_eq!(before, fixture.snapshot());
    }
}

#[test]
fn real_exclusive_lock_prevents_stranded_running_diagnosis() {
    let fixture = Fixture::new();
    fixture.modify("manifest.json", |value| {
        value["nodes"]["prepare"]["status"] = "running".into()
    });
    let path = fixture.run.join(".run.lock");
    fs::write(&path, b"\0").unwrap();
    let held = File::options().read(true).write(true).open(&path).unwrap();
    held.lock().unwrap();
    let before = fixture.snapshot();
    let active = fixture.inspect();
    assert_eq!(active.outcome, Outcome::ObservationUnavailable);
    assert_eq!(active.diagnostics.len(), 1);
    assert_eq!(active.diagnostics[0].code, "active_run_lock");
    assert!(active.nodes.is_empty());
    assert_eq!(before, fixture.snapshot());
    drop(held);
    assert_eq!(fixture.inspect().outcome, Outcome::RecoveryNeeded);
}

#[test]
fn nonempty_wal_and_hot_journal_suppress_counts_without_recovery() {
    let fixture = Fixture::new();
    fs::write(fixture.run.join("glossary.db-wal"), b"uncheckpointed bytes").unwrap();
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(observation.outcome, Outcome::ObservationUnavailable);
    assert!(observation.glossary.is_none());
    assert_eq!(before, fixture.snapshot());
    fs::remove_file(fixture.run.join("glossary.db-wal")).unwrap();
    let mut journal = vec![0; 1024];
    journal[..8].copy_from_slice(&[0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7]);
    fs::write(fixture.run.join("glossary.db-journal"), &journal).unwrap();
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(observation.outcome, Outcome::RecoveryNeeded);
    assert!(observation.glossary.is_none());
    assert_eq!(before, fixture.snapshot());
    fs::write(fixture.run.join("glossary.db-journal"), vec![0; 1024]).unwrap();
    fs::write(fixture.run.join("glossary.db-wal"), []).unwrap();
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::Current);
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn sqlite_null_and_locked_values_are_not_coerced_or_rewritten() {
    let fixture = Fixture::new();
    let path = fixture.run.join("glossary.db");
    let connection = Connection::open(&path).unwrap();
    connection.execute("INSERT INTO glossary (source,target,reading,type,gender,aliases,first_chapter,note,confidence,locked,status,updated_at) VALUES ('key','value',NULL,NULL,NULL,NULL,NULL,NULL,NULL,1,NULL,NULL)", []).unwrap();
    connection
        .execute(
            "INSERT INTO glossary (source,target,locked) VALUES ('nullable','target',NULL)",
            [],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO term_conflicts (source,resolved) VALUES ('key',0),('key',1),('key',NULL)",
            [],
        )
        .unwrap();
    drop(connection);
    let before = fixture.snapshot();
    let observation = fixture.inspect();
    assert_eq!(
        observation.outcome,
        Outcome::Current,
        "{:?}",
        observation.diagnostics
    );
    assert_eq!(observation.glossary.unwrap().terms, 2);
    assert_eq!(observation.glossary.unwrap().open_conflicts, 1);
    assert_eq!(before, fixture.snapshot());
    let mut uri = url::Url::from_file_path(&path).unwrap();
    uri.set_query(Some("mode=ro&immutable=1"));
    let connection = Connection::open_with_flags(
        uri.as_str(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
    .unwrap();
    let raw: (Value, Value, Value) = connection
        .query_row(
            "SELECT reading,locked,status FROM glossary WHERE source='key'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(raw, (Value::Null, Value::Integer(1), Value::Null));
    drop(connection);
    assert_eq!(before, snapshot(fixture.temporary.path()));
}

#[test]
fn missing_invalid_and_schema_inconsistent_database_are_distinct() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.run.join("glossary.db")).unwrap();
    assert_eq!(fixture.inspect().outcome, Outcome::ObservationUnavailable);
    fs::write(fixture.run.join("glossary.db"), b"not sqlite").unwrap();
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    fs::remove_file(fixture.run.join("glossary.db")).unwrap();
    let connection = Connection::open(fixture.run.join("glossary.db")).unwrap();
    connection
        .execute_batch("CREATE TABLE glossary(source TEXT PRIMARY KEY,target TEXT NOT NULL);")
        .unwrap();
    drop(connection);
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
}

#[test]
fn malformed_locked_value_is_invalid_without_coercion() {
    let fixture = Fixture::new();
    let connection = Connection::open(fixture.run.join("glossary.db")).unwrap();
    connection
        .execute(
            "INSERT INTO glossary(source,target,locked) VALUES('key','value',2)",
            [],
        )
        .unwrap();
    drop(connection);
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::InvalidState);
    assert_eq!(before, fixture.snapshot());
}

#[test]
fn historical_source_path_is_bookkeeping_not_an_input_to_open() {
    let fixture = Fixture::new();
    let unusable_source = fixture.temporary.path().join("not-a-source-file");
    fs::create_dir(&unusable_source).unwrap();
    fixture.modify("manifest.json", |value| {
        value["source_path"] = unusable_source.to_str().unwrap().into()
    });
    let before = fixture.snapshot();
    assert_eq!(fixture.inspect().outcome, Outcome::Current);
    assert_eq!(before, fixture.snapshot());
}
