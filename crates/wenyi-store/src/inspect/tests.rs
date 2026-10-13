use super::{ObservationOutcome, SourceIdentity, inspect_with_hook};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn copy(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn fixture() -> (tempfile::TempDir, SourceIdentity) {
    let temporary = tempfile::tempdir().unwrap();
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/migration");
    let source_bytes = fs::read(base.join("epub/structured-v1.epub")).unwrap();
    let source = SourceIdentity::new(
        format!("{:x}", Sha256::digest(source_bytes)),
        "Harbor Notes".to_owned(),
        "Harbor_Notes".to_owned(),
    )
    .unwrap();
    copy(
        &base.join("state/structured-epub-balanced-v1/run/state/Harbor_Notes"),
        &temporary.path().join("state/Harbor_Notes"),
    );
    (temporary, source)
}

#[test]
fn changing_artifacts_discard_unstable_semantics_and_diagnostics() {
    type Mutation = fn(&Path);
    let mutations: &[Mutation] = &[
        |run| {
            fs::write(run.join(".run.lock"), b"\0").unwrap();
        },
        |run| {
            fs::write(run.join("usage.wal.json"), b"not JSON").unwrap();
        },
        |run| {
            fs::write(run.join("glossary.db-wal"), b"new WAL").unwrap();
        },
        |run| {
            fs::remove_file(run.join("chapters_v2/ch0.json")).unwrap();
        },
        |run| {
            fs::write(run.join("chapters_v2/ch999.json"), b"new chapter").unwrap();
        },
        |run| {
            let path = run.join("manifest.json");
            let bytes = fs::read(&path).unwrap();
            let replacement = run.join("replacement.json");
            fs::write(&replacement, bytes).unwrap();
            fs::rename(replacement, path).unwrap();
        },
        |run| {
            let path = run.join("usage.json");
            let mut bytes = fs::read(&path).unwrap();
            bytes[0] = b'[';
            fs::write(path, bytes).unwrap();
        },
        #[cfg(unix)]
        |run| {
            let old = run.with_file_name("replaced-run");
            fs::rename(run, &old).unwrap();
            copy(&old, run);
        },
    ];
    for mutate in mutations {
        let (temporary, source) = fixture();
        let state = temporary.path().join("state");
        let run = state.join(&source.run_slug);
        let observation = inspect_with_hook(&state, &source, || mutate(&run)).unwrap();
        assert_eq!(
            observation.outcome,
            ObservationOutcome::ObservationUnavailable
        );
        assert_eq!(observation.diagnostics.len(), 1);
        assert_eq!(observation.diagnostics[0].code, "changing_files");
        assert!(observation.title.is_none());
        assert!(observation.chapters.is_empty());
        assert!(observation.nodes.is_empty());
        assert!(observation.glossary.is_none());
        assert!(observation.output_selection.is_none());
    }
}

#[test]
fn unstable_database_bytes_cannot_produce_a_corruption_diagnosis() {
    let (temporary, source) = fixture();
    let state = temporary.path().join("state");
    let run = state.join(&source.run_slug);
    let observation = inspect_with_hook(&state, &source, || {
        fs::write(run.join("glossary.db"), b"corrupt replacement").unwrap();
    })
    .unwrap();
    assert_eq!(
        observation.outcome,
        ObservationOutcome::ObservationUnavailable
    );
    assert_eq!(observation.diagnostics.len(), 1);
    assert_eq!(observation.diagnostics[0].code, "changing_files");
}

#[cfg(unix)]
#[test]
fn disappearance_of_run_is_observation_unavailable_not_missing_run() {
    let (temporary, source) = fixture();
    let state = temporary.path().join("state");
    let run = state.join(&source.run_slug);
    let observation = inspect_with_hook(&state, &source, || {
        fs::rename(&run, state.join("departed-run")).unwrap();
    })
    .unwrap();
    assert_eq!(
        observation.outcome,
        ObservationOutcome::ObservationUnavailable
    );
    assert_eq!(observation.diagnostics[0].code, "changing_files");
}

#[test]
fn empty_run_replacement_and_disappearance_discard_invalid_snapshot() {
    for replace in [false, true] {
        let (temporary, source) = fixture();
        let state = temporary.path().join("state");
        let run = state.join(&source.run_slug);
        fs::remove_dir_all(&run).unwrap();
        fs::create_dir(&run).unwrap();
        let observation = inspect_with_hook(&state, &source, || {
            fs::rename(&run, state.join("departed-run")).unwrap();
            if replace {
                fs::create_dir(&run).unwrap();
            }
        })
        .unwrap();
        assert_eq!(
            observation.outcome,
            ObservationOutcome::ObservationUnavailable
        );
        assert_eq!(observation.diagnostics.len(), 1);
        assert_eq!(observation.diagnostics[0].code, "changing_files");
        assert!(observation.title.is_none());
        assert!(observation.chapters.is_empty());
        assert!(observation.nodes.is_empty());
        assert!(observation.glossary.is_none());
        assert!(observation.output_selection.is_none());
    }
}

#[test]
fn report_content_is_not_an_observed_state_dependency() {
    let (temporary, source) = fixture();
    let state = temporary.path().join("state");
    let observation = inspect_with_hook(&state, &source, || {
        fs::write(
            state.join("Harbor_Notes/report.json"),
            b"not a state wire document",
        )
        .unwrap();
    })
    .unwrap();
    assert_eq!(observation.outcome, ObservationOutcome::Current);
}
