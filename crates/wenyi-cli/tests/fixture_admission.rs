#[path = "support/fixtures.rs"]
mod fixtures;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use fixtures::{FixtureError, FixtureWorkspace, admit_fixture};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const REFERENCE_PATH: &str = "state/structured-epub-balanced-v1/reference.json";

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/migration")
}

#[test]
fn admits_and_materializes_the_qualified_bundle_in_an_owned_workspace() {
    let fixture =
        admit_fixture(&fixture_root()).expect("qualified migration fixture must be admitted");
    assert_eq!(fixture.fixture_id(), "structured-epub-balanced-v1");
    assert!(
        fixture
            .workspace_mappings()
            .any(|(_, destination)| destination == "config.yaml")
    );
    assert!(
        fixture
            .workspace_mappings()
            .any(|(_, destination)| destination == "source/book.epub")
    );

    let workspace = FixtureWorkspace::new(&fixture).expect("fixture workspace must be isolated");
    assert!(workspace.root().join("config.yaml").is_file());
    assert!(workspace.root().join("source/book.epub").is_file());
    assert!(workspace.root().join("state").is_dir());
    assert!(!workspace.root().join("expected.json").exists());
}

#[test]
fn rejects_missing_hashed_unsafe_incompatible_and_unreferenced_inputs() {
    let missing = synthetic_fixture();
    fs::remove_file(missing.path().join("epub/structured-v1.epub"))
        .expect("remove source artifact");
    assert!(matches!(
        admit_fixture(missing.path()),
        Err(FixtureError::MissingArtifact(_))
    ));

    let changed = synthetic_fixture();
    alter_artifact_without_changing_size(&changed);
    assert!(matches!(
        admit_fixture(changed.path()),
        Err(FixtureError::HashMismatch { .. })
    ));

    let unsafe_path = synthetic_fixture();
    mutate_reference(&unsafe_path, |reference| {
        reference["files"][0]["path"] = json!("../escape");
    });
    assert!(matches!(
        admit_fixture(unsafe_path.path()),
        Err(FixtureError::UnsafePath(_))
    ));

    let wrong_schema = synthetic_fixture();
    mutate_reference(&wrong_schema, |reference| {
        reference["schema_version"] = json!(2)
    });
    assert!(matches!(
        admit_fixture(wrong_schema.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));

    let wrong_baseline = synthetic_fixture();
    mutate_reference(&wrong_baseline, |reference| {
        reference["baseline_commit"] = json!("0".repeat(40));
    });
    assert!(matches!(
        admit_fixture(wrong_baseline.path()),
        Err(FixtureError::BaselineMismatch { .. })
    ));

    let unlisted_receipt = synthetic_fixture();
    mutate_reference(&unlisted_receipt, |reference| {
        reference["outputs"][0]["receipt"] = json!("outputs/missing-receipt.json");
    });
    assert!(matches!(
        admit_fixture(unlisted_receipt.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));
}

#[test]
fn rejects_invalid_required_objects_roles_destinations_and_file_sizes() {
    let wrong_object = synthetic_fixture();
    mutate_reference(&wrong_object, |reference| {
        reference["wire_versions"] = json!([])
    });
    assert!(matches!(
        admit_fixture(wrong_object.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));

    let duplicate_role = synthetic_fixture();
    mutate_reference(&duplicate_role, |reference| {
        reference["outputs"][1]["role"] = json!("mono");
    });
    assert!(matches!(
        admit_fixture(duplicate_role.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));

    let duplicate_destination = synthetic_fixture();
    mutate_reference(&duplicate_destination, |reference| {
        reference["files"][1]["workspace_path"] = json!("config.yaml");
    });
    assert!(matches!(
        admit_fixture(duplicate_destination.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));

    let duplicate_path = synthetic_fixture();
    mutate_reference(&duplicate_path, |reference| {
        reference["files"][1]["path"] = reference["files"][0]["path"].clone();
    });
    assert!(matches!(
        admit_fixture(duplicate_path.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));

    let changed_size = synthetic_fixture();
    fs::write(
        changed_size.path().join("epub/structured-v1.epub"),
        b"short",
    )
    .expect("change source artifact size");
    assert!(matches!(
        admit_fixture(changed_size.path()),
        Err(FixtureError::ArtifactSizeMismatch { .. })
    ));
}

#[test]
fn rejects_projection_mismatches_and_revalidates_before_materialization() {
    let mismatched_projection = synthetic_fixture();
    mutate_reference(&mismatched_projection, |reference| {
        reference["semantic_projection"]["sha256"] = json!("0".repeat(64));
    });
    assert!(matches!(
        admit_fixture(mismatched_projection.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));

    let changed_after_admission = synthetic_fixture();
    let fixture =
        admit_fixture(changed_after_admission.path()).expect("synthetic fixture is valid");
    alter_artifact_without_changing_size(&changed_after_admission);
    assert!(matches!(
        FixtureWorkspace::new(&fixture),
        Err(FixtureError::HashMismatch { .. })
    ));
}

#[cfg(unix)]
#[test]
fn rejects_symlinked_artifacts_even_when_the_target_is_contained() {
    use std::os::unix::fs::symlink;

    let fixture = synthetic_fixture();
    let source = fixture.path().join("epub/structured-v1.epub");
    let real_source = fixture.path().join("epub/source-copy.epub");
    fs::rename(&source, &real_source).expect("move source artifact");
    symlink("source-copy.epub", &source).expect("replace artifact with a symlink");
    assert!(matches!(
        admit_fixture(fixture.path()),
        Err(FixtureError::UnsafePath(_))
    ));
}

#[test]
fn rejects_case_aliases_resolved_by_the_fixture_filesystem() {
    let fixture = synthetic_fixture();
    let source = fixture.path().join("epub/structured-v1.epub");
    let alias = fixture.path().join("EPUB/structured-v1.epub");
    if fs::canonicalize(&source).ok() != fs::canonicalize(&alias).ok() {
        return;
    }
    mutate_reference(&fixture, |reference| {
        let mut record = reference["files"]
            .as_array()
            .expect("file records")
            .iter()
            .find(|record| record["path"] == "epub/structured-v1.epub")
            .expect("source record")
            .clone();
        record["path"] = json!("EPUB/structured-v1.epub");
        record["workspace_path"] = Value::Null;
        reference["files"]
            .as_array_mut()
            .expect("file records")
            .push(record);
        reference["files"]
            .as_array_mut()
            .expect("file records")
            .sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    });
    assert!(matches!(
        admit_fixture(fixture.path()),
        Err(FixtureError::UnsafePath(_))
    ));
}

#[test]
fn rejects_workspace_case_aliases_resolved_by_the_workspace_filesystem() {
    let probe = tempfile::tempdir().expect("create case-sensitivity probe");
    let lowercase = probe.path().join("case-probe");
    fs::write(&lowercase, b"probe").expect("write case-sensitivity probe");
    if !probe.path().join("CASE-PROBE").exists() {
        return;
    }

    let fixture = synthetic_fixture();
    mutate_reference(&fixture, |reference| {
        let records = reference["files"].as_array_mut().expect("file records");
        let source_record = records
            .iter_mut()
            .find(|record| record["path"] == "epub/structured-v1.epub")
            .expect("source record");
        source_record["workspace_path"] = json!("Config.yaml");
    });
    assert!(matches!(
        admit_fixture(fixture.path()),
        Err(FixtureError::InvalidMetadata(_))
    ));
}

fn alter_artifact_without_changing_size(fixture: &TempDir) {
    let path = fixture.path().join("epub/structured-v1.epub");
    let mut bytes = fs::read(&path).expect("read source artifact");
    bytes[0] ^= 1;
    fs::write(path, bytes).expect("change source artifact bytes");
}

fn mutate_reference(fixture: &TempDir, mutate: impl FnOnce(&mut Value)) {
    let path = fixture.path().join(REFERENCE_PATH);
    let mut reference: Value = serde_json::from_slice(&fs::read(&path).expect("read reference"))
        .expect("decode reference");
    mutate(&mut reference);
    fs::write(
        path,
        serde_json::to_vec_pretty(&reference).expect("encode reference"),
    )
    .expect("write modified reference");
}

fn synthetic_fixture() -> TempDir {
    let root = tempfile::tempdir().expect("create synthetic fixture root");
    let mut artifacts = BTreeMap::<String, (Vec<u8>, Option<String>)>::new();
    artifacts.insert(
        "config.yaml".to_owned(),
        (
            b"quality: balanced\n".to_vec(),
            Some("config.yaml".to_owned()),
        ),
    );
    artifacts.insert(
        "epub/structured-v1.epub".to_owned(),
        (
            b"synthetic source bytes".to_vec(),
            Some("source/book.epub".to_owned()),
        ),
    );
    artifacts.insert(
        "expected.json".to_owned(),
        (b"{\"format_version\":1}\n".to_vec(), None),
    );
    artifacts.insert(
        "state/structured-epub-balanced-v1/run/state/book/manifest.json".to_owned(),
        (
            b"{\"schema_version\":4}\n".to_vec(),
            Some("state/book/manifest.json".to_owned()),
        ),
    );

    let roles = [
        ("mono", "mono"),
        ("bilingual_target_first", "target-first"),
        ("bilingual_source_first", "source-first"),
    ];
    let mut outputs = Vec::new();
    for (role, slug) in roles {
        let file = format!("outputs/{slug}.epub");
        let proof = format!("outputs/{slug}-proof.json");
        let receipt = format!("outputs/{slug}-receipt.json");
        artifacts.insert(
            file.clone(),
            (
                format!("epub bytes for {slug}").into_bytes(),
                Some(file.clone()),
            ),
        );
        artifacts.insert(proof.clone(), (b"{}\n".to_vec(), Some(proof.clone())));
        artifacts.insert(receipt.clone(), (b"{}\n".to_vec(), Some(receipt.clone())));
        outputs.push(json!({"role": role, "file": file, "proof": proof, "receipt": receipt}));
    }

    let mut file_records = Vec::new();
    for (path, (bytes, workspace_path)) in &artifacts {
        let artifact_path = root.path().join(path);
        fs::create_dir_all(artifact_path.parent().expect("artifact parent"))
            .expect("create artifact parent");
        fs::write(&artifact_path, bytes).expect("write synthetic artifact");
        let hash = format!("{:x}", Sha256::digest(bytes));
        file_records.push(json!({
            "path": path,
            "size_bytes": bytes.len(),
            "sha256": hash,
            "workspace_path": workspace_path,
        }));
    }
    let expected_hash = format!("{:x}", Sha256::digest(&artifacts["expected.json"].0));
    let reference = json!({
        "schema_version": 1,
        "fixture_id": "structured-epub-balanced-v1",
        "baseline_commit": "61835587606b5cb81cdcd81eec8427fd4f638df4",
        "reference_inputs": {"inventories": {}},
        "environment": {"platform": "synthetic"},
        "wire_versions": {"run_state": 4},
        "files": file_records,
        "outputs": outputs,
        "semantic_projection": {
            "path": "expected.json",
            "sha256": expected_hash,
            "format_version": 1,
        },
        "generation": {
            "origin_bound_fields": [],
            "omissions": [],
            "executed_checks": ["synthetic admission test"],
        },
    });
    let reference_path = root.path().join(REFERENCE_PATH);
    fs::create_dir_all(reference_path.parent().expect("reference parent"))
        .expect("create reference parent");
    fs::write(
        reference_path,
        serde_json::to_vec_pretty(&reference).expect("encode synthetic reference"),
    )
    .expect("write synthetic reference");
    root
}
