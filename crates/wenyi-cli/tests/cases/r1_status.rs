use super::{arguments, binary, fixture_root, support};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
};
use support::{
    fixtures::{FixtureWorkspace, admit_fixture},
    snapshot::snapshot_tree,
};
use wenyi_app::{
    Application, StatusRequest,
    config::{AssetReference, BilingualOrder},
};

fn workspace() -> FixtureWorkspace {
    let fixture = admit_fixture(&fixture_root()).unwrap();
    let source = fixture
        .workspace_mappings()
        .find(|(_, path)| *path == "source/book.epub");
    assert!(
        source.is_some(),
        "{} has no source mapping",
        fixture.fixture_id()
    );
    FixtureWorkspace::new(&fixture).unwrap()
}
fn run_dir(workspace: &FixtureWorkspace) -> PathBuf {
    workspace.root().join("state/Harbor_Notes")
}
fn request(workspace: &FixtureWorkspace) -> StatusRequest {
    StatusRequest {
        input: workspace.root().join("source/book.epub"),
        config_path: workspace.root().join("config.yaml"),
        state_root: workspace.root().join("state"),
    }
}
fn mutate(path: &Path, operation: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    operation(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}
fn status(workspace: &FixtureWorkspace) -> support::process::ProcessResult {
    binary(
        &arguments(&["status", "source/book.epub"]),
        workspace.root(),
    )
}
fn text(result: &support::process::ProcessResult) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    )
}

#[test]
fn current_paid_epub_uses_actual_source_and_preserves_all_files() {
    let workspace = workspace();
    mutate(&run_dir(&workspace).join("manifest.json"), |value| {
        value["source_path"] = json!("/never-open-historical-source.epub")
    });
    let before = snapshot_tree(workspace.root());
    let expected: Value = serde_json::from_slice(
        &fs::read(fixture_root().join("state/structured-epub-balanced-v1/expected.json")).unwrap(),
    )
    .unwrap();
    let report = Application::status(&request(&workspace)).unwrap();
    let paid_manifest: Value = serde_json::from_slice(
        &fs::read(
            fixture_root()
                .join("state/structured-epub-balanced-v1/run/state/Harbor_Notes/manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        report.exit_code(),
        0,
        "{:?}",
        report.observation.diagnostics
    );
    assert_eq!(report.source.identity.title, "Harbor Notes");
    assert_eq!(report.source.segments.len(), 17);
    assert_eq!(report.observation.chapters.len(), 2);
    for (chapter, saved) in report
        .observation
        .chapters
        .iter()
        .zip(expected["run"]["manifest"]["chapters"].as_array().unwrap())
    {
        assert_eq!(
            json!(chapter.title_translated),
            paid_manifest["chapters"][chapter.index]["title_translated"]
        );
        for (segment, paid) in chapter
            .segments
            .iter()
            .zip(saved["segments"].as_array().unwrap())
        {
            assert_eq!(json!(segment.target), paid["target"]);
            assert_eq!(
                serde_json::to_value(&segment.rich_target).unwrap(),
                paid["rich_target"]
            );
        }
    }
    let usage: Value =
        serde_json::from_slice(&fs::read(run_dir(&workspace).join("usage.json")).unwrap()).unwrap();
    assert_eq!(usage["totals"], expected["run"]["usage"]["totals"]);
    assert_eq!(usage["by_model"], expected["run"]["usage"]["by_model"]);
    let counts = report.observation.glossary.unwrap();
    assert_eq!(
        counts.terms,
        expected["run"]["sqlite"]["rows"]["glossary"]
            .as_array()
            .unwrap()
            .len() as u64
    );
    assert_eq!(counts.open_conflicts, 0);
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(0), "{}", text(&result));
    assert!(text(&result).contains("Harbor Notes") && text(&result).contains("Chapter 1"));
    assert_eq!(snapshot_tree(workspace.root()), before);
}

#[test]
fn config_failure_precedes_missing_source_and_global_path_works_after_command() {
    let workspace = tempfile::tempdir().unwrap();
    let path = workspace.path().join("broken config.yaml");
    fs::write(&path, "quality: [broken").unwrap();
    let before = snapshot_tree(workspace.path());
    let result = binary(
        &[
            OsString::from("status"),
            OsString::from("missing.epub"),
            OsString::from("-c"),
            path.clone().into_os_string(),
        ],
        workspace.path(),
    );
    assert_eq!(result.status.code(), Some(2));
    assert!(text(&result).contains("broken config.yaml"));
    assert_eq!(snapshot_tree(workspace.path()), before);
    let result = binary(
        &arguments(&["-c", "missing.yaml", "status", "missing.epub"]),
        workspace.path(),
    );
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(snapshot_tree(workspace.path()), before);
}

#[test]
fn init_creates_parents_refuses_overwrite_and_force_replaces_unparsed_config() {
    let workspace = tempfile::tempdir().unwrap();
    fs::write(workspace.path().join("keep.txt"), b"untouched").unwrap();
    let args = arguments(&["init", "--config", "nested/default.yaml"]);
    assert_eq!(binary(&args, workspace.path()).status.code(), Some(0));
    let path = workspace.path().join("nested/default.yaml");
    wenyi_app::config::load_config(&path).unwrap();
    fs::write(&path, b"not: [valid").unwrap();
    let before = snapshot_tree(workspace.path());
    assert_eq!(binary(&args, workspace.path()).status.code(), Some(1));
    assert_eq!(snapshot_tree(workspace.path()), before);
    assert_eq!(
        binary(
            &arguments(&["-c", "nested/default.yaml", "init", "--force"]),
            workspace.path()
        )
        .status
        .code(),
        Some(0)
    );
    wenyi_app::config::load_config(&path).unwrap();
    assert_eq!(
        fs::read(workspace.path().join("keep.txt")).unwrap(),
        b"untouched"
    );
    assert!(!workspace.path().join("state").exists());
}

#[test]
fn missing_run_does_not_create_state_and_current_pending_is_not_recovery() {
    let workspace = workspace();
    fs::remove_dir_all(workspace.root().join("state")).unwrap();
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(1));
    assert!(text(&result).contains("MissingRun"));
    assert_eq!(snapshot_tree(workspace.root()), before);
    let workspace = self::workspace();
    mutate(&run_dir(&workspace).join("manifest.json"), |value| {
        value["progress"]["0"]["status"] = json!("pending")
    });
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(0), "{}", text(&result));
    assert!(text(&result).contains("Pending"));
    assert_eq!(snapshot_tree(workspace.root()), before);
}

#[test]
fn active_writer_and_nonempty_wal_are_diagnosed_without_recovery() {
    let workspace = workspace();
    let lock_path = run_dir(&workspace).join(".run.lock");
    fs::write(&lock_path, b"writer").unwrap();
    let held = File::options()
        .read(true)
        .write(true)
        .open(&lock_path)
        .unwrap();
    held.lock().unwrap();
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(1));
    assert!(text(&result).contains("active_run_lock"));
    assert!(!text(&result).contains("stranded"));
    assert_eq!(snapshot_tree(workspace.root()), before);
    drop(held);
    fs::write(
        run_dir(&workspace).join("glossary.db-wal"),
        b"uncheckpointed",
    )
    .unwrap();
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(1));
    assert!(text(&result).contains("nonempty_glossary_wal"));
    assert!(text(&result).contains("Glossary: unavailable"));
    assert_eq!(snapshot_tree(workspace.root()), before);
}

#[test]
fn forged_source_locator_is_not_trusted_even_when_stored_hashes_are_valid() {
    let workspace = workspace();
    let path = run_dir(&workspace).join("chapters_v2/ch0.json");
    mutate(&path, |value| {
        value["segments"][0]["epub_state"]["block_path"] = json!([1, 1])
    });
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(1));
    assert!(
        text(&result).contains("source_binding_mismatch"),
        "{}",
        text(&result)
    );
    assert_eq!(snapshot_tree(workspace.root()), before);
}

#[test]
fn saved_output_uses_field_origins_and_explicit_config_null_clears_theme() {
    let workspace = workspace();
    let origin = workspace.root().join("asset-origin");
    mutate(
        &run_dir(&workspace).join("output_selection.json"),
        |value| {
            value["selection"]["mono"] = json!(false);
            value["selection"]["bilingual"]["enabled"] = json!(false);
            value["selection"]["bilingual"]["order"] = json!("source_first");
            value["selection"]["override_theme"] = json!({"styles":origin.join("never-open.css")});
            value["origins"]["override_theme.styles"] = json!(origin);
        },
    );
    fs::write(workspace.root().join("config.yaml"), b"quality: balanced\n").unwrap();
    let before = snapshot_tree(workspace.root());
    let report = Application::status(&request(&workspace)).unwrap();
    assert_eq!(
        report.exit_code(),
        0,
        "{:?}",
        report.observation.diagnostics
    );
    let output = report.output.unwrap();
    assert!(output.normalized && output.mono && !output.bilingual);
    assert_eq!(output.bilingual_order, BilingualOrder::SourceFirst);
    assert_eq!(
        output.override_theme,
        Some(AssetReference::File(origin.join("never-open.css")))
    );
    assert_eq!(output.origins["override_theme.styles"], origin);
    assert!(!origin.exists());
    assert_eq!(snapshot_tree(workspace.root()), before);
    fs::write(
        workspace.root().join("config.yaml"),
        b"output:\n  override_theme: null\n  mono: true\n",
    )
    .unwrap();
    let before = snapshot_tree(workspace.root());
    let report = Application::status(&request(&workspace)).unwrap();
    assert_eq!(report.output.unwrap().override_theme, None);
    assert_eq!(snapshot_tree(workspace.root()), before);
}

#[test]
fn completed_chapter_with_lost_segments_is_invalid_without_repair() {
    let workspace = workspace();
    mutate(&run_dir(&workspace).join("chapters_v2/ch0.json"), |value| {
        value["segments"] = json!([])
    });
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(1));
    assert!(
        text(&result).contains("source_binding_mismatch"),
        "{}",
        text(&result)
    );
    assert_eq!(snapshot_tree(workspace.root()), before);
}

#[test]
fn saved_relative_asset_is_invalid_instead_of_inventing_a_working_directory_origin() {
    let workspace = workspace();
    mutate(
        &run_dir(&workspace).join("output_selection.json"),
        |value| {
            value["selection"]["override_theme"] = json!({"styles":"unbound.css"});
            value["origins"] = json!({});
        },
    );
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    assert_eq!(result.status.code(), Some(1));
    assert!(text(&result).contains("invalid_output_selection"));
    assert_eq!(snapshot_tree(workspace.root()), before);
}

#[test]
fn unsafe_malformed_and_empty_spine_inputs_exit_two_without_creating_state() {
    use std::io::Write;
    use zip::{ZipWriter, write::SimpleFileOptions};
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.epub");
    fs::write(&input, b"not an archive").unwrap();
    let before = snapshot_tree(directory.path());
    assert_eq!(
        binary(&arguments(&["status", "source.epub"]), directory.path())
            .status
            .code(),
        Some(2)
    );
    assert_eq!(snapshot_tree(directory.path()), before);
    for unsafe_name in [false, true] {
        let mut zip = ZipWriter::new(File::create(&input).unwrap());
        for (name, bytes) in [
            (
                "META-INF/container.xml",
                "<container><rootfile full-path='book.opf'/></container>",
            ),
            (
                "book.opf",
                "<package><metadata><title>Empty Spine</title></metadata><manifest><item id='body' href='body.xhtml' media-type='application/xhtml+xml'/></manifest><spine/></package>",
            ),
            (
                "body.xhtml",
                "<html><body><p>Readable but not in the spine.</p></body></html>",
            ),
        ] {
            zip.start_file(name, SimpleFileOptions::default()).unwrap();
            zip.write_all(bytes.as_bytes()).unwrap();
        }
        if unsafe_name {
            zip.start_file("../outside", SimpleFileOptions::default())
                .unwrap();
            zip.write_all(b"never extracted").unwrap();
        }
        zip.finish().unwrap();
        let before = snapshot_tree(directory.path());
        let result = binary(&arguments(&["status", "source.epub"]), directory.path());
        assert_eq!(result.status.code(), Some(2), "{}", text(&result));
        assert_eq!(snapshot_tree(directory.path()), before);
        assert!(text(&result).contains(if unsafe_name {
            "unsafe_entry"
        } else {
            "missing_content_spine"
        }));
    }
}

#[test]
fn recovery_and_migration_diagnoses_do_not_execute_either_operation() {
    for migration in [false, true] {
        let workspace = workspace();
        mutate(&run_dir(&workspace).join("manifest.json"), |value| {
            if migration {
                value["run_state_schema"] = json!(3);
            } else {
                value["nodes"]["prepare"]["status"] = json!("running");
            }
        });
        let before = snapshot_tree(workspace.root());
        let result = status(&workspace);
        assert_eq!(result.status.code(), Some(1), "{}", text(&result));
        assert!(text(&result).contains(if migration {
            "MigrationNeeded"
        } else {
            "RecoveryNeeded"
        }));
        assert_eq!(snapshot_tree(workspace.root()), before);
    }
}

#[cfg(unix)]
#[test]
fn readonly_run_directory_needs_no_new_lock_or_sqlite_sidecar() {
    use std::os::unix::fs::PermissionsExt;
    let workspace = workspace();
    let run = run_dir(&workspace);
    let original = fs::metadata(&run).unwrap().permissions();
    fs::set_permissions(&run, fs::Permissions::from_mode(0o555)).unwrap();
    let before = snapshot_tree(workspace.root());
    let result = status(&workspace);
    let after = snapshot_tree(workspace.root());
    fs::set_permissions(&run, original).unwrap();
    assert_eq!(result.status.code(), Some(0), "{}", text(&result));
    assert_eq!(after, before);
}
