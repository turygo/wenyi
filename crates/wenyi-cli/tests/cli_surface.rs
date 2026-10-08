mod support;

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};
use support::fixtures::{FixtureWorkspace, admit_fixture};
use support::process::{ProcessError, ProcessResult, run_wenyi};
use tempfile::tempdir;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/migration")
}

fn binary(args: &[OsString], cwd: &Path) -> ProcessResult {
    run_wenyi(args, cwd, Duration::from_secs(10))
        .unwrap_or_else(|error| panic!("wenyi process failed: {error:?}"))
}

fn arguments(args: &[&str]) -> Vec<OsString> {
    args.iter().map(OsString::from).collect()
}

#[test]
fn config_without_an_implemented_action_is_not_successful() {
    let workspace = tempdir().expect("isolated native working directory");
    for option in ["--config", "-c"] {
        let result = binary(&arguments(&[option, "missing file.yaml"]), workspace.path());
        assert_eq!(result.status.code(), Some(2));
        let help = format!(
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(help.contains("--help") && help.contains("--config"));
        assert!(!workspace.path().join("missing file.yaml").exists());
    }
}

#[test]
fn actual_binary_obeys_the_r0_argument_surface_without_filesystem_effects() {
    let root = fixture_root();
    let fixture = admit_fixture(&root).expect("qualified migration fixture must be admitted");

    assert_eq!(fixture.fixture_id(), "structured-epub-balanced-v1");
    assert!(
        fixture
            .workspace_mappings()
            .any(|(_, destination)| destination == "source/book.epub")
    );
    let workspace = FixtureWorkspace::new(&fixture).expect("fixture workspace must be isolated");
    let fixture_before = snapshot_tree(&root);
    let workspace_before = snapshot_tree(workspace.root());
    let scratch = tempdir().expect("scratch directory");
    let missing_config = scratch.path().join("missing config.yaml");
    let malformed_config = scratch.path().join("malformed config.yaml");
    fs::write(&malformed_config, b"not: [valid yaml").expect("write malformed config");
    let spaced_config = scratch.path().join("path with spaces/config.yaml");

    let help = binary(&arguments(&["--help"]), workspace.root());
    assert_eq!(help.status.code(), Some(0));
    let help_text = String::from_utf8_lossy(&help.stdout);
    assert!(help_text.contains("wenyi"));
    assert!(help_text.contains("--help"));
    assert!(help_text.contains("--version"));
    assert!(help_text.contains("--config"));
    assert!(help_text.contains("-c"));

    let version = binary(&arguments(&["--version"]), workspace.root());
    assert_eq!(version.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&version.stdout).contains("wenyi 1.3.3"));

    let no_args = binary(&[], workspace.root());
    assert_eq!(no_args.status.code(), Some(2));
    let root_help = format!(
        "{}{}",
        String::from_utf8_lossy(&no_args.stdout),
        String::from_utf8_lossy(&no_args.stderr)
    );
    assert!(root_help.contains("--help") && root_help.contains("--config"));

    for config in [&missing_config, &malformed_config] {
        let result = run_wenyi(
            &[
                OsString::from("--config"),
                config.as_os_str().to_owned(),
                OsString::from("--help"),
            ],
            workspace.root(),
            Duration::from_secs(10),
        )
        .unwrap_or_else(|error| panic!("config/help invocation failed: {error:?}"));
        assert_eq!(result.status.code(), Some(0));
    }

    let path_with_spaces = run_wenyi(
        &[
            OsString::from("-c"),
            spaced_config.into_os_string(),
            OsString::from("--version"),
        ],
        workspace.root(),
        Duration::from_secs(10),
    )
    .unwrap_or_else(|error| panic!("path-with-spaces invocation failed: {error:?}"));
    assert_eq!(path_with_spaces.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&path_with_spaces.stdout).contains("wenyi 1.3.3"));

    for option in ["--config", "-c"] {
        let result = binary(&arguments(&[option]), workspace.root());
        assert_eq!(result.status.code(), Some(2));
        assert!(!result.stderr.is_empty());
    }

    for unsupported in ["--unknown-option", "translate"] {
        let result = binary(&arguments(&[unsupported]), workspace.root());
        assert_eq!(result.status.code(), Some(2));
        assert!(!result.stderr.is_empty());
    }

    assert_eq!(snapshot_tree(&root), fixture_before);
    assert_eq!(snapshot_tree(workspace.root()), workspace_before);
}

#[test]
fn zero_deadline_kills_and_reaps_the_actual_binary() {
    let fixture =
        admit_fixture(&fixture_root()).expect("qualified migration fixture must be admitted");
    let workspace = FixtureWorkspace::new(&fixture).expect("fixture workspace must be isolated");
    let fixture_before = snapshot_tree(&fixture_root());
    let workspace_before = snapshot_tree(workspace.root());
    let result = run_wenyi(
        &[OsString::from("--help")],
        workspace.root(),
        Duration::ZERO,
    );
    match result {
        Err(ProcessError::Timeout {
            child_pid,
            reaped,
            cleanup_error,
        }) => {
            assert!(child_pid > 0);
            assert!(reaped, "the owned Child::wait outcome must confirm reaping");
            assert!(cleanup_error.is_none());
        }
        other => panic!("zero deadline must time out after spawn, got {other:?}"),
    }
    assert_eq!(snapshot_tree(&fixture_root()), fixture_before);
    assert_eq!(snapshot_tree(workspace.root()), workspace_before);
}

#[derive(Debug, Eq, PartialEq)]
struct SnapshotEntry {
    is_directory: bool,
    size: u64,
    sha256: String,
}

fn snapshot_tree(root: &Path) -> BTreeMap<PathBuf, SnapshotEntry> {
    fn visit(root: &Path, directory: &Path, result: &mut BTreeMap<PathBuf, SnapshotEntry>) {
        let mut entries: Vec<_> = fs::read_dir(directory)
            .expect("read snapshot directory")
            .map(|entry| entry.expect("read directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            let metadata = fs::symlink_metadata(&path).expect("read snapshot metadata");
            assert!(
                !metadata.file_type().is_symlink(),
                "snapshot must not follow symlinks"
            );
            let relative = path
                .strip_prefix(root)
                .expect("path is below snapshot root")
                .to_path_buf();
            if metadata.is_dir() {
                result.insert(
                    relative,
                    SnapshotEntry {
                        is_directory: true,
                        size: 0,
                        sha256: String::new(),
                    },
                );
                visit(root, &path, result);
            } else {
                let mut file = File::open(&path).expect("open snapshot file");
                let mut hasher = Sha256::new();
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let count = file.read(&mut buffer).expect("read snapshot file");
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }
                result.insert(
                    relative,
                    SnapshotEntry {
                        is_directory: false,
                        size: metadata.len(),
                        sha256: format!("{:x}", hasher.finalize()),
                    },
                );
            }
        }
    }

    let mut result = BTreeMap::new();
    visit(root, root, &mut result);
    result
}
