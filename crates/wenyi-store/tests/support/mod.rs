use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};
use tempfile::TempDir;
use wenyi_core::SourceIdentity;
use wenyi_store::{RunObservation, inspect_run};

pub struct Fixture {
    pub temporary: TempDir,
    pub state: PathBuf,
    pub run: PathBuf,
    pub source: SourceIdentity,
}

impl Fixture {
    pub fn new() -> Self {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/migration/state/structured-epub-balanced-v1");
        let source_bytes = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/migration/epub/structured-v1.epub"),
        )
        .unwrap();
        let source = SourceIdentity::new(
            format!("{:x}", Sha256::digest(source_bytes)),
            "Harbor Notes".to_owned(),
            "Harbor_Notes".to_owned(),
        )
        .unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let state = temporary.path().join("state");
        let run = state.join(&source.run_slug);
        copy(&fixture.join("run/state/Harbor_Notes"), &run);
        Self {
            temporary,
            state,
            run,
            source,
        }
    }

    pub fn inspect(&self) -> RunObservation {
        inspect_run(&self.state, &self.source).unwrap()
    }
    pub fn read(&self, name: &str) -> Value {
        serde_json::from_slice(&fs::read(self.run.join(name)).unwrap()).unwrap()
    }
    pub fn write(&self, name: &str, value: &Value) {
        fs::write(
            self.run.join(name),
            serde_json::to_vec_pretty(value).unwrap(),
        )
        .unwrap();
    }
    pub fn modify(&self, name: &str, mutate: impl FnOnce(&mut Value)) {
        let mut value = self.read(name);
        mutate(&mut value);
        self.write(name, &value);
    }
    pub fn snapshot(&self) -> Snapshot {
        snapshot(self.temporary.path())
    }
}

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

pub type Snapshot = BTreeMap<PathBuf, (u64, SystemTime, Option<Vec<u8>>)>;
pub fn snapshot(root: &Path) -> Snapshot {
    fn visit(path: &Path, result: &mut Snapshot) {
        let metadata = fs::metadata(path).unwrap();
        let bytes = metadata.is_file().then(|| fs::read(path).unwrap());
        result.insert(
            path.to_owned(),
            (metadata.len(), metadata.modified().unwrap(), bytes),
        );
        if metadata.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                visit(&entry.unwrap().path(), result);
            }
        }
    }
    let mut result = BTreeMap::new();
    visit(root, &mut result);
    result
}
