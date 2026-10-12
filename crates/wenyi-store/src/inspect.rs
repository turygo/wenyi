use crate::{
    StoreError, StoreErrorKind, glossary,
    inventory::{self, Inventory, ObservationLock},
    observation::*,
    recovery, wire,
};
use serde_json::Value;
use std::path::{Component, Path};
use wenyi_core::SourceIdentity;

pub fn inspect_run(
    state_root: &Path,
    source: &SourceIdentity,
) -> Result<RunObservation, StoreError> {
    inspect_with_hook(state_root, source, || {})
}

fn inspect_with_hook(
    state_root: &Path,
    source: &SourceIdentity,
    after_inventory: impl FnOnce(),
) -> Result<RunObservation, StoreError> {
    validate_source(source, state_root)?;
    let run = state_root.join(&source.run_slug);
    let mut observation = RunObservation::empty(run.clone());
    if !inventory::exists(&run)? {
        observation.diagnose(
            ObservationOutcome::MissingRun,
            "missing_run",
            run,
            "source-bound run does not exist",
        );
        return Ok(observation);
    }
    let lock = match inventory::lock(&run) {
        Ok(lock) => lock,
        Err(error) => {
            record(&mut observation, error);
            return Ok(observation);
        }
    };
    if matches!(lock, ObservationLock::Contended) {
        observation.diagnose(
            ObservationOutcome::ObservationUnavailable,
            "active_run_lock",
            run.join(".run.lock"),
            "exclusive writer holds the run lock",
        );
        return Ok(observation);
    }
    let inventory = match Inventory::capture(&run) {
        Ok(inventory) => inventory,
        Err(error) => {
            observation.diagnose(
                ObservationOutcome::ObservationUnavailable,
                error.code,
                error.artifact,
                error.detail,
            );
            return Ok(observation);
        }
    };
    let lock_path = run.join(".run.lock");
    let consistent = match &lock {
        ObservationLock::Absent => Ok(inventory.size(&lock_path).is_none()),
        ObservationLock::Held(file) => inventory.matches_file(&lock_path, file),
        ObservationLock::Contended => unreachable!(),
    };
    match consistent {
        Ok(true) => {}
        Ok(false) => {
            observation.diagnose(
                ObservationOutcome::ObservationUnavailable,
                "changing_files",
                lock_path,
                "lock membership or identity changed before inventory admission",
            );
            return Ok(observation);
        }
        Err(error) => {
            record(&mut observation, error);
            return Ok(observation);
        }
    }
    after_inventory();
    inspect_artifacts(&inventory, source, &mut observation);
    match inventory.unchanged(&run) {
        Ok(true) => {}
        result => {
            observation.discard_data();
            observation.diagnostics.clear();
            observation.outcome = ObservationOutcome::Current;
            let detail = match result {
                Ok(false) => "relevant membership, identity, content, size or mtime changed during observation".to_owned(),
                Err(error) => error.to_string(),
                Ok(true) => unreachable!(),
            };
            observation.diagnose(
                ObservationOutcome::ObservationUnavailable,
                "changing_files",
                run,
                detail,
            );
        }
    }
    // 文件描述符持有到稳定性比较完成，释放时不改写锁文件。
    if let ObservationLock::Held(file) = lock {
        drop(file);
    }
    Ok(observation)
}

fn validate_source(source: &SourceIdentity, path: &Path) -> Result<(), StoreError> {
    let mut components = Path::new(&source.run_slug).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(StoreError::new(
            StoreErrorKind::SourceMismatch,
            "invalid_run_slug",
            path,
            "run slug must select exactly one child directory",
        ));
    }
    wire::sha256(&source.source_bytes_sha256, path)?;
    if source.title.is_empty() {
        return Err(StoreError::new(
            StoreErrorKind::SourceMismatch,
            "invalid_source_identity",
            path,
            "source title must not be empty",
        ));
    }
    if source.input_schema_version != 2 || source.source_slot_schema_version != 4 {
        return Err(wire::version_error(
            path,
            "unsupported_source_identity",
            "source identity requires input2/source4",
        ));
    }
    Ok(())
}

fn inspect_artifacts(
    inventory: &Inventory,
    source: &SourceIdentity,
    observation: &mut RunObservation,
) {
    let run = observation.run_dir.clone();
    let manifest_path = run.join("manifest.json");
    let manifest = inventory
        .get(&manifest_path)
        .ok_or_else(|| StoreError::wire(&manifest_path, "run lacks manifest"))
        .and_then(|bytes| wire::json(bytes, &manifest_path));
    let mut admitted = false;
    match manifest {
        Ok(mut value) => {
            let schema = bookkeeping_version(&value, inventory, &run);
            if schema.is_some_and(|version| (1..4).contains(&version)) {
                observation.diagnose(
                    ObservationOutcome::MigrationNeeded,
                    "historical_bookkeeping",
                    manifest_path.clone(),
                    "recognized bookkeeping layout requires migration",
                );
                if value.get("run_state_schema").is_none() {
                    value["run_state_schema"] = Value::from(1);
                }
            }
            match wire::manifest(&value, source, &manifest_path) {
                Ok(manifest) => {
                    admitted = admit_manifest(manifest, inventory, observation);
                }
                Err(error) => record(observation, error),
            }
        }
        Err(error) => record(observation, error),
    }
    inspect_glossary(inventory, observation);
    let selection_path = run.join("output_selection.json");
    if let Some(bytes) = inventory.get(&selection_path) {
        match wire::json(bytes, &selection_path)
            .and_then(|value| wire::selection(&value, &source.source_bytes_sha256, &selection_path))
        {
            Ok(selection) => observation.output_selection = selection,
            Err(error) => record(observation, error),
        }
    }
    inspect_recovery(inventory, source, admitted, observation);
}

fn bookkeeping_version(value: &Value, inventory: &Inventory, run: &Path) -> Option<u64> {
    if let Some(version) = value.get("run_state_schema") {
        return version.as_u64();
    }
    let legacy = value
        .get("chapters")
        .and_then(Value::as_array)
        .filter(|chapters| !chapters.is_empty())
        .is_some_and(|chapters| {
            value.get("progress").is_none()
                && value.get("nodes").is_none()
                && chapters.iter().all(|chapter| {
                    chapter.get("status").is_some()
                        && chapter
                            .get("index")
                            .and_then(Value::as_u64)
                            .is_some_and(|index| {
                                inventory
                                    .get(&run.join(format!("chapters/ch{index}.json")))
                                    .is_some()
                                    && inventory
                                        .get(&run.join(format!("chapters_v2/ch{index}.json")))
                                        .is_none()
                            })
                })
        });
    Some(if legacy { 1 } else { 4 })
}

fn admit_manifest(
    mut manifest: wire::Manifest,
    inventory: &Inventory,
    observation: &mut RunObservation,
) -> bool {
    let run = observation.run_dir.clone();
    observation.title = Some(manifest.title);
    observation.title_translated = manifest.title_translated;
    observation.source_lang = Some(manifest.source_lang);
    observation.target_lang = Some(manifest.target_lang);
    for (key, node) in manifest.nodes {
        if node.status == NodeStatus::Running {
            observation.diagnose(
                ObservationOutcome::RecoveryNeeded,
                "stranded_node",
                run.join("manifest.json"),
                format!("node {key} remains running without an exclusive writer"),
            );
        }
        observation.nodes.push(NodeObservation {
            key,
            node_id: node.node_id,
            status: node.status,
            attempts: node.attempts,
            input_fingerprint: node.input_fingerprint,
            failure: node.failure,
        });
    }
    let mut complete = true;
    for index in manifest.chapters {
        let directory = if manifest.run_state_schema == 1 {
            "chapters"
        } else {
            "chapters_v2"
        };
        let path = run.join(directory).join(format!("ch{}.json", index.index));
        let progress = manifest
            .progress
            .remove(&index.index.to_string())
            .unwrap_or_default();
        let result = wire::processing_progress(progress, index.index, &run.join("manifest.json"))
            .and_then(|mut progress| {
                if manifest.run_state_schema == 1 {
                    progress.status = index
                        .status
                        .as_ref()
                        .map(|status| wire::decode(status, &path))
                        .transpose()?
                        .unwrap_or_default();
                }
                let bytes = inventory
                    .get(&path)
                    .ok_or_else(|| StoreError::wire(&path, "listed chapter file is missing"))?;
                let value = wire::json(bytes, &path)?;
                wire::chapter(&value, &index, progress, &path)
            });
        match result {
            Ok(chapter) => {
                if chapter
                    .progress
                    .repair_ledger
                    .iter()
                    .any(|issue| issue.status == RepairStatus::Repairing)
                {
                    observation.diagnose(
                        ObservationOutcome::RecoveryNeeded,
                        "stranded_repair",
                        run.join("manifest.json"),
                        "repair ledger retains an active issue without an exclusive writer",
                    );
                }
                observation.chapters.push(chapter);
            }
            Err(error) => {
                record(observation, error);
                complete = false;
            }
        }
    }
    complete
}

fn inspect_glossary(inventory: &Inventory, observation: &mut RunObservation) {
    let run = observation.run_dir.clone();
    let path = run.join("glossary.db");
    let wal_path = run.join("glossary.db-wal");
    let journal_path = run.join("glossary.db-journal");
    let wal = inventory.size(&wal_path).is_some_and(|size| size != 0);
    let hot = inventory.get(&journal_path).is_some_and(|header| {
        glossary::hot_journal(header, inventory.size(&journal_path).unwrap_or(0))
    });
    if wal {
        observation.diagnose(
            ObservationOutcome::ObservationUnavailable,
            "nonempty_glossary_wal",
            wal_path,
            "immutable SQLite cannot expose uncheckpointed WAL counts",
        );
    }
    if hot {
        observation.diagnose(
            ObservationOutcome::RecoveryNeeded,
            "hot_glossary_journal",
            journal_path,
            "rollback journal requires recovery before querying",
        );
    }
    if inventory.get(&path).is_none() {
        observation.diagnose(
            ObservationOutcome::ObservationUnavailable,
            "missing_glossary",
            path,
            "glossary database is missing",
        );
    } else if !wal && !hot {
        match glossary::inspect(&path) {
            Ok(counts) => observation.glossary = Some(counts),
            Err(error) => record(observation, error),
        }
    }
}

fn inspect_recovery(
    inventory: &Inventory,
    source: &SourceIdentity,
    admitted: bool,
    observation: &mut RunObservation,
) {
    let run = observation.run_dir.clone();
    let usage_path = run.join("usage.json");
    if let Some(bytes) = inventory.get(&usage_path)
        && let Err(error) = wire::json(bytes, &usage_path)
            .and_then(|value| recovery::usage_snapshot(&value, &usage_path))
    {
        record(observation, error);
    }
    for name in ["journal.json", "note_migration.json", "usage.wal.json"] {
        let path = run.join(name);
        let Some(bytes) = inventory.get(&path) else {
            continue;
        };
        let result = wire::json(bytes, &path).and_then(|value| {
            match name {
                "journal.json" if admitted => {
                    recovery::checkpoint(&value, &observation.chapters, &path)
                }
                "journal.json" => {
                    recovery::checkpoint_shape(&value, &path)?;
                    return Ok(false);
                }
                "note_migration.json" => recovery::note(&value, inventory, &run, source, &path),
                _ => recovery::usage_wal(&value, inventory.get(&usage_path), &path),
            }
            .map(|()| true)
        });
        match result {
            Ok(true) => observation.diagnose(
                ObservationOutcome::RecoveryNeeded,
                "recovery_record",
                path,
                "valid durable transaction requires recovery",
            ),
            Ok(false) => {}
            Err(error) => record(observation, error),
        }
    }
}

fn record(observation: &mut RunObservation, error: StoreError) {
    let outcome = match error.kind {
        StoreErrorKind::Io | StoreErrorKind::Unavailable => {
            ObservationOutcome::ObservationUnavailable
        }
        _ => ObservationOutcome::InvalidState,
    };
    observation.diagnose(outcome, error.code, error.artifact, error.detail);
}

#[cfg(test)]
mod tests;
