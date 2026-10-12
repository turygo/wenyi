#![forbid(unsafe_code)]

mod binding;
mod source_watch;

use crate::config::{self, ConfigError, LoadedConfig, OutputFields, ResolvedOutput};
use source_watch::SourceWatch;
use std::{error::Error, fmt, path::PathBuf};
use wenyi_book::{BookError, BookErrorKind, admit_epub, document::SourceDocument};
use wenyi_store::{Diagnostic, ObservationOutcome, RunObservation, StoreError, inspect_run};

#[derive(Debug)]
pub struct StatusRequest {
    pub input: PathBuf,
    pub config_path: PathBuf,
    pub state_root: PathBuf,
}

#[derive(Debug)]
pub struct StatusReport {
    pub config: LoadedConfig,
    pub source: SourceDocument,
    pub observation: RunObservation,
    pub output: Option<ResolvedOutput>,
}

impl StatusReport {
    pub fn exit_code(&self) -> i32 {
        i32::from(self.observation.outcome != ObservationOutcome::Current)
    }

    pub fn outcome_label(&self) -> &'static str {
        match self.observation.outcome {
            ObservationOutcome::Current => "Current",
            ObservationOutcome::MissingRun => "MissingRun",
            ObservationOutcome::RecoveryNeeded => "RecoveryNeeded",
            ObservationOutcome::MigrationNeeded => "MigrationNeeded",
            ObservationOutcome::ObservationUnavailable => "ObservationUnavailable",
            ObservationOutcome::InvalidState => "InvalidState",
        }
    }
}

#[derive(Debug)]
pub enum StatusError {
    Config(ConfigError),
    Book(BookError),
    Store(StoreError),
    Io { path: PathBuf, detail: String },
}

impl StatusError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Config(_) => 2,
            Self::Book(error) if error.kind != BookErrorKind::Io => 2,
            _ => 1,
        }
    }
}

impl fmt::Display for StatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Book(error) => error.fmt(formatter),
            Self::Store(error) => error.fmt(formatter),
            Self::Io { path, detail } => write!(formatter, "{}: {detail}", path.display()),
        }
    }
}
impl Error for StatusError {}

pub(crate) fn inspect(request: &StatusRequest) -> Result<StatusReport, StatusError> {
    let config = config::load_config(&request.config_path).map_err(StatusError::Config)?;
    let mut watch = SourceWatch::capture(&request.input)?;
    let source = admit_epub(&request.input).map_err(StatusError::Book)?;
    let mut observation =
        inspect_run(&request.state_root, &source.identity).map_err(StatusError::Store)?;
    if matches!(
        observation.outcome,
        ObservationOutcome::Current | ObservationOutcome::RecoveryNeeded
    ) && let Err(detail) = binding::validate(&source, &observation)
    {
        diagnose(
            &mut observation,
            ObservationOutcome::InvalidState,
            "source_binding_mismatch",
            "manifest.json",
            detail,
        );
    }
    let output = resolve_selection(&config, &mut observation);
    let mut report = StatusReport {
        config,
        source,
        observation,
        output,
    };
    if let Err(detail) = watch.check(&report.source.identity.source_bytes_sha256) {
        report.observation.outcome = ObservationOutcome::ObservationUnavailable;
        report.observation.diagnostics.clear();
        diagnose(
            &mut report.observation,
            ObservationOutcome::ObservationUnavailable,
            "source_observation_changed",
            "manifest.json",
            detail,
        );
        report.observation.title = None;
        report.observation.title_translated = None;
        report.observation.source_lang = None;
        report.observation.target_lang = None;
        report.observation.chapters.clear();
        report.observation.nodes.clear();
        report.observation.glossary = None;
        report.observation.output_selection = None;
        report.output = None;
    }
    Ok(report)
}

fn resolve_selection(
    config: &LoadedConfig,
    observation: &mut RunObservation,
) -> Option<ResolvedOutput> {
    if !matches!(
        observation.outcome,
        ObservationOutcome::Current | ObservationOutcome::MissingRun
    ) {
        return None;
    }
    let saved = match observation
        .output_selection
        .as_ref()
        .map(config::admit_saved_output)
        .transpose()
    {
        Ok(saved) => saved,
        Err(error) => {
            diagnose(
                observation,
                ObservationOutcome::InvalidState,
                "invalid_output_selection",
                "output_selection.json",
                error.to_string(),
            );
            return None;
        }
    };
    Some(config::resolve_output(
        &OutputFields::default(),
        &config.output,
        saved.as_ref(),
    ))
}

fn diagnose(
    observation: &mut RunObservation,
    outcome: ObservationOutcome,
    code: &str,
    artifact: &str,
    detail: String,
) {
    observation.outcome = observation.outcome.max(outcome);
    observation.diagnostics.push(Diagnostic {
        outcome,
        code: code.to_owned(),
        artifact: Some(observation.run_dir.join(artifact)),
        detail,
    });
}
