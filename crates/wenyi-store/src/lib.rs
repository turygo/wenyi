#![forbid(unsafe_code)]

mod error;
mod glossary;
mod inspect;
mod inventory;
pub mod observation;
mod recovery;
pub mod wire;

pub use error::{StoreError, StoreErrorKind};
pub use inspect::inspect_run;
pub use observation::{
    AdmittedSourceState, ChapterProcessing, ChapterProgress, ChapterStatus, Diagnostic,
    GlossaryCounts, NodeFailure, NodeFailureKind, NodeObservation, NodeStatus, ObservationOutcome,
    ObservedChapter, ObservedSegment, PolishBatch, ProcessingAction, RepairIssue, RepairStatus,
    RunObservation, SavedOutputSelection,
};
