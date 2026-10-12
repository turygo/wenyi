use std::{
    fmt,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreErrorKind {
    Io,
    Wire,
    Version,
    SourceMismatch,
    Sqlite,
    Recovery,
    Migration,
    Unavailable,
}

#[derive(Debug)]
pub struct StoreError {
    pub kind: StoreErrorKind,
    pub code: &'static str,
    pub artifact: PathBuf,
    pub detail: String,
}

impl StoreError {
    pub(crate) fn new(
        kind: StoreErrorKind,
        code: &'static str,
        artifact: &Path,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            code,
            artifact: artifact.to_owned(),
            detail: detail.into(),
        }
    }
    pub(crate) fn wire(path: &Path, detail: impl Into<String>) -> Self {
        Self::new(StoreErrorKind::Wire, "invalid_wire", path, detail)
    }
    pub(crate) fn io(path: &Path, error: std::io::Error) -> Self {
        Self::new(StoreErrorKind::Io, "artifact_io", path, error.to_string())
    }
}
impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}: {}",
            self.code,
            self.artifact.display(),
            self.detail
        )
    }
}
impl std::error::Error for StoreError {}
