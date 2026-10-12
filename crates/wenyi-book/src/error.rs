#![forbid(unsafe_code)]
use std::{error::Error, fmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookErrorKind {
    Io,
    Archive,
    Package,
    Reference,
    XmlSyntax,
    XmlSafety,
    Encoding,
    Source,
    Css,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookError {
    pub kind: BookErrorKind,
    pub code: &'static str,
    pub resource: Option<String>,
    pub detail: String,
}
impl BookError {
    pub fn new(kind: BookErrorKind, code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            resource: None,
            detail: detail.into(),
        }
    }
    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }
}
impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code)?;
        if let Some(resource) = &self.resource {
            write!(f, " [{resource}]")?;
        }
        write!(f, ": {}", self.detail)
    }
}
impl Error for BookError {}
