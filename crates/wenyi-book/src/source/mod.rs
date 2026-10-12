#![deny(unsafe_code)]
mod input;
mod serialize;
mod tree;
mod xml_ffi;

use crate::error::{BookError, BookErrorKind};
use serde::Serialize;
pub use serialize::{serialize_block, serialize_source};
pub use tree::{
    Attribute, ElementData, NamespaceDecl, NodeId, NodeKind, QualifiedName, SourceNode, SourceTree,
};

pub(crate) const MAX_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const MAX_DEPTH: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XmlPurpose {
    Container,
    Package,
    Ncx,
    Nav,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ParseMode {
    #[serde(rename = "xml")]
    Xml,
    #[serde(rename = "recovered")]
    RecoveredHtml,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionPolicy {
    Strict,
    Replace,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BomKind {
    Utf8,
    Utf16Le,
    Utf16Be,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentEnvelope {
    pub mode: ParseMode,
    pub encoding: String,
    pub bom: Option<BomKind>,
    pub declaration: bool,
    pub doctype: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDiagnostic {
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub reason: String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDocument {
    pub tree: SourceTree,
    pub envelope: DocumentEnvelope,
    pub diagnostics: Vec<ParseDiagnostic>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeVersions {
    pub libxml2: String,
    pub expat: String,
    pub icu: String,
}

pub fn parse_xml(bytes: &[u8], purpose: XmlPurpose) -> Result<ParsedDocument, BookError> {
    if purpose == XmlPurpose::Nav {
        let envelope = DocumentEnvelope {
            mode: ParseMode::RecoveredHtml,
            encoding: "UTF-8".into(),
            bom: None,
            declaration: bytes.starts_with(b"<?xml"),
            doctype: None,
        };
        let mut document = xml_ffi::parse(bytes, false, true, envelope)?;
        recovered(&mut document)?;
        return Ok(document);
    }
    let envelope = input::source_envelope(bytes)?;
    xml_ffi::parse(bytes, true, false, envelope)
}
pub fn parse_source_markup(bytes: &[u8]) -> Result<ParsedDocument, BookError> {
    let envelope = input::source_envelope(bytes)?;
    match xml_ffi::parse(bytes, false, false, envelope.clone()) {
        Ok(document) => Ok(document),
        Err(error) if error.kind == BookErrorKind::XmlSyntax => {
            let mut document = xml_ffi::parse(bytes, false, true, envelope)?;
            document.diagnostics.truncate(19);
            document.diagnostics.insert(
                0,
                ParseDiagnostic {
                    line: None,
                    column: None,
                    reason: error.detail,
                },
            );
            recovered(&mut document)?;
            Ok(document)
        }
        Err(error) => Err(error),
    }
}
fn recovered(document: &mut ParsedDocument) -> Result<(), BookError> {
    if !document.tree.nodes.iter().any(
        |node| matches!(&node.kind, NodeKind::Element(element) if element.name.local == "body"),
    ) {
        return Err(BookError::new(
            BookErrorKind::XmlSyntax,
            "html-body",
            "recovered HTML has no body",
        ));
    }
    document.envelope.mode = ParseMode::RecoveredHtml;
    document.envelope.encoding = "UTF-8".into();
    document.envelope.bom = None;
    Ok(())
}
pub fn convert_encoding(
    bytes: &[u8],
    from: &str,
    to: &str,
    policy: ConversionPolicy,
) -> Result<Vec<u8>, BookError> {
    xml_ffi::convert(bytes, from, to, policy)
}
pub fn native_versions() -> NativeVersions {
    xml_ffi::versions()
}

#[cfg(test)]
mod tests;
