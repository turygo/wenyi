#![deny(unsafe_code)]

pub mod archive;
pub mod css;
pub mod document;
pub mod error;
pub mod ingest;
pub mod package;
pub mod references;
pub mod source;
mod style_projection;

pub use error::{BookError, BookErrorKind};
pub use ingest::admit_epub;
pub use source::{
    Attribute, BomKind, ConversionPolicy, DocumentEnvelope, ElementData, NamespaceDecl,
    NativeVersions, NodeId, NodeKind, ParseDiagnostic, ParseMode, ParsedDocument, QualifiedName,
    SourceNode, SourceTree, XmlPurpose, convert_encoding, native_versions, parse_source_markup,
    parse_xml, serialize_block, serialize_source,
};
