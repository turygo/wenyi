#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use wenyi_core::{RichSource, SourceIdentity, SourceSlot};

use crate::{archive::ArchiveInventory, package::PackageModel, source::ParsedDocument};

#[derive(Debug)]
pub struct SourceDocument {
    pub identity: SourceIdentity,
    pub inventory: ArchiveInventory,
    pub package: PackageModel,
    pub resources: Vec<ContentResource>,
    pub segments: Vec<SourceSegment>,
    pub chapters: Vec<SourceChapter>,
    pub navigation: Vec<NavigationEntry>,
    pub landmarks: Vec<Landmark>,
    pub notes: NoteRelations,
    pub diagnostics: Vec<AdmissionDiagnostic>,
}

#[derive(Debug)]
pub struct ContentResource {
    pub href: String,
    pub resource_sha256: String,
    pub original_bytes: Vec<u8>,
    pub parsed: ParsedDocument,
    pub title: String,
    pub segment_ids: Vec<usize>,
    pub fragment_anchors: BTreeMap<String, Option<usize>>,
}

#[derive(Debug)]
pub struct SourceSegment {
    pub source: String,
    pub kind: SegmentKind,
    pub anchor: String,
    pub resource_index: usize,
    pub block_path: Vec<i32>,
    pub block_fingerprint: String,
    pub slots: Vec<SourceSlot>,
    pub slot_contract_sha256: String,
    pub rich_source: RichSource,
    pub styled_source: Option<RichSource>,
    pub semantic_hints: Vec<String>,
    pub mirrored_toc_entry_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Heading,
    Text,
}
impl SegmentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Heading => "heading",
            Self::Text => "text",
        }
    }
}

#[derive(Debug)]
pub struct SourceChapter {
    pub index: usize,
    pub title: String,
    pub href: String,
    pub toc_entry_id: Option<String>,
    pub segment_ids: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct NavigationEntry {
    pub toc_path: String,
    pub entry_id: String,
    pub node_id: Option<String>,
    pub parent_entry_id: Option<String>,
    pub depth: usize,
    pub label: String,
    pub raw_href: String,
    pub resource_href: String,
    pub fragment: String,
    pub external: bool,
    pub boundary_segment: Option<usize>,
}

#[derive(Debug)]
pub struct Landmark {
    pub resource_href: String,
    pub semantic_type: String,
    pub role: String,
    pub label: String,
}

#[derive(Debug, Default)]
pub struct NoteRelations {
    pub markers: Vec<NoteMarker>,
    pub targets: Vec<NoteTarget>,
}
#[derive(Debug)]
pub struct NoteMarker {
    pub resource_index: usize,
    pub path: Vec<i32>,
    pub kind: wenyi_core::NoteKind,
    pub label: String,
    pub target_resource_index: usize,
    pub target_path: Vec<i32>,
}
#[derive(Debug)]
pub struct NoteTarget {
    pub resource_index: usize,
    pub path: Vec<i32>,
    pub kind: NoteBodyKind,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteBodyKind {
    Footnote,
    Endnote,
}

#[derive(Debug)]
pub struct AdmissionDiagnostic {
    pub code: String,
    pub resource: String,
    pub detail: String,
}
