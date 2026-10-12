#![forbid(unsafe_code)]

use wenyi_book::{
    ParseMode,
    document::{SourceDocument, SourceSegment},
};
use wenyi_core::{RichSource, SourceSlot};
use wenyi_store::{ChapterStatus, ObservedSegment, RunObservation};

pub(super) fn validate(source: &SourceDocument, observed: &RunObservation) -> Result<(), String> {
    if source.chapters.len() != observed.chapters.len() {
        return Err("chapter inventory differs from the actual EPUB source".into());
    }
    for (fresh, saved) in source.chapters.iter().zip(&observed.chapters) {
        if fresh.index != saved.index
            || fresh.title != saved.title
            || saved.href.as_ref().is_some_and(|href| *href != fresh.href)
            || saved
                .toc_entry_id
                .as_ref()
                .is_some_and(|id| Some(id) != fresh.toc_entry_id.as_ref())
        {
            return Err(format!(
                "chapter {} identity differs from the actual source",
                saved.index
            ));
        }
        if saved.segments.is_empty() && saved.progress.status != ChapterStatus::Done {
            continue;
        }
        if saved.segments.len() != fresh.segment_ids.len() {
            return Err(format!("chapter {} segment inventory differs", saved.index));
        }
        for (index, (&segment_id, segment)) in
            fresh.segment_ids.iter().zip(&saved.segments).enumerate()
        {
            if segment.index != index
                || !matches_source(source, &source.segments[segment_id], segment)
            {
                return Err(format!(
                    "chapter {} segment {index} source evidence differs",
                    saved.index
                ));
            }
        }
    }
    Ok(())
}

fn matches_source(
    document: &SourceDocument,
    fresh: &SourceSegment,
    saved: &ObservedSegment,
) -> bool {
    let Some(state) = &saved.epub_state else {
        return false;
    };
    let resource = &document.resources[fresh.resource_index];
    let mode = match resource.parsed.envelope.mode {
        ParseMode::Xml => "xml",
        ParseMode::RecoveredHtml => "recovered",
    };
    saved.source == fresh.source
        && saved.kind == fresh.kind.as_str()
        && saved
            .anchor
            .as_deref()
            .is_none_or(|anchor| anchor == fresh.anchor)
        && saved
            .resource_href
            .as_deref()
            .is_none_or(|href| href == resource.href)
        && state.resource_href == resource.href
        && state.resource_sha256 == resource.resource_sha256
        && state.block_path == fresh.block_path
        && state.block_fingerprint == fresh.block_fingerprint
        && state.parse_mode == mode
        && state.slot_contract_sha256 == fresh.slot_contract_sha256
        && equal_slots(&state.slots, &fresh.slots)
        && state
            .rich_source
            .as_ref()
            .is_none_or(|rich| equal_rich(rich, &fresh.rich_source))
}

fn equal_slots(saved: &[SourceSlot], fresh: &[SourceSlot]) -> bool {
    saved.len() == fresh.len()
        && saved.iter().zip(fresh).all(|(left, right)| {
            left.id == right.id
                && left.element_path == right.element_path
                && left.field == right.field
                && left.source_value == right.source_value
        })
}

fn equal_rich(saved: &RichSource, fresh: &RichSource) -> bool {
    saved.version == fresh.version
        && saved.atoms == fresh.atoms
        && saved.runs == fresh.runs
        && saved.note_references == fresh.note_references
        && saved.marks.len() == fresh.marks.len()
        && saved.marks.iter().zip(&fresh.marks).all(|(left, right)| {
            left.id == right.id
                && left.path == right.path
                && left.tag == right.tag
                && left.attributes == right.attributes
                && left.source_text == right.source_text
                && left.kind == right.kind
                && left.semantics == right.semantics
                && left.required == right.required
        })
}
