#![forbid(unsafe_code)]
use super::{rich, source_error, tree};
use crate::{
    document::{ContentResource, NoteRelations, SegmentKind, SourceSegment},
    error::BookError,
    source::{NodeId, SourceTree, serialize_block},
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use wenyi_core::slot_contract_digest;
const BLOCKS: &[&str] = &[
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "td",
    "th",
    "dt",
    "dd",
    "div",
];
const HEADINGS: &[&str] = &["h1", "h2", "h3", "h4", "h5", "h6"];
fn candidate(tree: &SourceTree, id: NodeId, resource: usize, notes: &NoteRelations) -> bool {
    tree.local_name(id).is_some_and(|tag| {
        BLOCKS.contains(&tag)
            || tag == "aside"
                && notes.targets.iter().any(|target| {
                    target.resource_index == resource && target.path == tree::path(tree, id)
                })
    })
}
fn direct_prose(
    tree: &SourceTree,
    id: NodeId,
    root: NodeId,
    resource: usize,
    notes: &NoteRelations,
    output: &mut String,
) {
    if id != root && candidate(tree, id, resource, notes)
        || rich::excluded(tree, id, resource, notes)
    {
        return;
    }
    let node = &tree.nodes[id.index()];
    if let Some(text) = &node.text {
        output.push_str(text);
    }
    for &child in &node.children {
        if tree.local_name(child).is_some() {
            direct_prose(tree, child, root, resource, notes, output);
        }
        if let Some(tail) = &tree.nodes[child.index()].tail {
            output.push_str(tail);
        }
    }
}
fn blocks(
    tree: &SourceTree,
    resource: usize,
    notes: &NoteRelations,
) -> Result<Vec<NodeId>, BookError> {
    let mut result = Vec::new();
    for id in tree::nodes(tree, tree.root) {
        if !candidate(tree, id, resource, notes)
            || tree::ancestor_tag(
                tree,
                id,
                &[
                    "head", "nav", "script", "style", "rt", "rp", "audio", "canvas", "embed",
                    "iframe", "math", "object", "svg", "video",
                ],
            )
        {
            continue;
        }
        let descendants = tree::nodes(tree, id);
        let has_blocks = descendants.iter().skip(1).any(|&child| {
            candidate(tree, child, resource, notes)
                && !rich::visible(tree, child, resource, notes).is_empty()
        });
        if has_blocks && tree.local_name(id) == Some("aside") {
            let mut direct = String::new();
            direct_prose(tree, id, id, resource, notes, &mut direct);
            if !direct.trim().is_empty() {
                return Err(source_error(
                    "mixed_note_aside",
                    "note aside has direct prose alongside child blocks",
                ));
            }
        }
        if tree.local_name(id) == Some("li")
            && let Some(link) = tree.element_children(id).find(|&child| {
                tree.local_name(child) == Some("a")
                    && !rich::visible(tree, child, resource, notes).is_empty()
            })
            && !tree::nodes(tree, link).into_iter().skip(1).any(|child| {
                candidate(tree, child, resource, notes)
                    && !rich::visible(tree, child, resource, notes).is_empty()
            })
        {
            result.push(link);
            continue;
        }
        if !has_blocks && !rich::visible(tree, id, resource, notes).is_empty() {
            result.push(id);
        }
    }
    Ok(result)
}
pub(super) fn extract(
    resource: &mut ContentResource,
    index: usize,
    notes: &NoteRelations,
    segments: &mut Vec<SourceSegment>,
) -> Result<(), BookError> {
    let tree = &resource.parsed.tree;
    let blocks = blocks(tree, index, notes).map_err(|error| error.with_resource(&resource.href))?;
    for (ordinal, &block) in blocks.iter().enumerate() {
        let anchor = format!("tn{index}_{ordinal}");
        let (slots, rich_source) = rich::extract(tree, block, index, notes, &anchor)
            .map_err(|error| error.with_resource(&resource.href))?;
        let source = tree::collapse(
            &slots
                .iter()
                .map(|slot| slot.source_value.as_str())
                .collect::<String>(),
        );
        if source.is_empty() {
            continue;
        }
        let slot_contract_sha256 = slot_contract_digest(&slots).map_err(|error| {
            source_error("slot_contract", error.to_string()).with_resource(&resource.href)
        })?;
        let fingerprint =
            serialize_block(tree, block).map_err(|error| error.with_resource(&resource.href))?;
        resource.segment_ids.push(segments.len());
        segments.push(SourceSegment {
            source,
            kind: if tree
                .local_name(block)
                .is_some_and(|tag| HEADINGS.contains(&tag))
            {
                SegmentKind::Heading
            } else {
                SegmentKind::Text
            },
            anchor,
            resource_index: index,
            block_path: tree::path(tree, block),
            block_fingerprint: format!("{:x}", Sha256::digest(&fingerprint)),
            slots,
            slot_contract_sha256,
            rich_source,
            styled_source: None,
            semantic_hints: hints(tree, block),
            mirrored_toc_entry_id: None,
        });
    }
    resource.fragment_anchors = fragments(tree, &resource.segment_ids, segments);
    resource.title = tree::nodes(tree, tree.root)
        .into_iter()
        .filter(|&id| {
            tree.local_name(id)
                .is_some_and(|tag| HEADINGS.contains(&tag))
        })
        .map(|id| rich::visible(tree, id, index, notes))
        .find(|text| !text.is_empty())
        .or_else(|| {
            tree::nodes(tree, tree.root)
                .into_iter()
                .find(|&id| tree.local_name(id) == Some("title"))
                .map(|id| tree::collapse(&tree.itertext(id).expect("admitted source tree")))
        })
        .unwrap_or_default();
    Ok(())
}
fn hints(tree: &SourceTree, mut id: NodeId) -> Vec<String> {
    let mut result = Vec::new();
    loop {
        for name in ["type", "role"] {
            let value = tree::attr(tree, id, name).trim();
            if !value.is_empty() {
                let hint = format!("xhtml:{}:{name}={value}", tree.local_name(id).unwrap_or(""));
                if !result.contains(&hint) {
                    result.push(hint);
                }
            }
        }
        match tree.nodes[id.index()].parent {
            Some(parent) => id = parent,
            None => return result,
        }
    }
}
fn fragments(
    tree: &SourceTree,
    segment_ids: &[usize],
    segments: &[SourceSegment],
) -> BTreeMap<String, Option<usize>> {
    let ordered = tree::nodes(tree, tree.root);
    let positions: BTreeMap<_, _> = ordered
        .iter()
        .enumerate()
        .map(|(position, id)| (id.index(), position))
        .collect();
    let blocks: Vec<_> = segment_ids
        .iter()
        .map(|&index| {
            (
                index,
                tree.resolve_element_path(tree.root, &segments[index].block_path)
                    .expect("extracted block path"),
            )
        })
        .collect();
    let mut result = BTreeMap::new();
    for id in ordered {
        if tree.local_name(id).is_none() {
            continue;
        }
        let target = blocks
            .iter()
            .find(|(_, block)| tree::within(tree, id, *block))
            .or_else(|| {
                blocks
                    .iter()
                    .find(|(_, block)| positions[&block.index()] > positions[&id.index()])
            })
            .map(|(index, _)| *index);
        for name in ["id", "name"] {
            let value = tree::attr(tree, id, name);
            if !value.is_empty() {
                result.entry(value.to_owned()).or_insert(target);
            }
        }
    }
    result
}
