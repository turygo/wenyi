#![forbid(unsafe_code)]
use super::tree;
use crate::{
    document::{ContentResource, NavigationEntry, SegmentKind, SourceSegment},
    references::resolve_href,
    source::{NodeId, SourceTree},
};
use std::collections::HashSet;
fn key(entry: &NavigationEntry) -> String {
    if entry.fragment.is_empty() {
        entry.resource_href.clone()
    } else {
        format!("{}#{}", entry.resource_href, entry.fragment)
    }
}
fn direct_link(source: &SourceTree, block: NodeId) -> Option<NodeId> {
    if source.local_name(block) == Some("a") {
        let parent = source.nodes[block.index()].parent?;
        if source.local_name(parent) != Some("li")
            || source.nodes[parent.index()]
                .text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
        {
            return None;
        }
        let children: Vec<_> = source.element_children(parent).collect();
        if children
            .iter()
            .filter(|&&id| source.local_name(id) == Some("a"))
            .count()
            != 1
            || children.iter().any(|&id| {
                !source
                    .local_name(id)
                    .is_some_and(|tag| ["a", "ol", "ul"].contains(&tag))
                    || source.nodes[id.index()]
                        .tail
                        .as_deref()
                        .is_some_and(|text| !text.trim().is_empty())
            })
        {
            return None;
        }
        return Some(block);
    }
    let children: Vec<_> = source.element_children(block).collect();
    if children.len() != 1 || source.local_name(children[0]) != Some("a") {
        return None;
    }
    let full = tree::collapse(&source.itertext(block).ok()?);
    (!full.is_empty() && full == tree::collapse(&source.itertext(children[0]).ok()?))
        .then_some(children[0])
}
pub(super) fn annotate(
    resources: &[ContentResource],
    segments: &mut [SourceSegment],
    navigation: &[NavigationEntry],
    declared: &HashSet<String>,
    canonical: Option<&str>,
    title: &str,
) {
    let Some(canonical) =
        canonical.or_else(|| navigation.first().map(|entry| entry.toc_path.as_str()))
    else {
        return;
    };
    let entries: Vec<_> = navigation
        .iter()
        .filter(|entry| {
            entry.toc_path == canonical
                && !entry.external
                && !entry.resource_href.is_empty()
                && !entry.label.trim().is_empty()
        })
        .collect();
    let expected: Vec<_> = entries.iter().map(|entry| key(entry)).collect();
    if expected.len() < 2 || expected.iter().collect::<HashSet<_>>().len() != expected.len() {
        return;
    }
    let root = tree::collapse(&entries[0].label).to_lowercase();
    let document = tree::collapse(title).to_lowercase();
    let omit_root = document == root || document.starts_with(&format!("{root}:"));
    for resource in resources
        .iter()
        .filter(|resource| !declared.contains(&resource.href))
    {
        if resource
            .segment_ids
            .iter()
            .filter(|&&id| segments[id].kind == SegmentKind::Heading)
            .count()
            > 1
        {
            continue;
        }
        let source = &resource.parsed.tree;
        let mut links = Vec::new();
        let mut rejected = false;
        for &id in &resource.segment_ids {
            if segments[id].kind == SegmentKind::Heading {
                continue;
            }
            let link = source
                .resolve_element_path(source.root, &segments[id].block_path)
                .ok()
                .and_then(|block| direct_link(source, block));
            let target = link
                .and_then(|link| {
                    resolve_href(&resource.href, tree::attr(source, link, "href")).ok()
                })
                .filter(|target| !target.external && !target.has_query);
            let Some(target) = target else {
                rejected = true;
                break;
            };
            links.push((id, target.target_key()));
        }
        let actual: Vec<_> = links.iter().map(|(_, target)| target).collect();
        let full: Vec<_> = expected.iter().collect();
        if rejected || links.len() < 2 || !(actual == full || omit_root && actual == full[1..]) {
            continue;
        }
        for (id, target) in links {
            let entry = entries[expected.iter().position(|key| key == &target).unwrap()];
            segments[id].mirrored_toc_entry_id = Some(entry.entry_id.clone());
        }
    }
}
