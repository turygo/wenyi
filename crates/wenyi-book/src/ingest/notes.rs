#![forbid(unsafe_code)]
use super::{source_error, tree};
use crate::{
    document::{ContentResource, Landmark, NoteBodyKind, NoteMarker, NoteRelations, NoteTarget},
    error::BookError,
    references::resolve_href,
    source::{NodeId, SourceTree},
};
use std::collections::{HashMap, HashSet};
use wenyi_core::{NoteKind, rich::note_label_key};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Location {
    resource: usize,
    node: NodeId,
}
struct Link {
    source: Location,
    target: Option<Location>,
    kind: Option<NoteKind>,
    label: String,
}

pub(super) fn detect(
    resources: &[ContentResource],
    excluded: &HashSet<String>,
    landmarks: &[Landmark],
) -> Result<NoteRelations, BookError> {
    let mut identifiers = HashMap::new();
    let mut result = NoteRelations::default();
    for (index, resource) in resources
        .iter()
        .enumerate()
        .filter(|(_, resource)| !excluded.contains(&resource.href))
    {
        let source = &resource.parsed.tree;
        for id in tree::nodes(source, source.root) {
            if protected(source, id) {
                continue;
            }
            if let Some(kind) =
                body_kind(source, id).map_err(|error| error.with_resource(&resource.href))?
            {
                add_target(
                    &mut result,
                    resources,
                    Location {
                        resource: index,
                        node: id,
                    },
                    kind,
                );
            }
            for name in ["id", "name"] {
                let value = tree::attr(source, id, name);
                if !value.is_empty() {
                    let location = Location {
                        resource: index,
                        node: id,
                    };
                    identifiers
                        .entry((index, value.to_owned()))
                        .and_modify(|previous| {
                            if *previous != Some(location) {
                                *previous = None;
                            }
                        })
                        .or_insert(Some(location));
                }
            }
        }
    }
    let mut links = Vec::new();
    for (index, resource) in resources
        .iter()
        .enumerate()
        .filter(|(_, resource)| !excluded.contains(&resource.href))
    {
        let source = &resource.parsed.tree;
        for id in tree::nodes(source, source.root)
            .into_iter()
            .filter(|&id| source.local_name(id) == Some("a") && !protected(source, id))
        {
            let target = resolve_href(&resource.href, tree::attr(source, id, "href"))
                .ok()
                .filter(|target| {
                    !target.external && !target.has_query && !target.fragment.is_empty()
                })
                .and_then(|target| {
                    resources
                        .iter()
                        .position(|resource| resource.href == target.resource_href)
                        .and_then(|index| {
                            identifiers
                                .get(&(index, target.fragment))
                                .copied()
                                .flatten()
                        })
                });
            links.push(Link {
                source: Location {
                    resource: index,
                    node: id,
                },
                target,
                kind: marker_kind(source, id)
                    .map_err(|error| error.with_resource(&resource.href))?,
                label: source
                    .itertext(id)
                    .map_err(|error| error.with_resource(&resource.href))?,
            });
        }
    }
    for link in &links {
        if let (Some(kind), Some(target)) = (link.kind, link.target) {
            add_marker(
                &mut result,
                resources,
                link.source,
                kind,
                target,
                &link.label,
            );
            let location = if kind == NoteKind::Noteref {
                target
            } else {
                link.source
            };
            if let Some(block) = note_block(resources, location)? {
                add_target(
                    &mut result,
                    resources,
                    block,
                    target_kind(resources, block, landmarks)?,
                );
            }
        }
    }
    infer(&mut result, resources, &links, landmarks)?;
    Ok(result)
}
fn protected(source: &SourceTree, id: NodeId) -> bool {
    tree::ancestor_tag(
        source,
        id,
        &[
            "head", "script", "style", "code", "math", "svg", "form", "nav",
        ],
    )
}
fn marker_kind(source: &SourceTree, id: NodeId) -> Result<Option<NoteKind>, BookError> {
    let reference = tree::semantic(source, id, "noteref", "doc-noteref");
    let backlink = tree::semantic(source, id, "backlink", "doc-backlink");
    if reference && backlink {
        return Err(source_error(
            "ambiguous_note_marker",
            "note marker declares both reference and backlink",
        ));
    }
    Ok(if reference {
        Some(NoteKind::Noteref)
    } else if backlink {
        Some(NoteKind::Backlink)
    } else {
        None
    })
}
fn body_kind(source: &SourceTree, id: NodeId) -> Result<Option<NoteBodyKind>, BookError> {
    let foot = tree::semantic(source, id, "footnote", "doc-footnote");
    let end = tree::semantic(source, id, "endnote", "doc-endnote");
    if foot && end {
        return Err(source_error(
            "ambiguous_note_body",
            "note body declares both footnote and endnote",
        ));
    }
    Ok(if foot {
        Some(NoteBodyKind::Footnote)
    } else if end {
        Some(NoteBodyKind::Endnote)
    } else {
        None
    })
}
fn note_block(
    resources: &[ContentResource],
    location: Location,
) -> Result<Option<Location>, BookError> {
    let source = &resources[location.resource].parsed.tree;
    let mut node = location.node;
    let mut fallback = None;
    loop {
        if body_kind(source, node)
            .map_err(|error| error.with_resource(&resources[location.resource].href))?
            .is_some()
        {
            return Ok(Some(Location { node, ..location }));
        }
        if fallback.is_none()
            && source
                .local_name(node)
                .is_some_and(|tag| ["p", "li", "aside"].contains(&tag))
        {
            fallback = Some(Location { node, ..location });
        }
        match source.nodes[node.index()].parent {
            Some(parent) => node = parent,
            None => return Ok(fallback),
        }
    }
}
fn target_kind(
    resources: &[ContentResource],
    location: Location,
    landmarks: &[Landmark],
) -> Result<NoteBodyKind, BookError> {
    if let Some(kind) = body_kind(&resources[location.resource].parsed.tree, location.node)
        .map_err(|error| error.with_resource(&resources[location.resource].href))?
    {
        return Ok(kind);
    }
    let mut foot = false;
    let mut end = false;
    for landmark in landmarks
        .iter()
        .filter(|landmark| landmark.resource_href == resources[location.resource].href)
    {
        for token in landmark
            .semantic_type
            .split_ascii_whitespace()
            .chain(landmark.role.split_ascii_whitespace())
        {
            foot |= matches!(token, "notes" | "footnotes" | "doc-footnote");
            end |= matches!(token, "endnotes" | "doc-endnotes");
        }
    }
    if foot && end {
        return Err(source_error(
            "ambiguous_note_resource",
            "note resource has conflicting landmark semantics",
        )
        .with_resource(&resources[location.resource].href));
    }
    Ok(if end {
        NoteBodyKind::Endnote
    } else {
        NoteBodyKind::Footnote
    })
}
fn add_marker(
    result: &mut NoteRelations,
    resources: &[ContentResource],
    source: Location,
    kind: NoteKind,
    target: Location,
    label: &str,
) {
    let path = tree::path(&resources[source.resource].parsed.tree, source.node);
    if result
        .markers
        .iter()
        .any(|marker| marker.resource_index == source.resource && marker.path == path)
    {
        return;
    }
    result.markers.push(NoteMarker {
        resource_index: source.resource,
        path,
        kind,
        label: label.to_owned(),
        target_resource_index: target.resource,
        target_path: tree::path(&resources[target.resource].parsed.tree, target.node),
    });
}
fn add_target(
    result: &mut NoteRelations,
    resources: &[ContentResource],
    location: Location,
    kind: NoteBodyKind,
) {
    let path = tree::path(&resources[location.resource].parsed.tree, location.node);
    if !result
        .targets
        .iter()
        .any(|target| target.resource_index == location.resource && target.path == path)
    {
        result.targets.push(NoteTarget {
            resource_index: location.resource,
            path,
            kind,
        });
    }
}
fn accepted_ref(resources: &[ContentResource], reference: Location, target: Location) -> bool {
    if reference.resource != target.resource {
        return false;
    }
    let source = &resources[reference.resource].parsed.tree;
    let mut top = reference.node;
    loop {
        if top == target.node {
            return true;
        }
        let Some(parent) = source.nodes[top.index()].parent else {
            return false;
        };
        if source
            .local_name(parent)
            .is_some_and(|tag| ["span", "sup"].contains(&tag))
            && source.element_children(parent).count() == 1
            && source.nodes[parent.index()]
                .text
                .as_deref()
                .is_none_or(|text| text.trim().is_empty())
            && source.nodes[top.index()]
                .tail
                .as_deref()
                .is_none_or(|text| text.trim().is_empty())
        {
            top = parent;
            continue;
        }
        let children: Vec<_> = source.element_children(parent).collect();
        let position = children.iter().position(|&child| child == top).unwrap();
        return position > 0
            && children[position - 1] == target.node
            && source.local_name(target.node) == Some("a")
            && (!tree::attr(source, target.node, "id").is_empty()
                || !tree::attr(source, target.node, "name").is_empty())
            && source
                .itertext(target.node)
                .is_ok_and(|text| text.trim().is_empty())
            && source.nodes[target.node.index()]
                .tail
                .as_deref()
                .is_none_or(|text| text.trim().is_empty());
    }
}
fn surrounding_body(resources: &[ContentResource], block: Location, backlink: Location) -> bool {
    let source = &resources[block.resource].parsed.tree;
    let mut before = String::new();
    let mut after = String::new();
    let mut seen = false;
    surrounding_text(
        source,
        block.node,
        backlink.node,
        &mut seen,
        &mut before,
        &mut after,
    );
    (before.trim().is_empty() || note_label_key(before.trim()).is_ok()) && !after.trim().is_empty()
}
fn surrounding_text(
    source: &SourceTree,
    id: NodeId,
    marker: NodeId,
    seen: &mut bool,
    before: &mut String,
    after: &mut String,
) {
    if id == marker {
        *seen = true;
        return;
    }
    let node = &source.nodes[id.index()];
    if let Some(text) = &node.text {
        if *seen {
            after.push_str(text);
        } else {
            before.push_str(text);
        }
    }
    for &child in &node.children {
        if source.local_name(child).is_some() {
            surrounding_text(source, child, marker, seen, before, after);
        }
        if let Some(tail) = &source.nodes[child.index()].tail {
            if *seen {
                after.push_str(tail);
            } else {
                before.push_str(tail);
            }
        }
    }
}
fn infer(
    result: &mut NoteRelations,
    resources: &[ContentResource],
    links: &[Link],
    landmarks: &[Landmark],
) -> Result<(), BookError> {
    let mut pairs = Vec::new();
    let mut inbound = HashMap::<Location, usize>::new();
    for (index, reference) in links.iter().enumerate() {
        if reference.kind == Some(NoteKind::Backlink) || note_label_key(&reference.label).is_err() {
            continue;
        }
        let Some(target) = reference.target else {
            continue;
        };
        let Some(block) = note_block(resources, target)? else {
            continue;
        };
        *inbound.entry(block).or_default() += 1;
        let backlinks: Vec<_> = links
            .iter()
            .enumerate()
            .filter(|(_, backlink)| {
                backlink.source.resource == block.resource
                    && tree::within(
                        &resources[block.resource].parsed.tree,
                        backlink.source.node,
                        block.node,
                    )
                    && backlink
                        .target
                        .is_some_and(|target| accepted_ref(resources, reference.source, target))
            })
            .collect();
        if backlinks.len() != 1 {
            continue;
        }
        let (back_index, backlink) = backlinks[0];
        if backlink.kind == Some(NoteKind::Noteref)
            || note_label_key(&backlink.label).ok() != note_label_key(&reference.label).ok()
            || !surrounding_body(resources, block, backlink.source)
        {
            continue;
        }
        let explicit = reference.kind.is_some()
            || backlink.kind.is_some()
            || body_kind(&resources[block.resource].parsed.tree, block.node)
                .map_err(|error| error.with_resource(&resources[block.resource].href))?
                .is_some()
            || landmarks.iter().any(|landmark| {
                landmark.resource_href == resources[block.resource].href
                    && landmark
                        .semantic_type
                        .split_ascii_whitespace()
                        .any(|token| matches!(token, "notes" | "footnotes" | "endnotes"))
            });
        if !explicit
            && note_label_key(&reference.label)
                .is_ok_and(|key| key.bytes().all(|byte| byte.is_ascii_digit()))
        {
            let Some(body) = note_block(resources, reference.source)? else {
                continue;
            };
            let mut before = String::new();
            let mut after = String::new();
            surrounding_text(
                &resources[body.resource].parsed.tree,
                body.node,
                reference.source.node,
                &mut false,
                &mut before,
                &mut after,
            );
            if body == block || before.trim().is_empty() || note_label_key(before.trim()).is_ok() {
                continue;
            }
        }
        pairs.push((index, back_index, block));
    }
    for &(reference_index, backlink_index, block) in &pairs {
        if inbound[&block] != 1
            || pairs.iter().any(|&(reference, backlink, _)| {
                reference == backlink_index || backlink == reference_index
            })
        {
            continue;
        }
        let reference = &links[reference_index];
        let backlink = &links[backlink_index];
        add_marker(
            result,
            resources,
            reference.source,
            NoteKind::Noteref,
            reference.target.unwrap(),
            &reference.label,
        );
        add_marker(
            result,
            resources,
            backlink.source,
            NoteKind::Backlink,
            backlink.target.unwrap(),
            &backlink.label,
        );
        add_target(
            result,
            resources,
            block,
            target_kind(resources, block, landmarks)?,
        );
    }
    Ok(())
}
