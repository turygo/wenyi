#![forbid(unsafe_code)]

use crate::archive::Archive;
use crate::css::{ElementStyle, SourceCss};
use crate::document::{ContentResource, SourceSegment};
use crate::source::{NodeId, NodeKind, SourceTree};
use crate::{BookError, BookErrorKind};
use std::collections::{BTreeSet, HashMap};
use wenyi_core::{
    InlineMark, MarkKind, OrderedAttributes, RichSource, SemanticRole, SlotField, semantic_roles,
};

const ROLES: [SemanticRole; 6] = [
    SemanticRole::Bold,
    SemanticRole::Italic,
    SemanticRole::Link,
    SemanticRole::Superscript,
    SemanticRole::Subscript,
    SemanticRole::Literal,
];

pub(crate) fn enrich(
    archive: &mut Archive,
    resources: &[ContentResource],
    segments: &mut [SourceSegment],
) -> Result<(), BookError> {
    for resource in resources {
        if resource.segment_ids.is_empty() {
            continue;
        }
        let tree = &resource.parsed.tree;
        let mut requested = BTreeSet::new();
        for &index in &resource.segment_ids {
            let segment = &segments[index];
            let block = resolve(tree, tree.root, &segment.block_path, &resource.href)?;
            request(tree, block, &mut requested);
            for slot in &segment.slots {
                let owner = resolve(tree, block, &slot.element_path, &resource.href)?;
                let owner = if slot.field == SlotField::Tail {
                    tree.nodes[owner.index()].parent.unwrap_or(block)
                } else {
                    owner
                };
                request(tree, owner, &mut requested);
            }
            for mark in &segment.rich_source.marks {
                request(
                    tree,
                    resolve(tree, block, &mark.path, &resource.href)?,
                    &mut requested,
                );
            }
        }
        let requested = requested
            .into_iter()
            .map(NodeId::new)
            .collect::<Result<Vec<_>, _>>()?;
        let css = SourceCss::load(archive, tree, &resource.href)?;
        let styles: HashMap<_, _> = css
            .evidence(tree, &requested, &resource.href)?
            .into_iter()
            .map(|style| (style.node, style))
            .collect();
        for &index in &resource.segment_ids {
            segments[index].styled_source =
                project(tree, &segments[index], &styles, &resource.href)?;
        }
    }
    Ok(())
}

fn request(tree: &SourceTree, mut node: NodeId, requested: &mut BTreeSet<usize>) {
    loop {
        if !requested.insert(node.index()) {
            break;
        }
        match tree.nodes[node.index()].parent {
            Some(parent) => node = parent,
            None => break,
        }
    }
}
fn resolve(tree: &SourceTree, root: NodeId, path: &[i32], href: &str) -> Result<NodeId, BookError> {
    let node = tree.resolve_element_path(root, path)?;
    if tree.local_name(node).is_none() {
        return Err(error(
            href,
            "source style locator does not resolve to an element",
        ));
    }
    Ok(node)
}
fn error(href: &str, detail: impl Into<String>) -> BookError {
    BookError::new(BookErrorKind::Source, "style_projection", detail).with_resource(href)
}
fn bit(role: SemanticRole) -> u8 {
    1 << role as u8
}
fn mask(roles: impl Iterator<Item = SemanticRole>) -> u8 {
    roles.fold(0, |mask, role| mask | bit(role))
}
fn roles(mask: u8) -> Vec<SemanticRole> {
    ROLES
        .into_iter()
        .filter(|role| mask & bit(*role) != 0)
        .collect()
}

struct Plan<'a> {
    node: NodeId,
    path: &'a [i32],
    raw: Option<&'a InlineMark>,
    desired: u8,
    members: Vec<(usize, u8)>,
    decoration: bool,
}

fn project(
    tree: &SourceTree,
    segment: &SourceSegment,
    styles: &HashMap<NodeId, ElementStyle>,
    href: &str,
) -> Result<Option<RichSource>, BookError> {
    let block = resolve(tree, tree.root, &segment.block_path, href)?;
    let slots: HashMap<_, _> = segment
        .slots
        .iter()
        .map(|slot| (slot.id.as_str(), slot))
        .collect();
    let owners: Vec<_> = segment
        .rich_source
        .runs
        .iter()
        .map(|run| {
            run.slot_id
                .as_deref()
                .and_then(|id| slots.get(id))
                .map(|slot| {
                    let node = resolve(tree, block, &slot.element_path, href)?;
                    Ok(if slot.field == SlotField::Tail {
                        tree.nodes[node.index()].parent.unwrap_or(block)
                    } else {
                        node
                    })
                })
                .transpose()
        })
        .collect::<Result<_, BookError>>()?;
    let mut plans = Vec::new();
    for raw in &segment.rich_source.marks {
        let node = resolve(tree, block, &raw.path, href)?;
        let style = &styles[&node];
        let desired = mask(style.roles.iter().copied());
        let members = segment
            .rich_source
            .runs
            .iter()
            .enumerate()
            .filter(|(_, run)| run.marks.contains(&raw.id))
            .map(|(index, _)| {
                (
                    index,
                    active_mask(desired, node, owners[index], styles, false, tree, block),
                )
            })
            .collect();
        plans.push(Plan {
            node,
            path: &raw.path,
            raw: Some(raw),
            desired,
            members,
            decoration: style.decoration,
        });
    }
    add_root_plan(tree, block, &owners, styles, &mut plans);
    let changed = plans.iter().any(|plan| match plan.raw {
        Some(raw) => {
            mask(semantic_roles(raw).iter()) != plan.desired
                || plan.members.iter().any(|(_, roles)| *roles != plan.desired)
                || plan.decoration && raw.kind != MarkKind::Decoration
        }
        None => !plan.members.is_empty(),
    });
    if !changed {
        return Ok(None);
    }
    let mut rich = segment.rich_source.clone();
    rich.marks.clear();
    for run in &mut rich.runs {
        run.marks.clear();
    }
    for plan in plans {
        materialize(tree, &plan, styles, &mut rich);
    }
    rich.validate()
        .map_err(|cause| error(href, cause.to_string()))?;
    Ok(Some(rich))
}

fn add_root_plan<'a>(
    tree: &SourceTree,
    block: NodeId,
    owners: &[Option<NodeId>],
    styles: &HashMap<NodeId, ElementStyle>,
    plans: &mut Vec<Plan<'a>>,
) {
    let mut root_roles = mask(styles[&block].roles.iter().copied());
    for &(origin, role) in &styles[&block].active_emphasis {
        if !within(tree, origin, block) {
            root_roles |= bit(role);
        }
    }
    let mut ancestor = tree.nodes[block.index()].parent;
    while let Some(node) = ancestor {
        root_roles |= mask(
            styles[&node]
                .roles
                .iter()
                .copied()
                .filter(|role| !matches!(role, SemanticRole::Bold | SemanticRole::Italic)),
        );
        ancestor = tree.nodes[node.index()].parent;
    }
    if root_roles != 0 || styles[&block].decoration {
        let members = owners
            .iter()
            .enumerate()
            .filter_map(|(index, owner)| {
                let active = match owner {
                    Some(_) => active_mask(root_roles, block, *owner, styles, true, tree, block),
                    None => root_roles & !(bit(SemanticRole::Bold) | bit(SemanticRole::Italic)),
                };
                (owner.is_some() || active != 0).then_some((index, active))
            })
            .collect();
        plans.insert(
            0,
            Plan {
                node: block,
                path: &[],
                raw: None,
                desired: root_roles,
                members,
                decoration: styles[&block].decoration,
            },
        );
    }
}

fn active_mask(
    desired: u8,
    origin: NodeId,
    owner: Option<NodeId>,
    styles: &HashMap<NodeId, ElementStyle>,
    root: bool,
    tree: &SourceTree,
    block: NodeId,
) -> u8 {
    let Some(owner) = owner else {
        return desired;
    };
    let mut result = desired;
    for role in [SemanticRole::Bold, SemanticRole::Italic] {
        if !styles[&owner]
            .active_emphasis
            .iter()
            .any(|&(node, active)| {
                active == role && (node == origin || root && !within(tree, node, block))
            })
        {
            result &= !bit(role);
        }
    }
    result
}
fn within(tree: &SourceTree, mut node: NodeId, ancestor: NodeId) -> bool {
    loop {
        if node == ancestor {
            return true;
        }
        match tree.nodes[node.index()].parent {
            Some(parent) => node = parent,
            None => return false,
        }
    }
}

fn materialize(
    tree: &SourceTree,
    plan: &Plan<'_>,
    styles: &HashMap<NodeId, ElementStyle>,
    rich: &mut RichSource,
) {
    let mut groups = [0_u8; 6];
    let mut count = 0;
    for role in ROLES {
        let role = bit(role);
        if !plan.members.iter().any(|(_, mask)| mask & role != 0) {
            continue;
        }
        if let Some(group) = groups[..count].iter_mut().find(|group| {
            let representative = 1 << group.trailing_zeros();
            plan.members
                .iter()
                .all(|(_, mask)| (mask & role != 0) == (mask & representative != 0))
        }) {
            *group |= role;
        } else {
            groups[count] = role;
            count += 1;
        }
    }
    let full = groups[..count].iter().position(|group| {
        let representative = 1 << group.trailing_zeros();
        plan.members
            .iter()
            .all(|(_, mask)| mask & representative != 0)
    });
    let id = plan
        .raw
        .map(|raw| raw.id.clone())
        .unwrap_or_else(|| "m:root".to_owned());
    if let Some(index) = full {
        let group = groups[index];
        record(
            tree,
            plan,
            styles,
            rich,
            id.clone(),
            group,
            Some(1 << group.trailing_zeros()),
        );
    } else {
        record(tree, plan, styles, rich, id.clone(), 0, None);
    }
    for (index, &group) in groups[..count].iter().enumerate() {
        if Some(index) != full {
            record(
                tree,
                plan,
                styles,
                rich,
                format!("{id}@{}", role_suffix(group)),
                group,
                Some(1 << group.trailing_zeros()),
            );
        }
    }
}
fn role_suffix(mask: u8) -> String {
    ROLES
        .into_iter()
        .filter(|role| mask & bit(*role) != 0)
        .map(|role| match role {
            SemanticRole::Bold => "bold",
            SemanticRole::Italic => "italic",
            SemanticRole::Link => "link",
            SemanticRole::Superscript => "superscript",
            SemanticRole::Subscript => "subscript",
            SemanticRole::Literal => "literal",
        })
        .collect::<Vec<_>>()
        .join("+")
}
fn record(
    tree: &SourceTree,
    plan: &Plan<'_>,
    styles: &HashMap<NodeId, ElementStyle>,
    rich: &mut RichSource,
    id: String,
    semantic: u8,
    membership: Option<u8>,
) {
    let NodeKind::Element(element) = &tree.nodes[plan.node.index()].kind else {
        unreachable!("admitted element")
    };
    let mut text = String::new();
    for &(index, active) in &plan.members {
        if membership.is_none_or(|role| active & role != 0) {
            text.push_str(&rich.runs[index].text);
            rich.runs[index].marks.push(id.clone());
        }
    }
    rich.marks.push(InlineMark {
        id,
        path: plan.path.to_vec(),
        tag: element.name.expanded(),
        attributes: OrderedAttributes(
            element
                .attributes
                .iter()
                .map(|attribute| (attribute.name.expanded(), attribute.value.clone()))
                .collect(),
        ),
        source_text: text,
        kind: if plan.decoration && semantic == 0 {
            MarkKind::Decoration
        } else {
            MarkKind::Style
        },
        style_evidence: styles[&plan.node].evidence.clone(),
        semantics: roles(semantic),
        required: semantic != 0,
    });
}
