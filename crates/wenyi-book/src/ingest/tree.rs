#![forbid(unsafe_code)]
use crate::source::{NodeId, NodeKind, SourceTree};

pub(super) fn nodes(tree: &SourceTree, root: NodeId) -> Vec<NodeId> {
    let mut pending = vec![root];
    let mut result = Vec::new();
    while let Some(id) = pending.pop() {
        result.push(id);
        pending.extend(tree.nodes[id.index()].children.iter().rev().copied());
    }
    result
}
pub(super) fn path(tree: &SourceTree, mut id: NodeId) -> Vec<i32> {
    let mut result = Vec::new();
    while let Some(parent) = tree.nodes[id.index()].parent {
        let step = if tree.local_name(id).is_some() {
            tree.element_children(parent)
                .position(|child| child == id)
                .unwrap() as i32
        } else {
            -1 - tree.nodes[parent.index()]
                .children
                .iter()
                .position(|child| *child == id)
                .unwrap() as i32
        };
        result.push(step);
        id = parent;
    }
    result.reverse();
    result
}
pub(super) fn within(tree: &SourceTree, mut id: NodeId, root: NodeId) -> bool {
    loop {
        if id == root {
            return true;
        }
        match tree.nodes[id.index()].parent {
            Some(parent) => id = parent,
            None => return false,
        }
    }
}
pub(super) fn attr<'a>(tree: &'a SourceTree, id: NodeId, name: &str) -> &'a str {
    match &tree.nodes[id.index()].kind {
        NodeKind::Element(element) => element
            .attributes
            .iter()
            .find(|attribute| attribute.name.local == name)
            .map_or("", |attribute| attribute.value.as_str()),
        _ => "",
    }
}
pub(super) fn token(tree: &SourceTree, id: NodeId, name: &str, value: &str) -> bool {
    attr(tree, id, name)
        .split_ascii_whitespace()
        .any(|part| part == value)
}
pub(super) fn semantic(tree: &SourceTree, id: NodeId, value: &str, role: &str) -> bool {
    token(tree, id, "type", value) || token(tree, id, "role", role)
}
pub(super) fn collapse(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut space = false;
    for ch in value.chars() {
        if matches!(ch, ' ' | '\t' | '\r' | '\n' | '\u{c}' | '\u{b}') {
            if !space {
                result.push(' ');
            }
            space = true;
        } else {
            result.push(ch);
            space = false;
        }
    }
    result.trim().to_owned()
}
pub(super) fn ancestor_tag(tree: &SourceTree, mut id: NodeId, tags: &[&str]) -> bool {
    loop {
        if tree.local_name(id).is_some_and(|tag| tags.contains(&tag)) {
            return true;
        }
        match tree.nodes[id.index()].parent {
            Some(parent) => id = parent,
            None => return false,
        }
    }
}
