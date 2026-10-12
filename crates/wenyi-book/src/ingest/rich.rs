#![forbid(unsafe_code)]
use super::{source_error, tree};
use crate::{
    document::NoteRelations,
    error::BookError,
    source::{NodeId, NodeKind, SourceTree},
};
use wenyi_core::{
    SlotField, SourceSlot,
    rich::{
        AtomKind, InlineAtom, InlineMark, InlineRun, MarkKind, OrderedAttributes, RichSource,
        tag_semantics,
    },
};

pub(super) const OPAQUE: &[&str] = &["script", "style", "rt", "rp"];
pub(super) const MEDIA: &[&str] = &[
    "audio", "canvas", "embed", "hr", "iframe", "img", "math", "object", "svg", "video",
];

pub(super) fn protected(
    tree: &SourceTree,
    id: NodeId,
    resource: usize,
    notes: &NoteRelations,
) -> bool {
    tree.local_name(id) == Some("a")
        && (tree::semantic(tree, id, "noteref", "doc-noteref")
            || tree::semantic(tree, id, "backlink", "doc-backlink")
            || notes.markers.iter().any(|marker| {
                marker.resource_index == resource && marker.path == tree::path(tree, id)
            }))
}
pub(super) fn excluded(
    tree: &SourceTree,
    id: NodeId,
    resource: usize,
    notes: &NoteRelations,
) -> bool {
    tree.local_name(id)
        .is_some_and(|tag| OPAQUE.contains(&tag) || MEDIA.contains(&tag))
        || protected(tree, id, resource, notes)
}
pub(super) fn visible(
    tree: &SourceTree,
    id: NodeId,
    resource: usize,
    notes: &NoteRelations,
) -> String {
    fn walk(
        tree: &SourceTree,
        id: NodeId,
        resource: usize,
        notes: &NoteRelations,
        output: &mut String,
    ) {
        if tree.local_name(id).is_none() || excluded(tree, id, resource, notes) {
            return;
        }
        let node = &tree.nodes[id.index()];
        if let Some(text) = &node.text {
            output.push_str(text);
        }
        for &child in &node.children {
            walk(tree, child, resource, notes, output);
            if let Some(tail) = &tree.nodes[child.index()].tail {
                output.push_str(tail);
            }
        }
    }
    let mut output = String::new();
    walk(tree, id, resource, notes, &mut output);
    tree::collapse(&output)
}

pub(super) fn extract(
    tree: &SourceTree,
    block: NodeId,
    resource: usize,
    notes: &NoteRelations,
    anchor: &str,
) -> Result<(Vec<SourceSlot>, RichSource), BookError> {
    let mut builder = Builder {
        tree,
        resource,
        notes,
        anchor,
        slots: Vec::new(),
        rich: RichSource {
            version: 2,
            marks: Vec::new(),
            atoms: Vec::new(),
            runs: Vec::new(),
            note_references: Vec::new(),
        },
    };
    builder.walk(block, &mut Vec::new(), &[]);
    for mark in &mut builder.rich.marks {
        mark.source_text = builder
            .rich
            .runs
            .iter()
            .filter(|run| run.marks.contains(&mark.id))
            .map(|run| run.text.as_str())
            .collect();
    }
    builder.rich.marks.retain(|mark| {
        builder
            .rich
            .runs
            .iter()
            .any(|run| run.marks.contains(&mark.id))
    });
    builder
        .rich
        .validate()
        .map_err(|error| source_error("rich_source", error.to_string()))?;
    Ok((builder.slots, builder.rich))
}
struct Builder<'a> {
    tree: &'a SourceTree,
    resource: usize,
    notes: &'a NoteRelations,
    anchor: &'a str,
    slots: Vec<SourceSlot>,
    rich: RichSource,
}
impl Builder<'_> {
    fn add(&mut self, id: NodeId, path: &[i32], field: SlotField, active: &[String]) {
        let node = &self.tree.nodes[id.index()];
        let value = match field {
            SlotField::Text => &node.text,
            SlotField::Tail => &node.tail,
        };
        if let Some(value) = value {
            let slot_id = format!("{}:s{}", self.anchor, self.slots.len() + 1);
            self.slots.push(SourceSlot {
                id: slot_id.clone(),
                element_path: path.to_vec(),
                field,
                source_value: value.clone(),
                target_value: None,
            });
            self.rich.runs.push(InlineRun {
                text: value.clone(),
                marks: active.to_vec(),
                slot_id: Some(slot_id),
                ..InlineRun::default()
            });
        }
    }
    fn atom_kind(&self, id: NodeId) -> Option<AtomKind> {
        let Some(tag) = self.tree.local_name(id) else {
            return Some(AtomKind::Opaque);
        };
        if protected(self.tree, id, self.resource, self.notes) {
            return Some(AtomKind::Note);
        }
        let children: Vec<_> = self.tree.element_children(id).collect();
        if tag == "sup"
            && children.len() == 1
            && self.tree.nodes[id.index()].text.is_none()
            && self.tree.nodes[children[0].index()].tail.is_none()
            && protected(self.tree, children[0], self.resource, self.notes)
        {
            return Some(AtomKind::Note);
        }
        if tag == "br" {
            return Some(AtomKind::Linebreak);
        }
        if MEDIA.contains(&tag) {
            return Some(AtomKind::Media);
        }
        if OPAQUE.contains(&tag) {
            return Some(AtomKind::Opaque);
        }
        if children.is_empty()
            && self.tree.nodes[id.index()]
                .text
                .as_deref()
                .is_none_or(str::is_empty)
        {
            return Some(AtomKind::Anchor);
        }
        None
    }
    fn walk(&mut self, id: NodeId, path: &mut Vec<i32>, active: &[String]) {
        if !path.is_empty()
            && let Some(kind) = self.atom_kind(id)
        {
            let atom_id = identifier("a", path);
            let label = match &self.tree.nodes[id.index()].kind {
                NodeKind::Comment(value) | NodeKind::EntityReference(value) => value.clone(),
                NodeKind::ProcessingInstruction { data, .. } => data.clone(),
                NodeKind::Element(_) => self.tree.itertext(id).expect("admitted source tree"),
            };
            self.rich.atoms.push(InlineAtom {
                id: atom_id.clone(),
                path: path.clone(),
                kind,
                label,
            });
            self.rich.runs.push(InlineRun {
                atom: Some(atom_id),
                marks: active.to_vec(),
                ..InlineRun::default()
            });
            return;
        }
        let mut own_active = active.to_vec();
        if !path.is_empty()
            && let NodeKind::Element(element) = &self.tree.nodes[id.index()].kind
        {
            let mark_id = identifier("m", path);
            let kind = match element.name.local.as_str() {
                "b" | "strong" => MarkKind::Bold,
                "i" | "em" => MarkKind::Italic,
                "a" => MarkKind::Link,
                _ => MarkKind::Style,
            };
            self.rich.marks.push(InlineMark {
                id: mark_id.clone(),
                path: path.clone(),
                tag: element.name.expanded(),
                attributes: OrderedAttributes(
                    element
                        .attributes
                        .iter()
                        .map(|attribute| (attribute.name.expanded(), attribute.value.clone()))
                        .collect(),
                ),
                source_text: String::new(),
                kind,
                style_evidence: Vec::new(),
                semantics: tag_semantics(&element.name.local).to_vec(),
                required: !tag_semantics(&element.name.local).is_empty(),
            });
            own_active.push(mark_id);
        }
        self.add(id, path, SlotField::Text, &own_active);
        let mut element_index = 0;
        let tree = self.tree;
        for (child_index, &child) in tree.nodes[id.index()].children.iter().enumerate() {
            let element = tree.local_name(child).is_some();
            path.push(if element {
                element_index
            } else {
                -1 - child_index as i32
            });
            self.walk(child, path, &own_active);
            if element {
                self.add(child, path, SlotField::Tail, &own_active);
                element_index += 1;
            }
            path.pop();
        }
    }
}
pub(super) fn identifier(prefix: &str, path: &[i32]) -> String {
    if path.is_empty() {
        return format!("{prefix}:root");
    }
    format!(
        "{prefix}:{}",
        path.iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(".")
    )
}
