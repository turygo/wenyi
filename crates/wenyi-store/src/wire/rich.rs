use super::{decode, object, strict};
use crate::StoreError;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::Path;
use wenyi_core::rich::{AtomKind, MarkKind, NoteKind, OmissionKind, SemanticRole};
use wenyi_core::{
    InlineAtom, InlineMark, InlineRun, NoteReference, OrderedAttributes, RichSource, RichTarget,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    #[serde(default)]
    text: String,
    #[serde(default)]
    marks: Vec<String>,
    #[serde(default)]
    atom: Option<String>,
    #[serde(default)]
    slot_id: Option<String>,
    #[serde(default)]
    omission: Option<String>,
    #[serde(default)]
    explanation: String,
    #[serde(default)]
    note: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mark {
    id: String,
    path: Vec<i32>,
    tag: String,
    #[serde(default)]
    attributes: Map<String, Value>,
    source_text: String,
    #[serde(default = "style")]
    kind: String,
    #[serde(default)]
    style_evidence: Vec<String>,
    #[serde(default)]
    semantics: Vec<String>,
    #[serde(default)]
    required: bool,
}
fn style() -> String {
    "style".to_owned()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Atom {
    id: String,
    path: Vec<i32>,
    kind: String,
    #[serde(default)]
    label: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    mark_id: String,
    label: String,
    kind: String,
    target_resource: String,
    target_path: Vec<i32>,
    content_marks: Vec<String>,
}

fn runs(value: &Map<String, Value>, path: &Path) -> Result<Vec<InlineRun>, StoreError> {
    let raw: Vec<Run> = match value.get("runs") {
        Some(value) => decode(value, path)?,
        None => Vec::new(),
    };
    raw.into_iter()
        .map(|run| {
            let omission = match run.omission.as_deref() {
                None => None,
                Some("implicit") => Some(OmissionKind::Implicit),
                Some("merged") => Some(OmissionKind::Merged),
                _ => return Err(StoreError::wire(path, "unknown rich omission")),
            };
            Ok(InlineRun {
                text: run.text,
                marks: run.marks,
                atom: run.atom,
                slot_id: run.slot_id,
                omission,
                explanation: run.explanation,
                note: run.note,
            })
        })
        .collect()
}

fn version(value: &Map<String, Value>, path: &Path) -> Result<u32, StoreError> {
    match value.get("version") {
        None => Ok(2),
        Some(value) => decode(value, path),
    }
}

pub(super) fn source(value: &Value, path: &Path) -> Result<RichSource, StoreError> {
    let map = object(value, path)?;
    strict(
        map,
        &["version", "marks", "atoms", "runs", "note_references"],
        path,
    )?;
    let marks = optional_list::<Mark>(map, "marks", path)?
        .into_iter()
        .map(|mark| admit_mark(mark, path))
        .collect::<Result<_, _>>()?;
    let atoms = optional_list::<Atom>(map, "atoms", path)?
        .into_iter()
        .map(|atom| admit_atom(atom, path))
        .collect::<Result<_, _>>()?;
    let note_references = optional_list::<Reference>(map, "note_references", path)?
        .into_iter()
        .map(|reference| admit_reference(reference, path))
        .collect::<Result<_, _>>()?;
    let source = RichSource {
        version: version(map, path)?,
        marks,
        atoms,
        runs: runs(map, path)?,
        note_references,
    };
    source
        .validate()
        .map_err(|error| StoreError::wire(path, error.to_string()))?;
    Ok(source)
}

pub(super) fn target(value: &Value, path: &Path) -> Result<RichTarget, StoreError> {
    let map = object(value, path)?;
    strict(map, &["version", "runs"], path)?;
    let target = RichTarget {
        version: version(map, path)?,
        runs: runs(map, path)?,
    };
    target
        .validate()
        .map_err(|error| StoreError::wire(path, error.to_string()))?;
    Ok(target)
}

fn optional_list<T: serde::de::DeserializeOwned>(
    map: &Map<String, Value>,
    key: &str,
    path: &Path,
) -> Result<Vec<T>, StoreError> {
    match map.get(key) {
        Some(value) => decode(value, path),
        None => Ok(Vec::new()),
    }
}

fn admit_mark(mark: Mark, path: &Path) -> Result<InlineMark, StoreError> {
    let kind = match mark.kind.as_str() {
        "bold" => MarkKind::Bold,
        "italic" => MarkKind::Italic,
        "link" => MarkKind::Link,
        "style" => MarkKind::Style,
        "decoration" => MarkKind::Decoration,
        _ => return Err(StoreError::wire(path, "unknown rich mark kind")),
    };
    let semantics = mark
        .semantics
        .into_iter()
        .map(|role| match role.as_str() {
            "bold" => Ok(SemanticRole::Bold),
            "italic" => Ok(SemanticRole::Italic),
            "link" => Ok(SemanticRole::Link),
            "superscript" => Ok(SemanticRole::Superscript),
            "subscript" => Ok(SemanticRole::Subscript),
            "literal" => Ok(SemanticRole::Literal),
            _ => Err(StoreError::wire(path, "unknown semantic role")),
        })
        .collect::<Result<_, _>>()?;
    let attributes = mark
        .attributes
        .into_iter()
        .map(|(key, value)| match value {
            Value::String(value) => Ok((key, value)),
            _ => Err(StoreError::wire(path, "rich attribute must be a string")),
        })
        .collect::<Result<_, StoreError>>()?;
    Ok(InlineMark {
        id: mark.id,
        path: mark.path,
        tag: mark.tag,
        attributes: OrderedAttributes(attributes),
        source_text: mark.source_text,
        kind,
        style_evidence: mark.style_evidence,
        semantics,
        required: mark.required,
    })
}

fn admit_atom(atom: Atom, path: &Path) -> Result<InlineAtom, StoreError> {
    let kind = match atom.kind.as_str() {
        "note" => AtomKind::Note,
        "anchor" => AtomKind::Anchor,
        "linebreak" => AtomKind::Linebreak,
        "media" => AtomKind::Media,
        "opaque" => AtomKind::Opaque,
        _ => return Err(StoreError::wire(path, "unknown rich atom kind")),
    };
    Ok(InlineAtom {
        id: atom.id,
        path: atom.path,
        kind,
        label: atom.label,
    })
}

fn admit_reference(reference: Reference, path: &Path) -> Result<NoteReference, StoreError> {
    let kind = match reference.kind.as_str() {
        "noteref" => NoteKind::Noteref,
        "backlink" => NoteKind::Backlink,
        _ => return Err(StoreError::wire(path, "unknown rich note kind")),
    };
    Ok(NoteReference {
        mark_id: reference.mark_id,
        label: reference.label,
        kind,
        target_resource: reference.target_resource,
        target_path: reference.target_path,
        content_marks: reference.content_marks,
    })
}

pub(super) fn coordinates(values: &[i32], path: &Path) -> Result<(), StoreError> {
    if values.iter().any(|index| *index < 0) {
        return Err(StoreError::wire(
            path,
            "source paths require nonnegative element indices",
        ));
    }
    Ok(())
}
