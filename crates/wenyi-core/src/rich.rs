mod validation;

use crate::{CoreError, CoreErrorKind};
use serde::{Serialize, Serializer, ser::SerializeMap};
use std::borrow::Cow;
use unicode_normalization::UnicodeNormalization;
pub use validation::validate_rich_target;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OrderedAttributes(pub Vec<(String, String)>);

impl OrderedAttributes {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }
}

impl Serialize for OrderedAttributes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SemanticRole {
    Bold,
    Italic,
    Link,
    Superscript,
    Subscript,
    Literal,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MarkKind {
    Bold,
    Italic,
    Link,
    Style,
    Decoration,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AtomKind {
    Note,
    Anchor,
    Linebreak,
    Media,
    Opaque,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NoteKind {
    Noteref,
    Backlink,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OmissionKind {
    Implicit,
    Merged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InlineMark {
    pub id: String,
    pub path: Vec<i32>,
    pub tag: String,
    pub attributes: OrderedAttributes,
    pub source_text: String,
    pub kind: MarkKind,
    pub style_evidence: Vec<String>,
    pub semantics: Vec<SemanticRole>,
    pub required: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InlineAtom {
    pub id: String,
    pub path: Vec<i32>,
    pub kind: AtomKind,
    pub label: String,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InlineRun {
    pub text: String,
    pub marks: Vec<String>,
    pub atom: Option<String>,
    pub slot_id: Option<String>,
    pub omission: Option<OmissionKind>,
    pub explanation: String,
    pub note: Option<String>,
}
impl Serialize for InlineRun {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(
            4 + if self.omission.is_some() { 2 } else { 0 } + usize::from(self.note.is_some()),
        ))?;
        map.serialize_entry("text", &self.text)?;
        map.serialize_entry("marks", &self.marks)?;
        map.serialize_entry("atom", &self.atom)?;
        map.serialize_entry("slot_id", &self.slot_id)?;
        if let Some(omission) = &self.omission {
            map.serialize_entry("omission", omission)?;
            map.serialize_entry("explanation", &self.explanation)?;
        }
        if let Some(note) = &self.note {
            map.serialize_entry("note", note)?;
        }
        map.end()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NoteReference {
    pub mark_id: String,
    pub label: String,
    pub kind: NoteKind,
    pub target_resource: String,
    pub target_path: Vec<i32>,
    pub content_marks: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RichSource {
    pub version: u32,
    pub marks: Vec<InlineMark>,
    pub atoms: Vec<InlineAtom>,
    pub runs: Vec<InlineRun>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub note_references: Vec<NoteReference>,
}
impl RichSource {
    pub fn validate(&self) -> Result<(), CoreError> {
        validation::validate_source(self)
    }
    pub fn text(&self) -> String {
        self.runs.iter().map(|run| run.text.as_str()).collect()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RichTarget {
    pub version: u32,
    pub runs: Vec<InlineRun>,
}
impl RichTarget {
    pub fn validate(&self) -> Result<(), CoreError> {
        validation::validate_target(self)
    }
    pub fn text(&self) -> String {
        self.runs.iter().map(|run| run.text.as_str()).collect()
    }
}

pub fn tag_semantics(tag: &str) -> &'static [SemanticRole] {
    match local_tag(tag).as_ref() {
        "b" | "strong" => &[SemanticRole::Bold],
        "i" | "em" => &[SemanticRole::Italic],
        "a" => &[SemanticRole::Link],
        "sup" => &[SemanticRole::Superscript],
        "sub" => &[SemanticRole::Subscript],
        "code" | "kbd" | "samp" | "pre" => &[SemanticRole::Literal],
        _ => &[],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticRoles(u8);

impl SemanticRoles {
    pub fn contains(self, role: &SemanticRole) -> bool {
        self.0 & (1 << *role as u8) != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = SemanticRole> {
        [
            SemanticRole::Bold,
            SemanticRole::Italic,
            SemanticRole::Link,
            SemanticRole::Superscript,
            SemanticRole::Subscript,
            SemanticRole::Literal,
        ]
        .into_iter()
        .filter(move |role| self.contains(role))
    }
}

pub fn semantic_roles(mark: &InlineMark) -> SemanticRoles {
    let mut roles = SemanticRoles(0);
    for role in tag_semantics(&mark.tag) {
        if !matches!(mark.kind, MarkKind::Style | MarkKind::Decoration)
            || (mark.semantics.is_empty()
                && matches!(role, SemanticRole::Link | SemanticRole::Literal))
        {
            roles.0 |= 1 << *role as u8;
        }
    }
    for role in &mark.semantics {
        if mark.kind != MarkKind::Decoration
            || matches!(role, SemanticRole::Link | SemanticRole::Literal)
        {
            roles.0 |= 1 << *role as u8;
        }
    }
    if mark.kind != MarkKind::Decoration {
        let kind = match mark.kind {
            MarkKind::Bold => Some(SemanticRole::Bold),
            MarkKind::Italic => Some(SemanticRole::Italic),
            MarkKind::Link => Some(SemanticRole::Link),
            _ => None,
        };
        if let Some(role) = kind {
            roles.0 |= 1 << role as u8;
        }
    }
    roles
}
pub fn note_label_key(label: &str) -> Result<String, CoreError> {
    let normalized: String = label.nfkc().collect();
    let value = normalized.trim();
    let number = value.strip_suffix('.').unwrap_or(value);
    let number = if number.starts_with('[') && number.ends_with(']')
        || number.starts_with('(') && number.ends_with(')')
    {
        &number[1..number.len() - 1]
    } else {
        number
    };
    if (1..=4).contains(&number.len()) && number.bytes().all(|byte| byte.is_ascii_digit()) {
        return Ok(number.to_owned());
    }
    let mut chars = value.chars();
    if let Some(first) = chars.next() {
        let mut count = 1;
        if matches!(first, '*' | '†' | '‡' | '§' | '¶')
            && chars.all(|ch| {
                count += 1;
                ch == first
            })
            && count <= 3
        {
            return Ok(value.to_owned());
        }
    }
    Err(invalid(
        "invalid_note_label",
        "rich note reference requires a short original label",
    ))
}
pub(crate) fn local_tag(tag: &str) -> Cow<'_, str> {
    let local = tag.rsplit('}').next().unwrap_or(tag);
    if local
        .bytes()
        .all(|byte| byte.is_ascii() && !byte.is_ascii_uppercase())
    {
        Cow::Borrowed(local)
    } else {
        Cow::Owned(local.to_lowercase())
    }
}
pub(crate) fn invalid(code: &'static str, detail: &str) -> CoreError {
    CoreError::new(CoreErrorKind::RichContract, code, detail)
}
