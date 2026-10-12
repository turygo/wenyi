#![forbid(unsafe_code)]
use crate::error::{BookError, BookErrorKind};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct NodeId(u32);
impl NodeId {
    pub fn new(index: usize) -> Result<Self, BookError> {
        Ok(Self(
            u32::try_from(index).map_err(|_| safety("source node index overflow"))?,
        ))
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedName {
    pub namespace: Option<String>,
    pub prefix: Option<String>,
    pub local: String,
}
impl QualifiedName {
    pub fn expanded(&self) -> String {
        self.namespace.as_ref().map_or_else(
            || self.local.clone(),
            |uri| format!("{{{uri}}}{}", self.local),
        )
    }
    pub fn local(&self) -> &str {
        &self.local
    }
    fn matches_expanded(&self, name: &str) -> bool {
        match &self.namespace {
            None => self.local == name,
            Some(uri) => {
                name.strip_prefix('{')
                    .and_then(|rest| rest.strip_prefix(uri.as_str()))
                    .and_then(|rest| rest.strip_prefix('}'))
                    == Some(self.local.as_str())
            }
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    pub name: QualifiedName,
    pub value: String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceDecl {
    pub prefix: Option<String>,
    pub uri: String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementData {
    pub name: QualifiedName,
    pub attributes: Vec<Attribute>,
    pub namespaces: Vec<NamespaceDecl>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Element(ElementData),
    Comment(String),
    ProcessingInstruction { target: String, data: String },
    EntityReference(String),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceNode {
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub text: Option<String>,
    pub tail: Option<String>,
    pub(crate) text_cdata: Vec<std::ops::Range<usize>>,
    pub(crate) tail_cdata: Vec<std::ops::Range<usize>>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTree {
    pub nodes: Vec<SourceNode>,
    pub root: NodeId,
    pub before_root: Vec<NodeId>,
    pub after_root: Vec<NodeId>,
}
impl SourceTree {
    pub fn node(&self, id: NodeId) -> Option<&SourceNode> {
        self.nodes.get(id.index())
    }
    pub fn element_children(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.node(id)
            .into_iter()
            .flat_map(|node| node.children.iter().copied())
            .filter(|child| {
                self.node(*child)
                    .is_some_and(|node| matches!(node.kind, NodeKind::Element(_)))
            })
    }
    pub fn local_name(&self, id: NodeId) -> Option<&str> {
        match &self.node(id)?.kind {
            NodeKind::Element(element) => Some(&element.name.local),
            _ => None,
        }
    }
    pub fn attribute(&self, id: NodeId, name: &str) -> Option<&str> {
        match &self.node(id)?.kind {
            NodeKind::Element(element) => element
                .attributes
                .iter()
                .find(|attribute| attribute.name.matches_expanded(name))
                .map(|attribute| attribute.value.as_str()),
            _ => None,
        }
    }
    pub fn resolve_element_path(&self, root: NodeId, path: &[i32]) -> Result<NodeId, BookError> {
        let mut current = root;
        for &step in path {
            let node = self
                .node(current)
                .ok_or_else(|| source("invalid locator node"))?;
            current = if step >= 0 {
                self.element_children(current).nth(step as usize)
            } else {
                let index = usize::try_from(-i64::from(step) - 1)
                    .map_err(|_| source("locator step overflow"))?;
                node.children.get(index).copied().filter(|child| {
                    self.node(*child)
                        .is_some_and(|node| !matches!(node.kind, NodeKind::Element(_)))
                })
            }
            .ok_or_else(|| source("source locator does not exist"))?;
        }
        self.node(current)
            .ok_or_else(|| source("invalid locator root"))?;
        Ok(current)
    }
    pub fn itertext(&self, root: NodeId) -> Result<String, BookError> {
        let mut output = String::new();
        let mut pending = vec![(root, false, 0_usize)];
        let mut visited = 0;
        while let Some((id, tail, depth)) = pending.pop() {
            let node = self
                .node(id)
                .ok_or_else(|| source("invalid text traversal node"))?;
            if tail {
                if let Some(text) = &node.tail {
                    output.push_str(text);
                }
            } else {
                visited += 1;
                if visited > self.nodes.len() || depth > self.nodes.len() {
                    return Err(safety("text traversal cycle detected"));
                }
                if matches!(node.kind, NodeKind::Element(_)) {
                    if let Some(text) = &node.text {
                        output.push_str(text);
                    }
                    for child in node.children.iter().rev() {
                        pending.push((*child, true, depth + 1));
                        pending.push((*child, false, depth + 1));
                    }
                }
            }
        }
        Ok(output)
    }
}
pub(crate) fn safety(detail: impl Into<String>) -> BookError {
    BookError::new(BookErrorKind::XmlSafety, "xml-safety", detail)
}
pub(crate) fn source(detail: impl Into<String>) -> BookError {
    BookError::new(BookErrorKind::Source, "source-tree", detail)
}
