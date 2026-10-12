use crate::source::{NodeId, NodeKind, SourceTree};
use cssparser::{CowRcStr, ParseError, ToCss};
use precomputed_hash::PrecomputedHash;
use selectors::attr::{AttrSelectorOperation, CaseSensitivity, NamespaceConstraint};
use selectors::bloom::BloomFilter;
use selectors::context::{
    MatchingContext, MatchingForInvalidation, MatchingMode, NeedsSelectorFlags, QuirksMode,
    SelectorCaches,
};
use selectors::matching::{ElementSelectorFlags, matches_selector};
use selectors::parser::{
    NonTSPseudoClass, ParseRelative, Parser, PseudoElement, SelectorImpl, SelectorList,
    SelectorParseErrorKind,
};
use selectors::{Element, OpaqueElement};
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Atom(String, u32);
impl From<&str> for Atom {
    fn from(value: &str) -> Self {
        let hash = value.bytes().fold(2166136261, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(16777619)
        });
        Self(value.into(), hash)
    }
}
impl Default for Atom {
    fn default() -> Self {
        Self::from("")
    }
}
impl AsRef<str> for Atom {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
impl std::borrow::Borrow<str> for Atom {
    fn borrow(&self) -> &str {
        &self.0
    }
}
impl PrecomputedHash for Atom {
    fn precomputed_hash(&self) -> u32 {
        self.1
    }
}
impl ToCss for Atom {
    fn to_css<W: fmt::Write>(&self, dest: &mut W) -> fmt::Result {
        cssparser::serialize_identifier(&self.0, dest)
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Pseudo(Atom);
impl ToCss for Pseudo {
    fn to_css<W: fmt::Write>(&self, dest: &mut W) -> fmt::Result {
        self.0.to_css(dest)
    }
}
impl NonTSPseudoClass for Pseudo {
    fn is_active_or_hover(&self) -> bool {
        matches!(self.0.as_ref(), "active" | "hover")
    }
    fn is_user_action_state(&self) -> bool {
        true
    }
}
impl PseudoElement for Pseudo {}
#[derive(Clone, Debug)]
pub(super) struct Standard;
impl SelectorImpl for Standard {
    type ExtraMatchingData<'a> = ();
    type AttrValue = Atom;
    type Identifier = Atom;
    type LocalName = Atom;
    type NamespaceUrl = Atom;
    type NamespacePrefix = Atom;
    type BorrowedNamespaceUrl = str;
    type BorrowedLocalName = str;
    type NonTSPseudoClass = Pseudo;
    type PseudoElement = Pseudo;
}
struct StandardParser<'a> {
    namespaces: &'a HashMap<String, String>,
    unknown: Cell<bool>,
    pseudo: Cell<bool>,
}
impl<'i> Parser<'i> for StandardParser<'_> {
    type Impl = Standard;
    type Error = SelectorParseErrorKind;
    fn parse_nth_child_of(&self) -> bool {
        true
    }
    fn parse_is_and_where(&self) -> bool {
        true
    }
    fn parse_has(&self) -> bool {
        true
    }
    fn allow_forgiving_selectors(&self) -> bool {
        false
    }
    fn parse_non_ts_pseudo_class(
        &self,
        name: CowRcStr<'i>,
    ) -> Result<Pseudo, ParseError<Self::Error>> {
        self.unknown.set(true);
        Ok(Pseudo(Atom::from(name.as_ref())))
    }
    fn parse_pseudo_element(&self, name: CowRcStr<'i>) -> Result<Pseudo, ParseError<Self::Error>> {
        self.pseudo.set(true);
        Ok(Pseudo(Atom::from(name.as_ref())))
    }
    fn default_namespace(&self) -> Option<Atom> {
        self.namespaces
            .get("")
            .map(|value| Atom::from(value.as_str()))
    }
    fn namespace_for_prefix(&self, prefix: &Atom) -> Option<Atom> {
        self.namespaces
            .get(prefix.as_ref())
            .map(|value| Atom::from(value.as_str()))
    }
}
pub(super) struct Selection {
    list: Option<SelectorList<Standard>>,
    unknown: bool,
    pseudo: bool,
}
impl Selection {
    pub(super) fn parse(text: &str, namespaces: &HashMap<String, String>) -> Self {
        let parser = StandardParser {
            namespaces,
            unknown: Cell::new(false),
            pseudo: Cell::new(false),
        };
        let mut input = cssparser::Parser::new(text);
        input.set_nested_block_limit(64);
        let list = SelectorList::parse(&parser, &mut input, ParseRelative::No).ok();
        let pseudo = list.as_ref().is_some_and(|list| {
            list.slice()
                .iter()
                .all(|selector| selector.has_pseudo_element())
        });
        Self {
            unknown: parser.unknown.get() || list.is_none(),
            pseudo,
            list,
        }
    }
    pub(super) fn specificity(&self, tree: &SourceTree, node: NodeId) -> Result<Option<u32>, ()> {
        if self.pseudo {
            return Ok(None);
        }
        if self.unknown {
            return Err(());
        }
        let mut caches = SelectorCaches::default();
        let mut context = MatchingContext::new(
            MatchingMode::Normal,
            None,
            &mut caches,
            QuirksMode::NoQuirks,
            NeedsSelectorFlags::No,
            MatchingForInvalidation::No,
        );
        let element = Dom { tree, node };
        Ok(self.list.as_ref().and_then(|list| {
            list.slice()
                .iter()
                .filter(|selector| {
                    !selector.has_pseudo_element()
                        && matches_selector(selector, 0, None, &element, &mut context)
                })
                .map(|selector| selector.specificity())
                .max()
        }))
    }
}
#[derive(Clone, Debug)]
struct Dom<'a> {
    tree: &'a SourceTree,
    node: NodeId,
}
impl<'a> Dom<'a> {
    fn at(&self, node: NodeId) -> Self {
        Self {
            tree: self.tree,
            node,
        }
    }
    fn sibling(&self, next: bool) -> Option<Self> {
        let parent = self.parent_element()?;
        let mut children = self.tree.element_children(parent.node);
        if next {
            children.find(|node| *node == self.node)?;
            children.next().map(|node| self.at(node))
        } else {
            let mut previous = None;
            for node in children {
                if node == self.node {
                    return previous.map(|node| self.at(node));
                }
                previous = Some(node);
            }
            None
        }
    }
}
impl Element for Dom<'_> {
    type Impl = Standard;
    fn opaque(&self) -> OpaqueElement {
        OpaqueElement::new(&self.tree.nodes[self.node.index()])
    }
    fn parent_element(&self) -> Option<Self> {
        self.tree
            .node(self.node)?
            .parent
            .filter(|node| self.tree.local_name(*node).is_some())
            .map(|node| self.at(node))
    }
    fn parent_node_is_shadow_root(&self) -> bool {
        false
    }
    fn containing_shadow_host(&self) -> Option<Self> {
        None
    }
    fn is_pseudo_element(&self) -> bool {
        false
    }
    fn prev_sibling_element(&self) -> Option<Self> {
        self.sibling(false)
    }
    fn next_sibling_element(&self) -> Option<Self> {
        self.sibling(true)
    }
    fn first_element_child(&self) -> Option<Self> {
        self.tree
            .element_children(self.node)
            .next()
            .map(|node| self.at(node))
    }
    fn is_html_element_in_html_document(&self) -> bool {
        false
    }
    fn has_local_name(&self, name: &str) -> bool {
        self.tree.local_name(self.node) == Some(name)
    }
    fn has_namespace(&self, namespace: &str) -> bool {
        matches!(&self.tree.nodes[self.node.index()].kind, NodeKind::Element(element) if element.name.namespace.as_deref().unwrap_or("") == namespace)
    }
    fn is_same_type(&self, other: &Self) -> bool {
        match (
            &self.tree.nodes[self.node.index()].kind,
            &other.tree.nodes[other.node.index()].kind,
        ) {
            (NodeKind::Element(left), NodeKind::Element(right)) => {
                left.name.local == right.name.local && left.name.namespace == right.name.namespace
            }
            _ => false,
        }
    }
    fn attr_matches(
        &self,
        namespace: &NamespaceConstraint<&Atom>,
        name: &Atom,
        operation: &AttrSelectorOperation<&Atom>,
    ) -> bool {
        let NodeKind::Element(element) = &self.tree.nodes[self.node.index()].kind else {
            return false;
        };
        element.attributes.iter().any(|attribute| {
            attribute.name.local == name.as_ref()
                && match namespace {
                    NamespaceConstraint::Any => true,
                    NamespaceConstraint::Specific(ns) => {
                        attribute.name.namespace.as_deref().unwrap_or("") == ns.as_ref()
                    }
                }
                && operation.eval_str(&attribute.value)
        })
    }
    fn match_non_ts_pseudo_class(&self, _: &Pseudo, _: &mut MatchingContext<Standard>) -> bool {
        false
    }
    fn match_pseudo_element(&self, _: &Pseudo, _: &mut MatchingContext<Standard>) -> bool {
        false
    }
    fn apply_selector_flags(&self, _: ElementSelectorFlags) {}
    fn is_link(&self) -> bool {
        matches!(self.tree.local_name(self.node), Some("a" | "area"))
            && self.tree.attribute(self.node, "href").is_some()
    }
    fn is_html_slot_element(&self) -> bool {
        false
    }
    fn has_id(&self, id: &Atom, case: CaseSensitivity) -> bool {
        self.tree
            .attribute(self.node, "id")
            .is_some_and(|value| case.eq(value.as_bytes(), id.as_ref().as_bytes()))
    }
    fn has_class(&self, name: &Atom, case: CaseSensitivity) -> bool {
        self.tree
            .attribute(self.node, "class")
            .is_some_and(|value| {
                value
                    .split_ascii_whitespace()
                    .any(|value| case.eq(value.as_bytes(), name.as_ref().as_bytes()))
            })
    }
    fn has_custom_state(&self, _: &Atom) -> bool {
        false
    }
    fn imported_part(&self, _: &Atom) -> Option<Atom> {
        None
    }
    fn is_part(&self, _: &Atom) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        let node = &self.tree.nodes[self.node.index()];
        self.tree.element_children(self.node).next().is_none()
            && node.text.as_deref().unwrap_or("").is_empty()
            && node.children.iter().all(|child| {
                self.tree.nodes[child.index()]
                    .tail
                    .as_deref()
                    .unwrap_or("")
                    .is_empty()
            })
    }
    fn is_root(&self) -> bool {
        self.node == self.tree.root
    }
    fn add_element_unique_hashes(&self, _: &mut BloomFilter) -> bool {
        false
    }
}
