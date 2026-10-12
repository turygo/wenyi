#![forbid(unsafe_code)]
mod resolve;
use crate::archive::Archive;
use crate::error::{BookError, BookErrorKind};
use crate::source::{NodeId, NodeKind, SourceTree, XmlPurpose, parse_xml};
use std::path::Path;
use wenyi_core::rich::OrderedAttributes;

#[derive(Debug, Clone)]
pub struct ManifestItem {
    pub id: String,
    pub href: String,
    pub media: String,
    pub properties: String,
    pub fallback: String,
    pub media_overlay: String,
    pub attrs: OrderedAttributes,
    pub path: Option<String>,
}
#[derive(Debug, Clone)]
pub struct ResolvedManifestItem {
    pub item_index: usize,
    pub path: String,
    pub media: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TocKind {
    Nav,
    Ncx,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TocResource {
    pub path: String,
    pub kind: TocKind,
}
#[derive(Debug, Clone)]
pub struct GuideEntry {
    pub entry_type: String,
    pub title: String,
    pub raw_href: String,
    pub resource_href: String,
}
#[derive(Debug)]
pub struct PackageModel {
    pub opf_path: String,
    pub title: String,
    pub items: Vec<ManifestItem>,
    pub resolved: Vec<ResolvedManifestItem>,
    pub spine_ids: Vec<String>,
    pub spine_toc: String,
    pub reading_order: Vec<String>,
    pub content_paths: Vec<String>,
    pub toc_paths: Vec<TocResource>,
    pub guide_entries: Vec<GuideEntry>,
}

pub fn load_package(archive: &mut Archive, input: &Path) -> Result<PackageModel, BookError> {
    const CONTAINER: &str = "META-INF/container.xml";
    if !archive.members().contains_key(CONTAINER) {
        return Err(package_error(
            "missing_container",
            "container.xml is required",
        ));
    }
    let container = parse_xml(&archive.read_resource(CONTAINER)?, XmlPurpose::Container)
        .map_err(|error| error.with_resource(CONTAINER))?;
    let roots: Vec<_> = elements(&container.tree)
        .into_iter()
        .filter(|&id| container.tree.local_name(id) == Some("rootfile"))
        .collect();
    if roots.len() != 1 {
        return Err(package_error(
            "invalid_rootfile",
            "exactly one rootfile is required",
        ));
    }
    let opf_path = attr(&container.tree, roots[0], "full-path");
    if !crate::archive::guard::safe_name(&opf_path) || !archive.members().contains_key(&opf_path) {
        return Err(package_error(
            "invalid_rootfile",
            "rootfile is unsafe or absent",
        ));
    }
    let document = parse_xml(&archive.read_resource(&opf_path)?, XmlPurpose::Package)
        .map_err(|error| error.with_resource(&opf_path))?;
    let tree = &document.tree;
    let nodes = elements(tree);
    let mut model = PackageModel {
        opf_path,
        title: String::new(),
        items: Vec::new(),
        resolved: Vec::new(),
        spine_ids: Vec::new(),
        spine_toc: String::new(),
        reading_order: Vec::new(),
        content_paths: Vec::new(),
        toc_paths: Vec::new(),
        guide_entries: Vec::new(),
    };
    for &id in &nodes {
        if tree.local_name(id) == Some("title") && model.title.is_empty() {
            model.title = tree
                .node(id)
                .and_then(|node| node.text.as_deref())
                .unwrap_or("")
                .trim()
                .to_owned();
        }
        if tree.local_name(id) == Some("item") {
            add_item(&mut model, tree, id, archive);
        }
    }
    if model.title.is_empty() {
        model.title = input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
    }
    if let Some(&spine) = nodes
        .iter()
        .find(|&&id| tree.local_name(id) == Some("spine"))
    {
        model.spine_toc = attr(tree, spine, "toc");
        model.spine_ids = tree
            .element_children(spine)
            .filter(|&id| tree.local_name(id) == Some("itemref"))
            .map(|id| attr(tree, id, "idref"))
            .collect();
    }
    resolve::orders(&mut model)?;
    for &guide in nodes
        .iter()
        .filter(|&&id| tree.local_name(id) == Some("guide"))
    {
        for id in descendants(tree, guide) {
            if tree.local_name(id) != Some("reference") {
                continue;
            }
            let raw_href = attr(tree, id, "href");
            if let Some(resource_href) = resolve::manifest_path(&model.opf_path, &raw_href) {
                model.guide_entries.push(GuideEntry {
                    entry_type: attr(tree, id, "type"),
                    title: attr(tree, id, "title"),
                    raw_href,
                    resource_href,
                });
            }
        }
    }
    Ok(model)
}
fn add_item(model: &mut PackageModel, tree: &SourceTree, id: NodeId, archive: &Archive) {
    let href = attr(tree, id, "href");
    let media = attr(tree, id, "media-type");
    let path = resolve::manifest_path(&model.opf_path, &href);
    if let Some(target) = path
        .as_ref()
        .filter(|target| archive.members().contains_key(*target))
    {
        let effective = if media.is_empty()
            && [".xhtml", ".html", ".htm"].iter().any(|suffix| {
                target
                    .get(target.len().saturating_sub(suffix.len())..)
                    .is_some_and(|ending| ending.eq_ignore_ascii_case(suffix))
            }) {
            "application/xhtml+xml".to_owned()
        } else {
            media.clone()
        };
        model.resolved.push(ResolvedManifestItem {
            item_index: model.items.len(),
            path: target.clone(),
            media: effective,
        });
    }
    let attrs = match &tree.node(id).unwrap().kind {
        NodeKind::Element(element) => OrderedAttributes(
            element
                .attributes
                .iter()
                .map(|attribute| (attribute.name.expanded(), attribute.value.clone()))
                .collect(),
        ),
        _ => OrderedAttributes::default(),
    };
    model.items.push(ManifestItem {
        id: attr(tree, id, "id"),
        href,
        media,
        properties: attr(tree, id, "properties"),
        fallback: attr(tree, id, "fallback"),
        media_overlay: attr(tree, id, "media-overlay"),
        attrs,
        path,
    });
}
fn attr(tree: &SourceTree, id: NodeId, name: &str) -> String {
    tree.attribute(id, name).unwrap_or("").trim().to_owned()
}
fn elements(tree: &SourceTree) -> Vec<NodeId> {
    descendants(tree, tree.root)
}
fn descendants(tree: &SourceTree, root: NodeId) -> Vec<NodeId> {
    let mut pending = vec![root];
    let mut result = Vec::new();
    while let Some(id) = pending.pop() {
        if tree.local_name(id).is_some() {
            result.push(id);
        }
        if let Some(node) = tree.node(id) {
            pending.extend(node.children.iter().rev().copied());
        }
    }
    result
}
fn package_error(code: &'static str, detail: &str) -> BookError {
    BookError::new(BookErrorKind::Package, code, detail)
}
