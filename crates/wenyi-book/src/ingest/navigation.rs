#![forbid(unsafe_code)]
use super::tree;
use crate::{
    archive::Archive,
    document::{AdmissionDiagnostic, Landmark, NavigationEntry},
    error::{BookError, BookErrorKind},
    package::{PackageModel, TocKind},
    references::resolve_href,
    source::{NodeId, SourceTree, XmlPurpose, parse_xml},
};

pub(super) fn load(
    archive: &mut Archive,
    package: &PackageModel,
    diagnostics: &mut Vec<AdmissionDiagnostic>,
) -> Result<(Vec<NavigationEntry>, Vec<Landmark>), BookError> {
    let mut entries = Vec::new();
    let mut landmarks: Vec<_> = package
        .guide_entries
        .iter()
        .map(|guide| Landmark {
            resource_href: guide.resource_href.clone(),
            semantic_type: guide.entry_type.clone(),
            role: String::new(),
            label: guide.title.clone(),
        })
        .collect();
    for toc in &package.toc_paths {
        let bytes = archive.read_resource(&toc.path)?;
        let parsed = parse_xml(
            &bytes,
            if toc.kind == TocKind::Nav {
                XmlPurpose::Nav
            } else {
                XmlPurpose::Ncx
            },
        );
        let admitted = parsed.and_then(|parsed| {
            let local = if toc.kind == TocKind::Nav {
                nav(&parsed.tree, &toc.path)
            } else {
                ncx(&parsed.tree, &toc.path)
            }?;
            let hints = if toc.kind == TocKind::Nav {
                nav_landmarks(&parsed.tree, &toc.path)?
            } else {
                Vec::new()
            };
            Ok((local, hints))
        });
        match admitted {
            Ok((local, hints)) => {
                entries.extend(local);
                landmarks.extend(hints);
            }
            Err(error)
                if matches!(
                    error.kind,
                    BookErrorKind::XmlSyntax | BookErrorKind::Reference
                ) =>
            {
                diagnostics.push(AdmissionDiagnostic {
                    code: error.code.to_owned(),
                    resource: toc.path.clone(),
                    detail: error.detail,
                })
            }
            Err(error) => return Err(error.with_resource(&toc.path)),
        }
    }
    Ok((entries, landmarks))
}
fn entry(
    tree: &SourceTree,
    id: NodeId,
    toc: &str,
    depth: usize,
    parent: Option<String>,
    content: (String, String),
    index: usize,
) -> Result<NavigationEntry, BookError> {
    let (label, raw_href) = content;
    let resolved = if raw_href.is_empty() {
        None
    } else {
        Some(resolve_href(toc, &raw_href)?)
    };
    let node_id = tree::attr(tree, id, "id");
    Ok(NavigationEntry {
        toc_path: toc.to_owned(),
        entry_id: format!("{toc}:{index}"),
        node_id: (!node_id.is_empty()).then(|| node_id.to_owned()),
        parent_entry_id: parent,
        depth,
        label,
        raw_href,
        resource_href: resolved
            .as_ref()
            .map_or_else(String::new, |value| value.resource_href.clone()),
        fragment: resolved
            .as_ref()
            .map_or_else(String::new, |value| value.fragment.clone()),
        external: resolved.is_some_and(|value| value.external),
        boundary_segment: None,
    })
}
fn child(tree: &SourceTree, id: NodeId, tag: &str) -> Option<NodeId> {
    tree.element_children(id)
        .find(|&child| tree.local_name(child) == Some(tag))
}
fn nav(tree: &SourceTree, toc: &str) -> Result<Vec<NavigationEntry>, BookError> {
    let all = tree::nodes(tree, tree.root);
    let navs: Vec<_> = all
        .iter()
        .copied()
        .filter(|&id| tree.local_name(id) == Some("nav"))
        .collect();
    let typed: Vec<_> = navs
        .iter()
        .copied()
        .filter(|&id| tree::token(tree, id, "type", "toc"))
        .collect();
    let scopes = if !typed.is_empty() {
        typed
    } else {
        navs.first()
            .copied()
            .map_or_else(|| vec![tree.root], |id| vec![id])
    };
    let mut result = Vec::new();
    for scope in scopes {
        let ol = child(tree, scope, "ol").or_else(|| {
            tree::nodes(tree, scope)
                .into_iter()
                .find(|&id| tree.local_name(id) == Some("ol"))
        });
        if let Some(ol) = ol {
            for li in tree
                .element_children(ol)
                .filter(|&id| tree.local_name(id) == Some("li"))
            {
                visit_li(tree, li, toc, 0, None, &mut result)?;
            }
        }
    }
    Ok(result)
}
fn visit_li(
    tree: &SourceTree,
    li: NodeId,
    toc: &str,
    depth: usize,
    parent: Option<String>,
    result: &mut Vec<NavigationEntry>,
) -> Result<(), BookError> {
    let mut current = parent;
    if let Some(label) = child(tree, li, "a").or_else(|| child(tree, li, "span")) {
        let value = entry(
            tree,
            li,
            toc,
            depth,
            current.clone(),
            (
                tree::collapse(&tree.itertext(label)?),
                if tree.local_name(label) == Some("a") {
                    tree::attr(tree, label, "href").to_owned()
                } else {
                    String::new()
                },
            ),
            result.len(),
        )?;
        current = Some(value.entry_id.clone());
        result.push(value);
    }
    if let Some(ol) = child(tree, li, "ol") {
        for nested in tree
            .element_children(ol)
            .filter(|&id| tree.local_name(id) == Some("li"))
        {
            visit_li(tree, nested, toc, depth + 1, current.clone(), result)?;
        }
    }
    Ok(())
}
fn ncx(tree: &SourceTree, toc: &str) -> Result<Vec<NavigationEntry>, BookError> {
    let mut result = Vec::new();
    if let Some(map) = tree::nodes(tree, tree.root)
        .into_iter()
        .find(|&id| tree.local_name(id) == Some("navMap"))
    {
        for point in tree
            .element_children(map)
            .filter(|&id| tree.local_name(id) == Some("navPoint"))
        {
            visit_point(tree, point, toc, 0, None, &mut result)?;
        }
    }
    Ok(result)
}
fn visit_point(
    tree: &SourceTree,
    point: NodeId,
    toc: &str,
    depth: usize,
    parent: Option<String>,
    result: &mut Vec<NavigationEntry>,
) -> Result<(), BookError> {
    let label = child(tree, point, "navLabel")
        .and_then(|id| child(tree, id, "text"))
        .map(|id| tree.itertext(id))
        .transpose()?
        .unwrap_or_default();
    let href = child(tree, point, "content").map_or("", |id| tree::attr(tree, id, "src"));
    let value = entry(
        tree,
        point,
        toc,
        depth,
        parent,
        (tree::collapse(&label), href.to_owned()),
        result.len(),
    )?;
    let current = value.entry_id.clone();
    result.push(value);
    for nested in tree
        .element_children(point)
        .filter(|&id| tree.local_name(id) == Some("navPoint"))
    {
        visit_point(tree, nested, toc, depth + 1, Some(current.clone()), result)?;
    }
    Ok(())
}
fn nav_landmarks(tree: &SourceTree, toc: &str) -> Result<Vec<Landmark>, BookError> {
    let mut result = Vec::new();
    for nav in tree::nodes(tree, tree.root).into_iter().filter(|&id| {
        tree.local_name(id) == Some("nav") && tree::token(tree, id, "type", "landmarks")
    }) {
        for link in tree::nodes(tree, nav)
            .into_iter()
            .filter(|&id| tree.local_name(id) == Some("a"))
        {
            let href = tree::attr(tree, link, "href");
            if href.is_empty() {
                continue;
            }
            let target = resolve_href(toc, href)?;
            if target.external {
                continue;
            }
            result.push(Landmark {
                resource_href: target.resource_href,
                semantic_type: tree::attr(tree, link, "type").to_owned(),
                role: tree::attr(tree, link, "role").to_owned(),
                label: tree::collapse(&tree.itertext(link)?),
            });
        }
    }
    Ok(result)
}
