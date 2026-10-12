#![forbid(unsafe_code)]
use super::{PackageModel, TocKind, TocResource, package_error};
use crate::{archive::guard::safe_name, error::BookError, references::resolve_href};
use indexmap::IndexMap;

pub(super) fn manifest_path(base: &str, href: &str) -> Option<String> {
    if href.is_empty() || href.contains('\0') {
        return None;
    }
    let reference = resolve_href(base, href).ok()?;
    if reference.external || !safe_name(&reference.resource_href) {
        None
    } else {
        Some(reference.resource_href)
    }
}
pub(super) fn orders(model: &mut PackageModel) -> Result<(), BookError> {
    for id in &model.spine_ids {
        let mut matches = model
            .resolved
            .iter()
            .filter(|item| !id.is_empty() && model.items[item.item_index].id == *id);
        let item = matches
            .next()
            .ok_or_else(|| package_error("invalid_spine", "spine ID has no resolved resource"))?;
        if matches.next().is_some() {
            return Err(package_error("invalid_spine", "spine ID is ambiguous"));
        }
        if !html(&item.media) {
            return Err(package_error("invalid_spine", "spine resource is not HTML"));
        }
        if !model.reading_order.contains(&item.path) {
            model.reading_order.push(item.path.clone());
        }
    }
    if model.reading_order.is_empty() {
        return Err(package_error(
            "missing_content_spine",
            "readable content spine is required",
        ));
    }
    let mut toc = IndexMap::new();
    for item in &model.resolved {
        if html(&item.media) && !model.content_paths.contains(&item.path) {
            model.content_paths.push(item.path.clone());
        }
        if model.items[item.item_index]
            .properties
            .split_whitespace()
            .any(|token| token == "nav")
        {
            toc.insert(
                item.path.clone(),
                if item.media == "application/x-dtbncx+xml" {
                    TocKind::Ncx
                } else {
                    TocKind::Nav
                },
            );
        }
    }
    for preferred in [true, false] {
        for item in &model.resolved {
            if item.media == "application/x-dtbncx+xml"
                && (!preferred || model.items[item.item_index].id == model.spine_toc)
            {
                toc.insert(item.path.clone(), TocKind::Ncx);
            }
        }
    }
    model.toc_paths = toc
        .into_iter()
        .map(|(path, kind)| TocResource { path, kind })
        .collect();
    Ok(())
}
fn html(media: &str) -> bool {
    matches!(media, "application/xhtml+xml" | "text/html")
}
