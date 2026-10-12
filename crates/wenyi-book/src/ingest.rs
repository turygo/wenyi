#![forbid(unsafe_code)]
mod chapters;
mod extraction;
mod mirrored;
mod navigation;
mod notes;
mod rich;
mod tree;

use crate::{
    archive::Archive,
    document::{AdmissionDiagnostic, ContentResource, SourceDocument},
    error::{BookError, BookErrorKind},
    package::load_package,
    source::parse_source_markup,
};
use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};
use wenyi_core::SourceIdentity;

pub fn admit_epub(input: &Path) -> Result<SourceDocument, BookError> {
    if !input
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("epub"))
    {
        return Err(source_error(
            "source_format",
            "source path must have an EPUB extension",
        ));
    }
    let mut archive = Archive::open(input)?;
    let package = load_package(&mut archive, input)?;
    let mut diagnostics = Vec::new();
    let (mut navigation, landmarks) = navigation::load(&mut archive, &package, &mut diagnostics)?;
    let mut resources = Vec::with_capacity(package.content_paths.len());
    for href in &package.content_paths {
        let original_bytes = archive.read_resource(href)?;
        let parsed =
            parse_source_markup(&original_bytes).map_err(|error| error.with_resource(href))?;
        for diagnostic in &parsed.diagnostics {
            diagnostics.push(AdmissionDiagnostic {
                code: "source_parse".to_owned(),
                resource: href.clone(),
                detail: diagnostic.reason.clone(),
            });
        }
        resources.push(ContentResource {
            href: href.clone(),
            resource_sha256: archive.members()[href].sha256.clone(),
            original_bytes,
            parsed,
            title: String::new(),
            segment_ids: Vec::new(),
            fragment_anchors: BTreeMap::new(),
        });
    }
    let declared: HashSet<_> = package
        .toc_paths
        .iter()
        .map(|toc| toc.path.clone())
        .collect();
    let notes = notes::detect(&resources, &declared, &landmarks)?;
    let mut segments = Vec::new();
    for (index, resource) in resources.iter_mut().enumerate() {
        extraction::extract(resource, index, &notes, &mut segments)?;
    }
    let (chapters, canonical) = chapters::assemble(
        &package.reading_order,
        &resources,
        &segments,
        &mut navigation,
    );
    if chapters.is_empty() {
        return Err(source_error(
            "empty_source",
            "reading order has no usable source blocks",
        ));
    }
    mirrored::annotate(
        &resources,
        &mut segments,
        &navigation,
        &declared,
        canonical.as_deref(),
        &package.title,
    );
    let identity = SourceIdentity::new(
        archive.source_bytes_sha256().to_owned(),
        package.title.clone(),
        slug(&package.title),
    )
    .map_err(|error| source_error("source_identity", error.to_string()))?;
    crate::style_projection::enrich(&mut archive, &resources, &mut segments)?;
    let inventory = archive.finish()?;
    Ok(SourceDocument {
        identity,
        inventory,
        package,
        resources,
        segments,
        chapters,
        navigation,
        landmarks,
        notes,
        diagnostics,
    })
}
pub(super) fn source_error(code: &'static str, detail: impl Into<String>) -> BookError {
    BookError::new(BookErrorKind::Source, code, detail)
}
fn slug(title: &str) -> String {
    let mut value = String::with_capacity(title.len());
    let mut separator = false;
    for ch in title.chars() {
        if ch.is_alphanumeric() || matches!(ch, '_' | '-') {
            value.push(ch);
            separator = false;
        } else if !separator {
            value.push('_');
            separator = true;
        }
    }
    let value = value.trim_matches('_');
    if value.is_empty() {
        "book".to_owned()
    } else {
        value.to_owned()
    }
}
