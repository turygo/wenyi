#![forbid(unsafe_code)]
#[path = "support/archive.rs"]
mod support;
use support::Temp;
use wenyi_book::{
    archive::Archive,
    error::BookErrorKind,
    package::{TocKind, load_package},
    references::resolve_href,
};
use zip::CompressionMethod;

fn package(items: &str, spine: &str, metadata: &str, guide: &str) -> Temp {
    let opf = format!(
        "<package><metadata>{metadata}</metadata><manifest>{items}</manifest>{spine}{guide}</package>"
    );
    Temp::new(&support::zip(
        &[
            (
                "META-INF/container.xml",
                b"<container><rootfile full-path=' OPS/book.opf '/></container>",
            ),
            ("OPS/book.opf", opf.as_bytes()),
            ("OPS/a", b"<html/>"),
            ("OPS/b.HTML", b"<html/>"),
            ("OPS/nav page.ncx", b"<html/>"),
            ("OPS/toc.ncx", b"<ncx/>"),
            ("OPS/second.ncx", b"<ncx/>"),
            ("OPS/note+one.xhtml", b"<html/>"),
        ],
        CompressionMethod::Deflated,
    ))
}
#[test]
fn declaration_evidence_media_orders_toc_overwrite_and_guides() {
    let temp = package(
        "<item id='a' href='a' media-type='text/html' extra=' original '/><item id='b' href='b.HTML'/><item id='nav' href='nav%20page.ncx' media-type='application/xhtml+xml' properties='nav'/><item id='first' href='toc.ncx' media-type='application/x-dtbncx+xml'/><item id='preferred' href='second.ncx' media-type='application/x-dtbncx+xml'/><item id='notes' href='note+one.xhtml'/><item id='missing' href='absent.xhtml'/><item id='external' href='https://example.org/x'/><item id='escape' href='../../a'/>",
        "<spine toc='preferred'><itemref idref='b'/><itemref idref='a'/><itemref idref='b'/></spine>",
        "<title> \u{2003} </title><title>  Actual title  </title>",
        "<guide><reference type='notes' title=' Missing hint ' href='missing.xhtml#n'/><reference href='https://example.org/'/></guide>",
    );
    let mut archive = Archive::open(&temp.path).unwrap();
    let model = load_package(&mut archive, &temp.path).unwrap();
    assert_eq!(model.title, "Actual title");
    assert_eq!(model.items.len(), 9);
    assert_eq!(model.resolved.len(), 6);
    assert_eq!(model.items[0].attrs.get("extra"), Some(" original "));
    assert_eq!(model.items[1].media, "");
    assert_eq!(model.resolved[1].media, "application/xhtml+xml");
    assert_eq!(model.items[6].path.as_deref(), Some("OPS/absent.xhtml"));
    for index in [7, 8] {
        assert_eq!(model.items[index].path, None);
    }
    assert_eq!(model.reading_order, ["OPS/b.HTML", "OPS/a"]);
    assert_eq!(
        model.content_paths,
        [
            "OPS/a",
            "OPS/b.HTML",
            "OPS/nav page.ncx",
            "OPS/note+one.xhtml"
        ]
    );
    assert_eq!(
        model
            .toc_paths
            .iter()
            .map(|item| item.path.as_str())
            .collect::<Vec<_>>(),
        ["OPS/nav page.ncx", "OPS/second.ncx", "OPS/toc.ncx"]
    );
    assert_eq!(model.toc_paths[0].kind, TocKind::Nav);
    assert_eq!(model.toc_paths[1].kind, TocKind::Ncx);
    assert_eq!(model.guide_entries.len(), 1);
    assert_eq!(model.guide_entries[0].resource_href, "OPS/missing.xhtml");
    archive.finish().unwrap();
    let temp = package(
        "<item id='a' href='a' media-type='text/html'/><item href='toc.ncx' media-type='text/html' properties='nav'/><item href='toc.ncx' media-type='application/x-dtbncx+xml'/>",
        "<spine><itemref idref='a'/></spine>",
        "",
        "",
    );
    let model = load_package(&mut Archive::open(&temp.path).unwrap(), &temp.path).unwrap();
    assert_eq!(model.toc_paths.len(), 1);
    assert_eq!(model.toc_paths[0].kind, TocKind::Ncx);
    assert_eq!(model.title, "fallback");
}
#[test]
fn required_spine_unresolved_duplicate_nonhtml_and_empty_reject() {
    for (items, spine, code) in [
        (
            "<item id='a' href='missing.xhtml'/>",
            "<spine><itemref idref='a'/></spine>",
            "invalid_spine",
        ),
        (
            "<item id='a' href='a' media-type='text/html'/><item id='a' href='b.HTML'/>",
            "<spine><itemref idref='a'/></spine>",
            "invalid_spine",
        ),
        (
            "<item id='a' href='b.HTML' media-type='image/png'/>",
            "<spine><itemref idref='a'/></spine>",
            "invalid_spine",
        ),
        (
            "<item id='a' href='a'/>",
            "<spine><itemref idref='a'/></spine>",
            "invalid_spine",
        ),
        (
            "<item id='a' href='b.HTML'/>",
            "<spine/>",
            "missing_content_spine",
        ),
        ("<item id='a' href='b.HTML'/>", "", "missing_content_spine"),
    ] {
        let temp = package(items, spine, "", "");
        let error = load_package(&mut Archive::open(&temp.path).unwrap(), &temp.path).unwrap_err();
        assert_eq!(error.kind, BookErrorKind::Package);
        assert_eq!(error.code, code);
    }
}
#[test]
fn required_container_rootfile_and_xml_errors_are_typed() {
    for (container, code) in [
        ("<container/>", "invalid_rootfile"),
        (
            "<container><rootfile full-path='OPS/book.opf'/><rootfile full-path='OPS/book.opf'/></container>",
            "invalid_rootfile",
        ),
        (
            "<container><rootfile full-path='../book.opf'/></container>",
            "invalid_rootfile",
        ),
        (
            "<container><rootfile full-path='OPS/%62ook.opf'/></container>",
            "invalid_rootfile",
        ),
    ] {
        let temp = Temp::new(&support::zip(
            &[
                ("META-INF/container.xml", container.as_bytes()),
                ("OPS/book.opf", b"<package/>"),
            ],
            CompressionMethod::Stored,
        ));
        let error = load_package(&mut Archive::open(&temp.path).unwrap(), &temp.path).unwrap_err();
        assert_eq!(error.kind, BookErrorKind::Package);
        assert_eq!(error.code, code);
    }
    for (name, bytes, kind) in [
        ("other", b"x".as_slice(), BookErrorKind::Package),
        (
            "META-INF/container.xml",
            b"<broken".as_slice(),
            BookErrorKind::XmlSyntax,
        ),
    ] {
        let temp = Temp::new(&support::zip(&[(name, bytes)], CompressionMethod::Stored));
        assert_eq!(
            load_package(&mut Archive::open(&temp.path).unwrap(), &temp.path)
                .unwrap_err()
                .kind,
            kind
        );
    }
}
#[test]
fn local_reference_mapping_and_external_classification() {
    for (href, resource, fragment, external, query) in [
        (
            "../text/a%20b.xhtml#n%20x",
            "text/a b.xhtml",
            "n x",
            false,
            false,
        ),
        (
            "note+one.xhtml#n+1",
            "OPS/note+one.xhtml",
            "n+1",
            false,
            false,
        ),
        ("/text/a.xhtml", "text/a.xhtml", "", false, false),
        ("#anchor", "OPS/book.opf", "anchor", false, false),
        ("x?", "OPS/x", "", false, false),
        ("x?q=1", "OPS/x", "", false, true),
        ("//example.org/a", "", "", true, false),
        ("mailto:a@example.org", "", "", true, false),
        ("//[::1]/x", "", "", true, false),
    ] {
        let reference = resolve_href("OPS/book.opf", href).unwrap();
        assert_eq!(
            (
                &*reference.resource_href,
                &*reference.fragment,
                reference.external,
                reference.has_query
            ),
            (resource, fragment, external, query)
        );
    }
    assert_eq!(resolve_href("", "").unwrap().target_key(), "");
    assert_eq!(resolve_href("a", "#x").unwrap().target_key(), "a#x");
}

#[test]
fn local_references_cannot_escape_or_introduce_unsafe_member_paths() {
    for href in ["../../outside", "%2e%2e/%2e%2e/outside", "a%5cb", "a%00b"] {
        assert_eq!(
            resolve_href("OPS/book.opf", href).unwrap_err().kind,
            BookErrorKind::Reference
        );
    }
}
