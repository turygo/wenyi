#![forbid(unsafe_code)]
use sha2::{Digest, Sha256};
use std::path::Path;
use wenyi_book::{
    archive::{Archive, MemberKind},
    package::{TocKind, load_package},
};

#[test]
fn real_r0_inventory_package_order_and_source_bytes_are_preserved() {
    let input = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/migration/epub/structured-v1.epub");
    let before = std::fs::read(&input).unwrap();
    let mut archive = Archive::open(&input).unwrap();
    assert_eq!(
        archive.source_bytes_sha256(),
        "c4b572020d268eb64fd02dd99eeed7408bd4904ffb8fd1e95b2c27a37fac2a10"
    );
    let expected = [
        (
            "mimetype",
            20,
            0,
            "e468e350d1143eb648f60c7b0bd6031101ec0544a361ca74ecef256ac901f48b",
        ),
        (
            "META-INF/container.xml",
            243,
            8,
            "869b0b198afbc0f3e16d3b3100973126bb9827bf6f80814140482150f96d3c95",
        ),
        (
            "OEBPS/content.opf",
            942,
            8,
            "5d38a70043e680ffcc4454b2d5545e038490b965f121a495bda93a5fd7fd26e4",
        ),
        (
            "OEBPS/nav.xhtml",
            371,
            8,
            "bcf8d7bf8bed2ccd6c36db197c83bd87a3a4f5affa8edd2991a89f963b5f5c93",
        ),
        (
            "OEBPS/toc.ncx",
            417,
            8,
            "c57ce634a3991262f741a529e0b3f60d1b1b7e6883f1f737bbbfbfd374f35444",
        ),
        (
            "OEBPS/style.css",
            29,
            8,
            "731b254b29bb9e08af36ac9dc85425fbc4405e241b515389c792d848c5b128da",
        ),
        (
            "OEBPS/images/map.png",
            70,
            8,
            "194bdb273fa55018b8e0e248714246a196283bab5b8cd20a1dedd0b74ec206fd",
        ),
        (
            "OEBPS/text/chapter-1.xhtml",
            1092,
            8,
            "b2872e917763f5a18c99d2224b8ce58307ec36a06b47788f244639732e29b91f",
        ),
        (
            "OEBPS/text/chapter-2.xhtml",
            444,
            8,
            "d8eff1593081c8185e692ea7682e817007359d9451c8638a74c2e3be564a4003",
        ),
    ];
    assert_eq!(archive.members().len(), expected.len());
    for (index, (name, size, method, hash)) in expected.into_iter().enumerate() {
        let (actual_name, member) = archive.members().get_index(index).unwrap();
        assert_eq!(actual_name, name);
        assert_eq!(member.index, index);
        assert_eq!(member.kind, MemberKind::File);
        assert_eq!(member.expanded_size, size);
        assert_eq!(member.compression_method, method);
        assert_eq!(member.sha256, hash);
        assert_eq!(
            format!("{:x}", Sha256::digest(archive.read_resource(name).unwrap())),
            hash
        );
    }
    let package = load_package(&mut archive, &input).unwrap();
    assert_eq!(package.opf_path, "OEBPS/content.opf");
    assert_eq!(package.title, "Harbor Notes");
    assert_eq!(package.spine_ids, ["chapter-one", "chapter-two"]);
    assert_eq!(
        package.reading_order,
        ["OEBPS/text/chapter-1.xhtml", "OEBPS/text/chapter-2.xhtml"]
    );
    assert_eq!(
        package.content_paths,
        [
            "OEBPS/nav.xhtml",
            "OEBPS/text/chapter-1.xhtml",
            "OEBPS/text/chapter-2.xhtml"
        ]
    );
    assert_eq!(
        package
            .toc_paths
            .iter()
            .map(|toc| (&*toc.path, toc.kind))
            .collect::<Vec<_>>(),
        [
            ("OEBPS/nav.xhtml", TocKind::Nav),
            ("OEBPS/toc.ncx", TocKind::Ncx)
        ]
    );
    archive.finish().unwrap();
    assert_eq!(std::fs::read(input).unwrap(), before);
}
