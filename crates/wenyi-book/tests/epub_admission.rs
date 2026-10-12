#![forbid(unsafe_code)]
#[path = "support/archive.rs"]
mod support;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::PathBuf};
use wenyi_book::{document::SegmentKind, error::BookErrorKind, ingest::admit_epub};
use wenyi_core::{SlotField, rich::AtomKind};
use zip::CompressionMethod;

fn book(body: &str, nav: &str, ncx: &str, extra: &[(&str, &str)]) -> support::Temp {
    let mut items = String::from(
        "<item id='nav' href='nav.xhtml' media-type='application/xhtml+xml' properties='nav'/><item id='a' href='a.xhtml' media-type='application/xhtml+xml'/>",
    );
    if !ncx.is_empty() {
        items.push_str("<item id='ncx' href='toc.ncx' media-type='application/x-dtbncx+xml'/>");
    }
    for (index, (name, _)) in extra.iter().enumerate() {
        items.push_str(&format!(
            "<item id='extra{index}' href='{name}' media-type='application/xhtml+xml'/>"
        ));
    }
    let opf = format!(
        "<package><metadata><title>Test Book</title></metadata><manifest>{items}</manifest><spine toc='ncx'><itemref idref='a'/><itemref idref='a'/></spine></package>"
    );
    let markup = format!(
        "<html xmlns='http://www.w3.org/1999/xhtml' xmlns:epub='http://www.idpf.org/2007/ops'><head><title>Source</title></head><body>{body}</body></html>"
    );
    let mut entries = vec![
        (
            "META-INF/container.xml",
            b"<container><rootfile full-path='OPS/book.opf'/></container>".as_slice(),
        ),
        ("OPS/book.opf", opf.as_bytes()),
        ("OPS/a.xhtml", markup.as_bytes()),
        ("OPS/nav.xhtml", nav.as_bytes()),
    ];
    if !ncx.is_empty() {
        entries.push(("OPS/toc.ncx", ncx.as_bytes()));
    }
    let paths: Vec<_> = extra
        .iter()
        .map(|(name, _)| format!("OPS/{name}"))
        .collect();
    for (path, (_, content)) in paths.iter().zip(extra) {
        entries.push((path, content.as_bytes()));
    }
    support::Temp::new(&support::zip(&entries, CompressionMethod::Deflated))
}
fn nav(links: &str) -> String {
    format!(
        "<html><head><title>Contents</title></head><body><nav type='toc'><ol>{links}</ol></nav></body></html>"
    )
}
fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/migration")
}
#[test]
fn frozen_paid_source_slots_blocks_and_raw_rich_geometry_are_preserved() {
    let root = fixture();
    let input = root.join("epub/structured-v1.epub");
    let document = admit_epub(&input).unwrap();
    assert_eq!(document.identity.title, "Harbor Notes");
    assert_eq!(document.identity.run_slug, "Harbor_Notes");
    assert_eq!(
        document.identity.source_bytes_sha256,
        format!("{:x}", Sha256::digest(std::fs::read(&input).unwrap()))
    );
    assert_eq!(
        document.package.reading_order,
        ["OEBPS/text/chapter-1.xhtml", "OEBPS/text/chapter-2.xhtml"]
    );
    assert_eq!(
        document
            .resources
            .iter()
            .map(|resource| resource.href.as_str())
            .collect::<Vec<_>>(),
        [
            "OEBPS/nav.xhtml",
            "OEBPS/text/chapter-1.xhtml",
            "OEBPS/text/chapter-2.xhtml"
        ]
    );
    assert!(document.resources[0].segment_ids.is_empty());
    let actual: BTreeMap<_, _> = document
        .segments
        .iter()
        .map(|segment| (segment.anchor.as_str(), segment))
        .collect();
    let mut expected_anchors = Vec::new();
    for chapter_index in 0..2 {
        let path = root.join(format!("state/structured-epub-balanced-v1/run/state/Harbor_Notes/chapters_v2/ch{chapter_index}.json"));
        let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let chapter = &document.chapters[chapter_index];
        assert_eq!(chapter.index, chapter_index);
        assert_eq!(json!(chapter.title), saved["title"]);
        assert_eq!(
            chapter.toc_entry_id.as_deref(),
            Some(format!("OEBPS/nav.xhtml:{chapter_index}").as_str())
        );
        let mut ids = Vec::new();
        for saved_segment in saved["segments"].as_array().unwrap() {
            let anchor = saved_segment["anchor"].as_str().unwrap();
            expected_anchors.push(anchor.to_owned());
            let segment = actual[anchor];
            ids.push(
                document
                    .segments
                    .iter()
                    .position(|candidate| candidate.anchor == anchor)
                    .unwrap(),
            );
            assert_eq!(json!(segment.source), saved_segment["source"]);
            assert_eq!(json!(segment.kind.as_str()), saved_segment["kind"]);
            let state = &saved_segment["epub_state"];
            assert_eq!(json!(segment.block_path), state["block_path"]);
            assert_eq!(json!(segment.block_fingerprint), state["block_fingerprint"]);
            assert_eq!(
                json!(segment.slot_contract_sha256),
                state["slot_contract_sha256"]
            );
            let expected_slots = state["slots"].as_array().unwrap();
            assert_eq!(segment.slots.len(), expected_slots.len());
            for (slot, saved) in segment.slots.iter().zip(expected_slots) {
                assert_eq!(json!(slot.id), saved["id"]);
                assert_eq!(json!(slot.element_path), saved["element_path"]);
                assert_eq!(serde_json::to_value(slot.field).unwrap(), saved["field"]);
                assert_eq!(json!(slot.source_value), saved["source_value"]);
            }
            let resource = &document.resources[segment.resource_index];
            assert_eq!(json!(resource.resource_sha256), state["resource_sha256"]);
            assert_eq!(
                resource.resource_sha256,
                format!("{:x}", Sha256::digest(&resource.original_bytes))
            );
            let mut expected_rich = state["rich_source"].clone();
            let mut raw = serde_json::to_value(&segment.rich_source).unwrap();
            for mark in expected_rich["marks"].as_array_mut().unwrap() {
                mark["style_evidence"] = json!([]);
            }
            for mark in raw["marks"].as_array_mut().unwrap() {
                mark["style_evidence"] = json!([]);
            }
            assert_eq!(raw, expected_rich, "{anchor}");
            assert!(
                segment
                    .rich_source
                    .marks
                    .iter()
                    .all(|mark| mark.style_evidence.is_empty())
            );
        }
        assert_eq!(chapter.segment_ids, ids);
    }
    assert_eq!(
        actual.keys().copied().collect::<Vec<_>>(),
        expected_anchors
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    );
    assert_eq!(document.segments.len(), 17);
    assert_eq!(
        actual["tn1_4"].slots[0].source_value,
        "A🙂 reader keeps cafe\u{301} exact."
    );
    assert_eq!(
        actual["tn1_5"].rich_source.atoms[0].kind,
        AtomKind::Linebreak
    );
    assert_eq!(actual["tn1_7"].rich_source.atoms[0].label, "1");
    assert_eq!(document.notes.markers[0].target_resource_index, 2);
}
#[test]
fn nav_preorder_group_boundaries_and_missing_end_fragments_are_not_remapped() {
    let toc = nav(
        "<li id='group'><span>Part</span><ol><li><a href='a.xhtml#inside'>One</a></li><li><a href='a.xhtml#missing'>Missing</a></li><li><a href='a.xhtml#between'>Two</a></li><li><a href='a.xhtml#end'>End</a></li></ol></li>",
    );
    let temp = book(
        "<p>Preface</p><h1 id='one'>One <span id='inside'>heading</span></h1><p id='inside'>first</p><a id='between'/><h2>Two</h2><p>second</p><a id='end'/>",
        &toc,
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    assert_eq!(document.package.reading_order, ["OPS/a.xhtml"]);
    assert_eq!(
        document
            .navigation
            .iter()
            .map(|entry| (
                entry.entry_id.as_str(),
                entry.parent_entry_id.as_deref(),
                entry.depth
            ))
            .collect::<Vec<_>>(),
        [
            ("OPS/nav.xhtml:0", None, 0),
            ("OPS/nav.xhtml:1", Some("OPS/nav.xhtml:0"), 1),
            ("OPS/nav.xhtml:2", Some("OPS/nav.xhtml:0"), 1),
            ("OPS/nav.xhtml:3", Some("OPS/nav.xhtml:0"), 1),
            ("OPS/nav.xhtml:4", Some("OPS/nav.xhtml:0"), 1)
        ]
    );
    let resource = &document.resources[1];
    assert_eq!(
        resource.fragment_anchors["inside"],
        Some(resource.segment_ids[1])
    );
    assert_eq!(
        resource.fragment_anchors["between"],
        Some(resource.segment_ids[3])
    );
    assert_eq!(resource.fragment_anchors["end"], None);
    assert!(!resource.fragment_anchors.contains_key("missing"));
    assert_eq!(
        document.navigation[1].boundary_segment,
        Some(resource.segment_ids[1])
    );
    assert_eq!(document.navigation[2].boundary_segment, None);
    assert_eq!(document.navigation[4].boundary_segment, None);
    assert_eq!(
        document
            .chapters
            .iter()
            .map(|chapter| chapter.title.as_str())
            .collect::<Vec<_>>(),
        ["", "Part"]
    );
    assert_eq!(document.chapters[0].segment_ids, [resource.segment_ids[0]]);
    assert_eq!(document.chapters[1].segment_ids, resource.segment_ids[1..]);
}
#[test]
fn faulty_optional_nav_uses_ordered_ncx_and_spine_fallback() {
    let bad_nav = nav("<li><a href='../../escape.xhtml'>Broken</a></li>");
    let ncx = "<ncx><navMap><navPoint id='one'><navLabel><text>First</text></navLabel><content src='a.xhtml#one'/></navPoint><navPoint id='two'><navLabel><text>Second</text></navLabel><content src='a.xhtml#two'/></navPoint></navMap></ncx>";
    let temp = book(
        "<h1 id='one'>One</h1><p>body one</p><h1 id='two'>Two</h1><p>body two</p>",
        &bad_nav,
        ncx,
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    assert_eq!(document.diagnostics[0].code, "unsafe_reference");
    assert_eq!(
        document
            .chapters
            .iter()
            .map(|chapter| chapter.title.as_str())
            .collect::<Vec<_>>(),
        ["First", "Second"]
    );
    assert_eq!(
        document.chapters[1].toc_entry_id.as_deref(),
        Some("OPS/toc.ncx:1")
    );
    assert_eq!(document.navigation[0].node_id.as_deref(), Some("one"));
    let temp = book("<h1>Only</h1><p>body</p>", &bad_nav, "<ncx><navMap>", &[]);
    let document = admit_epub(&temp.path).unwrap();
    assert_eq!(document.chapters[0].title, "Only");
    assert_eq!(document.chapters[0].toc_entry_id, None);
    assert_eq!(
        document.chapters[0].segment_ids,
        document.resources[1].segment_ids
    );
    assert_eq!(
        document
            .diagnostics
            .iter()
            .filter(|diagnostic| matches!(
                diagnostic.code.as_str(),
                "unsafe_reference" | "xml-syntax"
            ))
            .count(),
        2
    );
}
#[test]
fn unicode_ruby_page_and_non_element_atom_coordinates_remain_native() {
    let temp = book(
        "<p>A🙂 café<ruby>字<rt>zi</rt><rp>(zi)</rp></ruby><!--proof--><span type='pagebreak' id='page'/><br/><img src='map.png'/> tail</p>",
        &nav(""),
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    let segment = &document.segments[0];
    assert_eq!(segment.source, "A🙂 café字 tail");
    assert_eq!(segment.rich_source.text(), "A🙂 café字 tail");
    assert_eq!(segment.kind, SegmentKind::Text);
    assert_eq!(
        segment
            .rich_source
            .atoms
            .iter()
            .map(|atom| (atom.id.as_str(), atom.kind, atom.label.as_str()))
            .collect::<Vec<_>>(),
        [
            ("a:0.0", AtomKind::Opaque, "zi"),
            ("a:0.1", AtomKind::Opaque, "(zi)"),
            ("a:-2", AtomKind::Opaque, "proof"),
            ("a:1", AtomKind::Anchor, ""),
            ("a:2", AtomKind::Linebreak, ""),
            ("a:3", AtomKind::Media, "")
        ]
    );
    assert_eq!(
        segment
            .slots
            .iter()
            .map(|slot| (
                slot.element_path.as_slice(),
                slot.field,
                slot.source_value.as_str()
            ))
            .collect::<Vec<_>>(),
        [
            ([].as_slice(), SlotField::Text, "A🙂 café"),
            ([0].as_slice(), SlotField::Text, "字"),
            ([3].as_slice(), SlotField::Tail, " tail")
        ]
    );
    assert_eq!(document.resources[1].fragment_anchors["page"], Some(0));
}
#[test]
fn notes_are_protected_by_reciprocal_source_proof_and_asides_are_lossless() {
    let temp = book(
        "<p>Prose <a id='ref' href='#note'>*</a> returns.</p><aside id='note' type='footnote'><div><p><a href='#ref'>*</a> Note body.</p></div></aside>",
        &nav(""),
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    assert_eq!(document.notes.markers.len(), 2);
    assert_eq!(document.notes.targets[0].path, [1, 1]);
    assert_eq!(
        document
            .segments
            .iter()
            .map(|segment| segment.source.as_str())
            .collect::<Vec<_>>(),
        ["Prose returns.", "Note body."]
    );
    assert_eq!(
        document.segments[0].rich_source.atoms[0].kind,
        AtomKind::Note
    );
    assert_eq!(document.segments[0].rich_source.atoms[0].label, "*");
    assert_eq!(document.segments[1].block_path, [1, 1, 0, 0]);
    let temp = book(
        "<p>Prose <a id='ref' href='#note'>*</a>.</p><aside id='note' type='footnote'>Direct prose<p><a href='#ref'>*</a> child prose</p></aside>",
        &nav(""),
        "",
        &[],
    );
    let error = admit_epub(&temp.path).unwrap_err();
    assert_eq!(error.kind, BookErrorKind::Source);
    assert_eq!(error.code, "mixed_note_aside");
}
#[test]
fn repeated_note_targets_do_not_invent_inferred_protection() {
    let temp = book(
        "<p>Prose <a id='r1' href='#note'>*</a> <a id='r2' href='#note'>*</a>.</p><p id='note'><a href='#r1'>*</a> note text</p>",
        &nav(""),
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    assert!(document.notes.markers.is_empty());
    assert_eq!(document.segments[0].source, "Prose * *.");
    assert!(document.segments[0].rich_source.atoms.is_empty());
    assert_eq!(document.segments[0].rich_source.marks[0].source_text, "*");
}
#[test]
fn complete_unique_target_mirror_not_label_similarity_is_admitted() {
    let toc = nav("<li><a href='a.xhtml#one'>One</a></li><li><a href='a.xhtml#two'>Two</a></li>");
    let mirror = "<html><body><h1>Contents</h1><ol><li><a href='a.xhtml#one'>Different wording</a></li><li><a href='a.xhtml#two'>Other wording</a></li></ol></body></html>";
    let partial = "<html><body><p><a href='a.xhtml#one'>One</a></p><p><a href='a.xhtml#one'>Two</a></p></body></html>";
    let temp = book(
        "<h1 id='one'>One</h1><p>first</p><h1 id='two'>Two</h1><p>second</p>",
        &toc,
        "",
        &[("mirror.xhtml", mirror), ("partial.xhtml", partial)],
    );
    let document = admit_epub(&temp.path).unwrap();
    let mirror = &document.resources[2];
    assert_eq!(
        document.segments[mirror.segment_ids[1]]
            .mirrored_toc_entry_id
            .as_deref(),
        Some("OPS/nav.xhtml:0")
    );
    assert_eq!(
        document.segments[mirror.segment_ids[2]]
            .mirrored_toc_entry_id
            .as_deref(),
        Some("OPS/nav.xhtml:1")
    );
    assert!(
        document.resources[3]
            .segment_ids
            .iter()
            .all(|&id| document.segments[id].mirrored_toc_entry_id.is_none())
    );
}
#[test]
fn safety_and_empty_source_are_fatal_not_optional_navigation_diagnostics() {
    let temp = book(
        "<p>body</p>",
        &nav(""),
        "<!DOCTYPE ncx [<!ENTITY x SYSTEM 'file:///tmp/private'>]><ncx>&x;</ncx>",
        &[],
    );
    assert_eq!(
        admit_epub(&temp.path).unwrap_err().kind,
        BookErrorKind::XmlSafety
    );
    let temp = book("<script>not prose</script><p> </p>", &nav(""), "", &[]);
    assert_eq!(admit_epub(&temp.path).unwrap_err().code, "empty_source");
}
#[test]
fn chapter_depth_uses_scalar_median_not_utf8_byte_lengths() {
    let toc = nav(
        "<li><span>Part</span><ol><li><a href='a.xhtml#one'>One</a></li><li><a href='a.xhtml#two'>Two</a></li></ol></li>",
    );
    let short = "🙂".repeat(1600);
    let temp = book(
        &format!("<p id='one'>{short}</p><p id='two'>{short}</p>"),
        &toc,
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    assert_eq!(
        document
            .chapters
            .iter()
            .map(|chapter| chapter.title.as_str())
            .collect::<Vec<_>>(),
        ["Part"]
    );
    assert_eq!(
        document.chapters[0].segment_ids,
        document.resources[1].segment_ids
    );
    let long = "🙂".repeat(3100);
    let temp = book(
        &format!("<p id='one'>{long}</p><p id='two'>{long}</p>"),
        &toc,
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    assert_eq!(
        document
            .chapters
            .iter()
            .map(|chapter| chapter.title.as_str())
            .collect::<Vec<_>>(),
        ["One", "Two"]
    );
    assert_eq!(
        document.chapters[0].segment_ids,
        [document.resources[1].segment_ids[0]]
    );
    assert_eq!(
        document.chapters[1].segment_ids,
        [document.resources[1].segment_ids[1]]
    );
}
#[test]
fn explicit_endnote_roles_and_landmarks_preserve_original_marker_labels() {
    let toc = "<html><body><nav type='toc'><ol><li><a href='a.xhtml'>Body</a></li></ol></nav><nav type='landmarks'><ol><li><a type='endnotes' role='doc-endnotes' href='a.xhtml#endnote'>End notes</a></li></ol></nav></body></html>";
    let temp = book(
        "<p>Body<a id='reference' role='doc-noteref' href='#endnote'>[１]</a>.</p><aside id='endnote' role='doc-endnote'><p>Exact note.<a role='doc-backlink' href='#reference'>↩</a></p></aside>",
        toc,
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    assert_eq!(document.landmarks[0].resource_href, "OPS/a.xhtml");
    assert_eq!(document.landmarks[0].semantic_type, "endnotes");
    assert_eq!(document.landmarks[0].role, "doc-endnotes");
    assert_eq!(document.landmarks[0].label, "End notes");
    assert_eq!(
        document.notes.targets[0].kind,
        wenyi_book::document::NoteBodyKind::Endnote
    );
    assert_eq!(
        document
            .notes
            .markers
            .iter()
            .map(|marker| marker.label.as_str())
            .collect::<Vec<_>>(),
        ["[１]", "↩"]
    );
    assert_eq!(document.segments[0].rich_source.atoms[0].label, "[１]");
    assert_eq!(document.segments[1].rich_source.atoms[0].label, "↩");
    assert_eq!(
        document
            .segments
            .iter()
            .map(|segment| segment.source.as_str())
            .collect::<Vec<_>>(),
        ["Body.", "Exact note."]
    );
}

#[test]
fn real_admission_projects_nested_reset_without_changing_raw_scope_or_slots() {
    let temp = book(
        "<style>.normal {font-weight:normal}</style><p>Start <strong>A<span class='normal'>B</span>C</strong> End</p>",
        &nav(""),
        "",
        &[],
    );
    let original = std::fs::read(&temp.path).unwrap();
    let document = admit_epub(&temp.path).unwrap();
    let segment = &document.segments[0];
    let styled = segment.styled_source.as_ref().unwrap();
    assert_eq!(segment.rich_source.text(), "Start ABC End");
    assert_eq!(styled.text(), segment.rich_source.text());
    let original_bold = segment
        .rich_source
        .marks
        .iter()
        .find(|mark| mark.id == "m:0")
        .unwrap();
    assert_eq!(original_bold.source_text, "ABC");
    assert_eq!(original_bold.kind, wenyi_core::MarkKind::Bold);
    let emphasis = styled
        .marks
        .iter()
        .find(|mark| wenyi_core::semantic_roles(mark).contains(&wenyi_core::SemanticRole::Bold))
        .unwrap();
    assert_eq!(emphasis.id, "m:0@bold");
    assert_eq!(emphasis.source_text, "AC");
    for run in &styled.runs {
        let bold = run.marks.iter().any(|id| {
            styled.marks.iter().any(|mark| {
                &mark.id == id
                    && wenyi_core::semantic_roles(mark).contains(&wenyi_core::SemanticRole::Bold)
            })
        });
        assert_eq!(bold, matches!(run.text.as_str(), "A" | "C"));
    }
    assert_eq!(
        segment
            .slots
            .iter()
            .map(|slot| slot.source_value.as_str())
            .collect::<Vec<_>>(),
        ["Start ", "A", "B", "C", " End"]
    );
    assert_eq!(std::fs::read(&temp.path).unwrap(), original);
}

#[test]
fn root_link_crosses_atoms_and_partial_emphasis_without_reopening() {
    let temp = book(
        "<style>.emphasis {font-weight:bold;font-style:italic}.normal {font-weight:normal}</style><ol><li><a class='emphasis' href='a.xhtml'>A<br/><span class='normal'>B</span>C</a></li></ol>",
        &nav(""),
        "",
        &[],
    );
    let document = admit_epub(&temp.path).unwrap();
    let source = document.segments[0].styled_source.as_ref().unwrap();
    assert_eq!(source.atoms[0].kind, AtomKind::Linebreak);
    let link = source
        .marks
        .iter()
        .find(|mark| wenyi_core::semantic_roles(mark).contains(&wenyi_core::SemanticRole::Link))
        .unwrap();
    assert_eq!(link.id, "m:root");
    assert!(
        source
            .runs
            .iter()
            .all(|run| run.marks.first() == Some(&link.id))
    );
    let mut target = wenyi_core::RichTarget {
        version: 2,
        runs: source.runs.clone(),
    };
    for run in &mut target.runs {
        run.slot_id = None;
        run.text = match run.text.as_str() {
            "A" => "甲".into(),
            "B" => "乙".into(),
            "C" => "丙".into(),
            _ => run.text.clone(),
        };
    }
    wenyi_core::validate_rich_target(source, &target, Some("甲乙丙")).unwrap();
}
