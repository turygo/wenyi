"""译文语序与源格式身份的独立渲染契约。"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

from trans_novel.assemble.epub.rendering.richtext import render_rich_block
from trans_novel.assemble.epub.rendering.source_markup import render_source_resource
from trans_novel.assemble.epub.verification.richtext import (
    prove_and_restore_rich_blocks,
    prove_rich_target,
)
from trans_novel.assemble.epub.verification.slots import slot_proof
from trans_novel.epub.richtext import InlineRun, RichTarget
from trans_novel.epub.slots import (
    EpubSegmentState,
    normalized_source_text,
    slot_contract_digest,
    source_passthrough_transport,
)
from trans_novel.ingest import Segment
from trans_novel.ingest.epub.markup import block_slots, element_path, resource_fingerprint
from trans_novel.ingest.epub.richtext import extract_rich_sources


def _fixture(markup: str, *, split_br: bool = False):
    data = (
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="en"><head/><body>'
        + markup
        + "</body></html>"
    ).encode()
    root = etree.fromstring(data)
    block = root.find(".//{*}p")
    digest = hashlib.sha256(data).hexdigest()
    protected = {
        element_path(root, node) for node in block.iter() if node.get("role") == "doc-noteref"
    }
    slots = block_slots(
        root,
        block,
        resource_href="c.xhtml",
        resource_sha256=digest,
        parse_mode="xml",
        anchor="p",
        protected_paths=protected,
    )
    groups = [slots]
    if split_br:
        groups = [
            [slot for slot in slots if not slot.element_path],
            [slot for slot in slots if slot.element_path],
        ]
    states = [
        EpubSegmentState(
            resource_href="c.xhtml",
            resource_sha256=digest,
            block_path=element_path(root, block),
            block_fingerprint=resource_fingerprint(block),
            parse_mode="xml",
            slots=group,
            slot_contract_sha256=slot_contract_digest(group),
        )
        for group in groups
    ]
    for state, rich in zip(
        states, extract_rich_sources(root, block, states, protected_paths=protected), strict=True
    ):
        state.rich_source = rich
    segments = [
        Segment(
            index=index,
            source=normalized_source_text(state.slots),
            resource_href="c.xhtml",
            epub_state=state,
        )
        for index, state in enumerate(states)
    ]
    return data, root, block, segments


class TestRichTextRendering(unittest.TestCase):
    def test_comment_and_pi_keep_immutable_tail_and_separate_following_prose(self):
        _, _, block, segments = _fixture(
            "<p>Before<!--keep--> protected<?pagebreak original?> tail<b>bold</b>After</p>"
        )
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        inventory = segments[0].epub_state.rich_source
        comment, instruction = inventory.atoms
        mark = inventory.marks[0]
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="前"),
                    InlineRun(atom=comment.id),
                    InlineRun(text="后"),
                    InlineRun(atom=instruction.id),
                    InlineRun(text="粗体", marks=(mark.id,)),
                    InlineRun(text="。"),
                ]
            )
        )
        render_rich_block(block, segments)
        self.assertEqual(block[0].tail, " protected")
        self.assertEqual(block[1].attrib, {"data-tn-richtext": "text"})
        self.assertEqual(block[1].text, "后")
        self.assertEqual(block[2].tail, " tail")
        self.assertEqual(block[2].target, "pagebreak")
        prove_rich_target(source, block, segments)
        block[0].tail = " changed"
        with self.assertRaisesRegex(ValueError, "unexpected rich target node|mismatch"):
            prove_rich_target(source, block, segments)

    def test_tail_less_comment_can_precede_unformatted_target_prose(self):
        _, _, block, segments = _fixture("<p>Before<!--keep--><b>bold</b>After</p>")
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        inventory = segments[0].epub_state.rich_source
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(atom=inventory.atoms[0].id),
                    InlineRun(text="前"),
                    InlineRun(text="粗体", marks=(inventory.marks[0].id,)),
                    InlineRun(text="后"),
                ]
            )
        )
        render_rich_block(block, segments)
        self.assertEqual(block[0].tail, "前")
        prove_rich_target(source, block, segments)

    def test_relocated_note_mapping_tracks_actual_atom(self):
        data, root, _, segments = _fixture(
            '<p><b>First</b><sup><a id="ref" href="#n" role="doc-noteref">1</a></sup> second.</p>'
        )
        atom = segments[0].epub_state.rich_source.atoms[0]
        mark = segments[0].epub_state.rich_source.marks[0]
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="第一句。第二句。"),
                    InlineRun(atom=atom.id),
                    InlineRun(text="重点", marks=(mark.id,)),
                ]
            )
        )
        source_note = root.find(".//{*}a")
        source_path = element_path(root, source_note)
        scopes = {}
        rendered = render_source_resource(
            data,
            "c.xhtml",
            segments,
            expected_digest=hashlib.sha256(data).hexdigest(),
            expected_mode="xml",
            target_lang="zh",
            scope_sink=scopes,
            note_source_paths=(source_path,),
        )
        output = etree.fromstring(rendered)
        mapping = scopes["c.xhtml"].note_paths[0]
        self.assertEqual(mapping.source_path, source_path)
        self.assertNotEqual(mapping.source_path, mapping.target_path)
        node = output
        for index in mapping.target_path:
            node = list(node)[index]
        self.assertEqual(node.get("id"), "ref")

    def test_outer_dom_tamper_is_not_hidden_by_rich_restoration(self):
        data, _, _, segments = _fixture(
            '<p>Before<b>bold</b>After</p><div id="other">Untouched</div>'
        )
        mark = segments[0].epub_state.rich_source.marks[0]
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="前"),
                    InlineRun(text="粗体", marks=(mark.id,)),
                    InlineRun(text="后"),
                ]
            )
        )
        rendered = render_source_resource(
            data,
            "c.xhtml",
            segments,
            expected_digest=hashlib.sha256(data).hexdigest(),
            expected_mode="xml",
            target_lang="zh",
        )
        resource = {"resource_sha256": hashlib.sha256(data).hexdigest(), "parse_mode": "xml"}
        chapter = SimpleNamespace(segments=segments)
        store = SimpleNamespace(
            load_manifest=lambda: {
                "source_lang": "en",
                "target_lang": "zh",
                "meta": {},
                "chapters": [],
            }
        )
        for tamper in (False, True):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                output_root = etree.fromstring(rendered)
                if tamper:
                    output_root.find(".//{*}div").set("id", "changed")
                source_path, target_path = (
                    Path(directory) / "source.epub",
                    Path(directory) / "target.epub",
                )
                for path, content in (
                    (source_path, data),
                    (target_path, etree.tostring(output_root)),
                ):
                    with zipfile.ZipFile(path, "w") as archive:
                        archive.writestr("c.xhtml", content)
                failures = []
                slot_proof(
                    source_path,
                    target_path,
                    store,
                    {"c.xhtml": resource},
                    [chapter],
                    bilingual=False,
                    target_lang="zh",
                    failures=failures,
                    warnings=[],
                    checked={},
                )
                self.assertEqual(
                    any(item["code"] == "unauthorized_dom_change" for item in failures), tamper
                )
                if not tamper:
                    self.assertEqual(failures, [])

    def test_semantic_bold_reorders_and_note_follows_statement(self):
        _, _, block, segments = _fixture(
            '<p>All chapters remain intact.<sup><a id="ref" href="#n" role="doc-noteref">1</a></sup> I use <b>boldface type</b> for many passages.</p>'
        )
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        rich = segments[0].epub_state.rich_source
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="格雷厄姆的全部原始章节均完整保留。"),
                    InlineRun(atom=rich.atoms[0].id),
                    InlineRun(text="我还用"),
                    InlineRun(text="粗体", marks=(rich.marks[0].id,)),
                    InlineRun(text="标出了格雷厄姆的许多段落。"),
                ]
            )
        )
        render_rich_block(block, segments)
        self.assertEqual(block.find("{*}b").text, "粗体")
        self.assertEqual(block.text, "格雷厄姆的全部原始章节均完整保留。")
        self.assertEqual(block.find(".//{*}a").get("href"), "#n")
        prove_rich_target(source, block, segments)

    def test_whole_sentence_is_bold_including_punctuation(self):
        _, _, block, segments = _fixture(
            "<p><b>There is no sure and easy path to riches on Wall Street or anywhere else.</b></p>"
        )
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        mark = segments[0].epub_state.rich_source.marks[0]
        text = "在华尔街或任何其他地方，都不存在确定而轻松的致富之路。"
        segments[0].assign_translation(RichTarget(runs=[InlineRun(text=text, marks=(mark.id,))]))
        render_rich_block(block, segments)
        self.assertEqual(block.find("{*}b").text, text)
        prove_rich_target(source, block, segments)

    def test_italic_and_repeated_nested_mark_keep_one_identity(self):
        _, _, block, segments = _fixture(
            '<p><em id="emphasis"><b>First</b> and second</em> final.</p>'
        )
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        italic, bold = segments[0].epub_state.rich_source.marks
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="第一", marks=(italic.id, bold.id)),
                    InlineRun(text="间隔"),
                    InlineRun(text="第二", marks=(italic.id,)),
                    InlineRun(text="。"),
                ]
            )
        )
        render_rich_block(block, segments)
        self.assertEqual([node.text for node in block.findall("{*}em")], [None, "第二"])
        self.assertEqual(len(block.xpath('.//*[@id="emphasis"]')), 1)
        prove_rich_target(source, block, segments)

    def test_atom_tampering_is_rejected(self):
        _, _, block, segments = _fixture('<p>Before<img src="image.png"/>After</p>')
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        atom = segments[0].epub_state.rich_source.atoms[0]
        segments[0].assign_translation(
            RichTarget(runs=[InlineRun(text="前"), InlineRun(atom=atom.id), InlineRun(text="后")])
        )
        render_rich_block(block, segments)
        block.find("{*}img").set("src", "wrong.png")
        with self.assertRaisesRegex(ValueError, "unexpected rich target format|mismatch"):
            prove_rich_target(source, block, segments)

    def test_br_segments_render_once_and_bilingual_copy_proves(self):
        for order in ("source_first", "target_first"):
            with self.subTest(order=order):
                data, root, _, segments = _fixture('<p id="p">One<br/>Two</p>', split_br=True)
                segments[0].assign_translation(RichTarget(runs=[InlineRun(text="一")]))
                atom = segments[1].epub_state.rich_source.atoms[0]
                segments[1].assign_translation(
                    RichTarget(runs=[InlineRun(atom=atom.id), InlineRun(text="二")])
                )
                rendered = render_source_resource(
                    data,
                    "c.xhtml",
                    segments,
                    expected_digest=hashlib.sha256(data).hexdigest(),
                    expected_mode="xml",
                    target_lang="zh",
                    bilingual=True,
                    source_lang="en",
                    order=order,
                )
                output = etree.fromstring(rendered)
                self.assertEqual(len(output.xpath('//*[contains(@class,"tn-source")]')), 1)
                count = prove_and_restore_rich_blocks(
                    root,
                    output,
                    {segments[0].epub_state.block_path: segments},
                    bilingual=True,
                    source_lang="en",
                    order=order,
                )
                self.assertEqual(count, 1)
                self.assertEqual(output.find(".//{*}p").find("{*}br").tail, "Two")

    def test_missing_annotation_fails_closed(self):
        data, _, _, segments = _fixture("<p>Before<b>bold</b>After</p>")
        segments[0].target = "中文"
        with self.assertRaisesRegex(ValueError, "trusted targets"):
            render_source_resource(
                data,
                "c.xhtml",
                segments,
                expected_digest=hashlib.sha256(data).hexdigest(),
                expected_mode="xml",
                target_lang="zh",
            )

    def test_mixed_br_passthrough_and_translated_runs(self):
        data, root, _, segments = _fixture("<p>One<br/>Two</p>", split_br=True)
        segments[0].assign_translation(source_passthrough_transport(segments[0].epub_state))
        atom = segments[1].epub_state.rich_source.atoms[0]
        segments[1].assign_translation(
            RichTarget(runs=[InlineRun(atom=atom.id), InlineRun(text="二")])
        )
        rendered = render_source_resource(
            data,
            "c.xhtml",
            segments,
            expected_digest=hashlib.sha256(data).hexdigest(),
            expected_mode="xml",
            target_lang="zh",
        )
        output = etree.fromstring(rendered)
        self.assertEqual(output.find(".//{*}p").text, "One")
        from trans_novel.assemble.epub.rendering.richtext import rich_block_segments

        prove_and_restore_rich_blocks(
            root,
            output,
            rich_block_segments(segments),
            bilingual=False,
            source_lang="en",
            order="target_first",
        )

    def test_css_span_identity_and_scope_tampering(self):
        _, _, block, segments = _fixture(
            '<p>A<span class="strong" style="font-weight:700">bold</span>Z</p>'
        )
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        mark = segments[0].epub_state.rich_source.marks[0]
        mark.kind = "bold"
        mark.style_evidence = ["font-weight:700"]
        segments[0].assign_translation(
            RichTarget(runs=[InlineRun(text="加粗", marks=(mark.id,)), InlineRun(text="，其后。")])
        )
        render_rich_block(block, segments)
        prove_rich_target(source, block, segments)
        block.find("{*}span").set("style", "font-weight:400")
        with self.assertRaisesRegex(ValueError, "unexpected rich target format"):
            prove_rich_target(source, block, segments)


class TestRichSourceCoverage(unittest.TestCase):
    def test_deleted_source_field_and_rehashed_inventory_fail_before_restoration(self):
        data, root, _, segments = _fixture("<p>Before<b>bold</b>After</p>")
        original = deepcopy(root)
        segment = segments[0]
        state = segment.epub_state
        state.slots = state.slots[:-1]
        state.slot_contract_sha256 = slot_contract_digest(state.slots)
        state.rich_source = extract_rich_sources(root, root.find(".//{*}p"), [state])[0]
        segment.source = normalized_source_text(state.slots)
        mark = state.rich_source.marks[0]
        segment.assign_translation(
            RichTarget(runs=[InlineRun(text="前"), InlineRun(text="粗体", marks=(mark.id,))])
        )
        rendered = render_source_resource(
            data,
            "c.xhtml",
            segments,
            expected_digest=hashlib.sha256(data).hexdigest(),
            expected_mode="xml",
            target_lang="zh",
        )
        output = etree.fromstring(rendered)
        with self.assertRaisesRegex(ValueError, "source field coverage mismatch"):
            prove_and_restore_rich_blocks(
                original,
                output,
                {state.block_path: segments},
                bilingual=False,
                source_lang="en",
                order="target_first",
            )
        self.assertEqual("".join(output.find(".//{*}p").itertext()), "前粗体")

    def test_missing_atomic_inventory_does_not_hide_deleted_media(self):
        _, root, block, segments = _fixture('<p>Before<img src="image.png"/>After</p>')
        source = deepcopy(root).find(".//{*}p")
        segment = segments[0]
        inventory = segment.epub_state.rich_source
        segment.epub_state.rich_source = inventory.model_copy(
            update={"atoms": [], "runs": [run for run in inventory.runs if run.atom is None]}
        )
        segment.assign_translation(RichTarget(runs=[InlineRun(text="前后")]))
        render_rich_block(block, segments)
        with self.assertRaisesRegex(ValueError, "source inventory mismatch"):
            prove_rich_target(source, block, segments)

    def test_source_mirrors_deduplicate_geometry_but_reject_field_conflicts(self):
        from trans_novel.assemble.epub.verification.richtext import _source_inventory

        _, _, block, segments = _fixture("<p>Before<b>bold</b>After</p>")
        segment = segments[0]
        mark = segment.epub_state.rich_source.marks[0]
        segment.assign_translation(RichTarget(runs=[InlineRun(text="前后", marks=(mark.id,))]))
        mirror = segment.model_copy(deep=True)
        _source_inventory(block, [segment, mirror])
        mirror.epub_state.slots[0].source_value = "Conflicting source"
        with self.assertRaisesRegex(ValueError, "conflicting source field ownership"):
            _source_inventory(block, [segment, mirror])
