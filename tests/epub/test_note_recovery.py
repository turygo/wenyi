from __future__ import annotations

import unittest

from lxml import etree

from trans_novel.epub.notes import (
    detect_note_relations,
    detect_recoverable_note_relations,
    note_resources_by_semantics,
)


class TestNoteRecovery(unittest.TestCase):
    def _root(self, body: str) -> etree._Element:
        return etree.fromstring(
            (
                '<html xmlns="http://www.w3.org/1999/xhtml" '
                'xmlns:epub="http://www.idpf.org/2007/ops"><body>'
                f"{body}</body></html>"
            ).encode()
        )

    def _detect(self, body: str) -> dict:
        return detect_recoverable_note_relations({"chapter.xhtml": self._root(body)})

    def test_transparent_span_target_and_terminal_period_preserve_source(self) -> None:
        root = self._root(
            '<p>Body<span id="ref"><sup><a href="#note"> 12 </a></sup></span> tail</p>'
            '<p id="note"><a href="#ref">12.</a> Note body</p>'
        )
        before = etree.tostring(root)

        relations = detect_recoverable_note_relations({"chapter.xhtml": root})

        self.assertEqual(
            relations,
            {
                "version": 1,
                "markers": [
                    {
                        "resource_href": "chapter.xhtml",
                        "path": [0, 0, 0, 0, 0],
                        "kind": "noteref",
                        "label": " 12 ",
                        "target_resource": "chapter.xhtml",
                        "target_path": [0, 1],
                    },
                    {
                        "resource_href": "chapter.xhtml",
                        "path": [0, 1, 0],
                        "kind": "backlink",
                        "label": "12.",
                        "target_resource": "chapter.xhtml",
                        "target_path": [0, 0, 0],
                    },
                ],
                "targets": [{"resource_href": "chapter.xhtml", "path": [0, 1], "kind": "footnote"}],
            },
        )
        self.assertEqual(etree.tostring(root), before)
        self.assertEqual(detect_note_relations({"chapter.xhtml": root})["markers"], [])

    def test_numeric_pairs_without_note_semantics_use_strict_structure(self) -> None:
        for body in (
            '<p>Body<a id="ref" href="#note">[1234]</a></p>'
            '<li id="note"><a href="#ref">1234.</a> Note body</li>',
            '<p>Body<a id="ref"/><sup><a href="#note">(2)</a></sup></p>'
            '<aside id="note"><a href="#ref">2.</a> Note body</aside>',
        ):
            with self.subTest(body=body):
                self.assertEqual(len(self._detect(body)["markers"]), 2)
                self.assertEqual(
                    detect_note_relations({"chapter.xhtml": self._root(body)})["markers"], []
                )

    def test_explicit_noteref_allows_body_start_but_still_requires_same_short_label(self) -> None:
        accepted = (
            '<p><a id="ref" epub:type="noteref" href="#note">3</a> Body</p>'
            '<p id="note"><a href="#ref">3.</a> Note body</p>'
        )
        self.assertEqual(len(self._detect(accepted)["markers"]), 2)
        for labels in (("3", "4"), ("note", "return"), ("3", "3..")):
            with self.subTest(labels=labels):
                self.assertEqual(
                    self._detect(
                        '<p><a id="ref" role="doc-noteref" href="#note">'
                        f'{labels[0]}</a> Body</p><p id="note"><a role="doc-backlink" '
                        f'href="#ref">{labels[1]}</a> Note</p>'
                    )["markers"],
                    [],
                )

    def test_nontransparent_backlink_targets_are_rejected(self) -> None:
        for wrapper in (
            '<span id="ref">extra <a href="#note">1</a></span>',
            '<span id="ref"><a href="#note">1</a> extra</span>',
            '<span id="ref"><a href="#note">1</a><b>extra</b></span>',
            '<em id="ref"><a href="#note">1</a></em>',
        ):
            with self.subTest(wrapper=wrapper):
                self.assertEqual(
                    self._detect(
                        f'<p>Body{wrapper}</p><p id="note"><a href="#ref">1.</a> Note</p>'
                    )["markers"],
                    [],
                )

    def test_explicit_empty_notes_are_rejected(self) -> None:
        for label in ("1", "†"):
            with self.subTest(label=label):
                self.assertEqual(
                    self._detect(
                        '<p><a id="ref" role="doc-noteref" href="#note">'
                        f'{label}</a> Body</p><aside id="note" epub:type="footnote">'
                        f'<a role="doc-backlink" href="#ref">{label}</a> </aside>'
                    )["markers"],
                    [],
                )

    def test_different_numbers_duplicate_ids_and_extra_backlinks_are_rejected(self) -> None:
        for body in (
            '<p>Body<a id="ref" href="#note">1</a></p><p id="note"><a href="#ref">2.</a> Note</p>',
            '<p>Body<a id="ref" href="#note">1</a><a id="ref"/></p>'
            '<p id="note"><a href="#ref">1.</a> Note</p>',
            '<p>Body<a id="ref" href="#note">1</a></p>'
            '<p id="note"><a href="#ref">1.</a> <a href="#ref">1</a> Note</p>',
            '<p>Body<a id="ref" href="#note">1</a></p>'
            '<p id="note"><a href="#ref">1.</a> <a href="#ref">2</a> Note</p>',
            '<p>Body<a id="ref" href="#note">1</a> Other<a id="other" href="#note">1</a></p>'
            '<p id="note"><a href="#ref">1.</a> Note</p>',
            '<p>Body<a id="ref" href="#note">1</a></p>'
            '<p id="note"><a href="#ref">1.</a> Note</p><p id="note">Duplicate</p>',
        ):
            with self.subTest(body=body):
                self.assertEqual(self._detect(body)["markers"], [])

    def test_external_unsafe_protected_and_excluded_resources_are_rejected(self) -> None:
        for href in ("https://example.com/#note", "chapter.xhtml?x=1#note", "../missing#note"):
            with self.subTest(href=href):
                self.assertEqual(
                    self._detect(
                        f'<p>Body<a id="ref" href="{href}">1</a></p>'
                        '<p id="note"><a href="#ref">1.</a> Note</p>'
                    )["markers"],
                    [],
                )
        root = self._root(
            '<nav><p>Body<a id="ref" href="#note">1</a></p>'
            '<p id="note"><a href="#ref">1.</a> Note</p></nav>'
        )
        self.assertEqual(detect_recoverable_note_relations({"chapter.xhtml": root})["markers"], [])
        root = self._root(
            '<p>Body<a id="ref" href="#note">1</a></p><p id="note"><a href="#ref">1.</a> Note</p>'
        )
        self.assertEqual(
            detect_recoverable_note_relations(
                {"chapter.xhtml": root}, excluded_resources={"chapter.xhtml"}
            )["markers"],
            [],
        )

    def test_numeric_note_requires_body_and_unambiguous_direction_without_semantics(self) -> None:
        for body in (
            '<p><a id="ref" href="#note">1</a> Body</p><p id="note"><a href="#ref">1.</a> Note</p>',
            '<p>Body<a id="ref" href="#note">1</a></p><p id="note"><a href="#ref">1.</a> </p>',
            '<p>Body<a id="ref" href="#note">1</a></p>'
            '<p id="note">Text <a href="#ref">1.</a> Note</p>',
            '<p>Body<a id="ref" href="#note">1</a></p>'
            '<div id="note"><a href="#ref">1.</a> Note</div>',
        ):
            with self.subTest(body=body):
                self.assertEqual(self._detect(body)["markers"], [])

    def test_conflicting_roles_and_reciprocal_symbol_starts_are_rejected(self) -> None:
        for body in (
            '<p>Body<a id="ref" role="doc-noteref doc-backlink" href="#note">*</a></p>'
            '<p id="note"><a href="#ref">*</a> Note</p>',
            '<p><a id="left" href="#right">†</a> Left</p>'
            '<p><a id="right" href="#left">†</a> Right</p>',
            '<p>Body<a id="ref" href="#note">*</a></p>'
            '<p id="note" epub:type="footnote endnote"><a href="#ref">*</a> Note</p>',
        ):
            with self.subTest(body=body):
                self.assertEqual(self._detect(body)["markers"], [])

    def test_semantic_resource_helper_preserves_old_conflict_policy(self) -> None:
        self.assertEqual(
            note_resources_by_semantics(
                [
                    {"resource_href": "foot.xhtml", "type": "notes"},
                    {"resource_href": "end.xhtml", "nav_type": "endnotes"},
                    {"resource_href": "conflict.xhtml", "role": "doc-footnote"},
                    {"resource_href": "conflict.xhtml", "type": "endnotes"},
                    {"resource_href": "class.xhtml", "class": "footnotes"},
                    {"resource_href": "", "type": "notes"},
                ]
            ),
            {"foot.xhtml": "footnote", "end.xhtml": "endnote"},
        )

    def test_legacy_detector_still_preserves_multiple_symbol_backlinks(self) -> None:
        root = self._root(
            '<p>One<a id="r1" href="#note">*</a> Two<a id="r2" href="#note">*</a></p>'
            '<p id="note"><a href="#r1">*</a> <a href="#r2">*</a> Note</p>'
        )
        self.assertEqual(len(detect_note_relations({"chapter.xhtml": root})["markers"]), 4)
        self.assertEqual(detect_recoverable_note_relations({"chapter.xhtml": root})["markers"], [])


if __name__ == "__main__":
    unittest.main()
