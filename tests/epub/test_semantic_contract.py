"""必要语义与合法格式组合不依赖书名或源槽位切片。"""

import hashlib
import json
import unittest

from trans_novel.epub.richtext import (
    InlineMark,
    InlineRun,
    RichSource,
    RichTarget,
    rich_source_digest,
    validate_rich_target,
)


class TestSemanticContracts(unittest.TestCase):
    def test_required_tag_semantics_cannot_be_lost(self):
        for tag in ("sub", "sup", "code", "kbd", "samp", "b", "em"):
            for namespace in ("", "{http://www.w3.org/1999/xhtml}"):
                with self.subTest(tag=tag, namespace=namespace):
                    source = RichSource(
                        marks=[
                            InlineMark(
                                id="semantic",
                                path=(0,),
                                tag=namespace + tag,
                                source_text="value",
                            )
                        ]
                    )
                    with self.assertRaisesRegex(ValueError, "required semantic"):
                        validate_rich_target(source, RichTarget(runs=[InlineRun(text="中文")]))
                    validate_rich_target(
                        source,
                        RichTarget(
                            runs=[
                                InlineRun(text="中文", marks=("semantic",)),
                            ]
                        ),
                    )

    def test_combined_css_roles_and_normal_reset_are_required(self):
        for roles, required in [(("italic", "subscript"), False), ((), True)]:
            with self.subTest(roles=roles, required=required):
                source = RichSource(
                    marks=[
                        InlineMark(
                            id="css",
                            path=(0,),
                            tag="span",
                            source_text="value",
                            semantics=roles,
                            required=required,
                        )
                    ]
                )
                with self.assertRaisesRegex(ValueError, "required semantic"):
                    validate_rich_target(source, RichTarget(runs=[InlineRun(text="中文")]))

    def test_parallel_links_cannot_be_nested_or_reopened(self):
        source = RichSource(
            marks=[
                InlineMark(
                    id=key,
                    path=(index,),
                    tag="a",
                    source_text=key,
                    attributes={"href": "#destination"},
                )
                for index, key in enumerate(("one", "two"))
            ]
        )
        with self.assertRaisesRegex(ValueError, "nested links"):
            validate_rich_target(
                source,
                RichTarget(
                    runs=[
                        InlineRun(text="中文", marks=("one", "two")),
                    ]
                ),
            )
        with self.assertRaisesRegex(ValueError, "splits a source link"):
            validate_rich_target(
                source,
                RichTarget(
                    runs=[
                        InlineRun(text="一", marks=("one",)),
                        InlineRun(text="二", marks=("two",)),
                        InlineRun(text="三", marks=("one",)),
                    ]
                ),
            )

    def test_emphasis_may_be_discontinuous_and_reordered(self):
        source = RichSource(
            marks=[
                InlineMark(id="b", path=(0,), tag="strong", source_text="bold"),
                InlineMark(id="i", path=(1,), tag="i", source_text="italic"),
            ]
        )
        validate_rich_target(
            source,
            RichTarget(
                runs=[
                    InlineRun(text="甲", marks=("i",)),
                    InlineRun(text="乙", marks=("b",)),
                    InlineRun(text="丙", marks=("i", "b")),
                ]
            ),
        )

    def test_legacy_read_is_not_implicit_upgrade(self):
        source = RichSource(
            version=1,
            marks=[
                InlineMark(id="sub", path=(0,), tag="sub", source_text="90"),
            ],
        )
        old = RichTarget(version=1, runs=[InlineRun(text="90")])
        validate_rich_target(source, old)
        source.version = 2
        with self.assertRaisesRegex(ValueError, "explicit contract migration"):
            validate_rich_target(source, old)


class TestOmissionContracts(unittest.TestCase):
    def test_implicit_and_merged_accept_only_eligible_emphasis_or_style_reset(self):
        for tag, semantics, required in (
            ("b", (), False),
            ("em", (), False),
            ("span", ("bold", "italic"), False),
            ("span", (), True),
        ):
            for omission in ("implicit", "merged"):
                with self.subTest(tag=tag, semantics=semantics, omission=omission):
                    source = RichSource(
                        marks=[
                            InlineMark(
                                id="emphasis",
                                path=(0,),
                                tag=tag,
                                source_text="you",
                                semantics=semantics,
                                required=required,
                            )
                        ]
                    )
                    target = RichTarget(
                        version=3,
                        runs=[
                            InlineRun(text="吃库库！"),
                            InlineRun(
                                marks=("emphasis",),
                                omission=omission,
                                explanation="The imperative already implies the subject.",
                            ),
                        ],
                    )
                    validate_rich_target(source, target, expected_text="吃库库！")
                    restored = RichTarget.model_validate_json(target.model_dump_json())
                    self.assertEqual(restored, target)
                    self.assertEqual(restored.runs[-1].omission, omission)

    def test_ineligible_roles_decoration_and_empty_sources_cannot_be_omitted(self):
        cases = [{"tag": tag} for tag in ("a", "sup", "sub", "code", "kbd", "samp", "pre")]
        cases.extend(
            [
                {"tag": "span", "semantics": ("italic", "literal")},
                {"tag": "span", "kind": "decoration"},
                {"tag": "span"},
                {"tag": "em", "source_text": ""},
                {"tag": "em", "source_text": " "},
            ]
        )
        for case in cases:
            with self.subTest(case=case):
                mark = InlineMark.model_validate(
                    {"id": "m", "path": [0], "source_text": "you", **case}
                )
                target = RichTarget(
                    version=3,
                    runs=[InlineRun(marks=("m",), omission="implicit", explanation="Implied.")],
                )
                with self.assertRaisesRegex(ValueError, "ineligible"):
                    validate_rich_target(RichSource(marks=[mark]), target)

    def test_run_shape_requires_empty_marked_boundary_and_explanation(self):
        base = {"marks": ["m"], "omission": "implicit", "explanation": "Implied subject."}
        for update in (
            {"text": "你"},
            {"atom": "atom"},
            {"slot_id": "slot"},
            {"marks": []},
            {"explanation": ""},
            {"explanation": " \t"},
            {"omission": None},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                InlineRun.model_validate({**base, **update})

    def test_duplicate_or_meaningful_dual_representation_is_rejected(self):
        source = RichSource(marks=[InlineMark(id="m", path=(0,), tag="i", source_text="you")])
        omission = InlineRun(marks=("m",), omission="implicit", explanation="Implied subject.")
        for extra, error in (
            (omission.model_copy(deep=True), "duplicate omissions"),
            (InlineRun(text="你", marks=("m",)), "meaningful target range"),
            (InlineRun(atom="atom", marks=("m",)), "meaningful target range"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, error):
                validate_rich_target(source, RichTarget(version=3, runs=[omission, extra]))

    def test_omission_requires_source_two_target_three_and_other_required_ranges(self):
        mark = InlineMark(id="subject", path=(0,), tag="i", source_text="you")
        omission = InlineRun(
            marks=(mark.id,), omission="implicit", explanation="The subject is implied."
        )
        with self.assertRaisesRegex(ValueError, "source cannot contain"):
            RichSource(marks=[mark], runs=[omission])
        for version in (1, 2):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "version 3"):
                RichTarget(version=version, runs=[omission])
        with self.assertRaisesRegex(ValueError, "source version 2"):
            validate_rich_target(
                RichSource(version=1, marks=[mark]), RichTarget(version=3, runs=[omission])
            )
        source = RichSource(
            marks=[mark, InlineMark(id="verb", path=(1,), tag="b", source_text="Eat")]
        )
        with self.assertRaisesRegex(ValueError, "required semantic"):
            validate_rich_target(
                source, RichTarget(version=3, runs=[omission, InlineRun(text="吃库库！")])
            )
        validate_rich_target(
            source,
            RichTarget(version=3, runs=[omission, InlineRun(text="吃", marks=("verb",))]),
        )

    def test_legacy_run_json_and_source_digest_remain_exact(self):
        expected = '{"text":"Eat kuku!","marks":["m"],"atom":null,"slot_id":null}'
        run = InlineRun(text="Eat kuku!", marks=("m",))
        self.assertEqual(run.model_dump_json(), expected)
        self.assertEqual(run.model_dump(mode="json"), json.loads(expected))
        mark = InlineMark(id="m", path=(0,), tag="i", source_text="Eat kuku!")
        source_json = (
            '{"version":2,"marks":['
            + mark.model_dump_json()
            + '],"atoms":[],"runs":['
            + expected
            + "]}"
        )
        source = RichSource(marks=[mark], runs=[run])
        self.assertEqual(source.model_dump_json(), source_json)
        self.assertEqual(
            rich_source_digest(source), hashlib.sha256(source_json.encode()).hexdigest()
        )
        self.assertEqual(RichSource.model_validate_json(source_json), source)
