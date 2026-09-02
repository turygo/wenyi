"""Quality capability public API."""

from __future__ import annotations

from trans_novel.pipeline.quality.checks import LengthFlag, count_aligned, length_flags
from trans_novel.pipeline.quality.glossary import (
    audit_glossary,
    fix_latin_residue,
    rewrite_targets,
    target_corpus,
)
from trans_novel.pipeline.quality.lint import (
    LintIssue,
    PolishGateResult,
    drops_dialogue_quotes,
    evaluate_polish_gate,
    lint_targets,
    polish_gate,
)
from trans_novel.pipeline.quality.tools import lock, open_glossary, resolve

__all__ = [
    "LengthFlag",
    "LintIssue",
    "PolishGateResult",
    "audit_glossary",
    "count_aligned",
    "drops_dialogue_quotes",
    "evaluate_polish_gate",
    "fix_latin_residue",
    "length_flags",
    "lint_targets",
    "lock",
    "open_glossary",
    "polish_gate",
    "resolve",
    "rewrite_targets",
    "target_corpus",
]
