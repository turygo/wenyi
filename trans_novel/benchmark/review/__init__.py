"""Public review workflow facade."""

from __future__ import annotations

import os
from pathlib import Path

from trans_novel.benchmark.review.artifacts import (
    ReviewArtifactError,
    ReviewSpec,
    finalize_review_artifacts,
    prepare_review_artifacts,
    validate_review_artifacts,
)


def prepare_review(
    run_dir: str | os.PathLike[str],
    review_spec: str | os.PathLike[str] | dict[str, object],
    out_dir: str | os.PathLike[str],
) -> Path:
    return prepare_review_artifacts(run_dir, review_spec, out_dir)


def finalize_review(
    review_dir: str | os.PathLike[str],
    results_dir: str | os.PathLike[str],
) -> Path:
    return finalize_review_artifacts(review_dir, results_dir)


def validate_review(review_dir: str | os.PathLike[str]) -> dict[str, object]:
    return validate_review_artifacts(review_dir)


__all__ = [
    "ReviewArtifactError",
    "ReviewSpec",
    "finalize_review",
    "prepare_review",
    "validate_review",
    "validate_review_artifacts",
]
