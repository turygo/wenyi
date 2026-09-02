"""Thread-safe sinks for benchmark call-attempt telemetry."""

from __future__ import annotations

import json
import os
import threading
from os import PathLike
from pathlib import Path
from typing import Any

from trans_novel.llm.telemetry import CallAttemptTelemetry


def _context_value(value: object, *, name: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string" + (" or None" if nullable else ""))
    value = value.strip()
    if not value and not nullable:
        raise ValueError(f"{name} must not be blank")
    return value or None


class _StaticContext:
    def __init__(
        self,
        *,
        benchmark_id: str,
        candidate_id: str,
        run_id: str,
        book_id: str | None,
    ) -> None:
        self._context = {
            "benchmark_id": _context_value(benchmark_id, name="benchmark_id"),
            "candidate_id": _context_value(candidate_id, name="candidate_id"),
            "run_id": _context_value(run_id, name="run_id"),
            "book_id": _context_value(book_id, name="book_id", nullable=True),
        }

    def _envelope(self, attempt: CallAttemptTelemetry) -> dict[str, Any]:
        envelope = attempt.model_dump(mode="json")
        envelope.update(self._context)
        return envelope


class JsonlCallTelemetrySink:
    """Append one canonical JSON object per telemetry attempt."""

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        benchmark_id: str | None = None,
        candidate_id: str | None = None,
        run_id: str | None = None,
        book_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        if self.path.exists():
            try:
                self.records = [
                    json.loads(line)
                    for line in self.path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception as error:
                raise ValueError(f"invalid telemetry artifact {self.path}: {error}") from error
        self._context = None
        values = (benchmark_id, candidate_id, run_id)
        if any(value is not None for value in (*values, book_id)):
            self._context = _StaticContext(
                benchmark_id=benchmark_id or "",
                candidate_id=candidate_id or "",
                run_id=run_id or "",
                book_id=book_id,
            )
        self._lock = threading.Lock()

    def record(self, attempt: CallAttemptTelemetry) -> None:
        envelope = (
            self._context._envelope(attempt)
            if self._context is not None
            else attempt.model_dump(mode="python")
        )
        self.records.append(envelope)
        line = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())


class CollectingCallTelemetrySink(_StaticContext):
    """Collect enriched telemetry envelopes for deterministic offline tests."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        candidate_id: str,
        run_id: str,
        book_id: str | None = None,
    ) -> None:
        super().__init__(
            benchmark_id=benchmark_id, candidate_id=candidate_id, run_id=run_id, book_id=book_id
        )
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records]

    def record(self, attempt: CallAttemptTelemetry) -> None:
        envelope = self._envelope(attempt)
        with self._lock:
            self._records.append(envelope)


__all__ = ["CollectingCallTelemetrySink", "JsonlCallTelemetrySink"]
