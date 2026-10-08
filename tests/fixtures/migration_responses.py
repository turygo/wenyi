from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

from tests.fixtures.migration_epub import MARK_TARGETS


def _naming_candidates(user: str, chapters: dict[int, str]) -> list[str]:
    marker = "【候选列表（编号 表面形式（出现次数） 例句）】\n"
    if marker not in user:
        raise AssertionError("term naming request lacks its candidate boundary")
    body = user.split(marker, 1)[1].split("\n\n请为值得入表", 1)[0]
    sources = []
    for index, line in enumerate(body.splitlines()):
        match = re.fullmatch(r"\[(\d+)\] (.+?)（出现(\d+)次）(?: 例：(.*))?", line)
        if match is None or int(match[1]) != index or int(match[3]) < 1:
            raise AssertionError("term naming request contains an invalid candidate identity")
        source, contexts = match[2], match[4]
        if not any(source in chapter for chapter in chapters.values()):
            raise AssertionError("term naming candidate has no synthetic source evidence")
        if contexts and any(
            not any(context in chapter for chapter in chapters.values())
            for context in contexts.split(" / ")
        ):
            raise AssertionError("term naming context differs from the synthetic source")
        sources.append(source)
    if not sources or len(sources) != len(set(sources)):
        raise AssertionError("term naming request has no unique candidate set")
    return sources


def _annotation_runs(source: dict[str, Any], target: str) -> list[dict[str, Any]]:
    notes = source.get("note_references", [])
    note_marks = {key for item in notes for key in item.get("content_marks", [])}
    body = target
    note_runs = []
    for note in notes:
        label = note.get("label")
        if not isinstance(label, str) or not label or not body.endswith(label):
            raise AssertionError("fixture target does not preserve its source-bound note label")
        body = body[: -len(label)]
        note_runs.append({"text": label, "marks": note["content_marks"]})
    marks = source.get("marks", [])
    intervals = []
    for mark in marks:
        if mark["id"] in note_marks:
            continue
        fragment = MARK_TARGETS.get(mark.get("source_text", ""))
        start = body.find(fragment) if fragment else -1
        if start < 0 or body.count(fragment) != 1:
            raise AssertionError("source mark has no unique explicitly mapped target range")
        intervals.append((start, start + len(fragment), mark))
    boundaries = {0, len(body)}
    for start, end, _mark in intervals:
        boundaries.update((start, end))
    ordered = sorted(boundaries)
    runs: list[dict[str, Any]] = []
    for left, right in pairwise(ordered):
        text = body[left:right]
        if not text:
            continue
        active = [mark for start, end, mark in intervals if start <= left and right <= end]
        active.sort(key=lambda mark: (tuple(mark.get("path", [])), mark["id"]))
        run = {"text": text}
        if active:
            run["marks"] = [mark["id"] for mark in active]
        runs.append(run)
    linebreaks = [atom for atom in source.get("atoms", []) if atom.get("kind") == "linebreak"]
    remaining_atoms = [atom for atom in source.get("atoms", []) if atom.get("kind") != "linebreak"]
    if linebreaks and "\n" in body:
        before, after = body.split("\n", 1)
        runs = [
            {"text": before},
            {"atom": linebreaks[0]["id"], "marks": []},
            {"text": "\n" + after},
        ]
        remaining_atoms.extend(linebreaks[1:])
    for atom in remaining_atoms:
        runs.append({"atom": atom["id"], "marks": []})
    runs.extend(note_runs)
    return runs
