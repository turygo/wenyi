"""无模型的廉价校验：句段对齐、长度比异常（疑似漏译/失控）。

这些是不花 token 的第一道关，配合审校 agent 一起用。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LengthFlag:
    index: int
    ratio: float
    reason: str  # "too_short" | "too_long" | "empty"


def count_aligned(sources: list[str], targets: list[str]) -> bool:
    """段数是否一致（防整段漏译最硬的校验）。"""
    return len(sources) == len(targets)


def is_machine_literal(source: str) -> bool:
    """Whether a single token is machine-readable content that must pass through unchanged."""
    value = source.strip()
    if not value or any(char.isspace() for char in value):
        return False
    if value.upper().startswith("ISBN") or any(char in value for char in "@/#"):
        return True
    if any(char in value for char in "{}[]="):
        return True
    if ":" in value and ("_" in value or any(char.isdigit() for char in value)):
        return True
    if "." not in value or value.endswith("."):
        return False
    return all(
        part and all(char.isalnum() or char in "-_" for char in part) for part in value.split(".")
    )


def length_flags(
    sources: list[str],
    targets: list[str],
    *,
    too_short: float = 0.30,
    too_long: float = 3.0,
) -> list[LengthFlag]:
    """按 译文/原文 字符比标记可疑段。

    粗略按字符比抓异常；过小多半漏译，过大可能失控/增译。
    阈值偏宽松，只抓明显异常，避免误报。
    """
    flags: list[LengthFlag] = []
    for i, (s, t) in enumerate(zip(sources, targets, strict=False)):
        s_len = len(s.strip())
        t_len = len((t or "").strip())
        if s_len == 0:
            continue
        if t_len == 0:
            flags.append(LengthFlag(i, 0.0, "empty"))
            continue
        ratio = t_len / s_len
        if ratio < too_short:
            flags.append(LengthFlag(i, ratio, "too_short"))
        elif ratio > too_long:
            flags.append(LengthFlag(i, ratio, "too_long"))
    return flags
