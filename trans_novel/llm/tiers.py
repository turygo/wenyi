"""模型档位解析。"""

from __future__ import annotations

from typing import TypeVar

TierConfigT = TypeVar("TierConfigT")

_TIER_FALLBACK = {"fast": ("cheap", "strong"), "cheap": ("strong",), "strong": ()}


def resolve_tier(tiers: dict[str, TierConfigT], tier: str) -> TierConfigT:
    """按更便宜优先的回退链解析档位，缺 strong 时保留 KeyError。"""
    if tier in tiers:
        return tiers[tier]
    for fallback in _TIER_FALLBACK.get(tier, ("strong",)):
        if fallback in tiers:
            return tiers[fallback]
    return tiers["strong"]
