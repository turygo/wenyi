"""审校 Agent（廉价档）+ 回译抽检。

Reviewer：逐段比对原文/译文，报漏译、增译、误译、术语违例、人称错误。
BackTranslator：把译文回译成源语言，再与原文比对，抽样发现重大语义偏离。
"""

from __future__ import annotations

from typing import Any

from ..llm.base import parse_json_loose
from . import langprofile, prompts
from .base import Agent


def _backtrans_compare_system(src: str) -> str:
    lbl = langprofile.label(src)
    return (
        f"你是翻译保真度核查员。给定原文（{lbl}）与由译文回译得到的{lbl}，"
        "判断两者语义是否一致。只报实质性偏离（信息缺失、含义改变），忽略措辞差异。"
        '仅输出 JSON：{"issues":[{"index":整数,"detail":"偏离描述"}]}，无偏离则 {"issues":[]}。'
    )


class ReviewOutputError(ValueError):
    """模型返回无法安全审校的结果；reason 是稳定的协议错误标识。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


_REVIEW_TYPES = frozenset({"missing", "added", "mistranslation", "terminology", "pronoun"})


def _validate_review_output(data: Any, segment_count: int) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise ReviewOutputError("invalid_outer_schema")
    keys = list(data)
    if keys[-2:] != ["reviewed_segments", "complete"] or keys[:-2] != ["issues"]:
        raise ReviewOutputError("invalid_outer_schema")
    reviewed_segments = data.get("reviewed_segments")
    if (
        isinstance(reviewed_segments, bool)
        or not isinstance(reviewed_segments, int)
        or reviewed_segments != segment_count
    ):
        raise ReviewOutputError("invalid_receipt")
    if data.get("complete") is not True:
        raise ReviewOutputError("invalid_receipt")
    issues = data.get("issues")
    if not isinstance(issues, list):
        raise ReviewOutputError("invalid_issues")

    validated: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise ReviewOutputError("invalid_issue")
        index = issue.get("index")
        if isinstance(index, bool):
            raise ReviewOutputError("invalid_issue_index")
        if isinstance(index, str):
            try:
                index = int(index.strip())
            except (TypeError, ValueError):
                raise ReviewOutputError("invalid_issue_index") from None
        if not isinstance(index, int) or not 0 <= index < segment_count:
            raise ReviewOutputError("invalid_issue_index")
        if issue.get("type") not in _REVIEW_TYPES:
            raise ReviewOutputError("invalid_issue_type")
        detail = issue.get("detail")
        suggestion = issue.get("suggestion")
        if (
            not isinstance(detail, str)
            or not detail.strip()
            or not isinstance(suggestion, str)
            or not suggestion.strip()
        ):
            raise ReviewOutputError("invalid_issue_text")
        issue["index"] = index
        validated.append(issue)
    return validated


class Reviewer(Agent):
    def review(
        self, sources: list[str], targets: list[str], glossary_terms=None
    ) -> list[dict[str, Any]]:
        """返回通过完整回执和字段校验的问题列表。"""
        if not sources:
            return []
        system = prompts.render("reviewer_system", src=self.src, tgt=self.tgt)
        user = prompts.render(
            "reviewer_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(glossary_terms or []),
            n=len(sources),
            pairs=prompts.numbered_pairs(sources, targets),
        )
        # 直接调用 complete，让 provider 异常原样冒泡；只有后续解析/协议错误
        # 转换为 ReviewOutputError，供编排器进行可恢复拆分。
        raw = self.client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tier="cheap",
            json_mode=True,
            stage=type(self).__name__,
            operation="review.chapter",
        )
        try:
            data = parse_json_loose(raw)
        except Exception as exc:
            raise ReviewOutputError("invalid_json") from exc
        return _validate_review_output(data, len(sources))


class BackTranslator(Agent):
    """回译抽检（廉价档）。两步：译文→源语言，再与原文比对。"""

    def backtranslate(self, targets: list[str]) -> list[str]:
        if not targets:
            return []
        system = prompts.render("backtranslate_system", src=self.src, tgt=self.tgt)
        user = prompts.render(
            "backtranslate_user",
            src=self.src,
            tgt=self.tgt,
            n=len(targets),
            numbered_target=prompts.numbered(targets),
        )
        items = self._ask_json(
            system,
            user,
            tier="fast",  # 机械回译免思考；语义比对(check)仍走 cheap
            key="backtranslations",
            default=[],
            operation="backtranslate.translate",
        )
        return [str(x) for x in items] if isinstance(items, list) else []

    def check(self, sources: list[str], targets: list[str]) -> list[dict[str, Any]]:
        """对给定（已抽样的）段做回译并比对，返回偏离问题。index 为传入列表内的下标。"""
        back = self.backtranslate(targets)
        if len(back) != len(sources):
            return []  # 回译对齐失败则跳过，不阻塞
        pairs = "\n".join(
            f"[{i}] 原文：{s}\n    回译：{b}" for i, (s, b) in enumerate(zip(sources, back))
        )
        return self.dict_items(
            self._ask_json(
                _backtrans_compare_system(self.src),
                pairs,
                tier="cheap",
                key="issues",
                default=[],
                operation="backtranslate.check",
            )
        )
