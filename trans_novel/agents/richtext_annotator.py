"""为既有译文建立可换序的语义格式标注，不改写译文。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from trans_novel.agents.base import Agent, WorkflowProtocolError, retry_protocol
from trans_novel.ingest.models import InlineRun, RichSource, RichTarget, validate_rich_target


@dataclass(frozen=True, slots=True)
class AnnotationBatchResult:
    targets: tuple[RichTarget, ...]
    request_count: int


_SYSTEM = """你负责将原文内联格式和引用按含义映射到既有中文译文。
中文译文不可改写：所有 text 拼接必须逐字符等于 target，包括空格和标点。
text 只能复制 target 的原有片段，不能按原文或排版习惯规范化空白、单位间距或字形。
原文节点顺序和文字长度不能约束中文位置；按中文表达顺序输出 runs。
mark ID 标识原文强调、链接或样式；在对应含义的中文范围上填写 marks。
允许同一 mark 出现在多个不连续范围，允许格式范围换序。不得创造 ID、属性或链接。
每个 atom 必须恰好出现一次，放在对应语义位置；脚注放在所注释的表述之后，
不可嵌入中文词语内部。atom run 不含 text，标注其周围适用的 marks。
note_references 是在全书原始资源上严格证明的双向引用清单，mark_id 是源链接。
对应编号若已出现在中文（含 Unicode 上标数字），将该原有编号文字绑定到此链接和上下标，
不得重复恢复。若中文确实没有对应引用编号，使用 {"note":"源链接mark_id","marks":[...]}
作为独立零文字对象，在所注释中文表述之后放置，禁止把编号加入 text。
note 对象的 marks 必须以该条 content_marks 完整有序结尾，之前只允许原书的祖先 marks；
原书编号由对象呈现，不能同时用非空 text 表示同一引用。不得对普通链接创造 note 对象。
非空原文粗体、斜体、链接、上下标和字面量格式必须保留。
semantics 列出经源标签或可靠 CSS 确认的语义，一个节点可同时具有多种语义。
required=true 的节点也必须用于其对应中文范围；它可能用于取消继承的斜体或粗体，
即使 semantics 为空，也不能丢失这个格式恢复范围。
身份保存独立于格式必要性：所有 attributes 含 id/name 的 mark 都必须在 runs 中出现，
即使 kind=style、semantics 为空或 required=false，也不能丢失这个原书身份。
有对应中文含义时，将身份 mark 放在相应中文范围；多个源身份可对应同一中文范围。
源格式含义在中文中合并时，优先让多个 mark 共享实际非空中文范围，不要机械拆分中文。
仅当粗体、斜体或 required=true 的无语义样式恢复表达已隐含、没有独立中文文字时，
允许空文字 run 标注 omission="implicit" 或 "merged"，必须包含非空 explanation，说明
对应源表达如何在中文中隐含或合并；将该 run 放在正确中文语义边界，不得伪造文字。
同一 mark 只能有一次 omission，不能同时放在非空文字或 atom 上；空 run 仍保留原 id/name。
链接、上下标、字面量、首字装饰和空源标记不得使用 omission。解释不能代替原书身份保存。
原文字词或标点没有中文对应时，在其相邻表达的正确中文语义边界输出空文字身份 run，
不能统一放到段首，也不能为保留身份添加或改写译文。
保留中文斜体，不能转为粗体或删除；上下标必须保留其相应中文或数学范围。
style_evidence 是来源证据，必要语义由 semantics 指定，不能将首字排版当成整段强调。
链接不能嵌套，同一个源链接只能形成一个连续的实体；强调可以拆为不连续范围。
kind=decoration 的首字装饰不得应用到中文文字或原子对象；若 attributes 含 id/name，仍须通过
一个保留其 mark ID 的空文字 run 保存身份，不得丢失或伪造原书锚点。
page anchor 放在其原文对应的中文句界附近，不要放进词语内部。
linebreak 应尽量对应译文表达的实际断行位置；不要把原文断行当成翻译切片边界，
不得为迎合断行而改写、拆译、添加或删除文字。
若请求含 retry_feedback，按其中的拒绝原因修复标注，仍须返回全部请求记录。
返回 JSON：{"annotated":[{"id":0,"runs":[{"text":"文字","marks":["m1"]},
{"atom":"a1","marks":[]},{"text":"其余文字","marks":[]}]}]}。
每项必须包含原请求 id，不能缺失、重复或添加记录。不得输出 slot_id。"""


def _target_version(runs: list[dict]) -> int:
    return (
        3
        if any(run.get("omission") is not None or run.get("note") is not None for run in runs)
        else 2
    )


def _text_difference(expected: str, actual: str) -> dict:
    """提供目标坐标内的首个差异；只诊断，绝不修补或放宽文字守恒。"""
    offset = next(
        (
            index
            for index, (left, right) in enumerate(zip(expected, actual, strict=False))
            if left != right
        ),
        min(len(expected), len(actual)),
    )
    return {
        "offset": offset,
        "expected_length": len(expected),
        "actual_length": len(actual),
        "expected_code_point": f"U+{ord(expected[offset]):04X}"
        if offset < len(expected)
        else "end",
        "actual_code_point": f"U+{ord(actual[offset]):04X}" if offset < len(actual) else "end",
        "expected_context": expected[max(0, offset - 12) : offset + 13],
        "actual_context": actual[max(0, offset - 12) : offset + 13],
    }


class RichTextAnnotator(Agent):
    """通过既有 analyst 路由建立并严格校验语义格式契约。"""

    def annotate_batch(
        self, sources: list[RichSource], targets: list[str]
    ) -> AnnotationBatchResult:
        if len(sources) != len(targets):
            raise ValueError("rich annotation source/target count mismatch")
        annotated = [RichTarget(runs=[InlineRun(text=text)]) for text in targets]
        requested = [i for i, source in enumerate(sources) if source.marks or source.atoms]
        if not requested:
            return AnnotationBatchResult(tuple(annotated), 0)
        records = [
            {
                "id": index,
                "source": sources[index].model_dump(mode="json"),
                "target": targets[index],
            }
            for index in requested
        ]
        request = {"segments": records}
        count = 0

        def call() -> list[RichTarget]:
            nonlocal count
            count += 1
            user = json.dumps(request, ensure_ascii=False)
            data = self._ask_json(
                _SYSTEM, user, agent="analyst", operation="richtext.annotate", strict=True
            )
            try:
                return self._parse(data, requested, sources, targets, annotated)
            except WorkflowProtocolError as error:
                # 重试沿用冻结的输入，只补充上一响应被拒绝的具体原因。
                request["retry_feedback"] = {"reason": error.reason, "detail": str(error)}
                raise

        result = retry_protocol(call, retries=self.config.pipeline.protocol_retry_limit)
        return AnnotationBatchResult(tuple(result), count)

    @staticmethod
    def _parse(data, requested, sources, targets, defaults) -> list[RichTarget]:
        if not isinstance(data, dict) or set(data) != {"annotated"}:
            raise WorkflowProtocolError("rich_annotation_invalid_schema")
        records = data["annotated"]
        if not isinstance(records, list):
            raise WorkflowProtocolError("rich_annotation_invalid_schema")
        parsed: dict[int, RichTarget] = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {"id", "runs"}:
                raise WorkflowProtocolError("rich_annotation_invalid_record")
            index = record["id"]
            if type(index) is not int or index not in requested or index in parsed:
                raise WorkflowProtocolError("rich_annotation_invalid_id")
            runs = record["runs"]
            if not isinstance(runs, list) or any(
                not isinstance(run, dict) or "slot_id" in run for run in runs
            ):
                raise WorkflowProtocolError("rich_annotation_invalid_run")
            target = None
            try:
                target = RichTarget.model_validate({"version": _target_version(runs), "runs": runs})
                validate_rich_target(sources[index], target, expected_text=targets[index])
            except (ValueError, ValidationError) as error:
                detail = f"record {index}: {error}"
                if target is not None:
                    if target.text != targets[index]:
                        difference = _text_difference(targets[index], target.text)
                        detail += "; text difference: " + json.dumps(difference, ensure_ascii=False)
                    represented = {key for run in target.runs for key in run.marks}
                    missing = sorted(
                        mark.id for mark in sources[index].marks if mark.id not in represented
                    )
                    detail += f"; unrepresented source mark IDs: {missing}"
                raise WorkflowProtocolError("rich_annotation_invalid_target", detail) from error
            parsed[index] = target
        if set(parsed) != set(requested):
            raise WorkflowProtocolError("rich_annotation_missing_record")
        result = list(defaults)
        for index, target in parsed.items():
            result[index] = target
        return result
