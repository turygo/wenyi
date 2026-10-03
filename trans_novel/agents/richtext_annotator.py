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
原文节点顺序和文字长度不能约束中文位置；按中文表达顺序输出 runs。
mark ID 标识原文强调、链接或样式；在对应含义的中文范围上填写 marks。
允许同一 mark 出现在多个不连续范围，允许格式范围换序。不得创造 ID、属性或链接。
每个 atom 必须恰好出现一次，放在对应语义位置；脚注放在所注释的表述之后，
不可嵌入中文词语内部。atom run 不含 text，标注其周围适用的 marks。
非空原文粗体、斜体和链接必须保留。保留中文斜体，不能转为粗体或删除。
style_evidence 中字体粗细或斜体证据同样是语义格式，必须映射到相应中文含义范围。
kind=decoration 的首字装饰不得应用到中文文字或原子对象；若 attributes 含 id/name，仍须通过
一个保留其 mark ID 的空文字 run 保存身份，不得丢失或伪造原书锚点。
page anchor 放在其原文对应的中文句界附近，不要放进词语内部。
linebreak 应尽量对应译文表达的实际断行位置；不要把原文断行当成翻译切片边界，
不得为迎合断行而改写、拆译、添加或删除文字。
返回 JSON：{"annotated":[{"id":0,"runs":[{"text":"文字","marks":["m1"]},
{"atom":"a1","marks":[]},{"text":"其余文字","marks":[]}]}]}。
每项必须包含原请求 id，不能缺失、重复或添加记录。不得输出 slot_id。"""


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
        user = json.dumps({"segments": records}, ensure_ascii=False)
        count = 0

        def call() -> list[RichTarget]:
            nonlocal count
            count += 1
            data = self._ask_json(
                _SYSTEM, user, agent="analyst", operation="richtext.annotate", strict=True
            )
            return self._parse(data, requested, sources, targets, annotated)

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
            try:
                target = RichTarget.model_validate({"runs": runs})
                validate_rich_target(sources[index], target, expected_text=targets[index])
            except (ValueError, ValidationError) as error:
                raise WorkflowProtocolError("rich_annotation_invalid_target", str(error)) from error
            parsed[index] = target
        if set(parsed) != set(requested):
            raise WorkflowProtocolError("rich_annotation_missing_record")
        result = list(defaults)
        for index, target in parsed.items():
            result[index] = target
        return result
