"""测试用：按 agent 类型路由的 FakeClient handler，驱动整条流水线（离线）。"""

from __future__ import annotations

import json
import re


def fake_llm_dict(*, models=("p",), extra_agents=None) -> dict:
    """构造离线 fake provider 的四角色精简配置。"""
    if extra_agents is not None:
        raise ValueError("新配置不支持 Agent 路由覆盖")
    if not models or len(models) > 4:
        raise ValueError("fake 模型配置必须包含 1 至 4 个模型")
    qualified = [f"fake/{model}" for model in models]
    translator = qualified[:1]
    analyst = qualified[:1] if len(models) < 3 else qualified[1:2]
    editor = analyst if len(models) < 4 else qualified[2:3]
    fast = qualified[:1] if len(models) == 1 else qualified[-1:]
    return {
        "models": {
            "translator": translator,
            "analyst": analyst,
            "editor": editor,
            "fast": fast,
        },
    }


def _count_numbered(text: str) -> int:
    return len(re.findall(r"^\[(\d+)\]", text, re.M))


def _numbered_values(text: str) -> list[str]:
    return re.findall(r"^\[\d+\]\s*(.*)$", text, re.M)


def routing_handler(messages, agent, operation, json_mode):
    system = messages[0]["content"]
    user = messages[-1]["content"]

    if "语言识别器" in system:
        return json.dumps({"language": "ja"}, ensure_ascii=False)

    if "前期分析师" in system:
        return json.dumps(
            {
                "genre": "校园",
                "tone": "冷峻",
                "style_guide": "克制",
                "characters": [{"source": "綾小路", "target": "绫小路", "gender": "男"}],
                "terms": [],
                "conventions": "年代统一用'20世纪90年代'；星期统一用'星期X'。",
            },
            ensure_ascii=False,
        )

    if "标题翻译" in system:
        n = _count_numbered(user)
        return json.dumps({"titles": [f"标题{i}" for i in range(n)]}, ensure_ascii=False)

    if "文本边界对齐器" in system:
        try:
            records = json.loads(user.rsplit("\n\n", 1)[-1])
            aligned = []
            marker = "⟪TN_SLOT⟫"
            for record in records:
                translation = record["translation"]
                count = len(record["source_parts"])
                cuts = [round(len(translation) * index / count) for index in range(1, count)]
                parts = []
                start = 0
                for cut in cuts:
                    parts.append(translation[start:cut])
                    start = cut
                parts.append(translation[start:])
                aligned.append(marker.join(parts))
            return json.dumps({"aligned": aligned}, ensure_ascii=False)
        except (KeyError, TypeError, json.JSONDecodeError):
            return json.dumps({"aligned": []}, ensure_ascii=False)

    if "文学翻译" in system:
        if "【唯一待译" in user:
            source = user.split("【唯一待译", 1)[-1].split("\n", 1)[-1].split("\n\n", 1)[0]
            return json.dumps(
                {"translation": "译" + "文" * max(1, len(source) - 1)},
                ensure_ascii=False,
            )
        sources = _numbered_values(user.split("【待译", 1)[-1])
        translations = [
            f"译{i}" + "文" * max(0, len(source) - 2) for i, source in enumerate(sources)
        ]
        return json.dumps({"translations": translations}, ensure_ascii=False)

    if "中文润色编辑" in system:
        target_block = user.split("【待润色中文译文】", 1)[-1]
        targets = _numbered_values(target_block)
        polished = [f"润{i}" + "文" * max(0, len(target) - 2) for i, target in enumerate(targets)]
        return json.dumps({"polished": polished}, ensure_ascii=False)

    if "术语候选挖掘" in system:
        return json.dumps({"candidates": ["堀北"]}, ensure_ascii=False)

    if "全书定名" in system:
        surfaces = re.findall(r"^\[\d+\] (\S+?)（", user, re.M)
        return json.dumps(
            {
                "terms": [
                    {"source": s, "target": s, "type": "人物", "gender": "女"}
                    for s in dict.fromkeys(surfaces)
                ]
            },
            ensure_ascii=False,
        )

    if "术语" in system and "抽取器" in system:
        return json.dumps(
            {"terms": [{"source": "堀北", "target": "堀北", "type": "人物", "gender": "女"}]},
            ensure_ascii=False,
        )

    return "{}" if json_mode else ""
