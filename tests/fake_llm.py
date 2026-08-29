"""测试用：按 agent 类型路由的 FakeClient handler，驱动整条流水线（离线）。"""

from __future__ import annotations

import contextlib
import json
import re


def fake_llm_dict(*, models=("p",), extra_agents=None) -> dict:
    """构造离线 fake provider 的三角色精简配置。

    传入一个模型时，所有角色共用该模型；传入两个模型时，`primary` 和 `editor` 共用第一个，
    `fast` 使用第二个；传入三个模型时，依次对应 `primary`、`editor`、`fast`。
    """
    if extra_agents is not None:
        raise ValueError("新配置不支持 Agent 路由覆盖")
    if not models or len(models) > 3:
        raise ValueError("fake 模型配置必须包含 1 至 3 个模型")
    qualified = [f"fake/{model}" for model in models]
    primary = qualified[:1]
    editor = qualified[:1] if len(models) < 3 else qualified[1:2]
    fast = qualified[:1] if len(models) == 1 else qualified[-1:]
    return {
        "models": {"primary": primary, "editor": editor, "fast": fast},
    }


def _count_numbered(text: str) -> int:
    return len(re.findall(r"^\[(\d+)\]", text, re.M))


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

    if "文学翻译" in system:
        n = _count_numbered(user)
        marker = "【EPUB 槽位协议】\n"
        if marker in user:
            try:
                expected = json.loads(user.split(marker, 1)[1].split("\n", 1)[0])
                return json.dumps(
                    {
                        "translations": [
                            {
                                "slots": [
                                    {"id": slot["id"], "core": f"译{i}"} for slot in item["slots"]
                                ]
                            }
                            for i, item in enumerate(expected)
                        ]
                    },
                    ensure_ascii=False,
                )
            except (IndexError, KeyError, TypeError, json.JSONDecodeError):
                return json.dumps({"translations": []}, ensure_ascii=False)
        return json.dumps({"translations": [f"译{i}" for i in range(n)]}, ensure_ascii=False)

    if "中文润色编辑" in system:
        target_block = user.split("【待润色中文译文】", 1)[-1]
        n = _count_numbered(target_block)
        if "【EPUB 槽位协议】" in user:
            records = []
            for line in target_block.splitlines():
                match = re.match(r"^\[\d+\] (.+)$", line)
                if match:
                    with contextlib.suppress(json.JSONDecodeError):
                        records.append(json.loads(match.group(1)))
            return json.dumps(
                {
                    "polished": [
                        {"slots": [{"id": slot["id"], "core": "润"} for slot in item]}
                        for item in records
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps({"polished": [f"润{i}" for i in range(n)]}, ensure_ascii=False)

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
