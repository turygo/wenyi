"""测试用：按 agent 类型路由的 FakeClient handler，驱动整条流水线（离线）。"""

from __future__ import annotations

import json
import re


def _count_numbered(text: str) -> int:
    return len(re.findall(r"^\[(\d+)\]", text, re.M))


def routing_handler(messages, tier, json_mode):
    system = messages[0]["content"]
    user = messages[-1]["content"]

    if "语言识别器" in system:
        return json.dumps({"language": "ja"}, ensure_ascii=False)

    if "前期分析师" in system:
        return json.dumps({
            "genre": "校园", "tone": "冷峻", "style_guide": "克制",
            "characters": [{"source": "綾小路", "target": "绫小路", "gender": "男"}],
            "terms": [],
            "conventions": "年代统一用'20世纪90年代'；星期统一用'星期X'。",
        }, ensure_ascii=False)

    if "标题翻译" in system:
        n = _count_numbered(user)
        return json.dumps({"titles": [f"标题{i}" for i in range(n)]}, ensure_ascii=False)

    if "文学翻译" in system:
        n = _count_numbered(user)
        return json.dumps({"translations": [f"译{i}" for i in range(n)]}, ensure_ascii=False)

    if "中文润色编辑" in system:
        # prompt 含【源文对照】+【待润色中文译文】两个编号块；只按待润色块计数。
        target_block = user.split("【待润色中文译文】", 1)[-1]
        n = _count_numbered(target_block)
        return json.dumps({"polished": [f"润{i}" for i in range(n)]}, ensure_ascii=False)

    if "译文审校" in system:
        return json.dumps({"issues": []}, ensure_ascii=False)

    if "术语候选挖掘" in system:
        return json.dumps({"candidates": ["堀北"]}, ensure_ascii=False)

    if "全书定名" in system:
        surfaces = re.findall(r"^\[\d+\] (\S+?)（", user, re.M)
        return json.dumps({"terms": [
            {"source": s, "target": s, "type": "人物", "gender": "女"}
            for s in dict.fromkeys(surfaces)
        ]}, ensure_ascii=False)

    if "术语" in system and "抽取器" in system:
        return json.dumps({"terms": [
            {"source": "堀北", "target": "堀北", "type": "人物", "gender": "女"}
        ]}, ensure_ascii=False)

    if "回译译者" in system:
        n = _count_numbered(user)
        return json.dumps({"backtranslations": [f"逆{i}" for i in range(n)]}, ensure_ascii=False)

    if "保真度" in system:
        return json.dumps({"issues": []}, ensure_ascii=False)

    if "章节梗概员" in system:
        return "本章梗概：人物登场，情节推进。"

    if "全书概览员" in system:
        return "全书概览：主线与人物关系，整体基调。"

    return "{}" if json_mode else ""
