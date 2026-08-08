"""模型 JSON 输出的宽松解析。"""

from __future__ import annotations

import json
import re
from typing import Any


def _repair_unescaped_quotes(text: str) -> str:
    """转义 JSON 字符串值内部未转义的 ASCII 双引号。"""
    out: list[str] = []
    in_str = False
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if not in_str:
            if char == '"':
                in_str = True
            out.append(char)
        elif char == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        elif char == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in ",:]}":
                in_str = False
                out.append(char)
            else:
                out.append('\\"')
        else:
            out.append(char)
        i += 1
    return "".join(out)


def parse_json_loose(text: str) -> Any:
    """从模型输出里尽力解析 JSON。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            text = inner
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if starts:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
            return value
        except Exception:
            pass
    repaired = _repair_unescaped_quotes(text)
    starts = [index for index in (repaired.find("{"), repaired.find("[")) if index != -1]
    if starts:
        try:
            value, _ = json.JSONDecoder().raw_decode(repaired[min(starts) :])
            return value
        except Exception:
            pass
    for candidate in (
        text,
        *(
            text[start : end + 1]
            for open_char, close_char in (("[", "]"), ("{", "}"))
            for start, end in [(text.find(open_char), text.rfind(close_char))]
            if start != -1 and end > start
        ),
    ):
        try:
            return json.loads(_repair_unescaped_quotes(candidate))
        except Exception:
            continue
    raise ValueError(f"无法解析为 JSON：{text[:200]!r}")
