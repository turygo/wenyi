"""根据连续译文上下文产生可追溯的局部文本编辑。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CJK = re.compile(r"[一-鿿぀-ヿ＀-￯“”‘’（）《》【】、，。！？：；…—]")
_HALF = {",": "，", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；"}
_QUOTES = {"「": "“", "」": "”", "『": "‘", "』": "’"}
_EMAIL = (
    r"(?:mailto:)?[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+"
)
_LITERAL = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\u3000-\u303f，。！？；：“”‘’]+|" + _EMAIL + r"|`[^`\n]*`",
    re.IGNORECASE,
)
_COLLAPSE = re.compile(r"。{3,}|・{2,}|\.{3,}|…+|-{2,}|—+")
_CLEAN_SPACE = frozenset("，。！？：；、”’》】")


@dataclass(frozen=True)
class TextEdit:
    """以原始目标文本的半开区间表示一项编辑。"""

    start: int
    end: int
    replacement: str


def validate_text_edits(text: str, edits: list[TextEdit]) -> None:
    """拒绝越界、逆序、重叠和非法类型的编辑。"""
    previous = 0
    for edit in edits:
        if (
            type(edit.start) is not int
            or type(edit.end) is not int
            or not isinstance(edit.replacement, str)
            or not 0 <= edit.start <= edit.end <= len(text)
            or edit.start < previous
        ):
            raise ValueError("invalid or overlapping target text edit")
        previous = edit.end


def apply_text_edits(text: str, edits: list[TextEdit]) -> str:
    """只替换声明的目标区间，保持区间外文字完全不变。"""
    validate_text_edits(text, edits)
    pieces: list[str] = []
    previous = 0
    for edit in edits:
        pieces.extend((text[previous : edit.start], edit.replacement))
        previous = edit.end
    pieces.append(text[previous:])
    return "".join(pieces)


def literal_ranges(text: str) -> list[tuple[int, int]]:
    """识别网址、邮箱和显式代码字面量，避免中文排印改变其内容。"""
    return [(match.start(), match.end()) for match in _LITERAL.finditer(text)]


def _normalize_token_spacing(
    tokens: list[tuple[int, int, str]], protected: list[bool]
) -> list[tuple[int, int, str]]:
    """用删除空白后的左侧上下文处理半角标点，保持原始坐标。"""
    result: list[tuple[int, int, str]] = []
    previous = ""
    index = 0
    while index < len(tokens):
        start, end, value = tokens[index]
        if value in _HALF and not protected[start]:
            stop = index + 1
            while (
                stop < len(tokens) and tokens[stop][2] in _HALF and not protected[tokens[stop][0]]
            ):
                stop += 1
            right = tokens[stop][2][:1] if stop < len(tokens) else ""
            convert = _CJK.fullmatch(previous) or _CJK.fullmatch(right)
            for token_start, token_end, token_value in tokens[index:stop]:
                rendered = _HALF[token_value] if convert else token_value
                result.append((token_start, token_end, rendered))
                previous = rendered[-1]
            index = stop
            continue
        if value.isspace() and previous in _CLEAN_SPACE and not protected[start]:
            value = ""
        elif value:
            previous = value[-1]
        result.append((start, end, value))
        index += 1
    return result


def _collapse_token_groups(
    tokens: list[tuple[int, int, str]], protected: list[bool], *, points: bool
) -> list[tuple[int, int, str]]:
    """合并空白清理后新相邻的折叠片段，不吞掉孤立句号。"""
    result: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        start, _end, value = tokens[index]
        eligible = value in {".", "。"} if points else value in {"……", "——"}
        if not eligible or protected[start]:
            result.append(tokens[index])
            index += 1
            continue
        stop, count, last = index + 1, 1, index
        while stop < len(tokens):
            next_start, _, next_value = tokens[stop]
            same = next_value in {".", "。"} if points else next_value == value
            if protected[next_start] or (next_value and not same):
                break
            if next_value:
                count += 1
                last = stop
            stop += 1
        if count >= (3 if points else 2):
            result.append((start, tokens[last][1], "……" if points else value))
        else:
            result.extend(tokens[index : last + 1])
        index = last + 1
    return result


def punctuation_edits(
    text: str,
    *,
    quote_state: dict[str, bool] | None = None,
    protected_ranges: list[tuple[int, int]] | None = None,
) -> list[TextEdit]:
    """在整段上下文中识别标点与空白，每个结果保留原始编辑坐标。"""
    protected = [False] * len(text)
    for start, end in [*literal_ranges(text), *(protected_ranges or [])]:
        if not 0 <= start <= end <= len(text):
            raise ValueError("invalid protected literal range")
        protected[start:end] = [True] * (end - start)
    state = quote_state if quote_state is not None else {}
    double, single = state.get("double", True), state.get("single", True)
    tokens: list[tuple[int, int, str]] = []
    position = 0
    while position < len(text):
        char = text[position]
        if protected[position]:
            tokens.append((position, position + 1, char))
            position += 1
            continue
        collapse = _COLLAPSE.match(text, position)
        if collapse is not None and not any(protected[position : collapse.end()]):
            value = "——" if char in "-—" else "……"
            tokens.append((position, collapse.end(), value))
            position = collapse.end()
            continue
        value = _QUOTES.get(char, char)
        if char == '"':
            value = "“" if double else "”"
            double = not double
        elif char == "'":
            apostrophe = (
                position > 0
                and position + 1 < len(text)
                and text[position - 1].isascii()
                and text[position - 1].isalpha()
                and text[position + 1].isascii()
                and text[position + 1].isalpha()
            )
            value = "’" if apostrophe else ("‘" if single else "’")
            if not apostrophe:
                single = not single
        tokens.append((position, position + 1, value))
        position += 1
    state.update(double=double, single=single)
    tokens = _normalize_token_spacing(tokens, protected)
    tokens = _collapse_token_groups(tokens, protected, points=True)
    tokens = _collapse_token_groups(tokens, protected, points=False)
    edits: list[TextEdit] = []
    for start, end, value in tokens:
        if value != text[start:end]:
            edits.append(TextEdit(start, end, value))
    return edits
