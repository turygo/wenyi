"""根据连续译文上下文产生可追溯的局部文本编辑。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CJK = re.compile(r"[一-鿿぀-ヿ＀-￯“”‘’（）《》【】、，。！？：；…—]")
_HALF = {",": "，", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；"}
_QUOTES = {"「": "“", "」": "”", "『": "‘", "』": "’"}
_LITERAL = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\u3000-\u303f，。！？；：“”‘’]+|`[^`\n]*`")
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
    """识别网址和显式代码字面量，避免中文排印改变其内容。"""
    return [(match.start(), match.end()) for match in _LITERAL.finditer(text)]


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
    index = 0
    while index < len(tokens):
        if tokens[index][2] not in _HALF or protected[tokens[index][0]]:
            index += 1
            continue
        end_index = index + 1
        while (
            end_index < len(tokens)
            and tokens[end_index][2] in _HALF
            and not protected[tokens[end_index][0]]
        ):
            end_index += 1
        left = tokens[index - 1][2][-1:] if index else ""
        right = tokens[end_index][2][:1] if end_index < len(tokens) else ""
        if _CJK.fullmatch(left) or _CJK.fullmatch(right):
            for token_index in range(index, end_index):
                start, end, value = tokens[token_index]
                tokens[token_index] = (start, end, _HALF[value])
        index = end_index
    edits: list[TextEdit] = []
    previous = ""
    for start, end, value in tokens:
        if value.isspace() and previous in _CLEAN_SPACE and not protected[start]:
            value = ""
        elif value:
            previous = value[-1]
        if value != text[start:end]:
            edits.append(TextEdit(start, end, value))
    return edits
