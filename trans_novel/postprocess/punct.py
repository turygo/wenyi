"""译文标点规范化、章节标题数字风格规范化。

确定性兜底（提示词已要求，这里再保一道）：
- 日式引号 「」→ “”，『』→ ‘’；
- 英式直引号 "→ “/”（按出现次序配对），' → ‘/’（按次序配对，撇号尽量保留）；
- 半角 , . ! ? : ; 在中文语境（相邻为 CJK）→ 全角 ，。！？：；；
- 连续点号 ... / 。。。 / ・・・ → ……；-- 或 — → ——。

策略保守：英文/数字串内部的半角标点（如 9.11、Mr. Smith）不误伤——
仅当半角标点紧邻 CJK 字符时才转全角。
"""

from __future__ import annotations

import re

_CJK = (
    "一-鿿"      # CJK 统一汉字
    "぀-ヿ"      # 假名（保险）
    "＀-￯"      # 全角符号
    "“”‘’（）《》【】、，。！？：；…—"
)
_CJK_RE = f"[{_CJK}]"

# 半角标点 → 全角
_HALF_TO_FULL = {",": "，", ".": "。", "!": "！", "?": "？", ":": "：", ";": "；"}


def _convert_quotes(text: str) -> str:
    # 日式引号直接映射
    text = text.translate(str.maketrans({"「": "“", "」": "”", "『": "‘", "』": "’"}))

    # 英式直双引号：按出现次序交替配对 → “ ”
    out = []
    open_dq = True
    for ch in text:
        if ch == '"':
            out.append("“" if open_dq else "”")
            open_dq = not open_dq
        else:
            out.append(ch)
    text = "".join(out)

    # 直单引号：仅当成对出现于引用语境时转弯引号；撇号（被字母包夹）保留为 ’
    def _single(m: re.Match) -> str:
        return "’"  # 英文撇号统一为右单引号字形
    text = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", _single, text)
    # 其余成对单引号交替配对
    out, open_sq = [], True
    for ch in text:
        if ch == "'":
            out.append("‘" if open_sq else "’")
            open_sq = not open_sq
        else:
            out.append(ch)
    return "".join(out)


def _convert_ellipsis_dash(text: str) -> str:
    text = re.sub(r"。{3,}", "……", text)
    text = re.sub(r"・{2,}", "……", text)
    text = re.sub(r"\.{3,}", "……", text)
    text = re.sub(r"…+", "……", text)          # 单个/多个 … → ……
    text = re.sub(r"-{2,}", "——", text)
    text = re.sub(r"—{1,}", "——", text)        # — / —— 归一为 ——
    text = re.sub(r"——(——)+", "——", text)
    return text


def _convert_halfwidth(text: str) -> str:
    """半角 ,.!?:; 紧邻 CJK 时转全角。"""
    def repl(m: re.Match) -> str:
        return _HALF_TO_FULL[m.group(0)]

    # 标点左侧或右侧是 CJK 即转（避免误伤英文/数字内部）
    pattern = re.compile(
        rf"(?<={_CJK_RE})[,.!?:;]|[,.!?:;](?={_CJK_RE})"
    )
    return pattern.sub(repl, text)


def normalize_zh(text: str) -> str:
    """把一段中文译文的标点规范化为简体中文通用全角标点。"""
    if not text:
        return text
    text = _convert_quotes(text)
    text = _convert_ellipsis_dash(text)
    text = _convert_halfwidth(text)
    # 全角标点后的多余空格清理（中文标点后不留空格）
    text = re.sub(r"([，。！？：；、”’》】])\s+", r"\1", text)
    return text


# ── 章节标题数字风格规范化 ──────────────────────────────────────────────────
# 背景：标题译文有的走“正文首段复用”（LLM 逐段翻译，数字风格随句而定，可能
# 落成阿拉伯数字），有的走独立标题 agent（倾向汉字数字），导致同一本书里
# 「第5章」「第六章」混用。这里只在消费/输出侧做确定性规范化：统一转汉字
# 位值读法，不改状态、不影响正文中部出现的数字。
_FULL_TO_HALF_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_HEADING_NUM_RE = re.compile(r"第([0-9０-９]+)([章节部卷回])")
_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ("", "十", "百", "千")


def _int_to_cn(n: int) -> str:
    """把 1..9999 的整数转为汉字位值读法（如 105→一百零五，1024→一千零二十四）。"""
    s = str(n)
    length = len(s)
    parts: list[str] = []
    pending_zero = False
    for i, ch in enumerate(s):
        d = int(ch)
        pos = length - i - 1  # 0=个位，1=十位，2=百位，3=千位
        if d == 0:
            pending_zero = True
            continue
        if pending_zero:
            parts.append("零")
            pending_zero = False
        parts.append(_CN_DIGITS[d])
        if pos > 0:
            parts.append(_CN_UNITS[pos])
    cn = "".join(parts)
    if length == 2 and cn.startswith("一十"):
        cn = cn[1:]  # 10~19 说“十X”，不说“一十X”
    return cn


def normalize_heading_numbering(text: str) -> str:
    """把标题开头的「第<阿拉伯数字><章|节|部|卷|回>」转为汉字位值读法。

    仅处理字符串（去除首尾空白后）开头的编号，中部出现的编号不动；已是汉字
    数字或不匹配模式的输入原样返回；范围外（0 或 >9999）原样返回；幂等。
    """
    if not text:
        return text
    lstripped = text.lstrip()
    prefix = text[: len(text) - len(lstripped)]
    m = _HEADING_NUM_RE.match(lstripped)
    if not m:
        return text
    digits = m.group(1).translate(_FULL_TO_HALF_DIGITS)
    n = int(digits)
    if not (1 <= n <= 9999):
        return text
    quant = m.group(2)
    rest = lstripped[m.end():]
    return f"{prefix}第{_int_to_cn(n)}{quant}{rest}"
