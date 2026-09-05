"""按源语言提供提示词的语言相关片段。

各 agent 的提示词主体语言无关，差异通过这里的 `$src_label`、`$lang_guidance`、
`$term_guidance` 占位注入。prompts.render() 会按 `src` 自动填入默认值，
调用方（如 translator 需带入敬称策略）可显式覆盖。
"""

from __future__ import annotations

import re

LABELS = {
    "ja": "日文",
    "en": "英文",
    "zh": "中文",
    "ru": "俄文",
    "ko": "韩文",
    "fr": "法文",
    "de": "德文",
    "es": "西班牙文",
    "it": "意大利文",
    "pt": "葡萄牙文",
}

_ROMAN_PAGE = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")


def _roman_page_part(value: str) -> bool:
    upper = value.upper()
    if not value.islower() or not upper or not _ROMAN_PAGE.fullmatch(upper):
        return False
    numbers = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for index, char in enumerate(upper):
        number = numbers[char]
        total += (
            -number if index + 1 < len(upper) and number < numbers[upper[index + 1]] else number
        )
    return total < 100


def label(src: str) -> str:
    return LABELS.get(src, f"{src}文" if src else "原文")


def honorific_rule(strategy: str) -> str:
    return {
        "keep_style": "体现敬称所含的人物关系与语气（如 先輩→前辈、ちゃん→小X、君→可酌情保留），译法全书统一。",
        "normalize": "按统一规则处理敬称，避免同一敬称多种译法。",
        "drop": "在不影响语义和人物关系的前提下省略敬称。",
    }.get(strategy, "体现敬称语气并保持全书统一。")


def translate_guidance(src: str, honorific_strategy: str = "keep_style") -> str:
    """翻译/润色用：源语言相关的译法要点。"""
    if src == "ja":
        return (
            "- 敬称：" + honorific_rule(honorific_strategy) + "\n"
            "- 依据【角色信息】与第一人称（私/僕/俺/あたし 等）体现的语域，正确选择"
            "“他/她”等代词与说话口吻。\n"
            "- 拟声拟态词按中文小说习惯自然表达，不生硬直译。\n"
            "- 汉字词不等于中文词，按语义译，勿照搬日文汉字写法。"
        )
    if src == "en":
        return (
            "- 英文无敬称体系；Mr./Ms./Sir 等称谓按中文习惯自然处理，全书统一。\n"
            "- 依据人名性别与上下文正确选择“他/她/它”；英文不显性别处须联系上下文判断。\n"
            "- 时态、关系从句、长句按中文表达重组断句；被动语态酌情转主动，避免翻译腔。\n"
            "- 英文专有名词按通行译名规范音译/意译，并沿用对照表，全书统一。"
        )
    return "- 忠实传达原意，符合中文小说表达习惯。"


def term_guidance(src: str) -> str:
    """分析/术语抽取用：reading 字段与性别判断的语言相关说明。"""
    if src == "ja":
        return "reading 填假名读音（用于音译消歧）；人物依语气/第一人称判断性别。"
    if src == "en":
        return "reading 可留空（英文无需读音）；人物依姓名常识与上下文判断性别。"
    return "reading 可留空；人物依上下文判断性别。"


def is_machine_literal(source: str) -> bool:
    """Whether a single token is machine-readable content that must pass through unchanged."""
    value = source.strip()
    if not value or any(char.isspace() for char in value):
        return False
    if value.upper().startswith("ISBN") or any(char in value for char in "@/#"):
        return True
    if any(char in value for char in "{}[]="):
        return True
    if ":" in value and ("_" in value or any(char.isdigit() for char in value)):
        return True
    if all(_roman_page_part(part) for part in re.split(r"[-–—]", value)):
        return True
    if "." not in value or value.endswith("."):
        return False
    return all(
        part and all(char.isalnum() or char in "-_" for char in part) for part in value.split(".")
    )


def needs_translation(source: str) -> bool:
    """Whether a segment contains natural-language text that should be sent to a model."""
    return any(character.isalpha() for character in source) and not is_machine_literal(source)
