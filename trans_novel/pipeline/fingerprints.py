"""节点输入指纹的纯函数公式：planner 对账与节点记录共用同一套。

指纹只覆盖影响该节点的稳定输入（config/持久化产物），刻意排除两类输入：
- 术语表全文/上下文等翻译期持续增长的输入（防中途失效，见 book_synopsis 先例）；
- 与节点无关的质量档位与批次预算（改 polish/review/naturalize/consistency_qa/
  backtranslate_sample/max_chars_per_batch 不得清空已确认译文，见
  ``translate_input_fingerprint`` 的输入集合）。

LLM 消费型节点一律纳入 ``model_profile``（provider + primary/fast 模型选择）：
换模型续跑必须失效对应节点及其后代，不能复用旧模型产物。翻译 target 是权威
续跑标记：translate 指纹包含 source/语言/全书概览/风格/标点/敬称/术语作用域/
模型配置，但不含批次预算与质量档位；附属章旁路不消费概览/风格，指纹相应收缩
（升档重开由 planner 的 back_matter 重开扫描负责，不靠指纹）。
"""

from __future__ import annotations

from trans_novel.pipeline.state import input_fingerprint, normalize_lang_code


def model_profile(config) -> str:
    """已解析的模型路由（provider + primary/fast 双角色），用于混合角色节点。"""
    return f"{_provider(config)}|{_role(config, 'primary')}|{_role(config, 'fast')}"


def primary_model_profile(config) -> str:
    """provider + primary 模型（translator/editor/analyst 等主模型节点）。"""
    return f"{_provider(config)}|{_role(config, 'primary')}"


def fast_model_profile(config) -> str:
    """provider + fast 模型（reviewer/preparer/light-translator 等快模型节点）。"""
    return f"{_provider(config)}|{_role(config, 'fast')}"


def _provider(config) -> str:
    llm = getattr(config, "llm", None)
    if llm is None:
        return ""
    return str(getattr(llm, "provider", "") or "")


def _role(config, role: str) -> str:
    llm = getattr(config, "llm", None)
    if llm is None:
        return ""
    models = getattr(llm, "models", None)
    if models is None:
        return ""
    return str(getattr(models, role, "") or "")


def glossary_semantic_fingerprint_part(terms) -> str:
    """一致性 QA 消费的术语语义稳定序列化（source/target/aliases/type/lock/status）。

    `tools glossary resolve|lock|audit` 只改既有词条的语义字段而不新增 source 时，
    QA 指纹必须随之变化（重新扫描），不能只按 source 拼写判断。
    """
    parts: list[str] = []
    for t in terms or []:
        parts.append(
            "|".join(
                [
                    str(getattr(t, "source", "") or ""),
                    str(getattr(t, "target", "") or ""),
                    ",".join(str(a) for a in (getattr(t, "aliases", None) or [])),
                    str(getattr(t, "type", "") or ""),
                    "1" if getattr(t, "locked", False) else "0",
                    str(getattr(t, "status", "") or ""),
                ]
            )
        )
    return "\n".join(sorted(parts))


def prepare_input_fingerprint(source_sha: str, src_lang: str, tgt_lang: str) -> str:
    """准备节点：源文件字节哈希 + 解析后的语言对（即运行身份的输入面）。"""
    return input_fingerprint(
        source_sha, normalize_lang_code(src_lang), normalize_lang_code(tgt_lang)
    )


def analyze_input_fingerprint(sample: str, model: str = "") -> str:
    """风格/角色分析：样章文本 + 模型路由。"""
    return input_fingerprint(sample, model)


def translate_input_fingerprint(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    *,
    book_synopsis: str,
    style_brief: str,
    punctuation_normalize: bool,
    honorific_strategy: str,
    glossary_scope: str,
    model: str = "",
) -> str:
    """正文翻译：源文 + 语言 + 全书概览 + 风格 + 模型 + 直接改变译文/提示的配置。

    不含：批次预算（max_chars_per_batch）、质量档位（polish/review/naturalize/
    backtranslate_sample/consistency_qa）——这些变化不得清空已确认译文。
    """
    return input_fingerprint(
        source_text,
        normalize_lang_code(src_lang),
        normalize_lang_code(tgt_lang),
        book_synopsis,
        style_brief,
        punctuation_normalize,
        honorific_strategy,
        glossary_scope,
        model,
    )


def back_matter_translate_input_fingerprint(
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    *,
    punctuation_normalize: bool,
    model: str = "",
) -> str:
    """附属章旁路翻译（skip/light）：只消费源文/语言/标点配置/模型。"""
    return input_fingerprint(
        source_text,
        normalize_lang_code(src_lang),
        normalize_lang_code(tgt_lang),
        punctuation_normalize,
        model,
    )


def polish_input_fingerprint(
    source_text: str,
    src_lang: str,
    style_brief: str,
    *,
    punctuation_normalize: bool,
    model: str = "",
) -> str:
    """润色：源文 + 风格 + 标点配置 + 模型（译文本体由 translate 级联驱动）。"""
    return input_fingerprint(
        source_text,
        normalize_lang_code(src_lang),
        style_brief,
        punctuation_normalize,
        model,
    )


def naturalize_input_fingerprint(
    source_text: str, *, punctuation_normalize: bool, model: str = ""
) -> str:
    return input_fingerprint(source_text, punctuation_normalize, model)


def review_input_fingerprint(
    source_text: str,
    *,
    autofix_severe: bool,
    review_output_retries: int,
    model: str = "",
) -> str:
    return input_fingerprint(source_text, autofix_severe, review_output_retries, model)


def backtranslate_input_fingerprint(
    source_text: str, *, backtranslate_sample: float, model: str = ""
) -> str:
    return input_fingerprint(source_text, backtranslate_sample, model)


def titles_input_fingerprint(
    titles: list[str], src_lang: str, tgt_lang: str, book_synopsis: str, model: str = ""
) -> str:
    return input_fingerprint(
        titles, normalize_lang_code(src_lang), normalize_lang_code(tgt_lang), book_synopsis, model
    )


def consistency_input_fingerprint(
    targets_text: str, glossary_semantic: str, model: str = ""
) -> str:
    """跨章一致性：全书译文摘要 + 术语语义 + 模型路由（两者变化都需重扫）。"""
    return input_fingerprint(targets_text, glossary_semantic, model)


def report_input_fingerprint(
    review_issues: list[dict],
    backtranslation_issues: list[dict],
    consistency_issues: list[dict],
    glossary_terms: list[str],
    titles: list[str],
) -> str:
    return input_fingerprint(
        review_issues, backtranslation_issues, consistency_issues, glossary_terms, titles
    )


def assemble_input_fingerprint(
    targets_text: str,
    *,
    mono: bool,
    bilingual: bool,
    out_format: str,
    bilingual_order: str,
) -> str:
    return input_fingerprint(targets_text, mono, bilingual, out_format, bilingual_order)
