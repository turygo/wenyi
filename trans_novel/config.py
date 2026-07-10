"""配置加载。读取 config.yaml，提供带默认值的类型化访问（pydantic v2）。"""

from __future__ import annotations

import os
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class TierConfig(BaseModel):
    model: str
    reasoning_effort: str = "high"
    thinking: bool = True


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout: int = 600
    max_retries: int = 4
    tiers: dict[str, TierConfig] = Field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class SegmentConfig(BaseModel):
    max_chars_per_batch: int = 1800
    max_chars_per_segment: int = 1200


class PipelineConfig(BaseModel):
    review: bool = True
    autofix_severe: bool = True      # 章末审校后自动重译严重项（漏译/误译）；关闭则仅上报留人工
    align_retry_limit: int = 2       # 批次翻译段数不符时的整批重试次数，超限后逐段兜底
    polish: bool = False             # 默认关：润色=用强档把全书再翻一遍，最烧钱；需要时显式开
    backtranslate_sample: float = 0.05
    consistency_qa: bool = True
    rolling_context_segments: int = 6
    # 翻译前预扫源文，生成全书概览+逐章梗概注入翻译 prompt（让译者对全书有理解）。
    # fast 档（免思考），且全局概览为恒定前缀可命中缓存复用；关掉可省去预扫成本。
    book_understanding: bool = True
    prescan_concurrency: int = 4     # 预扫逐章梗概的并发线程数（各章独立，1=串行）
    glossary_scope: str = "chapter"  # chapter=只注入本章出现的词条+锁定人物（省 token）；full=全量表
    # 附属章（Notes/Index/参考文献/致谢等，按标题关键词+全书首尾位置识别）处理档位：
    # skip=原文直通（零成本）；light=fast 档粗翻，跳过审校/润色/回译（省成本）；
    # full=完整翻译流水线。任何档位都不从附属章抽术语（引文人名/书名会污染全书术语表）。
    # 非法值启动即报错（成本开关必须 fail-fast）；升档重跑自动重开已完成的附属章，降档不回退。
    back_matter: Literal["skip", "light", "full"] = "light"
    # 术语供给策略：False（默认）=只用翻译前一次性定名的结果，翻译期术语表只读，
    # 不再从译后 (源文,译文) 对里抽词回灌（避免把首译直译固化成铁律、避免伪冲突）；
    # True=保留旧的"译后逐批+章末抽取"行为（日文轻小说的称呼变体/口癖场景仍需要译后确认）。
    inflight_glossary: bool = False


class OutputConfig(BaseModel):
    mono: bool = True  # 产出单语版
    bilingual: bool = True  # 产出双语版
    bilingual_order: str = (
        "target_first"  # target_first=译文在上原文在下(默认); source_first=原文在上
    )


class Config(BaseModel):
    source_lang: str = "auto"        # auto | ja | en | …（auto 时由模型检测）
    target_lang: str = "zh"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    honorific_strategy: str = "keep_style"
    punctuation_normalize: bool = True  # 译文标点规范化为简体中文通用
    state_dir: str = "state"

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        lang = raw.get("language", {})
        llm_raw = raw.get("llm", {})
        tiers = {
            name: TierConfig.model_validate(t)
            for name, t in (llm_raw.get("tiers", {}) or {}).items()
        }
        llm = LLMConfig(
            provider=llm_raw.get("provider", "deepseek"),
            base_url=llm_raw.get("base_url", "https://api.deepseek.com"),
            api_key_env=llm_raw.get("api_key_env", "DEEPSEEK_API_KEY"),
            timeout=llm_raw.get("timeout", 600),
            max_retries=llm_raw.get("max_retries", 4),
            tiers=tiers,
        )
        segment = SegmentConfig.model_validate(raw.get("segment", {}) or {})
        pipeline = PipelineConfig.model_validate(raw.get("pipeline", {}) or {})
        output = OutputConfig.model_validate(raw.get("output", {}) or {})
        punct = raw.get("punctuation", {}) or {}
        return cls(
            source_lang=lang.get("source", "auto"),
            target_lang=lang.get("target", "zh"),
            llm=llm,
            segment=segment,
            pipeline=pipeline,
            output=output,
            honorific_strategy=raw.get("honorific", {}).get("strategy", "keep_style"),
            punctuation_normalize=bool(punct.get("normalize", True)),
            state_dir=raw.get("paths", {}).get("state_dir", "state"),
        )
