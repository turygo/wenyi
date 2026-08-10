"""配置加载。读取 config.yaml，提供带默认值的类型化访问（pydantic v2）。

LLM 配置为显式两段式：`llm.providers`（Provider→命名模型目录）与
`llm.agents`（Agent→primary/fallback 路由）。所有模型引用形如
`<provider-alias>:<model-alias>`，按第一个冒号切分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# 六个可配置的 Agent 路由键：AgentRouter 只按 Agent 选模型路由；内部
# operation（业务标签）仅作用量/调试归因，不参与路由。每个生产 LLM 调用
# 同时显式携带 agent 与 operation。Config.load 强制 llm.agents 恰好声明这六个
# Agent（缺失/未知键在加载时即失败）；仓库随附的 config.yaml 由
# tests/test_routing.py 的离线验收测试保证。
PRODUCTION_AGENT_IDS: tuple[str, ...] = (
    "translator",
    "editor",
    "reviewer",
    "analyst",
    "preparer",
    "light-translator",
)

_PROVIDER_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_SUPPORTED_PROVIDER_TYPES = frozenset(
    {"deepseek", "openai", "openrouter", "openai-compatible", "custom", "ollama", "vllm", "fake"}
)
# 由系统生成的请求字段，request_overrides 不得覆盖。
_RESERVED_OVERRIDE_KEYS = frozenset({"model", "messages", "stream"})


@dataclass(frozen=True)
class ModelRef:
    """不可变的 <provider-alias>:<model-alias> 引用，按第一个冒号切分。

    `local:qwen3:32b` 解析为 provider=`local`、model=`qwen3:32b`。
    """

    provider: str
    model: str

    @classmethod
    def parse(cls, ref: str) -> "ModelRef":
        """从字符串解析引用：解析前先去除首尾空白，并拒绝空段、非法 Provider 别名以及包含逗号或空白的模型段。"""
        if not isinstance(ref, str):
            raise ValueError(f"模型引用必须是字符串：{ref!r}")
        provider, sep, model = ref.partition(":")
        provider = provider.strip()
        model = model.strip()
        if not sep or not provider or not model:
            raise ValueError(
                f"模型引用 {ref!r} 必须形如 <provider-alias>:<model-alias>（按第一个冒号切分）"
            )
        if not _PROVIDER_ALIAS_RE.match(provider):
            raise ValueError(
                f"模型引用 {ref!r}：provider 别名 {provider!r} 必须匹配 [a-z][a-z0-9_-]*"
            )
        if "," in model or any(ch.isspace() for ch in model):
            raise ValueError(f"模型引用 {ref!r}：model 别名 {model!r} 不得包含逗号或空白")
        return cls(provider=provider, model=model)

    @property
    def full_name(self) -> str:
        return f"{self.provider}:{self.model}"


class ReasoningConfig(BaseModel):
    """推理开关：enabled 控制是否启用推理；enabled=false 时不使用 effort，但仍保留其配置值。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    effort: Literal["low", "medium", "high"] = "high"


class ProviderModelConfig(BaseModel):
    """Provider 模型目录中的一条：id 是发给服务的精确模型 ID，alias 是目录键。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    # 推理强度在模型条目中显式配置；省略 reasoning 时默认关闭推理（enabled=False）。
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    request_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("模型服务 ID（id）不能为空")
        return value

    @field_validator("request_overrides")
    @classmethod
    def _no_reserved_overrides(cls, value: dict[str, Any]) -> dict[str, Any]:
        for key in value:
            if key in _RESERVED_OVERRIDE_KEYS:
                raise ValueError(
                    f"request_overrides 不得覆盖保留字段 {key!r}"
                    "（model/messages/stream 由系统生成）"
                )
        return value


class ProviderConfig(BaseModel):
    """一个 Provider 别名（一个端点/账号）及其命名模型目录。"""

    model_config = ConfigDict(extra="forbid")

    type: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: int = Field(default=600, gt=0)
    max_retries: int = Field(default=4, ge=0)
    models: dict[str, ProviderModelConfig] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _type_supported(cls, value: str) -> str:
        if value not in _SUPPORTED_PROVIDER_TYPES:
            raise ValueError(
                f"未知 provider 类型 {value!r}"
                f"（支持 {', '.join(sorted(_SUPPORTED_PROVIDER_TYPES))}）"
            )
        return value


class AgentRouteConfig(BaseModel):
    """Agent → primary/fallback 路由；fallback 接受 YAML 列表或逗号分隔标量。"""

    model_config = ConfigDict(extra="forbid")

    model: ModelRef
    fallback: tuple[ModelRef, ...] = ()

    @field_validator("model", mode="before")
    @classmethod
    def _parse_model(cls, value: Any) -> Any:
        if isinstance(value, ModelRef):
            return value
        if isinstance(value, str):
            return ModelRef.parse(value)
        raise ValueError(f"model 必须是 <provider>:<model> 字符串或 ModelRef：{value!r}")

    @field_validator("fallback", mode="before")
    @classmethod
    def _parse_fallback(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [part for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            parsed: list[ModelRef] = []
            for item in value:
                if isinstance(item, ModelRef):
                    parsed.append(item)
                elif isinstance(item, str):
                    parsed.append(ModelRef.parse(item))
                else:
                    raise ValueError(f"fallback 元素必须是 <provider>:<model> 字符串：{item!r}")
            return tuple(parsed)
        raise ValueError(f"fallback 必须是列表或逗号分隔字符串：{value!r}")

    @model_validator(mode="after")
    def _no_duplicate_refs(self) -> "AgentRouteConfig":
        seen = {self.model}
        for ref in self.fallback:
            if ref in seen:
                raise ValueError(
                    f"fallback 重复引用 {ref.full_name}（与 model 或其他 fallback 候选重复）"
                )
            seen.add(ref)
        return self


class LLMConfig(BaseModel):
    """LLM 顶层配置：providers 与 agents 均非空；旧 llm.provider/llm.tiers 在此拒绝。"""

    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    agents: dict[str, AgentRouteConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_schema(cls, data: Any) -> Any:
        if isinstance(data, dict) and ("provider" in data or "tiers" in data):
            raise ValueError(
                "llm.provider / llm.tiers 已废弃：请迁移到 llm.providers"
                "（Provider→命名模型目录）与 llm.agents（Agent→primary/fallback 路由）。"
                "参考仓库 config.yaml。"
            )
        return data

    @model_validator(mode="after")
    def _validate_catalog_and_routes(self) -> "LLMConfig":
        if not self.providers:
            raise ValueError("llm.providers：必须至少配置一个 provider")
        if not self.agents:
            raise ValueError("llm.agents：必须至少配置一个 Agent 路由")
        for alias, provider in self.providers.items():
            path = f"llm.providers.{alias}"
            if not alias or alias != alias.strip():
                raise ValueError(f"{path}：provider 别名不得为空或含首尾空白")
            if not _PROVIDER_ALIAS_RE.match(alias):
                raise ValueError(
                    f"{path}：provider 别名 {alias!r} 必须匹配 [a-z][a-z0-9_-]*"
                    "（必须使用小写形式，不会自动改写）"
                )
            if (
                provider.type in {"openai-compatible", "custom"}
                and not (provider.base_url or "").strip()
            ):
                raise ValueError(
                    f"{path}.base_url：provider 类型 {provider.type!r} 必须显式配置 base_url"
                )
            if not provider.models:
                raise ValueError(f"{path}.models：模型目录不能为空")
            for model_alias in provider.models:
                if not model_alias or model_alias != model_alias.strip():
                    raise ValueError(f"{path}.models：模型别名不得为空或含首尾空白")
                if "," in model_alias or any(ch.isspace() for ch in model_alias):
                    raise ValueError(f"{path}.models：模型别名 {model_alias!r} 不得包含逗号或空白")
        for agent, route in self.agents.items():
            agent_path = f"llm.agents.{agent}"
            for label, ref in _route_refs(route):
                provider = self.providers.get(ref.provider)
                if provider is None:
                    raise ValueError(
                        f"{agent_path}.{label}：引用了未配置的 provider {ref.provider!r}"
                        f"（{ref.full_name}）"
                    )
                if ref.model not in provider.models:
                    raise ValueError(
                        f"{agent_path}.{label}：provider {ref.provider!r} 没有模型别名"
                        f" {ref.model!r}（{ref.full_name}）"
                    )
        return self


def _route_refs(route: AgentRouteConfig) -> list[tuple[str, ModelRef]]:
    refs: list[tuple[str, ModelRef]] = [("model", route.model)]
    refs.extend((f"fallback[{i}]", ref) for i, ref in enumerate(route.fallback))
    return refs


class SegmentConfig(BaseModel):
    max_chars_per_batch: int = 1800
    max_chars_per_segment: int = 1200


class PipelineConfig(BaseModel):
    review: bool = True
    autofix_severe: bool = True  # 章末审校后自动重译严重项（漏译/误译）；关闭则仅上报留人工
    align_retry_limit: int = 2  # 批次翻译段数不符时的整批重试次数，超限后逐段兜底
    review_output_retries: int = Field(default=2, ge=0, le=5)
    polish: bool = False  # 默认关：润色=用 pro 模型把全书再翻一遍，最烧钱；需要时显式开
    backtranslate_sample: float = 0.05
    consistency_qa: bool = True
    rolling_context_segments: int = 6
    # 翻译前预扫源文，生成全书概览+逐章梗概注入翻译 prompt（让译者对全书有理解）。
    # 全局概览为恒定前缀可命中缓存复用；关掉可省去预扫成本。
    book_understanding: bool = True
    prescan_concurrency: int = 4  # 预扫逐章梗概的并发线程数（各章独立，1=串行）
    glossary_scope: str = (
        "chapter"  # chapter=只注入本章出现的词条+锁定人物（省 token）；full=全量表
    )
    # 附属章（Notes/Index/参考文献/致谢等，按标题关键词+全书首尾位置识别）处理档位：
    # skip=原文直通（零成本）；light=快速粗翻，跳过审校/润色/回译（省成本）；
    # full=完整翻译流水线。任何档位都不从附属章抽术语（引文人名/书名会污染全书术语表）。
    # 非法值启动即报错（成本开关必须 fail-fast）；升档重跑自动重开已完成的附属章，降档不回退。
    back_matter: Literal["skip", "light", "full"] = "light"
    # 术语供给策略：False（默认）=只用翻译前一次性定名的结果，翻译期术语表只读，
    # 不再从译后 (源文,译文) 对里抽词回灌（避免把首译直译固化成铁律、避免伪冲突）；
    # True=保留旧的"译后逐批+章末抽取"行为（日文轻小说的称呼变体/口癖场景仍需要译后确认）。
    inflight_glossary: bool = False
    # 去翻译腔（单语审读+改写，三道关卡把关采纳；实验证明轻度直译修复有效）；章级插入点在
    # 润色写回完成之后、章末审校之前，续跑靠 chapter.meta["naturalized"] 幂等——
    # 已完成章不自动补跑；存量已译书用 `tools naturalize` 手动补跑。
    naturalize: bool = True


class OutputConfig(BaseModel):
    mono: bool = True  # 产出单语版
    bilingual: bool = True  # 产出双语版
    bilingual_order: str = (
        "target_first"  # target_first=译文在上原文在下(默认); source_first=原文在上
    )


class Config(BaseModel):
    source_lang: str = "auto"  # auto | ja | en | …（auto 时由模型检测）
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
        llm = LLMConfig.model_validate(raw.get("llm") or {})
        segment = SegmentConfig.model_validate(raw.get("segment", {}) or {})
        pipeline = PipelineConfig.model_validate(raw.get("pipeline", {}) or {})
        output = OutputConfig.model_validate(raw.get("output", {}) or {})
        punct = raw.get("punctuation", {}) or {}
        # 生产配置必须恰好声明六个 Agent 路由；缺失/未知键立即失败（不猜测、不补默认）。
        missing = set(PRODUCTION_AGENT_IDS) - set(llm.agents)
        unknown = set(llm.agents) - set(PRODUCTION_AGENT_IDS)
        if missing or unknown:
            raise ValueError(
                "llm.agents 必须恰好声明六个 Agent 路由："
                + ", ".join(PRODUCTION_AGENT_IDS)
                + f"（缺失：{sorted(missing)}；未知：{sorted(unknown)}）"
            )
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
