"""用户配置与运行时策略。

公开 YAML 表达模型、模型 thinking 级别，以及质量/成本档位。
Agent 路由、重试、切分等实现细节由代码统一管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trans_novel.model_profiles import ReasoningEffort, validate_model_selection

PRODUCTION_AGENT_IDS: tuple[str, ...] = (
    "translator",
    "editor",
    "reviewer",
    "analyst",
    "preparer",
    "light-translator",
)

ProviderType = Literal[
    "deepseek",
    "opencode-go",
    "bailian",
    "openai",
    "openrouter",
    "openai-compatible",
    "ollama",
    "vllm",
    "fake",
]
QualityPreset = Literal["economy", "balanced", "quality"]

_DEFAULT_PRIMARY_MODEL = "deepseek-v4-flash:high"
_DEFAULT_FAST_MODEL = "deepseek-v4-flash:off"
_DEPRECATED_ROOT_KEYS = frozenset(
    {"language", "segment", "pipeline", "honorific", "punctuation", "paths", "output"}
)
_DEPRECATED_LLM_KEYS = frozenset({"providers", "agents", "tiers"})


@dataclass(frozen=True)
class ModelRef:
    """一次已解析模型请求；model 是发送给服务的精确模型 ID。"""

    provider: str
    model: str
    reasoning_enabled: bool = False
    reasoning_effort: ReasoningEffort = "high"

    @property
    def full_name(self) -> str:
        return f"{self.provider}:{self.model}"


class ModelRoles(BaseModel):
    """用户可选的 primary、editor、fast 三个模型角色。"""

    model_config = ConfigDict(extra="forbid")

    primary: str = _DEFAULT_PRIMARY_MODEL
    editor: str
    fast: str = _DEFAULT_FAST_MODEL

    @model_validator(mode="before")
    @classmethod
    def _inherit_editor(cls, value: Any) -> Any:
        if isinstance(value, dict) and "editor" not in value:
            return {**value, "editor": value.get("primary", _DEFAULT_PRIMARY_MODEL)}
        return value

    @field_validator("primary", "editor", "fast")
    @classmethod
    def _model_id_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("模型 ID 不能为空")
        return value


class LLMConfig(BaseModel):
    """单 Provider、三模型角色的公开 LLM 配置。"""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderType = "opencode-go"
    models: ModelRoles = Field(default_factory=ModelRoles)
    base_url: str | None = None
    api_key_env: str | None = None

    @model_validator(mode="after")
    def _validate_llm(self) -> LLMConfig:
        if self.provider == "openai-compatible":
            if not (self.base_url or "").strip():
                raise ValueError("llm.base_url：openai-compatible 必须配置服务地址")
        elif self.base_url is not None or self.api_key_env is not None:
            raise ValueError(
                "llm.base_url / llm.api_key_env 只用于 openai-compatible；"
                "标准 Provider 使用内置地址和密钥环境变量"
            )
        for role in ("primary", "editor", "fast"):
            value = getattr(self.models, role)
            try:
                validate_model_selection(self.provider, value)
            except ValueError as error:
                raise ValueError(f"llm.models.{role}：{error}") from None
        return self


class _FileConfig(BaseModel):
    """严格的公开 YAML schema。"""

    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    quality: QualityPreset = "balanced"


class SegmentConfig(BaseModel):
    """内部切分策略，不属于用户配置。"""

    max_chars_per_batch: int = 1800
    max_chars_per_segment: int = 1200


class PipelineConfig(BaseModel):
    """由质量档位展开的内部流水线策略。"""

    review: bool
    autofix_severe: bool
    polish: bool
    backtranslate_sample: float
    consistency_qa: bool
    book_understanding: bool
    naturalize: bool
    align_retry_limit: int = 2
    review_output_retries: int = 2
    rolling_context_segments: int = 6
    prescan_concurrency: int = 4
    glossary_scope: Literal["chapter", "full"] = "chapter"
    back_matter: Literal["skip", "light", "full"] = "light"
    inflight_glossary: bool = False

    @classmethod
    def for_quality(cls, quality: QualityPreset) -> PipelineConfig:
        common = {
            "align_retry_limit": 2,
            "review_output_retries": 2,
            "rolling_context_segments": 6,
            "prescan_concurrency": 4,
            "glossary_scope": "chapter",
            "back_matter": "light",
            "inflight_glossary": False,
        }
        profiles: dict[str, dict[str, Any]] = {
            "economy": {
                "review": False,
                "autofix_severe": False,
                "polish": False,
                "backtranslate_sample": 0,
                "consistency_qa": False,
                "book_understanding": False,
                "naturalize": False,
            },
            "balanced": {
                "review": True,
                "autofix_severe": True,
                "polish": True,
                "backtranslate_sample": 0,
                "consistency_qa": False,
                "book_understanding": True,
                "naturalize": True,
            },
            "quality": {
                "review": True,
                "autofix_severe": True,
                "polish": True,
                "backtranslate_sample": 0.05,
                "consistency_qa": True,
                "book_understanding": True,
                "naturalize": True,
            },
        }
        return cls.model_validate({**common, **profiles[quality]})


class OutputConfig(BaseModel):
    """单次运行的输出选择，由 CLI 覆盖。"""

    mono: bool = True
    bilingual: bool = True
    bilingual_order: Literal["target_first", "source_first"] = "target_first"


@dataclass
class Config:
    """严格文件配置，加上不暴露到 YAML 的运行时状态。"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    quality: QualityPreset = "balanced"
    source_lang: str = "auto"
    target_lang: str = "zh"
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    pipeline: PipelineConfig = field(default_factory=lambda: PipelineConfig.for_quality("balanced"))
    output: OutputConfig = field(default_factory=OutputConfig)
    honorific_strategy: Literal["keep_style", "normalize", "drop"] = "keep_style"
    punctuation_normalize: bool = True
    state_dir: str = "state"

    @classmethod
    def defaults(cls) -> Config:
        return cls()

    def apply_quality(self, quality: str) -> None:
        parsed = _FileConfig.model_validate({"quality": quality}).quality
        self.quality = parsed
        self.pipeline = PipelineConfig.for_quality(parsed)

    @staticmethod
    def default_config_text() -> str:
        return (
            resources.files("trans_novel")
            .joinpath("config.example.yaml")
            .read_text(encoding="utf-8")
        )

    @classmethod
    def create_default_file(cls, path: str = "config.yaml", *, overwrite: bool = False) -> bool:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        template = cls.default_config_text()
        if overwrite:
            target.write_text(template, encoding="utf-8")
            return True
        try:
            with target.open("x", encoding="utf-8") as stream:
                stream.write(template)
        except FileExistsError:
            return False
        return True

    @classmethod
    def load(cls, path: str = "config.yaml") -> Config:
        target = Path(path).expanduser()
        if not target.is_file():
            return cls.defaults()
        with target.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        if not isinstance(raw, dict):
            raise ValueError("配置文件顶层必须是 YAML 映射")
        deprecated = sorted(_DEPRECATED_ROOT_KEYS.intersection(raw))
        llm_raw = raw.get("llm")
        if isinstance(llm_raw, dict):
            deprecated.extend(
                f"llm.{key}" for key in sorted(_DEPRECATED_LLM_KEYS.intersection(llm_raw))
            )
        if deprecated:
            raise ValueError(
                "配置文件使用了已废弃的格式（"
                + ", ".join(deprecated)
                + "）。请删除 config.yaml 后直接运行，或执行 `wenyi init --force`。"
            )
        parsed = _FileConfig.model_validate(raw)
        return cls(
            llm=parsed.llm,
            quality=parsed.quality,
            pipeline=PipelineConfig.for_quality(parsed.quality),
        )
