"""精简配置、LLM 公共接口与 Provider 请求方言的离线契约测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import ClassVar

from pydantic import ValidationError

from tests.fake_llm import fake_llm_dict
from trans_novel.config import Config, LLMConfig, ModelRef, PipelineConfig
from trans_novel.llm import (
    AgentRouter,
    FakeClient,
    GenerationOptions,
    build_client,
    parse_json_loose,
)
from trans_novel.llm.errors import JSONParseError


class TestParseJsonLoose(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json_loose('{"a":1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(parse_json_loose("```json\n[1,2,3]\n```"), [1, 2, 3])

    def test_surrounded_by_prose(self):
        text = '思考结束。结果如下：["译文1","译文2"] 完毕。'
        self.assertEqual(parse_json_loose(text), ["译文1", "译文2"])

    def test_failure(self):
        with self.assertRaises(JSONParseError) as caught:
            parse_json_loose("没有任何 JSON 内容")
        # 保持与 ValueError 兼容：既有调用方仍可按 ValueError 捕获该异常
        self.assertIsInstance(caught.exception, ValueError)

    def test_inner_ascii_quotes_repaired(self):
        raw = '{"translations":["磨到那份锱铢必较里暗含的"小气"二字无声地烫上面颊。"]}'
        got = parse_json_loose(raw)
        self.assertEqual(
            got["translations"][0], '磨到那份锱铢必较里暗含的"小气"二字无声地烫上面颊。'
        )

    def test_trailing_extra_brace(self):
        self.assertEqual(parse_json_loose('{"a": 1}\n}'), {"a": 1})

    def test_valid_json_untouched(self):
        self.assertEqual(parse_json_loose('{"a": "b, c: d"}'), {"a": "b, c: d"})


class TestConfigValidation(unittest.TestCase):
    def test_zero_config_defaults(self):
        cfg = Config.from_dict({})
        self.assertEqual(cfg.llm.provider, "opencode-go")
        self.assertEqual(cfg.llm.models.primary, "deepseek-v4-flash:high")
        self.assertEqual(cfg.llm.models.editor, "deepseek-v4-flash:high")
        self.assertEqual(cfg.llm.models.fast, "deepseek-v4-flash:off")
        self.assertEqual(cfg.quality, "balanced")

    def test_omitted_editor_inherits_selected_primary(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "fake",
                    "models": {"primary": "custom-primary", "fast": "custom-fast"},
                }
            }
        )
        self.assertEqual(cfg.llm.models.editor, "custom-primary")

    def test_explicit_editor_null_or_empty_is_rejected(self):
        for editor in (None, ""):
            with (
                self.subTest(editor=editor),
                self.assertRaisesRegex(ValidationError, r"llm\.models\.editor"),
            ):
                Config.from_dict(
                    {
                        "llm": {
                            "provider": "fake",
                            "models": {
                                "primary": "custom-primary",
                                "editor": editor,
                                "fast": "custom-fast",
                            },
                        }
                    }
                )

    def test_missing_file_uses_defaults_without_creating_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.yaml")
            cfg = Config.load(path)
            self.assertEqual(cfg.quality, "balanced")
            self.assertFalse(os.path.exists(path))

    def test_opencode_go_has_built_in_endpoint_credentials_and_model_profile(self):
        from trans_novel.llm.registry import ProviderRegistry
        from trans_novel.llm.usage import UsageTracker
        from trans_novel.model_profiles import DIALECT_DEEPSEEK, DIALECT_GENERIC

        transport = ProviderRegistry(Config.defaults().llm, UsageTracker()).transport()
        self.assertEqual(transport.provider, "opencode-go")
        self.assertEqual(transport.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(transport.api_key_env, "OPENCODE_API_KEY")
        flash = transport.capabilities_for("deepseek-v4-flash")
        self.assertEqual(flash.request_dialect, DIALECT_DEEPSEEK)
        self.assertEqual(flash.reasoning_efforts, frozenset({"high", "max"}))
        unknown = transport.capabilities_for("unknown-model")
        self.assertEqual(unknown.request_dialect, DIALECT_GENERIC)
        self.assertEqual(unknown.reasoning_efforts, frozenset())

    def test_bailian_has_built_in_endpoint_credentials_and_model_profiles(self):
        from trans_novel.llm.registry import ProviderRegistry
        from trans_novel.llm.usage import UsageTracker
        from trans_novel.model_profiles import DIALECT_BAILIAN

        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "bailian",
                    "models": {
                        "primary": "deepseek-v4-flash:high",
                        "fast": "qwen3.7-flash:off",
                    },
                }
            }
        )
        transport = ProviderRegistry(cfg.llm, UsageTracker()).transport()
        self.assertEqual(transport.provider, "bailian")
        self.assertEqual(
            transport.base_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(transport.api_key_env, "BAILIAN_API_KEY")
        deepseek = transport.capabilities_for("deepseek-v4-flash")
        self.assertEqual(deepseek.request_dialect, DIALECT_BAILIAN)
        self.assertEqual(
            deepseek.reasoning_efforts,
            frozenset({"low", "medium", "high", "max"}),
        )
        qwen = transport.capabilities_for("qwen3.7-flash")
        self.assertEqual(qwen.request_dialect, DIALECT_BAILIAN)
        self.assertEqual(qwen.reasoning_efforts, frozenset())

    def test_model_thinking_suffix_is_validated_against_capabilities(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "opencode-go",
                    "models": {
                        "primary": "deepseek-v4-flash:max",
                        "fast": "deepseek-v4-flash:off",
                    },
                }
            }
        )
        self.assertEqual(cfg.llm.models.primary, "deepseek-v4-flash:max")
        with self.assertRaisesRegex(
            ValidationError,
            "deepseek-v4-flash 不支持 thinking 级别 'low'.*支持：off, high, max",
        ):
            Config.from_dict(
                {
                    "llm": {
                        "provider": "opencode-go",
                        "models": {
                            "primary": "deepseek-v4-flash:low",
                            "fast": "deepseek-v4-flash:off",
                        },
                    }
                }
            )

    def test_editor_model_selection_validation_is_field_specific(self):
        with self.assertRaisesRegex(ValidationError, r"llm\.models\.editor"):
            Config.from_dict(
                {
                    "llm": {
                        "provider": "opencode-go",
                        "models": {
                            "primary": "deepseek-v4-flash:high",
                            "editor": "deepseek-v4-flash:low",
                            "fast": "deepseek-v4-flash:off",
                        },
                    }
                }
            )

    def test_known_model_rejects_unknown_thinking_suffix(self):
        with self.assertRaisesRegex(ValidationError, "未知 thinking 级别 'turbo'"):
            Config.from_dict(
                {
                    "llm": {
                        "provider": "opencode-go",
                        "models": {
                            "primary": "deepseek-v4-flash:turbo",
                            "fast": "deepseek-v4-flash:off",
                        },
                    }
                }
            )

    def test_non_thinking_colon_remains_part_of_model_id(self):
        from trans_novel.model_profiles import parse_model_selection

        selection = parse_model_selection("qwen3:32b")
        self.assertEqual(selection.model, "qwen3:32b")
        self.assertIsNone(selection.thinking)

    def test_custom_models(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "openai",
                    "models": {"primary": "gpt-5", "fast": "gpt-5-mini"},
                },
                "quality": "quality",
            }
        )
        self.assertEqual(cfg.llm.models.primary, "gpt-5")
        self.assertTrue(cfg.pipeline.polish)
        self.assertTrue(cfg.pipeline.consistency_qa)

    def test_openai_compatible_requires_base_url(self):
        with self.assertRaisesRegex(ValidationError, "base_url"):
            LLMConfig.model_validate(
                {
                    "provider": "openai-compatible",
                    "models": {"primary": "a", "fast": "b"},
                }
            )

    def test_standard_provider_rejects_endpoint_overrides(self):
        with self.assertRaisesRegex(ValidationError, "只用于 openai-compatible"):
            LLMConfig.model_validate({"provider": "deepseek", "base_url": "https://example.com"})

    def test_unknown_and_old_fields_fail_fast(self):
        with self.assertRaisesRegex(ValidationError, "Extra inputs"):
            Config.from_dict({"unknown": True})
        for raw in (
            {"pipeline": {"polish": True}},
            {"llm": {"providers": {}, "agents": {}}},
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "已废弃"):
                Config.from_dict(raw)

    def test_quality_profiles(self):
        economy = PipelineConfig.for_quality("economy")
        balanced = PipelineConfig.for_quality("balanced")
        quality = PipelineConfig.for_quality("quality")
        self.assertFalse(economy.review)
        self.assertTrue(balanced.review)
        self.assertFalse(balanced.polish)
        self.assertTrue(quality.polish)
        self.assertEqual(quality.backtranslate_sample, 0.05)

    def test_fake_provider_usable_without_credentials(self):
        cfg = Config.from_dict({"llm": fake_llm_dict()})
        router = build_client(cfg)
        self.assertIsInstance(router, AgentRouter)
        self.assertEqual(
            router.complete(
                [{"role": "user", "content": "x"}],
                agent="preparer",
                operation="prescan.digest",
            ),
            "",
        )

    def test_fake_model_tuple_roles(self):
        self.assertEqual(
            fake_llm_dict(models=("same",))["models"],
            {"primary": "same", "editor": "same", "fast": "same"},
        )
        self.assertEqual(
            fake_llm_dict(models=("primary", "fast"))["models"],
            {"primary": "primary", "editor": "primary", "fast": "fast"},
        )
        self.assertEqual(
            fake_llm_dict(models=("primary", "editor", "fast"))["models"],
            {"primary": "primary", "editor": "editor", "fast": "fast"},
        )
        with self.assertRaises(ValueError):
            fake_llm_dict(models=("one", "two", "three", "four"))


class TestRoleProfiles(unittest.TestCase):
    def test_profiles_invalidate_only_consuming_roles(self):
        from trans_novel.pipeline.fingerprints import (
            editor_fast_model_profile,
            editor_model_profile,
            fast_model_profile,
            primary_fast_model_profile,
            primary_model_profile,
        )

        primary = Config.from_dict(
            {
                "llm": {
                    "provider": "fake",
                    "models": {"primary": "p", "editor": "e", "fast": "f"},
                }
            }
        )
        editor_changed = Config.from_dict(
            {
                "llm": {
                    "provider": "fake",
                    "models": {"primary": "p", "editor": "e2", "fast": "f"},
                }
            }
        )
        primary_changed = Config.from_dict(
            {
                "llm": {
                    "provider": "fake",
                    "models": {"primary": "p2", "editor": "e", "fast": "f"},
                }
            }
        )
        self.assertEqual(primary_model_profile(primary), primary_model_profile(editor_changed))
        self.assertNotEqual(editor_model_profile(primary), editor_model_profile(editor_changed))
        self.assertNotEqual(
            editor_fast_model_profile(primary), editor_fast_model_profile(editor_changed)
        )
        self.assertEqual(
            primary_fast_model_profile(primary), primary_fast_model_profile(editor_changed)
        )
        self.assertNotEqual(primary_model_profile(primary), primary_model_profile(primary_changed))
        self.assertEqual(editor_model_profile(primary), editor_model_profile(primary_changed))
        self.assertEqual(
            editor_fast_model_profile(primary), editor_fast_model_profile(primary_changed)
        )
        self.assertEqual(primary_model_profile(primary), "fake|p")
        self.assertEqual(editor_model_profile(primary), "fake|e")
        self.assertEqual(fast_model_profile(primary), "fake|f")
        self.assertEqual(editor_fast_model_profile(primary), "fake|e|f")
        self.assertEqual(primary_fast_model_profile(primary), "fake|p|f")
        self.assertNotEqual(
            primary_fast_model_profile(primary), primary_fast_model_profile(primary_changed)
        )
        self.assertEqual(fast_model_profile(primary), fast_model_profile(editor_changed))


class TestFakeClient(unittest.TestCase):
    def test_default(self):
        client = FakeClient()
        self.assertEqual(
            client.complete(
                [{"role": "user", "content": "x"}],
                agent="translator",
                operation="translate.batch",
            ),
            "",
        )
        self.assertEqual(
            client.complete_json(
                [{"role": "user", "content": "x"}],
                agent="translator",
                operation="translate.batch",
            ),
            [],
        )

    def test_handler_preserves_call_metadata(self):
        client = FakeClient(handler=lambda messages, agent, operation, json_mode: "hello")
        self.assertEqual(
            client.complete(
                [{"role": "user", "content": "x"}],
                stage="Translator",
                agent="translator",
                operation="translate.batch",
            ),
            "hello",
        )
        self.assertEqual(client.calls[0]["agent"], "translator")
        self.assertEqual(client.calls[0]["operation"], "translate.batch")


class TestProviderRequestCapabilities(unittest.TestCase):
    messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "translate"}]

    @staticmethod
    def _model(*, enabled: bool = True, effort: str = "high") -> ModelRef:
        return ModelRef(
            "deepseek",
            "deepseek-model",
            reasoning_enabled=enabled,
            reasoning_effort=effort,
        )

    def test_deepseek_enabled_and_disabled_thinking(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import DIALECT_DEEPSEEK, ModelCapabilities

        capabilities = ModelCapabilities(
            request_dialect=DIALECT_DEEPSEEK,
            reasoning_efforts=frozenset({"high"}),
        )
        enabled = build_request_kwargs(
            capabilities,
            self._model(),
            self.messages,
            json_mode=True,
            max_tokens=10,
        )
        disabled = build_request_kwargs(
            capabilities,
            self._model(enabled=False),
            self.messages,
            max_tokens=10,
        )
        self.assertEqual(enabled["model"], "deepseek-model")
        self.assertEqual(enabled["response_format"], {"type": "json_object"})
        self.assertEqual(enabled["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(enabled["reasoning_effort"], "high")
        self.assertEqual(enabled["max_tokens"], 4096)
        self.assertEqual(disabled["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", disabled)
        self.assertEqual(disabled["max_tokens"], 10)

    def test_bailian_enabled_and_disabled_thinking(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import DIALECT_BAILIAN, ModelCapabilities

        enabled = build_request_kwargs(
            ModelCapabilities(DIALECT_BAILIAN, frozenset({"high"})),
            self._model(),
            self.messages,
            max_tokens=10,
        )
        disabled = build_request_kwargs(
            ModelCapabilities(DIALECT_BAILIAN),
            self._model(enabled=False),
            self.messages,
            max_tokens=10,
        )
        self.assertEqual(enabled["extra_body"], {"enable_thinking": True})
        self.assertEqual(enabled["reasoning_effort"], "high")
        self.assertEqual(enabled["max_tokens"], 4096)
        self.assertEqual(disabled["extra_body"], {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", disabled)
        self.assertEqual(disabled["max_tokens"], 10)

    def test_openai_and_openrouter_capabilities(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import (
            DIALECT_OPENAI,
            DIALECT_OPENROUTER,
            ModelCapabilities,
        )

        efforts = frozenset({"high"})
        openai = build_request_kwargs(
            ModelCapabilities(DIALECT_OPENAI, efforts),
            self._model(),
            self.messages,
        )
        self.assertEqual(openai["reasoning_effort"], "high")
        router = build_request_kwargs(
            ModelCapabilities(DIALECT_OPENROUTER, efforts),
            self._model(enabled=False),
            self.messages,
        )
        self.assertEqual(router["extra_body"], {"reasoning": {"enabled": False}})

    def test_unknown_capabilities_do_not_send_or_budget_reasoning(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import ModelCapabilities

        kwargs = build_request_kwargs(
            ModelCapabilities(),
            self._model(),
            self.messages,
            max_tokens=100,
        )
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertNotIn("extra_body", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)

    def test_unsupported_effort_is_disabled_instead_of_upgraded(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import DIALECT_DEEPSEEK, ModelCapabilities

        kwargs = build_request_kwargs(
            ModelCapabilities(DIALECT_DEEPSEEK, frozenset({"high"})),
            self._model(effort="low"),
            self.messages,
        )
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", kwargs)


class TestGenerationOptions(unittest.TestCase):
    def test_validation_boundaries_and_types(self):
        for temperature in (0.0, 2.0, 1, 0):
            GenerationOptions(temperature=temperature)
        for temperature in (float("nan"), float("inf"), -0.1, 2.1, True, "0.1"):
            with self.subTest(temperature=temperature), self.assertRaises((TypeError, ValueError)):
                GenerationOptions(temperature=temperature)
        GenerationOptions(seed=-(2**100))
        for seed in (True, 1.0, "1"):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError)):
                GenerationOptions(seed=seed)
        for field in ("require_catalogued_model", "require_thinking_disabled"):
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                GenerationOptions(**{field: 1})

    def test_is_immutable(self):
        options = GenerationOptions(temperature=0.1)
        with self.assertRaises((AttributeError, TypeError)):
            options.temperature = 0.2


class TestBailianGenerationCapabilities(unittest.TestCase):
    messages: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "translate"}]
    model_ids: ClassVar[tuple[str, ...]] = (
        "qwen3.7-flash",
        "qwen3.7-flash-2026-07-15",
        "qwen3.7-plus",
        "qwen3.7-plus-us",
        "qwen3.7-plus-2026-05-26",
        "qwen3.8-max",
        "deepseek-v4-flash",
        "deepseek-v4-flash-0731",
        "deepseek-v4-pro",
        "deepseek-v4-pro-us",
        "deepseek-v4-pro-0813",
    )

    def test_every_official_id_accepts_controlled_disabled_generation(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import capabilities_for

        options = GenerationOptions(
            temperature=0.1,
            require_catalogued_model=True,
            require_thinking_disabled=True,
        )
        for model in self.model_ids:
            with self.subTest(model=model):
                model_ref = ModelRef("bailian", model, reasoning_enabled=False)
                kwargs = build_request_kwargs(
                    capabilities_for("bailian", model),
                    model_ref,
                    self.messages,
                    generation_options=options,
                )
                self.assertEqual(kwargs["temperature"], 0.1)
                self.assertEqual(kwargs["extra_body"], {"enable_thinking": False})

    def test_unknown_and_unsupported_seed_fail_closed(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import capabilities_for

        unknown = ModelRef("bailian", "not-catalogued", reasoning_enabled=False)
        with self.assertRaises(ValueError):
            build_request_kwargs(
                capabilities_for("bailian", unknown.model),
                unknown,
                self.messages,
                generation_options=GenerationOptions(
                    temperature=0.1, require_catalogued_model=True
                ),
            )
        candidate = ModelRef("bailian", "qwen3.8-max", reasoning_enabled=False)
        with self.assertRaises(ValueError):
            build_request_kwargs(
                capabilities_for("bailian", candidate.model),
                candidate,
                self.messages,
                generation_options=GenerationOptions(seed=1),
            )

    def test_disabled_requirement_rejects_enabled_ref(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import capabilities_for

        model_ref = ModelRef("bailian", "qwen3.8-max", reasoning_enabled=True)
        with self.assertRaises(ValueError):
            build_request_kwargs(
                capabilities_for("bailian", model_ref.model),
                model_ref,
                self.messages,
                generation_options=GenerationOptions(require_thinking_disabled=True),
            )

    def test_absent_options_preserve_request_kwargs(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import capabilities_for

        model_ref = ModelRef("bailian", "qwen3.8-max", reasoning_enabled=False)
        capabilities = capabilities_for("bailian", model_ref.model)
        self.assertEqual(
            build_request_kwargs(capabilities, model_ref, self.messages),
            build_request_kwargs(
                capabilities,
                model_ref,
                self.messages,
                generation_options=None,
            ),
        )

    def test_qwen_enabled_thinking_omits_unverified_effort(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import capabilities_for

        for model in ("qwen3.7-plus", "qwen3.8-max"):
            with self.subTest(model=model):
                model_ref = ModelRef("bailian", model, reasoning_enabled=True)
                kwargs = build_request_kwargs(
                    capabilities_for("bailian", model_ref.model),
                    model_ref,
                    self.messages,
                )
                self.assertEqual(kwargs["extra_body"], {"enable_thinking": True})
                self.assertNotIn("reasoning_effort", kwargs)

    def test_qwen_explicit_off_disables_thinking(self):
        from trans_novel.llm.providers.transport import build_request_kwargs
        from trans_novel.model_profiles import capabilities_for

        for model in ("qwen3.7-plus", "qwen3.8-max"):
            with self.subTest(model=model):
                model_ref = ModelRef("bailian", model, reasoning_enabled=False)
                kwargs = build_request_kwargs(
                    capabilities_for("bailian", model_ref.model),
                    model_ref,
                    self.messages,
                )
                self.assertEqual(kwargs["extra_body"], {"enable_thinking": False})
                self.assertNotIn("reasoning_effort", kwargs)


if __name__ == "__main__":
    unittest.main()
