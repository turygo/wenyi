"""精简配置、LLM 公共接口与 Provider 请求方言的离线契约测试。"""

from __future__ import annotations

import unittest
from typing import ClassVar

from pydantic import ValidationError

from tests.fake_llm import fake_llm_dict
from trans_novel.config import Config, LLMConfig, ModelRef, PipelineConfig
from trans_novel.llm import (
    FakeClient,
    GenerationOptions,
    build_client,
    parse_json_loose,
)
from trans_novel.llm.errors import JSONParseError
from trans_novel.pipeline.fingerprints import primary_model_profile


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
        self.assertEqual(cfg.llm.models.primary, ["opencode-go/deepseek-v4-flash:high"])
        self.assertEqual(cfg.llm.models.editor, ["opencode-go/deepseek-v4-flash:high"])
        self.assertEqual(cfg.llm.models.fast, ["opencode-go/deepseek-v4-flash:off"])

    def test_omitted_editor_inherits_complete_primary_chain(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/first", "fake/second"],
                        "fast": ["fake/fast"],
                    }
                }
            }
        )
        self.assertEqual(cfg.llm.models.editor, ["fake/first", "fake/second"])

    def test_lists_are_non_empty_and_unique(self):
        for models in ([], ["fake/a", "fake/a"]):
            with self.subTest(models=models), self.assertRaises(ValidationError):
                Config.from_dict({"llm": {"models": {"primary": models}}})

    def test_old_provider_and_scalar_models_are_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"llm": {"provider": "fake", "models": {"primary": ["fake/a"]}}})
        with self.assertRaises(ValidationError):
            Config.from_dict({"llm": {"models": {"primary": "fake/a"}}})

    def test_provider_model_parsing_rules(self):
        from trans_novel.model_profiles import parse_model_selection, parse_provider_model

        self.assertEqual(
            parse_provider_model("openrouter/google/gemini:high"),
            ("openrouter", "google/gemini:high"),
        )
        self.assertEqual(parse_provider_model("ollama/qwen3:32b"), ("ollama", "qwen3:32b"))
        with self.assertRaises(ValueError):
            parse_provider_model("unknown/model")
        with self.assertRaises(ValueError):
            parse_provider_model("fake/")
        self.assertEqual(parse_model_selection("qwen3:32b").model, "qwen3:32b")

    def test_provider_endpoints_and_capabilities_are_keyed(self):
        from trans_novel.llm.registry import ProviderRegistry
        from trans_novel.llm.usage import UsageTracker
        from trans_novel.model_profiles import DIALECT_BAILIAN, DIALECT_DEEPSEEK

        cfg = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": [
                            "opencode-go/deepseek-v4-flash:high",
                            "bailian/qwen3.7-flash:off",
                        ],
                        "fast": ["bailian/qwen3.7-flash:off"],
                    }
                }
            }
        )
        registry = ProviderRegistry(cfg.llm, UsageTracker())
        go = registry.transport("opencode-go")
        bailian = registry.transport("bailian")
        self.assertIs(go, registry.transport("opencode-go"))
        self.assertEqual(go.provider, "opencode-go")
        self.assertEqual(bailian.provider, "bailian")
        self.assertEqual(go.capabilities_for("deepseek-v4-flash").request_dialect, DIALECT_DEEPSEEK)
        self.assertEqual(bailian.capabilities_for("qwen3.7-flash").request_dialect, DIALECT_BAILIAN)

    def test_model_thinking_suffix_is_validated_against_capabilities(self):
        with self.assertRaises(ValidationError):
            Config.from_dict(
                {"llm": {"models": {"primary": ["opencode-go/deepseek-v4-flash:low"]}}}
            )

    def test_openai_compatible_requires_base_url(self):
        with self.assertRaisesRegex(ValidationError, "base_url"):
            LLMConfig.model_validate({"models": {"primary": ["openai-compatible/a"]}})
        cfg = LLMConfig.model_validate(
            {
                "models": {"primary": ["openai-compatible/a"], "fast": ["fake/f"]},
                "base_url": "https://example.com/v1",
            }
        )
        self.assertEqual(cfg.base_url, "https://example.com/v1")

    def test_standard_provider_rejects_endpoint_overrides(self):
        with self.assertRaisesRegex(ValidationError, "只用于 openai-compatible"):
            LLMConfig.model_validate(
                {"models": {"primary": ["fake/a"]}, "base_url": "https://example.com"}
            )

    def test_unknown_and_deprecated_fields_fail_fast(self):
        with self.assertRaisesRegex(ValidationError, "Extra inputs"):
            Config.from_dict({"unknown": True})
        with self.assertRaisesRegex(ValueError, "已废弃"):
            Config.from_dict({"llm": {"provider": "fake", "models": {"primary": ["fake/a"]}}})

    def test_quality_profiles(self):
        self.assertTrue(PipelineConfig.for_quality("quality").polish)

    def test_fake_provider_usable_without_credentials(self):
        cfg = Config.from_dict({"llm": fake_llm_dict()})
        self.assertEqual(
            build_client(cfg).complete(
                [{"role": "user", "content": "x"}],
                agent="preparer",
                operation="terms.mine",
            ),
            "",
        )

    def test_fake_model_roles_are_qualified_lists(self):
        self.assertEqual(
            fake_llm_dict(models=("primary", "editor", "fast"))["models"],
            {
                "primary": ["fake/primary"],
                "editor": ["fake/editor"],
                "fast": ["fake/fast"],
            },
        )


class TestProviderTransportConfiguration(unittest.TestCase):
    def test_openai_compatible_overrides_do_not_redirect_builtin_provider(self):
        from trans_novel.llm.registry import ProviderRegistry
        from trans_novel.llm.usage import UsageTracker

        cfg = LLMConfig(
            models={
                "primary": ["deepseek/custom-chain", "openai-compatible/custom-model"],
            },
            base_url="https://custom.example/v1",
            api_key_env="CUSTOM_API_KEY",
        )
        registry = ProviderRegistry(cfg, UsageTracker())

        builtin = registry.transport("deepseek")
        custom = registry.transport("openai-compatible")

        self.assertEqual(builtin.base_url, "https://api.deepseek.com")
        self.assertEqual(builtin.api_key_env, "DEEPSEEK_API_KEY")
        self.assertEqual(custom.base_url, "https://custom.example/v1")
        self.assertEqual(custom.api_key_env, "CUSTOM_API_KEY")


class TestRoleProfiles(unittest.TestCase):
    def test_profiles_invalidate_only_consuming_roles(self):
        from trans_novel.pipeline.fingerprints import (
            editor_fast_model_profile,
            editor_model_profile,
            fast_model_profile,
            primary_fast_model_profile,
        )

        primary = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/p", "fake/p2"],
                        "editor": ["fake/e"],
                        "fast": ["fake/f"],
                    }
                }
            }
        )
        editor_changed = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/p", "fake/p2"],
                        "editor": ["fake/e2"],
                        "fast": ["fake/f"],
                    }
                }
            }
        )
        primary_changed = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/p2"],
                        "editor": ["fake/e"],
                        "fast": ["fake/f"],
                    }
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
        self.assertIn('"fake/p","fake/p2"', primary_model_profile(primary))
        reordered = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/p2", "fake/p"],
                        "editor": ["fake/e"],
                        "fast": ["fake/f"],
                    }
                }
            }
        )
        self.assertNotEqual(primary_model_profile(primary), primary_model_profile(reordered))
        self.assertEqual(fast_model_profile(primary), fast_model_profile(editor_changed))

    def test_role_profile_serializes_candidate_boundaries(self):
        single = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/a|fake/b"],
                        "editor": ["fake/e"],
                        "fast": ["fake/f"],
                    }
                }
            }
        )
        multiple = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/a", "fake/b"],
                        "editor": ["fake/e"],
                        "fast": ["fake/f"],
                    }
                }
            }
        )
        self.assertNotEqual(primary_model_profile(single), primary_model_profile(multiple))

    def test_role_profile_includes_base_url_for_whitespace_custom_provider(self):
        first = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["openai-compatible /model"],
                        "editor": ["fake/e"],
                        "fast": ["fake/f"],
                    },
                    "base_url": "https://one.example/v1",
                }
            }
        )
        second = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["openai-compatible /model"],
                        "editor": ["fake/e"],
                        "fast": ["fake/f"],
                    },
                    "base_url": "https://two.example/v1",
                }
            }
        )
        self.assertNotEqual(primary_model_profile(first), primary_model_profile(second))
        self.assertIn('"base_url":"https://one.example/v1"', primary_model_profile(first))


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

    def test_responses_api_conversion_preserves_controls(self):
        from trans_novel.llm.providers.transport import (
            build_request_kwargs,
            build_responses_request_kwargs,
        )
        from trans_novel.model_profiles import DIALECT_OPENAI, ModelCapabilities

        chat = build_request_kwargs(
            ModelCapabilities(
                request_dialect=DIALECT_OPENAI,
                reasoning_efforts=frozenset({"low"}),
                supports_temperature=True,
            ),
            self._model(effort="low"),
            self.messages,
            json_mode=True,
            max_tokens=8,
            generation_options=GenerationOptions(temperature=0.1),
        )
        responses = build_responses_request_kwargs(chat)
        self.assertEqual(responses["input"], self.messages)
        self.assertEqual(responses["max_output_tokens"], 4096)
        self.assertEqual(responses["reasoning"], {"effort": "low"})
        self.assertEqual(responses["text"], {"format": {"type": "json_object"}})
        self.assertEqual(responses["temperature"], 0.1)
        self.assertNotIn("messages", responses)

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
