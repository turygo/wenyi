"""LLM 公共接口、ModelRef、配置校验与 provider 请求方言的离线契约测试。"""

from __future__ import annotations

import unittest

from tests.fake_llm import fake_llm_dict
from trans_novel.config import (
    LLMConfig,
    ModelRef,
    ProviderModelConfig,
    ReasoningConfig,
)
from trans_novel.llm import AgentRouter, FakeClient, build_client, parse_json_loose


class TestParseJsonLoose(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json_loose('{"a":1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(parse_json_loose("```json\n[1,2,3]\n```"), [1, 2, 3])

    def test_surrounded_by_prose(self):
        text = '思考结束。结果如下：["译文1","译文2"] 完毕。'
        self.assertEqual(parse_json_loose(text), ["译文1", "译文2"])

    def test_failure(self):
        with self.assertRaises(ValueError):
            parse_json_loose("没有任何 JSON 内容")

    def test_inner_ascii_quotes_repaired(self):
        raw = '{"translations":["磨到那份锱铢必较里暗含的"小气"二字无声地烫上面颊。"]}'
        got = parse_json_loose(raw)
        self.assertEqual(
            got["translations"][0], '磨到那份锱铢必较里暗含的"小气"二字无声地烫上面颊。'
        )

    def test_trailing_extra_brace(self):
        self.assertEqual(parse_json_loose('{"a": 1}\n}'), {"a": 1})

    def test_unescaped_quotes_with_trailing_extra_brace_keeps_object(self):
        raw = '{"translations":["他说"好"。"]}\n}'
        self.assertEqual(parse_json_loose(raw), {"translations": ['他说"好"。']})

    def test_valid_json_untouched(self):
        self.assertEqual(parse_json_loose('{"a": "b, c: d"}'), {"a": "b, c: d"})

    def test_escaped_quotes_still_work(self):
        self.assertEqual(parse_json_loose('{"a": "he said \\"hi\\""}'), {"a": 'he said "hi"'})


class TestModelRef(unittest.TestCase):
    def test_parse_splits_on_first_colon(self):
        ref = ModelRef.parse("local:qwen3:32b")
        self.assertEqual((ref.provider, ref.model), ("local", "qwen3:32b"))
        self.assertEqual(ref.full_name, "local:qwen3:32b")

    def test_parse_trims_whitespace(self):
        ref = ModelRef.parse("  deepseek : pro  ")
        self.assertEqual((ref.provider, ref.model), ("deepseek", "pro"))

    def test_parse_rejects_missing_parts(self):
        for bad in ("", "deepseek", ":pro", "deepseek:", "  :  "):
            with self.assertRaises(ValueError, msg=bad):
                ModelRef.parse(bad)

    def test_parse_rejects_invalid_provider_alias(self):
        for bad in ("DeepSeek:pro", "1deep:pro", "deep seek:pro", "deep!seek:pro"):
            with self.assertRaises(ValueError, msg=bad):
                ModelRef.parse(bad)

    def test_parse_rejects_comma_or_space_in_model_alias(self):
        for bad in ("deepseek:pro,flash", "deepseek:pro flash"):
            with self.assertRaises(ValueError, msg=bad):
                ModelRef.parse(bad)

    def test_immutable_and_equal_by_value(self):
        a, b = ModelRef("deepseek", "pro"), ModelRef("deepseek", "pro")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        with self.assertRaises(Exception):
            a.provider = "other"  # frozen dataclass


class TestConfigValidation(unittest.TestCase):
    @staticmethod
    def _cfg(**llm_kwargs) -> LLMConfig:
        return LLMConfig.model_validate(llm_kwargs)

    def test_minimal_valid_config(self):
        cfg = LLMConfig.model_validate(
            {
                "providers": {
                    "deepseek": {
                        "type": "deepseek",
                        "models": {"pro": {"id": "deepseek-v4-pro"}},
                    }
                },
                "agents": {"translator": {"model": "deepseek:pro", "fallback": []}},
            }
        )
        self.assertEqual(cfg.agents["translator"].model.full_name, "deepseek:pro")

    def test_fallback_accepts_comma_scalar(self):
        def _cfg(fallback):
            return LLMConfig.model_validate(
                {
                    "providers": {
                        "a": {"type": "fake", "models": {"m1": {"id": "m1"}, "m2": {"id": "m2"}}}
                    },
                    "agents": {"op": {"model": "a:m1", "fallback": fallback}},
                }
            )

        cfg = _cfg("a:m2")
        self.assertEqual([r.full_name for r in cfg.agents["op"].fallback], ["a:m2"])
        # 重复引用（含 primary 重复）必须在加载时拒绝
        with self.assertRaises(Exception):
            _cfg("a:m2, a:m1")
        with self.assertRaises(Exception):
            _cfg("a:m1")

    def test_rejects_empty_providers_or_agents(self):
        with self.assertRaises(Exception):
            self._cfg(providers={}, agents={"op": {"model": "a:m1"}})
        with self.assertRaises(Exception):
            self._cfg(
                providers={"a": {"type": "fake", "models": {"m1": {"id": "m1"}}}},
                agents={},
            )

    def test_rejects_unknown_provider_type(self):
        with self.assertRaisesRegex(Exception, "未知 provider 类型"):
            self._cfg(
                providers={"a": {"type": "bogus", "models": {"m1": {"id": "m1"}}}},
                agents={"op": {"model": "a:m1"}},
            )

    def test_rejects_unknown_provider_or_model_reference(self):
        with self.assertRaisesRegex(Exception, "未配置的 provider"):
            self._cfg(
                providers={"a": {"type": "fake", "models": {"m1": {"id": "m1"}}}},
                agents={"op": {"model": "nope:m1"}},
            )
        with self.assertRaisesRegex(Exception, "没有模型别名"):
            self._cfg(
                providers={"a": {"type": "fake", "models": {"m1": {"id": "m1"}}}},
                agents={"op": {"model": "a:missing"}},
            )

    def test_rejects_uppercase_or_whitespace_provider_alias(self):
        with self.assertRaisesRegex(Exception, "必须匹配"):
            self._cfg(
                providers={"DeepSeek": {"type": "fake", "models": {"m1": {"id": "m1"}}}},
                agents={"op": {"model": "DeepSeek:m1"}},
            )
        with self.assertRaisesRegex(Exception, "必须匹配"):
            self._cfg(
                providers={"deep seek": {"type": "fake", "models": {"m1": {"id": "m1"}}}},
                agents={"op": {"model": "deep seek:m1"}},
            )

    def test_rejects_blank_model_alias_keys_and_service_ids(self):
        with self.assertRaisesRegex(Exception, "模型别名"):
            self._cfg(
                providers={"a": {"type": "fake", "models": {"": {"id": "m1"}}}},
                agents={"op": {"model": "a:m1"}},
            )
        with self.assertRaisesRegex(Exception, "不能为空"):
            self._cfg(
                providers={"a": {"type": "fake", "models": {"m1": {"id": "  "}}}},
                agents={"op": {"model": "a:m1"}},
            )

    def test_rejects_model_alias_with_comma_or_whitespace(self):
        with self.assertRaisesRegex(Exception, "不得包含逗号或空白"):
            self._cfg(
                providers={"a": {"type": "fake", "models": {"m 1": {"id": "m1"}}}},
                agents={"op": {"model": "a:m 1"}},
            )

    def test_rejects_reserved_request_overrides(self):
        for reserved in ("model", "messages", "stream"):
            with self.subTest(reserved=reserved), self.assertRaisesRegex(Exception, "保留字段"):
                self._cfg(
                    providers={
                        "a": {
                            "type": "fake",
                            "models": {"m1": {"id": "m1", "request_overrides": {reserved: "x"}}},
                        }
                    },
                    agents={"op": {"model": "a:m1"}},
                )

    def test_rejects_unknown_llm_or_provider_or_model_keys(self):
        with self.assertRaisesRegex(Exception, "Extra inputs"):
            LLMConfig.model_validate(
                {
                    "providers": {"a": {"type": "fake", "models": {"m1": {"id": "m1"}}}},
                    "agents": {"op": {"model": "a:m1"}},
                    "bogus_key": 1,
                }
            )
        with self.assertRaisesRegex(Exception, "Extra inputs"):
            LLMConfig.model_validate(
                {
                    "providers": {
                        "a": {"type": "fake", "models": {"m1": {"id": "m1"}}, "bogus": 1}
                    },
                    "agents": {"op": {"model": "a:m1"}},
                }
            )

    def test_rejects_legacy_llm_provider_and_tiers(self):
        for raw in ({"provider": "deepseek"}, {"tiers": {"strong": {"model": "x"}}}):
            with self.subTest(raw=raw), self.assertRaisesRegex(Exception, "已废弃"):
                LLMConfig.model_validate(raw)

    def test_openai_compatible_requires_base_url(self):
        with self.assertRaisesRegex(Exception, "base_url"):
            LLMConfig.model_validate(
                {
                    "providers": {
                        "bailian": {"type": "openai-compatible", "models": {"m": {"id": "m"}}}
                    },
                    "agents": {"op": {"model": "bailian:m"}},
                }
            )

    def test_timeout_and_max_retries_bounds(self):
        with self.assertRaises(Exception):
            self._cfg(
                providers={"a": {"type": "fake", "timeout": 0, "models": {"m": {"id": "m"}}}},
                agents={"op": {"model": "a:m"}},
            )
        with self.assertRaises(Exception):
            self._cfg(
                providers={"a": {"type": "fake", "max_retries": -1, "models": {"m": {"id": "m"}}}},
                agents={"op": {"model": "a:m"}},
            )

    def test_rejects_invalid_reasoning_effort(self):
        with self.assertRaises(Exception):
            LLMConfig.model_validate(
                {
                    "providers": {
                        "a": {
                            "type": "fake",
                            "models": {"m": {"id": "m", "reasoning": {"effort": "ultra"}}},
                        }
                    },
                    "agents": {"op": {"model": "a:m"}},
                }
            )

    def test_fake_provider_usable_without_credentials(self):
        from trans_novel.config import Config

        cfg = Config.from_dict({"llm": fake_llm_dict()})
        router = build_client(cfg)
        self.assertIsInstance(router, AgentRouter)
        self.assertEqual(
            router.complete(
                [{"role": "user", "content": "x"}], agent="preparer", operation="prescan.digest"
            ),
            "",
        )


class TestFakeClient(unittest.TestCase):
    def test_default(self):
        client = FakeClient()
        self.assertEqual(
            client.complete(
                [{"role": "user", "content": "x"}], agent="translator", operation="translate.batch"
            ),
            "",
        )
        self.assertEqual(
            client.complete_json(
                [{"role": "user", "content": "x"}], agent="translator", operation="translate.batch"
            ),
            [],
        )

    def test_empty_handler_response_is_preserved_without_retry(self):
        client = FakeClient(handler=lambda messages, agent, operation, json_mode: "")
        self.assertEqual(
            client.complete(
                [{"role": "user", "content": "x"}], agent="translator", operation="translate.batch"
            ),
            "",
        )
        self.assertEqual(len(client.calls), 1)

    def test_handler_preserves_call_metadata(self):
        def handler(messages, agent, operation, json_mode):
            return '["A","B"]' if json_mode else "hello"

        client = FakeClient(handler=handler)
        self.assertEqual(
            client.complete(
                [{"role": "user", "content": "x"}],
                stage="Translator",
                agent="translator",
                operation="translate.batch",
            ),
            "hello",
        )
        self.assertEqual(
            client.complete_json(
                [{"role": "user", "content": "x"}], agent="translator", operation="translate.batch"
            ),
            ["A", "B"],
        )
        self.assertEqual(client.calls[0]["agent"], "translator")
        self.assertEqual(client.calls[0]["operation"], "translate.batch")
        self.assertEqual(client.calls[0]["stage"], "Translator")


class TestProviderRequestDialects(unittest.TestCase):
    messages = [{"role": "user", "content": "translate"}]

    @staticmethod
    def _model(**overrides) -> ProviderModelConfig:
        base = dict(id="deepseek-model", reasoning=ReasoningConfig(enabled=True, effort="high"))
        base.update(overrides)
        return ProviderModelConfig.model_validate(base)

    def test_deepseek_enabled_and_disabled_thinking(self):
        from trans_novel.llm.providers.transport import (
            DIALECT_DEEPSEEK,
            build_request_kwargs,
        )

        enabled = build_request_kwargs(
            DIALECT_DEEPSEEK,
            self._model(),
            self.messages,
            json_mode=True,
            max_tokens=10,
        )
        disabled = build_request_kwargs(
            DIALECT_DEEPSEEK,
            self._model(reasoning=ReasoningConfig(enabled=False)),
            self.messages,
            max_tokens=10,
        )
        self.assertEqual(enabled["model"], "deepseek-model")
        self.assertEqual(enabled["messages"], self.messages)
        self.assertIs(enabled["stream"], False)
        self.assertEqual(enabled["response_format"], {"type": "json_object"})
        self.assertEqual(enabled["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(enabled["reasoning_effort"], "high")
        self.assertEqual(enabled["max_tokens"], 4096)  # 思考模型抬到安全下限
        self.assertEqual(disabled["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", disabled)
        self.assertEqual(disabled["max_tokens"], 10)  # 关思考不抬上限

    def test_openai_and_openrouter_dialects(self):
        from trans_novel.llm.providers.transport import (
            DIALECT_OPENAI,
            DIALECT_OPENROUTER,
            build_request_kwargs,
        )

        openai = build_request_kwargs(DIALECT_OPENAI, self._model(), self.messages, max_tokens=10)
        self.assertEqual(openai["reasoning_effort"], "high")
        self.assertNotIn("extra_body", openai)
        openai_off = build_request_kwargs(
            DIALECT_OPENAI,
            self._model(reasoning=ReasoningConfig(enabled=False)),
            self.messages,
        )
        self.assertNotIn("reasoning_effort", openai_off)

        router = build_request_kwargs(
            DIALECT_OPENROUTER,
            self._model(reasoning=ReasoningConfig(enabled=False)),
            self.messages,
            max_tokens=123,
        )
        self.assertEqual(router["extra_body"], {"reasoning": {"enabled": False}})
        self.assertEqual(router["max_tokens"], 123)
        router_on = build_request_kwargs(
            DIALECT_OPENROUTER,
            self._model(reasoning=ReasoningConfig(enabled=True, effort="low")),
            self.messages,
        )
        self.assertEqual(router_on["extra_body"], {"reasoning": {"effort": "low"}})

    def test_generic_dialect_only_controls_max_tokens_floor(self):
        from trans_novel.llm.providers.transport import (
            DIALECT_GENERIC,
            build_request_kwargs,
        )

        gen = build_request_kwargs(DIALECT_GENERIC, self._model(), self.messages, max_tokens=100)
        self.assertEqual(gen["max_tokens"], 4096)
        self.assertNotIn("extra_body", gen)
        self.assertNotIn("reasoning_effort", gen)
        gen_off = build_request_kwargs(
            DIALECT_GENERIC,
            self._model(reasoning=ReasoningConfig(enabled=False)),
            self.messages,
            max_tokens=100,
        )
        self.assertEqual(gen_off["max_tokens"], 100)

    def test_request_overrides_immutable_recursive_merge(self):
        from trans_novel.llm.providers.transport import (
            DIALECT_DEEPSEEK,
            build_request_kwargs,
        )

        model = self._model(
            request_overrides={
                "extra_body": {"thinking": {"type": "enabled"}, "custom": {"deep": 1}},
                "temperature": None,
                "top_p": 0.9,
            }
        )
        overrides_snapshot = {
            k: (dict(v) if isinstance(v, dict) else v) for k, v in model.request_overrides.items()
        }
        kwargs = build_request_kwargs(DIALECT_DEEPSEEK, model, self.messages)
        # 递归合并：extra_body 的 thinking 与 custom 都被保留
        self.assertEqual(
            kwargs["extra_body"], {"thinking": {"type": "enabled"}, "custom": {"deep": 1}}
        )
        # None 整体替换生成值
        self.assertIsNone(kwargs["temperature"])
        # 非 dict 值整体替换
        self.assertEqual(kwargs["top_p"], 0.9)
        # 配置不被改写
        self.assertEqual(model.request_overrides, overrides_snapshot)


if __name__ == "__main__":
    unittest.main()
