"""LLM 架构、provider 和 JSON 解析的离线契约测试。"""

from __future__ import annotations

import unittest

from trans_novel.config import Config, LLMConfig, TierConfig
from trans_novel.llm import FakeClient, build_client, parse_json_loose
from trans_novel.llm.tiers import resolve_tier


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


class TestResolveTier(unittest.TestCase):
    def test_fallback_chain(self):
        strong = TierConfig(model="pro")
        cheap = TierConfig(model="flash")
        fast = TierConfig(model="flash", options={"thinking": False})

        tiers = {"strong": strong, "cheap": cheap, "fast": fast}
        self.assertIs(resolve_tier(tiers, "fast"), fast)
        self.assertIs(resolve_tier(tiers, "cheap"), cheap)
        self.assertIs(resolve_tier(tiers, "strong"), strong)
        tiers_without_fast = {"strong": strong, "cheap": cheap}
        self.assertIs(resolve_tier(tiers_without_fast, "fast"), cheap)
        tiers_only_strong = {"strong": strong}
        self.assertIs(resolve_tier(tiers_only_strong, "fast"), strong)
        self.assertIs(resolve_tier(tiers_only_strong, "cheap"), strong)
        self.assertIs(resolve_tier(tiers, "unknown"), strong)


class TestFakeClient(unittest.TestCase):
    def test_default(self):
        client = FakeClient()
        self.assertEqual(client.complete([{"role": "user", "content": "x"}]), "")
        self.assertEqual(client.complete_json([{"role": "user", "content": "x"}]), [])

    def test_empty_handler_response_is_preserved_without_retry(self):
        client = FakeClient(handler=lambda messages, tier, json_mode: "")
        self.assertEqual(client.complete([{"role": "user", "content": "x"}]), "")
        self.assertEqual(len(client.calls), 1)

    def test_handler_preserves_call_metadata(self):
        def handler(messages, tier, json_mode):
            return '["A","B"]' if json_mode else "hello"

        client = FakeClient(handler=handler)
        self.assertEqual(
            client.complete(
                [{"role": "user", "content": "x"}],
                stage="Translator",
                operation="translate.batch",
            ),
            "hello",
        )
        self.assertEqual(client.complete_json([{"role": "user", "content": "x"}]), ["A", "B"])
        self.assertEqual(client.calls[0]["operation"], "translate.batch")
        self.assertEqual(client.calls[0]["stage"], "Translator")


class TestProviderFactory(unittest.TestCase):
    @staticmethod
    def _config(provider: str) -> Config:
        return Config(
            llm=LLMConfig(
                provider=provider,
                base_url="https://example.test/v1"
                if provider in {"custom", "openai-compatible"}
                else None,
                tiers={"strong": TierConfig(model="test-model")},
            )
        )

    def test_dispatches_every_supported_provider(self):
        from trans_novel.llm.providers.deepseek import DeepSeekClient
        from trans_novel.llm.providers.ollama import OllamaClient
        from trans_novel.llm.providers.openai import OpenAIClient
        from trans_novel.llm.providers.openai_compatible import OpenAICompatibleClient
        from trans_novel.llm.providers.openrouter import OpenRouterClient
        from trans_novel.llm.providers.vllm import VLLMClient

        cases = {
            "deepseek": DeepSeekClient,
            "openai": OpenAIClient,
            "openrouter": OpenRouterClient,
            "openai-compatible": OpenAICompatibleClient,
            "custom": OpenAICompatibleClient,
            "ollama": OllamaClient,
            "vllm": VLLMClient,
            "fake": FakeClient,
        }
        for provider, client_type in cases.items():
            with self.subTest(provider=provider):
                self.assertIsInstance(build_client(self._config(provider)), client_type)

    def test_unknown_provider_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "未知 provider：unknown"):
            build_client(self._config("unknown"))

    def test_deepseek_uses_defaults_when_tiers_are_omitted(self):
        client = build_client(Config(llm=LLMConfig(provider="deepseek")))
        self.assertEqual(client.tiers["strong"].model, "deepseek-v4-pro")
        self.assertFalse(client.tiers["fast"].options.thinking)

    def test_custom_provider_requires_base_url(self):
        with self.assertRaisesRegex(ValueError, "llm.base_url"):
            build_client(
                Config(
                    llm=LLMConfig(
                        provider="custom", tiers={"strong": TierConfig(model="test-model")}
                    )
                )
            )


class TestProviderRequestWiring(unittest.TestCase):
    messages = [{"role": "user", "content": "translate"}]

    def test_deepseek_explicitly_controls_thinking_and_json_mode(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.deepseek import DeepSeekTierOptions, build_request_kwargs

        enabled = build_request_kwargs(
            ResolvedTier("deepseek-model", DeepSeekTierOptions()),
            self.messages,
            json_mode=True,
            max_tokens=10,
        )
        disabled = build_request_kwargs(
            ResolvedTier("deepseek-model", DeepSeekTierOptions(thinking=False)),
            self.messages,
        )
        self.assertEqual(enabled["response_format"], {"type": "json_object"})
        self.assertEqual(enabled["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(enabled["reasoning_effort"], "high")
        self.assertEqual(enabled["max_tokens"], 4096)
        self.assertEqual(disabled["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", disabled)

    def test_openrouter_and_custom_request_dialects(self):
        from trans_novel.llm.providers._openai_compatible import ResolvedTier
        from trans_novel.llm.providers.openai_compatible import (
            OpenAICompatibleTierOptions,
        )
        from trans_novel.llm.providers.openai_compatible import (
            build_request_kwargs as build_custom_request,
        )
        from trans_novel.llm.providers.openrouter import (
            OpenRouterTierOptions,
        )
        from trans_novel.llm.providers.openrouter import (
            build_request_kwargs as build_openrouter_request,
        )

        router = build_openrouter_request(
            ResolvedTier("router-model", OpenRouterTierOptions(thinking=False)),
            self.messages,
            max_tokens=123,
        )
        custom = build_custom_request(
            ResolvedTier(
                "custom-model",
                OpenAICompatibleTierOptions(extra_body={"provider": {"route": "x"}}),
            ),
            self.messages,
            json_mode=True,
            max_tokens=123,
        )
        self.assertEqual(router["extra_body"], {"reasoning": {"enabled": False}})
        self.assertEqual(router["max_tokens"], 123)
        self.assertEqual(custom["response_format"], {"type": "json_object"})
        self.assertEqual(custom["extra_body"], {"provider": {"route": "x"}})
        self.assertEqual(custom["max_tokens"], 123)

    def test_local_provider_default_base_urls(self):
        from trans_novel.llm.providers.ollama import DEFAULT_BASE_URL as ollama_url
        from trans_novel.llm.providers.vllm import DEFAULT_BASE_URL as vllm_url

        ollama = build_client(self._local_config("ollama"))
        vllm = build_client(self._local_config("vllm"))
        self.assertEqual(ollama.base_url, ollama_url)
        self.assertEqual(vllm.base_url, vllm_url)

    @staticmethod
    def _local_config(provider: str) -> Config:
        return Config(llm=LLMConfig(provider=provider, tiers={"strong": TierConfig(model="local")}))


if __name__ == "__main__":
    unittest.main()
