"""CLI 配置覆盖行为测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from trans_novel.cli import app
from trans_novel.config import Config


class TestCliConfig(unittest.TestCase):
    def test_translate_defaults_keep_config_switches(self):
        cfg = Config.from_dict(
            {
                "llm": {"provider": "fake", "tiers": {"strong": {"model": "p"}}},
                "pipeline": {"polish": True, "consistency_qa": False},
            }
        )
        captured = {}

        class FakeOrchestrator:
            def __init__(self, config):
                captured["polish"] = config.pipeline.polish

            def run_all(self, input_path, **kwargs):
                captured["run_all"] = kwargs
                return {
                    "report": {
                        "summary": {
                            "chapters_done": 1,
                            "chapters_total": 1,
                            "terms": 0,
                        }
                    },
                    "audit": [],
                    "qa_issues": [],
                    "output": "out.epub",
                }

        with (
            patch("trans_novel.cli._load_config", return_value=cfg),
            patch("trans_novel.pipeline.orchestrator.Orchestrator", FakeOrchestrator),
            patch("trans_novel.cli.os.path.isfile", return_value=True),
        ):
            result = CliRunner().invoke(app, ["translate", "input.txt"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(captured["polish"])
        self.assertIsNone(captured["run_all"]["do_qa"])

    def test_translate_flags_override_config_switches(self):
        cfg = Config.from_dict(
            {
                "llm": {"provider": "fake", "tiers": {"strong": {"model": "p"}}},
                "pipeline": {"polish": True, "consistency_qa": False},
            }
        )
        captured = {}

        class FakeOrchestrator:
            def __init__(self, config):
                captured["polish"] = config.pipeline.polish

            def run_all(self, input_path, **kwargs):
                captured["run_all"] = kwargs
                return {
                    "report": {
                        "summary": {
                            "chapters_done": 1,
                            "chapters_total": 1,
                            "terms": 0,
                        }
                    },
                    "audit": [],
                    "qa_issues": [],
                    "output": "out.epub",
                }

        with (
            patch("trans_novel.cli._load_config", return_value=cfg),
            patch("trans_novel.pipeline.orchestrator.Orchestrator", FakeOrchestrator),
            patch("trans_novel.cli.os.path.isfile", return_value=True),
        ):
            result = CliRunner().invoke(
                app,
                ["translate", "input.txt", "--no-polish", "--qa"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertFalse(captured["polish"])
        self.assertTrue(captured["run_all"]["do_qa"])

    def test_resume_delegates_to_translate_without_audit_argument(self):
        cfg = Config.from_dict(
            {
                "llm": {"provider": "fake", "tiers": {"strong": {"model": "p"}}},
                "pipeline": {"polish": True, "consistency_qa": False},
            }
        )
        captured = {}

        class FakeOrchestrator:
            def __init__(self, config):
                captured["polish"] = config.pipeline.polish

            def run_all(self, input_path, **kwargs):
                captured["input_path"] = input_path
                captured["run_all"] = kwargs
                return {
                    "report": {
                        "summary": {
                            "chapters_done": 1,
                            "chapters_total": 1,
                            "terms": 0,
                        }
                    },
                    "qa_issues": [],
                    "output": "out.txt",
                }

        with (
            patch("trans_novel.cli._load_config", return_value=cfg),
            patch("trans_novel.pipeline.orchestrator.Orchestrator", FakeOrchestrator),
            patch("trans_novel.cli.os.path.isfile", return_value=True),
        ):
            result = CliRunner().invoke(
                app,
                ["resume", "input.txt", "--format", "txt"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(captured["input_path"], "input.txt")
        self.assertEqual(captured["run_all"]["out_format"], "txt")
        self.assertIsNone(captured["run_all"]["out_path"])
        self.assertIsNone(captured["run_all"]["do_qa"])
        self.assertTrue(captured["polish"])

    def test_translate_missing_input_exits_before_loading_config(self):
        missing = os.path.join(tempfile.gettempdir(), "trans-novel-missing.epub")
        with patch(
            "trans_novel.cli._load_config",
            side_effect=AssertionError("config should not load"),
        ):
            result = CliRunner().invoke(app, ["translate", missing])

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("输入文件不存在", result.output)

    def test_tools_naturalize_rejects_non_integer_chapters(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "novel.txt")
            state_dir = os.path.join(d, "state")
            with open(src, "w", encoding="utf-8") as f:
                f.write("第一段。\n\n第二段。\n")
            cfg = Config.from_dict(
                {
                    "language": {"source": "ja", "target": "zh"},
                    "paths": {"state_dir": state_dir},
                }
            )
            run_dir = os.path.join(state_dir, "novel")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
                f.write('{"title": "novel", "chapters": []}')

            with patch("trans_novel.cli._load_config", return_value=cfg):
                result = CliRunner().invoke(
                    app, ["tools", "naturalize", src, "--chapters", "1,x,3"]
                )

            self.assertNotEqual(result.exit_code, 0, result.output)
            self.assertFalse(
                isinstance(result.exception, ValueError),
                "非法 --chapters 不应以未捕获的 ValueError 泄漏",
            )
            self.assertIn("--chapters", result.output)

    def test_status_does_not_create_state_directory(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "novel.txt")
            state_dir = os.path.join(d, "state")
            with open(src, "w", encoding="utf-8") as f:
                f.write("第一段。\n")
            cfg = Config.from_dict(
                {
                    "language": {"source": "ja", "target": "zh"},
                    "paths": {"state_dir": state_dir},
                }
            )

            with patch("trans_novel.cli._load_config", return_value=cfg):
                result = CliRunner().invoke(app, ["status", src])

            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("尚无进度", result.output)
            self.assertFalse(os.path.exists(state_dir))


if __name__ == "__main__":
    unittest.main()
