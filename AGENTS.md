# AGENTS.md

## Project
- 多 Agent 长篇小说翻译系统（多语言 EPUB/FB2/TXT → 中文），Python ≥3.10，uv 管理，包名 `trans-novel`。
- 源码在 `trans_novel/`，唯一入口是 CLI（`trans_novel.cli:main`）。
- 子系统：`ingest/` 输入解析、`llm/` 模型抽象+DeepSeek+FakeClient、`glossary/` SQLite 术语库、`agents/` 翻译/审校/润色/定名/提示词、`pipeline/` 编排+断点+确定性 lint、`postprocess/` 标点、`assemble/` 回填导出。

## Commands
- 安装依赖：`uv sync`
- 跑测试：`uv run python -m unittest discover -s tests`（uv 缓存不可写时前置 `UV_CACHE_DIR=/tmp/uv-cache`）
- 本地跑：`uv run trans-novel translate book.epub`（需 `export DEEPSEEK_API_KEY=...`）

## Constraints
- `state/` 是运行产物（断点状态、章节 JSON、SQLite 术语库 `glossary.db`、报告），已 gitignore，由续跑逻辑管理——切勿手改。
- 离线测试/调试切勿真发网络请求：把 `config.yaml` 的 `llm.provider` 设为 `fake`（测试统一走 `FakeClient`）。
- 调翻译行为优先改 `config.yaml`（模型档位、流水线开关、切分阈值），而不是改代码。
- 术语库在翻译期只读、按配置注入提示词；改动定名或注入逻辑前先读 `README.md`「一致性机制」，否则易破坏全书专名一致性。
- Commit message 的标题与正文一律用英文编写；代码注释可用中文。

## Context Routing
- 翻译全流程、一致性机制、模型档位分工 → 读 `README.md`（「工作流程」「一致性机制」「模型档位」）。
- 各配置项含义与默认值 → `config.yaml` 内联注释 + `README.md`「配置」。
