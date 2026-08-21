# 文译

专注于将多语言 EPUB、FB2 或 TXT 小说翻译成中文，并尽量保留 EPUB 原排版、图片、目录和跳转。

项目的日常入口只有一个命令：`translate`。它会完成预扫、分析、翻译、可选润色、章末审校、标点规范化和 EPUB 导出；中断后可以继续跑。

## 快速开始

从 [GitHub Releases](https://github.com/turygo/wenyi/releases/latest) 下载与系统匹配的压缩包，并用同一页面的 `SHA256SUMS.txt` 校验文件：

- Windows x64：`wenyi-windows-x64.zip`
- Linux x64 / ARM64：`wenyi-linux-x64.tar.gz` / `wenyi-linux-arm64.tar.gz`
- macOS Intel / Apple Silicon：`wenyi-macos-x64.tar.gz` / `wenyi-macos-arm64.tar.gz`

解压后直接运行：

```bash
export OPENCODE_API_KEY=sk-...
./wenyi translate book.epub
```

在 Windows PowerShell 中，先运行 `$env:OPENCODE_API_KEY = "sk-..."` 设置环境变量，
再运行 `.\wenyi.exe translate .\book.epub`。API Key 来自
[OpenCode Go](https://dev.opencode.ai/docs/go/) 订阅。没有 `config.yaml` 时程序直接使用
OpenCode Go 的 DeepSeek V4 Flash + `balanced` 默认值，不会自动写文件；需要改模型时运行 `wenyi init`。

从源码安装仍可使用 `uv tool install .`，命令名为 `trans-novel`。运行
`trans-novel --help` 或 `wenyi --help` 可查看全部命令。

翻译完成后，默认会在源文件目录下生成译文 EPUB。运行状态、章节 JSON、术语库和报告会放在 `state/` 目录下。

中断后继续：

```bash
trans-novel resume book.epub
```

查看进度：

```bash
trans-novel status book.epub
```

仅重新导出 EPUB：

```bash
trans-novel tools assemble book.epub
```

## 输入和输出

- 输入：EPUB、FB2、TXT。
- 默认输出：中文 EPUB。
- EPUB 输入会按原 XHTML 模板回填译文，尽量保留原书样式、图片、目录和锚点。
- TXT 输入会生成新的 EPUB。
- 需要纯文本时使用 `--format txt`。

示例：

```bash
trans-novel translate book.epub
trans-novel translate book.epub --format txt
trans-novel translate book.epub --chapter 3
```

## 常用开关

```bash
trans-novel translate book.epub --quality economy
trans-novel translate book.epub --quality quality
trans-novel translate book.epub --polish
trans-novel translate book.epub --no-qa
trans-novel translate book.epub --source-language ja
trans-novel translate book.epub --back-matter full
```

`--quality` 只覆盖本次运行。`balanced` 默认开启全书预扫、章末审校、严重问题自动重译和
去翻译腔；`economy` 关闭这些额外模型阶段；`quality` 额外开启全文润色、跨章一致性 QA
和 5% 回译抽检。`--polish/--no-polish` 与 `--qa/--no-qa` 可以继续覆盖单个高成本阶段。
已经翻译完成的批次会被断点续跑跳过，后来改变档位不会自动重跑旧译文。

## 配置

配置文件只回答“使用什么模型”和“选择哪个质量档位”。没有配置文件也可以直接运行；
`trans-novel init` 会生成以下精简配置：

```yaml
llm:
  provider: opencode-go
  models:
    primary: deepseek-v4-flash:high
    # editor 省略时继承 primary；显式配置可使用不同模型
    editor: deepseek-v4-flash:high
    fast: deepseek-v4-flash:off

quality: balanced
```

- `primary`：正文翻译、全局分析和定名；默认 thinking 级别为 `high`。
- `editor`：润色和自然化改写；省略时继承 `primary`，默认 thinking 级别为 `high`。
- `fast`：审校、预扫、梗概、术语抽取、回译和附属章粗翻；默认 thinking 级别为 `off`。
- `quality`：`economy`、`balanced` 或 `quality`。

模型规格可在模型 ID 最右侧追加 `:off`、`:low`、`:medium`、`:high` 或 `:max`。
程序启动时会根据 Provider/模型能力校验；不支持的级别直接报错并列出可选值，不会静默
升级或降级。没有级别后缀时，`primary` 和 `editor` 默认 `high`，`fast` 默认 `off`。

Agent 路由、重试、超时、切分、上下文窗口和并发数都是内部策略，不接受 YAML 覆盖。
Provider 使用固定的官方地址和密钥环境变量：

| Provider | 密钥环境变量 |
| --- | --- |
| `opencode-go` | `OPENCODE_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `bailian` | `BAILIAN_API_KEY` |
| `ollama` | 无 |
| `vllm` | 无 |

百炼使用华北 2（北京）的共享 OpenAI 兼容端点。正文模型保留 DeepSeek V4 Flash、
快任务改用更便宜的千问 3.7 Flash 时，可写：
```yaml
llm:
  provider: bailian
  models:
    primary: deepseek-v4-flash:high
    editor: deepseek-v4-flash:high
    fast: qwen3.7-flash:off

quality: balanced
```

OpenAI 兼容服务使用单端点配置：

```yaml
llm:
  provider: openai-compatible
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key_env: DASHSCOPE_API_KEY
  models:
    primary: qwen-max
    editor: qwen-max
    fast: qwen-turbo

quality: balanced
```

配置中只写密钥环境变量名，不能写明文密钥。旧的 `llm.providers`、`llm.agents`、
`pipeline`、`segment` 等格式已废弃，加载时会直接报错；删除旧文件后直接运行，或执行
`wenyi init --force` 生成新配置。

## 工作流程

默认连续流程大致是：

```text
读取输入
→ 解析章节、正文段落和 EPUB 目录
→ 模型识别源语言（或使用配置指定语言）
→ 分析样章，建立风格指南与初始术语表
→ 预扫整本书：逐章梗概 → 源文侧术语候选挖掘 → 一次性全书定名 → 全书概览
→ 按章、按批翻译（批后确定性 lint：引号/数字/锁定专名/未译，命中即带反馈定向重译）
→ 可选润色（润色若引入 lint 回归，该段回退润色前译文）
→ 标点规范化
→ 章末 review
→ 可选严重项自动重译
→ 可选一致性 QA
→ 回填导出 EPUB/TXT
```

每个批次翻译完成后都会写入 `state/`，所以长书中断后可以续跑。已经有译文的批次会跳过，只补未完成部分。

## 一致性机制

- **术语库**：翻译前从源文挖掘专名候选（英文走确定性统计，其他语言走 LLM 挖掘），由定名 Agent 一次性统一定名后写入 SQLite 术语库，翻译期只读；人物条目锁定后由 lint 硬校验。
- **全书理解**：翻译前预扫源文，生成全书概览和章节梗概，让早期章节也能参考全书走向。
- **滚动上下文**：章内批次串行处理，后一个批次能看到前面最近几段译文。
- **段数对齐**：每批输入 N 段，要求模型输出 N 段 JSON；段数不符会重试，仍失败则逐段兜底。
- **确定性 lint**：零成本机器校验直接引语引号保留、数字一致、锁定专名命中、整段未译；翻译后与润色后各跑一遍，命中即定向重译或回退，其余记录进报告。
- **章末 review**：`balanced` 和 `quality` 档位按章检查漏译、误译、专名、人称等语义问题，并自动重译严重项。
- **标点规范化**：译文统一为简体中文大陆常用全角标点。

## 常用工具

```bash
trans-novel tools glossary book.epub list
trans-novel tools glossary book.epub conflicts
trans-novel tools qa book.epub
trans-novel tools report book.epub
trans-novel tools assemble book.epub
```

这些工具主要用于查看术语库、检查一致性、生成报告或重新导出成品。QA 和报告默认只汇总问题，不会自动改正文。

## 模型路由

默认使用 OpenCode Go，通过其 OpenAI-compatible 端点
`https://opencode.ai/zen/go/v1` 调用 DeepSeek V4 Flash。直接使用 DeepSeek、OpenAI 或
OpenRouter 时走各自官方地址；Ollama 与 vLLM 使用本地默认地址；
`openai-compatible` 必须设置 `llm.base_url`。

内部 Agent 固定映射到三个用户模型角色：

- `primary`：正文翻译、定向重译、标题翻译、全局分析、定名和术语审计。
- `editor`：润色和自然化改写；省略时继承 `primary`。
- `fast`：章末审校、一致性检查、语言识别、预扫、梗概、术语抽取、回译和附属章粗翻。

thinking 级别由模型规格最右侧的后缀决定。请求字段由内置的逐 Provider/模型能力表
生成：OpenCode Go 的 `deepseek-v4-flash:high` 下发 `thinking=enabled` 与
`reasoning_effort=high`，`:off` 下发 `thinking=disabled`。该模型不支持 `low`，
配置为 `deepseek-v4-flash:low` 会在启动时直接报错。

## 项目结构

```text
trans_novel/
  config.example.yaml  默认配置的唯一来源，首次运行时复制到工作目录
  ingest/       输入解析、EPUB/FB2/TXT 切分
  llm/          LLM 抽象接口、provider factory、内置 providers、FakeClient
  glossary/     SQLite 术语库、源文候选挖掘、译后抽取（可选）、冲突处理
  agents/       分析、翻译、审校、润色、定名、一致性、提示词
  pipeline/     workflow（声明式节点/planner/runner）、断点状态、滚动上下文、确定性 lint、校验
  postprocess/  标点规范化
  assemble/     EPUB/TXT 回填导出、QA 报告
tests/          离线测试
```

## 测试

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

如果本机 `uv` 缓存目录可写，也可以直接运行：

```bash
uv run python -m unittest discover -s tests
```
## 本地基准语料库

基准语料库工具完全离线运行，既不调用模型，也不访问网络。`BOOK_SPEC.yaml` 中的源书籍路径按文件
所在目录解析；请仅处理已获授权的书籍，并注意扫描目录会包含原文片段。

```bash
trans-novel tools benchmark corpus scan BOOK_SPEC.yaml --out benchmark_data/inventory
trans-novel tools benchmark corpus build BOOK_SPEC.yaml SELECTION.yaml --out benchmark_runs/corpus
trans-novel tools benchmark corpus validate benchmark_runs/corpus
```

指定的输出目录必须尚未创建，工具不会覆盖或删除已有目录。`challenge_keys.jsonl` 含有人审答案和理由，
仅供评估器使用，不得作为运行器的输入。`benchmark_data/` 和 `benchmark_runs/` 已被 Git
忽略；不要提交受版权保护的书籍、扫描结果或冻结语料。

Layer one consumes a frozen `PREPARATION.json` and strict `CANDIDATES.yaml`; it performs
only translation and optional production-gated polish. The same primary translation
artifact is shared by all editor branches and unpolished controls. Context passages
produce C0/C1/C2 diagnostics; screen and other non-context passages use C2 preparation
context. Resume is allowed only with matching immutable hashes:

Phase six freezes real production preparation once, then reuses it for the full quality
DAG on formal continuous passages. Preparation is an internal source protocol, not a
provider or model transport; controls are skipped by the full layer. Raw translation
stores are closed before cloning editor/fast branches, and branch telemetry must contain
no new translation calls:

```bash
trans-novel tools benchmark prepare freeze CORPUS_DIR BOOK_SPEC.yaml PREPARATION_SPEC.yaml --out benchmark_runs/preparation
trans-novel tools benchmark prepare validate benchmark_runs/preparation
trans-novel tools benchmark run full CORPUS_DIR CANDIDATES.yaml benchmark_runs/preparation --out benchmark_runs/full
```

All preparation and full-run directories are create-or-resume only with exact immutable hashes. They contain local artifacts and source identities only; never commit source books, prompts, provider credentials, or generated translations.

## 离线人工评估包

Phase seven generates deterministic, model-blind packs from a validated corpus and
completed attribution/full run. The evaluator receives only the local files under
`raters/<opaque-id>/`; the secret mapping stays at the pack root with restrictive
permissions and is never copied into a rater directory.

```bash
trans-novel tools benchmark evaluate pack CORPUS_DIR RUN_DIR EVALUATION_SPEC.yaml --out PACK_DIR
trans-novel tools benchmark evaluate validate PACK_DIR
trans-novel tools benchmark evaluate import PACK_DIR RESPONSES_DIR --out EVALUATION_DIR
```

Pack generation, validation, and import are offline and create-only. The static UI
loads no external resource and stores autosaves in local storage scoped by pack and
rater. It displays eligibility, consent, compensation, and retention notices and
requires explicit consent before work; no name, email, IP, or device identifier is
collected. Imports are strict and incremental: each rater file is hash-bound,
revalidated, and atomically incorporated into canonical JSONL outputs. A completion
manifest is emitted only after every expected rater has been accepted.
Rater assignment/response envelopes bind to `pack_semantic_sha256`, which covers the
specification, source/run hashes, task counts, and secret assignment semantics.
`pack_sha256` is a separate final manifest hash covering the immutable manifest,
secret mapping hash, and every rater asset byte hash; it is never embedded in the
assets, avoiding a self-referential hash cycle.


## Phase 8 报告与发布门禁

报告构建过程完全离线，只读取冻结后的语料库、运行结果、准备数据、评估包、评估导入结果和人民币价格快照，既不调用模型，也不访问网络。工具不会覆盖不匹配的已有输出目录；已有结果的完整哈希一致时，重复调用不会重新构建。

```bash
trans-novel tools benchmark report build CORPUS_DIR RUN_DIR PREP_DIR PACK_DIR EVALUATION_DIR PRICE.yaml REPORT_SPEC.yaml --out REPORT_DIR
trans-novel tools benchmark report build CORPUS_DIR RUN_DIR PREP_DIR PACK_DIR EVALUATION_DIR PRICE.yaml REPORT_SPEC.yaml --out REPORT_DIR --integration INTEGRATION.json
trans-novel tools benchmark report reprice REPORT_DIR NEW_PRICE.yaml --out NEW_REPORT_DIR
trans-novel tools benchmark report validate REPORT_DIR
```

`INTEGRATION.json` 必须与同目录的 `integration_complete.json` 配对；报告会校验候选
结果文件路径和原始字节哈希，并只把 Phase 9 选中的非对照候选纳入最终门禁。
`candidates.csv` 的 `effective_cost_rate_*` 列根据 `REPORT_SPEC` 中设定的每位编辑的时薪生成，结果确定且可复现。

缺少集成完成文件时，报告保持 `provisional` 状态，只发布质量表和诊断性 `Pareto` 前沿，不选出获胜者。所有待裁决项和未达标的校准 `alpha` 指标，都会明确列入 `gates` 和 `withheld_reasons`。提供有效的终态集成结果后，报告才可能进入 `final` 状态；每个候选仍须通过人工、系统、统计和逐书门禁。计费用量未知不会降低质量分数，但相关候选会被排除在成本/有效成本 `Pareto` 前沿及推荐结果之外，避免把成本下界误当作实际成本。`reprice` 只读取报告中冻结的人工评估数据、系统评估数据和规范化定价数据，因此即使删除原始源文件、运行目录和准备目录仍可执行；质量相关 `CSV` 的字节保持不变。`report.html` 自包含、无外链且不嵌入原文段落。

## Phase 9 隐藏 EPUB 中断续跑集成

Phase 9 只处理一本已获许可且 `split=hidden` 的 EPUB，并在通过全部校验后才调用模型。每个候选的状态、输出、遥测和金丝雀测试数据相互独立，不共享翻译或准备状态，也不设后备路径。只允许在正文翻译批次完整持久化后中断；请勿强制终止进程，也不要在写入过程中人为触发崩溃。

```bash
trans-novel tools benchmark integration run CORPUS_DIR BOOK_SPEC.yaml CANDIDATES.yaml \
  INTEGRATION_SPEC.yaml --out benchmark_runs/integration
```

命令采用 `create/resume` 语义：输入与已完成记录完全一致时，命令会先校验哈希，随后不再执行其他操作；任务尚未完成时，工具会新建 `Application`，复用 `RunStore` 状态并只翻译剩余批次。如果请求与状态不一致、数据损坏或隐藏源文件哈希不匹配，命令会拒绝继续。实际运行会消耗 `Provider` 配额。

每个候选都必须通过基于合成输入的三角色金丝雀测试、生产质量流水线的就绪门禁，以及源 `EPUB`、单语 `EPUB` 和双语 `EPUB` 的结构校验。
输出包括 `integration_request.json`、
`integration_state.json`、`candidates/<id>/result.json`、`integration.json` 和
`integration_complete.json`；最终清单只在所有候选进入终态后写入，并记录单语与双语
输出路径及其原始字节哈希。
## 提交与发版

启用提交钩子：

```bash
uv run pre-commit install
```

提交涉及代码、测试、构建脚本、GitHub Actions 工作流或依赖的变更时，必须在同一次提交中更新
`CHANGELOG.md` 的 `[Unreleased]` 小节。否则，提交钩子会拒绝该提交，Pull Request 的 CI
也会执行相同检查。

发布 `X.Y.Z` 版本时：

1. 把 `pyproject.toml` 中的 `project.version` 修改为 `X.Y.Z`。
2. 在 `CHANGELOG.md` 中保留新的 `[Unreleased]` 小节，并把本次发布内容移入
   `## [X.Y.Z] - YYYY-MM-DD` 小节。
3. 提交对版本号和 `CHANGELOG.md` 的修改。
4. 创建并推送与版本号完全匹配的标签：`git tag vX.Y.Z && git push origin vX.Y.Z`。

标签推送会触发 GitHub Actions。只有当标签、`project.version` 与 `CHANGELOG.md` 中的版本号
完全一致时，工作流才会创建 GitHub Release，并以对应版本小节作为发布说明；随后附上各平台的
可执行文件和 `SHA256SUMS.txt`。

## 憧憬与不足

本项目为作者个人兴趣所开发，仅在于针对长文本书籍的译介做出一份微薄的努力，未来想让翻译在够准确的前提下更加顺畅，努力从可读向好读迈进。现阶段翻译文本一些口头禅前后翻译不一致，专有名词翻译不准确的问题，已经改进！如果还有什么问题，可以提交issue，如果你有什么想法，欢迎在讨论区提出，如果你有一定的编程能力，欢迎给我提交PR，让这个项目变得更好。👏

## 星标历史

<a href="https://www.star-history.com/?repos=BigDawnGhost%2FWenyi&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=BigDawnGhost/Wenyi&type=date&theme=dark&legend=top-left&sealed_token=VFuKZdjDh-9e2mG4qlvqeSpCkWCoRf9ZRy0hIDLdaECFQeoNNlQ20QxSD4PuvTZp1RJg7J2s5hr57Eq66paMrhikuuI3kc41uZZCYb-bTqsUafeSB7AVdhw7bmz70NhkVXABHtSIHdw0DROZaInmznYJ651gP2klEeW8OOM8EkfJnXgDld6f0xn8mIJ9" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=BigDawnGhost/Wenyi&type=date&legend=top-left&sealed_token=VFuKZdjDh-9e2mG4qlvqeSpCkWCoRf9ZRy0hIDLdaECFQeoNNlQ20QxSD4PuvTZp1RJg7J2s5hr57Eq66paMrhikuuI3kc41uZZCYb-bTqsUafeSB7AVdhw7bmz70NhkVXABHtSIHdw0DROZaInmznYJ651gP2klEeW8OOM8EkfJnXgDld6f0xn8mIJ9" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=BigDawnGhost/Wenyi&type=date&legend=top-left&sealed_token=VFuKZdjDh-9e2mG4qlvqeSpCkWCoRf9ZRy0hIDLdaECFQeoNNlQ20QxSD4PuvTZp1RJg7J2s5hr57Eq66paMrhikuuI3kc41uZZCYb-bTqsUafeSB7AVdhw7bmz70NhkVXABHtSIHdw0DROZaInmznYJ651gP2klEeW8OOM8EkfJnXgDld6f0xn8mIJ9" />
 </picture>
</a>
