# 模型 Benchmark 操作指南

本指南覆盖生产等价的章节 EPUB benchmark、自动分片评审和确定性报告。只处理已获授权的书籍；源书、运行状态、模型输出、评审 shard 和报告均是本地数据，不得提交。

## 核心不变量

- benchmark 直接调用生产 `Application.run_all()`，输入是 `BOOK_SPEC.yaml` 指向的原始章节 EPUB。
- 每个候选、章节和 replicate 使用独立状态目录；polish arm 只从同模型对的已关闭 minimal arm 克隆初译状态。
- 不存在 benchmark 专用翻译 prompt、共享 preparation 或独立重译的 polish control。
- minimal 使用 `balanced`（不润色），polish 使用 `quality`（仅新增润色及必要收尾）。
- 每个候选必须配置 `pipeline_variant: minimal|polish`、`primary_model`、`editor_model` 和 thinking 级别。
- canary 使用一个真实 `screen` 章节；full 只使用全部 6 个 `formal` 章节。
- 自动评审只接受带原文和译文真实子串证据的严格 JSON，不接收人工表单或后编辑计时。

## 文件布局

```text
BOOK_SPEC.yaml
CANDIDATES.yaml
REVIEW_SPEC.yaml
PRICE.yaml
benchmark_data/
  sources/
    screen-01.epub
    ...
    formal-01.epub
    ...
    formal-06.epub
    hidden-01.epub
benchmark_runs/
  canary/
  full/
  review/
  review-results/
  report/
```

所有 `benchmark_runs/` 输出都应使用新目录。不要把旧架构生成的 preparation、attribution、evaluation pack 或 report 目录与新运行混用。

## 1. 准备书籍规范

`BOOK_SPEC.yaml` 的路径相对于规范文件解析。语料约束是至少 3 个 `screen`、至少 6 个 `formal` 和至少 1 个 `hidden`；full runner 进一步要求 formal 恰好为 6 个。正式源文件应是已经切出的章节 EPUB，而不是整本原书。

```yaml
schema_version: 1
source_language: en
target_language: zh
books:
  - book_id: screen-01
    path: benchmark_data/sources/screen-01.epub
    split: screen
  - book_id: screen-02
    path: benchmark_data/sources/screen-02.epub
    split: screen
  - book_id: screen-03
    path: benchmark_data/sources/screen-03.epub
    split: screen
  - book_id: formal-01
    path: benchmark_data/sources/formal-01.epub
    split: formal
  # formal-02 ... formal-06
  - book_id: hidden-01
    path: benchmark_data/sources/hidden-01.epub
    split: hidden
```

同一物理路径或相同源文件字节不能重复出现在规范中。

## 2. 配置候选

`CANDIDATES.yaml` 决定实际 Provider、模型和 minimal/polish 变体，不继承本地 `config.yaml`：

```yaml
schema_version: 2
benchmark_id: wenyi-benchmark
provider: opencode-go
fast_model: mimo-v2.5:off
temperature: 0.1
seed: null
replicates: 1
candidates:
  - candidate_id: deepseek-v4-flash-minimal
    primary_model: deepseek-v4-flash:off
    editor_model: deepseek-v4-flash:off
    pipeline_variant: minimal
  - candidate_id: deepseek-v4-flash-polish
    primary_model: deepseek-v4-flash:off
    editor_model: deepseek-v4-flash:off
    pipeline_variant: polish
```

运行前在当前 shell 设置凭据，并在 OpenCode Go workspace 中为
`muse-spark-1.2-contributor` 显式启用数据贡献：

```bash
export OPENCODE_API_KEY='<secret>'
```

不要把密钥写进 YAML、状态目录或命令历史。

## 3. 运行 canary

canary 从按 `book_id` 排序后的第一个 `screen` 章节中选择一个；也可显式指定：

```bash
uv run trans-novel tools benchmark run canary \
  BOOK_SPEC.yaml CANDIDATES.yaml \
  --book-id screen-01 \
  --out benchmark_runs/canary
```

成功条件：候选都走完整生产质量流水线，生成 EPUB、`segments.jsonl` 和物理调用遥测，并写入完成状态。canary 不是缩短 prompt 的模拟请求。

## 4. 运行全部 6 个 formal 章节

```bash
uv run trans-novel tools benchmark run full \
  BOOK_SPEC.yaml CANDIDATES.yaml \
  --out benchmark_runs/full
```

输出结构：

```text
full/
  run.json
  run_state.json
  candidates.json
  candidates/<candidate>/<book>/r1/
    state/<book-slug>/
    outputs/<book-id>.epub
    segments.jsonl
    telemetry.jsonl
```

`run.json` 冻结书籍规范哈希、候选规范哈希、生成参数、质量档位、书籍集合和 replicate 数。输入身份不一致时拒绝续跑；已完成分支必须通过状态、输出、segment 和 telemetry 哈希校验。

## 5. 准备自动评审

先计算 `run.json` 的原始字节 SHA-256，并创建 `REVIEW_SPEC.yaml`：

```bash
shasum -a 256 benchmark_runs/full/run.json
```

```yaml
schema_version: 1
benchmark_id: wenyi-benchmark
run_sha256: '<64-hex-sha256>'
seed: 17
segments_per_book: 24
shard_count: 8
```

然后生成盲化 shard：

```bash
uv run trans-novel tools benchmark evaluate prepare \
  benchmark_runs/full REVIEW_SPEC.yaml \
  --out benchmark_runs/review
```

抽样对每本 formal 章节独立执行，优先覆盖：

- lint 或确定性 QA 发现过问题的风险段；
- 对话；
- 专名和术语；
- 长句；
- 普通叙事补足样本。

同一 seed、run 和规范必定产生相同 unit、A/B 朝向和 shard。`secret_mapping.json` 只由主流程在汇总时读取，reviewer 不应读取它。

## 6. 并行执行 reviewer shard

每个 reviewer 只接收：

- `review/prompt.json`；
- 一个 `review/shards/shard-NNN.json`；
- 该 shard 的输出 JSON Schema。

不同 reviewer 的 unit 集合互不重叠。结果保存为：

```text
benchmark_runs/review-results/shard-001.json
benchmark_runs/review-results/shard-002.json
...
```

每个结果必须满足以下规则：

- `review_sha256` 和 `shard_id` 与输入完全一致；
- 对 shard 中每个 unit 恰好返回一个 review，不得缺失、重复或越界；
- A/B 获胜时必须至少有一条针对败方的 finding；
- `source_quote` 必须是该 unit 原文的非空真实子串；
- `target_quote` 必须是 finding 指定 side 译文的非空真实子串；
- error type 和 severity 只能使用 prompt 声明的枚举。

## 7. 汇总自动评审

```bash
uv run trans-novel tools benchmark evaluate finalize \
  benchmark_runs/review benchmark_runs/review-results
uv run trans-novel tools benchmark evaluate validate benchmark_runs/review
```

汇总器校验所有输入后解除 A/B 盲化，输出：

- `findings.jsonl`：逐条、可追溯到 unit/book/segment/candidate 的证据；
- `comparison.json`：严重度、错误类型、每万原文词加权错误、胜负和逐书结果；
- `summary.json`：候选排序、review 数和 finding 数；
- `review_complete.json`：reviewer 结果和衍生文件哈希。

排序依次最小化 critical、major、加权错误，再最大化 wins。指标完全相同时 `winner` 为 `null`，不会按候选名虚构获胜者。

## 8. 冻结价格并生成报告

`PRICE.yaml` 使用 `opencode-go:<resolved-model>` 作为模型键。仓库中的快照来自
OpenCode Go 官方价格表，币种为美元、单价为每百万 token。DeepSeek V4 Flash
按官方 UTC 峰谷时段逐请求计价；其他两个模型使用全天价格：

```yaml
schema_version: 1
provider: opencode-go
region: global
currency: USD
retrieved_at: '<UTC timestamp>'
source_urls:
  - https://opencode.ai/docs/go/
models:
  opencode-go:deepseek-v4-flash:
    model_id: opencode-go:deepseek-v4-flash
    rules:
      - min_prompt_tokens: 0
        max_prompt_tokens: null
        time_band: off_peak
        input_uncached_per_million: '0.22'
        input_cached_per_million: '0.007'
        output_per_million: '0.66'
      - min_prompt_tokens: 0
        max_prompt_tokens: null
        time_band: peak
        input_uncached_per_million: '0.44'
        input_cached_per_million: '0.014'
        output_per_million: '1.32'
  # muse-spark-1.2-contributor and mimo-v2.5 follow the committed PRICE.yaml
```

```bash
uv run trans-novel tools benchmark report build \
  benchmark_runs/full benchmark_runs/review PRICE.yaml \
  --out benchmark_runs/report
uv run trans-novel tools benchmark report validate benchmark_runs/report
```

报告生成：

- `summary.json`：状态、获胜者、排序和输入身份；
- `comparison.json`、`findings.jsonl`：原样保留自动评审事实；
- `costs.json`：规范化 usage、逐模型报价和候选 API 总成本；
- `system.json`：分支、书籍、输出、失败请求和思考 token 状态；
- `report.html`：无外链的简明比较页；
- `report.json`：全部报告文件的字节哈希。

出现失败请求、缺失输出或未定价模型时，报告为 `provisional` 且不宣布 winner。原生 reasoning token 会进入系统与成本事实，但不会单独阻止质量比较。

## 9. 可选隐藏 EPUB 集成

隐藏 EPUB 的中断续跑和结构校验仍使用独立命令：

```bash
uv run trans-novel tools benchmark integration run \
  CORPUS_DIR BOOK_SPEC.yaml CANDIDATES.yaml INTEGRATION_SPEC.yaml \
  --out benchmark_runs/integration
```

它不替代 formal 章节 benchmark，也不参与自动评审抽样。需要把终态集成事实复制进报告时，可向 `report build` 传入 `--integration INTEGRATION.json`。

## 故障处理

- `immutable benchmark run identity mismatch`：输出目录属于不同输入；使用新的空目录。
- `full benchmark requires exactly six formal chapter EPUBs`：修正 `BOOK_SPEC.yaml`，不要传整本原书或额外 formal 条目。
- `candidate segment sets do not match`：某个候选分支缺段或使用了不同源文件；不要手工补齐，重跑损坏分支。
- `finding ... quote is not present`：reviewer 证据不是原文/对应译文的逐字子串；修正该 shard JSON。
- `report ... provisional`：检查 `system.json` 和 `costs.json`，定位失败调用、缺失输出或未定价模型。
