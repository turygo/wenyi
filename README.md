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
export DEEPSEEK_API_KEY=sk-...
./wenyi translate book.epub
```

在 Windows PowerShell 中，先运行 `$env:DEEPSEEK_API_KEY = "sk-..."` 设置环境变量，
再运行 `.\wenyi.exe translate .\book.epub`。程序首次运行时会在当前目录自动生成可修改的
`config.yaml`；`wenyi init --force` 可以将它重置为随当前版本发布的默认配置。

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
trans-novel translate book.epub --polish
trans-novel translate book.epub --no-polish
trans-novel translate book.epub --qa
trans-novel translate book.epub --no-qa
```

`--polish/--no-polish` 会覆盖 `config.yaml` 里的 `pipeline.polish`。随程序发布的默认配置写的是 `polish: true`，所以不加参数时默认会润色；代码层面的缺省值是 `false`，只在配置文件没写该字段时生效。

润色会让每个翻译批次增加一次经 `editor` Agent 路由发起的模型调用；默认配置中，
`editor` 与 `translator` 绑定同一模型（`deepseek:pro`）。质量可能更稳，但会明显增加
耗时和成本。已经翻译完成的批次会被断点续跑跳过，后来再开关润色不会自动重跑旧译文。

## 配置

主要配置都在 `config.yaml`：

- `language.source`: `auto` 由模型识别源语言，也可以写死语言代码，如 `ja`、`en`、`ko`、`ru`、`de` 等。
- `llm.providers`: Provider 别名 → 端点 / 凭据环境变量 / 超时 / 重试 / 命名模型目录。Provider 类型：`deepseek`、`openai`、`openrouter`、`openai-compatible`（或 `custom`）、`ollama`、`vllm`、`fake`。
- `llm.agents`: 六个功能 Agent 路由（`translator` / `editor` / `reviewer` / `analyst` / `preparer` / `light-translator`），每个 Agent → `model`（primary）+ `fallback`（有序降级列表）。模型引用形如 `<provider-alias>:<model-alias>`，按**第一个冒号**切分，所以 `local:qwen3:32b` 解析为 provider=`local`、model=`qwen3:32b`。配置必须恰好声明这六个 Agent，缺失或未知键会在加载时直接报错。
- `pipeline.review`: 章末审校。
- `pipeline.autofix_severe`: 对严重问题自动重译并采纳通过校验的结果。
- `pipeline.polish`: 翻译后再做中文润色。
- `pipeline.backtranslate_sample`: 回译抽检比例，`0` 为关闭。
- `pipeline.consistency_qa`: 全书跨章一致性扫描。
- `pipeline.book_understanding`: 翻译前预扫整本书，生成全书概览和逐章梗概。
- `pipeline.rolling_context_segments`: 每批翻译时带入的前文译文段数。
- `segment.max_chars_per_batch`: 每个翻译批次的大小。
- `segment.max_chars_per_segment`: 超长段落的拆分阈值。

离线测试或调试流程时，可以把 provider 类型改成 `fake`，不会发网络请求。

### Provider → 模型目录

每个 Provider 别名代表一个端点/账号，`models` 里是它提供的命名模型；目录键是
用户别名，`id` 是发给服务的精确模型 ID。同一个服务 ID 需要不同推理设置时必须
拆成多个模型别名：

```yaml
llm:
  providers:
    deepseek:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key_env: DEEPSEEK_API_KEY
      timeout: 600
      max_retries: 4
      models:
        pro:
          id: deepseek-v4-pro
          reasoning: {enabled: true, effort: high}
        flash-thinking:
          id: deepseek-v4-flash
          reasoning: {enabled: true, effort: high}
        flash-fast:
          id: deepseek-v4-flash
          reasoning: {enabled: false}
```

`reasoning.enabled=true` 会按不同 Provider 的请求格式启用推理（DeepSeek/OpenAI 使用
`reasoning_effort`，OpenRouter 使用 `extra_body.reasoning.effort`），并把请求的正数
`max_tokens` 抬到至少 4096；`enabled=false` 则关闭推理、不抬上限。`request_overrides`
可递归覆盖请求字段（`model`/`messages`/`stream` 由系统生成，不可覆盖）。

### Agent → primary / fallback 路由

路由键是六个功能 Agent 之一：每个 Agent 先调用 `model`。只有发生可重试、可切换
候选的错误（空响应、限流、超时、连接错误、5xx 等），且当前候选的
`max_retries + 1` 次实际请求均失败后，才会尝试 `fallback` 中的下一个候选；
配置错误、凭据缺失、鉴权失败、请求被拒等不可重试错误会立即报错，不会切换候选。
所有候选均失败后抛出 `AllModelsFailedError`（只包含 `<provider>:<model>` 与
归一化原因）。内部 operation（业务标签，如 `translate.batch`）不参与路由，只做
用量/调试归因。

```yaml
llm:
  agents:
    translator:
      model: deepseek:pro
      fallback: []
    reviewer:
      model: deepseek:flash-thinking
      fallback: [deepseek:pro]   # 判断类任务的主模型失败时改用 pro
```

### 自定义 provider 示例

Bailian（阿里云百炼）使用 OpenAI 兼容端点；OpenCode-Go 使用其兼容端点
（`https://opencode-go.example.com/v1` 仅为占位示例 URL，请替换为你的实际网关地址）：

```yaml
llm:
  providers:
    bailian:
      type: openai-compatible
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key_env: DASHSCOPE_API_KEY
      models:
        qwen-max:
          id: qwen-max
          reasoning: {enabled: false}
    opencode-go:
      type: openai-compatible
      base_url: https://opencode-go.example.com/v1
      api_key_env: OPENCODE_GO_API_KEY
      models:
        "qwen3:32b":
          id: qwen3-32b
          reasoning: {enabled: false}
  agents:
    translator:
      model: bailian:qwen-max
      fallback: [opencode-go:qwen3:32b]
```

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

- **术语库**：翻译前从源文挖掘专名候选（英文走确定性统计，其他语言走 LLM 挖掘），由定名 Agent 一次性统一定名后写入 SQLite 术语库，翻译期只读、按配置注入提示词；人物条目锁定后由 lint 硬校验。日文轻小说等需要译后确认称呼变体的场景可开 `pipeline.inflight_glossary` 保留旧的译后抽取。
- **全书理解**：翻译前预扫源文，生成全书概览和章节梗概，让早期章节也能参考全书走向。
- **滚动上下文**：章内批次串行处理，后一个批次能看到前面最近几段译文。
- **段数对齐**：每批输入 N 段，要求模型输出 N 段 JSON；段数不符会重试，仍失败则逐段兜底。
- **确定性 lint**：零成本机器校验直接引语引号保留、数字一致、锁定专名命中、整段未译；翻译后与润色后各跑一遍，命中即定向重译或回退，其余记录进报告。
- **章末 review**：按章检查漏译、误译、专名、人称等语义问题（机械问题已由 lint 兜住）；是否自动重译严重项由 `autofix_severe` 控制。
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

默认配置使用 DeepSeek，并通过 OpenAI SDK 调用 `https://api.deepseek.com`。其他 provider 也使用兼容
Chat Completions 的接口：OpenAI 与 OpenRouter 使用各自默认的 API URL 和环境变量；Ollama 与 vLLM
使用本地默认 URL；`openai-compatible`/`custom` 必须设置 `llm.providers.<别名>.base_url`。

每个 Agent 的模型由 `llm.agents` 路由决定（六键之一，operation 不参与路由）：

- `translator` → `deepseek:pro`（开思考）：正文翻译、定向重译、标题翻译。
- `editor` → `deepseek:pro`（开思考）：润色、去翻译腔改写。
- `reviewer` → `deepseek:flash-thinking`（开思考）：章末 review、一致性 QA、回译比对、去翻译腔检测、语言识别。
- `analyst` → `deepseek:pro`（开思考）：全局分析、一次性定名、术语审计。
- `preparer` → `deepseek:flash-fast`（免思考）：全书预扫、章节梗概、术语候选挖掘/抽取等机械任务（英文候选挖掘与 lint 为纯本地计算，零 token）。
- `light-translator` → `deepseek:flash-fast`（免思考）：回译、附属章粗翻。

改模型只动 `config.yaml` 的 `llm.providers.<别名>.models`（改 `id` 或推理设置）与
`llm.agents`（改路由）；provider 专属请求字段写在模型条目的 `request_overrides`。

## 项目结构

```text
trans_novel/
  config.example.yaml  默认配置的唯一来源，首次运行时复制到工作目录
  ingest/       输入解析、EPUB/FB2/TXT 切分
  llm/          LLM 抽象接口、provider factory、内置 providers、FakeClient
  glossary/     SQLite 术语库、源文候选挖掘、译后抽取（可选）、冲突处理
  agents/       分析、翻译、审校、润色、定名、一致性、提示词
  pipeline/     编排器、断点状态、滚动上下文、确定性 lint、校验
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
