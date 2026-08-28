# Changelog

本项目的所有重要变更都记录在此文件中，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Fixed

- Reject review autofixes that introduce deterministic lint regressions or remove preserved dialogue quotes.

## [0.1.4] - 2026-08-28
### Added

- 新增公开 `editor` 模型角色，支持为润色和自然化改写配置独立模型；省略时继承 `primary`。
- 新增经官方目录核验的百炼候选模型能力元数据，并支持通过构造器注入温度、思考开关等受控生成选项。
- 节点输入指纹按实际模型角色组合计算，模型变更只使受影响的流水线阶段失效。
- 新增可选的遥测功能，记录每次实际的 `LLM` 调用，可写入 `JSONL` 或接入收集器；同时新增不可变的价格快照和基于 `Decimal` 的成本报价基础设施。
- 新增完全离线的基准语料库扫描、人工筛选、冻结和校验命令；所有产物均包含稳定哈希，并检查跨书配额和数据泄漏。
- 新增生产等价的章节 `EPUB` benchmark：金丝雀和正式运行都直接调用 `Application.run_all()`，每个候选和章节隔离状态、术语库、翻译和润色结果，正式运行固定覆盖全部六个 `formal` 章节。
- 新增确定性自动评审：按风险、对话、术语、长句和叙事抽样，枚举二至六个候选的全部候选对并分别盲化为 A/B，再拆分成互不重叠的 reviewer shard；汇总时严格校验完整单元集合和原文/译文逐字证据。
- 新增自动评审报告：输出候选严重度、错误类型、每万词加权错误、逐书胜负、证据明细、生产系统状态和基于冻结价格快照的 API 成本。
- 新增第九阶段的隐藏 `EPUB` 集成测试框架：严格校验输入和哈希，执行三角色金丝雀测试；支持在批次完整提交后安全中断，续跑时为每个候选分别新建 `Application`，确保彼此独立；同时提供单语、双语结构门禁、资源链接校验，以及与第八阶段兼容的清单。
- EPUB source runs now persist schema-3 lxml text-slot contracts and write translations back to reopened source XHTML while preserving inline structure, resources, and vertical layout.
- EPUB 导出统一经过独立磁盘重开验证、确定性 `epub_verification.json` 报告和原子发布门禁；失败时保留既有文件并记录稳定事件。
- Schema-3 bilingual EPUB 回填改为单次 lxml 资源渲染：共享严格源文清洗、容器/直 `<br>` 配对与双语样式契约，支持 `target_first` 和 `source_first` 并在发布前拒绝映射与保留标记冲突。

### Changed

- `balanced` 与 `quality` 质量档位现在始终执行润色，`economy` 仍保持不润色。
- 基准候选必须显式配置 `editor_model`；移除 attribution、共享 preparation、未润色 control、人工评估包、后编辑计时、人力成本和重新定价路径。
- 正式 benchmark Provider 从百炼切换到 OpenCode Go，比较 `deepseek-v4-flash`、`muse-spark-1.2-contributor` 和 `mimo-v2.5` 三个各自完成初译与润色的候选，并按 OpenCode Go 官方价格逐请求计费。
- OpenAI Responses API 模型现在复用统一的请求控制、响应解析和 token 用量归一化。

- Removed the legacy template-based source EPUB fallback; source EPUB assembly now accepts only schema 3 slot state.

### Fixed

- OpenAI SDK 客户端禁用内置重试，避免与集中式 Provider 重试叠加；缺失 `choices` 的畸形聊天响应按空响应重试；严格润色遇到段数协议错误时先携带精确数量约束重试，仍失败则逐段恢复；正式 benchmark 在发出请求前拒绝任何包含多个逻辑章节的 formal EPUB。
- 严格润色在批量响应错位时保留逐段恢复结果，不再直接丢弃整批输出。
- 修复 EPUB schema 3 槽位回填、双语源文配对、跨槽位标点规范化及脚注标记识别，并补齐严格流水线的断点续跑与审校统计。

## [0.1.3] - 2026-08-15

### Fixed

- Windows 打包冒烟检查确认缺失输入会按预期报错后，现显式返回成功，以免预期的非零退出码导致发布流程被误判为失败。

## [0.1.2] - 2026-08-15

### Fixed

- 模型语言识别请求失败时不再吞掉原始异常，缺少 API key 等配置错误会直接报告。
- 正文翻译仅在模型输出违反协议时整批重试并逐段兜底；Provider 异常和业务异常不再触发上层重复调用，也不会被错误包装。
- 修复多平台可执行文件打包工作流仍假定 `translate` 会自动生成配置文件的问题；打包冒烟检查现改为显式执行 `init`，以验证内置默认配置。

### Changed

- 事件日志升级为 schema 2：例行翻译、跳过、润色、issue 和用量载荷仅保留稳定的 SHA-256 摘要与计数；改写审计事件保留 before/after，并在正文落盘后才发出；事件追加失败时只告警，不阻断流程。同时移除正文翻译、附属章、自然化和审校流程中冗余的整章写入。
- 开发检查启用额外 Ruff 质量规则，包内导入统一改为绝对导入并禁止新增相对导入。

## [0.1.1] - 2026-08-15

### Added

- 要求提交代码时同步更新 `CHANGELOG.md`，并由提交钩子和 CI 校验。
- 支持在推送版本标签后自动创建 GitHub Release，并以对应版本的变更记录作为发布说明。
- 支持阿里云百炼 Provider，以及 DeepSeek V4 Flash 和千问 3.7 Flash 的思考开关。

### Changed

- 配置文件精简为 `llm.provider`、`llm.models.primary`、`llm.models.fast` 和 `quality`；旧的 Provider 目录、Agent 路由与流水线调参格式不再兼容。
- 缺少 `config.yaml` 时直接使用 OpenCode Go、DeepSeek V4 Flash 与 `balanced` 内置默认值，不再自动创建文件。
- 新增 `--quality`、`--source-language`、`--back-matter` 和 `--honorifics` 单次运行参数。
- 默认 `primary` 与 `fast` 均改用 OpenCode Go 提供的 `deepseek-v4-flash`，降低长篇翻译成本。
- 模型规格支持以 `:<thinking-level>` 后缀选择 `off` / `low` / `medium` / `high` / `max`；程序按逐 Provider/模型能力表校验，不支持的级别启动即报错。默认 OpenCode Go 的 `deepseek-v4-flash:high` 用于 `primary`，`:off` 用于 `fast`。
