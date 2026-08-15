# Changelog

本项目的所有重要变更都记录在此文件中，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [Unreleased]

### Fixed

- 模型语言识别请求失败时不再吞掉原始异常，缺少 API key 等配置错误会直接报告。
- 正文翻译仅在模型输出违反协议时整批重试并逐段兜底；Provider 异常和业务异常不再触发上层重复调用，也不会被错误包装。

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
