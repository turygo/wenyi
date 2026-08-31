# Python 架构与代码规模治理

本文定义 `trans-novel` 的架构治理规则。它帮助开发者和编码 Agent 在实现功能前确定职责边界，在实现过程中控制代码规模，并在合并前通过自动检查阻止架构退化。

> [!IMPORTANT]
> 本文解决新增架构债务的根因。它不规定如何重构现有大文件，也不要求在启用治理前完成存量重构。

## 状态

**提议中。** 规则接入 `AGENTS.md`、pre-commit 和 CI 后生效。

## 目标

本治理方案具有以下目标：

- 阻止新增超过规模上限的 Python 文件、类和函数。
- 让每个子系统拥有清晰、可验证的职责。
- 让 Agent 在编码前推导架构，而不是把代码追加到最近的现有文件。
- 阻止新增循环依赖、反向依赖和跨模块私有调用。
- 通过棘轮基线逐步减少存量债务，而不是用一次大重构替代治理。
- 对生产代码和测试代码应用相同的质量标准。

本治理方案不要求：

- 为普通修复编写长篇设计文档。
- 引入新的第三方质量工具。
- 创建没有当前用途的接口、工厂或抽象层。
- 立即拆分所有现有大文件。

## 为什么需要治理

现有仓库包含架构说明，但缺少可执行的边界。`AGENTS.md` 说明了目录和主要组件的职责，却没有定义允许的依赖方向、模块不负责的事项、规模预算或自动门禁。

历史记录还表明，大文件不只来自长期累加。多个文件在首次提交时就已超过合理规模：

- 提交 `e3857a1` 一次增加 10,194 行。`benchmark/runner.py` 首次提交为 3,923 行，`benchmark/corpus.py` 首次提交为 938 行。
- 提交 `474c007` 一次增加 6,649 行。`benchmark/integration.py` 首次提交为 2,175 行，两个集成测试文件首次提交分别为 1,436 行和 1,019 行。
- 提交 `853540c` 创建了 3,041 行的 `assemble/epub_verifier.py`，并向对应测试增加 663 行。

这些结果说明，治理必须在设计和合并阶段生效。事后拆分无法阻止同类问题再次出现。

## 当前基线

### 文件规模

生产代码包含 84 个 Python 文件：

| 指标 | 行数或数量 |
| --- | ---: |
| 中位数 | 155 行 |
| P75 | 346 行 |
| P90 | 694 行 |
| P95 | 920 行 |
| 超过 500 行 | 13 个文件 |
| 超过 800 行 | 7 个文件 |
| 最大文件 | 3,383 行 |

测试代码包含 35 个 Python 文件：

| 指标 | 行数或数量 |
| --- | ---: |
| 中位数 | 274 行 |
| P75 | 538 行 |
| P90 | 979 行 |
| P95 | 996 行 |
| 超过 500 行 | 10 个文件 |
| 超过 800 行 | 5 个文件 |
| 最大文件 | 1,723 行 |

### 函数和类规模

生产代码包含 932 个函数或方法：

- P90 为 44 行。
- P95 为 84 行。
- 25 个函数或方法超过 120 行。

测试代码包含 731 个函数或方法：

- P90 为 35 行。
- P95 为 48 行。
- 3 个函数或方法超过 120 行。

生产代码和测试代码共有 9 个超过 400 行的类。

### 依赖边界

按静态 import 计算，`agents`、`assemble`、`epub`、`glossary`、`ingest` 和 `pipeline` 共同形成一个包级强连通分量。

模块级至少存在以下互相依赖组：

- `trans_novel.assemble.writer` 和 `trans_novel.assemble.epub_verifier`
- `trans_novel.epub.slots` 和 `trans_novel.ingest.models`

第二组的一条边位于 `TYPE_CHECKING` 下。它不一定构成运行时循环，但仍说明类型所有权不清晰。

生产代码还包含 12 处跨模块私有符号导入。例如：

- `epub_verifier.py` 导入 `writer._epub_lang`。
- `epub_verifier.py` 导入 `writer._translated_toc_title`。
- `writer.py` 导入 `epub_reader._resource_parser`。
- `integration.py` 导入 `runner._model_client`。
- `benchmark/runner.py` 导入 `corpus._resolve_books`。

### 自动检查

当前 CI 执行 changelog 检查和离线测试，但不执行以下检查：

- `ruff check`
- `ruff format --check`
- Python 文件、类或函数规模检查
- 包和模块依赖检查
- 循环依赖检查
- 跨模块私有符号检查
- 单文件变更集中度检查

本地 pre-commit 运行 Ruff，但本地 hook 可以被跳过。CI 必须提供最终门禁。

## 治理原则

### 把行数上限用作保险丝

行数上限能阻止明显失控，但不能单独证明设计合理。把一个 1,000 行文件机械拆成两个 500 行文件，不会自动满足 KISS 或单一职责原则。

每次拆分都必须形成清晰的职责、公开契约和依赖方向。不要创建 `part1.py`、`helpers2.py` 或其他没有业务含义的容器。

### 为每项职责指定一个所有者

每项行为、状态和不变量必须有唯一所有者。所有者可以委托机制，但仍负责公开契约。

不要因为某个文件已经包含相似代码，就默认该文件应继续拥有新行为。

### 区分四种代码

实现前，先区分以下代码类型：

- **Orchestration**：决定调用顺序和生命周期。
- **Policy**：决定业务规则和取舍。
- **Mechanism**：完成解析、转换、校验或计算。
- **I/O**：访问文件、ZIP、模型服务或持久化存储。

入口和 composition root 只保留 orchestration。Policy、mechanism 和 I/O 应由各自的子系统拥有。

### 依赖公开契约

生产模块只能依赖另一个模块的公开契约。以下划线开头的符号属于实现细节，生产模块不得跨模块导入。

如果两个模块需要共享内部函数，请先确定该函数的真实所有者。把它提升为所有者的公开契约，或移动到双方都可以依赖的中立模块。不要复制实现，也不要通过局部 import 隐藏循环依赖。

### 对测试应用相同规则

测试文件不是规模和职责规则的例外。按行为契约组织测试，不要按阶段或里程碑把不相关测试持续追加到同一个文件。

测试可以访问被测模块的私有符号进行必要的白盒验证，但生产模块不能使用这项例外。

## 子系统所有权

每个子系统必须在 `AGENTS.md` 中声明以下内容：

- **Owns**：该子系统负责的行为、状态和不变量。
- **Does not own**：该子系统明确不负责的内容。
- **Public contract**：其他子系统可以使用的入口。
- **Allowed dependencies**：该子系统可以依赖的方向。

以下表格定义初始目标。存量违反项进入基线，但不得新增。

| 子系统 | Owns | Does not own | Public contract | 依赖规则 |
| --- | --- | --- | --- | --- |
| `cli` | 参数解析、用户交互和命令分派 | 翻译策略、状态恢复、EPUB 细节 | CLI 命令 | 依赖 `config`、`pipeline.bootstrap` 和 benchmark 命令入口 |
| `pipeline` | 工作流编排、状态转换、恢复、并发提交顺序 | EPUB/XML 解析、Provider 传输细节 | Application、Planner、Runner 和 node contracts | 依赖各子系统的公开 API |
| `ingest` | 输入解析、源结构提取和逻辑章节生成 | 输出渲染、运行状态和 LLM 调用 | `read_*` 和 `Document` 契约 | 依赖数据模型和 EPUB primitives，不依赖具体 Runner |
| `assemble` | 输出渲染、序列化、验证和发布 | 工作流调度、重试策略和状态迁移 | `assemble()` 和 publication API | 依赖数据模型和 EPUB primitives，不依赖具体 Runner |
| `epub` | ingest 和 assemble 共享的 EPUB 值对象与确定性机制 | workflow 和持久化 | EPUB primitives | 不反向依赖 ingest 或 assemble 的实现 |
| `agents` | Prompt 驱动的翻译、分析、润色和定名行为 | Provider 路由、状态存储和文件输出 | Agent contracts | 依赖 `llm` 和明确的数据契约 |
| `glossary` | 术语提取、解析和存储规则 | Agent 生命周期和 pipeline 调度 | Glossary contracts | 不依赖 Agent 的具体实现 |
| `llm` | Provider 构造、路由、重试、用量和 telemetry | 翻译业务策略 | Client、router 和 transport contracts | 依赖 `config` 和 model profiles |
| `benchmark` | 离线评估、候选隔离和评估产物 | 生产翻译实现 | 生产 Application facade | 可以依赖生产公开 API；生产代码不得依赖 benchmark |

## 依赖规则

新代码必须满足以下规则：

1. `cli` 通过公开入口调用应用和 benchmark。
2. `pipeline` 编排其他子系统，但不复制子系统机制。
3. ingest、assemble、agents、glossary 和 llm 不依赖具体的 Runner、RunStore 或 node 实现。
4. benchmark 依赖生产公开 facade。生产包不依赖 benchmark。
5. ingest 和 assemble 通过中立 EPUB contract 或 primitives 共享机制。
6. 生产模块不跨模块导入私有符号。
7. 新依赖不得创建模块循环或包级反向依赖。
8. composition root 负责构造和注入，不负责业务决策。

## 代码规模预算

生产代码、测试代码和脚本使用同一硬上限。

| 对象 | 预警线 | 硬上限 | 要求 |
| --- | ---: | ---: | --- |
| Python 文件 | 500 行 | 800 行 | 超过预警线时先说明职责分解；超过硬上限时 CI 失败 |
| 函数或方法 | 80 行 | 120 行 | 超过预警线时检查职责；超过硬上限时 CI 失败 |
| 类 | 250 行 | 400 行 | 超过预警线时检查状态和行为是否可分；超过硬上限时 CI 失败 |
| 单文件单次净新增 | 200 行 | 300 行 | 超过硬上限时必须提供 Architecture Delta |
| 新模块循环 | 0 | 0 | CI 失败 |
| 新生产跨模块私有导入 | 0 | 0 | CI 失败 |

行数按 Ruff 格式化后的物理行计算，包括注释和空行。这样可以保持规则简单、稳定且可复现。

不要通过压缩代码、删除必要注释或合并语句规避限制。Ruff 格式化和评审共同防止这类规避。

## Agent 工作流

### 开始编码前

Agent 必须完成以下步骤：

1. 找出需求对应的 capability。
2. 确定唯一 owner。
3. 列出必须保持的不变量。
4. 区分 orchestration、policy、mechanism 和 I/O。
5. 确认可以复用的公开契约。
6. 列出新增依赖边。
7. 检查目标文件、类和函数的当前规模。
8. 预测每个文件将增加的职责和行数。

### 使用 Architecture Delta

以下情况必须先提供 Architecture Delta：

- 新增功能或 CLI 行为。
- 修改持久化状态、恢复语义、并发或发布流程。
- 修改公开 contract。
- 涉及三个以上生产包。
- 新增依赖方向。
- 预计单文件增加超过 300 行。
- 目标文件已经超过 500 行。
- 需要跨模块使用私有符号。

普通的局部修复不需要 Architecture Delta，除非它触发以上条件。

使用以下模板：

```text
Capability:
Owner:
Invariants:
State and I/O:
Existing public contract reused:
New dependency edges:
Files to change:
Responsibility added to each file:
Expected size change:
```

### 编码过程中

Agent 必须遵循以下规则：

- 把行为放到 owner 中，不放到最近的文件中。
- 复用现有公开契约，不创建平行实现。
- 不为一个实现创建接口、工厂或插件层。
- 不把大型函数原样移动到新文件。
- 不创建没有明确职责的通用 helper 模块。
- 如果目标文件已超过 800 行，新行为必须进入职责明确的新模块。原文件只能接线，且净行数不得增长。

### 停止并重新设计

出现以下情况时，Agent 必须停止当前方案并重新分解：

- 新文件预计超过 500 行。
- 任意文件将超过 800 行。
- 新函数预计超过 120 行。
- 新类预计超过 400 行。
- 模块将获得第二种独立变化原因。
- 方案将创建循环依赖或反向依赖。
- 方案需要导入另一个生产模块的私有符号。
- 超限文件的净行数将增加。
- 单文件承载超过 300 行新增代码，但没有 Architecture Delta。

不要通过请求豁免继续实现。先重新确定 owner、contract 和依赖方向。

### 完成实现后

Agent 必须：

1. 运行 Ruff 检查和格式检查。
2. 运行架构检查。
3. 运行覆盖变更行为的验证。
4. 确认没有新增基线条目。
5. 报告新增或删除的依赖边。
6. 报告规模预算结果。

## 测试组织

按稳定行为契约创建测试文件。例如：

- archive safety
- navigation validation
- bilingual proof
- publication transaction
- resume integrity
- usage accounting

不要为新测试创建 `test_phaseN_*`、`test_newfeatures.py` 或类似的聚合桶。现有文件可以保留，但不得继续获得不相关职责。

共享 fixture 和生成器应放在已有的 `tests/sample_data.py` 或职责明确的 fixture 模块中。不要为了缩短测试文件而把测试断言隐藏在通用 helper 中。

## 自动架构检查

新增 `scripts/check_architecture.py`。使用 Python 标准库实现，不增加依赖。

脚本使用：

- `pathlib` 统计物理行数。
- `ast` 统计函数和类跨度并解析 import。
- Git diff 识别新增和修改内容。
- 图遍历检测新增循环依赖。

脚本必须检查：

1. 新 Python 文件不超过 800 行。
2. 新增或扩大的函数不超过 120 行。
3. 新增或扩大的类不超过 400 行。
4. 不新增生产跨模块私有导入。
5. 不新增模块循环或违反允许的包依赖方向。
6. 存量超限文件、类和函数不扩大。
7. 单文件变更集中度满足 Architecture Delta 要求。
8. 基线项目只能减少，不能增加。

## 管理存量债务

### 建立基线

新增 `architecture-baseline.json`，记录：

- 当前超限文件及行数。
- 当前超限函数和类及跨度。
- 当前模块循环。
- 当前跨模块私有导入。
- 当前不允许的包依赖边。

基线只用于迁移。它不是永久豁免列表。

### 使用棘轮规则

应用以下规则：

- 允许现有基线项目暂时保留。
- 基线数值只能下降，不能上升。
- 不得增加新的基线项目。
- 修改超限文件时，该文件净行数不得增长。
- 移除违反项后，必须从基线删除。
- 已删除的违反项不得重新加入基线。

这套规则先阻止恶化，再让日常修改逐步减少债务。

### 单独安排存量重构

不要把存量重构作为启用治理的前置条件。根据以下信号单独排序：

- 修改频率
- 故障率
- 最大函数跨度
- 循环依赖
- 跨模块私有耦合
- 未来需求压力

本文件不规定现有大文件的具体拆分方案。

## CI 和 pre-commit

### CI

CI 必须按以下顺序运行：

```bash
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_architecture.py
uv run --no-sync python -m unittest discover -s tests
```

CI 只检查，不自动修复。失败输出必须指出文件、符号、当前值、允许值和修复方向。

### Pre-commit

在现有 Ruff hooks 之后运行架构检查。pre-commit 可以自动格式化，但架构检查只报告错误。

CI 是最终门禁。不要依赖开发者是否安装了本地 hook。

## 架构决策记录

普通变更使用 Architecture Delta，不创建永久文档。

出现以下变化时，创建或更新持久化架构决策记录：

- 新增子系统。
- 修改允许的包依赖方向。
- 修改公开 contract。
- 修改持久化 schema 或恢复语义。
- 修改并发、锁或提交顺序。
- 新增外部 I/O 边界。
- 修改架构规模预算。
- 允许新的架构例外。

决策记录必须说明问题、约束、选择、拒绝的替代方案和可验证结果。不要记录代码可以直接表达的实现细节。

## 例外策略

默认不允许人工豁免规模和依赖规则。

只为以下内容考虑例外：

- 受外部格式约束且必须提交的生成代码。
- 无法修改的 vendored code。

当前仓库没有需要这类例外的 Python 源码。将来如需例外，必须记录 owner、原因、适用路径和删除条件。例外不得覆盖手写生产代码或测试代码。

## 分阶段启用

### 阶段 1：阻止新增债务

1. 把规模预算和 Agent 停止条件加入 `AGENTS.md`。
2. 添加架构检查脚本。
3. 生成当前基线。
4. 把脚本接入 pre-commit 和 CI。
5. 在 CI 中补充 Ruff 检查。

本阶段不修改业务模块。

### 阶段 2：执行依赖边界

1. 在 `AGENTS.md` 中声明各子系统的 Owns、Does not own、Public contract 和 Allowed dependencies。
2. 将存量循环、私有导入和反向依赖加入基线。
3. 阻止新增同类问题。
4. 对触发条件内的变更要求 Architecture Delta。

### 阶段 3：通过日常修改降低基线

1. 修改超限文件时保持净行数不增长。
2. 把新增行为放入职责明确的 owner。
3. 只在入口文件保留编排和接线。
4. 在违反项消失后立即更新基线。

### 阶段 4：单独治理高风险存量代码

根据变更频率、缺陷、依赖和函数规模选择重构目标。不要用一次全仓拆分替代前三个阶段。

## 验收标准

治理生效后，仓库必须满足以下指标：

| 指标 | 目标 |
| --- | ---: |
| 新增超过 800 行的 Python 文件 | 0 |
| 新增超过 120 行的函数或方法 | 0 |
| 新增超过 400 行的类 | 0 |
| 新增模块循环 | 0 |
| 新增生产跨模块私有导入 | 0 |
| 新增不允许的包依赖边 | 0 |
| 存量超限文件总行数变化 | 不增加 |
| CI 中 Ruff、架构检查和测试覆盖 | 100% |
| 触发条件内的 Architecture Delta 覆盖 | 100% |

存量超限文件数量不作为启用治理的前置验收条件。治理必须先阻止新增债务。

## 维护规则

架构规则本身也必须保持简单：

- 优先使用标准库检查。
- 只检查可以明确判定的事实。
- 不用单一复杂度分数替代工程判断。
- 不为可能出现的未来架构创建抽象。
- 发现规则可以被轻易规避时，修正规则或检查器。
- 每次修改预算、依赖方向或例外策略时，更新本文和自动检查。

## 参考资料

- [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/)
- [Top 10 tips for Microsoft style and voice](https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice)
