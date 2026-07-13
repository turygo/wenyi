# Token 与翻译耗时优化审查

> 日期：2026-07-13
> 项目：wenyi / `trans-novel`
> 目标：在不劣化翻译质量的前提下，识别可减少 Token 消耗和墙钟时间的优化机会。

## 1. 结论摘要

当前流水线已经具备几项有效优化：

- 批级断点续跑，已完成批次不会重复翻译；
- 逐章梗概与术语挖掘支持并发；
- 润色在批间异步执行；
- 默认按章裁剪术语表；
- 使用确定性 lint 拦截引号、数字、锁定专名和未译残留；
- prompt 采用“稳定内容在前、动态内容在后”的顺序，能够利用 DeepSeek 前缀缓存。

剩余优化分为两类：

1. **严格无损优化**：只改变独立任务的调度或增加观测，不改变 prompt、模型、采纳条件和输出语义。可以优先实施。
2. **Token 优化实验**：合并请求、裁剪上下文或改变输出协议。虽然潜在收益明确，但任何 prompt 变化都可能影响模型行为，必须通过真实章节 A/B 和质量门禁后才能上线。

推荐顺序：

1. Reviewer chunk 并发；
2. Naturalizer 正反 pair vote 并发；
3. 逐章梗概与术语挖掘重叠执行；
4. 增加 operation 级 Token、延迟、重试和采纳率统计；
5. 再验证上下文裁剪、预扫合并和润色增量输出。

不能把以下做法宣称为“不降质优化”：关闭润色、关闭去翻译腔、关闭审校、降低模型档位、关闭 thinking、减少现有质量门禁，或直接并行章内翻译批次。

---

## 2. 审查范围与证据

### 2.1 源码范围

本次审查覆盖了以下主要 Module：

- `trans_novel/pipeline/orchestrator.py`
- `trans_novel/pipeline/context.py`
- `trans_novel/pipeline/runstore.py`
- `trans_novel/llm/base.py`
- `trans_novel/config.py`
- `trans_novel/assemble/translator.py`
- `trans_novel/agents/base.py`
- `trans_novel/agents/analyzer.py`
- `trans_novel/agents/synopsis.py`
- `trans_novel/agents/namer.py`
- `trans_novel/agents/polisher.py`
- `trans_novel/agents/reviewer.py`
- `trans_novel/agents/naturalizer.py`
- `trans_novel/agents/prompts.py`
- `trans_novel/glossary/miner.py`
- `trans_novel/glossary/store.py`
- `trans_novel/pipeline/lint.py`
- usage、术语裁剪、翻译、润色、审校相关测试。

### 2.2 运行数据范围

读取了 `state/` 中以下三本书的持久化事件与 usage 快照：

- Atomic Habits
- The Wedding People
- Chip War

没有修改 `state/`。

### 2.3 数据限制

当前统计存在以下限制：

- `UsageTracker` 不单独记录 reasoning tokens；
- 不记录单次请求耗时；
- 不记录失败的网络尝试；
- 同一个 Agent 的不同 operation 使用同一个 stage 名；
- 部分书的 usage 是旧统计格式，只能按 tier 归因；
- Atomic Habits 的阶段明细来自一次续跑快照，不包含首次预扫的完整全书账单；
- 润色增量分析只统计事件中能够正确配对的 `batch_translated` / `batch_polished` 样本。

因此本文把已观察事实和待验证推断分开陈述。

---

## 3. 当前完整流程

### 3.1 准备阶段

```text
读取输入
→ 解析章节、段落和目录
→ 可选模型语言检测
→ 初始化 RunStore / manifest / chapter JSON
→ 强档分析样章
→ 生成风格指南、角色信息和初始术语
```

关键代码：

- `Orchestrator.prepare()`：`pipeline/orchestrator.py:193–254`
- `_detect_language_ai()`：`pipeline/orchestrator.py:256–276`
- `_sample_text()`：`pipeline/orchestrator.py:278–299`
- `Analyzer`：`agents/analyzer.py`

### 3.2 全书理解与定名

```text
逐章梗概：fast × 正文章数，并发
→ 逐章术语候选挖掘：fast × 正文章数，并发
→ 候选分组定名：strong × 候选组数，并发
→ 全书概览归并：fast × map-reduce 轮次
```

关键代码：

- `_build_understanding()`：`pipeline/orchestrator.py:447–577`
- `Synopsizer`：`agents/synopsis.py`
- `mine_candidates()`：`glossary/miner.py:422–438`
- `CastNamer`：`agents/namer.py`

注意：README 仍描述英文候选挖掘为确定性、零 Token；当前代码实际采用：

```text
英文确定性大写通道 ∪ 每章 fast LLM 通道
```

证据：`glossary/miner.py:422–438`。因此英文书也会产生每正文章一次术语挖掘请求。

### 3.3 逐章翻译关键路径

```text
逐章
  → 按字符预算切成 batch
  → 读取最新 rolling context
  → strong 批翻译，章内严格串行
  → 段数对齐校验
  → 确定性 lint
  → actionable lint 命中时逐段定向重译
  → 保存 raw 译文和 pending_polish 标记
  → 后台提交 strong 润色
  → 下一批继续翻译
```

关键代码：

- `_translate_chapter()`：`pipeline/orchestrator.py:819–1211`
- `_process_batch()`：`pipeline/orchestrator.py:1459–1483`
- `Translator.translate_batch()`：`assemble/translator.py:116–155`
- `RollingContext`：`pipeline/context.py`

章内翻译不能直接并发。后一个 batch 依赖前一个 batch 的最新译文，用于：

- 指代与人物称谓；
- 跨段句意；
- 语气和文体连续性；
- 前文未入术语表的译名延续。

并行章内 batch 会改变质量合同。

### 3.4 润色、去翻译腔和审校

章末流程：

```text
排干本章润色 futures
→ 标点规范化
→ 润色结果确定性 lint；引入新 issue 的段回退 raw
→ Naturalizer 单语筛查
    → strong 单段改写
    → 确定性 lint
    → cheap 双语忠实度检查
    → cheap 正反两次成对判断
→ 章末分块审校
→ 对 missing / mistranslation 严重项定向重译
→ 可选回译抽检
→ 标记章节完成
```

关键代码：

- `_drain_chapter_polish()`：`pipeline/orchestrator.py:1213–1302`
- `naturalize_chapter()`：`agents/naturalizer.py:125–244`
- `_review_chapter()`：`pipeline/orchestrator.py:1351–1372`
- `_autofix_severe()`：`pipeline/orchestrator.py:1391–1457`

### 3.5 收尾

```text
翻译章节标题和额外目录项
→ 可选全书一致性 QA
→ 生成报告
→ 回填 EPUB / TXT
→ 可选生成双语版
```

关键代码：

- `_translate_titles()`：`pipeline/orchestrator.py:580–686`
- `run_steps()`：`pipeline/orchestrator.py:1488–1583`
- `run_all()`：`pipeline/orchestrator.py:1585–1600`

---

## 4. 真实 Token 与耗时基线

### 4.1 三本书整体数据

| 书 | 总 Token | LLM Calls | 翻译时段 | 吞吐 |
|---|---:|---:|---:|---:|
| Atomic Habits，续跑快照 | 2.92M | 676 | 1.75h / 1,179 段 | 674 段/h |
| The Wedding People | 6.84M | 1,850 | 4.86h / 4,667 段 | 960 段/h |
| Chip War | 4.84M | 1,316 | 3.50h / 2,223 段 | 634 段/h |

三本书的 strong 档占总 Token：

| 书 | strong 占比 |
|---|---:|
| Atomic Habits | 82.2% |
| The Wedding People | 74.1% |
| Chip War | 74.9% |

结论：成本优化必须优先关注 Translator 和 Polisher；只优化 fast 预扫无法显著改变总账。

### 4.2 Atomic Habits 阶段分布

Atomic Habits 的续跑快照能够完整按 stage 归因：

| 阶段 | Calls | Prompt Tokens | Completion Tokens | Total Tokens | 总 Token 占比 | Cache Hit |
|---|---:|---:|---:|---:|---:|---:|
| Translator | 234 | 1,224,474 | 226,021 | 1,450,495 | 49.7% | 86.2% |
| Polisher | 152 | 606,211 | 304,744 | 910,955 | 31.2% | 75.9% |
| Reviewer | 79 | 255,105 | 113,712 | 368,817 | 12.6% | 48.0% |
| Naturalizer | 210 | 94,111 | 80,650 | 174,761 | 6.0% | 20.1% |
| Title translator | 1 | 11,311 | 951 | 12,262 | 0.4% | 0% |

Translator + Polisher 合计占总 Token：

\[
49.7\% + 31.2\% = 80.9\%
\]

主要判断：

- Translator 的缓存命中率已经达到 86.2%，继续只调整前缀顺序不是第一优先级；
- Polisher 是第二大成本中心；
- Reviewer 占 12.6%，且存在不改变任何判断逻辑的并发机会；
- Naturalizer 总 Token 占比不算最高，但 210 次调用会显著增加请求等待；
- 标题全量术语注入明显浪费，但单次请求对全书总账影响有限。

---

## 5. 当前已经有效的优化

这些机制应保留，不应在后续改造中被破坏。

### 5.1 批级断点续跑

`_translate_chapter()` 会检查整批是否已有非空 target。已完成批次：

- 不重新翻译；
- 只重建 rolling context；
- 重新运行零成本 lint；
- 保留未完成润色标记供章末恢复。

这避免长书中断后重复消耗 strong Token。

### 5.2 幂等标记

当前至少包含：

| 标记 | 范围 | 作用 |
|---|---|---|
| `source_digest` | 章 | 跳过已生成的章节梗概 |
| `term_mining_done` | 全书 | 避免重复全书定名 |
| `pending_polish` | batch | 恢复中断前未写回的润色 |
| `naturalized` | 章 | 避免重复去翻译腔 |
| `review_pending` | 章 | 恢复异步审校 |

### 5.3 前缀缓存顺序

Translator user prompt 的顺序大致是：

```text
风格指南
→ 全书概览
→ 本章梗概
→ 术语表
→ 滚动上下文
→ 当前源文
```

稳定内容位于动态内容之前。Atomic Translator 的 86.2% cache hit 说明该设计有效。

### 5.4 术语按章裁剪

默认配置：

```yaml
pipeline:
  glossary_scope: chapter
```

`_chapter_term_snapshot()` 只保留：

- 本章源文实际命中的术语；
- 本章以部分姓名形式出现的锁定人物。

代码注释记录过一个真实极端：357 个锁定人物约 1.1 万字符，全量注入会让 prompt 七成以上都是术语噪声。

### 5.5 确定性 lint

以下检查不消耗 Token：

- 直接引语引号丢失；
- 数字不一致；
- 锁定专名漂移；
- 未译残留；
- 长度异常提示。

只有 actionable issue 才触发定向重译，避免用 LLM 做机器可确定的检查。

---

## 6. 严格无损的耗时优化

本节优化保持以下条件完全不变：

- 模型和 tier 不变；
- prompt 字节不变；
- 请求数量和 Token 不变；
- 采纳条件不变；
- 最终结果合并顺序不变；
- RunStore 和 SQLite 写入仍由主线程执行。

### 6.1 Reviewer chunk 有界并发

现状：

```python
for chunk in self._pack_contiguous(pairs, budget):
    for issue in self.reviewer.review(srcs, tgts, terms):
        ...
```

位置：`pipeline/orchestrator.py:1351–1372`。

每个 chunk：

- 只读取固定的 `(source, target)` 数据副本；
- 使用同一个术语快照；
- 不修改正文；
- issue index 可以在结果返回后确定性映射回章内段号。

建议：

1. 按现有 `_pack_contiguous()` 切块；
2. 有界并发提交全部 review chunk；
3. 按原 chunk 顺序合并结果；
4. 全部 review 完成后，继续按段号顺序执行 autofix。

收益：

- Token 不变；
- 审校结论逻辑不变；
- 墙钟时间从 chunk 延迟之和下降到接近若干并发波次。

这是最适合优先落地的优化之一。

### 6.2 Naturalizer 正反 pair vote 并发

现状：

```python
order1 = self.judge_pair(orig, rewritten)
order2 = self.judge_pair(rewritten, orig)
return order1 == "B" and order2 == "A"
```

位置：`agents/naturalizer.py:89–93`。

两次判断互不依赖，只在最后组合布尔结果。可以在 fidelity 通过后同时发送两个请求，再使用原判定式。

收益：

- 两次判断仍全部执行；
- 顺序偏差保护不变；
- Token 不变；
- pairwise 等待时间接近减半。

不建议把 fidelity 和两个 pair vote 同时提前执行。当前 fidelity 失败会短路 pairwise；全部提前并发会增加无效请求和 Token。

### 6.3 Digest 与 term mining 重叠执行

当前 `_build_understanding()` 先等待所有 digest，再启动 term mining。但二者只依赖章节源文，不相互依赖。

可改为：

```text
chapter digest × C ─┐
                    ├→ CastNamer → book synopsis
term mining × C ────┘
```

收益：

- 保留原来的两套 prompt；
- 输出和质量合同不变；
- Token 不变；
- 预扫减少一个完整并发波次。

当前 `prescan_concurrency` 默认已经是 4。优化重点不是把 1 改为 4，而是让两类预扫彼此重叠。

---

## 7. 必须先补的成本观测

### 7.1 当前观测 interface 的问题

`UsageTracker` 当前记录：

```text
calls
prompt_tokens
completion_tokens
total_tokens
cache_hit_tokens
cache_miss_tokens
```

`Agent._ask_json()` 和 `_ask_text()` 统一传入：

```python
stage=type(self).__name__
```

因此以下调用无法分开：

#### Translator

- 正常 batch 翻译；
- alignment 整批重试；
- 逐段 fallback；
- lint 定向重译；
- review autofix。

#### Naturalizer

- screen；
- rewrite；
- fidelity；
- 正反 pair vote。

当前只能回答“哪个 Agent 贵”，不能回答“哪个 operation、失败重试或拒收最贵”。

### 7.2 建议补充字段

一次 LLM invocation 建议记录：

- `stage`
- `operation`
- `tier`
- `model`
- `elapsed_ms`
- `attempt`
- `retry_reason`
- `fallback`
- `prompt_tokens`
- `completion_tokens`
- `reasoning_tokens`，provider 提供时
- `cache_hit_tokens`
- `cache_miss_tokens`
- `accepted` / `rejected`，适用于润色、去腔和定向重译

### 7.3 Alignment 请求放大上界

默认：

```yaml
pipeline:
  align_retry_limit: 2
llm:
  max_retries: 4
```

一次 batch 最多先执行：

\[
2 + 1 = 3
\]

次整批逻辑调用。仍不对齐时，再对批内每段单独翻译。

对于包含 \(N\) 段的批次，逻辑模型调用上界为：

\[
3 + N
\]

每个逻辑调用底层最多尝试：

\[
4 + 1 = 5
\]

次网络请求，因此理论请求尝试上界为：

\[
5(3 + N)
\]

这是代码上界，不代表真实发生。当前统计不记录失败尝试和 alignment 原因，不能据此估算实际浪费；应先补 operation telemetry。

---

## 8. 低风险 Token 优化候选

这些优化不删除质量阶段，但会修改 prompt 或上下文，仍需真实章节回归。

### 8.1 删除 Naturalizer 未消费的输出字段

当前 prompt 要求：

| Operation | 输出字段 | 代码是否使用 |
|---|---|---|
| screen | `index` | 使用 |
| screen | `quote` | 使用 |
| screen | `reason` | 使用，传给 rewrite |
| screen | `rewrite` | **未使用** |
| pair vote | `winner` | 使用 |
| pair vote | `reason` | **未使用** |
| fidelity | `faithful` | 使用 |
| fidelity | `detail` | **未使用** |

证据：

- prompt：`agents/prompts.py:352–419`
- 消费代码：`agents/naturalizer.py:63–101, 169–208`

可以把输出合同缩减为：

```json
{"issues":[{"index":0,"quote":"……","reason":"……"}]}
```

```json
{"winner":"A"}
```

```json
{"faithful":true}
```

潜在收益：

- 减少 cheap completion token；
- 减少 JSON 解析失败面；
- 不删除任何质量判断。

风险：输出理由或 rewrite 要求可能间接影响模型判断。cheap 档已经启用内部 thinking，但仍应对 screen 召回率、fidelity 通过率和最终人工盲评做 A/B。

### 8.2 标题术语按标题命中裁剪

`_translate_titles()` 当前使用：

```python
prompts.render_glossary(glossary.all_terms())
```

位置：`pipeline/orchestrator.py:638–654`。

Atomic Habits 的术语库实际有 724 条：

| 类型 | 锁定 | 数量 |
|---|---:|---:|
| 人物 | 1 | 357 |
| 术语 | 0 | 193 |
| 组织 | 0 | 117 |
| 地名 | 0 | 52 |
| 作品 | 0 | 2 |
| 物品 | 0 | 2 |
| 组织 | 1 | 1 |

标题翻译单次 prompt 实测为 11,311 tokens。

可以按所有待译标题统一筛选：

- source 或 alias 实际命中；
- 部分姓名实际出现时保留对应锁定人物；
- 不出现在任何标题里的普通术语不注入。

这是降噪，不是删除标题所需信息。但必须测试部分姓名、别名和非空格语言的命中规则。

### 8.3 CastNamer 按候选出现章节选择 digest

当前：

```python
digest_text = "\n".join(digests)
if len(digest_text) > 6000:
    digest_text = digest_text[:6000]
```

然后每个候选组重复注入同一段 digest。位置：`agents/namer.py:57–64`。

`Candidate` 已记录：

```python
chapters: list[int]
contexts: list[str]
```

因此每个候选组可以选择候选真实出现章节的 digest，而不是永远使用全书开头 6,000 字。

潜在收益：

- 减少重复输入；
- 长书后半部分候选得到更相关的上下文；
- 可能同时提高定名质量。

需要验证跨章人物、同名对象和伏笔类候选是否仍有足够全局信息。

### 8.4 Reviewer 按 chunk 选择术语

当前 `_review_chapter()` 对每个 review chunk 重复注入整章术语快照。

可以针对 chunk 原文筛选实际命中的术语。但必须保留：

- 部分姓名命中；
- 当前 chunk 需要的人物性别信息；
- alias；
- 可能影响 pronoun 审校的人物条目。

不建议直接调用简单的精确字符串过滤，否则可能降低 pronoun 与称谓检查质量。应把选择规则集中在一个 prompt-context Module 中，复用 `_chapter_term_snapshot()` 已有的姓名匹配语义。

---

## 9. 需要正式 A/B 的较大优化

### 9.1 合并逐章 digest 与 term mining

当前两个调用都读取：

```python
source_text[:8000]
```

证据：

- `agents/synopsis.py:22–31`
- `glossary/miner.py:373–419`

当前每正文章：

```text
源文前 8,000 字 → fast digest
源文前 8,000 字 → fast candidates
```

候选方案：一次请求返回：

```json
{
  "digest": "……",
  "candidates": ["……", "……"]
}
```

理论收益：

- 每正文章少一次请求；
- 每正文章最多少发送约 8,000 个源文字符；
- 预扫调用数从约 \(2C\) 降为 \(C\)；
- 墙钟时间减少一个预扫波次。

质量风险：多任务 prompt 可能使 digest 或 candidates 的质量下降。

验收指标：

- digest 是否保留剧情转折、人物关系、结局和伏笔；
- term candidate precision / recall；
- 最终锁定人物名覆盖率；
- 领域术语覆盖率；
- review 中 terminology issue 数；
- 人工抽检早期章节是否因全书理解不足而误译。

如果 A/B 不通过，保留两次调用，只实施二者并发。

### 9.2 Polisher 增量输出

当前 Polisher 必须返回完整等长数组，即使某段完全未改，也重新输出全文。

可配对事件样本统计：

| 书 | 配对段数 | 实际改动段比例 | 仅输出改动文本的理论 completion 字符节省 |
|---|---:|---:|---:|
| Atomic Habits | 2,457 | 55.7% | 16.2% |
| The Wedding People | 4,640 | 54.7% | 22.3% |
| Chip War | 954 | 89.6% | 2.4% |

候选输出：

```json
{
  "polished": [
    "改写后的第0段",
    null,
    null,
    "改写后的第3段"
  ]
}
```

`null` 表示沿用原译。代码复原等长数组后，继续执行现有确定性 lint 和回退。

优点：

- 不删除润色阶段；
- 对改动较少的小说，可能减少约两成 Polisher completion；
- 保留逐段对齐。

风险：

- 模型可能把应改段错误标为 `null`；
- 输出 interface 更复杂；
- 不同文体收益波动大；
- 可能增加 alignment 或 JSON 错误。

必须用原协议作为对照组进行人工盲评，并比较 lint 回退率和段数对齐失败率。

### 9.3 Batch 大小参数扫描

默认：

```yaml
segment:
  max_chars_per_batch: 1800
```

扩大 batch 理论上可以：

- 减少请求数；
- 减少 system、风格、概览、梗概、术语和 rolling context 的重复输入；
- 提高缓存利用率。

但也可能：

- 增加段数对齐失败；
- 降低长上下文注意力；
- 使单次重试更昂贵；
- 影响长段落的文学质量。

不能直接修改默认值。应在 operation telemetry 完成后，用同一组真实章节比较例如 1,800 / 2,400 / 3,000 字符：

- prompt tokens / source character；
- completion tokens / source character；
- calls / 1,000 source characters；
- alignment retry rate；
- lint issue rate；
- review severe issue rate；
- 人工盲评。

只有质量指标不退化时才能调整默认值。

---

## 10. 明确不建议的做法

### 10.1 关闭 `pipeline.polish`

Polisher 占比较高，但关闭它等于移除整次文学性加工，不能满足“不降质”。

### 10.2 关闭 `pipeline.naturalize`

Naturalizer 有明确的翻译腔识别、忠实度和双顺序判断合同；关闭会改变最终中文自然度。

### 10.3 删除正反 pairwise 中的一次判断

两次判断用于降低 A/B 顺序偏差。可以并发，但不能在没有质量实验的情况下只保留一次。

### 10.4 关闭 strong / cheap thinking

当前 completion token 中可能包含大量推理消耗，但项目没有 operation 级数据证明哪些判断可以无损关闭 thinking。

### 10.5 并行章内翻译 batch

会破坏 rolling context、指代、称谓和跨段句意。不能作为严格无损优化。

### 10.6 只追求更高 cache hit

Atomic Translator 已达到 86.2%。缓存能够降低缓存命中部分的计费和延迟，但无法消除翻译、润色和审校对正文的实际处理。

### 10.7 优化默认关闭的 inflight glossary

当前默认：

```yaml
pipeline:
  inflight_glossary: false
```

因此异步化批后术语抽取对默认流程没有收益。除非目标配置明确开启该功能，否则不应优先投入。

---

## 11. 推荐实施路线

### 第一阶段：严格无损

1. 增加 operation 级 telemetry；
2. Reviewer chunks 有界并发；
3. Naturalizer 正反 pair vote 并发；
4. Digest 与 term mining 同时启动；
5. 验证最终 issue 顺序、落盘顺序和续跑行为不变。

### 第一阶段实施状态（2026-07-13）

以下五项已在 `trans_novel/llm/base.py`、`trans_novel/agents/base.py`、全部生产 LLM callsite、
`trans_novel/pipeline/orchestrator.py`、`trans_novel/agents/naturalizer.py` 实现并通过聚焦测试：

1. **operation 级 telemetry**：`UsageTracker` 新增 `by_operation`，在原有 token/cache/calls
   字段之上记录 `logical_calls`、`attempts`、`failed_attempts`、`elapsed_ms`、
   `reasoning_tokens`、`accepted`、`rejected`；`by_tier`/`by_stage`/`totals` 字段形状不变，
   历史快照缺 `by_operation` 按 0 合并。20 个生产 operation 名（`translate.batch`、
   `translate.lint_fix`、`translate.review_fix`、`naturalize.screen/rewrite/fidelity/pair`、
   `review.chapter`、`polish.batch`、`prescan.digest/term_mine/name_terms/book_synopsis`、
   `language.detect`、`title.translate`、`analyzer.analyze`、`glossary.extract`、
   `glossary.audit`、`backtranslate.translate/check`、`consistency.check/autofix`）已逐一标注，
   `Agent._ask_json`/`_ask_text` 把 `operation` 设为必填参数，杜绝遗漏。
2. **Reviewer chunk 有界并发**：`Orchestrator.run()` 新建 book-wide 专用 `review_executor`
   （4-worker，与章级 `executor` 分离避免嵌套死锁），贯穿同步（`autofix_severe=true`）与
   异步（`review_pending`）两条路径；`_review_chapter()` 按提交顺序（非完成顺序）取
   `fut.result()`，保证并发下合并结果仍严格保持原 chunk 顺序。
3. **Naturalizer 正反 pair vote 并发**：`naturalize_chapter()` 为整章复用一个
   `ThreadPoolExecutor(max_workers=2)`，仅当 `fidelity_check(...) is True` 时才把两次
   `judge_pair` 提交进该池；fidelity 未过仍严格短路，不提前发出成对判断请求。
4. **digest 与 term mining 重叠执行**：`_build_understanding()` 把 `mine_candidates()`
   提交进独立后台线程池后立即返回 future，主线程随即进入 digest 的 `as_completed`
   落盘循环，二者真正同时在跑；naming 仍等待两条分支都收尾才开始，`term_mining_done`
   只在两者都成功时落盘，digest 异常仍整体冒泡且优先于 mining 异常。

**审查中两轮修复**：

- 初始化失败的 telemetry 缺口：`DeepSeekClient.complete()` 原先只在重试包裹的
  `create()` 调用外层计时，`resolve_tier` 缺档、`_ensure_client` 缺 API key/SDK 等
  入口即失败的路径不会记 `logical_calls`/`elapsed_ms`。已把 try/finally 计时范围
  扩大到整个 `complete()` 方法体，任意环节异常都先记账再原样重抛；`attempts` 仍
  只在真正调用 `create()` 时计数。
- `Orchestrator._flush_usage()` 持久化门控缺口：原判断仅看
  `increment["totals"]["calls"]`，导致"整次逻辑调用失败、无成功 token 响应"的
  operation-only 增量（`attempts`/`failed_attempts`/`logical_calls` 增长但
  `by_tier`/`by_stage`/`totals.calls` 全零）不会触发 `save_usage`，续跑后这类失败
  尝试的底层统计永久丢失。已把跳过条件改为
  `bool(increment["totals"]["calls"]) or bool(increment.get("by_operation"))`。

**最终验证**：`uv run ruff format .`、`uv run ruff check --fix .`（清出 1 处 E731 lambda
赋值并改写为 `def`）、`uv run python -m unittest discover -s tests` 全部通过（308 个测试）。

**尚未做的事**：本轮全部验证基于离线 `FakeClient`，**没有跑过任何真实联网翻译基准**，
因此不对本次改动声称具体的墙钟加速比例或 Token 降幅——这类数字目前无从谈起。
下一次对三本书之一（或新书）跑真实翻译时，应先用改动前的 `git stash`/上一个 tag
跑一次留档，再用本次改动跑一次，用新增的 `by_operation` 快照（`elapsed_ms`、
`attempts`、`accepted`/`rejected`）与本文档第 4 节的旧基线对照，才能建立可信的
第一阶段收益基线，供第二、三阶段决策参考。

### 第二阶段：低风险 Token 优化

1. 删除 Naturalizer 未消费的 JSON 字段；
2. 标题术语按实际命中裁剪；
3. CastNamer 按 `Candidate.chapters` 选择 digest；
4. Reviewer chunk 使用保守的相关术语选择；
5. 对真实章节做 A/B 和人工盲评。

### 第三阶段：实验性优化

1. 合并 digest 与 term mining；
2. Polisher 增量输出；
3. 扫描 batch 大小；
4. 在有 operation 级数据后，评估个别判断任务的 reasoning effort；
5. 只有质量指标不退化才切换默认路径。

---

## 12. 验证原则

任何声称“不降质”的优化都应至少通过以下检查。

### 12.1 确定性回归

- 输入段数与输出段数一致；
- 引号保留；
- 数字一致；
- 锁定专名一致；
- 无新增未译残留；
- 续跑不重复发已完成请求；
- 并发完成顺序不影响最终落盘顺序。

### 12.2 模型质量回归

在固定真实章节集上比较：

- review issue 数和类型；
- severe issue 数；
- autofix 采纳率；
- polish lint 回退率；
- naturalize fidelity 通过率；
- naturalize pairwise 通过率；
- 术语 precision / recall；
- 人工双盲偏好。

### 12.3 成本与性能指标

- 总 prompt tokens；
- 总 completion tokens；
- reasoning tokens，provider 提供时；
- cache hit / miss tokens；
- calls；
- retry / fallback 次数；
- 每 operation P50 / P95 延迟；
- Token / 1,000 source characters；
- 秒 / 1,000 source characters。

### 12.4 上线门槛

建议采用以下原则：

```text
确定性质量指标不得退化
AND severe issue 不增加
AND 人工盲评不劣于基线
AND Token 或墙钟时间有稳定收益
```

若不能同时满足，保留当前实现。

---

## 13. 最终建议

第一步不应修改模型档位或关闭质量阶段，而应：

1. 深化 LLM invocation telemetry；
2. 把 Reviewer chunk、Naturalizer pair vote、两类预扫按真实依赖并发；
3. 获得 operation 级 Token、延迟、重试与采纳率基线；
4. 再用真实章节评估 prompt 上下文裁剪和请求合并。

这一顺序先获得不改变翻译结果的墙钟收益，同时为后续 Token 优化建立可信证据，避免凭感觉降低 reasoning、减少审校或扩大 batch 后才发现质量回归。
