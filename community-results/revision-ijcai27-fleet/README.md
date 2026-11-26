# 第二轮实验：IJCAI-27 补充实验（E0–E7）

**本页是第二轮实验的统一入口。** 第二轮指 2026-11-24 启动的 IJCAI 修订实验，
数据分支为 `ijcai27-revision`，完整原始证据首次发布于 `3772e7a`。
这里的“第二轮”不是早期 README 中的 `Phase 2` 加载实验，也不是 `K=2` 中的“2”；
`K=2` 表示每种模板生成两条想象文本。

## 先区分三组数据

| 数据组 | 用途 | 位置 | 与第二轮的关系 |
|---|---|---|---|
| 早期 Qwen4 / Phase 1、Phase 2、Phase 2c | 历史探索与单模型结果 | [早期实验记录](../../docs/05-results.md)、各模型 `k4/` 历史归档 | 不作为第二轮的默认主表；不同协议、split、指标不能直接拼接 |
| 修订前主实验（第一轮基准） | 完整 HySkill 与外部方法的正式比较 | 各模型 `k2/`、`baselines-runtime-matched/`；[K=2 汇总](../k2-fleet/summary.json)、[外部基线比较](../baselines-runtime-matched-fleet/paired_comparisons.json) | 原样沿用；第二轮没有重跑整套 Gated / Bare / BM25+Select / 原生 Rerank 答案 |
| 第二轮 IJCAI 补充实验 | 新检索重算、内部分析、新增 Hy+Rerank 推理 | 各模型 `revision-ijcai27/` 和本目录 | E0–E7 的唯一发布位置；补充主实验，不覆盖第一轮 |

`k2-gate-recalibration-v2/` 是独立敏感性检查，不能代替本文主引用的 `k2/`。
`k-ablation/` 是 K={1,2,4,8,10} 联合消融，也不属于本页的第二轮 E0–E7 编号。

## 完整 HySkill 与外部方法：沿用第一轮主实验

本节的 **HySkill 主方法 = K=2 路由检索 + Gated 加载**。外部对手为 Bare、
BM25→LLM Rerank→加载 top-1、BM25→LLM Select→加载选中技能。
`Hy+Select` 是 HySkill 的另一种加载配置；`Hy+Rerank` 是第二轮新增的候选来源对照。
两者都不能悄悄替换本节的 Gated 主方法。

### A. 论文主表：按题准确率

四个规则领域，每模型 2,265 道 held-out 题；先在每模型内按题汇总，再等权平均模型。
由于每个模型题数相同，所列总体数值也等于共同模型集合内的 pooled accuracy。
下表每一行只在同一模型集合内比较，不跨行比较七模型和五模型均值。

| 模型范围 | 外部方法 | 外部方法准确率 | HySkill（Gated）准确率 | 按题差值 |
|---|---|---:|---:|---:|
| 七模型 | Bare | 37.44% | 48.85% | +11.41 pp |
| 五个 Select/Rerank 可用模型 | 原生 Rerank | 52.13% | 56.33% | +4.19 pp |
| 五个 Select/Rerank 可用模型 | BM25+Select | 51.69% | 56.33% | +4.64 pp |

来源：[HySkill 按题汇总](../k2-fleet/summary.json) 的 `answers.fleet_model_macro`，
[外部基线按题汇总](../baselines-runtime-matched-fleet/metrics_summary.json) 的
`fleet_model_macro`，均取 `heldout`。
差值从未舍入的原始值计算；例如该重排对比的未舍入差值约为 4.1943 pp（显示为 +4.19 pp），
不能用已舍入的 56.33 − 52.13 反推更多小数位。

单模型的高分仍然存在：Qwen3.5-4B 的 Gated 是 **72.89%**，Qwen3.5-9B 是
**73.64%**；Qwen9 的 **75.98%** 属于 `Hy+Select` 配置。它们是单模型按题
准确率，不是七模型平均，也不是检索 Recall。

### B. 显著性检验：模型与领域等权

同一批 held-out 答案，先对每个模型×领域单元算准确率，再等权平均这些单元；
分层 bootstrap 按模型→领域→配对题目重采样 10,000 次。这与 A 表按题加权不同。

| 比较 | HySkill（Gated） | 对手 | 等权差值 | 差值 95% CI | p |
|---|---:|---:|---:|---|---:|
| vs Bare，七模型 | 45.47% | 37.24% | +8.23 pp | [+3.53, +13.87] | <0.0001 |
| vs 原生 Rerank，五模型 | 52.38% | 48.77% | +3.61 pp | [+0.79, +6.27] | 0.0154 |
| vs BM25+Select，五模型 | 52.38% | 48.94% | +3.44 pp | [+0.28, +7.38] | 0.0302 |

来源：[runtime-matched 配对比较](../baselines-runtime-matched-fleet/paired_comparisons.json)
中的 `fleet_hierarchical`。三项区间都不含零。`p<0.0001` 对应 10,000 次 bootstrap
中记录的 `p_two_sided=0.0`，不表示真实概率严格为零。
**A 表的均值/差值不能配上 B 表的 CI，也不能把 48.85% 与 45.47% 解释为两轮性能变化。**

## 第二轮具体做了什么

模型集合：七模型为 DeepSeek-7B、GLM-4-9B、Llama-3.1-8B、Mistral-7B、
Qwen3.5-4B、Qwen3.5-9B、Yi-1.5-9B；五模型排除 DeepSeek 和 Yi。
四个规则领域是 TheoremQA、LogicBench、MedCalc-Bench、CHAMP；五领域再加
BigCodeBench（只做检索）。七模型五领域的检索每模型有 3,177 道 held-out 题。

| 编号 | 实验问题与性质 | 第二轮新增计算 / 沿用数据 | 范围与结果入口 |
|---|---|---|---|
| E0 | 重算能否复现原检索结果？复现检查 | 用已发布 K=2 想象缓存、原主编码器 MiniLM 重算；不重新生成想象或答案 | 七模型×五领域；各模型 `reproduction.json`、`metrics_all-MiniLM-L6-v2.*`、`rankings_all-MiniLM-L6-v2.*` |
| E1 | HySkill 检索与外部 BM25 / Rerank 在不同深度如何比较？ | 重新统计第一轮排名与 runtime-matched 重排决策；不新增答题 | 主表五模型×四规则领域；[深度比较](e1_depth.md)、[JSON](e1_depth.json)。BigCodeBench 补充表仅四个 FastBM25 模型 |
| E2 | 路由 HySkill 是否优于外部 HyDE 式 passage 基线？ | 用 E0 的 MiniLM 逐题指标做配对比较；`naive_passage` 是本项目的 HyDE 式全文单向量基线 | 七模型×五领域；[HyDE 比较](e2_hyde.json)、[解读](router.md) |
| E3 | 路由的贡献是什么？内部配置分析 | 比较路由、全局固定配置、逐领域事后最佳固定配置；不新增推理 | 七模型×五领域；[路由分析](e3_router.json)、[解读](router.md) |
| E4 | 更换编码器后结论是否保持？稳健性检验 | 在 BGE-base、GTE-base、BGE-large 上重算同一 K=2 缓存的五个固定配置及路由；每个编码器重新用验证集选路由；不重跑下游答案 | HyDE/配置比较七模型×五领域；BM25 比较仅四个 FastBM25 模型×五领域；[编码器结果](e4_encoders.md)、[JSON](e4_encoders.json) |
| E5 | 想象是否更接近 gold 技能？机制线索 | 比较 query 与各模板 K=2 想象均值向量到 gold 的余弦；不新增推理，不是端到端准确率 | 七模型×五领域；[相似度分析](e5_mechanism.md)、[JSON](e5_mechanism.json) |
| E6 | 能否减少目标领域标注依赖？内部替代方案 | 由 MiniLM 排名计算五配置 RRF 融合及留一领域配置选择；LODO 仍使用其他四个领域的验证标签，不是全程无标注 | 七模型×五领域，仅检索；[替代方案](e6_label_free.json)、[解读](router.md) |
| E7 | 候选池改善是否能转化为下游收益？候选来源×下游的 2×2 对照 | **仅 Hy+Rerank 是本轮新增 LLM 推理**；使用第一轮冻结的 K=2 路由候选。其余三格沿用已发布答案，只重新汇总统计 | 五模型×四规则领域；[2×2 结果](e7_pool_grid.md)、[JSON](e7_pool_grid.json)、各模型 `hy_rerank/` |
| E8 | 直接加载 gold 的 oracle 上限 | 可选，第二轮未运行；早期 Phase 2 的 oracle 不能当作本轮 E8 结果 | 无本轮结果 |

E0 不是“七个自动门槛全部通过”：Llama 和 Qwen9 的原始 `passed=false` 保留，
其一实例数值/并列排序漂移经作者接受，分别有
[Llama 决定记录](../llama31-8b/revision-ijcai27/reproduction_decision.md) 和
[Qwen9 决定记录](../qwen35-9b/revision-ijcai27/reproduction_decision.md)。
最初误用 BGE 对 MiniLM 做复现检查的 DeepSeek 诊断单独归档，不进入正式 E0/E4 汇总。

### E7 四个单元格的来源

| 候选来源 | Select | Rerank |
|---|---|---|
| BM25 top-50 | 第一轮 `baselines-runtime-matched/` 的 `select_bm25` | 第一轮 `baselines-runtime-matched/` 的 `always_rerank` |
| HySkill K=2 路由 top-50 | 第一轮 `k2/` 的 `routed_select`（Hy+Select） | **第二轮新增** `revision-ijcai27/hy_rerank/` 的 `hy_rerank` |

Gated 和 Bare 是参照，不是 2×2 的两个轴。E7 用于解释候选来源，不能替代前面的
完整 HySkill 对外部系统主表。特别是：

- 主系统 **Gated vs 原生 Rerank：+3.61 pp，p=0.0154，显著**（第一轮比较）。
- 补充实验 **Hy+Rerank vs BM25+Rerank：+2.78 pp，95% CI [-0.03, +5.58]，p=0.0524，未显著**（第二轮 E7）。
- E7 的 Hy+Select 对 BM25+Select 为 +4.24 pp、p=0.0158；第一轮同一对比的
  p=0.0160。两者使用相同已发布答案，但 bootstrap 的比较名称/派生 seed 不同。
  每次引用均值、CI、p 时，必须从同一个比较文件整组取值。

## 第二轮结论的适用范围

- E2：MiniLM 下路由相对 HyDE 式 passage 的 nDCG@10 +0.045、Recall@50 +0.057，
  均显著；35 个模型×领域组合中有 12 个直接选择 passage。
- E1：主表 hit@10 和 nDCG@10 显著高于原生 Rerank，整体 hit@1 未显著；
  LogicBench 的 top-1 及 BigCodeBench 补充表的头部排序存在负结果。
- E3：路由优于全局固定配置；相对逐领域事后最佳固定配置未检测到显著差异，
  不能写成“等价”或“不劣”。
- E4：GTE-base、BGE-large 相对 HyDE 的 nDCG@10 提升显著，BGE-base 未显著；
  Recall@50 在这三个额外编码器中仅 BGE-large 显著。
- E5：余弦变化是机制线索，不是生成文本真实性、人评质量或因果机制的证明。
- E6：RRF 与路由未检测到显著差异；LODO 显著低于目标领域标注路由。
- E7：下游增益明确支持 Select；Rerank 下的均值为正但区间含零，应如实报告。

所有测试结论排除冻结的 20% 验证集。未支持的模型单元格是 `unavailable`，不是 0；
`method_failure` 保留在答题分母中。检索 nDCG/Recall、加载精度、加载率、gold-load
rate、答题准确率分别命名，不统一写成“准确率”。

## 当前完成状态

E0–E7 产物和完整原始证据已发布；E0 的两个接受例外如上。完整证据上传时通过了
28 项相关测试、204 个发布文件 SHA 校验及下文的原始文件对齐核验。
这不等同于原计划 Task 12 所要求的主仓库与 Hy+Rerank 工作树全套测试和 smoke
均已重新验收；该项仍需单独核实。E8 未运行，论文正文尚需纳入第二轮结果。

原[设计](../../docs/superpowers/specs/2026-11-24-ijcai27-revision-experiments-design.md)
和[实施计划](../../docs/superpowers/plans/2026-11-24-ijcai27-revision-experiments.md)
保留为计划记录；实际编码器、共同模型范围、复现例外和已公开排名，以本页及绑定的
机器可读证据为准。实验目录与文件名保持原样，避免破坏既有引用和输入哈希。

## 完整数据文件与原始证据

This index covers the E0–E7 revision artifacts. The complete source supplement
adds full retrieval rankings and raw E7 evidence to the existing metric and
summary packs. Original file bytes are preserved, including model responses,
method failures, interrupted attempts, and recorded runtime metadata.

## Data locations

Paths below are relative to `community-results/`.

| Path | Contents |
|---|---|
| `<TAG>/revision-ijcai27/metrics_<ENCODER>.jsonl.gz` | Per-instance metrics for all four encoders |
| `<TAG>/revision-ijcai27/rankings_<ENCODER>.jsonl.gz` | Full ordered top-50 skill IDs and gold IDs; original sibling manifests |
| `<TAG>/revision-ijcai27/reproduction.json` | Original MiniLM E0 reproduction report |
| `<TAG>/revision-ijcai27/hy_rerank/` | Compact E7 answers, decisions, usage, runtime identity, and validation |
| `<TAG>/revision-ijcai27/hy_rerank_source.tar.gz` | Raw E7 answers and reranker responses, evaluation, audits, runtime manifests, and logs |
| `revision-ijcai27-fleet/e*.json` and `e*.md` | Existing E1–E7 fleet analyses and their manifests |
| `revision-ijcai27-fleet/source-evidence/e7-inputs.tar.gz` | Exact E7 candidate pools, their manifests, and frozen validation IDs |
| `revision-ijcai27-fleet/source-evidence/e7-code.tar.gz` | Eleven source files bound by the E7 runtime manifests |
| `revision-ijcai27-fleet/source-evidence/operations.tar.gz` | E4/E7 launch scripts and orchestration logs |
| `revision-ijcai27-fleet/source-evidence/e0-operations.tar.gz` | E0 launch scripts, logs, and encoder provenance snapshots |
| `revision-ijcai27-fleet/source-evidence/encoders.e0.json` and `encoders.e4.json` | Exact encoder provenance used for MiniLM and the additional encoders |

The seven retrieval tags are `deepseek7b`, `glm4-9b`, `llama31-8b`, `mistral7b`,
`qwen3.5-4b-reference`, `qwen35-9b`, and `yi15-9b`. The encoders are
`all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `gte-base`, and `bge-large-en-v1.5`.
Each of the 28 ranking files has 23,820 rows: 3,970 instances × six arms,
including validation and held-out instances. Total formal ranking rows: 666,960.
Ranking rows have no split field; the corresponding metric rows retain the split.

E7 covers the five Select/Rerank-eligible models and four rule-scored domains.
It contains 14,150 final answers and 14,150 final decisions, with 11,325 held-out
answers. DeepSeek and Yi remain unavailable. The 127 final method failures remain
in the denominator. The five raw archives contain 347 files, including the GLM
`archive-interrupted-20261125/` directory; those interrupted files are preserved
history and must not be added to final answer counts. The 60 candidate/validation
input files are stored separately in `e7-inputs.tar.gz`.

## Integrity and provenance

`data_manifest.json` records every revision publication file with its size and
SHA-256, each archive member with its original size and SHA-256, source locations,
coverage, and validation results. Its inventory excludes itself and `SHA256SUMS`
to avoid recursive hashes. `SHA256SUMS` includes the manifest itself and all
inventory files; paths are relative to the repository root. Verify after cloning:

```sh
shasum -a 256 -c community-results/revision-ijcai27-fleet/SHA256SUMS
```

All 56 source ranking/manifest files and 441 source evidence files were compared
with server SHA-256 values. All formal ranking metrics were recomputed from the
published ranked IDs and matched the existing metric rows. E7 raw answer hashes,
decision fields, evaluation labels, candidate order, and validation membership
were matched to the existing compact packs. All 20 final domain audits are valid.
The 300 code-file references in the final E7 runtime manifests match the eleven
files in `e7-code.tar.gz`. Archive members were read back after compression.

Original manifests retain their historical server paths and generation commit.
They have not been rewritten to imply the new publication location was the
execution location. For rankings, map the original `results/revision-ijcai27/`
or `/data/hyskill-ijcai27/rankings/` location to the same model and filename under
`community-results/<TAG>/revision-ijcai27/`. The older MiniLM manifests lack
output hashes; the publication inventory supplies independently verified hashes.
Their encoder input hash resolves to `source-evidence/encoders.e0.json`, while
the E4 encoder input hash resolves to `source-evidence/encoders.e4.json`.
For raw E7 archives, prepend the recorded `source_root` to each member path to
recover its original location. Runtime paths, loopback endpoints, and device
identifiers are retained as provenance; no credentials were found in the
published source files.

The corpus, benchmark inputs, and previously published imagination caches retain
the source identities recorded in the manifests. Downloaded model weights,
virtual environments, package caches, and derived embedding caches are not part
of this experiment-output supplement. The runtime code snapshot is the exact
manifest-bound subset, not a replacement for the repository or its dependencies.

## Interpretation boundaries

MiniLM is the primary encoder. Original E0 reports retain their recorded gate
outcomes, including the accepted one-instance numerical/tie drift for Llama and
Qwen9; this upload does not change those outcomes. The initial DeepSeek BGE
diagnostic is retained under
`deepseek7b/revision-ijcai27/diagnostics/`, with its original reproduction report
at `deepseek7b/revision-ijcai27/reproduction_bge-base-en-v1.5.json`. Its 23,820
ranking rows include 70 rows that differ from the final E4 BGE ranking and are
excluded from the formal total above. It is not the MiniLM E0 gate.

The E7 candidate pools come from the frozen published K=2 routed retrieval,
not from the additional-encoder E4 rankings. E4 has seven-model retrieval support;
E7 has five-model downstream support. Preserve the scorer-specific support in
the fleet analyses, including the separate SR-Agents BM25 scorer for Qwen4.
Unchanged K=2, K=4, K-ablation, and runtime-matched baseline packs remain their
own protocols and are not replaced by this supplement.
