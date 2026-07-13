# HySkill：假想技能生成用于技能检索与加载

> 把 HyDE（ACL 2023）的"假想文档嵌入"首次迁移到 LLM 智能体的技能（SKILL.md）检索与加载上：
> **给定任务，先让模型"想象"出需要的技能，用想象的嵌入去技能库检索真技能，并用想象↔现实的差距做加载门控。**

## Phase 1 全量结果（2026-07-13，全部 3,970 实例 + 显著性检验；36/45 单元完成，终版更新中）

SRA-Bench 全量 × 26,262 技能全库，本地 Qwen3.5-4B 生成、MiniLM 编码（完整数据与 p 值见 [docs/05-results.md](docs/05-results.md)）：

| 域（题数） | 冠军方法 | nDCG@10 | 对最强基线 | p |
|---|---|---|---|---|
| theoremqa (747) | **两阶段（融合召回+单路精排）** | **0.819** | +0.070 | <0.0001 |
| logicbench (760) | **四路融合** | **0.368** | +0.142 | <0.0001 |
| medcalcbench (1,100) | **想象·一段话** | **0.995**（近乎解决） | **+0.549** | <0.0001 |
| champ (223) | **想象·完整技能** | **0.442** | +0.097 | 0.0004 |
| bigcodebench (1,140) | 融合系跑完后终裁 | — | — | — |

已完赛 4 域全部由想象类方法**显著**夺冠；同一个模型，"想象"显著优于 SRA 认证最强的"LLM 重排"（+0.070，p<0.0001）。

### 全量确立的规律

- **规律 1（主结论）**：查询侧假想生成显著优于全部基线与 LLM 重排；语义鸿沟越大增益越大（medcalc +0.549）；
- **规律 2（映射 × 域难度）**：难域融合胜（广撒网），强信号域单路胜（精排）——试点的"金标结构"解释已被全量修正；
- **规律 3（两阶段边界）**：强信号域最优（theoremqa 显著登顶），难域让位纯融合——非万能主方法，附带清晰适用判据；
- **规律 4（粒度）**：一句话在代码域塌方、在强信号域与长文无差——粒度只在中等难度域重要。

## 三层创新定位

1. **对象层**：查询侧假想生成首次进入技能领域（此前 ToolDreamer 限于 API 工具、SkillDAG 限于库侧索引）；
2. **方法层**：双挡位映射（单向量/四路融合）+ 挂挡规律 + 想象粒度规律；
3. **加载层（Phase 2）**：假想技能兼任"参数化知识外化快照"，驱动 S1/S2 加载门控——直击 SRA 证实的"无差别加载"瓶颈，HyDE/ToolDreamer 均止步于检索。

## 进度

- [x] 文献调研与新颖性排查（查询侧 HyDE-on-skills 无人做，见 `docs/01-background.md` §5）
- [x] 方法设计与实现（`hyskill` 包，SR-Agents 插件零改动接入，23 单元测试 + 冒烟）
- [x] **Phase 0 试点**：5 域 × 20 实例 × 8 方法，GO
- [ ] Phase 1 全量（5,400 实例）+ 显著性 + 自适应/两阶段消融 + 生成器规模消融 + 补齐外部基线（SkillRouter 等）
- [ ] Phase 2 加载门控（S1 覆盖 / S2 增益）
- [ ] arXiv 占位稿

## 仓库结构（阅读顺序即编号顺序）

```
docs/
  01-background.md                研究背景：退化现象、加载方法版图、新颖性逐篇排查
  02-innovations.md               创新点详述：三层创新 × 最近邻区分 × 证据 × 论文章节映射
  03-benchmarks-and-competitors.md 基准介绍 + 竞争方法论文自报数据 + 同场直测对比
  04-experiment-design.md         实验思路：SRA-Bench 协议、借鉴与改造、归因设计
  05-results.md                   实验结果全记录：运行台账、主表、分域全表、三规律、待补清单
  06-idea-and-decisions.md        idea 演化与决策记录（含新颖性防御问答）
  07-hyde-method.md               HyDE 原文精读：公式链、消融、到 HySkill 的映射
  superpowers/                    设计规格与实现计划
hyskill/                 实现：parser / generator(+缓存) / embedder / fusion /
                         retriever(四路) / naive_hyde(单路三粒度) / plugin
scripts/                 smoke.sh 冒烟自检 · run_phase0.sh 批跑 · analyze.py 汇总
tests/                   23 个单元测试 + 微型语料
```

## 快速复现

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" -e external/SR-Agents sentence-transformers openai
.venv/bin/pytest -q                # 23 passed
./scripts/smoke.sh                 # 离线冒烟（mock 生成器，无需端点）
MODEL=<模型> API_BASE=<OpenAI兼容地址> PILOT=1 ./scripts/run_phase0.sh   # 试点
```

想象粒度变体：`--retriever naive_hyde --retriever-arg template={passage|skill|sentence}`；四路融合：`--retriever hyskill`；Qwen3/3.5 系加 `--retriever-arg no_think=1`。
