# HySkill：假想技能生成用于技能检索与加载

> 把 HyDE（ACL 2023）的"假想文档嵌入"首次迁移到 LLM 智能体的技能（SKILL.md）检索与加载上：
> **给定任务，先让模型"想象"出需要的技能，用想象的嵌入去技能库检索真技能，并用想象↔现实的差距做加载门控。**

## Phase 0 试点结果（2026-07-13）：GO ✅

SRA-Bench 5 数据集 × 26,262 技能全库，本地 Qwen3.5-4B 生成、MiniLM 编码（全部数据见 [docs/05-results.md](docs/05-results.md)）：

| 方法 | 平均 nDCG@10 | 备注 |
|---|---|---|
| bm25 / dense（基线） | 0.429 / 0.351 | 基准原装 |
| LLM 重排（SRA 报告的最强基线） | 被全面超越 | 同一个 Qwen，"想象">"重排" |
| 想象·一句话（成本 1/10） | 0.615 | 抽象域冠军 |
| 想象·一段话（HyDE 忠实移植） | 0.631 | 单金标域冠军 |
| **想象·完整技能（单向量）** | **0.675** | **最鲁棒，无塌方域** |
| 想象·四路融合 | 0.626 | 多金标域冠军 |

**五个域的冠军全部是想象类方法**；各域冠军组合平均 0.738 ≈ 最强基线的 1.7 倍。

### 三条经验规律（试点级证据，全量待验证）

- **规律 A**：映射方式跟着金标结构走——单金标域单向量胜（R@1 强），多金标域四路融合胜（R@10 强）→ 两阶段（融合召回+单路精排）/自适应挂挡；
- **规律 B**：想象粒度跟着域抽象度走——抽象域一句话就够（且便宜 10 倍），技术域要长文；完整技能形态最稳；
- **规律 C**：查询↔技能语义鸿沟越大，想象增益越大（medcalc：dense 0.139 → 想象满分 1.000）。

## 三层创新定位

1. **对象层**：查询侧假想生成首次进入技能领域（此前 ToolDreamer 限于 API 工具、SkillDAG 限于库侧索引）；
2. **方法层**：双挡位映射（单向量/四路融合）+ 挂挡规律 + 想象粒度规律；
3. **加载层（Phase 2）**：假想技能兼任"参数化知识外化快照"，驱动 S1/S2 加载门控——直击 SRA 证实的"无差别加载"瓶颈，HyDE/ToolDreamer 均止步于检索。

## 进度

- [x] 文献调研与新颖性排查（查询侧 HyDE-on-skills 无人做，见 `docs/background.md` §5）
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
