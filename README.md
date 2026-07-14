# HySkill：假想技能生成用于技能检索与加载

> 把 HyDE（ACL 2023）的"假想文档嵌入"首次迁移到 LLM 智能体的技能（SKILL.md）检索与加载上：
> **给定任务，先让模型"想象"出需要的技能，用想象的嵌入去技能库检索真技能，并用想象↔现实的差距做加载门控。**

## Phase 1 全量结果（终版：45/45 单元，全部 3,970 实例 + 显著性检验）

SRA-Bench 全量 × 26,262 技能全库，本地 Qwen3.5-4B 生成、MiniLM 编码（完整数据与 p 值见 [docs/05-results.md](docs/05-results.md)）：

| 域（题数） | 冠军方法 | nDCG@10 | 关键对比 |
|---|---|---|---|
| theoremqa (747) | **两阶段（想象系）** | **0.819** | 超重排 +0.070，p<0.0001 |
| logicbench (760) | LLM 重排 | 0.391 | 对我方融合 +0.023，n.s. |
| medcalcbench (1,100) | **想象·一段话** | **0.995**（近乎解决） | 超重排 +0.097，p<0.0001；超最强基线 +0.549 |
| champ (223) | **两阶段（想象系）** | **0.445** | 对重排 +0.048，n.s. |
| bigcodebench (1,140) | LLM 重排 | 0.628 | 对我方融合 +0.053，p<0.0001 |

**终局**：域冠军想象系 3 : 重排 2；五域均值两强几乎打平（两阶段 0.604 vs 重排 0.613），**单次在线成本同数量级（实测 ~2.7k vs ~3.9k tokens/题），但想象 token 一批三用（检索/路由/门控）、可离线预生成、与库解耦，且无重排的 BM25 召回天花板**（medcalc 实证封顶）。全部想象变体对传统基线（hybrid 均值 0.456）五域全部显著胜出。

## Phase 2 门控结果（终版：4 规则域 × 5 臂 × 2,830 题端到端做题）

检索固定为我方想象 top-1，只变加载策略；门控信号全部复用检索阶段已缓存的想象文档，**0 次新 LLM 调用**：

| 臂（做题准确率 %） | theoremqa | logicbench | medcalc | champ | 合并 |
|---|---|---|---|---|---|
| bare 不给技能 | 60.8 | **72.1** | 62.4 | 48.9 | 63.5 |
| always 无脑塞 | 67.7 | 67.5 | 77.5 | 44.0 | 69.6 |
| **gated 门控（我方）** | 68.0 | 70.7 | **78.2** | 45.7 | **70.9** |
| select 模型自选（每题 +1 次 LLM） | **70.0** | 69.5 | 75.5 | **51.6** | 70.5 |
| oracle 天花板 | 74.0 | 84.2 | 79.6 | 56.5 | 77.6 |

**三个答案**：① 无差别加载在 2/4 域净伤害（logicbench −4.6 p=0.012、champ −4.9）——遮蔽效应实锤，且 oracle 84.2 证明是"检索失败"而非"库里没书"；② 门控四域全部不降反升（合并 +1.34 vs always，对裸考 +7.4 p<0.001），同一保守标定自动适应库质量（拦截率 1.3%–98.8%）；③ 与模型自选统计打平但**成本差一个数量级**（0 vs 5,660 次额外推理）。

## 终局：粒度路由 + 门控（v2 完整流水线）

用与门控同一个 20% 验证集给每域选最优想象变体（"路由"，5/5 命中全量冠军），门控对新 top-1 重标定：

- **检索端**：路由想象 **显著反超 LLM 重排**（合并 nDCG@10 0.687 vs 0.662，+2.5，p<0.0001）——同等在线成本下显著更准，且想象 token 一批三用、可离线缓存、与库解耦；
- **做题端**：**gated_r 合并 72.5%，显著击败所有可部署对手**（vs 固定门控 +1.6 p=0.027、vs 模型自选 +2.0 p=0.018、vs 裸考 +9.0 p<0.0001）；medcalc 81.6 触及 oracle 天花板，champ 被路由从遮蔽域治愈（塞书 −4.9 → +3.6）；
- **互补性**：路由（油门）在 logicbench 单用有毒（更像的错书更误导），门控（刹车）完全兜住——**合体后无一域落败**。

完整表与显著性见 [docs/05-results.md](docs/05-results.md) §5.4。

### 全量确立的规律

- **规律 1**：想象显著优于全部传统基线（五域 p<0.001）；在强信号域对 LLM 重排亦显著取胜；
- **规律 2**：想象 vs 重排——固定变体打平、路由后显著反超；单次在线成本同数量级，但想象 token 一批三用且与库解耦；重排优势集中于"嵌入弱、字面尚可"的域；组合（想象召回+LLM 精排）为自然后续；
- **规律 3**：两阶段在想象信号可靠的域登顶,信号不可靠时破坏融合原序——附清晰适用判据；
- **规律 4**：想象粒度只在中等难度域重要；完整技能是均值最高且最稳的单路形态。

## 三层创新定位

1. **对象层**：查询侧假想生成首次进入技能领域（此前 ToolDreamer 限于 API 工具、SkillDAG 限于库侧索引）；
2. **方法层**：双挡位映射（单向量/四路融合）+ 挂挡规律 + 想象粒度规律；
3. **加载层（Phase 2 已验证）**：假想技能兼任"参数化知识外化快照"，驱动 S1/S2 加载门控——直击"无差别加载"瓶颈（本实验实测 2/4 域净伤害），门控四域全不降、合并 +1.34/+7.4（vs always/bare），零新增推理；HyDE/ToolDreamer 均止步于检索。

## 进度

- [x] 文献调研与新颖性排查（查询侧 HyDE-on-skills 无人做，见 `docs/01-background.md` §5）
- [x] 方法设计与实现（`hyskill` 包 + `scripts/gate.py`，SR-Agents 插件零改动接入，34 单元测试 + 冒烟）
- [x] **Phase 0 试点**：5 域 × 20 实例 × 8 方法，GO
- [x] **Phase 1 全量**：45/45 单元、3,970 实例 + 配对 bootstrap 显著性
- [x] **Phase 2 加载门控**：5 臂 × 4 规则域端到端做题 + 显著性（S1/S2 信号零新调用）
- [ ] 消融（K、生成器规模、编码器）+ 外部基线（SkillRouter）+ bigcode Stage 3 + ToolQA
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

## 🤝 多模型协作实验（招募中）

现有结果均出自 Qwen3.5-4B 单一生成模型——我们正在把想象检索 + 门控在 **Llama / Mistral / GLM / Gemma / Yi** 等家族上众包复跑，证明方法与模型无关。**认领一个模型 + 一张 24GB 显卡 + 一条命令**：

```bash
(TAG=<模型tag> MODEL=<模型tag> API_BASE=http://localhost:8000/v1 TRACKB=1 \
  nohup ./scripts/run_multimodel.sh > run.log 2>&1 &)
```

全流程多线程（生成 32 并发 / 做题 48 并发）、断点续跑；跑完自动生成 `community-results/<TAG>/summary.json`，提 PR 即完成回传。**认领表、环境步骤、耗时估算、常见坑见 [docs/08-multimodel-plan.md](docs/08-multimodel-plan.md)。**
