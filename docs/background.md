# 研究背景

> 调研日期：2026-07-13。所有论文均经原文/摘要核实，非仅凭搜索摘要转述。

## 1. 现实背景：GPT-5.6 与"复杂技能反而伤性能"的社区感知

- GPT-5.6 于 2026 年 6 月下旬发布（[TechTimes](https://www.techtimes.com/articles/318799/20260621/gpt-56-launch-window-starts-monday-alignment-fix-15m-token-context-inside.htm)），主打对齐修复（alignment fix）与 1.5M token 上下文。
- 社区有体验下降反馈：[openai/codex Issue #32161](https://github.com/openai/codex/issues/32161)（2026-07-10）报告 GPT-5.6 在 Codex 中丢失已给信息、小任务触发大范围仓库探索、token 消耗上升。**仅一名报告者，自述"体验性反馈而非受控基准"，OpenAI 未回应。**
- 此前 GPT-5.5 也有[官方承认的性能退化事故](https://status.openai.com/incidents/01KRP6FM6HSKWB3MS1EJTY6AT4)。"用户感知的退化"反复出现，但从未被受控实验验证。

## 2. 技能导致性能退化：已被证实的现象

### 2.1 More Skills, Worse Agents?（arXiv [2605.24050](https://arxiv.org/abs/2605.24050)，2026-05）

- 技能库从小规模扩到 202 个技能，性能最多下降 **21%**（SkillsBench，88 任务；Claude Haiku 4.5 / Sonnet 4.6；2,545 条轨迹）。
- 方法：先经验性定义 oracle 技能（单独配给任务、相对无技能基线通过率提升 ≥ 4%），再把轨迹按调用模式分三类事件（纯 oracle / 混入干扰 / 未用技能），条件分解 Δ = Δ_ctx + Δ_shd。
- **结论：选择错误（skill shadowing）是主要瓶颈，上下文变长（Δ_ctx）影响可忽略。**
- 弱点（对我们有用）：oracle 定义忽略组合效应、τ=4% 阈值随意、"混入即污染"过粗、有效样本仅 38 个 (任务,模型) 对、纯行为学推断无内部证据。

### 2.2 How Well Do Agentic Skills Work in the Wild（arXiv [2604.04323](https://arxiv.org/abs/2604.04323)，2026-04）

- 让智能体从 **34,000 个真实技能**中自行检索，Terminal-Bench 2.0，含 Claude Opus 4.6，结论跨模型一致。
- **场景越现实，技能收益衰减越厉害；最难场景下收益逼近零**（≈无技能基线）。
- "查询级技能精炼"（按当前任务改写检回的技能）可恢复大部分损失。

### 2.3 机制线索（均为行为学推断，无白盒证据，且互有矛盾）

| 假说 | 来源 | 要点 |
|---|---|---|
| 上下文长度副作用（context rot） | [Chroma 研究](https://www.trychroma.com/research/context-rot) | 输入越长性能越不可靠，即使任务简单；但与 2605.24050 的"Δ_ctx 可忽略"存在张力 |
| 注意力分散/技能竞争 | SkillsInjector（[2605.29794](https://arxiv.org/abs/2605.29794)） | 注入越多，注意力摊薄、相似技能互相竞争 |
| 过度服从 | When the Tool Decides（[2606.14476](https://arxiv.org/abs/2606.14476)）；MESA-S（[2604.16753](https://arxiv.org/abs/2604.16753)） | 越强的模型越盲从工具（Qwen2.5 1.5B→7B 一致率 0.60→0.98）；SRA 发现智能体不分需不需要都加载技能 |

## 3. 技能/工具加载方法版图（五条路线）

### 3.1 渐进式披露（Progressive Disclosure，业界默认范式）

启动时只预载技能名 + 一行描述，SKILL.md 正文触发时才加载。见 [Agent Skills 综述 2602.12430](https://arxiv.org/pdf/2602.12430)、[Externalization 综述 2604.08224](https://arxiv.org/html/2604.08224v1)。**正被 SkillRouter 的发现挑战**（只看元数据路由掉 31–44 个百分点）。

### 3.2 检索增强式加载

| 工作 | arXiv | 要点 |
|---|---|---|
| SRA / SRA-Bench | [2604.24594](https://arxiv.org/html/2604.24594v1) | 首个全流程基准（5,400 测例 / 636 金标技能 / 26,262 技能库）；发现**无差别加载**瓶颈：不管检回的对不对、任务需不需要，都以相近概率加载 |
| SkillRouter | [2603.22455](https://arxiv.org/html/2603.22455v1) | 1.2B 检索+重排路由器读**技能全文**，Hit@1 74%；藏正文只留元数据 → 路由准确率暴跌 31–44 pt；[代码](https://github.com/zhengyanzhao1997/SkillRouter) |
| Skill Is Not Document | [2606.03565](https://arxiv.org/pdf/2606.03565) | 技能检索是**能力匹配而非语义相似**；查询条件化两阶段检索器 + 基准 |
| SkillResolve-Bench | [2606.10388](https://arxiv.org/pdf/2606.10388) | 同能力歧义（多个功能几乎相同的技能怎么选） |
| SkillRet | [2605.05726](https://arxiv.org/pdf/2605.05726) | 大规模技能检索基准 + 专训嵌入模型（0.6B/8B） |
| SkillFlow | [2504.06188](https://arxiv.org/html/2504.06188v2) | 35,866 个真实 SKILL.md；四阶段流水线；查询生成为**显式搜索查询**（单/多查询，GPT-4o-mini），论文明确与 HyDE 划界，无 HyDE 基线；SkillsBench Pass@1 9.2%→16.4%（oracle 上限的 84.1%）；**oracle 技能代码块占比显著更高（39% vs 24%）**；Terminal-Bench 无显著增益（库中缺高质量可执行技能时检索无力回天） |
| Graph-of-Skills | [2604.05333](https://arxiv.org/html/2604.05333) | 类型化技能图上的反向感知 PPR 扩散；**不用任何伪文档生成** |
| Group of Skills | [2605.06978](https://arxiv.org/html/2605.06978v1) | 组结构化检索单元 |
| SkillDAG | [2606.03056](https://arxiv.org/abs/2606.03056) | 见第 5 节（对本项目最关键） |

### 3.3 组合式路由

SkillWeaver / Compositional Skill Routing（[2606.18051](https://arxiv.org/html/2606.18051)）：分解→检索→DAG 编排，2,209 个真实 MCP 技能；技能感知分解（SAD）把分解准确率 51.0%→67.7%；上下文占用比全量枚举减 99.9%。局限：Qwen 系为主、模板合成查询。

### 3.4 动态注入

SkillsInjector（[2605.29794](https://arxiv.org/pdf/2605.29794)）：上下文规划器按任务自适应决定注入数量；注入描述按共同注入邻居改写。Tau2-bench/SkillsBench/ALFWorld +3.9~7.3 pt。

### 3.5 编译式加载

SkillSmith（[2605.15215](https://arxiv.org/pdf/2605.15215)）：技能包离线编译为带操作边界的最小可执行接口，运行时按边界加载。SkillsBench 上 token −57%、迭代 −43%、提速 2×；强模型编译产物可供小模型使用。

## 4. 顶会/顶刊接收状态分层

### 4.1 已正式接收（peer-reviewed，引用主干）

| 论文 | Venue | 与本项目关系 |
|---|---|---|
| [HyDE](https://aclanthology.org/2023.acl-long.99/)（Gao, Ma, Lin, Callan） | **ACL 2023 主会长文** | 方法源头 |
| ToolLLM | ICLR 2024 | 大规模 API 工具学习 + 检索器 |
| [Re-Invoke](https://aclanthology.org/2024.findings-emnlp.270.pdf) | EMNLP 2024 Findings | **反向**假想：索引期给每个工具生成合成查询 |
| AvaTaR | NeurIPS 2024 | 工具使用提示自动优化 |
| [ToolGen](https://arxiv.org/abs/2410.03439) | ICLR 2025 | 生成式检索：工具=词表 token |
| [ToolDreamer](https://arxiv.org/abs/2510.19791) | **EACL 2026 口头** | **查询侧假想工具描述检索——最直接前驱，但仅 API 工具** |
| [Tool-DE](https://openreview.net/forum?id=g9D9MgG7iW) | ICLR 2026（网页显示接收，引用前需再核对 OpenReview 决定） | 库侧文档扩写 + Tool-Embed/Tool-Rank |
| RL for Self-Improving Agent with Skill Library | ACL 2026 主会 | 技能库 + RL 自提升 |
| The Confidence Dichotomy | ACL 2026 主会 | 工具使用智能体**置信度失准**——门控贡献可直接借其问题定义 |

### 4.2 预印本/在审（引用时标注 concurrent/preprint）

SkillRouter（标注 under review）、SkillFlow（OpenReview 在审）、SkillsInjector、SkillSmith、SRA-Bench、Skill Is Not Document、SkillRet、SkillResolve-Bench、Graph-of-Skills、Group of Skills、SkillDAG、SkillWeaver、[SING](https://arxiv.org/pdf/2606.16591)（合成意图图做工具发现——同带"合成生成"味道，相关工作需划界）。

**含义**：正式接收的加载方法全部停留在 API 工具层面；SKILL.md 式技能加载研究基本无一走完同行评审——本项目有机会成为最早一批 peer-reviewed 的技能加载方法。

## 5. HyDE 谱系与新颖性排查（核心结论）

### 5.1 HyDE 原始论文

Gao, Ma, Lin, Callan. *Precise Zero-Shot Dense Retrieval without Relevance Labels*. **ACL 2023**（[2023.acl-long.99](https://aclanthology.org/2023.acl-long.99/)，[arXiv:2212.10496](https://arxiv.org/abs/2212.10496)，[代码](https://github.com/texttron/hyde)）。
方法：查询 → 指令模型零样本生成"假想文档"（允许事实编造）→ Contriever 嵌入 → 按向量相似度检索真文档。洞察：假文档虽失真，但落在正确答案文档的嵌入邻域内；"文档↔文档"匹配远易于"查询↔文档"。

### 5.2 方向区分（本项目定位的关键）

- **查询侧（HyDE 本尊）**：推理时，任务查询 → 生成假想文档 → 检索。在线、任务条件化。
- **库侧（Doc2Query/Re-Invoke 方向）**：索引时，给库中每个条目生成假想查询/需求。离线、逐条目、不感知当前任务。

### 5.3 逐篇排查结果（2026-07-13 核实）

| 工作 | 查询侧假想生成？ | 核实方式 |
|---|---|---|
| SkillDAG（2606.03056，v2 2026-07-02） | ❌ **仅库侧**：索引期对每技能生成 e_needs（"LLM imagines a few invoking tasks and summarizes the shared prerequisites in one sentence (HyDE-style [gao2023hyde])"）；**推理时查询原样嵌入**（"top-K skills by query–node embedding cosine"）；已引用 HyDE | 全文核对 |
| SkillFlow（2504.06188） | ❌ 生成显式搜索查询，论文自己与 HyDE 划界 | 全文核对，基线无 HyDE |
| SRA-Bench（2604.24594） | ❌ 基线仅 BM25/TF-IDF/BGE/Contriever/混合/LLM 重排 | 基线列表核对 |
| SkillRet（2605.05726） | ❌ 稠密/重排/LLM 基线 | 基线列表核对 |
| Graph-of-Skills（2604.05333） | ❌ 明确不用伪文档生成 | 方法核对 |
| SkillRouter（2603.22455） | ❌ 全文重排，无生成 | 方法核对 |
| Skill Is Not Document（2606.03565） | ❌ 两阶段检索器，无假想生成 | 方法核对 |
| ToolDreamer（2510.19791，EACL 2026） | ✅ 查询侧——**但对象是 API 工具，非技能** | 全文核对 |

### 5.4 结论

> **"任务查询 → 生成假想技能文档 → 检索真技能"（查询侧 HyDE on skills）截至 2026-07-13 无人做过。**
> 最近邻：ToolDreamer（查询侧、API 工具）与 SkillDAG（技能、库侧）。二者分别验证了"查询侧假想生成有效"与"假想文本在技能嵌入空间有效"两个前提，且均不占用本项目的组合位置。
> 时效警示：SkillDAG 已把 HyDE 引入技能文献，v2 更新于 2026-07-02；技能检索赛道 3 个月约 10 篇论文。**窗口真实但正在收窄。**
