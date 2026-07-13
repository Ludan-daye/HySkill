# HySkill 设计文档：查询侧假想技能生成用于技能检索与加载门控

日期：2026-07-13。方法动机与文献依据见 `docs/background.md`、`docs/idea.md`、`docs/hyde-method.md`。

## 1. 目标

实现并验证 HySkill-RG：
1. **检索**：给定任务查询，生成 K 份假想 SKILL.md，分字段嵌入取质心，与真实技能库做字段对字段匹配，四路 RRF 融合；
2. **门控**：用覆盖信号 S1 与增益信号 S2 决定 加载 / 跳过 / 不加载；
3. **评测**：在 SRA-Bench 官方框架（[SR-Agents](https://github.com/oneal2000/SR-Agents)，MIT 协议）上按其三段协议对比基线。

非目标（本期不做）：多技能组合检索；技能精炼融合（留扩展）；SkillsBench/Terminal-Bench 端到端（Phase 3 再做）。

## 2. 系统架构

独立 Python 包 `hyskill`，通过 SR-Agents 的外部插件机制接入（`sragents --plugin hyskill.plugin`），**不 fork 不改动 SR-Agents 源码**。

```
hyskill/
  parser.py      # SKILL.md → {meta, body, code} 三字段（含缺字段处理）
  generator.py   # OpenAI 兼容端点调用，任务查询 → K 份假想 SKILL.md（温度 0.7）
  embedder.py    # sentence-transformers 封装（默认 BAAI/bge-base-en-v1.5，与 SR-Agents 的 bge 基线一致）
  fusion.py      # 每字段排名 + BM25 排名 → RRF（k=60）
  gating.py      # S1/S2 计算与三区间决策；阈值校准工具
  retriever.py   # HySkillRetriever：实现 SR-Agents Retriever 协议
                 #   build_index(ids, texts)：解析字段、建三个稠密索引 + BM25 索引
                 #   retrieve(queries, top_k)：生成→质心→查询锚→四路融合
  plugin.py      # @register("hyskill") / @register("naive_hyde") 注册入口
  cache.py       # 假想技能生成结果磁盘缓存（按 query 哈希；实验可复现、省 API 费）
  configs/       # 实验配置（YAML）：K、生成器、编码器、融合权重、消融开关
tests/
  fixtures/      # 10 个合成技能 + 5 个合成查询的微型语料
  test_parser.py test_fusion.py test_gating.py test_retriever_mock.py（mock LLM）
scripts/
  run_phase0.sh  # 六数据集 × {bm25, bge, hybrid, naive_hyde, hyskill} 的 sragents retrieve 批跑
  analyze.py     # 汇总 Recall@K / nDCG@10 表格
```

### 关键接口约束（来自 SR-Agents 实测）

- Retriever 协议：`build_index(corpus_ids: list[str], corpus_texts: list[str])` + `retrieve(queries, top_k) -> list[list[(skill_id, score)]]`；
- `corpus_texts` 由 SR-Agents 拼好传入。**注意**：需核对其 CLI 拼接方式（name+description+content 还是仅部分字段）；若拼接丢字段，用 `--corpus` 原始 JSON 在 `build_index` 前自行重载解析（retriever 构造参数传 corpus 路径）；
- 检索输出 JSON schema 固定（instance_id / gold_skill_ids / retrieved），下游 infer/evaluate 直接消费；
- 语料条目：`{skill_id, name, description, content}`；实例：`{instance_id, question, skill_annotations, eval_data}`。

## 3. 方法规格

### 3.1 生成（generator.py）

- 提示模板（v1，Phase 0 期间可调，模板存 configs 并进版本控制）：
  > You are writing a SKILL.md that an agent would use to solve the task below. Output: (1) frontmatter with `name` and a one-line `description`; (2) numbered procedure steps; (3) a minimal code skeleton in a fenced block. Be concise, ≤300 tokens. Factual precision is not required — capture what the right skill would look like. Task: {q}
- K 份采样，温度 0.7；默认 K=4（成本折中），消融 K∈{1,4,8}；
- 生成器经 OpenAI 兼容端点（支持 vLLM 本地 Qwen 与托管 API）；默认小模型，消融换档；
- 失败处理：单份生成失败重试 1 次；全部失败该查询退化为"纯查询检索"（等价 bge 基线）并在输出 metadata 里计数。

### 3.2 嵌入与融合（embedder.py / fusion.py）

- 假想技能解析成 meta/body/code，逐字段嵌入，K 份取均值质心；
- 查询锚：f(q) 以 1/(K+1) 权重混入 **meta 路**质心（HyDE 公式 8 的适配）；
- 库侧三索引 + BM25(query→全文) 共四路，各出排名，RRF：score = Σ 1/(60+rank)；
- 缺字段处理：技能缺 code 字段 → 该技能在 code 路不参与排名（RRF 缺项即不加分），不做惩罚。

### 3.3 门控（gating.py，Phase 2 启用）

- S1 = mean_F cos(ĥ^F, v_top1^F)（缺字段跳过取均值）；
- S2 = top-1 技能按句/步骤切分后，对假想技能句子集最大相似度 < τ_c 的单元占比；
- 决策：S1<τ1 → 不加载；S1≥τ1 且 S2≥τ2 → 加载；否则跳过；
- τ1/τ2/τ_c 在各数据集验证切分上网格校准（真值：金标是否在候选 / eval_data 正误 + 无技能基线正误）；
- 输出与 SR-Agents infer 阶段对接：门控结果转成 provider 参数（加载→topk k=1；不加载/跳过→无技能 direct）。

### 3.4 基线

Phase 0 全部走 sragents 原装 CLI：bm25 / bge / hybrid（原装）+ naive_hyde（我们注册：生成普通解答段落而非结构化技能，单向量，其余同 HyDE 原文）+ hyskill。对照设计：**hyskill / naive_hyde / bge 共用同一编码器**，增益可逐级归因（结构化生成的增量 vs 假想生成的增量）。

## 4. 实验计划

| Phase | 内容 | 产出/判据 |
|---|---|---|
| 0（本期） | 六数据集 × 5 检索器，Recall@{1,5,10,50}、nDCG@10 | hyskill 显著 > bge/hybrid/naive_hyde → go，挂 arXiv；hyskill≈naive_hyde → 结构化生成无效，回炉 |
| 1 | 消融：K、生成器规模、逐路去除、查询锚、单向量 vs 分字段、编码器 | 每个设计决策的数字支撑 |
| 2 | 门控 vs SR-Agents 三种原装暴露策略，relevance/need-aware 分离度 + 端到端 accuracy | 治好 SRA 诊断的"无差别加载" |
| 3 | SkillsBench(+Terminal-Bench) 端到端，Pass@1 / token / shadowing 率 | 完整论文故事 |

Phase 0 成本估算：约 5,400 查询 × K=4 × ~300 token 生成 ≈ 6.5M token（小模型，数十美元量级）；嵌入本地 GPU。生成缓存后消融只付增量。

## 5. 测试策略

- 单元测试全部离线（mock LLM、固定嵌入种子的微型语料）：parser 字段切分与缺字段、fusion 的 RRF 正确性（手工可验的小例）、gating 三区间边界、retriever 协议合规（返回排序、长度）；
- 集成冒烟测试：微型语料 + mock 生成器走通 build_index→retrieve→schema 校验；
- 真实小样：每数据集抽 20 实例先跑通三段 CLI，再放全量。

## 6. 风险与对策

- SR-Agents 的 corpus_texts 拼接不含结构分隔 → retriever 直读原始 corpus.json（构造参数）；
- 小生成器质量不足（HyDE Table 4 风险）→ Phase 0 同时跑一档中模型对照，早发现；
- ToolQA 需外部语料（Google Drive）→ Phase 0 先跑其余五个数据集，ToolQA 后补；
- BigCodeBench 评测需执行环境 → 检索层指标不受影响，端到端推迟到 Phase 2/3。
