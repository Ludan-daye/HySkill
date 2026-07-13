# HyDE 原文方法精读：相似度比较机制

> 来源：Gao, Ma, Lin, Callan. *Precise Zero-Shot Dense Retrieval without Relevance Labels*. ACL 2023 主会长文（[2023.acl-long.99](https://aclanthology.org/2023.acl-long.99/)）。本笔记逐页核对正式版 PDF 整理（2026-07-13），公式编号与原文一致。

## 1. 问题设定

传统稠密检索的相似度是"查询↔文档"内积：

```
sim(q, d) = ⟨enc_q(q), enc_d(d)⟩        (公式1)
```

零样本困境：没有相关性标注，学不出让内积恰好编码相关性的两个编码器。

## 2. 关键转身：只算"文档↔文档"相似度

原文："query-document similarity scores are no longer explicitly modeled or computed."

检索拆成两个免标注任务：

- **生成任务**：`g(q, INST) = InstructLM(q, INST)`（公式4），INST = "write a paragraph that answers the question"（各数据集模板见原文附录 A.1）。生成的文档 *is not real*，允许幻觉，只需捕捉"相关性模式"；
- **文档-文档相似度**：无监督对比编码器（Contriever）本就以文档-文档相似度为训练目标。`f = enc_d = enc_con`（公式2），全库离线编码 `v_d = f(d)`（公式3）。

## 3. 相似度计算公式链

```
E[v_q] = E[ f(g(q, INST)) ]                          (公式5)  查询向量 := 假想文档嵌入的期望
v̂_q = (1/N) Σ_{k=1..N} f(d̂_k)                       (公式6-7) 采样 N 份取平均（单峰假设）
v̂_q = 1/(N+1) [ Σ_{k=1..N} f(d̂_k) + f(q) ]          (公式8)  查询自身作为"第 N+1 份假设"混入
sim(q, d) = ⟨v̂_q, v_d⟩   ∀d ∈ D                     (公式9)  对全库内积最近邻（MIPS）
```

要点：
- 公式 8 中查询用**文档编码器 f** 编码——查询被降格为一份假设。这是 HySkill 三路融合中"查询锚"一路的出处；
- 工具链 Pyserini；温度 0.7；官方实现默认采样 8 份假想文档。

## 4. 幻觉为何无害：两道保险

1. **稠密瓶颈 = 有损压缩器**（原文："the encoder function f serves as a lossy compressor... extra details are filtered and left out of the vector"）：编造的具体细节在定长向量压缩中丢失，保留主题/实体/话语结构等相关性模式；向量被真实语料"接地"；
2. **多份采样平均 + 查询锚**：抹掉单份生成跑偏的方差。

原文 Figure 3（t-SNE）：假想文档向量（红）落在相关文档簇（蓝）内部，原始查询向量（绿）距离较远——"pivot"的直观证明。

## 5. 实验配置与对照设计

- 生成器 InstructGPT（text-davinci-003）；编码器 Contriever / mContriever，**开箱即用零训练**；
- **对照设计精髓（预实验必须复制）**：HyDE 与 Contriever 基线共享同一语料索引与编码器，**唯一差异是查询向量构造方式**——增益可完全归因于假想生成。

## 6. 主结果与关键消融

| 数据集 | Contriever | HyDE | 参照 |
|---|---|---|---|
| TREC DL19 nDCG@10 | 44.5 | **61.3** | Contriever-ft（微调）62.1，打平 |
| TREC DL20 nDCG@10 | 42.1 | **57.9** | BM25 48.0 |
| DL19 Recall@1k | 74.6 | **88.0** | 全场最佳 |
| BEIR 7 个低资源集 | — | 全面提升 | 仅 TREC-COVID 输 BM25 0.2（59.3 vs 59.5） |
| Mr.TyDi（sw/ko/ja/bn） | — | 超 mContriever | 多语言成立 |

**关键消融**：
- **生成器规模（Table 4，DL19 nDCG@10）**：FLAN-T5-xxl 48.9 < Cohere-52B 53.8 < InstructGPT 61.3。生成器越大增益越大 → HySkill"小模型生成"卖点必须靠消融自证（赌 2026 小模型 ≫ 当年 FLAN-T5）；
- **微调编码器叠加（Table 6）**：GTR-XL 69.6 → +HyDE 71.9（DL19），HyDE 与强编码器不冲突，但增益缩小；
- **非指令模型（Table 5）**：3-shot GPT-3 base 不稳定——生成器需要指令跟随能力。

## 7. 逐组件映射到 HySkill

| HyDE 组件 | HySkill 对应 | 改动理由 |
|---|---|---|
| "write a paragraph…" 指令 | "为该任务写一份 SKILL.md" | 技能是结构化程序性文档；生成含名称/描述/步骤/代码骨架（SkillFlow：oracle 技能代码块 39% vs 24%） |
| 单向量平均（公式8） | 分字段质心 + 查询锚 + BM25 三路 RRF | 技能异构，单向量糊掉结构信号 |
| f = Contriever | 现代嵌入模型（BGE 等），需消融 | HyDE 对编码器敏感 |
| 内积 MIPS | 字段对字段打分后融合 | 同上 |
| N 份采样平均 | k 份假想技能质心 | 直接继承 |
| 稠密瓶颈滤幻觉 | 可行性核心依赖，原样继承 | 假想技能中编错的 API 名被压缩掉，留下能力轮廓 |
| 生成器越大越好 | 小生成器成本卖点 = 风险点 | 预实验早测 |
