# qwen3.5-4b 主实验参考包

**模型**：Qwen/Qwen3.5-4B（全部 Phase 0/1/2/2c 主实验的生成器与执行器）｜ 数据覆盖：Phase 1 全量 9 方法检索网格 + Phase 2 五臂 + Phase 2c 路由臂（含 select、oracle）。这是跨模型对比的**基准列**，也是数据最全的一份（唯一有 7 臂做题与双门信号的包）。

## 文件清单（全部已入库）

| 文件 | 规模 | 内容 |
|---|---|---|
| `retrieval_top10.jsonl.gz` | 38,560 行 | （实例 × 方法）：bm25/dense/hybrid/llm_rerank/三粒度想象/四路/两阶段 **9 方法** + routed，金标 + top-10 + 逐题 nDCG@10 |
| `gating_per_instance.jsonl.gz` | 2,830 行 | **双门信号**（固定门与路由门各一套 S1/S2/τ/拦截/标定集标记）+ **7 臂逐题对错**（bare/always/gated/select/oracle/always_r/gated_r） |
| `imagination_samples.jsonl.gz` | 50 行 | 与舰队包同一批 50 题（seed 0）：3 模板 × K=4 想象全文 + 各方法 top-3（名称/简介/分数/命中） |
| `router_decisions.json` | 4 域 | 路由决策与验证集全变体比分 |
| `metrics_flat.jsonl.gz` | 448 行 | 全部（域 × 方法 × 指标）数字拍平：Recall@1/5/10/50、nDCG@k、各臂 accuracy/n |
| `significance.json` | — | Phase 2 全套配对 bootstrap（分域 + 合并 + 剔标定集净测试版） |
| `MANIFEST.md` | — | 自动清单 + pandas 读取示例 |

## 与舰队包的对齐

- imagination_samples 的 50 题与所有舰队模型完全相同（seed 0），可逐题横向比"不同模型想象了什么、检回了什么"；
- 列名与舰队包一致，`pd.concat` 前加一列 `model` 即可合并分析。

原始超大件（top-50 全榜、做题 jsonl、6.3 万份想象缓存、Phase 0 试点存档）在实验室服务器 `results/phase1|phase2|hyp_cache`。
