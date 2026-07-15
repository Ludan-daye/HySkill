# qwen3.5-4b 主实验参考包（自动生成）

主实验（Phase 1 全量 9 方法 + Phase 2 五臂 + Phase 2c 路由臂）的逐题级数据。

| 文件 | 规模 | 内容 |
|---|---|---|
| retrieval_top10.jsonl.gz | 38560 行 | （实例 × 方法）×{9 方法 + routed}：金标、top-10、逐题 nDCG@10 |
| gating_per_instance.jsonl.gz | 2830 行 | 固定门+路由门双份 S1/S2/τ/拦截决定/标定集标记 + **7 臂逐题对错**（bare/always/gated/select/oracle/always_r/gated_r） |
| imagination_samples.jsonl.gz | 50 行 | 与舰队包同一批 50 题：3 模板 × K=4 想象全文 + 各方法 top-3 |
| router_decisions.json | 4 域 | 路由决策与验证集比分 |
| metrics_flat.jsonl.gz | 448 行 | 全部（域 × 方法 × 指标）数字拍平 |
| significance.json | — | Phase 2 全部配对 bootstrap（分域 + 合并 + 剔标定集版） |

pandas: `pd.read_json("<file>.jsonl.gz", lines=True)`
