# qwen35-9b 分析数据包清单（自动生成）

| 文件 | 行数 | 内容 |
|---|---|---|
| retrieval_top10.jsonl.gz | 26467 | 每行=（实例×变体）：金标、top-10（id/分数/是否金标）、逐题 nDCG@10。变体含 5 想象变体 + routed+llm_rerank |
| gating_per_instance.jsonl.gz | 2830 | 每行=实例：S1/S2、top1、检索是否错、τ、门控是否拦截、是否标定集、各臂对错（bare/always/gated） |
| imagination_samples.jsonl.gz | 50 | 每域固定 10 题（seed 0，跨模型同题可比）：查询原文、K=4 份想象 SKILL.md 全文、naive_skill top-3（含名称/简介/分数/命中）、金标 |
| summary.json | — | 聚合指标（检索/路由/门控/成本审计） |

## 用 pandas 读取

```python
import pandas as pd
top10 = pd.read_json("retrieval_top10.jsonl.gz", lines=True)
gate  = pd.read_json("gating_per_instance.jsonl.gz", lines=True)
imag  = pd.read_json("imagination_samples.jsonl.gz", lines=True)
```

全量原始件（top-50 榜单、做题 jsonl、日志、完整想象缓存）留存跑批服务器 `results/multimodel/qwen35-9b/`，见本目录 README。
