# mistral7b 分析数据包清单

| 文件 | 行数 | 内容 |
|---|---|---|
| retrieval_top10.jsonl.gz | 27790 | 每行=（实例×变体）：金标、top-10（id/分数/是否金标）、逐题 nDCG@10。变体含 5 想象变体 + routed+llm_rerank |
| retrieval_top50.jsonl.gz | 31760 | 每行=（实例×方法）：金标及完整 top-50 `[skill_id, score, is_gold]` 三元组 |
| gating_per_instance.jsonl.gz | 2830 | 每行=实例：S1/S2、top1、检索是否错、τ、门控是否拦截、是否标定集、六个协议臂 + oracle 的逐题对错 |
| loading_per_instance.jsonl.gz | 16980 | 每行=（实例×六个协议臂）：实际装载技能 id、金标及是否命中；bare/门控拦截为空 |
| imagination_samples.jsonl.gz | 50 | 每域固定 10 题（seed 0，跨模型同题可比）：查询原文、**三种模板 × K=4 份想象全文**、每个变体（含 routed/rerank）的 top-3（名称/简介/分数/命中）、金标 |
| router_decisions.json | 5 域 | 路由决策账：每域选中的变体 + 全部变体的验证集 nDCG 比分 + 切分参数 |
| metrics_flat.jsonl.gz | 376 | **全部分数拍平**：（域 × 方法 × 指标）一行一个数——检索 Recall@1/5/10/50、nDCG@k 全量 + 各臂 accuracy/n |
| summary.json | — | 聚合指标（检索/路由/门控/成本审计） |

## 用 pandas 读取

```python
import pandas as pd
top10 = pd.read_json("retrieval_top10.jsonl.gz", lines=True)
top50 = pd.read_json("retrieval_top50.jsonl.gz", lines=True)
gate  = pd.read_json("gating_per_instance.jsonl.gz", lines=True)
load  = pd.read_json("loading_per_instance.jsonl.gz", lines=True)
imag  = pd.read_json("imagination_samples.jsonl.gz", lines=True)
```

未压缩的原始件（top-50 检索 JSON、做题 JSONL、日志、完整想象缓存）留存跑批服务器 `results/multimodel/mistral7b/`；本目录只提交上表所列的规范化分析包，见 README。
