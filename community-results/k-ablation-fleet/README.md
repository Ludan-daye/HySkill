# 七模型 K 消融 fleet 数据包

本目录是 `K_img ∈ {1,2,4,8,10}` 检索消融的严格共同支持汇总。逐模型数据位于
`community-results/<model-tag>/k-ablation/`；正式分析见
[docs/10-k-ablation-analysis.md](../../docs/10-k-ablation-analysis.md)。

## 目录关系

```text
community-results/
├── <model-tag>/
│   ├── imagination_full_k{1,2,4,8,10}.jsonl.gz
│   ├── imagination_full_k{1,2,4,8,10}.manifest.json
│   └── k-ablation/
│       ├── metrics_long.jsonl.gz
│       ├── summary.json
│       ├── paired_vs_k4.json
│       ├── cost.json
│       └── manifest.json
└── k-ablation-fleet/
    ├── metrics_long.jsonl.gz
    ├── summary.json
    ├── paired_vs_k4.json
    ├── cost.json
    ├── manifest.json
    └── README.md
```

## 文件说明

| 文件 | 作用 | 关键约束 |
|---|---|---|
| `metrics_long.jsonl.gz` | 七模型共同支持的规范化长表 | 5,040 行；不以零填补缺失 |
| `summary.json` | 逐域及跨域的 K 曲线 | 7 模型、5 K、5 域、6 变体、3 split |
| `paired_vs_k4.json` | fleet 层配对对比 | 10,000 次 bootstrap，seed 0 |
| `cost.json` | 效果--成本配套数据 | 真实文本字符数 / 3.8 的 token 估算 |
| `manifest.json` | 来源和完整性 | 逐模型 manifest、runner 和四个输出哈希 |

墙钟字段对七个模型均为 `unavailable`，不能从 token 估算反推或补造。

## 复现汇总

在仓库根目录运行：

```bash
PYTHONPATH="$PWD" python3 scripts/summarize_k_ablation_fleet.py \
  --community-root community-results \
  --output-dir community-results/k-ablation-fleet \
  --bootstrap-samples 10000 \
  --bootstrap-seed 0
```

汇总器会先验证七个逐模型 manifest、150 文件计数、共同指标支持、代码身份及
输出哈希，再写入 fleet 文件。当前机器可读 manifest 的 SHA-256 为
`62f97a5311f2987bdbefadae955f1e48778823f7d90094ef36bbbe26d5034eb1`。
