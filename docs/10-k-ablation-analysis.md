# 七模型生成采样数 K 消融分析

> 状态：完成并验收（2026-07-23）
>
> 主指标：routed test nDCG@10
>
> 对照：同模型、同域、同实例的 `K_img=4`
>
> 机器可读来源：[`community-results/k-ablation-fleet/`](../community-results/k-ablation-fleet/)

## 1. 结论摘要

`K_img=2` 是当前七模型上的最佳效果--成本折中：routed test nDCG@10
宏平均为 54.43%，比 `K=4` 的 54.17% 高 0.26 个百分点，同时每题估算
生成 token 从 2,747 降至 1,385，减少约 49.6%。模型--领域层级 bootstrap
的 95% CI 为 [-0.52, 1.25] 个百分点，包含 0；因此结论是“未检测到相对
`K=4` 的下降”，不是形式化等价或严格无损。

`K=8/10` 分别消耗约 `K=4` 的 1.99/2.47 倍 token，却没有总体收益。
主实验可以继续保留 `K=4` 作为固定 reference，同时把 `K=2` 报告为推荐的
效率设置。现有证据支持“性能在 `K=2--4` 进入平台区”，不支持“样本越多越好”
或“`K=4` 对所有模型都最优”。

## 2. 协议与支持集

| 维度 | 设置 |
|---|---|
| 模型 | Qwen3.5-4B、Qwen3.5-9B、GLM-4-9B、Llama-3.1-8B、DeepSeek-7B、Yi-1.5-9B、Mistral-7B |
| 领域 | theoremqa、logicbench、medcalcbench、champ、bigcodebench |
| `K_img` | 1、2、4、8、10 |
| fixed 变体 | `naive_sentence`、`naive_passage`、`naive_skill`、`hyskill`、`two_stage` |
| 自适应变体 | `routed`：每个模型、K、领域在固定 20% validation split 上重新选 fixed 变体 |
| 主评估 | 剩余 80% test split；routed nDCG@10 |
| 生成协议 | temperature 0.7；三模板；K 使用同一 K=10 序列的嵌套前缀 |
| 检索协议 | all-MiniLM-L6-v2；top-50；validation seed 0 |

本实验只回答检索阶段的 K 敏感性，没有重跑 Track B 做题阶段，因此不能把
nDCG 变化写成端到端回答准确率变化。

## 3. 数据完整性与复现身份

| 检查项 | 验证结果 |
|---|---|
| 正式结果矩阵 | 7 模型 × 5 K × 5 域 ×（5 fixed + 1 routed）= 1,050/1,050 |
| 逐模型结果 | 每模型 125 fixed + 25 routed = 150；失败标记 0 |
| 逐模型分析包 | 7 × 5 = 35 个机器可读文件；服务器与本地 SHA-256 完全一致 |
| 指标记录 | 每模型 5,040 行；七模型共同支持汇总 5,040 行 |
| 想象前缀包 | 35 个 gzip + 35 个 manifest；`rows=3970`、`unique_queries=3968`、哈希与字节数复验通过 |
| repository commit | `b642012bd86ca098f828a0c06f6f926375d74b7f` |
| runner bundle | `efc7d610d25b29e28fdb14ab58da099e3e7b2b0b0b7fb55d2f7817f8306cf69f` |
| fleet manifest | `62f97a5311f2987bdbefadae955f1e48778823f7d90094ef36bbbe26d5034eb1` |
| bootstrap | 10,000 次，seed 0 |

公开仓库保留规范化逐题指标、汇总、配对检验、成本和 manifest。服务器端
1,050 个原始 top-50 检索 JSON、缓存散文件和日志不重复上传。

## 4. 主效果--成本结果

下表使用七模型、五领域等权的 routed test nDCG@10 宏平均。区间为
模型--领域层级 bootstrap 的 \(K-K4\) 95% CI。token 来自真实 prompt 和
缓存输出文本的字符数，并按 3.8 characters/token 估算；它不是模型 tokenizer
精确计数或 API 计费读数。

| \(K_{\mathrm{img}}\) | routed nDCG@10 | 相对 K=4（百分点） | 95% CI（百分点） | 每题 token 估算 | 相对 K=4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.00 | -0.17 | [-1.65, 2.00] | 702 | 25.6% |
| 2 | **54.43** | +0.26 | [-0.52, 1.25] | 1,385 | 50.4% |
| 4 | 54.17 | 0.00 | reference | 2,747 | 100.0% |
| 8 | 54.11 | -0.06 | [-0.79, 0.82] | 5,479 | 199.4% |
| 10 | 53.99 | -0.17 | [-0.89, 0.46] | 6,784 | 246.9% |

micro 平均给出相同排序：`K={1,2,4,8,10}` 分别为
58.93%、59.27%、59.10%、58.83%、58.78%。主结论不依赖 macro 或 micro
口径的选择。

## 5. 模型异质性

| 模型 | K=1 | K=2 | K=4 | K=8 | K=10 | 点估计最佳 K |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-7B | **42.77** | 40.31 | 37.70 | 36.04 | 35.86 | 1 |
| GLM-4-9B | 52.87 | 53.92 | **54.48** | 54.09 | 53.92 | 4 |
| Llama-3.1-8B | 54.76 | 55.49 | 55.62 | 55.74 | **56.03** | 10 |
| Mistral-7B | 52.71 | 53.53 | **53.74** | 53.40 | 53.32 | 4 |
| Qwen3.5-4B | 60.70 | 61.04 | 60.45 | **61.73** | 60.83 | 8 |
| Qwen3.5-9B | 60.56 | **62.13** | 61.59 | 61.81 | 61.89 | 2 |
| Yi-1.5-9B | 53.61 | 54.57 | 55.59 | 55.97 | **56.10** | 10 |

经验最佳 K 覆盖 1、2、4、8、10，说明 K 效应高度依赖生成模型。最明显的
反例是 DeepSeek-7B：五个 fixed 变体均随 K 增大明显下降，routed 从 K=1
的 42.77% 降到 K=10 的 35.86%。这说明额外样本可能引入噪声或稀释表示；
现有结果能证明退化模式，但未直接测量每条想象的语义质量，不能把机制写成
已被因果验证。

表中“最佳 K”来自同一组测试点的事后最大值，适合说明异质性，不应被当作
逐模型调参后的无偏主结果。逐模型配对区间保留在各自的
`paired_vs_k4.json`。

## 6. 领域异质性

| 领域 | K=1 | K=2 | K=4 | K=8 | K=10 |
|---|---:|---:|---:|---:|---:|
| theoremqa | **77.26** | 76.75 | 76.59 | 76.26 | 76.32 |
| logicbench | **19.34** | 18.24 | 17.85 | 17.21 | 17.08 |
| medcalcbench | 88.43 | 90.96 | **91.21** | 90.91 | 90.82 |
| champ | 35.54 | 37.28 | 36.62 | **37.87** | 37.38 |
| bigcodebench | **49.43** | 48.92 | 48.55 | 48.31 | 48.36 |

`K=1` 在 LogicBench 相对 `K=4` 提高 1.49 个百分点，95% CI
[0.74, 2.18]；但在 MedCalcBench 降低 2.78 个百分点，95% CI
[-4.00, -1.67]。所以 `K=1` 虽然成本最低，却不是稳健的统一替代。

`K=8/10` 在 LogicBench 分别比 `K=4` 低 0.64/0.77 个百分点，区间均不含
0；其他领域没有给出足以抵消额外成本的稳定收益。领域比较数量较多且未做
family-wise 多重检验校正，宜把单域结果作为异质性证据，不扩写成普遍定律。

## 7. 固定分支与路由适应

| 变体 | K=1 | K=2 | K=4 | K=8 | K=10 |
|---|---:|---:|---:|---:|---:|
| `naive_sentence` | **45.80** | 44.67 | 43.04 | 41.90 | 41.65 |
| `naive_passage` | **50.06** | 49.97 | 49.78 | 49.58 | 49.60 |
| `naive_skill` | **49.49** | 48.91 | 47.96 | 47.30 | 47.11 |
| `hyskill` | **49.96** | 49.74 | 49.06 | 48.24 | 48.11 |
| `two_stage` | **50.95** | 50.26 | 49.26 | 48.47 | 48.28 |
| `routed` | 54.00 | **54.43** | 54.17 | 54.11 | 53.99 |

五个 fixed 分支在 `K=8/10` 的点估计都低于 `K=4`。fixed `hyskill` 在
`K=10` 低 0.95 个百分点，跨模型 95% CI 为 [-1.51, -0.44]。因此
routed 曲线平坦不能解释成每个固定分支都对 K 不敏感。

路由选择计数揭示了补偿机制：

| K | `naive_sentence` | `naive_passage` | `naive_skill` | `hyskill` | `two_stage` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 11 | 2 | 20 | 2 |
| 2 | 2 | 12 | 1 | 17 | 3 |
| 4 | 2 | 15 | 0 | 15 | 3 |
| 8 | 1 | 16 | 0 | 13 | 5 |
| 10 | 2 | 16 | 0 | 13 | 4 |

K 增大时，路由器把选择从 `hyskill` 转向对 K 较平坦的
`naive_passage`。所以 routed 结果证明的是完整自适应系统能够吸收一部分 K
敏感性，而不是“更多想象不会破坏检索”。

## 8. 论文可用结论与边界

推荐正文结论：

> 在七模型严格共同支持上，`K_img=2` 将估算生成 token 减少约一半，且未检测到
> routed nDCG@10 相对 `K_img=4` 的下降；将 K 提高到 8 或 10 没有带来总体收益。

必须保留的边界：

- CI 包含 0 时写“未检测到差异/下降”，不写“等价”“严格无损”。
- `K=2` 是跨模型平均的效率推荐，不是每个模型各自的最优 K。
- `K=4` 可作为与既有主实验一致的保守 reference，但不是唯一饱和点。
- token 为真实文本驱动的估算；七模型 warmup 墙钟均为 `unavailable`，不得补造。
- 本轮只评估检索，不据此声称端到端回答准确率同步变化。

## 9. 数据与复现入口

- 数据结构与字段：
  [`community-results/k-ablation-fleet/README.md`](../community-results/k-ablation-fleet/README.md)
- fleet 汇总：
  [`summary.json`](../community-results/k-ablation-fleet/summary.json)
- 配对检验：
  [`paired_vs_k4.json`](../community-results/k-ablation-fleet/paired_vs_k4.json)
- 成本：
  [`cost.json`](../community-results/k-ablation-fleet/cost.json)
- 完整性：
  [`manifest.json`](../community-results/k-ablation-fleet/manifest.json)
- 执行计划与服务器验收：
  [`docs/superpowers/plans/2026-07-22-multimodel-k-ablation.md`](superpowers/plans/2026-07-22-multimodel-k-ablation.md)

复现 fleet 汇总：

```bash
PYTHONPATH="$PWD" python3 scripts/summarize_k_ablation_fleet.py \
  --community-root community-results \
  --output-dir community-results/k-ablation-fleet \
  --bootstrap-samples 10000 \
  --bootstrap-seed 0
```
