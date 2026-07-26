# K=2 与 K=4 的装载指标对比

论文主线是 **K=2 + 门控**。本目录回答一个直接的问题：换成 K=4，装载效果会
更好吗？答案是不会——**K=2 在三个臂上全部不劣于 K=4**。

支持集为七模型（Hy+Select 为五模型）× 四个规则域，held-out。

## 先看一个会导致错误结论的陷阱

K=4 的 Always 臂**并非恒为 100% 加载**：

| 模型 | 域 | 空加载比例 |
|---|---|---:|
| yi15-9b | medcalcbench | **38.0%** |
| yi15-9b | logicbench | **36.8%** |
| glm4-9b | logicbench | 3.6% |

七模型宏平均下，K=4 Always 的加载率只有 **92.13%**，而 K=2 是 100%。

后果是 **`loaded_skill_precision` 在两者间不可比**：K=4 的分母只统计了成功
检索到候选的那部分实例，等于自动剔除了检索失败的困难样本，把精度抬高了。
按这个指标会得出"K=4 Always 更好（55.37% vs 54.16%）"的错误结论。

**可比的指标是 `gold_load_rate`**，它的分母是全部实例，不受空加载影响。

## 结果（held-out，`gold_load_rate`）

| 臂 | K=4 | K=2 | 差 |
|---|---:|---:|---:|
| Always（不加门控） | 51.65% | **54.16%** | **+2.51 pp** |
| Gated（加门控） | 49.42% | **51.61%** | **+2.19 pp** |
| Hy+Select | 62.00% | **62.28%** | +0.28 pp |

同时 K=2 每题生成 token 为 1,385，K=4 为 2,747（省 49.6%，见
[K 消融分析](../../docs/10-k-ablation-analysis.md)）。

### 逐模型差异很大

K=2 的优势集中在两个弱模型上。yi15-9b 的 Gated 金标装载率从 40.49% 升到
**53.07%**（+12.6 pp），deepseek7b 从 23.40% 升到 **27.15%**。llama31-8b 和
mistral7b 上 K=4 反而微弱领先（0.5 pp 量级）。逐模型数据见
`loading_by_model.jsonl.gz`。

## 门控在 K=2 上补回了检索的损失

K=2 只想象两次，检索本身弱于 K=4——这符合直觉。但门控把这个劣势补上了：

- 不加门控时，K=2 领先 2.51 pp
- 加门控后，两者的**载入技能精度**从 K=4 领先转为 K=2 领先（73.10% → 73.64%）

yi15-9b 是最清楚的例子：K=4 下检索有近四成空结果，门控随之把加载率压到
51.83%；K=2 下检索稳定，门控加载率回到 66.49%，金标装载率因此高出 12.6 pp。

这与 HySkill 的核心主张一致——门控的价值在于吸收检索的不确定性。

## 文件

| 文件 | 内容 |
|---|---|
| `loading_by_model.jsonl.gz` | 19 行：每个 (模型, 臂) 的 K=4 与 K=2 三项装载指标与原始计数 |
| `summary.json` | 三个臂的跨模型宏平均 |

## 口径说明

- K=2 的 held-out 由 `loading_per_instance.jsonl.gz` 的 `is_validation` 直接给出。
- **K=4 的 `loading_per_instance` 没有该字段**，held-out 由同包
  `gating_per_instance.jsonl.gz` 的 `in_calibration_split` 按
  `(domain, instance_id)` 关联得到。两个文件同源同批次，但这一步是重建而非
  原始记录，引用时应注明。
- 三项指标定义：`loaded_skill_precision = gold_loaded / loaded`，
  `loading_rate = loaded / instances`，`gold_load_rate = gold_loaded / instances`。
  三者是不同的量，不可互相替代，也都不等于答题准确率。
