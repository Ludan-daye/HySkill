# K=2 门控重标定（v2）

> 本目录是 **`<TAG>/k2/` 的补充，不是替代**。`k2/` 包保持 `aa5020c` 发布时的状态不变。

## 这是什么

关闭 K2M001（baseline runtime identity gate）需要用与正式 K=2 相同的
checkpoint / tokenizer / chat template / vLLM / BF16 / 8K context 重跑 Bare。

而 `scripts/gate.py` 里 `tau2` 的正标签恰好是「**Bare 在这道题上是否答对**」：

```
tau2 = 最大的 t，使得 precision(Bare 答对 | S2 < t) >= 0.9
```

所以重跑 Bare 会改变标定标签，进而改变 `tau2`，最终让部分实例的
「加载 / 不加载」决策翻转。本目录记录这条依赖链的完整后果。

`tau1` 只依赖检索标签，与 Bare 无关，**9 个 job 全部未变**。

## 结果

32 个 gate（28 routed + 4 Qwen fixed）全部重标定，`valid=true`。其中：

- **9 个 job 的 `tau2` 改变**，
- **609 条决策翻转**并已重新推理，
- 22,031 / 22,640 行保留原答案（payload 未变），40 条既有 method failure 原样保留。

变化最集中的是 Qwen3.5-4B 的 medcalcbench：`tau2` 从 `null` 变为非空，
一个域就贡献 463 / 609（76%）。这正是设计文档预判的情形——
「即使旧 `tau2=null`，新 Bare 标签也可能产生非空阈值」。

### 对答案准确率的影响

受影响 8 个单元（held-out）：**67.71% → 67.32%**（−0.39pp，净 −17 题）。

### 对加载指标的影响（held-out，模型 macro）

| 支持集 / 臂 | 载入技能精度 | 加载率 | 金标加载率 |
|---|---:|---:|---:|
| 七模型 Gated | 73.64% → **73.19%** | 69.43% → **68.01%** | 51.61% → **50.17%** |
| 五模型 Gated | 76.43% → **75.82%** | 73.76% → **71.68%** | 56.21% → **54.15%** |
| Always / Hy+Select | 不变 | 不变 | 不变 |

Always 与 Hy+Select 一个数都没变——它们不经过门控，这是正确性自检。
新 `tau2` 让门控更保守：加载率下降 1.4–2.1pp。

### 对四条主对比的影响

分层 bootstrap，10,000 次重采样，seed 0，排除冻结校准 ID：

| 对比 | 旧 | 新 |
|---|---|---|
| Gated vs Always（七模型） | +1.38pp, p=0.2300 | **+1.52pp, p=0.1860** |
| Gated vs Hy+Select（五模型） | −0.80pp, p=0.5540 | −0.69pp, p=0.6048 |
| Hy+Select vs Always（五模型） | +1.76pp, p=0.3050 | 完全不变 |
| Qwen4 routed vs fixed Gated | −0.13pp, p=0.8808 | **+0.18pp**, p=0.8548 |

**四条 CI 依旧全部包含零，显著性判定没有任何变化。**

最后一条的符号发生了翻转（−0.13 → +0.18），但两侧 p 值都在 0.85 以上，
属于「未检测到差异」内部的噪声漂移。**不得**据此声称 routed 反超 fixed。

## 文件

| 文件 | 内容 |
|---|---|
| `taus_recalibrated.json` | 9 个 job 的新旧 `tau1`/`tau2`、决策计数、变化数 |
| `changed_decisions.jsonl.gz` | 609 条翻转决策，含 `answer_payload_hash` |
| `rerun_answers.jsonl.gz` | 609 条重新推理的答案 |
| `loading_metrics_long.jsonl.gz` | 重算的 210 行加载指标长表 |
| `answer_metrics_long.jsonl.gz` | 重算的 210 行答案指标长表 |
| `loading_summary.json` / `answer_summary.json` | 重算汇总 |
| `paired_comparisons.json` | 重算的四条配对对比 |
| `manifest.json` | 上述文件的 SHA-256 与行数 |

## 为什么不直接覆盖 `k2/`

三条理由：

1. `hyskill/k2_answer_provenance.py` 硬编码了每个模型的 provenance 期望计数
   （`formal_direct` / `posthoc_structural` / `formal_retry_after_import`），
   这是**刻意的冻结设计**。门控重标定产生的 609 条属于新的来源类别，
   塞进去就必须改动这些冻结常量与导出器白名单。
2. 这 609 条不是对旧数据的**修正**，而是一个**新的实验事实**
   （Fresh Bare → 新 `tau2` → 新决策）。旧答案在旧 `tau2` 下依然是有效记录。
3. 仓库既定原则是不覆盖已发布数据。`k2/` 包已在 `aa5020c` 推送并逐字节读回。

## 论文如何引用

**主引用仍是 `<TAG>/k2/`。** 本目录是对那些数字的**稳健性检验**，不是替代：
门控重标定使七模型 Gated 全局移动 −0.13pp，四条主对比的显著性判定
一个都没变。既然结论不受影响，就没有理由让论文里出现两个版本的 Gated。

这样也让 baseline 对比表保持自洽——那张表的 Gated 列取自 `k2/`，全文统一
引用 `k2/` 就不会在同一篇论文里混用两代 Gated 数字。

本目录的价值在于回答一个必然会被问到的问题：*用 runtime-matched 的 Bare
重新标定门控，结论会变吗？* 答案是不会，而这里有完整的逐题证据。

## 复核入口

`taus_recalibrated.json` 的 `old/new_decision_counts` 与
`changed_decisions.jsonl.gz` 的行数必须一致（609）。
`rerun_answers.jsonl.gz` 行数应等于各 job `rerun_required_count` 之和（609）。
加载指标可由 `k2/` 的 `loading_per_instance.jsonl.gz` 叠加本目录决策独立重算。
