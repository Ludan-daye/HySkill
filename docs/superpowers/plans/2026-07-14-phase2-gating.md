# Phase 2 加载门控实验计划

> 2026-07-14 制定。前置：Phase 1 检索结果 45/45（results/phase1/）与假想文档缓存（~6 万条）全部复用——**P2 不需要任何新检索**，成本集中在 Stage 3 做题推理。

## 命题

模型不会判断"该不该用检索来的技能"（SRA：有无金标加载率 21.8% vs 16.9%）。用假想技能免费计算 S1/S2 两信号替模型判断，将弱检索域的"塞错书伤害"转为中性，并省 token。

## 对照线（主表骨架）

| # | 线 | 实现 |
|---|---|---|
| ① | 裸做题 | sragents infer --engine direct（无技能） |
| ② | 无脑塞 top-1 | --provider topk k=1（源 = naive_skill 检索文件） |
| ③ | **S1/S2 门控** | --provider topk k=1（源 = gate.py 产出的门控后检索文件,被拦实例 retrieved 置空） |
| ④ | Oracle 塞金标 | --provider topk（源 = 金标构造的检索文件），上界 |
| ⑤ | 模型自主决定 | SRA 原装 LLM Selection 暴露策略——③ vs ⑤ = 外部门控 vs 模型自觉 |

## 门控信号（零 LLM 调用，全部来自缓存向量）

- S1 = cos(假想技能质心, top-1 技能向量)；< τ₁ → 不塞（库中无合适，防 shadowing）
- S2 = top-1 内容中假想覆盖不到的句子占比；< τ₂ → 跳过（参数化知识已足够）
- 检索底座：naive_skill（完整技能单向量，R@1 全场最稳）

## 校准与防泄漏

每域 20% 验证集网格搜 τ₁/τ₂，80% 测试集报告。真值：τ₁ ← 金标==top-1；τ₂ ← SRA skill-free correct/wrong 标注。附门控自身 precision/recall 表。初始策略：保守（高精度低召回），保证最坏 ≈ ②。

## 指标

1. 主：Stage 3 accuracy（规则判分域）；
2. 诊断：SRA relevance-aware / need-aware 分离度（门控 vs 模型自主）；
3. 成本：平均注入 token、加载率。

## 范围与排期

- 一期：theoremqa/logicbench/medcalc/champ（2,830 题 × 5 线 ≈ 14k 次推理 ≈ 3–5h A100，可过夜）；
- 二期：bigcodebench（需补装其执行判分依赖 django 等）；
- 代码三小件：scripts/gate.py（门控后检索文件生成）、scripts/calibrate_gate.py、scripts/run_phase2.sh；
- 时间线：代码半天 → 校准 1h → 批跑过夜 → 分析半天，两个工作日出主表。

## 预注册预期

- 增益最大：logicbench/champ（top-1 错误率 60–90%，拦错书收益大）；medcalc 门控 ≈ 无脑塞；
- 若 ③≈②：错书无害假说成立 → shadowing 主张需弱化,门控贡献转为纯成本节省（负结果照发）；
- ③ vs ⑤ 是第三层创新的决胜局。
