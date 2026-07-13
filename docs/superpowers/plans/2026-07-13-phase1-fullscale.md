# Phase 1 全量实验计划

> 状态：2026-07-13 制定。执行环境：GPU 服务器（A100，Qwen3.5-4B/9B 本地 vLLM，MiniLM 编码）。
> 目标：把试点的三条规律升级为全量+显著性支撑的正式结论，并验证两阶段架构。

## 规模

- 实例：3,970（theoremqa 747 / logicbench 760 / medcalc 1,100 / champ 223 / bigcode 1,140；ToolQA 缓）
- 生成：3 模板 × K=4 × 3,970 ≈ 47.6k 次（并行预热 1–1.5h）；llm_rerank 3,970 次（~2h）
- 磁盘：<0.5GB 增量；API 费：0

## 步骤

### Step 1 代码（本地写+测，推送后服务器拉取）
1. `scripts/warm_cache.py`：并行（32 workers）批量生成假想文档灌 `results/hyp_cache`——复用 HypotheticalGenerator 的缓存键，参数：instances 列表、模板集、K、模型端点；
2. `hyskill/two_stage.py` + 插件注册 `two_stage`：四路融合取 top-50 → 用"完整技能"单向量对候选重排——规律 A 的方法化；纯向量运算复用缓存；
3. `scripts/significance.py`：从检索结果 JSON 的逐题记录计算逐题 nDCG@10/R@K，方法两两配对 bootstrap（10k 重采样）报 p 值与置信区间。

### Step 2 主网格（服务器，nohup+落盘日志）
预热缓存 → 5 域 × {bm25, dense, hybrid, llm_rerank, sentence, passage, skill, hyskill, two_stage} × 全量实例，top-50。

### Step 3 消融
- K∈{1,2,4} 缓存子集分析 + K=8 最佳模板补采；
- 生成器规模：Qwen3.5-9B 重跑最佳模板（换 vLLM 服务或双服务）；
- 编码器 bge-base（**待批准下载 440MB**）；
- SkillRouter 同场（**待批准下载 2.5GB**）。

### Step 4 汇总
significance 报表 → 更新 `docs/05-results.md`（全量主表替换试点表，试点表降级为附录）→ README 结果区更新 → 推送。

## 判定标准

- 三条规律在全量下仍显著（p<0.05）→ 写入论文主张；
- 两阶段 ≥ max(单路, 四路) 且差距显著 → 定为主方法；
- 想象类 vs llm_rerank 全量优势显著 → 保留"想象胜过重排"主张。

## 风险

- vLLM 长时并发稳定性 → 预热器带重试与断点续灌（缓存天然断点）；
- 9B 显存：与 4B 分时服务（跑完 4B 网格再换 9B）；
- champ/bigcode 多金标的逐题指标计算需与 SR-Agents 官方口径一致（直接复用其 metrics 函数）。
