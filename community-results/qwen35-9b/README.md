# qwen35-9b 数据档案

**模型**：Qwen/Qwen3.5-9B（阿里）｜ 跑批机：实验室主服务器（A100-80G PCIe，端口 8311）｜ 特殊配置：`NO_THINK=1`（思考模式关闭）；重排臂已跑（长上下文）。

## 本文件夹应存放（入库 GitHub）

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | v2.1 全菜单汇总：5 变体 × 5 域检索指标、路由选择与验证集比分、bare/always/gated 做题准确率、门控 τ 与拦截统计、成本审计块 | ✅ 已入库（2026-07-15） |

## 留在跑批服务器不入库的原始件（显著性复核时 scp 取）

位置 `/home/vicuna/ludan/HySkill/results/multimodel/qwen35-9b/`：
- `<域>-<变体>.json` × 25：各变体完整检索榜单（逐题 top-50）
- `<域>-routed.json` × 5 与 `<域>-signals/taus/gated.json`：路由与门控中间产物
- `<域>-{bare,always,gated}.jsonl` + `.eval.json`：做题原始输出与逐题判分
- `logs/`：预热/重排/各域流日志
- 想象缓存在共享 `results/hyp_cache`（按模型 tag 检索）

## 数据去向

- docs/09-summary.md §〇 跨模型矩阵「qwen35-9b」列（已填）
- 已知注意事项：重排臂数字带"思考截断"疑点（sragents rerank 无 no_think 开关），待禁思考重跑对照后方可入论文主表
