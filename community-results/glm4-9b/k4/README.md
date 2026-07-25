# glm4-9b 数据档案

**模型**：ZhipuAI/glm-4-9b-chat（智谱，ModelScope 源）｜ 跑批机：A100-80G 服务器 A（端口 8002，与 llama31-8b 同卡分显存）｜ 特殊配置：无 no_think；重排臂已跑（长上下文）；vllm 0.19.1 + enforce-eager。

## 本文件夹应存放（入库 GitHub）

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | v2.1 全菜单汇总（检索/路由/门控/成本，格式同 qwen35-9b） | ✅ 已入库（ALL-DONE 2026-07-15） |
| `retrieval_top10.jsonl.gz` | 匹配明细：（实例 × 变体）金标 + top-10 + 逐题 nDCG@10（26,467 行） | ✅ 已入库 |
| `gating_per_instance.jsonl.gz` | 门控逐题：S1/S2、τ、拦截决定、各臂逐题对错（2,830 行） | ✅ 已入库 |
| `imagination_samples.jsonl.gz` | 想象原文样本：每域固定 10 题（seed 0，跨模型同题）× K=4 份想象 + top-3 匹配（50 行） | ✅ 已入库 |
| `router_decisions.json` + `metrics_flat.jsonl.gz` | 路由决策与验证集比分；全指标拍平（312 行） | ✅ 已入库 |
| `MANIFEST.md` | 自动清单 + pandas 读取示例 | ✅ 已入库 |

## 留在跑批服务器不入库的原始件

位置 `/root/HySkill/results/multimodel/glm4-9b/`：检索榜单 ×25、routed/signals/taus/gated、做题 jsonl+eval、logs/。想象缓存在 `/root/HySkill/results/hyp_cache`。

## 数据去向

- docs/09-summary.md §〇 跨模型矩阵「glm4-9b」列（已填）
- 运行备注：预热 0 空生成；曾因 corpus.json.zip 未解压导致检索段返工一次（预热资产无损，不影响数据有效性）
- ⚠️ 数据异常待复查：logicbench 的 naive_skill nDCG@10 = 0.000（全域 760 题 top-10 无一命中金标）；路由在验证集上探测到并把该域交给 hyskill（0.112）。复查入口：`imagination_samples.jsonl.gz` 里 logicbench 10 题的 skill 模板想象原文
