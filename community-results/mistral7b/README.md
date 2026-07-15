# mistral7b 数据档案（外部协作者）

**模型**：mistralai/Mistral-7B-Instruct-v0.3（Mistral）｜ 跑批方：外部协作者自备机器 ｜ 特殊配置：BF16、8K 上下文、五域重排；无需 `NO_THINK`。

## 本次回传文件

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | 检索、路由、门控与成本审计汇总 | ✅ 完成 |
| `retrieval_top10.jsonl.gz` | 27,790 行实例×变体明细：金标、top-10、逐题 nDCG@10 | ✅ 完成 |
| `gating_per_instance.jsonl.gz` | 2,830 行门控逐题数据：S1/S2、τ、拦截决定及五臂对错 | ✅ 完成 |
| `imagination_samples.jsonl.gz` | 每域固定 10 题，三模板 × K=4 的非空想象文本及各变体 top-3 | ✅ 完成 |
| `router_decisions.json` | 五域验证集路由选择及候选变体比分 | ✅ 完成 |
| `metrics_flat.jsonl.gz` | 360 行域×方法×指标的扁平表 | ✅ 完成 |
| `MANIFEST.md` | 自动生成的数据包清单与 pandas 读取示例 | ✅ 完成 |

## 协议与验证

- HySkill 实验快照：`c8f732c5378c4c5ebe99766917e8b7351ad39331`；
- 模型 revision：`c170c708c41dac9275d15a8fff4eca08d52bab71`；
- 冻结参数：K=4、temperature=0.7、all-MiniLM-L6-v2、top-k=50、20% 标定集 / seed 0；
- Track A 覆盖五域 3,970 题，`RERANK_DOMAINS=all`；Track B 覆盖四个规则域 2,830 题；
- bare / always / gated / select / oracle 五臂齐全；SELECT 与 oracle 使用相同冻结模型 revision 和端点补跑；
- 官方 analysis pack 逐行复核通过：JSON/gzip 均可解析、无空想象样本、signals `cache_misses=0`。

## 注意

- 原始 top-50 榜单、答题 JSONL、日志及完整想象缓存按协作协议留在跑批服务器，不提交入库；
- `summary.json` 的 routed/gating 聚合仍包含标定集；独立 held-out 重算和配对显著性审计已通过，复核件留存服务器；
- rerank 成本沿用项目统一的 `chars/3.8` 字符估算口径；
- 运行方式与数据流说明见 [docs/08-multimodel-plan.md](../../docs/08-multimodel-plan.md)。

## 数据去向

- `docs/09-summary.md` 的跨模型矩阵由维护者在合并后统一更新。


## v2.2 协议补跑提示（2026-07-15）

本包缺三个新增做题臂：`select`（同源消融）、`select_bm25`（原装自选基线）、`always_rerank`（原装重排基线）。补跑方式：`git pull` 后用原命令重跑（断点逻辑自动只补缺臂，约 3–4 小时），再 `export_analysis_pack.py mistral7b mistral7b` + `export_top50.py mistral7b` + `export_loading.py mistral7b`（完整 top-50 榜单与装载数据，新入库要求）重导并提 PR 覆盖。规则见 docs/08 §5「基线不混搭」。
