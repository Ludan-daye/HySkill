# mistral7b 数据档案（外部协作者）

**模型**：mistralai/Mistral-7B-Instruct-v0.3（Mistral）｜ 跑批方：外部协作者自备机器 ｜ 特殊配置：BF16、8K 主协议、五域重排；无需 `NO_THINK`。

## 本次回传文件

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | 检索、路由、门控与成本审计汇总 | ✅ 完成 |
| `retrieval_top10.jsonl.gz` | 27,790 行实例×变体明细：金标、top-10、逐题 nDCG@10 | ✅ 完成 |
| `retrieval_top50.jsonl.gz` | 31,760 行实例×方法明细：完整 top-50 技能 id / 分数 / 金标标记 | ✅ 完成 |
| `gating_per_instance.jsonl.gz` | 2,830 行门控逐题数据：S1/S2、τ、拦截决定、六个协议臂及 oracle 对错 | ✅ 完成 |
| `loading_per_instance.jsonl.gz` | 16,980 行实例×六臂的实际装载技能、金标与命中记录 | ✅ 完成 |
| `imagination_samples.jsonl.gz` | 每域固定 10 题，三模板 × K=4 的非空想象文本及各变体 top-3 | ✅ 完成 |
| `router_decisions.json` | 五域验证集路由选择及候选变体比分 | ✅ 完成 |
| `metrics_flat.jsonl.gz` | 376 行域×方法×指标的扁平表 | ✅ 完成 |
| `MANIFEST.md` | 数据包清单与 pandas 读取示例 | ✅ 完成 |

## 协议与验证

- HySkill 主实验快照：`c8f732c5378c4c5ebe99766917e8b7351ad39331`；原装基线回填及最终数据包导出基于 `8b7755fa388c193aa092f34448e646750cc5b9ac`；
- 模型 revision：`c170c708c41dac9275d15a8fff4eca08d52bab71`；
- 冻结参数：K=4、temperature=0.7、all-MiniLM-L6-v2、top-k=50、20% 标定集 / seed 0；
- Track A 覆盖五域 3,970 题，`RERANK_DOMAINS=all`；Track B 覆盖四个规则域 2,830 题；
- 六个协议臂齐全：bare / always / gated / select / always_rerank / select_bm25；另保留 oracle 天花板；
- `always_rerank` 为 bm25→LLM rerank→top-1 直装，`select_bm25` 为 bm25 候选→官方 selector；`select` 使用 HySkill routed 候选，只是同源消融，不作为原装基线；
- `always_rerank` 使用相同冻结模型 revision 与 8K 主协议；超长技能 400 原样保留并按错计，不做幸存者过滤；
- `select_bm25` 的 8K 超限行使用同一模型 revision 的原生 32K 服务恢复；仍有 3 题的 BM25 50 候选 prompt 达到 39–42K 输入 token，超过模型原生窗口，原样保留并按错计；8K 原件及恢复日志完整归档在跑批服务器；
- 官方 analysis pack 逐行复核通过：JSON/gzip 均可解析、无空想象样本、signals `cache_misses=0`。

## 注意

- 未压缩的原始 top-50 检索 JSON、答题 JSONL、日志及完整想象缓存按协作协议留在跑批服务器，不提交入库；本目录仅提交规范要求的紧凑版 `retrieval_top50.jsonl.gz`；
- `summary.json` 的 routed/gating 聚合仍包含标定集；独立 held-out 重算和配对显著性审计已通过，复核件留存服务器；
- rerank 成本沿用项目统一的 `chars/3.8` 字符估算口径；
- 运行方式与数据流说明见 [docs/08-multimodel-plan.md](../../docs/08-multimodel-plan.md)。

## 数据去向

- `docs/09-summary.md` 的跨模型矩阵由维护者在合并后统一更新。


## v2.2 协议回填（2026-07-16）

✅ 本次实际新增并补齐 `select_bm25`（原装自选基线）和 `always_rerank`（原装重排基线）；原有 `select` 作为同源消融保留。随后用 current main 重新执行 `export_analysis_pack.py`、`export_top50.py`、`export_loading.py`。本目录现已包含六个协议臂、oracle 天花板、完整 top-50 紧凑榜单及逐题装载记录。
