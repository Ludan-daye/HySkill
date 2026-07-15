# deepseek7b 数据档案

**模型**：deepseek-ai/deepseek-llm-7b-chat（深度求索，ModelScope 源；原计划 gemma-2-9b 因 HF gated 墙替换）｜ 跑批机：2×A100-40G 服务器 B 的 GPU0（端口 8001）｜ 特殊配置：**上下文 4K → 重排臂跳过**（`RERANK_DOMAINS=""`，50 候选长 prompt 装不下，协议注明）；vllm 0.19.1 + enforce-eager；重跑段 WORKERS=16。

## 本文件夹应存放（入库 GitHub）

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | v2.1 菜单汇总（检索/路由/门控/成本；无重排块） | ⏳ 跑批中 |

## 留在跑批服务器不入库的原始件

位置 `/root/HySkill/results/multimodel/deepseek7b/`：检索榜单 ×25、routed/signals/taus/gated、做题 jsonl+eval、logs/。

## 数据去向

- docs/09-summary.md §〇 跨模型矩阵「deepseek7b」列
- 运行备注：4K 上下文使做题长尾少量实例可能溢出（各臂均匀承受，臂间对比有效）；检索段曾因双模型嵌入进程挤 GPU0 冻结返工（断点续跑，无数据损失）
