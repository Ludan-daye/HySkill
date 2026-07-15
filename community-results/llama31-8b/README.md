# llama31-8b 数据档案

**模型**：LLM-Research/Meta-Llama-3.1-8B-Instruct（Meta，经 ModelScope 无门槛镜像获取——HF 原仓 gated）｜ 跑批机：A100-80G 服务器 A（端口 8001，与 glm4-9b 同卡分显存）｜ 特殊配置：无 no_think；重排臂按默认三争议域；vllm 0.19.1 + enforce-eager；WORKERS=16（15GB 小内存机减压）。

## 本文件夹应存放（入库 GitHub）

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | v2.1 全菜单汇总（检索/路由/门控/成本） | ⏳ 跑批中（此槽位原计划即 llama，中途曾由 internlm2.5 代打后换回——internlm 与 vllm 0.19.1 不兼容，见 docs/08 坑表） |

## 留在跑批服务器不入库的原始件

位置 `/root/HySkill/results/multimodel/llama31-8b/`：检索榜单 ×25、routed/signals/taus/gated、做题 jsonl+eval、logs/。

## 数据去向

- docs/09-summary.md §〇 跨模型矩阵「llama31-8b」列
