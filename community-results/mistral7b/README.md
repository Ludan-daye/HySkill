# mistral7b 数据档案（外部协作者）

**模型**：mistralai/Mistral-7B-Instruct-v0.3（Mistral）｜ 跑批方：外部协作者自备机器 ｜ 特殊配置：无 no_think；长上下文可跑重排臂（建议默认三争议域）。

## 本文件夹应存放（协作者提 PR）

| 文件 | 内容 | 状态 |
|---|---|---|
| `summary.json` | 由 `scripts/summarize_multimodel.py` 自动生成（跑完 `run_multimodel.sh` 即有），含检索/路由/门控/成本审计 | ⏳ 等待协作者回传 |

## 协作者注意

- 原始大文件（results/multimodel/mistral7b/ 下的 *.json / *.jsonl / logs/）**不要**提交，请本地留存备显著性复核；
- 运行方式与常见坑见 [docs/08-multimodel-plan.md](../../docs/08-multimodel-plan.md)（重点：国内机器用 ModelScope/清华源；驱动 550 类机器需 vllm cu12x 构建 + --enforce-eager）。

## 数据去向

- docs/09-summary.md §〇 跨模型矩阵「mistral7b」列（回传后由维护者填入）
