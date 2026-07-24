# community-results：多模型协作实验回传区

每个认领模型的运行者把规范化分析包通过 PR 提交到
`community-results/<TAG>/`。这里接收可直接复核的小型 JSON / gzip JSONL，
不接收 `results/hyp_cache` 的数万个散文件、原始做题 jsonl 或运行日志。认领表、
导出命令和完整验收步骤见
[docs/08-multimodel-plan.md](../docs/08-multimodel-plan.md)。

## K=2/K=4 主实验目录契约

当前论文活跃主实验统一使用 K=2，新结果上传到
`community-results/<TAG>/k2/`；历史 K=4 主实验包归档到
`community-results/<TAG>/k4/`。两者不得把逐题证据、汇总指标或 manifest
混在同一目录。

`k-ablation/` 继续保存 K={1,2,4,8,10} 的联合分析包；
`imagination_full_k{1,2,4,8,10}.*` 继续作为完整 prefix cache 保持在模型
目录根部。它们不归入 `k2/` 或 `k4/`。Qwen reference 的
`baselines-native/` 不依赖 imagination K，也保持为共享证据。

截至 2026-07-23，只读审计确认旧 K=4 主实验文件仍在模型目录根部。迁移必须
等 K=2 包验收完成后采用一次 `git mv`，不得复制整个 399 MB 目录。详细文件
映射、SHA 门禁和回滚规则见
[GitHub K 目录迁移清单](../docs/superpowers/plans/2026-07-23-community-results-k-layout-migration.md)。

## K=2 协作者必传文件

完成统一主实验后上传到 `community-results/<TAG>/k2/`：

- `retrieval_top50.jsonl.gz`、`router_decisions.json`
- `gating_per_instance.jsonl.gz` 与 `loading_per_instance.jsonl.gz`
- `selection_per_instance.jsonl.gz`（仅上下文支持 Select 的模型）
- `answer_per_instance.jsonl.gz` 与 `answer_metrics.json`
- `metrics_flat.jsonl.gz`、`significance.json`
- `reuse_manifest.json`、`manifest.json` 与 `README.md`

无法运行 Select 的模型必须在 manifest 中标记 unavailable 和原因，不得上传
空文件或以零值代替。

为支持生成采样数 K 消融，2026-07-22 起还必须按嵌套前缀补传：

- `imagination_full_k{1,2,4,8,10}.jsonl.gz`：全部 3,970 题对应 K
  前缀的三模板想象；
- 同名 `.manifest.json`：模型/代码 revision 与完整性哈希。

使用 `scripts/export_full_imagination_cache.py --k <K>` 逐个生成。验收值固定为
`rows=3970`、`unique_queries=3968`、
`verified_cache_files=11904*K`；任一想象为空或缺失时脚本会失败，不得用空字符串
或重新生成的文本悄悄补位。完整命令见
[多模型计划 §7](../docs/08-multimodel-plan.md#7-结果回传三步)。

完成检索矩阵后，每个模型还必须在 `k-ablation/` 下提交以下五个紧凑文件：

| 文件 | 内容 |
|---|---|
| `metrics_long.jsonl.gz` | 5 个 K × 5 域 × 6 变体 × 3 split 的规范化 Recall/nDCG 长表 |
| `summary.json` | 逐域、跨域 micro/macro 的完整 K 曲线 |
| `paired_vs_k4.json` | `K={1,2,8,10}` 相对 `K=4` 的逐题配对 bootstrap |
| `cost.json` | 基于真实 prompt 和缓存输出文本的 token 估算及墙钟可用性 |
| `manifest.json` | 模型、代码、缓存、150 个结果文件和上述输出的完整性证据 |

服务器端 1,050 个原始检索 JSON、缓存散文件和日志仍按协议留在运行服务器，
不进入 GitHub；上述逐题指标长表和哈希清单是公开复核入口。

## 档案索引（每个模型一个文件夹，夹内 README 说明应存放的文件与服务器端原始件位置）

| TAG | 家族 | 跑批方 | summary.json | 重排臂 |
|---|---|---|---|---|
| [qwen35-9b](qwen35-9b/) | 阿里 | 实验室主服务器 | ✅ 满配入库（base+select+五域 rerank） | ✅ 五域全跑（带思考截断疑点待复核） |
| [glm4-9b](glm4-9b/) | 智谱 | 服务器 A | ✅ 全套分析包已入库 | ✅ |
| [llama31-8b](llama31-8b/) | Meta | 服务器 A | ✅ 满配入库（base+select+五域 rerank） | ✅ 五域全跑 |
| [deepseek7b](deepseek7b/) | 深度求索 | 服务器 B | ✅ 全套分析包已入库 | ➖ 跳过（4K 上下文） |
| [yi15-9b](yi15-9b/) | 零一万物 | 服务器 B | ✅ 全套分析包已入库 | ➖ 跳过（4K 上下文) |
| [mistral7b](mistral7b/) | Mistral | 外部协作者（EXPLORER41，PR #1+#2） | ✅ v2.2 满配（六臂+oracle+top50+装载数据） | ✅ 五域全跑 |
| [qwen3.5-4b-reference](qwen3.5-4b-reference/) | 阿里（主实验基准列） | 实验室主服务器 | ✅ 全套参考包已入库（9 方法检索明细 + 7 臂逐题 + 双门信号 + 显著性） | ✅ |

## K 消融归档索引

| 范围 | 数据目录 | 状态 |
|---|---|---|
| Qwen3.5-4B | [qwen3.5-4b-reference/k-ablation](qwen3.5-4b-reference/k-ablation/) | ✅ 150 结果汇总包 |
| Qwen3.5-9B | [qwen35-9b/k-ablation](qwen35-9b/k-ablation/) | ✅ 150 结果汇总包 |
| GLM-4-9B | [glm4-9b/k-ablation](glm4-9b/k-ablation/) | ✅ 150 结果汇总包 |
| Llama-3.1-8B | [llama31-8b/k-ablation](llama31-8b/k-ablation/) | ✅ 150 结果汇总包 |
| DeepSeek-7B | [deepseek7b/k-ablation](deepseek7b/k-ablation/) | ✅ 150 结果汇总包 |
| Yi-1.5-9B | [yi15-9b/k-ablation](yi15-9b/k-ablation/) | ✅ 150 结果汇总包 |
| Mistral-7B | [mistral7b/k-ablation](mistral7b/k-ablation/) | ✅ 150 结果汇总包 |
| 七模型共同支持 | [k-ablation-fleet](k-ablation-fleet/) | ✅ 5,040 指标行、10,000 次 bootstrap |

论文级结论、效果--成本表和解释边界见
[七模型 K 消融分析报告](../docs/10-k-ablation-analysis.md)。
