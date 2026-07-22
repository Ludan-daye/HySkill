# community-results：多模型协作实验回传区

每个认领模型的运行者把规范化分析包通过 PR 提交到
`community-results/<TAG>/`。这里接收可直接复核的小型 JSON / gzip JSONL，
不接收 `results/hyp_cache` 的数万个散文件、原始做题 jsonl 或运行日志。认领表、
导出命令和完整验收步骤见
[docs/08-multimodel-plan.md](../docs/08-multimodel-plan.md)。

## 协作者必传文件

完成主实验后上传：

- `summary.json`
- `retrieval_top10.jsonl.gz` 与 `retrieval_top50.jsonl.gz`
- `gating_per_instance.jsonl.gz` 与 `loading_per_instance.jsonl.gz`
- `metrics_flat.jsonl.gz`、`router_decisions.json`
- `imagination_samples.jsonl.gz`（每域 10 题的定性样例）
- `MANIFEST.md` 与模型目录内的 `README.md`

为支持生成采样数 K 消融，2026-07-22 起还必须按嵌套前缀补传：

- `imagination_full_k{1,2,4,8,10}.jsonl.gz`：全部 3,970 题对应 K
  前缀的三模板想象；
- 同名 `.manifest.json`：模型/代码 revision 与完整性哈希。

使用 `scripts/export_full_imagination_cache.py --k <K>` 逐个生成。验收值固定为
`rows=3970`、`unique_queries=3968`、
`verified_cache_files=11904*K`；任一想象为空或缺失时脚本会失败，不得用空字符串
或重新生成的文本悄悄补位。完整命令见
[多模型计划 §7](../docs/08-multimodel-plan.md#7-结果回传三步)。

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
