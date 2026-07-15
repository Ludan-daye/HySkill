# community-results：多模型协作实验回传区

每个认领模型的运行者把 `scripts/run_multimodel.sh` 自动生成的
`community-results/<TAG>/summary.json` 通过 PR 提交到这里（只收小结 JSON，
原始 jsonl 请自行留存备查）。认领表与完整操作步骤见
[docs/08-multimodel-plan.md](../docs/08-multimodel-plan.md)。

## 档案索引（每个模型一个文件夹，夹内 README 说明应存放的文件与服务器端原始件位置）

| TAG | 家族 | 跑批方 | summary.json | 重排臂 |
|---|---|---|---|---|
| [qwen35-9b](qwen35-9b/) | 阿里 | 实验室主服务器 | ✅ 已入库 | ✅（带思考截断疑点待复核） |
| [glm4-9b](glm4-9b/) | 智谱 | 服务器 A | ✅ 全套分析包已入库 | ✅ |
| [llama31-8b](llama31-8b/) | Meta | 服务器 A | ⏳ 跑批中 | ✅ |
| [deepseek7b](deepseek7b/) | 深度求索 | 服务器 B | ⏳ 跑批中 | ➖ 跳过（4K 上下文） |
| [yi15-9b](yi15-9b/) | 零一万物 | 服务器 B | ⏳ 跑批中 | ➖ 跳过（4K 上下文) |
| [mistral7b](mistral7b/) | Mistral | 外部协作者 | ⏳ 等待回传 | 建议跑 |
| [qwen3.5-4b-reference](qwen3.5-4b-reference/) | 阿里（主实验基准列） | 实验室主服务器 | ✅ 全套参考包已入库（9 方法检索明细 + 7 臂逐题 + 双门信号 + 显著性） | ✅ |
