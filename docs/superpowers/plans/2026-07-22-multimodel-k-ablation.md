# 全模型 K 消融三服务器并行计划

> 日期：2026-07-22
> 状态：`[~]` 三台服务器已验证；本地六模型首波服务启动中，Mistral 由协作者回传。
> 目标：在七个正式模型上统一评估生成采样数
> \(K_{\mathrm{img}}\in\{1,2,4,8,10\}\)，回答多样本想象的收益、成本拐点与饱和位置。

## 1. 实验边界

本实验中的 \(K_{\mathrm{img}}\) 是每题生成的想象样本数，不是
Recall@5/10/50 的检索截断深度。正文、结果文件和图中统一写作
`K_img`，避免两个 K 混淆。

七个正式模型全部纳入：

- Qwen3.5-4B
- Qwen3.5-9B
- GLM-4-9B
- Llama-3.1-8B
- DeepSeek-7B
- Yi-1.5-9B
- Mistral-7B

每个 `K_img` 重跑五个受采样数影响的检索变体：

1. `naive_sentence`
2. `naive_passage`
3. `naive_skill`
4. `hyskill`
5. `two_stage`

五个变体完成后，使用与主实验完全相同的 20% 验证集、seed 0 和
validation nDCG@10 规则，为每个 `(model, K_img, domain)` 重新产生
`routed` 结果。这样既能报告固定变体对 K 的敏感性，也能报告论文主系统
`HySkill routed` 对 K 的敏感性。

本轮不重跑下列与 `K_img` 无关或成本不成比例的臂：BM25、dense、hybrid、
LLM rerank、bare、native Select、native Rerank 和回答阶段 Track B。
如果检索曲线表明 `K_img=1` 或 `2` 足以替代 `4`，再单独决定是否只在
Qwen3.5-4B 上补一个端到端确认；不预先扩成七模型回答大矩阵。

## 2. 冻结协议

除 `K_img` 外，以下设置全部固定：

| 项目 | 固定值 |
|---|---|
| 数据 | theoremqa 747 / logicbench 760 / medcalcbench 1,100 / champ 223 / bigcodebench 1,140 |
| 总实例数 | 3,970；内容寻址后有 3,968 个不同查询字符串 |
| 模板 | sentence / passage / skill |
| 温度 | 0.7 |
| 编码器 | `sentence-transformers/all-MiniLM-L6-v2` |
| 检索深度 | top-50 |
| 路由验证集 | 每域 20%，sorted ids，seed 0 |
| 路由指标 | validation nDCG@10 argmax |
| 主评估集 | 每域剩余 80% test split |
| 主指标 | routed nDCG@10 |
| 次指标 | Recall@5/10/50、五个固定变体 nDCG@10、生成 token 与墙钟 |
| 比较基准 | 同模型、同域、同实例上的 `K_img=4` |

`K_img=1/2/4/8/10` 使用嵌套前缀样本：`1` 使用样本 0，`2` 使用样本
0--1，依此类推。缓存键包含样本编号而不包含总 K，因此这种设计能保证
不同 K 之间共享已有样本，仅增加新样本，避免把不同随机样本误当成 K 效应。

## 3. 为什么不能直接并行启动现有主脚本

当前 `scripts/run_multimodel.sh` 不适合作为 K 消融 runner：

- `GEN_ARGS`、`warm_cache.py --k` 和 `gate.py --k` 均写死为 4；
- 输出写入 `results/multimodel/<TAG>/`，不同 K 会相互跳过或覆盖；
- 脚本还会重复运行 rerank、Track B 和 Select，这些不是本轮问题；
- 同时启动五份脚本会让它们并发写相同的前四个缓存项，产生重复调用和竞态；
- 每份脚本内部又开五个域流，五份脚本会膨胀为 25 个未受控进程。

因此先实现独立的 K 消融 runner，再上服务器。主脚本保持不动，避免影响
已经完成的正式结果。

## 4. 2026-07-22 服务器实测盘点

凭据、口令和 token 不写入仓库。下表只记录复现实验需要的资源事实。

| 资源位 | 已验证主机 | GPU / CPU / 内存 | 项目与缓存 | 已确认模型 | 状态 |
|---|---|---|---|---|---|
| S1 | `ubuntu22` | A100-SXM4 80GB；8 vCPU；15 GiB RAM | `/root/HySkill`，HEAD `291db6e`；`hyp_cache` 95,232 文件 / 383 MB | DeepSeek-7B、Yi-1.5-9B | `[x]` GPU 空闲；两个模型和 K=4 三模板缓存均在 |
| S2 | `user-SYS-7049GP-TRT` | A100 80GB PCIe；64 vCPU；502 GiB RAM | `/home/vicuna/ludan/HySkill`，HEAD `24a3127`；`hyp_cache` 111,104 文件 / 446 MB | Qwen3.5-4B、Qwen3.5-9B | `[x]` GPU 空闲；两个模型 K=4 缓存均在，另有一组单模板 K=8 增量缓存 |
| S3 | `ubuntu22` | 2×A100-SXM4 40GB；16 vCPU；31 GiB RAM | `/root/HySkill`，HEAD `291db6e`；`hyp_cache` 95,232 文件 / 383 MB | GLM-4-9B、Llama-3.1-8B | `[x]` 两张 GPU 空闲；两个模型和 K=4 三模板缓存均在 |

S1 的 95,232 个缓存文件恰好等于
`2 models × 3,968 unique queries × 3 templates × K=4`。S2 比这个数多
15,872，恰好等于 `3,968 × 1 template × 4 additional samples`，与已有
单模板 K=8 预热记录一致。正式启动前仍需用缓存键审计脚本把这些文件归属到
具体模型和模板，不能只按总数猜测。

两台已访问服务器的工作树均有历史实验修改和未跟踪产物。不得直接
`git pull`、清理或覆盖；实验代码必须部署到独立干净工作树，并只链接已有
`external/SR-Agents`、`hyp_cache` 和 `emb_cache`。

## 5. 模型分配与全局波次

模型服务阶段一张 GPU 同时只加载一个生成模型。三个服务器并行，服务器内部
按模型顺序补齐缓存：

| 波次 | S1 | S2 | S3（待验证） |
|---|---|---|---|
| G1 | DeepSeek-7B | Qwen3.5-4B | GLM-4-9B |
| G2 | Yi-1.5-9B | Qwen3.5-9B | Llama-3.1-8B |
| G3 | — | — | Mistral-7B（协作者回传，本地暂不运行） |

若 S3 实际模型分布不同，只调整资源位，不改变模型 tag、checkpoint 或缓存键。
禁止为了均衡负载换用另一个同家族 checkpoint；K 消融必须与主实验模型完全
一致。

每台服务器采用两个阶段：

### G：生成缓存阶段

1. 用主实验完全相同的 served-model name 启动一个模型端点；
2. 先运行 `warm_cache.py --templates passage,skill,sentence --k 8`；
3. 审计通过后导出并上传 K=1/2/4/8 的严格前缀包；
4. 再运行同一缓存键序列到 `--k 10`，仅补 sample 8--9；
5. 审计通过后导出并上传 K=10 包；
6. 停止本次明确记录 PID 的模型服务，再加载该服务器的下一个模型。

每个完整 K=4 模型补到三模板 K=10，新增缓存上限为：

```text
3,968 unique queries × 3 templates × 6 new samples = 71,424
```

七模型最大新增量为 499,968。S2 已有的单模板 K=8 缓存经归属审计确认后，
可再减少 15,872 次调用。不得同时为同一模型启动多个 K warmup；K=10 一次
预热已经覆盖全部较小 K。

### R：检索与路由阶段

本机所有模型补齐 K=10 后关闭 vLLM，把以下 25 个任务放入一个扁平队列：

```text
(K_img ∈ {1,2,4,8,10}) × (5 domains)
```

一个队列任务串行完成该 `(model, K_img, domain)` 的五个检索变体，避免
嵌套并行。全部变体完成后，再按 K 并行执行五域路由和汇总。

资源上限按实测配置设置：

| 资源位 | `MAX_RETRIEVAL_JOBS` | 理由 |
|---|---:|---|
| S1 | 4 | 仅 8 vCPU / 15 GiB；超过 4 容易因多个 MiniLM 与索引副本耗尽内存 |
| S2 | 12 | 64 vCPU / 502 GiB；12 路能并行多个 K 和域，同时避免 25 路 GPU 争用 |
| S3 | 启动前根据 `nproc` 与 `free -h` 设定 | 未盘点前不猜并发数；上限不得直接照抄 S2 |

这使并行发生在三个层面，但每层都有界：三服务器并行、各服务器模型缓存
顺序补齐、缓存完成后多个 `(K, domain)` 任务并行。

## 6. 需要先实现的最小代码

### K001：缓存审计器

新增 `scripts/audit_k_cache.py`：

- 输入必须显式给出 model、instances、templates、temperature、cache-dir 和 K；
- 先从 `hyskill/generator.py` 提取带完整类型标注的纯缓存键函数，让
  `HypotheticalGenerator` 与审计器共同调用；不得在审计器中复制哈希逻辑；
- 输出每个 `(model, template, sample_index)` 的 expected / present / missing；
- 任何缺失、空文件或不可读文件均以非零状态退出；
- 生成 JSON 审计结果供 manifest 引用，不能只打印人读日志。

### K002：专用 runner

新增 `scripts/run_k_ablation.sh`：

- 必填参数：`TAG`、`MODEL`、`API_BASE`、`K_VALUES`、`MAX_RETRIEVAL_JOBS`；
- 生成阶段只预热 `max(K_VALUES)`；
- 检索阶段按 `(K, domain)` 扁平有界并发，每个任务内部五变体串行；
- 不调用 rerank、infer、evaluate、Select 或 gate；
- 每条命令失败立即令总任务失败，日志保留准确的 model/K/domain/variant；
- 已完成文件只有通过 schema、实例数和 metadata 校验后才能跳过，不能仅凭
  “文件存在”跳过；
- 捕获并汇总所有后台任务退出码，不允许静默缺臂。

### K003：K 汇总器

新增 `scripts/summarize_k_ablation.py`，产出：

- `metrics_long.jsonl`：model / domain / K / variant / split / n / nDCG / Recall；
- `summary.json`：逐模型、跨域与跨模型聚合；
- `paired_vs_k4.json`：每个 K 相对 K=4 的逐题配对 bootstrap 差异、95% CI 和 p；
- `cost.json`：每个模板与 K 的实际输入/输出 token 估计和预热墙钟；
- `manifest.json`：代码 commit、模型 checkpoint、served-model name、encoder、
  数据文件 hash、缓存审计文件和完成时间。

## 7. 输出隔离

原有 `results/multimodel/<TAG>/` 与 `community-results/<TAG>/summary.json`
保持不动。新产物固定写入：

```text
results/k-ablation/<TAG>/
├── k1/<domain>-<variant>.json
├── k2/<domain>-<variant>.json
├── k4/<domain>-<variant>.json
├── k8/<domain>-<variant>.json
├── k10/<domain>-<variant>.json
├── routed/k1/...k10/
├── logs/
└── audits/

community-results/<TAG>/k-ablation/
├── metrics_long.jsonl.gz
├── summary.json
├── paired_vs_k4.json
├── cost.json
└── manifest.json
```

每模型应有 `5 K × 5 domains × 5 variants = 125` 个固定变体结果，以及
`5 K × 5 domains = 25` 个 routed 结果，共 150 个检索结果。七模型完整矩阵
应有 1,050 个结果文件；汇总器必须按这个矩阵验收，不能用缺失项平均。

## 8. 启动前门禁

每台服务器、每个模型开始前必须全部通过：

1. GPU 无未知占用；
2. 干净实验工作树指向同一个代码 commit；
3. `external/SR-Agents` 数据实例数分别为 747 / 760 / 1,100 / 223 / 1,140；
4. MiniLM 可在 CUDA 上加载；
5. `/v1/models` 返回的 served-model name 与主实验缓存 tag 完全一致；
6. K=4 缓存审计无缺失；
7. 单域 `K_img=1` 与 `K_img=10` pilot 均能写入隔离目录，metadata 中 K 正确；
8. `bash -n`、现有 `pytest -q` 和 `scripts/smoke.sh` 通过。

服务器实验不在本机运行。本机只负责代码、静态检查和结果汇总脚本验证；真实
pilot、生成和检索均在对应 GPU 服务器执行。

## 9. 完成标准与任务状态

| ID | 状态 | 任务 | 优先级 | 相关文件 | 完成标准 | 验证结果 | 更新时间 |
|---|---|---|---|---|---|---|---|
| K001 | `[~]` | 实现缓存审计器 | 高 | `scripts/audit_k_cache.py` | 七模型均可按模板和样本编号证明 K=10 缓存完整 | GLM K=4 实测：47,616/47,616，缺失/空/不可读均为 0 | 2026-07-22 |
| K002 | `[ ]` | 实现专用 K runner | 高 | `scripts/run_k_ablation.sh` | 无输出冲突、无 K 无关臂、后台失败能传递 | 待实现 | 2026-07-22 |
| K003 | `[ ]` | 实现汇总与配对检验 | 高 | `scripts/summarize_k_ablation.py` | 能拒绝缺失矩阵并输出效果与成本 | 待实现 | 2026-07-22 |
| K004 | `[x]` | 补齐 S3 盘点 | 高 | 本计划第 4 节 | 获得可用入口并确认 GPU、CPU、RAM、项目、模型与缓存 | 2×A100 40GB、GLM/Llama 与两模型完整 K=4 缓存已验证 | 2026-07-22 |
| K005 | `[~]` | 三服务器补齐 K=10 缓存 | 高 | `results/hyp_cache`、审计 JSON | 七模型三模板 sample 0--9 全部存在、empty=0 | 本地六模型首波服务启动中；Mistral 等协作者回传 | 2026-07-22 |
| K006 | `[ ]` | 跑完七模型 K 检索矩阵 | 高 | `results/k-ablation/` | 每模型 150 个结果，实例数和 metadata 全通过 | 未启动 | 2026-07-22 |
| K007 | `[ ]` | 汇总论文级结论 | 高 | `community-results/*/k-ablation/` | K 曲线、相对 K=4 配对 CI、token/墙钟同时齐全 | 未启动 | 2026-07-22 |

## 10. 最终判读规则

- 若 `K_img=1` 或 `2` 相对 `4` 的跨模型 routed nDCG@10 差异很小且 CI
  不显示稳定退化，则主结论是可以显著降低生成成本；不写成形式化等价，除非
  另行设定并完成等价界检验。
- 若 `8` 或 `10` 相对 `4` 无稳定收益，则 `K_img=4` 是经验饱和点；`10`
  作为上界证据保留，不因结果平坦而删除。
- 若最佳 K 随模型或域明显变化，报告异质性，不只报七模型总平均。
- 任何效果表必须同行给出生成 token 或墙钟；K 消融的核心是效果--成本曲线，
  不能只报精度。
- 缺模型、缺域或缺 K 时不计算“全模型平均”，也不把缺失项当作零。
