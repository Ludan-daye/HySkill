# 全模型 K 消融三服务器并行计划

> 日期：2026-07-22
> 状态：`[x]` 七模型 K 消融矩阵、逐模型汇总和严格共同支持 fleet 汇总均已完成并验收。
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

正式结果的代码身份固定为：

```text
repository commit: b642012
runner bundle SHA-256: efc7d610d25b29e28fdb14ab58da099e3e7b2b0b0b7fb55d2f7817f8306cf69f
SOURCE_REVISION: b642012+bundle-efc7d610d25b29e28fdb14ab58da099e3e7b2b0b0b7fb55d2f7817f8306cf69f
```

六个 runner/validator 文件的组合哈希由 `summarize_k_ablation.py` 按文件
SHA-256 重新计算，不手工缩写。已落盘 fixed 结果均使用上述 64 位标识；
逐模型迁移证据保存在 `audits/source-revision-migration.json`。旧标识生成的
6 个 routed pilot 文件保留在
`quarantine/source-revision-typo-20260723/routed/`，不计入正式矩阵，并由
正确 fixed 文件重新路由。

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
| S1 | `ubuntu22` | A100-SXM4 80GB；8 vCPU；15 GiB RAM | `/root/HySkill-k-run-20260723`，基线 `b642012`；共享 `hyp_cache` 238,080 文件 | DeepSeek-7B、Yi-1.5-9B | `[x]` 两模型各 150 个结果和 5 个汇总文件均完成；K=10 全域缓存审计均为 119,040/119,040 |
| S2 | `user-SYS-7049GP-TRT` | A100 80GB PCIe；64 vCPU；502 GiB RAM | `/home/vicuna/ludan/HySkill-k-run-20260723`，基线 `b642012`；共享 `hyp_cache` 238,080 文件 | Qwen3.5-4B、Qwen3.5-9B | `[x]` 两模型各 150 个结果和 5 个汇总文件均完成；K=10 全域缓存审计均为 119,040/119,040 |
| S3 | `ubuntu22` | 2×A100-SXM4 40GB；16 vCPU；31 GiB RAM | `/root/HySkill-k-run-20260723`，基线 `b642012`；共享 `hyp_cache` 357,120 文件 | GLM-4-9B、Llama-3.1-8B、Mistral-7B | `[x]` 三模型各 150 个结果和 5 个汇总文件均完成；K=10 全域缓存审计均为 119,040/119,040 |

缓存归属已经由缓存键审计确认，不再依据目录文件总数推测。七个模型各自的
全域 K=10 缓存均为 `119,040/119,040`，即
`3,968 unique queries × 3 templates × 10 samples`；missing、empty 和
unreadable 均为 0。各个 K=1/2/4/8/10 manifest 也已验证为同一 K=10
样本序列的严格前缀。

两台已访问服务器的工作树均有历史实验修改和未跟踪产物。不得直接
`git pull`、清理或覆盖；实验代码必须部署到独立干净工作树，并只链接已有
`external/SR-Agents`、`hyp_cache` 和 `emb_cache`。

## 5. 模型分配与全局波次

模型服务阶段一张 GPU 同时只加载一个生成模型。三个服务器并行，服务器内部
按模型顺序补齐缓存：

| 波次 | S1 | S2 | S3 |
|---|---|---|---|
| G1 | DeepSeek-7B | Qwen3.5-4B | GLM-4-9B |
| G2 | Yi-1.5-9B | Qwen3.5-9B | Llama-3.1-8B |
| G3 | — | — | Mistral-7B |

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
| S1 | 2（每模型 1） | 实测 4 个 hyskill 进程各占约 3.5--3.9 GiB，并触发 swap；降为 2 后恢复 14 GiB 可用内存 |
| S2 | 12（每模型 6） | 64 vCPU / 502 GiB；12 路仍保留充足 CPU、RAM 与 GPU 余量 |
| S3 | 6（每模型 2） | 16 vCPU / 31 GiB；实测 6 路时仍有约 6 GiB 可用内存 |

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
| K001 | `[x]` | 实现缓存审计器 | 高 | `scripts/audit_k_cache.py`、`scripts/validate_k_cache.py` | 七模型均可按模板和样本编号证明 K=10 缓存完整 | 七模型全域 K=10 审计均为 119,040/119,040；missing、empty、unreadable 均为 0 | 2026-07-23 |
| K002 | `[x]` | 实现专用 K runner | 高 | `scripts/run_k_ablation.sh`、`scripts/route_k_retrieval.py`、`hyskill/k_ablation.py` | 无输出冲突、无 K 无关臂、后台失败能传递 | S1/S3 的 K=1/K=10 theoremqa 实机 pilot 均完成 10 固定结果 + 2 routed，所有结果经 schema、gold、top-50、重算指标、缓存与来源哈希复验 | 2026-07-23 |
| K003 | `[x]` | 实现汇总与配对检验 | 高 | `scripts/summarize_k_ablation.py`、`scripts/summarize_k_ablation_fleet.py` | 能拒绝缺失矩阵并输出效果与成本 | 七个真实模型包均通过 150 文件验收；fleet 汇总验证 7 模型、5,040 条共同支持指标、10,000 次 bootstrap 和四个输出哈希 | 2026-07-23 |
| K004 | `[x]` | 补齐 S3 盘点 | 高 | 本计划第 4 节 | 获得可用入口并确认 GPU、CPU、RAM、项目、模型与缓存 | 2×A100 40GB、GLM/Llama/Mistral 与三模型完整 K=10 缓存已验证 | 2026-07-23 |
| K005 | `[x]` | 三服务器补齐 K=10 缓存 | 高 | `results/hyp_cache`、`community-results/*/imagination_full_k*.manifest.json` | 七模型三模板 sample 0--9 全部存在、empty=0 | 七模型 K=1/2/4/8/10 前缀包均已验证并入库，GitHub `main=b642012` | 2026-07-23 |
| K006 | `[x]` | 跑完七模型 K 检索矩阵 | 高 | `results/k-ablation/` | 每模型 150 个结果，实例数和 metadata 全通过 | 1,050/1,050 个正式结果完成；每模型 125 fixed + 25 routed、5,040 条指标记录、失败标记 0，来源与结果哈希均通过验收 | 2026-07-23 |
| K007 | `[x]` | 汇总论文级结论 | 高 | `community-results/*/k-ablation/`、`community-results/k-ablation-fleet/`、`docs/10-k-ablation-analysis.md` | K 曲线、相对 K=4 配对 CI 和实际文本 token 估算齐全；墙钟可用性显式记录 | 七模型 35 个文件与服务器 SHA-256 完全一致；fleet manifest 为 `62f97a5...`；正式分析报告已整理，七模型墙钟均标记 unavailable，未做事后估造 | 2026-07-23 |

## 10. 最终判读规则

- 若 `K_img=1` 或 `2` 相对 `4` 的跨模型 routed nDCG@10 差异很小且 CI
  不显示稳定退化，则主结论是可以显著降低生成成本；不写成形式化等价，除非
  另行设定并完成等价界检验。
- 若 `8` 或 `10` 相对 `4` 无稳定收益，则只写性能在 `K_img=2--4` 进入
  经验平台区；`K=4` 保留为主实验 reference，`10` 作为上界证据保留。
- 若最佳 K 随模型或域明显变化，报告异质性，不只报七模型总平均。
- 任何效果表必须同行给出生成 token 或墙钟；K 消融的核心是效果--成本曲线，
  不能只报精度。
- 缺模型、缺域或缺 K 时不计算“全模型平均”，也不把缺失项当作零。

## 11. 最终结果入口

机器可读的逐模型数据与 fleet 汇总位于
`community-results/<model-tag>/k-ablation/` 和
`community-results/k-ablation-fleet/`。规范化主表、模型/领域异质性、
路由选择变化、成本解释和论文措辞边界统一维护在
[七模型 K 消融分析报告](../../10-k-ablation-analysis.md)，避免计划与结果报告
重复维护数字。

一句话结论：在七模型共同支持上，`K=2` 将估算生成 token 减半，且未检测到
routed nDCG@10 相对 `K=4` 的下降；更大的 K 没有带来总体收益。
