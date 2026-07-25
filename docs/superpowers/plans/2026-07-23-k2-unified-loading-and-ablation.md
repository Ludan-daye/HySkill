# 全模型 K=2 统一装载与消融实验计划

> 日期：2026-07-23  
> 状态：`[~]` 方案已获用户批准；代码实现、N1 部署和 K=2 下游实验正在并行执行；32/32 Gate 已完成。  
> 目标：除 K 消融本身外，将当前论文仍在使用的所有想象相关实验统一为
> \(K_{\mathrm{img}}=2\)，优先交付装载正确率，再完成端到端回答、消融、
> 显著性和论文数据更新。

## 0. 已批准的执行决策

本计划固定以下决策，执行过程中不得静默改变：

1. **统一标准**：K 消融继续保留
   \(K_{\mathrm{img}}\in\{1,2,4,8,10\}\)；除此之外，当前论文活跃实验统一使用
   \(K_{\mathrm{img}}=2\)。
2. **不靠删实验加速**：七模型 routed Always/Gated、五模型 routed
   Hy+Select，以及 Qwen3.5-4B fixed `naive_skill` + Gated 组件消融全部纳入。
3. **K 无关结果经身份门禁后复用**：Bare、Oracle、BM25、Dense、Hybrid、
   native Rerank、BM25+native Select 不因 K 改变；只有旧/新模型与 runtime
   身份一致时才不重新调用模型。
4. **选择与回答解耦**：Hy+Select 先运行 selection-only，持久化每题的选择
   决策；回答阶段只读取该决策，禁止再次调用 selector。
5. **严格同臂复用**：仅在同一 arm 的完整回答请求哈希完全相同时复用旧
   K=4 回答；禁止 Bare → Gated、Always → Gated 等跨臂拼接。
6. **按实测五槽调度**：当前四个 SSH 节点共有五个 GPU 槽；N1 承载
   Qwen3.5-9B，Mistral-7B 在 S3-0 完成 GLM 后串行运行。未经盘点不得虚构
   第五个物理节点或第六个 GPU 槽。
7. **保持 Select 行为不变**：沿用现有 pool=50、temperature=0、
   max_tokens=64、最多三次解析和失败后回退 rank-1 的协议。本轮不增加
   abstain；论文后续把 “choose or abstain” 修正为实际的强制单选语义。
8. **输出完全隔离**：K=2 下游结果不得覆盖
   `results/multimodel/`、`results/phase2/` 或现有 K=4 产物；公开主实验包
   分别进入 `community-results/<tag>/k2/` 和
   `community-results/<tag>/k4/`。联合 `k-ablation/` 与
   `imagination_full_k{1,2,4,8,10}.*` 前缀缓存保持独立，不归入任一主实验
   目录。

这里的“所有实验”指**当前论文活跃且仍会被引用的实验**。Phase 0、旧 Qwen
K=4 五臂过程记录和已被新实验取代的中间结果保留为 `legacy K4 / not cited`，
不要求为历史归档重新制造一套 K=2 数据。若 `docs/05-results.md`、
`docs/09-summary.md` 或论文仍引用这些旧数字，最终一致性审计必须更新引用或
明确标为 legacy，不能把 K=4 与 K=2 混入同一结论。

## 1. 研究问题与交付顺序

### RQ1：K=2 下各加载策略是否正确装载？

在同一 K=2 routed 候选源上比较 Always、Gated 和 Hy+Select，并同时报告：

- Loaded-Skill Precision：已加载样本中，加载 gold skill 的比例；
- Loading Rate：全部样本中，策略决定加载技能的比例；
- Gold-Load Rate：全部样本中，最终加载 gold skill 的比例。

只报告条件装载精度会让低加载率方法显得虚高，因此三项必须同时出现。

### RQ2：Hy+Select 的收益来自候选源还是 selector？

保持 selector、回答模型和回答协议不变，比较：

- BM25 + native Select：K 无关，复用现有结果；
- K=2 routed HySkill + 同一 Select：本轮新结果。

五个上下文合格模型统一使用 routed HySkill 候选。旧图中 Qwen3.5-4B 使用
fixed full-skill、其他模型使用 routed 的混合口径不再沿用。

### RQ3：在相同 HySkill 候选上，哪种加载策略有效？

在 K=2 routed top-50 上比较：

- Always：加载 routed top-1；
- Gated：由 K=2 S1/S2 门控决定加载 top-1 或不加载；
- Select：由模型在同一 ordered top-50 中选一个技能。

这组实验闭合“同候选源、不同加载策略”的消融。

### RQ4：K=2 端到端主结论和组件结论是否保持？

- 七模型：K=2 routed Gated 与 Bare 比较；
- 五模型严格共同支持：K=2 routed Gated 与 native Rerank、native Select
  比较；
- Qwen3.5-4B held-out：routed+Gated 对 fixed
  `naive_skill`+Gated，以及 routed+Gated 对 routed+Always。

交付顺序固定为：

1. 七模型 Always/Gated 装载结果；
2. 五模型 Hy+Select 装载结果；
3. 严格复用总审计和实际待推理清单；
4. 全部 K=2 端到端回答；
5. 显著性、公开数据包、图表和论文更新。

为缩短墙钟，复用审计按 arm 流水执行：Always/Gated 决策在 G0 冻结后立即
审计并可开始补跑；Select 决策在 L1 冻结后立即审计。第二项交付结束后汇总为
统一复用报告，不要求所有模型等到同一时刻才开始回答。

## 2. 实验范围

### 2.1 纳入的 K=2 活跃实验

| 候选源 | 加载策略 | 模型范围 | 是否需要新回答 |
|---|---|---:|---|
| K=2 routed HySkill | Always | 7 模型 × 4 规则域 | 是，可同臂哈希复用 |
| K=2 routed HySkill | Gated | 7 模型 × 4 规则域 | 是，可同臂哈希复用 |
| K=2 routed HySkill | Select | 5 模型 × 4 规则域 | 是；先新建选择缓存 |
| K=2 fixed `naive_skill` | Gated | Qwen3.5-4B × 4 规则域 | 是，可同臂哈希复用 |

Qwen3.5-4B 的 fixed 组件消融只把 K 从 4 改为 2，不改变候选源这个实验因子。
fleet 主线使用 routed，fixed 组件线使用 `naive_skill`，两套文件名和 manifest
必须显式区分。

### 2.2 K 无关、直接复用的结果

- Bare；
- Oracle；
- BM25、Dense、Hybrid 检索；
- native LLM Rerank 检索与 `always_rerank` 回答；
- BM25 候选上的 native Select 与 `select_bm25` 回答；
- 其他没有读取想象缓存、K=2 retrieval 或 K=2 gate 决策的基线。

“K 无关”只说明方法输入不受 K 影响，不自动证明旧结果与本轮模型身份一致。
复用这些结果前仍需验证实例集合、checkpoint、tokenizer、chat template、
runtime、数据版本和原始文件哈希，不得只凭文件名认定可用。K2M001 是阻断
门禁：若任一模型无法证明旧基线与本轮 endpoint 身份一致，立即停止该模型并
报告需要扩展重跑的 Bare/native baseline jobs；在更新本计划工作量前不得继续
声称 80 jobs 覆盖该模型的完整比较。旧 Bare eval 只有通过同一身份门禁后才能
用于 K=2 S2 校准。

### 2.3 明确不纳入

- 不重跑 K 消融已经完成的检索矩阵；
- 不为 DeepSeek-7B 和 Yi-1.5-9B 强行运行 50-candidate Select；
- 不增加 Select abstain、pool-size、候选内容格式或新 prompt 消融；
- 不换模型 checkpoint、tokenizer、chat template、精度或量化方式；
- 不把 BigCodeBench 扩展为新的端到端执行实验；
- 不覆盖、删除或重命名 K=4 和 K 消融原始产物。

DeepSeek-7B 和 Yi-1.5-9B 的 Select/Rerank 长 prompt 仍标记
`unavailable`，不记作零分，也不进入五模型共同支持分母。

## 3. 冻结协议

| 项目 | 固定值 |
|---|---|
| 想象样本数 | \(K_{\mathrm{img}}=2\) |
| 想象缓存 | 已完成的 K=10 嵌套序列前两个样本 |
| 想象温度 | 0.7 |
| 检索深度 | ordered top-50 |
| 编码器 | 各模型 K 消融 manifest 中记录的同一 MiniLM checkpoint |
| 规则域 | TheoremQA 747 / LogicBench 760 / MedCalc-Bench 1,100 / CHAMP 223 |
| 回答实例总数 | 每模型 2,830 |
| 路由/门控验证集 | 各域 sorted IDs、seed 0、20% |
| 门控目标精度 | `p_min=0.9` |
| Selector | pool=50、temperature=0、max_tokens=64、thinking off |
| Selector 失败行为 | 最多三次解析；仍失败则选择 candidate rank-1 |
| 回答引擎 | `direct`，temperature=0.7、max_tokens=2048、thinking off |
| 工具题 | 保留技能 tools 与现有 tool loop，不改调用语义 |
| 回答显著性 | paired bootstrap 10,000，seed 0 |
| 检索显著性 | paired bootstrap 5,000，seed 0 |

门控必须基于 K=2 想象重新产生 signals、阈值和 gated 文件。K=4 gate decision、
阈值或 gated retrieval 不能直接复用；通过旧/新身份门禁的 K 无关 Bare eval
可以作为 S2 校准输入复用。

K=2 retrieval 输入的冻结代码身份为：

```text
source revision:
b642012+bundle-efc7d610d25b29e28fdb14ab58da099e3e7b2b0b0b7fb55d2f7817f8306cf69f
```

本轮新增 downstream runner 在实现完成后另行计算完整 bundle SHA，并写入所有
服务器 manifest；不要求为了部署提前创建 Git commit，但任何正式输出必须能够
定位到同一份源代码。

### 3.1 结果 tag、缓存 model tag 与模型身份

`result_tag` 决定输出目录；`cache_model_tag` 必须读取 K 消融 manifest 的
`model` 字段，并原样传给 `gate.py --model`。两者不得由字符串猜测。

| result tag | cache model tag | 冻结模型 revision | 部署要求 |
|---|---|---|---|
| `qwen3.5-4b-reference` | `qwen3.5-4b` | `huggingface:Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 使用精确 HF revision |
| `qwen35-9b` | `qwen35-9b` | `huggingface:Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 使用精确 HF revision |
| `mistral7b` | `mistral7b` | `modelscope:LLM-Research/Mistral-7B-Instruct-v0.3@c8cfccbcfd71d4e3479498c30b2823bab19c4687` | 使用精确 ModelScope revision |
| `deepseek7b` | `deepseek7b` | `modelscope:deepseek-ai/deepseek-llm-7b-chat@snapshots/master` | 从原服务器复制并冻结实际 checkpoint 哈希 |
| `glm4-9b` | `glm4-9b` | `modelscope:ZhipuAI/glm-4-9b-chat@snapshots/master` | 从原服务器复制并冻结实际 checkpoint 哈希 |
| `llama31-8b` | `llama31-8b` | `modelscope:LLM-Research/Meta-Llama-3.1-8B-Instruct@snapshots/master` | 从原服务器复制并冻结实际 checkpoint 哈希 |
| `yi15-9b` | `yi15-9b` | `modelscope:01ai/Yi-1.5-9B-Chat@snapshots/master` | 从原服务器复制并冻结实际 checkpoint 哈希 |

对只记录 `snapshots/master` 的模型，若无法证明旧回答与当前 endpoint 使用相同
weights、config、tokenizer 和 chat template，则该模型的 K=4 回答不得进入同臂
复用池。禁止重新下载“当前 master”后声称它等同于旧 checkpoint。

### 3.2 中国镜像门禁

本轮所有服务器上的外部下载统一使用中国镜像，包括模型 checkpoint、Python
包、Conda 包、系统依赖和容器镜像。具体约束如下：

- Hugging Face 模型只通过已记录的中国镜像端点获取；ModelScope 模型使用其
  国内服务。无国内镜像时停止并报告，不得静默回退到海外源；
- pip、Conda、APT 和容器拉取均显式配置中国镜像，runtime manifest 记录实际
  镜像 URL、包版本和安装命令；
- 镜像只改变传输来源，不改变模型身份。模型仍须固定 repo 与精确 revision，
  并对 checkpoint、tokenizer、config 和 chat template 做身份校验；
- 已在服务器存在且通过身份门禁的本地文件直接复用，不为满足镜像规则重复
  下载；服务器间复制须校验 SHA-256；
- 下载或安装失败必须保留错误和请求目标并停止对应 staging，不允许通过海外
  地址、浮动 revision、替代模型或替代依赖版本兜底。

## 4. 精确工作量与支持集

### 4.1 已完成输入

单个 K 值包含：

```text
7 models × 5 domains × (5 fixed variants + 1 routed) = 210 result files
```

这些 K=2 retrieval 文件已经完成；正式下游使用服务器上的原始 top-50 文件，
不是只含汇总指标的 GitHub compact 包。启动前需对 210 个文件建立只读输入
manifest。

### 4.2 Gate 与装载记录

| 阶段 | pipeline / job | 逻辑记录 | 新 LLM 调用 |
|---|---:|---:|---:|
| routed signals→calibrate→apply | 7×4 = 28 pipelines，84 条命令 | 19,810 | 0 |
| Qwen fixed signals→calibrate→apply | 1×4 = 4 pipelines，12 条命令 | 2,830 | 0 |
| routed Always loading | — | 19,810 | 0 |
| routed Gated loading | — | 19,810 | 0 |
| routed Hy+Select selection-only | 5×4 = 20 jobs | 14,150 | 14,150 起 |

14,150 是 selector 决策数，不包含解析失败产生的重试请求。每次重试必须记录，
最终 manifest 同时报告 logical decisions 和实际 HTTP requests。

### 4.3 回答阶段逻辑工作量

| Arm | 回答 jobs | 逻辑回答记录 |
|---|---:|---:|
| routed Always | 28 | 19,810 |
| routed Gated | 28 | 19,810 |
| routed Hy+Select | 20 | 14,150 |
| Qwen fixed `naive_skill`+Gated | 4 | 2,830 |
| **合计** | **80** | **56,600** |

在所有 K 无关基线身份门禁通过的前提下，复用前的逻辑任务量是：

```text
56,600 answer records + 14,150 selector decisions = 70,750 logical units
```

70,750 不是 HTTP request 上限：selector 可能重试，MedCalc tool loop 也可能
为一条回答记录产生多次请求。“逻辑记录数”“实际新 answer records”和“实际
HTTP requests”必须分栏报告。实际待推理量只能在严格哈希审计结束后确定，
不预先承诺复用比例或 HTTP 调用上限。

### 4.4 严格支持集

| 比较 | 描述性完整支持 | 推断性 held-out 支持 |
|---|---:|---:|
| routed Gated vs Always | 7 模型 × 2,830 = 19,810 | 7 × 2,265 = 15,855 |
| routed Always vs Gated vs Hy+Select | 5 × 2,830 = 14,150 | 5 × 2,265 = 11,325 |
| BM25+Select vs routed Hy+Select | 5 × 2,830 = 14,150 | 5 × 2,265 = 11,325 |
| Gated vs native Rerank/native Select | 5 × 2,830 = 14,150 | 5 × 2,265 = 11,325 |
| Gated vs Bare | 7 模型结果；另给五模型共同支持结果 | 对应模型均排除共享 validation IDs |
| Qwen task component ablation | calibration 565；完整 2,830 | held-out test 2,265 |
| Qwen routed retrieval comparison | 完整 3,970 | held-out test 3,177 |

图中的 “available-model average” 不得把七模型 Gated/Always 均值直接与五模型
Rerank/Select 均值比较。manifest 必须分别提供：

- 七模型 Gated/Always average；
- 五模型四臂 strict-common-support average。

## 5. 指标与预注册比较

### 5.1 装载指标

第一阶段在回答前得到的是 provider/gate/selector 的
**decision-to-load（预期装载决定）**，不是回答引擎已经成功执行后的
`skill_ids_used`。主装载指标以预期决定计算，从而只衡量加载策略，不把 endpoint
或 tool loop 失败混进策略质量。

设总题数为 \(N\)，决定加载技能的题数为 \(L\)，决定加载 gold skill 的题数为
\(H\)：

```text
Loaded-Skill Precision = H / L
Loading Rate           = L / N
Gold-Load Rate         = H / N
```

当 \(L=0\) 时 Loaded-Skill Precision 记为不可定义，不得填零。聚合口径冻结为：

- selector `method_failure` 仍计入 \(N\)，但不计入 \(L\) 或 \(H\)，并单独报告
  Selection Failure Rate；
- per-domain：在该模型、该域内按实例计数的 micro 指标；
- per-model pooled：跨四域合并 2,830 个实例后，以 \(H/L\)、\(L/N\)、
  \(H/N\) 计算 micro 指标；
- 七模型 fleet 图：先得到每模型 pooled 指标，再对七模型作等权 macro；
- 五模型严格四臂比较：先限制到相同五模型，再对五个 per-model pooled 指标
  作等权 macro；
- 14,150/19,810 实例直接合并得到的 fleet micro 仅作为附录诊断，不替代主图
  的 model-macro 口径。

所有指标保留：

- per-model × per-domain；
- per-model pooled；
- strict-common-support fleet aggregate；
- 每题预期装载决定、gold、selected/loaded skill ID。

回答结束后逐题把预期决定与 `skill_ids_used` 对账，报告 injection match rate。
成功执行的记录必须完全一致；确定性方法失败保留预期决定、实际注入状态和失败
类别，不得反向改写第一阶段装载指标。早期交付明确标为
“decision-level loading metrics”，不能描述为引擎已成功注入的技能。

回答准确率必须另列，不能称作 loading accuracy。

### 5.2 端到端确认比较

正式报告预注册以下 contrasts：

1. K=2 Gated vs Bare；
2. K=2 Gated vs native Rerank；
3. K=2 Gated vs native Select；
4. K=2 Hy+Select vs BM25+Select；
5. K=2 Gated vs K=2 Always；
6. K=2 Gated vs K=2 Hy+Select；
7. Qwen held-out routed+Gated vs fixed `naive_skill`+Gated；
8. Qwen held-out routed+Gated vs routed+Always。

所有使用 validation-selected routed variant、K=2 gate 或 routed Hy+Select 的
推断性比较均排除共享 validation IDs，包括 candidate-source 和
Gated-vs-Hy+Select。完整 2,830 题只用于描述性主表，不能代替 held-out 推断。

### 5.3 显著性与措辞门禁

- 逐模型、逐域对比使用 paired bootstrap；
- task comparison 使用 10,000 次重采样；
- retrieval comparison 使用 5,000 次重采样；
- fleet pooled CI 使用 model→domain→instance 分层重采样，不能把 14,150 题
  当成完全独立同分布样本；
- 同时保存 point estimate、差值、95% CI、双侧 p 值和有效样本数；
- CI 包含 0 时只写“未检测到差异”，不得写“显著”“无损”或“严格更优”；
- 结果完成标准是协议和数据完整，不是分数必须提高。若 K=2 推翻旧结论，必须
  如实更新论文。

## 6. 最小代码设计

实现前先搜索并复用 `scripts/gate.py`、`scripts/export_loading.py`、
`scripts/summarize_multimodel.py`、`scripts/phase2_significance.py` 和
SR-Agents 现有 provider/engine 逻辑。新增代码保持薄、显式和单一职责。

### 6.1 纯哈希与复用逻辑

新增 `hyskill/downstream_reuse.py`，仅包含严格类型的纯函数：

- canonical JSON serialization；
- SHA-256 计算；
- selector request fingerprint；
- answer execution fingerprint；
- Always/Gated/Select 预期装载决定构造；
- 同臂 preseed eligibility 判断；
- coverage、duplicate、error 检查。

canonical JSON 固定为：

```python
json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
)
```

模块不读写文件、不调用 endpoint、不修改输入对象。

### 6.2 Selection-only

新增 `scripts/run_select_only.py`。它必须复用现有 `llm_select` 的 prompt、
候选格式、解析和 fallback 行为，逐题保存：

- instance ID；
- ordered candidate IDs；
- 完整 selector request hash；
- selected skill ID 与 rank；
- raw response；
- request/attempt 数；
- parse success 或 rank-1 fallback；
- model/runtime/code revision；
- error。

同时生成标准 top-1 retrieval JSON：

```text
<domain>-routed-select-source.json
```

后续回答使用现有 `topk(k=1) + direct`，不得再次运行 `llm_select`。

若需要修改
`external/SR-Agents/src/sragents/infer/providers/llm_select.py` 才能记录 retry
和 fallback，只允许做无行为变化的函数提取；必须用同一真实 endpoint pilot
验证拆分前后 selected skill 完全一致。

外部 endpoint 调用使用有界重试并记录结构化 warning；网络/服务重试耗尽后抛出
包含 model、instance、attempt、status 和 response body 的明确异常。selector
协议规定的“文本解析失败后回退 rank-1”与基础设施异常必须使用不同状态字段。

### 6.3 失败类别

所有 selection 和 answer 记录必须使用以下互斥类别：

- `success`：正常得到可评估输出；
- `selector_fallback`：selector 请求成功，但三次文本均无法解析，按协议选择
  rank-1；这不是基础设施错误；
- `infra_transient`：网络、HTTP 429/5xx、endpoint crash、OOM 等可重试故障；
- `method_failure`：在冻结协议下可稳定复现的 context-length overflow、工具协议
  失败或其他方法自身失败；
- `unclassified_error`：尚未归因的异常，必须阻断阶段完成。

`infra_transient` 只在有界重试后仍失败时落盘，并进入补跑清单；
`method_failure` 保留原始错误、按预注册 evaluator 计 incorrect，不重复补跑、
不删除、不排除。最终门禁要求 unresolved `infra_transient=0` 且
`unclassified_error=0`，而不是把所有确定性失败“重试到消失”。

### 6.4 同臂复用审计与 preseed

新增 `scripts/audit_k2_reuse.py`，所有输入均显式指定：

- K=2 routed/gated/selected source；
- K=4 同臂 JSONL；
- instances、corpus；
- model/runtime manifest；
- arm；
- 输出 audit 和 preseed 路径。

逐实例输出：

- `reused_same_arm`；
- `needs_inference`；
- `rejected` 与明确原因。

复用的原回答记录保持不变。来源文件 SHA、源行号、旧/新 request hash 写入独立
sidecar，不把 provenance 混进模型原始回答。

### 6.5 完整性 validator

新增 `scripts/validate_k2_downstream.py`，每个阶段和每个 arm 结束后运行。
出现以下任意问题必须非零退出：

- 缺失或重复 instance ID；
- unresolved `infra_transient` 或 `unclassified_error`；
- 没有明确 `method_failure` 分类却回答为空；
- model、arm、domain、runtime identity 不符；
- `success` 记录的 `skill_ids_used` 与 K=2 预期决定不符；
- `method_failure` 没有同时保留 expected skill IDs、actual injection state 和
  failure category；
- reused source hash 或 request hash 不符；
- selector 选择不在对应 ordered top-50；
- gate `cache_misses != 0`；
- evaluation coverage 不完整；
- strict-support denominator 不符。

现有 inference runner 会把异常写成 error record，而 resume 只看到
instance ID 就认为完成。因此正式 wrapper 必须由 validator 只为
missing/infra-transient 生成补跑输入，不能依赖原始 resume 自动修复，也不能
把确定性 `method_failure` 放入无限补跑循环。

### 6.6 编排与汇总

新增薄编排脚本 `scripts/run_k2_main.sh`：

- 必填 `RESULT_TAG`、`CACHE_MODEL_TAG`、`SERVED_MODEL`、endpoint、输入根目录、
  输出根目录和全局并发；
- `gate.py --model` 只能使用 `CACHE_MODEL_TAG`，输出目录只能使用
  `RESULT_TAG`；
- 只执行本计划需要的 gate、selection、preseed、answer、evaluate；
- 不重新生成想象或检索；
- 每个后台任务退出码都必须汇总；
- validator 通过后才进入下一阶段；
- 支持按 model/domain/arm 精确恢复，不以“文件存在”作为完成条件。

新增或扩展汇总入口，统一生成 loading、answer、significance 和 manifest。
优先抽取现有脚本中的纯逻辑复用，避免复制统计实现。

所有新 Python 函数使用严格类型、显式参数和明确异常；不使用 flag 参数让一个
函数切换多种逻辑，不设置默认参数。日志使用结构化字段。注释和 docstring
使用英文。测试以 fixture-backed integration/smoke 为主，不增加无必要 mock
或覆盖率测试。

## 7. Canonical hash 与复用规则

### 7.1 Selector request hash

至少包含：

- schema version；
- `arm=select`；
- instance ID、完整实例内容和实例 SHA；
- 完整渲染后的 selector prompt；
- ordered candidate skill IDs；
- 每个候选实际展示的 name/description；
- corpus SHA；
- model checkpoint、tokenizer、chat template revision；
- served-model name、vLLM version、dtype、context length；
- temperature=0、max_tokens=64、thinking/extra-body；
- selector prompt/parser/retry/fallback 代码 bundle SHA。

即使渲染文本相同，也必须保存 candidate IDs，因为数字位置最终映射到具体
skill。

### 7.2 Answer execution hash

至少包含：

- schema version 和 arm；
- 完整 system/user messages；
- question 与实例内容 SHA；
- loaded skill ID、完整 skill content；
- 完整 tools 定义；
- instances/corpus SHA；
- model、tokenizer、chat template、vLLM、dtype、context length；
- temperature=0.7、max_tokens=2048、thinking/extra-body；
- `prompts.py`、`direct.py`、`llm.py`、`tool_loop.py` 的代码 bundle SHA。

K、检索分数和 retrieval variant 不进入 answer request hash，因为它们不一定
进入模型实际请求；但必须记录在 provenance 中。只有模型真正收到的请求和运行
身份完全相同，K=2 才可能命中旧 K=4 回答。

### 7.3 允许的 preseed

hash 中使用规范语义 arm，而不是直接使用历史文件 label：

| 新语义 arm | Qwen3.5-4B 旧源 | 其他 fleet 旧源 | 说明 |
|---|---|---|---|
| `routed_always` | `always_r` | `always` | Qwen 的无后缀 `always` 是 fixed |
| `routed_gated` | `gated_r` | `gated` | Qwen 的无后缀 `gated` 是 fixed |
| `routed_select` | 无可用旧源 | `select`（eligible models） | Qwen 旧 `select` 是 fixed，禁止复用 |
| `fixed_gated` | `gated` | 不适用 | 仅 Qwen fixed `naive_skill` 组件线 |

source adapter 只有在 retrieval/gate provenance 与上表一致时，才可把历史 label
映射到规范语义 arm。未经映射的 `always_r` 不能因名字不同被漏掉，Qwen
`always/gated/select` 也不能因名字相同被误当成 routed。

允许的同语义复用仅为：

- K4 `routed_always` → K2 `routed_always`；
- K4 `routed_gated` → K2 `routed_gated`；
- K4 `routed_select` → K2 `routed_select`；
- K4 `fixed_gated` → K2 `fixed_gated`。

并且必须同时满足：

- answer execution hash 完全一致；
- 原记录类别为 `success`，`raw_output` 非空；
- `skill_ids_used` 与 K=2 决定一致；
- checkpoint、tokenizer、runtime identity 可证明相同；
- 复用规则不读取 correctness，不根据结果好坏选择样本。

旧 JSONL 若没有 `failure_category`，legacy adapter 只能在 `error` 缺失或为空且
`raw_output` 非空时派生 `success`；派生规则和原始行 SHA 写入 sidecar，不修改
旧文件。其他旧记录一律拒绝复用。

禁止：

- Bare → Gated；
- Always → Gated；
- fixed → routed；
- 其他“恰好注入同一技能”的跨臂复用；
- 只比较 skill ID、文件名或 served-model name 就复用。

MedCalc 技能可能进入多轮 tool loop，tools 和 tool-loop 代码身份必须进入
hash。对无法证明 immutable model/runtime identity 的旧记录，直接归入
`rejected`，不得静默降低校验标准。

## 8. 输出隔离与公开数据结构

### 8.1 服务器原始结果

K=2 检索原件保持只读：

```text
results/k-ablation/<tag>/k2/
results/k-ablation/<tag>/routed/k2/
```

新下游结果写入：

```text
results/k2-main/<tag>/
├── <domain>-routed-signals.json
├── <domain>-routed-taus.json
├── <domain>-routed-gated.json
├── <domain>-routed-always.loading.jsonl
├── <domain>-routed-gated.loading.jsonl
├── <domain>-routed-select.selection.jsonl
├── <domain>-routed-select-source.json
├── <domain>-routed-select.loading.jsonl
├── <domain>-routed-always.jsonl
├── <domain>-routed-gated.jsonl
├── <domain>-routed-select.jsonl
├── <domain>-routed-always.eval.json
├── <domain>-routed-gated.eval.json
├── <domain>-routed-select.eval.json
├── <domain>-fixed-signals.json                 # Qwen3.5-4B only
├── <domain>-fixed-taus.json                    # Qwen3.5-4B only
├── <domain>-fixed-gated.json                   # Qwen3.5-4B only
├── <domain>-fixed-gated.jsonl                  # Qwen3.5-4B only
├── <domain>-fixed-gated.eval.json              # Qwen3.5-4B only
├── audits/
│   ├── input-manifest.json
│   ├── model-runtime-manifest.json
│   ├── reuse-per-instance.jsonl
│   └── completion.json
├── logs/
└── manifest.json
```

manifest 记录每个只读输入的路径和 SHA，但不复制或修改 K 消融原件。

### 8.2 GitHub 协作包

每模型：

```text
community-results/<tag>/
├── k2/
│   ├── retrieval_top50.jsonl.gz
│   ├── router_decisions.json
│   ├── gating_per_instance.jsonl.gz
│   ├── loading_per_instance.jsonl.gz
│   ├── selection_per_instance.jsonl.gz         # eligible models only
│   ├── answer_per_instance.jsonl.gz
│   ├── answer_metrics.json
│   ├── metrics_flat.jsonl.gz
│   ├── significance.json
│   ├── reuse_manifest.json
│   ├── manifest.json
│   └── README.md
├── k4/                                         # legacy main-result archive
├── k-ablation/                                 # joint K=1/2/4/8/10 analysis
└── imagination_full_k{1,2,4,8,10}.*            # prefix caches, kept separate
```

fleet：

```text
community-results/k2-fleet/
├── loading_metrics_long.jsonl.gz
├── answer_metrics_long.jsonl.gz
├── summary.json
├── paired_comparisons.json
└── manifest.json
```

其中：

- `retrieval_top50` 保存 K=2 routed 的逐题 ordered candidates、gold 和输入
  source hash；
- `router_decisions` 保存每域 validation scores、chosen variant 和 validation
  IDs；
- `gating_per_instance` 保存 S1/S2、tau、top-1、gold、block/keep 和 calibration
  标记；
- `answer_per_instance` 至少保存每个 active arm 的 correctness、failure
  category、request hash 和 loaded skill，使 paired bootstrap 可独立复算；
- `selection_per_instance` 保存 ordered-candidate hash、选择和 fallback。

公开包必须足以从逐题证据复算 loading metrics、task accuracy、支持集与 paired
bootstrap。它保留逐题决策、指标和 provenance；不上传 checkpoint、私有
endpoint、token、原始散装缓存或服务器凭据。

### 8.3 K=4 非破坏迁移

截至本计划冻结时，历史 K=4 主实验包仍位于各模型目录根部。不得为了建立新
目录而复制 399 MB 的整个 `community-results/`，也不得在 K=2 包尚未验收时
提前移动旧数据。最终 GitHub 整理采用一次可审查的 `git mv`：

- 根部旧 `summary.json`、`retrieval_top10.jsonl.gz`、
  `retrieval_top50.jsonl.gz`、`gating_per_instance.jsonl.gz`、
  `loading_per_instance.jsonl.gz`、`metrics_flat.jsonl.gz`、
  `router_decisions.json`、`imagination_samples.jsonl.gz` 及其 K=4
  README/manifest 归档到 `<tag>/k4/`；
- 新验收通过的统一实验包只写入 `<tag>/k2/`；
- `k-ablation/`、五档 `imagination_full_k*`、以及不依赖想象 K 的
  `baselines-native/` 保持原位；
- 迁移前后逐文件 SHA-256 必须相同，gzip 必须可读，模型根目录改写为只含
  K=2/K=4/联合消融入口的索引；
- 不存在独立验证过的 K=4 fleet 包时，不得由旧文档数字拼造
  `k4-fleet/`。

完整文件映射、门禁和回滚步骤见
[GitHub K 目录迁移清单](2026-07-23-community-results-k-layout-migration.md)。

## 9. GPU 部署与并行波次

### 9.1 四个物理节点、五个 GPU 资源位

| 资源位 | GPU | 模型 | 正式任务 |
|---|---|---|---|
| S1 | A100 80GB | DeepSeek-7B → Yi-1.5-9B | Gate、Always、Gated |
| S2 | A100 80GB | Qwen3.5-4B | Gate、Select、Always、Gated、fixed 组件消融 |
| S3-0 | A100 40GB | GLM-4-9B → Mistral-7B | Gate、Select、Always、Gated |
| S3-1 | A100 40GB | Llama-3.1-8B | Gate、Select、Always、Gated |
| N1 | RTX 4090，驱动报告 49,140 MiB | Qwen3.5-9B | pilot 通过后运行 Select、Always、Gated |

当前只有四个已验证 SSH 节点和五个 GPU 槽；旧 `180.127.11.169:25720`
在认证前关闭，不能计作 N2。N1 必须先通过 BF16、8K 上下文和真实 endpoint
pilot。禁止量化、换 checkpoint、缩短正式上下文或用 TP=2 把两个可独立运行
的模型绑到一起。若后续提供第五个物理节点，必须先按服务器清单完成独立验收，
再调整调度。

### 9.2 Staging

Gate 在原服务器完成，N1 不迁移完整 `hyp_cache`。N1 只接收：

- 冻结代码和 SR-Agents；
- 四个规则域 instances 与 corpus；
- K=2 routed/gated 文件；
- K 无关 bare eval；
- selection-only 生成的 selected-skill 文件；
- 精确 checkpoint、tokenizer 和 chat template。

N1 使用 Qwen3.5-9B 精确 HF revision，并严格执行 3.2 节中国镜像门禁。
Mistral 保留在已有精确 ModelScope revision 的 S3。staging 与本地 runner
开发并行进行；任何 checkpoint 下载/复制超过 90 分钟都应报告具体瓶颈，
不能静默换模型或切换到海外源。

### 9.3 G0：Gate 与装载基线

在原服务器并行产生 K=2 gate：

- S1：DeepSeek、Yi 串行；
- S2：Qwen3.5-4B 和 Qwen3.5-9B 的 gate 数据；
- S3：GLM、Llama、Mistral 有界并行；
- Qwen3.5-4B 额外产生 fixed `naive_skill` gate。

G0 完成后立即导出七模型 Always/Gated 的三项装载指标，不等待回答。
每个模型的 Always/Gated 决策一旦通过 validator，就立即运行这两个 arm 的
同臂复用审计，产生 `needs_inference`，与其他模型的 Gate/Selection 并行。

### 9.4 L1：Selection-only

五模型按五个 GPU 槽尽量并行运行：

- S2：Qwen3.5-4B；
- S3-0：GLM-4-9B 完成后切换 Mistral-7B；
- S3-1：Llama-3.1-8B；
- N1：Qwen3.5-9B；

S1 在 DeepSeek、Yi 的 Always/Gated 复用审计通过后，同时顺序运行两模型的
`needs_inference`。L1 完成后立即验证 14,150 条 selection、审计 Select
同臂复用，并交付完整 Always/Gated/Hy+Select 装载结果。

### 9.5 A1：回答阶段

五个 eligible 模型保持 checkpoint 常驻，同卡依次运行：

1. routed Gated；
2. routed Always；
3. routed Select-answer。

四域并行、域内各臂串行；每张 GPU 使用一个全局并发预算，禁止多个臂各自开启
一套最大并发。Qwen3.5-4B 最后补 fixed `naive_skill`+Gated。

实际运行只提交复用审计中的 `needs_inference`。不得让两台机器重复运行同一个
model-domain-arm-instance。

### 9.6 V1：边跑边验收

每个 model-domain-arm 完成后立即：

1. 运行 completeness validator；
2. 只对 missing/infra-transient 输入补跑；
3. 验证 unresolved infra/unclassified 为零，并保留 method failures；
4. evaluate；
5. 写入阶段 manifest；
6. 才允许释放 endpoint 或开始该模型下一波任务。

CPU 可以在 GPU 回答期间并行执行 evaluate、装载导出、hash 和汇总。

## 10. 并发 pilot

每模型先用固定、预注册的 200 题分别测试 selector 和 direct-answer 两类负载。
pilot 题目仅用于吞吐选择，不参与结果筛选。

| GPU | 初始全局 in-flight | 对照档 |
|---|---:|---:|
| A100 80GB | 96（四域各 24） | 128（各 32） |
| A100 40GB / L40S 48GB | 48（各 12） | 64（各 16） |

选择规则：

- 以端到端 output tokens/s 为主；
- bounded retry 后 unresolved HTTP/network/OOM 与 unclassified error 必须为 0；
- deterministic method failure 不隐藏，单独计数并保留；
- 两档吞吐差不足 5% 时选择较低并发；
- 40/48GB 只有在 64 稳定后才允许测试 96；
- S3 两个 endpoint 分别绑定独立 CUDA device；
- 正式日志记录开始/结束时间、逻辑实例数、实际请求数、输入/输出 token、
  吞吐和失败数。

## 11. 阶段验收门禁

### 11.1 输入门禁

- 七模型各有 K=2 fixed 5/5 域、routed 5/5 域；
- 规则域实例数严格为 747 / 760 / 1,100 / 223；
- 每个 routed 结果实例集合完整且每题 top-50 完整；
- source SHA 与对应 K 消融 manifest 一致；
- instances、corpus、encoder、模型身份可追溯；
- manifest 同时记录 `result_tag`、`cache_model_tag`、`served_model`；Qwen
  reference 的前两者分别为 `qwen3.5-4b-reference` / `qwen3.5-4b`；
- 所有拟复用的 Bare/native baseline 通过旧/新 model/runtime identity 门禁；
- 原始 K=2 文件只读，工作树与 K=4 产物不被覆盖。

### 11.2 Gate 门禁

- 28 个 routed pipelines 和 4 个 Qwen fixed pipelines 全部完成；
- 每个 pipeline 的 signals、taus、gated ID 集合完整；
- 所有 `cache_misses=0`；
- `gate.py --model` 与对应 K 消融 manifest 的 `model` 字段完全一致；
- 阈值明确标记为 K=2 新标定；
- validation IDs 与 sorted IDs、seed 0、20% 规则一致；
- gated 文件与 routed 文件实例集合完全相同。

### 11.3 Selection 门禁

- 五模型各 2,830 条，共 14,150 个唯一 instance records；
- `success`/`selector_fallback` 的 selected skill 必须属于对应题 K=2 ordered
  top-50；
- candidate hash 与输入 routed 文件一致；
- raw response、attempt、parse/fallback、model/runtime identity 完整；
- missing、unresolved infra 和 unclassified error 均为 0；
- deterministic method failure 保留并计入端到端 incorrect；在装载统计中按
  “未形成加载决定”处理，并额外报告 failure rate；
- DeepSeek、Yi 没有伪造 Select 记录。

### 11.4 回答门禁

- 四类活跃实验共有 56,600 条逻辑回答记录；
- 每个 model-domain-arm 数量等于对应域规模；
- duplicate=0、missing=0、unresolved infra=0、unclassified error=0；
- `success` 记录的 empty=0；deterministic method failure 原样保留并按
  evaluator 计 incorrect；
- Always 的 expected decision 每题恰好一个 routed skill；
- Gated 的 expected decision 在被拦题为空，保留题恰好一个 routed top-1；
- Select 的 expected decision 与 selection-only 记录完全一致；
- 仅对 `success` 强制实际 `skill_ids_used == expected_skill_ids`；
- `method_failure` 可以为空或部分注入，但必须保留 expected、actual 和失败类别；
- fixed/routed source 不混淆；
- reused/new/rejected 数量与 sidecar、request hash 一致。

### 11.5 分析与公开包门禁

- 三项 loading 指标、task accuracy 和成本分开；
- `retrieval_top50`、router decisions、逐题 gate、selection、loading 和 answer
  correctness/failure category 全部存在；
- strict-support denominator 由程序验证；
- paired comparisons、CI、p、n 和 seed 可由公开逐题包独立复算；
- 每模型与 fleet manifest 均包含输入/输出 SHA；
- 旧 K=4 引用已更新或标为 legacy；
- missing arm 保持 unavailable，不补零；
- 分数方向不作为通过条件。

### 11.6 N1 回传门禁

- N1 结果回传到对应原服务器；
- 源端重新计算 SHA-256；
- 逐字节一致且 fleet 汇总可读后才释放 N1；
- checkpoint 不进入 GitHub，服务器地址、密码和 token 不进入日志或仓库。

## 12. 任务状态表

| ID | 状态 | 任务 | 优先级 | 相关文件 | 完成标准 | 验证结果 | 更新时间 |
|---|---|---|---|---|---|---|---|
| K2M000 | `[x]` | 冻结 K=2 统一方案 | 高 | 本计划、K 消融计划、`paper/STATE.md` | 用户确认全量 K=2、selection-only、严格同臂复用和新增两卡 | 用户已确认；本计划已记录边界 | 2026-07-23 |
| K2M001 | `[ ]` | 冻结 K=2 输入、旧基线与 model/runtime identity | 高 | `results/k-ablation/`、旧基线、各模型 manifest | 210 个 retrieval 输入和所有拟复用基线均通过身份/SHA 门禁；失败则停止并扩展计划 | 待执行 | 2026-07-23 |
| K2M002 | `[~]` | 实现 hash、selection-only、preseed、validator 和 runner | 高 | `hyskill/downstream_reuse.py`、`hyskill/loading_metrics.py`、`scripts/`、`tests/` | 静态检查、fixture-backed smoke、真实 endpoint 200 题 pilot 通过 | 14 文件已冻结并部署 S2；50 tests 与离线 smoke 通过，等待真实 Qwen4 200 题 pilot | 2026-07-23 |
| K2M003 | `[~]` | 并行 staging N1 | 高 | model/runtime manifest、部署日志 | 中国镜像下载的精确 checkpoint 与环境验收通过，输入传输 SHA 一致 | N1 已盘点并开始部署；等待 BF16/8K/endpoint pilot | 2026-07-23 |
| K2M004 | `[x]` | 计算 32 个 K=2 gate pipelines | 高 | `results/k2-main/<tag>/` | 96 条 gate 命令完成，cache_misses=0，ID 完整 | 32/32 signals、calibrate/apply 全部完成；signals 均 cache_misses=0；Mistral 四域 Bare 行数/唯一 ID/输入 ID 集合完整且 error=0、empty=0，随后 Gate 行数与唯一 ID 复验通过 | 2026-07-23 |
| K2M005 | `[ ]` | 运行五模型 selection-only | 高 | selection JSONL、selected-source JSON | 14,150 条唯一记录，infra/unclassified 为零，选择或确定性失败均可审计 | 待执行 | 2026-07-23 |
| K2M006 | `[ ]` | 交付 decision-level 装载结果 | 高 | loading per-instance、loading summary | 三项指标按冻结的 micro/model-macro 口径和严格支持集汇总，后续可与实际注入对账 | 待执行 | 2026-07-23 |
| K2M007 | `[ ]` | 同臂精确复用审计 | 高 | reuse sidecar、preseed、audit summary | 每题进入 reused/needs/rejected 且原因可追溯 | 待执行 | 2026-07-23 |
| K2M008 | `[ ]` | 五个 GPU 槽完成 K=2 回答 | 高 | 80 个 answer jobs | 56,600 条逻辑回答完整，实际新请求数有 manifest | 待执行 | 2026-07-23 |
| K2M009 | `[ ]` | 评估、显著性和 fleet 验收 | 高 | metrics、paired comparisons、fleet manifest | 支持集、CI、p、n、hash 全部机器验证 | 待执行 | 2026-07-23 |
| K2M010 | `[ ]` | 整理 GitHub 包并更新论文 | 高 | `community-results/<tag>/k2/`、`community-results/<tag>/k4/`、中英文论文、图表 | K=2/K=4 主结果目录隔离且迁移前后 SHA 一致；所有活跃实验统一 K=2，旧 K=4 不再混入结论，中英文同步 | 待执行 | 2026-07-23 |

## 13. 时间预算

仓库没有保存足以给出精确 SLA 的七模型历史墙钟，以下是保守调度区间。正式
运行必须用日志替换估计值。

| 阶段 | 预计墙钟 |
|---|---:|
| runner 开发、staging、Gate、五槽 pilot（互相重叠） | 1–2 小时 |
| 七模型 Always/Gated 装载结果 | 正式点火后 0.5–1 小时 |
| 完整 Hy+Select 装载结果 | 正式点火后 1.5–3 小时 |
| 五槽全部回答推理 | 4.5–7.5 小时 |
| evaluate、汇总、hash 与回传验收 | 0.5–1.5 小时 |
| staging 已完成后的总墙钟 | 6–10 小时 |
| 从空白 N1 开始 | 保守 7–12 小时 |

以上时间按当前五个 GPU 槽估算；Mistral 在 S3-0 形成串行尾部。若后续新增并
验收一个独立 GPU 槽，才可重新估计缩短后的墙钟。

## 14. 完成定义

本计划只有在以下条件全部满足时才能从 `[~]` 改为 `[x]`：

1. K=2 原始输入、模型和 runtime identity 全部冻结；
2. 七模型 Always/Gated 与五模型 Hy+Select 装载数据完整；
3. 80 个回答 jobs 对应的 56,600 条逻辑记录全部可追溯；
4. unresolved infra、unclassified error、无类别空输出、重复和缺失均清零，
   deterministic method failures 被完整保留并计入结果；
5. 同臂复用规则、实际新调用数和来源 hash 完整公开；
6. strict-common-support 指标和预注册显著性比较通过机器验收；
7. GitHub 的每模型 `k2/` 与 `k4/` 主结果目录完整隔离，联合 K 消融和五档
   imagination cache 保持独立，迁移前后 SHA-256 一致；
8. 当前论文活跃实验、图表和中英文文字统一为 K=2；
9. 任何不再使用的 K=4 数字已删除引用或明确标记 legacy；
10. 新结果无论正负均按真实数据更新，不以复现旧优势作为完成条件。
