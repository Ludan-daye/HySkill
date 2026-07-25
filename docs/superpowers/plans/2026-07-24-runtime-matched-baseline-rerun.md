# Runtime-Matched Baseline 重跑计划

> 日期：2026-07-24  
> 状态：`[~]` 协议、依赖图、工作量、证据格式和验收门禁已冻结；
> runtime-matched runner 已实现并通过 38 项聚焦测试，Fresh Bare、Gate
> 重校准和 changed-Gated 补跑正在并发执行。  
> K=2 环境参照：
> [2026-07-24-k2-runtime-environment.md](../2026-07-24-k2-runtime-environment.md)  
> 目标：用与正式 K=2 相同的模型/runtime/答题协议重跑 Bare、native
> Rerank 和 BM25+native Select，关闭 K2M001。

## 1. 冻结决策

1. 旧 baseline 的 48,110 条回答全部只读保留，但不进入新结果；新 baseline
   `reused_same_arm=0`。
2. 所有模型使用 K=2 formal 表中相同 checkpoint、tokenizer、chat template、
   served-model、vLLM、BF16 和 8K context。
3. 答题统一使用 native direct engine，temperature 0.7、
   `max_tokens=2048`、thinking off；不改 prompt、tools 或 tool loop。
4. Mistral 不再使用旧 32K recovery。8K 下的选择、重排或答题 overflow
   是 preregistered method failure，保留并计错。
5. DeepSeek-7B 和 Yi-1.5-9B 只重跑 Bare。50-candidate Rerank/Select 继续
   标记 `unavailable`，不填零。
6. BM25 top-50 从冻结 corpus/instances 确定性重建并与旧候选 SHA/ID 顺序
   对账；不从 compact 指标反造逐题候选。
7. 不根据 K=2 结果或 held-out 正确率为 baseline 调参；Rerank/Select 使用冻结的
   native prompt、parser、retry 和 generation 参数。分数方向不是验收条件，新
   结果即使推翻旧 baseline 优势也原样更新。
8. 每次 API attempt 保存真实 `prompt_tokens`、`completion_tokens` 和
   `total_tokens`；服务未返回 usage 时保存 `null` 和明确原因，不能用
   `max_tokens` 冒充实际消耗。
9. 不覆盖 `k2/`、`k4/`、`baselines-native/` 或旧服务器结果。使用全新的隔离
   工作树和输出目录。
10. K=2 public packs 已在固定提交 `aa5020c` 推送并逐字节 read back。baseline
    实现和远端 staging 必须从该提交建立新的隔离工作树，再叠加显式哈希的
    baseline 代码；当前未提交的 K4 migration、reader 和论文改动不得混入
    baseline 输出，也不再作为科学上无关的点火前置条件。

## 2. 三类 baseline 的精确定义

### 2.1 Bare

- provider=`none`；
- 不加载技能；
- 直接使用正式 K=2 answer engine 和参数；
- 七模型、四规则域、每模型 2,830 题。

### 2.2 Native Rerank + Always

- 用 frozen BM25 top-50 作为候选；
- 使用 SR-Agents commit `277fd8d...` 的 listwise rerank prompt；
- 候选只展示 name/description；
- temperature 0、thinking off；
- `hyskill.plugin` 将 upstream `max_tokens=4096` 固定限制为 1,024；
- 最多三次；只有成功响应但排名解析不足时才把遗漏候选按 BM25 原顺序补回；
- 加载 reranked top-1，再用正式 K=2 direct engine 答题；
- endpoint/context failure 不回退 BM25 top-1，记录 method failure 并计错。

### 2.3 BM25 + Native Select

- 用同一 frozen BM25 top-50；
- 使用与 K=2 Hy+Select 相同的 native selector prompt/parser；
- temperature 0、`max_tokens=64`、thinking off；
- 最多三次解析；解析失败 deterministic rank-1 fallback；
- context/endpoint failure不属于解析失败，不做 rank-1 fallback，记录
  method failure 并计错；
- 成功时加载所选一项，再用正式 K=2 direct engine 答题。

这样 `Hy+Select vs BM25+Select` 只改变候选源；selector、回答模型和回答协议
保持一致。

## 3. 精确工作量

| Baseline | 模型支持 | Decision jobs | Decisions | Answer jobs | Answer records |
|---|---:|---:|---:|---:|---:|
| Bare | 7 models | 0 | 0 | 28 | 19,810 |
| Native Rerank + Always | 5 models | 20 | 14,150 | 20 | 14,150 |
| BM25 + Native Select | 5 models | 20 | 14,150 | 20 | 14,150 |
| **合计** | | **40** | **28,300** | **68** | **48,110** |

计数恒等式为：

- answer jobs：`7×4 Bare + 5×4 Rerank + 5×4 Select = 68`；
- answer rows：`(7+5+5)×2,830 = 48,110`；
- decision jobs/rows：`5×4×2 = 40` jobs，
  `5×2×2,830 = 28,300` rows。

逻辑 LLM units 为 76,410，但不是 HTTP request 数。decision retry、answer retry
和 MedCalc tool loop 会增加实际调用；decision-stage method failure 则不会再
发起对应 answer call，但仍必须写一条 zero-call answer `method_failure` record，
所以 48,110 条 answer records 的完整性不受影响。

每个 eligible model 有 12 个 answer jobs、8,490 条回答；DeepSeek 和 Yi
各有 4 个 Bare jobs、2,830 条回答。

## 4. Bare 对 K=2 Gate 的前置审计

Fresh Bare 不能只用于最后一张比较表。现有 K=2 S2 gate calibration 使用了旧
Bare correctness，因此执行顺序冻结为：

```text
fresh Bare + evaluate
          |
          v
32 个 gate 重新 calibrate/apply（零 LLM）
          |
          +-- decisions 全同 --> 保留现有 K=2 answers，只更新身份审计
          |
          +-- decisions 有变 --> 仅对 request-hash 改变的 Gated rows 新推理
                                  然后重建 K=2 pack/current4
```

所有 28 个 routed gate 和 4 个 Qwen fixed gate 都要重算。即使旧 `tau2=null`，
新 Bare labels 也可能产生非空 threshold，不能只检查当前 13 个 active tau2。

### 4.1 为什么 Fresh Bare 会影响已有 Gate

对 calibration split 中的每一道题，Gate 已保存一个 `S2`，但 `tau2` 的正标签
不是检索标签，而是“Bare 在这道题上是否答对”。`gate.py` 实际计算：

```text
tau2 = largest t such that
       precision(Bare-correct | S2 < t) >= 0.9
```

因此 Fresh Bare 可能改变 calibration 标签，继而改变 `tau2`，最后让某些题从
“加载 top-1 skill”变成“不加载”，或反过来。它不是在检查 Gate 代码要不要重写，
也不是默认判定现有 K=2 answers 无效。

本地保存证据已经做过一次可执行回放：用 32 份 K=2 signals/`val_ids` 和旧
K-independent Bare correctness 重算 28 个 routed 加 4 个 fixed Gate，
`32/32` 的 `tau1/tau2` 都与保存值一致，mismatch 为 0。这证明依赖链和输入文件
完整；Fresh Bare 到达后只需替换 correctness labels 再执行同一纯 CPU 计算。
回放的源文件固定为：

- signals/taus：
  `results/k2-preservation-20260724/remote-sources/*/results/k2-main/<tag>/`；
- Bare labels：
  固定提交 `aa5020c` 的
  `community-results/<tag>/gating_per_instance.jsonl.gz` 中的
  `correct_bare`（Bare 与 `K_img` 无关；当前 K4 migration review state 中同一
  SHA 的文件位于 `<tag>/k4/`）；
- calibration membership：每份 K=2 taus 的 `val_ids`，而不是重新随机切分。

若发生决策变化：

- 不跨臂复用 fresh Bare 输出；
- 同一 Gated arm 中，加载 skill IDs、渲染 messages 和 tools 都未改变的旧答案
  可以保留；
- Gate 原因从 `skipped_s1` 变成 `skipped_s2`、但最终仍是不加载 skill 时，
  answer payload 不变，不需要重答；
- 只有 answer payload hash 改变的 rows 才重新回答、评估和写 provenance；
- 重新计算装载、answer、current4 和所有 baseline comparisons；
- GitHub K=2 pack 与论文数据必须重新发布并 read back。

最坏情况下需要重新审计 32 个 Gated jobs、22,640 个逻辑记录；实际模型调用只
针对 changed request hashes，预计远小于该上限。

## 5. Runner 与证据设计

不扩展旧 K=2 runner 的 arm choices，也不给一个新 runner 增加切换多种实验语义
的 flag。实现以下五个单一职责入口：

- `run_runtime_matched_bare.py`：Bare answer；
- `run_runtime_matched_rerank_decisions.py`：native Rerank decision；
- `run_runtime_matched_rerank_answers.py`：Rerank top-1 answer；
- `run_runtime_matched_select_decisions.py`：BM25 native Select decision；
- `run_runtime_matched_select_answers.py`：Select chosen-skill answer。

共享的 request rendering、direct-engine execution、usage normalization、
failure classification 和 manifest 纯函数放在 `hyskill/`。

Gate 审计必须区分两个哈希：

1. `answer_payload_hash`：只覆盖 instance、渲染 messages、加载 skill 内容、
   tools 和 generation params，用来判断已有 K=2 answer 是否真的需要重答；
2. `execution_request_hash`：再绑定 runtime manifest 和新代码 bundle，用于新
   baseline/changed-row 调用的完整 provenance。

不能直接用含代码 bundle 的新 execution hash 判断 Gate 是否改变，否则仅仅新增
usage instrumentation 就会错误地把所有旧 K=2 rows 标成需重跑。
payload 对比还必须强制 model、domain 和 semantic arm 一致。payload 不变的旧行
保留原 request hash、原 provenance 和原 outcome；这既包括 success，也包括现有
routed Gated 的 40 条 method failures。当前 `same_arm_preseed_eligibility()` 只
接受旧 success，不能直接拿来合并 Gate diff，必须实现专用的 Gate
merge/validator，避免把原方法失败重新随机采样。

除五个 LLM runner 外，还必须实现以下单一职责数据入口：

- `build_runtime_matched_bm25.py`：从冻结 corpus/instances 构建 deterministic
  top-50，并写 candidate manifest；
- `validate_runtime_matched_bm25.py`：核对 50 个 ordered IDs、重复、coverage、
  corpus/instance SHA 和旧候选回归差异；
- `evaluate_runtime_matched_baselines.py`：使用冻结 evaluator 评估 Bare、
  Rerank 和 Select answer records；
- `summarize_runtime_matched_baselines.py`：只读取新 runtime-matched
  per-instance eval，不再读取 legacy compact baseline；
- `audit_runtime_matched_gate.py`、`validate_runtime_matched_baselines.py` 和
  `export_runtime_matched_baselines.py`：分别负责 Gate diff、全局验收和公开包。

正式点火前必须完成：

1. request-render golden test：新 instrumentation 与 K=2 native engine 对同一
   fixture 产生逐字节相同 system/user messages、tools 和 generation params；
2. selector golden test：BM25 与 routed 两个入口只允许 source/candidate hash
   不同；
3. reranker golden test：prompt、parser、retry 和 omitted-candidate append
   与 SR-Agents `277fd8d...` 一致；
4. usage capture smoke：真实 vLLM response 的 usage 字段进入 per-attempt log；
5. failure smoke：8K overflow、parse failure、empty output 和 transient
   infrastructure error 被分到不同类别；
6. gate replay fixture：当前 32 个旧 threshold 必须全部重现，mismatch=0；
7. final-bound canary：canary 使用正式 manifest 和正式输出路径，验证后原样进入
   final rows 并 resume，不能丢弃后对同一随机样本重新调用。

每个模型、每个将运行的 arm 的 canary 固定为 20 个最终实验 IDs：四域各取
sorted-ID 前 5 个。为保持“七模型 Bare 全部先完成”的全局顺序，点火前只执行
Bare canary；Rerank/Select canary 必须等 32-Gate audit 结束后再执行。若需要覆盖
tool loop，额外选择的 MedCalc ID 也必须在 canary 前写入 manifest，并在通过后
保留为正式 row，不能临时换题。

usage capture 必须包住 OpenAI-compatible client 的
`chat.completions.create`，让原生 direct engine、selector、reranker 和 tool loop
看到完全相同的请求，同时把每个成功 response 的
`prompt_tokens/completion_tokens/total_tokens` 绑定到逻辑 attempt。服务未返回
usage 或请求失败时记录 `null` 和具体原因；不得重新 tokenizer 估算后冒充实际值。
共享 client 在多线程和 tool loop 下必须用 `contextvars.ContextVar` 或等价的
context-local call ID，分别绑定 `job_id`、`model`、`domain`、`arm`、
`instance_id`、`logical_attempt` 和 `http_subcall`；禁止用一个共享“当前
instance”变量，否则 usage 会串到别的 row。

### Job-bound runtime manifest

每个 endpoint 点火前记录：

- checkpoint repo/revision/path 和全目录 files-manifest SHA；
- tokenizer/chat-template hashes；
- served-model 与 `/v1/models` 回读；
- vLLM/Python/PyTorch/Transformers/CUDA/driver；
- GPU model、UUID 和 endpoint process command line；
- dtype、max-model-len、quantization 和 tensor-parallel；
- SR-Agents revision、answer/selector/reranker code members 与 bundle SHA；
- corpus、instances、BM25 candidates 和 evaluation code SHA；
- generation parameters；
- China-hosted mirror provenance或已验证本地 checkpoint 来源。

每条 decision/answer row 保存 `runtime_manifest_sha256`。任务结束后再生成
post-run manifest、完整 `SHA256SUMS` 和敏感信息扫描结果。

## 6. 输出目录

服务器原始输出：

```text
results/baselines-runtime-matched-v1/<tag>/
├── runtime/
├── bm25/
├── decisions/
├── answers/
├── eval/
├── audits/
└── FORMAL_COMPLETE
```

公开输出：

```text
community-results/<tag>/baselines-runtime-matched/
community-results/baselines-runtime-matched-fleet/
```

旧 `baselines-native/` 保持原位并在 README 中标记 legacy/runtime-unproven。

## 7. 五槽调度与并发

| Slot | Wave A: Bare | Wave C: native baseline | Answer jobs | Decision jobs |
|---|---|---|---:|---:|
| S1 A100 80GB | DeepSeek → Yi | 无 | 8 | 0 |
| S2 A100 80GB | Qwen3.5-4B | Qwen3.5-4B | 12 | 8 |
| S3-0 A100 40GB | GLM → Mistral | Mistral → GLM | 24 | 16 |
| S3-1 A100 40GB | Llama | Llama | 12 | 8 |
| N1 RTX 4090 | Qwen3.5-9B | Qwen3.5-9B | 12 | 8 |

S3-0 在 Wave A 结束时已驻留 Mistral，因此 Wave C 从 Mistral 开始再切回 GLM，
比两波都写 `GLM→Mistral` 少一次模型重载。

使用三个全局 wave，不能在某个快模型的 Bare 完成后提前点火该模型的 native
baseline：

1. **Wave A — Fresh Bare**：七模型 endpoint/runtime preflight、Bare
   final-bound canary、28 个 Bare jobs、evaluate 和 completeness；
2. **Wave B — Gate integrity barrier**：用全部 Fresh Bare 重算 32 个 Gate，
   输出逐题 tau/decision/payload diff；若 payload 有变化，受影响模型先完成
   changed-row Gated 补跑，最终统计前完成全局 K=2 pack/current4 重建与验证；
3. **Wave C — Native baselines**：五模型 Rerank/Select final-bound canary，
   40 个 decision jobs、40 个 non-Bare answer jobs、evaluate/manifest。

wave 内不同模型并行。S1 的 DeepSeek→Yi 和 S3-0 的 GLM→Mistral 是两个串行
链；Wave A 的主要尾部预计是 S1，Wave C 的主要尾部预计是 S3-0 或慢速 Qwen
endpoint。全局 barrier 会牺牲少量 GPU overlap，但保证 Fresh Bare→32 Gate→其余
baseline 的审计顺序没有歧义。

### 7.1 阶段内部并发

1. 四个物理节点的 read-only preflight 并行执行；每个模型 endpoint 的 identity
   验证仍必须独立出具 manifest。
2. Wave A 中五个 GPU slot 同时工作；每个 endpoint 的四域 Bare 队列并发，但只
   使用一个由 canary 实测确定的 endpoint-wide in-flight cap，不能把一个安全的
   worker 数机械地乘四。
3. 一个 Bare 域完成后立即在 CPU 上 evaluate/validate；该模型四域完成后可提前
   预计算自己的 Gate diff，但 Wave C 仍等全部 32 Gate diff 出齐。
4. 32 个 Gate 在本地 CPU process pool 并发重放。全局 diff 完成后，
   `payload_change=0` 的模型立即进入 Wave C；有变化的模型先补自己的 changed
   Gated rows，再进入该模型的 Wave C。所有最终统计仍等待 K=2 patch 完整验证。
5. Wave C 中 Rerank 和 Select decision 使用两个独立队列并发提交；总并发仍受
   endpoint-wide cap 约束。某个 domain 的 decision file 完成且 validator 通过后，
   立即启动同一 domain 的 answer job，不等待其他 domain 或另一 decision arm。
6. 每个模型完成后立即并行执行 evaluate、gzip、SHA、敏感扫描和传回；fleet
   bootstrap、最终 manifest 与 GitHub fixed-commit readback 等最后一个模型。

runner 实现也按文件所有权并发：共享 usage/manifest/Bare、Rerank/BM25、
Select/evaluator/comparison 三条实现流并行，主流程负责 Gate merge、全局 validator
和集成。预计 3–6 engineer-hours 可压到约 1.5–3 h 墙钟，但最终集成测试不能省略。

### 7.2 2026-07-24 23:55 CST 实时并发调度

实际执行在不改变科学依赖的前提下，把模型内可证明独立的工作提前：

| Lane | 当前任务 | 已验证状态 | 当前尾部 |
|---|---|---|---|
| S1 A100 80GB | Yi Fresh Bare | final-bound canary 20/20 success；2,064/2,830 rows | 约 20–30 min |
| S5 GPU | Llama Fresh Bare | 四域 full 正常；1,604/2,830 rows；usage null=0 | 约 15–18 min |
| N2 GPU | Mistral checkpoint identity/点火 | 冻结 revision 与运行路径正在做最终校验 | 尚未计入确定 ETA |
| S2 A100 80GB | Qwen4 changed Gated | 15/15 canary success；474 rows 总量，459 rows 正在 resume | 与用户训练并存，endpoint-wide cap=6 |
| N1 RTX 4090 | Qwen9 changed Gated | 10/10 canary success；109 rows 总量，99 rows 正在 resume | endpoint-wide cap=6 |
| Local CPU | eval、completeness、Gate audit、runner tests | 4/7 Fresh Bare 已闭环；20/32 Gate valid；38 tests passed | 不占 GPU |

当前 Fresh Bare 已完整闭环 DeepSeek、GLM、Qwen4 和 Qwen9，共
11,320/19,810 rows。对应 20 个 Gate task 已审计：DeepSeek/GLM 的 payload
change 均为 0；Qwen4 为 474，Qwen9 为 109。

模型之间的 Gate calibration 彼此不共享 Bare labels 或 thresholds。因此某模型的
Fresh Bare、eval 和全部本模型 Gate audit 通过后，可以提前补该模型自己的 changed
Gated rows；这不会读取其他模型尚未完成的结果。全局 Wave C 仍保持原 barrier：
必须等七模型 Fresh Bare、32/32 Gate audit 和全部 changed-Gated merge 验证通过，
才启动 Rerank/Select。S2/N1 在 changed rows 完成前不切模型。

## 8. 推断与统计

运行完整 2,830 题用于描述表和公开 pack；正式比较只用 frozen held-out：

| Contrast | Support | Held-out n |
|---|---:|---:|
| Gated vs Bare | 7 models | 15,855 |
| Gated vs native Rerank | 5 models | 11,325 |
| Gated vs BM25 Select | 5 models | 11,325 |
| Hy+Select vs BM25 Select | 5 models | 11,325 |

使用 10,000 bootstrap samples、seed 0，fleet 按
model→domain→paired instance 分层重采样。DeepSeek/Yi 的 Rerank/Select
保持 unavailable；不把七模型 Gated 均值与五模型 baseline 均值直接相减。

旧四项 baseline contrast 只作为回归诊断，不是新结果必须达到的目标。

## 9. 验收门禁

### Runtime

- 七模型逐项匹配 K=2 runtime identity；
- context=8192、dtype=BF16、quantization=none；
- checkpoint/tokenizer/chat template SHA 全部通过；
- endpoint process、GPU 和包版本 job-bound；
- Mistral 不存在 32K recovery rows。

### Completeness

- 28,300 decisions 和 48,110 answer records 数量完整；
- duplicate=0、missing=0、unresolved infra=0、unclassified=0；
- deterministic method failures保留并计错；
- Select/Rerank unsupported cells不存在伪造记录；
- all baseline answer rows `reused_same_arm=0`。

### Reproducibility

- actual usage、attempts、request hash、raw output、evaluation 和 runtime
  manifest 全部绑定；
- gzip/JSON/JSONL/schema/ID/SHA 验证通过；
- 四项 held-out comparison 可从 public per-instance files 独立复算；
- 重新检查 Gate 后，K=2 pack 要么逐字节保持，要么按 changed rows 完整重建；
- GitHub commit 固定下载回读与本地逐字节一致；
- 论文只使用新 runtime-matched baseline。

## 10. 时间预算

Gate 代码检查和离线重算本身不需要 2 小时：当前 32 个 Gate 的回放在本地为秒级，
Fresh Bare 到齐后的正式 recalibrate/apply/diff 预计 5–15 分钟。耗时来自
19,810 条 Fresh Bare 和后续 28,300 条带 skill 的 baseline answers。

| 阶段 | 预计墙钟 |
|---|---:|
| 并发实现 runners、BM25/evaluator/comparison、usage/provenance、golden/smoke | 1.5–3 h 墙钟（3–6 engineer-hours） |
| 四节点并行 preflight、endpoint staging、分阶段 final-bound canary | 0.5–1.0 h，可重叠 |
| Fresh Bare 19,810 answers | 1.5–4 h |
| 32 Gate recalibration/apply/payload diff | 5–15 min |
| 并发/流水化 Rerank+Select 28,300 decisions + 28,300 answers | 5–10 h |
| 增量 evaluate/pack 已重叠；最终 bootstrap、manifest、GitHub readback | 0.5–1.5 h |
| **无 Gate payload 变化时，从 runner 实现开始** | **约 9–20 h** |
| **runner/preflight 完成后，纯正式执行** | **约 7–17 h** |

这个区间不是把“看代码”估成数小时。保存的 K=2 attempt logs 中，eligible
models 的 selector response 中位数约为 7.4–28.9 秒，answer response 中位数约为
12.2–52.9 秒；Yi 的可用 answer log 样本较少且中位数约 140 秒。旧 Rerank 没有
同等级完整 timing evidence，因此最终墙钟仍需由 final-bound canary 收紧。

Gate changed-row contingency 不能在看到 diff 前固定写成 0–4 小时：

- payload change=0：额外 0；
- 不超过 5%（最多 1,132/22,640 rows）：通常约 0.25–1.5 h；
- 5%–25%：约 1–5 h；
- 更高比例或集中在慢模型：立即按实际模型/域/rows 重估，不能声称仍在 4 小时内。

以上假设五个 GPU 槽可用、checkpoint 无需重新下载且 endpoint 无环境漂移。

## 11. 任务表

| ID | 状态 | 任务 | 优先级 | 相关文件 | 完成标准 | 验证结果 | 更新时间 |
|---|---|---|---|---|---|---|---|
| BMR000 | `[x]` | 冻结 K=2 reference environment | 高 | 环境冻结表、formal rows | 七模型 runtime/tokenizer/chat-template/code identity 唯一且完整 | 已从 56,600 answer rows 和 14,150 selection rows核对 | 2026-07-24 |
| BMR001 | `[x]` | 冻结 baseline 协议与工作量 | 高 | 本计划、环境冻结表、`gate.py`、保存 signals/taus | 三臂、8K failure、Bare→Gate 审计、hash 边界与 68 jobs 均明确 | 68 jobs/48,110 rows 算式复核；32/32 Gate threshold 回放一致 | 2026-07-24 |
| BMR002 | `[~]` | 实现 runner、BM25、evaluator、comparison、usage 与 manifest | 高 | `hyskill/`、`scripts/`、`tests/` | 五个 runner、BM25 builder/validator、evaluator、新 comparison adapter、context-local usage、Gate merge 的 golden/smoke 通过 | changed-Gated 专用 runner 的 artifact inventory、usage、final-bound canary/resume 已覆盖；runtime-matched 聚焦测试 38 passed | 2026-07-24 |
| BMR003 | `[~]` | 五槽 runtime preflight | 高 | server runtime manifests | endpoint 与 K=2 身份逐项相同 | DeepSeek、GLM、Qwen4、Qwen9 已随完整 Bare 闭环；Yi 8K endpoint/canary 已通过；Llama 正在 full；Mistral 最终 identity 校验中 | 2026-07-24 |
| BMR004 | `[~]` | Fresh Bare | 高 | 28 answer jobs | 19,810 rows，fresh-only，eval 完整 | 4/7 模型、11,320/19,810 rows 已完成并通过 eval/completeness；Yi 2,064/2,830、Llama 1,604/2,830 正在运行 | 2026-07-24 |
| BMR005 | `[~]` | Gate recalibration audit | 高 | 32 tau/gate/payload diffs | payload hash 逐题相同，或 changed rows 完整补跑；不能因新代码 hash 误判 | 20/32 audit valid；DeepSeek/GLM change=0；Qwen4=474、Qwen9=109；25/25 changed-row canary success，剩余 558 正在 resume | 2026-07-24 |
| BMR006 | `[ ]` | Fresh Rerank/Select decisions | 高 | 40 decision jobs | 28,300 rows，usage/failure/provenance 完整 | 未开始 | 2026-07-24 |
| BMR007 | `[ ]` | Fresh baseline answers | 高 | 40 non-Bare answer jobs | 28,300 rows；总 baseline=48,110 | 未开始 | 2026-07-24 |
| BMR008 | `[~]` | 评估与四项比较 | 高 | eval、paired comparisons | held-out 支持、CI、p、seed 可复算 | 已完成 4/7 模型的 16 份 Bare eval；最终四项比较等待全部 baseline arms | 2026-07-24 |
| BMR009 | `[ ]` | Public pack、GitHub、论文 | 高 | community-results、双语论文 | SHA/readback/敏感扫描通过，新 baseline 完全替换旧 claim | 未开始 | 2026-07-24 |
