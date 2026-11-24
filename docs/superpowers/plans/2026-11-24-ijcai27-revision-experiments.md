# IJCAI-27 修改实验 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 2026-12-21 前产出 E0–E7（E8 可选）的结果和可提交的证据包，支撑新定位"想象提升候选池质量，路由器按领域选粒度，整体显著优于 HyDE 原版"。

**Architecture:** 
- **重算部分（E0–E6）**：在主仓库新分支 `ijcai27-revision` 上实现。核心是一个"缓存生成器"：它把已提交的 `imagination_full_k2` 喂给现有检索器，在本地重算所有配置的逐题排名。每题的指标行提交到 `community-results/`；完整排名只留在 gitignored 的 `results/revision-ijcai27/`。
- **新推理部分（E7）**：在基线工作树分支 `codex/runtime-matched-baselines-20260724` 上切出新分支，扩展现有的 runtime-matched 重排流水线，新增 `hy_rerank` 臂。协议不变，只换候选来源。

**Tech Stack:** Python 3.12（重建的 `.venv`）、numpy、sentence-transformers、`BAAI/bge-base-en-v1.5`（E4 另加 `thenlper/gte-base`、`BAAI/bge-large-en-v1.5`）、vLLM（远程）、pytest。

**Spec:** `docs/superpowers/specs/2026-11-24-ijcai27-revision-experiments-design.md`（`6e4f0d9`）

## Global Constraints

- `unavailable` 不等于 0：DeepSeek-7B 和 Yi-1.5-9B 没有 Select/Rerank 单元格，不进 E1/E7 的表。
- `method_failure` 行留在分母里，按错误计分。
- 置信区间含 0 时只写"未检测到差异"。
- 不得把七模型平均和五模型平均放在一起比较。
- 加载精度、加载率、gold 加载率、答题准确率是四个不同的量。
- 验证集是 `hyskill.k_ablation.validation_ids(frozenset(domain_instance_ids))`，按领域划分，占 20%，seed 为 0。所有测试报告都要排除这些 id。
- 统计：分层 bootstrap，依次重采样模型、领域、题目；10,000 次；基础 seed 为 0。每个比较的 seed 用 `comparison_seed(0, name)` 派生，与 `scripts/summarize_k_ablation.py:581` 的实现一致。
- 指标定义：
  - nDCG@k 和 Recall@k 与 `hyskill.k_ablation.metric_for_record` 完全一致。
  - hit@k 指前 k 名中至少有一个 gold。
- E7 协议：冻结的 SR-Agents `277fd8d` 提示词、`max_tokens=1024` 补丁和 K=2 运行参数都不变，只把候选来源从 BM25 top-50 换成路由后的 HySkill top-50。
- 不得写入或改动 `k2/`、`k4/`、`k-ablation/`、`baselines-native/`、`baselines-runtime-matched/` 和原始服务器输出。新证据只放在：
  - `community-results/<TAG>/revision-ijcai27/`
  - `community-results/revision-ijcai27-fleet/`
- 下载走国内镜像，PyPI 用清华源，模型用 hf-mirror 或 ModelScope。记录确切 revision 和每个权重文件的 SHA-256，不得静默回退到境外源。
- 远程服务器：先只读检查主机名、进程、GPU、磁盘和结果路径；不得停止别人的服务；每次传输都重算 SHA。
- Git：
  - 只按白名单 `git add`，一个逻辑改动一个提交。
  - 提交信息用小写范围前缀，并以 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` 结尾。
  - 不经用户同意不 push。
- 任何一项结果不利时，按 spec §7 的写法如实报告，不得改协议或挑子集。

## Review Focus

1. **验证集泄漏进测试表。** 每个汇总脚本都要断言测试集题数：四个规则领域每个模型 2,265 题；五个领域时每个领域的题数等于总数减去 `validation_ids` 的数量。由 Task 3 的 `test_split_counts` 覆盖。
2. **不同来源的题目集合不一致**（Qwen3.5-4B 的 BM25 差异可能就是这个原因）。比较任意两臂之前，必须断言 instance_id 集合完全相同，不一致就报错，不能取交集。由 Task 3 的 `test_paired_requires_identical_ids` 覆盖。
3. **多个 gold 技能。** hit@k 取"任一命中"；nDCG 的 IDCG 用 `min(len(gold), depth)`。由 Task 1 的 `test_metrics_multi_gold` 覆盖。
4. **重排补位。** 补位位置在严格口径下算作未命中；补位数缺失时报错，不默认为 0。由 Task 4 的 `test_strict_hit_ignores_filled_positions` 和 `test_missing_omitted_count_raises` 覆盖。
5. **Mac（MPS/CPU）和 CUDA 的浮点差异导致名次互换。** E0 的门槛是三位小数的指标一致，另外报告 top-50 id 的重合率；不要求逐 id 完全一致。由 Task 2 的 `test_gate_tolerates_tie_reorder` 覆盖。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `hyskill/revision_rankings.py` | 缓存生成器；重算某个模型、领域、配置的排名；加载已提交的 BM25、Rerank、路由排名；计算逐题指标 |
| `hyskill/revision_stats.py` | 测试集划分、配对对齐、分层 bootstrap |
| `scripts/recompute_revision_rankings.py` | CLI：一个模型 × 一个编码器，写出指标行和本地排名 |
| `scripts/check_revision_reproduction.py` | E0 复现门槛和 BM25 差异报告 |
| `scripts/summarize_revision_depth.py` | E1 |
| `scripts/summarize_revision_router.py` | E2、E3、E6 |
| `scripts/summarize_revision_mechanism.py` | E5 |
| `scripts/summarize_revision_encoders.py` | E4 |
| `tests/test_revision_rankings.py`、`tests/test_revision_stats.py`、`tests/test_revision_summaries.py` | 对应测试 |
| 基线工作树 `hyskill/runtime_matched_rerank.py` 等 | E7：`hy_rerank` 臂 |

---

### Task 0: 环境与分支

**Files:** 无需提交的代码；记录写进 Task 2 的 reproduction manifest。

- [ ] **Step 1:** 在主仓库执行 `git switch -c ijcai27-revision`（从 `6e4f0d9` 开始）。
- [ ] **Step 2:** 用 `python3.12 -m venv .venv` 重建环境，再执行 `.venv/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev,full]" -e external/SR-Agents`。
- [ ] **Step 3:** 从镜像下载 `BAAI/bge-base-en-v1.5`，记录 revision 和每个权重文件的 SHA-256，写到 `results/revision-ijcai27/encoders.json`。
- [ ] **Step 4:** 运行 `.venv/bin/pytest -q`，预期 `68 passed`，只有在 sragents 和 sentence-transformers 装好后原本 skip 的那一项才会转为通过。之后运行 `./scripts/smoke.sh`，必须通过。

### Task 1: 缓存生成器与排名重算

**Files:**
- Create: `hyskill/revision_rankings.py`、`scripts/recompute_revision_rankings.py`
- Test: `tests/test_revision_rankings.py`（沿用 `tests/test_naive_hyde.py` 里的 `FakeST` 和 `tests/fixtures/tiny_corpus.json`）

**Interfaces:**
- Produces:
  - `class CachedImaginationGenerator`：构造参数为 `(rows: dict[str, dict[str, list[str]]], template: str, k: int)`。`rows` 的键是查询文本，值是 `imaginations[template]`。`generate(query: str) -> list[str]` 返回前 k 条；查询不在缓存里时抛 `KeyError`，因为不允许回退去现场生成。`n_failures = 0`。
  - `load_imagination_cache(path: Path) -> dict[str, dict[str, list[str]]]`
  - `RankingRow = TypedDict("RankingRow", {"model": str, "domain": str, "arm": str, "instance_id": str, "gold_skill_ids": list[str], "ranked_ids": list[str], "omitted_count": int | None})`
  - `recompute_rankings(tag: str, domain: str, variant: str, encoder_name: str, k: int, corpus_path: Path, instances_dir: Path, cache_path: Path, emb_cache_dir: Path, st_model: object | None = None) -> list[RankingRow]`
    - `variant` 必须属于 `k_ablation.FIXED_VARIANTS`；
    - 检索器按 `k_ablation.VARIANT_RETRIEVERS` 构造，参数与 `hyskill/plugin.py` 的工厂默认值一致；
    - `top_k=50`。
  - `route(rows_by_variant: dict[str, list[RankingRow]], validation: frozenset[str]) -> tuple[str, list[RankingRow]]`：按验证集 nDCG@10 选配置，平局规则复用 `k_ablation.expected_router_pick`；返回的行把 arm 改为 `"routed"`。
  - `load_committed_rankings(tag: str, arm: Literal["bm25", "llm_rerank", "routed_k2"]) -> list[RankingRow]`：
    - `bm25` 和 `llm_rerank` 读 `<TAG>/k4/retrieval_top50.jsonl.gz`，字段为 `gold` 和 `top50` 三元组；
    - `routed_k2` 读 `<TAG>/k2/retrieval_top50.jsonl.gz`。
  - `load_runtime_rerank_rankings(raw_root: Path, tag: str) -> list[RankingRow]`：读基线工作树的 `results/baselines-unified-20260725/<TAG>/decisions/*always_rerank*.jsonl`，取 `reranked_candidate_ids` 和 `omitted_candidate_count`。
  - `instance_metrics(row: RankingRow) -> dict[str, float]`：键为 `hit@{1,3,5,10,20,50}`、`Recall@{1,5,10,50}`、`nDCG@{1,5,10,50}`。

- [ ] **Step 1: 写失败测试**
  - `test_cached_generator_returns_prefix`：缓存里有 4 条 skill 想象，`k=2` 时返回前 2 条。
  - `test_cached_generator_missing_query_raises`：不在缓存里的查询抛 `KeyError`。
  - `test_metrics_multi_gold`：gold 为 `["a","b"]`，排名为 `["x","b","a"]`。断言 `hit@1 == 0`、`hit@3 == 1`、`Recall@5 == 1.0`，`nDCG@10 == (1/log2(3) + 1/log2(4)) / (1 + 1/log2(3))`。
  - `test_recompute_naive_passage_tiny`：用 FakeST 和 tiny corpus，结果有 50 个以内的 id、按分数降序、不重复。
  - `test_route_picks_validation_winner`：两个配置中验证集 nDCG@10 更高的那个被选中。
  - `test_load_committed_rankings_schemas`：用两条 fixture 行，分别覆盖 k4 三元组格式和 k2 的 `retrieved` 格式。
- [ ] **Step 2:** 运行 `.venv/bin/pytest tests/test_revision_rankings.py -q`，预期 FAIL（模块不存在）。
- [ ] **Step 3:** 实现上面列出的函数。
- [ ] **Step 4:** 实现 `scripts/recompute_revision_rankings.py`。
  - 参数全部必填：`--tag --encoder --k 2 --corpus --instances-dir --cache --emb-cache-dir --out-metrics --out-rankings`。
  - 依次跑五个配置和路由。
  - 逐题指标行写到 `--out-metrics`（`.jsonl.gz`，含 `split` 字段），完整排名写到 `--out-rankings`，两者都附 manifest：输入 SHA、编码器 revision、代码 commit。
- [ ] **Step 5:** 运行测试，预期 PASS。
- [ ] **Step 6:** 提交：`git add hyskill/revision_rankings.py scripts/recompute_revision_rankings.py tests/test_revision_rankings.py tests/fixtures/<新增 fixture>`，提交信息 `feat: recompute variant rankings from committed imagination caches`。

### Task 2: E0 复现门槛

**Files:**
- Create: `scripts/check_revision_reproduction.py`
- Test: 并入 `tests/test_revision_rankings.py`

**Interfaces:**
- Consumes: Task 1 的指标行、`load_committed_rankings`。
- Produces: `community-results/<TAG>/revision-ijcai27/reproduction.json`，字段为 `{"passed": bool, "metric_mismatches": [...], "router_pick_mismatches": [...], "routed_top50_overlap": {domain: float}, "bm25_report": {...}}`。退出码为 0 当且仅当 `passed`。

- [ ] **Step 1: 写失败测试**
  - `test_gate_passes_on_equal_metrics`
  - `test_gate_fails_on_fourth_decimal_only_when_third_differs`：0.5034 对 0.5031 通过，0.504 对 0.503 失败，比较方式是 `round(x, 3)`。
  - `test_gate_tolerates_tie_reorder`：两个同分 id 互换位置，指标不变，判定通过。
- [ ] **Step 2:** 运行测试，预期 FAIL。
- [ ] **Step 3:** 实现脚本。
  - 对照 `k-ablation/metrics_long.jsonl.gz` 中 `k_samples == 2` 的全部 `(variant, domain, split, metric)` 行。
  - 路由选择对照 `k2/router_decisions.json` 的 `domains[d].pick`。
  - BM25 报告：对七个模型分别用 k4 的 `bm25` 行算 test-split hit@50 和 nDCG@10，并比较各模型的 instance_id 集合与 gold 是否相同。差异的原因写进 `bm25_report.explanation`；查不出原因就写 `"unexplained"`，并把 `passed` 设为 false。
- [ ] **Step 4:** 测试通过后，对七个模型运行 Task 1 的 CLI（bge-base）和本脚本。**七个都 `passed` 才能进入 Task 4–8。** 只要有一个失败，就停下来报告给用户。
- [ ] **Step 5:** 提交代码，以及七个 `reproduction.json` 和对应的指标行文件。提交信息 `data: add bge-base recomputation and reproduction gate`。

### Task 3: 统计模块

**Files:**
- Create: `hyskill/revision_stats.py`
- Test: `tests/test_revision_stats.py`

**Interfaces:**
- Produces:
  - `test_ids(domain_ids: frozenset[str]) -> frozenset[str]`：返回 `domain_ids - validation_ids(domain_ids)`。
  - `paired_cells(rows_a: list[dict], rows_b: list[dict], metric: str, ids: Callable[[str, str], frozenset[str]]) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]`：键为 `(model, domain)`。任何一格两边 id 集合不同就抛 `ValueError`。
  - `hierarchical_paired_bootstrap(cells: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], samples: int, seed: int, name: str) -> dict[str, object]`：算法照搬基线工作树 `scripts/summarize_runtime_matched_baselines.py:927` 的 `hierarchical_bootstrap`（依次重采样模型、领域、题目）。返回 `mean_a`、`mean_b`、`difference`、`ci95`、`p_two_sided`、`n_models`、`n_domains`、`n_instances`、`samples`、`seed`。
  - `comparison_seed(base_seed: int, key: str) -> int`：与 `scripts/summarize_k_ablation.py:581` 的实现逐字一致。

- [ ] **Step 1: 写失败测试**
  - `test_split_counts`：用 theoremqa（747 题）的 id，断言 `len(test_ids) == 747 - 149`。
  - `test_paired_requires_identical_ids`
  - `test_bootstrap_zero_difference`：两边相同，结果 `ci95 == [0.0, 0.0]`。
  - `test_bootstrap_deterministic`：同一个 seed 和 name，两次结果相同。
  - `test_comparison_seed_matches_k_ablation`：与 `scripts.summarize_k_ablation.comparison_seed` 的结果相等。
- [ ] **Step 2:** 运行，预期 FAIL；**Step 3:** 实现；**Step 4:** 运行，预期 PASS。
- [ ] **Step 5:** 提交，信息 `feat: add paired hierarchical bootstrap for revision analyses`。

### Task 4: E1 分深度命中率表

**Files:**
- Create: `scripts/summarize_revision_depth.py`
- Test: `tests/test_revision_summaries.py`

**口径：**
- 范围：五个模型（glm4-9b、llama31-8b、mistral7b、qwen3.5-4b-reference、qwen35-9b），只用测试集。
- **主表**：四个规则领域，三臂为 BM25（k4 `bm25`）、Rerank（`load_runtime_rerank_rankings`，下游实际用的就是这次重排）、路由后的 HySkill（`routed_k2`）。
- **补充行**：BigCodeBench 用 k4 的 `llm_rerank`。它没有补位信息，在表注里写明。
- 这是对 spec §6 E1 的细化：只有 runtime-matched 重排记录了补位数，所以主表用四个领域；第五个领域作为补充。

**Interfaces:**
- Produces: `community-results/revision-ijcai27-fleet/e1_depth.json` 和 `e1_depth.md`。内容是每个深度、每臂的均值，Hy 对 Rerank、Hy 对 BM25 的 `hierarchical_paired_bootstrap` 结果，每个深度的补位比例，以及严格口径的结果。
- `strict_hit(row: RankingRow, k: int) -> float`：只看前 `50 - omitted_count` 个位置；`omitted_count is None` 时抛 `ValueError`。

- [ ] **Step 1: 写失败测试**
  - `test_strict_hit_ignores_filled_positions`：gold 在第 48 位，`omitted_count=5`，此时 strict hit@50 为 0，原样 hit@50 为 1。
  - `test_missing_omitted_count_raises`
- [ ] **Step 2–4:** 测试先失败，然后实现，再通过。
- [ ] **Step 5:** 运行脚本产出结果，把各深度的差值和 CI 与会话中的初步数字对照：k=10 约 +.100 [+.039, +.159]。主表改为四个领域后数字会变；如有明显出入，在 `e1_depth.md` 中说明原因。
- [ ] **Step 6:** 提交，信息 `data: add held-out depth table for candidate pools`。

### Task 5: E2、E3、E6 路由与 HyDE 对比

**Files:**
- Create: `scripts/summarize_revision_router.py`
- Test: 并入 `tests/test_revision_summaries.py`

**口径：** 七个模型 × 五个领域，测试集，指标 nDCG@10 和 Recall@50，数据来自 Task 1 的 bge-base 指标行。
- **E2**：`routed` 对 `naive_passage`。输出中加字段 `passage_pick_count`，统计 35 个组合里路由选了 passage 的次数（预期 12）。
- **E3**：
  - (a) 全局固定配置：按七模型 × 五领域合并后的测试集 nDCG@10 选出一个配置；
  - (b) 每领域固定配置：按该领域七个模型合并后的测试集选；
  - 另报：路由选择分布；`hyskill − naive_passage` 在每个领域的差值。
- **E6a**：用 `hyskill.fusion.rrf` 融合五个配置的排名，`k=60, top_k=50`，完整排名从 `results/` 读取。
- **E6b**：留一个领域。对每个模型，在其余四个领域的验证集上合并计算 nDCG@10，选出最好的配置，在留出领域的测试集上评估。与路由结果、E3(b) 一起报告。

**Interfaces:**
- Produces: `community-results/revision-ijcai27-fleet/e2_hyde.json`、`e3_router.json`、`e6_label_free.json`，以及一个合并的 `router.md`。
- `best_fixed_global(metrics_rows: list[dict]) -> str`、`best_fixed_per_domain(metrics_rows: list[dict]) -> dict[str, str]`、`lodo_pick(metrics_rows: list[dict], model: str, held_out: str) -> str`。

- [ ] **Step 1: 写失败测试**
  - `test_best_fixed_global_pooled`：用构造数据，验证合并后的均值决定选择，而不是各格均值的平均。
  - `test_lodo_excludes_held_out_domain`：留出领域的验证集数据改成极端值，选择结果不变。
- [ ] **Step 2–4:** 测试先失败，然后实现，再通过。
- [ ] **Step 5:** 运行脚本。与初步数字对照：E2 nDCG@10 约 +.045 [+.011, +.094]，Recall@50 约 +.057。
- [ ] **Step 6:** 提交，信息 `data: add HyDE, router and label-free comparisons`。

### Task 6: E5 机制证据

**Files:**
- Create: `scripts/summarize_revision_mechanism.py`
- Test: 并入 `tests/test_revision_summaries.py`

**口径：**
- 对每道题计算：
  - `cos(query, gold)`；
  - 对每个模板 t ∈ {skill, passage, sentence}，`cos(mean(imagination_t[:2]), gold)`。均值向量做 L2 归一化，不混入查询向量。
- gold 向量用 naive_hyde 全文索引里同一份语料文本的嵌入，可以直接读 Task 1 的 npz 缓存。多个 gold 时取最大值。
- 七个模型 × 五个领域，测试集。按领域报告均值，以及"想象减查询"的配对 CI。另报路由所选模板的汇总结果。

**Interfaces:**
- Produces: `community-results/revision-ijcai27-fleet/e5_mechanism.json`、`e5_mechanism.md`。
- `gold_similarity(vec: np.ndarray, gold_vecs: np.ndarray) -> float`

- [ ] **Step 1:** 写失败测试 `test_gold_similarity_takes_max`、`test_template_mean_excludes_query`。
- [ ] **Step 2–4:** 测试先失败，然后实现，再通过。
- [ ] **Step 5:** 运行脚本，然后提交，信息 `data: add query vs imagination gold-similarity analysis`。

### Task 7: E4 换编码器

**Files:**
- Create: `scripts/summarize_revision_encoders.py`
- Modify: 无。复用 Task 1 的 CLI，传 `--encoder`。

**口径：**
- 编码器选 `thenlper/gte-base` 和 `BAAI/bge-large-en-v1.5`，两者都不需要 e5 那样的前缀，可以直接用现有的 `Embedder`。
- 下载要求同 Task 0 Step 3。
- 七个模型都跑；路由在新编码器的验证集上重新选择。
- BM25 与编码器无关，直接沿用。
- 汇总：每个编码器一张表，内容是各配置和路由在测试集上的 nDCG@10、Recall@50，以及路由对 BM25、路由对 passage 的 CI。

**Interfaces:**
- Produces: `community-results/<TAG>/revision-ijcai27/metrics_<encoder-slug>.jsonl.gz`、`community-results/revision-ijcai27-fleet/e4_encoders.json` 和 `e4_encoders.md`。

- [ ] **Step 1:** 写失败测试 `test_encoder_table_rejects_mixed_model_sets`：两个编码器的模型集合不同时报错。
- [ ] **Step 2–4:** 测试先失败，然后实现，再通过。
- [ ] **Step 5:** 运行两个编码器 × 七个模型。Mac 上太慢的话，可以在空闲 GPU 服务器上跑，需遵守远程规则，并核对结果文件的 SHA。
- [ ] **Step 6:** 提交，信息 `data: add retrieval results under two additional encoders`。

### Task 8: E7 `hy_rerank` 臂（代码）

**Files：** 在基线仓库新建工作树：`git -C /Users/a1-6/importantfile/Research/skill-LLM-baseline-run-20260724 worktree add ../skill-LLM-hy-rerank-20261124 -b ijcai27-hy-rerank 35b2f80`。
- Modify:
  - `hyskill/runtime_matched_rerank.py`：新增臂常量 `HY_RERANK_ARM = "hy_rerank"`。
  - `scripts/run_runtime_matched_rerank_decisions.py`、`scripts/run_runtime_matched_rerank_answers.py`：新增 `--arm {always_rerank,hy_rerank}`，并把 `--bm25-source` 改名为 `--candidate-source`，旧名保留为别名。
  - 评估、审计、校验、导出脚本中的臂白名单：`runtime_matched_public.NATIVE_ARMS` 等。
- Create: `scripts/export_routed_candidates.py`，把 `community-results/<TAG>/k2/retrieval_top50.jsonl.gz` 转成 `load_retrieval_records` 能读的 `{"results": [...]}` 格式，保持顺序，并写出源文件和输出文件的 SHA。
- Test: 在该工作树的 `tests/` 下新增 `test_hy_rerank_arm.py`。

**Interfaces:**
- Produces: 与 `always_rerank` 结构相同的决策行和答案行，`arm == "hy_rerank"`，`candidate_source_sha256` 指向导出的路由候选文件。

- [ ] **Step 1: 写失败测试**
  - `test_routed_export_preserves_order`
  - `test_hy_rerank_identity_differs_from_always_rerank`：同一道题、相同运行参数，候选不同、臂不同，所以 `execution_request_hash` 不同。
  - `test_hy_rerank_prompt_identical_given_same_candidates`：候选相同时，两臂生成的提示词逐字节相同。这一项用来证明只换了候选来源。
- [ ] **Step 2–4:** 测试先失败，然后实现，再通过；该工作树的全部测试也要通过。
- [ ] **Step 5:** 提交，信息 `feat: add hy_rerank arm over routed candidates`。

### Task 9: E7 部署与运行

- [ ] **Step 1:** 对 `AGENTS.md` 清单中的五台主机只读检查：主机名、`nvidia-smi`、进程、磁盘、权重路径、vLLM 版本。把结果写进 `results/revision-ijcai27/servers-20261124.md`。哪些模型能在哪台机器上跑、和原 runtime manifest 是否一致，以及部署方案，都要先给用户确认。
- [ ] **Step 2:** 用 `scripts/build_runtime_matched_runtime_manifest.py` 为每个模型、每个领域生成 manifest，运行参数必须与该模型 `always_rerank` 的 manifest 完全一致。先在 4 个领域上估算 token 量，报告给用户。
- [ ] **Step 3:** 传输代码包和路由候选文件，传输前后重算 SHA。
- [ ] **Step 4:** 每个模型先跑 `--max-new-records 20` 的试跑，核对提示词与 `always_rerank` 在格式上一致、`omitted_candidate_count` 已记录、失败分类正常；然后正式跑 4 个领域的全部 2,830 题，其中测试集 2,265 题。
- [ ] **Step 5:** 用 Monitor 轮询 vLLM 的 `Running / Waiting / KV cache usage` 三项指标，不要看 nvidia-smi 的利用率。确实有轮询在跑时，才能告诉用户"在监控"。
- [ ] **Step 6:** 取回结果，核对 SHA，然后依次跑 `evaluate_*`、`audit_*`、`validate_*`，要求 `valid=true`。最后导出到 `community-results/<TAG>/revision-ijcai27/hy_rerank/`。

### Task 10: E7 汇总（2×2 表）

**Files:** 在主仓库新建 `scripts/summarize_revision_pool_grid.py`，测试并入 `tests/test_revision_summaries.py`。

**口径：**
- 范围：五个模型 × 四个领域，测试集，指标是答题准确率，`method_failure` 算错。
- 单元格：
  - BM25+Select：`select_bm25`；
  - BM25+Rerank：`always_rerank`；
  - Hy+Select：`routed_select`；
  - Hy+Rerank：`hy_rerank`。
- 参照：Bare、Gated。
- 配对 CI：Hy+Rerank 对 BM25+Rerank，Hy+Select 对 BM25+Select，另加两个下游之间的对比。

- [ ] **Step 1:** 写失败测试 `test_grid_counts_method_failure_as_incorrect` 和 `test_grid_rejects_unavailable_models`：DeepSeek 或 Yi 出现在输入里时报错。
- [ ] **Step 2–4:** 测试先失败，然后实现，再通过。
- [ ] **Step 5:** 运行脚本，产出 `revision-ijcai27-fleet/e7_pool_grid.json` 和 `.md`。如果 Hy+Rerank 的 CI 含 0 或为负，在 md 中按 spec §7 写明收窄后的主张。
- [ ] **Step 6:** 提交，信息 `data: add candidate-source by downstream grid`。

### Task 11（可选）：E8 oracle 上限

E7 跑完且服务器有空闲时才做。做法：五个模型 × 四个领域，加载第一个 gold 技能后答题，臂名为 `oracle_gold`。复用 Task 8 工作树里的答案流水线，测试用 `test_oracle_loads_first_gold`。结果放进 2×2 表作为上限行。

### Task 12: 证据包收尾

- [ ] **Step 1:** 给每个新目录写 `README.md`，内容包括：来源、口径（测试集定义、五模型还是七模型、领域数）、SHA 清单，以及对 spec §6 的偏离（E1 主表改为四个领域）。
- [ ] **Step 2:** 运行 `.venv/bin/pytest -q`（主仓库和 hy-rerank 工作树都要跑），再运行 `./scripts/smoke.sh`，全部必须通过。
- [ ] **Step 3:** 在 `paper/STATE.md` 增加一节"IJCAI-27 revision experiments"，列出各实验的结论和证据路径。
- [ ] **Step 4:** 提交，信息 `docs: document IJCAI-27 revision evidence`。push 等用户决定。
