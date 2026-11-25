# E1: Held-Out Candidate-Pool Hit Rate by Depth

- **Support (main table):** 5 Select/Rerank-eligible models (`glm4-9b`, `llama31-8b`, `mistral7b`, `qwen3.5-4b-reference`, `qwen35-9b`) × 4 rule-scored domains (`theoremqa`, `logicbench`, `medcalcbench`, `champ`), held-out split only (`2,265` instances/model). DeepSeek-7B and Yi-1.5-9B are `unavailable`.
- **Rerank source (main table):** Runtime-matched `always_rerank` decisions (records `omitted_candidate_count`).
- **BM25 source (main table):** `ordered_candidate_ids` of the same `always_rerank` decisions, i.e. the shared FastBM25 candidate file every Select/Rerank arm received. BM25 and Rerank therefore score one candidate pool per instance, and their `hit@50` must coincide.
- **BM25 scorer note:** `qwen3.5-4b-reference`'s `k4` BM25 came from SR-Agents `BM25Retriever`; the other four models' `k4` BM25 is FastBM25. The main table does not use `k4` BM25. The BigCodeBench supplement and the 5-domain reference use only those four FastBM25 models.

## Main Table (4 Rule Domains, Macro Mean Across 5×4 Cells)

| Metric | BM25 | Native Rerank | Rerank (Strict Unfilled) | Routed HySkill (K=2) | Hy − Rerank (95% CI, p) | Hy − BM25 (95% CI, p) | Rerank Filled-Slot % |
|---|---:|---:|---:|---:|---|---|---:|
| `hit@1` | 0.245 | 0.435 | 0.435 | 0.489 | +0.055 [-0.004, +0.110] (p=0.0676) | +0.244 [+0.127, +0.369] (p=0.0000) | 0.1% |
| `hit@3` | 0.352 | 0.536 | 0.536 | 0.614 | +0.078 [+0.004, +0.146] (p=0.0396) | +0.262 [+0.149, +0.372] (p=0.0000) | 2.1% |
| `hit@5` | 0.412 | 0.572 | 0.562 | 0.672 | +0.100 [+0.022, +0.164] (p=0.0136) | +0.260 [+0.155, +0.355] (p=0.0000) | 10.9% |
| `hit@10` | 0.485 | 0.597 | 0.580 | 0.750 | +0.153 [+0.072, +0.216] (p=0.0006) | +0.265 [+0.168, +0.343] (p=0.0000) | 28.2% |
| `hit@20` | 0.560 | 0.616 | 0.594 | 0.805 | +0.190 [+0.112, +0.256] (p=0.0000) | +0.245 [+0.152, +0.319] (p=0.0000) | 41.9% |
| `hit@50` | 0.643 | 0.643 | 0.610 | 0.861 | +0.218 [+0.133, +0.291] (p=0.0000) | +0.218 [+0.134, +0.292] (p=0.0000) | 56.3% |
| `nDCG@10` | 0.344 | 0.500 | — | 0.592 | +0.091 [+0.029, +0.146] (p=0.0050) | +0.248 [+0.145, +0.347] (p=0.0000) | — |

## Per-Domain Top-1 Hit (`hit@1`, 5 Models)

| Domain | BM25 | Native Rerank | Routed HySkill | Hy − Rerank (95% CI) |
|---|---:|---:|---:|---|
| `theoremqa` | 0.453 | 0.555 | 0.672 | +0.117 [+0.066, +0.171] |
| `logicbench` | 0.110 | 0.192 | 0.078 | -0.114 [-0.140, -0.088] |
| `medcalcbench` | 0.256 | 0.764 | 0.938 | +0.174 [+0.146, +0.203] |
| `champ` | 0.162 | 0.229 | 0.270 | +0.041 [-0.028, +0.116] |

## BigCodeBench Supplement (4 FastBM25 Models, `k4` BM25 and `llm_rerank`, No Omission Count Stored)

| Metric | BM25 | `k4` LLM Rerank | Routed HySkill | Hy − Rerank (95% CI) |
|---|---:|---:|---:|---|
| `hit@1` | 0.453 | 0.556 | 0.447 | -0.109 [-0.141, -0.076] |
| `hit@3` | 0.626 | 0.793 | 0.700 | -0.094 [-0.116, -0.072] |
| `hit@5` | 0.712 | 0.850 | 0.791 | -0.059 [-0.081, -0.035] |
| `hit@10` | 0.812 | 0.894 | 0.884 | -0.010 [-0.033, +0.011] |
| `hit@20` | 0.891 | 0.921 | 0.953 | +0.032 [+0.015, +0.050] |
| `hit@50` | 0.955 | 0.955 | 0.987 | +0.032 [+0.022, +0.041] |
| `nDCG@10` | 0.411 | 0.557 | 0.490 | -0.067 [-0.087, -0.046] |

## Reconciliation Note

- 5-domain reference (4 FastBM25 models, `k4` `llm_rerank`, `hit@10`): diff `+0.112` 95% CI `[+0.032, +0.183]` (the earlier session check, +0.100 [+0.039, +0.159], pooled qwen3.5-4b-reference with its SR-Agents rerank pool and is superseded; the main table scopes to the 4 rule domains where runtime-matched rerank recorded `omitted_candidate_count`).
- Note on tail positions: deeper ranks (`k >= 20`) increasingly overlap positions appended in BM25 order when listwise output truncated early; report both deployed rankings and the filled-slot diagnostic.
