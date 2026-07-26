# Unified K=2 Results

This document summarizes the active K=2 experiment from the public per-instance
packs. Unless explicitly marked `full`, all downstream results use the frozen
held-out split. Calibration instances are excluded.

## Scope and metric definitions

- Rule-scored downstream domains: TheoremQA, LogicBench, MedCalc-Bench, and
  CHAMP.
- Held-out instances: 2,265 per model.
- Seven-model support: DeepSeek-7B, GLM-4-9B, Llama-3.1-8B, Mistral-7B,
  Qwen3.5-4B, Qwen3.5-9B, and Yi-1.5-9B.
- Five-model Select support excludes DeepSeek-7B and Yi-1.5-9B because their
  8,192-token endpoints cannot support the frozen 50-candidate Select prompt.
- Loaded-skill precision is `gold loaded / loaded`.
- Loading rate is `loaded / instances`.
- Gold-load rate is `gold loaded / instances`.
- Hy+Select always loads exactly one selected skill. Its selection accuracy,
  loaded-skill precision, and gold-load rate are therefore identical.
- Deterministic method failures remain in the answer denominator and are
  scored incorrect.

## Held-out loading results

### Fleet aggregates

Fleet rates are macro-averaged across models within each stated support set.

| Support | Arm | Loaded-skill precision | Loading rate | Gold-load rate |
|---|---|---:|---:|---:|
| Seven models | Always | 54.160% | 100.000% | 54.160% |
| Seven models | Gated | 73.638% | 69.429% | 51.611% |
| Five models | Always | 58.393% | 100.000% | 58.393% |
| Five models | Gated | 76.426% | 73.757% | 56.212% |
| Five models | Hy+Select | 62.278% | 100.000% | 62.278% |

Gating raises conditional precision because it abstains on some instances. It
does not raise the fleet gold-load rate above Always: on seven-model support,
73.638% conditional precision is paired with a 69.429% loading rate and a
51.611% gold-load rate.

### Per model

| Model | Arm | Loaded-skill precision | Loading rate | Gold-load rate |
|---|---|---:|---:|---:|
| DeepSeek-7B | Always | 31.921% | 100.000% | 31.921% |
| DeepSeek-7B | Gated | 53.525% | 50.728% | 27.152% |
| DeepSeek-7B | Hy+Select | unavailable | unavailable | unavailable |
| GLM-4-9B | Always | 54.658% | 100.000% | 54.658% |
| GLM-4-9B | Gated | 75.297% | 70.596% | 53.157% |
| GLM-4-9B | Hy+Select | 66.181% | 100.000% | 66.181% |
| Llama-3.1-8B | Always | 57.881% | 100.000% | 57.881% |
| Llama-3.1-8B | Gated | 77.431% | 72.185% | 55.894% |
| Llama-3.1-8B | Hy+Select | 51.479% | 100.000% | 51.479% |
| Mistral-7B | Always | 55.320% | 100.000% | 55.320% |
| Mistral-7B | Gated | 75.228% | 72.539% | 54.570% |
| Mistral-7B | Hy+Select | 59.161% | 100.000% | 59.161% |
| Qwen3.5-4B | Always | 62.031% | 100.000% | 62.031% |
| Qwen3.5-4B | Gated | 82.855% | 69.272% | 57.395% |
| Qwen3.5-4B | Hy+Select | 61.810% | 100.000% | 61.810% |
| Qwen3.5-9B | Always | 62.075% | 100.000% | 62.075% |
| Qwen3.5-9B | Gated | 71.316% | 84.194% | 60.044% |
| Qwen3.5-9B | Hy+Select | 72.759% | 100.000% | 72.759% |
| Yi-1.5-9B | Always | 55.232% | 100.000% | 55.232% |
| Yi-1.5-9B | Gated | 79.814% | 66.490% | 53.068% |
| Yi-1.5-9B | Hy+Select | unavailable | unavailable | unavailable |

## Select behavior

### Overall

| Split | Decisions | Parsed normally | Rank-1 fallback | Gold selected | Parsed-only accuracy | Fallback accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Full | 14,150 | 14,018 (99.067%) | 132 (0.933%) | 8,861 (62.622%) | 62.291% | 97.727% |
| Held-out | 11,325 | 11,223 (99.099%) | 102 (0.901%) | 7,053 (62.278%) | 61.953% | 98.039% |

The high fallback accuracy is not a selector-recovery result. Frozen fallback
always chooses routed rank 1, and almost all fallback cases occur in domains
where routed rank 1 is already strong.

### Held-out by model

| Model | Decisions | Parse success | Rank-1 fallback | Selection accuracy | Mean selected rank |
|---|---:|---:|---:|---:|---:|
| GLM-4-9B | 2,265 | 100.000% | 0.000% | 66.181% | 3.211 |
| Llama-3.1-8B | 2,265 | 100.000% | 0.000% | 51.479% | 6.817 |
| Mistral-7B | 2,265 | 99.956% | 0.044% | 59.161% | 5.143 |
| Qwen3.5-4B | 2,265 | 96.645% | 3.355% | 61.810% | 5.489 |
| Qwen3.5-9B | 2,265 | 98.896% | 1.104% | 72.759% | 4.963 |

### Held-out by domain over five models

| Domain | Decisions | Parse success | Rank-1 fallback | Selection accuracy |
|---|---:|---:|---:|---:|
| TheoremQA | 2,990 | 100.000% | 0.000% | 70.167% |
| LogicBench | 3,040 | 99.967% | 0.033% | 22.039% |
| MedCalc-Bench | 4,400 | 97.705% | 2.295% | 90.909% |
| CHAMP | 895 | 100.000% | 0.000% | 31.844% |

The selector issued 14,458 model calls for 14,150 full-split decisions:
13,988 decisions used one parse attempt, 16 used two, and 146 used three.
There were no unresolved infrastructure or unclassified failures.

## Held-out answer accuracy

| Model | Always | Gated | Hy+Select |
|---|---:|---:|---:|
| DeepSeek-7B | 13.554% | 16.909% | unavailable |
| GLM-4-9B | 48.079% | 49.845% | 52.804% |
| Llama-3.1-8B | 50.773% | 50.552% | 48.918% |
| Mistral-7B | 33.907% | 34.702% | 33.863% |
| Qwen3.5-4B | 72.097% | 72.892% | 73.377% |
| Qwen3.5-9B | 72.406% | 73.642% | 75.982% |
| Yi-1.5-9B | 40.088% | 43.400% | unavailable |
| Seven-model aggregate | 47.272% | 48.849% | unavailable |
| Five-model aggregate | 55.453% | 56.327% | 56.989% |

Qwen3.5-4B fixed-candidate Gated accuracy is 73.024%. The held-out active-arm
table retains 377 deterministic method failures. The complete K=2 archive
retains 467 method failures among 56,600 answer records.

The four completed K=2 paired contrasts all have confidence intervals that
contain zero. They support only "no detected difference", not equivalence or
strict superiority.

## K=2 routed retrieval

The following are held-out macro averages over seven models and five retrieval
domains, or 35 model-domain cells.

| Recall@1 | Recall@5 | Recall@10 | Recall@50 | nDCG@10 |
|---:|---:|---:|---:|---:|
| 38.243% | 57.533% | 66.202% | 82.196% | 54.428% |

BigCodeBench is included in these retrieval metrics but not in downstream
answer denominators.

## Token accounting

### Routed K=2 imagination estimate

These are estimates, not tokenizer-reported API usage. The published estimator
uses the actual saved prompts and generated outputs with a fixed conversion of
3.8 characters per token. Each model has 3,970 five-domain queries and 7,940
K=2 generations.

| Model | Input/query | Output/query | Total/query | Estimated total |
|---|---:|---:|---:|---:|
| DeepSeek-7B | 911.882 | 432.480 | 1,344.362 | 5.337M |
| GLM-4-9B | 881.405 | 443.360 | 1,324.765 | 5.259M |
| Llama-3.1-8B | 829.614 | 524.942 | 1,354.556 | 5.378M |
| Mistral-7B | 883.356 | 534.056 | 1,417.412 | 5.627M |
| Qwen3.5-4B | 877.320 | 544.257 | 1,421.577 | 5.644M |
| Qwen3.5-9B | 853.850 | 583.139 | 1,436.989 | 5.705M |
| Yi-1.5-9B | 846.843 | 548.942 | 1,395.785 | 5.541M |
| Seven-model total/mean | 869.181 | 515.882 | 1,385.064 | 38.491M |

The seven-model estimate is 24.155M input tokens plus 14.336M output tokens,
38.491M in total.

### Select and answer token status

Actual `prompt_tokens`, `completion_tokens`, and `total_tokens` were not saved
for the Select or answer calls. The records preserve request hashes, responses,
attempt counts, runtime identities, and elapsed time, but not API `usage`.

| Stage | Logical records | Model calls | Configured output cap | Actual token usage |
|---|---:|---:|---:|---|
| Select | 14,150 | 14,458 | 64 tokens/call | not recorded |
| Answer | 56,600 | 56,814 | 2,048 tokens/call | not recorded |

The caps imply only loose allocation ceilings of 925,312 Select completion
tokens and 116,355,072 answer completion tokens. They are not consumption
estimates and must not be reported as measured usage. Exact stage totals require
offline reconstruction with each frozen tokenizer and chat template; until
that is completed, the correct public value is `not_recorded`.

## Baseline comparisons under runtime-matched baselines (K2M001 closed)

Bare, native Rerank, and BM25+Select were re-run for all seven models under the
same checkpoint, tokenizer, chat template, served model, vLLM version, BF16, and
8K context as the formal K=2 answers — 48,110 answers and 28,300 decisions over
108 jobs, 98,207 HTTP calls, 250,356,288 tokens, `valid=true`.

This retires the `baseline_runtime_identity_gate=not_proven_by_this_script` flag
that had kept these four contrasts out of `k2-fleet/`. Held-out, hierarchical
bootstrap, 10,000 samples, seed 0:

| Contrast | A | B | A−B | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| Gated vs Bare, seven models | 45.47% | 37.24% | **+8.23 pp** | [+3.53, +13.87] | 0.0000 |
| Gated vs native Rerank, five models | 52.38% | 48.77% | **+3.61 pp** | [+0.79, +6.27] | 0.0154 |
| Gated vs BM25+Select, five models | 52.38% | 48.94% | **+3.44 pp** | [+0.28, +7.38] | 0.0302 |
| Hy+Select vs BM25+Select, five models | 53.18% | 48.94% | **+4.24 pp** | [+0.81, +8.20] | 0.0160 |

**All four CIs exclude zero.** Every contrast also keeps the sign and magnitude
of the earlier unverified numbers (+8.06 / +2.88 / +3.98 / +4.78), so the concern
that unproven runtime identity had inflated them does not hold.

Two caveats that must travel with these numbers:

- The `Gated vs native Rerank` contrast is weakened as a claim about reranking by
  the degradation documented below: on GLM and Qwen3.5-4B most instances compare
  against BM25 order rather than a working reranker.
- The Gated column comes from the published `k2/` packs, which is the source the
  paper cites throughout. The gate recalibration in the next section is a
  robustness check on those figures, not a replacement: it moves seven-model
  Gated by −0.13 pp and changes no significance verdict.

Evidence: `community-results/baselines-runtime-matched-fleet/` and
`community-results/<model>/baselines-runtime-matched/`.

## Gate recalibration under runtime-matched Bare

Closing K2M001 required re-running Bare under the frozen K=2 runtime. `gate.py`
calibrates `tau2` against "did **Bare** answer this correctly", so a fresh Bare
changes the calibration labels, then `tau2`, then the load/skip decision. All 32
gates were recalibrated (`valid=true`). `tau1` is unchanged everywhere — it
depends only on retrieval labels.

`tau2` moved on 9 jobs, flipping **609** decisions, which were re-inferred;
22,031 rows kept their original answer because the payload did not change, and
the 40 pre-existing method failures were preserved rather than resampled.
Qwen3.5-4B medcalcbench alone contributes 463 of the 609, because its `tau2`
went from `null` to a non-empty threshold.

Held-out effects: the 8 affected units go 67.71% → 67.32% (−0.39pp, net −17
items). Seven-model Gated loading precision 73.64→73.19, loading rate
69.43→68.01, gold-load rate 51.61→50.17. **Always and Hy+Select are numerically
unchanged** — they never pass through the gate, which serves as the correctness
check on the rebuild.

All four paired comparisons keep their significance verdict; every CI still
contains zero. The Qwen4 routed-vs-fixed contrast flips sign (−0.13 → +0.18pp)
at p=0.8548 — noise inside "no detected difference", **not** a direction
reversal.

**Which numbers the paper cites.** The published `k2/` packs remain the primary
source for Gated figures. This recalibration is reported as a robustness check,
not a replacement: it moves seven-model Gated by −0.13 pp globally and changes
no significance verdict, so the published figures stand. Evidence for the check:
`community-results/k2-gate-recalibration-v2/`.

Reporting it this way also keeps the baseline comparison table internally
consistent — that table's Gated column comes from `k2/`, so citing `k2/`
throughout avoids mixing two Gated vintages in one paper.

## Why the paper is built on K=2 rather than K=4

Held-out, seven models (five for Hy+Select), four rule domains. Using
`gold_load_rate` — the only cross-K comparable loading metric, see the warning
below — K=2 is no worse than K=4 on any arm:

| Arm | K=4 | K=2 | K=2 − K=4 |
|---|---:|---:|---:|
| Always (no gate) | 51.65% | **54.16%** | **+2.51 pp** |
| Gated | 49.42% | **51.61%** | **+2.19 pp** |
| Hy+Select | 62.00% | **62.28%** | +0.28 pp |

K=2 also costs 1,385 generation tokens per item against K=4's 2,747 (−49.6%),
and the K ablation finds routed retrieval nDCG@10 macro-average 54.43% for K=2
against 54.17% for K=4.

### The metric trap that reverses this conclusion

**K=4's Always arm is not 100% loaded.** yi15-9b returns no candidate on 38.0%
of medcalcbench and 36.8% of logicbench instances; across seven models the K=4
Always loading rate is only 92.13%, against 100% for K=2.

That makes `loaded_skill_precision` **incomparable across K**: K=4's denominator
silently excludes exactly the instances where retrieval failed, inflating the
figure. Read that way K=4 appears to win Always, 55.37% vs 54.16%. It does not.
`gold_load_rate` keeps every instance in the denominator and is the metric any
cross-K loading comparison must use.

### Gating absorbs K=2's weaker retrieval

K=2 imagines twice instead of four times, so its retrieval is genuinely weaker —
and the gate compensates. On **loaded-skill precision** the ordering flips once
the gate is applied: K=4 leads on Always but K=2 leads on Gated (73.10% →
73.64%).

yi15-9b shows the mechanism most clearly. Under K=4 nearly 40% of its retrievals
come back empty, so the gate is forced down to a 51.83% loading rate; under K=2
retrieval is stable, the gate returns to 66.49%, and gold-load rate is 12.6 pp
higher. This is the paper's central claim in miniature: the gate's value is
absorbing retrieval uncertainty.

### Where K=4 still wins

K=2's margin comes almost entirely from the two weak models — yi15-9b (+12.6 pp
gold-load rate) and deepseek7b (+3.8 pp). On **llama31-8b and mistral7b, K=4
leads by roughly 0.5 pp**. The claim is "K=2 is the better cost–effect tradeoff
across models", not "K=2 wins on every model".

Evidence: `community-results/k2-vs-k4-loading-analysis/`.

## Native rerank degrades into BM25 order on smaller models

The frozen protocol lets a listwise rerank that parses incompletely fall back to
appending the omitted candidates in their original BM25 order. Measuring how
often that happens shows the native Rerank baseline is not a working reranker on
most models:

| Model | omitted mean | fully ranked | ≥41 of 50 appended | used all 3 parse attempts |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 9.8 | 33.3% | 6.5% | 17.3% |
| Mistral-7B | 24.7 | 7.7% | 30.6% | 55.1% |
| Qwen3.5-4B | 32.6 | 5.1% | 62.7% | 69.4% |
| GLM-4-9B | 37.4 | 2.7% | 72.3% | 80.5% |

Every row is `failure_category=success`; nothing here violates the protocol, and
Llama's 33% full-ranking rate shows the pipeline itself is correct. This is a
capability difference, not a bug.

The consequence for interpretation: on GLM and Qwen3.5-4B, roughly two thirds of
the "Gated vs native Rerank" comparison is effectively **Gated vs BM25 order**.
Report the contrast with that caveat rather than as a comparison against a
functioning LLM reranker.

## Public evidence

- `community-results/k2-gate-recalibration-v2/` (gate recalibration — robustness
  check on the `k2/` Gated figures, not a replacement for them)
- `community-results/k2-fleet/loading_metrics_long.jsonl.gz`
- `community-results/k2-fleet/answer_metrics_long.jsonl.gz`
- `community-results/k2-fleet/summary.json`
- `community-results/k-ablation-fleet/summary.json`
- `community-results/k-ablation-fleet/cost.json`
- `community-results/<model>/k2/selection_per_instance.jsonl.gz`
- `community-results/<model>/k2/loading_per_instance.jsonl.gz`
- `community-results/<model>/k2/answer_per_instance.jsonl.gz`
- `community-results/<model>/k-ablation/cost.json`
