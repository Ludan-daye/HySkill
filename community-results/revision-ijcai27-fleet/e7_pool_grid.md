# E7: Candidate Source × Downstream Strategy (2×2 Held-Out Accuracy Grid)

- **Support:** 5 models (`glm4-9b`, `llama31-8b`, `mistral7b`, `qwen3.5-4b-reference`, `qwen35-9b`) × 4 rule domains (`2,265` held-out instances/model). `method_failure` rows remain in the denominator and score `0`.

## 2×2 Grid (Cell-Macro % / Instance-Weighted %)

| Candidate Pool | + Select | + Rerank |
|---|---:|---:|
| **BM25 top-50** | 48.94% (51.69%) | 48.77% (52.13%) |
| **Routed HySkill (K=2) top-50** | 53.18% (56.99%) | 51.55% (55.18%) |

- **References (5-model support):** `Bare` = 43.27% (43.94%), `Routed Gated (0 extra calls)` = 52.38% (56.33%).

## Key Paired Contrasts (Hierarchical Bootstrap, 10,000 Samples)

- **Hy+Select vs. BM25+Select:** `+4.24 pp` 95% CI `[+0.80, +8.17]` (`p=0.0158`)
- **Hy+Rerank vs. BM25+Rerank:** `+2.78 pp` 95% CI `[-0.03, +5.58]` (`p=0.0524`)

## Claim Guidance

- The 95% CI for Hy+Rerank minus BM25+Rerank contains zero or is negative; per Spec §7, scope the downstream candidate-pool claim to Select (Hy+Select > BM25+Select) and report no detected improvement (or negative difference) under listwise Rerank.
