# E4: Retrieval Robustness Across Embedding Encoders

- **Variant & HyDE comparisons:** 7 models × 5 domains (`3,177` held-out instances/model).
- **BM25 comparisons:** 4 FastBM25 models × 5 domains. `deepseek7b` and `yi15-9b` store `theoremqa` only in `k4`; `qwen3.5-4b-reference`'s `k4` BM25 used SR-Agents `BM25Retriever`. None of the three is mixed into BM25 contrasts.

## Encoder: `BAAI/bge-base-en-v1.5`

| Arm (7-Model Macro) | nDCG@10 | Recall@50 |
|---|---:|---:|
| `naive_sentence` | 0.477 | 0.736 |
| `naive_passage` | 0.523 | 0.779 |
| `naive_skill` | 0.521 | 0.767 |
| `hyskill` | 0.505 | 0.800 |
| `two_stage` | 0.530 | 0.800 |
| `routed` | 0.557 | 0.809 |

- **Routed vs. `naive_passage` (7 models):** nDCG@10 `+0.034` `[-0.001, +0.077]` (`p=0.0594`); Recall@50 `+0.030` `[-0.013, +0.081]` (`p=0.1872`).
- **Routed vs. `bm25` (4 FastBM25 models):** nDCG@10 `+0.219` `[+0.120, +0.320]` (`p=0.0000`); Recall@50 `+0.192` `[+0.105, +0.274]` (`p=0.0000`).

## Encoder: `thenlper/gte-base`

| Arm (7-Model Macro) | nDCG@10 | Recall@50 |
|---|---:|---:|
| `naive_sentence` | 0.470 | 0.723 |
| `naive_passage` | 0.532 | 0.796 |
| `naive_skill` | 0.512 | 0.751 |
| `hyskill` | 0.499 | 0.807 |
| `two_stage` | 0.521 | 0.807 |
| `routed` | 0.568 | 0.827 |

- **Routed vs. `naive_passage` (7 models):** nDCG@10 `+0.035` `[+0.002, +0.084]` (`p=0.0312`); Recall@50 `+0.032` `[-0.010, +0.092]` (`p=0.1664`).
- **Routed vs. `bm25` (4 FastBM25 models):** nDCG@10 `+0.228` `[+0.130, +0.327]` (`p=0.0000`); Recall@50 `+0.203` `[+0.123, +0.278]` (`p=0.0000`).

## Encoder: `BAAI/bge-large-en-v1.5`

| Arm (7-Model Macro) | nDCG@10 | Recall@50 |
|---|---:|---:|
| `naive_sentence` | 0.492 | 0.761 |
| `naive_passage` | 0.539 | 0.808 |
| `naive_skill` | 0.529 | 0.776 |
| `hyskill` | 0.525 | 0.826 |
| `two_stage` | 0.542 | 0.826 |
| `routed` | 0.584 | 0.849 |

- **Routed vs. `naive_passage` (7 models):** nDCG@10 `+0.045` `[+0.011, +0.095]` (`p=0.0004`); Recall@50 `+0.041` `[+0.003, +0.095]` (`p=0.0310`).
- **Routed vs. `bm25` (4 FastBM25 models):** nDCG@10 `+0.241` `[+0.146, +0.336]` (`p=0.0000`); Recall@50 `+0.229` `[+0.146, +0.308]` (`p=0.0000`).
