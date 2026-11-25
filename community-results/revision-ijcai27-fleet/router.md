# E2, E3, and E6: HyDE Baseline, Router Necessity, and Label-Free Alternatives

- **Support:** 7 models × 5 domains (`3,177` held-out instances/model), encoder `all-MiniLM-L6-v2`.

## E2: Routed HySkill vs. HyDE (`naive_passage`)

- **Router selected `naive_passage` in `12 / 35` model-domain cells** (must be disclosed alongside the paired gain).
- `nDCG@10`: `0.544` vs `0.500`, diff `+0.045` 95% CI `[+0.011, +0.094]` (`p=0.0006`).
- `Recall@50`: `0.822` vs `0.765`, diff `+0.057` 95% CI `[+0.015, +0.119]` (`p=0.0014`).

## E3: Router Evidence

- **(a) vs. Global Best Fixed (`two_stage`):** `nDCG@10` diff `+0.042` 95% CI `[+0.019, +0.075]` (`p=0.0000`).
- **(b) vs. Per-Domain Post-Hoc Best Fixed (`{'bigcodebench': 'hyskill', 'champ': 'naive_passage', 'logicbench': 'hyskill', 'medcalcbench': 'two_stage', 'theoremqa': 'two_stage'}`):** `nDCG@10` diff `+0.014` 95% CI `[-0.001, +0.044]` (`p=0.0686`).
- **Router Pick Counts (out of 7 models per domain):**

| Domain | `naive_sentence` | `naive_passage` | `naive_skill` | `hyskill` | `two_stage` | `hyskill − passage` nDCG@10 (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| `theoremqa` | 0 | 5 | 0 | 1 | 1 | +0.004 [-0.084, +0.154] |
| `logicbench` | 1 | 0 | 0 | 6 | 0 | +0.079 [+0.021, +0.139] |
| `medcalcbench` | 0 | 4 | 0 | 1 | 2 | -0.097 [-0.157, -0.029] |
| `champ` | 1 | 3 | 1 | 2 | 0 | -0.037 [-0.079, +0.008] |
| `bigcodebench` | 0 | 0 | 0 | 7 | 0 | +0.040 [+0.021, +0.061] |

## E6: Label-Free Alternatives

- **(a) 5-Variant RRF Fusion (`k=60`) vs. Routed:** `nDCG@10` `0.526` vs `0.544`, diff `-0.019` 95% CI `[-0.048, +0.001]` (`p=0.0656`).
- **(b) Leave-One-Domain-Out (LODO) vs. Routed:** `nDCG@10` `0.504` vs `0.544`, diff `-0.040` 95% CI `[-0.071, -0.012]` (`p=0.0008`).
