# E0 reproduction: author decision for qwen35-9b

Decision (2026-11-25, author): **accepted as passed**.

`reproduction.json` keeps `passed=false` unmodified. Every mismatch below is a one-instance flip at a rank boundary caused by
floating-point near-ties between hardware runs (or an exact value sitting on a 3-decimal rounding boundary). No router pick changed.

reproduction.json sha256: 53c2696cdbbd3558ca7e86060050f7b60db62081aa08cd2273515666701fa484

| variant | domain | split | metric | published | recomputed | n |
|---|---|---|---|---|---|---|
| hyskill | __all_macro__ | all | Recall@10 | 0.76026 | 0.76053 | 5 |
| hyskill | logicbench | all | Recall@10 | 0.63026 | 0.63158 | 760 |
| hyskill | logicbench | test | Recall@10 | 0.62664 | 0.62829 | 608 |
| routed | logicbench | all | Recall@10 | 0.63026 | 0.63158 | 760 |
| routed | logicbench | test | Recall@10 | 0.62664 | 0.62829 | 608 |

router_pick_mismatches: []
