# E0 reproduction: author decision for llama31-8b

Decision (2026-11-25, author): **accepted as passed**.

`reproduction.json` keeps `passed=false` unmodified. Every mismatch below is a one-instance flip at a rank boundary caused by
floating-point near-ties between hardware runs (or an exact value sitting on a 3-decimal rounding boundary). No router pick changed.

reproduction.json sha256: 8ca931a34e01695e0b8aea76c87822027dc7fd6da39ee900b1930ced85406aa1

| variant | domain | split | metric | published | recomputed | n |
|---|---|---|---|---|---|---|
| hyskill | __all_macro__ | all | Recall@50 | 0.82772 | 0.82746 | 5 |
| two_stage | __all_macro__ | all | Recall@50 | 0.82772 | 0.82746 | 5 |
| naive_passage | champ | validation | Recall@10 | 0.56250 | 0.56250 | 44 |
| routed | champ | validation | Recall@10 | 0.56250 | 0.56250 | 44 |
| hyskill | logicbench | all | Recall@50 | 0.49868 | 0.49737 | 760 |
| hyskill | logicbench | all | nDCG@50 | 0.21162 | 0.21139 | 760 |
| hyskill | logicbench | test | Recall@50 | 0.49671 | 0.49507 | 608 |
| two_stage | logicbench | all | Recall@50 | 0.49868 | 0.49737 | 760 |
| two_stage | logicbench | test | Recall@50 | 0.49671 | 0.49507 | 608 |

router_pick_mismatches: []
