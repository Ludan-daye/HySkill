# E5: Query vs. Imagination Cosine Similarity to Gold Skills

- **Support:** 7 models × 5 domains (`3,177` held-out instances/model), encoder `sentence-transformers/all-MiniLM-L6-v2` (`modelscope:m`).
- **Protocol:** Each template uses the L2-normalized mean of `K=2` cached imaginations **without** mixing the raw query vector and **without** multi-view field splitting; multi-gold instances take `max` cosine over gold skills.

## Overall (Macro Across 7×5 Cells)

- Raw query mean cosine to gold: **`0.3553`**

| Representation | Mean Cosine | Imagination − Query | 95% CI | p |
|---|---:|---:|---|---:|
| `sentence` | 0.4111 | +0.0558 | [-0.0307, +0.1320] | 0.1988 |
| `passage` | 0.4506 | +0.0952 | [+0.0061, +0.1722] | 0.0382 |
| `skill` | 0.4810 | +0.1256 | [+0.0304, +0.2009] | 0.0110 |
| `routed_template` | 0.4804 | +0.1250 | [+0.0323, +0.2041] | 0.0104 |

## By Domain (Imagination − Query, 7 Models)

| Domain | Query | `sentence − Q` | `passage − Q` | `skill − Q` | `routed_tpl − Q` (95% CI) |
|---|---:|---:|---:|---:|---|
| `theoremqa` | 0.4416 | -0.0286 | +0.0216 | +0.0586 | +0.0412 [-0.0841, +0.1162] |
| `logicbench` | 0.1517 | +0.0729 | +0.0105 | +0.1118 | +0.1116 [+0.0362, +0.1889] |
| `medcalcbench` | 0.3797 | +0.2695 | +0.3194 | +0.3235 | +0.3429 [+0.2028, +0.4346] |
| `champ` | 0.3118 | +0.0455 | +0.0864 | +0.1068 | +0.1021 [+0.0339, +0.1511] |
| `bigcodebench` | 0.4920 | -0.0804 | +0.0382 | +0.0273 | +0.0273 [-0.0045, +0.0484] |
