# HySkill Current State and Agent Handoff

> Updated: 2026-07-26, after the K=2 manuscript migration. Read this file before changing experiments, figures, or the paper. It records the current state, not a history log. Do not copy claims from older status paragraphs without checking the sources below.

## Source-of-Truth Order

1. **Live manuscripts:** `paper/latex/` (English) and `paper/latex-zh/` (Chinese), each a separate Git repository connected to Overleaf.
2. **Active per-model evidence:** `community-results/<model-tag>/k2/`. Since the `d64a77f` migration, per-instance files live under `k2/` (active) and `k4/` (archived reference), **not** at the model root. Files that remain at the model root are `imagination_full_k{1,2,4,8,10}.*`, plus the `k-ablation/`, `baselines-runtime-matched/`, and (Qwen only) `baselines-native/` directories.
3. **Native baselines:** `community-results/<model-tag>/baselines-runtime-matched/` and the fleet aggregate `community-results/baselines-runtime-matched-fleet/`. These re-ran Bare / native Rerank / BM25+Select under the K=2 runtime and are the only baselines the paper may cite.
4. **Fleet significance:** `community-results/baselines-runtime-matched-fleet/paired_comparisons.json` (four baseline contrasts) and `community-results/k2-fleet/paired_comparisons.json` (internal-arm contrasts).
5. **Result prose:** `docs/10-k2-results.md` for K=2, `docs/10-k-ablation-analysis.md` for the K ablation. `docs/05-results.md` is the **K=4 Phase 1/2 record** and must not be cited for current numbers.
6. **Implementation:** `hyskill/` and `scripts/`; do not infer behavior from prose when code or saved outputs answer the question.

## The manuscript is now entirely K=2

As of 2026-07-26 both manuscripts carry K=2 numbers throughout, including the abstract, introduction, and all source comments. This was verified mechanically, not by reading: an audit checked every value the migration was supposed to replace and every value it was supposed to introduce, across `main.tex`, `intro.tex`, and `experiments.tex` (English) and `main.tex` (Chinese). Result: no surviving K=4 value, all required K=2 values present.

Pushed commits: English `f75c3c0`, Chinese `6dce6c2`, both live on Overleaf; repository `8e50f75` on GitHub.

### What each number now is

| Location | Was (K=4) | Is (K=2) |
|---|---|---|
| Main table | Always / Gated / Hy+Select only | **plus Bare / Rerank / BM25+Select**, from the runtime-matched rerun |
| `tab:retrieval-common` imagination rows | .568 / .593 / .680 / .838 | .567 / .596 / .679 / .836 |
| `tab:variant-ablation` | .434 / .501 / .482 / .497 / .495 | .450 / .503 / .491 / .502 / **.505** |
| Nine-method grid, best fixed variant | two-stage .604 | two-stage .596 |
| Efficiency | 1,760 in / 976 out | 877 / 544 (Qwen3.5-4B); 869 / 516 (seven-model mean) |
| Qwen retrieval paired test | .687 vs .662, $p<0.0001$ | .662 vs .662, $p=0.8904$ |
| Select-source ablation gain | 5.9 points | 5.1 points |
| Abstract headline | Recall@50 .838; +13.5 / +4.8 points | Recall@50 .836; **+8.2 / +3.6 points**, CIs excluding zero |

### Three claims that changed direction

These are not transcription updates. The K=2 data says something different from the K=4 data, and the prose was rewritten to follow it.

1. **The best fixed variant is two-stage, not passage.** At K=4 passage led (.501); at K=2 two-stage leads (.505) with passage and field fusion within .003. The manuscript no longer names a winner — it argues the top three are separated by less than cross-model spread, which is the actual motivation for routing.
2. **Routed retrieval no longer beats reranking on nDCG@10.** The Qwen3.5-4B held-out paired test over the same 3,177 instances gives $-.001$, 95% CI $[-.013, .011]$, $p=0.8904$. The manuscript now reports no detected difference at the top of the ranking and rests the retrieval argument on Recall@50 (.638 → .836), where reranking is structurally capped because it can only reorder the BM25 pool. **Do not restore any "routed beats rerank on nDCG" claim.**
3. **Native Select does not abstain.** The old text said it may "choose or abstain". All 14,150 decisions across the five eligible models resolve to a choice — it is a forced single pick at temperature 0 with a deterministic rank-1 fallback. Gating is the only arm in the comparison that can decline to load.

### The four baseline contrasts now in the paper

Held-out, hierarchical bootstrap (models → domains → paired instances), 10,000 samples, seed 0. All four intervals exclude zero.

| Contrast | A | B | A−B | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| Gated vs Bare, 7 models | 45.47 | 37.24 | +8.23 | [+3.53, +13.87] | 0.0000 |
| Gated vs native Rerank, 5 | 52.38 | 48.77 | +3.61 | [+0.79, +6.27] | 0.0154 |
| Gated vs BM25+Select, 5 | 52.38 | 48.94 | +3.44 | [+0.28, +7.38] | 0.0302 |
| Hy+Select vs BM25+Select, 5 | 53.18 | 48.94 | +4.24 | [+0.81, +8.20] | 0.0160 |

The internal-arm contrasts (Gated vs Always, Gated vs Hy+Select, Hy+Select vs Always, routed vs fixed Gated) all have CIs containing zero and are reported as no detected difference. The paper separates HySkill from the native pipelines but does not rank routing, gating, and selection against each other.

## Figures

All three generators were re-run on 2026-07-26 and their outputs copied into both manuscript repositories.

| Script | Source | Status |
|---|---|---|
| `generate_loading_precision_by_model.py` | `<tag>/k2/loading_per_instance.jsonl.gz` | fine; already K=2 |
| `generate_retrieval_recall_by_model.py` | `<tag>/k2/` (routed) + `<tag>/k4/` (BM25, rerank) | **was broken**, fixed in `8e50f75` |
| `generate_select_candidate_source_ablation.py` | `<tag>/k2/` + `<tag>/baselines-runtime-matched/` | fine |

The retrieval generator had all seven paths dead after the `d64a77f` migration moved per-instance files into `k2/`/`k4/` — the figure simply could not be regenerated. It now reads routed retrieval from `k2/` and the two K-independent baselines from `k4/`, which is the only pack publishing their top50, and records that split in its manifest.

**The passage-imagination curve was removed.** It is K-dependent and `k2/` publishes no per-instance passage pool, so plotting it would mix K inside one figure. Curves are now routed (7 models), LLM rerank (5), BM25 (5) — DeepSeek-7B and Yi-1.5-9B have no complete BM25 or rerank pool in `k4/`, and the script refuses to impute.

### Why the retrieval figure reads two files out of `k4/`

This looks wrong at a glance and was questioned once already, so it is recorded here rather than rediscovered.

`K_img` controls how many hypothetical skill documents the LLM writes. Only methods that consume those documents can change with it. BM25 is lexical matching on the raw query and never calls the LLM at all; LLM reranking reorders BM25's 50 candidates and reads a candidate list, not an imagination. Neither takes `K` as an input, so their rankings are byte-identical at K=1, K=2 and K=4.

That is why the K=2 campaign only re-ran the imagination-dependent arms: re-running the baselines would have burned compute to reproduce identical files. `k4/` is therefore not "the K=4 numbers" — it is "everything the K=4 round emitted", part of which is shared across all K. `k2/` publishes `routed` only.

**Considered and rejected: copying the baseline top50 into `k2/` to make it self-contained.** Reasons: `k2/` is a published frozen pack whose `manifest.json` binds an 11-file list with a per-file SHA, so adding files breaks the directory contract and the exporter that writes it; and the two schemas differ (`k2` uses `gold_skill_ids`/`retrieved`/`k_samples`, the archive uses `gold`/`top50`). The `k_samples` field is the blocking one — BM25 has no such quantity, so writing 2 would assert it consumed two imaginations, and writing null would violate the schema. The current arrangement is safe because both the generator code (`K_INDEPENDENT_METHODS`) and the emitted manifest (`k_provenance`) state the split explicitly, and the figure caption tells the reader.

## Completion Snapshot

Status marks: `[x]` complete for the current scope, `[~]` usable draft with remaining work, `[ ]` not drafted, `[!]` waiting on user input or a new experiment.

| ID | Status | Area | Current state | Completion criterion |
|---|---|---|---|---|
| E001 | `[x]` | Core comparison data | Seven K=2 packs plus seven runtime-matched baseline packs. DeepSeek-7B and Yi-1.5-9B cannot fit the 50-candidate prompt; marked unavailable, never zero. | Analysis-pack files present under `community-results/`. |
| F001 | `[x]` | Result figures | Retrieval, loading, and select-ablation figures regenerated from K=2 packs and synced to both repos. | All PDFs present in both manuscript repositories. |
| W001 | `[x]` | Abstract, Introduction, Related Work | Drafted bilingually; headline numbers migrated to K=2. | Final citation pass after all sections exist. |
| W002 | `[~]` | Method | Five variants, offline routing, and the loading gate are written and synchronized. | The two-lane overview figure is still absent; the user said they will draw it. |
| W003 | `[x]` | Experiments | Fully migrated to K=2, native baselines restored, three reversed claims rewritten, mechanically audited. | Done for the current evidence. Revisit only if new experiments land. |
| W004 | `[ ]` | Analysis | `sections/analysis.tex` is a 12-line skeleton. | Evidence-backed patterns, counterexamples, and scope without repeating main tables. |
| W005 | `[ ]` | Conclusion and limitations | `sections/conclusion.tex` is a 13-line skeleton. | Conclusion plus explicit limitations consistent with the experiments. |
| W006 | `[ ]` | Reproducibility checklist | Not included by `main.tex`. | Add the AAAI-27 checklist before submission. |
| L001 | `[!]` | Final layout | **Not compiled since the K=2 edits.** The main table went from 4 to 7 columns. | Compile on Overleaf and check the main table does not overflow `table*`. |

The paper remains **not submission-ready**: Analysis and Conclusion are empty, and the method overview and checklist are missing. The experimental content itself is complete and internally consistent.

## Known open risk

**The K=2 manuscript has never been compiled.** The local TeX install is TeX Live Basic and lacks `newtxtext.sty` (English, AAAI) and `placeins.sty` (Chinese); installing them needs sudo, which was not taken. What *was* verified is that all three table environments in each language compile standalone under `booktabs` with zero errors and zero overfull boxes.

The specific thing to watch is `tab:e2e-main`, which grew from 4 to 7 columns. `tabcolsep` was reduced to 5pt (English) and 3.4pt (Chinese); estimated width is about 5.75in against roughly 7.0in of `table*` width, but AAAI's newtx metrics are wider than the Computer Modern used in the standalone check. If it overflows, the cheapest fix is one decimal place instead of three.

## Metric and Baseline Semantics

- **Retrieval accuracy** means nDCG/Recall against gold skills.
- **Loaded-skill precision** is P(loaded skill is gold | a skill was loaded). It is not answer accuracy. Loading rate and gold-load rate are two further distinct quantities; all four must never share the label "accuracy".
- **End-to-end accuracy** is rule-scored task-answer accuracy after the complete pipeline.
- `always_rerank` is the native BM25 → LLM-rerank → load-top-1 pipeline.
- `select_bm25` is the native BM25-candidate LLM-selection pipeline and is the external Select baseline.
- `select` fed with HySkill candidates is a same-source loading ablation, never an external baseline. It and the Hy+Select column are the same construction viewed from two sides.
- `loaded_skill_precision` **is not comparable across K**. Use `gold_load_rate` for any cross-K statement.
- `omitted_candidate_count` measures the ability to emit a complete 50-item ordering, a format-following capability. It does **not** measure whether reranking worked, and is if anything inversely correlated with whether the model changed BM25's top-1. See the correction in `dbe6278`.
- Cross-model averages must use strict common support. Missing curves stay missing; never replace them with zero. Never average a seven-model figure against a five-model one.

## Evidence Boundaries

- Routed HySkill leads the strict-common-support retrieval comparison on macro-averaged nDCG@10 and Recall@50, but is statistically indistinguishable from reranking on the Qwen3.5-4B per-instance nDCG@10 test. State the reach argument, not an ordering win.
- The independent effect of signal $S_2$ is not established: conservative calibration left it inactive.
- K=2 was chosen because `gold_load_rate` is no worse than K=4 on any arm at half the generation cost — not because K=2 is uniformly better. Its margin comes mostly from yi15-9b and deepseek7b; llama31-8b and mistral7b favour K=4 by about 0.5 pp. Claim the tradeoff, not a per-model sweep.
- Known failure boundary: imagination quality can collapse on LogicBench for non-Qwen families, and higher retrieval scores can fail to improve answers in a shadowing domain. Preserve these negative results.
- Unfinished extensions: encoder and generator-scale ablations, SkillRouter direct comparison, ToolQA, and BigCodeBench end-to-end execution.

## Next Tasks

| ID | Status | Priority | Task | Done when |
|---|---|---|---|---|
| T000 | `[!]` | High | Compile both manuscripts on Overleaf | No errors; `tab:e2e-main` fits without overflow; page budget re-checked. |
| T001 | `[ ]` | High | Draft Analysis | Each pattern has direct evidence, a counterexample or boundary, and no unsupported universal wording. |
| T002 | `[ ]` | High | Draft Conclusion and Limitations | Closes the problem-method-result loop; names the unrun settings, the inactive $S_2$, and the undetected internal-arm differences. |
| T003 | `[!]` | Medium | Insert method overview figure | Wait for the user's drawing, then place it with an ordinary AuthorKit float. |
| T004 | `[ ]` | High | Final consistency and citation audit | English/Chinese content, terminology, support sets, references, and claims agree. |
| T005 | `[ ]` | High | Final AAAI packaging | Checklist included, page budget checked, figures readable, forbidden layout constructs absent. |
| T006 | `[ ]` | Low | Refresh stale repo docs | `CLAUDE.md` and `AGENTS.md` still describe `experiments.tex` as an entirely K=4 manuscript asserting significant routing and gating on Qwen4. Both statements are now wrong. |

## Writing and Layout Rules

- Explain the method directly. Reporting superiority to evaluated baselines is fine; mysterious or absolute language such as "works everywhere" is not.
- Preserve the distinction between retrieval, loading precision, gold-load rate, and end-to-end answer accuracy.
- A confidence interval containing zero supports only "no detected difference" — never equivalence, "no loss", or superiority.
- Content changes stay synchronized in English and Chinese. The standing exception is float/page positioning, which the user limited to English.
- Follow AAAI-27 AuthorKit before optimizing placement. Do not use `float`, `[H]`, `cuted`, `strip`, `captionof`, manual page breaks, negative spacing, or `minipage` figure grouping. Use `figure`, `figure*`, `table`, `table*`, source order, and sparing section-boundary `\FloatBarrier`.
- Figure captions go below figures; table captions go above tables.

## Git and Overleaf Workflow

- English: `paper/latex/`, Overleaf project `6a54ae13df54d74bd8edb90f`, branch **`main`** (not `master`).
- Chinese: `paper/latex-zh/`, Overleaf project `6a54b0de6f1125784434a635`, branch `main`.
- `.gitignore` excludes only `paper/latex/` and `paper/latex-zh/`. Everything else under `paper/` — including `figures/scripts/` — is committable to the root repository, and the figure generators now are. Generated PDFs/PNGs under `paper/figures/` are still untracked by choice; the manuscript copies live in the two Overleaf repositories.
- Never write credentials or tokens into repository files. The 2026-07-26 push used a one-time Overleaf token supplied in chat; **it should be revoked and rotated**. Prefer the Keychain form:

  ```bash
  /opt/homebrew/bin/git -c credential.helper= -c credential.helper=osxkeychain \
    -c http.proxy= -c https.proxy= push overleaf main
  ```

- GitHub pushes may fail on the LFS lock-verify call behind a local proxy. The repository now sets `lfs.<url>.locksverify false` to bypass it.
- Overleaf prohibits force pushes. Fetch and rebase when the remote advances; if conflict ownership is ambiguous, preserve both versions and ask.
