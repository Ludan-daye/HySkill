# Repository Guidelines

## Current Research Handoff

Before changing experiments, figures, or either manuscript, read the
2026-07-24 live operational handoff later in this file, then read
[paper/STATE.md](paper/STATE.md). The latter remains the manuscript baseline but
is stale for the completed K=2 execution and packaging progress.

## Project Structure & Module Organization

`hyskill/` contains the installable Python package: retrieval, generation, parsing, fusion, embedding, BM25, and SR-Agents plugin integration. Keep reusable logic there and keep command-line experiment orchestration in `scripts/`. Tests live in `tests/`; small deterministic inputs belong in `tests/fixtures/`. Research rationale, experiment protocols, and finalized results are organized in `docs/`. Submitted model outputs belong under `community-results/<model-tag>/`; paper material is kept in `paper/`. Do not commit local caches, virtual environments, or raw experiment outputs unless the collaboration protocol explicitly requires them.

## Build, Test, and Development Commands

Use Python 3.10 or newer and a project-local environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,full]"
.venv/bin/pytest -q
./scripts/smoke.sh
```

The editable install exposes package changes immediately. `pytest -q` runs the test suite; `scripts/smoke.sh` validates the SR-Agents plugin and retrieval schema with the fixture corpus. For a small experiment pilot, configure an OpenAI-compatible endpoint and run:

```bash
MODEL=<model> API_BASE=<url> PILOT=1 ./scripts/run_phase0.sh
```

Some workflows also require the sibling `external/SR-Agents` checkout; consult `README.md` and `docs/08-multimodel-plan.md` before full runs.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, type annotations, top-level imports, and short single-purpose functions. Use `snake_case` for modules, functions, and variables; use `PascalCase` only for external-system interfaces where a class is warranted. Prefer pure functions and explicit parameters over mutable global state or default arguments. Comments and docstrings must be in English. Raise specific, actionable exceptions; do not silently fall back or hide service failures. No formatter is configured, so match neighboring code and keep changes minimal.

## Testing Guidelines

Tests use `pytest` and follow `tests/test_<module>.py` naming. Add only focused coverage needed for changed behavior, favoring fixture-backed integration or smoke tests over mocks. Run the complete suite and the smoke script when changing retrieval, plugin, or output-schema behavior. There is no numeric coverage target.

## Commit & Pull Request Guidelines

Recent history uses concise, scoped subjects such as `docs: ...` and `data: ...`; use an imperative summary with an appropriate lowercase prefix. Keep commits limited to one logical change. Pull requests should explain the method or data change, list validation commands and results, link relevant issues or experiment plans, and identify model tags, endpoints, and output artifacts when results change. Include tables or screenshots only when they clarify documentation or result differences. Never include credentials or private endpoint tokens.

## Live Operational Handoff — 2026-07-24 09:55 CST

This section is the current handoff for the K=2 experiment, result packaging, and
paper update. It supersedes the old K=2 task statuses in
`paper/STATE.md` and in the 2026-07-23 plan, but it does not replace their
methodological details. Do not restart completed inference from an older status
table.

When sources disagree, use this order:

1. Machine-verifiable raw outputs, manifests, hashes, and validators.
2. This live handoff.
3. `docs/superpowers/plans/2026-07-23-k2-unified-loading-and-ablation.md`.
4. `docs/superpowers/plans/2026-07-23-community-results-k-layout-migration.md`.
5. `docs/superpowers/2026-07-23-server-inventory.md`.
6. `paper/STATE.md`, which remains the manuscript baseline but was last updated
   on 2026-07-21 and is stale for K=2 execution progress.

Status marks in this handoff are `[x]` complete and verified, `[~]` in progress
or complete with a remaining gate, `[ ]` not started, and `[!]` blocked.

### Project Objective and Frozen Decisions

HySkill retrieves task skills by generating model-conditioned hypothetical
queries, routing among retrieval variants, and deciding whether or which skill
to load before answering. The current goal is to make every active
imagination-dependent paper experiment use `K_img=2`, while preserving the
separate `K={1,2,4,8,10}` ablation.

The following decisions are frozen:

- Active downstream experiments use K=2. The joint K ablation and all five
  nested-prefix caches remain separate and unchanged.
- Seven-model routed Always and Gated, five-model routed Hy+Select, and the
  Qwen3.5-4B fixed `naive_skill` + Gated component line are in scope.
- Selection and answering are separate. Hy+Select reads the persisted
  selection decision and must not call the selector again during answering.
- Select uses 50 ordered candidates, temperature 0, `max_tokens=64`, at most
  three parse attempts, and deterministic rank-1 fallback. It does not abstain.
- Answer generation uses the existing direct engine, temperature 0.7,
  `max_tokens=2048`, and thinking disabled.
- DeepSeek-7B and Yi-1.5-9B cannot support the 50-candidate Select/Rerank prompt.
  Their cells are `unavailable`, never zero and never failures.
- K-independent old baselines may be reused only after an independent
  checkpoint/tokenizer/chat-template/runtime identity gate. File identity alone
  is not runtime identity.
- All external downloads must use China-hosted mirrors with no silent overseas
  fallback. Mirror provenance does not replace exact revision and SHA checks.
- Never overwrite K=4, the K ablation, raw server outputs, or another dirty
  server worktree.

The active model tags are:

| Result tag | Runtime model tag | Active arms |
|---|---|---|
| `deepseek7b` | `deepseek7b` | routed Always, routed Gated |
| `glm4-9b` | `glm4-9b` | routed Always, routed Gated, routed Select |
| `llama31-8b` | `llama31-8b` | routed Always, routed Gated, routed Select |
| `mistral7b` | `mistral7b` | routed Always, routed Gated, routed Select |
| `qwen3.5-4b-reference` | `qwen3.5-4b` | routed Always/Gated/Select and fixed Gated |
| `qwen35-9b` | `qwen35-9b` | routed Always, routed Gated, routed Select |
| `yi15-9b` | `yi15-9b` | routed Always, routed Gated |

The four rule domains contain TheoremQA 747, LogicBench 760, MedCalc-Bench
1,100, and CHAMP 223 instances, or 2,830 instances per model. Calibration uses
the frozen sorted-ID, seed-0, 20% split; held-out inference uses 2,265 instances
per model. BigCodeBench is the fifth retrieval-only domain and is not part of
the 2,830 rule-scored answer denominator.

### Authoritative K=2 Artifacts

The completed formal result tree is currently in an ephemeral local directory:

```text
/tmp/hyskill-k2-formal-all.B3vAc4
```

It currently contains 742 files and approximately 422 MB. Do not delete,
reformat, or mutate it. Because `/tmp` is not durable, exporting and
hash-verifying the public packs is the immediate preservation priority.

The verified fleet aggregation is:

```text
/tmp/hyskill-k2-fleet-v1.OlCp4z
```

It currently contains:

- `loading_metrics_long.jsonl`: 210 rows;
- `loading_summary.json`;
- `answer_metrics_long.jsonl`: 210 rows;
- `answer_summary.json`;
- `paired_comparisons.current4.json`;
- `paired_comparisons.baselines4.json`;
- `paired_comparisons.baselines4.json.sha256`;
- `SHA256SUMS` for the original five aggregate products.

The aggregate directory is approximately 332 KB. The original five entries in
`SHA256SUMS` all verify. The additional baseline comparison has its own verified
sidecar and SHA-256
`d21f8085dfdb5b60b3b9a77894bef1fcdca1f2ac1f02cce439ba0feee3c28afa`.
Before publication, merge the eight comparisons into the public
`paired_comparisons.json` and regenerate a single complete fleet manifest.

### Completed Formal-Result Gates

| Item | Verified result |
|---|---|
| Retrieval inputs | 210 K=2 raw result files: seven models × five domains × five fixed variants plus one routed result |
| Gate | 32/32 pipelines complete: 28 routed plus four Qwen fixed; signals, calibration, and apply completed with `cache_misses=0` |
| Select decisions | 20 files, 14,150 records: 14,018 success and 132 deterministic `selector_fallback`; unresolved infra/unclassified = 0 |
| Answer jobs | 80/80 jobs, exactly 56,600 answer records |
| Loading files | 48 files, 53,770 decision records |
| Evaluation files | 80/80 eval JSON files |
| Completion audits | 80/80 completion JSON files, all `valid=true` |
| Answer reuse | `reused_same_arm=0`, `needs_inference=56,600`, non-empty preseed files = 0; all reuse sidecars say `legacy_record_missing`, so no old K=4 answer entered the K=2 result |
| Answer outcomes | 56,133 success and 467 preregistered `method_failure`; infra/unclassified = 0 |

There are 100 top-level non-loading JSONL files because the 80 answer files and
20 `*.selection.jsonl` files coexist. Any answer counter must exclude selection
files; do not report 70,750 as the number of answers.

The 467 method failures remain in the denominator and are scored incorrect.
They are method behavior, not missing data:

| Model and arm | Context overflow | Empty after the allowed attempts |
|---|---:|---:|
| DeepSeek Always / Gated | 86 / 17 | 0 / 0 |
| GLM Always / Select | 29 / 21 | 0 / 0 |
| Llama Always / Select | 20 / 26 | 0 / 0 |
| Mistral Always / Select | 42 / 38 | 0 / 0 |
| Qwen3.5-4B Always / Select | 14 / 17 | 0 / 0 |
| Qwen3.5-9B Always / Select | 10 / 3 | 0 / 0 |
| Yi Always / Gated | 36 / 1 | 85 / 22 |
| **Total** | **360** | **107** |

Qwen3.5-9B Always/Gated answers produced on N1 and the Lab copy were compared
byte-for-byte and had identical answer SHA values. Qwen9 Gate was already
complete before N1 answering and must not be rerun.

### Provenance Decision for the 16,980-Record Early-Raw Cohort

The real-time audit identified a 16,980-record logical cohort generated fresh
under K=2. It is three models × two arms × 2,830 instances:

| Model | Arms | Logical rows | Direct raw-to-schema imports | Later formal retries |
|---|---|---:|---:|---:|
| DeepSeek-7B | routed Always + routed Gated | 5,660 | 5,660 | 0 |
| Llama-3.1-8B | routed Always + routed Gated | 5,660 | 5,660 | 0 |
| Yi-1.5-9B | routed Always + routed Gated | 5,660 | 5,553 | 107 |
| **Total** | | **16,980** | **16,873** | **107** |

The 16,980 figure is therefore the early-raw logical cohort, not literally the
number of final rows carrying a `provisional_source` field. The 107 Yi rows were
retried by the formal runner; each records three empty model outputs and remains
`method_failure=EmptyModelOutput`.

The 16,873 directly imported rows are post-generation structural conversions.
They are not reused K=4 answers:

- all have K=2 inputs and `reuse=0`;
- instance ID, selected/loaded skill, raw model answer, and evaluation target
  are unchanged;
- the conversion changes representation and provenance strength, not the
  observed answer.

Do not rerun this cohort. Rerunning would create a new stochastic sample and
would not validate the original output. Record the 16,873 direct conversions
as:

```text
provenance_level=posthoc_structural
```

Record the whole 16,980 as the `early_raw_k2` logical cohort and distinguish the
107 Yi rows as `formal_retry_after_import`. Do not falsely describe those 107
rows as direct structural conversions.

The presence of a `provisional_source` field is broader than the
`posthoc_structural` classification. Across all 56,600 answers, 25,363 rows
carry it: the 16,873 direct conversions above plus 8,490 Qwen3.5-4B reference
fixed-Gated/routed-Always/routed-Gated rows. Qwen reference provenance must be
classified from its run history; the field alone is insufficient. No current
answer row contains `provenance_level`.

Preserve the original raw files, conversion source path, source line number,
per-line SHA, whole-file SHA, logs, completion files, and converted output SHA.
The public exporter still needs an explicit cohort and row-set manifest.

The local formal bundle does not contain the referenced original
`raw-answers/*.jsonl` files or the original runtime manifests; its
`provisional_source.source_path` values point to server paths. Before public
release or SSH-key cleanup:

1. Retrieve the 24 DeepSeek/Llama/Yi raw source files, associated logs, runtime
   manifests, and checkpoint evidence.
2. Verify every whole-file SHA and stored source-line SHA.
3. Compare final `raw_output`, instance ID, dataset, arm, and skill IDs to the
   source row.
4. Write the 16,980 / 16,873 / 107 cohort manifest and add provenance labels
   without altering answer text or evaluation outcomes.
5. Generate a complete top-level `SHA256SUMS`.

Until this is done, do not claim independent raw-to-final verification. The
existing completion artifacts use `k2-answer-validation-v1`; do not describe
them as v2-validated unless the v2 validator is actually run successfully.

Rerun only if a concrete integrity check finds changed answer text, an
unresolvable ID or skill mapping, unverifiable K=2 input identity, missing raw
source, mismatched SHA, or actual K=4 contamination. A missing pre-run immutable
manifest alone lowers the provenance level but is not a scientific reason to
regenerate the answer.

The new provenance patch constrains future runs. It must not be applied
retroactively as a reason to invalidate otherwise verified current samples.

### Runtime and Code Identity

All formal answer rows report BF16 and an 8,192-token context. Their per-row
`runtime_identity` is authoritative for the full tokenizer and chat-template
hashes.

| Model | Frozen checkpoint identity | vLLM | Answer code bundle |
|---|---|---:|---|
| DeepSeek-7B | ModelScope `snapshots/master`, files manifest `25b7f08040a12a38ed6a4fdca625063e18091926a30813d56a3c87e3cbe1f03b` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| GLM-4-9B | ModelScope `snapshots/master`, files manifest `cd37e55587031d4dbc51bf768f83268669e196434f209b1bd0e6245991e038be` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| Llama-3.1-8B | ModelScope `snapshots/master`, files manifest `a8e51a9052d5cfe3faea783aa90837c6ba39d04f438eb6eca344a0f4b1e44630` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| Mistral-7B | ModelScope revision `c8cfccbcfd71d4e3479498c30b2823bab19c4687`, files manifest `559840283ece7b8cbbb937d74d5ce47aff520cda4a453a3331ac3e8f26bfa6df` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| Qwen3.5-4B | HF revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, files manifest `7447e4e49652e2eb494c53d808d9b4e005838b1430aecb6df8181b2105d177dc` | 0.17.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |
| Qwen3.5-9B | China HF mirror revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, files manifest `daf8a250ee437249688f839397f7908ed75e10eba31ab9a5663456c36c46b595` | 0.17.1 | `f796f20537fe63a484b4a302ebf3c2d5131d15aaff051404c2362be8afbe8d86` |
| Yi-1.5-9B | ModelScope `snapshots/master`, files manifest `45eb2167b36e6209f26a897a440cf27bf002f4b1368556d9105fbe76341addca` | 0.19.1 | `05e7bbb12b8d836db8fbec2d4cc9651ece9bac6a22228c0d78d19c824e265682` |

Do not confuse these three code identities:

- K=2 retrieval input identity:
  `b642012+bundle-efc7d610d25b29e28fdb14ab58da099e3e7b2b0b0b7fb55d2f7817f8306cf69f`;
- answer-generation bundle stored per row: `05e7...` for six result tags and
  `f796...` for Qwen9;
- current post-run provenance/manifest patch bundle:
  `5d451511e9a84b69eed186fa7da0da45c0e54f4ddca4dbb1b96e6f3cb8678962`,
  with the recorded 14-file double-hash
  `2c4e836336c1c91d817dc894eb20248434077ed0ea64640455847614562d71c7`.

### Verified K=2 Results

Descriptive held-out answer accuracy is instance-weighted within the stated
support:

| Support | Always | Gated | Hy+Select |
|---|---:|---:|---:|
| Seven models | 47.27% | 48.85% | unavailable as a seven-model aggregate |
| Five Select-eligible models | 55.45% | 56.33% | 56.99% |
| Qwen3.5-4B only | 72.10% | 72.89% | 73.38% |

Qwen3.5-4B fixed Gated held-out accuracy is 73.02%.

Held-out loading metrics must always report all three quantities:

| Support and arm | Loaded-skill precision | Loading rate | Gold-load rate |
|---|---:|---:|---:|
| Seven-model Always | 54.16% | 100.00% | 54.16% |
| Seven-model Gated | 73.64% | 69.43% | 51.61% |
| Five-model Always | 58.39% | 100.00% | 58.39% |
| Five-model Gated | 76.43% | 73.76% | 56.21% |
| Five-model Hy+Select | 62.28% | 100.00% | 62.28% |

The paired inference uses 10,000 bootstrap samples, seed 0, excludes the frozen
calibration IDs, and for fleet contrasts resamples model, then domain, then
paired instances. Its model/domain-weighted point estimate can differ from the
descriptive instance-weighted accuracy above.

| Contrast, held-out | A | B | A-B | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| Gated vs Always, seven models | 45.47% | 44.09% | +1.38 pp | [-0.83, 3.85] | 0.2300 |
| Gated vs Hy+Select, five models | 52.38% | 53.18% | -0.80 pp | [-3.53, 1.84] | 0.5540 |
| Hy+Select vs Always, five models | 53.18% | 51.42% | +1.76 pp | [-1.62, 5.22] | 0.3050 |
| Qwen routed Gated vs fixed Gated | 72.89% | 73.02% | -0.13 pp | [-1.68, 1.32] | 0.8808 |
| Gated vs Bare, seven models | 45.47% | 37.41% | +8.06 pp | [3.48, 13.52] | 0.0002 |
| Gated vs native Rerank, five models | 52.38% | 49.50% | +2.88 pp | [0.55, 5.13] | 0.0174 |
| Gated vs native BM25 Select, five models | 52.38% | 48.40% | +3.98 pp | [0.41, 9.10] | 0.0284 |
| Hy+Select vs BM25+Select, five models | 53.18% | 48.40% | +4.78 pp | [0.84, 9.74] | 0.0172 |

The first four contrasts are in `paired_comparisons.current4.json`, SHA-256
`4547f8e49996e36c84df15457ec3e5560b42bd174b97b3ab476fc62d15333a4c`.
The last four are in `paired_comparisons.baselines4.json`.

The last four baseline contrasts are statistically computed but not yet cleared
for an unconditional paper claim. Their artifact explicitly records
`baseline_runtime_identity_gate=not_proven_by_this_script`: the legacy
per-instance files and held-out IDs match, but the compact packs cannot prove
checkpoint, tokenizer, chat-template, or runtime equivalence to the K=2
endpoints. Pass K2M001 independently. If identity cannot be proven, rerun only
the affected K-independent baseline jobs needed for the claim, not the 16,980
fresh K=2 converted answers.

This is a material, not merely administrative, risk: the legacy DeepSeek and Yi
documentation records 4K services while the formal K=2 rows record 8K, and the
legacy Mistral native Select recovery used the same model revision with a 32K
service for overlength cases while the formal K=2 endpoint is 8K. Audit the
exact answer-generation and selection stages before deciding which baseline
jobs need a controlled rerun.

Claim rules:

- A CI containing zero supports only “no detected difference,” not
  equivalence, “no loss,” or strict superiority.
- Do not compare a seven-model average directly with a five-model average.
- Missing Select/Rerank arms remain unavailable, not zero.
- Loading precision, loading rate, gold-load rate, and answer accuracy are
  different quantities and must not share the label “accuracy.”
- Deterministic method failures remain incorrect; do not silently filter them.

### Current Task Table

| ID | Status | Task | Verified progress | Remaining completion criterion |
|---|---|---|---|---|
| K2M000 | `[x]` | Freeze unified K=2 protocol | User-approved protocol and support sets are recorded in the plan | None |
| K2M001 | `[!]` | Freeze old-baseline runtime identity | K2 runtime identities, K2 eval SHA values, baseline file SHA values, and held-out ID equality are recorded | Independently prove old Bare/Rerank/Select checkpoint, tokenizer, chat-template, and runtime equality, or rerun only affected baselines |
| K2M002 | `[~]` | Runner, validator, and provenance hardening | Core K2 pipeline ran successfully; the stricter manifest-v2 patch and tests are implemented | Review and preserve the uncommitted patch; repair local optional dependencies before claiming a clean full suite |
| K2M003 | `[x]` | N1 Qwen9 staging and execution | Qwen9 Select/Always/Gated completed, returned, evaluated, and SHA-matched against the Lab copy | Do not restage or rerun; only final key cleanup remains |
| K2M004 | `[x]` | K=2 Gate | 32/32 pipelines completed with zero cache misses | None |
| K2M005 | `[x]` | Five-model selection-only | 14,150/14,150 decisions; 132 auditable rank-1 fallbacks; no unresolved infra/unclassified | Export into public packs |
| K2M006 | `[x]` | Loading results | 48 files and 53,770 rows aggregated; `loading_summary.valid=true` | Export and independently recompute from the public pack |
| K2M007 | `[~]` | Reuse and provenance audit | All 56,600 answers have `reuse=0`; source references and completion audits are present | Retrieve and verify the 24 early raw files; add the explicit 16,980 / 16,873 / 107 cohort manifest and row-level provenance |
| K2M008 | `[x]` | K=2 answers and evaluation | 80/80 jobs, 56,600/56,600 records, 80 eval files; infra/unclassified zero | Export and preserve |
| K2M009 | `[~]` | Fleet statistics | 210 loading rows, 210 answer rows, and all eight paired contrasts generated and hash-checked | Pass K2M001, combine the two comparison files, and create the final fleet manifest |
| K2M010 | `[~]` | GitHub packs, K4 migration, and paper update | Directory contract and non-destructive migration plan are frozen; `community-results/README.md` has an uncommitted contract update | Finish public exporter, recover missing retrieval/router/gate metadata, validate packs, then migrate K4, update readers/docs/paper, commit, and push |
| CLEAN001 | `[ ]` | Remove staging SSH key | Cleanup target is known | After all transfers verify, remove the key comment `codex-k2-stage-20260723` from the three servers where it was installed, then remove `/tmp/hyskill-k2-stage.5SHbTG` |

### Current Uncommitted Implementation

The root repository is on `main` at
`5bc21aca567d671d92472b294f8c867a2a0e43b1`, equal to `origin/main` at the
time of this handoff. No K=2 implementation or data commit has been created.

The stricter provenance patch currently covers:

- `hyskill/downstream_reuse.py`;
- `scripts/build_k2_runtime_manifest.py`;
- `scripts/audit_k2_reuse.py`;
- `scripts/validate_k2_downstream.py`;
- `scripts/run_k2_main.sh`;
- `scripts/run_k2_select_main.sh`;
- `tests/test_downstream_reuse.py`;
- `tests/test_k2_downstream_pipeline.py`.

Its behavior includes manifest v2 per-file code SHA values, aggregate hash
recomputation, explicit legacy JSONL identity binding, unknown-model rejection,
DeepSeek/Yi Select rejection, and answer validation directly against legacy raw
rows rather than trusting an audit sidecar.

Verification recorded for this patch:

- targeted tests: 12 passed;
- `py_compile`: passed;
- `bash -n`: passed;
- full suite before the baseline adapter: 39 passed, 2 skipped, 10 failed;
- the 10 failures are from missing existing optional dependencies `bm25s` and
  `rank_bm25`, not from the patch;
- local smoke is additionally blocked by an existing `.venv/bin/sragents`
  shebang pointing to an obsolete `/root/...` interpreter.

The baseline adapter is newly implemented and uncommitted:

- `scripts/summarize_k2_baseline_comparisons.py`;
- `tests/test_k2_baseline_comparisons.py`.

It enforces all 80 K2 eval files and 56,600 rows, reads 48,110 old baseline
rows, uses the K2 `is_validation` IDs as held-out authority, handles Qwen4's
eight native eval files explicitly, and never aliases `always_r` to native
Rerank. The new fixture plus existing K2 pipeline regression passed 3 tests in
8.30 seconds. The full suite after adding it is 40 passed, 2 skipped, 10 failed
for the same missing BM25 dependencies.

The public-pack exporter is a draft in progress:

- `scripts/export_k2_public_pack.py` and
  `tests/test_k2_public_pack.py` now exist and are uncommitted;
- `py_compile` passes;
- it has explicit `FORMAL_COMPLETE` marker validation, requires an
  `--answer-provenance` manifest, and is designed to verify original raw-file
  and source-line SHA values before assigning per-row provenance;
- its latest targeted test snapshot is **8 failed** because the fixture
  `run_cli()` had not yet been wired to the new required
  `--answer-provenance` argument;
- no production public pack has been generated or accepted;
- even after tests pass, production must remain blocked until the 24 early raw
  files and explicit run-history/provenance manifests are recovered.

Do not mark the exporter or the 82-file public delivery complete merely because
the draft files exist.

The worktree also contains substantial pre-existing untracked `paper/` content
and many untracked K2 scripts/tests. These belong to the user's review state.
Never use `git add .`, `git clean`, reset, or a broad commit. Stage only an
explicit reviewed whitelist. Do not add the whole `paper/` tree from the root;
the English and Chinese manuscript directories have their own repositories.

### Manuscript State Before K=2 Integration

The manuscript is not submission-ready. The 2026-07-21 paper snapshot records:

- Abstract, Introduction, and Related Work drafted in English and Chinese.
- Method and Experiments are usable drafts, but they predate the completed K=2
  result integration.
- Analysis and Conclusion/Limitations are still skeletons.
- The user-planned method overview figure and the AAAI-27 reproducibility
  checklist are still missing.
- The English layout was last reported as seven pages before the missing
  sections were written, so the final page budget is unknown.
- The old state file says English Overleaf was at `4552b66` and Chinese
  Overleaf at `e1c7ef7`, but these are no longer the current local heads.

The live nested-repository audit on 2026-07-24 found:

- `paper/latex` at `3747e0440241ef7205e836af1f424071b8c42ea2`,
  with only `tmp/` untracked;
- `paper/latex-zh` at
  `a269d8bbcd3a85500fa75a85e7f1a3bf421610c2`, with a clean worktree.

Do not assume either local head has been read back from Overleaf without an
explicit fetch/status check.

The current English `experiments.tex` is still an old K=4 manuscript:

- setup says default K=4;
- loading, end-to-end, cost, and component numbers are the old results;
- the candidate-source figure uses fixed full-skill candidates for Qwen4 but
  routed candidates for the other models;
- it claims both routing and gating are significant on Qwen4, whereas the new
  routed-Gated vs fixed-Gated result is -0.13 pp with CI crossing zero and
  `p=0.8808`;
- its Select description and conclusions must be reconciled with the frozen
  forced-single-choice behavior and new five-model routed comparison.

After the public K=2 evidence is frozen, update tables, figures, significance,
support-set wording, Analysis, Conclusion, limitations, and checklist. Keep
English and Chinese content synchronized unless the user explicitly limits a
layout-only change to one language. English layout must remain AAAI 2027
AuthorKit-compliant; use normal floats and do not use force-placement hacks.
Completed manuscript changes are pushed to both Overleaf repositories under
the established workflow, with no force push.

### Servers and Standing Runtime Rules

The last verified inventory is four physical nodes and five GPU slots:

| Node | Login command without credential | Hardware | Frozen role | Main working location |
|---|---|---|---|---|
| S1 | `ssh root@180.127.11.169 -p 32940` | 1× A100 80 GB | DeepSeek-7B, then Yi-1.5-9B | `/root/HySkill-k-run-20260723` |
| S2 | `ssh vicuna@8.138.30.52 -p 6007` | 1× A100 80 GB | Qwen3.5-4B and Gate; historical Qwen9 source | `/home/vicuna/ludan/HySkill-k-run-20260723` |
| S3 | `ssh root@180.127.11.167 -p 22624` | 2× A100 40 GB | GLM, Llama, then Mistral | `/root/HySkill-k-run-20260723` |
| N1 | `ssh root@180.127.11.167 -p 27244` | 1× RTX 4090, driver reported about 49 GB | Qwen3.5-9B answers only | isolated staged K2 workspace |

Passwords are intentionally not stored here. Obtain them from the user or the
approved local credential channel. The old port 25720 entry is invalid and is
not a fifth node. There is no confirmed fifth physical server.

The table is an inventory snapshot, not permission to restart jobs. Formal
inference is complete. Before any remote operation, inspect hostname, current
processes, GPU use, disk, working tree, and result paths read-only. Never kill a
service, overwrite a result directory, or modify an unrelated dirty worktree.

If a future remote job is active, monitor it continuously, report an anomaly
immediately, and report normal progress approximately every ten minutes. Do not
claim real-time monitoring when no polling mechanism is active.

### GitHub Public-Pack and K=4 Migration Contract

The intended K=2 public delivery is:

- five Select-eligible models × 12 files;
- DeepSeek and Yi × 11 files, with Select explicitly unavailable;
- 82 per-model K=2 files total;
- one `community-results/k2-fleet/` pack with five public products.

Each eligible model needs routed K=2 retrieval, router decisions, gate
decisions, loading, selection, answers, answer metrics, flat metrics,
significance, reuse/provenance, manifest, and README evidence. DeepSeek and Yi
omit only the selection file and must explain the unavailable arm.

The formal local tree does not contain the seven-model raw routed retrieval,
signals, taus, router metadata, or standalone gated retrieval files required by
the public contract. Recover them read-only from:

```text
results/k-ablation/<tag>/routed/k2/
results/k2-main/<tag>/
```

Recompute transfer SHA values. Do not synthesize these files from aggregate
metrics.

The historical K=4 migration is frozen but not started:

- 70 whitelisted files;
- 70,917,990 bytes;
- all seven destination `k4/` directories were absent at the last audit;
- `k-ablation/`, `imagination_full_k{1,2,4,8,10}.*`, and
  `baselines-native/` must remain in place.

Do not move K4 until every K2 pack is complete and validated. Then use an
explicit whitelist and `git mv`, verify every pre/post SHA, update all readers
and writers listed in the migration plan, and check that no large file was
duplicated or recompressed.

### Immediate Continuation Order

1. Preserve the ephemeral formal and fleet trees by finishing the deterministic
   public exporter and manifests.
2. Retrieve and hash the missing routed retrieval, signals, taus, gated, router
   artifacts, and the 24 DeepSeek/Llama/Yi early raw answer files plus their
   runtime evidence from the servers. Do not rerun inference.
3. Emit seven per-model K2 packs and one fleet pack; validate gzip, JSON/JSONL,
   schemas, row counts, unique IDs, support sets, provenance levels, and all
   manifest SHA values.
4. Complete K2M001 independently. Do not present the four old-baseline
   comparisons as final until runtime identity is proven.
5. Combine and independently reproduce all eight held-out comparisons from the
   public per-instance files.
6. Only after K2 passes, perform the whitelisted K4 `git mv` and prove
   pre/post SHA equality.
7. Update repository readers/writers, `README.md`, `paper/STATE.md`, result
   documentation, figures, tables, and both manuscript languages. Remove or
   label every stale K4 active-paper number.
8. Review the exact diff, create only scoped commits when authorized, push the
   GitHub data, and push completed manuscript changes to both Overleaf
   repositories without force-pushing.
9. After all transfers and read-back checks, remove the temporary staging key
   from the three affected servers and delete its local temporary directory.

### Definition of Done

Do not mark the unified K=2 project complete until all of the following are
machine-verified:

- K2M001 passes, or every required baseline identity mismatch is resolved by a
  scoped rerun and re-evaluation.
- All 82 per-model K2 files and all five fleet products exist.
- Gzip integrity, JSON/JSONL parsing, schemas, counts, IDs, support sets,
  provenance, SHA manifests, and sensitive-information scans pass.
- The 16,980 / 16,873 / 107 early-raw provenance split is reproducible from the
  preserved raw sources.
- All eight comparisons reproduce from public held-out per-instance evidence
  under the documented estimand.
- All 70 K4 files retain identical SHA-256 values after migration.
- Every reader, writer, link, table, figure, and both manuscript languages use
  the new directory and K=2 evidence contract.
- GitHub and both applicable Overleaf repositories are pushed and read back
  without force pushes or credentials in history.
- Temporary remote authorization and local staging keys are removed only after
  the final transfer verification.

The K=2 experiment is computationally complete but is not publication-complete.
The remaining work is provenance classification, baseline runtime identity,
durable public packaging, non-destructive K4 migration, and manuscript
integration.
