# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

HySkill is a **research repository**, not a product. It ports HyDE's query-side
hypothetical-document idea to agent *skill* (`SKILL.md`) retrieval and loading:
generate K model-conditioned hypothetical skills for a task, retrieve real skills
with those embeddings, then use the imagination↔reality gap as a **loading gate**
that decides whether (and which) skill to inject before answering.

Most of the value in the tree is **frozen experimental evidence and its
provenance**, not application code. Code exists to produce, validate, and package
that evidence. Treat committed results as immutable artifacts.

An AAAI-2027 manuscript is in flight (`paper/`). **The K=2 experiments and their
control arms are complete and published**; K2M001 (baseline runtime identity) is
closed. The remaining work is manuscript integration — `experiments.tex` is still
a K=4 document.

## Read this before changing experiments, data, or the paper

Source-of-truth order (highest first):

1. Machine-verifiable raw outputs, manifests, SHA files, and validators.
2. **`AGENTS.md`** — the live operational handoff. It carries the frozen protocol
   decisions, the K2M0xx task table, verified counts, model tags, server
   inventory, and the claim rules. It is authoritative over any prose below.
3. `docs/superpowers/plans/` — dated implementation plans, notably
   `2026-07-23-k2-unified-loading-and-ablation.md`,
   `2026-07-24-runtime-matched-baseline-rerun.md`, and
   `2026-07-23-community-results-k-layout-migration.md`.
4. `docs/10-k2-results.md` (active K=2 numbers, the four closed baseline
   contrasts, gate recalibration, cross-K loading, rerank degradation),
   `docs/05-results.md` (Phase 1/2), `docs/10-k-ablation-analysis.md` (K ablation).
5. `paper/STATE.md` — manuscript baseline, last updated 2026-07-21 and stale for
   K=2 execution progress.

`docs/01-…` through `docs/10-…` are numbered in intended reading order.

## Environment and commands

**The checked-in `.venv` is broken on this machine.** Its console-script shebangs
point at `/root/ludandaye/reaserch/skill-LLM/.venv/bin/python3` (this tree was
moved from a Linux server), and `.venv/bin/python3` symlinks to system Python
3.9. Do not diagnose test failures through `.venv`.

Working local invocation (Anaconda Python 3.13 on PATH as `python3`, with numpy /
rank_bm25 / bm25s / openai available):

```bash
PYTHONPATH=. python3 -m pytest -q                      # 68 passed, 1 skipped
PYTHONPATH=. python3 -m pytest tests/test_gate.py -q   # single file
PYTHONPATH=. python3 -m pytest -q -k route_variant     # single test / pattern
```

The 1 skip and anything touching `sragents` or `sentence_transformers` need those
packages installed; they are absent from the Anaconda interpreter. Consequently
`./scripts/smoke.sh` (hardcodes `.venv/bin/sragents`) and `scripts/run_k_ablation.sh`
(hard-requires executable `.venv/bin/sragents` and `.venv/bin/python`) **cannot run
locally as-is**. To restore them, rebuild the venv rather than editing the scripts:

```bash
python3.12 -m venv .venv                               # 3.10–3.12; K=2 runners reject others
.venv/bin/pip install -e ".[dev,full]" -e external/SR-Agents
.venv/bin/pytest -q && ./scripts/smoke.sh
```

`external/` and `results/` are gitignored. `external/SR-Agents` is a separate
checkout pinned at `277fd8d` — the exact commit the frozen rerank/select prompts
reference — and supplies the benchmark corpus (`data/bench/corpus/corpus.json`,
232 MB, 26,262 skills) and instances. Scripts that shell out to SR-Agents export
`PYTHONPATH="$REPO:$REPO/external/SR-Agents/src"`.

Experiment entry points all take configuration through required environment
variables and fail loudly on omission (`: "${VAR:?…}"`); read the script header
before invoking:

```bash
MODEL=<tag> API_BASE=<url> PILOT=1 ./scripts/run_phase0.sh
TAG=<tag> MODEL=<tag> API_BASE=http://localhost:8000/v1 TRACKB=1 ./scripts/run_multimodel.sh
```

`MODEL` must equal `TAG` and must never change mid-run — it is baked into
generation cache keys.

## Architecture

### Layer 1 — retrieval core (`hyskill/`)

Pure, injectable, dependency-light. Network clients and sentence-transformers are
constructed lazily so unit tests never import them.

- `generator.py` — hypothetical `SKILL.md` generation over an OpenAI-compatible
  endpoint. Three prompt templates (`SKILL_TEMPLATE`, `PASSAGE_TEMPLATE`,
  `SENTENCE_TEMPLATE`) define the three "imagination granularities".
  `hypothetical_cache_key(model_tag, temperature, template, query, sample_index)`
  is the content-addressed key that makes every later stage a cache hit — the
  whole nested-prefix K design (`K=1` is a prefix of `K=2` …) depends on it.
- `parser.py` — splits `SKILL.md` text into `meta`/`body`/`code` fields; tolerates
  LLM-added outer ```` ```markdown ```` wrappers.
- `embedder.py`, `fusion.py` (RRF), `bm25.py` (`bm25s` with `rank_bm25` fallback;
  rank order is identical, absolute scores are not).
- Four retrieval variants, all speaking the SR-Agents retriever protocol
  (`build_index` / `retrieve`):
  - `naive_hyde.py` — faithful HyDE port: single vector, mean of K passage
    embeddings + query anchor, full-text corpus index. Three granularities via
    `template={skill|passage|sentence}`.
  - `retriever.py` (`HySkillRetriever`) — 4-path multi-view: meta↔meta,
    body↔body, code↔code (code-bearing skills only), BM25(query→full text),
    fused with RRF. Query vector is mixed into the meta centroid with weight
    `1/(K+1)` (HyDE eq. 8). Falls back to dense+BM25 when generation fails.
  - `two_stage.py` — fusion recall (`recall_k=50`) then single-vector precision
    rerank; generation is a cache hit from stage 1.
- `plugin.py` — registers `hyskill` and `naive_hyde` with SR-Agents and, on
  import, **monkey-patches upstream `llm_rerank`'s hardcoded `max_tokens=4096`
  down to 1024** so long rerank prompts don't 400 on an 8K-context server. This
  bugfix is versioned here deliberately; the baseline protocol depends on it.

Corpus embeddings are cached as `.npz` keyed on corpus file identity + encoder id
+ id set, so index building is paid once per machine.

### Layer 2 — gating and routing (`scripts/`)

- `scripts/gate.py` — the S1/S2 loading gate, three subcommands:
  `signals` (S1 = cos(hypothetical centroid, top-1 skill); S2 = fraction of top-1
  sentences *not* covered by the imagination) → `calibrate` (largest conservative
  τ with `precision ≥ p_min` on a 20% validation split) → `apply` (writes a gated
  retrieval file where blocked instances get `retrieved=[]`, which makes
  SR-Agents inject nothing = bare behaviour). **Zero new LLM calls** — every
  signal reuses the cached imagination. This is the paper's cost argument.
  Note `tau2`'s positive label is "**Bare** answered this correctly", which is why
  re-running Bare forces gate recalibration (see the baseline-rerun plan).
- `scripts/route_variant.py` — domain-level granularity router: score each
  variant by mean nDCG@10 on the same 20%/seed-0 validation split, pick the
  winner, copy it to a routed file with `metadata.router` recording the decision.
  Validation ids must be excluded from all downstream test reporting.
- `warm_cache.py` (pre-fill generations, resumable), `significance.py` /
  `phase2_significance.py` (paired bootstrap), `analyze.py` /
  `summarize_multimodel.py` (aggregation + cost audit).

### Layer 3 — K=2 downstream pipeline and provenance

This is where most recent code lives, and it is written defensively: every stage
binds a **request identity hash** and refuses to reuse or emit anything it cannot
prove. Pure typed helpers live in `hyskill/`, CLIs are thin shells in `scripts/`.

- `hyskill/downstream_reuse.py` — request identity, reuse eligibility, and failure
  taxonomy (`success` / `selector_fallback` / `infra_transient` / `method_failure`
  / `unclassified_error`). The taxonomy is load-bearing: `method_failure` stays in
  the denominator and scores incorrect.
- `hyskill/loading_metrics.py` — decision-level loading metrics. Keeps
  loaded-skill precision, loading rate, and gold-load rate as **three distinct
  quantities**.
- `hyskill/k2_answer_provenance.py` — strict provenance reconstruction
  (`formal_direct` / `posthoc_structural` / `formal_retry_after_import`) with
  per-line and whole-file SHA binding.
- `hyskill/k_ablation.py` — frozen constants and validators for the K ablation
  (`K_VALUES=(1,2,4,8,10)`, five domains, six variants, `EXPECTED_TOTAL_ROWS=3970`,
  `VAL_FRACTION=0.2`, `VAL_SEED=0`).
- Orchestration: `scripts/run_k2_main.sh` (routed Always/Gated per model-domain)
  and `run_k2_select_main.sh` chain
  `export_k2_loading_decisions` → `audit_k2_reuse` → `run_k2_answers` →
  `evaluate_k2_answers` → `validate_k2_downstream`, then
  `summarize_k2_*` → `export_k2_public_pack` / `export_k2_fleet_pack`.

Selection and answering are **separate stages**: Hy+Select reads the persisted
selection decision and must not call the selector again while answering.

### Data flow

```
external/SR-Agents/data/bench/   frozen corpus + instances (gitignored)
   → results/                    local-only run tree, huge, gitignored
     (hyp_cache, multimodel/<TAG>, k-ablation, k2-main, k2-formal-v1,
      k2-preservation-20260724)
   → community-results/<TAG>/    small, committed, PR-reviewable evidence
```

`community-results/` directory contract (see `community-results/README.md`):

| Path | Contents |
|---|---|
| `<TAG>/k2/` | active paper experiment: 12 files (11 for DeepSeek/Yi, no Select) |
| `<TAG>/k4/` | archived historical K=4 main experiment (fixed reference) |
| `<TAG>/baselines-runtime-matched/` | Bare / native Rerank / BM25+Select under the K=2 runtime |
| `<TAG>/k-ablation/` | joint K={1,2,4,8,10} pack: 5 files |
| `<TAG>/imagination_full_k{1,2,4,8,10}.*` | full nested prefix caches — stay at model root |
| `qwen3.5-4b-reference/baselines-native/` | K-independent shared evidence |
| `baselines-runtime-matched-fleet/` | the four baseline contrasts (K2M001 evidence) |
| `k2-gate-recalibration-v2/` | gate recalibration + sensitivity check |
| `k2-vs-k4-loading-analysis/` | cross-K loading comparison |
| `k2-fleet/`, `k-ablation-fleet/` | cross-model aggregates |

Never mix per-instance evidence, summaries, and manifests across those
directories, and never move `k-ablation/`, `imagination_full_k*`, or
`baselines-native/` into `k2/` or `k4/`.

## Current state (as of 2026-07-25, end of the K2M001 round)

`main` is at `28c11ef` and everything below is pushed. The K=2 project is
**experimentally complete**: 48,110 baseline answers + 28,300 decisions across
108 jobs, `valid=true`, and the four baseline contrasts are all significant on
held-out data. K2M001 is closed.

Committed this round, in order: `d64a77f` K=4 migration (70 files, SHA-verified)
· `c6a3206` K=2 downstream pipeline · `58fa20e` handoff docs · `0ec7537` gate
recalibration pack · `a349ba5` results doc · `4915baa` AGENTS.md refresh ·
`9c0ddc3` runtime-matched baseline packs · `4f5542a`+`eee8a9f` citation scoping ·
`28c11ef` K=2 vs K=4 loading analysis.

The second worktree
`/Users/a1-6/importantfile/Research/skill-LLM-baseline-run-20260724` (branch
`codex/runtime-matched-baselines-20260724`, at `35b2f80`) holds the baseline
implementation: six `hyskill/runtime_matched_*.py` modules, twenty
`scripts/*runtime_matched*.py` CLIs, eleven test files, `runtime-staging/`. It is
pushed and clean. Its `results/` (gitignored) holds the raw 409 MB recovered from
the five servers plus the v2 formal tree — that is local-only evidence, not a
deliverable.

Four untracked paths in `main` are **deliberately not committed**: `paper/`
(contains two nested Overleaf repos), `error.log`, `package.json`,
`package-lock.json`.

### Frozen decisions from this round

- **The paper is built on K=2 + gating**; K=4 remains a fixed reference in
  `<TAG>/k4/`. Do not switch the main line back — every baseline in this round
  was run against K=2 retrieval inputs.
- **Gated figures are cited from the published `k2/` packs**, not from
  `k2-gate-recalibration-v2/`. The recalibration is a sensitivity check: it
  changes no verdict and makes three of the four contrasts stronger, so `k2/` is
  the conservative side.
- **Runtime identity is parameter-only.** The formal K=2 answers' own
  `runtime_identity` has eight fields and no hardware column, so
  `runtime_identity_key` was narrowed to match; `hardware`/`source` are reported
  via `runtime_context_key` and a `runtime_context_varies` record. This is the
  only validation gate narrowed this round (branch commit `35b2f80`).

### Two metric traps

- **`loaded_skill_precision` is not comparable across K.** K=4's Always arm is
  not 100% loaded (yi15-9b returns nothing on 38.0% of medcalcbench), so its
  denominator silently drops retrieval failures and the metric reads too high.
  Use `gold_load_rate` for any cross-K loading comparison.
- **`nvidia-smi` utilization cannot tell you whether a vLLM job is saturated.**
  It shows ~100% whenever a kernel is resident. Read vLLM's own
  `Running / Waiting / KV cache usage` triple instead. Getting this wrong left a
  job running at 1/20 speed for hours this round.

### Remaining work (outside the K2M001 scope)

- `paper/latex/sections/experiments.tex` is still an entirely K=4 manuscript and
  asserts that routing and gating are both significant on Qwen4 — the K=2 result
  is −0.13 pp with a CI crossing zero, p=0.8808.
- Analysis and Conclusion sections are skeletons; the method overview figure and
  the AAAI-27 reproducibility checklist are missing.
- Both Overleaf repos need syncing; keep EN/ZH content aligned.
- CLEAN001: remove the temporary staging SSH key (`codex-k2-stage-20260723`)
  from the three servers where it was installed.

## Non-negotiable rules

Git hygiene:

- **Never** `git add .`, `git add paper/`, `git clean`, or a broad reset. Stage an
  explicit reviewed whitelist. `paper/latex/` and `paper/latex-zh/` are nested
  separate git repos wired to Overleaf and are gitignored from the root.
- Commit subjects are imperative with a lowercase scope prefix (`docs: …`,
  `data: …`), one logical change each. Push only when asked; never force-push.

Scientific integrity (full statements in `AGENTS.md`):

- **`unavailable` ≠ 0.** DeepSeek-7B and Yi-1.5-9B cannot fit the 50-candidate
  Select/Rerank prompt; their cells are `unavailable`, never zero, never failures.
- Deterministic `method_failure` rows stay in the denominator and score incorrect.
  Do not silently filter them.
- A confidence interval containing zero supports only "no detected difference" —
  never equivalence, "no loss", or superiority.
- Never compare a seven-model average against a five-model average.
- Loaded-skill precision, loading rate, gold-load rate, and answer accuracy are
  four different quantities and must not share the label "accuracy".
- Do not re-run completed inference to repair provenance. Missing pre-run
  manifests lower the provenance level; they are not a reason to resample. Rerun
  only on a concrete integrity failure (changed answer text, unresolvable ID/skill
  mapping, SHA mismatch, proven K=4 contamination).
- Never overwrite `k4/`, `k-ablation/`, `baselines-native/`, raw server outputs, or
  another worktree's dirty state. Recompute SHA on every transfer.
- External downloads use China-hosted mirrors with no silent overseas fallback;
  mirror provenance does not replace exact revision + SHA checks.
- Answer counting trap: the formal tree's 100 top-level non-loading JSONL files
  are 80 answer files **plus** 20 `*.selection.jsonl`. Answer counters must exclude
  selection files — the answer total is 56,600, never 70,750.

Remote servers (inventory in `AGENTS.md`): the table is a snapshot, not permission
to restart jobs. Formal K=2 inference is complete. Inspect hostname, processes,
GPU, disk, and result paths read-only before any remote operation; never kill a
service or touch an unrelated dirty worktree. Passwords are deliberately not
stored in the repo. Do not claim real-time monitoring unless a polling mechanism
is actually running.

## Code and test conventions

`AGENTS.md` holds the full style guide. In short: Python 3.10+, four-space
indent, type annotations (the K=2 layer uses `TypedDict` / `Literal` / `TypeAlias`
heavily), top-level imports, short single-purpose functions, `snake_case`,
English comments, no formatter configured — match neighbouring code. Raise
specific actionable exceptions; **never silently fall back or hide a service
failure**, since a hidden fallback silently corrupts an experiment.

Tests are `pytest`, named `tests/test_<module>.py`, and favour fixture-backed
integration over mocks; deterministic inputs live in `tests/fixtures/`. Add
focused coverage for changed behaviour only — there is no coverage target. Run the
full suite plus the smoke script when touching retrieval, the plugin, or an output
schema.
