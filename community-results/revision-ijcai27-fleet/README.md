# IJCAI-27 revision experiment data

This index covers the E0–E7 revision artifacts. The complete source supplement
adds full retrieval rankings and raw E7 evidence to the existing metric and
summary packs. Original file bytes are preserved, including model responses,
method failures, interrupted attempts, and recorded runtime metadata.

## Data locations

Paths below are relative to `community-results/`.

| Path | Contents |
|---|---|
| `<TAG>/revision-ijcai27/metrics_<ENCODER>.jsonl.gz` | Per-instance metrics for all four encoders |
| `<TAG>/revision-ijcai27/rankings_<ENCODER>.jsonl.gz` | Full ordered top-50 skill IDs and gold IDs; original sibling manifests |
| `<TAG>/revision-ijcai27/reproduction.json` | Original MiniLM E0 reproduction report |
| `<TAG>/revision-ijcai27/hy_rerank/` | Compact E7 answers, decisions, usage, runtime identity, and validation |
| `<TAG>/revision-ijcai27/hy_rerank_source.tar.gz` | Raw E7 answers and reranker responses, evaluation, audits, runtime manifests, and logs |
| `revision-ijcai27-fleet/e*.json` and `e*.md` | Existing E1–E7 fleet analyses and their manifests |
| `revision-ijcai27-fleet/source-evidence/e7-inputs.tar.gz` | Exact E7 candidate pools, their manifests, and frozen validation IDs |
| `revision-ijcai27-fleet/source-evidence/e7-code.tar.gz` | Eleven source files bound by the E7 runtime manifests |
| `revision-ijcai27-fleet/source-evidence/operations.tar.gz` | E4/E7 launch scripts and orchestration logs |
| `revision-ijcai27-fleet/source-evidence/e0-operations.tar.gz` | E0 launch scripts, logs, and encoder provenance snapshots |
| `revision-ijcai27-fleet/source-evidence/encoders.e0.json` and `encoders.e4.json` | Exact encoder provenance used for MiniLM and the additional encoders |

The seven retrieval tags are `deepseek7b`, `glm4-9b`, `llama31-8b`, `mistral7b`,
`qwen3.5-4b-reference`, `qwen35-9b`, and `yi15-9b`. The encoders are
`all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `gte-base`, and `bge-large-en-v1.5`.
Each of the 28 ranking files has 23,820 rows: 3,970 instances × six arms,
including validation and held-out instances. Total formal ranking rows: 666,960.
Ranking rows have no split field; the corresponding metric rows retain the split.

E7 covers the five Select/Rerank-eligible models and four rule-scored domains.
It contains 14,150 final answers and 14,150 final decisions, with 11,325 held-out
answers. DeepSeek and Yi remain unavailable. The 127 final method failures remain
in the denominator. The five raw archives contain 347 files, including the GLM
`archive-interrupted-20261125/` directory; those interrupted files are preserved
history and must not be added to final answer counts. The 60 candidate/validation
input files are stored separately in `e7-inputs.tar.gz`.

## Integrity and provenance

`data_manifest.json` records every revision publication file with its size and
SHA-256, each archive member with its original size and SHA-256, source locations,
coverage, and validation results. Its inventory excludes itself and `SHA256SUMS`
to avoid recursive hashes. `SHA256SUMS` includes the manifest itself and all
inventory files; paths are relative to the repository root. Verify after cloning:

```sh
shasum -a 256 -c community-results/revision-ijcai27-fleet/SHA256SUMS
```

All 56 source ranking/manifest files and 441 source evidence files were compared
with server SHA-256 values. All formal ranking metrics were recomputed from the
published ranked IDs and matched the existing metric rows. E7 raw answer hashes,
decision fields, evaluation labels, candidate order, and validation membership
were matched to the existing compact packs. All 20 final domain audits are valid.
The 300 code-file references in the final E7 runtime manifests match the eleven
files in `e7-code.tar.gz`. Archive members were read back after compression.

Original manifests retain their historical server paths and generation commit.
They have not been rewritten to imply the new publication location was the
execution location. For rankings, map the original `results/revision-ijcai27/`
or `/data/hyskill-ijcai27/rankings/` location to the same model and filename under
`community-results/<TAG>/revision-ijcai27/`. The older MiniLM manifests lack
output hashes; the publication inventory supplies independently verified hashes.
Their encoder input hash resolves to `source-evidence/encoders.e0.json`, while
the E4 encoder input hash resolves to `source-evidence/encoders.e4.json`.
For raw E7 archives, prepend the recorded `source_root` to each member path to
recover its original location. Runtime paths, loopback endpoints, and device
identifiers are retained as provenance; no credentials were found in the
published source files.

The corpus, benchmark inputs, and previously published imagination caches retain
the source identities recorded in the manifests. Downloaded model weights,
virtual environments, package caches, and derived embedding caches are not part
of this experiment-output supplement. The runtime code snapshot is the exact
manifest-bound subset, not a replacement for the repository or its dependencies.

## Interpretation boundaries

MiniLM is the primary encoder. Original E0 reports retain their recorded gate
outcomes, including the accepted one-instance numerical/tie drift for Llama and
Qwen9; this upload does not change those outcomes. The initial DeepSeek BGE
diagnostic is retained under
`deepseek7b/revision-ijcai27/diagnostics/`, with its original reproduction report
at `deepseek7b/revision-ijcai27/reproduction_bge-base-en-v1.5.json`. Its 23,820
ranking rows include 70 rows that differ from the final E4 BGE ranking and are
excluded from the formal total above. It is not the MiniLM E0 gate.

The E7 candidate pools come from the frozen published K=2 routed retrieval,
not from the additional-encoder E4 rankings. E4 has seven-model retrieval support;
E7 has five-model downstream support. Preserve the scorer-specific support in
the fleet analyses, including the separate SR-Agents BM25 scorer for Qwen4.
Unchanged K=2, K=4, K-ablation, and runtime-matched baseline packs remain their
own protocols and are not replaced by this supplement.
