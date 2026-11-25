#!/usr/bin/env python3
"""E7: Candidate-source × downstream (2×2) end-to-end accuracy grid.

Compares {BM25, Routed HySkill} × {Select, Rerank} across the five
50-candidate-eligible models and four rule-scored domains on the held-out
split (2,265 instances/model), alongside Bare and Gated references.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from hyskill.k_ablation import (
    require_object,
    require_string,
    sha256_file,
)
from hyskill.revision_stats import (
    hierarchical_paired_bootstrap,
    paired_cells,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
FIVE_MODELS: tuple[str, ...] = (
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
)
UNAVAILABLE_MODELS: frozenset[str] = frozenset(
    {"deepseek7b", "deepseek-7b", "yi15-9b", "yi-1.5-9b"}
)
RULE_DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
)
EXPECTED_HELD_OUT: dict[str, int] = {
    "theoremqa": 598,
    "logicbench": 608,
    "medcalcbench": 880,
    "champ": 179,
}
EXPECTED_MODEL_TOTAL: int = 2265


def score_answer_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_arm: str,
) -> list[dict[str, object]]:
    """Filter held-out rows for ``expected_arm`` and score method_failure as 0.0."""

    scored: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        context = f"answer[{index}]"
        model = require_string(row.get("model"), f"{context}.model")
        if model.lower() in UNAVAILABLE_MODELS or model not in FIVE_MODELS:
            raise ValueError(
                f"model is unavailable or outside the 5-model E7 support: model={model}"
            )
        arm = require_string(row.get("arm"), f"{context}.arm")
        if arm != expected_arm:
            continue
        is_val = row.get("is_validation")
        if not isinstance(is_val, bool):
            raise ValueError(f"{context}.is_validation must be bool: {is_val!r}")
        if is_val:
            continue
        domain = require_string(row.get("domain"), f"{context}.domain")
        if domain not in RULE_DOMAINS:
            raise ValueError(f"unexpected domain in E7 grid: domain={domain}")
        iid = require_string(row.get("instance_id"), f"{context}.instance_id")
        key = (model, domain, iid)
        if key in seen:
            raise ValueError(f"duplicate answer row: {key}")
        seen.add(key)
        category = require_string(row.get("failure_category"), f"{context}.failure_category")
        if category not in ("success", "method_failure", "selector_fallback"):
            raise ValueError(f"unresolved failure_category={category} at {key}")
        correct_raw = row.get("correct")
        if not isinstance(correct_raw, bool):
            raise ValueError(f"{context}.correct must be bool: {correct_raw!r}")
        accuracy = 0.0 if category == "method_failure" else (1.0 if correct_raw else 0.0)
        scored.append(
            {
                "model": model,
                "domain": domain,
                "arm": expected_arm,
                "instance_id": iid,
                "failure_category": category,
                "accuracy": accuracy,
            }
        )
    return scored


def _read_jsonl_gz(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing answer file: path={path}")
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(require_object(json.loads(line), f"{path}:{line_no}"))
    return rows


def _assert_support_counts(rows: Sequence[Mapping[str, object]], arm_label: str) -> dict[tuple[str, str], frozenset[str]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["model"]), str(row["domain"]))].add(str(row["instance_id"]))
    out: dict[tuple[str, str], frozenset[str]] = {}
    for model in FIVE_MODELS:
        total = 0
        for domain in RULE_DOMAINS:
            ids = frozenset(grouped[(model, domain)])
            if len(ids) != EXPECTED_HELD_OUT[domain]:
                raise ValueError(
                    f"held-out count mismatch for {arm_label} {model}:{domain}: "
                    f"expected={EXPECTED_HELD_OUT[domain]}, actual={len(ids)}"
                )
            out[(model, domain)] = ids
            total += len(ids)
        if total != EXPECTED_MODEL_TOTAL:
            raise ValueError(f"model held-out total mismatch for {arm_label} {model}: {total}")
    return out


def _arm_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_model: dict[str, list[float]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    method_failures = 0
    for row in rows:
        acc = float(row["accuracy"])  # type: ignore[arg-type]
        m = str(row["model"])
        d = str(row["domain"])
        by_model[m].append(acc)
        by_cell[(m, d)].append(acc)
        if row.get("failure_category") == "method_failure":
            method_failures += 1
    per_model = {m: sum(by_model[m]) / len(by_model[m]) for m in FIVE_MODELS}
    cell_macro = sum(sum(v) / len(v) for v in by_cell.values()) / len(by_cell)
    instance_weighted = sum(per_model.values()) / len(FIVE_MODELS)
    return {
        "cell_macro_accuracy": cell_macro,
        "instance_weighted_accuracy": instance_weighted,
        "per_model_accuracy": per_model,
        "method_failures_in_denominator": method_failures,
        "n_instances": len(rows),
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rel_path_str(raw: str | Path) -> str:
    p = Path(raw).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        parts = p.parts
        if "results" in parts:
            idx = parts.index("results")
            return "/".join(parts[idx:])
        return p.name


def _write_with_manifest(path: Path, content: str, manifest_meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

    sanitized_meta = dict(manifest_meta)
    if isinstance(sanitized_meta.get("input_sha256"), dict):
        sanitized_meta["input_sha256"] = {
            _rel_path_str(k): v
            for k, v in sanitized_meta["input_sha256"].items()  # type: ignore[union-attr]
        }

    m_path = Path(str(path) + ".manifest.json")
    manifest = {
        **sanitized_meta,
        "output_path": _rel_path_str(path),
        "output_sha256": sha256_file(path),
        "output_bytes": path.stat().st_size,
    }
    tmp_m = m_path.with_name(f"{m_path.name}.tmp")
    tmp_m.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_m.replace(m_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E7 2x2 candidate-pool grid.")
    parser.add_argument("--community-root", type=Path, default=Path("community-results"))
    parser.add_argument(
        "--hy-rerank-relpath",
        default="revision-ijcai27/hy_rerank/answer_per_instance.jsonl.gz",
        help="Path under <community-root>/<TAG>/ to the exported hy_rerank answer file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("community-results/revision-ijcai27-fleet"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    community_root = args.community_root.resolve()
    out_dir = args.output_dir.resolve()
    samples = int(args.bootstrap_samples)
    seed = int(args.bootstrap_seed)

    raw_k2: list[dict[str, object]] = []
    raw_base: list[dict[str, object]] = []
    raw_hy_rerank: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}

    for model in FIVE_MODELS:
        k2_p = community_root / model / "k2" / "answer_per_instance.jsonl.gz"
        base_p = community_root / model / "baselines-runtime-matched" / "answer_per_instance.jsonl.gz"
        hyr_p = community_root / model / str(args.hy_rerank_relpath)
        for p in (k2_p, base_p, hyr_p):
            input_hashes[str(p)] = sha256_file(p)
        raw_k2.extend(_read_jsonl_gz(k2_p))
        raw_base.extend(_read_jsonl_gz(base_p))
        raw_hy_rerank.extend(_read_jsonl_gz(hyr_p))

    arms: dict[str, list[dict[str, object]]] = {
        "bare": score_answer_rows(raw_base, expected_arm="bare"),
        "select_bm25": score_answer_rows(raw_base, expected_arm="select_bm25"),
        "always_rerank": score_answer_rows(raw_base, expected_arm="always_rerank"),
        "routed_gated": score_answer_rows(raw_k2, expected_arm="routed_gated"),
        "routed_select": score_answer_rows(raw_k2, expected_arm="routed_select"),
        "hy_rerank": score_answer_rows(raw_hy_rerank, expected_arm="hy_rerank"),
    }

    reference_allowed: dict[tuple[str, str], frozenset[str]] | None = None
    for arm_name, arm_rows in arms.items():
        allowed = _assert_support_counts(arm_rows, arm_name)
        if reference_allowed is None:
            reference_allowed = allowed
        elif allowed != reference_allowed:
            raise ValueError(f"held-out ID mismatch between arms: {arm_name}")

    assert reference_allowed is not None

    def allow_five(model: str, domain: str) -> frozenset[str]:
        return reference_allowed.get((model, domain), frozenset())

    contrasts = {
        "hy_rerank_vs_always_rerank": hierarchical_paired_bootstrap(
            paired_cells(arms["hy_rerank"], arms["always_rerank"], "accuracy", allow_five),
            samples=samples,
            seed=seed,
            name="e7:hy_rerank_vs_always_rerank",
        ),
        "routed_select_vs_select_bm25": hierarchical_paired_bootstrap(
            paired_cells(arms["routed_select"], arms["select_bm25"], "accuracy", allow_five),
            samples=samples,
            seed=seed,
            name="e7:routed_select_vs_select_bm25",
        ),
        "routed_select_vs_hy_rerank": hierarchical_paired_bootstrap(
            paired_cells(arms["routed_select"], arms["hy_rerank"], "accuracy", allow_five),
            samples=samples,
            seed=seed,
            name="e7:routed_select_vs_hy_rerank",
        ),
        "select_bm25_vs_always_rerank": hierarchical_paired_bootstrap(
            paired_cells(arms["select_bm25"], arms["always_rerank"], "accuracy", allow_five),
            samples=samples,
            seed=seed,
            name="e7:select_bm25_vs_always_rerank",
        ),
        "hy_rerank_vs_routed_gated": hierarchical_paired_bootstrap(
            paired_cells(arms["hy_rerank"], arms["routed_gated"], "accuracy", allow_five),
            samples=samples,
            seed=seed,
            name="e7:hy_rerank_vs_routed_gated",
        ),
        "hy_rerank_vs_bare": hierarchical_paired_bootstrap(
            paired_cells(arms["hy_rerank"], arms["bare"], "accuracy", allow_five),
            samples=samples,
            seed=seed,
            name="e7:hy_rerank_vs_bare",
        ),
    }

    hyr_ci = contrasts["hy_rerank_vs_always_rerank"]["ci95"]  # type: ignore[index]
    hyr_significant_positive = bool(hyr_ci[0] > 0.0)
    claim_guidance = (
        "Hy+Rerank significantly improves over BM25+Rerank on held-out 5-model support; "
        "both Select and Rerank benefit from the routed HySkill candidate pool."
        if hyr_significant_positive
        else "The 95% CI for Hy+Rerank minus BM25+Rerank contains zero or is negative; "
        "per Spec §7, scope the downstream candidate-pool claim to Select (Hy+Select > BM25+Select) "
        "and report no detected improvement (or negative difference) under listwise Rerank."
    )

    summaries = {name: _arm_summary(rows) for name, rows in arms.items()}
    payload: dict[str, object] = {
        "experiment": "E7_candidate_source_by_downstream_grid",
        "models": list(FIVE_MODELS),
        "domains": list(RULE_DOMAINS),
        "held_out_instances_per_model": EXPECTED_MODEL_TOTAL,
        "arms": summaries,
        "paired_contrasts": contrasts,
        "hy_rerank_beats_bm25_rerank_significantly": hyr_significant_positive,
        "claim_guidance": claim_guidance,
    }

    c_r = contrasts["hy_rerank_vs_always_rerank"]
    c_s = contrasts["routed_select_vs_select_bm25"]
    md_lines = [
        "# E7: Candidate Source × Downstream Strategy (2×2 Held-Out Accuracy Grid)",
        "",
        "- **Support:** 5 models (`glm4-9b`, `llama31-8b`, `mistral7b`, `qwen3.5-4b-reference`, `qwen35-9b`) × 4 rule domains (`2,265` held-out instances/model). `method_failure` rows remain in the denominator and score `0`.",
        "",
        "## 2×2 Grid (Cell-Macro % / Instance-Weighted %)",
        "",
        "| Candidate Pool | + Select | + Rerank |",
        "|---|---:|---:|",
        f"| **BM25 top-50** | {summaries['select_bm25']['cell_macro_accuracy']*100:.2f}% ({summaries['select_bm25']['instance_weighted_accuracy']*100:.2f}%) | {summaries['always_rerank']['cell_macro_accuracy']*100:.2f}% ({summaries['always_rerank']['instance_weighted_accuracy']*100:.2f}%) |",
        f"| **Routed HySkill (K=2) top-50** | {summaries['routed_select']['cell_macro_accuracy']*100:.2f}% ({summaries['routed_select']['instance_weighted_accuracy']*100:.2f}%) | {summaries['hy_rerank']['cell_macro_accuracy']*100:.2f}% ({summaries['hy_rerank']['instance_weighted_accuracy']*100:.2f}%) |",
        "",
        f"- **References (5-model support):** `Bare` = {summaries['bare']['cell_macro_accuracy']*100:.2f}% ({summaries['bare']['instance_weighted_accuracy']*100:.2f}%), `Routed Gated (0 extra calls)` = {summaries['routed_gated']['cell_macro_accuracy']*100:.2f}% ({summaries['routed_gated']['instance_weighted_accuracy']*100:.2f}%).",
        "",
        "## Key Paired Contrasts (Hierarchical Bootstrap, 10,000 Samples)",
        "",
        f"- **Hy+Select vs. BM25+Select:** `{c_s['difference']*100:+.2f} pp` 95% CI `[{c_s['ci95'][0]*100:+.2f}, {c_s['ci95'][1]*100:+.2f}]` (`p={c_s['p_two_sided']:.4f}`)",
        f"- **Hy+Rerank vs. BM25+Rerank:** `{c_r['difference']*100:+.2f} pp` 95% CI `[{c_r['ci95'][0]*100:+.2f}, {c_r['ci95'][1]*100:+.2f}]` (`p={c_r['p_two_sided']:.4f}`)",
        "",
        "## Claim Guidance",
        "",
        f"- {claim_guidance}",
        "",
    ]

    manifest_meta: dict[str, object] = {
        "git_head": _git_head(REPO_ROOT),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "input_sha256": input_hashes,
    }
    _write_with_manifest(out_dir / "e7_pool_grid.json", json.dumps(payload, indent=2) + "\n", manifest_meta)
    _write_with_manifest(out_dir / "e7_pool_grid.md", "\n".join(md_lines), manifest_meta)


if __name__ == "__main__":
    main()
