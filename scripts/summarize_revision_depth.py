#!/usr/bin/env python3
"""E1: Held-out depth hit-rate and nDCG@10 comparison for candidate pools.

Main table covers the five 50-candidate-eligible models across the four
rule-scored downstream domains, comparing BM25, runtime-matched Rerank
(with omitted_candidate_count diagnostics and strict unfilled hit@k), and
K=2 routed HySkill. BigCodeBench is reported separately as a supplement
row using committed k4 llm_rerank rankings (no omission counts stored).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from hyskill.k_ablation import (
    require_integer,
    sha256_file,
)
from hyskill.revision_rankings import (
    HIT_DEPTHS,
    RankingRow,
    instance_metrics,
    load_committed_rankings,
    load_runtime_bm25_rankings,
    load_runtime_rerank_rankings,
)
from hyskill.revision_stats import (
    hierarchical_paired_bootstrap,
    paired_cells,
    test_ids,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
FIVE_MODELS: tuple[str, ...] = (
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
)
UNAVAILABLE_MODELS: frozenset[str] = frozenset({"deepseek7b", "yi15-9b"})
# k4 BM25 for qwen3.5-4b-reference came from SR-Agents BM25Retriever; these four
# used FastBM25, the scorer behind the shared runtime candidate file.
FASTBM25_K4_MODELS: tuple[str, ...] = ("glm4-9b", "llama31-8b", "mistral7b", "qwen35-9b")
RULE_DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
)
SUPPLEMENT_DOMAIN: str = "bigcodebench"
EXPECTED_TEST_COUNTS: dict[str, int] = {
    "theoremqa": 598,
    "logicbench": 608,
    "medcalcbench": 880,
    "champ": 179,
    "bigcodebench": 912,
}
EXPECTED_RULE_TEST_TOTAL: int = 2265
DEPTH_METRICS: tuple[str, ...] = tuple(f"hit@{k}" for k in HIT_DEPTHS) + ("nDCG@10",)


def strict_hit(row: RankingRow, k: int) -> float:
    """Return 1.0 iff a gold skill appears in the unfilled prefix of top-k.

    Positions beyond ``len(ranked_ids) - omitted_count`` were appended in BM25
    order when the listwise output parsed incompletely and are treated as misses
    under the strict auxiliary definition.
    """

    if k <= 0:
        raise ValueError(f"depth k must be positive: k={k}")
    omitted_raw: int | None = row.get("omitted_count")
    if omitted_raw is None:
        raise ValueError(
            "strict_hit requires a non-null omitted_count: "
            f"model={row.get('model')}, domain={row.get('domain')}, "
            f"instance_id={row.get('instance_id')}"
        )
    omitted: int = require_integer(omitted_raw, "omitted_count")
    ranked: list[str] = row["ranked_ids"]
    if omitted < 0 or omitted > len(ranked):
        raise ValueError(
            f"omitted_count out of range: omitted={omitted}, ranked_len={len(ranked)}"
        )
    parsed_count: int = len(ranked) - omitted
    window: list[str] = ranked[: min(k, parsed_count)]
    gold: set[str] = set(row["gold_skill_ids"])
    return 1.0 if any(skill_id in gold for skill_id in window) else 0.0


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


def _sanitize_manifest_meta(manifest_meta: dict[str, object]) -> dict[str, object]:
    sanitized = dict(manifest_meta)
    if isinstance(sanitized.get("input_sha256"), dict):
        sanitized["input_sha256"] = {
            _rel_path_str(k): v
            for k, v in sanitized["input_sha256"].items()  # type: ignore[union-attr]
        }
    return sanitized


def _write_json_with_manifest(
    path: Path,
    payload: dict[str, object],
    manifest_meta: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)

    manifest_path = Path(str(path) + ".manifest.json")
    manifest = {
        **_sanitize_manifest_meta(manifest_meta),
        "output_path": _rel_path_str(path),
        "output_sha256": sha256_file(path),
        "output_bytes": path.stat().st_size,
    }
    tmp_m = manifest_path.with_name(f"{manifest_path.name}.tmp")
    tmp_m.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_m.replace(manifest_path)


def _write_text_with_manifest(
    path: Path,
    text: str,
    manifest_meta: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

    manifest_path = Path(str(path) + ".manifest.json")
    manifest = {
        **_sanitize_manifest_meta(manifest_meta),
        "output_path": _rel_path_str(path),
        "output_sha256": sha256_file(path),
        "output_bytes": path.stat().st_size,
    }
    tmp_m = manifest_path.with_name(f"{manifest_path.name}.tmp")
    tmp_m.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_m.replace(manifest_path)


def _score_rows(
    rows: Sequence[RankingRow],
    *,
    include_strict: bool,
) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for row in rows:
        if row["model"] in UNAVAILABLE_MODELS:
            raise ValueError(f"unavailable model must not enter E1: model={row['model']}")
        metrics = instance_metrics(row)
        record: dict[str, object] = {
            "model": row["model"],
            "domain": row["domain"],
            "arm": row["arm"],
            "instance_id": row["instance_id"],
            "omitted_count": row["omitted_count"],
            "ranked_len": len(row["ranked_ids"]),
            **metrics,
        }
        if include_strict:
            for depth in HIT_DEPTHS:
                record[f"strict_hit@{depth}"] = strict_hit(row, depth)
        scored.append(record)
    return scored


def _domain_test_id_map(
    rows: Sequence[dict[str, object]],
    expected_domains: Sequence[str],
    models: Sequence[str] = FIVE_MODELS,
) -> dict[tuple[str, str], frozenset[str]]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        model = str(row["model"])
        domain = str(row["domain"])
        if domain in expected_domains:
            grouped[(model, domain)].add(str(row["instance_id"]))
    out: dict[tuple[str, str], frozenset[str]] = {}
    for model in models:
        total_model_test = 0
        for domain in expected_domains:
            full_ids = frozenset(grouped[(model, domain)])
            if not full_ids:
                raise ValueError(f"missing domain rows: model={model}, domain={domain}")
            held_out = test_ids(full_ids)
            expected_n = EXPECTED_TEST_COUNTS[domain]
            if len(held_out) != expected_n:
                raise ValueError(
                    f"test split count mismatch: model={model}, domain={domain}, "
                    f"expected={expected_n}, actual={len(held_out)}"
                )
            out[(model, domain)] = held_out
            total_model_test += len(held_out)
        if tuple(expected_domains) == RULE_DOMAINS and total_model_test != EXPECTED_RULE_TEST_TOTAL:
            raise ValueError(
                f"rule-domain held-out total mismatch: model={model}, "
                f"expected={EXPECTED_RULE_TEST_TOTAL}, actual={total_model_test}"
            )
    return out


def _macro_cell_means(
    rows: Sequence[dict[str, object]],
    allowed_map: dict[tuple[str, str], frozenset[str]],
    metrics: Sequence[str],
) -> dict[str, float]:
    by_cell: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model"]), str(row["domain"]))
        if key in allowed_map and str(row["instance_id"]) in allowed_map[key]:
            by_cell[key].append(row)
    if set(by_cell) != set(allowed_map):
        raise ValueError("missing cells in macro_cell_means")
    means: dict[str, float] = {}
    for metric in metrics:
        cell_vals = [
            sum(float(r[metric]) for r in cell_rows) / len(cell_rows)
            for _key, cell_rows in sorted(by_cell.items())
        ]
        means[metric] = sum(cell_vals) / len(cell_vals)
    return means


def _fill_diagnostics(
    rerank_rows: Sequence[dict[str, object]],
    allowed_map: dict[tuple[str, str], frozenset[str]],
) -> dict[str, dict[str, float]]:
    by_cell: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rerank_rows:
        key = (str(row["model"]), str(row["domain"]))
        if key in allowed_map and str(row["instance_id"]) in allowed_map[key]:
            by_cell[key].append(row)
    result: dict[str, dict[str, float]] = {}
    for depth in HIT_DEPTHS:
        slot_fracs: list[float] = []
        tail_rates: list[float] = []
        for _key, cell_rows in sorted(by_cell.items()):
            cell_slot: list[float] = []
            cell_tail: list[float] = []
            for row in cell_rows:
                omitted = require_integer(row.get("omitted_count"), "omitted_count")
                ranked_len = require_integer(row.get("ranked_len"), "ranked_len")
                parsed = ranked_len - omitted
                filled_slots = max(0, min(depth, ranked_len) - parsed)
                cell_slot.append(filled_slots / depth)
                cell_tail.append(1.0 if parsed < depth else 0.0)
            slot_fracs.append(sum(cell_slot) / len(cell_slot))
            tail_rates.append(sum(cell_tail) / len(cell_tail))
        result[f"hit@{depth}"] = {
            "mean_filled_slot_fraction": sum(slot_fracs) / len(slot_fracs),
            "rank_k_is_filled_rate": sum(tail_rates) / len(tail_rates),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E1 candidate-pool depth metrics.")
    parser.add_argument("--community-root", type=Path, default=Path("community-results"))
    parser.add_argument(
        "--runtime-rerank-root",
        type=Path,
        required=True,
        help="Path to baselines-unified-20260725 containing <TAG>/decisions/*always_rerank*.jsonl",
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
    community_root: Path = args.community_root.resolve()
    rerank_root: Path = args.runtime_rerank_root.resolve()
    out_dir: Path = args.output_dir.resolve()
    samples: int = int(args.bootstrap_samples)
    seed: int = int(args.bootstrap_seed)

    bm25_all: list[dict[str, object]] = []
    bm25_rule: list[dict[str, object]] = []
    routed_all: list[dict[str, object]] = []
    k4_rerank_all: list[dict[str, object]] = []
    runtime_rerank_rule: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}

    for tag in FIVE_MODELS:
        k4_path = community_root / tag / "k4" / "retrieval_top50.jsonl.gz"
        k2_path = community_root / tag / "k2" / "retrieval_top50.jsonl.gz"
        input_hashes[str(k4_path)] = sha256_file(k4_path)
        input_hashes[str(k2_path)] = sha256_file(k2_path)
        for dec_path in sorted((rerank_root / tag / "decisions").glob("*always_rerank*.jsonl")):
            input_hashes[str(dec_path.resolve())] = sha256_file(dec_path)

        routed_all.extend(_score_rows(load_committed_rankings(tag, "routed_k2", community_root), include_strict=False))
        bm25_rule.extend(_score_rows(load_runtime_bm25_rankings(rerank_root, tag), include_strict=False))
        if tag in FASTBM25_K4_MODELS:
            bm25_all.extend(
                _score_rows(load_committed_rankings(tag, "bm25", community_root), include_strict=False)
            )
            k4_rerank_all.extend(
                _score_rows(load_committed_rankings(tag, "llm_rerank", community_root), include_strict=False)
            )
        runtime_rerank_rule.extend(
            _score_rows(load_runtime_rerank_rankings(rerank_root, tag), include_strict=True)
        )

    rule_allowed = _domain_test_id_map(routed_all, RULE_DOMAINS)
    routed_fastbm25: list[dict[str, object]] = [
        row for row in routed_all if row["model"] in FASTBM25_K4_MODELS
    ]
    bigcode_allowed = _domain_test_id_map(
        routed_fastbm25, (SUPPLEMENT_DOMAIN,), FASTBM25_K4_MODELS
    )
    five_domain_allowed = _domain_test_id_map(
        routed_fastbm25, RULE_DOMAINS + (SUPPLEMENT_DOMAIN,), FASTBM25_K4_MODELS
    )

    def allow_rule(model: str, domain: str) -> frozenset[str]:
        return rule_allowed.get((model, domain), frozenset())

    def allow_bigcode(model: str, domain: str) -> frozenset[str]:
        return bigcode_allowed.get((model, domain), frozenset())

    def allow_five(model: str, domain: str) -> frozenset[str]:
        return five_domain_allowed.get((model, domain), frozenset())

    bm25_means = _macro_cell_means(bm25_rule, rule_allowed, DEPTH_METRICS)
    rerank_means = _macro_cell_means(runtime_rerank_rule, rule_allowed, DEPTH_METRICS)
    rerank_strict_means = _macro_cell_means(
        runtime_rerank_rule,
        rule_allowed,
        tuple(f"strict_hit@{k}" for k in HIT_DEPTHS),
    )
    routed_means = _macro_cell_means(routed_all, rule_allowed, DEPTH_METRICS)
    fill_diag = _fill_diagnostics(runtime_rerank_rule, rule_allowed)

    hy_vs_rerank: dict[str, object] = {}
    hy_vs_bm25: dict[str, object] = {}
    hy_vs_rerank_strict: dict[str, object] = {}
    for metric in DEPTH_METRICS:
        hy_vs_rerank[metric] = hierarchical_paired_bootstrap(
            paired_cells(routed_all, runtime_rerank_rule, metric, allow_rule),
            samples=samples,
            seed=seed,
            name=f"e1:routed_vs_runtime_rerank:{metric}",
        )
        hy_vs_bm25[metric] = hierarchical_paired_bootstrap(
            paired_cells(routed_all, bm25_rule, metric, allow_rule),
            samples=samples,
            seed=seed,
            name=f"e1:routed_vs_bm25:{metric}",
        )
    for depth in HIT_DEPTHS:
        strict_alias_rows = [
            {**row, f"hit@{depth}": row[f"strict_hit@{depth}"]}
            for row in runtime_rerank_rule
        ]
        hy_vs_rerank_strict[f"hit@{depth}"] = hierarchical_paired_bootstrap(
            paired_cells(routed_all, strict_alias_rows, f"hit@{depth}", allow_rule),
            samples=samples,
            seed=seed,
            name=f"e1:routed_vs_runtime_rerank_strict:hit@{depth}",
        )

    per_domain_top1: dict[str, object] = {}
    for domain in RULE_DOMAINS:
        def allow_single(m: str, d: str, target: str = domain) -> frozenset[str]:
            return rule_allowed.get((m, d), frozenset()) if d == target else frozenset()

        vs_rerank = hierarchical_paired_bootstrap(
            paired_cells(routed_all, runtime_rerank_rule, "hit@1", allow_single),
            samples=samples,
            seed=seed,
            name=f"e1:top1:{domain}:routed_vs_runtime_rerank",
        )
        vs_bm25 = hierarchical_paired_bootstrap(
            paired_cells(routed_all, bm25_rule, "hit@1", allow_single),
            samples=samples,
            seed=seed,
            name=f"e1:top1:{domain}:routed_vs_bm25",
        )
        per_domain_top1[domain] = {
            "routed_hit@1": vs_rerank["mean_a"],
            "runtime_rerank_hit@1": vs_rerank["mean_b"],
            "bm25_hit@1": vs_bm25["mean_b"],
            "routed_vs_runtime_rerank": vs_rerank,
            "routed_vs_bm25": vs_bm25,
        }

    # Supplement: BigCodeBench (k4 llm_rerank, no omitted_count stored)
    bigcode_bm25 = _macro_cell_means(bm25_all, bigcode_allowed, DEPTH_METRICS)
    bigcode_rerank = _macro_cell_means(k4_rerank_all, bigcode_allowed, DEPTH_METRICS)
    bigcode_routed = _macro_cell_means(routed_all, bigcode_allowed, DEPTH_METRICS)
    bigcode_hy_vs_rerank: dict[str, object] = {}
    bigcode_hy_vs_bm25: dict[str, object] = {}
    for metric in DEPTH_METRICS:
        bigcode_hy_vs_rerank[metric] = hierarchical_paired_bootstrap(
            paired_cells(routed_all, k4_rerank_all, metric, allow_bigcode),
            samples=samples,
            seed=seed,
            name=f"e1:bigcodebench:routed_vs_k4_rerank:{metric}",
        )
        bigcode_hy_vs_bm25[metric] = hierarchical_paired_bootstrap(
            paired_cells(routed_all, bm25_all, metric, allow_bigcode),
            samples=samples,
            seed=seed,
            name=f"e1:bigcodebench:routed_vs_bm25:{metric}",
        )

    # 5-domain k4 reference check against preliminary session hit@10 (+0.100 [+0.039, +0.159])
    five_domain_k4_hit10 = hierarchical_paired_bootstrap(
        paired_cells(routed_all, k4_rerank_all, "hit@10", allow_five),
        samples=samples,
        seed=seed,
        name="e1:five_domain_k4_reference:hit@10",
    )

    payload: dict[str, object] = {
        "experiment": "E1_candidate_pool_depth",
        "models": list(FIVE_MODELS),
        "rule_domains": list(RULE_DOMAINS),
        "held_out_instances_per_model_rule": EXPECTED_RULE_TEST_TOTAL,
        "main_four_domains": {
            "means": {
                "bm25": bm25_means,
                "always_rerank": rerank_means,
                "always_rerank_strict": rerank_strict_means,
                "routed_k2": routed_means,
            },
            "fill_diagnostics_always_rerank": fill_diag,
            "routed_vs_always_rerank": hy_vs_rerank,
            "routed_vs_always_rerank_strict": hy_vs_rerank_strict,
            "routed_vs_bm25": hy_vs_bm25,
            "per_domain_top1": per_domain_top1,
        },
        "supplement_bigcodebench": {
            "domain": SUPPLEMENT_DOMAIN,
            "held_out_instances_per_model": EXPECTED_TEST_COUNTS[SUPPLEMENT_DOMAIN],
            "note": "Uses committed k4 llm_rerank rankings; omitted_candidate_count is not recorded in k4.",
            "means": {
                "bm25": bigcode_bm25,
                "llm_rerank_k4": bigcode_rerank,
                "routed_k2": bigcode_routed,
            },
            "routed_vs_llm_rerank_k4": bigcode_hy_vs_rerank,
            "routed_vs_bm25": bigcode_hy_vs_bm25,
        },
        "five_domain_k4_hit10_reference": five_domain_k4_hit10,
    }

    lines: list[str] = [
        "# E1: Held-Out Candidate-Pool Hit Rate by Depth",
        "",
        "- **Support (main table):** 5 Select/Rerank-eligible models (`glm4-9b`, `llama31-8b`, `mistral7b`, `qwen3.5-4b-reference`, `qwen35-9b`) × 4 rule-scored domains (`theoremqa`, `logicbench`, `medcalcbench`, `champ`), held-out split only (`2,265` instances/model). DeepSeek-7B and Yi-1.5-9B are `unavailable`.",
        "- **Rerank source (main table):** Runtime-matched `always_rerank` decisions (records `omitted_candidate_count`).",
        "- **BM25 source (main table):** `ordered_candidate_ids` of the same `always_rerank` decisions, i.e. the shared FastBM25 candidate file every Select/Rerank arm received. BM25 and Rerank therefore score one candidate pool per instance, and their `hit@50` must coincide.",
        "- **BM25 scorer note:** `qwen3.5-4b-reference`'s `k4` BM25 came from SR-Agents `BM25Retriever`; the other four models' `k4` BM25 is FastBM25. The main table does not use `k4` BM25. The BigCodeBench supplement and the 5-domain reference use only those four FastBM25 models.",
        "",
        "## Main Table (4 Rule Domains, Macro Mean Across 5×4 Cells)",
        "",
        "| Metric | BM25 | Native Rerank | Rerank (Strict Unfilled) | Routed HySkill (K=2) | Hy − Rerank (95% CI, p) | Hy − BM25 (95% CI, p) | Rerank Filled-Slot % |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for depth in HIT_DEPTHS:
        m = f"hit@{depth}"
        comp_r = hy_vs_rerank[m]  # type: ignore[index]
        comp_b = hy_vs_bm25[m]  # type: ignore[index]
        fill_pct = fill_diag[m]["mean_filled_slot_fraction"] * 100.0
        lines.append(
            f"| `{m}` | {bm25_means[m]:.3f} | {rerank_means[m]:.3f} | "
            f"{rerank_strict_means[f'strict_{m}']:.3f} | {routed_means[m]:.3f} | "
            f"{comp_r['difference']:+.3f} [{comp_r['ci95'][0]:+.3f}, {comp_r['ci95'][1]:+.3f}] (p={comp_r['p_two_sided']:.4f}) | "
            f"{comp_b['difference']:+.3f} [{comp_b['ci95'][0]:+.3f}, {comp_b['ci95'][1]:+.3f}] (p={comp_b['p_two_sided']:.4f}) | "
            f"{fill_pct:.1f}% |"
        )
    comp_ndcg_r = hy_vs_rerank["nDCG@10"]  # type: ignore[index]
    comp_ndcg_b = hy_vs_bm25["nDCG@10"]  # type: ignore[index]
    lines.append(
        f"| `nDCG@10` | {bm25_means['nDCG@10']:.3f} | {rerank_means['nDCG@10']:.3f} | — | "
        f"{routed_means['nDCG@10']:.3f} | "
        f"{comp_ndcg_r['difference']:+.3f} [{comp_ndcg_r['ci95'][0]:+.3f}, {comp_ndcg_r['ci95'][1]:+.3f}] (p={comp_ndcg_r['p_two_sided']:.4f}) | "
        f"{comp_ndcg_b['difference']:+.3f} [{comp_ndcg_b['ci95'][0]:+.3f}, {comp_ndcg_b['ci95'][1]:+.3f}] (p={comp_ndcg_b['p_two_sided']:.4f}) | — |"
    )
    lines.extend(
        [
            "",
            "## Per-Domain Top-1 Hit (`hit@1`, 5 Models)",
            "",
            "| Domain | BM25 | Native Rerank | Routed HySkill | Hy − Rerank (95% CI) |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for domain in RULE_DOMAINS:
        d_info = per_domain_top1[domain]  # type: ignore[index]
        vr = d_info["routed_vs_runtime_rerank"]
        lines.append(
            f"| `{domain}` | {d_info['bm25_hit@1']:.3f} | {d_info['runtime_rerank_hit@1']:.3f} | "
            f"{d_info['routed_hit@1']:.3f} | "
            f"{vr['difference']:+.3f} [{vr['ci95'][0]:+.3f}, {vr['ci95'][1]:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## BigCodeBench Supplement (4 FastBM25 Models, `k4` BM25 and `llm_rerank`, No Omission Count Stored)",
            "",
            "| Metric | BM25 | `k4` LLM Rerank | Routed HySkill | Hy − Rerank (95% CI) |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric in DEPTH_METRICS:
        br = bigcode_hy_vs_rerank[metric]  # type: ignore[index]
        lines.append(
            f"| `{metric}` | {bigcode_bm25[metric]:.3f} | {bigcode_rerank[metric]:.3f} | "
            f"{bigcode_routed[metric]:.3f} | "
            f"{br['difference']:+.3f} [{br['ci95'][0]:+.3f}, {br['ci95'][1]:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Reconciliation Note",
            "",
            f"- 5-domain reference (4 FastBM25 models, `k4` `llm_rerank`, `hit@10`): diff `{five_domain_k4_hit10['difference']:+.3f}` "
            f"95% CI `[{five_domain_k4_hit10['ci95'][0]:+.3f}, {five_domain_k4_hit10['ci95'][1]:+.3f}]` "
            "(the earlier session check, +0.100 [+0.039, +0.159], pooled qwen3.5-4b-reference with its SR-Agents rerank pool and is superseded; the main table scopes to the 4 rule domains where runtime-matched rerank recorded `omitted_candidate_count`).",
            "- Note on tail positions: deeper ranks (`k >= 20`) increasingly overlap positions appended in BM25 order when listwise output truncated early; report both deployed rankings and the filled-slot diagnostic.",
            "",
        ]
    )

    manifest_meta: dict[str, object] = {
        "git_head": _git_head(REPO_ROOT),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "input_sha256": input_hashes,
    }
    _write_json_with_manifest(out_dir / "e1_depth.json", payload, manifest_meta)
    _write_text_with_manifest(out_dir / "e1_depth.md", "\n".join(lines), manifest_meta)


if __name__ == "__main__":
    main()
