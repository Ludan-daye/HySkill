#!/usr/bin/env python3
"""E2, E3, and E6: HyDE comparison, router necessity, and label-free alternatives.

Consumes per-instance recomputed K=2 metrics and rankings across all seven
models and five benchmark domains on the held-out test split.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from hyskill.fusion import rrf
from hyskill.k_ablation import (
    DOMAINS,
    FIXED_VARIANTS,
    expected_router_pick,
    require_float,
    require_object,
    require_string,
    sha256_file,
)
from hyskill.revision_rankings import RankingRow, instance_metrics
from hyskill.revision_stats import (
    hierarchical_paired_bootstrap,
    paired_cells,
    test_ids,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS: tuple[str, ...] = (
    "deepseek7b",
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
    "yi15-9b",
)
EXPECTED_TEST_COUNTS: dict[str, int] = {
    "theoremqa": 598,
    "logicbench": 608,
    "medcalcbench": 880,
    "champ": 179,
    "bigcodebench": 912,
}
REPORT_METRICS: tuple[str, ...] = ("nDCG@10", "Recall@50")


def best_fixed_global(
    metrics_rows: Sequence[Mapping[str, object]],
    metric: str = "nDCG@10",
) -> str:
    """Pick the single fixed variant maximizing pooled held-out mean."""

    by_variant: dict[str, list[float]] = {variant: [] for variant in FIXED_VARIANTS}
    for row in metrics_rows:
        if row.get("split") != "test":
            continue
        arm = str(row.get("arm", ""))
        if arm in by_variant:
            by_variant[arm].append(require_float(row.get(metric), f"row.{metric}"))
    missing = [variant for variant, vals in by_variant.items() if not vals]
    if missing:
        raise ValueError(f"best_fixed_global missing test rows for variants: {missing}")
    scores = {variant: sum(vals) / len(vals) for variant, vals in by_variant.items()}
    winner, _deg = expected_router_pick(scores)
    return winner


def best_fixed_per_domain(
    metrics_rows: Sequence[Mapping[str, object]],
    metric: str = "nDCG@10",
) -> dict[str, str]:
    """Pick one fixed variant per domain maximizing pooled 7-model held-out mean."""

    by_dom_var: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {variant: [] for variant in FIXED_VARIANTS}
    )
    for row in metrics_rows:
        if row.get("split") != "test":
            continue
        domain = str(row.get("domain", ""))
        arm = str(row.get("arm", ""))
        if domain and arm in FIXED_VARIANTS:
            by_dom_var[domain][arm].append(require_float(row.get(metric), f"row.{metric}"))
    if not by_dom_var:
        raise ValueError("best_fixed_per_domain received no test rows")
    picks: dict[str, str] = {}
    for domain in sorted(by_dom_var):
        missing = [v for v, vals in by_dom_var[domain].items() if not vals]
        if missing:
            raise ValueError(f"domain {domain} missing test rows for variants: {missing}")
        scores = {v: sum(vals) / len(vals) for v, vals in by_dom_var[domain].items()}
        winner, _deg = expected_router_pick(scores)
        picks[domain] = winner
    return picks


def lodo_pick(
    metrics_rows: Sequence[Mapping[str, object]],
    model: str,
    held_out: str,
    metric: str = "nDCG@10",
) -> str:
    """Pick a fixed variant for ``(model, held_out)`` using validation rows of the other 4 domains."""

    if not model or not held_out:
        raise ValueError("model and held_out must be non-empty")
    by_variant: dict[str, list[float]] = {variant: [] for variant in FIXED_VARIANTS}
    seen_domains: set[str] = set()
    for row in metrics_rows:
        if row.get("model") != model or row.get("split") != "validation":
            continue
        domain = str(row.get("domain", ""))
        if domain == held_out:
            continue
        arm = str(row.get("arm", ""))
        if arm in by_variant:
            seen_domains.add(domain)
            by_variant[arm].append(require_float(row.get(metric), f"row.{metric}"))
    if not seen_domains:
        raise ValueError(f"lodo_pick found no other-domain validation rows: model={model}, held_out={held_out}")
    missing = [variant for variant, vals in by_variant.items() if not vals]
    if missing:
        raise ValueError(f"lodo_pick missing validation variants for model={model}: {missing}")
    scores = {variant: sum(vals) / len(vals) for variant, vals in by_variant.items()}
    winner, _deg = expected_router_pick(scores)
    return winner


def _read_jsonl_gz(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing jsonl.gz file: path={path}")
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                rows.append(require_object(json.loads(line), f"{path}:{line_no}"))
    return rows


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


def _derive_validation_router_picks(
    metrics_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in metrics_rows:
        if row.get("split") != "validation":
            continue
        arm = str(row.get("arm", ""))
        if arm in FIXED_VARIANTS:
            key = (str(row["model"]), str(row["domain"]), arm)
            grouped[key].append(require_float(row.get("nDCG@10"), "nDCG@10"))
    picks: dict[tuple[str, str], str] = {}
    for model in MODELS:
        for domain in DOMAINS:
            scores = {
                v: sum(grouped[(model, domain, v)]) / len(grouped[(model, domain, v)])
                for v in FIXED_VARIANTS
            }
            winner, _deg = expected_router_pick(scores)
            picks[(model, domain)] = winner
    return picks


def _build_rrf_test_rows(
    rankings_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    allowed_test: Mapping[tuple[str, str], frozenset[str]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in MODELS:
        by_dom_inst_var: dict[tuple[str, str], dict[str, RankingRow]] = defaultdict(dict)
        for raw in rankings_by_model[model]:
            arm = require_string(raw.get("arm"), "ranking.arm")
            if arm not in FIXED_VARIANTS:
                continue
            domain = require_string(raw.get("domain"), "ranking.domain")
            iid = require_string(raw.get("instance_id"), "ranking.instance_id")
            if iid not in allowed_test[(model, domain)]:
                continue
            by_dom_inst_var[(domain, iid)][arm] = {
                "model": model,
                "domain": domain,
                "arm": arm,
                "instance_id": iid,
                "gold_skill_ids": [str(x) for x in raw["gold_skill_ids"]],  # type: ignore[index]
                "ranked_ids": [str(x) for x in raw["ranked_ids"]],  # type: ignore[index]
                "omitted_count": None,
            }
        for (domain, iid), var_map in sorted(by_dom_inst_var.items()):
            if set(var_map) != set(FIXED_VARIANTS):
                raise ValueError(f"missing fixed variants for RRF: {model}:{domain}:{iid}")
            paths = [var_map[v]["ranked_ids"] for v in FIXED_VARIANTS]
            fused = rrf(paths, k=60, top_k=50)
            sample_row = var_map[FIXED_VARIANTS[0]]
            fused_row: RankingRow = {
                "model": model,
                "domain": domain,
                "arm": "rrf_5variants",
                "instance_id": iid,
                "gold_skill_ids": sample_row["gold_skill_ids"],
                "ranked_ids": [skill_id for skill_id, _score in fused],
                "omitted_count": None,
            }
            out.append(
                {
                    "model": model,
                    "domain": domain,
                    "arm": "rrf_5variants",
                    "instance_id": iid,
                    "split": "test",
                    **instance_metrics(fused_row),
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize E2, E3, and E6 router/HyDE results.")
    parser.add_argument("--community-root", type=Path, default=Path("community-results"))
    parser.add_argument("--rankings-root", type=Path, default=Path("results/revision-ijcai27"))
    parser.add_argument("--encoder-slug", default="all-MiniLM-L6-v2")
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
    rankings_root = args.rankings_root.resolve()
    out_dir = args.output_dir.resolve()
    slug = str(args.encoder_slug)
    samples = int(args.bootstrap_samples)
    seed = int(args.bootstrap_seed)

    all_metrics: list[dict[str, object]] = []
    rankings_by_model: dict[str, list[dict[str, object]]] = {}
    input_hashes: dict[str, str] = {}

    for model in MODELS:
        m_path = community_root / model / "revision-ijcai27" / f"metrics_{slug}.jsonl.gz"
        r_path = rankings_root / model / f"rankings_{slug}.jsonl.gz"
        input_hashes[str(m_path)] = sha256_file(m_path)
        input_hashes[str(r_path)] = sha256_file(r_path)
        all_metrics.extend(_read_jsonl_gz(m_path))
        rankings_by_model[model] = _read_jsonl_gz(r_path)

    # Verify split counts strictly against validation_ids
    full_ids_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    test_ids_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in all_metrics:
        if row.get("arm") == "routed":
            cell = (str(row["model"]), str(row["domain"]))
            full_ids_by_cell[cell].add(str(row["instance_id"]))
            if row.get("split") == "test":
                test_ids_seen[cell].add(str(row["instance_id"]))

    allowed_test: dict[tuple[str, str], frozenset[str]] = {}
    for model in MODELS:
        for domain in DOMAINS:
            expected_held_out = test_ids(frozenset(full_ids_by_cell[(model, domain)]))
            if len(expected_held_out) != EXPECTED_TEST_COUNTS[domain]:
                raise ValueError(
                    f"unexpected test count for {model}:{domain}: {len(expected_held_out)}"
                )
            if test_ids_seen[(model, domain)] != set(expected_held_out):
                raise ValueError(f"split field disagrees with test_ids for {model}:{domain}")
            allowed_test[(model, domain)] = expected_held_out

    def allow_all(model: str, domain: str) -> frozenset[str]:
        return allowed_test.get((model, domain), frozenset())

    by_arm_test: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_metrics:
        if row.get("split") == "test":
            by_arm_test[str(row["arm"])].append(row)

    picks = _derive_validation_router_picks(all_metrics)
    passage_pick_count = sum(1 for pick in picks.values() if pick == "naive_passage")

    # E2: routed vs naive_passage
    e2_contrasts = {
        metric: hierarchical_paired_bootstrap(
            paired_cells(by_arm_test["routed"], by_arm_test["naive_passage"], metric, allow_all),
            samples=samples,
            seed=seed,
            name=f"e2:routed_vs_naive_passage:{metric}",
        )
        for metric in REPORT_METRICS
    }
    e2_payload: dict[str, object] = {
        "experiment": "E2_hyskill_vs_hyde_passage",
        "encoder_slug": slug,
        "n_models": len(MODELS),
        "n_domains": len(DOMAINS),
        "passage_pick_count": passage_pick_count,
        "total_cells": len(picks),
        "router_picks": {f"{m}:{d}": p for (m, d), p in sorted(picks.items())},
        "routed_vs_naive_passage": e2_contrasts,
    }

    # E3: router vs global best fixed & per-domain best fixed; pick distribution; hyskill - passage
    global_winner = best_fixed_global(all_metrics, "nDCG@10")
    domain_winners = best_fixed_per_domain(all_metrics, "nDCG@10")
    per_domain_oracle_rows = [
        row
        for row in all_metrics
        if row.get("split") == "test" and row.get("arm") == domain_winners.get(str(row["domain"]))
    ]
    dist_by_domain: dict[str, dict[str, int]] = {}
    for domain in DOMAINS:
        counts = Counter(picks[(m, domain)] for m in MODELS)
        dist_by_domain[domain] = {v: counts.get(v, 0) for v in FIXED_VARIANTS}
    overall_counts = Counter(picks.values())

    hyskill_vs_passage_by_domain: dict[str, object] = {}
    for domain in DOMAINS:
        def allow_dom(m: str, d: str, target: str = domain) -> frozenset[str]:
            return allowed_test.get((m, d), frozenset()) if d == target else frozenset()

        hyskill_vs_passage_by_domain[domain] = {
            metric: hierarchical_paired_bootstrap(
                paired_cells(by_arm_test["hyskill"], by_arm_test["naive_passage"], metric, allow_dom),
                samples=samples,
                seed=seed,
                name=f"e3:hyskill_vs_passage:{domain}:{metric}",
            )
            for metric in REPORT_METRICS
        }

    e3_payload: dict[str, object] = {
        "experiment": "E3_router_evidence",
        "encoder_slug": slug,
        "global_best_fixed_variant": global_winner,
        "per_domain_best_fixed_variant": domain_winners,
        "routed_vs_global_best_fixed": {
            metric: hierarchical_paired_bootstrap(
                paired_cells(by_arm_test["routed"], by_arm_test[global_winner], metric, allow_all),
                samples=samples,
                seed=seed,
                name=f"e3:routed_vs_global_fixed:{metric}",
            )
            for metric in REPORT_METRICS
        },
        "routed_vs_per_domain_best_fixed": {
            metric: hierarchical_paired_bootstrap(
                paired_cells(by_arm_test["routed"], per_domain_oracle_rows, metric, allow_all),
                samples=samples,
                seed=seed,
                name=f"e3:routed_vs_per_domain_fixed:{metric}",
            )
            for metric in REPORT_METRICS
        },
        "router_pick_distribution_overall": {v: overall_counts.get(v, 0) for v in FIXED_VARIANTS},
        "router_pick_distribution_by_domain": dist_by_domain,
        "hyskill_vs_naive_passage_overall": {
            metric: hierarchical_paired_bootstrap(
                paired_cells(by_arm_test["hyskill"], by_arm_test["naive_passage"], metric, allow_all),
                samples=samples,
                seed=seed,
                name=f"e3:hyskill_vs_passage:overall:{metric}",
            )
            for metric in REPORT_METRICS
        },
        "hyskill_vs_naive_passage_by_domain": hyskill_vs_passage_by_domain,
    }

    # E6: (a) 5-variant RRF; (b) Leave-one-domain-out (LODO)
    rrf_rows = _build_rrf_test_rows(rankings_by_model, allowed_test)
    lodo_picks = {
        (model, domain): lodo_pick(all_metrics, model, domain, "nDCG@10")
        for model in MODELS
        for domain in DOMAINS
    }
    lodo_rows = [
        row
        for row in all_metrics
        if row.get("split") == "test"
        and row.get("arm") == lodo_picks[(str(row["model"]), str(row["domain"]))]
    ]
    e6_payload: dict[str, object] = {
        "experiment": "E6_label_free_alternatives",
        "encoder_slug": slug,
        "rrf_5variants": {
            "rrf_vs_routed": {
                metric: hierarchical_paired_bootstrap(
                    paired_cells(rrf_rows, by_arm_test["routed"], metric, allow_all),
                    samples=samples,
                    seed=seed,
                    name=f"e6:rrf_vs_routed:{metric}",
                )
                for metric in REPORT_METRICS
            },
            "rrf_vs_naive_passage": {
                metric: hierarchical_paired_bootstrap(
                    paired_cells(rrf_rows, by_arm_test["naive_passage"], metric, allow_all),
                    samples=samples,
                    seed=seed,
                    name=f"e6:rrf_vs_naive_passage:{metric}",
                )
                for metric in REPORT_METRICS
            },
        },
        "leave_one_domain_out": {
            "lodo_picks": {f"{m}:{d}": p for (m, d), p in sorted(lodo_picks.items())},
            "lodo_vs_routed": {
                metric: hierarchical_paired_bootstrap(
                    paired_cells(lodo_rows, by_arm_test["routed"], metric, allow_all),
                    samples=samples,
                    seed=seed,
                    name=f"e6:lodo_vs_routed:{metric}",
                )
                for metric in REPORT_METRICS
            },
            "lodo_vs_per_domain_best_fixed": {
                metric: hierarchical_paired_bootstrap(
                    paired_cells(lodo_rows, per_domain_oracle_rows, metric, allow_all),
                    samples=samples,
                    seed=seed,
                    name=f"e6:lodo_vs_per_domain_fixed:{metric}",
                )
                for metric in REPORT_METRICS
            },
            "lodo_vs_naive_passage": {
                metric: hierarchical_paired_bootstrap(
                    paired_cells(lodo_rows, by_arm_test["naive_passage"], metric, allow_all),
                    samples=samples,
                    seed=seed,
                    name=f"e6:lodo_vs_naive_passage:{metric}",
                )
                for metric in REPORT_METRICS
            },
        },
    }

    e2_n = e2_contrasts["nDCG@10"]
    e2_r = e2_contrasts["Recall@50"]
    e3_g = e3_payload["routed_vs_global_best_fixed"]["nDCG@10"]  # type: ignore[index]
    e3_d = e3_payload["routed_vs_per_domain_best_fixed"]["nDCG@10"]  # type: ignore[index]
    e6_rrf = e6_payload["rrf_5variants"]["rrf_vs_routed"]["nDCG@10"]  # type: ignore[index]
    e6_lodo = e6_payload["leave_one_domain_out"]["lodo_vs_routed"]["nDCG@10"]  # type: ignore[index]

    md_lines = [
        "# E2, E3, and E6: HyDE Baseline, Router Necessity, and Label-Free Alternatives",
        "",
        f"- **Support:** 7 models × 5 domains (`3,177` held-out instances/model), encoder `{slug}`.",
        "",
        "## E2: Routed HySkill vs. HyDE (`naive_passage`)",
        "",
        f"- **Router selected `naive_passage` in `{passage_pick_count} / {len(picks)}` model-domain cells** (must be disclosed alongside the paired gain).",
        f"- `nDCG@10`: `{e2_n['mean_a']:.3f}` vs `{e2_n['mean_b']:.3f}`, diff `{e2_n['difference']:+.3f}` 95% CI `[{e2_n['ci95'][0]:+.3f}, {e2_n['ci95'][1]:+.3f}]` (`p={e2_n['p_two_sided']:.4f}`).",
        f"- `Recall@50`: `{e2_r['mean_a']:.3f}` vs `{e2_r['mean_b']:.3f}`, diff `{e2_r['difference']:+.3f}` 95% CI `[{e2_r['ci95'][0]:+.3f}, {e2_r['ci95'][1]:+.3f}]` (`p={e2_r['p_two_sided']:.4f}`).",
        "",
        "## E3: Router Evidence",
        "",
        f"- **(a) vs. Global Best Fixed (`{global_winner}`):** `nDCG@10` diff `{e3_g['difference']:+.3f}` 95% CI `[{e3_g['ci95'][0]:+.3f}, {e3_g['ci95'][1]:+.3f}]` (`p={e3_g['p_two_sided']:.4f}`).",
        f"- **(b) vs. Per-Domain Post-Hoc Best Fixed (`{domain_winners}`):** `nDCG@10` diff `{e3_d['difference']:+.3f}` 95% CI `[{e3_d['ci95'][0]:+.3f}, {e3_d['ci95'][1]:+.3f}]` (`p={e3_d['p_two_sided']:.4f}`).",
        "- **Router Pick Counts (out of 7 models per domain):**",
        "",
        "| Domain | `naive_sentence` | `naive_passage` | `naive_skill` | `hyskill` | `two_stage` | `hyskill − passage` nDCG@10 (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for domain in DOMAINS:
        d_cnt = dist_by_domain[domain]
        hp = hyskill_vs_passage_by_domain[domain]["nDCG@10"]  # type: ignore[index]
        md_lines.append(
            f"| `{domain}` | {d_cnt['naive_sentence']} | {d_cnt['naive_passage']} | "
            f"{d_cnt['naive_skill']} | {d_cnt['hyskill']} | {d_cnt['two_stage']} | "
            f"{hp['difference']:+.3f} [{hp['ci95'][0]:+.3f}, {hp['ci95'][1]:+.3f}] |"
        )
    md_lines.extend(
        [
            "",
            "## E6: Label-Free Alternatives",
            "",
            f"- **(a) 5-Variant RRF Fusion (`k=60`) vs. Routed:** `nDCG@10` `{e6_rrf['mean_a']:.3f}` vs `{e6_rrf['mean_b']:.3f}`, diff `{e6_rrf['difference']:+.3f}` 95% CI `[{e6_rrf['ci95'][0]:+.3f}, {e6_rrf['ci95'][1]:+.3f}]` (`p={e6_rrf['p_two_sided']:.4f}`).",
            f"- **(b) Leave-One-Domain-Out (LODO) vs. Routed:** `nDCG@10` `{e6_lodo['mean_a']:.3f}` vs `{e6_lodo['mean_b']:.3f}`, diff `{e6_lodo['difference']:+.3f}` 95% CI `[{e6_lodo['ci95'][0]:+.3f}, {e6_lodo['ci95'][1]:+.3f}]` (`p={e6_lodo['p_two_sided']:.4f}`).",
            "",
        ]
    )

    manifest_meta: dict[str, object] = {
        "git_head": _git_head(REPO_ROOT),
        "encoder_slug": slug,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "input_sha256": input_hashes,
    }
    _write_with_manifest(out_dir / "e2_hyde.json", json.dumps(e2_payload, indent=2) + "\n", manifest_meta)
    _write_with_manifest(out_dir / "e3_router.json", json.dumps(e3_payload, indent=2) + "\n", manifest_meta)
    _write_with_manifest(out_dir / "e6_label_free.json", json.dumps(e6_payload, indent=2) + "\n", manifest_meta)
    _write_with_manifest(out_dir / "router.md", "\n".join(md_lines), manifest_meta)


if __name__ == "__main__":
    main()
