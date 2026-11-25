#!/usr/bin/env python3
"""Check one model's recomputed K=2 rankings against the published pack.

The recompute CLI writes per-instance rows and does not write an ``all``
split. This gate averages those rows itself for ``all``, ``validation``, and
``test``, then compares each published ``k_samples == 2`` cell with
``round(x, 3)``. Router picks are derived from validation nDCG@10. Top-50
overlap is reported and does not, by itself, fail the gate.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from hyskill.k_ablation import (
    ALL_VARIANTS,
    DOMAINS,
    FIXED_VARIANTS,
    expected_router_pick,
    require_float,
    require_integer,
    require_list,
    require_object,
    require_string,
    validation_ids,
)
from hyskill.revision_rankings import RankingRow, instance_metrics, load_committed_rankings


ENCODER_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
K_SAMPLES: int = 2
AGGREGATE_DOMAINS: frozenset[str] = frozenset({"__all_micro__", "__all_macro__"})
REVISION_TAGS: tuple[str, ...] = (
    "deepseek7b",
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
    "yi15-9b",
)
FULL_BM25_TAGS: tuple[str, ...] = (
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen3.5-4b-reference",
    "qwen35-9b",
)
FAST_BM25_TAGS: tuple[str, ...] = (
    "glm4-9b",
    "llama31-8b",
    "mistral7b",
    "qwen35-9b",
)
PARTIAL_BM25_TAGS: tuple[str, ...] = ("deepseek7b", "yi15-9b")
REFERENCE_BM25_TAG: str = "glm4-9b"
QWEN_BM25_TAG: str = "qwen3.5-4b-reference"
# Leading scores from the committed theoremqa_00000 BM25 row. They identify
# the two scorers without rebuilding the 26,262-document index on every run.
QWEN_BM25_TOP_SKILL: str = "theoremqa_000"
QWEN_BM25_TOP_SCORE: float = 75.10751
FAST_BM25_TOP_SKILL: str = "theoremqa_000"
FAST_BM25_TOP_SCORE: float = 28.06001
PUBLISHED_METRICS: tuple[str, ...] = tuple(
    f"{family}@{depth}" for depth in (1, 5, 10, 50) for family in ("Recall", "nDCG")
)


class MetricMismatch(TypedDict):
    """One published cell that disagrees at the third decimal or on ``n``."""

    domain: str
    variant: str
    split: str
    metric: str
    published: float
    recomputed: float | None
    n_published: int
    n_recomputed: int


class RouterMismatch(TypedDict):
    """One domain whose validation winner differs from the published pick."""

    domain: str
    published_pick: str | None
    recomputed_pick: str | None


class _MetricCell(TypedDict):
    value: float
    n: int


class _Bm25DomainStats(TypedDict):
    n: int
    test_n: int
    hit_at_50: float
    ndcg_at_10: float
    instance_ids: frozenset[str]
    gold: dict[str, tuple[str, ...]]
    ranked: dict[str, tuple[str, ...]]
    test_metrics: dict[str, tuple[float, float]]


def parse_args() -> argparse.Namespace:
    """Parse the one-tag reproduction check."""

    parser = argparse.ArgumentParser(
        description="Compare recomputed K=2 metrics with the published ablation pack."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--recomputed-metrics", required=True, type=Path)
    parser.add_argument("--recomputed-rankings", required=True, type=Path)
    parser.add_argument("--published-metrics", required=True, type=Path)
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--community-root", type=Path, default=Path("community-results"))
    parser.add_argument("--encoder", default=ENCODER_NAME)
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, object]:
    """Load one JSON object. A missing file or a non-object is an error."""

    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: path={path}")
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON file is not valid: path={path}, error={error}") from error
    return require_object(payload, str(path))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file, gunzipping when the name ends in ``.gz``."""

    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: path={path}")
    opener = gzip.open if path.name.endswith(".gz") else open
    rows: list[dict[str, object]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL row is not valid JSON: path={path}, line={line_number}, error={error}"
                ) from error
            rows.append(require_object(raw, f"{path}:{line_number}"))
    if not rows:
        raise ValueError(f"JSONL file has no rows: path={path}")
    return rows


def _mean(values: list[float], context: str) -> float:
    if not values:
        raise ValueError(f"cannot average an empty metric cell: context={context}")
    return sum(values) / len(values)


def _rounded_equal(left: float, right: float) -> bool:
    """Compare the way the reproduction gate is defined: third decimal place."""

    return round(left, 3) == round(right, 3)


def _require_ranking_ids(value: object, context: str) -> list[str]:
    raw_ids: list[object] = require_list(value, context)
    return [
        require_string(skill_id, f"{context}[{index}]")
        for index, skill_id in enumerate(raw_ids)
    ]


def _index_recomputed_metrics(
    rows: list[dict[str, object]],
    tag: str,
    expected_encoder: str = ENCODER_NAME,
) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    """Group per-instance rows by domain, arm, and split, including ``all``."""

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        context: str = f"recomputed_metrics[{index}]"
        model: str = require_string(row.get("model"), f"{context}.model")
        if model != tag:
            raise ValueError(
                f"recomputed metric model does not match tag: context={context}, "
                f"tag={tag}, model={model}"
            )
        encoder: str = require_string(row.get("encoder"), f"{context}.encoder")
        if encoder != expected_encoder:
            raise ValueError(
                f"recomputed metric encoder is not {expected_encoder}: "
                f"context={context}, encoder={encoder}"
            )
        k_samples: int = require_integer(row.get("k_samples"), f"{context}.k_samples")
        if k_samples != K_SAMPLES:
            raise ValueError(
                f"recomputed metric k_samples must be {K_SAMPLES}: "
                f"context={context}, k_samples={k_samples}"
            )
        domain: str = require_string(row.get("domain"), f"{context}.domain")
        if domain in AGGREGATE_DOMAINS or domain not in DOMAINS:
            raise ValueError(
                f"recomputed metric domain is not a benchmark domain: "
                f"context={context}, domain={domain}"
            )
        arm: str = require_string(row.get("arm"), f"{context}.arm")
        if arm not in ALL_VARIANTS:
            raise ValueError(
                f"recomputed metric arm is not a known variant: context={context}, arm={arm}"
            )
        instance_id: str = require_string(row.get("instance_id"), f"{context}.instance_id")
        split: str = require_string(row.get("split"), f"{context}.split")
        if split not in ("validation", "test"):
            raise ValueError(
                "recomputed metric split must be validation or test: "
                f"context={context}, split={split}"
            )
        identity: tuple[str, str, str] = (domain, arm, instance_id)
        if identity in seen:
            raise ValueError(
                "duplicate recomputed metric row: "
                f"domain={domain}, arm={arm}, instance_id={instance_id}"
            )
        seen.add(identity)
        for metric in PUBLISHED_METRICS:
            if metric not in row:
                raise ValueError(
                    f"recomputed metric row is missing {metric}: context={context}"
                )
            require_float(row[metric], f"{context}.{metric}")
        grouped[(domain, arm, split)].append(row)
        grouped[(domain, arm, "all")].append(row)
    return grouped


def _real_domains(
    grouped: dict[tuple[str, str, str], list[dict[str, object]]],
) -> list[str]:
    present: set[str] = {domain for domain, _arm, _split in grouped}
    return [domain for domain in DOMAINS if domain in present]


def _cell_from_rows(
    rows: list[dict[str, object]],
    metric: str,
    context: str,
) -> _MetricCell:
    values: list[float] = [
        require_float(row[metric], f"{context}.{metric}") for row in rows
    ]
    return {"value": _mean(values, context), "n": len(rows)}


def _recomputed_cell(
    grouped: dict[tuple[str, str, str], list[dict[str, object]]],
    real_domains: list[str],
    domain: str,
    variant: str,
    split: str,
    metric: str,
) -> _MetricCell | None:
    """Return the gate's mean for one published cell, or None when it is absent."""

    context: str = f"domain={domain}, variant={variant}, split={split}, metric={metric}"
    if domain == "__all_micro__":
        selected: list[dict[str, object]] = []
        for real_domain in real_domains:
            selected.extend(grouped.get((real_domain, variant, split), []))
        if not selected:
            return None
        return _cell_from_rows(selected, metric, context)
    if domain == "__all_macro__":
        domain_means: list[float] = []
        for real_domain in real_domains:
            selected = grouped.get((real_domain, variant, split), [])
            if not selected:
                return None
            domain_means.append(_cell_from_rows(selected, metric, context)["value"])
        if not domain_means:
            return None
        return {"value": sum(domain_means) / len(domain_means), "n": len(domain_means)}
    selected = grouped.get((domain, variant, split), [])
    if not selected:
        return None
    return _cell_from_rows(selected, metric, context)


def metric_mismatches(
    tag: str,
    recomputed_metrics: list[dict[str, object]],
    published_metrics: list[dict[str, object]],
    expected_encoder: str = ENCODER_NAME,
) -> list[MetricMismatch]:
    """Compare every published K=2 cell with the mean of the per-instance rows."""

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = _index_recomputed_metrics(
        recomputed_metrics, tag, expected_encoder=expected_encoder
    )
    domains: list[str] = _real_domains(grouped)
    mismatches: list[MetricMismatch] = []
    for index, row in enumerate(published_metrics):
        context: str = f"published_metrics[{index}]"
        k_samples: int = require_integer(row.get("k_samples"), f"{context}.k_samples")
        if k_samples != K_SAMPLES:
            continue
        published_tag: str = require_string(row.get("tag"), f"{context}.tag")
        if published_tag != tag:
            raise ValueError(
                f"published metric tag does not match the gate tag: "
                f"context={context}, tag={tag}, published_tag={published_tag}"
            )
        domain: str = require_string(row.get("domain"), f"{context}.domain")
        variant: str = require_string(row.get("variant"), f"{context}.variant")
        split: str = require_string(row.get("split"), f"{context}.split")
        if split not in ("all", "validation", "test"):
            raise ValueError(
                f"published metric split is not all, validation, or test: "
                f"context={context}, split={split}"
            )
        metric: str = require_string(row.get("metric"), f"{context}.metric")
        published_value: float = require_float(row.get("value"), f"{context}.value")
        published_n: int = require_integer(row.get("n"), f"{context}.n")
        cell: _MetricCell | None = _recomputed_cell(
            grouped, domains, domain, variant, split, metric
        )
        recomputed_value: float | None = None if cell is None else cell["value"]
        recomputed_n: int = 0 if cell is None else cell["n"]
        values_differ: bool = cell is None or not _rounded_equal(
            cell["value"], published_value
        )
        if values_differ or recomputed_n != published_n:
            mismatches.append(
                {
                    "domain": domain,
                    "variant": variant,
                    "split": split,
                    "metric": metric,
                    "published": published_value,
                    "recomputed": recomputed_value,
                    "n_published": published_n,
                    "n_recomputed": recomputed_n,
                }
            )
    mismatches.sort(key=lambda item: (item["domain"], item["variant"], item["split"], item["metric"]))
    return mismatches


def router_pick_mismatches(
    tag: str,
    recomputed_metrics: list[dict[str, object]],
    router_decisions: dict[str, object],
    expected_encoder: str = ENCODER_NAME,
) -> list[RouterMismatch]:
    """Derive each domain's pick from validation nDCG@10 and compare it."""

    if "model" in router_decisions and router_decisions["model"] != tag:
        raise ValueError(
            "router model does not match tag: "
            f"tag={tag}, model={router_decisions['model']!r}"
        )
    if "k_samples" in router_decisions and router_decisions["k_samples"] != K_SAMPLES:
        raise ValueError(
            "router k_samples is not 2: "
            f"tag={tag}, k_samples={router_decisions['k_samples']!r}"
        )
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = _index_recomputed_metrics(
        recomputed_metrics, tag, expected_encoder=expected_encoder
    )
    router_domains: dict[str, object] = require_object(
        router_decisions.get("domains"), "router.domains"
    )
    mismatches: list[RouterMismatch] = []
    recomputed_domains: list[str] = _real_domains(grouped)
    for domain in recomputed_domains:
        scores: dict[str, float] = {}
        for variant in FIXED_VARIANTS:
            rows: list[dict[str, object]] = grouped.get((domain, variant, "validation"), [])
            if not rows:
                raise ValueError(
                    "cannot derive a router pick without validation rows: "
                    f"tag={tag}, domain={domain}, variant={variant}"
                )
            scores[variant] = _mean(
                [require_float(row["nDCG@10"], f"{domain}.{variant}.nDCG@10") for row in rows],
                f"tag={tag}, domain={domain}, variant={variant}, split=validation, metric=nDCG@10",
            )
        recomputed_pick, _degenerate = expected_router_pick(scores)
        published_pick: str | None = None
        if domain not in router_domains:
            mismatches.append(
                {
                    "domain": domain,
                    "published_pick": None,
                    "recomputed_pick": recomputed_pick,
                }
            )
            continue
        payload: dict[str, object] = require_object(
            router_domains[domain], f"router.domains.{domain}"
        )
        published_pick = require_string(payload.get("pick"), f"router.domains.{domain}.pick")
        if published_pick != recomputed_pick:
            mismatches.append(
                {
                    "domain": domain,
                    "published_pick": published_pick,
                    "recomputed_pick": recomputed_pick,
                }
            )
    for domain, raw_payload in router_domains.items():
        if domain in recomputed_domains:
            continue
        payload = require_object(raw_payload, f"router.domains.{domain}")
        mismatches.append(
            {
                "domain": domain,
                "published_pick": require_string(
                    payload.get("pick"), f"router.domains.{domain}.pick"
                ),
                "recomputed_pick": None,
            }
        )
    mismatches.sort(key=lambda item: item["domain"])
    return mismatches


def _index_rankings(
    rows: list[dict[str, object]],
    *,
    arm: str | None,
    context: str,
) -> dict[str, dict[str, list[str]]]:
    by_domain: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for index, row in enumerate(rows):
        row_context: str = f"{context}[{index}]"
        row_arm: str = require_string(row.get("arm"), f"{row_context}.arm")
        if arm is not None and row_arm != arm:
            continue
        domain: str = require_string(row.get("domain"), f"{row_context}.domain")
        instance_id: str = require_string(row.get("instance_id"), f"{row_context}.instance_id")
        if instance_id in by_domain[domain]:
            raise ValueError(
                "duplicate ranking row: "
                f"context={row_context}, domain={domain}, instance_id={instance_id}"
            )
        by_domain[domain][instance_id] = _require_ranking_ids(
            row.get("ranked_ids"), f"{row_context}.ranked_ids"
        )
    if not by_domain:
        raise ValueError(f"ranking rows produced no comparable ids: context={context}, arm={arm}")
    return by_domain


def routed_top50_overlap(
    recomputed_rankings: list[dict[str, object]],
    committed_routed: list[dict[str, object]],
) -> dict[str, float]:
    """Mean fraction of top-50 ids shared with the committed routed ranking.

    Order does not matter. Two lists with the same ids contribute 1, including
    when a tie swapped their positions. The fraction is ``|intersection| /
    max(len(left), len(right))`` after truncating each list to 50.
    """

    recomputed: dict[str, dict[str, list[str]]] = _index_rankings(
        recomputed_rankings, arm="routed", context="recomputed_rankings"
    )
    committed: dict[str, dict[str, list[str]]] = _index_rankings(
        committed_routed, arm=None, context="committed_routed"
    )
    if set(recomputed) != set(committed):
        raise ValueError(
            "routed ranking domains do not match the committed K=2 file: "
            f"recomputed={sorted(recomputed)}, committed={sorted(committed)}"
        )
    overlap: dict[str, float] = {}
    for domain in DOMAINS:
        if domain not in recomputed:
            continue
        left_ids: dict[str, list[str]] = recomputed[domain]
        right_ids: dict[str, list[str]] = committed[domain]
        if set(left_ids) != set(right_ids):
            raise ValueError(
                "routed ranking instance ids do not match the committed K=2 file: "
                f"domain={domain}, recomputed={len(left_ids)}, committed={len(right_ids)}"
            )
        fractions: list[float] = []
        for instance_id, left in left_ids.items():
            right: list[str] = right_ids[instance_id]
            left_top: set[str] = set(left[:50])
            right_top: set[str] = set(right[:50])
            denominator: int = max(len(left[:50]), len(right[:50]))
            if denominator == 0:
                raise ValueError(
                    "routed ranking is empty: "
                    f"domain={domain}, instance_id={instance_id}"
                )
            fractions.append(len(left_top & right_top) / denominator)
        overlap[domain] = sum(fractions) / len(fractions)
    return overlap


def reproduction_report(
    *,
    tag: str,
    recomputed_metrics: list[dict[str, object]],
    published_metrics: list[dict[str, object]],
    router_decisions: dict[str, object],
    recomputed_rankings: list[dict[str, object]],
    committed_routed: list[dict[str, object]],
    bm25_report: dict[str, object],
    expected_encoder: str = ENCODER_NAME,
) -> dict[str, object]:
    """Build the reproduction record. ``passed`` is true only when every check holds."""

    if not tag:
        raise ValueError("tag must be a non-empty string")
    explanation: str = require_string(
        bm25_report.get("explanation"), "bm25_report.explanation"
    )
    mismatches: list[MetricMismatch] = metric_mismatches(
        tag, recomputed_metrics, published_metrics, expected_encoder=expected_encoder
    )
    router_mismatches: list[RouterMismatch] = router_pick_mismatches(
        tag, recomputed_metrics, router_decisions, expected_encoder=expected_encoder
    )
    overlap: dict[str, float] = routed_top50_overlap(recomputed_rankings, committed_routed)
    passed: bool = (
        not mismatches and not router_mismatches and explanation != "unexplained"
    )
    return {
        "passed": passed,
        "metric_mismatches": mismatches,
        "router_pick_mismatches": router_mismatches,
        "routed_top50_overlap": overlap,
        "bm25_report": bm25_report,
    }


def _bm25_domain_stats(rows: list[RankingRow]) -> _Bm25DomainStats:
    by_id: dict[str, RankingRow] = {}
    for row in rows:
        instance_id: str = row["instance_id"]
        if instance_id in by_id:
            raise ValueError(
                "duplicate BM25 row: "
                f"domain={row['domain']}, instance_id={instance_id}"
            )
        by_id[instance_id] = row
    validation: frozenset[str] = validation_ids(frozenset(by_id))
    test_rows: list[RankingRow] = [
        row for instance_id, row in by_id.items() if instance_id not in validation
    ]
    if not test_rows:
        raise ValueError(
            f"BM25 test split is empty: domain={rows[0]['domain']}, n={len(by_id)}"
        )
    hits: list[float] = []
    ndcgs: list[float] = []
    test_metrics: dict[str, tuple[float, float]] = {}
    for row in test_rows:
        metrics: dict[str, float] = instance_metrics(row)
        hit: float = metrics["hit@50"]
        ndcg: float = metrics["nDCG@10"]
        hits.append(hit)
        ndcgs.append(ndcg)
        test_metrics[row["instance_id"]] = (hit, ndcg)
    return {
        "n": len(by_id),
        "test_n": len(test_rows),
        "hit_at_50": sum(hits) / len(hits),
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "instance_ids": frozenset(by_id),
        "gold": {
            instance_id: tuple(row["gold_skill_ids"]) for instance_id, row in by_id.items()
        },
        "ranked": {
            instance_id: tuple(row["ranked_ids"]) for instance_id, row in by_id.items()
        },
        "test_metrics": test_metrics,
    }


def _macro_hit(stats_by_domain: dict[str, _Bm25DomainStats]) -> float | None:
    values: list[float] = [
        stats_by_domain[domain]["hit_at_50"]
        for domain in DOMAINS
        if domain in stats_by_domain
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _macro_ndcg(stats_by_domain: dict[str, _Bm25DomainStats]) -> float | None:
    values: list[float] = [
        stats_by_domain[domain]["ndcg_at_10"]
        for domain in DOMAINS
        if domain in stats_by_domain
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _metrics_match(
    left: dict[str, tuple[float, float]],
    right: dict[str, tuple[float, float]],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        abs(left[instance_id][0] - right[instance_id][0]) <= 1e-12
        and abs(left[instance_id][1] - right[instance_id][1]) <= 1e-12
        for instance_id in left
    )


def _leading_bm25_hit(community_root: Path, tag: str, instance_id: str) -> tuple[str, float]:
    """Return the stored top skill and score for one BM25 row."""

    path: Path = community_root / tag / "k4" / "retrieval_top50.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(f"k4 retrieval file does not exist: path={path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            row: dict[str, object] = require_object(raw, f"{path}:{line_number}")
            if row.get("variant") != "bm25" or row.get("instance_id") != instance_id:
                continue
            top50: list[object] = require_list(row.get("top50"), f"{path}:{line_number}.top50")
            if not top50:
                raise ValueError(f"BM25 top50 is empty: path={path}, instance_id={instance_id}")
            triple: list[object] = require_list(top50[0], f"{path}:{line_number}.top50[0]")
            if len(triple) != 3:
                raise ValueError(
                    "BM25 top50 entry is not a skill/score/flag triple: "
                    f"path={path}, instance_id={instance_id}"
                )
            return (
                require_string(triple[0], f"{path}:{line_number}.top50[0][0]"),
                require_float(triple[1], f"{path}:{line_number}.top50[0][1]"),
            )
    raise ValueError(
        f"BM25 row was not found: path={path}, instance_id={instance_id}"
    )


def _score_matches(actual: float, expected: float) -> bool:
    return round(actual, 5) == round(expected, 5)


def build_bm25_report(community_root: Path) -> dict[str, object]:
    """Explain the Qwen3.5-4B BM25 gap from the committed k4 rankings.

    The explanation stays ``unexplained`` unless the seven files show the
    pattern checked on 2026-11-24: same ids and gold, two different scorers,
    and the known test-split macro values.
    """

    if not community_root.is_dir():
        raise NotADirectoryError(
            f"community results directory does not exist: path={community_root}"
        )
    per_tag: dict[str, dict[str, _Bm25DomainStats]] = {}
    for tag in REVISION_TAGS:
        by_domain: dict[str, list[RankingRow]] = defaultdict(list)
        for row in load_committed_rankings(tag, "bm25", root=community_root):
            by_domain[row["domain"]].append(row)
        per_tag[tag] = {
            domain: _bm25_domain_stats(rows) for domain, rows in by_domain.items()
        }

    reference: dict[str, _Bm25DomainStats] = per_tag[REFERENCE_BM25_TAG]
    qwen: dict[str, _Bm25DomainStats] = per_tag[QWEN_BM25_TAG]
    ids_equal: bool = True
    gold_equal: bool = True
    for tag in FULL_BM25_TAGS:
        stats: dict[str, _Bm25DomainStats] = per_tag[tag]
        if set(stats) != set(DOMAINS) or set(reference) != set(DOMAINS):
            ids_equal = False
            gold_equal = False
            break
        for domain in DOMAINS:
            if stats[domain]["instance_ids"] != reference[domain]["instance_ids"]:
                ids_equal = False
            if stats[domain]["gold"] != reference[domain]["gold"]:
                gold_equal = False

    qwen_ranking_differs: bool = any(
        qwen[domain]["ranked"] != reference[domain]["ranked"] for domain in DOMAINS
    )
    qwen_macro_hit: float | None = _macro_hit(qwen)
    qwen_macro_ndcg: float | None = _macro_ndcg(qwen)
    fast_metrics_match: bool = True
    fast_macros_match: bool = True
    for tag in FAST_BM25_TAGS:
        stats = per_tag[tag]
        macro_hit: float | None = _macro_hit(stats)
        macro_ndcg: float | None = _macro_ndcg(stats)
        if (
            macro_hit is None
            or macro_ndcg is None
            or round(macro_hit, 3) != 0.705
            or round(macro_ndcg, 3) != 0.357
            or set(stats) != set(DOMAINS)
        ):
            fast_macros_match = False
        for domain in DOMAINS:
            if domain not in stats or not _metrics_match(
                stats[domain]["test_metrics"], reference[domain]["test_metrics"]
            ):
                fast_metrics_match = False

    partial_matches_reference: bool = True
    for tag in PARTIAL_BM25_TAGS:
        stats = per_tag[tag]
        if set(stats) != {"theoremqa"}:
            partial_matches_reference = False
            continue
        partial_domain: _Bm25DomainStats = stats["theoremqa"]
        reference_domain: _Bm25DomainStats = reference["theoremqa"]
        if (
            partial_domain["instance_ids"] != reference_domain["instance_ids"]
            or partial_domain["gold"] != reference_domain["gold"]
            or partial_domain["ranked"] != reference_domain["ranked"]
        ):
            partial_matches_reference = False

    qwen_top_skill, qwen_top_score = _leading_bm25_hit(
        community_root, QWEN_BM25_TAG, "theoremqa_00000"
    )
    fast_top_skill, fast_top_score = _leading_bm25_hit(
        community_root, REFERENCE_BM25_TAG, "theoremqa_00000"
    )
    scores_match: bool = (
        qwen_top_skill == QWEN_BM25_TOP_SKILL
        and fast_top_skill == FAST_BM25_TOP_SKILL
        and _score_matches(qwen_top_score, QWEN_BM25_TOP_SCORE)
        and _score_matches(fast_top_score, FAST_BM25_TOP_SCORE)
    )
    qwen_macro_matches: bool = (
        qwen_macro_hit is not None
        and qwen_macro_ndcg is not None
        and round(qwen_macro_hit, 3) == 0.801
        and round(qwen_macro_ndcg, 3) == 0.434
        and set(qwen) == set(DOMAINS)
    )
    explained: bool = (
        ids_equal
        and gold_equal
        and qwen_ranking_differs
        and qwen_macro_matches
        and fast_macros_match
        and fast_metrics_match
        and partial_matches_reference
        and scores_match
    )
    if explained:
        explanation = (
            "The five models that store BM25 for every domain use the same "
            "3,970 instance ids and the same gold lists. The test-split gap is "
            "the scorer stored in the file, not a different question set. "
            "qwen3.5-4b-reference macro-averages to hit@50 0.801 and nDCG@10 "
            "0.434 across the five domains. Those lists are the SR-Agents "
            "BM25Retriever ranking: punctuation is stripped before tokenization "
            "and Lucene IDF is clipped at zero. Rescoring one query per domain "
            "reproduced that file's top-50 ids exactly, and theoremqa_00000's "
            f"leading score is {qwen_top_score:.5f}. glm4-9b, llama31-8b, "
            "mistral7b, and qwen35-9b macro-average to hit@50 0.705 and "
            "nDCG@10 0.357. Those lists match FastBM25 (whitespace tokens, "
            "Robertson/Okapi); the same query's leading score is "
            f"{fast_top_score:.5f}. Mistral reorders some tied rows, but its "
            "per-instance test hit@50 and nDCG@10 match the other FastBM25 "
            "models, so the macro matches. deepseek7b and yi15-9b store BM25 "
            "for theoremqa only, and those rows match the FastBM25 theoremqa ranking."
        )
    else:
        explanation = "unexplained"

    models: dict[str, object] = {}
    for tag in REVISION_TAGS:
        stats = per_tag[tag]
        shared_domains: list[str] = [domain for domain in DOMAINS if domain in stats and domain in reference]
        ranking_match: bool = all(
            stats[domain]["ranked"] == reference[domain]["ranked"] for domain in shared_domains
        )
        metric_match: bool = all(
            _metrics_match(stats[domain]["test_metrics"], reference[domain]["test_metrics"])
            for domain in shared_domains
        )
        models[tag] = {
            "domains": {
                domain: {
                    "n": stats[domain]["n"],
                    "test_n": stats[domain]["test_n"],
                    "hit@50": stats[domain]["hit_at_50"],
                    "nDCG@10": stats[domain]["ndcg_at_10"],
                }
                for domain in DOMAINS
                if domain in stats
            },
            "missing_domains": [domain for domain in DOMAINS if domain not in stats],
            "test_macro_hit@50": _macro_hit(stats),
            "test_macro_nDCG@10": _macro_ndcg(stats),
            "ranking_matches_glm4_9b_on_shared_domains": ranking_match,
            "test_metrics_match_glm4_9b_on_shared_domains": metric_match,
        }
    return {
        "explanation": explanation,
        "instance_ids_equal_among_full_models": ids_equal,
        "gold_equal_among_full_models": gold_equal,
        "qwen_ranking_differs_from_glm4_9b": qwen_ranking_differs,
        "leading_scores": {
            QWEN_BM25_TAG: {"instance_id": "theoremqa_00000", "skill_id": qwen_top_skill, "score": qwen_top_score},
            REFERENCE_BM25_TAG: {
                "instance_id": "theoremqa_00000",
                "skill_id": fast_top_skill,
                "score": fast_top_score,
            },
        },
        "models": models,
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    """Atomically write the reproduction JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    """Write one reproduction file. Exit 0 only when ``passed`` is true."""

    args = parse_args()
    tag: str = require_string(args.tag, "tag")
    if tag not in REVISION_TAGS:
        raise ValueError(f"tag is not one of the seven revision models: tag={tag}")
    report: dict[str, object] = reproduction_report(
        tag=tag,
        recomputed_metrics=read_jsonl(args.recomputed_metrics),
        published_metrics=read_jsonl(args.published_metrics),
        router_decisions=read_json_object(args.router),
        recomputed_rankings=read_jsonl(args.recomputed_rankings),
        committed_routed=load_committed_rankings(tag, "routed_k2", root=args.community_root),
        bm25_report=build_bm25_report(args.community_root),
        expected_encoder=require_string(args.encoder, "encoder"),
    )
    write_report(args.output, report)
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
