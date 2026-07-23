#!/usr/bin/env python3
"""Aggregate seven complete model-level K-ablation summary packs."""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np

from hyskill.k_ablation import (
    ALL_VARIANTS,
    DOMAINS,
    K_VALUES,
    require_float,
    require_integer,
    require_object,
    require_string,
    sha256_file,
)
from scripts.summarize_k_ablation import (
    COMPARISON_K_VALUES,
    METRIC_NAMES,
    SPLITS,
    MetricRow,
    bootstrap_stats,
    comparison_seed,
    metric_rows_summary,
    write_json_atomic,
    write_jsonl_gzip_atomic,
)


MODEL_BY_TAG: dict[str, str] = {
    "deepseek7b": "deepseek7b",
    "glm4-9b": "glm4-9b",
    "llama31-8b": "llama31-8b",
    "mistral7b": "mistral7b",
    "qwen3.5-4b-reference": "qwen3.5-4b",
    "qwen35-9b": "qwen35-9b",
    "yi15-9b": "yi15-9b",
}
TAGS: tuple[str, ...] = tuple(MODEL_BY_TAG)


class PackedMetric(TypedDict):
    """Validated metric row fields needed for fleet aggregation."""

    unit: str
    n: int
    value: float


def parse_args() -> argparse.Namespace:
    """Parse explicit community-pack paths and bootstrap settings."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--community-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", required=True, type=int)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    return parser.parse_args()


def metric_key(row: dict[str, object], path: Path, line_number: int) -> tuple[str, int, str, str, str]:
    """Validate and return the identity key of one model metric row."""

    domain: str = require_string(row.get("domain"), f"{path}:{line_number}.domain")
    allowed_domains: set[str] = {*DOMAINS, "__all_micro__", "__all_macro__"}
    if domain not in allowed_domains:
        raise ValueError(
            f"Unsupported metric-row domain: path={path}, line={line_number}, domain={domain}"
        )
    k_samples: int = require_integer(
        row.get("k_samples"), f"{path}:{line_number}.k_samples"
    )
    if k_samples not in K_VALUES:
        raise ValueError(
            f"Unsupported metric-row K: path={path}, line={line_number}, k={k_samples}"
        )
    variant: str = require_string(row.get("variant"), f"{path}:{line_number}.variant")
    if variant not in ALL_VARIANTS:
        raise ValueError(
            f"Unsupported metric-row variant: path={path}, line={line_number}, "
            f"variant={variant}"
        )
    split: str = require_string(row.get("split"), f"{path}:{line_number}.split")
    if split not in SPLITS:
        raise ValueError(
            f"Unsupported metric-row split: path={path}, line={line_number}, split={split}"
        )
    metric: str = require_string(row.get("metric"), f"{path}:{line_number}.metric")
    if metric not in METRIC_NAMES:
        raise ValueError(
            f"Unsupported metric name: path={path}, line={line_number}, metric={metric}"
        )
    return domain, k_samples, variant, split, metric


def load_model_metrics(
    community_root: Path, tag: str, model: str
) -> dict[tuple[str, int, str, str, str], PackedMetric]:
    """Load exactly 5,040 unique normalized rows from one verified model pack."""

    path: Path = community_root / tag / "k-ablation" / "metrics_long.jsonl.gz"
    metrics: dict[tuple[str, int, str, str, str], PackedMetric] = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            raw_row: object = json.loads(line)
            row: dict[str, object] = require_object(raw_row, f"{path}:{line_number}")
            if row.get("schema_version") != 1:
                raise ValueError(
                    f"Metric row schema mismatch: path={path}, line={line_number}, "
                    f"actual={row.get('schema_version')!r}"
                )
            if row.get("tag") != tag or row.get("model") != model:
                raise ValueError(
                    f"Metric row identity mismatch: path={path}, line={line_number}, "
                    f"expected_tag={tag}, actual_tag={row.get('tag')!r}, "
                    f"expected_model={model}, actual_model={row.get('model')!r}"
                )
            key: tuple[str, int, str, str, str] = metric_key(row, path, line_number)
            if key in metrics:
                raise ValueError(
                    f"Duplicate normalized metric key: path={path}, line={line_number}, key={key}"
                )
            unit: str = require_string(row.get("unit"), f"{path}:{line_number}.unit")
            n: int = require_integer(row.get("n"), f"{path}:{line_number}.n")
            if n <= 0:
                raise ValueError(
                    f"Metric row n must be positive: path={path}, line={line_number}, n={n}"
                )
            metrics[key] = {
                "unit": unit,
                "n": n,
                "value": require_float(row.get("value"), f"{path}:{line_number}.value"),
            }
    if len(metrics) != 5040:
        raise ValueError(
            f"Model metric row count mismatch: path={path}, expected=5040, "
            f"actual={len(metrics)}"
        )
    return metrics


def load_all_metrics(
    community_root: Path,
) -> dict[str, dict[tuple[str, int, str, str, str], PackedMetric]]:
    """Load seven strict model packs and require identical row support."""

    by_tag: dict[str, dict[tuple[str, int, str, str, str], PackedMetric]] = {
        tag: load_model_metrics(community_root, tag, MODEL_BY_TAG[tag]) for tag in TAGS
    }
    reference_keys: set[tuple[str, int, str, str, str]] = set(by_tag[TAGS[0]])
    for tag in TAGS[1:]:
        actual_keys: set[tuple[str, int, str, str, str]] = set(by_tag[tag])
        if actual_keys != reference_keys:
            raise ValueError(
                f"Model metric support differs: tag={tag}, "
                f"missing_sample={sorted(reference_keys - actual_keys)[:10]}, "
                f"extra_sample={sorted(actual_keys - reference_keys)[:10]}"
            )
    return by_tag


def fleet_metric_rows(
    by_tag: dict[str, dict[tuple[str, int, str, str, str], PackedMetric]]
) -> list[MetricRow]:
    """Aggregate every strict-common-support row across all seven models."""

    reference_keys: list[tuple[str, int, str, str, str]] = sorted(by_tag[TAGS[0]])
    rows: list[MetricRow] = []
    for domain, k_samples, variant, split, metric in reference_keys:
        model_rows: list[PackedMetric] = [
            by_tag[tag][(domain, k_samples, variant, split, metric)] for tag in TAGS
        ]
        units: set[str] = {row["unit"] for row in model_rows}
        if len(units) != 1:
            raise ValueError(
                f"Metric units differ across models: key={(domain, k_samples, variant, split, metric)}, "
                f"units={sorted(units)}"
            )
        if domain == "__all_macro__":
            value: float = sum(row["value"] for row in model_rows) / len(model_rows)
            n: int = sum(row["n"] for row in model_rows)
            unit: str = "model_domains"
        else:
            n = sum(row["n"] for row in model_rows)
            value = sum(row["value"] * row["n"] for row in model_rows) / n
            unit = "model_instances"
        rows.append(
            {
                "schema_version": 1,
                "tag": "__fleet__",
                "model": "__fleet__",
                "domain": domain,
                "k_samples": k_samples,
                "variant": variant,
                "split": split,
                "unit": unit,
                "n": n,
                "metric": metric,
                "value": value,
            }
        )
    if len(rows) != 5040:
        raise ValueError(
            f"Fleet metric row count mismatch: expected=5040, actual={len(rows)}"
        )
    return rows


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON object with a path-aware root check."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return require_object(raw, str(path))


def nested_object(value: dict[str, object], keys: tuple[str, ...], context: str) -> dict[str, object]:
    """Traverse a required sequence of JSON object keys."""

    current: dict[str, object] = value
    for key in keys:
        current = require_object(current.get(key), f"{context}.{key}")
    return current


def model_comparison_stats(
    paired_by_tag: dict[str, dict[str, object]],
    variant: str,
    comparison: str,
    scope: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Bootstrap paired model-level means for one comparison scope."""

    values_k: list[float] = []
    values_k4: list[float] = []
    for tag in TAGS:
        block: dict[str, object] = nested_object(
            paired_by_tag[tag],
            ("comparisons", variant, comparison, scope),
            f"paired:{tag}",
        )
        values_k.append(require_float(block.get("mean_k"), f"paired:{tag}.mean_k"))
        values_k4.append(require_float(block.get("mean_k4"), f"paired:{tag}.mean_k4"))
    return bootstrap_stats(
        np.asarray(values_k, dtype=np.float64),
        np.asarray(values_k4, dtype=np.float64),
        bootstrap_samples,
        comparison_seed(
            bootstrap_seed, f"fleet:{variant}:{comparison}:{scope}:models"
        ),
    )


def routed_domain_stats(
    paired_by_tag: dict[str, dict[str, object]],
    comparison: str,
    domain: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Bootstrap seven paired model means within one routed domain."""

    values_k: list[float] = []
    values_k4: list[float] = []
    for tag in TAGS:
        block: dict[str, object] = nested_object(
            paired_by_tag[tag],
            ("comparisons", "routed", comparison, "by_domain", domain),
            f"paired:{tag}",
        )
        values_k.append(require_float(block.get("mean_k"), f"paired:{tag}.mean_k"))
        values_k4.append(require_float(block.get("mean_k4"), f"paired:{tag}.mean_k4"))
    return bootstrap_stats(
        np.asarray(values_k, dtype=np.float64),
        np.asarray(values_k4, dtype=np.float64),
        bootstrap_samples,
        comparison_seed(
            bootstrap_seed, f"fleet:routed:{comparison}:domain:{domain}:models"
        ),
    )


def hierarchical_bootstrap(
    values_k: np.ndarray,
    values_k4: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Resample models, then domains within sampled models, for routed effects."""

    if values_k.shape != (len(TAGS), len(DOMAINS)) or values_k4.shape != values_k.shape:
        raise ValueError(
            f"Hierarchical arrays have unexpected shapes: k={values_k.shape}, "
            f"k4={values_k4.shape}"
        )
    differences: np.ndarray = values_k - values_k4
    rng: np.random.Generator = np.random.default_rng(seed)
    distribution: np.ndarray = np.empty(bootstrap_samples, dtype=np.float64)
    chunk_size: int = min(1024, bootstrap_samples)
    offset: int = 0
    while offset < bootstrap_samples:
        current: int = min(chunk_size, bootstrap_samples - offset)
        model_indices: np.ndarray = rng.integers(
            0, len(TAGS), size=(current, len(TAGS)), endpoint=False
        )
        domain_indices: np.ndarray = rng.integers(
            0,
            len(DOMAINS),
            size=(current, len(TAGS), len(DOMAINS)),
            endpoint=False,
        )
        sampled_models: np.ndarray = np.broadcast_to(
            model_indices[:, :, None], domain_indices.shape
        )
        distribution[offset : offset + current] = differences[
            sampled_models, domain_indices
        ].mean(axis=(1, 2))
        offset += current
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    p_value: float = min(
        1.0,
        2.0
        * min(
            float(np.mean(distribution <= 0.0)),
            float(np.mean(distribution >= 0.0)),
        ),
    )
    return {
        "n_models": len(TAGS),
        "n_domains_per_model": len(DOMAINS),
        "mean_k": float(values_k.mean()),
        "mean_k4": float(values_k4.mean()),
        "difference_k_minus_k4": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": p_value,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "resampling_unit": "models, then domains within sampled models",
    }


def routed_hierarchical_stats(
    paired_by_tag: dict[str, dict[str, object]],
    comparison: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Build the routed 7-model by 5-domain hierarchical comparison."""

    values_k: np.ndarray = np.empty((len(TAGS), len(DOMAINS)), dtype=np.float64)
    values_k4: np.ndarray = np.empty_like(values_k)
    for model_index, tag in enumerate(TAGS):
        for domain_index, domain in enumerate(DOMAINS):
            block: dict[str, object] = nested_object(
                paired_by_tag[tag],
                ("comparisons", "routed", comparison, "by_domain", domain),
                f"paired:{tag}",
            )
            values_k[model_index, domain_index] = require_float(
                block.get("mean_k"), f"paired:{tag}.{domain}.mean_k"
            )
            values_k4[model_index, domain_index] = require_float(
                block.get("mean_k4"), f"paired:{tag}.{domain}.mean_k4"
            )
    return hierarchical_bootstrap(
        values_k,
        values_k4,
        bootstrap_samples,
        comparison_seed(
            bootstrap_seed, f"fleet:routed:{comparison}:hierarchical"
        ),
    )


def fleet_paired_report(
    community_root: Path, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, object]:
    """Aggregate model-level paired effects without hiding model heterogeneity."""

    paired_by_tag: dict[str, dict[str, object]] = {
        tag: load_json(
            community_root / tag / "k-ablation" / "paired_vs_k4.json"
        )
        for tag in TAGS
    }
    comparisons: dict[str, object] = {}
    for variant in ALL_VARIANTS:
        variant_block: dict[str, object] = {}
        for k_samples in COMPARISON_K_VALUES:
            comparison: str = f"k{k_samples}_vs_k4"
            result: dict[str, object] = {
                "across_models_micro": model_comparison_stats(
                    paired_by_tag,
                    variant,
                    comparison,
                    "cross_domain_micro",
                    bootstrap_samples,
                    bootstrap_seed,
                ),
                "across_models_macro": model_comparison_stats(
                    paired_by_tag,
                    variant,
                    comparison,
                    "cross_domain_macro_stratified",
                    bootstrap_samples,
                    bootstrap_seed,
                ),
            }
            if variant == "routed":
                result["by_domain_across_models"] = {
                    domain: routed_domain_stats(
                        paired_by_tag,
                        comparison,
                        domain,
                        bootstrap_samples,
                        bootstrap_seed,
                    )
                    for domain in DOMAINS
                }
                result["model_domain_hierarchical"] = routed_hierarchical_stats(
                    paired_by_tag,
                    comparison,
                    bootstrap_samples,
                    bootstrap_seed,
                )
            variant_block[comparison] = result
        comparisons[variant] = variant_block
    return {
        "schema_version": 1,
        "metric": "nDCG@10",
        "split": "test",
        "contrast": "K minus K=4",
        "models": list(TAGS),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "inference_note": (
            "Fleet intervals resample model-level paired means; routed hierarchical "
            "intervals resample models and domains. Per-model files retain the "
            "within-instance paired bootstrap intervals."
        ),
        "comparisons": comparisons,
    }


def cost_value(
    cost: dict[str, object], k_samples: int, template_name: str, field: str
) -> float:
    """Extract one per-query cost estimate from a model cost report."""

    block: dict[str, object] = nested_object(
        cost,
        (
            "by_k",
            f"k{k_samples}",
            "templates",
            template_name,
            "per_evaluated_instance",
        ),
        "cost",
    )
    return require_float(block.get(field), f"cost.k{k_samples}.{template_name}.{field}")


def aggregate_values(values: list[float]) -> dict[str, float]:
    """Return mean and observed range for seven model-specific costs."""

    if len(values) != len(TAGS):
        raise ValueError(
            f"Cost aggregation requires seven values: expected={len(TAGS)}, actual={len(values)}"
        )
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def fleet_cost_report(community_root: Path) -> dict[str, object]:
    """Aggregate actual-text token estimates and timing availability."""

    costs: dict[str, dict[str, object]] = {
        tag: load_json(community_root / tag / "k-ablation" / "cost.json")
        for tag in TAGS
    }
    by_k: dict[str, object] = {}
    for k_samples in K_VALUES:
        templates: dict[str, object] = {}
        for template_name in ("sentence", "passage", "skill"):
            templates[template_name] = {
                field: aggregate_values(
                    [cost_value(costs[tag], k_samples, template_name, field) for tag in TAGS]
                )
                for field in (
                    "input_tokens_per_query_estimate",
                    "output_tokens_per_query_estimate",
                    "total_tokens_per_query_estimate",
                )
            }
        routed_values: list[float] = []
        for tag in TAGS:
            routed: dict[str, object] = nested_object(
                costs[tag],
                ("by_k", f"k{k_samples}", "variants", "routed"),
                f"cost:{tag}",
            )
            routed_values.append(
                require_float(
                    routed.get("total_tokens_per_query_estimate"),
                    f"cost:{tag}.k{k_samples}.routed.total_tokens_per_query_estimate",
                )
            )
        by_k[f"k{k_samples}"] = {
            "templates": templates,
            "routed_total_tokens_per_query_estimate": aggregate_values(routed_values),
        }
    timing_status: dict[str, str] = {}
    for tag in TAGS:
        timing: dict[str, object] = require_object(
            costs[tag].get("warmup_wall_clock"), f"cost:{tag}.warmup_wall_clock"
        )
        timing_status[tag] = require_string(
            timing.get("status"), f"cost:{tag}.warmup_wall_clock.status"
        )
    return {
        "schema_version": 1,
        "models": list(TAGS),
        "estimator": {
            "characters_per_token": 3.8,
            "source": "actual prompts and cached generation outputs in each model pack",
        },
        "by_k": by_k,
        "warmup_wall_clock_status": timing_status,
    }


def verify_model_manifests(community_root: Path) -> dict[str, object]:
    """Verify model-pack output hashes and return strict fleet provenance."""

    provenance: dict[str, object] = {}
    source_revisions: set[str] = set()
    runner_bundle_hashes: set[str] = set()
    for tag in TAGS:
        pack_dir: Path = community_root / tag / "k-ablation"
        manifest_path: Path = pack_dir / "manifest.json"
        manifest: dict[str, object] = load_json(manifest_path)
        if manifest.get("tag") != tag or manifest.get("model") != MODEL_BY_TAG[tag]:
            raise ValueError(
                f"Model manifest identity mismatch: path={manifest_path}, tag={tag}, "
                f"model={MODEL_BY_TAG[tag]}"
            )
        if manifest.get("result_files") != 150:
            raise ValueError(
                f"Model manifest result count mismatch: path={manifest_path}, "
                f"actual={manifest.get('result_files')!r}"
            )
        source_revision: str = require_string(
            manifest.get("source_revision"), f"{manifest_path}.source_revision"
        )
        source_revisions.add(source_revision)
        runner_bundle: dict[str, object] = require_object(
            manifest.get("runner_bundle"), f"{manifest_path}.runner_bundle"
        )
        runner_bundle_hashes.add(
            require_string(runner_bundle.get("sha256"), f"{manifest_path}.runner_bundle.sha256")
        )
        outputs: dict[str, object] = require_object(
            manifest.get("outputs"), f"{manifest_path}.outputs"
        )
        for filename in (
            "metrics_long.jsonl.gz",
            "summary.json",
            "paired_vs_k4.json",
            "cost.json",
        ):
            expected_digest: str = require_string(
                outputs.get(filename), f"{manifest_path}.outputs.{filename}"
            )
            actual_digest: str = sha256_file(pack_dir / filename)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"Model pack output digest mismatch: path={pack_dir / filename}, "
                    f"expected={expected_digest}, actual={actual_digest}"
                )
        provenance[tag] = {
            "model": MODEL_BY_TAG[tag],
            "model_revision": require_string(
                manifest.get("model_revision"), f"{manifest_path}.model_revision"
            ),
            "encoder": require_string(
                manifest.get("encoder"), f"{manifest_path}.encoder"
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    if len(source_revisions) != 1 or len(runner_bundle_hashes) != 1:
        raise ValueError(
            f"Model packs used different runner identities: "
            f"source_revisions={sorted(source_revisions)}, "
            f"bundle_hashes={sorted(runner_bundle_hashes)}"
        )
    return {
        "models": provenance,
        "source_revision": next(iter(source_revisions)),
        "runner_bundle_sha256": next(iter(runner_bundle_hashes)),
    }


def main() -> None:
    """Verify seven packs and write strict-common-support fleet artifacts."""

    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError(
            f"bootstrap-samples must be positive: value={args.bootstrap_samples}"
        )
    provenance: dict[str, object] = verify_model_manifests(args.community_root)
    model_metrics = load_all_metrics(args.community_root)
    rows: list[MetricRow] = fleet_metric_rows(model_metrics)
    summary: dict[str, object] = {
        "schema_version": 1,
        "models": list(TAGS),
        "model_count": len(TAGS),
        "strict_common_support": True,
        "protocol": {
            "k_values": list(K_VALUES),
            "domains": list(DOMAINS),
            "variants": list(ALL_VARIANTS),
            "splits": list(SPLITS),
        },
        "metric_rows": len(rows),
        "metrics": metric_rows_summary(rows),
    }
    paired: dict[str, object] = fleet_paired_report(
        args.community_root, args.bootstrap_samples, args.bootstrap_seed
    )
    cost: dict[str, object] = fleet_cost_report(args.community_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path: Path = args.output_dir / "metrics_long.jsonl.gz"
    summary_path: Path = args.output_dir / "summary.json"
    paired_path: Path = args.output_dir / "paired_vs_k4.json"
    cost_path: Path = args.output_dir / "cost.json"
    manifest_path: Path = args.output_dir / "manifest.json"
    write_jsonl_gzip_atomic(metrics_path, rows)
    write_json_atomic(summary_path, summary)
    write_json_atomic(paired_path, paired)
    write_json_atomic(cost_path, cost)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "seven_model_k_ablation_fleet",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(TAGS),
        "models": list(TAGS),
        "strict_common_support": True,
        "provenance": provenance,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "metrics_long.jsonl.gz": sha256_file(metrics_path),
            "summary.json": sha256_file(summary_path),
            "paired_vs_k4.json": sha256_file(paired_path),
            "cost.json": sha256_file(cost_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "model_count": len(TAGS),
                "metric_rows": len(rows),
                "output_dir": str(args.output_dir),
                "manifest_sha256": sha256_file(manifest_path),
                "verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
