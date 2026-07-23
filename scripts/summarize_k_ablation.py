#!/usr/bin/env python3
"""Validate and summarize one complete model-level K-ablation matrix."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np

from hyskill.k_ablation import (
    ALL_VARIANTS,
    DOMAINS,
    EXPECTED_CACHE_FILES_PER_SAMPLE,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_UNIQUE_QUERIES,
    FIXED_VARIANTS,
    K_VALUES,
    METRIC_DEPTHS,
    TEMPLATE_TEXTS,
    KResultSpec,
    RetrievalPayload,
    compute_metrics,
    load_expected_gold,
    load_json_object,
    metric_for_record,
    require_float,
    require_integer,
    require_list,
    require_object,
    require_sha256,
    require_string,
    sha256_file,
    validate_cache_artifact,
    validate_final_result,
    validation_ids,
)


SPLITS: tuple[str, ...] = ("all", "validation", "test")
METRIC_NAMES: tuple[str, ...] = tuple(
    f"{family}@{depth}"
    for depth in METRIC_DEPTHS
    for family in ("Recall", "nDCG")
)
COMPARISON_K_VALUES: tuple[int, ...] = (1, 2, 8, 10)
CHARS_PER_TOKEN: float = 3.8
RESULT_BUNDLE_PATHS: tuple[str, ...] = (
    "hyskill/k_ablation.py",
    "scripts/finalize_k_retrieval.py",
    "scripts/validate_k_retrieval.py",
    "scripts/validate_k_cache.py",
    "scripts/route_k_retrieval.py",
    "scripts/run_k_ablation.sh",
)
VARIANT_TEMPLATES: dict[str, str] = {
    "naive_sentence": "sentence",
    "naive_passage": "passage",
    "naive_skill": "skill",
    "hyskill": "skill",
    "two_stage": "skill",
}


class MetricRow(TypedDict):
    """One normalized metric observation."""

    schema_version: int
    tag: str
    model: str
    domain: str
    k_samples: int
    variant: str
    split: str
    unit: str
    n: int
    metric: str
    value: float


class ResultData(TypedDict):
    """Memory-bounded summary of one validated retrieval result."""

    path: str
    sha256: str
    records_sha256: str
    metrics_sha256: str
    split_n: dict[str, int]
    split_metrics: dict[str, dict[str, float]]
    ndcg10_by_id: dict[str, float]
    router: dict[str, object] | None


class TextTotals(TypedDict):
    """Character totals for one imagination template."""

    prompt_chars: int
    output_chars: int
    generations: int


class CacheArtifactStats(TypedDict):
    """Streaming statistics and prefix identities for one cache artifact."""

    artifact_path: str
    rows: int
    unique_queries: int
    query_rows_sha256: str
    sample_sha256: dict[str, list[str]]
    by_template: dict[str, TextTotals]
    by_unique_query_template: dict[str, TextTotals]
    by_domain_template: dict[str, dict[str, TextTotals]]


class Digest(Protocol):
    """Minimal hash object interface used by streaming validators."""

    def update(self, value: bytes) -> None:
        """Consume bytes into the digest state."""

    def hexdigest(self) -> str:
        """Return the current lowercase hexadecimal digest."""


def parse_args() -> argparse.Namespace:
    """Parse explicit model identity, paths, and bootstrap settings."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--instances-dir", required=True, type=Path)
    parser.add_argument("--community-root", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", required=True, type=int)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    return parser.parse_args()


def sha256_json(value: object) -> str:
    """Return a deterministic digest for a JSON-compatible value."""

    encoded: str = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_json_with_sha256(path: Path) -> tuple[dict[str, object], str]:
    """Load a JSON object while hashing the exact file bytes once."""

    raw: bytes = path.read_bytes()
    parsed: object = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"JSON root must be an object: path={path}")
    return cast(dict[str, object], parsed), hashlib.sha256(raw).hexdigest()


def repository_commit(repository_root: Path) -> str:
    """Return the checked-out Git commit or raise with command context."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Failed to resolve repository commit: root={repository_root}, "
            f"stderr={error.stderr.strip()}"
        ) from error
    commit: str = result.stdout.strip()
    if not commit:
        raise RuntimeError(f"Git returned an empty commit: root={repository_root}")
    return commit


def result_bundle_manifest(repository_root: Path) -> dict[str, object]:
    """Return exact runner file digests and their combined bundle identity."""

    file_hashes: dict[str, str] = {}
    digest_lines: list[str] = []
    for relative_path in RESULT_BUNDLE_PATHS:
        digest: str = sha256_file(repository_root / relative_path)
        file_hashes[relative_path] = digest
        digest_lines.append(f"{digest}  {relative_path}\n")
    bundle_sha256: str = hashlib.sha256("".join(digest_lines).encode()).hexdigest()
    return {"sha256": bundle_sha256, "files": file_hashes}


def fixed_result_path(results_dir: Path, k_samples: int, domain: str, variant: str) -> Path:
    """Return the canonical fixed-variant result path."""

    return results_dir / f"k{k_samples}" / f"{domain}-{variant}.json"


def routed_result_path(results_dir: Path, k_samples: int, domain: str) -> Path:
    """Return the canonical routed result path."""

    return results_dir / "routed" / f"k{k_samples}" / f"{domain}-routed.json"


def expected_result_paths(results_dir: Path) -> set[Path]:
    """Return the exact 150-file model-level result matrix."""

    expected: set[Path] = set()
    for k_samples in K_VALUES:
        for domain in DOMAINS:
            for variant in FIXED_VARIANTS:
                expected.add(fixed_result_path(results_dir, k_samples, domain, variant))
            expected.add(routed_result_path(results_dir, k_samples, domain))
    return expected


def validate_result_inventory(results_dir: Path) -> set[Path]:
    """Reject missing or extra result JSON files in the matrix directories."""

    expected: set[Path] = expected_result_paths(results_dir)
    actual: set[Path] = set(results_dir.glob("k*/*.json"))
    actual.update(results_dir.glob("routed/k*/*.json"))
    missing: list[str] = sorted(str(path) for path in expected - actual)
    extra: list[str] = sorted(str(path) for path in actual - expected)
    if missing or extra:
        raise ValueError(
            f"K-ablation result inventory mismatch: expected={len(expected)}, "
            f"actual={len(actual)}, missing_sample={missing[:10]}, extra_sample={extra[:10]}"
        )
    return expected


def result_identity(value: dict[str, object], path: Path) -> tuple[str, str]:
    """Extract encoder and source revision from a finalized result stamp."""

    metadata: dict[str, object] = require_object(value.get("metadata"), f"{path}.metadata")
    stamp: dict[str, object] = require_object(
        metadata.get("k_ablation"), f"{path}.metadata.k_ablation"
    )
    encoder: str = require_string(stamp.get("encoder"), f"{path}.metadata.k_ablation.encoder")
    source_revision: str = require_string(
        stamp.get("source_revision"), f"{path}.metadata.k_ablation.source_revision"
    )
    return encoder, source_revision


def split_instance_ids(instances_path: Path, domain: str) -> dict[str, frozenset[str]]:
    """Return protocol all, validation, and test ID sets for one domain."""

    all_ids: frozenset[str] = frozenset(load_expected_gold(instances_path, domain))
    selected_validation_ids: frozenset[str] = validation_ids(all_ids)
    test_ids: frozenset[str] = all_ids - selected_validation_ids
    if not test_ids:
        raise ValueError(f"Test split is empty: domain={domain}, path={instances_path}")
    return {
        "all": all_ids,
        "validation": selected_validation_ids,
        "test": test_ids,
    }


def result_spec(
    tag: str,
    model: str,
    encoder: str,
    source_revision: str,
    k_samples: int,
    domain: str,
    variant: str,
) -> KResultSpec:
    """Return a complete result identity for validation."""

    return {
        "tag": tag,
        "model": model,
        "k_samples": k_samples,
        "domain": domain,
        "variant": variant,
        "encoder": encoder,
        "source_revision": source_revision,
    }


def summarize_result_payload(
    path: Path,
    value: dict[str, object],
    file_sha256: str,
    payload: RetrievalPayload,
    split_ids: dict[str, frozenset[str]],
) -> ResultData:
    """Reduce one validated payload to metrics and paired-test scores."""

    split_metrics: dict[str, dict[str, float]] = {
        split: compute_metrics(payload["results"], split_ids[split])
        for split in SPLITS
    }
    split_n: dict[str, int] = {split: len(split_ids[split]) for split in SPLITS}
    ndcg10_by_id: dict[str, float] = {
        record["instance_id"]: metric_for_record(record, "nDCG@10")
        for record in payload["results"]
    }
    metadata: dict[str, object] = payload["metadata"]
    raw_router: object = metadata.get("router")
    router: dict[str, object] | None = None
    if raw_router is not None:
        router = copy.deepcopy(require_object(raw_router, f"{path}.metadata.router"))
    return {
        "path": str(path),
        "sha256": file_sha256,
        "records_sha256": sha256_json(value.get("results")),
        "metrics_sha256": sha256_json(value.get("metrics")),
        "split_n": split_n,
        "split_metrics": split_metrics,
        "ndcg10_by_id": ndcg10_by_id,
        "router": router,
    }


def validate_matrix(
    tag: str,
    model: str,
    results_dir: Path,
    instances_dir: Path,
    community_root: Path,
    repository_root: Path,
) -> tuple[dict[tuple[int, str, str], ResultData], str, str, list[MetricRow]]:
    """Validate all 150 files and return memory-bounded metric summaries."""

    inventory: set[Path] = validate_result_inventory(results_dir)
    if len(inventory) != 150:
        raise ValueError(f"Expected exactly 150 result files: actual={len(inventory)}")
    first_path: Path = fixed_result_path(results_dir, K_VALUES[0], DOMAINS[0], FIXED_VARIANTS[0])
    first_value, _ = load_json_with_sha256(first_path)
    encoder, source_revision = result_identity(first_value, first_path)
    commit: str = repository_commit(repository_root)
    runner_bundle: dict[str, object] = result_bundle_manifest(repository_root)
    expected_source_revision: str = f"{commit[:7]}+bundle-{runner_bundle['sha256']}"
    if source_revision != expected_source_revision:
        raise ValueError(
            f"Result source revision does not match current runner bundle: "
            f"expected={expected_source_revision}, actual={source_revision}"
        )

    splits_by_domain: dict[str, dict[str, frozenset[str]]] = {
        domain: split_instance_ids(instances_dir / f"{domain}.json", domain)
        for domain in DOMAINS
    }
    matrix: dict[tuple[int, str, str], ResultData] = {}
    metric_rows: list[MetricRow] = []
    for k_samples in K_VALUES:
        cache_manifest: Path = (
            community_root / tag / f"imagination_full_k{k_samples}.manifest.json"
        )
        validate_cache_artifact(cache_manifest, repository_root)
        for domain in DOMAINS:
            instances_path: Path = instances_dir / f"{domain}.json"
            for variant in ALL_VARIANTS:
                path: Path = (
                    routed_result_path(results_dir, k_samples, domain)
                    if variant == "routed"
                    else fixed_result_path(results_dir, k_samples, domain, variant)
                )
                value, file_digest = load_json_with_sha256(path)
                actual_encoder, actual_source_revision = result_identity(value, path)
                if actual_encoder != encoder or actual_source_revision != source_revision:
                    raise ValueError(
                        f"Result identity drift: path={path}, encoder={actual_encoder!r}, "
                        f"source_revision={actual_source_revision!r}, "
                        f"expected_encoder={encoder!r}, expected_source_revision={source_revision!r}"
                    )
                payload: RetrievalPayload = validate_final_result(
                    value,
                    instances_path,
                    cache_manifest,
                    result_spec(
                        tag,
                        model,
                        encoder,
                        source_revision,
                        k_samples,
                        domain,
                        variant,
                    ),
                )
                data: ResultData = summarize_result_payload(
                    path,
                    value,
                    file_digest,
                    payload,
                    splits_by_domain[domain],
                )
                matrix[(k_samples, domain, variant)] = data
                for split in SPLITS:
                    for metric_name in METRIC_NAMES:
                        metric_rows.append(
                            {
                                "schema_version": 1,
                                "tag": tag,
                                "model": model,
                                "domain": domain,
                                "k_samples": k_samples,
                                "variant": variant,
                                "split": split,
                                "unit": "instances",
                                "n": data["split_n"][split],
                                "metric": metric_name,
                                "value": data["split_metrics"][split][metric_name],
                            }
                        )
    validate_routed_sources(matrix, results_dir)
    return matrix, encoder, source_revision, metric_rows


def validate_routed_sources(
    matrix: dict[tuple[int, str, str], ResultData], results_dir: Path
) -> None:
    """Prove every routed payload is an exact copy of its recorded source arm."""

    for k_samples in K_VALUES:
        for domain in DOMAINS:
            routed: ResultData = matrix[(k_samples, domain, "routed")]
            router: dict[str, object] = require_object(
                routed["router"], f"routed:k={k_samples},domain={domain}.router"
            )
            pick: str = require_string(
                router.get("pick"), f"routed:k={k_samples},domain={domain}.router.pick"
            )
            source: ResultData = matrix[(k_samples, domain, pick)]
            source_result: dict[str, object] = require_object(
                router.get("source_result"),
                f"routed:k={k_samples},domain={domain}.router.source_result",
            )
            expected_path: Path = fixed_result_path(results_dir, k_samples, domain, pick)
            recorded_path: str = require_string(
                source_result.get("path"),
                f"routed:k={k_samples},domain={domain}.router.source_result.path",
            )
            if Path(recorded_path) != expected_path:
                raise ValueError(
                    f"Routed source path mismatch: k={k_samples}, domain={domain}, "
                    f"expected={expected_path}, actual={recorded_path}"
                )
            recorded_digest: str = require_sha256(
                source_result.get("sha256"),
                f"routed:k={k_samples},domain={domain}.router.source_result.sha256",
            )
            if recorded_digest != source["sha256"]:
                raise ValueError(
                    f"Routed source digest mismatch: k={k_samples}, domain={domain}, "
                    f"expected={source['sha256']}, actual={recorded_digest}"
                )
            if routed["records_sha256"] != source["records_sha256"]:
                raise ValueError(
                    f"Routed records differ from selected source: k={k_samples}, "
                    f"domain={domain}, pick={pick}"
                )
            if routed["metrics_sha256"] != source["metrics_sha256"]:
                raise ValueError(
                    f"Routed full-data metrics differ from selected source: k={k_samples}, "
                    f"domain={domain}, pick={pick}"
                )


def aggregate_metric_rows(
    tag: str,
    model: str,
    matrix: dict[tuple[int, str, str], ResultData],
) -> list[MetricRow]:
    """Build strict cross-domain micro and macro rows from domain cells."""

    rows: list[MetricRow] = []
    for k_samples in K_VALUES:
        for variant in ALL_VARIANTS:
            for split in SPLITS:
                domain_data: list[ResultData] = [
                    matrix[(k_samples, domain, variant)] for domain in DOMAINS
                ]
                total_n: int = sum(data["split_n"][split] for data in domain_data)
                for metric_name in METRIC_NAMES:
                    weighted_sum: float = sum(
                        data["split_metrics"][split][metric_name]
                        * data["split_n"][split]
                        for data in domain_data
                    )
                    macro_value: float = sum(
                        data["split_metrics"][split][metric_name]
                        for data in domain_data
                    ) / len(DOMAINS)
                    rows.append(
                        {
                            "schema_version": 1,
                            "tag": tag,
                            "model": model,
                            "domain": "__all_micro__",
                            "k_samples": k_samples,
                            "variant": variant,
                            "split": split,
                            "unit": "instances",
                            "n": total_n,
                            "metric": metric_name,
                            "value": weighted_sum / total_n,
                        }
                    )
                    rows.append(
                        {
                            "schema_version": 1,
                            "tag": tag,
                            "model": model,
                            "domain": "__all_macro__",
                            "k_samples": k_samples,
                            "variant": variant,
                            "split": split,
                            "unit": "domains",
                            "n": len(DOMAINS),
                            "metric": metric_name,
                            "value": macro_value,
                        }
                    )
    return rows


def metric_rows_summary(metric_rows: list[MetricRow]) -> dict[str, object]:
    """Convert normalized metric rows into deterministic nested JSON."""

    summary: dict[str, object] = {}
    for row in metric_rows:
        raw_domain_block: object = summary.get(row["domain"])
        if raw_domain_block is None:
            raw_domain_block = {}
            summary[row["domain"]] = raw_domain_block
        domain_block: dict[str, object] = require_object(
            raw_domain_block, f"summary.{row['domain']}"
        )
        k_key: str = f"k{row['k_samples']}"
        raw_k_block: object = domain_block.get(k_key)
        if raw_k_block is None:
            raw_k_block = {}
            domain_block[k_key] = raw_k_block
        k_block: dict[str, object] = require_object(
            raw_k_block, f"summary.{row['domain']}.{k_key}"
        )
        raw_variant_block: object = k_block.get(row["variant"])
        if raw_variant_block is None:
            raw_variant_block = {}
            k_block[row["variant"]] = raw_variant_block
        variant_block: dict[str, object] = require_object(
            raw_variant_block,
            f"summary.{row['domain']}.{k_key}.{row['variant']}",
        )
        raw_split_block: object = variant_block.get(row["split"])
        if raw_split_block is None:
            raw_split_block = {
                "unit": row["unit"],
                "n": row["n"],
                "metrics": {},
            }
            variant_block[row["split"]] = raw_split_block
        split_block: dict[str, object] = require_object(
            raw_split_block,
            f"summary.{row['domain']}.{k_key}.{row['variant']}.{row['split']}",
        )
        metrics: dict[str, object] = require_object(
            split_block.get("metrics"),
            f"summary.{row['domain']}.k{row['k_samples']}.{row['variant']}.{row['split']}",
        )
        metrics[row["metric"]] = row["value"]
    return summary


def comparison_seed(base_seed: int, key: str) -> int:
    """Derive an order-independent NumPy seed for one named comparison."""

    digest: bytes = hashlib.sha256(key.encode()).digest()
    return (int.from_bytes(digest[:8], byteorder="big") ^ base_seed) % (2**63)


def bootstrap_distribution(
    differences: np.ndarray, bootstrap_samples: int, seed: int
) -> np.ndarray:
    """Return paired bootstrap means without allocating a full index cube."""

    if differences.ndim != 1 or differences.size == 0:
        raise ValueError(
            f"Bootstrap differences must be a non-empty vector: shape={differences.shape}"
        )
    if bootstrap_samples <= 0:
        raise ValueError(
            f"Bootstrap sample count must be positive: value={bootstrap_samples}"
        )
    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = int(differences.size)
    chunk_size: int = max(1, min(512, 8_000_000 // n))
    distribution: np.ndarray = np.empty(bootstrap_samples, dtype=np.float64)
    offset: int = 0
    while offset < bootstrap_samples:
        current: int = min(chunk_size, bootstrap_samples - offset)
        indices: np.ndarray = rng.integers(0, n, size=(current, n), endpoint=False)
        distribution[offset : offset + current] = differences[indices].mean(axis=1)
        offset += current
    return distribution


def bootstrap_stats(
    values_k: np.ndarray,
    values_k4: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Return paired K-minus-K4 effect statistics and a two-sided p-value."""

    if values_k.shape != values_k4.shape:
        raise ValueError(
            f"Paired arrays have different shapes: k={values_k.shape}, k4={values_k4.shape}"
        )
    differences: np.ndarray = values_k - values_k4
    distribution: np.ndarray = bootstrap_distribution(
        differences, bootstrap_samples, seed
    )
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    probability_nonpositive: float = float(np.mean(distribution <= 0.0))
    probability_nonnegative: float = float(np.mean(distribution >= 0.0))
    p_value: float = min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
    return {
        "n": int(values_k.size),
        "mean_k": float(values_k.mean()),
        "mean_k4": float(values_k4.mean()),
        "difference_k_minus_k4": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": p_value,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def paired_arrays(
    data_k: ResultData, data_k4: ResultData, selected_ids: frozenset[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-order nDCG@10 arrays for an explicit paired ID set."""

    ordered_ids: list[str] = sorted(selected_ids)
    missing_k: list[str] = [
        instance_id
        for instance_id in ordered_ids
        if instance_id not in data_k["ndcg10_by_id"]
    ]
    missing_k4: list[str] = [
        instance_id
        for instance_id in ordered_ids
        if instance_id not in data_k4["ndcg10_by_id"]
    ]
    if missing_k or missing_k4:
        raise ValueError(
            f"Paired score coverage mismatch: missing_k={missing_k[:5]}, "
            f"missing_k4={missing_k4[:5]}"
        )
    values_k: np.ndarray = np.asarray(
        [data_k["ndcg10_by_id"][instance_id] for instance_id in ordered_ids],
        dtype=np.float64,
    )
    values_k4: np.ndarray = np.asarray(
        [data_k4["ndcg10_by_id"][instance_id] for instance_id in ordered_ids],
        dtype=np.float64,
    )
    return values_k, values_k4


def macro_bootstrap_stats(
    domain_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    bootstrap_samples: int,
    seed: int,
    comparison_key: str,
) -> dict[str, object]:
    """Return a domain-stratified bootstrap for the macro-average effect."""

    domain_distributions: list[np.ndarray] = []
    mean_k: float = 0.0
    mean_k4: float = 0.0
    total_n: int = 0
    for domain in DOMAINS:
        values_k, values_k4 = domain_pairs[domain]
        domain_seed: int = comparison_seed(seed, f"{comparison_key}:{domain}")
        domain_distributions.append(
            bootstrap_distribution(
                values_k - values_k4,
                bootstrap_samples,
                domain_seed,
            )
        )
        mean_k += float(values_k.mean())
        mean_k4 += float(values_k4.mean())
        total_n += int(values_k.size)
    distribution: np.ndarray = np.stack(domain_distributions).mean(axis=0)
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
        "n_instances": total_n,
        "n_domains": len(DOMAINS),
        "mean_k": mean_k / len(DOMAINS),
        "mean_k4": mean_k4 / len(DOMAINS),
        "difference_k_minus_k4": float(distribution.mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": p_value,
        "bootstrap_samples": bootstrap_samples,
        "seed_scheme": "independent paired resampling within each domain",
    }


def paired_vs_k4(
    matrix: dict[tuple[int, str, str], ResultData],
    instances_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Build test-split paired comparisons for every variant and routed domains."""

    test_ids_by_domain: dict[str, frozenset[str]] = {
        domain: split_instance_ids(instances_dir / f"{domain}.json", domain)["test"]
        for domain in DOMAINS
    }
    comparisons: dict[str, object] = {}
    for variant in ALL_VARIANTS:
        variant_comparisons: dict[str, object] = {}
        for k_samples in COMPARISON_K_VALUES:
            domain_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for domain in DOMAINS:
                domain_pairs[domain] = paired_arrays(
                    matrix[(k_samples, domain, variant)],
                    matrix[(4, domain, variant)],
                    test_ids_by_domain[domain],
                )
            micro_k: np.ndarray = np.concatenate(
                [domain_pairs[domain][0] for domain in DOMAINS]
            )
            micro_k4: np.ndarray = np.concatenate(
                [domain_pairs[domain][1] for domain in DOMAINS]
            )
            comparison_key: str = f"{variant}:k{k_samples}-k4:test:ndcg10"
            result: dict[str, object] = {
                "cross_domain_micro": bootstrap_stats(
                    micro_k,
                    micro_k4,
                    bootstrap_samples,
                    comparison_seed(bootstrap_seed, f"{comparison_key}:micro"),
                ),
                "cross_domain_macro_stratified": macro_bootstrap_stats(
                    domain_pairs,
                    bootstrap_samples,
                    bootstrap_seed,
                    f"{comparison_key}:macro",
                ),
            }
            if variant == "routed":
                result["by_domain"] = {
                    domain: bootstrap_stats(
                        domain_pairs[domain][0],
                        domain_pairs[domain][1],
                        bootstrap_samples,
                        comparison_seed(
                            bootstrap_seed, f"{comparison_key}:domain:{domain}"
                        ),
                    )
                    for domain in DOMAINS
                }
            variant_comparisons[f"k{k_samples}_vs_k4"] = result
        comparisons[variant] = variant_comparisons
    return {
        "schema_version": 1,
        "metric": "nDCG@10",
        "split": "test",
        "contrast": "K minus K=4",
        "validation_fraction": 0.2,
        "validation_seed": 0,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "comparisons": comparisons,
    }


def empty_text_totals() -> TextTotals:
    """Return zeroed character totals."""

    return {"prompt_chars": 0, "output_chars": 0, "generations": 0}


def update_text_totals(
    totals: TextTotals, prompt_chars: int, output_chars: int, generations: int
) -> None:
    """Accumulate non-negative character counts into one local result object."""

    if prompt_chars < 0 or output_chars < 0 or generations < 0:
        raise ValueError(
            f"Text totals must be non-negative: prompt={prompt_chars}, "
            f"output={output_chars}, generations={generations}"
        )
    totals["prompt_chars"] += prompt_chars
    totals["output_chars"] += output_chars
    totals["generations"] += generations


def digest_update(digest: Digest, *values: str) -> None:
    """Update a digest with length-delimited UTF-8 strings."""

    for value in values:
        encoded: bytes = value.encode()
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)


def stream_cache_artifact(artifact_path: Path, k_samples: int) -> CacheArtifactStats:
    """Validate cache row structure and collect actual-text cost statistics."""

    by_template: dict[str, TextTotals] = {
        template_name: empty_text_totals() for template_name in TEMPLATE_TEXTS
    }
    unique_totals: dict[str, TextTotals] = {
        template_name: empty_text_totals() for template_name in TEMPLATE_TEXTS
    }
    by_domain_template: dict[str, dict[str, TextTotals]] = {
        domain: {
            template_name: empty_text_totals() for template_name in TEMPLATE_TEXTS
        }
        for domain in DOMAINS
    }
    query_rows_digest = hashlib.sha256()
    sample_digests: dict[str, list[Digest]] = {
        template_name: [hashlib.sha256() for _ in range(k_samples)]
        for template_name in TEMPLATE_TEXTS
    }
    seen_instance_ids: set[str] = set()
    query_payload_sha: dict[str, str] = {}
    domain_rows: dict[str, int] = {domain: 0 for domain in DOMAINS}
    row_count: int = 0
    with gzip.open(artifact_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            raw_row: object = json.loads(line)
            row: dict[str, object] = require_object(
                raw_row, f"cache_artifact:{artifact_path}[{line_number}]"
            )
            instance_id: str = require_string(
                row.get("instance_id"),
                f"cache_artifact:{artifact_path}[{line_number}].instance_id",
            )
            if instance_id in seen_instance_ids:
                raise ValueError(
                    f"Duplicate cache artifact instance ID: artifact={artifact_path}, "
                    f"instance_id={instance_id}"
                )
            seen_instance_ids.add(instance_id)
            domain: str = require_string(
                row.get("domain"),
                f"cache_artifact:{artifact_path}[{line_number}].domain",
            )
            if domain not in DOMAINS:
                raise ValueError(
                    f"Unsupported cache artifact domain: artifact={artifact_path}, "
                    f"domain={domain}"
                )
            query: str = require_string(
                row.get("query"),
                f"cache_artifact:{artifact_path}[{line_number}].query",
            )
            imaginations: dict[str, object] = require_object(
                row.get("imaginations"),
                f"cache_artifact:{artifact_path}[{line_number}].imaginations",
            )
            if set(imaginations) != set(TEMPLATE_TEXTS):
                raise ValueError(
                    f"Cache artifact template mismatch: artifact={artifact_path}, "
                    f"line={line_number}, actual={sorted(imaginations)}"
                )
            digest_update(query_rows_digest, instance_id, domain, query)
            domain_rows[domain] += 1
            row_imagination_identity: dict[str, list[str]] = {}
            is_unique_query: bool = query not in query_payload_sha
            for template_name, template_text in TEMPLATE_TEXTS.items():
                raw_samples: list[object] = require_list(
                    imaginations.get(template_name),
                    f"cache_artifact:{artifact_path}[{line_number}].imaginations.{template_name}",
                )
                if len(raw_samples) != k_samples:
                    raise ValueError(
                        f"Cache artifact sample count mismatch: artifact={artifact_path}, "
                        f"line={line_number}, template={template_name}, "
                        f"expected={k_samples}, actual={len(raw_samples)}"
                    )
                samples: list[str] = [
                    require_string(
                        sample,
                        f"cache_artifact:{artifact_path}[{line_number}]."
                        f"imaginations.{template_name}[{sample_index}]",
                    )
                    for sample_index, sample in enumerate(raw_samples)
                ]
                row_imagination_identity[template_name] = samples
                prompt_chars: int = len(template_text.format(q=query))
                output_chars: int = sum(len(sample) for sample in samples)
                update_text_totals(
                    by_template[template_name], prompt_chars, output_chars, len(samples)
                )
                update_text_totals(
                    by_domain_template[domain][template_name],
                    prompt_chars,
                    output_chars,
                    len(samples),
                )
                if is_unique_query:
                    update_text_totals(
                        unique_totals[template_name],
                        prompt_chars,
                        output_chars,
                        len(samples),
                    )
                for sample_index, sample in enumerate(samples):
                    digest_update(
                        sample_digests[template_name][sample_index],
                        instance_id,
                        sample,
                    )
            imagination_sha: str = sha256_json(row_imagination_identity)
            if is_unique_query:
                query_payload_sha[query] = imagination_sha
            elif query_payload_sha[query] != imagination_sha:
                raise ValueError(
                    f"Identical query has inconsistent cached imaginations: "
                    f"artifact={artifact_path}, instance_id={instance_id}"
                )
            row_count += 1
    if row_count != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Cache artifact row count mismatch: artifact={artifact_path}, "
            f"expected={EXPECTED_TOTAL_ROWS}, actual={row_count}"
        )
    if len(query_payload_sha) != EXPECTED_UNIQUE_QUERIES:
        raise ValueError(
            f"Cache artifact unique-query count mismatch: artifact={artifact_path}, "
            f"expected={EXPECTED_UNIQUE_QUERIES}, actual={len(query_payload_sha)}"
        )
    expected_domain_rows: dict[str, int] = {
        "theoremqa": 747,
        "logicbench": 760,
        "medcalcbench": 1100,
        "champ": 223,
        "bigcodebench": 1140,
    }
    if domain_rows != expected_domain_rows:
        raise ValueError(
            f"Cache artifact domain counts mismatch: artifact={artifact_path}, "
            f"expected={expected_domain_rows}, actual={domain_rows}"
        )
    return {
        "artifact_path": str(artifact_path),
        "rows": row_count,
        "unique_queries": len(query_payload_sha),
        "query_rows_sha256": query_rows_digest.hexdigest(),
        "sample_sha256": {
            template_name: [digest.hexdigest() for digest in digests]
            for template_name, digests in sample_digests.items()
        },
        "by_template": by_template,
        "by_unique_query_template": unique_totals,
        "by_domain_template": by_domain_template,
    }


def validate_cache_prefixes(stats_by_k: dict[int, CacheArtifactStats]) -> None:
    """Prove every smaller cache artifact is the exact prefix of K=10."""

    reference: CacheArtifactStats = stats_by_k[10]
    for k_samples in K_VALUES:
        stats: CacheArtifactStats = stats_by_k[k_samples]
        if stats["query_rows_sha256"] != reference["query_rows_sha256"]:
            raise ValueError(
                f"Cache artifact query rows differ across K: k={k_samples}, "
                f"reference_k=10"
            )
        for template_name in TEMPLATE_TEXTS:
            expected_prefix: list[str] = reference["sample_sha256"][template_name][
                :k_samples
            ]
            actual_prefix: list[str] = stats["sample_sha256"][template_name]
            if actual_prefix != expected_prefix:
                raise ValueError(
                    f"Cache artifact is not an exact K=10 prefix: k={k_samples}, "
                    f"template={template_name}"
                )


def token_cost(totals: TextTotals, query_count: int) -> dict[str, object]:
    """Convert actual text lengths into the protocol's token estimate."""

    if query_count <= 0 or totals["generations"] <= 0:
        raise ValueError(
            f"Token cost requires positive counts: queries={query_count}, "
            f"generations={totals['generations']}"
        )
    input_tokens_total: float = totals["prompt_chars"] / CHARS_PER_TOKEN
    output_tokens_total: float = totals["output_chars"] / CHARS_PER_TOKEN
    return {
        "queries": query_count,
        "generations": totals["generations"],
        "input_tokens_total_estimate": input_tokens_total,
        "output_tokens_total_estimate": output_tokens_total,
        "input_tokens_per_query_estimate": input_tokens_total / query_count,
        "output_tokens_per_query_estimate": output_tokens_total / query_count,
        "total_tokens_per_query_estimate": (
            input_tokens_total + output_tokens_total
        )
        / query_count,
        "output_tokens_per_generation_estimate": output_tokens_total
        / totals["generations"],
    }


def routed_cost(
    stats: CacheArtifactStats,
    matrix: dict[tuple[int, str, str], ResultData],
    k_samples: int,
) -> dict[str, object]:
    """Return row-weighted generation cost for per-domain routed templates."""

    totals: TextTotals = empty_text_totals()
    picks: dict[str, str] = {}
    for domain in DOMAINS:
        router: dict[str, object] = require_object(
            matrix[(k_samples, domain, "routed")]["router"],
            f"cost.router:k={k_samples},domain={domain}",
        )
        pick: str = require_string(
            router.get("pick"), f"cost.router:k={k_samples},domain={domain}.pick"
        )
        template_name: str = VARIANT_TEMPLATES[pick]
        domain_totals: TextTotals = stats["by_domain_template"][domain][template_name]
        update_text_totals(
            totals,
            domain_totals["prompt_chars"] * k_samples,
            domain_totals["output_chars"],
            domain_totals["generations"],
        )
        picks[domain] = pick
    return {"picks": picks, **token_cost(totals, EXPECTED_TOTAL_ROWS)}


def warmup_timing(results_dir: Path, tag: str, model: str) -> dict[str, object]:
    """Load recorded warmup timing, or state why wall-clock evidence is absent."""

    path: Path = results_dir / "audits" / "warmup-timing.json"
    if not path.is_file():
        return {
            "status": "unavailable",
            "reason": (
                "The completed cache export did not record generation wall-clock time; "
                "no retrospective estimate is substituted."
            ),
        }
    value: dict[str, object] = load_json_object(path)
    if value.get("tag") != tag or value.get("model") != model:
        raise ValueError(
            f"Warmup timing identity mismatch: path={path}, expected_tag={tag}, "
            f"actual_tag={value.get('tag')!r}, expected_model={model}, "
            f"actual_model={value.get('model')!r}"
        )
    stages: list[object] = require_list(value.get("stages"), f"{path}.stages")
    if not stages:
        raise ValueError(f"Warmup timing has no stages: path={path}")
    total_seconds: float = 0.0
    for stage_index, raw_stage in enumerate(stages):
        stage: dict[str, object] = require_object(raw_stage, f"{path}.stages[{stage_index}]")
        seconds: float = require_float(
            stage.get("wall_clock_seconds"),
            f"{path}.stages[{stage_index}].wall_clock_seconds",
        )
        if seconds < 0.0:
            raise ValueError(
                f"Warmup timing must be non-negative: path={path}, "
                f"stage={stage_index}, seconds={seconds}"
            )
        total_seconds += seconds
    return {
        "status": "recorded",
        "path": str(path),
        "sha256": sha256_file(path),
        "total_wall_clock_seconds": total_seconds,
        "stages": stages,
    }


def build_cost_report(
    tag: str,
    model: str,
    community_root: Path,
    repository_root: Path,
    results_dir: Path,
    matrix: dict[tuple[int, str, str], ResultData],
) -> dict[str, object]:
    """Audit exact cache prefixes and estimate tokens from their actual text."""

    stats_by_k: dict[int, CacheArtifactStats] = {}
    for k_samples in K_VALUES:
        manifest_path: Path = (
            community_root / tag / f"imagination_full_k{k_samples}.manifest.json"
        )
        artifact_path: Path = validate_cache_artifact(manifest_path, repository_root)
        stats_by_k[k_samples] = stream_cache_artifact(artifact_path, k_samples)
    validate_cache_prefixes(stats_by_k)

    by_k: dict[str, object] = {}
    for k_samples in K_VALUES:
        stats: CacheArtifactStats = stats_by_k[k_samples]
        templates: dict[str, object] = {}
        for template_name in TEMPLATE_TEXTS:
            row_totals: TextTotals = stats["by_template"][template_name]
            scaled_row_totals: TextTotals = {
                "prompt_chars": row_totals["prompt_chars"] * k_samples,
                "output_chars": row_totals["output_chars"],
                "generations": row_totals["generations"],
            }
            unique_totals: TextTotals = stats["by_unique_query_template"][template_name]
            scaled_unique_totals: TextTotals = {
                "prompt_chars": unique_totals["prompt_chars"] * k_samples,
                "output_chars": unique_totals["output_chars"],
                "generations": unique_totals["generations"],
            }
            templates[template_name] = {
                "per_evaluated_instance": token_cost(
                    scaled_row_totals, EXPECTED_TOTAL_ROWS
                ),
                "actual_unique_cache": token_cost(
                    scaled_unique_totals, EXPECTED_UNIQUE_QUERIES
                ),
            }
        variant_costs: dict[str, object] = {
            variant: templates[template_name]
            for variant, template_name in VARIANT_TEMPLATES.items()
        }
        variant_costs["routed"] = routed_cost(stats, matrix, k_samples)
        by_k[f"k{k_samples}"] = {
            "artifact_path": stats["artifact_path"],
            "templates": templates,
            "variants": variant_costs,
        }
    return {
        "schema_version": 1,
        "tag": tag,
        "model": model,
        "estimator": {
            "kind": "actual UTF-8 text character count divided by a fixed ratio",
            "characters_per_token": CHARS_PER_TOKEN,
            "scope": "generation prompt plus cached generation output",
        },
        "cache_prefix_validation": {
            "reference_k": 10,
            "validated_k_values": list(K_VALUES),
            "exact_prefix": True,
        },
        "by_k": by_k,
        "warmup_wall_clock": warmup_timing(results_dir, tag, model),
    }


def validate_full_cache_audit(results_dir: Path, model: str) -> dict[str, object]:
    """Validate the full five-domain K=10 raw-cache audit."""

    path: Path = results_dir / "audits" / "cache-k10.json"
    audit: dict[str, object] = load_json_object(path)
    expected_fields: dict[str, object] = {
        "schema_version": 1,
        "model": model,
        "k_samples": 10,
        "temperature": 0.7,
        "unique_queries": EXPECTED_UNIQUE_QUERIES,
        "expected": EXPECTED_CACHE_FILES_PER_SAMPLE * 10,
        "present": EXPECTED_CACHE_FILES_PER_SAMPLE * 10,
        "missing": 0,
        "empty": 0,
        "unreadable": 0,
        "complete": True,
    }
    for field, expected_value in expected_fields.items():
        actual_value: object = audit.get(field)
        if actual_value != expected_value:
            raise ValueError(
                f"Full cache audit mismatch: path={path}, field={field}, "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    raw_templates: list[object] = require_list(audit.get("templates"), f"{path}.templates")
    templates: set[str] = {
        require_string(value, f"{path}.templates") for value in raw_templates
    }
    if templates != set(TEMPLATE_TEXTS):
        raise ValueError(
            f"Full cache audit template mismatch: path={path}, "
            f"expected={sorted(TEMPLATE_TEXTS)}, actual={sorted(templates)}"
        )
    instance_rows: dict[str, object] = require_object(
        audit.get("instance_rows"), f"{path}.instance_rows"
    )
    actual_counts: list[int] = sorted(
        require_integer(value, f"{path}.instance_rows")
        for value in instance_rows.values()
    )
    if actual_counts != sorted((747, 760, 1100, 223, 1140)):
        raise ValueError(
            f"Full cache audit instance counts mismatch: path={path}, "
            f"actual={actual_counts}"
        )
    return {"path": str(path), "sha256": sha256_file(path), "complete": True}


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    """Atomically replace one derived JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, mode="w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def write_jsonl_gzip_atomic(path: Path, rows: list[MetricRow]) -> None:
    """Atomically write deterministic gzip JSONL metric rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, mode="wb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as archive:
                for row in rows:
                    line: bytes = (
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                    archive.write(line)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def result_matrix_digest(matrix: dict[tuple[int, str, str], ResultData]) -> str:
    """Return a stable digest over every result path and exact file hash."""

    rows: list[dict[str, object]] = [
        {
            "k_samples": key[0],
            "domain": key[1],
            "variant": key[2],
            "path": data["path"],
            "sha256": data["sha256"],
        }
        for key, data in sorted(matrix.items())
    ]
    return sha256_json(rows)


def cache_provenance(
    tag: str, community_root: Path, instances_dir: Path
) -> tuple[dict[str, object], str]:
    """Return cache manifest identities, dataset hashes, and model revision."""

    manifests: dict[str, object] = {}
    model_revisions: set[str] = set()
    generation_commits: set[str] = set()
    data_hashes: dict[str, str] = {}
    for k_samples in K_VALUES:
        path: Path = community_root / tag / f"imagination_full_k{k_samples}.manifest.json"
        manifest: dict[str, object] = load_json_object(path)
        model_revisions.add(
            require_string(manifest.get("model_revision"), f"{path}.model_revision")
        )
        generation_commits.add(
            require_string(
                manifest.get("generation_code_commit"),
                f"{path}.generation_code_commit",
            )
        )
        output: dict[str, object] = require_object(manifest.get("output"), f"{path}.output")
        manifests[f"k{k_samples}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "artifact_sha256": require_sha256(
                output.get("sha256"), f"{path}.output.sha256"
            ),
        }
        instance_entries: dict[str, object] = require_object(
            manifest.get("instances"), f"{path}.instances"
        )
        for domain in DOMAINS:
            entry: dict[str, object] = require_object(
                instance_entries.get(domain), f"{path}.instances.{domain}"
            )
            manifest_hash: str = require_sha256(
                entry.get("sha256"), f"{path}.instances.{domain}.sha256"
            )
            actual_hash: str = sha256_file(instances_dir / f"{domain}.json")
            if manifest_hash != actual_hash:
                raise ValueError(
                    f"Dataset hash drift: domain={domain}, manifest={manifest_hash}, "
                    f"actual={actual_hash}"
                )
            data_hashes[domain] = actual_hash
    if len(model_revisions) != 1:
        raise ValueError(
            f"Cache manifests disagree on model revision: revisions={sorted(model_revisions)}"
        )
    return (
        {
            "manifests": manifests,
            "generation_code_commits": sorted(generation_commits),
            "data_sha256": data_hashes,
        },
        next(iter(model_revisions)),
    )


def router_summary(
    matrix: dict[tuple[int, str, str], ResultData]
) -> dict[str, object]:
    """Return every validated per-K, per-domain routing decision."""

    return {
        f"k{k_samples}": {
            domain: matrix[(k_samples, domain, "routed")]["router"]
            for domain in DOMAINS
        }
        for k_samples in K_VALUES
    }


def main() -> None:
    """Validate, summarize, test, cost, and manifest one model matrix."""

    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError(
            f"bootstrap-samples must be positive: value={args.bootstrap_samples}"
        )
    repository_root: Path = Path(__file__).resolve().parents[1]
    results_dir: Path = args.results_dir
    instances_dir: Path = args.instances_dir
    community_root: Path = args.community_root
    matrix, encoder, source_revision, domain_metric_rows = validate_matrix(
        args.tag,
        args.model,
        results_dir,
        instances_dir,
        community_root,
        repository_root,
    )
    aggregate_rows: list[MetricRow] = aggregate_metric_rows(
        args.tag, args.model, matrix
    )
    metric_rows: list[MetricRow] = domain_metric_rows + aggregate_rows
    if len(metric_rows) != 5040:
        raise ValueError(
            f"Unexpected normalized metric row count: expected=5040, actual={len(metric_rows)}"
        )
    paired: dict[str, object] = paired_vs_k4(
        matrix,
        instances_dir,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    cost: dict[str, object] = build_cost_report(
        args.tag,
        args.model,
        community_root,
        repository_root,
        results_dir,
        matrix,
    )
    cache_audit: dict[str, object] = validate_full_cache_audit(results_dir, args.model)
    cache_info, model_revision = cache_provenance(
        args.tag, community_root, instances_dir
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "tag": args.tag,
        "model": args.model,
        "model_revision": model_revision,
        "encoder": encoder,
        "source_revision": source_revision,
        "protocol": {
            "k_values": list(K_VALUES),
            "domains": list(DOMAINS),
            "variants": list(ALL_VARIANTS),
            "splits": list(SPLITS),
            "validation_fraction": 0.2,
            "validation_seed": 0,
            "primary_metric": "routed test nDCG@10",
        },
        "result_files": len(matrix),
        "metric_rows": len(metric_rows),
        "metrics": metric_rows_summary(metric_rows),
        "router": router_summary(matrix),
    }
    output_dir: Path = community_root / args.tag / "k-ablation"
    metrics_path: Path = output_dir / "metrics_long.jsonl.gz"
    summary_path: Path = output_dir / "summary.json"
    paired_path: Path = output_dir / "paired_vs_k4.json"
    cost_path: Path = output_dir / "cost.json"
    manifest_path: Path = output_dir / "manifest.json"
    write_jsonl_gzip_atomic(metrics_path, metric_rows)
    write_json_atomic(summary_path, summary)
    write_json_atomic(paired_path, paired)
    write_json_atomic(cost_path, cost)
    runner_bundle: dict[str, object] = result_bundle_manifest(repository_root)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "complete_model_k_ablation",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "model": args.model,
        "model_revision": model_revision,
        "encoder": encoder,
        "repository_commit": repository_commit(repository_root),
        "source_revision": source_revision,
        "runner_bundle": runner_bundle,
        "summarizer_sha256": sha256_file(Path(__file__).resolve()),
        "result_files": len(matrix),
        "result_matrix_sha256": result_matrix_digest(matrix),
        "cache_audit": cache_audit,
        "cache": cache_info,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
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
                "tag": args.tag,
                "model": args.model,
                "result_files": len(matrix),
                "metric_rows": len(metric_rows),
                "output_dir": str(output_dir),
                "manifest_sha256": sha256_file(manifest_path),
                "verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
