#!/usr/bin/env python3
"""Summarize K=2 answer accuracy and preregistered paired comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np

from hyskill.downstream_reuse import (
    DownstreamDataError,
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_file,
)
from scripts.audit_k2_reuse import (
    require_list,
    require_object,
    require_string,
)
from scripts.summarize_k_ablation import bootstrap_stats


DOMAINS: tuple[str, ...] = (
    "theoremqa",
    "logicbench",
    "medcalcbench",
    "champ",
)
DOMAIN_COUNTS: dict[str, int] = {
    "theoremqa": 747,
    "logicbench": 760,
    "medcalcbench": 1100,
    "champ": 223,
}
DOMAIN_VALIDATION_COUNTS: dict[str, int] = {
    domain: max(1, int(count * 0.2))
    for domain, count in DOMAIN_COUNTS.items()
}
ArmName = Literal[
    "routed_always",
    "routed_gated",
    "routed_select",
    "fixed_gated",
]
SplitName = Literal["full", "heldout"]


class AnswerEvaluationRow(TypedDict):
    """Fields required for answer metrics and paired tests."""

    model: str
    domain: str
    arm: ArmName
    instance_id: str
    correct: bool
    failure_category: str
    is_validation: bool


class ComparisonSpec(TypedDict):
    """One named paired comparison and its strict model support."""

    name: str
    arm_a: ArmName
    arm_b: ArmName
    models: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Parse explicit supports, bootstrap settings, and output paths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--seven-models", required=True)
    parser.add_argument("--five-models", required=True)
    parser.add_argument("--fixed-model", required=True)
    parser.add_argument("--expected-total-per-model", required=True, type=int)
    parser.add_argument("--expected-heldout-per-model", required=True, type=int)
    parser.add_argument("--bootstrap-samples", required=True, type=int)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    parser.add_argument("--output-long", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-comparisons", required=True, type=Path)
    return parser.parse_args()


def parse_model_list(value: str, name: str) -> tuple[str, ...]:
    """Parse one ordered, unique comma-separated model support."""

    models: tuple[str, ...] = tuple(
        model.strip() for model in value.split(",") if model.strip()
    )
    if not models:
        raise ValueError(f"{name} must contain at least one model")
    if len(models) != len(set(models)):
        raise ValueError(f"{name} contains duplicate models: models={models}")
    return models


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return one JSON Boolean."""

    if not isinstance(value, bool):
        raise DownstreamDataError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


def require_arm(value: JsonValue | None, context: str) -> ArmName:
    """Return one supported K=2 answer arm."""

    arm: str = require_string(value, context)
    if arm not in (
        "routed_always",
        "routed_gated",
        "routed_select",
        "fixed_gated",
    ):
        raise DownstreamDataError(
            f"Unknown K=2 answer arm: context={context}, arm={arm}"
        )
    return cast(ArmName, arm)


def load_json(path: Path) -> JsonObject:
    """Load one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Answer evaluation input does not exist: path={path}"
        )
    try:
        raw_value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise DownstreamDataError(
            "Answer evaluation JSON is malformed: "
            f"path={path}, line={error.lineno}, column={error.colno}, "
            f"message={error.msg}"
        ) from error
    return require_object(raw_value, f"evaluation:{path}")


def parse_row(
    raw_row: JsonValue,
    path: Path,
    row_index: int,
) -> AnswerEvaluationRow:
    """Validate one per-instance answer evaluation row."""

    context: str = f"evaluation:{path}:details[{row_index}]"
    row: JsonObject = require_object(raw_row, context)
    domain: str = require_string(row.get("domain"), f"{context}.domain")
    if domain not in DOMAINS:
        raise DownstreamDataError(
            f"Unknown answer domain: context={context}, domain={domain}"
        )
    return {
        "model": require_string(row.get("model"), f"{context}.model"),
        "domain": domain,
        "arm": require_arm(row.get("arm"), f"{context}.arm"),
        "instance_id": require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        ),
        "correct": require_boolean(
            row.get("correct"),
            f"{context}.correct",
        ),
        "failure_category": require_string(
            row.get("failure_category"),
            f"{context}.failure_category",
        ),
        "is_validation": require_boolean(
            row.get("is_validation"),
            f"{context}.is_validation",
        ),
    }


def load_rows(paths: Sequence[Path]) -> list[AnswerEvaluationRow]:
    """Load unique answer rows from explicit per-job evaluations."""

    rows: list[AnswerEvaluationRow] = []
    seen_keys: set[tuple[str, str, ArmName, str]] = set()
    for path in paths:
        resolved_path: Path = path.resolve()
        payload: JsonObject = load_json(resolved_path)
        if payload.get("schema_version") != "k2-answer-evaluation-v1":
            raise DownstreamDataError(
                "Unexpected answer evaluation schema: "
                f"path={resolved_path}, "
                f"schema={payload.get('schema_version')!r}"
            )
        raw_details: list[JsonValue] = require_list(
            payload.get("details"),
            f"evaluation:{resolved_path}.details",
        )
        for row_index, raw_row in enumerate(raw_details):
            row: AnswerEvaluationRow = parse_row(
                raw_row,
                resolved_path,
                row_index,
            )
            key: tuple[str, str, ArmName, str] = (
                row["model"],
                row["domain"],
                row["arm"],
                row["instance_id"],
            )
            if key in seen_keys:
                raise DownstreamDataError(
                    f"Duplicate answer evaluation row: key={key}"
                )
            seen_keys.add(key)
            rows.append(row)
    if not rows:
        raise DownstreamDataError("No answer evaluation rows were loaded")
    return rows


def split_rows(
    rows: Sequence[AnswerEvaluationRow],
    split: SplitName,
) -> list[AnswerEvaluationRow]:
    """Return full or held-out answer support."""

    if split == "full":
        return list(rows)
    return [row for row in rows if not row["is_validation"]]


def metric_record(
    rows: Sequence[AnswerEvaluationRow],
    level: str,
    split: SplitName,
    arm: ArmName,
    model: str | None,
    domain: str | None,
    support: str | None,
) -> JsonObject:
    """Build one long-form accuracy record."""

    if not rows:
        raise DownstreamDataError(
            "Cannot compute answer metrics on empty support: "
            f"level={level}, split={split}, arm={arm}, "
            f"model={model}, domain={domain}, support={support}"
        )
    correct: int = sum(row["correct"] for row in rows)
    categories: dict[str, int] = {}
    for row in rows:
        category: str = row["failure_category"]
        categories[category] = categories.get(category, 0) + 1
    return {
        "schema_version": "k2-answer-metrics-long-v1",
        "level": level,
        "split": split,
        "model": model,
        "domain": domain,
        "arm": arm,
        "support": support,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "failure_categories": categories,
    }


def rows_for(
    rows: Sequence[AnswerEvaluationRow],
    models: Sequence[str],
    arm: ArmName,
    split: SplitName,
    domain: str | None,
) -> list[AnswerEvaluationRow]:
    """Filter rows to an explicit support."""

    model_set: set[str] = set(models)
    selected: list[AnswerEvaluationRow] = [
        row
        for row in rows
        if row["model"] in model_set
        and row["arm"] == arm
        and (domain is None or row["domain"] == domain)
    ]
    return split_rows(selected, split)


def expected_arms(
    model: str,
    five_models: Sequence[str],
    fixed_model: str,
) -> tuple[ArmName, ...]:
    """Return the exact active K=2 arm set for one model."""

    arms: list[ArmName] = ["routed_always", "routed_gated"]
    if model in set(five_models):
        arms.append("routed_select")
    if model == fixed_model:
        arms.append("fixed_gated")
    return tuple(arms)


def verify_coverage(
    rows: Sequence[AnswerEvaluationRow],
    seven_models: Sequence[str],
    five_models: Sequence[str],
    fixed_model: str,
    expected_total_per_model: int,
    expected_heldout_per_model: int,
) -> None:
    """Verify exact arm, domain, and held-out denominators."""

    allowed_models: set[str] = set(seven_models)
    unexpected_models: list[str] = sorted(
        {row["model"] for row in rows} - allowed_models
    )
    if unexpected_models:
        raise DownstreamDataError(
            f"Unexpected models in answer evaluations: models={unexpected_models}"
        )
    for model in seven_models:
        for arm in expected_arms(model, five_models, fixed_model):
            model_rows: list[AnswerEvaluationRow] = rows_for(
                rows,
                (model,),
                arm,
                "full",
                None,
            )
            heldout_rows: list[AnswerEvaluationRow] = split_rows(
                model_rows,
                "heldout",
            )
            if len(model_rows) != expected_total_per_model:
                raise DownstreamDataError(
                    "Answer model denominator mismatch: "
                    f"model={model}, arm={arm}, "
                    f"expected={expected_total_per_model}, "
                    f"actual={len(model_rows)}"
                )
            if len(heldout_rows) != expected_heldout_per_model:
                raise DownstreamDataError(
                    "Answer held-out denominator mismatch: "
                    f"model={model}, arm={arm}, "
                    f"expected={expected_heldout_per_model}, "
                    f"actual={len(heldout_rows)}"
                )
            for domain in DOMAINS:
                domain_rows: list[AnswerEvaluationRow] = [
                    row for row in model_rows if row["domain"] == domain
                ]
                expected_domain_count: int = DOMAIN_COUNTS[domain]
                if len(domain_rows) != expected_domain_count:
                    raise DownstreamDataError(
                        "Answer domain denominator mismatch: "
                        f"model={model}, arm={arm}, domain={domain}, "
                        f"expected={expected_domain_count}, "
                        f"actual={len(domain_rows)}"
                    )
                validation_count: int = sum(
                    row["is_validation"] for row in domain_rows
                )
                expected_validation_count: int = DOMAIN_VALIDATION_COUNTS[
                    domain
                ]
                if validation_count != expected_validation_count:
                    raise DownstreamDataError(
                        "Answer validation denominator mismatch: "
                        f"model={model}, arm={arm}, domain={domain}, "
                        f"expected={expected_validation_count}, "
                        f"actual={validation_count}"
                    )
    for model in seven_models:
        allowed_arms: set[ArmName] = set(
            expected_arms(model, five_models, fixed_model)
        )
        extra_arms: list[str] = sorted(
            {
                row["arm"]
                for row in rows
                if row["model"] == model and row["arm"] not in allowed_arms
            }
        )
        if extra_arms:
            raise DownstreamDataError(
                f"Unexpected answer arms: model={model}, arms={extra_arms}"
            )


def build_long_metrics(
    rows: Sequence[AnswerEvaluationRow],
    seven_models: Sequence[str],
    five_models: Sequence[str],
    fixed_model: str,
) -> list[JsonObject]:
    """Build domain, model, and strict-support fleet metrics."""

    records: list[JsonObject] = []
    for split in ("full", "heldout"):
        split_name: SplitName = cast(SplitName, split)
        for model in seven_models:
            for arm in expected_arms(model, five_models, fixed_model):
                for domain in DOMAINS:
                    records.append(
                        metric_record(
                            rows_for(
                                rows,
                                (model,),
                                arm,
                                split_name,
                                domain,
                            ),
                            "model_domain",
                            split_name,
                            arm,
                            model,
                            domain,
                            None,
                        )
                    )
                records.append(
                    metric_record(
                        rows_for(
                            rows,
                            (model,),
                            arm,
                            split_name,
                            None,
                        ),
                        "model_pooled",
                        split_name,
                        arm,
                        model,
                        None,
                        None,
                    )
                )
        for support_name, models, arms in (
            (
                "seven_model",
                seven_models,
                ("routed_always", "routed_gated"),
            ),
            (
                "five_model",
                five_models,
                ("routed_always", "routed_gated", "routed_select"),
            ),
        ):
            for raw_arm in arms:
                arm: ArmName = cast(ArmName, raw_arm)
                model_accuracies: list[float] = []
                total_n: int = 0
                total_correct: int = 0
                categories: dict[str, int] = {}
                for model in models:
                    selected: list[AnswerEvaluationRow] = rows_for(
                        rows,
                        (model,),
                        arm,
                        split_name,
                        None,
                    )
                    correct: int = sum(row["correct"] for row in selected)
                    model_accuracies.append(correct / len(selected))
                    total_n += len(selected)
                    total_correct += correct
                    for row in selected:
                        category: str = row["failure_category"]
                        categories[category] = (
                            categories.get(category, 0) + 1
                        )
                records.append(
                    {
                        "schema_version": "k2-answer-metrics-long-v1",
                        "level": "fleet_model_macro",
                        "split": split_name,
                        "model": None,
                        "domain": None,
                        "arm": arm,
                        "support": support_name,
                        "n": total_n,
                        "correct": total_correct,
                        "accuracy": sum(model_accuracies)
                        / len(model_accuracies),
                        "fleet_micro_accuracy": total_correct / total_n,
                        "failure_categories": categories,
                    }
                )
    return records


def comparison_seed(base_seed: int, key: str) -> int:
    """Derive one order-independent NumPy seed."""

    digest: bytes = hashlib.sha256(key.encode()).digest()
    return (int.from_bytes(digest[:8], byteorder="big") ^ base_seed) % (
        2**63
    )


def paired_values(
    rows: Sequence[AnswerEvaluationRow],
    model: str,
    domain: str | None,
    arm_a: ArmName,
    arm_b: ArmName,
) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out correctness arrays in identical instance order."""

    selected: list[AnswerEvaluationRow] = [
        row
        for row in rows
        if row["model"] == model
        and not row["is_validation"]
        and (domain is None or row["domain"] == domain)
        and row["arm"] in (arm_a, arm_b)
    ]
    by_arm: dict[ArmName, dict[tuple[str, str], float]] = {
        arm_a: {},
        arm_b: {},
    }
    for row in selected:
        key: tuple[str, str] = (row["domain"], row["instance_id"])
        by_arm[row["arm"]][key] = float(row["correct"])
    keys_a: set[tuple[str, str]] = set(by_arm[arm_a])
    keys_b: set[tuple[str, str]] = set(by_arm[arm_b])
    if keys_a != keys_b or not keys_a:
        raise DownstreamDataError(
            "Paired answer support mismatch: "
            f"model={model}, domain={domain}, arm_a={arm_a}, arm_b={arm_b}, "
            f"missing_a={sorted(keys_b - keys_a)[:10]}, "
            f"missing_b={sorted(keys_a - keys_b)[:10]}"
        )
    ordered_keys: list[tuple[str, str]] = sorted(keys_a)
    values_a: np.ndarray = np.asarray(
        [by_arm[arm_a][key] for key in ordered_keys],
        dtype=np.float64,
    )
    values_b: np.ndarray = np.asarray(
        [by_arm[arm_b][key] for key in ordered_keys],
        dtype=np.float64,
    )
    return values_a, values_b


def instance_bootstrap_distribution(
    differences: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> np.ndarray:
    """Return paired instance-bootstrap means for one model-domain cell."""

    if differences.ndim != 1 or differences.size == 0:
        raise ValueError(
            "Instance-bootstrap differences must be non-empty: "
            f"shape={differences.shape}"
        )
    rng: np.random.Generator = np.random.default_rng(seed)
    output: np.ndarray = np.empty(bootstrap_samples, dtype=np.float64)
    sample_count: int = int(differences.size)
    chunk_size: int = max(1, min(256, 4_000_000 // sample_count))
    offset: int = 0
    while offset < bootstrap_samples:
        current: int = min(chunk_size, bootstrap_samples - offset)
        indices: np.ndarray = rng.integers(
            0,
            sample_count,
            size=(current, sample_count),
            endpoint=False,
        )
        output[offset : offset + current] = differences[indices].mean(axis=1)
        offset += current
    return output


def hierarchical_bootstrap(
    rows: Sequence[AnswerEvaluationRow],
    models: Sequence[str],
    arm_a: ArmName,
    arm_b: ArmName,
    bootstrap_samples: int,
    seed: int,
    comparison_name: str,
) -> JsonObject:
    """Resample models, domains, then paired instances within each cell."""

    model_count: int = len(models)
    domain_count: int = len(DOMAINS)
    cell_distributions: np.ndarray = np.empty(
        (model_count, domain_count, bootstrap_samples),
        dtype=np.float64,
    )
    means_a: np.ndarray = np.empty(
        (model_count, domain_count),
        dtype=np.float64,
    )
    means_b: np.ndarray = np.empty_like(means_a)
    total_instances: int = 0
    for model_index, model in enumerate(models):
        for domain_index, domain in enumerate(DOMAINS):
            values_a, values_b = paired_values(
                rows,
                model,
                domain,
                arm_a,
                arm_b,
            )
            differences: np.ndarray = values_a - values_b
            means_a[model_index, domain_index] = values_a.mean()
            means_b[model_index, domain_index] = values_b.mean()
            total_instances += int(values_a.size)
            cell_distributions[model_index, domain_index, :] = (
                instance_bootstrap_distribution(
                    differences,
                    bootstrap_samples,
                    comparison_seed(
                        seed,
                        f"{comparison_name}:{model}:{domain}:instances",
                    ),
                )
            )

    hierarchy_rng: np.random.Generator = np.random.default_rng(
        comparison_seed(seed, f"{comparison_name}:hierarchy")
    )
    model_indices: np.ndarray = hierarchy_rng.integers(
        0,
        model_count,
        size=(bootstrap_samples, model_count),
        endpoint=False,
    )
    domain_indices: np.ndarray = hierarchy_rng.integers(
        0,
        domain_count,
        size=(bootstrap_samples, model_count, domain_count),
        endpoint=False,
    )
    sampled_models: np.ndarray = np.broadcast_to(
        model_indices[:, :, None],
        domain_indices.shape,
    )
    bootstrap_indices: np.ndarray = np.broadcast_to(
        np.arange(bootstrap_samples)[:, None, None],
        domain_indices.shape,
    )
    distribution: np.ndarray = cell_distributions[
        sampled_models,
        domain_indices,
        bootstrap_indices,
    ].mean(axis=(1, 2))
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    probability_nonpositive: float = float(np.mean(distribution <= 0.0))
    probability_nonnegative: float = float(np.mean(distribution >= 0.0))
    p_value: float = min(
        1.0,
        2.0 * min(probability_nonpositive, probability_nonnegative),
    )
    return {
        "schema_version": "k2-answer-hierarchical-bootstrap-v1",
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_models": model_count,
        "n_domains_per_model": domain_count,
        "n_unique_instances": total_instances,
        "mean_a": float(means_a.mean()),
        "mean_b": float(means_b.mean()),
        "difference_a_minus_b": float((means_a - means_b).mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": p_value,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "resampling_unit": "models, then domains, then paired instances",
    }


def paired_report(
    rows: Sequence[AnswerEvaluationRow],
    comparison: ComparisonSpec,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> JsonObject:
    """Build per-domain, pooled, and fleet held-out paired statistics."""

    by_model: JsonObject = {}
    for model in comparison["models"]:
        by_domain: JsonObject = {}
        for domain in DOMAINS:
            values_a, values_b = paired_values(
                rows,
                model,
                domain,
                comparison["arm_a"],
                comparison["arm_b"],
            )
            by_domain[domain] = cast(
                JsonValue,
                bootstrap_stats(
                    values_a,
                    values_b,
                    bootstrap_samples,
                    comparison_seed(
                        bootstrap_seed,
                        f"{comparison['name']}:{model}:{domain}",
                    ),
                ),
            )
        pooled_a, pooled_b = paired_values(
            rows,
            model,
            None,
            comparison["arm_a"],
            comparison["arm_b"],
        )
        by_model[model] = {
            "by_domain": by_domain,
            "pooled_instances": cast(
                JsonValue,
                bootstrap_stats(
                    pooled_a,
                    pooled_b,
                    bootstrap_samples,
                    comparison_seed(
                        bootstrap_seed,
                        f"{comparison['name']}:{model}:pooled",
                    ),
                ),
            ),
        }
    output: JsonObject = {
        "schema_version": "k2-answer-paired-comparison-v1",
        "split": "heldout",
        "arm_a": comparison["arm_a"],
        "arm_b": comparison["arm_b"],
        "models": list(comparison["models"]),
        "by_model": by_model,
    }
    if len(comparison["models"]) > 1:
        output["fleet_hierarchical"] = hierarchical_bootstrap(
            rows,
            comparison["models"],
            comparison["arm_a"],
            comparison["arm_b"],
            bootstrap_samples,
            bootstrap_seed,
            comparison["name"],
        )
    return output


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Write one JSON object atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[JsonObject]) -> None:
    """Write canonical JSONL atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        for row in rows:
            output_file.write(canonical_json(row))
            output_file.write("\n")
    temporary_path.replace(path)


def summary_from_long(rows: Sequence[JsonObject]) -> JsonObject:
    """Build a compact nested summary from long-form metrics."""

    summary: JsonObject = {
        "schema_version": "k2-answer-summary-v1",
        "model_pooled": {},
        "fleet_model_macro": {},
    }
    model_block: JsonObject = cast(JsonObject, summary["model_pooled"])
    fleet_block: JsonObject = cast(JsonObject, summary["fleet_model_macro"])
    for row in rows:
        level: str = require_string(row.get("level"), "metric.level")
        split: str = require_string(row.get("split"), "metric.split")
        arm: str = require_string(row.get("arm"), "metric.arm")
        if level == "model_pooled":
            model: str = require_string(row.get("model"), "metric.model")
            raw_model: JsonValue | None = model_block.get(model)
            current_model: JsonObject
            if raw_model is None:
                current_model = {}
                model_block[model] = current_model
            else:
                current_model = require_object(
                    raw_model,
                    f"summary.model_pooled.{model}",
                )
            raw_split: JsonValue | None = current_model.get(split)
            current_split: JsonObject
            if raw_split is None:
                current_split = {}
                current_model[split] = current_split
            else:
                current_split = require_object(
                    raw_split,
                    f"summary.model_pooled.{model}.{split}",
                )
            current_split[arm] = row
        elif level == "fleet_model_macro":
            support: str = require_string(
                row.get("support"),
                "metric.support",
            )
            raw_support: JsonValue | None = fleet_block.get(support)
            current_support: JsonObject
            if raw_support is None:
                current_support = {}
                fleet_block[support] = current_support
            else:
                current_support = require_object(
                    raw_support,
                    f"summary.fleet_model_macro.{support}",
                )
            raw_split = current_support.get(split)
            if raw_split is None:
                current_split = {}
                current_support[split] = current_split
            else:
                current_split = require_object(
                    raw_split,
                    f"summary.fleet_model_macro.{support}.{split}",
                )
            current_split[arm] = row
    return summary


def main() -> None:
    """Validate coverage, summarize accuracy, and run paired bootstraps."""

    args = parse_args()
    seven_models: tuple[str, ...] = parse_model_list(
        str(args.seven_models),
        "seven-models",
    )
    five_models: tuple[str, ...] = parse_model_list(
        str(args.five_models),
        "five-models",
    )
    fixed_model: str = str(args.fixed_model)
    if len(seven_models) != 7:
        raise ValueError(
            f"seven-models must contain exactly 7 models: "
            f"actual={len(seven_models)}"
        )
    if len(five_models) != 5:
        raise ValueError(
            f"five-models must contain exactly 5 models: "
            f"actual={len(five_models)}"
        )
    if not set(five_models).issubset(set(seven_models)):
        raise ValueError(
            "five-models must be a subset of seven-models: "
            f"five={five_models}, seven={seven_models}"
        )
    if fixed_model not in set(five_models):
        raise ValueError(
            f"fixed-model must belong to five-models: model={fixed_model}"
        )
    bootstrap_samples: int = int(args.bootstrap_samples)
    bootstrap_seed: int = int(args.bootstrap_seed)
    if bootstrap_samples <= 0:
        raise ValueError(
            f"bootstrap-samples must be positive: value={bootstrap_samples}"
        )
    expected_total_per_model: int = int(args.expected_total_per_model)
    expected_heldout_per_model: int = int(args.expected_heldout_per_model)
    input_paths: list[Path] = [
        cast(Path, path).resolve() for path in cast(list[Path], args.inputs)
    ]
    rows: list[AnswerEvaluationRow] = load_rows(input_paths)
    verify_coverage(
        rows,
        seven_models,
        five_models,
        fixed_model,
        expected_total_per_model,
        expected_heldout_per_model,
    )
    long_metrics: list[JsonObject] = build_long_metrics(
        rows,
        seven_models,
        five_models,
        fixed_model,
    )
    comparison_specs: tuple[ComparisonSpec, ...] = (
        {
            "name": "gated_vs_always_seven_model",
            "arm_a": "routed_gated",
            "arm_b": "routed_always",
            "models": seven_models,
        },
        {
            "name": "gated_vs_select_five_model",
            "arm_a": "routed_gated",
            "arm_b": "routed_select",
            "models": five_models,
        },
        {
            "name": "select_vs_always_five_model",
            "arm_a": "routed_select",
            "arm_b": "routed_always",
            "models": five_models,
        },
        {
            "name": "routed_gated_vs_fixed_gated",
            "arm_a": "routed_gated",
            "arm_b": "fixed_gated",
            "models": (fixed_model,),
        },
    )
    comparisons: JsonObject = {
        "schema_version": "k2-answer-paired-comparisons-v1",
        "split": "heldout",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "comparisons": {
            specification["name"]: paired_report(
                rows,
                specification,
                bootstrap_samples,
                bootstrap_seed,
            )
            for specification in comparison_specs
        },
        "notes": (
            "Full support is descriptive. All paired inference excludes "
            "the frozen calibration IDs. Fleet intervals resample models, "
            "then domains, then paired instances."
        ),
    }
    input_manifest: list[JsonObject] = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for path in sorted(input_paths)
    ]
    summary: JsonObject = summary_from_long(long_metrics)
    summary["seven_models"] = list(seven_models)
    summary["five_models"] = list(five_models)
    summary["fixed_model"] = fixed_model
    summary["expected_total_per_model"] = expected_total_per_model
    summary["expected_heldout_per_model"] = expected_heldout_per_model
    summary["inputs"] = input_manifest

    output_long: Path = cast(Path, args.output_long).resolve()
    output_summary: Path = cast(Path, args.output_summary).resolve()
    output_comparisons: Path = cast(Path, args.output_comparisons).resolve()
    write_jsonl_atomic(output_long, long_metrics)
    write_json_atomic(output_summary, summary)
    write_json_atomic(output_comparisons, comparisons)
    print(
        canonical_json(
            {
                "event": "k2_answer_summary_complete",
                "evaluation_inputs": len(input_paths),
                "evaluation_rows": len(rows),
                "metric_rows": len(long_metrics),
                "comparisons": len(comparison_specs),
                "output_long": str(output_long),
                "output_long_sha256": sha256_file(output_long),
                "output_summary": str(output_summary),
                "output_summary_sha256": sha256_file(output_summary),
                "output_comparisons": str(output_comparisons),
                "output_comparisons_sha256": sha256_file(
                    output_comparisons
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
