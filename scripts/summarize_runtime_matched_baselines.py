#!/usr/bin/env python3
"""Summarize fresh runtime-matched baselines and compare them with K=2."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

import numpy as np

from hyskill.runtime_matched_execution import (
    canonical_json,
    sha256_file,
)
from scripts.evaluate_runtime_matched_baselines import (
    EVALUATION_ROW_SCHEMA_VERSION,
    EVALUATION_SCHEMA_VERSION,
)


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
SplitName: TypeAlias = Literal["full", "heldout"]

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
K2_PUBLIC_ROW_SCHEMA_VERSION: str = "k2-public-answer-row-v1"
SUMMARY_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-summary-v1"
)
LONG_METRIC_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-metric-v1"
)
COMPARISONS_SCHEMA_VERSION: str = (
    "runtime-matched-baseline-paired-comparisons-v1"
)
BASELINE_ARMS: tuple[str, ...] = (
    "bare",
    "always_rerank",
    "select_bm25",
)
K2_ARMS: tuple[str, ...] = (
    "routed_gated",
    "routed_select",
)


class BaselineSummaryError(ValueError):
    """Raised when comparison evidence violates the frozen support."""


class ComparisonRow(TypedDict):
    """One per-instance correctness observation."""

    model: str
    domain: str
    arm: str
    instance_id: str
    correct: bool
    failure_category: str
    is_validation: bool
    runtime_manifest_sha256: str | None


class ComparisonSpec(TypedDict):
    """One preregistered paired contrast and its strict model support."""

    name: str
    arm_a: str
    arm_b: str
    models: tuple[str, ...]


RowKey: TypeAlias = tuple[str, str, str, str]
PairKey: TypeAlias = tuple[str, str]


def parse_args() -> argparse.Namespace:
    """Parse explicit fresh baseline and public K=2 evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--k2-answers", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--baseline-evals",
        required=True,
        nargs="+",
        type=Path,
    )
    parser.add_argument("--seven-models", required=True)
    parser.add_argument("--five-models", required=True)
    parser.add_argument("--expected-k2-files", required=True, type=int)
    parser.add_argument(
        "--expected-baseline-eval-files",
        required=True,
        type=int,
    )
    parser.add_argument("--expected-total-per-model", required=True, type=int)
    parser.add_argument(
        "--expected-heldout-per-model",
        required=True,
        type=int,
    )
    parser.add_argument("--bootstrap-samples", required=True, type=int)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    parser.add_argument("--output-long", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-comparisons", required=True, type=Path)
    return parser.parse_args()


def require_object(value: JsonValue | None, context: str) -> JsonObject:
    """Return one JSON object with source context."""

    if not isinstance(value, dict):
        raise BaselineSummaryError(
            "Expected JSON object: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_list(value: JsonValue | None, context: str) -> list[JsonValue]:
    """Return one JSON list with source context."""

    if not isinstance(value, list):
        raise BaselineSummaryError(
            "Expected JSON list: "
            f"context={context}, value_type={type(value).__name__}"
        )
    return value


def require_string(value: JsonValue | None, context: str) -> str:
    """Return one non-empty string with source context."""

    if not isinstance(value, str) or not value:
        raise BaselineSummaryError(
            f"Expected non-empty string: context={context}, value={value!r}"
        )
    return value


def require_boolean(value: JsonValue | None, context: str) -> bool:
    """Return one Boolean with source context."""

    if not isinstance(value, bool):
        raise BaselineSummaryError(
            f"Expected Boolean: context={context}, value={value!r}"
        )
    return value


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


def resolve_exact_paths(
    paths: Sequence[Path],
    expected_count: int,
    context: str,
) -> list[Path]:
    """Resolve one exact, duplicate-free file inventory."""

    if expected_count <= 0:
        raise ValueError(
            f"Expected file count must be positive: context={context}, "
            f"value={expected_count}"
        )
    resolved: list[Path] = [path.resolve() for path in paths]
    if len(resolved) != expected_count:
        raise BaselineSummaryError(
            f"{context} file count mismatch: "
            f"expected={expected_count}, actual={len(resolved)}"
        )
    if len(resolved) != len(set(resolved)):
        raise BaselineSummaryError(
            f"{context} inputs contain duplicate paths"
        )
    missing: list[str] = [
        str(path) for path in resolved if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{context} inputs do not exist: sample={missing[:20]}"
        )
    return resolved


def load_json(path: Path, context: str) -> JsonObject:
    """Load one UTF-8 JSON object."""

    try:
        raw_value: JsonValue = cast(
            JsonValue,
            json.loads(path.read_text(encoding="utf-8")),
        )
    except json.JSONDecodeError as error:
        raise BaselineSummaryError(
            f"{context} JSON is malformed: path={path}, "
            f"line={error.lineno}, column={error.colno}, message={error.msg}"
        ) from error
    return require_object(raw_value, f"{context}:{path}")


def load_jsonl(path: Path, context: str) -> list[JsonObject]:
    """Load regular or gzip JSONL based only on the explicit suffix."""

    rows: list[JsonObject] = []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    raw_row: JsonValue = cast(
                        JsonValue,
                        json.loads(line),
                    )
                except json.JSONDecodeError as error:
                    raise BaselineSummaryError(
                        f"{context} JSONL is malformed: path={path}, "
                        f"line={line_number}, column={error.colno}, "
                        f"message={error.msg}"
                    ) from error
                rows.append(
                    require_object(
                        raw_row,
                        f"{context}:{path}:{line_number}",
                    )
                )
    except gzip.BadGzipFile as error:
        raise BaselineSummaryError(
            f"{context} input is not valid gzip: path={path}"
        ) from error
    if not rows:
        raise BaselineSummaryError(
            f"{context} input is empty: path={path}"
        )
    return rows


def parse_common_row(
    raw_row: Mapping[str, JsonValue],
    context: str,
    allowed_arms: frozenset[str],
    runtime_manifest_required: bool,
) -> ComparisonRow:
    """Parse the common correctness fields from one evidence row."""

    domain: str = require_string(raw_row.get("domain"), f"{context}.domain")
    if domain not in DOMAIN_COUNTS:
        raise BaselineSummaryError(
            f"Unknown domain: context={context}, domain={domain}"
        )
    arm: str = require_string(raw_row.get("arm"), f"{context}.arm")
    if arm not in allowed_arms:
        raise BaselineSummaryError(
            f"Unexpected arm: context={context}, arm={arm}, "
            f"allowed={sorted(allowed_arms)}"
        )
    runtime_manifest_value: JsonValue | None = raw_row.get(
        "runtime_manifest_sha256"
    )
    runtime_manifest_sha256: str | None = None
    if runtime_manifest_required:
        runtime_manifest_sha256 = require_string(
            runtime_manifest_value,
            f"{context}.runtime_manifest_sha256",
        )
    elif runtime_manifest_value is not None:
        runtime_manifest_sha256 = require_string(
            runtime_manifest_value,
            f"{context}.runtime_manifest_sha256",
        )
    return {
        "model": require_string(raw_row.get("model"), f"{context}.model"),
        "domain": domain,
        "arm": arm,
        "instance_id": require_string(
            raw_row.get("instance_id"),
            f"{context}.instance_id",
        ),
        "correct": require_boolean(
            raw_row.get("correct"),
            f"{context}.correct",
        ),
        "failure_category": require_string(
            raw_row.get("failure_category"),
            f"{context}.failure_category",
        ),
        "is_validation": require_boolean(
            raw_row.get("is_validation"),
            f"{context}.is_validation",
        ),
        "runtime_manifest_sha256": runtime_manifest_sha256,
    }


def load_k2_rows(paths: Sequence[Path]) -> list[ComparisonRow]:
    """Load only current public K=2 Gated and Hy+Select evidence."""

    rows: list[ComparisonRow] = []
    seen_keys: set[RowKey] = set()
    for path in paths:
        for row_index, raw_row in enumerate(
            load_jsonl(path, "K=2 public answers")
        ):
            if raw_row.get("schema_version") != K2_PUBLIC_ROW_SCHEMA_VERSION:
                raise BaselineSummaryError(
                    "Unexpected K=2 public row schema: "
                    f"path={path}, row={row_index}, "
                    f"schema={raw_row.get('schema_version')!r}"
                )
            raw_arm: JsonValue | None = raw_row.get("arm")
            if raw_arm not in K2_ARMS:
                continue
            row: ComparisonRow = parse_common_row(
                raw_row,
                f"K=2:{path}:{row_index}",
                frozenset(K2_ARMS),
                False,
            )
            key: RowKey = (
                row["model"],
                row["domain"],
                row["arm"],
                row["instance_id"],
            )
            if key in seen_keys:
                raise BaselineSummaryError(
                    f"Duplicate K=2 comparison row: key={key}"
                )
            seen_keys.add(key)
            rows.append(row)
    if not rows:
        raise BaselineSummaryError("No K=2 comparison rows were loaded")
    return rows


def load_baseline_rows(paths: Sequence[Path]) -> list[ComparisonRow]:
    """Load only fresh runtime-matched per-instance evaluations."""

    rows: list[ComparisonRow] = []
    seen_keys: set[RowKey] = set()
    for path in paths:
        payload: JsonObject = load_json(path, "baseline evaluation")
        if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise BaselineSummaryError(
                "Unexpected baseline evaluation schema: "
                f"path={path}, schema={payload.get('schema_version')!r}"
            )
        provenance: JsonObject = require_object(
            payload.get("provenance"),
            f"baseline:{path}.provenance",
        )
        if provenance.get("legacy_compact_baseline_read") is not False:
            raise BaselineSummaryError(
                "Baseline evaluation does not explicitly reject legacy "
                f"compact evidence: path={path}"
            )
        details: list[JsonValue] = require_list(
            payload.get("details"),
            f"baseline:{path}.details",
        )
        if not details:
            raise BaselineSummaryError(
                f"Baseline evaluation contains no details: path={path}"
            )
        for row_index, raw_value in enumerate(details):
            raw_row: JsonObject = require_object(
                raw_value,
                f"baseline:{path}.details[{row_index}]",
            )
            if (
                raw_row.get("schema_version")
                != EVALUATION_ROW_SCHEMA_VERSION
            ):
                raise BaselineSummaryError(
                    "Unexpected fresh baseline row schema: "
                    f"path={path}, row={row_index}, "
                    f"schema={raw_row.get('schema_version')!r}"
                )
            row: ComparisonRow = parse_common_row(
                raw_row,
                f"baseline:{path}:{row_index}",
                frozenset(BASELINE_ARMS),
                True,
            )
            key: RowKey = (
                row["model"],
                row["domain"],
                row["arm"],
                row["instance_id"],
            )
            if key in seen_keys:
                raise BaselineSummaryError(
                    f"Duplicate fresh baseline row: key={key}"
                )
            seen_keys.add(key)
            rows.append(row)
    if not rows:
        raise BaselineSummaryError(
            "No fresh runtime-matched baseline rows were loaded"
        )
    return rows


def expected_baseline_arms(
    model: str,
    five_models: Sequence[str],
) -> tuple[str, ...]:
    """Return the exact fresh baseline support for one model."""

    arms: list[str] = ["bare"]
    if model in set(five_models):
        arms.extend(("always_rerank", "select_bm25"))
    return tuple(arms)


def expected_k2_arms(
    model: str,
    five_models: Sequence[str],
) -> tuple[str, ...]:
    """Return the exact K=2 support needed by the comparisons."""

    arms: list[str] = ["routed_gated"]
    if model in set(five_models):
        arms.append("routed_select")
    return tuple(arms)


def rows_for(
    rows: Sequence[ComparisonRow],
    model: str,
    arm: str,
    domain: str | None,
) -> list[ComparisonRow]:
    """Filter rows to one explicit model, arm, and optional domain."""

    return [
        row
        for row in rows
        if row["model"] == model
        and row["arm"] == arm
        and (domain is None or row["domain"] == domain)
    ]


def verify_one_arm(
    rows: Sequence[ComparisonRow],
    model: str,
    arm: str,
    expected_total_per_model: int,
    expected_heldout_per_model: int,
) -> None:
    """Verify exact total, held-out, domain, and split denominators."""

    selected: list[ComparisonRow] = rows_for(rows, model, arm, None)
    heldout: list[ComparisonRow] = [
        row for row in selected if not row["is_validation"]
    ]
    if len(selected) != expected_total_per_model:
        raise BaselineSummaryError(
            "Model-arm denominator mismatch: "
            f"model={model}, arm={arm}, expected={expected_total_per_model}, "
            f"actual={len(selected)}"
        )
    if len(heldout) != expected_heldout_per_model:
        raise BaselineSummaryError(
            "Model-arm held-out denominator mismatch: "
            f"model={model}, arm={arm}, "
            f"expected={expected_heldout_per_model}, actual={len(heldout)}"
        )
    for domain in DOMAINS:
        domain_rows: list[ComparisonRow] = rows_for(
            selected,
            model,
            arm,
            domain,
        )
        expected_domain_count: int = DOMAIN_COUNTS[domain]
        if len(domain_rows) != expected_domain_count:
            raise BaselineSummaryError(
                "Model-arm-domain denominator mismatch: "
                f"model={model}, arm={arm}, domain={domain}, "
                f"expected={expected_domain_count}, actual={len(domain_rows)}"
            )
        validation_count: int = sum(
            row["is_validation"] for row in domain_rows
        )
        expected_validation: int = DOMAIN_VALIDATION_COUNTS[domain]
        if validation_count != expected_validation:
            raise BaselineSummaryError(
                "Model-arm-domain validation denominator mismatch: "
                f"model={model}, arm={arm}, domain={domain}, "
                f"expected={expected_validation}, actual={validation_count}"
            )


def verify_coverage(
    k2_rows: Sequence[ComparisonRow],
    baseline_rows: Sequence[ComparisonRow],
    seven_models: Sequence[str],
    five_models: Sequence[str],
    expected_total_per_model: int,
    expected_heldout_per_model: int,
) -> None:
    """Verify exact fresh and K=2 supports and identical split flags."""

    allowed_models: set[str] = set(seven_models)
    for context, rows in (
        ("K=2", k2_rows),
        ("baseline", baseline_rows),
    ):
        unexpected_models: list[str] = sorted(
            {row["model"] for row in rows} - allowed_models
        )
        if unexpected_models:
            raise BaselineSummaryError(
                f"{context} evidence has unexpected models: "
                f"models={unexpected_models}"
            )
    for model in seven_models:
        for arm in expected_k2_arms(model, five_models):
            verify_one_arm(
                k2_rows,
                model,
                arm,
                expected_total_per_model,
                expected_heldout_per_model,
            )
        for arm in expected_baseline_arms(model, five_models):
            verify_one_arm(
                baseline_rows,
                model,
                arm,
                expected_total_per_model,
                expected_heldout_per_model,
            )
    k2_split_reference: dict[tuple[str, str, str], bool] = {}
    for row in k2_rows:
        if row["arm"] != "routed_gated":
            continue
        key: tuple[str, str, str] = (
            row["model"],
            row["domain"],
            row["instance_id"],
        )
        k2_split_reference[key] = row["is_validation"]
    for row in baseline_rows:
        key = (row["model"], row["domain"], row["instance_id"])
        expected_split: bool | None = k2_split_reference.get(key)
        if expected_split is None:
            raise BaselineSummaryError(
                "Fresh baseline row is outside K=2 Gated support: "
                f"key={key}, arm={row['arm']}"
            )
        if row["is_validation"] != expected_split:
            raise BaselineSummaryError(
                "Fresh baseline split flag differs from K=2 authority: "
                f"key={key}, arm={row['arm']}, "
                f"baseline={row['is_validation']}, k2={expected_split}"
            )
    allowed_k2: dict[str, set[str]] = {
        model: set(expected_k2_arms(model, five_models))
        for model in seven_models
    }
    allowed_baseline: dict[str, set[str]] = {
        model: set(expected_baseline_arms(model, five_models))
        for model in seven_models
    }
    for context, rows, allowed in (
        ("K=2", k2_rows, allowed_k2),
        ("baseline", baseline_rows, allowed_baseline),
    ):
        extras: list[RowKey] = [
            (
                row["model"],
                row["domain"],
                row["arm"],
                row["instance_id"],
            )
            for row in rows
            if row["arm"] not in allowed[row["model"]]
        ]
        if extras:
            raise BaselineSummaryError(
                f"{context} evidence has unexpected arms: sample={extras[:20]}"
            )


def metric_record(
    rows: Sequence[ComparisonRow],
    level: str,
    split: SplitName,
    arm: str,
    model: str | None,
    domain: str | None,
    support: str | None,
) -> JsonObject:
    """Build one long-form fresh baseline accuracy record."""

    selected: list[ComparisonRow] = (
        list(rows)
        if split == "full"
        else [row for row in rows if not row["is_validation"]]
    )
    if not selected:
        raise BaselineSummaryError(
            "Cannot summarize empty support: "
            f"level={level}, split={split}, arm={arm}, "
            f"model={model}, domain={domain}, support={support}"
        )
    correct: int = sum(row["correct"] for row in selected)
    categories: dict[str, int] = {}
    for row in selected:
        category: str = row["failure_category"]
        categories[category] = categories.get(category, 0) + 1
    return {
        "schema_version": LONG_METRIC_SCHEMA_VERSION,
        "level": level,
        "split": split,
        "model": model,
        "domain": domain,
        "arm": arm,
        "support": support,
        "n": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected),
        "failure_categories": categories,
    }


def build_long_metrics(
    rows: Sequence[ComparisonRow],
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> list[JsonObject]:
    """Build model-domain, model-pooled, and strict-support metrics."""

    records: list[JsonObject] = []
    for raw_split in ("full", "heldout"):
        split: SplitName = cast(SplitName, raw_split)
        for model in seven_models:
            for arm in expected_baseline_arms(model, five_models):
                for domain in DOMAINS:
                    records.append(
                        metric_record(
                            rows_for(rows, model, arm, domain),
                            "model_domain",
                            split,
                            arm,
                            model,
                            domain,
                            None,
                        )
                    )
                records.append(
                    metric_record(
                        rows_for(rows, model, arm, None),
                        "model_pooled",
                        split,
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
                ("bare",),
            ),
            (
                "five_model",
                five_models,
                BASELINE_ARMS,
            ),
        ):
            for arm in arms:
                support_rows: list[ComparisonRow] = [
                    row
                    for row in rows
                    if row["model"] in set(models) and row["arm"] == arm
                ]
                model_accuracies: list[float] = []
                for model in models:
                    model_rows: list[ComparisonRow] = [
                        row
                        for row in support_rows
                        if row["model"] == model
                        and (
                            split == "full"
                            or not row["is_validation"]
                        )
                    ]
                    model_accuracies.append(
                        sum(row["correct"] for row in model_rows)
                        / len(model_rows)
                    )
                record: JsonObject = metric_record(
                    support_rows,
                    "fleet_model_macro",
                    split,
                    arm,
                    None,
                    None,
                    support_name,
                )
                record["accuracy"] = (
                    sum(model_accuracies) / len(model_accuracies)
                )
                records.append(record)
    return records


def comparison_specs(
    seven_models: tuple[str, ...],
    five_models: tuple[str, ...],
) -> tuple[ComparisonSpec, ...]:
    """Return the four preregistered fresh baseline comparisons."""

    return (
        {
            "name": "gated_vs_bare_seven_model",
            "arm_a": "routed_gated",
            "arm_b": "bare",
            "models": seven_models,
        },
        {
            "name": "gated_vs_native_rerank_five_model",
            "arm_a": "routed_gated",
            "arm_b": "always_rerank",
            "models": five_models,
        },
        {
            "name": "gated_vs_bm25_select_five_model",
            "arm_a": "routed_gated",
            "arm_b": "select_bm25",
            "models": five_models,
        },
        {
            "name": "hyskill_select_vs_bm25_select_five_model",
            "arm_a": "routed_select",
            "arm_b": "select_bm25",
            "models": five_models,
        },
    )


def comparison_seed(base_seed: int, key: str) -> int:
    """Derive one order-independent NumPy seed."""

    digest: bytes = hashlib.sha256(key.encode()).digest()
    return (int.from_bytes(digest[:8], byteorder="big") ^ base_seed) % (
        2**63
    )


def paired_values(
    rows: Sequence[ComparisonRow],
    model: str,
    domain: str | None,
    arm_a: str,
    arm_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out correctness arrays in identical instance order."""

    by_arm: dict[str, dict[PairKey, float]] = {
        arm_a: {},
        arm_b: {},
    }
    for row in rows:
        if (
            row["model"] != model
            or row["is_validation"]
            or (domain is not None and row["domain"] != domain)
            or row["arm"] not in (arm_a, arm_b)
        ):
            continue
        key: PairKey = (row["domain"], row["instance_id"])
        if key in by_arm[row["arm"]]:
            raise BaselineSummaryError(
                "Duplicate row on paired support: "
                f"model={model}, arm={row['arm']}, key={key}"
            )
        by_arm[row["arm"]][key] = float(row["correct"])
    keys_a: set[PairKey] = set(by_arm[arm_a])
    keys_b: set[PairKey] = set(by_arm[arm_b])
    if keys_a != keys_b or not keys_a:
        raise BaselineSummaryError(
            "Paired support mismatch: "
            f"model={model}, domain={domain}, arm_a={arm_a}, arm_b={arm_b}, "
            f"missing_a={sorted(keys_b - keys_a)[:20]}, "
            f"missing_b={sorted(keys_a - keys_b)[:20]}"
        )
    ordered_keys: list[PairKey] = sorted(keys_a)
    return (
        np.asarray(
            [by_arm[arm_a][key] for key in ordered_keys],
            dtype=np.float64,
        ),
        np.asarray(
            [by_arm[arm_b][key] for key in ordered_keys],
            dtype=np.float64,
        ),
    )


def bootstrap_distribution(
    differences: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> np.ndarray:
    """Return paired instance-bootstrap means."""

    if differences.ndim != 1 or differences.size == 0:
        raise ValueError(
            "Bootstrap differences must be a non-empty vector: "
            f"shape={differences.shape}"
        )
    if bootstrap_samples <= 0:
        raise ValueError(
            "Bootstrap sample count must be positive: "
            f"value={bootstrap_samples}"
        )
    generator: np.random.Generator = np.random.default_rng(seed)
    sample_count: int = int(differences.size)
    chunk_size: int = max(1, min(256, 4_000_000 // sample_count))
    distribution: np.ndarray = np.empty(
        bootstrap_samples,
        dtype=np.float64,
    )
    offset: int = 0
    while offset < bootstrap_samples:
        current: int = min(chunk_size, bootstrap_samples - offset)
        indices: np.ndarray = generator.integers(
            0,
            sample_count,
            size=(current, sample_count),
            endpoint=False,
        )
        distribution[offset : offset + current] = differences[
            indices
        ].mean(axis=1)
        offset += current
    return distribution


def paired_stats(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> JsonObject:
    """Return one paired effect, interval, and two-sided bootstrap p-value."""

    if values_a.shape != values_b.shape:
        raise ValueError(
            "Paired arrays have different shapes: "
            f"a={values_a.shape}, b={values_b.shape}"
        )
    differences: np.ndarray = values_a - values_b
    distribution: np.ndarray = bootstrap_distribution(
        differences,
        bootstrap_samples,
        seed,
    )
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    probability_nonpositive: float = float(
        np.mean(distribution <= 0.0)
    )
    probability_nonnegative: float = float(
        np.mean(distribution >= 0.0)
    )
    return {
        "n": int(values_a.size),
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "difference_a_minus_b": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": min(
            1.0,
            2.0
            * min(
                probability_nonpositive,
                probability_nonnegative,
            ),
        ),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def hierarchical_bootstrap(
    rows: Sequence[ComparisonRow],
    models: Sequence[str],
    arm_a: str,
    arm_b: str,
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
            cell_distributions[
                model_index,
                domain_index,
                :,
            ] = bootstrap_distribution(
                differences,
                bootstrap_samples,
                comparison_seed(
                    seed,
                    f"{comparison_name}:{model}:{domain}:instances",
                ),
            )
    hierarchy_generator: np.random.Generator = np.random.default_rng(
        comparison_seed(seed, f"{comparison_name}:hierarchy")
    )
    model_indices: np.ndarray = hierarchy_generator.integers(
        0,
        model_count,
        size=(bootstrap_samples, model_count),
        endpoint=False,
    )
    domain_indices: np.ndarray = hierarchy_generator.integers(
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
    probability_nonpositive: float = float(
        np.mean(distribution <= 0.0)
    )
    probability_nonnegative: float = float(
        np.mean(distribution >= 0.0)
    )
    return {
        "schema_version": (
            "runtime-matched-baseline-hierarchical-bootstrap-v1"
        ),
        "arm_a": arm_a,
        "arm_b": arm_b,
        "n_models": model_count,
        "n_domains_per_model": domain_count,
        "n_unique_instances": total_instances,
        "mean_a": float(means_a.mean()),
        "mean_b": float(means_b.mean()),
        "difference_a_minus_b": float((means_a - means_b).mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": min(
            1.0,
            2.0
            * min(
                probability_nonpositive,
                probability_nonnegative,
            ),
        ),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "resampling_unit": (
            "models, then domains, then paired instances"
        ),
    }


def paired_report(
    rows: Sequence[ComparisonRow],
    specification: ComparisonSpec,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> JsonObject:
    """Build per-domain, pooled, and fleet held-out statistics."""

    by_model: JsonObject = {}
    for model in specification["models"]:
        by_domain: JsonObject = {}
        for domain in DOMAINS:
            values_a, values_b = paired_values(
                rows,
                model,
                domain,
                specification["arm_a"],
                specification["arm_b"],
            )
            by_domain[domain] = paired_stats(
                values_a,
                values_b,
                bootstrap_samples,
                comparison_seed(
                    bootstrap_seed,
                    f"{specification['name']}:{model}:{domain}",
                ),
            )
        pooled_a, pooled_b = paired_values(
            rows,
            model,
            None,
            specification["arm_a"],
            specification["arm_b"],
        )
        by_model[model] = {
            "by_domain": by_domain,
            "pooled_instances": paired_stats(
                pooled_a,
                pooled_b,
                bootstrap_samples,
                comparison_seed(
                    bootstrap_seed,
                    f"{specification['name']}:{model}:pooled",
                ),
            ),
        }
    return {
        "schema_version": (
            "runtime-matched-baseline-paired-comparison-v1"
        ),
        "split": "heldout",
        "arm_a": specification["arm_a"],
        "arm_b": specification["arm_b"],
        "models": list(specification["models"]),
        "by_model": by_model,
        "fleet_hierarchical": hierarchical_bootstrap(
            rows,
            specification["models"],
            specification["arm_a"],
            specification["arm_b"],
            bootstrap_samples,
            bootstrap_seed,
            specification["name"],
        ),
    }


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Write one formatted JSON object atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[JsonObject],
) -> None:
    """Write canonical JSONL atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(canonical_json(row))
            output_file.write("\n")
    temporary_path.replace(path)


def summary_from_long(
    rows: Sequence[JsonObject],
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> JsonObject:
    """Build a compact nested summary from long-form metrics."""

    summary: JsonObject = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "seven_models": list(seven_models),
        "five_models": list(five_models),
        "model_pooled": {},
        "fleet_model_macro": {},
    }
    model_block: JsonObject = cast(JsonObject, summary["model_pooled"])
    fleet_block: JsonObject = cast(
        JsonObject,
        summary["fleet_model_macro"],
    )
    for row in rows:
        level: str = require_string(row.get("level"), "metric.level")
        split: str = require_string(row.get("split"), "metric.split")
        arm: str = require_string(row.get("arm"), "metric.arm")
        if level == "model_pooled":
            model: str = require_string(row.get("model"), "metric.model")
            model_value: JsonValue | None = model_block.get(model)
            model_summary: JsonObject = (
                {}
                if model_value is None
                else require_object(
                    model_value,
                    f"summary.model_pooled.{model}",
                )
            )
            model_block[model] = model_summary
            split_value: JsonValue | None = model_summary.get(split)
            split_summary: JsonObject = (
                {}
                if split_value is None
                else require_object(
                    split_value,
                    f"summary.model_pooled.{model}.{split}",
                )
            )
            model_summary[split] = split_summary
            split_summary[arm] = row
        elif level == "fleet_model_macro":
            support: str = require_string(
                row.get("support"),
                "metric.support",
            )
            support_value: JsonValue | None = fleet_block.get(support)
            support_summary: JsonObject = (
                {}
                if support_value is None
                else require_object(
                    support_value,
                    f"summary.fleet_model_macro.{support}",
                )
            )
            fleet_block[support] = support_summary
            split_value = support_summary.get(split)
            split_summary = (
                {}
                if split_value is None
                else require_object(
                    split_value,
                    f"summary.fleet_model_macro.{support}.{split}",
                )
            )
            support_summary[split] = split_summary
            split_summary[arm] = row
    return summary


def main() -> None:
    """Validate exact supports and write fresh metrics and comparisons."""

    args = parse_args()
    seven_models: tuple[str, ...] = parse_model_list(
        str(args.seven_models),
        "seven-models",
    )
    five_models: tuple[str, ...] = parse_model_list(
        str(args.five_models),
        "five-models",
    )
    if len(seven_models) != 7:
        raise ValueError(
            "seven-models must contain exactly seven models: "
            f"actual={len(seven_models)}"
        )
    if len(five_models) != 5:
        raise ValueError(
            "five-models must contain exactly five models: "
            f"actual={len(five_models)}"
        )
    if not set(five_models).issubset(set(seven_models)):
        raise ValueError(
            "five-models must be a subset of seven-models: "
            f"five={five_models}, seven={seven_models}"
        )
    expected_total_per_model: int = int(args.expected_total_per_model)
    expected_heldout_per_model: int = int(
        args.expected_heldout_per_model
    )
    if expected_total_per_model != sum(DOMAIN_COUNTS.values()):
        raise ValueError(
            "Expected total does not match the frozen four-domain protocol: "
            f"value={expected_total_per_model}"
        )
    frozen_heldout: int = expected_total_per_model - sum(
        DOMAIN_VALIDATION_COUNTS.values()
    )
    if expected_heldout_per_model != frozen_heldout:
        raise ValueError(
            "Expected held-out total does not match the frozen split: "
            f"value={expected_heldout_per_model}, frozen={frozen_heldout}"
        )
    bootstrap_samples: int = int(args.bootstrap_samples)
    bootstrap_seed: int = int(args.bootstrap_seed)
    if bootstrap_samples <= 0:
        raise ValueError(
            "bootstrap-samples must be positive: "
            f"value={bootstrap_samples}"
        )
    k2_paths: list[Path] = resolve_exact_paths(
        cast(list[Path], args.k2_answers),
        int(args.expected_k2_files),
        "K=2 public answer",
    )
    baseline_paths: list[Path] = resolve_exact_paths(
        cast(list[Path], args.baseline_evals),
        int(args.expected_baseline_eval_files),
        "fresh baseline evaluation",
    )
    k2_rows: list[ComparisonRow] = load_k2_rows(k2_paths)
    baseline_rows: list[ComparisonRow] = load_baseline_rows(
        baseline_paths
    )
    verify_coverage(
        k2_rows,
        baseline_rows,
        seven_models,
        five_models,
        expected_total_per_model,
        expected_heldout_per_model,
    )
    long_metrics: list[JsonObject] = build_long_metrics(
        baseline_rows,
        seven_models,
        five_models,
    )
    all_rows: list[ComparisonRow] = [*k2_rows, *baseline_rows]
    specifications: tuple[ComparisonSpec, ...] = comparison_specs(
        seven_models,
        five_models,
    )
    comparisons: JsonObject = {
        "schema_version": COMPARISONS_SCHEMA_VERSION,
        "split": "heldout",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "comparison_count": len(specifications),
        "comparisons": {
            specification["name"]: paired_report(
                all_rows,
                specification,
                bootstrap_samples,
                bootstrap_seed,
            )
            for specification in specifications
        },
        "support": {
            "seven_models": list(seven_models),
            "five_models": list(five_models),
            "expected_total_per_model": expected_total_per_model,
            "expected_heldout_per_model": expected_heldout_per_model,
            "fresh_baseline_rows": len(baseline_rows),
            "k2_comparison_rows": len(k2_rows),
        },
        "provenance": {
            "heldout_authority": (
                "K=2 public answer_per_instance is_validation flags; "
                "fresh baseline flags were required to match exactly."
            ),
            "legacy_compact_baseline_read": False,
            "baseline_runtime_identity_gate": "fresh_job_bound_manifests",
            "k2_answers": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in sorted(k2_paths)
            ],
            "baseline_evaluations": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in sorted(baseline_paths)
            ],
        },
        "notes": (
            "All paired inference excludes frozen K=2 calibration IDs. "
            "Fleet intervals resample model, then domain, then paired "
            "instances. Deterministic method failures remain incorrect."
        ),
    }
    summary: JsonObject = summary_from_long(
        long_metrics,
        seven_models,
        five_models,
    )
    summary["expected_total_per_model"] = expected_total_per_model
    summary["expected_heldout_per_model"] = expected_heldout_per_model
    summary["fresh_baseline_rows"] = len(baseline_rows)
    summary["legacy_compact_baseline_read"] = False
    summary["inputs"] = {
        "k2_answers": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in sorted(k2_paths)
        ],
        "baseline_evaluations": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in sorted(baseline_paths)
        ],
    }

    output_long: Path = cast(Path, args.output_long).resolve()
    output_summary: Path = cast(Path, args.output_summary).resolve()
    output_comparisons: Path = cast(
        Path,
        args.output_comparisons,
    ).resolve()
    write_jsonl_atomic(output_long, long_metrics)
    write_json_atomic(output_summary, summary)
    write_json_atomic(output_comparisons, comparisons)
    print(
        canonical_json(
            {
                "event": "runtime_matched_baseline_summary_complete",
                "fresh_baseline_rows": len(baseline_rows),
                "k2_comparison_rows": len(k2_rows),
                "comparisons": len(specifications),
                "output_long": str(output_long),
                "output_summary": str(output_summary),
                "output_comparisons": str(output_comparisons),
            }
        )
    )


if __name__ == "__main__":
    main()
