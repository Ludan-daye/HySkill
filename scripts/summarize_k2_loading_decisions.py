#!/usr/bin/env python3
"""Summarize K=2 decision-level loading metrics on strict support sets."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from hyskill.downstream_reuse import (
    DownstreamDataError,
    FailureCategory,
    JsonLike,
    JsonObject,
    JsonValue,
    canonical_json,
    sha256_file,
    validate_failure_category,
)
from hyskill.loading_metrics import (
    LoadingArm,
    LoadingDecisionRow,
    LoadingMetrics,
    compute_loading_metrics,
    mean_defined,
)
from scripts.audit_k2_reuse import require_list, require_object, require_string


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
SplitName = Literal["full", "heldout"]
SupportName = Literal["seven_model", "five_model"]


def parse_args() -> argparse.Namespace:
    """Parse explicit strict support and output arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--seven-models", required=True)
    parser.add_argument("--five-models", required=True)
    parser.add_argument("--expected-total-per-model", required=True, type=int)
    parser.add_argument("--expected-heldout-per-model", required=True, type=int)
    parser.add_argument("--output-long", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    return parser.parse_args()


def parse_model_list(value: str, name: str) -> tuple[str, ...]:
    """Parse one ordered, unique comma-separated model support set."""

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


def require_optional_boolean(
    value: JsonValue | None,
    context: str,
) -> bool | None:
    """Return one JSON Boolean or null."""

    if value is None or isinstance(value, bool):
        return value
    raise DownstreamDataError(
        f"Expected Boolean or null: context={context}, value={value!r}"
    )


def parse_loading_row(
    raw_row: JsonValue,
    path: Path,
    line_number: int,
) -> LoadingDecisionRow:
    """Validate one exported loading decision."""

    context: str = f"loading:{path}:{line_number}"
    row: JsonObject = require_object(raw_row, context)
    arm_value: str = require_string(row.get("arm"), f"{context}.arm")
    if arm_value not in ("routed_always", "routed_gated", "routed_select"):
        raise DownstreamDataError(
            f"Unknown loading arm: context={context}, arm={arm_value}"
        )
    raw_expected: list[JsonValue] = require_list(
        row.get("expected_skill_ids"),
        f"{context}.expected_skill_ids",
    )
    expected: list[str] = [
        require_string(value, f"{context}.expected_skill_ids[{index}]")
        for index, value in enumerate(raw_expected)
    ]
    raw_gold: list[JsonValue] = require_list(
        row.get("gold_skill_ids"),
        f"{context}.gold_skill_ids",
    )
    gold: list[str] = [
        require_string(value, f"{context}.gold_skill_ids[{index}]")
        for index, value in enumerate(raw_gold)
    ]
    loaded: bool = require_boolean(row.get("loaded"), f"{context}.loaded")
    hit: bool | None = require_optional_boolean(row.get("hit"), f"{context}.hit")
    gold_loaded: bool = require_boolean(
        row.get("gold_loaded"),
        f"{context}.gold_loaded",
    )
    if loaded != bool(expected):
        raise DownstreamDataError(
            "Loading flag disagrees with expected skill IDs: "
            f"context={context}, loaded={loaded}, expected={expected}"
        )
    expected_hit: bool | None = (
        any(skill_id in set(gold) for skill_id in expected)
        if expected
        else None
    )
    if hit != expected_hit or gold_loaded != (expected_hit is True):
        raise DownstreamDataError(
            "Loading hit fields disagree with expected and gold skills: "
            f"context={context}, expected_hit={expected_hit}, "
            f"hit={hit}, gold_loaded={gold_loaded}"
        )
    category: FailureCategory = validate_failure_category(
        row.get("failure_category")
    )
    if category in ("infra_transient", "unclassified_error"):
        raise DownstreamDataError(
            "Loading rows cannot contain unresolved failures: "
            f"context={context}, failure_category={category}"
        )
    return {
        "schema_version": require_string(
            row.get("schema_version"),
            f"{context}.schema_version",
        ),
        "instance_id": require_string(
            row.get("instance_id"),
            f"{context}.instance_id",
        ),
        "model": require_string(row.get("model"), f"{context}.model"),
        "domain": require_string(row.get("domain"), f"{context}.domain"),
        "arm": cast(LoadingArm, arm_value),
        "expected_skill_ids": expected,
        "gold_skill_ids": gold,
        "loaded": loaded,
        "hit": hit,
        "gold_loaded": gold_loaded,
        "is_validation": require_boolean(
            row.get("is_validation"),
            f"{context}.is_validation",
        ),
        "failure_category": category,
        "decision_source_sha256": require_string(
            row.get("decision_source_sha256"),
            f"{context}.decision_source_sha256",
        ),
    }


def load_rows(paths: Sequence[Path]) -> list[LoadingDecisionRow]:
    """Load unique loading rows from explicit JSONL inputs."""

    rows: list[LoadingDecisionRow] = []
    seen_keys: set[tuple[str, str, LoadingArm, str]] = set()
    for path in paths:
        resolved_path: Path = path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Loading decision input does not exist: path={resolved_path}"
            )
        with resolved_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    raw_row: JsonValue = cast(JsonValue, json.loads(line))
                except json.JSONDecodeError as error:
                    raise DownstreamDataError(
                        "Loading JSONL is malformed: "
                        f"path={resolved_path}, line={line_number}, "
                        f"column={error.colno}, message={error.msg}"
                    ) from error
                row: LoadingDecisionRow = parse_loading_row(
                    raw_row,
                    resolved_path,
                    line_number,
                )
                key: tuple[str, str, LoadingArm, str] = (
                    row["model"],
                    row["domain"],
                    row["arm"],
                    row["instance_id"],
                )
                if key in seen_keys:
                    raise DownstreamDataError(
                        f"Duplicate loading decision row: key={key}"
                    )
                seen_keys.add(key)
                rows.append(row)
    if not rows:
        raise DownstreamDataError("No loading decision rows were loaded")
    return rows


def split_rows(
    rows: Sequence[LoadingDecisionRow],
    split: SplitName,
) -> list[LoadingDecisionRow]:
    """Return full or held-out decision support."""

    if split == "full":
        return list(rows)
    return [row for row in rows if not row["is_validation"]]


def metric_record(
    level: str,
    split: SplitName,
    metrics: LoadingMetrics,
    model: str | None,
    domain: str | None,
    arm: LoadingArm,
    support: SupportName | None,
) -> JsonObject:
    """Build one long-form metric record."""

    return {
        "schema_version": "k2-loading-metrics-v1",
        "level": level,
        "split": split,
        "model": model,
        "domain": domain,
        "arm": arm,
        "support": support,
        **metrics,
    }


def verify_domain_coverage(
    rows: Sequence[LoadingDecisionRow],
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> None:
    """Verify exact per-domain denominators for every supported arm."""

    for model in seven_models:
        for domain in DOMAINS:
            for arm in ("routed_always", "routed_gated"):
                selected: list[LoadingDecisionRow] = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["domain"] == domain
                    and row["arm"] == arm
                ]
                expected_count: int = DOMAIN_COUNTS[domain]
                if len(selected) != expected_count:
                    raise DownstreamDataError(
                        "Seven-model loading denominator mismatch: "
                        f"model={model}, domain={domain}, arm={arm}, "
                        f"expected={expected_count}, actual={len(selected)}"
                    )
                validation_count: int = sum(
                    1 for row in selected if row["is_validation"]
                )
                if validation_count != DOMAIN_VALIDATION_COUNTS[domain]:
                    raise DownstreamDataError(
                        "Validation denominator mismatch: "
                        f"model={model}, domain={domain}, arm={arm}, "
                        f"expected={DOMAIN_VALIDATION_COUNTS[domain]}, "
                        f"actual={validation_count}"
                    )
    for model in five_models:
        for domain in DOMAINS:
            selected = [
                row
                for row in rows
                if row["model"] == model
                and row["domain"] == domain
                and row["arm"] == "routed_select"
            ]
            expected_count = DOMAIN_COUNTS[domain]
            if len(selected) != expected_count:
                raise DownstreamDataError(
                    "Five-model Select denominator mismatch: "
                    f"model={model}, domain={domain}, expected={expected_count}, "
                    f"actual={len(selected)}"
                )
            validation_count = sum(
                1 for row in selected if row["is_validation"]
            )
            if validation_count != DOMAIN_VALIDATION_COUNTS[domain]:
                raise DownstreamDataError(
                    "Select validation denominator mismatch: "
                    f"model={model}, domain={domain}, "
                    f"expected={DOMAIN_VALIDATION_COUNTS[domain]}, "
                    f"actual={validation_count}"
                )


def model_macro_record(
    split: SplitName,
    arm: LoadingArm,
    support: SupportName,
    model_metrics: Sequence[LoadingMetrics],
) -> JsonObject:
    """Build equal-model-weighted loading metrics."""

    precision, precision_models = mean_defined(
        [metrics["loaded_skill_precision"] for metrics in model_metrics]
    )
    loading_rate, loading_rate_models = mean_defined(
        [metrics["loading_rate"] for metrics in model_metrics]
    )
    gold_load_rate, gold_load_rate_models = mean_defined(
        [metrics["gold_load_rate"] for metrics in model_metrics]
    )
    selection_failure_rate, failure_rate_models = mean_defined(
        [metrics["selection_failure_rate"] for metrics in model_metrics]
    )
    return {
        "schema_version": "k2-loading-metrics-v1",
        "level": "fleet_model_macro",
        "split": split,
        "model": None,
        "domain": None,
        "arm": arm,
        "support": support,
        "models": len(model_metrics),
        "instances": sum(metrics["instances"] for metrics in model_metrics),
        "loaded": sum(metrics["loaded"] for metrics in model_metrics),
        "gold_loaded": sum(metrics["gold_loaded"] for metrics in model_metrics),
        "method_failures": sum(
            metrics["method_failures"] for metrics in model_metrics
        ),
        "loaded_skill_precision": precision,
        "loading_rate": loading_rate,
        "gold_load_rate": gold_load_rate,
        "selection_failure_rate": selection_failure_rate,
        "metric_model_denominators": {
            "loaded_skill_precision": precision_models,
            "loading_rate": loading_rate_models,
            "gold_load_rate": gold_load_rate_models,
            "selection_failure_rate": failure_rate_models,
        },
    }


def build_metric_records(
    rows: Sequence[LoadingDecisionRow],
    seven_models: Sequence[str],
    five_models: Sequence[str],
) -> list[JsonObject]:
    """Build per-domain, per-model, fleet-micro, and model-macro records."""

    output: list[JsonObject] = []
    all_models: tuple[str, ...] = tuple(seven_models)
    for split in ("full", "heldout"):
        split_name: SplitName = cast(SplitName, split)
        active_rows: list[LoadingDecisionRow] = split_rows(rows, split_name)
        for model in all_models:
            available_arms: tuple[LoadingArm, ...] = (
                ("routed_always", "routed_gated", "routed_select")
                if model in set(five_models)
                else ("routed_always", "routed_gated")
            )
            for arm in available_arms:
                pooled: list[LoadingDecisionRow] = [
                    row
                    for row in active_rows
                    if row["model"] == model and row["arm"] == arm
                ]
                output.append(
                    metric_record(
                        "per_model_pooled",
                        split_name,
                        compute_loading_metrics(pooled),
                        model,
                        None,
                        arm,
                        None,
                    )
                )
                for domain in DOMAINS:
                    domain_rows: list[LoadingDecisionRow] = [
                        row for row in pooled if row["domain"] == domain
                    ]
                    output.append(
                        metric_record(
                            "per_model_domain",
                            split_name,
                            compute_loading_metrics(domain_rows),
                            model,
                            domain,
                            arm,
                            None,
                        )
                    )
        support_specs: tuple[
            tuple[SupportName, Sequence[str], tuple[LoadingArm, ...]], ...
        ] = (
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
        )
        for support, models, arms in support_specs:
            for arm in arms:
                fleet_rows: list[LoadingDecisionRow] = [
                    row
                    for row in active_rows
                    if row["model"] in set(models) and row["arm"] == arm
                ]
                output.append(
                    metric_record(
                        "fleet_micro",
                        split_name,
                        compute_loading_metrics(fleet_rows),
                        None,
                        None,
                        arm,
                        support,
                    )
                )
                per_model: list[LoadingMetrics] = [
                    compute_loading_metrics(
                        [
                            row
                            for row in fleet_rows
                            if row["model"] == model
                        ]
                    )
                    for model in models
                ]
                output.append(
                    model_macro_record(
                        split_name,
                        arm,
                        support,
                        per_model,
                    )
                )
    return output


def write_jsonl_atomic(path: Path, rows: Sequence[JsonObject]) -> None:
    """Atomically write long-form metrics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(canonical_json(row) + "\n")
    temporary_path.replace(path)


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    """Atomically write a formatted summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Validate strict supports and emit all registered loading aggregates."""

    args = parse_args()
    input_paths: list[Path] = [
        path.resolve() for path in cast(list[Path], args.inputs)
    ]
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
            f"seven-models must contain exactly 7 models: actual={len(seven_models)}"
        )
    if len(five_models) != 5:
        raise ValueError(
            f"five-models must contain exactly 5 models: actual={len(five_models)}"
        )
    if not set(five_models).issubset(set(seven_models)):
        raise ValueError(
            "five-models must be a subset of seven-models: "
            f"five_models={five_models}, seven_models={seven_models}"
        )
    expected_total_per_model: int = int(args.expected_total_per_model)
    expected_heldout_per_model: int = int(args.expected_heldout_per_model)
    protocol_total: int = sum(DOMAIN_COUNTS.values())
    protocol_heldout: int = protocol_total - sum(
        DOMAIN_VALIDATION_COUNTS.values()
    )
    if expected_total_per_model != protocol_total:
        raise ValueError(
            "Explicit total denominator disagrees with protocol domains: "
            f"explicit={expected_total_per_model}, protocol={protocol_total}"
        )
    if expected_heldout_per_model != protocol_heldout:
        raise ValueError(
            "Explicit held-out denominator disagrees with protocol split: "
            f"explicit={expected_heldout_per_model}, protocol={protocol_heldout}"
        )
    rows: list[LoadingDecisionRow] = load_rows(input_paths)
    observed_models: set[str] = {row["model"] for row in rows}
    if observed_models != set(seven_models):
        raise DownstreamDataError(
            "Loading rows do not match the seven-model support: "
            f"missing={sorted(set(seven_models) - observed_models)}, "
            f"unexpected={sorted(observed_models - set(seven_models))}"
        )
    verify_domain_coverage(rows, seven_models, five_models)
    metric_rows: list[JsonObject] = build_metric_records(
        rows,
        seven_models,
        five_models,
    )
    output_long_path: Path = cast(Path, args.output_long).resolve()
    output_summary_path: Path = cast(Path, args.output_summary).resolve()
    write_jsonl_atomic(output_long_path, metric_rows)
    fleet_model_macro: list[JsonObject] = [
        row for row in metric_rows if row["level"] == "fleet_model_macro"
    ]
    summary: JsonObject = {
        "schema_version": "k2-loading-summary-v1",
        "valid": True,
        "input_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in input_paths
        ],
        "input_records": len(rows),
        "seven_model_support": list(seven_models),
        "five_model_support": list(five_models),
        "expected_total_per_model": expected_total_per_model,
        "expected_heldout_per_model": expected_heldout_per_model,
        "fleet_model_macro": fleet_model_macro,
        "long_metrics_path": str(output_long_path),
        "long_metrics_sha256": sha256_file(output_long_path),
    }
    write_json_atomic(output_summary_path, summary)
    print(
        canonical_json(
            {
                "event": "k2_loading_summary_complete",
                "input_records": len(rows),
                "metric_records": len(metric_rows),
                "output_long": str(output_long_path),
                "output_summary": str(output_summary_path),
                "output_summary_sha256": sha256_file(output_summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()
